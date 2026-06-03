"""Semantic chunking: sentence-embedding breakpoints, content-only text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

MAX_WORDS = 140
MIN_WORDS = 40
BREAKPOINT_PCTL = 20
TITLE_PREPEND_THRESHOLD = 50
SLIDING_FALLBACK_OVERLAP = 30

# AD years 1000-2099; word boundaries prevent partial matches inside longer numbers
_YEAR_RE = re.compile(r"\b((?:1[0-9]|20)\d{2})\b")
# BC/BCE years stored as negative ints: "300 BC" -> -300, "44 BCE" -> -44
_BC_YEAR_RE = re.compile(r"\b(\d{1,4})\s+BC(?:E)?\b", re.IGNORECASE)
# Optional-suffix form covers full names, 3-letter abbrs, and "Sept" in one pattern
_MONTH_RE = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May"
    r"|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?"
    r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\b",
    re.IGNORECASE,
)
_MONTH_NORM: Dict[str, str] = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def extract_years(text: str) -> Set[int]:
    ad = {int(m) for m in _YEAR_RE.findall(text)}
    bc = {-int(m) for m in _BC_YEAR_RE.findall(text)}
    return ad | bc


def extract_months(text: str) -> Set[str]:
    return {_MONTH_NORM[m[:3].lower()] for m in _MONTH_RE.findall(text)}


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str
    title: str = ""
    years: Set[int] = field(default_factory=set)
    months: Set[str] = field(default_factory=set)


def _split_sentences(content: str) -> Tuple[List[str], Set[int]]:
    """Return (sentences, forced_break_indices).

    Forced breaks correspond to indices i such that sentence i begins a new
    paragraph (i.e. there was a blank-line boundary before it).
    """
    paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(content) if p.strip()]
    sentences: List[str] = []
    forced: Set[int] = set()
    for para in paragraphs:
        if sentences:
            forced.add(len(sentences))
        parts = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        if not parts:
            continue
        sentences.extend(parts)
    return sentences, forced


def _sliding_split(sentence: str, max_words: int, overlap: int) -> List[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    step = max_words - overlap
    out: List[str] = []
    for start in range(0, max(1, len(words) - overlap), step):
        out.append(" ".join(words[start : start + max_words]))
    return out


def _assemble_chunks(
    sentences: Sequence[str],
    breakpoints: Set[int],
    max_words: int,
) -> List[str]:
    """Greedy: walk sentences, cut at breakpoint OR when adding next exceeds max_words.

    Sentences longer than max_words are sliding-window split in place.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0
    for i, sent in enumerate(sentences):
        sent_words = len(sent.split())
        if sent_words > max_words:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
            chunks.extend(_sliding_split(sent, max_words, SLIDING_FALLBACK_OVERLAP))
            continue
        is_break = i in breakpoints
        if buf and (is_break or buf_words + sent_words > max_words):
            chunks.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(sent)
        buf_words += sent_words
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _merge_residuals(chunks: List[str], min_words: int, max_words: int) -> List[str]:
    """Merge any chunk < min_words into a neighbor (prefer previous)."""
    if not chunks:
        return chunks
    out: List[str] = [chunks[0]]
    for ch in chunks[1:]:
        if len(out[-1].split()) < min_words:
            merged = out[-1] + " " + ch
            if len(merged.split()) <= max_words + min_words:
                out[-1] = merged
                continue
        out.append(ch)
    if len(out) >= 2 and len(out[-1].split()) < min_words:
        merged = out[-2] + " " + out[-1]
        if len(merged.split()) <= max_words + min_words:
            out[-2] = merged
            out.pop()
    return out


def _semantic_breakpoints(
    sent_vecs: np.ndarray,
    forced: Set[int],
    pctl: float,
) -> Set[int]:
    n = sent_vecs.shape[0]
    if n < 2:
        return set(forced)
    sims = np.sum(sent_vecs[:-1] * sent_vecs[1:], axis=1)
    if sims.size == 0:
        return set(forced)
    threshold = float(np.percentile(sims, pctl))
    breaks = {i + 1 for i, s in enumerate(sims) if s < threshold}
    return breaks | forced


def chunk_entry(record: Dict[str, Any], *, sent_encoder=None) -> List[Chunk]:
    """Semantic-chunk one record.

    sent_encoder: callable(list[str]) -> np.ndarray of L2-normalized vectors.
    If None, falls back to single-chunk-per-page (used only when caller wants
    a degenerate path; production builds always pass encoder).
    """
    page_id = int(record["page_id"])
    title = record.get("title", "")
    content = record.get("content", "") or ""

    sentences, forced = _split_sentences(content)
    if not sentences:
        return []

    if len(sentences) == 1 or sent_encoder is None:
        bodies = (
            _sliding_split(
                sentences[0] if sentences else "", MAX_WORDS, SLIDING_FALLBACK_OVERLAP
            )
            if len(sentences) == 1
            else [" ".join(sentences)]
        )
    else:
        sent_vecs = sent_encoder(sentences)
        breaks = _semantic_breakpoints(sent_vecs, forced, BREAKPOINT_PCTL)
        bodies = _assemble_chunks(sentences, breaks, MAX_WORDS)
        bodies = _merge_residuals(bodies, MIN_WORDS, MAX_WORDS)

    chunks: List[Chunk] = []
    for i, body in enumerate(bodies):
        body = body.strip()
        if not body:
            continue
        if len(body.split()) < TITLE_PREPEND_THRESHOLD and title:
            text = f"{title}. {body}"
        else:
            text = body
        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=i,
                text=text,
                title=title,
                years=extract_years(body),
                months=extract_months(body),
            )
        )
    return chunks


def chunk_corpus(
    records: List[Dict[str, Any]],
    *,
    sent_encoder=None,
) -> List[Chunk]:
    """Semantic-chunk corpus. sent_encoder lazy-built if not provided.

    The encoder is invoked per record (ST batches internally) so memory stays bounded.
    """
    if sent_encoder is None:
        from embed import embed_texts

        def sent_encoder(texts):
            return embed_texts(texts, show_progress=False)

    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record, sent_encoder=sent_encoder))
    return chunks

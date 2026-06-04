"""Chunking strategies: semantic (embedding breakpoints) or sliding window."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from embed import embed_texts, get_model

MAX_WORDS = 170
MIN_WORDS = 50
BREAKPOINT_PCTL = 20
TITLE_PREPEND_THRESHOLD = 60   # words
SLIDING_FALLBACK_OVERLAP = 40  # words
ENCODE_BATCH_SIZE = 27000
_SLIDING_WINDOW_MAX_TOKENS = 200  # token cap for debug sliding strategy

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


def _count_words(text: str) -> int:
    return len(text.split())


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


def _sliding_split(sentence: str, max_words: int, overlap_words: int) -> List[str]:
    """Split a too-long sentence by sliding a word window."""
    words = sentence.split()
    n = len(words)
    if n <= max_words:
        return [sentence]
    step = max_words - overlap_words
    out: List[str] = []
    start = 0
    while start + max_words < n:
        out.append(" ".join(words[start : start + max_words]))
        start += step
    # Anchor final window to end so tail is exactly max_words wide.
    out.append(" ".join(words[n - max_words : n]))
    return out


def _assemble_chunks(
    sentences: Sequence[str],
    sent_word_counts: Sequence[int],
    breakpoints: Set[int],
    max_words: int,
) -> List[str]:
    """Greedy: walk sentences, cut at breakpoint OR when adding next exceeds max_words.

    Sentences longer than max_words are sliding-window split in place.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0
    for i, (sent, sent_toks) in enumerate(zip(sentences, sent_word_counts)):
        if sent_toks > max_words:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
            chunks.extend(_sliding_split(sent, max_words, SLIDING_FALLBACK_OVERLAP))
            continue
        is_break = i in breakpoints
        if buf and (is_break or buf_words + sent_toks > max_words):
            chunks.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(sent)
        buf_words += sent_toks
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _merge_residuals(chunks: List[str], min_words: int, max_words: int) -> List[str]:
    """Merge any chunk < min_words into a neighbor (prefer previous).

    Cap is max_words: merging past the safe margin would silently truncate
    at embed time. Unmergeable residuals stay tiny and are rescued by
    title-prepend in _make_chunks.
    """
    if not chunks:
        return chunks
    out: List[str] = [chunks[0]]
    out_tc: List[int] = [_count_words(chunks[0])]
    for ch in chunks[1:]:
        ch_tc = _count_words(ch)
        if out_tc[-1] < min_words and out_tc[-1] + ch_tc <= max_words:
            out[-1] = out[-1] + " " + ch
            out_tc[-1] += ch_tc
        else:
            out.append(ch)
            out_tc.append(ch_tc)
    if (
        len(out) >= 2
        and out_tc[-1] < min_words
        and out_tc[-2] + out_tc[-1] <= max_words
    ):
        out[-2] = out[-2] + " " + out[-1]
        out.pop()
    return out


_MIN_SENTS_FOR_SEMANTIC_BREAK = 5


def _semantic_breakpoints(
    sent_vecs: np.ndarray,
    forced: Set[int],
    pctl: float,
) -> Set[int]:
    # Below ~5 sentences the percentile threshold is degenerate and forces
    # a break regardless of how cohesive the page actually is. Fall back
    # to paragraph-only (forced) breaks for short pages.
    n = sent_vecs.shape[0]
    if n < _MIN_SENTS_FOR_SEMANTIC_BREAK:
        return set(forced)
    sims = np.sum(sent_vecs[:-1] * sent_vecs[1:], axis=1)
    threshold = float(np.percentile(sims, pctl))
    breaks = {i + 1 for i, s in enumerate(sims) if s < threshold}
    return breaks | forced


def _sliding_window_page_chunks(content: str, page_id: int, title: str) -> List[Chunk]:
    """Chunk a single page with a fixed token-size sliding window (no embeddings)."""
    tokenizer = get_model().tokenizer
    token_ids = tokenizer.encode(content, add_special_tokens=False)
    n = len(token_ids)
    if n == 0:
        return []
    cap = _SLIDING_WINDOW_MAX_TOKENS
    step = cap - SLIDING_FALLBACK_OVERLAP
    windows: List[str] = []
    if n <= cap:
        windows.append(tokenizer.decode(token_ids))
    else:
        start = 0
        while start + cap < n:
            windows.append(tokenizer.decode(token_ids[start : start + cap]))
            start += step
        windows.append(tokenizer.decode(token_ids[n - cap : n]))
    return _make_chunks(windows, page_id, title)


def _make_chunks(bodies: List[str], page_id: int, title: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, body in enumerate(bodies):
        body = body.strip()
        if not body:
            continue
        # Tiny chunks embed as noise; prepend title to anchor the vector
        # topically. Metadata (years/months) still extracted from body
        # alone so title tokens don't leak into filters.
        if title and _count_words(body) < TITLE_PREPEND_THRESHOLD:
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
    show_progress: bool = True,
    encode_batch: int = ENCODE_BATCH_SIZE,
    strategy: str = "semantic",
) -> List[Chunk]:
    """Chunk corpus using ``strategy``.

    ``"semantic"``: sentence-embedding breakpoints (default).
    ``"sliding"``: fixed token-size sliding window, no embeddings — fast, for debugging.
    """
    if strategy == "sliding":
        chunks: List[Chunk] = []
        for record in records:
            content = record.get("content", "") or ""
            page_id = int(record["page_id"])
            title = record.get("title", "")
            chunks.extend(_sliding_window_page_chunks(content, page_id, title))
        return chunks

    chunks = []
    for batch_start in range(0, len(records), encode_batch):
        batch = records[batch_start : batch_start + encode_batch]
        batch_sentences: List[str] = []
        batch_spans: List[Tuple[Dict[str, Any], int, int, Set[int]]] = []
        for record in batch:
            content = record.get("content", "") or ""
            sents, forced = _split_sentences(content)
            lo = len(batch_sentences)
            batch_sentences.extend(sents)
            batch_spans.append((record, lo, len(batch_sentences), forced))
        if not batch_sentences:
            continue
        batch_vecs = embed_texts(batch_sentences)
        batch_word_counts = [len(s.split()) for s in batch_sentences]
        for record, lo, hi, forced in batch_spans:
            sents = batch_sentences[lo:hi]
            if not sents:
                continue
            page_id = int(record["page_id"])
            title = record.get("title", "")
            word_counts = batch_word_counts[lo:hi]
            if len(sents) == 1:
                bodies = _sliding_split(sents[0], MAX_WORDS, SLIDING_FALLBACK_OVERLAP)
            else:
                breaks = _semantic_breakpoints(
                    batch_vecs[lo:hi], forced, BREAKPOINT_PCTL
                )
                bodies = _assemble_chunks(sents, word_counts, breaks, MAX_WORDS)
                bodies = _merge_residuals(bodies, MIN_WORDS, MAX_WORDS)
            chunks.extend(_make_chunks(bodies, page_id, title))
    return chunks

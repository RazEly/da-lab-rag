"""Chunking: word-count sliding window, paragraph-aware single-sentence overlap.

Sentences are greedily accumulated into ~``MAX_WORDS`` windows; each window
borrows one neighbor sentence per side as overlap, but never across a paragraph
break. Embedded/BM25 text is ``[title] body`` for every chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from utils import header_text, normalize_page_id, peel_headers

MAX_WORDS = 120
SLIDING_FALLBACK_OVERLAP = 40  # words

# Any run of digits; word boundaries avoid splitting numbers embedded in words.
_NUMBER_RE = re.compile(r"\b\d+\b")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def extract_numbers(text: str) -> Set[int]:
    """All integer numbers in `text` (powers query-time number pre/post-filter)."""
    return {int(m) for m in _NUMBER_RE.findall(text)}


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str
    title: str = ""
    section: str = ""  # recovered section heading; "" = page lead/intro
    section_number: int = 0  # 0 = lead; 1,2,... in document order per page
    numbers: Set[int] = field(default_factory=set)


def _count_words(text: str) -> int:
    return len(text.split())


def _split_sentences(
    content: str,
) -> Tuple[List[str], Set[int], List[int], Dict[int, str]]:
    """Return (body_sentences, forced_breaks, sent_sections, section_names).

    Flattened section headings are peeled off each paragraph tail (see
    ``utils.peel_headers``) and excluded from ``body_sentences`` — they are
    labels, not content. Each peeled heading opens a new section:
    ``sent_sections[i]`` is the section number of body sentence ``i`` and
    ``section_names`` maps that number to its title (number 0 = the page lead,
    before any heading).

    Forced breaks = indices where a sentence begins a new paragraph. A heading
    sits at one paragraph's tail and its body starts in the next, so every
    section change coincides with a forced break — no chunk straddles sections.
    """
    paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(content) if p.strip()]
    sentences: List[str] = []
    forced: Set[int] = set()
    sent_sections: List[int] = []
    section_names: Dict[int, str] = {0: ""}
    cur = 0
    for para in paragraphs:
        parts = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        if not parts:
            continue
        body, headings = peel_headers(parts)
        if body:
            if sentences:
                forced.add(len(sentences))
            for s in body:
                sentences.append(s)
                sent_sections.append(cur)
        for h in headings:
            cur += 1
            section_names[cur] = header_text(h)
    return sentences, forced, sent_sections, section_names


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


def _make_chunks(
    bodies: List[str],
    page_id: int,
    title: str,
    sections: Sequence[Tuple[int, str]],
) -> List[Chunk]:
    """Wrap chunk bodies as ``Chunk``s with ``[title] body`` embedded text.

    Title is prepended to the dense/BM25 text of every chunk. Metadata
    (numbers) is extracted from the body alone.
    """
    chunks: List[Chunk] = []
    for i, body in enumerate(bodies):
        body = body.strip()
        if not body:
            continue
        sec_num, sec_name = sections[i]
        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=i,
                text=f"[{title}] {body}",
                title=title,
                section=sec_name,
                section_number=sec_num,
                numbers=extract_numbers(body),
            )
        )
    return chunks


def _sliding_para_page_chunks(
    sents: Sequence[str],
    forced: Set[int],
    sent_sections: Sequence[int],
    section_names: Dict[int, str],
    page_id: int,
    title: str,
) -> List[Chunk]:
    """Word-count sliding window with paragraph-aware single-sentence overlap.

    Cutting ignores paragraph boundaries: greedily accumulate whole sentences
    until adding the next would exceed ``max_words``, then cut.

    Overlap padding adds one neighbor sentence per side but never across a
    paragraph break (``forced`` holds the indices that begin a paragraph):

    * break *before* the segment start (``start in forced``) -> no left pad
    * break *before* the segment end (``end in forced``) -> no right pad

    Over-long single sentences are sliding-window split in place and emitted
    unpadded. Section tag = section of the segment's first sentence (a size-cut
    window may span sections; first wins).
    """
    n = len(sents)
    word_counts = [_count_words(s) for s in sents]
    # (start_idx, end_idx_exclusive, sentence_list, padded?)
    raw: List[Tuple[int, int, List[str], bool]] = []
    buf: List[str] = []
    buf_start = 0
    buf_words = 0

    def _flush() -> None:
        nonlocal buf, buf_words
        if buf:
            raw.append((buf_start, buf_start + len(buf), list(buf), True))
            buf, buf_words = [], 0

    for i, (sent, wc) in enumerate(zip(sents, word_counts)):
        if wc > MAX_WORDS:
            _flush()
            for piece in _sliding_split(sent, MAX_WORDS, SLIDING_FALLBACK_OVERLAP):
                raw.append((i, i + 1, [piece], False))  # no overlap for split pieces
            continue
        if buf and buf_words + wc > MAX_WORDS:
            _flush()
        if not buf:
            buf_start = i
        buf.append(sent)
        buf_words += wc
    _flush()

    bodies: List[str] = []
    sections: List[Tuple[int, str]] = []
    for start, end, slist, padded in raw:
        parts = slist  # owned by this tuple, discarded after; safe to mutate
        if padded:
            # left pad: previous sentence, skipped when a paragraph break starts
            # the segment (start in forced).
            if (
                start > 0
                and start not in forced
                and word_counts[start - 1] <= MAX_WORDS
            ):
                parts.insert(0, sents[start - 1])
            # right pad: next sentence, skipped when a paragraph break starts at `end`.
            if end < n and end not in forced and word_counts[end] <= MAX_WORDS:
                parts.append(sents[end])
        sec = sent_sections[start]
        bodies.append(" ".join(parts))
        sections.append((sec, section_names.get(sec, "")))
    return _make_chunks(bodies, page_id, title, sections)


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Chunk corpus with the paragraph-aware word-count sliding window.

    No embeddings at chunk time. Each chunk's embedded/BM25 text is
    ``[title] body``.
    """
    chunks: List[Chunk] = []
    for record in records:
        content = record.get("content", "") or ""
        page_id = normalize_page_id(record["page_id"])
        title = record.get("title", "")
        sents, forced, sent_sections, section_names = _split_sentences(content)
        if not sents:
            continue
        chunks.extend(
            _sliding_para_page_chunks(
                sents,
                forced,
                sent_sections,
                section_names,
                page_id,
                title,
            )
        )
    return chunks

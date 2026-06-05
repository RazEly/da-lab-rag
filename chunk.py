"""Chunking strategies: semantic (embedding breakpoints) or sliding window."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from embed import embed_texts, get_model
from headers import header_text, peel_headers
from utils import normalize_page_id

MAX_WORDS = 120
MIN_WORDS = 50
BREAKPOINT_PCTL = 20
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
    section: str = ""  # recovered section heading; "" = page lead/intro
    section_number: int = 0  # 0 = lead; 1,2,... in document order per page
    years: Set[int] = field(default_factory=set)
    months: Set[str] = field(default_factory=set)


def _count_words(text: str) -> int:
    return len(text.split())


def _split_sentences(
    content: str,
) -> Tuple[List[str], Set[int], List[int], Dict[int, str]]:
    """Return (body_sentences, forced_breaks, sent_sections, section_names).

    Flattened section headings are peeled off each paragraph tail (see
    ``headers.peel_headers``) and excluded from ``body_sentences`` — they are
    labels, not content, so they never get embedded. Instead each peeled
    heading opens a new section: ``sent_sections[i]`` is the section number of
    body sentence ``i`` and ``section_names`` maps that number to its title
    (number 0 = the page lead, before any heading).

    Forced breaks correspond to indices i where sentence i begins a new
    paragraph. Because a heading always sits at the tail of one paragraph and
    its section body starts in the next, every section change coincides with a
    forced break — so no assembled chunk ever straddles two sections.
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


@dataclass
class _Segment:
    """A chunk-in-progress: its sentences + whether the cut *before* it was semantic.

    ``cut_before_semantic`` drives overlap padding: a segment is padded across
    a boundary only when that boundary was a size cut, never a semantic break.
    """

    sents: List[str]
    word_count: int
    cut_before_semantic: bool
    section_num: int = 0  # section of this segment's sentences (uniform: see below)


def _assemble_chunks(
    sentences: Sequence[str],
    sent_word_counts: Sequence[int],
    sent_sections: Sequence[int],
    breakpoints: Set[int],
    max_words: int,
) -> List[_Segment]:
    """Greedy: walk sentences, cut at breakpoint OR when adding next exceeds max_words.

    Sentences longer than max_words are sliding-window split in place. Each
    emitted segment records whether the boundary that started it was a semantic
    break (``True``) or a size cut (``False``); sliding-split pieces and the
    boundaries around them count as semantic (no overlap).

    Each segment is tagged with the section of its first sentence. Section
    changes are forced breaks (in ``breakpoints``), so a buffer never mixes
    sections and that one tag describes all of the segment's sentences.
    """
    segments: List[_Segment] = []
    buf: List[str] = []
    buf_words = 0
    buf_cut_semantic = False  # boundary type preceding the current buffer
    buf_section = 0
    for i, (sent, sent_toks) in enumerate(zip(sentences, sent_word_counts)):
        if sent_toks > max_words:
            if buf:
                segments.append(_Segment(buf, buf_words, buf_cut_semantic, buf_section))
                buf, buf_words = [], 0
            for piece in _sliding_split(sent, max_words, SLIDING_FALLBACK_OVERLAP):
                segments.append(
                    _Segment([piece], _count_words(piece), True, sent_sections[i])
                )
            buf_cut_semantic = True  # next real chunk must not pad across this
            continue
        is_break = i in breakpoints
        if buf and (is_break or buf_words + sent_toks > max_words):
            segments.append(_Segment(buf, buf_words, buf_cut_semantic, buf_section))
            buf, buf_words = [], 0
            buf_cut_semantic = is_break
        if not buf:
            buf_section = sent_sections[i]
        buf.append(sent)
        buf_words += sent_toks
    if buf:
        segments.append(_Segment(buf, buf_words, buf_cut_semantic, buf_section))
    return segments


def _merge_residuals(
    segments: List[_Segment], min_words: int, max_words: int
) -> List[_Segment]:
    """Merge any segment < min_words into a neighbor (prefer previous).

    Cap is max_words: merging past the safe margin would silently truncate
    at embed time. The merged segment keeps the previous segment's preceding
    boundary type. Unmergeable residuals stay tiny and are rescued by
    title-prepend in _make_chunks.
    """
    if not segments:
        return segments
    out: List[_Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if (
            prev.word_count < min_words
            and prev.word_count + seg.word_count <= max_words
            and prev.section_num == seg.section_num  # keep sections pure
        ):
            prev.sents = prev.sents + seg.sents
            prev.word_count += seg.word_count
        else:
            out.append(seg)
    if (
        len(out) >= 2
        and out[-1].word_count < min_words
        and out[-2].word_count + out[-1].word_count <= max_words
        and out[-2].section_num == out[-1].section_num
    ):
        out[-2].sents = out[-2].sents + out[-1].sents
        out[-2].word_count += out[-1].word_count
        out.pop()
    return out


def _pad_and_join(segments: Sequence[_Segment]) -> List[str]:
    """Join each segment to text, padding with one neighbor sentence per side.

    Left pad = last sentence of the previous segment; right pad = first
    sentence of the next segment. A side is padded only when the boundary it
    crosses was a size cut (``cut_before_semantic`` False), so semantic breaks
    are never bridged. By the example: if chunk 3 split from chunk 2 on a
    semantic break, chunk 2 only borrows from chunk 1 and chunk 3 only from
    chunk 4 — never across the 2|3 boundary.
    """
    bodies: List[str] = []
    n = len(segments)
    for k, seg in enumerate(segments):
        parts = list(seg.sents)
        if k > 0 and not seg.cut_before_semantic and segments[k - 1].sents:
            parts.insert(0, segments[k - 1].sents[-1])
        if k < n - 1 and not segments[k + 1].cut_before_semantic and segments[k + 1].sents:
            parts.append(segments[k + 1].sents[0])
        bodies.append(" ".join(parts))
    return bodies


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


def _format_text(title: str, sec_name: str, body: str, prepend: str) -> str:
    """Build the dense/BM25 chunk text under the chosen prepend policy.

    ``"none"``          -> ``body``
    ``"title"``         -> ``[title] body``
    ``"title_section"`` -> ``[title] [section] body`` (section dropped if absent)
    """
    if prepend == "none":
        return body
    if prepend == "title":
        return f"[{title}] {body}"
    # title_section
    if sec_name:
        return f"[{title}] [{sec_name}] {body}"
    return f"[{title}] {body}"


def _make_chunks(
    bodies: List[str],
    page_id: int,
    title: str,
    sections: Sequence[Tuple[int, str]] | None = None,
    prepend: str = "title_section",
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, body in enumerate(bodies):
        body = body.strip()
        if not body:
            continue
        sec_num, sec_name = sections[i] if sections is not None else (0, "")
        # ``prepend`` controls what is glued onto the embedded text. BM25 always
        # also concatenates ``c.title`` independently (see bm25.build_bm25), so
        # ``"none"`` drops title/section from the DENSE vector only.
        # Metadata (years/months) extracted from body alone.
        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=i,
                text=_format_text(title, sec_name, body, prepend),
                title=title,
                section=sec_name,
                section_number=sec_num,
                years=extract_years(body),
                months=extract_months(body),
            )
        )
    return chunks


def _sliding_para_page_chunks(
    sents: Sequence[str],
    word_counts: Sequence[int],
    forced: Set[int],
    sent_sections: Sequence[int],
    section_names: Dict[int, str],
    page_id: int,
    title: str,
    max_words: int,
    cross_para_overlap: bool = False,
    prepend: str = "title_section",
) -> List[Chunk]:
    """Word-count sliding window with single-sentence overlap.

    Cutting ignores paragraph/semantic boundaries: greedily accumulate whole
    sentences until adding the next would exceed ``max_words``, then cut.

    Overlap padding adds one neighbor sentence per side. With
    ``cross_para_overlap=False`` (paragraph-aware) it never pads across a
    paragraph break (``forced`` holds the indices that begin a paragraph):

    * break *before* the segment start (``start in forced``) -> no left pad
      (append only the next sentence, not the previous one)
    * break *before* the segment end (``end in forced``) -> no right pad
      (append only the previous sentence, not the next one)

    With ``cross_para_overlap=True`` the ``forced`` checks are dropped: overlap
    is applied at every size-cut boundary regardless of paragraph breaks.

    Over-long single sentences are sliding-window split in place and emitted
    unpadded (mirrors the semantic path). Section tag = section of the
    segment's first sentence (a size-cut window may span sections; first wins).
    """
    n = len(sents)
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
        if wc > max_words:
            _flush()
            for piece in _sliding_split(sent, max_words, SLIDING_FALLBACK_OVERLAP):
                raw.append((i, i + 1, [piece], False))  # no overlap for split pieces
            continue
        if buf and buf_words + wc > max_words:
            _flush()
        if not buf:
            buf_start = i
        buf.append(sent)
        buf_words += wc
    _flush()

    bodies: List[str] = []
    sections: List[Tuple[int, str]] = []
    for start, end, slist, padded in raw:
        parts = list(slist)
        if padded:
            # left pad: previous sentence. Paragraph-aware mode skips it when a
            # paragraph break starts the segment (start in forced).
            left_ok = cross_para_overlap or start not in forced
            if start > 0 and left_ok and word_counts[start - 1] <= max_words:
                parts.insert(0, sents[start - 1])
            # right pad: next sentence. Skipped when a paragraph break starts at
            # `end` (paragraph-aware mode only).
            right_ok = cross_para_overlap or end not in forced
            if end < n and right_ok and word_counts[end] <= max_words:
                parts.append(sents[end])
        sec = sent_sections[start] if start < n else 0
        bodies.append(" ".join(parts))
        sections.append((sec, section_names.get(sec, "")))
    return _make_chunks(bodies, page_id, title, sections, prepend)


def chunk_corpus(
    records: List[Dict[str, Any]],
    show_progress: bool = True,
    encode_batch: int = ENCODE_BATCH_SIZE,
    strategy: str = "semantic",
    prepend: str = "title_section",
) -> List[Chunk]:
    """Chunk corpus using ``strategy``.

    ``"semantic"``: sentence-embedding breakpoints (default).
    ``"sliding"``: fixed token-size sliding window, no embeddings — fast, for debugging.
    ``"sliding_para"``: word-count sliding window, paragraph-aware single-sentence
    overlap (never pads across a paragraph break). No embeddings at chunk time.
    ``"sliding_overlap"``: same word-count window but overlap is applied at every
    size-cut boundary, including across paragraph breaks. No embeddings.

    ``prepend`` controls the embedded text: ``"none"`` | ``"title"`` |
    ``"title_section"`` (default). See :func:`_format_text`.
    """
    if strategy == "sliding":
        chunks: List[Chunk] = []
        for record in records:
            content = record.get("content", "") or ""
            page_id = normalize_page_id(record["page_id"])
            title = record.get("title", "")
            chunks.extend(_sliding_window_page_chunks(content, page_id, title))
        return chunks

    if strategy in ("sliding_para", "sliding_overlap"):
        cross = strategy == "sliding_overlap"
        chunks = []
        for record in records:
            content = record.get("content", "") or ""
            page_id = normalize_page_id(record["page_id"])
            title = record.get("title", "")
            sents, forced, sent_sections, section_names = _split_sentences(content)
            if not sents:
                continue
            word_counts = [_count_words(s) for s in sents]
            chunks.extend(
                _sliding_para_page_chunks(
                    sents, word_counts, forced, sent_sections, section_names,
                    page_id, title, MAX_WORDS,
                    cross_para_overlap=cross, prepend=prepend,
                )
            )
        return chunks

    chunks = []
    for batch_start in range(0, len(records), encode_batch):
        batch = records[batch_start : batch_start + encode_batch]
        batch_sentences: List[str] = []
        batch_spans: List[
            Tuple[Dict[str, Any], int, int, Set[int], List[int], Dict[int, str]]
        ] = []
        for record in batch:
            content = record.get("content", "") or ""
            sents, forced, sent_sections, section_names = _split_sentences(content)
            lo = len(batch_sentences)
            batch_sentences.extend(sents)
            batch_spans.append(
                (record, lo, len(batch_sentences), forced, sent_sections, section_names)
            )
        if not batch_sentences:
            continue
        batch_vecs = embed_texts(batch_sentences)
        batch_word_counts = [_count_words(s) for s in batch_sentences]
        for record, lo, hi, forced, sent_sections, section_names in batch_spans:
            sents = batch_sentences[lo:hi]
            if not sents:
                continue
            page_id = normalize_page_id(record["page_id"])
            title = record.get("title", "")
            word_counts = batch_word_counts[lo:hi]
            if len(sents) == 1:
                bodies = _sliding_split(sents[0], MAX_WORDS, SLIDING_FALLBACK_OVERLAP)
                # single sentence -> whole page is one section (its first body sent)
                sec = sent_sections[0] if sent_sections else 0
                sections = [(sec, section_names.get(sec, "")) for _ in bodies]
            else:
                breaks = _semantic_breakpoints(
                    batch_vecs[lo:hi], forced, BREAKPOINT_PCTL
                )
                segments = _assemble_chunks(
                    sents, word_counts, sent_sections, breaks, MAX_WORDS
                )
                segments = _merge_residuals(segments, MIN_WORDS, MAX_WORDS)
                bodies = _pad_and_join(segments)
                sections = [
                    (seg.section_num, section_names.get(seg.section_num, ""))
                    for seg in segments
                ]
            chunks.extend(_make_chunks(bodies, page_id, title, sections, prepend))
    return chunks

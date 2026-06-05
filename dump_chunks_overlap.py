"""Dump one page's chunks to a .txt, marking overlap (padded) sentences.

Replicates chunk.py's semantic pipeline but, instead of silently joining the
borrowed neighbor sentences, wraps them so overlaps are visible:

    <<PREV| ...borrowed last sentence of previous segment... |PREV>>
    ...core sentences of this chunk...
    <<NEXT| ...borrowed first sentence of next segment... |NEXT>>

Run: part-2/.venv/bin/python dump_chunks_overlap.py 39477
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence

from chunk import (
    MAX_WORDS,
    MIN_WORDS,
    BREAKPOINT_PCTL,
    SLIDING_FALLBACK_OVERLAP,
    _Segment,
    _assemble_chunks,
    _count_words,
    _merge_residuals,
    _semantic_breakpoints,
    _sliding_split,
    _split_sentences,
)
from embed import embed_texts
from utils import ENTRIES_DIR, normalize_page_id


def _pad_and_join_marked(segments: Sequence[_Segment]) -> List[str]:
    """Same logic as chunk._pad_and_join but tags borrowed neighbor sentences."""
    bodies: List[str] = []
    n = len(segments)
    for k, seg in enumerate(segments):
        parts: List[str] = []
        if k > 0 and not seg.cut_before_semantic and segments[k - 1].sents:
            parts.append(f"<<PREV| {segments[k - 1].sents[-1]} |PREV>>")
        parts.extend(seg.sents)
        if k < n - 1 and not segments[k + 1].cut_before_semantic and segments[k + 1].sents:
            parts.append(f"<<NEXT| {segments[k + 1].sents[0]} |NEXT>>")
        bodies.append(" ".join(parts))
    return bodies


def main() -> None:
    page_id_arg = sys.argv[1] if len(sys.argv) > 1 else "39477"
    path = ENTRIES_DIR / f"{page_id_arg}.json"
    record = json.loads(path.read_text())
    page_id = normalize_page_id(record.get("page_id", path.stem))
    title = record.get("title", "")
    content = record.get("content", "") or ""

    sents, forced = _split_sentences(content)
    if not sents:
        print("no sentences")
        return

    if len(sents) == 1:
        bodies = _sliding_split(sents[0], MAX_WORDS, SLIDING_FALLBACK_OVERLAP)
    else:
        vecs = embed_texts(sents)
        word_counts = [_count_words(s) for s in sents]
        breaks = _semantic_breakpoints(vecs, forced, BREAKPOINT_PCTL)
        segments = _assemble_chunks(sents, word_counts, breaks, MAX_WORDS)
        segments = _merge_residuals(segments, MIN_WORDS, MAX_WORDS)
        bodies = _pad_and_join_marked(segments)

    out_path = Path(__file__).parent / f"chunks_{page_id}_overlap.txt"
    lines: List[str] = []
    lines.append(
        f"page_id={page_id}  title={title}  n_chunks={len(bodies)}  "
        f"(MAX_WORDS={MAX_WORDS} MIN_WORDS={MIN_WORDS} PCTL={BREAKPOINT_PCTL})"
    )
    lines.append(
        "overlap markers: <<PREV|...|PREV>> = borrowed tail of previous chunk, "
        "<<NEXT|...|NEXT>> = borrowed head of next chunk"
    )
    lines.append("")
    for i, body in enumerate(bodies):
        dense = f"[{title}] {body}"
        lines.append(f"### chunk {i}  words={_count_words(dense)}")
        lines.append(dense)
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}  ({len(bodies)} chunks)")


if __name__ == "__main__":
    main()

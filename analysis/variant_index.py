"""Parametrized dense-index builder for measurement (throwaway indexes).

Builds a dense-only index (vectors + meta; no BM25) into ``indexes/<tag>/``,
reusing production chunking + embedding. Two variant axes:

  (a) MAX_WORDS   — set chunk.MAX_WORDS in-process, re-chunk, embed ``[title] body``.
  (b) text-mode   — chunk once, rewrite each chunk's embedded text:
                      vanilla       -> body
                      title         -> [title] body          (production)
                      title_header  -> [title] [section] body (empty section -> [title] body)

Idempotent: skips if ``indexes/<tag>/index_vectors.npy`` exists unless --force.
Promoting a winner to production is the separate index-affecting rebuild
(scripts/build_index.py + re-commit LFS) — this only builds for measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common import INDEXES_DIR, bootstrap

bootstrap()

import chunk as chunk_mod  # noqa: E402
from chunk import Chunk, chunk_corpus  # noqa: E402
from embed import embed_texts  # noqa: E402
from index import INDEX_META_NAME, INDEX_VECTORS_NAME, _select_subset  # noqa: E402
from utils import iter_entries  # noqa: E402

TEXT_MODES = ("vanilla", "title", "title_header")


def _strip_title(text: str, title: str) -> str:
    """Recover the body from a ``[title] body`` chunk text."""
    prefix = f"[{title}] "
    return text[len(prefix):] if text.startswith(prefix) else text


def _chunk_text(c: Chunk, mode: str) -> str:
    body = _strip_title(c.text, c.title)
    if mode == "vanilla":
        return body
    if mode == "title":
        return f"[{c.title}] {body}"
    if mode == "title_header":
        if c.section:
            return f"[{c.title}] [{c.section}] {body}"
        return f"[{c.title}] {body}"
    raise ValueError(f"unknown text mode: {mode}")


def _records(subset: Optional[int]) -> List[Dict[str, Any]]:
    records = list(iter_entries())
    if subset is not None:
        records = _select_subset(records, subset)
    return records


def _save(out_dir: Path, vectors: np.ndarray, chunks: List[Chunk], extra: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta: Dict[str, Any] = {
        "page_ids": [c.page_id for c in chunks],
        "chunk_ids": [c.chunk_id for c in chunks],
        "page_titles": [c.title for c in chunks],
        "chunk_sections": [c.section for c in chunks],
        "chunk_section_numbers": [c.section_number for c in chunks],
        "chunk_word_counts": [len(c.text.split()) for c in chunks],
        "chunk_numbers": [sorted(c.numbers) for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(chunks),
    }
    meta.update(extra)
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_maxwords(mw: int, *, subset: Optional[int], force: bool) -> Path:
    """Index with MAX_WORDS=mw, text mode = title ([title] body)."""
    tag = f"mw{mw}"
    out_dir = INDEXES_DIR / tag
    if (out_dir / INDEX_VECTORS_NAME).exists() and not force:
        print(f"[variant] skip {tag} (exists)")
        return out_dir
    prev = chunk_mod.MAX_WORDS
    chunk_mod.MAX_WORDS = mw
    try:
        chunks = chunk_corpus(_records(subset))
    finally:
        chunk_mod.MAX_WORDS = prev
    print(f"[variant] {tag}: {len(chunks)} chunks; embedding ...")
    vectors = embed_texts([c.text for c in chunks], show_progress=True)
    _save(out_dir, vectors, chunks, {"max_words": mw, "text_mode": "title", "chunk_count": len(chunks)})
    print(f"[variant] saved {tag} ({len(chunks)} vecs)")
    return out_dir


def build_textmode(
    mode: str, *, mw: Optional[int] = None, subset: Optional[int], force: bool
) -> Path:
    """Index with a text mode (optionally at a given MAX_WORDS)."""
    tag = f"mode_{mode}" if mw is None else f"mw{mw}_{mode}"
    out_dir = INDEXES_DIR / tag
    if (out_dir / INDEX_VECTORS_NAME).exists() and not force:
        print(f"[variant] skip {tag} (exists)")
        return out_dir
    prev = chunk_mod.MAX_WORDS
    if mw is not None:
        chunk_mod.MAX_WORDS = mw
    try:
        chunks = chunk_corpus(_records(subset))
    finally:
        chunk_mod.MAX_WORDS = prev
    texts = [_chunk_text(c, mode) for c in chunks]
    print(f"[variant] {tag}: {len(chunks)} chunks; embedding ...")
    vectors = embed_texts(texts, show_progress=True)
    _save(
        out_dir,
        vectors,
        chunks,
        {"max_words": mw or prev, "text_mode": mode, "chunk_count": len(chunks)},
    )
    print(f"[variant] saved {tag} ({len(chunks)} vecs)")
    return out_dir


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maxwords", type=int, help="build MAX_WORDS variant")
    ap.add_argument("--mode", choices=TEXT_MODES, help="build text-mode variant")
    ap.add_argument("--at-maxwords", type=int, default=None, help="MAX_WORDS for --mode")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--subset", type=int, help="GT-inclusive subset size")
    scope.add_argument("--full", action="store_true", help="full 27k corpus")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = _parse()
    subset = None if a.full else a.subset
    if a.maxwords is not None:
        build_maxwords(a.maxwords, subset=subset, force=a.force)
    if a.mode is not None:
        build_textmode(a.mode, mw=a.at_maxwords, subset=subset, force=a.force)
    if a.maxwords is None and a.mode is None:
        raise SystemExit("specify --maxwords and/or --mode")


if __name__ == "__main__":
    main()

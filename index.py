"""Offline index build and load (not timed at grading)."""

from __future__ import annotations

import json
from chunk import Chunk, chunk_corpus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bm25 import build_bm25
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
FAISS_INDEX_NAME = "faiss.index"


def _select_subset(records: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """
    Pick an n-page subset for fast local dev, always keeping the public queries'
    ground-truth pages. Without this, none of the relevant pages get indexed and
    local NDCG is structurally 0 — the corpus is 27k pages, GT is 100 of them.
    """
    from utils import load_public_queries

    gt: set = set()
    for row in load_public_queries():
        gt.update(row["relevant_page_ids"])

    gt_recs = [r for r in records if r["page_id"] in gt]
    other = [r for r in records if r["page_id"] not in gt]

    if n <= len(gt_recs):
        print(
            f"[index] WARN: subset {n} <= {len(gt_recs)} ground-truth pages; "
            "keeping GT pages only — some queries still unscorable below 100."
        )
        return gt_recs[:n]

    fill = other[: n - len(gt_recs)]
    print(
        f"[index] subset: {len(gt_recs)} GT + {len(fill)} filler = {len(gt_recs) + len(fill)} pages"
    )
    return gt_recs + fill


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    subset: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Embed the full corpus and persist artifacts.

    subset: if set, index only N pages (GT-inclusive) for fast local iteration.
            Leave None for the real submission build.
    Returns (vectors, page_ids) where row i corresponds to page_ids[i].
    Chunk-level vectors; aggregation to page happens in retrieve.py.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    print(f"[index] loaded {len(records)} records")
    if subset is not None:
        records = _select_subset(records, subset)
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    print(f"[index] embedding {len(texts)} chunk texts...")
    vectors = embed_texts(texts, show_progress=True)
    print(f"[index] saving {len(vectors)} vectors to {out_dir}")
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "page_titles": [c.title for c in chunks],
        "chunk_sections": [c.section for c in chunks],
        "chunk_section_numbers": [c.section_number for c in chunks],
        "chunk_word_counts": [len(c.text.split()) for c in chunks],
        # Per-chunk extracted numbers (sorted int lists).
        # Powers query-time number pre/post-filtering in retrieve.py.
        "chunk_numbers": [sorted(c.numbers) for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    build_bm25(chunks, out_dir)
    _build_faiss(vectors, out_dir)

    return vectors, page_ids


def _build_faiss(vectors: np.ndarray, out_dir: Path) -> None:
    """Build and save FAISS IndexFlatIP (exact cosine for L2-normalised vectors)."""
    try:
        import faiss  # type: ignore

        dim = vectors.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(vectors)
        faiss.write_index(idx, str(out_dir / FAISS_INDEX_NAME))
        print(f"[index] FAISS index saved → {out_dir / FAISS_INDEX_NAME}")
    except Exception as exc:
        print(f"[index] FAISS build skipped: {exc}")


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed vectors and page_id map from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids


def load_meta(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load full index meta dict (for BM25, reranking, diagnostics)."""
    root = artifacts_dir or ARTIFACTS_DIR
    return json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))

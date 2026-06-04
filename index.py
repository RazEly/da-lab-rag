"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bm25 import build_bm25
from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
FAISS_INDEX_NAME = "faiss.index"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    chunking_strategy: str = "semantic",
) -> Tuple[np.ndarray, List[int]]:
    """
    Embed the full corpus and persist artifacts.

    chunking_strategy: ``"semantic"`` (default) or ``"sliding"`` (fixed token window).
    Returns (vectors, page_ids) where row i corresponds to page_ids[i].
    Chunk-level vectors; aggregation to page happens in retrieve.py.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    print(f"[index] loaded {len(records)} records")
    chunks: List[Chunk] = chunk_corpus(records, show_progress=True, strategy=chunking_strategy)
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
        "chunk_word_counts": [len(c.text.split()) for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

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

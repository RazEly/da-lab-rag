"""Query-time retrieval (timed portion includes query embedding)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from bm25 import load_bm25
from bm25 import score_batch as bm25_score_batch
from embed import embed_queries
from index import FAISS_INDEX_NAME, load_index
from utils import ARTIFACTS_DIR, K_EVAL

# S6: max+mean blend alpha over chunk scores per page. 1.0=pure max, 0.0=pure mean.
ALPHA = 0.7

# S2: BM25 hybrid weight (BM25 share of final score). Tuned on public set: 0.75 optimal.
BM25_WEIGHT = 0.00

# Normalization candidate pool: min/max computed over top-K only to avoid
# outlier compression from non-retrieved documents setting the range.
TOPK_NORM = 1000


def _load_faiss(artifacts_dir: Path):
    """Load FAISS index from disk; return None if unavailable."""
    try:
        import faiss  # type: ignore

        idx_path = artifacts_dir / FAISS_INDEX_NAME
        if not idx_path.exists():
            return None
        idx = faiss.read_index(str(idx_path))
        try:
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                idx = faiss.index_cpu_to_gpu(res, 0, idx)
        except Exception:
            pass  # faiss-cpu has no GPU bindings
        return idx
    except Exception:
        return None


def _minmax_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Per-row min-max normalisation using only the top-k values to define [lo, hi].

    Avoids outlier compression: a single very-high-scoring document would
    collapse all others toward 0 if we used global min/max.
    Rows where all top-k values are equal (no BM25 signal) stay at 0.
    """
    k = min(k, scores.shape[1])
    out = np.zeros_like(scores)
    for i, row in enumerate(scores):
        topk_idx = np.argpartition(row, -k)[-k:]
        lo = float(row[topk_idx].min())
        hi = float(row[topk_idx].max())
        if hi - lo < 1e-9:
            continue
        out[i] = np.clip((row - lo) / (hi - lo), 0.0, 1.0)
    return out


def _aggregate_page_scores(
    chunk_scores: np.ndarray,
    page_ids: List[int],
    top_k: int,
    alpha: float,
) -> List[int]:
    """Max+mean blend aggregation over chunk scores → ranked page_id list."""
    page_chunks: Dict[int, List[float]] = {}
    for idx, score in enumerate(chunk_scores):
        pid = page_ids[idx]
        page_chunks.setdefault(pid, []).append(float(score))

    page_scores: Dict[int, float] = {}
    for pid, chunk_list in page_chunks.items():
        top3 = sorted(chunk_list, reverse=True)[:3]
        page_scores[pid] = alpha * top3[0] + (1.0 - alpha) * (sum(top3) / len(top3))

    ranked = sorted(page_scores, key=page_scores.__getitem__, reverse=True)
    return ranked[:top_k]


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    alpha: float = ALPHA,
    bm25_weight: float = BM25_WEIGHT,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query."""
    root = artifacts_dir or ARTIFACTS_DIR
    corpus_vectors, page_ids = load_index(root)
    bm25 = load_bm25(root)
    query_vectors = embed_queries(queries)

    if query_vectors.size == 0:
        return [[] for _ in queries]

    num_chunks = corpus_vectors.shape[0]

    # Dense scores: FAISS if available (GPU-accelerated), else exact numpy matmul.
    faiss_idx = _load_faiss(root)
    if faiss_idx is not None:
        D, I = faiss_idx.search(query_vectors, num_chunks)
        dense_scores = np.zeros((len(queries), num_chunks), dtype=np.float32)
        for qi in range(len(queries)):
            dense_scores[qi, I[qi]] = D[qi]
    else:
        dense_scores = query_vectors @ corpus_vectors.T  # (Q, C)

    # BM25 sparse scores: (Q, C) accumulated from inverted index.
    sparse_scores = bm25_score_batch(bm25, queries)  # (Q, C)

    # Normalise each retriever over its own top-TOPK_NORM candidates to avoid
    # a single high-scoring outlier compressing the rest of the range.
    dense_norm = _minmax_topk(dense_scores, TOPK_NORM)
    sparse_norm = _minmax_topk(sparse_scores, TOPK_NORM)

    # Convex combination: tuned on public set (sweep BM25_WEIGHT).
    final = (1.0 - bm25_weight) * dense_norm + bm25_weight * sparse_norm

    return [_aggregate_page_scores(row, page_ids, top_k, alpha) for row in final]

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
# Tuned on public subset: mean-heavy over many chunks rewards pages consistently
# on-topic (long GT articles) instead of one lucky chunk. Lifted dense-only
# 0.3555 -> 0.4213, fused -> 0.4351. Old (0.7, top3) hobbled both retrievers.
ALPHA = 0.2

# Top chunks per page averaged in the mean term (max term always uses #1).
# ~30 ≈ "all chunks" on the tuning plateau; capped for speed.
AGG_TOPN = 30

# Fusion mode: "rrf" (rank fusion) | "blend" (legacy convex score blend).
# Score-blend monotonically degraded to BM25-only because min-max stretches MiniLM's
# tightly-bunched cosines into noise on BM25's scale. RRF fuses by rank position only,
# so dense's complementary recall returns without polluting BM25's order.
FUSION = "rrf"

# RRF params (S3). Each retriever contributes its top-RRF_DEPTH page ranking;
# fused score = sum 1/(RRF_K + rank). Toggle either retriever off to A/B.
RRF_K = 60
RRF_DEPTH = 200
INCLUDE_DENSE = True
INCLUDE_SPARSE = True

# --- legacy "blend" mode only ---
# S2: BM25 hybrid weight (BM25 share of final score).
BM25_WEIGHT = 0.5
# Normalization candidate pool: min/max computed over top-K only to avoid
# outlier compression from non-retrieved documents setting the range.
TOPK_NORM = 1000


def _load_faiss(artifacts_dir: Path, corpus_vectors: np.ndarray):
    """
    Return a FAISS index, or None if faiss is unavailable.

    faiss.index is NOT shipped in the repo (it's ~400MB and fully reconstructible).
    If the file is absent we rebuild IndexFlatIP from corpus_vectors in-memory —
    .add() of L2-normalised vectors is sub-second and exact, well within the 60s wall.
    """
    try:
        import faiss  # type: ignore

        idx_path = artifacts_dir / FAISS_INDEX_NAME
        if idx_path.exists():
            idx = faiss.read_index(str(idx_path))
        else:
            idx = faiss.IndexFlatIP(corpus_vectors.shape[1])
            idx.add(corpus_vectors)
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
        ranked_chunks = sorted(chunk_list, reverse=True)
        topn = ranked_chunks[:AGG_TOPN]
        page_scores[pid] = alpha * ranked_chunks[0] + (1.0 - alpha) * (sum(topn) / len(topn))

    ranked = sorted(page_scores, key=page_scores.__getitem__, reverse=True)
    return ranked[:top_k]


def _rrf(rankings: List[List[int]], k: int, top_k: int) -> List[int]:
    """Reciprocal Rank Fusion over per-retriever page rankings."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]


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
    faiss_idx = _load_faiss(root, corpus_vectors)
    if faiss_idx is not None:
        D, I = faiss_idx.search(query_vectors, num_chunks)
        dense_scores = np.zeros((len(queries), num_chunks), dtype=np.float32)
        for qi in range(len(queries)):
            dense_scores[qi, I[qi]] = D[qi]
    else:
        dense_scores = query_vectors @ corpus_vectors.T  # (Q, C)

    # BM25 sparse scores: (Q, C) accumulated from inverted index.
    sparse_scores = bm25_score_batch(bm25, queries)  # (Q, C)

    if FUSION == "rrf":
        # Rank each retriever independently to page level, then fuse by rank.
        # Immune to dense/BM25 score-scale mismatch (the cause of blend collapse).
        results: List[List[int]] = []
        for qi in range(len(queries)):
            rankings: List[List[int]] = []
            if INCLUDE_DENSE:
                rankings.append(
                    _aggregate_page_scores(dense_scores[qi], page_ids, RRF_DEPTH, alpha)
                )
            if INCLUDE_SPARSE:
                rankings.append(
                    _aggregate_page_scores(
                        sparse_scores[qi], page_ids, RRF_DEPTH, alpha
                    )
                )
            results.append(_rrf(rankings, RRF_K, top_k))
        return results

    # Legacy "blend": normalise each retriever over its own top-TOPK_NORM
    # candidates (avoids one outlier compressing the range), then convex-combine.
    dense_norm = _minmax_topk(dense_scores, TOPK_NORM)
    sparse_norm = _minmax_topk(sparse_scores, TOPK_NORM)
    final = (1.0 - bm25_weight) * dense_norm + bm25_weight * sparse_norm

    return [_aggregate_page_scores(row, page_ids, top_k, alpha) for row in final]

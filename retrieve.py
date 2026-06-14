"""Query-time retrieval (timed portion includes query embedding)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from bm25 import load_bm25
from bm25 import score_batch as bm25_score_batch
from embed import embed_queries
from index import FAISS_INDEX_NAME, load_index, load_meta
from utils import ARTIFACTS_DIR, K_EVAL

# Number pre-filter: if a query names a number, restrict the candidate pool to
# pages having >=1 chunk whose extracted metadata contains that number.
# Toggle off to A/B. Empty result after filtering -> fall back to unfiltered
# (an over-eager filter that wipes the pool is worse than no filter).
ENABLE_NUMBER_FILTER = True
# "pre"  : restrict the candidate pool before ranking/RRF (non-matching pages
#          never compete). "post": rank the full corpus, then drop non-matching
#          pages from the final list. Pre is strictly stronger when the number
#          signal is trustworthy; post is safer when metadata recall is partial.
FILTER_MODE = "post"
_QUERY_NUMBER_RE = re.compile(r"\b\d+\b")

# S6: max+mean blend per page. alpha 1.0=pure max, 0.0=pure mean.
# CRITICAL (2026-06-05, full 27k): the two retrievers want OPPOSITE aggregation.
#   Dense  : mean-heavy — gold is a long article whose chunks are uniformly
#            on-topic; max alone picks one lucky chunk and buries it.
#   BM25   : max-heavy — exact-match signal lives in ONE chunk; averaging dilutes
#            the rank-1 hit.
# Forcing a single alpha on both (old behaviour) starved BM25, the stronger leg.
# Per-retriever agg + RRF lifted full-corpus NDCG@10 0.155 -> 0.275.
#
# Fused-optimal agg (2026-06-14 sweep with adaptive IDF-filtered BM25).
#   Dense : pure mean (alpha=0.0), topn=100.
#   BM25  : near-mean (alpha=0.25), topn=30 — IDF filtering concentrates signal in
#           fewer high-quality chunks; a wider topn captures multi-chunk GT pages.
# BM25_QUERY_MIN_IDF=4.0 filters generic query terms (Robertson IDF < 4.0), keeping
# only discriminative terms. BM25_FALLBACK_THRESHOLD=15.0: if the max filtered score
# for a query is below this value, rescore unfiltered (recovers broad/generic queries).
# NDCG@10 0.5433 (up from 0.5343 at idf-only, 0.5215 baseline).
DENSE_ALPHA, DENSE_TOPN = 0.0, 100
SPARSE_ALPHA, SPARSE_TOPN = 0.25, 30

# BM25 query-time IDF threshold: skip query terms with Robertson IDF below this.
# Low-IDF terms (e.g. "founded", "company") match many pages indiscriminately;
# filtering them sharpens BM25 precision without sacrificing high-IDF recall.
# Optimal range 3.7-4.1 (plateau). 4.0 chosen as conservative centre.
BM25_QUERY_MIN_IDF: float = 4.0
# If IDF-filtered max chunk score < this threshold, fall back to unfiltered BM25.
# Triggers only for queries where filtering removes all discriminative signal.
# Threshold=15 targets exactly one such query type in the public set (broad city queries).
BM25_FALLBACK_THRESHOLD: float = 15.0

# RRF params: asymmetric depths (2026-06-13).
# Dense=100 preserves semantic breadth; BM25=30 is tight enough that only
# strong IDF-filtered matches compete (reduces noise from weak BM25 hits).
RRF_K = 60  # was 28
RRF_DEPTH = 100  # dense depth
SPARSE_RRF_DEPTH = (
    60  # was 25  # BM25 depth (tighter with IDF-filtered, high-quality top-25)
)
INCLUDE_DENSE = True
INCLUDE_SPARSE = True


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


def _extract_query_numbers(query: str) -> Set[int]:
    """All integer numbers mentioned in a query (matches stored metadata)."""
    return {int(m) for m in _QUERY_NUMBER_RE.findall(query)}


def _build_page_numbers(
    page_ids: List[int], chunk_numbers: List[List[int]]
) -> Dict[int, Set[int]]:
    """page_id → union of numbers across all its chunks (row-aligned inputs)."""
    page_numbers: Dict[int, Set[int]] = {}
    for pid, nums in zip(page_ids, chunk_numbers):
        if nums:
            page_numbers.setdefault(pid, set()).update(nums)
    return page_numbers


def _allowed_pages_for_query(
    query: str, page_numbers: Dict[int, Set[int]]
) -> Optional[Set[int]]:
    """
    Pages allowed for `query` under the number filter, or None for no filtering.

    None means "don't filter" — either the query names no number, or filtering
    would empty the pool (no page carries the queried number); both fall back to
    the full corpus rather than returning nothing.
    """
    nums = _extract_query_numbers(query)
    if not nums:
        return None
    allowed = {pid for pid, pn in page_numbers.items() if pn & nums}
    return allowed or None


def _aggregate_page_scores(
    chunk_scores: np.ndarray,
    page_ids: List[int],
    top_k: int,
    alpha: float,
    agg_topn: int = 30,
    allowed_pages: Optional[Set[int]] = None,
) -> List[int]:
    """Max+mean blend aggregation over chunk scores → ranked page_id list.

    allowed_pages: if set, chunks on pages outside this set are dropped before
    aggregation (number pre-filter). None = no filtering.
    """
    page_chunks: Dict[int, List[float]] = {}
    for idx, score in enumerate(chunk_scores):
        pid = page_ids[idx]
        if allowed_pages is not None and pid not in allowed_pages:
            continue
        page_chunks.setdefault(pid, []).append(float(score))

    page_scores: Dict[int, float] = {}
    for pid, chunk_list in page_chunks.items():
        ranked_chunks = sorted(chunk_list, reverse=True)
        topn = ranked_chunks[:agg_topn]
        page_scores[pid] = alpha * ranked_chunks[0] + (1.0 - alpha) * (
            sum(topn) / len(topn)
        )

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
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query."""
    root = artifacts_dir or ARTIFACTS_DIR
    corpus_vectors, page_ids = load_index(root)
    bm25 = load_bm25(root)
    query_vectors = embed_queries(queries)

    # Number filter: build page→numbers map once. Disabled if artifacts predate
    # the chunk_numbers field (stale build) so old indexes keep working unchanged.
    page_numbers: Optional[Dict[int, Set[int]]] = None
    if ENABLE_NUMBER_FILTER:
        chunk_numbers = load_meta(root).get("chunk_numbers")
        if chunk_numbers is not None:
            page_numbers = _build_page_numbers(page_ids, chunk_numbers)

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
    # min_query_idf drops low-discriminativity query terms (see BM25_QUERY_MIN_IDF).
    sparse_scores = bm25_score_batch(
        bm25,
        queries,
        min_query_idf=BM25_QUERY_MIN_IDF,
        fallback_threshold=BM25_FALLBACK_THRESHOLD,
    )  # (Q, C)

    # Rank each retriever independently to page level, then fuse by rank (RRF).
    # Immune to dense/BM25 score-scale mismatch (the cause of blend collapse).
    results: List[List[int]] = []
    for qi in range(len(queries)):
        allowed = (
            _allowed_pages_for_query(queries[qi], page_numbers)
            if page_numbers is not None
            else None
        )
        # pre-filter applies inside per-retriever aggregation; post-filter
        # leaves ranking untouched and prunes the fused list afterwards.
        pre = allowed if FILTER_MODE == "pre" else None
        rankings: List[List[int]] = []
        if INCLUDE_DENSE:
            rankings.append(
                _aggregate_page_scores(
                    dense_scores[qi],
                    page_ids,
                    RRF_DEPTH,
                    DENSE_ALPHA,
                    DENSE_TOPN,
                    allowed_pages=pre,
                )
            )
        if INCLUDE_SPARSE:
            rankings.append(
                _aggregate_page_scores(
                    sparse_scores[qi],
                    page_ids,
                    SPARSE_RRF_DEPTH,
                    SPARSE_ALPHA,
                    SPARSE_TOPN,
                    allowed_pages=pre,
                )
            )
        if FILTER_MODE == "post" and allowed is not None:
            # Fuse the full pool, then keep only number-matching pages.
            # Fall back to the unfiltered order if the filter empties it.
            fused = _rrf(rankings, RRF_K, len(page_ids))
            kept = [p for p in fused if p in allowed]
            results.append((kept or fused)[:top_k])
        else:
            results.append(_rrf(rankings, RRF_K, top_k))
    return results

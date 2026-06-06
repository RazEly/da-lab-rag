"""Query-time retrieval strategy sweep on the 50 public queries.

Loads the COMMITTED artifacts once, computes dense + BM25 score matrices once,
then sweeps fusion/aggregation params over the cached matrices. No index rebuild.

Query-text variants (lowercase / strip question words) re-embed + re-tokenize the
50 queries only (cheap). Everything else reuses the fixed score matrices.

    python sweep_retrieval.py
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from bm25 import load_bm25, score_batch as bm25_score_batch, tokenize
from embed import embed_queries
from eval import load_query_file, mean_ndcg_at_k
from index import load_index, load_meta
from retrieve import (
    _aggregate_page_scores,
    _build_page_numbers,
    _extract_query_numbers,
    _load_faiss,
    _rrf,
)
from utils import PUBLIC_QUERIES_PATH

TOP_K = 10
RRF_DEPTH = 200

_QUESTION_WORDS = {"who", "what", "which", "when", "where", "why", "how", "whose", "whom"}


def _allowed(query: str, page_numbers: Optional[Dict[int, Set[int]]]) -> Optional[Set[int]]:
    if page_numbers is None:
        return None
    nums = _extract_query_numbers(query)
    if not nums:
        return None
    allowed = {pid for pid, pn in page_numbers.items() if pn & nums}
    return allowed or None


def weighted_rrf(rankings_w: List[Tuple[List[int], float]], k: int, top_k: int) -> List[int]:
    """RRF with a per-ranking weight on each 1/(k+rank) contribution."""
    scores: Dict[int, float] = {}
    for ranking, w in rankings_w:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]


def fuse(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    page_ids: List[int],
    queries: List[str],
    page_numbers: Optional[Dict[int, Set[int]]],
    *,
    dense_alpha: float,
    dense_topn: int,
    sparse_alpha: float,
    sparse_topn: int,
    rrf_k: int,
    dense_w: float = 1.0,
    sparse_w: float = 1.0,
    include_dense: bool = True,
    include_sparse: bool = True,
    filter_mode: str = "post",  # "off" | "pre" | "post"
) -> List[List[int]]:
    results: List[List[int]] = []
    for qi in range(len(queries)):
        allowed = _allowed(queries[qi], page_numbers) if filter_mode != "off" else None
        pre = allowed if filter_mode == "pre" else None
        rankings_w: List[Tuple[List[int], float]] = []
        if include_dense:
            rankings_w.append((
                _aggregate_page_scores(
                    dense_scores[qi], page_ids, RRF_DEPTH, dense_alpha, dense_topn,
                    allowed_pages=pre,
                ), dense_w,
            ))
        if include_sparse:
            rankings_w.append((
                _aggregate_page_scores(
                    sparse_scores[qi], page_ids, RRF_DEPTH, sparse_alpha, sparse_topn,
                    allowed_pages=pre,
                ), sparse_w,
            ))
        if filter_mode == "post" and allowed is not None:
            fused = weighted_rrf(rankings_w, rrf_k, len(page_ids))
            kept = [p for p in fused if p in allowed]
            results.append((kept or fused)[:TOP_K])
        else:
            results.append(weighted_rrf(rankings_w, rrf_k, TOP_K))
    return results


def main() -> None:
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    print("loading artifacts (once) ...")
    t0 = time.perf_counter()
    corpus_vectors, page_ids = load_index()
    bm25 = load_bm25()
    meta = load_meta()
    chunk_numbers = meta.get("chunk_numbers")
    page_numbers = (
        _build_page_numbers(page_ids, chunk_numbers) if chunk_numbers is not None else None
    )
    num_chunks = corpus_vectors.shape[0]
    print(f"  loaded {num_chunks} chunks in {time.perf_counter()-t0:.1f}s")

    def dense_for(qtexts: List[str]) -> np.ndarray:
        qv = embed_queries(qtexts)
        faiss_idx = _load_faiss(None, corpus_vectors)
        if faiss_idx is not None:
            D, I = faiss_idx.search(qv, num_chunks)
            ds = np.zeros((len(qtexts), num_chunks), dtype=np.float32)
            for qi in range(len(qtexts)):
                ds[qi, I[qi]] = D[qi]
            return ds
        return qv @ corpus_vectors.T

    print("computing base dense + sparse score matrices (once) ...")
    t0 = time.perf_counter()
    dense_scores = dense_for(queries)
    sparse_scores = bm25_score_batch(bm25, queries)
    print(f"  scored in {time.perf_counter()-t0:.1f}s")

    results: List[Tuple[str, float]] = []

    def run(label: str, **kw) -> None:
        ranked = fuse(dense_scores, sparse_scores, page_ids, queries, page_numbers, **kw)
        ndcg = mean_ndcg_at_k(ranked, gt, k=10)
        results.append((label, ndcg))
        print(f"  {ndcg:.4f}  {label}")

    BASE = dict(
        dense_alpha=0.0, dense_topn=50, sparse_alpha=0.8, sparse_topn=3,
        rrf_k=60, filter_mode="post",
    )

    print("\n=== A. current baseline ===")
    run("baseline (D0.0/50 S0.8/3 k60 post)", **BASE)

    print("\n=== B. solo retrievers ===")
    run("dense-only", **{**BASE, "include_sparse": False})
    run("bm25-only", **{**BASE, "include_dense": False})

    print("\n=== C. number filter mode ===")
    for fm in ("off", "pre", "post"):
        run(f"filter={fm}", **{**BASE, "filter_mode": fm})

    print("\n=== D. dense aggregation (alpha/topn) ===")
    for da in (0.0, 0.2, 0.4, 0.7):
        for dt in (30, 50, 100):
            run(f"dense_alpha={da} topn={dt}", **{**BASE, "dense_alpha": da, "dense_topn": dt})

    print("\n=== E. sparse aggregation (alpha/topn) ===")
    for sa in (0.5, 0.8, 1.0):
        for st in (1, 3, 5):
            run(f"sparse_alpha={sa} topn={st}", **{**BASE, "sparse_alpha": sa, "sparse_topn": st})

    print("\n=== F. RRF k ===")
    for k in (10, 30, 60, 100, 200):
        run(f"rrf_k={k}", **{**BASE, "rrf_k": k})

    print("\n=== G. weighted RRF (dense_w:sparse_w) ===")
    for dw, sw in [(2.0, 1.0), (1.5, 1.0), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0), (1.0, 3.0)]:
        run(f"dense_w={dw} sparse_w={sw}", **{**BASE, "dense_w": dw, "sparse_w": sw})

    # --- query-text variants: re-embed / re-tokenize the 50 queries ---
    print("\n=== H. query-text variants (re-embed) ===")

    def lower(q: str) -> str:
        return q.lower()

    def strip_punct(q: str) -> str:
        return re.sub(r"[^\w\s]", " ", q.lower())

    def strip_qwords(q: str) -> str:
        toks = q.split()
        while toks and toks[0].lower().strip("?,") in _QUESTION_WORDS:
            toks = toks[1:]
        return " ".join(toks)

    for vname, fn in [("lowercase", lower), ("strip_punct", strip_punct), ("strip_qwords", strip_qwords)]:
        qv = [fn(q) for q in queries]
        ds = dense_for(qv)
        ss = bm25_score_batch(bm25, qv)
        ranked = fuse(ds, ss, page_ids, qv, page_numbers, **BASE)
        ndcg = mean_ndcg_at_k(ranked, gt, k=10)
        results.append((f"qvariant={vname}", ndcg))
        print(f"  {ndcg:.4f}  qvariant={vname}")

    # --- I. combine independent winners ---
    print("\n=== I. combined winners ===")
    COMBO = {**BASE, "sparse_alpha": 0.5, "sparse_topn": 5, "dense_topn": 100}
    run("combo (S0.5/5 D0.0/100 k60 post)", **COMBO)
    run("combo + dense_w=1.5", **{**COMBO, "dense_w": 1.5})
    for k in (30, 60, 100):
        run(f"combo + rrf_k={k}", **{**COMBO, "rrf_k": k})
    for st in (5, 8, 12):
        run(f"combo sparse_topn={st}", **{**COMBO, "sparse_topn": st})
    for sa in (0.3, 0.5, 0.6):
        run(f"combo sparse_alpha={sa}", **{**COMBO, "sparse_alpha": sa})

    # combo on strip_qwords query text
    qv = [strip_qwords(q) for q in queries]
    ds = dense_for(qv)
    ss = bm25_score_batch(bm25, qv)
    for label, kw in [("combo+strip_qwords", COMBO), ("combo+strip_qwords+dw1.5", {**COMBO, "dense_w": 1.5})]:
        ranked = fuse(ds, ss, page_ids, qv, page_numbers, **kw)
        ndcg = mean_ndcg_at_k(ranked, gt, k=10)
        results.append((label, ndcg))
        print(f"  {ndcg:.4f}  {label}")

    # --- J. push BM25 mean-heavy + dense_w ---
    print("\n=== J. refine BM25 mean-heavy ===")
    for sa in (0.0, 0.1, 0.2, 0.3):
        for st in (5, 8, 12, 20):
            run(f"S{sa}/{st}", **{**BASE, "sparse_alpha": sa, "sparse_topn": st, "dense_topn": 100})
    print("  -- best-region + dense_w --")
    BEST = {**BASE, "sparse_alpha": 0.2, "sparse_topn": 12, "dense_topn": 100}
    for dw in (1.0, 1.3, 1.5, 2.0):
        run(f"S0.2/12 dense_w={dw}", **{**BEST, "dense_w": dw})

    print("\n================ TOP 12 ================")
    for label, ndcg in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ndcg:.4f}  {label}")


if __name__ == "__main__":
    main()

"""First-stage retrieval eval on the 50 public queries (committed artifacts/).

Complements scripts/eval_public.py (which reports NDCG@10): this reports the
pre-rerank pool metrics from eval_retrieval — recall@k, hit@k, MRR, R-precision
and the first-relevant-rank depth. Reads the committed artifacts; no rebuild.

    python eval_public_retrieval.py
"""
from __future__ import annotations

import time

from eval import load_query_file, mean_ndcg_at_k
from eval_retrieval import DEFAULT_KS, retrieval_metrics
from retrieve import search_batch
from utils import PUBLIC_QUERIES_PATH


def main() -> None:
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    depth = max(DEFAULT_KS)
    t0 = time.perf_counter()
    ranked = search_batch(queries, top_k=depth)
    elapsed = time.perf_counter() - t0

    ndcg = mean_ndcg_at_k(ranked, gt, k=10)
    ret = retrieval_metrics(ranked, gt)

    print(f"public_queries={len(queries)}  pool_depth={depth}")
    print(f"query_phase_time={elapsed:.2f}s")
    print(f"NDCG@10           = {ndcg:.4f}   (reranker metric, for reference)")
    print("--- first-stage (pre-rerank) ---")
    for k in DEFAULT_KS:
        print(f"recall@{k:<4d}       = {ret[f'recall@{k}']:.4f}    hit@{k:<4d} = {ret[f'hit@{k}']:.4f}")
    print(f"mrr@{depth}          = {ret[f'mrr@{depth}']:.4f}")
    print(f"r_precision       = {ret['r_precision']:.4f}")
    if "first_rel_rank_median" in ret:
        print(
            f"first_rel_rank    = median {ret['first_rel_rank_median']:.0f}, "
            f"p90 {ret['first_rel_rank_p90']:.0f}, "
            f"found_any {ret['found_any_rate']:.2f}"
        )


if __name__ == "__main__":
    main()

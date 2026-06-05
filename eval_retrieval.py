"""First-stage (pre-rerank) retrieval metrics — complement NDCG@10.

NDCG@10 scores the *reranker*: it is top-heavy and order-sensitive, so it
punishes a good candidate pool that merely has relevant pages in the wrong
order. Before reranking, what matters is the pool itself — did we retrieve the
relevant pages at all (recall), and how deep do they sit (so we know where to
cut the candidate pool). These metrics measure exactly that and are mostly
order-insensitive within the cutoff.

Metrics (binary relevance; ranked lists are de-duplicated preserving order,
matching ``eval.ndcg_at_k``; queries with no ground truth are skipped):

* ``recall@k``    — fraction of a query's relevant pages present in top-k.
                    THE first-stage metric: the reranker can only reorder what
                    recall captured, so recall@(pool depth) is the score ceiling.
* ``hit@k``       — fraction of queries with >=1 relevant page in top-k
                    (a.k.a. success@k). For single-answer (Type A) queries this
                    equals recall@k; for multi-answer (Type B) it is laxer.
* ``mrr@k``       — mean reciprocal rank of the FIRST relevant page (0 if none
                    in top-k). Order-light; rewards one early hit.
* ``r_precision`` — precision at R, where R = #relevant for that query. Balances
                    precision/recall without NDCG's positional discount.
* first-relevant-rank ``median`` / ``p90`` — depth diagnostic over queries that
                    have any hit in the returned list: where to set the pool cut.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence, Set, Tuple

DEFAULT_KS: Tuple[int, ...] = (10, 20, 50, 100)


def _dedup(ranked_ids: Sequence[int]) -> List[int]:
    """Drop duplicate page_ids, preserving first-seen order (mirrors eval.py)."""
    seen: Set[int] = set()
    out: List[int] = []
    for pid in ranked_ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def recall_at_k(ranked_ids: Sequence[int], relevant_ids: Set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for pid in _dedup(ranked_ids)[:k] if pid in relevant_ids)
    return hits / len(relevant_ids)


def hit_at_k(ranked_ids: Sequence[int], relevant_ids: Set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return 1.0 if any(pid in relevant_ids for pid in _dedup(ranked_ids)[:k]) else 0.0


def reciprocal_rank_at_k(
    ranked_ids: Sequence[int], relevant_ids: Set[int], k: int
) -> float:
    for rank, pid in enumerate(_dedup(ranked_ids)[:k], start=1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def r_precision(ranked_ids: Sequence[int], relevant_ids: Set[int]) -> float:
    r = len(relevant_ids)
    if r == 0:
        return 0.0
    hits = sum(1 for pid in _dedup(ranked_ids)[:r] if pid in relevant_ids)
    return hits / r


def first_relevant_rank(
    ranked_ids: Sequence[int], relevant_ids: Set[int]
) -> Optional[int]:
    """1-based rank of the first relevant page in the full list, or None."""
    for rank, pid in enumerate(_dedup(ranked_ids), start=1):
        if pid in relevant_ids:
            return rank
    return None


def retrieval_metrics(
    all_ranked: Sequence[Sequence[int]],
    all_relevant: Sequence[Set[int]],
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    """Macro-average first-stage metrics over queries (empty-GT queries skipped)."""
    pairs = [(r, rel) for r, rel in zip(all_ranked, all_relevant) if rel]
    n = len(pairs)
    out: Dict[str, float] = {"num_queries": float(n)}
    if n == 0:
        return out

    max_k = max(ks)
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in pairs) / n
        out[f"hit@{k}"] = sum(hit_at_k(r, rel, k) for r, rel in pairs) / n
    out[f"mrr@{max_k}"] = sum(
        reciprocal_rank_at_k(r, rel, max_k) for r, rel in pairs
    ) / n
    out["r_precision"] = sum(r_precision(r, rel) for r, rel in pairs) / n

    ranks = [fr for r, rel in pairs if (fr := first_relevant_rank(r, rel)) is not None]
    if ranks:
        ranks.sort()
        out["first_rel_rank_median"] = float(statistics.median(ranks))
        idx = max(0, int(round(0.9 * (len(ranks) - 1))))
        out["first_rel_rank_p90"] = float(ranks[idx])
        out["found_any_rate"] = len(ranks) / n
    return out


def evaluate_retrieval(
    queries: List[str],
    ground_truth: List[Set[int]],
    run_fn,
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    """Call ``run_fn(queries)`` (should return lists deep enough for max(ks)).

    Returns the first-stage metric dict. ``run_fn`` must rank at least ``max(ks)``
    page_ids per query, else recall at large k is understated — pass a closure
    like ``lambda qs: search_batch(qs, top_k=max(ks), ...)``.
    """
    ranked = run_fn(queries)
    if len(ranked) != len(queries):
        raise ValueError(
            f"run() returned {len(ranked)} lists for {len(queries)} queries"
        )
    return retrieval_metrics(ranked, ground_truth, ks=ks)

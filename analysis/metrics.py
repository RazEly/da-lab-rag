"""Self-contained retrieval metrics for the analysis package.

Reimplements the retrieval-quality metrics that the deleted ``eval_retrieval``
module used to provide, so analysis scripts depend only on the read-only
``eval.py`` (for the grader's exact NDCG) and stdlib/numpy.

Binary relevance, multi-page ground truth. Ground truth is passed as a set of
relevant page_ids per query; ranked lists are page_ids best-first.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set

# eval.py is read-only and allowed; it carries the grader's exact NDCG@10.
from eval import mean_ndcg_at_k, ndcg_at_k  # noqa: F401  (re-exported)

# Default k ladder shared across the analysis.
KS = (1, 3, 5, 10, 20, 50, 100)


def _dedup(ranked: Sequence[int]) -> List[int]:
    """Stable de-dup, mirroring the grader (duplicates score once)."""
    seen: Set[int] = set()
    out: List[int] = []
    for pid in ranked:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def recall_at_k(ranked: Sequence[int], gt: Set[int], k: int) -> float:
    """Fraction of relevant pages found in the top-k."""
    if not gt:
        return 0.0
    top = set(_dedup(ranked)[:k])
    return len(top & gt) / len(gt)


def hit_at_k(ranked: Sequence[int], gt: Set[int], k: int) -> float:
    """1.0 if any relevant page appears in the top-k, else 0.0."""
    if not gt:
        return 0.0
    return 1.0 if set(_dedup(ranked)[:k]) & gt else 0.0


def first_rel_rank(ranked: Sequence[int], gt: Set[int]) -> int:
    """1-indexed rank of the first relevant page, or 0 if none found."""
    for i, pid in enumerate(_dedup(ranked), start=1):
        if pid in gt:
            return i
    return 0


def mrr(ranked: Sequence[int], gt: Set[int], k: int = 100) -> float:
    """Reciprocal rank of the first relevant page within depth k (0 if none)."""
    r = first_rel_rank(ranked, gt)
    if r == 0 or r > k:
        return 0.0
    return 1.0 / r


def r_precision(ranked: Sequence[int], gt: Set[int]) -> float:
    """Precision at depth R = |gt| (relevant found in the top-|gt|)."""
    r = len(gt)
    if r == 0:
        return 0.0
    top = set(_dedup(ranked)[:r])
    return len(top & gt) / r


def mean_recall_curve(
    all_ranked: Sequence[Sequence[int]],
    all_gt: Sequence[Set[int]],
    ks: Sequence[int] = KS,
) -> Dict[int, float]:
    """Mean recall@k over all queries, for each k in ks."""
    out: Dict[int, float] = {}
    for k in ks:
        vals = [recall_at_k(r, g, k) for r, g in zip(all_ranked, all_gt)]
        out[k] = float(sum(vals) / len(vals)) if vals else 0.0
    return out


def mean_hit_curve(
    all_ranked: Sequence[Sequence[int]],
    all_gt: Sequence[Set[int]],
    ks: Sequence[int] = KS,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k in ks:
        vals = [hit_at_k(r, g, k) for r, g in zip(all_ranked, all_gt)]
        out[k] = float(sum(vals) / len(vals)) if vals else 0.0
    return out


def per_query_table(
    queries: Sequence[str],
    all_ranked: Sequence[Sequence[int]],
    all_gt: Sequence[Set[int]],
    ks: Sequence[int] = (10, 50, 100),
) -> List[Dict[str, object]]:
    """One row per query with NDCG@10, recall@k ladder, first-rel-rank, mrr."""
    rows: List[Dict[str, object]] = []
    for q, ranked, gt in zip(queries, all_ranked, all_gt):
        row: Dict[str, object] = {
            "query": q,
            "n_relevant": len(gt),
            "ndcg@10": ndcg_at_k(list(ranked), gt, k=10),
            "first_rel_rank": first_rel_rank(ranked, gt),
            "mrr@100": mrr(ranked, gt, 100),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, gt, k)
        rows.append(row)
    return rows


def summary(
    all_ranked: Sequence[Sequence[int]],
    all_gt: Sequence[Set[int]],
    ks: Sequence[int] = KS,
) -> Dict[str, float]:
    """Headline scalar metrics for a run."""
    n = len(all_ranked)
    recalls = mean_recall_curve(all_ranked, all_gt, ks)
    hits = mean_hit_curve(all_ranked, all_gt, ks)
    franks = [first_rel_rank(r, g) for r, g in zip(all_ranked, all_gt)]
    found = [f for f in franks if f > 0]
    found.sort()

    def pct(vals: List[int], p: float) -> float:
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, int(round(p * (len(vals) - 1))))
        return float(vals[idx])

    out: Dict[str, float] = {
        "ndcg@10": mean_ndcg_at_k(all_ranked, all_gt, k=10),
        "mrr@100": float(sum(mrr(r, g, 100) for r, g in zip(all_ranked, all_gt)) / n)
        if n
        else 0.0,
        "found_any": float(len(found) / n) if n else 0.0,
        "first_rel_median": pct(found, 0.5),
        "first_rel_p90": pct(found, 0.9),
    }
    for k in ks:
        out[f"recall@{k}"] = recalls[k]
        out[f"hit@{k}"] = hits[k]
    return out

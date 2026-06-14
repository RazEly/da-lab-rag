"""Step E (Slide 7): RRF vs score-blend; RRF_K sweep.

(1) NDCG@10 for dense-only, bm25-only, RRF-fused, min-max score-blend baseline
    (shows the dense/BM25 scale mismatch that collapses score blending).
(2) NDCG@10 vs RRF_K grid (incl. shipped). Production target: RRF_K.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as C
import harness as H
import metrics as M

RESULT = "s7_rrf.json"
RRF_KS = (5, 10, 20, 28, 60, 100, 200)


def _page_scores(scores, page_ids, alpha, topn, depth):
    """page_id → blended chunk score (no truncation), for score-blend baseline."""
    from collections import defaultdict
    pc = defaultdict(list)
    for i, s in enumerate(scores):
        pc[page_ids[i]].append(float(s))
    out = {}
    for pid, lst in pc.items():
        lst.sort(reverse=True)
        tn = lst[:topn]
        out[pid] = alpha * lst[0] + (1 - alpha) * (sum(tn) / len(tn))
    return out


def _score_blend(dense, sparse, page_ids, queries, gt, p):
    """Min-max normalize each leg's page scores, sum, rank — the rejected baseline."""
    ranked_all = []
    for qi in range(len(queries)):
        ds = _page_scores(dense[qi], page_ids, p["DENSE_ALPHA"], p["DENSE_TOPN"], 100)
        sp = _page_scores(sparse[qi], page_ids, p["SPARSE_ALPHA"], p["SPARSE_TOPN"], 100)
        pages = set(ds) | set(sp)

        def norm(d):
            vals = [d.get(x, 0.0) for x in pages]
            lo, hi = min(vals), max(vals)
            rng = hi - lo or 1.0
            return {x: (d.get(x, 0.0) - lo) / rng for x in pages}

        nd, ns = norm(ds), norm(sp)
        blend = {x: nd[x] + ns[x] for x in pages}
        ranked_all.append(sorted(blend, key=blend.__getitem__, reverse=True)[:100])
    return M.mean_ndcg_at_k(ranked_all, gt, 10)


def compute() -> dict:
    ctx = H.load_prod()
    p = C.current_prod_params()
    ss = H.prod_bm25(p["BM25_QUERY_MIN_IDF"], p["BM25_FALLBACK_THRESHOLD"])
    fk = C.prod_fuse_kwargs()

    def fused(**ov):
        r = C.fuse_prod(ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers,
                        top_k=100, **{**fk, **ov})
        return M.mean_ndcg_at_k(r, ctx.gt, 10), M.mean_recall_curve(r, ctx.gt, [100])[100]

    dense_only = fused(include_sparse=False)[0]
    bm25_only = fused(include_dense=False)[0]
    rrf_fused, rrf_recall = fused()
    blend = _score_blend(ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.gt, p)

    kcurve = {}
    krecall = {}
    for k in RRF_KS:
        n, r = fused(rrf_k=k)
        kcurve[str(k)] = n
        krecall[str(k)] = r
    best_k = int(max(kcurve, key=kcurve.__getitem__))
    return {
        "comparison": {"dense-only": dense_only, "bm25-only": bm25_only,
                       "rrf-fused": rrf_fused, "score-blend": blend},
        "rrf_k_curve": kcurve, "rrf_k_recall100": krecall,
        "rrf_recall100": rrf_recall,
        "shipped_k": p["RRF_K"], "best_k": best_k,
    }


def plot(data: dict) -> None:
    import plot_style as PS
    import matplotlib.pyplot as plt

    PS.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    comp = data["comparison"]
    names = list(comp)
    bars = ax1.bar(names, [comp[n] for n in names],
                   color=[PS.PALETTE[i] for i in range(len(names))], edgecolor="white")
    ax1.bar_label(bars, fmt="%.3f", fontsize=8)
    ax1.set_ylabel("NDCG@10"); ax1.set_title("Fusion strategy")
    ax1.tick_params(axis="x", rotation=20)

    ks = sorted(int(k) for k in data["rrf_k_curve"])
    ax2.plot(ks, [data["rrf_k_curve"][str(k)] for k in ks], "o-", color=PS.PALETTE[0])
    ax2.axvline(data["shipped_k"], color=PS.SHIPPED_COLOR, ls="--",
                label=f"shipped k={data['shipped_k']}")
    ax2.set_xlabel("RRF_K"); ax2.set_ylabel("NDCG@10")
    ax2.set_title("RRF_K sweep"); ax2.legend()
    fig.suptitle(f"RRF vs score-blend ({PS.QUERY_NOTE})")
    PS.save(fig, "s7_rrf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    path = C.RESULTS_DIR / RESULT
    if path.exists() and not a.force:
        print(f"[s7-rrf] cached")
        data = C.load_raw(RESULT)
    else:
        data = compute()
        C.save_raw(RESULT, data)
    print(f"  comparison: {data['comparison']}")
    print(f"  shipped k={data['shipped_k']} best k={data['best_k']}")
    prod_ndcg = data["rrf_k_curve"][str(data["shipped_k"])]
    cand_ndcg = data["rrf_k_curve"][str(data["best_k"])]
    H.gate(
        "E. RRF_K (slide 7)",
        cand_ndcg=cand_ndcg, prod_ndcg=prod_ndcg,
        cand_recall100=data["rrf_k_recall100"][str(data["best_k"])],
        prod_recall100=data["rrf_k_recall100"][str(data["shipped_k"])],
        cand_params=f"RRF_K={data['best_k']}", prod_params=f"RRF_K={data['shipped_k']}",
    )
    plot(data)


if __name__ == "__main__":
    main()

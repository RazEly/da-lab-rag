"""Step C (Slide 5): hybrid fusion vs either leg alone.

Diagnostic — confirms fusing dense + BM25 beats shipping one leg. recall@{5,10,20}
+ NDCG@10 for dense-only, bm25-only, fused (first-stage pool, number filter off,
shipped aggregation). No production param. If a solo leg unexpectedly wins the
gate, that's a structural finding logged to CHANGELOG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as C
import harness as H
import metrics as M

RESULT = "s5_retrievers.json"
KS = (5, 10, 20)


def compute() -> dict:
    ctx = H.load_prod()
    p = C.current_prod_params()
    ss = H.prod_bm25(p["BM25_QUERY_MIN_IDF"], p["BM25_FALLBACK_THRESHOLD"])
    fk = C.prod_fuse_kwargs()
    fk["filter_mode"] = "off"  # first-stage pool, pre-filter

    def run(include_dense: bool, include_sparse: bool) -> dict:
        ranked = C.fuse_prod(
            ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers,
            include_dense=include_dense, include_sparse=include_sparse,
            top_k=100, **fk,
        )
        out = {"ndcg@10": M.mean_ndcg_at_k(ranked, ctx.gt, 10)}
        for k in KS:
            out[f"recall@{k}"] = M.mean_recall_curve(ranked, ctx.gt, [k])[k]
        out["recall@100"] = M.mean_recall_curve(ranked, ctx.gt, [100])[100]
        return out

    legs = {
        "dense-only": run(True, False),
        "bm25-only": run(False, True),
        "fused": run(True, True),
    }
    return {"legs": legs, "ks": list(KS), "n_queries": len(ctx.queries)}


def plot(data: dict) -> None:
    import numpy as np
    import plot_style as PS
    import matplotlib.pyplot as plt

    PS.apply_style()
    legs = data["legs"]
    metrics_order = [f"recall@{k}" for k in data["ks"]] + ["ndcg@10"]
    names = ["dense-only", "bm25-only", "fused"]
    x = np.arange(len(metrics_order))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, nm in enumerate(names):
        vals = [legs[nm][m] for m in metrics_order]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=nm,
                      color=PS.PALETTE[i], edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_order)
    ax.set_ylabel("score")
    ax.set_title(f"Hybrid fusion vs single leg ({PS.QUERY_NOTE})")
    ax.legend()
    PS.save(fig, "s5_retrievers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    path = C.RESULTS_DIR / RESULT
    if path.exists() and not a.force:
        print(f"[s5] using cached {RESULT}")
        data = C.load_raw(RESULT)
    else:
        data = compute()
        C.save_raw(RESULT, data)
    for nm, v in data["legs"].items():
        print(f"  {nm:12s} ndcg@10={v['ndcg@10']:.4f} recall@10={v['recall@10']:.4f} recall@100={v['recall@100']:.4f}")
    fused, dense, bm25 = data["legs"]["fused"], data["legs"]["dense-only"], data["legs"]["bm25-only"]
    best_solo = max(dense, bm25, key=lambda d: d["ndcg@10"])
    H.gate(
        "C. hybrid vs solo (slide 5)",
        cand_ndcg=best_solo["ndcg@10"], prod_ndcg=fused["ndcg@10"],
        cand_recall100=best_solo["recall@100"], prod_recall100=fused["recall@100"],
        cand_params="best solo leg", prod_params="fused (shipped)",
        note="diagnostic; CHANGE here would mean disabling a leg (unexpected)",
    )
    plot(data)


if __name__ == "__main__":
    main()

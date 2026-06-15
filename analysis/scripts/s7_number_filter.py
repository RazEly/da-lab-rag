"""Step G (Slide 7): number filter mode off / pre / post.

NDCG@10 + recall@10 for each mode. Production target: FILTER_MODE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as C
import harness as H
import metrics as M

RESULT = "number_filter.json"
MODES = ("off", "pre", "post")


def compute() -> dict:
    ctx = H.load_prod()
    p = C.current_prod_params()
    ss = H.prod_bm25()
    fk = C.prod_fuse_kwargs()
    out = {}
    for m in MODES:
        r = C.fuse_prod(ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers,
                        top_k=100, **{**fk, "filter_mode": m})
        out[m] = {
            "ndcg@10": M.mean_ndcg_at_k(r, ctx.gt, 10),
            "recall@10": M.mean_recall_curve(r, ctx.gt, [10])[10],
            "recall@100": M.mean_recall_curve(r, ctx.gt, [100])[100],
        }
    return {"modes": out, "shipped": p["FILTER_MODE"]}


def plot(data: dict) -> None:
    import numpy as np
    import plot_style as PS
    import matplotlib.pyplot as plt

    PS.apply_style()
    modes = list(data["modes"])
    x = np.arange(len(modes)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, met in enumerate(("ndcg@10", "recall@10")):
        vals = [data["modes"][m][met] for m in modes]
        b = ax.bar(x + (i - 0.5) * w, vals, w, label=met,
                   color=PS.PALETTE[i], edgecolor="white")
        ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(
        [f"{m}{' (shipped)' if m==data['shipped'] else ''}" for m in modes])
    ax.set_ylabel("score"); ax.legend()
    ax.set_title(f"Number filter mode ({PS.QUERY_NOTE})")
    PS.save(fig, "s7_number_filter")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    path = C.RESULTS_DIR / RESULT
    if path.exists() and not a.force:
        print("[s7-numfilter] cached")
        data = C.load_raw(RESULT)
    else:
        data = compute()
        C.save_raw(RESULT, data)
    for m, v in data["modes"].items():
        print(f"  {m:5s} ndcg={v['ndcg@10']:.4f} recall@10={v['recall@10']:.4f}")
    shipped = data["shipped"]
    best = max(data["modes"], key=lambda m: data["modes"][m]["ndcg@10"])
    H.gate(
        "G. number filter mode (slide 7)",
        cand_ndcg=data["modes"][best]["ndcg@10"],
        prod_ndcg=data["modes"][shipped]["ndcg@10"],
        cand_recall100=data["modes"][best]["recall@100"],
        prod_recall100=data["modes"][shipped]["recall@100"],
        cand_params=f"filter_mode={best}", prod_params=f"filter_mode={shipped}",
    )
    plot(data)


if __name__ == "__main__":
    main()

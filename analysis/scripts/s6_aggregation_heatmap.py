"""Step D (Slide 6): per-retriever chunk→page aggregation (alpha/topn).

Solo-retriever NDCG@10 heatmaps over alpha × topn, dense and BM25 separately;
then evaluate the best-cell pair FUSED (solo-optimal ≠ fused-optimal — gate on the
fused score). Production target: DENSE_ALPHA/TOPN, SPARSE_ALPHA/TOPN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as C
import harness as H
import metrics as M

RESULT = "s6_agg.json"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TOPNS = (1, 3, 10, 30, 100)


def _solo_grid(scores, page_ids, gt) -> dict:
    grid = {}
    for a in ALPHAS:
        for t in TOPNS:
            ranked = [C.agg(scores[qi], page_ids, depth=100, alpha=a, topn=t)
                      for qi in range(len(gt))]
            grid[f"{a}|{t}"] = M.mean_ndcg_at_k(ranked, gt, 10)
    return grid


def _best_cell(grid: dict):
    key = max(grid, key=grid.__getitem__)
    a, t = key.split("|")
    return float(a), int(t), grid[key]


def compute() -> dict:
    ctx = H.load_prod()
    p = C.current_prod_params()
    ss = H.prod_bm25()
    dense_grid = _solo_grid(ctx.dense, ctx.page_ids, ctx.gt)
    bm25_grid = _solo_grid(ss, ctx.page_ids, ctx.gt)
    d_a, d_t, d_best = _best_cell(dense_grid)
    s_a, s_t, s_best = _best_cell(bm25_grid)

    fk = C.prod_fuse_kwargs()
    fused_cand = C.fuse_prod(
        ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers,
        **{**fk, "dense_alpha": d_a, "dense_topn": d_t,
           "sparse_alpha": s_a, "sparse_topn": s_t}, top_k=100,
    )
    fused_prod = C.fuse_prod(
        ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers, **fk, top_k=100,
    )
    return {
        "alphas": list(ALPHAS), "topns": list(TOPNS),
        "dense_grid": dense_grid, "bm25_grid": bm25_grid,
        "dense_solo_best": {"alpha": d_a, "topn": d_t, "ndcg": d_best},
        "bm25_solo_best": {"alpha": s_a, "topn": s_t, "ndcg": s_best},
        "fused_cand_ndcg": M.mean_ndcg_at_k(fused_cand, ctx.gt, 10),
        "fused_cand_recall100": M.mean_recall_curve(fused_cand, ctx.gt, [100])[100],
        "fused_prod_ndcg": M.mean_ndcg_at_k(fused_prod, ctx.gt, 10),
        "fused_prod_recall100": M.mean_recall_curve(fused_prod, ctx.gt, [100])[100],
        "prod": {"dense": (p["DENSE_ALPHA"], p["DENSE_TOPN"]),
                 "bm25": (p["SPARSE_ALPHA"], p["SPARSE_TOPN"])},
    }


def _dump_csv(data: dict) -> None:
    for leg in ("dense", "bm25"):
        rows = []
        for a in data["alphas"]:
            for t in data["topns"]:
                rows.append({"alpha": a, "topn": t,
                             "ndcg@10": data[f"{leg}_grid"][f"{a}|{t}"]})
        C.save_csv(f"s6_agg_{leg}.csv", rows)


def plot(data: dict) -> None:
    import numpy as np
    import plot_style as PS
    import matplotlib.pyplot as plt

    PS.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, leg, prodkey in zip(axes, ("dense", "bm25"), ("dense", "bm25")):
        grid = data[f"{leg}_grid"]
        mat = np.array([[grid[f"{a}|{t}"] for t in data["topns"]]
                        for a in data["alphas"]])
        im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(data["topns"])))
        ax.set_xticklabels(data["topns"])
        ax.set_yticks(range(len(data["alphas"])))
        ax.set_yticklabels(data["alphas"])
        ax.set_xlabel("topn"); ax.set_ylabel("alpha (1=max,0=mean)")
        ax.set_title(f"{leg} solo NDCG@10")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
        pa, pt = data["prod"][prodkey]
        if pa in data["alphas"] and pt in data["topns"]:
            ax.add_patch(plt.Rectangle(
                (data["topns"].index(pt) - 0.5, data["alphas"].index(pa) - 0.5),
                1, 1, fill=False, edgecolor=PS.SHIPPED_COLOR, lw=2.5))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Per-retriever aggregation ({PS.QUERY_NOTE}; red = shipped)")
    PS.save(fig, "s6_aggregation")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    path = C.RESULTS_DIR / RESULT
    if path.exists() and not a.force:
        print(f"[s6] using cached {RESULT}")
        data = C.load_raw(RESULT)
    else:
        data = compute()
        C.save_raw(RESULT, data)
    _dump_csv(data)
    print(f"  dense solo best: {data['dense_solo_best']}")
    print(f"  bm25  solo best: {data['bm25_solo_best']}")
    print(f"  fused cand NDCG={data['fused_cand_ndcg']:.4f}  prod NDCG={data['fused_prod_ndcg']:.4f}")
    H.gate(
        "D. per-retriever aggregation (slide 6)",
        cand_ndcg=data["fused_cand_ndcg"], prod_ndcg=data["fused_prod_ndcg"],
        cand_recall100=data["fused_cand_recall100"],
        prod_recall100=data["fused_prod_recall100"],
        cand_params=f"D({data['dense_solo_best']['alpha']}/{data['dense_solo_best']['topn']}) "
                    f"S({data['bm25_solo_best']['alpha']}/{data['bm25_solo_best']['topn']})",
        prod_params=f"D{data['prod']['dense']} S{data['prod']['bm25']}",
        note="fused-optimal gate, not solo-optimal",
    )
    plot(data)


if __name__ == "__main__":
    main()

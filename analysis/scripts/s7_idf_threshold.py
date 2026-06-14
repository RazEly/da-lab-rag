"""Step F (Slide 7): BM25 query-term IDF filter threshold.

NDCG@10 vs BM25_QUERY_MIN_IDF over [0..6] (re-score BM25 per threshold, fuse with
shipped params, fallback held at shipped). Report where the plateau is; prefer the
plateau centre over a noisy peak. Production target: BM25_QUERY_MIN_IDF.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as C
import harness as H
import metrics as M

RESULT = "s7_idf.json"
THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)


def compute() -> dict:
    ctx = H.load_prod()
    p = C.current_prod_params()
    fk = C.prod_fuse_kwargs()
    fb = p["BM25_FALLBACK_THRESHOLD"]
    curve, recall = {}, {}
    for t in THRESHOLDS:
        ss = H.prod_bm25(t, fb)
        r = C.fuse_prod(ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers,
                        top_k=100, **fk)
        curve[str(t)] = M.mean_ndcg_at_k(r, ctx.gt, 10)
        recall[str(t)] = M.mean_recall_curve(r, ctx.gt, [100])[100]
    best_t = float(max(curve, key=curve.__getitem__))
    return {"curve": curve, "recall100": recall, "shipped": p["BM25_QUERY_MIN_IDF"],
            "best": best_t, "fallback": fb}


def plot(data: dict) -> None:
    import plot_style as PS
    import matplotlib.pyplot as plt

    PS.apply_style()
    ts = sorted(float(t) for t in data["curve"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ts, [data["curve"][str(t)] for t in ts], "o-", color=PS.PALETTE[0])
    ax.axvline(data["shipped"], color=PS.SHIPPED_COLOR, ls="--",
               label=f"shipped={data['shipped']}")
    ax.set_xlabel("BM25_QUERY_MIN_IDF"); ax.set_ylabel("NDCG@10")
    ax.set_title(f"BM25 IDF filter ({PS.QUERY_NOTE})"); ax.legend()
    PS.save(fig, "s7_idf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    path = C.RESULTS_DIR / RESULT
    if path.exists() and not a.force:
        print("[s7-idf] cached")
        data = C.load_raw(RESULT)
    else:
        data = compute()
        C.save_raw(RESULT, data)
    for t in sorted(float(x) for x in data["curve"]):
        print(f"  idf>={t:<4} ndcg={data['curve'][str(t)]:.4f}")
    H.gate(
        "F. BM25 IDF threshold (slide 7)",
        cand_ndcg=data["curve"][str(data["best"])],
        prod_ndcg=data["curve"][str(data["shipped"])],
        cand_recall100=data["recall100"][str(data["best"])],
        prod_recall100=data["recall100"][str(data["shipped"])],
        cand_params=f"min_idf={data['best']}", prod_params=f"min_idf={data['shipped']}",
        note="prefer plateau centre over noisy peak",
    )
    plot(data)


if __name__ == "__main__":
    main()

"""Steps H–K (supporting evidence) against the FINAL shipped config.

H. Per-query NDCG@10 + zero-score count   -> results/per_query_ndcg.csv
I. First-relevant-rank dense vs BM25       -> results/first_rel_rank.csv
J. Latency vs 60 s wall (≥3 warm run())    -> results/latency.json
K. Headline scorecard                       -> results/headline.json

Uses the production search_batch (truth) for the shipped ranking, and fuse_prod
solo legs for the dense-vs-BM25 first-rel comparison.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as C
import harness as H
import metrics as M


def step_HK(skip_latency: bool) -> dict:
    ctx = H.load_prod()
    p = C.current_prod_params()
    ss = H.prod_bm25(p["BM25_QUERY_MIN_IDF"], p["BM25_FALLBACK_THRESHOLD"])
    fk = C.prod_fuse_kwargs()

    # Shipped fused ranking (deep) for recall + per-query.
    shipped = C.fuse_prod(ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers,
                          top_k=100, **fk)

    # H. per-query
    pq = M.per_query_table(ctx.queries, shipped, ctx.gt, ks=(10, 50, 100))
    zero = sum(1 for r in pq if r["ndcg@10"] == 0.0)
    C.save_csv("per_query_ndcg.csv", pq)

    # I. first-relevant-rank dense vs bm25 (solo legs, deep)
    dense_solo = C.fuse_prod(ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers,
                             include_sparse=False, top_k=200, **{**fk, "filter_mode": "off"})
    bm25_solo = C.fuse_prod(ctx.dense, ss, ctx.page_ids, ctx.queries, ctx.page_numbers,
                            include_dense=False, top_k=200, **{**fk, "filter_mode": "off"})
    frr_rows = []
    for i, q in enumerate(ctx.queries):
        frr_rows.append({
            "query": q,
            "dense_first_rel": M.first_rel_rank(dense_solo[i], ctx.gt[i]),
            "bm25_first_rel": M.first_rel_rank(bm25_solo[i], ctx.gt[i]),
            "fused_first_rel": M.first_rel_rank(shipped[i], ctx.gt[i]),
        })
    C.save_csv("first_rel_rank.csv", frr_rows)

    def med_p90(key):
        found = sorted(r[key] for r in frr_rows if r[key] > 0)
        if not found:
            return {"median": 0, "p90": 0, "found_frac": 0.0}
        med = found[len(found) // 2]
        p90 = found[min(len(found) - 1, int(0.9 * (len(found) - 1)))]
        return {"median": med, "p90": p90, "found_frac": len(found) / len(frr_rows)}

    # J. latency
    latency = {}
    if not skip_latency:
        from main import run
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            run(ctx.queries)
            ts.append(time.perf_counter() - t0)
        latency = {"runs_s": ts, "mean_s": statistics.mean(ts), "max_s": max(ts),
                   "wall_s": 60.0, "n_queries": len(ctx.queries),
                   "note": "run() reloads artifacts each call (grader-equivalent)"}

    # K. headline
    summ = M.summary(shipped, ctx.gt, M.KS)
    headline = {
        "ndcg@10": summ["ndcg@10"],
        "recall@10": summ["recall@10"], "recall@50": summ["recall@50"],
        "recall@100": summ["recall@100"],
        "mrr@100": summ["mrr@100"], "found_any": summ["found_any"],
        "first_rel_median": summ["first_rel_median"], "first_rel_p90": summ["first_rel_p90"],
        "n_queries": len(ctx.queries), "n_chunks": len(ctx.page_ids),
        "n_pages": len(set(ctx.page_ids)),
        "reorder_headroom": summ["recall@100"] - summ["ndcg@10"],
        "query_time_s": latency.get("mean_s") if latency else None,
    }

    return {
        "per_query_zero_count": zero,
        "per_query_ndcg_mean": summ["ndcg@10"],
        "first_rel_dense": med_p90("dense_first_rel"),
        "first_rel_bm25": med_p90("bm25_first_rel"),
        "first_rel_fused": med_p90("fused_first_rel"),
        "latency": latency,
        "headline": headline,
    }


def plots(data: dict) -> None:
    import numpy as np
    import plot_style as PS
    import matplotlib.pyplot as plt

    PS.apply_style()
    # H: per-query NDCG strip
    import csv
    pq = list(csv.DictReader((C.RESULTS_DIR / "per_query_ndcg.csv").open()))
    vals = sorted(float(r["ndcg@10"]) for r in pq)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(range(len(vals)), vals, color=PS.PALETTE[0], edgecolor="white")
    ax.axhline(np.mean(vals), color=PS.SHIPPED_COLOR, ls="--",
               label=f"mean={np.mean(vals):.3f}")
    ax.set_xlabel("query (sorted)"); ax.set_ylabel("NDCG@10")
    ax.set_title(f"Per-query NDCG@10 — {data['per_query_zero_count']} zero ({PS.QUERY_NOTE})")
    ax.legend()
    PS.save(fig, "h_per_query_ndcg")

    # I: first-rel CDF dense vs bm25
    frr = list(csv.DictReader((C.RESULTS_DIR / "first_rel_rank.csv").open()))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for key, col, lbl in (("dense_first_rel", PS.PALETTE[0], "dense"),
                          ("bm25_first_rel", PS.PALETTE[1], "bm25"),
                          ("fused_first_rel", PS.PALETTE[2], "fused")):
        ranks = sorted(int(r[key]) for r in frr if int(r[key]) > 0)
        if not ranks:
            continue
        ys = np.arange(1, len(ranks) + 1) / len(frr)
        ax.step(ranks, ys, where="post", color=col, label=lbl, lw=2)
    ax.set_xscale("log"); ax.set_xlabel("first relevant rank")
    ax.set_ylabel("fraction of queries"); ax.set_title(f"First-relevant-rank CDF ({PS.QUERY_NOTE})")
    ax.legend()
    PS.save(fig, "i_first_rel_cdf")

    # J: latency bar
    if data["latency"]:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.bar(["mean run()"], [data["latency"]["mean_s"]], color=PS.PALETTE[0],
               edgecolor="white", yerr=[[data["latency"]["mean_s"] - min(data["latency"]["runs_s"])],
                                        [data["latency"]["max_s"] - data["latency"]["mean_s"]]])
        ax.axhline(60, color=PS.SHIPPED_COLOR, ls="--", label="60 s wall")
        ax.set_ylabel("seconds"); ax.set_title("run() latency vs wall"); ax.legend()
        PS.save(fig, "j_latency")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    a = ap.parse_args()
    path = C.RESULTS_DIR / "extra_stats.json"
    if path.exists() and not a.force:
        print("[extra] cached")
        data = C.load_raw("extra_stats.json")
    else:
        data = step_HK(a.skip_latency)
        C.save_raw("extra_stats.json", data)
        C.save_raw("headline.json", data["headline"])
        if data["latency"]:
            C.save_raw("latency.json", data["latency"])
    print(f"  headline: {data['headline']}")
    print(f"  zero-score queries: {data['per_query_zero_count']}")
    print(f"  first-rel dense={data['first_rel_dense']} bm25={data['first_rel_bm25']}")
    plots(data)


if __name__ == "__main__":
    main()

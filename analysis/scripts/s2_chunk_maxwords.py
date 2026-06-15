"""Step A (Slide 2, plot A): chunk MAX_WORDS — index-affecting.

Dense-only recall@KS + NDCG@10 over MAX_WORDS ∈ {80,120,160,200} (mode=title index
per value; shipped dense aggregation). Records chunk_count (cost). The gate compares
each candidate's dense-only score against the shipped MAX_WORDS=120 on the SAME
dense-only basis; a clear win flags the expensive full artifact rebuild.

Deck note: the slide says "MIN_WORDS" — no such production knob exists; we plot
MAX_WORDS and footnote the rename.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as C
import harness as H
import metrics as M
from index import INDEX_META_NAME
from variant_index import build_textmode

RESULT = "s2_maxwords.json"
MAXWORDS = (80, 120, 160, 200)
SHIPPED = 120


def _dense_only(index_dir: Path, queries, gt, tag) -> dict:
    meta = json.loads((index_dir / INDEX_META_NAME).read_text())
    page_ids = [int(x) for x in meta["page_ids"]]
    dense = C.dense_score_matrix(index_dir, queries, tag=tag, variant="base")
    ranked = [C.agg(dense[qi], page_ids, depth=100, alpha=C.current_prod_params()["DENSE_ALPHA"],
                    topn=C.current_prod_params()["DENSE_TOPN"]) for qi in range(len(gt))]
    out = {"ndcg@10": M.mean_ndcg_at_k(ranked, gt, 10),
           "chunk_count": meta.get("chunk_count", len(page_ids))}
    rc = M.mean_recall_curve(ranked, gt, M.KS)
    for k in M.KS:
        out[f"recall@{k}"] = rc[k]
    return out


def compute(subset, force) -> dict:
    queries, gt = C.load_eval()
    rows = {}
    for mw in MAXWORDS:
        # mode=title == production [title] body; shares the index B also builds.
        d = build_textmode("title", mw=mw, subset=subset, force=force)
        rows[str(mw)] = _dense_only(d, queries, gt, f"mw{mw}_title")
        print(f"  mw{mw}: ndcg@10={rows[str(mw)]['ndcg@10']:.4f} "
              f"recall@10={rows[str(mw)]['recall@10']:.4f} chunks={rows[str(mw)]['chunk_count']}")
    return {"maxwords": list(MAXWORDS), "rows": rows, "shipped": SHIPPED, "ks": list(M.KS)}


def plot(data: dict) -> None:
    import plot_style as PS
    import matplotlib.pyplot as plt

    PS.apply_style()
    ks = [k for k in data["ks"] if k in (1, 3, 5, 10)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, mw in enumerate(data["maxwords"]):
        r = data["rows"][str(mw)]
        ys = [r[f"recall@{k}"] for k in ks]
        lbl = f"MAX_WORDS={mw}" + (" (shipped)" if mw == data["shipped"] else "")
        ax.plot(ks, ys, "o-", color=PS.PALETTE[i % len(PS.PALETTE)],
                lw=2.5 if mw == data["shipped"] else 1.5, label=lbl)
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("k"); ax.set_ylabel("dense-only recall@k")
    ax.legend(loc="lower right")
    fig.suptitle("recall@k across different chunk sizes")
    PS.save(fig, "s2_maxwords")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    subset = None if a.full else a.subset
    path = C.RESULTS_DIR / RESULT
    if path.exists() and not a.force:
        print("[s2-maxwords] cached")
        data = C.load_raw(RESULT)
    else:
        data = compute(subset, a.force)
        C.save_raw(RESULT, data)
    data["scope"] = "" if a.full else f"; subset {subset} (directional)"
    rows = data["rows"]
    best = max(rows, key=lambda m: rows[m]["ndcg@10"])
    H.gate(
        "A. chunk MAX_WORDS (slide 2) [index-affecting]",
        cand_ndcg=rows[best]["ndcg@10"], prod_ndcg=rows[str(data["shipped"])]["ndcg@10"],
        cand_recall100=rows[best]["recall@100"], prod_recall100=rows[str(data["shipped"])]["recall@100"],
        cand_params=f"MAX_WORDS={best} (dense-only)",
        prod_params=f"MAX_WORDS={data['shipped']} (dense-only)",
        note="dense-only basis; a pass flags full artifact rebuild before shipping",
    )
    plot(data)


if __name__ == "__main__":
    main()

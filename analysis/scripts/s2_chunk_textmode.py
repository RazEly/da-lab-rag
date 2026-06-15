"""Step B (Slide 2, plot B): title / header injection — index-affecting.

Dense-only recall@KS + NDCG@10 for embedded-text modes vanilla / title /
title_header, then crossed with MAX_WORDS (joint chunking + injection decision).
Production target: chunk text assembly in chunk._make_chunks (ships [title] body).
A clear winning cell flags the full artifact rebuild.
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

RESULT = "s2_textmode.json"
MODES = ("vanilla", "title", "title_header")
MAXWORDS = (80, 120, 160, 200)
SHIPPED_MODE = "title"
SHIPPED_MW = 120


def _dense_only(index_dir: Path, queries, gt, tag) -> dict:
    meta = json.loads((index_dir / INDEX_META_NAME).read_text())
    page_ids = [int(x) for x in meta["page_ids"]]
    p = C.current_prod_params()
    dense = C.dense_score_matrix(index_dir, queries, tag=tag, variant="base")
    ranked = [C.agg(dense[qi], page_ids, depth=100, alpha=p["DENSE_ALPHA"], topn=p["DENSE_TOPN"])
              for qi in range(len(gt))]
    out = {"ndcg@10": M.mean_ndcg_at_k(ranked, gt, 10)}
    rc = M.mean_recall_curve(ranked, gt, M.KS)
    for k in M.KS:
        out[f"recall@{k}"] = rc[k]
    return out


def compute(subset, force) -> dict:
    queries, gt = C.load_eval()
    # main: 3 modes at shipped MAX_WORDS
    modes = {}
    for m in MODES:
        d = build_textmode(m, mw=SHIPPED_MW, subset=subset, force=force)
        modes[m] = _dense_only(d, queries, gt, f"mw{SHIPPED_MW}_{m}")
        print(f"  mode {m}: ndcg@10={modes[m]['ndcg@10']:.4f} recall@10={modes[m]['recall@10']:.4f}")
    # A×B grid
    grid = {}
    for mw in MAXWORDS:
        for m in MODES:
            d = build_textmode(m, mw=mw, subset=subset, force=force)
            r = _dense_only(d, queries, gt, f"mw{mw}_{m}")
            grid[f"{mw}|{m}"] = {"ndcg@10": r["ndcg@10"], "recall@10": r["recall@10"],
                                 "recall@100": r["recall@100"]}
            print(f"  grid mw{mw}/{m}: ndcg@10={r['ndcg@10']:.4f}")
    return {"modes": modes, "grid": grid, "maxwords": list(MAXWORDS),
            "mode_names": list(MODES), "shipped_mode": SHIPPED_MODE, "shipped_mw": SHIPPED_MW,
            "ks": list(M.KS)}


def plot(data: dict) -> None:
    import numpy as np
    import plot_style as PS
    import matplotlib.pyplot as plt

    PS.apply_style()
    ks = [k for k in data["ks"] if k <= 10]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, m in enumerate(data["mode_names"]):
        ys = [data["modes"][m][f"recall@{k}"] for k in ks]
        lbl = m + (" (shipped)" if m == data["shipped_mode"] else "")
        ax1.plot(ks, ys, "o-", color=PS.PALETTE[i],
                 lw=2.5 if m == data["shipped_mode"] else 1.5, label=lbl)
    ax1.set_xscale("log"); ax1.set_xticks(ks); ax1.set_xticklabels(ks)
    ax1.set_xlabel("k"); ax1.set_ylabel("dense-only recall@k")
    ax1.set_title("Text-mode injection"); ax1.legend()

    mat = np.array([[data["grid"][f"{mw}|{m}"]["ndcg@10"] for m in data["mode_names"]]
                    for mw in data["maxwords"]])
    im = ax2.imshow(mat, aspect="auto", cmap="viridis", origin="lower")
    ax2.set_xticks(range(len(data["mode_names"]))); ax2.set_xticklabels(data["mode_names"])
    ax2.set_yticks(range(len(data["maxwords"]))); ax2.set_yticklabels(data["maxwords"])
    ax2.set_xlabel("text mode"); ax2.set_ylabel("MAX_WORDS")
    ax2.set_title("A×B grid NDCG@10")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax2.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    si, sj = data["maxwords"].index(data["shipped_mw"]), data["mode_names"].index(data["shipped_mode"])
    ax2.add_patch(plt.Rectangle((sj - 0.5, si - 0.5), 1, 1, fill=False,
                                edgecolor=PS.SHIPPED_COLOR, lw=2.5))
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.suptitle(f"Chunk text injection × MAX_WORDS ({PS.QUERY_NOTE}{data.get('scope','')}; red = shipped)")
    PS.save(fig, "s2_textmode")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    subset = None if a.full else a.subset
    path = C.RESULTS_DIR / RESULT
    if path.exists() and not a.force:
        print("[s2-textmode] cached")
        data = C.load_raw(RESULT)
    else:
        data = compute(subset, a.force)
        C.save_raw(RESULT, data)
    data["scope"] = "" if a.full else f"; subset {subset} (directional)"
    grid = data["grid"]
    best = max(grid, key=lambda c: grid[c]["ndcg@10"])
    ship = f"{data['shipped_mw']}|{data['shipped_mode']}"
    H.gate(
        "B. text-mode × MAX_WORDS (slide 2) [index-affecting]",
        cand_ndcg=grid[best]["ndcg@10"], prod_ndcg=grid[ship]["ndcg@10"],
        cand_recall100=grid[best]["recall@100"], prod_recall100=grid[ship]["recall@100"],
        cand_params=f"{best} (dense-only)", prod_params=f"{ship} (dense-only)",
        note="dense-only joint grid; a pass flags full artifact rebuild",
    )
    plot(data)


if __name__ == "__main__":
    main()

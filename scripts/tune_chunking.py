"""Grid-search chunk.MAX_WORDS x chunk.SLIDING_FALLBACK_OVERLAP.

For each (max_words, overlap) pair: rebuild the index into a throwaway dir,
run the 50 public queries, and report NDCG@10 + first-stage recall so the two
chunking params can be tuned without touching the shipped artifacts/.

    # fast directional sweep on a GT-inclusive subset (default 3000 pages)
    python scripts/tune_chunking.py --max-words 100,120,140 --overlap 20,40,60

    # the number that actually counts (full 27k; ~45 min/combo on CPU, no GPU)
    python scripts/tune_chunking.py --full --max-words 120,140

CAVEAT: subset NDCG is DIRECTIONAL ONLY. The corpus is 27k pages; a subset has
far fewer distractors, so it has previously mis-ranked design choices vs. the
full corpus (see CLAUDE.md). Confirm a subset winner with --full before adopting.

SLIDING_FALLBACK_OVERLAP only affects single sentences longer than MAX_WORDS
(sliding-window split); on a corpus with few such sentences its effect is small
or zero — watch the chunk_count column to see whether it moves anything.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import time
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

import chunk  # noqa: E402  (tuned in place per combo)
from eval import load_query_file, mean_ndcg_at_k  # noqa: E402
from eval_retrieval import retrieval_metrics  # noqa: E402
from index import build_index, load_meta  # noqa: E402
from retrieve import search_batch  # noqa: E402
from utils import PUBLIC_QUERIES_PATH  # noqa: E402

EVAL_KS = (10, 20, 50)
POOL_DEPTH = max(EVAL_KS)


def _ints(csv_str: str) -> list[int]:
    return [int(x) for x in csv_str.split(",") if x.strip()]


def _run_combo(
    max_words: int,
    overlap: int,
    queries: list[str],
    gt: list[list[int]],
    subset: int | None,
) -> dict:
    """Build a fresh index with these params and eval the public queries."""
    chunk.MAX_WORDS = max_words
    chunk.SLIDING_FALLBACK_OVERLAP = overlap

    tmp = Path(tempfile.mkdtemp(prefix=f"tune_{max_words}_{overlap}_"))
    try:
        t0 = time.perf_counter()
        build_index(artifacts_dir=tmp, subset=subset)
        build_s = time.perf_counter() - t0
        n_chunks = int(load_meta(tmp)["num_vectors"])

        t1 = time.perf_counter()
        ranked = search_batch(queries, top_k=POOL_DEPTH, artifacts_dir=tmp)
        query_s = time.perf_counter() - t1

        ndcg = mean_ndcg_at_k(ranked, gt, k=10)
        ret = retrieval_metrics(ranked, [set(g) for g in gt], ks=EVAL_KS)
        return {
            "max_words": max_words,
            "overlap": overlap,
            "ndcg@10": round(ndcg, 4),
            "recall@10": round(ret.get("recall@10", 0.0), 4),
            "recall@20": round(ret.get("recall@20", 0.0), 4),
            "recall@50": round(ret.get("recall@50", 0.0), 4),
            "chunks": n_chunks,
            "build_s": round(build_s, 1),
            "query_s": round(query_s, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-words", default="100,120,140", help="comma list")
    ap.add_argument("--overlap", default="20,40,60", help="comma list")
    ap.add_argument(
        "--subset",
        type=int,
        default=3000,
        help="GT-inclusive subset size for fast iteration (ignored with --full)",
    )
    ap.add_argument("--full", action="store_true", help="build full 27k corpus")
    ap.add_argument("--out", default="tune_chunking_results.csv")
    args = ap.parse_args()

    subset = None if args.full else args.subset
    max_words_grid = _ints(args.max_words)
    overlap_grid = _ints(args.overlap)

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    scope = "FULL 27k" if args.full else f"subset={subset} (DIRECTIONAL ONLY)"
    print(f"[tune] scope: {scope}")
    print(f"[tune] grid: max_words={max_words_grid} x overlap={overlap_grid}")

    results: list[dict] = []
    for mw in max_words_grid:
        for ov in overlap_grid:
            if ov >= mw:
                print(f"[tune] skip max_words={mw} overlap={ov} (overlap>=max_words)")
                continue
            print(f"[tune] building max_words={mw} overlap={ov} ...", flush=True)
            res = _run_combo(mw, ov, queries, gt, subset)
            results.append(res)
            print(
                f"       NDCG@10={res['ndcg@10']:.4f}  "
                f"R@10={res['recall@10']:.4f} R@50={res['recall@50']:.4f}  "
                f"chunks={res['chunks']}  build={res['build_s']}s"
            )

    results.sort(key=lambda r: r["ndcg@10"], reverse=True)

    print(f"\n=== results ({scope}), best first ===")
    hdr = ["max_words", "overlap", "ndcg@10", "recall@10", "recall@20",
           "recall@50", "chunks", "build_s", "query_s"]
    print("  ".join(f"{h:>9}" for h in hdr))
    for r in results:
        print("  ".join(f"{r[h]!s:>9}" for h in hdr))

    out_path = STUDENT_ROOT / args.out
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(results)
    print(f"\n[tune] wrote {out_path}")
    if results:
        b = results[0]
        print(f"[tune] best: max_words={b['max_words']} overlap={b['overlap']} "
              f"NDCG@10={b['ndcg@10']}")
        if not args.full:
            print("[tune] confirm on --full before editing chunk.py defaults.")


if __name__ == "__main__":
    main()

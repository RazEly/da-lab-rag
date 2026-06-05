"""Sweep chunking strategy x prepend policy, report mean NDCG@10 on public queries.

Matrix (2 x 3 = 6 builds):
    strategy : sliding_para     (overlap, never across a paragraph break)
               sliding_overlap  (overlap at every size-cut, incl. across paragraphs)
    prepend  : none | title | title_section

Each config is built into its own temp artifacts dir (the committed artifacts/ is
never touched) and evaluated with a search_batch closure. Run the heavy full build
on a strong machine:

    python sweep_chunking.py --full          # all 27k pages (slow: 6 full embeds)
    python sweep_chunking.py --subset 500    # GT-inclusive dev build (fast, directional)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import load_query_file, mean_ndcg_at_k
from eval_retrieval import DEFAULT_KS, retrieval_metrics
from index import build_index
from retrieve import search_batch
from utils import PUBLIC_QUERIES_PATH

STRATEGIES = ["sliding_para", "sliding_overlap"]
PREPENDS = ["none", "title", "title_section"]
POOL_DEPTH = max(DEFAULT_KS)  # retrieve this deep so recall@max(ks) is exact


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true", help="build on the full 27k corpus")
    g.add_argument("--subset", type=int, default=500, help="GT-inclusive subset size")
    args = ap.parse_args()
    subset = None if args.full else args.subset
    scope = "FULL 27k" if args.full else f"subset={subset} (directional)"

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).resolve().parent / f"sweep_results_{stamp}.txt"
    out_fh = out_path.open("w", encoding="utf-8")

    def emit(line: str = "") -> None:
        """Print to stdout AND append to the results file (flushed, crash-durable)."""
        print(line)
        out_fh.write(line + "\n")
        out_fh.flush()

    def fmt_row(tag, ndcg, ret, dt) -> str:
        return (
            f"  {tag:30s} {ndcg:8.4f} {ret['recall@50']:6.3f} {ret['recall@100']:6.3f} "
            f"{ret['hit@50']:7.3f} {ret[f'mrr@{POOL_DEPTH}']:6.3f} {ret['r_precision']:6.3f} "
            f"{ret.get('first_rel_rank_median', float('nan')):6.1f} {dt:6.2f}s"
        )

    hdr = (
        f"  {'strategy / prepend':30s} {'NDCG@10':>8s} {'R@50':>6s} {'R@100':>6s} "
        f"{'Hit@50':>7s} {'MRR':>6s} {'Rprec':>6s} {'medRk':>6s} {'qtime':>7s}"
    )

    results = []
    try:
        emit(f"# chunking sweep ({scope})  started {stamp}")
        emit(f"# pool_depth={POOL_DEPTH}  queries={len(queries)}")
        emit("\n# per-config results (in run order; failures marked ERR):")
        emit(hdr)
        for strat in STRATEGIES:
            for prep in PREPENDS:
                d = Path(tempfile.mkdtemp(prefix=f"sweep_{strat}_{prep}_"))
                tag = f"{strat} / {prep}"
                print(f"\n### building [{tag}]  subset={subset}  -> {d}", flush=True)
                try:
                    build_index(
                        chunking_strategy=strat,
                        prepend=prep,
                        artifacts_dir=d,
                        subset=subset,
                    )
                    t0 = time.perf_counter()
                    # one deep retrieval feeds both NDCG@10 and first-stage metrics
                    ranked = search_batch(queries, top_k=POOL_DEPTH, artifacts_dir=d)
                    dt = time.perf_counter() - t0
                    ndcg = mean_ndcg_at_k(ranked, gt, k=10)
                    ret = retrieval_metrics(ranked, gt)
                    results.append((tag, ndcg, ret, dt))
                    emit(fmt_row(tag, ndcg, ret, dt))  # durable: written as it finishes
                except Exception as exc:  # one bad config must not kill the sweep
                    emit(f"  {tag:30s}  ERR: {exc!r}")
                finally:
                    shutil.rmtree(d, ignore_errors=True)

        emit(f"\n================ summary, ranked by recall@50 ({scope}) ================")
        emit(hdr)
        for row in sorted(results, key=lambda r: -r[2].get("recall@50", 0)):
            emit(fmt_row(*row))
    finally:
        out_fh.close()
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()

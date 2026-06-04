"""
Section B entry point.

The autograder calls run(queries) once with all evaluation queries (batch of 50).
Query embedding + retrieval must complete within the time limit (GPU available).
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """
    Rank corpus pages for each query.

    Parameters
    ----------
    queries : list[str]
        Batch of query strings (e.g. 50 hidden queries at grading time).

    Returns
    -------
    list[list[int]]
        One ranked list of page_id per query (most relevant first).
        Only the first 10 IDs per list are scored.
    """
    return search_batch(queries)


def build_offline_index(
    chunking_strategy: str = "semantic",
    subset: int | None = None,
) -> None:
    """Run once locally to create artifacts/ (not timed at grading).

    chunking_strategy: ``"semantic"`` (default) or ``"sliding"`` (debugging).
    subset: index only N pages (ground-truth-inclusive) for fast local dev.
            Leave None for the real submission build.
    """
    build_index(chunking_strategy=chunking_strategy, subset=subset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", default="semantic", choices=["semantic", "sliding"],
        help="Chunking strategy (default: semantic)"
    )
    parser.add_argument(
        "--subset", type=int, default=None, metavar="N",
        help="Index only N pages (always includes query ground-truth pages) "
             "for fast local iteration. Omit for the full submission build."
    )
    args = parser.parse_args()
    build_offline_index(chunking_strategy=args.strategy, subset=args.subset)
    print("Index built under artifacts/. Run: python scripts/eval_public.py")

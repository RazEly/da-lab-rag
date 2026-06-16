"""Section B entry point.

The autograder calls run(queries) once with all 50 eval queries; query embedding
+ retrieval must finish within the time limit (GPU available).
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Rank corpus pages per query → one page_id list each (best first).

    Only the first 10 IDs per list are scored.
    """
    return search_batch(queries)


def build_offline_index(subset: int | None = None) -> None:
    """Run once locally to create artifacts/ (untimed).

    subset: index only N pages (GT-inclusive) for fast local dev; None = full build.
    """
    build_index(subset=subset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset", type=int, default=None, metavar="N",
        help="Index only N pages (always includes query ground-truth pages) "
             "for fast local iteration. Omit for the full submission build."
    )
    args = parser.parse_args()
    build_offline_index(subset=args.subset)
    print("Index built under artifacts/. Run: python scripts/eval_public.py")

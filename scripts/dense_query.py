"""Run a single query against the DENSE retriever only and print top page scores.

Dense-only: no BM25, no RRF fusion. Uses the same FAISS IndexFlatIP search and
per-page max+mean aggregation as the shipped pipeline (retrieve.DENSE_ALPHA/TOPN),
so scores match dense's contribution inside the real run().

Usage (from part-2/):
    .venv/bin/python scripts/dense_query.py "who coached the LA franchise in the 80s"
    .venv/bin/python scripts/dense_query.py -k 20 "your query here"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/dense_query.py` from part-2/ to import the modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embed import embed_queries
from index import load_index, load_meta
from retrieve import DENSE_ALPHA, DENSE_TOPN, _load_faiss
from utils import ARTIFACTS_DIR


def dense_search(query: str, top_k: int):
    """Return [(page_id, score, title), ...] ranked by dense page score."""
    corpus_vectors, page_ids = load_index(ARTIFACTS_DIR)
    meta = load_meta(ARTIFACTS_DIR)
    titles = meta["page_titles"]  # aligned with chunk rows / page_ids

    qvec = embed_queries([query])
    if qvec.size == 0:
        return []

    num_chunks = corpus_vectors.shape[0]

    # Dense chunk scores (cosine == dot, vectors are L2-normalized).
    faiss_idx = _load_faiss(ARTIFACTS_DIR, corpus_vectors)
    if faiss_idx is not None:
        D, I = faiss_idx.search(qvec, num_chunks)
        chunk_scores = np.zeros(num_chunks, dtype=np.float32)
        chunk_scores[I[0]] = D[0]
    else:
        chunk_scores = (qvec @ corpus_vectors.T)[0]

    # Per-page max+mean blend (same as retrieve._aggregate_page_scores).
    page_chunks: dict[int, list[float]] = {}
    title_of: dict[int, str] = {}
    for idx, score in enumerate(chunk_scores):
        pid = page_ids[idx]
        page_chunks.setdefault(pid, []).append(float(score))
        title_of.setdefault(pid, titles[idx])

    page_scores: dict[int, float] = {}
    for pid, chunks in page_chunks.items():
        ranked = sorted(chunks, reverse=True)
        topn = ranked[:DENSE_TOPN]
        page_scores[pid] = DENSE_ALPHA * ranked[0] + (1.0 - DENSE_ALPHA) * (
            sum(topn) / len(topn)
        )

    ranked_pids = sorted(page_scores, key=page_scores.__getitem__, reverse=True)
    return [(pid, page_scores[pid], title_of[pid]) for pid in ranked_pids[:top_k]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dense-only single-query retrieval.")
    ap.add_argument("query", help="query string")
    ap.add_argument("-k", "--top-k", type=int, default=10, help="results to show")
    args = ap.parse_args()

    results = dense_search(args.query, args.top_k)
    print(f"\nquery: {args.query!r}  (dense-only, alpha={DENSE_ALPHA}, topn={DENSE_TOPN})\n")
    print(f"{'rank':>4}  {'page_id':>8}  {'score':>8}  title")
    for rank, (pid, score, title) in enumerate(results, 1):
        print(f"{rank:>4}  {pid:>8}  {score:>8.4f}  {title}")


if __name__ == "__main__":
    main()

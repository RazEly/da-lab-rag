"""BM25 sparse index: build, load, score. Numpy-only (no scipy)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import ARTIFACTS_DIR

BM25_INDEX_NAME = "bm25.npz"

# b=0.5 (lower than default 0.75): chunks have uniform length (~40-140 words),
# so length normalization contributes little — standard b over-penalises short chunks.
K1: float = 1.5
B: float = 0.5

_PUNCT_RE = re.compile(r"[^\w\s]")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return _PUNCT_RE.sub(" ", text.lower()).split()


@dataclass
class BM25Index:
    vocab: np.ndarray            # (V,) object — term strings
    idf: np.ndarray              # (V,) float32 — Robertson IDF per term
    chunk_ids: np.ndarray        # COO int32 — row indices
    term_ids: np.ndarray         # COO int32 — col indices
    tf_weights: np.ndarray       # COO float32 — BM25 TF-saturation values
    num_chunks: int
    term_to_id: Dict[str, int] = field(default_factory=dict)
    inverted: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def build_bm25(chunks, out_dir: Optional[Path] = None) -> None:
    """
    Build BM25 index from a list of Chunk objects and save to out_dir/bm25.npz.

    BM25 document text is title + chunk.text. For short chunks chunk.text already
    has title prepended; the minor duplication is acceptable (slight title boost).
    """
    out_dir = out_dir or ARTIFACTS_DIR
    n = len(chunks)
    print(f"[bm25] tokenizing {n} chunks ...")

    docs: List[List[str]] = [tokenize(f"{c.title} {c.text}") for c in chunks]

    doc_lengths = np.array([len(d) for d in docs], dtype=np.float32)
    avgdl = float(doc_lengths.mean()) if n > 0 else 1.0

    # Document frequencies
    df: Dict[str, int] = {}
    for tokens in docs:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    vocab_list = sorted(df.keys())
    term_to_id = {t: i for i, t in enumerate(vocab_list)}
    V = len(vocab_list)
    print(f"[bm25] vocab size: {V}")

    # Robertson IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    idf = np.array(
        [math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0) for t in vocab_list],
        dtype=np.float32,
    )

    # Build COO sparse matrix of TF-saturation weights
    print(f"[bm25] building sparse TF matrix ...")
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for chunk_idx, tokens in enumerate(docs):
        dl = float(doc_lengths[chunk_idx])
        tf_raw: Dict[int, int] = {}
        for t in tokens:
            tid = term_to_id.get(t)
            if tid is not None:
                tf_raw[tid] = tf_raw.get(tid, 0) + 1
        for tid, freq in tf_raw.items():
            tf_sat = freq * (K1 + 1.0) / (freq + K1 * (1.0 - B + B * dl / avgdl))
            rows.append(chunk_idx)
            cols.append(tid)
            vals.append(tf_sat)

    nnz = len(rows)
    print(f"[bm25] saving {nnz} non-zeros → {out_dir / BM25_INDEX_NAME}")
    np.savez(
        out_dir / BM25_INDEX_NAME,
        vocab=np.array(vocab_list, dtype=object),
        idf=idf,
        chunk_ids=np.array(rows, dtype=np.int32),
        term_ids=np.array(cols, dtype=np.int32),
        tf_weights=np.array(vals, dtype=np.float32),
        num_chunks=np.array(n, dtype=np.int64),
    )


def load_bm25(artifacts_dir: Optional[Path] = None) -> BM25Index:
    """Load bm25.npz and build an in-memory inverted index for O(1) term lookup."""
    root = artifacts_dir or ARTIFACTS_DIR
    data = np.load(root / BM25_INDEX_NAME, allow_pickle=True)

    vocab = data["vocab"]
    idf = data["idf"].astype(np.float32)
    chunk_ids = data["chunk_ids"]
    term_ids = data["term_ids"]
    tf_weights = data["tf_weights"].astype(np.float32)
    num_chunks = int(data["num_chunks"])

    term_to_id: Dict[str, int] = {str(t): int(i) for i, t in enumerate(vocab)}

    # Group COO by term_id → inverted index
    order = np.argsort(term_ids, kind="stable")
    s_terms = term_ids[order]
    s_chunks = chunk_ids[order]
    s_vals = tf_weights[order]

    unique_terms, starts = np.unique(s_terms, return_index=True)
    ends = np.append(starts[1:], len(s_terms))

    inverted: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        int(uid): (s_chunks[start:end], s_vals[start:end])
        for uid, start, end in zip(unique_terms, starts, ends)
    }

    return BM25Index(
        vocab=vocab,
        idf=idf,
        chunk_ids=chunk_ids,
        term_ids=term_ids,
        tf_weights=tf_weights,
        num_chunks=num_chunks,
        term_to_id=term_to_id,
        inverted=inverted,
    )


def score_batch(bm25: BM25Index, queries: List[str]) -> np.ndarray:
    """
    Return raw BM25 scores, shape (n_queries, num_chunks).

    For each query term, looks up its posting list and accumulates idf * tf_sat
    into the chunk score vector. Duplicate query tokens counted once (BM25 standard).
    """
    n_queries = len(queries)
    out = np.zeros((n_queries, bm25.num_chunks), dtype=np.float32)

    for qi, query in enumerate(queries):
        tokens = tokenize(query)
        seen: set = set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            tid = bm25.term_to_id.get(t)
            if tid is None:
                continue
            cids, tfs = bm25.inverted[tid]
            out[qi, cids] += bm25.idf[tid] * tfs

    return out

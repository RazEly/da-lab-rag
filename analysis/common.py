"""Shared plumbing for the analysis package.

Imports production modules (never copies their logic) so every sweep computes
fusion/aggregation exactly as ``retrieve.py`` ships it. Caches the expensive
artifacts (dense/BM25 score matrices) as ``.npy`` keyed by (index_tag, variant).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"
INDEXES_DIR = ANALYSIS_DIR / "indexes"
CACHE_DIR = ANALYSIS_DIR / "cache"


def bootstrap() -> None:
    """Put the part-2 repo root on sys.path so production modules import."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    for d in (RESULTS_DIR, FIGURES_DIR, INDEXES_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


bootstrap()

# Production imports (after bootstrap).
from bm25 import load_bm25, score_batch as bm25_score_batch  # noqa: E402
from embed import embed_queries  # noqa: E402
from eval import load_query_file  # noqa: E402
from index import (  # noqa: E402
    INDEX_VECTORS_NAME,
    load_index,
    load_meta,
)
from retrieve import (  # noqa: E402
    _aggregate_page_scores,
    _allowed_pages_for_query,
    _build_page_numbers,
    _load_faiss,
    _rrf,
)
from utils import ARTIFACTS_DIR, PUBLIC_QUERIES_PATH  # noqa: E402

# Fusion math reused verbatim from the dev sweep harness (symmetric-depth variant).
from scripts.sweep_retrieval import fuse  # noqa: E402,F401


# --------------------------------------------------------------------------- #
# Eval data
# --------------------------------------------------------------------------- #
def load_eval(path: Optional[Path] = None) -> Tuple[List[str], List[Set[int]]]:
    """Return (queries, ground_truth_sets) from the public query file."""
    rows = load_query_file(path or PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [set(r["relevant_page_ids"]) for r in rows]
    return queries, gt


# --------------------------------------------------------------------------- #
# Raw IO
# --------------------------------------------------------------------------- #
def save_raw(name: str, obj: Any) -> Path:
    """Persist a json result under results/ (source of truth for plots)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_raw(name: str) -> Any:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def save_csv(name: str, rows: Sequence[Dict[str, Any]]) -> Path:
    """Flat CSV dump of a list of dict rows (header = union of keys)."""
    import csv

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


# --------------------------------------------------------------------------- #
# Score-matrix cache  (key = index_tag + query_variant + query hash)
# --------------------------------------------------------------------------- #
def _qhash(queries: Sequence[str]) -> str:
    h = hashlib.sha1("\n".join(queries).encode("utf-8")).hexdigest()
    return h[:12]


def _cache_path(kind: str, tag: str, variant: str, queries: Sequence[str]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{kind}__{tag}__{variant}__{_qhash(queries)}.npy"


def dense_score_matrix(
    index_dir: Path,
    queries: Sequence[str],
    *,
    tag: str,
    variant: str = "base",
    cache: bool = True,
    force: bool = False,
) -> np.ndarray:
    """(Q, C) dense cosine scores for an index dir. Cached as .npy."""
    cpath = _cache_path("dense", tag, variant, queries)
    if cache and cpath.exists() and not force:
        return np.load(cpath)
    vectors = np.load(index_dir / INDEX_VECTORS_NAME)
    qv = embed_queries(list(queries))
    num_chunks = vectors.shape[0]
    faiss_idx = _load_faiss(index_dir, vectors)
    if faiss_idx is not None:
        D, I = faiss_idx.search(qv, num_chunks)
        ds = np.zeros((len(queries), num_chunks), dtype=np.float32)
        for qi in range(len(queries)):
            ds[qi, I[qi]] = D[qi]
    else:
        ds = (qv @ vectors.T).astype(np.float32)
    if cache:
        np.save(cpath, ds)
    return ds


def bm25_score_matrix(
    queries: Sequence[str],
    *,
    tag: str = "prod",
    variant: str = "base",
    index_dir: Optional[Path] = None,
    min_idf: float = 0.0,
    fallback: float = 0.0,
    cache: bool = True,
    force: bool = False,
) -> np.ndarray:
    """(Q, C) BM25 scores from a bm25.npz. Cached per (idf, fallback)."""
    vtag = f"{variant}_idf{min_idf}_fb{fallback}"
    cpath = _cache_path("bm25", tag, vtag, queries)
    if cache and cpath.exists() and not force:
        return np.load(cpath)
    bm25 = load_bm25(index_dir or ARTIFACTS_DIR)
    ss = bm25_score_batch(
        bm25, list(queries), min_query_idf=min_idf, fallback_threshold=fallback
    )
    if cache:
        np.save(cpath, ss)
    return ss


def page_numbers_for(index_dir: Optional[Path] = None) -> Optional[Dict[int, Set[int]]]:
    """page_id → {numbers} map from an index's meta (None if absent)."""
    root = index_dir or ARTIFACTS_DIR
    meta = load_meta(root)
    page_ids = [int(x) for x in meta["page_ids"]]
    chunk_numbers = meta.get("chunk_numbers")
    if chunk_numbers is None:
        return None
    return _build_page_numbers(page_ids, chunk_numbers)


def page_ids_from_meta(index_dir: Optional[Path] = None) -> List[int]:
    """page_ids without loading the (large) vectors file."""
    meta = load_meta(index_dir or ARTIFACTS_DIR)
    return [int(x) for x in meta["page_ids"]]


def agg(
    chunk_scores: np.ndarray,
    page_ids: List[int],
    *,
    depth: int,
    alpha: float,
    topn: int,
    allowed: Optional[Set[int]] = None,
) -> List[int]:
    """Thin wrapper over production per-page aggregation."""
    return _aggregate_page_scores(
        chunk_scores, page_ids, depth, alpha, topn, allowed_pages=allowed
    )


def fuse_prod(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    page_ids: List[int],
    queries: Sequence[str],
    page_numbers: Optional[Dict[int, Set[int]]],
    *,
    dense_alpha: float,
    dense_topn: int,
    sparse_alpha: float,
    sparse_topn: int,
    rrf_k: int,
    rrf_depth: int,
    sparse_rrf_depth: int,
    filter_mode: str = "post",
    include_dense: bool = True,
    include_sparse: bool = True,
    top_k: int = 10,
) -> List[List[int]]:
    """Production-faithful fusion: mirrors retrieve.search_batch's per-query loop
    EXACTLY (asymmetric RRF depths, post/pre/off number filter, _rrf), with every
    knob overridable. Unlike sweep_retrieval.fuse this honours separate dense vs
    BM25 RRF depths, matching what ships.
    """
    results: List[List[int]] = []
    for qi in range(len(queries)):
        allowed = (
            _allowed_pages_for_query(queries[qi], page_numbers)
            if (page_numbers is not None and filter_mode != "off")
            else None
        )
        pre = allowed if filter_mode == "pre" else None
        rankings: List[List[int]] = []
        if include_dense:
            rankings.append(
                _aggregate_page_scores(
                    dense_scores[qi], page_ids, rrf_depth, dense_alpha, dense_topn,
                    allowed_pages=pre,
                )
            )
        if include_sparse:
            rankings.append(
                _aggregate_page_scores(
                    sparse_scores[qi], page_ids, sparse_rrf_depth, sparse_alpha,
                    sparse_topn, allowed_pages=pre,
                )
            )
        if filter_mode == "post" and allowed is not None:
            fused = _rrf(rankings, rrf_k, len(page_ids))
            kept = [p for p in fused if p in allowed]
            results.append((kept or fused)[:top_k])
        else:
            results.append(_rrf(rankings, rrf_k, top_k))
    return results


def prod_fuse_kwargs() -> Dict[str, Any]:
    """fuse_prod kwargs that reproduce the live production config."""
    p = current_prod_params()
    return {
        "dense_alpha": p["DENSE_ALPHA"],
        "dense_topn": p["DENSE_TOPN"],
        "sparse_alpha": p["SPARSE_ALPHA"],
        "sparse_topn": p["SPARSE_TOPN"],
        "rrf_k": p["RRF_K"],
        "rrf_depth": p["RRF_DEPTH"],
        "sparse_rrf_depth": p["SPARSE_RRF_DEPTH"],
        "filter_mode": p["FILTER_MODE"] if p["ENABLE_NUMBER_FILTER"] else "off",
    }


def invalidate_cache(tag: str) -> int:
    """Drop cached matrices made stale by a production change. Returns count."""
    n = 0
    for p in CACHE_DIR.glob(f"*__{tag}__*.npy"):
        p.unlink()
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Live production constants (gate always compares against the real module)
# --------------------------------------------------------------------------- #
def current_prod_params() -> Dict[str, Any]:
    import bm25 as _bm25
    import chunk as _chunk
    import retrieve as _r

    return {
        "DENSE_ALPHA": _r.DENSE_ALPHA,
        "DENSE_TOPN": _r.DENSE_TOPN,
        "SPARSE_ALPHA": _r.SPARSE_ALPHA,
        "SPARSE_TOPN": _r.SPARSE_TOPN,
        "RRF_K": _r.RRF_K,
        "RRF_DEPTH": _r.RRF_DEPTH,
        "SPARSE_RRF_DEPTH": _r.SPARSE_RRF_DEPTH,
        "BM25_QUERY_MIN_IDF": _r.BM25_QUERY_MIN_IDF,
        "BM25_FALLBACK_THRESHOLD": _r.BM25_FALLBACK_THRESHOLD,
        "FILTER_MODE": _r.FILTER_MODE,
        "ENABLE_NUMBER_FILTER": _r.ENABLE_NUMBER_FILTER,
        "MAX_WORDS": _chunk.MAX_WORDS,
        "K1": _bm25.K1,
        "B": _bm25.B,
    }

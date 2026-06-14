"""Shared run/gate plumbing for the step scripts (§0.1 update gate)."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

import common as C

CHANGELOG = C.RESULTS_DIR / "CHANGELOG.md"

# §0.1 gate thresholds.
NDCG_MIN_DELTA = 0.005  # candidate must beat prod NDCG@10 by ≥ this
WALL_SECONDS = 60.0


@dataclass
class ProdContext:
    queries: List[str]
    gt: List[Set[int]]
    page_ids: List[int]
    page_numbers: Optional[Dict[int, Set[int]]]
    dense: np.ndarray  # (Q,C) on prod artifacts
    # bm25 cached per (min_idf, fallback) on demand


def load_prod() -> ProdContext:
    queries, gt = C.load_eval()
    page_ids = C.page_ids_from_meta()
    page_numbers = C.page_numbers_for()
    dense = C.dense_score_matrix(C.ARTIFACTS_DIR, queries, tag="prod", variant="base")
    return ProdContext(queries, gt, page_ids, page_numbers, dense)


def prod_bm25(min_idf: float, fallback: float) -> np.ndarray:
    queries, _ = C.load_eval()
    return C.bm25_score_matrix(queries, tag="prod", min_idf=min_idf, fallback=fallback)


def gate(
    name: str,
    cand_ndcg: float,
    prod_ndcg: float,
    cand_recall100: float,
    prod_recall100: float,
    *,
    cand_params: str = "",
    prod_params: str = "",
    within_wall: bool = True,
    note: str = "",
) -> Tuple[bool, str]:
    """Apply §0.1: change only if ndcg win ≥ +0.005, no recall@100 regression,
    within the 60s wall. Returns (change?, reason)."""
    delta = cand_ndcg - prod_ndcg
    ok_ndcg = delta >= NDCG_MIN_DELTA
    ok_recall = cand_recall100 >= prod_recall100 - 1e-9
    change = bool(ok_ndcg and ok_recall and within_wall)
    if change:
        reason = f"CHANGE: NDCG {prod_ndcg:.4f}->{cand_ndcg:.4f} (Δ{delta:+.4f} ≥ {NDCG_MIN_DELTA}); recall@100 {prod_recall100:.4f}->{cand_recall100:.4f} ok; within wall"
    else:
        why = []
        if not ok_ndcg:
            why.append(f"ΔNDCG {delta:+.4f} < {NDCG_MIN_DELTA}")
        if not ok_recall:
            why.append(f"recall@100 regress {prod_recall100:.4f}->{cand_recall100:.4f}")
        if not within_wall:
            why.append("over 60s wall")
        reason = "KEEP: " + "; ".join(why)
    log_decision(name, change, reason, cand_params, prod_params, note)
    return change, reason


def log_decision(
    name: str,
    changed: bool,
    reason: str,
    cand_params: str,
    prod_params: str,
    note: str,
) -> None:
    CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
    if not CHANGELOG.exists():
        CHANGELOG.write_text(
            "# Analysis §0.1 gate decisions\n\n"
            "Per-step: candidate vs live production, change-if-better.\n\n",
            encoding="utf-8",
        )
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tag = "**CHANGED**" if changed else "kept"
    block = (
        f"## {name} — {tag}  ({ts})\n\n"
        f"- {reason}\n"
        f"- production: `{prod_params}`\n"
        f"- candidate:  `{cand_params}`\n"
    )
    if note:
        block += f"- note: {note}\n"
    block += "\n"
    # Idempotent per step: drop any prior block for the same step name, then append.
    text = CHANGELOG.read_text(encoding="utf-8")
    head = f"## {name} — "
    if head in text:
        import re as _re
        text = _re.sub(
            _re.escape(head) + r".*?(?=\n## |\Z)", "", text, flags=_re.S
        )
        text = _re.sub(r"\n{3,}", "\n\n", text)
        CHANGELOG.write_text(text.rstrip() + "\n\n", encoding="utf-8")
    with CHANGELOG.open("a", encoding="utf-8") as fh:
        fh.write(block)

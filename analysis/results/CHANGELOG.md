# Analysis §0.1 gate decisions

Per-step: candidate vs live production, change-if-better.

## D. per-retriever aggregation (slide 6) — kept  (2026-06-14 15:58)

- KEEP: ΔNDCG -0.0182 < 0.005; recall@100 regress 0.9174->0.8942
- production: `D(0.0, 100) S(0.25, 30)`
- candidate:  `D(0.0/100) S(0.75/100)`
- note: fused-optimal gate, not solo-optimal

## C. hybrid vs solo (slide 5) — kept  (2026-06-14 18:46)

- KEEP: ΔNDCG -0.0991 < 0.005; recall@100 regress 0.9174->0.8132
- production: `fused (shipped)`
- candidate:  `best solo leg`
- note: diagnostic; CHANGE here would mean disabling a leg (unexpected)

## E. RRF_K (slide 7) — **CHANGED** 60→28  (2026-06-14 16:03)

- CHANGE: NDCG 0.5363->0.5428 (Δ+0.0065 ≥ 0.005); recall@100 0.9174->0.9174 ok; within wall.
- production (before): `RRF_K=60` → (after): `RRF_K=28`
- sweep {5,10,20,28,60,100,200} → NDCG peaks at the 20–28 region (k20=0.5390,
  k28=0.5428), falls to 0.5363 by k=60. Applied to `retrieve.py`; re-verified
  `eval_public.py` (0.5428) + `validate_submission.py` (pass), 55s < 60s wall.
- re-gate at the new baseline (28 vs 28) confirms stable: no further change.

## F. BM25 IDF threshold (slide 7) — kept  (2026-06-14 18:57)

- KEEP: ΔNDCG +0.0000 < 0.005
- production: `min_idf=4.0`
- candidate:  `min_idf=4.0`
- note: prefer plateau centre over noisy peak

## G. number filter mode (slide 7) — kept  (2026-06-14 18:59)

- KEEP: ΔNDCG +0.0000 < 0.005
- production: `filter_mode=post`
- candidate:  `filter_mode=post`

## B. text-mode × MAX_WORDS (slide 2) [index-affecting] — kept  (2026-06-14 18:59)

- KEEP: ΔNDCG +0.0000 < 0.005
- production: `120|title (dense-only)`
- candidate:  `120|title (dense-only)`
- note: dense-only joint grid; a pass flags full artifact rebuild

## A. chunk MAX_WORDS (slide 2) [index-affecting] — kept  (2026-06-14 19:12)

- KEEP: ΔNDCG +0.0000 < 0.005
- production: `MAX_WORDS=120 (dense-only)`
- candidate:  `MAX_WORDS=120 (dense-only)`
- note: dense-only basis; a pass flags full artifact rebuild before shipping


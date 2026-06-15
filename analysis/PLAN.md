# Implementation Plan — Empirical Evidence + Production Tuning

This is an execution checklist for me. **Run each step, measure, and if the
empirical finding beats the current production setting, update production
immediately — then continue to the next step.** The analysis exists both to fill
every `[PLACEHOLDER]` in `part-2/presentation.md` AND to act as a tuning loop:
evidence that contradicts a shipped design decision triggers a production change,
not just a slide footnote.

No measured values are hardcoded here: each script *computes* them and persists
raw output. Steps run in order; a production change in an early step becomes the
baseline for every later step.

---

## 0. Hard rules

1. **Analysis code stays in `analysis/`.** Sweep/plot scripts live under
   `part-2/analysis/` and only *import* production modules — never copy logic into
   production. Production changes (below) are deliberate, minimal edits to the real
   modules (`chunk.py`, `retrieve.py`, `bm25.py`) or their artifacts, made through
   the §0.1 gate. Nothing else in production is touched.
2. **Raw-first.** Every script writes numeric output to `analysis/results/<name>.json`
   (+ `.csv` where tabular) **before** plotting. Plot code reads `results/` and
   never recomputes. Recomputing a sweep is opt-in (`--force`).
3. **Cache the expensive steps.** Index variants (per `MAX_WORDS`, per text mode)
   and the dense/BM25 score matrices (Q×C) are the cost. Build each index once into
   `analysis/indexes/<tag>/`; cache score matrices as `.npy`.
4. **Two scopes.** `--subset N` (GT-inclusive, fast, directional only) for
   iteration; `--full` (27k) for slides AND before any production change. A subset
   has far fewer distractors and can mis-rank choices — never change production off
   a subset result.
5. **Eval set.** `data/public_queries.json` = 29 queries (hidden grader uses 50).
   Label figures "29 public queries"; treat results as directional for the hidden
   score (this is why §0.1 has an overfitting guard).
6. **Plotting deps stay out of the graded path.** `matplotlib`/`pandas` only here;
   keep them out of `part-2/requirements.txt`; install into `analysis/.venv-analysis`.

### 0.1 Per-step update gate (run → evaluate → update-if-better → continue)

After each artifact's sweep, compare the measured best against the current
production setting and decide whether to change production:

1. **Compute the delta on `--full`** (the 29 public queries, full 27k corpus),
   primary metric = NDCG@10, with recall@10 as tiebreak. Subset results may
   nominate a candidate but never justify a change.
2. **Update only if the win is real**, all of:
   - candidate beats current production NDCG@10 by **≥ +0.005** (guard against
     noise on 29 queries / overfitting the public set), and
   - it does not regress recall@100 (don't trade away pool recall for top-10), and
   - it keeps `run()` within the 60 s wall.
   Ties or sub-threshold wins → **keep production**, record the result, move on.
3. **Apply the change** to the real module (see each artifact's "Production
   target"). Two change classes:
   - **param-only** (`retrieve.py` / `bm25.py` constants): no rebuild.
   - **index-affecting** (`chunk.py` `MAX_WORDS` or chunk text-mode): requires a
     **full artifact rebuild** (`scripts/build_index.py`) + re-commit via Git LFS.
     Flagged per artifact; only do it when the gate clearly passes (rebuild is
     expensive and changes the shipped artifacts).
4. **Re-verify after the edit:** rerun `scripts/eval_public.py` (and
   `eval_public_retrieval.py`) and confirm NDCG@10 moved as predicted and nothing
   regressed; confirm `validate_submission.py` still passes.
5. **Record + propagate:** append the decision (changed / kept, before→after,
   delta) to `analysis/results/CHANGELOG.md`. If production changed, update
   `CLAUDE.md` → Active hyperparameters / Current results, and invalidate any
   cached score matrices / indexes that the change makes stale so later steps
   rebaseline against the new production.
6. **Continue** to the next step against the updated production.

---

## 1. Folder layout

```
part-2/analysis/
  PLAN.md
  metrics.py              # self-contained recall@k / hit@k / mrr / r_precision / first_rel_rank
  common.py               # query+GT load, sys.path bootstrap, dense/bm25 score helpers, raw IO
  variant_index.py        # build a dense index under a chunk-text mode or MAX_WORDS into indexes/<tag>/
  plot_style.py           # shared mpl style; save(fig, name) -> figures/<name>.png + .svg
  results/                # RAW outputs (json + csv) + CHANGELOG.md — committed, source of truth
  figures/                # rendered png + svg — regenerated from results/
  indexes/                # cached throwaway dense indexes (gitignored; large)
  scripts/                # one script per artifact (see §4–§5)
  run_all.sh              # build indexes -> sweeps (each gated) -> plots
```

`analysis/indexes/` and `analysis/.venv-analysis/` → local `.gitignore`.
`results/` and `figures/` are committed.

---

## 2. `metrics.py` — self-contained (avoids a deleted dependency)

`scripts/eval_public_retrieval.py` and `tune_chunking.py` import
`eval_retrieval.retrieval_metrics`, but that source no longer exists (only stale
`.pyc`). Do not depend on it. Reimplement here, importing only `eval.py`
(read-only, allowed) for NDCG. Binary relevance, multi-page GT:

- `recall_at_k`, `hit_at_k`, `mrr`, `r_precision`, `first_rel_rank`
- `mean_recall_curve(all_ranked, all_gt, ks)`, `per_query_table(...)`
- NDCG@10 via `eval.mean_ndcg_at_k` (grader's exact definition)
- `KS = (1, 3, 5, 10, 20, 50, 100)` default ladder

Each function's output dumped to `results/` (json + flat csv).

## 3. `common.py` — shared plumbing

- `bootstrap()` — put `part-2` root on `sys.path`.
- `load_eval()` → `(queries, gt)` via `eval.load_query_file` (GT as sets).
- `dense_score_matrix(index_dir, queries, cache)` — `embed.embed_queries` + FAISS
  (`retrieve._load_faiss`) → `(Q,C)`; cache `.npy`, reload if present.
- `bm25_score_matrix(index_dir, queries, *, min_idf, fallback, cache)` —
  `bm25.load_bm25` + `bm25.score_batch`; cached.
- `agg(...)` wraps `retrieve._aggregate_page_scores`; `fuse_rrf(...)` reuses
  `sweep_retrieval.fuse` / `retrieve._rrf` so fusion math == production.
- `save_raw` / `load_raw`; `current_prod_params()` reads the live constants from
  `retrieve`/`chunk`/`bm25` so the gate always compares against real production.
- `invalidate_cache(tag)` — drop cached matrices/indexes made stale by a change.

Cache keys = `(index_tag, query_variant)` so matrices never collide.

## 4. `variant_index.py` — parametrized dense index builder

Dense-only index (vectors + meta; skip BM25 unless needed) into `indexes/<tag>/`,
reusing `chunk.chunk_corpus`, `embed.embed_texts`, `index.build_index`'s meta
schema. Idempotent: skip if `indexes/<tag>/index_vectors.npy` exists unless
`--force`. Honors `--subset N` / `--full`.

- **(a) `MAX_WORDS`.** Set `chunk.MAX_WORDS = mw` in-process, `chunk_corpus`,
  embed `[title] body`. Tag `mw{mw}`. Record `chunk_count`.
- **(b) text-mode.** `chunk_corpus` once, rewrite each chunk's text before
  embedding (body = strip `[title]` prefix; section = `chunk.section`): `vanilla`
  → `body`; `title` → `[title] body`; `title_header` → `[title] [section] body`
  (empty section → `[title] body`). Tag `mode_{vanilla|title|title_header}`.

These build throwaway indexes for *measurement*. Promoting a winner to production
is the index-affecting rebuild in §0.1 (rerun `build_index.py`, re-commit LFS).

---

## 5. Steps — each: sweep → §0.1 gate → (update prod if better) → plot

All metrics on the 29 public queries unless noted. Run in this order; each step
sees production as left by the previous step.

### A. Chunk `MAX_WORDS` (Slide 2, plot A) — index-affecting
- **Decision under test:** sliding-window chunks at `MAX_WORDS=120`.
- **Sweep:** dense-only recall@`KS` over `MAX_WORDS ∈ {80,120,160,200}`
  (`mode=title` index per value; dense agg = shipped dense setting). Record
  `chunk_count` (cost).
- **Production target:** `chunk.MAX_WORDS`. If a value clears the §0.1 gate at
  `--full`, set it and rebuild artifacts; else keep 120.
- `scripts/s2_chunk_maxwords.py` → `results/s2_maxwords.json` → line plot
  recall vs `k` (lines = `MAX_WORDS`, shipped marked) + twin axis chunk_count.
- *Deck note:* slide says "MIN_WORDS" — no such production knob; plot `MAX_WORDS`,
  footnote the rename.

### B. Title / header injection (Slide 2, plot B) — index-affecting
- **Decision under test:** embed `[title] body`; does adding the recovered section
  header (`[title] [section] body`) help further?
- **Sweep:** dense-only recall@`KS` for `vanilla` vs `title` vs `title_header`.
  Then **combine with A**: cross `MAX_WORDS × {none, title, title+header}` so the
  chunking + injection decision is made jointly (the winning cell is the candidate).
- **Production target:** chunk text assembly in `chunk._make_chunks` (currently
  `[title] body`). If `title_header` wins the gate, change the embedded/BM25 text
  to include the section and rebuild artifacts; if `vanilla` somehow wins, drop the
  title prefix. Else keep `[title] body`.
- `scripts/s2_chunk_textmode.py` → `results/s2_textmode.json` → line plot (3 lines)
  + the A×B grid heatmap; annotate recall@10 gaps.

### C. Hybrid beats either leg (Slide 5) — diagnostic (no single param)
- **Decision under test:** fuse dense + BM25 vs shipping one leg.
- **Sweep:** recall@{5,10,20} + NDCG@10 for dense-only, bm25-only, fused
  ("before reranking" = first-stage pool, `filter_mode="off"`, shipped agg).
- **Production action:** if one solo leg beats fused at the gate (unexpected),
  that's a structural finding — record it, and only then consider disabling a leg
  via `INCLUDE_DENSE`/`INCLUDE_SPARSE`. Normally confirms fusion → no change.
- `scripts/s5_retriever_grouped.py` (prod `artifacts/`, no rebuild; via
  `sweep_retrieval.fuse`) → `results/s5_retrievers.json` → grouped bar.

### D. Per-retriever aggregation (Slide 6) — param-only
- **Decision under test:** dense and BM25 use different `alpha`/`topn`
  (`DENSE_ALPHA/TOPN`, `SPARSE_ALPHA/TOPN`).
- **Sweep:** solo-retriever NDCG@10 heatmaps over `alpha ∈ {0,0.25,0.5,0.75,1.0}`
  × `topn ∈ {1,3,10,30,100}`, dense and BM25 separately; then evaluate the
  best-cell pair **fused** (solo-optimal ≠ fused-optimal — gate on the fused score).
- **Production target:** `DENSE_ALPHA/DENSE_TOPN`, `SPARSE_ALPHA/SPARSE_TOPN` in
  `retrieve.py`. Update the pair if the fused gate passes.
- `scripts/s6_aggregation_heatmap.py` → `results/s6_agg_{dense,bm25}.csv` → two
  heatmaps (shipped cell marked). If optima aren't distinct, report and revise slide.

### E. RRF vs score-blend; `RRF_K` (Slide 7) — param-only
- **Decision under test:** rank-fusion (RRF) over normalized score-blend; shipped
  `RRF_K`.
- **Sweep:** (1) NDCG@10 for dense-only, bm25-only, RRF-fused, min-max score-blend
  baseline. (2) NDCG@10 vs `RRF_K` grid (incl. shipped value).
- **Production target:** `RRF_K` (+ `RRF_DEPTH`/`SPARSE_RRF_DEPTH` if swept).
  Update if the curve's argmax clears the gate; else keep.
- `scripts/s7_rrf_ablation.py` → `results/s7_rrf.json` → bar + line.

### F. BM25 IDF filter — REMOVED (2026-06-15)
- A hard query-term IDF cutoff (`BM25_QUERY_MIN_IDF` / fallback) was tried and
  removed from the codebase. It only helped the 29-query public dev set it was fit
  on (+0.013 NDCG@10) and regressed both held-out sets — `llm_queries` (-0.004),
  `squad_queries` (-0.023). BM25's own Robertson-IDF weight already down-weights
  generic terms continuously; a hard cutoff (and a stopword-list variant, which
  lost on all three sets) only discarded small-but-nonzero signal. `bm25.score_batch`
  no longer accepts a filter argument. Record kept: `results/bm25_idf_filter_ab.json`,
  `bm25_idf_filter_sweep.json`, `bm25_stopwords_ab.json` (generating scripts deleted).

### G. Number filter mode (Slide 7) — param-only
- **Decision under test:** `FILTER_MODE="post"` vs `pre`/`off`.
- **Sweep:** NDCG@10 + recall@10 for `filter_mode ∈ {off, pre, post}`.
- **Production target:** `FILTER_MODE` / `ENABLE_NUMBER_FILTER`. Update if a mode
  clears the gate.
- `scripts/s7_number_filter.py` → `results/number_filter.json` → grouped bar.

### H–K. Supporting evidence (diagnostic; no production change expected)
- **H. Per-query NDCG@10 + zero-score queries** (`results/per_query_ndcg.csv`):
  box/strip + zero-score count; backs the remaining-headroom story.
- **I. First-relevant-rank CDF dense vs BM25** (`results/first_rel_rank.csv`):
  report measured medians/p90.
- **J. Latency vs 60 s wall** (`results/latency.json`): mean of ≥3 warm
  `search_batch` runs; bar with the 60 s line. (If a production change pushed time
  up, this is the regression check.)
- **K. Headline scorecard** (`results/headline.json`): final NDCG@10, recall@10/100,
  query time, corpus/chunk counts — computed from `results/`, fills Slide 1's blank;
  annotate the recall@100-vs-NDCG@10 reordering headroom.
- All in `scripts/extra_stats.py`. Rerun H–K last so they reflect the final shipped
  config after any A–G updates.

### Not empirical (no script)
- Slide 1 block diagram: hand-drawn. Slide 8 (extra dataset, LLM-as-judge):
  future-work narrative; LLM-judge needs an external model → out of scope.

---

## 6. Execution order (`run_all.sh`)

```
For A then B (index-affecting): build variant indexes (--full) -> sweep -> §0.1 gate
  -> if win: edit chunk.py + rebuild artifacts + re-verify + invalidate caches.
Then cache dense+bm25 score matrices for the (possibly rebuilt) prod artifacts.
For C..G (mostly param-only): sweep -> §0.1 gate -> if win: edit retrieve.py/bm25.py
  -> re-verify (eval_public + validate_submission).
Then H..K against the final config.
Finally: render all figures from results/.
```

Run A→K in order so each step baselines on the updated production. Every script:
`argparse` `--subset/--full`, `--force`, `--out-dir`; idempotent (present raw file
⇒ skip compute, just redraw). The gate decision for each step is logged to
`results/CHANGELOG.md`.

## 7. Done when

- [ ] `metrics.py` self-contained (no `eval_retrieval` import)
- [ ] every step A–K ran at `--full`; each has raw json/csv written before plots
- [ ] each step's §0.1 gate decision (changed/kept, before→after, delta) in
      `results/CHANGELOG.md`
- [ ] every production change re-verified (`eval_public.py`,
      `validate_submission.py`) with no regression and within 60 s
- [ ] `CLAUDE.md` (Active hyperparameters / Current results) reflects the final
      shipped config; slides + figures reflect it too
- [ ] `figures/` png+svg for each artifact, shipped settings marked, "29 queries" labelled
- [ ] `run_all.sh` reproduces the whole loop from a clean `indexes/`
```

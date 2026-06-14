# `analysis/` — empirical evidence + production tuning loop

Self-contained analysis package that (a) fills every empirical placeholder in
`../presentation.md` and (b) acts as a **per-step tuning loop**: each sweep is
compared against live production via the §0.1 gate and, if a candidate clears it,
production is changed and re-verified. See `PLAN.md` for the full spec.

Isolated from the graded path: these scripts only **import** production modules
(`chunk`, `retrieve`, `bm25`, `embed`, `index`, `eval`) — never copy their logic.
Plotting deps (`matplotlib`, `pandas`) are deliberately **kept out of**
`../requirements.txt`; they live in the project venv used to run the analysis only.

## Layout

```
metrics.py        self-contained recall/hit/mrr/r_precision/first_rel + NDCG via eval.py
common.py         query+GT load, dense/BM25 score-matrix cache, fuse_prod (== search_batch), raw IO, gate inputs
harness.py        load_prod(), prod_bm25(), §0.1 gate() + CHANGELOG writer (idempotent per step)
variant_index.py  parametrized dense-only index builder (MAX_WORDS / text-mode) into indexes/<tag>/
plot_style.py     shared mpl style + save(fig, name) -> figures/<name>.{png,svg}
scripts/          one script per artifact (A..K)
results/          RAW json/csv (source of truth) + CHANGELOG.md  (committed)
figures/          rendered png+svg (committed; regenerated from results/)
indexes/  cache/  throwaway dense indexes + score-matrix cache (gitignored)
run_all.sh        build indexes -> gated sweeps -> plots
```

## Run

```bash
# from part-2/analysis/, using the project venv (production deps + matplotlib/pandas)
bash run_all.sh --full          # full 27k corpus (param-only steps + diagnostics)
# index-affecting steps A/B are expensive on --full (each variant ≈ full re-embed);
# they default to directional subset evidence:
../.venv/bin/python scripts/s2_chunk_maxwords.py  --subset 6000
../.venv/bin/python scripts/s2_chunk_textmode.py  --subset 6000
```

Every script: `--full | --subset N`, `--force` (recompute; default reuses the raw
json), writes raw to `results/` **before** plotting. Re-running is idempotent
(present raw ⇒ skip compute, just redraw; gate decisions de-dup by step name).

## Steps → production target

| step | script | decision | production target | outcome |
|---|---|---|---|---|
| A | `s2_chunk_maxwords` | chunk `MAX_WORDS` | `chunk.MAX_WORDS` (index-affecting) | kept 120 |
| B | `s2_chunk_textmode` | title/header injection | `chunk._make_chunks` text (index-affecting) | kept `[title] body` |
| C | `s5_retriever_grouped` | fuse vs single leg | (diagnostic) | fused wins |
| D | `s6_aggregation_heatmap` | per-retriever alpha/topn | `DENSE_/SPARSE_ALPHA,TOPN` | kept |
| E | `s7_rrf_ablation` | RRF vs blend; `RRF_K` | `retrieve.RRF_K` | **changed 60 → 28** |
| F | `s7_idf_threshold` | BM25 IDF filter | `BM25_QUERY_MIN_IDF` | kept 4.0 |
| G | `s7_number_filter` | number filter mode | `FILTER_MODE` | kept `post` |
| H–K | `extra_stats` | per-query / first-rel / latency / headline | (diagnostic) | — |

## The §0.1 gate (`harness.gate`)

Change production only if **all** hold on the 29 public queries / full 27k corpus:
candidate beats prod NDCG@10 by **≥ +0.005**, no recall@100 regression, within the
60 s wall. Each decision (changed/kept, before→after, Δ) is appended to
`results/CHANGELOG.md`. Index-affecting wins (A/B) additionally require a full
`scripts/build_index.py` rebuild + LFS re-commit before shipping — the subset sweep
only nominates; it never promotes (subset = directional, far fewer distractors).

## Production change made by this loop

`retrieve.RRF_K` 60 → 28 (step E): NDCG@10 0.5363 → **0.5428** on 29 public queries,
recall@100 unchanged, 55 s < 60 s wall; re-verified with `scripts/eval_public.py`
and `validate_submission.py`. Full provenance in `results/CHANGELOG.md`.

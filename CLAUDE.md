# Project A — Section B (Retrieval Pipeline)

This file documents **Section B only**. Section A is out of scope here.

## What Section B is

End-to-end retrieval over ~27k Wikipedia-style JSON pages. The autograder calls
`run(queries: list[str]) -> list[list[int]]` once with 50 hidden queries and
scores **mean NDCG@10** with binary relevance. Multiple pages may be relevant
per query. Submission is a public GitHub repo containing code **and** a
prebuilt index under `artifacts/`.

- Submission deadline: **2026-06-12, 22:55**
- Code lives in `part-2/` (a git repo).
- Submit a `Repo Link` txt file pointing to the public GitHub repo.

## Hard constraints

- **Embedding model**: only `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- **Imports**: stdlib + `numpy`, `sentence-transformers`, `faiss-cpu` (or `faiss`). No other packages in the graded path.
- **Grading runtime**: one `run(queries)` call must finish in **≤60 s on GPU**, including query embedding and retrieval. Index build (`scripts/build_index.py`) is untimed.
- **Artifacts must ship in the repo**: the grader does not rebuild. `scripts/eval_public.py` must succeed on a fresh clone *without* rebuilding the index.
- `eval.py`, `scripts/eval_public.py`, `scripts/build_index.py` are read-only.
- Use Git LFS for any large artifact files.

## Scoring (Section B = 50% of total grade)

- **Functional (50% of B)**: mean NDCG@10 over 50 hidden queries, ranked vs. peers (1st = +5 bonus, 2nd = +3, 3rd = 100%, 4th+ = `min(1, NDCG_team/NDCG_3rd) * 100%`).
- **GitHub repo (25% of B)**: seamless eval run (50), modular code (20), README (10), in-code docs (10), pair collaboration in git history (10).

## Current results

Measured on the 29 labelled public queries (`scripts/eval_public.py`), committed
`artifacts/`, no rebuild (no-filter BM25, 2026-06-15):

| Metric | Value |
|---|---|
| mean NDCG@10 | **0.5297** |
| query-phase time | ~39 s (under the 60 s wall; includes artifact load) |
| recall@10 / @50 / @100 | 0.6321 / 0.8539 / 0.9338 |
| hit@10 / @50 | 0.8621 / 0.9655 |
| mrr@100 | 0.5404 |
| first relevant rank | median 2, p90 12, found_any 1.00 |
| zero-score queries | 4 / 29 |

First-stage retrieval almost always finds a relevant page in the top 100
(recall@100 ≈ 0.92) but NDCG@10 trails it — the remaining headroom is in the
**final ordering of the retrieved pool**, not in recall.

> The public dev set has **29** queries; the hidden grading set has 50. Treat
> public NDCG as directional for the hidden score.

## Files (`part-2/`)

| File | Role |
|---|---|
| `main.py` | `run()` entrypoint + `build_offline_index()` |
| `chunk.py` | word-count sliding-window chunking; `Chunk` w/ `section`/`section_number`/`numbers` metadata; dense+BM25 text = `[title] body` (`MAX_WORDS=120`, paragraph-aware 1-sentence overlap) |
| `headers.py` | recovers flattened `== Section ==` headings from paragraph tails (closed-class filter, ~95% precision); feeds `chunk.peel_headers` |
| `embed.py` | MiniLM wrapper, L2-normalized; `embed_texts` == `embed_queries` (symmetric model) |
| `index.py` | builds `index_vectors.npy` + `index_meta.json` + `bm25.npz` + `faiss.index`; `--subset N` GT-inclusive dev build |
| `retrieve.py` | `search_batch`: FAISS `IndexFlatIP` dense + BM25 sparse, per-retriever `max+mean` agg, RRF fusion; number post-filter |
| `bm25.py` | numpy-only BM25 (k1=1.5, b=0.5): build/load/score, inverted index; `score_batch` supports `min_query_idf` + `fallback_threshold` |
| `utils.py` | paths, corpus iteration, `entry_text(record)`, `load_public_queries`, `normalize_page_id` |
| `eval.py` | read-only NDCG@10 scorer (`mean_ndcg_at_k`, `load_query_file`) |
| `scripts/` | `build_index.py`, `eval_public.py` (read-only); `eval_public_retrieval.py`, `sweep_retrieval.py`, `tune_chunking.py`, `dense_query.py`, `dump_chunks.py` (dev tooling) |
| `analysis/` | presentation empirics (plots/stats); isolated from the graded path — see `analysis/PLAN.md` |
| `data/Wikipedia Entries/` | 27,074 JSON pages (`page_id`, `title`, `content`) — corpus |
| `data/public_queries.json` | 29 labelled public queries (dev set; binary relevance, multi-page GT) |
| `artifacts/` | `index_vectors.npy` (~576k chunk vecs) + `index_meta.json` + `bm25.npz` (committed via Git LFS); `faiss.index` gitignored, rebuilt in-memory at load |

## How the pipeline works

### Chunking (`chunk.py`)

Word-count sliding window, paragraph-aware, no embeddings at chunk time.

1. **Split**: `\n\n` → paragraphs; regex → sentences. Paragraph starts are **forced breaks**.
2. **Header recovery** (`headers.peel_headers`): wiki `== Section ==` markup was flattened into short phrases glued to paragraph tails. A closed-class filter (1–6 words, leading cap, ends `.`, no auxiliary/pronoun, sentence/title-case) peels these into `section`/`section_number` labels, **excluded from body** (~95% precision).
3. **Assemble**: greedily accumulate whole sentences until adding the next exceeds `MAX_WORDS = 120`, then cut. Each window pads **one** neighbor sentence per side as overlap, **never across a paragraph break**. A single over-long sentence is sliding-window split (`SLIDING_FALLBACK_OVERLAP = 40` words), emitted unpadded. Section tag = section of the window's first sentence.
4. **Text = `[title] body`** for every chunk. Title is prepended to the embedded text AND tokenized into BM25 (no separate title concat — would double-count). `numbers` (all integers via `\b\d+\b`) extracted from **body alone**.
5. `index_meta.json` carries (row-aligned with vectors): `page_ids`, `chunk_ids`, `page_titles`, `chunk_sections`, `chunk_section_numbers`, `chunk_word_counts`, `chunk_numbers`.

### Embed + index (`embed.py`, `index.py`)

MiniLM, L2-normalized, symmetric (query encoder == doc encoder). FAISS
`IndexFlatIP` (exact; IP == cosine for normalized vectors). `faiss.index` is
gitignored and rebuilt in-memory at load (`retrieve._load_faiss`, sub-second,
exact) so a stale on-disk index can never shadow fresh vectors.

### Retrieve + fuse (`retrieve.py`)

Two retrievers (dense + BM25) are ranked **independently to page level**, then
fused by **rank** (RRF), never by raw score — this sidesteps the dense/BM25
score-scale mismatch (cosine ∈ [−1,1] vs unbounded BM25 sums).

- **Per-retriever chunk→page aggregation** (`max+mean` blend; `alpha=1`→pure max, `0`→pure mean). Both legs' `alpha`/`topn` are tuned against the **fused** NDCG@10 (`scripts/sweep_retrieval.py`), not each leg's solo score — solo-optimal ≠ fused-optimal (analysis §D / `s6_aggregation`):
  - **Dense → mean** (`alpha=0.0`) — gold is a long, uniformly on-topic article; max picks one lucky chunk. (This is also dense's solo optimum.)
  - **BM25 → near-mean** (`alpha=0.25`, wider `topn=30`). BM25 *solo* peaks near max (`alpha≈0.75`: exact-match signal lives in one chunk), but fused it ships near-mean — a wider mean captures multi-chunk GT pages and complements dense (shipped fused 0.5363 vs solo-best pair fused 0.5182).
  - A single shared alpha starves BM25 — this is why an earlier shared-setting build scored far lower.
- **RRF**: a page at rank `r` (0-indexed) contributes `1/(RRF_K + r + 1)`; contributions summed across legs; re-sorted. `RRF_K` damps the top so agreement across both legs beats one leg's lucky #1. Depths are asymmetric (dense broad, BM25 tight).
- **No BM25 query-term filtering**: BM25 scores every query term. A hard IDF cutoff (`BM25_QUERY_MIN_IDF` / fallback) was tried and **removed** (2026-06-15) — it only helped the public dev set it was fit on and regressed both held-out sets; BM25's own IDF weight already down-weights generic terms.
- **Number post-filter**: every integer in a chunk body is stored per-chunk; unioned to `page_id → {numbers}` at load. Per query, pull all integers (`\b\d+\b`; decades like "1820s" expand to {1820…1829}). In `post` mode, RRF-rank the full corpus, then drop pages whose number set doesn't intersect the query's. Skipped if the query names no number, or no page carries it, or `chunk_numbers` is absent.

## Active hyperparameters

### `retrieve.py`

| Param | Value | Notes |
|---|---|---|
| `DENSE_ALPHA` / `DENSE_TOPN` | 0.0 / 100 | pure mean over dense chunks |
| `SPARSE_ALPHA` / `SPARSE_TOPN` | 0.25 / 30 | near-mean; wider mean captures multi-chunk GT pages |
| `RRF_K` | 28 | damps the top (analysis §0.1 gate 2026-06-14: 60→28, +0.0065 NDCG) |
| `RRF_DEPTH` | 100 | dense RRF candidate depth |
| `SPARSE_RRF_DEPTH` | 60 | BM25 RRF candidate depth |
| `ENABLE_NUMBER_FILTER` / `FILTER_MODE` | True / "post" | rank full pool, then drop non-number-matching pages |

### `chunk.py` / `bm25.py`

| Param | Value |
|---|---|
| `chunk.MAX_WORDS` | 120 |
| `chunk.SLIDING_FALLBACK_OVERLAP` | 40 |
| `bm25.K1` / `bm25.B` | 1.5 / 0.5 |

## Corpus & query analysis

**Page length** (sampled 500): min 4, median ~1,434, mean ~2,466, p90 ~6,523,
max ~16,601 words. MiniLM's 256-token limit ≈ 180 words, so whole-page
embedding would truncate most pages — chunking is required.

**Query patterns** (`data/public_queries.json`):

- **Type A**: single relevant page, highly specific paraphrased/obfuscated facts ("1820s" vs "1826", demonym shifts, franchise paraphrases).
- **Type B**: 2–4 relevant pages, relationship-style — each relevant page matches one facet. Top-10 must catch all of them.
- Several queries share identical text but different relevant pages — the model needs broad recall, not just top-1 precision.

## What's been ruled out (don't re-try)

- **Semantic-embedding chunking** (cosine breakpoint cuts): A/B-lost to the current sliding-window chunker; `chunk.py` computes no embeddings at chunk time.
- **Single shared aggregation alpha** on both retrievers: starves BM25.
- **Score-blend fusion** (normalize + weighted sum): collapses on the dense/BM25 scale mismatch — RRF is used instead.
- **Candidate-pool rerank** of the merged union: tied full RRF, no gain.
- **PRF** (centroid re-query): all variants regressed.
- **Asymmetric RRF k / heavy BM25 weighting**: recovering a few BM25-strong queries costs more overall.

## Remaining opportunities

- **Final-pool reordering**: recall@100 ≈ 0.92 but NDCG@10 ≈ 0.52 — the gap is the ordering of pages already retrieved. Biggest lever.
- **Zero-score queries**: a few queries whose GT ranks well in BM25 but poorly in dense never reach top-10 under fusion. No fix within `retrieve.py`/`bm25.py` alone — would need an index rebuild or model change.
- **Multi-granularity index** (sentence + paragraph, RRF-merged): only if an index rebuild is in budget.

## Submission readiness

- Commit `artifacts/index_vectors.npy` + `index_meta.json` + `bm25.npz` (Git LFS).
- README in `part-2/` must document: `pip install`, how to rebuild the index, artifact paths/formats, how to run `eval_public.py`. The rubric rewards "fresh clone runs without rebuild".
- Verify in a fresh checkout: clone → `pip install -r requirements.txt` → `python scripts/eval_public.py` works with no extra steps.
- Both partners need real commits in the git history.

## Environment

- Python venv at `part-2/.venv`. Always use `part-2/.venv/bin/python` (not system `pip`/`python`).

## Working notes

- `part-2/` is the working dir for graded code. `Project A.pdf` is the source of truth for scoring/grading questions.
- `chunk.py` shadows a stdlib name; that's the template's choice — don't rename.
- Don't add packages outside `numpy / sentence-transformers / faiss-cpu` to the graded path. The autograder will fail imports. (Plotting/analysis deps belong in `analysis/`, not `requirements.txt`.)
- Don't modify `eval.py`, `scripts/eval_public.py`, `scripts/build_index.py`.
- `run()` must return `list[list[int]]` of page_ids, deduped, max-10-scored.
- Measure on `data/public_queries.json` after every change; don't fly blind.

# Section B — Retrieval pipeline

End-to-end retrieval over ~27k Wikipedia-style JSON pages. The grader calls
`run(queries: list[str]) -> list[list[int]]` once and scores **mean NDCG@10**
(binary relevance, top-10 per query). A prebuilt index ships in `artifacts/` —
**the grader does not rebuild**; a fresh clone runs the eval directly.

## Setup

```bash
cd part-2
pip install -r requirements.txt
```

Dependencies (stdlib +): `numpy`, `sentence-transformers`, `faiss-cpu`.
Embedding model: **`sentence-transformers/all-MiniLM-L6-v2`** (384-dim, L2-normalized).

Corpus lives at **`data/Wikipedia Entries/`** (one JSON per page: `page_id`, `title`, `content`).

## Architecture

A **hybrid dense + sparse** retriever, fused by Reciprocal Rank Fusion (RRF):

1. **Chunking** (`chunk.py`) — sliding word-count window. Sentences are greedily
   accumulated into `MAX_WORDS=120` windows, each borrowing one neighbor sentence
   per side as overlap but never across a paragraph break (`\n\n`). Sentences over
   the cap fall back to a sliding split (`SLIDING_FALLBACK_OVERLAP=40`). Both the
   embedded text and the BM25 text are **`[title] body`** for every chunk — title
   tokens feed *both* retrievers. Per-chunk integer numbers are extracted
   (`extract_numbers`) and stored for the query-time number filter.
2. **Dense** (`embed.py` + FAISS `IndexFlatIP`) — MiniLM embeddings, exact
   cosine (= inner product on normalized vectors).
3. **Sparse** (`bm25.py`) — numpy-only BM25 indexed over the same `[title] body`
   text, so title/entity signal is in both retrievers.
4. **Fusion** (`retrieve.py`) — each retriever ranks pages independently via a
   `max+mean` chunk aggregation, then the two page rankings are RRF-merged
   (`RRF_K=60`, `RRF_DEPTH=200`). Once RRF-fused, both legs want mean-heavy
   aggregation:
   - **Dense**: pure mean (`DENSE_ALPHA=0.0`) over top `DENSE_TOPN=100` chunks.
   - **Sparse**: near-pure mean (`SPARSE_ALPHA=0.1`) over top `SPARSE_TOPN=12`.

   Both retrievers are enabled (`INCLUDE_DENSE`/`INCLUDE_SPARSE`); dense is a
   primary ranker, not a tie-breaker. RRF is immune to the dense/BM25 score-scale
   mismatch that collapses convex score-blending (the legacy `"blend"` path).
5. **Number filter** (`retrieve.py`) — if a query names integers, pages whose
   chunk metadata carries none of those numbers are pruned. `FILTER_MODE="post"`
   fuses the full pool first, then drops non-matching pages, falling back to the
   unfiltered order if the filter would empty the list.

`run()` (in `main.py`) is a thin wrapper over `search_batch()` and returns
deduped page-id lists (only the first 10 are scored).

## Build index (offline, untimed — your machine only)

Run once locally to (re)create `artifacts/`. **Commit these files** to the repo;
staff do not rebuild at grading time.

```bash
python scripts/build_index.py        # full submission build
# or, for fast local iteration on a GT-inclusive subset:
python main.py --subset 500
```

> Rebuild whenever `chunk.py` / `embed.py` / `index.py` change — the shipped
> vectors must match the current chunking logic.

## Artifacts (`artifacts/`)

| File | Format | Contents |
|---|---|---|
| `index_vectors.npy` | `np.float32 [N, 384]` | L2-normalized chunk embeddings, row `i` ↔ chunk `i` |
| `faiss.index` | `faiss.IndexFlatIP` | same vectors, for fast inner-product search |
| `index_meta.json` | JSON | per-row `page_ids`, `chunk_ids`, `page_titles`, `chunk_word_counts`, `chunk_numbers` (aligned to vector rows) |
| `bm25.npz` | `np.savez` | BM25 inverted index over `[title] body` (df, idf, term weights, doc lengths) |

Large artifacts are tracked with **Git LFS**.

## Public self-test

After building, verify a fresh run loads the submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

Reports mean NDCG@10 over `data/public_queries.json` (50 labelled queries) plus
the query-phase time. The full `run(queries)` call must finish in **≤60 s on GPU**.

---

## Optimizing the query-time pipeline

- **`retrieve.py`** — the whole timed path: load artifacts, embed queries, FAISS
  dense search + BM25 sparse search, chunk→page aggregation, RRF fusion, the
  number filter, and any reranking you add. `main.py:run()` is just a wrapper.

- **The eval loop** — run `python scripts/eval_public.py` after *every* change and
  record the NDCG@10 delta. Never change two knobs at once without measuring in
  between. `python eval_public_retrieval.py` reports the *pre-rerank pool* metrics
  (recall@k, hit@k, MRR, first-relevant rank) — use it to tell a *recall* problem
  (the relevant page isn't in the pool) from a *ranking* problem (it's in the pool
  but ranked too low). They need different fixes.

You can iterate **without ever rebuilding the index** — the committed `artifacts/`
are fixed inputs. Only touch `chunk.py` / `index.py` (Person A's files) if you've
proven the ceiling is in chunking, not ranking, and then you must rebuild + recommit.

### Measurement loop

```bash
# 1. edit a knob in retrieve.py
# 2. measure
python scripts/eval_public.py          # mean NDCG@10 + wall time  ← the number that matters
python eval_public_retrieval.py        # pool recall/MRR — diagnoses WHY a change helped/hurt
# 3. keep it only if NDCG@10 went up AND wall time stays well under 60 s
```

### Tunable hyperparameters (all in `retrieve.py`)

| Knob | Current | What it does |
|---|---|---|
| `DENSE_ALPHA`, `DENSE_TOPN` | `0.0`, `100` | dense page agg: `alpha*max + (1-alpha)*mean` over top-N chunks |
| `SPARSE_ALPHA`, `SPARSE_TOPN` | `0.1`, `12` | same blend, BM25 leg |
| `RRF_K` | `60` | RRF rank-discount constant (higher = flatter contribution) |
| `RRF_DEPTH` | `200` | how deep each retriever's ranking feeds the fusion |
| `INCLUDE_DENSE` / `INCLUDE_SPARSE` | `True` / `True` | toggle a leg off to A/B it solo |
| `FUSION` | `"rrf"` | `"rrf"` (rank fusion) or `"blend"` (legacy convex score blend) |
| `ENABLE_NUMBER_FILTER` | `True` | prune pages missing a number named in the query |
| `FILTER_MODE` | `"post"` | `"post"` (filter after fusion) or `"pre"` (filter the pool first) |

- Once RRF-fused, **both** legs want **mean-heavy** aggregation (not max). BM25's
  "pure max" only wins *solo*; in fusion a page with several on-topic chunks should
  outrank a one-lucky-keyword page (helps the multi-page Type-B queries).

### Dead ends

- **Candidate-pool rerank** (union top-50, rerank only the pool): *tied* full RRF. The pool was never the bottleneck.
- **Asymmetric / weighted RRF** (favor BM25's top): equal weight ≥ asymmetric here.
- **Convex score-blend** (`FUSION="blend"`): monotonically degrades to BM25-only —
  min-max stretches MiniLM's tightly-bunched cosines into noise on BM25's scale.
  This is why the pipeline uses rank fusion.

### Next directions

1. **Pseudo-relevance feedback (PRF).** Take the centroid of the top-k dense chunks,
   interpolate with the query vector (`0.7*q + 0.3*centroid`), re-normalize, do a
   second dense pass, RRF-merge both passes. Targets Type-B multi-page recall. Watch
   the 60 s wall — it's a second pass.
2. **Query variants (`S9`).** Lowercase + strip punctuation; strip leading question
   words ("who/what/which/when/where"); try the MiniLM asymmetric prefix
   `"Represent this sentence for searching relevant passages: {q}"`. Each is a 2-line
   change — measure and keep only what helps.
3. **Re-sweep fusion** as anything upstream changes — the current agg was tuned on 50
   public queries, so re-confirm the plateau is still flat before trusting it.
4. **Multi-granularity** (sentence + paragraph indexes, RRF-merged) — only if 1–3
   plateau; it needs a rebuild and grows the artifact.

## Validation and public error analysis

We added a lightweight validation and diagnostics workflow for Section B.

`validate_submission.py` checks that the required prebuilt artifacts are present, that the vector artifact is a real NumPy matrix rather than a Git LFS pointer, and that `run(queries)` returns valid top-10 page ID lists.

`analyze_public_errors.py` computes per-query NDCG@10, first relevant rank, relevant page IDs, and the system's returned top-10 page IDs. This helped identify whether weak cases were caused by retrieval failure or by ranking/reranking errors.

### Public baseline

| Metric | Value |
|---|---:|
| Public queries | 29 |
| Mean NDCG@10 | 0.5203 |
| Query phase time | 37.83s |
| Recall@10 | 0.6234 |
| Recall@50 | 0.8548 |
| Recall@100 | 0.9166 |
| Hit@100 | 1.0000 |
| MRR@100 | 0.5141 |
| First relevant rank | median 2, p90 12 |

The diagnostics suggest that the system usually retrieves relevant pages somewhere in the candidate pool, but some relevant pages are not ranked high enough in the final top 10. Therefore, the main opportunity for improvement is reranking rather than rebuilding the embedding index.


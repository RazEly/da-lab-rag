# End-to-End Retrieval Pipeline for Wikipedia-style Corpus

## Overview

Video overview of the project: <https://www.youtube.com/watch?v=zOn-_WBD7iE>

## Setup

Setting up the environment:

```bash
pip install -r requirements.txt
```

Running the evaluation:

```bash
python scripts/eval_public.py
```

Dependencies: `numpy`, `sentence-transformers`, `faiss-cpu`.
Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Project Structure

```
.
├── main.py             
├── chunk.py            
├── embed.py            
├── bm25.py             
├── index.py            
├── retrieve.py         
├── utils.py            # paths, corpus iteration, query loading, section-header recovery
├── eval.py             
├── requirements.txt    
├── scripts/
│   ├── build_index.py            
│   ├── eval_public.py            
│   ├── analyze_public_errors.py  
│   ├── enrich_validation.py      
│   ├── validate_submission.py    
│   ├── dense_query.py            # dev: ad-hoc dense query probe
│   └── dump_chunks.py            # dev: inspect chunker output
├── artifacts/          # prebuilt index, shipped via Git LFS 
│   ├── index_vectors.npy
│   ├── index_meta.json
│   └── bm25.npz        # faiss.index gitignored, rebuilt in-memory at load
├── data/
│   ├── public_queries.json       # 29 labelled dev queries
│   ├── llm_queries.json          # LLM-generated validation set
│   ├── squad_queries.json        # SQuAD v2 hybrid validation set
│   └── Wikipedia Entries/        # JSON corpus pages (not committed)
└── presentation/       # slides + figures
```

## Architecture

A **hybrid dense + sparse** retriever, fused by Reciprocal Rank Fusion (RRF):

1. **Chunking** (`chunk.py`) — sliding word-count window. Sentences are greedily
   accumulated into `MAX_WORDS=120` windows, each borrowing one neighbor sentence
   per side as overlap but never across a paragraph break (`\n\n`). Sentences over
   the cap fall back to a sliding split (`SLIDING_FALLBACK_OVERLAP=40`). Both the
   embedded text and the BM25 text are **`[title] body`** for every chunk — title
   tokens feed *both* retrievers. Per-chunk integer numbers are extracted
   (`extract_numbers`) and stored for the query-time number filter.
2. **Dense** (`embed.py` + FAISS `IndexFlatIP`) — MiniLM embedding.
3. **Sparse** (`bm25.py`) — numpy-only BM25 indexed over the same `[title] body`
   text.
4. **Fusion** (`retrieve.py`) — each retriever ranks pages independently via a
   `max+mean` chunk aggregation, then the two page rankings are RRF-merged
   (`RRF_K=28`) with asymmetric depth (`RRF_DEPTH=100` dense, `SPARSE_RRF_DEPTH=28`
   BM25).
   - **Dense**: pure mean (`DENSE_ALPHA=0.0`) over top `DENSE_TOPN=100` chunks.
   - **Sparse**: near-mean (`SPARSE_ALPHA=0.25`) over top `SPARSE_TOPN=30`.

5. **Post-filter** (`retrieve.py`) — if a query names integers, pages whose
   chunk metadata carries none of those numbers are pruned.

## Build Index

Run once locally to recreate `artifacts/`.
Note that the Wikipedia corpus is **not** committed to the repo.
Running the project does not require the raw JSON corpus, only the included `artifacts/` directory.

```bash
python scripts/build_index.py
```

## Artifacts (`artifacts/`)

| File | Format | Contents |
|---|---|---|
| `index_vectors.npy` | `np.float32 [N, 384]` | L2-normalized chunk embeddings, row `i` ↔ chunk `i` |
| `faiss.index` | `faiss.IndexFlatIP` | same vectors, for fast inner-product search |
| `index_meta.json` | JSON | per-row `page_ids`, `chunk_ids`, `page_titles`, `chunk_sections`, `chunk_section_numbers`, `chunk_word_counts`, `chunk_numbers` (aligned to vector rows) |
| `bm25.npz` | `np.savez` | BM25 inverted index over `[title] body` (df, idf, term weights, doc lengths) |

Large artifacts are tracked with **Git LFS**.

## Tunable hyperparameters (all in `retrieve.py`)

| Knob | Current | What it does |
|---|---|---|
| `DENSE_ALPHA`, `DENSE_TOPN` | `0.0`, `100` | dense page agg: `alpha*max + (1-alpha)*mean` over top-N chunks |
| `SPARSE_ALPHA`, `SPARSE_TOPN` | `0.25`, `30` | same blend, BM25 leg |
| `RRF_K` | `28` | RRF rank-discount constant (higher = flatter contribution) |
| `RRF_DEPTH` | `100` | dense RRF candidate depth |
| `SPARSE_RRF_DEPTH` | `60` | BM25 RRF candidate depth (asymmetric: dense broad, BM25 tight) |
| `INCLUDE_DENSE` / `INCLUDE_SPARSE` | `True` / `True` | toggle a leg off to A/B it solo |
| `ENABLE_NUMBER_FILTER` | `True` | prune pages missing a number named in the query |
| `FILTER_MODE` | `"post"` | `"post"` (filter after fusion) or `"pre"` (filter the pool first) |

## Results

Official benchmarks on `public_queries.json`:

| Metric | Value |
|---|---:|
| Public queries | 29 |
| Mean NDCG@10 | 0.5297 |
| Recall@10 / @50 / @100 | 0.6321 / 0.8539 / 0.9338 |

We have created additional validation sets, generated both by an LLM and by harmonizing with the SQuAD v2 dataset.
Those can be found in the `data` directory.

| Dataset | Queries | mean NDCG@10 |
|---|---:|---:|
| `public_queries` (provided dev set) | 29 | 0.5297 |
| `llm_queries` (LLM-generated) | 30 | 0.8597 |
| `squad_queries` (SQuAD v2 hybrid) | 119 | 0.4618 |

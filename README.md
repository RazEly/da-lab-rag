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

1. **Semantic chunking** (`chunk.py`) — pages are split into chunks at
   sentence-embedding breakpoints (low adjacent-cosine cuts), capped at
   `MAX_WORDS=170`. **Dense chunk text is content-only**; the title is *not*
   prepended into the embedded text. Per-chunk `years` / `months` metadata is
   extracted from the body.
2. **Dense** (`embed.py` + FAISS `IndexFlatIP`) — MiniLM embeddings, exact
   cosine (= inner product on normalized vectors).
3. **Sparse** (`bm25.py`) — numpy-only BM25 indexed over **`title + content`**,
   so all title/entity signal lives here (the dense side stays pure content).
4. **Fusion** (`retrieve.py`) — each retriever's per-page ranking (via
   `max+mean` chunk aggregation, `ALPHA=0.2` over the top `AGG_TOPN=30` chunks)
   is RRF-merged (`RRF_K=60`). Both retrievers are enabled; dense is a primary
   ranker, not a tie-breaker.

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
| `index_meta.json` | JSON | per-row `page_ids`, `chunk_ids`, `page_titles`, `chunk_word_counts` (aligned to vector rows) |
| `bm25.npz` | `np.savez` | BM25 inverted index over `title + content` (df, idf, term weights, doc lengths) |

Large artifacts are tracked with **Git LFS**.

## Public self-test

After building, verify a fresh run loads the submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

Reports mean NDCG@10 over `data/public_queries.json` (50 labelled queries) plus
the query-phase time. The full `run(queries)` call must finish in **≤60 s on GPU**.

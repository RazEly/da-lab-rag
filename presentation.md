# Project A — Section B: Retrieval Pipeline (video script)

---

## Slide 1 — Pipeline

**On slide (sparse):**

> _[PLACEHOLDER: block diagram — corpus → chunk (paragraphs)→ embed → dense index, embed -> bm25 index (offline) ‖
> query → dense retrieval, query -> sparse retrieval -> rff -> retrieve]_

- NDCG@10 final score:
-

## Slide 2 — chunk.py

- chunking strategy: sliding window (PARAM) + no-overlap per paragraphs
- file `header.py` used to extract paragraph headers deterministically
- proved to be faster and more effective than semantic embedding.
- metadata saved: exact numbers, titles, paragraphs.

> _[PLACEHOLDER: comparison between dense-only retrieval recall@k results with different MIN_WORDS parameter,
> comparison between dense-only retrieval recall@k - vanilla vs. injected title vs. injected title + paragraph header]_

---

## Slide 4 — embed.py

-

---

## Slide 5 — index.py

- two steps: dense and sparse

> _[PLACEHOLDER: grouped bar - (recall@k, ndcg@10) — dense-only vs BM25-only vs fused. before reranking]
>_[PLACEHOLDER: grouped bar — (recall@k, ndcg@10) - dense-only vs BM25-only vs fused. before reranking]
---

## Slide 6 — retrieve.py

- TODO: explain the retrieval pipeline (dense, sparse, how each one is treated (max vs mean aggregation) => post-filtering with extended numbers)
- Two retrievers (dense + BM25), ranked **independently**, fused by rank
- **Key finding:** the legs want **opposite** chunk→page aggregation
  - Dense → **mean** (gold = long uniform article)
  - BM25 → **max-ish** (exact match in one chunk)
- One shared setting starved BM25 → caused the original 0.155

---

## Slide 7 — retrieve.py: Reciprocal Rank Fusion

- **Reciprocal Rank Fusion (RRF):** combine the two ranked lists using **rank position only**, never raw scores — this sidesteps the dense/BM25 score-scale mismatch that collapsed earlier blend attempts (cosine ∈ [−1, 1] vs unbounded BM25 sums).
  - Each retriever returns an independently ranked page list. A page at rank _r_ (0-indexed) in a list contributes `1 / (k + r + 1)` to its fused score; contributions across both lists are summed; pages re-sorted by total.
  - `k = 60` (`RRF_K`) damps the top: it shrinks the gap between rank 0 and rank 1 so no single retriever's #1 can dominate — agreement across both legs is rewarded over one leg's lucky top hit.
  - Fusion depth is asymmetric: dense contributes its top **100** (`RRF_DEPTH`, semantic breadth), BM25 its top **60** (`SPARSE_RRF_DEPTH`, IDF-filtered so only strong exact matches compete).
  - Effect: a page ranked decently by _both_ retrievers beats a page ranked #1 by one and missing from the other — exactly the multi-facet Type-B queries we need.

- **Extracted numbers (metadata) in retrieval — number filter:**
  - At chunk time, every integer in the body is stored per-chunk (`chunk_numbers`); at load, unioned to a `page_id → {numbers}` map.
  - Per query, pull all integers (`\b\d+\b`); decade tokens like "1820s" expand to the full year range {1820…1829} (the trailing "s" blocks a plain match).
  - **Post-filter mode** (`FILTER_MODE="post"`): RRF-rank the full corpus first, _then_ drop pages whose number set doesn't intersect the query's. Ranking is untouched; only the final list is pruned.
  - Two safety fallbacks: if the query names no number, or no page carries it (empty pool), filtering is skipped — over-eager pruning is worse than none. Also auto-disabled on pre-`chunk_numbers` artifacts.
- **BM25 IDF filter:** drop generic query terms (IDF < 4.0); keep discriminative ones
- Adaptive fallback if filtering kills all signal

> _[PLACEHOLDER: TODO: add appropriate graphics OR empirical findings by sweeping different settings / showcasing how important the RFF is]
> _[PLACEHOLDER: line plot — NDCG@10 vs IDF threshold [1-5], plateau.]_

---

## Slide 8 - Evaluation

We used the following methods to avoid overfitting for the `public_queries.json` provided validation set:

- Enriching with another dataset
  - Extracted data from the SQUAD v2 dataset - an IR benchmark dataset
  - Created a hybrid set of queries by matching the wikipedia titles in 'SQUAD' to entries of the provided dataset
  - We reached NDCG@10 = 0.42 for 120 queries with a single answer.
- LLM as a judge

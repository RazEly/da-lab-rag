# Project A — Section B: Retrieval Pipeline (video script)

---

## Slide 1 — Pipeline

**On slide (sparse):**

> _[PLACEHOLDER: block diagram — corpus → chunk (paragraphs)→ embed → dense index, embed -> bm25 index (offline) ‖
> query → dense retrieval, query -> sparse retrieval -> rff -> retrieve]_

**Headline (29 public queries, full 27k corpus, committed artifacts — no rebuild):**

| metric | value |
|---|---|
| **mean NDCG@10** | **0.5428** |
| recall@10 / @50 / @100 | 0.632 / 0.856 / 0.917 |
| mrr@100 | 0.596 |
| first-relevant rank | median 2, p90 12, found 100% |
| query time (incl. artifact load) | ~55 s < 60 s wall |
| corpus / chunks | 27,068 pages / 576,321 chunks |

> Reorder headroom: recall@100 ≈ 0.917 vs NDCG@10 ≈ 0.543 — a relevant page is in
> the top-100 almost always; the gap is final ordering, not recall.
> (figures: `analysis/figures/h_per_query_ndcg`, `i_first_rel_cdf`, `j_latency`;
> raw: `analysis/results/headline.json`)

## Slide 2 — chunk.py

- chunking strategy: sliding window (PARAM) + no-overlap per paragraphs
- file `header.py` used to extract paragraph headers deterministically
- proved to be faster and more effective than semantic embedding.
- metadata saved: exact numbers, titles, paragraphs.

**(A) chunk size — dense-only over MAX_WORDS** (figure: `analysis/figures/s2_maxwords`).
Footnote: the slide's "MIN_WORDS" is `chunk.MAX_WORDS` — there is no MIN_WORDS knob.
Dense-only NDCG@10 (GT-inclusive 6k-page subset, directional): MAX_WORDS **120 = 0.511**
(best) vs 80 = 0.460, 160 = 0.485, 200 = 0.443. 120 ships; smaller over-fragments,
larger truncates context. Chunk count falls 234k→85k across 80→200 (cost twin-axis).

**(B) title / header injection — dense-only** (figure: `analysis/figures/s2_textmode`,
incl. MAX_WORDS×mode grid). At MAX_WORDS=120: **`[title] body` = 0.511** (ships) >
vanilla `body` = 0.495 > `[title] [section] body` = 0.493. Title injection helps;
adding the recovered section header does **not** — the grid's best cell is 120×title.
Both A and B cleared the §0.1 gate with **no change** (shipped settings already
optimal); measured on subset (directional), never promoted off subset.

---

## Slide 4 — embed.py

-

---

## Slide 5 — index.py

- two steps: dense and sparse

**Hybrid fusion beats either leg** (figure: `analysis/figures/s5_retrievers`;
first-stage pool, number filter off, shipped aggregation):

| | recall@5 | recall@10 | recall@20 | NDCG@10 |
|---|---|---|---|---|
| dense-only | 0.41 | 0.52 | 0.62 | 0.390 |
| bm25-only | 0.46 | 0.54 | 0.62 | 0.388 |
| **fused** | **0.54** | **0.60** | **0.69** | **0.489** |

Fused dominates on every metric — the legs recover different relevant pages
(dense = semantic paraphrase, BM25 = exact-term), so the union lifts recall and
NDCG together (recall@100 0.81/0.87 solo → **0.92** fused).
---

## Slide 6 — retrieve.py

- TODO: explain the retrieval pipeline (dense, sparse, how each one is treated (max vs mean aggregation) => post-filtering with extended numbers)
- Two retrievers (dense + BM25), ranked **independently**, fused by rank
- **Key finding:** the legs want **opposite** chunk→page aggregation
  - Dense → **mean** (gold = long uniform article)
  - BM25 → **max-ish** (exact match in one chunk)
- One shared setting starved BM25 → caused the original 0.155

**Evidence** (figure: `analysis/figures/s6_aggregation`, solo NDCG@10 over alpha×topn):
solo-optimal cells are **dense (alpha 0.0 = pure mean, topn 100)** and **BM25
(alpha 0.75 = max-heavy, topn 100)** — opposite ends of the alpha axis, confirming
the legs want different aggregation. Solo-optimal ≠ fused-optimal: the shipped pair
**D(0.0/100) S(0.25/30)** gates higher fused (0.5428) than the solo-best pair fused
(0.5182), so production is kept.

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

**RRF vs score-blend, and RRF_K** (figure: `analysis/figures/s7_rrf`):

| fusion | NDCG@10 |
|---|---|
| dense-only | 0.464 |
| bm25-only | 0.449 |
| min-max score-blend | 0.443 |
| **RRF (shipped)** | **0.543** |

RRF beats the normalized score-blend by **+0.10** — blending collapses on the
dense/BM25 scale mismatch (cosine ∈ [−1,1] vs unbounded BM25 sums). **RRF_K sweep**
{5,10,20,28,60,100,200} peaks at the 20–28 region (k28 = 0.5428) vs 0.5363 at k=60:
a tighter k rewards pages both legs rank decently. **This sweep changed production
`RRF_K` 60 → 28** (§0.1 gate, Δ+0.0065; re-verified eval+validate; see
`analysis/results/CHANGELOG.md`).

**BM25 IDF filter threshold** (figure: `analysis/figures/s7_idf`): NDCG@10 vs
`BM25_QUERY_MIN_IDF` over [0..6] peaks sharply at **4.0 = 0.5428** (shipped); 0.0
(no filter) = 0.530. Dropping generic low-IDF query terms is worth ~+0.013.

**Number filter mode** (figure: `analysis/figures/s7_number_filter`): off = 0.489,
pre = 0.542, **post = 0.5428** (shipped). Post edges pre and is safer when number
metadata recall is partial; both far beat off.

---

## Slide 8 - Evaluation

We used the following methods to avoid overfitting for the `public_queries.json` provided validation set:

- Enriching with another dataset
  - Extracted data from the SQUAD v2 dataset - an IR benchmark dataset
  - Created a hybrid set of queries by matching the wikipedia titles in 'SQUAD' to entries of the provided dataset
  - We reached NDCG@10 = 0.42 for 120 queries with a single answer.
- LLM as a judge

#import "@preview/touying:0.7.3": *
#import themes.metropolis: *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Project A — Section B: Retrieval Pipeline],
    subtitle: [End-to-end retrieval over ~27k Wikipedia-style pages],
    author: "",
  ),
)

#set text(size: 20pt)

#title-slide()

// ============================================================
== Pipeline

#align(center)[
  #set text(size: 14pt)
  #diagram(
    node-stroke: 0.5pt,
    node-corner-radius: 3pt,
    node-inset: 7pt,
    node-shape: fletcher.shapes.rect,
    spacing: (1.7em, 2.1em),
    edge-stroke: 0.6pt,

    node((0.55, -1.2), text(size: 0.82em, fill: rgb("#447"))[*OFFLINE — `build_index`*]),
    node((0, 0), [corpus\ #text(0.72em)[]], fill: rgb("#eef")),
    node((1.1, 0), [chunk\ #text(0.72em)[`[title] body`]], fill: rgb("#eef")),
    node((2.3, -0.55), [embed\ #text(0.72em)[all-MiniLM-L6-v2]], fill: rgb("#eef")),
    node((3.55, -0.55), [dense index\ #text(0.72em)[FAISS]], fill: rgb("#dfe")),
    node((2.3, 0.55), [BM25 build\ #text(0.72em)[tokenize]], fill: rgb("#eef")),
    node((3.55, 0.55), [sparse index\ #text(0.72em)[]], fill: rgb("#dfe")),
    edge((0, 0), (1.1, 0), "->"),
    edge((1.1, 0), (2.3, -0.55), "->"),
    edge((2.3, -0.55), (3.55, -0.55), "->"),
    edge((1.1, 0), (2.3, 0.55), "->"),
    edge((2.3, 0.55), (3.55, 0.55), "->"),

    // ---- QUERY-TIME: search_batch (<=60s) ----
    node((0.7, 1.15), text(size: 0.82em, fill: rgb("#744"))[*QUERY-TIME — `search_batch`*]),
    node((0, 2.4), [query], fill: rgb("#fee")),
    node((1.35, 1.9), [dense vector\ search], fill: rgb("#fee")),
    node((1.35, 2.9), [BM25 score], fill: rgb("#fee")),
    node((2.75, 1.9), [chunk→page\ #text(0.7em)[mean · top100]], fill: rgb("#fee")),
    node((2.75, 2.9), [chunk→page\ #text(0.7em)[α=.25 · top30]], fill: rgb("#fee")),
    node((4.05, 2.4), [RRF\ #text(0.7em)[k=28]], fill: rgb("#fed")),
    node((5.15, 2.4), [post-filter], fill: rgb("#fed")),
    node((6.2, 2.4), [top 10\ retrieval\ #text(0.7em)[]], fill: rgb("#fed")),
    edge((0, 2.4), (1.35, 1.9), "->"),
    edge((0, 2.4), (1.35, 2.9), "->"),
    edge((1.35, 1.9), (2.75, 1.9), "->"),
    edge((1.35, 2.9), (2.75, 2.9), "->"),
    edge((2.75, 1.9), (4.05, 2.4), "->"),
    edge((2.75, 2.9), (4.05, 2.4), "->"),
    edge((4.05, 2.4), (5.15, 2.4), "->"),
    edge((5.15, 2.4), (6.2, 2.4), "->"),

    // artifacts loaded at query time (dashed)
    edge((3.55, -0.55), (1.35, 1.9), "-->", stroke: rgb("#99a"), label: text(0.62em)[load]),
    edge((3.55, 0.55), (1.35, 2.9), "-->", stroke: rgb("#99a")),
  )
]

#v(0.5em)
#align(center)[#text(size: 16pt)[*Final results:* NDCG\@10 = 0.53]]

// ============================================================
== chunk.py

- *chunking strategy:* sliding window + no-overlap per paragraphs
  - Wikipedia articles are _already_ split semantically
  - file `header.py` used to extract paragraph headers deterministically, using heuristics:
    - Search for short sentences (<= 6 words) before double line breaks `\n\n`
    - Leading capital, ends with a period, no quotes.
    - Reject any phrase carrying an auxiliary/pronoun token (`is`, `was`, `it`, `they`...)

- Used recall\@k (k $in [1,10]$) to measure embedding effectiveness:

== chunk.py: max_words sweep

#align(center)[
  #image("analysis/figures/s2_maxwords.png", height: 80%)
]

== chunk.py: title injection

#grid(
  columns: (1fr, 1.2fr),
  column-gutter: 1.2em,
  align: horizon,
  [
    - We have found that some chunks lack context to the overall theme.
    - Solution: inject the title into every chunk: `[TITLE]: BODY`
    - Tested across the validation set with:
      - original chunk
      - injected article title
      - injected article title + paragraph title
  ],
  [
    #image("analysis/figures/s2_textmode.png", width: 100%)
  ],
)

// ============================================================
== embed.py

- No changes where made from the original implementation
- Optimized batch size for faster GPU run times.

// ============================================================
== index.py

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1.2em,
  align: horizon,
  [
    - Artifacts:
      - Dense `all-MiniLM-L6-v2` embeddings
      - metadata saved: exact numbers (used later for filtering), titles, paragraphs.
      - Sparse index (BM25)
    - Implemented a classic BM25 index to work alongside the dense embeddings
      - Per query term: score `tf-idf`, summed over chunk.
      - Applied length normalization
    - Validated on sparse vs dense vs fused
  ],
  [
    #image("analysis/figures/s5_retrievers.png", width: 100%)
  ],
)

// ============================================================
== retrieve.py

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1.2em,
  align: horizon,
  [
    - Each index aggregates its chunk scores up to a page score with a `max`/`mean` blend (`alpha`: 1=max, 0=mean).
    - Grid search: find optimal `alpha` and `top_n` values for each index.
    - Final hyperparameters:
      - Dense → *mean* ($alpha = 0.0$, `top_n=100`).
      - BM25 → *near-mean* ($alpha = 0.25$, `top_n=30`).
  ],
  [
    #image("analysis/figures/s6_aggregation.png", width: 100%)
  ],
)

// ============================================================
== retrieve.py: Reciprocal Rank Fusion

#grid(
  columns: (1.2fr, 1fr),
  column-gutter: 1.2em,
  align: horizon,
  [
    #set text(size: 17pt)
    - *Reciprocal Rank Fusion (RRF):* combine the two ranked lists using *rank position only*,
    - $ "score"(p) = sum_(r in R) 1 / (k + "rank"_r (p) + 1) $
      (sum over each retriever list $r$, with $k = 28$)
    - Each retriever returns an independently ranked page list. A page at rank $r$ in a list contributes $1 \/ (k + r + 1)$ to its fused score
    - Asymmetric depth:
      - dense contributes its top *100*
      - sparse contributes its top *60* (`SPARSE_RRF_DEPTH`)
    - pages re-sorted by total score.
  ],
  [
    #image("analysis/figures/s7_rrf.png", width: 100%)
  ],
)

== retrieve.py: Post-filtering

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1.2em,
  align: horizon,
  [
    - During chunking, numbers are saved to the metadata.
    - Queries containing numbers will trigger a post-filtering of the final list.
      - Years are expanded: "... in the 1920s" => expanded to {1920, 1921, ..., 1929}
    - `post` mode: rank the full corpus first, only then drop non-matching pages
  ],
  [
    #image("analysis/figures/s7_number_filter.png", width: 100%)
  ],
)

// ============================================================
== Evaluation

#set text(size: 16pt)

We used the following methods to avoid overfitting for the `public_queries.json` provided validation set:

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1.2em,
  [
    - Public dataset enrichment
      - Extracted data from the SQUaD v2 dataset - an IR benchmark dataset
      - Created a hybrid set of queries by matching the Wikipedia titles in SQUaD to entries of the provided dataset
      - We reached NDCG\@10 = 0.42 for 120 queries with a single answer.
  ],
  [
    - LLM Enrichment:
      - Used a short script to pick some articles at random from the corpus
      - prompted an LLM to create a query fitting of that article
      - verified the validity of the queries and saved as JSON.
  ],
)

Final scores across all datasets (mean NDCG\@10, shipped fusion, `RRF_K=28`):

#align(center)[
  #table(
    columns: 3,
    align: (left, center, center),
    table.header([*Dataset*], [*Queries*], [*mean NDCG\@10*]),
    [`public_queries` (provided dev set)], [29], [0.5297],
    [`llm_queries` (LLM-generated)], [30], [0.8597],
    [`squad_queries` (SQuAD v2 hybrid)], [119], [0.4618],
  )
]

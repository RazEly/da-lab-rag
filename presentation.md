# Project A — Section B: Retrieval Pipeline (video script)

**Format (from rubric, p.8):** video ≤ **3:00**, **≤10 slides**, **both members speak**,
cover chunk/embed/index/retrieve, show **empirical results visually**.
**No code scrolling / no pasted code on slides.** Penalties: 0.1 pt/sec over 3:00,
0.1 pt/slide over 10.

**Speakers:** A = member 1, B = member 2 (alternating so both speak meaningfully).
**Time budget:** ~18 s/slide → 180 s total (see per-slide `⏱`). Keep on-slide text
sparse; the prose under each slide is the **spoken script**, not slide content.

> **Headline: mean NDCG@10 = 0.5433 on the 50 public queries** (full 27k corpus,
> ~22 s query phase). From 0.1554 at first wiring → 3.5× via measured iteration.

---

## Slide 1 — Problem & pipeline  · ⏱ 15 s · Speaker A

**On slide (sparse):**
- 27k pages → `run(50 queries)` → ranked page_ids · scored mean **NDCG@10**
- Constraints: MiniLM-384d only · ≤60 s on GPU · prebuilt index shipped
- 4 stages: **chunk → embed → index → retrieve**

> _[PLACEHOLDER: clean block diagram — corpus → chunk → embed → index (offline) ‖
> query → retrieve (timed ≤60s). One graphic, no code.]_

**Script:** "We retrieve relevant Wikipedia pages for 50 hidden queries, scored on
mean NDCG@10, under a fixed model and a 60-second budget. Pipeline is four stages —
chunk, embed, index, retrieve. Headline: we reached 0.54, 3.5× our first wiring."

---

## Slide 2 — chunk.py: why & how  · ⏱ 18 s · Speaker A

**On slide (sparse):**
- Pages span **4 → 16,601 words**; MiniLM truncates at ~180 → whole-page loses most text
- **Sliding window, 120 words**, 1-sentence overlap, never crossing paragraphs
- Recover flattened `Section` headings (~95% precision) · prepend **title** to every chunk

> _[PLACEHOLDER: histogram of page word-counts, MiniLM ~180-word truncation line
> overlaid — visual proof of how much text whole-page embedding drops.]_

**Script:** "Pages vary 100×. MiniLM only sees ~180 words, so embedding whole pages
silently truncates most of the corpus. We split into 120-word windows with overlap,
keep title with every chunk, and recover section headings."

---

## Slide 3 — chunk.py: the decision that paid  · ⏱ 18 s · Speaker A

**On slide (sparse):**
- Built **semantic chunking** (embedding breakpoints) first — then **reverted it**
- Simple sliding window **won the A/B**

| Chunker | NDCG@10 (subset-500) |
|---|---|
| semantic breakpoints | baseline |
| **sliding window** | **~+12%** |

> _[PLACEHOLDER: bar chart sliding vs semantic, subset-500 + full-27k.
> Full-27k A/B still pending — confirm before final claim.]_

**Script:** "We tried the clever option — semantic breakpoints — and A/B-tested it.
The simple sliding window won by ~12%, so we deleted the semantic path entirely.
Theme of this project: measure, don't assume."

---

## Slide 4 — embed.py: model & vectors  · ⏱ 16 s · Speaker B

**On slide (sparse):**
- MiniLM-L6-v2, **L2-normalized** → inner product = cosine (enables exact FAISS)
- **Symmetric** query/doc encoder → no prefix tricks (tested, no gain)
- Pipeline **audited clean**: norms=1.0, alignment exact, FAISS==numpy

> _[PLACEHOLDER: metric card — dense GT-recall@10 = 37/50 vs BM25 36/50;
> GT median rank dense 5.0 vs BM25 3.5. Shows dense is a real retriever.]_

**Script:** "We normalize so cosine becomes a dot product, letting FAISS do exact
search. The model is symmetric, so query prefixes gave nothing. We fully audited
the vectors — they're correct; dense recall matches BM25."

---

## Slide 5 — index.py: offline build  · ⏱ 16 s · Speaker B

**On slide (sparse):**
- Embed corpus **once** offline; grader **never rebuilds**
- Ship: vectors (`.npy`, Git LFS) + meta JSON + BM25 · **FAISS rebuilt at load**
- `--subset N` GT-inclusive build for fast local iteration

> _[PLACEHOLDER: metric card — # chunks (~576k), artifact sizes on disk,
> offline build time, FAISS rebuild time at load (sub-second).]_

**Script:** "We build everything offline and commit it via LFS, so a fresh clone runs
with no rebuild. The 400 MB FAISS index isn't shipped — it reconstructs in under a
second at load. A subset build kept local iteration fast."

---

## Slide 6 — retrieve.py: hybrid + opposite aggregation  · ⏱ 20 s · Speaker B

**On slide (sparse):**
- Two retrievers (dense + BM25), ranked **independently**, fused by rank
- **Key finding:** the legs want **opposite** chunk→page aggregation
  - Dense → **mean** (gold = long uniform article)
  - BM25 → **max-ish** (exact match in one chunk)
- One shared setting starved BM25 → caused the original 0.155

> _[PLACEHOLDER: grouped bar — dense-only vs BM25-only vs fused; and
> single-α vs per-retriever-α. Visual of why per-retriever matters.]_

**Script:** "Our biggest structural insight: dense and BM25 need opposite
aggregation. Dense wants the page mean; BM25 wants its single best chunk. Forcing
one setting on both starved the stronger leg — that was the original 0.15."

---

## Slide 7 — retrieve.py: fusion choice  · ⏱ 14 s · Speaker A

**On slide (sparse):**
- Fuse by **Reciprocal Rank Fusion**, not score blending
- Why: dense & BM25 scores live on different scales → blending collapses
- RRF uses only **rank** → scale-immune

> _[PLACEHOLDER: line plot — NDCG@10 vs RRF k {28,40,60,80}; mark chosen k=60.]_

**Script:** "We fuse by rank, not raw score. Dense and BM25 scores aren't
comparable, so blending them collapses. Rank fusion ignores scale and just rewards
pages both retrievers like."

---

## Slide 8 — retrieve.py: the biggest single win  · ⏱ 20 s · Speaker A

**On slide (sparse):**
- **BM25 IDF filter:** drop generic query terms (IDF < 4.0); keep discriminative ones
- Adaptive fallback if filtering kills all signal

| Step | NDCG@10 |
|---|---|
| fused, optimal aggregation | 0.3161 |
| **+ BM25 IDF filter** | **0.5343** |
| + adaptive fallback + re-sweep | **0.5433** |

> _[PLACEHOLDER: line plot — NDCG@10 vs IDF threshold {2,3,3.7,4,4.1,5}, plateau.]_

**Script:** "The single largest gain: filtering generic query words out of BM25.
'Founded' and 'city' match everything; rare nouns and years carry the signal.
That one change jumped us from 0.32 to 0.53."

---

## Slide 9 — retrieve.py: number filter + a real bug  · ⏱ 16 s · Speaker B

**On slide (sparse):**
- **Number filter:** queries name years/stats → drop pages missing that number (with fallback)
- **Bug found via process:** a stale FAISS index zeroed dense for ~98% of corpus

| State | NDCG@10 |
|---|---|
| stale FAISS + shared α | 0.1554 |
| both fixed | 0.3161 |

> _[PLACEHOLDER: before/after bar — dense coverage % of corpus, stale vs fixed.]_

**Script:** "Two more wins. A number filter exploits that queries cite years. And our
process exposed a bug — a stale index zeroed 98% of dense scores. Fixing it doubled
the score before any tuning."

---

## Slide 10 — Results, dead ends, ceiling  · ⏱ 25 s · Speaker B → A

**On slide (sparse):**
- **0.1554 → 0.5433** by measured iteration
- **Rejected (measured):** PRF, asymmetric RRF, pool rerank, pre-mode filter
- **Ceiling:** 4 queries score 0 — GT strong in BM25, ~2000+ in dense

> _[PLACEHOLDER: cumulative line — NDCG@10 across milestones (the 0.155→0.5433 curve).]_

> _[PLACEHOLDER: per-query NDCG@10 bar (q001–q050), highlight the 4 zeros + 60s-budget timing card.]_

**Script (B):** "From 0.15 to 0.54 — every step measured on the public set. We also
killed ideas that sounded good but lost: PRF, asymmetric fusion, pool reranking."
**Script (A):** "Remaining ceiling is four queries dense ranks ~2000 — fixable only
by changing the index or model, which the constraints lock. Thanks for watching."

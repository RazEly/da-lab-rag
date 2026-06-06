"""Self-supervised domain fine-tune of all-MiniLM-L6-v2 (build-time, UNTIMED).

Why this exists
---------------
Public/hidden queries are paraphrased / oblique (short titles, demonyms,
obfuscated dates). Stock MiniLM is weaker *at the very top* of the ranking on
this corpus. We adapt it with contrastive learning on signal mined from the
27k corpus itself -- NO labelled queries -- so it generalises to the hidden 50.

Method
------
MultipleNegativesRankingLoss (in-batch negatives) on (anchor, positive) pairs:

  * (title, chunk_body)      title is a short oblique probe -> full passage.
                             Directly targets the Type-A paraphrase gap.
  * (chunk_body, chunk_body) same page, optional (--intra-page): two facets of
                             one page -> helps Type-B multi-page recall.

Hard constraints honoured
-------------------------
  * Imports: ``sentence_transformers`` + ``torch`` (+ stdlib). torch is a
    transitive dep of sentence-transformers (no new package). It is needed for
    the manual training loop: ST v5's ``model.fit`` now routes through
    ``SentenceTransformerTrainer``, which hard-requires HF ``datasets`` -- we
    avoid that by training by hand. This is a BUILD-TIME script, NOT imported
    by ``run()``, so it does not touch the graded import constraint. A custom
    no-duplicate batcher keeps two same-text anchors out of one batch (in-batch
    -negative false-negative guard, same role as ST's old loader).
  * NEVER trains on data/public_queries.json -- that set stays a clean
    held-out NDCG proxy for the hidden queries.
  * Same architecture / 384-dim / same run() speed. Output weights ship in
    artifacts/ (Git LFS) and are loaded by embed.py.

After running
-------------
  1. Point embed.py at the fine-tuned dir:  export FINETUNED_MODEL_DIR=artifacts/minilm_ft
  2. Rebuild the index:  python scripts/build_index.py
  3. DELETE artifacts/faiss.index   (stale .index shadows fresh .npy -> dead dense)
  4. Re-sweep agg params, then eval_public_retrieval.py.
  5. KEEP only if public NDCG@10 clearly beats 0.3695 (margin > noise; 50 queries).

⚠ Compliance: project says "only all-MiniLM-L6-v2". Fine-tuned weights are a
  MODIFIED checkpoint. Confirm the autograder allows modified weights (vs the
  exact published checkpoint) BEFORE shipping. See Project A.pdf.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow running as `python scripts/finetune.py` from part-2/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402 -- transitive dep of sentence-transformers; build-time only
from sentence_transformers import (  # noqa: E402
    InputExample,
    SentenceTransformer,
    losses,
)

from chunk import chunk_corpus  # noqa: E402
from utils import (  # noqa: E402
    ARTIFACTS_DIR,
    EMBEDDING_MODEL_NAME,
    iter_entries,
)


def _strip_title_prefix(text: str, title: str) -> str:
    """Recover bare body from a chunk's ``[title] body`` embedded text."""
    prefix = f"[{title}] "
    return text[len(prefix):] if text.startswith(prefix) else text


def build_pairs(*, intra_page: bool, max_pairs: int | None) -> list[InputExample]:
    """Mine (anchor, positive) pairs from the corpus. No public-query data."""
    records = list(iter_entries())
    print(f"[ft] loaded {len(records)} records")
    chunks = chunk_corpus(records)
    print(f"[ft] {len(chunks)} chunks")

    examples: list[InputExample] = []

    # (title, body): short oblique anchor -> dense passage.
    for c in chunks:
        title = (c.title or "").strip()
        if not title:
            continue
        body = _strip_title_prefix(c.text, c.title).strip()
        if not body:
            continue
        examples.append(InputExample(texts=[title, body]))

    # (body_i, body_j) same page, consecutive chunks: two facets of one page.
    if intra_page:
        by_page: dict[int, list[str]] = {}
        for c in chunks:
            body = _strip_title_prefix(c.text, c.title).strip()
            if body:
                by_page.setdefault(c.page_id, []).append(body)
        pre = len(examples)
        for bodies in by_page.values():
            for a, b in zip(bodies, bodies[1:]):
                examples.append(InputExample(texts=[a, b]))
        print(f"[ft] intra-page pairs added: {len(examples) - pre}")

    print(f"[ft] total pairs: {len(examples)}")
    if max_pairs is not None and len(examples) > max_pairs:
        # Deterministic stride subsample -> keeps coverage across the corpus.
        stride = len(examples) / max_pairs
        examples = [examples[int(i * stride)] for i in range(max_pairs)]
        print(f"[ft] subsampled to {len(examples)} pairs (--max-pairs)")
    return examples


def iter_no_dup_batches(
    examples: list[InputExample], batch_size: int, *, seed: int = 0
):
    """Yield batches with no repeated text within a batch (false-negative guard).

    Mirrors ST's old ``NoDuplicatesDataLoader``: floor(n/bs) batches per call;
    rotates a pointer through a shuffled copy, reshuffling on wrap-around. Two
    pairs sharing an anchor/positive never co-occur, so in-batch negatives are
    never accidentally positives.
    """
    data = list(examples)
    rng = random.Random(seed)
    rng.shuffle(data)
    n = len(data)
    ptr = 0
    for _ in range(n // batch_size):
        batch: list[InputExample] = []
        seen: set[str] = set()
        while len(batch) < batch_size:
            ex = data[ptr]
            if all(t not in seen for t in ex.texts):
                batch.append(ex)
                seen.update(ex.texts)
            ptr += 1
            if ptr >= n:
                ptr = 0
                rng.shuffle(data)
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=1, help="1-2 max (forgetting risk)")
    ap.add_argument("--batch-size", type=int, default=64, help="bigger = more in-batch negatives")
    ap.add_argument("--lr", type=float, default=2e-5, help="low: preserve pretrained space")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--intra-page", action="store_true", help="add same-page body pairs (Type-B)")
    ap.add_argument("--max-pairs", type=int, default=None, help="cap pairs for a quick run")
    ap.add_argument(
        "--out",
        type=str,
        default=str(ARTIFACTS_DIR / "minilm_ft"),
        help="output dir for fine-tuned model",
    )
    args = ap.parse_args()

    examples = build_pairs(intra_page=args.intra_page, max_pairs=args.max_pairs)
    if not examples:
        raise SystemExit("[ft] no training pairs built; aborting.")

    print(f"[ft] loading base model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    device = model.device
    loss_fn = losses.MultipleNegativesRankingLoss(model).to(device)

    n_batches = len(examples) // args.batch_size
    total_steps = n_batches * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    print(f"[ft] {n_batches} batches/epoch x {args.epochs} = {total_steps} steps; warmup {warmup}")
    if total_steps == 0:
        raise SystemExit("[ft] batch_size > #pairs; nothing to train.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for batch in iter_no_dup_batches(examples, args.batch_size, seed=epoch):
            # One tokenized feature dict per column (anchors, positives).
            feats = [
                {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in model.tokenize(col).items()}
                for col in ([ex.texts[0] for ex in batch],
                            [ex.texts[1] for ex in batch])
            ]
            labels = torch.zeros(len(batch), dtype=torch.long, device=device)  # MNRL ignores

            loss_value = loss_fn(feats, labels)
            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            running += loss_value.item()
            if step % 50 == 0:
                print(f"[ft] epoch {epoch} step {step}/{total_steps} "
                      f"loss {running / 50:.4f} lr {scheduler.get_last_lr()[0]:.2e}")
                running = 0.0
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"[ft] saved fine-tuned model -> {out}")
    print("[ft] NEXT: export FINETUNED_MODEL_DIR=%s ; rebuild index ; "
          "delete artifacts/faiss.index ; re-sweep ; eval." % out)


if __name__ == "__main__":
    main()

import csv
import json
import math
from pathlib import Path

from main import run


ROOT = Path(__file__).resolve().parent
PUBLIC_QUERIES = ROOT / "data" / "public_queries.json"
OUT_CSV = ROOT / "public_error_analysis.csv"


def dcg_at_k(relevance, k=10):
    score = 0.0
    for i, rel in enumerate(relevance[:k]):
        if rel:
            score += 1.0 / math.log2(i + 2)
    return score


def ndcg_at_k(result_ids, relevant_ids, k=10):
    relevance = [1 if int(pid) in relevant_ids else 0 for pid in result_ids[:k]]
    dcg = dcg_at_k(relevance, k)

    ideal_hits = min(len(relevant_ids), k)
    ideal = dcg_at_k([1] * ideal_hits, k)

    if ideal == 0:
        return 0.0

    return dcg / ideal


def first_relevant_rank(result_ids, relevant_ids):
    for i, pid in enumerate(result_ids, start=1):
        if int(pid) in relevant_ids:
            return i
    return None


def main():
    with PUBLIC_QUERIES.open("r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [item["query"] for item in data]
    outputs = run(queries)

    rows = []

    for item, result in zip(data, outputs):
        relevant = set(map(int, item["relevant_page_ids"]))
        ndcg = ndcg_at_k(result, relevant, k=10)
        first_rank = first_relevant_rank(result, relevant)

        rows.append(
            {
                "query_id": item.get("query_id", ""),
                "query": item["query"],
                "ndcg_at_10": round(ndcg, 6),
                "first_relevant_rank": first_rank if first_rank is not None else "",
                "relevant_page_ids": " ".join(map(str, sorted(relevant))),
                "top10_returned": " ".join(map(str, result[:10])),
            }
        )

    rows.sort(key=lambda row: row["ndcg_at_10"])

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query",
                "ndcg_at_10",
                "first_relevant_rank",
                "relevant_page_ids",
                "top10_returned",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV}")
    print()
    print("Worst 10 public queries:")
    for row in rows[:10]:
        print("-" * 80)
        print(f"{row['query_id']} | NDCG@10={row['ndcg_at_10']}")
        print(row["query"])
        print(f"Relevant: {row['relevant_page_ids']}")
        print(f"Returned: {row['top10_returned']}")
        print(f"First relevant rank: {row['first_relevant_rank']}")


if __name__ == "__main__":
    main()
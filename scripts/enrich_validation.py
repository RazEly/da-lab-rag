import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def clean_title(s):
    s = s.replace("_", " ").lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


squad = load_dataset("rajpurkar/squad_v2", split="validation")

titles = [row["title"] for row in squad]
queries = [row["question"] for row in squad]

df = pd.DataFrame({"query": queries, "title": titles})
df["len"] = df["query"].str.len()

# Build cleaned-title -> page_id map from data folder
DATA_DIR = Path("part-2/data/Wikipedia Entries")
title_to_id = {}
for fp in DATA_DIR.glob("*.json"):
    rec = json.loads(fp.read_text())
    title_to_id[clean_title(rec["title"])] = rec["page_id"]

# Keep only titles present in data folder (cleaned match), attach page_id
df["page_id"] = df["title"].map(lambda t: title_to_id.get(clean_title(t)))
df = df[df["page_id"].notna()]

# Sample 7 random queries per subject (title); emit in public_queries.json format
PER_SUBJECT = 7
SEED = 42
sampled = (
    df.groupby("title", group_keys=False)
    .apply(lambda g: g.sample(PER_SUBJECT, random_state=SEED), include_groups=False)
    .reset_index(drop=True)
)

out = [
    {
        "query_id": f"q_val_{i}",
        "query": row["query"],
        "relevant_page_ids": [str(row["page_id"])],
    }
    for i, row in sampled.iterrows()
]

out_path = Path(__file__).parent / "validation_queries.json"
out_path.write_text(json.dumps(out, indent=4))
print(f"wrote {len(out)} queries -> {out_path}")

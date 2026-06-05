"""Debug: dump the chunk split for a single corpus JSON file.

Usage:
    python scripts/dump_chunks.py data/Wikipedia\\ Entries/10000.json [out.txt]

Writes each chunk's text separated by a blank line. If no output path is
given, writes <stem>.chunks.txt next to this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunk import chunk_corpus
from utils import normalize_page_id


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <page.json> [out.txt]")
    in_path = Path(sys.argv[1])
    record = json.loads(in_path.read_text(encoding="utf-8"))
    record["page_id"] = normalize_page_id(record.get("page_id", in_path.stem))

    chunks = chunk_corpus([record])

    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.with_suffix(".chunks.txt")
    blocks = []
    for c in chunks:
        header = f"### chunk {c.chunk_id}  ({len(c.text.split())} words)"
        blocks.append(f"{header}\n{c.text}")
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    print(f"page_id={record['page_id']} title={record.get('title','')!r}")
    print(f"{len(chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()

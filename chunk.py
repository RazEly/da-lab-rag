"""Chunking: sliding window over page words, title prepended to every chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from utils import entry_text

CHUNK_WORDS = 150
OVERLAP_WORDS = 30

# AD years 1000-2099; word boundaries prevent partial matches inside longer numbers
_YEAR_RE = re.compile(r"\b((?:1[0-9]|20)\d{2})\b")
# BC/BCE years stored as negative ints: "300 BC" → -300, "44 BCE" → -44
_BC_YEAR_RE = re.compile(r"\b(\d{1,4})\s+BC(?:E)?\b", re.IGNORECASE)
# Optional-suffix form covers full names, 3-letter abbrs, and "Sept" in one pattern
_MONTH_RE = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May"
    r"|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?"
    r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\b",
    re.IGNORECASE,
)
_MONTH_NORM: Dict[str, str] = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}


def extract_years(text: str) -> Set[int]:
    ad = {int(m) for m in _YEAR_RE.findall(text)}
    bc = {-int(m) for m in _BC_YEAR_RE.findall(text)}
    return ad | bc


def extract_months(text: str) -> Set[str]:
    return {_MONTH_NORM[m[:3].lower()] for m in _MONTH_RE.findall(text)}


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str
    title: str = ""
    years: Set[int] = field(default_factory=set)
    months: Set[str] = field(default_factory=set)


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title = record.get("title", "")
    words = entry_text(record).split()
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for i, start in enumerate(range(0, max(1, len(words) - OVERLAP_WORDS), step)):
        body = " ".join(words[start : start + CHUNK_WORDS])
        text = title + "\n\n" + body
        years = extract_years(body)
        months = extract_months(body)
        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=i,
                text=text,
                title=title,
                years=years,
                months=months,
            )
        )
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks

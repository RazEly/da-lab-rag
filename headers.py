"""Section-header recovery for flattened wiki text.

The corpus had its wiki markup stripped: every ``== Section ==`` / ``=== Sub ===``
heading was flattened into a short Title/sentence-case phrase glued to the tail
of the preceding paragraph, just before the blank-line break. These helpers
recover those headings structurally — no markup, no model, no extra deps.

Detection peels trailing sentences off a paragraph while each passes
:func:`is_header`:

* 1-6 words, leading capital, ends with ``.`` (not ``?``/``!``), >2 chars, no quote
* carries no auxiliary/pronoun token — those mark a leaked real sentence
  (``It was formed``, ``they had abandoned``), never a heading
* capitalization is sentence-case (``Early life``) or title-case (``South Korea``),
  never "mixed" (a stray proper-noun cap mid-clause: ``This worried Drs``)

Measured on the 27k-page corpus against a cross-page frequency gazetteer:
~95% precision. The residual false positives all carry a lexical verb with no
auxiliary (``Custom also plays a role``), which the closed-class filter can't see.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

HEADER_MAX_WORDS = 6

# Function words allowed lowercase inside an otherwise-capitalized title
# (so "Awards and honors" / "Coat of arms" stay title/sentence-case).
_FUNC = {
    "a", "an", "the", "of", "and", "or", "for", "in", "on", "to", "at", "by",
    "with", "as", "from", "de", "la", "le", "vs", "per",
}
# Auxiliaries + pronouns. Their presence in a short tail phrase means a real
# sentence leaked through, not a heading. Note: bare "i" is deliberately absent
# (it collides with the Roman numeral I, e.g. "World War I").
_AUX = {
    "is", "was", "were", "are", "am", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must",
}
_PRON = {
    "it", "its", "he", "him", "his", "she", "her", "they", "them", "their",
    "we", "us", "our", "you", "your", "this", "that", "these", "those",
    "who", "which",
}
_STOP = _AUX | _PRON


def _cap_label(s: str) -> str:
    """Classify capitalization as ``sentence`` | ``title`` | ``mixed``."""
    words = s.rstrip(".").split()
    if len(words) == 1:
        return "sentence"
    interior = words[1:]
    interior_caps = [t for t in interior if t[:1].isupper()]
    interior_low_content = [
        t for t in interior if t[:1].islower() and t.lower() not in _FUNC
    ]
    if not interior_caps:
        return "sentence"  # first word cap, rest lowercase (+ function words)
    if (
        all(t[:1].isupper() or t.lower() in _FUNC for t in words)
        and not interior_low_content
    ):
        return "title"  # every content word capitalized
    return "mixed"  # stray interior cap beside lowercase content -> leaked sentence


def _has_stopword(s: str) -> bool:
    return any(t.lower().strip("().,") in _STOP for t in s.rstrip(".").split())


def is_header(s: str) -> bool:
    """True if ``s`` looks like a recovered flattened section heading."""
    words = s.split()
    if not 1 <= len(words) <= HEADER_MAX_WORDS:
        return False
    if not s[:1].isupper():
        return False
    if not s.endswith(".") or s.endswith(("?", "!")):
        return False
    if '"' in s:
        return False
    if len(s.rstrip(".")) <= 2:  # drop bare initials: "D.", "U."
        return False
    if _has_stopword(s):
        return False
    return _cap_label(s) != "mixed"


def header_text(s: str) -> str:
    """Normalize a header sentence to its bare title (strip trailing period)."""
    return s.rstrip(".").strip()


def peel_headers(sentences: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split one paragraph's sentences into (body, trailing_headers).

    Headings sit at the paragraph tail (the flattening artifact), so walk from
    the end while sentences pass :func:`is_header` and stop at the first body
    sentence. Order is preserved; stacked headings (``Overview. Influences.``)
    come back as ``["Overview.", "Influences."]``.
    """
    cut = len(sentences)
    for s in reversed(sentences):
        if is_header(s):
            cut -= 1
        else:
            break
    return list(sentences[:cut]), list(sentences[cut:])

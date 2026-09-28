"""T3f: deterministic query-aware projection of an already-selected passage.

Sees only a passage and the query: no model, no benchmark label. Query
phrases are matched with ZOMAH's own lexical matcher (the helpers behind
``zomah.knowledge.localize``, imported unchanged), so matching is
case- and diacritic-insensitive, whole-word, and ``tool-result`` means the
word sequence tool, result.
"""

from __future__ import annotations

from dataclasses import dataclass

from zomah.knowledge import _line_matches, _query_phrases

from embedding_bench.passages import Excerpt, Passage

# Leading context kept before the anchor: a quarter of the budget, as localize() keeps 40 of 160.
LEAD_DIVISOR = 4


@dataclass(frozen=True, slots=True)
class QueryAnchor:
    """Where a projection is centred, relative to the start of the passage text.

    ``mode`` is "match" (earliest match in the strongest line) or "center"
    (no query phrase occurs in the passage).
    """

    mode: str
    offset: int
    line: int | None  # 1-based document line of the strongest line, for "match"
    distinct: int  # distinct query phrases on that line
    occurrences: int  # query-phrase occurrences on that line


def _trimmed_bounds(text: str) -> tuple[int, int]:
    return len(text) - len(text.lstrip()), len(text.rstrip())


def query_anchor(passage: Passage, query: str) -> QueryAnchor:
    """The strongest line by distinct phrases, then occurrences, then earliest; else the passage centre."""

    phrases = _query_phrases(query)
    best: tuple[tuple[int, int, int], QueryAnchor] | None = None
    line_start = 0
    for index, line in enumerate(passage.text.split("\n")):
        matches = _line_matches(line, phrases) if phrases else []
        if matches:
            distinct, occurrences = len({phrase for phrase, _ in matches}), len(matches)
            key = (-distinct, -occurrences, index)
            if best is None or key < best[0]:
                offset = line_start + min(offset for _, offset in matches)
                best = (key, QueryAnchor("match", offset, passage.start_line + index, distinct, occurrences))
        line_start += len(line) + 1
    if best is not None:
        return best[1]
    lo, hi = _trimmed_bounds(passage.text)
    return QueryAnchor("center", (lo + hi) // 2, None, 0, 0)


def project_query_aware(passage: Passage, query: str, budget: int) -> Excerpt:
    """At most ``budget`` source characters of ``passage`` around the query anchor.

    The whole trimmed passage when it fits. Otherwise a window that starts a
    quarter budget before the earliest match in the strongest line (or is
    centred on the passage when nothing matches), shifted back inside the
    passage when it reaches an edge. A cut that would split a word moves to
    the word boundary when that keeps at least half the budget; edge
    whitespace is dropped. "…" marks omitted passage text and is not counted.
    """

    if budget < 1:
        raise ValueError("budget must be positive")
    text = passage.text
    lo, hi = _trimmed_bounds(text)
    if hi <= lo:
        return Excerpt("", passage.char_start, passage.char_start, False)
    if hi - lo <= budget:
        return Excerpt(text[lo:hi], passage.char_start + lo, passage.char_start + hi, False)

    anchor = query_anchor(passage, query)
    wanted = anchor.offset - (budget // LEAD_DIVISOR if anchor.mode == "match" else budget // 2)
    start = max(lo, wanted)
    end = min(hi, start + budget)
    start = max(lo, end - budget)

    # A word is a run of letters/digits, as in ZOMAH's matcher. A match starts a word, so moving
    # the start forward never passes the anchor; the half-budget floor keeps the end beyond it.
    floor = max(1, budget // 2)
    if start > lo and text[start - 1].isalnum() and text[start].isalnum():
        moved = start
        while moved < end and text[moved].isalnum():
            moved += 1
        if end - moved >= floor:
            start = moved
    if end < hi and text[end - 1].isalnum() and text[end].isalnum():
        moved = end
        while moved > start and text[moved - 1].isalnum():
            moved -= 1
        if moved - start >= floor:
            end = moved
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    body = ("…" if start > lo else "") + text[start:end] + ("…" if end < hi else "")
    return Excerpt(body, passage.char_start + start, passage.char_start + end, True)

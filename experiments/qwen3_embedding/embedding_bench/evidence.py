"""T3e: score localized passages against benchmark evidence, after localization.

Standard library only. Nothing here chooses a passage; it measures where the
chosen passage and its excerpts sit relative to the evidence anchor.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from embedding_bench.manifest import normalize_whitespace


def anchor_span(text: str, anchor: str) -> tuple[int, int] | None:
    """Raw character span of a whitespace-normalized anchor in ``text``, or None."""

    chars: list[str] = []
    raw_index: list[int] = []
    previous_space = True
    for index, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                chars.append(" ")
                raw_index.append(index)
            previous_space = True
        else:
            chars.append(char)
            raw_index.append(index)
            previous_space = False
    target = normalize_whitespace(anchor)
    found = "".join(chars).find(target) if target else -1
    if found < 0:
        return None
    return raw_index[found], raw_index[found + len(target) - 1] + 1


def span_lines(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """1-based lines (split on newline) spanned by a raw character span."""

    return text.count("\n", 0, span[0]) + 1, text.count("\n", 0, max(span[0], span[1] - 1)) + 1


def score_span(span: tuple[int, int], anchor: tuple[int, int]) -> dict[str, Any]:
    """How a selected span relates to the anchor span.

    ``survives``: the span holds the whole anchor, or the anchor is longer and
    the span lies entirely inside it (the excerpt is all evidence).
    """

    overlap = max(0, min(span[1], anchor[1]) - max(span[0], anchor[0]))
    whole = span[0] <= anchor[0] and anchor[1] <= span[1]
    inside = anchor[0] <= span[0] and span[1] <= anchor[1] and span[1] > span[0]
    return {
        "contains_anchor": whole,
        "survives": whole or inside,
        "coverage": overlap / (anchor[1] - anchor[0]),
    }


def lines_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def evaluate(result: Mapping[str, Any], text: str, anchor: tuple[int, int]) -> dict[str, Any]:
    """Evaluate one localizer's result (passage span + excerpt spans) against one anchor."""

    anchor_lines = span_lines(text, anchor)
    passage = result["passage"]
    evaluation = {
        "line_overlap": lines_overlap((passage["start_line"], passage["end_line"]), anchor_lines),
        "passage": score_span((passage["char_start"], passage["char_end"]), anchor),
        "excerpts": {
            budget: score_span((excerpt["char_start"], excerpt["char_end"]), anchor)
            for budget, excerpt in result["excerpts"].items()
        },
    }
    native = result.get("native_excerpt")
    if native is not None:
        evaluation["native_excerpt"] = score_span((native["char_start"], native["char_end"]), anchor)
    return evaluation


# --- aggregation ---------------------------------------------------------------------------


def hit_measures(evaluation: Mapping[str, Any]) -> dict[str, bool]:
    """The boolean measures reported for every localizer."""

    measures = {
        "line_overlap": evaluation["line_overlap"],
        "passage_contains_anchor": evaluation["passage"]["contains_anchor"],
    }
    for budget, score in evaluation["excerpts"].items():
        measures[f"excerpt_{budget}_survives"] = score["survives"]
        measures[f"excerpt_{budget}_contains_anchor"] = score["contains_anchor"]
    if "native_excerpt" in evaluation:
        measures["native_excerpt_survives"] = evaluation["native_excerpt"]["survives"]
        measures["native_excerpt_contains_anchor"] = evaluation["native_excerpt"]["contains_anchor"]
    return measures


def coverage_measures(evaluation: Mapping[str, Any]) -> dict[str, float]:
    measures = {"passage": evaluation["passage"]["coverage"]}
    for budget, score in evaluation["excerpts"].items():
        measures[f"excerpt_{budget}"] = score["coverage"]
    if "native_excerpt" in evaluation:
        measures["native_excerpt"] = evaluation["native_excerpt"]["coverage"]
    return measures


def summarize(rows: Sequence[Mapping[str, Any]], localizer: str, categories: Sequence[str]) -> dict[str, Any]:
    """Counts of every hit measure, overall and per category, plus mean anchor coverage."""

    def block(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        hits = [hit_measures(m[localizer]["evaluation"]) for m in members]
        coverage = [coverage_measures(m[localizer]["evaluation"]) for m in members]
        keys = list(hits[0]) if hits else []
        return {
            "n": len(members),
            "hits": {key: sum(1 for h in hits if h[key]) for key in keys},
            "mean_coverage": {
                key: sum(c[key] for c in coverage) / len(coverage) for key in (coverage[0] if coverage else {})
            },
        }

    return {
        "overall": block(rows),
        "by_category": {category: block([r for r in rows if r["category"] == category]) for category in categories},
    }


def pairwise(rows: Iterable[Mapping[str, Any]], measure: str, a: str = "lexical", b: str = "semantic") -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"both_hit": [], f"{a}_only": [], f"{b}_only": [], "both_miss": []}
    for row in rows:
        hit_a = hit_measures(row[a]["evaluation"])[measure]
        hit_b = hit_measures(row[b]["evaluation"])[measure]
        key = "both_hit" if hit_a and hit_b else f"{a}_only" if hit_a else f"{b}_only" if hit_b else "both_miss"
        buckets[key].append(row["id"])
    return buckets

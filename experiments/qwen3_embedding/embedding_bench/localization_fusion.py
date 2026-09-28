"""T3f: deterministic evidence projection and bounded localization fusion over a frozen T3e run.

Reads one T3e ``results.json`` and nothing else: no model, no corpus, no
re-embedding. Three strictly ordered phases:

1. check only the label-free fields (identity, config, ids, queries, paths,
   passages, native excerpt, semantic similarity and windows) and that the
   recorded leading excerpts reproduce;
2. build label-free cases and freeze every projection and packet;
3. only then read categories, anchors and recorded evaluation: validate
   them, reproduce the recorded T3e evaluation, check it against the
   accepted live run, and score the frozen decisions.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from embedding_bench.evidence import lines_overlap, pairwise, score_span, summarize
from embedding_bench.passages import EXCERPT_BUDGETS, STRIDE, WINDOW_LINES, Excerpt, Passage, bounded_excerpt
from embedding_bench.projection import LEAD_DIVISOR, project_query_aware, query_anchor

EXPERIMENT = "T3f: deterministic evidence projection + bounded localization fusion over frozen T3e passages"
# T3e's identity and top-window depth, as written by localization_runner (kept here so reading an
# artifact never imports ZOMAH's capabilities; a test pins them to T3e's values).
T3E_EXPERIMENT = "T3e: semantic within-document evidence localization vs lexical localize()"
TOP_K = 3
SOURCES = ("lexical", "semantic")
METHODS = ("leading", "query_aware")
PROJECTION_BUDGETS = EXCERPT_BUDGETS
PACKET_TOTALS = (160, 240, 320)
FUSION = "semantic_primary_dual_on_disagreement"
CONTROL = "semantic_only"  # the same total budget spent on the semantic passage alone
STRATEGIES = (FUSION, CONTROL)
# Label-bearing keys of a T3e artifact. Phases 1 and 2 never read them (a test enforces this).
LABEL_KEYS = frozenset({
    "category", "anchor_lines", "anchor_chars", "anchor_length",  # per case
    "evaluation", "first_hit_rank", "overlaps_anchor",  # per localizer / top window
    "metrics", "pairwise", "top3", "lexical_reference", "drift",  # artifact level
})
REFERENCE_MEASURES = (
    "line_overlap",
    "passage_contains_anchor",
    "excerpt_80_survives",
    "excerpt_120_survives",
    "excerpt_160_survives",
)

# The accepted real-machine T3e run (T3E-LIVE-RESULTS.md). The baselines recomputed from the
# input artifact must equal these, or the run stops.
T3E_ACCEPTED: dict[str, Any] = {
    "n": 30,
    "hits": {
        "lexical": {"line_overlap": 10, "passage_contains_anchor": 10, "excerpt_80_survives": 2,
                    "excerpt_120_survives": 3, "excerpt_160_survives": 6},
        "semantic": {"line_overlap": 20, "passage_contains_anchor": 20, "excerpt_80_survives": 6,
                     "excerpt_120_survives": 8, "excerpt_160_survives": 12},
    },
    "line_overlap_by_category": {
        "lexical": {"exact": [7, 7], "paraphrase": [0, 10], "conceptual": [0, 5], "distractor": [3, 8]},
        "semantic": {"exact": [6, 7], "paraphrase": [6, 10], "conceptual": [2, 5], "distractor": [6, 8]},
    },
    "pairwise": {
        "line_overlap": {"both_hit": 9, "lexical_only": 1, "semantic_only": 11, "both_miss": 9},
        "excerpt_160_survives": {"both_hit": 5, "lexical_only": 1, "semantic_only": 7, "both_miss": 17},
    },
    "first_hit_rank_histogram": {"1": 20, "2": 1, "3": 1, ">3": 8},
    "hit_in_top3": 22,
}


class T3eInputError(ValueError):
    """The T3e artifact is missing data, malformed, or internally inconsistent. Nothing was scored."""


class T3eReferenceDrift(RuntimeError):
    """The baselines recomputed from the artifact differ from the accepted T3e run."""


# --- phase 1: the artifact, label-free fields only ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class T3eArtifact:
    source: str
    sha256: str | None
    data: Mapping[str, Any]


def load_t3e(path: str | Path) -> T3eArtifact:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise T3eInputError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T3eInputError(f"{path} is not JSON: {exc}") from exc
    return parse_t3e(data, source=str(path), sha256=hashlib.sha256(raw).hexdigest())


def parse_t3e(data: Any, *, source: str = "<memory>", sha256: str | None = None) -> T3eArtifact:
    """Phase 1: check the label-free fields and that the recorded leading excerpts reproduce.

    Never reads a key in ``LABEL_KEYS``; those are validated by ``validate_t3e_labels`` in phase 3.
    """

    if not isinstance(data, Mapping):
        raise T3eInputError("a T3e results.json is a JSON object")
    experiment = _get(data, "experiment", str, "artifact")
    if experiment != T3E_EXPERIMENT:
        raise T3eInputError(f"not a T3e localization artifact: experiment is {experiment!r}")
    config = _get(data, "config", Mapping, "artifact")
    expected_config = {"window_lines": WINDOW_LINES, "stride": STRIDE, "excerpt_budgets": list(EXCERPT_BUDGETS)}
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise T3eInputError(f"config.{key} is {config.get(key)!r}, T3f expects {value!r}")
    cases = _get(data, "cases", list, "artifact")
    if not cases:
        raise T3eInputError("the artifact has no scored cases")
    seen: set[str] = set()
    for number, row in enumerate(cases):
        where = f"cases[{number}]"
        case_id = _get(row, "id", str, where)
        where = f"case {case_id}"
        if case_id in seen:
            raise T3eInputError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        if not _get(row, "query", str, where).strip():
            raise T3eInputError(f"{where}: empty query")
        _get(row, "expected_path", str, where)

        lexical = _get(row, "lexical", Mapping, where)
        lexical_passage = _passage(_get(lexical, "passage", Mapping, f"{where}.lexical"), f"{where}.lexical.passage")
        _check_leading(lexical, lexical_passage, f"{where}.lexical")
        _native_excerpt(lexical, lexical_passage, f"{where}.lexical")

        semantic = _get(row, "semantic", Mapping, where)
        semantic_passage = _passage(_get(semantic, "passage", Mapping, f"{where}.semantic"), f"{where}.semantic.passage")
        _check_leading(semantic, semantic_passage, f"{where}.semantic")
        _get(semantic, "similarity", (int, float), f"{where}.semantic")
        top = _get(semantic, "top", list, f"{where}.semantic")
        ranking = _get(semantic, "ranking", list, f"{where}.semantic")
        if not top or not ranking:
            raise T3eInputError(f"{where}.semantic: empty top windows or ranking")
        for index, window in enumerate(top):
            _get(window, "similarity", (int, float), f"{where}.semantic.top[{index}]")
            _line_range(window, f"{where}.semantic.top[{index}]")
        for index, entry in enumerate(ranking):
            if not (isinstance(entry, list) and len(entry) == 3 and all(_is_int(v) for v in entry[:2])
                    and isinstance(entry[2], (int, float))):
                raise T3eInputError(f"{where}.semantic.ranking[{index}] is not [start_line, end_line, similarity]")
        selected = (semantic_passage.start_line, semantic_passage.end_line)
        if _line_range(top[0], f"{where}.semantic.top[0]") != selected or tuple(ranking[0][:2]) != selected:
            raise T3eInputError(f"{where}.semantic: the selected passage is not the top-ranked window")
    return T3eArtifact(source, sha256, data)


def _check_leading(localized: Mapping[str, Any], passage: Passage, where: str) -> None:
    excerpts = _get(localized, "excerpts", Mapping, where)
    for budget in EXCERPT_BUDGETS:
        recorded = _get(excerpts, str(budget), Mapping, f"{where}.excerpts")
        expected = _excerpt_dict(bounded_excerpt(passage, budget))
        if {key: recorded.get(key) for key in expected} != expected:
            raise T3eInputError(
                f"{where}: the recorded {budget}-character leading excerpt does not reproduce from its passage "
                "(the T3e excerpt rule or the passage text drifted)"
            )


def _native_excerpt(lexical: Mapping[str, Any], passage: Passage, where: str) -> Excerpt:
    native = _get(lexical, "native_excerpt", Mapping, where)
    text = _get(native, "text", str, f"{where}.native_excerpt")
    start = _get(native, "char_start", int, f"{where}.native_excerpt")
    end = _get(native, "char_end", int, f"{where}.native_excerpt")
    core = text.removeprefix("…").removesuffix("…")
    inside = passage.char_start <= start <= end <= passage.char_end
    if not inside or passage.text[start - passage.char_start : end - passage.char_start] != core:
        raise T3eInputError(f"{where}: the native excerpt is not the passage text at its recorded span")
    return Excerpt(text, start, end, core != text)


def _passage(obj: Mapping[str, Any], where: str) -> Passage:
    start_line, end_line = _line_range(obj, where)
    text = _get(obj, "text", str, where)
    char_start = _get(obj, "char_start", int, where)
    char_end = _get(obj, "char_end", int, where)
    if char_start < 0 or char_end - char_start != len(text):
        raise T3eInputError(f"{where}: char span does not match the passage text length")
    if text.count("\n") != end_line - start_line:
        raise T3eInputError(f"{where}: line range does not match the passage text")
    return Passage(start_line, end_line, text, char_start, char_end)


def _line_range(obj: Mapping[str, Any], where: str) -> tuple[int, int]:
    start, end = _get(obj, "start_line", int, where), _get(obj, "end_line", int, where)
    if start < 1 or end < start:
        raise T3eInputError(f"{where}: invalid line range {start}-{end}")
    return start, end


def _pair(row: Mapping[str, Any], key: str, where: str) -> tuple[int, int]:
    value = _get(row, key, list, where)
    if len(value) != 2 or not all(_is_int(v) for v in value) or value[1] < value[0]:
        raise T3eInputError(f"{where}.{key} is not an ordered [start, end] pair")
    return value[0], value[1]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _get(mapping: Any, key: str, kind: Any, where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise T3eInputError(f"{where}: missing {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        raise T3eInputError(f"{where}.{key} has the wrong type ({type(value).__name__})")
    return value


# --- phase 2: label-free decisions ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProjectionCase:
    """Everything projection and fusion may see. No category, anchor, or evaluation."""

    id: str
    query: str
    path: str
    lexical_passage: Passage
    lexical_native_excerpt: Excerpt
    semantic_passage: Passage
    semantic_similarity: float
    semantic_top: tuple[tuple[int, int, float], ...]  # (start_line, end_line, similarity), best first


def label_free_cases(artifact: T3eArtifact) -> tuple[ProjectionCase, ...]:
    cases = []
    for row in artifact.data["cases"]:
        lexical, semantic = row["lexical"], row["semantic"]
        where = f"case {row['id']}"
        lexical_passage = _passage(lexical["passage"], where)
        cases.append(
            ProjectionCase(
                id=row["id"],
                query=row["query"],
                path=row["expected_path"],
                lexical_passage=lexical_passage,
                lexical_native_excerpt=_native_excerpt(lexical, lexical_passage, where),
                semantic_passage=_passage(semantic["passage"], where),
                semantic_similarity=float(semantic["similarity"]),
                semantic_top=tuple((w["start_line"], w["end_line"], float(w["similarity"])) for w in semantic["top"]),
            )
        )
    return tuple(cases)


def passages_overlap(a: Passage, b: Passage) -> bool:
    return lines_overlap((a.start_line, a.end_line), (b.start_line, b.end_line))


def build_packet(case: ProjectionCase, strategy: str, total: int) -> dict[str, Any]:
    """An ordered evidence packet of at most ``total`` source characters.

    ``semantic_primary_dual_on_disagreement``: the semantic passage alone with the whole budget when
    the two selected line ranges overlap; otherwise semantic then lexical, half the budget each.
    ``semantic_only``: the semantic passage alone with the whole budget. Contents are query-aware
    projections.
    """

    if strategy not in STRATEGIES:
        raise ValueError(f"unknown packet strategy {strategy!r}")
    parts = [("semantic", case.semantic_passage)]
    dual = strategy == FUSION and not passages_overlap(case.semantic_passage, case.lexical_passage)
    if dual:
        parts.append(("lexical", case.lexical_passage))
    share = total // len(parts)
    segments = []
    for source, passage in parts:
        excerpt = project_query_aware(passage, case.query, share)
        segments.append(
            {"source": source, "start_line": passage.start_line, "end_line": passage.end_line, **_excerpt_dict(excerpt)}
        )
    return {
        "dual": dual,
        "budget_per_segment": share,
        "segments": segments,
        "source_chars": sum(s["char_end"] - s["char_start"] for s in segments),
    }


def decide(case: ProjectionCase) -> dict[str, Any]:
    """Every projection and packet for one case: a pure function of the label-free case."""

    passages = {"lexical": case.lexical_passage, "semantic": case.semantic_passage}
    return {
        "id": case.id,
        "passages": {
            source: {"start_line": p.start_line, "end_line": p.end_line, "char_start": p.char_start, "char_end": p.char_end}
            for source, p in passages.items()
        },
        "anchors": {source: asdict(query_anchor(p, case.query)) for source, p in passages.items()},
        "projections": {
            source: {
                "leading": {str(b): _excerpt_dict(bounded_excerpt(p, b)) for b in PROJECTION_BUDGETS},
                "query_aware": {str(b): _excerpt_dict(project_query_aware(p, case.query, b)) for b in PROJECTION_BUDGETS},
            }
            for source, p in passages.items()
        },
        "native_excerpt": _excerpt_dict(case.lexical_native_excerpt),
        "packets": {strategy: {str(t): build_packet(case, strategy, t) for t in PACKET_TOTALS} for strategy in STRATEGIES},
    }


def freeze(decisions: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """A detached copy of every decision and its SHA-256, taken before any label is read."""

    canonical = json.dumps(list(decisions), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return json.loads(canonical), hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _excerpt_dict(excerpt: Excerpt) -> dict[str, Any]:
    return {"text": excerpt.text, "char_start": excerpt.char_start, "char_end": excerpt.char_end}


# --- phase 3: labels -----------------------------------------------------------------------------------


def score_spans(spans: Sequence[tuple[int, int]], anchor: tuple[int, int]) -> dict[str, Any]:
    """``score_span`` over the union of several emitted source spans.

    The evidence survives when the union holds the whole anchor, or one span is
    entirely anchor text (as for a single excerpt).
    """

    covered, reach = 0, anchor[0]
    for start, end in sorted(s for s in spans if s[1] > s[0]):
        start, end = max(start, reach), min(end, anchor[1])
        if end > start:
            covered += end - start
            reach = end
    whole = covered == anchor[1] - anchor[0]
    inside = any(anchor[0] <= s and e <= anchor[1] and e > s for s, e in spans)
    return {"contains_anchor": whole, "survives": whole or inside, "coverage": covered / (anchor[1] - anchor[0])}


def validate_t3e_labels(artifact: T3eArtifact) -> tuple[str, ...]:
    """Phase 3: check the label-bearing fields and return the benchmark categories."""

    data = artifact.data
    metrics = _get(data, "metrics", Mapping, "artifact")
    for localizer in SOURCES:
        _get(_get(_get(metrics, localizer, Mapping, "metrics"), "overall", Mapping, f"metrics.{localizer}"),
             "hits", Mapping, f"metrics.{localizer}.overall")
    categories = tuple(_get(_get(metrics, "lexical", Mapping, "metrics"), "by_category", Mapping, "metrics.lexical"))
    _get(data, "pairwise", Mapping, "artifact")
    _get(data, "top3", Mapping, "artifact")
    for row in data["cases"]:
        where = f"case {row['id']}"
        if _get(row, "category", str, where) not in categories:
            raise T3eInputError(f"{where}: category is not in the recorded metrics")
        anchor_lines = _pair(row, "anchor_lines", where)
        anchor_chars = _pair(row, "anchor_chars", where)
        if anchor_lines[0] < 1 or anchor_chars[0] < 0 or anchor_chars[1] <= anchor_chars[0]:
            raise T3eInputError(f"{where}: anchor span is empty or out of range")
        if _get(row, "anchor_length", int, where) != anchor_chars[1] - anchor_chars[0]:
            raise T3eInputError(f"{where}: anchor_length disagrees with anchor_chars")
        for localizer in SOURCES:
            _get(row[localizer], "evaluation", Mapping, f"{where}.{localizer}")
    return categories


def reproduce_t3e(artifact: T3eArtifact, categories: Sequence[str] | None = None) -> dict[str, Any]:
    """Re-score the recorded T3e selections, require the recorded evaluation, return the baseline."""

    if categories is None:
        categories = validate_t3e_labels(artifact)
    data = artifact.data
    rows = data["cases"]
    for row in rows:
        anchor, anchor_lines = tuple(row["anchor_chars"]), tuple(row["anchor_lines"])
        for localizer in SOURCES:
            recorded = row[localizer]
            if _t3e_evaluation(recorded, anchor, anchor_lines) != recorded["evaluation"]:
                raise T3eInputError(f"case {row['id']}: the recorded {localizer} evaluation does not reproduce")
        semantic = row["semantic"]
        overlaps = [lines_overlap((w["start_line"], w["end_line"]), anchor_lines) for w in semantic["top"]]
        if overlaps != [w.get("overlaps_anchor") for w in semantic["top"]]:
            raise T3eInputError(f"case {row['id']}: the recorded top-window evidence flags do not reproduce")
        first = next((rank for rank, (s, e, _) in enumerate(semantic["ranking"], 1) if lines_overlap((s, e), anchor_lines)), None)
        if first != semantic.get("first_hit_rank"):
            raise T3eInputError(f"case {row['id']}: the recorded first-hit rank does not reproduce")

    categories = list(categories)
    metrics = {localizer: summarize(rows, localizer, categories) for localizer in SOURCES}
    pairs = {measure: pairwise(rows, measure) for measure in ("line_overlap", "excerpt_160_survives")}
    top3 = {
        "first_hit_rank_histogram": _histogram(row["semantic"]["first_hit_rank"] for row in rows),
        "hit_in_top3": sum(1 for row in rows if any(w["overlaps_anchor"] for w in row["semantic"]["top"])),
    }
    for name, recomputed in (("metrics", metrics), ("pairwise", pairs), ("top3", top3)):
        recorded = {key: data[name].get(key) for key in recomputed}
        if recomputed != recorded:
            raise T3eInputError(f"the recorded T3e {name} do not reproduce from the recorded cases")

    return {
        "n": len(rows),
        "hits": {loc: {m: metrics[loc]["overall"]["hits"][m] for m in REFERENCE_MEASURES} for loc in SOURCES},
        "line_overlap_by_category": {
            loc: {c: [metrics[loc]["by_category"][c]["hits"].get("line_overlap", 0), metrics[loc]["by_category"][c]["n"]]
                  for c in categories}
            for loc in SOURCES
        },
        "pairwise": {measure: {bucket: len(ids) for bucket, ids in buckets.items()} for measure, buckets in pairs.items()},
        **top3,
    }


def check_reference(baseline: Mapping[str, Any], reference: Mapping[str, Any] | None) -> dict[str, Any]:
    if reference is None:
        return {"checked": False, "recomputed": dict(baseline)}
    differences = _differences(reference, baseline)
    if differences:
        raise T3eReferenceDrift(
            "the T3e baselines recomputed from this artifact differ from the accepted live run: "
            + "; ".join(f"{path}: accepted {want!r}, recomputed {got!r}" for path, want, got in differences)
        )
    return {"checked": True, "matches": True, "recomputed": dict(baseline)}


def _differences(want: Any, got: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    if isinstance(want, Mapping) and isinstance(got, Mapping):
        out: list[tuple[str, Any, Any]] = []
        for key in sorted(set(want) | set(got), key=str):
            out += _differences(want.get(key), got.get(key), f"{path}.{key}" if path else str(key))
        return out
    return [] if want == got else [(path, want, got)]


def _t3e_evaluation(localized: Mapping[str, Any], anchor: tuple[int, int], anchor_lines: tuple[int, int]) -> dict[str, Any]:
    """evidence.evaluate() from recorded spans and the recorded anchor lines (no document text)."""

    passage = localized["passage"]
    evaluation = {
        "line_overlap": lines_overlap((passage["start_line"], passage["end_line"]), anchor_lines),
        "passage": score_span((passage["char_start"], passage["char_end"]), anchor),
        "excerpts": {b: score_span((e["char_start"], e["char_end"]), anchor) for b, e in localized["excerpts"].items()},
    }
    native = localized.get("native_excerpt")
    if native is not None:
        evaluation["native_excerpt"] = score_span((native["char_start"], native["char_end"]), anchor)
    return evaluation


def evaluate_decision(decision: Mapping[str, Any], anchor: tuple[int, int], anchor_lines: tuple[int, int]) -> dict[str, Any]:
    """Score one case's frozen decisions against its anchor."""

    def span(excerpt: Mapping[str, Any]) -> tuple[int, int]:
        return excerpt["char_start"], excerpt["char_end"]

    projection = {}
    for source in SOURCES:
        passage = decision["passages"][source]
        projection[source] = {
            "line_overlap": lines_overlap((passage["start_line"], passage["end_line"]), anchor_lines),
            "passage": score_span(span(passage), anchor),
            **{
                method: {b: score_span(span(e), anchor) for b, e in decision["projections"][source][method].items()}
                for method in METHODS
            },
        }
    packets = {}
    for strategy in STRATEGIES:
        packets[strategy] = {}
        for total, packet in decision["packets"][strategy].items():
            segments = packet["segments"]
            packets[strategy][total] = {
                **score_spans([span(s) for s in segments], anchor),
                "line_overlap": any(lines_overlap((s["start_line"], s["end_line"]), anchor_lines) for s in segments),
                "dual": packet["dual"],
                "source_chars": packet["source_chars"],
            }
    return {
        "projection": projection,
        "native_excerpt": score_span(span(decision["native_excerpt"]), anchor),
        "packets": packets,
    }


# --- aggregation ---------------------------------------------------------------------------------------


def _block(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(scores)
    return {
        "n": n,
        "survives": sum(1 for s in scores if s["survives"]),
        "contains_anchor": sum(1 for s in scores if s["contains_anchor"]),
        "mean_coverage": sum(s["coverage"] for s in scores) / n if n else None,
    }


def _by_category(rows: Sequence[Mapping[str, Any]], categories: Sequence[str], pick) -> dict[str, Any]:
    return {
        "overall": _block([pick(r) for r in rows]),
        "by_category": {c: _block([pick(r) for r in rows if r["category"] == c]) for c in categories},
    }


def _buckets(rows: Sequence[Mapping[str, Any]], a: str, b: str, hit_a, hit_b) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"both_hit": [], f"{a}_only": [], f"{b}_only": [], "both_miss": []}
    for row in rows:
        x, y = hit_a(row), hit_b(row)
        buckets["both_hit" if x and y else f"{a}_only" if x else f"{b}_only" if y else "both_miss"].append(row["id"])
    return buckets


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "max": max(values)}


def aggregate(rows: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> dict[str, Any]:
    def projected(source: str, method: str, budget: str):
        return lambda r: r["evaluation"]["projection"][source][method][budget]

    def packet(strategy: str, total: str):
        return lambda r: r["evaluation"]["packets"][strategy][total]

    def chars(source: str, method: str, budget: str) -> list[int]:
        return [e["char_end"] - e["char_start"] for e in (r["decision"]["projections"][source][method][budget] for r in rows)]

    projection = {
        source: {
            method: {
                str(b): {**_by_category(rows, categories, projected(source, method, str(b))),
                         "source_chars": _stats(chars(source, method, str(b)))}
                for b in PROJECTION_BUDGETS
            }
            for method in METHODS
        }
        for source in SOURCES
    }
    projection_pairwise = {
        source: {
            str(b): _buckets(
                rows, "leading", "query_aware",
                lambda r, s=source, b=str(b): projected(s, "leading", b)(r)["survives"],
                lambda r, s=source, b=str(b): projected(s, "query_aware", b)(r)["survives"],
            )
            for b in PROJECTION_BUDGETS
        }
        for source in SOURCES
    }

    packets = {}
    for strategy in STRATEGIES:
        packets[strategy] = {}
        for total in (str(t) for t in PACKET_TOTALS):
            scores = [packet(strategy, total)(r) for r in rows]
            packets[strategy][total] = {
                **_by_category(rows, categories, packet(strategy, total)),
                "line_overlap": sum(1 for s in scores if s["line_overlap"]),
                "dual_cases": sum(1 for s in scores if s["dual"]),
                "source_chars": _stats([s["source_chars"] for s in scores]),
            }
    packet_pairwise = {
        str(t): _buckets(
            rows, "fusion", "semantic_only",
            lambda r, t=str(t): packet(FUSION, t)(r)["survives"],
            lambda r, t=str(t): packet(CONTROL, t)(r)["survives"],
        )
        for t in PACKET_TOTALS
    }

    retention = {}
    for source in SOURCES:
        hits = [r for r in rows if r["evaluation"]["projection"][source]["line_overlap"]]
        retention[source] = {
            "line_overlap": len(hits),
            "line_overlap_ids": [r["id"] for r in hits],
            "passage_contains_anchor": sum(
                1 for r in rows if r["evaluation"]["projection"][source]["passage"]["contains_anchor"]
            ),
            "survives_among_line_hits": {
                method: {str(b): sum(1 for r in hits if projected(source, method, str(b))(r)["survives"]) for b in PROJECTION_BUDGETS}
                for method in METHODS
            },
        }

    anchors = {
        source: {
            "center_fallback": sum(1 for r in rows if r["decision"]["anchors"][source]["mode"] == "center"),
            "center_fallback_by_category": {
                c: sum(1 for r in rows if r["category"] == c and r["decision"]["anchors"][source]["mode"] == "center")
                for c in categories
            },
        }
        for source in SOURCES
    }

    lexical_hit = [r["evaluation"]["projection"]["lexical"]["line_overlap"] for r in rows]
    semantic_hit = [r["evaluation"]["projection"]["semantic"]["line_overlap"] for r in rows]
    headroom = {
        "top1_oracle_line_overlap": sum(1 for x, y in zip(lexical_hit, semantic_hit) if x or y),
        "semantic_top1_line_overlap": sum(semantic_hit),
        "lexical_top1_line_overlap": sum(lexical_hit),
        "semantic_top3_line_overlap": sum(1 for r in rows if r["semantic_top3_hit"]),
    }

    return {
        "projection": projection,
        "projection_pairwise": projection_pairwise,
        "native_excerpt": _by_category(rows, categories, lambda r: r["evaluation"]["native_excerpt"]),
        "retention": retention,
        "packets": packets,
        "packet_pairwise": packet_pairwise,
        "anchors": anchors,
        "headroom": headroom,
    }


# --- the run -------------------------------------------------------------------------------------------


def run_t3f(t3e_path: str | Path, *, reference: Mapping[str, Any] | None = T3E_ACCEPTED) -> dict[str, Any]:
    created_at = _now()
    artifact = load_t3e(t3e_path)
    return run_t3f_artifact(artifact, reference=reference, created_at=created_at)


def run_t3f_artifact(
    artifact: T3eArtifact, *, reference: Mapping[str, Any] | None = T3E_ACCEPTED, created_at: str | None = None
) -> dict[str, Any]:
    # Phase 2: every decision is made and frozen from label-free cases.
    cases = label_free_cases(artifact)
    decisions, decisions_sha256 = freeze([decide(case) for case in cases])

    # Phase 3: labels from here on.
    categories = list(validate_t3e_labels(artifact))
    baseline = reproduce_t3e(artifact, categories)
    reference_check = check_reference(baseline, reference)
    rows = []
    for row, decision in zip(artifact.data["cases"], decisions, strict=True):
        anchor, anchor_lines = tuple(row["anchor_chars"]), tuple(row["anchor_lines"])
        rows.append(
            {
                "id": row["id"],
                "category": row["category"],
                "query": row["query"],
                "expected_path": row["expected_path"],
                "anchor_lines": list(anchor_lines),
                "anchor_chars": list(anchor),
                "anchor_length": row["anchor_length"],
                "semantic_top3_hit": any(w["overlaps_anchor"] for w in row["semantic"]["top"][:TOP_K]),
                "decision": decision,
                "evaluation": evaluate_decision(decision, anchor, anchor_lines),
            }
        )
    metrics = aggregate(rows, categories)
    _check_consistency(metrics, baseline)
    source_config = artifact.data["config"]
    return {
        "experiment": EXPERIMENT,
        "created_at": created_at or _now(),
        "input": {
            "t3e_results": artifact.source,
            "t3e_sha256": artifact.sha256,
            "t3e_started_at": artifact.data.get("started_at"),
            "t3e_embedder": source_config.get("embedder"),
            "cases": len(rows),
        },
        "t3e_reference": reference_check,
        "decisions_sha256": decisions_sha256,
        "config": {
            "projection_budgets": list(PROJECTION_BUDGETS),
            "packet_totals": list(PACKET_TOTALS),
            "leading_projection": "T3e bounded_excerpt, unchanged: " + str(source_config.get("excerpt_rule")),
            "query_aware_projection": (
                "whole trimmed passage if it fits; else anchor at the earliest query match in the strongest line "
                "(distinct phrases, then occurrences, then earliest line), keeping a quarter of the budget before "
                "it; with no match, centre on the passage; shift back inside the passage at an edge; move a cut "
                "that splits a word (letters/digits) to the word boundary if that keeps half the budget; drop "
                "edge whitespace; '…' marks omitted passage text and is not counted"
            ),
            "lead_share": f"1/{LEAD_DIVISOR} of the budget",
            "query_matcher": "zomah.knowledge._query_phrases and _line_matches, unchanged (case- and "
            "diacritic-insensitive, whole words, compound tokens as word sequences)",
            "fusion": {
                FUSION: "semantic passage first; when the lexical and semantic line ranges are disjoint, add the "
                "lexical passage second and split the total budget evenly; otherwise the semantic passage alone "
                "gets the whole total budget",
                CONTROL: "the semantic passage alone gets the whole total budget",
            },
            "survives_definition": "the emitted spans together hold the whole anchor, or one span is entirely "
            "anchor text (T3e's definition, applied to the union of spans)",
            "labels": "read only after all decisions were frozen (decisions_sha256 is taken before)",
        },
        "metrics": metrics,
        "categories": categories,
        "cases": rows,
    }


def _histogram(values) -> dict[str, int]:
    """T3e's first-hit-rank histogram (localization_runner._histogram)."""

    histogram: dict[str, int] = {}
    for value in values:
        key = "none" if value is None else str(value) if value <= TOP_K else f">{TOP_K}"
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


def _check_consistency(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    """T3f's own scoring of the frozen T3e selections must equal the reproduced T3e baseline."""

    observed = {
        source: {
            "line_overlap": metrics["retention"][source]["line_overlap"],
            "passage_contains_anchor": metrics["retention"][source]["passage_contains_anchor"],
            **{
                f"excerpt_{b}_survives": metrics["projection"][source]["leading"][str(b)]["overall"]["survives"]
                for b in PROJECTION_BUDGETS
            },
        }
        for source in SOURCES
    }
    headroom = metrics["headroom"]
    pairs = baseline["pairwise"]["line_overlap"]
    expected_headroom = {
        "top1_oracle_line_overlap": pairs["both_hit"] + pairs["lexical_only"] + pairs["semantic_only"],
        "semantic_top1_line_overlap": baseline["hits"]["semantic"]["line_overlap"],
        "lexical_top1_line_overlap": baseline["hits"]["lexical"]["line_overlap"],
        "semantic_top3_line_overlap": baseline["hit_in_top3"],
    }
    if observed != baseline["hits"] or headroom != expected_headroom:
        raise RuntimeError("T3f scoring of the frozen T3e selections disagrees with the reproduced T3e baseline")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

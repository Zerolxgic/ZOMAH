"""T3f: query-aware projection, packet fusion, artifact validation, label isolation, and baselines."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

pytest.importorskip("zomah")

from embedding_bench import localization_fusion as t3f  # noqa: E402
from embedding_bench.evidence import score_span  # noqa: E402
from embedding_bench.localization_fusion import (  # noqa: E402
    CONTROL,
    FUSION,
    PACKET_TOTALS,
    PROJECTION_BUDGETS,
    T3E_ACCEPTED,
    ProjectionCase,
    T3eInputError,
    T3eReferenceDrift,
    build_packet,
    check_reference,
    decide,
    freeze,
    label_free_cases,
    parse_t3e,
    reproduce_t3e,
    run_t3f_artifact,
    score_spans,
)
from embedding_bench.localization_fusion_report import render_markdown, render_summary  # noqa: E402
from embedding_bench.passages import Excerpt, Passage, bounded_excerpt  # noqa: E402
from embedding_bench.projection import project_query_aware, query_anchor  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
OFFSET = 1000


def passage(*lines: str, start_line: int = 10, offset: int = OFFSET) -> Passage:
    text = "\n".join(lines)
    return Passage(start_line, start_line + len(lines) - 1, text, offset, offset + len(text))


def core(excerpt: Excerpt) -> str:
    return excerpt.text.removeprefix("…").removesuffix("…")


def check_invariants(p: Passage, query: str, budget: int) -> Excerpt:
    """Properties every query-aware projection must have."""

    excerpt = project_query_aware(p, query, budget)
    body = core(excerpt)
    start, end = excerpt.char_start - p.char_start, excerpt.char_end - p.char_start
    trimmed = p.text.strip()
    assert p.text[start:end] == body  # exact source span
    assert len(body) <= budget
    assert body == body.strip()
    assert excerpt.text.startswith("…") == bool(p.text[:start].strip())  # "…" only for real omitted text
    assert excerpt.text.endswith("…") == bool(p.text[end:].strip())
    assert excerpt.truncated == (body != trimmed)
    if len(trimmed) <= budget:
        assert excerpt.text == trimmed and not excerpt.truncated
    anchor = query_anchor(p, query)
    if anchor.mode == "match" and excerpt.truncated:
        assert start <= anchor.offset < end  # the anchoring match is always shown
    return excerpt


# --- projection ---------------------------------------------------------------------------------------


def test_short_passages_are_returned_whole_without_ellipses() -> None:
    p = passage("  Short evidence line.", "second line  ", "")
    for budget in PROJECTION_BUDGETS:
        excerpt = check_invariants(p, "evidence", budget)
        assert excerpt.text == "Short evidence line.\nsecond line"
        assert (excerpt.char_start, excerpt.char_end) == (OFFSET + 2, OFFSET + 2 + len(excerpt.text))
        assert not excerpt.truncated


def test_the_window_keeps_a_quarter_budget_lead_before_the_earliest_match() -> None:
    p = passage("aaaa " * 40 + "target" + " bbbb" * 40)
    excerpt = check_invariants(p, "target", 80)
    assert excerpt.char_start == OFFSET + 200 - 80 // 4
    assert core(excerpt).startswith("aaaa ") and "target" in core(excerpt) and core(excerpt).endswith("bbbb")
    assert excerpt.text.startswith("…") and excerpt.text.endswith("…")
    for budget in PROJECTION_BUDGETS:
        assert check_invariants(p, "target", budget).char_start == OFFSET + 200 - budget // 4


def test_windows_backfill_at_either_passage_edge() -> None:
    near_end = passage("x" * 3 + " filler" * 40 + " target end")
    excerpt = check_invariants(near_end, "target", 80)
    assert excerpt.char_end == near_end.char_end and not excerpt.text.endswith("…")
    assert len(core(excerpt)) >= 70  # the unused tail was spent before the match
    near_start = passage("target begins" + " filler" * 40)
    excerpt = check_invariants(near_start, "target", 80)
    assert excerpt.char_start == near_start.char_start and not excerpt.text.startswith("…")
    assert len(core(excerpt)) >= 70


def test_cuts_move_to_word_boundaries_unless_that_loses_half_the_budget() -> None:
    p = passage("alphabet " * 10 + "target " + "omegaword " * 20)
    excerpt = check_invariants(p, "target", 80)
    start, end = excerpt.char_start - OFFSET, excerpt.char_end - OFFSET
    assert not (p.text[start - 1].isalnum() and p.text[start].isalnum())
    assert not (p.text[end - 1].isalnum() and p.text[end].isalnum())
    unbroken = passage("z" * 300)  # no boundary within reach: the cut must split
    excerpt = check_invariants(unbroken, "nothing", 80)
    assert len(core(excerpt)) == 80


def test_no_match_centres_on_the_passage_instead_of_the_leading_characters() -> None:
    p = passage("first line words " * 4, "middle line words " * 4, "last line words " * 4)
    excerpt = check_invariants(p, "absent zebra", 80)
    assert query_anchor(p, "absent zebra").mode == "center"
    start, end = excerpt.char_start - OFFSET, excerpt.char_end - OFFSET
    middle = len(p.text.strip()) / 2
    assert abs((start + end) / 2 - middle) <= 12  # centred, within word-boundary movement
    assert excerpt.text.startswith("…") and excerpt.text.endswith("…")
    assert excerpt.char_start != bounded_excerpt(p, 80).char_start


def test_projection_is_deterministic_and_position_independent() -> None:
    lines = ("The Tool-Result handler records every result.", "Nothing here.", "Tool results are stored in the trace.")
    for budget in (20, *PROJECTION_BUDGETS):
        first = project_query_aware(passage(*lines), "tool-result trace", budget)
        assert project_query_aware(passage(*lines), "tool-result trace", budget) == first
        moved = project_query_aware(passage(*lines, offset=5), "tool-result trace", budget)
        assert moved.text == first.text and moved.char_start == first.char_start - OFFSET + 5


WORDS = ("tool", "result", "tool-result", "tool_result", "toolkit", "café", "CAFE", "trace", "evidence", "the", "a",
         "https://example.org/" + "x" * 90, "résumé", "line.", "(note)", "—", "state", "", "  ")


def test_projection_invariants_hold_across_generated_passages() -> None:
    rng = random.Random(20260928)
    for _ in range(400):
        lines = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(0, 30))) for _ in range(rng.randint(1, 3))]
        p = passage(*lines, offset=rng.randint(0, 5000))
        query = " ".join(rng.choice(WORDS[:12]) for _ in range(rng.randint(1, 4)))
        for budget in (1, 7, *PROJECTION_BUDGETS, 240, 320):
            check_invariants(p, query, budget)
    with pytest.raises(ValueError):
        project_query_aware(passage("x"), "x", 0)


# --- query matching (ZOMAH's lexical matcher) ------------------------------------------------------------


def anchor_line(query: str, *lines: str) -> int | None:
    return query_anchor(passage(*lines, start_line=1), query).line


def test_matching_uses_zomah_s_own_lexical_matcher() -> None:
    import zomah.knowledge
    from embedding_bench import projection

    assert projection._line_matches is zomah.knowledge._line_matches
    assert projection._query_phrases is zomah.knowledge._query_phrases


def test_matching_is_case_and_diacritic_insensitive() -> None:
    assert anchor_line("tool", "alpha beta", "gamma TOOL delta") == 2
    assert anchor_line("TOOL", "alpha beta", "gamma tool delta") == 2
    assert anchor_line("cafe", "nothing", "Le Café ouvre") == 2
    assert anchor_line("Café", "nothing", "the cafe opens") == 2


def test_compound_tokens_match_as_word_sequences_and_words_are_whole() -> None:
    assert anchor_line("tool-result", "tool here", "a tool result") == 2
    assert anchor_line("tool-result", "tool here", "a tool_result") == 2
    assert anchor_line("tool-result", "result tool", "toolresult", "tool") is None
    assert anchor_line("tool", "toolkit", "retool", "tools") is None
    assert query_anchor(passage("toolkit and tool"), "tool").offset == len("toolkit and ")


def test_strongest_line_ranks_distinct_phrases_then_occurrences_then_earliest() -> None:
    assert anchor_line("tool result", "tool tool tool tool", "tool result") == 2  # distinct beats occurrences
    assert anchor_line("tool result", "tool result", "tool result tool") == 2  # then occurrences
    assert anchor_line("tool result", "tool result", "result tool") == 1  # then the earliest line
    anchor = query_anchor(passage("x", "result then tool", start_line=4), "tool result")
    assert (anchor.mode, anchor.line, anchor.distinct, anchor.occurrences) == ("match", 5, 2, 2)
    assert anchor.offset == len("x\n")  # the earliest match on that line


# --- packets -------------------------------------------------------------------------------------------


def case(semantic: Passage, lexical: Passage, query: str = "tool result") -> ProjectionCase:
    return ProjectionCase(
        id="c", query=query, path="doc.md", lexical_passage=lexical,
        lexical_native_excerpt=Excerpt(lexical.text, lexical.char_start, lexical.char_end, False),
        semantic_passage=semantic, semantic_similarity=0.5, semantic_top=((semantic.start_line, semantic.end_line, 0.5),),
    )


LONG = "tool result " + "words in a long line " * 20


def test_overlapping_selections_emit_one_semantic_excerpt_with_the_whole_budget() -> None:
    semantic = passage(LONG, LONG, LONG, start_line=4, offset=0)
    lexical = passage(LONG, start_line=5, offset=len(LONG) + 1)
    for total in PACKET_TOTALS:
        packet = build_packet(case(semantic, lexical), FUSION, total)
        assert not packet["dual"] and [s["source"] for s in packet["segments"]] == ["semantic"]
        assert packet["budget_per_segment"] == total
        assert packet["source_chars"] <= total
        assert packet == build_packet(case(semantic, lexical), CONTROL, total)


def test_disjoint_selections_emit_semantic_first_then_lexical_on_half_budgets() -> None:
    semantic = passage(LONG, LONG, LONG, start_line=4, offset=0)
    lexical = passage(LONG, start_line=7, offset=3 * (len(LONG) + 1))  # adjacent, not overlapping
    for total in PACKET_TOTALS:
        packet = build_packet(case(semantic, lexical), FUSION, total)
        assert packet["dual"] and [s["source"] for s in packet["segments"]] == ["semantic", "lexical"]
        assert packet["budget_per_segment"] == total // 2
        for segment, source in zip(packet["segments"], (semantic, lexical)):
            projected = project_query_aware(source, "tool result", total // 2)
            assert (segment["text"], segment["char_start"], segment["char_end"]) == (projected.text, projected.char_start, projected.char_end)
            assert segment["char_end"] - segment["char_start"] <= total // 2
        assert packet["source_chars"] <= total
        control = build_packet(case(semantic, lexical), CONTROL, total)
        assert not control["dual"] and control["segments"][0]["char_end"] - control["segments"][0]["char_start"] <= total
    with pytest.raises(ValueError):
        build_packet(case(semantic, lexical), "oracle", 160)


def test_packet_evaluation_unions_emitted_source_spans() -> None:
    anchor = (100, 200)
    assert score_spans([(100, 150), (150, 200)], anchor) == {"contains_anchor": True, "survives": True, "coverage": 1.0}
    assert score_spans([(90, 160), (140, 210)], anchor)["coverage"] == 1.0  # overlap is not double counted
    gap = score_spans([(90, 140), (160, 210)], anchor)
    assert gap == {"contains_anchor": False, "survives": False, "coverage": 0.8}
    assert score_spans([(120, 140), (300, 400)], anchor)["survives"]  # one span entirely evidence, as in T3e
    assert score_spans([(150, 150)], anchor) == {"contains_anchor": False, "survives": False, "coverage": 0.0}
    for start in range(80, 221, 7):
        for end in range(start, 231, 11):
            assert score_spans([(start, end)], anchor) == score_span((start, end), anchor)


# --- the frozen T3e artifact (real corpus, fake embedder) -------------------------------------------------


@pytest.fixture(scope="module")
def t3e_data(tmp_path_factory) -> dict:
    from embedding_bench.localization_runner import run_t3e
    from embedding_bench.retrieval import HashingEmbedder

    result, _ = run_t3e(
        root=REPO_ROOT, manifest_path=EXPERIMENT_DIR / "benchmark.json", embedder_factory=HashingEmbedder,
        workdir=tmp_path_factory.mktemp("t3e-work"),
    )
    return json.loads(json.dumps(result))  # exactly what results.json holds


@pytest.fixture(scope="module")
def t3e_file(t3e_data, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("t3e") / "results.json"
    path.write_text(json.dumps(t3e_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def result(t3e_data) -> dict:
    return run_t3f_artifact(parse_t3e(copy.deepcopy(t3e_data)), reference=None)


def test_t3e_identity_is_pinned_to_the_t3e_runner() -> None:
    from embedding_bench import localization_runner

    assert t3f.T3E_EXPERIMENT == localization_runner.EXPERIMENT
    assert t3f.TOP_K == localization_runner.TOP_K
    ranks = [1, 2, 3, 4, 9, None, 1]
    assert t3f._histogram(ranks) == localization_runner._histogram(ranks)


def test_accepted_reference_is_internally_consistent() -> None:
    ref = T3E_ACCEPTED
    lines, cut = ref["pairwise"]["line_overlap"], ref["pairwise"]["excerpt_160_survives"]
    assert sum(lines.values()) == sum(cut.values()) == ref["n"] == sum(ref["first_hit_rank_histogram"].values())
    for name in ("lexical", "semantic"):
        assert ref["hits"][name]["line_overlap"] == lines["both_hit"] + lines[f"{name}_only"]
        assert ref["hits"][name]["excerpt_160_survives"] == cut["both_hit"] + cut[f"{name}_only"]
        assert sum(hit for hit, _ in ref["line_overlap_by_category"][name].values()) == ref["hits"][name]["line_overlap"]
        assert sum(n for _, n in ref["line_overlap_by_category"][name].values()) == ref["n"]
    assert ref["hit_in_top3"] == sum(ref["first_hit_rank_histogram"][k] for k in ("1", "2", "3"))
    assert ref["hits"]["semantic"]["line_overlap"] == ref["first_hit_rank_histogram"]["1"]


def test_frozen_t3e_baselines_reproduce_from_the_artifact(t3e_data, result) -> None:
    baseline = reproduce_t3e(parse_t3e(copy.deepcopy(t3e_data)))
    # The lexical side is real localize() on this corpus: it must equal the accepted live run.
    assert baseline["hits"]["lexical"] == T3E_ACCEPTED["hits"]["lexical"]
    assert baseline["line_overlap_by_category"]["lexical"] == T3E_ACCEPTED["line_overlap_by_category"]["lexical"]
    # T3f's own scoring of the frozen selections equals what T3e recorded.
    metrics = result["metrics"]
    for source in ("lexical", "semantic"):
        recorded = t3e_data["metrics"][source]["overall"]["hits"]
        for budget in PROJECTION_BUDGETS:
            assert metrics["projection"][source]["leading"][str(budget)]["overall"]["survives"] == recorded[f"excerpt_{budget}_survives"]
            assert metrics["projection"][source]["leading"][str(budget)]["overall"]["contains_anchor"] == recorded[f"excerpt_{budget}_contains_anchor"]
        assert metrics["retention"][source]["line_overlap"] == recorded["line_overlap"]
    assert metrics["native_excerpt"]["overall"]["survives"] == t3e_data["metrics"]["lexical"]["overall"]["hits"]["native_excerpt_survives"]
    pairs = t3e_data["pairwise"]["line_overlap"]
    assert metrics["headroom"] == {
        "top1_oracle_line_overlap": len(pairs["both_hit"]) + len(pairs["lexical_only"]) + len(pairs["semantic_only"]),
        "semantic_top1_line_overlap": len(pairs["both_hit"]) + len(pairs["semantic_only"]),
        "lexical_top1_line_overlap": len(pairs["both_hit"]) + len(pairs["lexical_only"]),
        "semantic_top3_line_overlap": t3e_data["top3"]["hit_in_top3"],
    }


def test_reference_check_passes_on_a_match_and_names_every_drift() -> None:
    same = copy.deepcopy(T3E_ACCEPTED)
    assert check_reference(same, T3E_ACCEPTED)["matches"]
    assert check_reference(same, None) == {"checked": False, "recomputed": same}
    same["hits"]["semantic"]["excerpt_160_survives"] = 11
    same["first_hit_rank_histogram"].pop("3")
    with pytest.raises(T3eReferenceDrift) as drift:
        check_reference(same, T3E_ACCEPTED)
    assert "hits.semantic.excerpt_160_survives: accepted 12, recomputed 11" in str(drift.value)
    assert "first_hit_rank_histogram.3: accepted 1, recomputed None" in str(drift.value)


def test_a_non_accepted_artifact_stops_before_scoring(t3e_data) -> None:
    artifact = parse_t3e(copy.deepcopy(t3e_data))
    with pytest.raises(T3eReferenceDrift, match="hits.semantic.line_overlap"):
        run_t3f_artifact(artifact)
    matching = run_t3f_artifact(artifact, reference=reproduce_t3e(artifact))
    assert matching["t3e_reference"]["checked"] and matching["t3e_reference"]["matches"]


# --- label isolation --------------------------------------------------------------------------------------


def test_projection_and_fusion_inputs_carry_no_labels() -> None:
    assert [f.name for f in dataclasses.fields(ProjectionCase)] == [
        "id", "query", "path", "lexical_passage", "lexical_native_excerpt",
        "semantic_passage", "semantic_similarity", "semantic_top",
    ]
    assert list(inspect.signature(decide).parameters) == ["case"]
    assert list(inspect.signature(build_packet).parameters) == ["case", "strategy", "total"]
    assert list(inspect.signature(project_query_aware).parameters) == ["passage", "query", "budget"]
    assert list(inspect.signature(query_anchor).parameters) == ["passage", "query"]


def relabelled(data: dict) -> dict:
    """Same passages and queries; different categories, anchors, and every evaluation field."""

    data = copy.deepcopy(data)
    names = list(data["metrics"]["lexical"]["by_category"])
    for i, row in enumerate(data["cases"]):
        row["category"] = names[(names.index(row["category"]) + 1 + i) % len(names)]
        row["anchor_chars"] = [0, 1]
        row["anchor_lines"] = [1, 1]
        row["anchor_length"] = 1
        for side in ("lexical", "semantic"):
            row[side]["evaluation"] = {"line_overlap": i % 2 == 0}
        row["semantic"]["first_hit_rank"] = None
        for window in row["semantic"]["top"]:
            window["overlaps_anchor"] = not window.get("overlaps_anchor")
    return data


def decisions_sha(data: dict) -> str:
    return freeze([decide(c) for c in t3f.label_free_cases(parse_t3e(data))])[1]


def test_labels_never_change_a_decision(t3e_data, result) -> None:
    assert decisions_sha(relabelled(t3e_data)) == decisions_sha(copy.deepcopy(t3e_data)) == result["decisions_sha256"]


def test_mutation_a_label_leak_would_be_caught(t3e_data, monkeypatch) -> None:
    honest = t3f.label_free_cases

    def leaky(artifact):
        # Steers projection with the evidence text under the anchor, as a label leak would.
        by_id = {row["id"]: row for row in artifact.data["cases"]}
        leaked = []
        for c in honest(artifact):
            start, end = by_id[c.id]["anchor_chars"]
            p = c.semantic_passage
            evidence = p.text[max(0, start - p.char_start) : max(0, end - p.char_start)]
            leaked.append(dataclasses.replace(c, query=f"{c.query} {evidence}"))
        return tuple(leaked)

    monkeypatch.setattr(t3f, "label_free_cases", leaky)
    assert decisions_sha(relabelled(t3e_data)) != decisions_sha(copy.deepcopy(t3e_data))


class LabelReadBeforeFreeze(AssertionError):
    pass


class Tripwire:
    def __init__(self) -> None:
        self.armed = True
        self.read_after_freeze: set[str] = set()


class Guarded(Mapping):
    """Read-only view of artifact JSON: any label-bearing key raises while the tripwire is armed."""

    def __init__(self, data: dict, trip: Tripwire) -> None:
        self._data, self._trip = data, trip

    def _touch(self, key: str) -> None:
        if key in t3f.LABEL_KEYS:
            if self._trip.armed:
                raise LabelReadBeforeFreeze(key)
            self._trip.read_after_freeze.add(key)

    def __getitem__(self, key):
        self._touch(key)
        return guard(self._data[key], self._trip)

    def __contains__(self, key) -> bool:
        self._touch(key)
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def guard(value, trip: Tripwire):
    if isinstance(value, dict):
        return Guarded(value, trip)
    if isinstance(value, list):
        return [guard(item, trip) for item in value]
    return value


def test_no_label_bearing_key_is_read_before_decisions_are_frozen(t3e_data, result, monkeypatch) -> None:
    trip = Tripwire()
    real_freeze = t3f.freeze
    frozen: list[str] = []

    def freeze_then_disarm(decisions):
        out = real_freeze(decisions)
        frozen.append(out[1])
        trip.armed = False  # labels become readable only once freeze has returned
        return out

    monkeypatch.setattr(t3f, "freeze", freeze_then_disarm)
    artifact = parse_t3e(guard(copy.deepcopy(t3e_data), trip))  # phase 1, armed
    guarded = run_t3f_artifact(artifact, reference=None)  # phase 2 armed until freeze returns, then phase 3

    assert frozen == [guarded["decisions_sha256"]] == [result["decisions_sha256"]]
    # Phase 3 reads labels through the same view, so the tripwire does see these access paths.
    assert {"category", "anchor_lines", "anchor_chars", "anchor_length", "evaluation", "first_hit_rank",
            "overlaps_anchor", "metrics", "pairwise", "top3"} <= trip.read_after_freeze
    # And label validation run early would have tripped it.
    with pytest.raises(LabelReadBeforeFreeze):
        t3f.validate_t3e_labels(parse_t3e(guard(copy.deepcopy(t3e_data), Tripwire())))


def test_decisions_are_frozen_before_evaluation_and_recorded_verbatim(t3e_data, result) -> None:
    cases = label_free_cases(parse_t3e(copy.deepcopy(t3e_data)))
    frozen, sha = freeze([decide(c) for c in cases])
    assert sha == result["decisions_sha256"]
    assert [row["decision"] for row in result["cases"]] == frozen
    canonical = json.dumps(frozen, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == sha


def test_every_packet_on_the_real_passages_respects_its_budget(result) -> None:
    duals = 0
    for row in result["cases"]:
        decision = row["decision"]
        sem, lex = decision["passages"]["semantic"], decision["passages"]["lexical"]
        disjoint = sem["end_line"] < lex["start_line"] or lex["end_line"] < sem["start_line"]
        for total in PACKET_TOTALS:
            packet = decision["packets"][FUSION][str(total)]
            assert packet["dual"] == disjoint and packet["segments"][0]["source"] == "semantic"
            assert len(packet["segments"]) == (2 if disjoint else 1)
            assert packet["source_chars"] == sum(s["char_end"] - s["char_start"] for s in packet["segments"]) <= total
            assert all(s["char_end"] - s["char_start"] <= packet["budget_per_segment"] for s in packet["segments"])
            assert not decision["packets"][CONTROL][str(total)]["dual"]
        for source in ("lexical", "semantic"):
            for budget in PROJECTION_BUDGETS:
                for method in ("leading", "query_aware"):
                    excerpt = decision["projections"][source][method][str(budget)]
                    assert excerpt["char_end"] - excerpt["char_start"] <= budget
        duals += disjoint
    assert result["metrics"]["packets"][FUSION]["160"]["dual_cases"] == duals


def test_repeated_runs_are_identical(t3e_data, result) -> None:
    again = run_t3f_artifact(parse_t3e(copy.deepcopy(t3e_data)), reference=None, created_at=result["created_at"])
    assert again == result


# --- malformed or drifted input -------------------------------------------------------------------------------


def _set(path: str, value):
    def mutate(data: dict) -> None:
        *parents, last = path.split(".")
        target = data
        for key in parents:
            target = target[int(key)] if isinstance(target, list) else target[key]
        if isinstance(target, list):
            target[int(last)] = value
        else:
            target[last] = value
    return mutate


def _edit_text(side: str):
    def mutate(data: dict) -> None:
        passage = data["cases"][0][side]["passage"]
        passage["text"] = passage["text"][::-1]  # same length and line count, different text
    return mutate


def _duplicate_id(data: dict) -> None:
    data["cases"][1]["id"] = data["cases"][0]["id"]


def _delete(path: str):
    def mutate(data: dict) -> None:
        *parents, last = path.split(".")
        target = data
        for key in parents:
            target = target[int(key)] if isinstance(target, list) else target[key]
        del target[last]
    return mutate


def _shift_top_window(data: dict) -> None:
    window = data["cases"][0]["semantic"]["top"][0]
    window["start_line"] += 1
    window["end_line"] += 1


MALFORMED = [
    (_set("experiment", "T3a: something else"), "not a T3e localization artifact"),
    (_set("config.window_lines", 4), "config.window_lines"),
    (_set("config.excerpt_budgets", [80, 160]), "config.excerpt_budgets"),
    (_set("cases", []), "no scored cases"),
    (_delete("cases.0.query"), "missing 'query'"),
    (_duplicate_id, "duplicate case id"),
    (_edit_text("semantic"), "leading excerpt does not reproduce"),
    (_edit_text("lexical"), "leading excerpt does not reproduce"),
    (_set("cases.0.semantic.passage.char_end", -1), "char span does not match"),
    (_set("cases.0.lexical.native_excerpt.char_start", -1), "native excerpt"),
    (_shift_top_window, "not the top-ranked window"),
    (_set("cases.0.semantic.ranking.0", [1, 3]), "ranking[0]"),
    (_set("cases.0.semantic.similarity", "0.9"), "wrong type"),
    (_set("cases.0.semantic.top", []), "empty top windows"),
]


@pytest.mark.parametrize("mutate, message", MALFORMED)
def test_malformed_or_incompatible_artifacts_fail_clearly(t3e_data, mutate, message) -> None:
    data = copy.deepcopy(t3e_data)
    mutate(data)
    with pytest.raises(T3eInputError, match=message.replace("[", r"\[").replace("]", r"\]")):
        parse_t3e(data)


def _flip_first_evaluation(data: dict) -> None:
    evaluation = data["cases"][0]["lexical"]["evaluation"]
    evaluation["line_overlap"] = not evaluation["line_overlap"]


def _bump_metric(data: dict) -> None:
    data["metrics"]["semantic"]["overall"]["hits"]["excerpt_160_survives"] += 1


def _bump_top3(data: dict) -> None:
    data["top3"]["hit_in_top3"] += 1


LABEL_PROBLEMS = [
    (_set("cases.0.category", "unknown"), "category is not in the recorded metrics"),
    (_set("cases.0.anchor_length", 0), "anchor_length"),
    (_set("cases.0.anchor_chars", [5, 5]), "anchor span is empty"),
    (_delete("cases.0.anchor_lines"), "missing 'anchor_lines'"),
    (_delete("cases.0.semantic.evaluation"), "missing 'evaluation'"),
    (_delete("metrics"), "missing 'metrics'"),
    (_flip_first_evaluation, "recorded lexical evaluation does not reproduce"),
    (_set("cases.0.semantic.first_hit_rank", 999), "first-hit rank does not reproduce"),
    (_bump_metric, "recorded T3e metrics do not reproduce"),
    (_bump_top3, "recorded T3e top3 do not reproduce"),
]


@pytest.mark.parametrize("mutate, message", LABEL_PROBLEMS)
def test_label_problems_fail_in_phase_3_before_scoring(t3e_data, mutate, message) -> None:
    data = copy.deepcopy(t3e_data)
    mutate(data)
    artifact = parse_t3e(data)  # phase 1 never looks at labels, so it accepts this
    with pytest.raises(T3eInputError, match=message):
        run_t3f_artifact(artifact, reference=None)


def test_unreadable_or_non_json_input_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(T3eInputError, match="cannot read"):
        t3f.load_t3e(tmp_path / "missing.json")
    bad = tmp_path / "results.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(T3eInputError, match="is not JSON"):
        t3f.load_t3e(bad)
    with pytest.raises(T3eInputError, match="JSON object"):
        parse_t3e([])


# --- report, CLI, isolation ---------------------------------------------------------------------------------


def test_report_and_summary_flag_unchecked_reference_and_fake_input(result) -> None:
    report = render_markdown(result)
    assert report.startswith("# T3f") and "T3e REFERENCE NOT CHECKED" in report and "FAKE EMBEDDER" in report
    for heading in ("## Selection headroom", "## Projection of the same selected passage", "## Evidence packets", "## Cases"):
        assert heading in report
    assert "| semantic | query-aware | " in report and "| lexical | leading (T3e) | " in report
    assert "| fusion (semantic primary, dual on disagreement) | 320 | " in report and "| semantic only (control) | 160 | " in report
    assert all(f"| {row['id']} | {row['category']} | " in report for row in result["cases"])
    summary = render_summary(result, paths={"json": "r.json"})
    assert "NOT CHECKED" in summary and "semantic passage line hit" in summary and "json:" in summary


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(EXPERIMENT_DIR / "run_localization_fusion.py"), *args],
                          capture_output=True, text=True, timeout=120)


def test_cli_writes_results_and_report_from_a_t3e_artifact(t3e_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "t3f"
    completed = cli("--t3e-results", str(t3e_file), "--output-dir", str(out), "--skip-t3e-reference")
    assert completed.returncode == 0, completed.stderr
    assert "T3e reference: NOT CHECKED" in completed.stdout
    written = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert written["input"]["t3e_sha256"] == hashlib.sha256(t3e_file.read_bytes()).hexdigest()
    assert len(written["cases"]) == 30 and (out / "report.md").read_text(encoding="utf-8").startswith("# T3f")


def test_cli_stops_on_reference_drift_and_malformed_input_without_writing(t3e_file: Path, tmp_path: Path) -> None:
    drifted = cli("--t3e-results", str(t3e_file), "--output-dir", str(tmp_path / "a"))
    assert drifted.returncode == 3 and "differ from the accepted live run" in drifted.stderr
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"experiment": "nope"}), encoding="utf-8")
    malformed = cli("--t3e-results", str(bad), "--output-dir", str(tmp_path / "b"))
    assert malformed.returncode == 2 and "cannot run T3f" in malformed.stderr
    same_dir = cli("--t3e-results", str(t3e_file), "--output-dir", str(t3e_file.parent), "--skip-t3e-reference")
    assert same_dir.returncode == 2
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_t3f_needs_no_model_and_only_zomah_s_lexical_matcher() -> None:
    probe = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(EXPERIMENT_DIR)!r}); "
         "import embedding_bench.localization_fusion, embedding_bench.localization_fusion_report, embedding_bench.projection; "
         "print(sorted(m for m in sys.modules if m.split('.')[0] in "
         "{'torch', 'sentence_transformers', 'transformers', 'zomah', 'pydantic', 'numpy', 'laya', 'typesafe_sdk'}))"],
        capture_output=True, text=True, timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "['zomah', 'zomah.access', 'zomah.knowledge']"

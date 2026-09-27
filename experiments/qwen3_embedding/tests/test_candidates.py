"""T3c: candidate union, deterministic resolution classes, and the audit report."""

from __future__ import annotations

import inspect
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from embedding_bench.candidates import (
    Candidate,
    audit_case,
    build_candidates,
    classify,
    dominates,
    pareto_front,
    run_audit,
    size_stats,
)
from embedding_bench.candidates_report import render_markdown, render_summary
from embedding_bench.fusion import check_labels, parse_t3a
from t3a_artifacts import t3a_artifact, write_manifest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

# (id, category, expected, lexical ranking, semantic ranking); outcomes worked by hand.
SCENARIOS = [
    ("agree-right", "exact", "a.md", ["a.md", "b.md", "c.md"], ["a.md", "c.md", "b.md", "d.md", "e.md"]),
    ("agree-wrong-in", "exact", "b.md", ["a.md", "b.md", "c.md"], ["a.md", "b.md", "c.md", "d.md", "e.md"]),
    ("agree-wrong-out", "exact", "e.md", ["a.md", "b.md", "c.md"], ["a.md", "c.md", "b.md", "d.md", "e.md"]),
    ("lex-only", "paraphrase", "a.md", ["a.md", "b.md", "c.md"], ["b.md", "a.md", "c.md", "d.md", "e.md"]),
    ("sem-only", "paraphrase", "d.md", ["a.md", "b.md", "c.md"], ["d.md", "a.md", "b.md", "c.md", "e.md"]),
    ("other", "paraphrase", "c.md", ["a.md", "c.md", "b.md"], ["b.md", "c.md", "a.md", "d.md", "e.md"]),
    ("fail", "paraphrase", "e.md", ["a.md", "b.md"], ["b.md", "a.md", "c.md", "d.md", "e.md"]),
]


def audit(tmp_path: Path, scenarios=SCENARIOS) -> dict:
    artifact = t3a_artifact(scenarios)
    parsed = parse_t3a(artifact)
    return run_audit(parsed, label_check=check_labels(parsed, write_manifest(tmp_path / "m.json", artifact)))


def cand(path: str, lexical: int | None, semantic: int | None) -> Candidate:
    return Candidate(path, lexical, semantic, lexical is not None and lexical <= 3, semantic is not None and semantic <= 3)


# --- candidate union ---------------------------------------------------------------------------


def test_union_of_top3_lists_is_deduplicated_and_keeps_full_ranks() -> None:
    candidates = build_candidates(["a.md", "b.md", "c.md", "d.md"], ["c.md", "a.md", "e.md", "b.md", "d.md"])
    assert [(c.path, c.lexical_rank, c.semantic_rank) for c in candidates] == [
        ("a.md", 1, 2),
        ("c.md", 3, 1),
        ("b.md", 2, 4),  # in lexical top 3 only; keeps its semantic rank 4
        ("e.md", None, 3),  # lexical never returned it
    ]
    by_path = {c.path: c for c in candidates}
    assert by_path["b.md"].in_lexical_top and not by_path["b.md"].in_semantic_top
    assert not by_path["e.md"].in_lexical_top and by_path["e.md"].in_semantic_top
    assert "d.md" not in by_path


def test_union_size_ranges_from_n_to_2n() -> None:
    assert len(build_candidates(["a", "b", "c"], ["c", "b", "a"])) == 3
    assert len(build_candidates(["a", "b", "c"], ["d", "e", "f"])) == 6
    assert len(build_candidates(["a"], ["a", "b", "c"], n=1)) == 1


def test_missing_lexical_and_semantic_ranks() -> None:
    no_lexical = build_candidates([], ["b.md", "a.md", "c.md"])
    assert all(c.lexical_rank is None and not c.in_lexical_top for c in no_lexical)
    partial_semantic = {c.path: c for c in build_candidates(["a.md", "b.md", "c.md"], ["b.md"])}
    assert partial_semantic["a.md"].semantic_rank is None
    assert partial_semantic["b.md"].semantic_rank == 1


def test_candidate_order_is_symmetric_and_deterministic() -> None:
    # (1, 3) and (3, 1) have the same best and worse rank: path decides.
    first = build_candidates(["z.md", "m.md", "a.md"], ["a.md", "m.md", "z.md"])
    second = build_candidates(["a.md", "m.md", "z.md"], ["z.md", "m.md", "a.md"])
    assert [c.path for c in first] == [c.path for c in second] == ["a.md", "z.md", "m.md"]


def test_rankings_listing_a_document_twice_are_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_candidates(["a.md", "a.md"], ["a.md"])
    with pytest.raises(ValueError):
        build_candidates(["a.md"], ["a.md"], n=0)


# --- resolution classes ----------------------------------------------------------------------------


def test_agreement() -> None:
    resolution = classify(build_candidates(["a.md", "b.md", "c.md"], ["a.md", "c.md", "b.md"]))
    assert (resolution.kind, resolution.path) == ("agreement", "a.md")


def test_dominance_rule_selects_a_unique_dominator() -> None:
    assert dominates(cand("a", 2, 2), cand("b", 3, 2))
    assert dominates(cand("a", 2, None), cand("b", 3, None))
    assert dominates(cand("a", 5, 1), cand("b", None, 1))  # absent is worse than any rank
    assert not dominates(cand("a", 1, 3), cand("b", 3, 1))
    resolution = classify([cand("x", 2, 2), cand("y", 3, 4), cand("z", None, 3)])
    assert (resolution.kind, resolution.path) == ("dominance", "x")


def test_dominance_resolves_when_lexical_returns_nothing() -> None:
    resolution = classify(build_candidates([], ["b.md", "a.md", "c.md"]))
    assert (resolution.kind, resolution.path) == ("dominance", "b.md")


def test_ambiguous_when_the_two_first_places_differ() -> None:
    candidates = build_candidates(["a.md", "b.md", "c.md"], ["b.md", "a.md", "c.md"])
    assert classify(candidates).kind == "ambiguous"
    assert [c.path for c in pareto_front(candidates)] == ["a.md", "b.md"]  # c.md (3,3) is dominated


def test_identical_rank_pairs_never_dominate_each_other() -> None:
    twins = [cand("x", 2, 2), cand("y", 2, 2)]
    assert not dominates(twins[0], twins[1]) and not dominates(twins[1], twins[0])
    assert classify(twins).kind == "ambiguous"
    assert classify([]).kind == "ambiguous"


def test_unique_dominance_over_this_union_only_ever_means_agreement() -> None:
    docs = ["a.md", "b.md", "c.md", "d.md"]
    for lexical in itertools.permutations(docs):
        for semantic in itertools.permutations(docs):
            for cut in (1, 2, 4):  # lexical may omit documents with no matching term
                resolution = classify(build_candidates(lexical[:cut], semantic))
                assert resolution.kind != "dominance"
                assert (resolution.kind == "agreement") == (lexical[0] == semantic[0])


# --- labels are for evaluation only ---------------------------------------------------------------


def test_construction_and_classification_take_no_labels() -> None:
    assert list(inspect.signature(build_candidates).parameters) == ["lexical", "semantic", "n"]
    assert list(inspect.signature(classify).parameters) == ["candidates"]


def test_relabelling_changes_evaluation_but_never_candidates_or_resolution(tmp_path: Path) -> None:
    relabelled = [(cid, "exact", "e.md", lex, sem) for cid, _, _, lex, sem in SCENARIOS]
    original, changed = audit(tmp_path, SCENARIOS), audit(tmp_path, relabelled)
    for a, b in zip(original["cases"], changed["cases"], strict=True):
        assert (a["candidates"], a["resolution"], a["resolved_path"]) == (
            b["candidates"],
            b["resolution"],
            b["resolved_path"],
        )
    assert original["outcomes"] != changed["outcomes"]


def test_expected_labels_pass_through_unchanged(tmp_path: Path) -> None:
    result = audit(tmp_path)
    assert [(c["id"], c["category"], c["expected_path"]) for c in result["cases"]] == [
        (cid, category, expected) for cid, category, expected, _, _ in SCENARIOS
    ]


# --- recall, sizes, outcomes -----------------------------------------------------------------------------


def test_candidate_recall_overall_and_by_category(tmp_path: Path) -> None:
    recall = audit(tmp_path)["candidate_recall"]
    overall = recall["overall"]
    assert (overall["lexical_top"]["count"], overall["semantic_top"]["count"], overall["union"]["count"]) == (4, 5, 5)
    assert overall["union"]["rate"] == pytest.approx(5 / 7)
    exact, paraphrase = recall["by_category"]["exact"], recall["by_category"]["paraphrase"]
    assert (exact["n"], exact["lexical_top"]["count"], exact["semantic_top"]["count"], exact["union"]["count"]) == (3, 2, 2, 2)
    assert (paraphrase["lexical_top"]["count"], paraphrase["semantic_top"]["count"], paraphrase["union"]["count"]) == (2, 3, 3)


def test_candidate_size_statistics(tmp_path: Path) -> None:
    assert size_stats([3, 4, 4, 6]) == {
        "n": 4,
        "mean": 4.25,
        "median": 4.0,
        "min": 3,
        "max": 6,
        "histogram": {"3": 1, "4": 2, "6": 1},
    }
    assert size_stats([])["mean"] is None
    size = audit(tmp_path)["candidate_size"]
    assert (size["mean"], size["median"], size["max"], size["histogram"]) == (pytest.approx(22 / 7), 3, 4, {"3": 6, "4": 1})


def test_resolution_counts_accuracy_and_outcomes(tmp_path: Path) -> None:
    result = audit(tmp_path)
    counts = result["resolution"]["counts"]
    assert counts == {
        "agreement": ["agree-right", "agree-wrong-in", "agree-wrong-out"],
        "dominance": [],
        "ambiguous": ["lex-only", "sem-only", "other", "fail"],
    }
    assert result["resolution"]["resolved_accuracy"]["correct"] == 1
    assert result["resolution"]["resolved_accuracy"]["rate"] == pytest.approx(1 / 3)
    assert result["outcomes"] == {
        "resolved_correct": ["agree-right"],
        "resolved_wrong_in_union": ["agree-wrong-in"],
        "resolved_retrieval_failure": ["agree-wrong-out"],
        "ambiguous_in_union": ["lex-only", "sem-only", "other"],
        "ambiguous_retrieval_failure": ["fail"],
    }


def test_retrieval_failures_are_kept_separate_from_judgment_scope(tmp_path: Path) -> None:
    result = audit(tmp_path)
    assert result["retrieval_failures"] == ["agree-wrong-out", "fail"]
    ambiguous = result["ambiguous"]
    assert ambiguous["expected_in_union"] == 3  # the only cases a judge could fix
    assert ambiguous["expected_is"] == {
        "lexical_top1": ["lex-only"],
        "semantic_top1": ["sem-only"],
        "other_candidate": ["other"],
        "not_in_union": ["fail"],
    }
    refs = result["end_to_end_references"]
    assert refs["deterministic_plus_perfect_judge"] == 4
    assert (refs["deterministic_plus_lexical_top1"], refs["deterministic_plus_semantic_top1"], refs["rrf_top1"]) == (2, 2, 2)


def test_ambiguous_diagnostics_pareto_front_and_selectors(tmp_path: Path) -> None:
    ambiguous = audit(tmp_path)["ambiguous"]
    assert ambiguous["pareto_front_size"]["histogram"] == {"2": 3, "3": 1}
    assert ambiguous["expected_on_pareto_front"] == 3
    selectors = ambiguous["selector_diagnostics"]
    assert [selectors[key]["correct"] for key in ("lexical_top1", "semantic_top1", "rrf_top1")] == [1, 1, 1]


def test_t3a_buckets_inside_the_union(tmp_path: Path) -> None:
    wins = audit(tmp_path)["t3a_wins_in_union"]
    assert (wins["lexical_correct_semantic_wrong"]["n"], wins["lexical_correct_semantic_wrong"]["in_union"]) == (1, 1)
    assert (wins["semantic_correct_lexical_wrong"]["n"], wins["semantic_correct_lexical_wrong"]["in_union"]) == (1, 1)
    assert (wins["both_wrong"]["n"], wins["both_wrong"]["in_union"]) == (4, 2)
    assert wins["both_correct"]["resolution"]["agreement"] == ["agree-right"]


def test_audit_case_marks_candidates_for_inspection() -> None:
    artifact = parse_t3a(t3a_artifact(SCENARIOS))
    row = audit_case(next(c for c in artifact.cases if c.id == "sem-only"))
    assert row["candidate_count"] == 4
    assert {c["path"]: c["on_pareto_front"] for c in row["candidates"]} == {
        "a.md": True,
        "d.md": True,
        "b.md": False,
        "c.md": False,
    }
    assert row["evaluation"]["expected_ranks"] == {"lexical": None, "semantic": 1}


# --- reports, CLI, and dependencies ---------------------------------------------------------------------------


def test_reports_list_every_ambiguous_case_and_retrieval_failure(tmp_path: Path) -> None:
    result = audit(tmp_path)
    markdown = render_markdown(result)
    for case_id in ("lex-only", "sem-only", "other", "fail"):
        assert f"### {case_id} (paraphrase, ambiguous)" in markdown
    assert "### agree-wrong-out (exact, agreement → `a.md`)" in markdown
    assert "## Deterministic resolutions that picked the wrong candidate" in markdown
    assert "- `d.md` — lexical —, semantic 1 (expected, front)" in markdown
    assert "| overall | 7 | 4/7 (57.1%) | 5/7 (71.4%) | 5/7 (71.4%) |" in markdown
    assert "Jev and Laya would be alternative implementations of that single role" in markdown
    for case_id, *_ in SCENARIOS:
        assert f"| {case_id} |" in markdown
    summary = render_summary(result)
    assert "agreement 3 (correct 1), dominance 0 (correct 0), ambiguous 4 (expected is a candidate in 3)" in summary
    assert "retrieval failures: agree-wrong-out, fail" in summary


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    # -I -S: no site-packages, so torch, sentence-transformers, numpy, pydantic and ZOMAH are unimportable.
    return subprocess.run(
        [sys.executable, "-I", "-S", str(EXPERIMENT_DIR / "run_candidate_audit.py"), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_runs_on_the_standard_library_alone(tmp_path: Path) -> None:
    artifact = t3a_artifact(SCENARIOS)
    source = tmp_path / "t3a" / "results.json"
    source.parent.mkdir()
    source.write_text(json.dumps(artifact), encoding="utf-8")
    manifest = write_manifest(tmp_path / "m.json", artifact)
    out = tmp_path / "audit"
    completed = run_cli("--input", str(source), "--output-dir", str(out), "--manifest", str(manifest))
    assert completed.returncode == 0, completed.stderr
    assert "T3c candidate audit" in completed.stdout
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["ambiguous"]["n"] == 4
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# T3c")
    assert json.loads(source.read_text(encoding="utf-8")) == artifact


def test_cli_refuses_the_t3a_directory_and_reports_missing_rankings(tmp_path: Path) -> None:
    artifact = t3a_artifact(SCENARIOS)
    source = tmp_path / "results.json"
    source.write_text(json.dumps(artifact), encoding="utf-8")
    refused = run_cli("--input", str(source), "--output-dir", str(tmp_path))
    assert refused.returncode == 2 and "overwrite" in refused.stderr

    del artifact["cases"][0]["lexical"]["ranking"]
    source.write_text(json.dumps(artifact), encoding="utf-8")
    missing = run_cli("--input", str(source), "--output-dir", str(tmp_path / "out"))
    assert missing.returncode == 2
    assert "lexical ranking is missing" in missing.stderr
    assert not (tmp_path / "out").exists()


def test_real_t3a_runner_output_satisfies_the_structural_invariants(tmp_path: Path) -> None:
    pytest.importorskip("zomah")
    from embedding_bench.retrieval import HashingEmbedder
    from embedding_bench.runner import run_benchmark

    work = tmp_path / "work"
    work.mkdir()
    t3a = json.loads(
        json.dumps(
            run_benchmark(
                root=EXPERIMENT_DIR.parents[1],
                manifest_path=EXPERIMENT_DIR / "benchmark.json",
                embedder_factory=HashingEmbedder,
                workdir=work,
            )
        )
    )
    artifact = parse_t3a(t3a)
    result = run_audit(artifact, label_check=check_labels(artifact, EXPERIMENT_DIR / "benchmark.json"))
    rows = {row["id"]: row for row in result["cases"]}
    for case_id in t3a["agreement"]["both_correct"]:
        assert rows[case_id]["evaluation"]["outcome"] == "resolved_correct"
    for bucket in ("lexical_correct_semantic_wrong", "semantic_correct_lexical_wrong"):
        for case_id in t3a["agreement"][bucket]:
            assert rows[case_id]["resolution"] == "ambiguous"
            assert rows[case_id]["evaluation"]["expected_in_union"]
    overall = result["candidate_recall"]["overall"]
    assert overall["union"]["count"] >= max(overall["lexical_top"]["count"], overall["semantic_top"]["count"])
    assert result["resolution"]["counts"]["dominance"] == []

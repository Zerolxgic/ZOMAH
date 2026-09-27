"""T3b: RRF over frozen T3a rankings, tested with standard-library inputs only."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from embedding_bench.fusion import (
    RRF_K,
    FusionInputError,
    check_labels,
    compare_to_inputs,
    load_t3a,
    parse_t3a,
    rrf_contribution,
    rrf_fuse,
    run_fusion,
)
from embedding_bench.fusion_report import render_markdown, render_summary
from t3a_artifacts import write_manifest
from t3a_artifacts import t3a_artifact as build_t3a_artifact

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

# (id, category, expected, lexical ranking, semantic ranking); outcomes worked by hand at k=60.
SCENARIOS = [
    ("both", "exact", "a.md", ["a.md", "b.md", "c.md"], ["a.md", "c.md", "b.md", "d.md", "e.md"]),
    ("lex-kept", "exact", "a.md", ["a.md", "c.md", "b.md"], ["b.md", "a.md", "c.md", "d.md", "e.md"]),
    ("lex-lost", "exact", "a.md", ["a.md", "b.md"], ["b.md", "c.md", "d.md", "e.md", "a.md"]),
    ("sem-kept", "exact", "a.md", ["b.md", "a.md"], ["a.md", "c.md", "d.md", "e.md", "b.md"]),
    ("sem-lost", "paraphrase", "a.md", ["b.md", "c.md"], ["a.md", "b.md", "c.md", "d.md", "e.md"]),
    ("recovered", "paraphrase", "a.md", ["b.md", "a.md"], ["c.md", "a.md", "d.md", "e.md", "b.md"]),
    ("worse", "paraphrase", "a.md", ["b.md", "a.md", "c.md"], ["c.md", "a.md", "b.md", "d.md", "e.md"]),
    ("tie", "paraphrase", "c.md", ["c.md", "b.md"], ["b.md", "c.md", "a.md", "d.md", "e.md"]),
]


def t3a_artifact(scenarios=SCENARIOS, *, drift=()) -> dict:
    return build_t3a_artifact(scenarios, drift=drift)


def fused_result(tmp_path: Path, artifact: dict | None = None) -> dict:
    artifact = artifact or t3a_artifact()
    parsed = parse_t3a(artifact)
    return run_fusion(parsed, label_check=check_labels(parsed, write_manifest(tmp_path / "m.json", artifact)))


# --- RRF scoring ----------------------------------------------------------------------


def test_rrf_contribution_is_one_over_k_plus_rank() -> None:
    assert RRF_K == 60
    assert rrf_contribution(1) == Fraction(1, 61)
    assert rrf_contribution(13) == Fraction(1, 73)
    assert rrf_contribution(None) == 0
    assert rrf_contribution(2, k=0) == Fraction(1, 2)
    with pytest.raises(ValueError):
        rrf_contribution(0)
    with pytest.raises(ValueError):
        rrf_contribution(1, k=-1)


def test_rrf_sums_both_sides_exactly() -> None:
    fused = {doc.path: doc for doc in rrf_fuse(["a.md", "b.md", "c.md"], ["c.md", "a.md", "b.md"])}
    assert fused["a.md"].score == Fraction(1, 61) + Fraction(1, 62)
    assert fused["c.md"].score == Fraction(1, 63) + Fraction(1, 61)
    assert (fused["b.md"].lexical_rank, fused["b.md"].semantic_rank) == (2, 3)


def test_missing_lexical_rank_contributes_zero() -> None:
    fused = {doc.path: doc for doc in rrf_fuse(["b.md"], ["a.md", "b.md"])}
    assert fused["a.md"].score == Fraction(1, 61)
    assert fused["a.md"].lexical_rank is None and fused["a.md"].semantic_rank == 1


def test_missing_semantic_rank_contributes_zero() -> None:
    fused = {doc.path: doc for doc in rrf_fuse(["a.md", "b.md"], ["b.md"])}
    assert fused["a.md"].score == Fraction(1, 61)
    assert fused["a.md"].semantic_rank is None


def test_equal_rank_pairs_tie_exactly_and_unequal_pairs_do_not() -> None:
    fused = rrf_fuse(["x.md", "m.md", "y.md"], ["y.md", "m.md", "x.md"])
    scores = {doc.path: doc.score for doc in fused}
    assert scores["x.md"] == scores["y.md"]  # (1,3) and (3,1): exact tie
    assert scores["m.md"] < scores["x.md"]  # (2,2) loses to (1,3): no float noise decides this
    assert [doc.path for doc in fused] == ["x.md", "y.md", "m.md"]


def test_ties_break_by_path_whatever_the_input_order() -> None:
    first = rrf_fuse(["z.md", "a.md"], ["a.md", "z.md"])
    second = rrf_fuse(["a.md", "z.md"], ["z.md", "a.md"])
    assert [doc.path for doc in first] == [doc.path for doc in second] == ["a.md", "z.md"]


def test_full_ranking_covers_the_union_in_score_order() -> None:
    fused = rrf_fuse(["c.md", "a.md"], ["a.md", "b.md", "c.md", "d.md"])
    assert [doc.path for doc in fused] == ["a.md", "c.md", "b.md", "d.md"]
    assert all(fused[i].score >= fused[i + 1].score for i in range(len(fused) - 1))
    assert rrf_fuse([], []) == ()


def test_rankings_listing_a_document_twice_are_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        rrf_fuse(["a.md", "a.md"], ["a.md"])


def test_compare_to_inputs() -> None:
    assert compare_to_inputs(1, 2, 2) == "better_than_both"
    assert compare_to_inputs(1, 1, 5) == "matches_best"
    assert compare_to_inputs(3, 1, 5) == "between"
    assert compare_to_inputs(3, 1, None) == "between"
    assert compare_to_inputs(3, 2, 2) == "worse_than_both"


# --- metrics, categories, preservation ------------------------------------------------------


def test_metrics_for_all_three_systems_and_categories(tmp_path: Path) -> None:
    result = fused_result(tmp_path)
    rrf = result["metrics"]["rrf"]
    assert rrf["overall"]["n"] == 8
    assert rrf["overall"]["top1"] == pytest.approx(4 / 8)
    assert rrf["overall"]["top3"] == pytest.approx(1.0)
    assert rrf["overall"]["mrr"] == pytest.approx((1 + 1 + 1 / 2 + 1 + 1 / 3 + 1 + 1 / 3 + 1 / 2) / 8)
    assert rrf["by_category"]["exact"]["top1"] == pytest.approx(3 / 4)
    assert rrf["by_category"]["exact"]["mrr"] == pytest.approx(3.5 / 4)
    assert rrf["by_category"]["paraphrase"]["top1"] == pytest.approx(1 / 4)
    assert rrf["by_category"]["paraphrase"]["mrr"] == pytest.approx((1 / 3 + 1 + 1 / 3 + 1 / 2) / 4)
    # The input systems' metrics are exactly the stored T3a metrics.
    artifact = t3a_artifact()
    for side in ("lexical", "semantic"):
        assert result["metrics"][side] == artifact["metrics"][side]
    assert [case["ranks"]["rrf"] for case in result["cases"]] == [1, 1, 2, 1, 3, 1, 3, 2]


def test_preservation_rescue_and_rank_comparisons(tmp_path: Path) -> None:
    result = fused_result(tmp_path)
    transitions = result["top1_transitions"]
    assert transitions["both_correct"] == {"n": 1, "rrf_correct": ["both"], "rrf_wrong": []}
    assert transitions["lexical_correct_semantic_wrong"] == {
        "n": 3,
        "rrf_correct": ["lex-kept"],
        "rrf_wrong": ["lex-lost", "tie"],
    }
    assert transitions["semantic_correct_lexical_wrong"] == {
        "n": 2,
        "rrf_correct": ["sem-kept"],
        "rrf_wrong": ["sem-lost"],
    }
    assert transitions["both_wrong"] == {"n": 2, "rrf_correct": ["recovered"], "rrf_wrong": ["worse"]}
    assert result["oracle_top1_union"]["correct"] == 6
    assert result["rank_vs_inputs"]["better_than_both"] == ["recovered"]
    assert result["rank_vs_inputs"]["worse_than_both"] == ["worse"]
    assert result["top3_lost"] == {"vs_lexical": [], "vs_semantic": []}
    assert result["ties"]["rrf_top1_decided_by_path"] == ["worse", "tie"]
    assert result["ties"]["expected_rank_decided_by_path"] == ["tie"]


def test_top3_losses_are_reported(tmp_path: Path) -> None:
    # Semantic has a.md third; lexical never matches it, so every lexically matched document outranks it.
    scenario = ("lost", "exact", "a.md", ["b.md", "c.md", "d.md"], ["b.md", "c.md", "a.md", "d.md", "e.md"])
    result = fused_result(tmp_path, t3a_artifact([scenario]))
    assert result["cases"][0]["ranks"] == {"lexical": None, "semantic": 3, "rrf": 4}
    assert result["top3_lost"]["vs_semantic"] == ["lost"]


# --- labels -----------------------------------------------------------------------------------


def test_expected_labels_pass_through_unchanged(tmp_path: Path) -> None:
    artifact = t3a_artifact()
    result = fused_result(tmp_path, artifact)
    fields = ("id", "category", "query", "expected_path")
    assert [tuple(c[f] for f in fields) for c in result["cases"]] == [
        tuple(c[f] for f in fields) for c in artifact["cases"]
    ]
    assert result["manifest_check"]["labels_match"] is True


def test_labels_and_categories_never_influence_the_fused_ranking(tmp_path: Path) -> None:
    relabelled = [(cid, "exact", "e.md", lex, sem) for cid, _, _, lex, sem in SCENARIOS]
    original = fused_result(tmp_path, t3a_artifact())
    changed = fused_result(tmp_path, t3a_artifact(relabelled))
    assert [c["rrf_ranking"] for c in original["cases"]] == [c["rrf_ranking"] for c in changed["cases"]]


def test_fusion_ignores_raw_scores(tmp_path: Path) -> None:
    artifact = t3a_artifact()
    scrambled = copy.deepcopy(artifact)
    for case in scrambled["cases"]:
        for side in ("lexical", "semantic"):
            for entry in case[side]["ranking"]:
                entry["score"] = -12345.0
    assert fused_result(tmp_path, artifact)["cases"] == fused_result(tmp_path, scrambled)["cases"]


def test_label_mismatch_with_the_manifest_is_rejected(tmp_path: Path) -> None:
    artifact = t3a_artifact()
    manifest = write_manifest(tmp_path / "m.json", artifact)
    data = json.loads(manifest.read_text())
    data["cases"][0]["expected_path"] = "b.md"
    manifest.write_text(json.dumps(data))
    with pytest.raises(FusionInputError, match="expected_path differ"):
        check_labels(parse_t3a(artifact), manifest)


def test_manifest_cases_missing_from_t3a_must_be_recorded_drift(tmp_path: Path) -> None:
    extra = {"id": "gone", "category": "exact", "query": "q", "expected_path": "z.md", "evidence_anchor": "x"}
    artifact = t3a_artifact()
    manifest = write_manifest(tmp_path / "m.json", artifact, extra_cases=[extra])
    with pytest.raises(FusionInputError, match="without being recorded as drift"):
        check_labels(parse_t3a(artifact), manifest)
    drifted = t3a_artifact(drift=["gone"])
    check = check_labels(parse_t3a(drifted), manifest)
    assert check["not_scored_in_t3a"] == ["gone"]
    assert check["sha256_matches"] is False


# --- malformed or incompatible input ------------------------------------------------------------


def _drop_semantic_ranking(d):
    del d["cases"][2]["semantic"]["ranking"]


def _drop_lexical_side(d):
    del d["cases"][0]["lexical"]


def _duplicate_lexical_path(d):
    d["cases"][0]["lexical"]["ranking"].append({"path": "b.md", "score": 0.0})


def _unknown_path(d):
    d["cases"][0]["semantic"]["ranking"][4]["path"] = "zzz.md"


def _incomplete_semantic(d):
    d["cases"][0]["semantic"]["ranking"].pop()


def _wrong_stored_rank(d):
    d["cases"][0]["semantic"]["rank"] = 2


def _wrong_metrics(d):
    d["metrics"]["semantic"]["overall"]["mrr"] += 0.01


def _wrong_category_metrics(d):
    d["metrics"]["lexical"]["by_category"]["exact"]["top1"] = 0.0


def _wrong_agreement(d):
    d["agreement"]["both_correct"] = []


def _unknown_category(d):
    d["cases"][0]["category"] = "vibes"


def _duplicate_case(d):
    d["cases"].append(copy.deepcopy(d["cases"][0]))


def _no_cases(d):
    d["cases"] = []


def _no_corpus(d):
    del d["corpus"]


def _expected_outside_corpus(d):
    d["cases"][0]["expected_path"] = "zzz.md"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_semantic_ranking, "semantic ranking is missing"),
        (_drop_lexical_side, "lexical ranking is missing"),
        (_duplicate_lexical_path, "more than once"),
        (_unknown_path, "outside the corpus"),
        (_incomplete_semantic, "does not cover the whole corpus"),
        (_wrong_stored_rank, "disagrees with its ranking"),
        (_wrong_metrics, "semantic metrics disagree"),
        (_wrong_category_metrics, "lexical category metrics disagree"),
        (_wrong_agreement, "agreement bucket 'both_correct'"),
        (_unknown_category, "unknown category"),
        (_duplicate_case, "appears more than once"),
        (_no_cases, "no scored cases"),
        (_no_corpus, "corpus"),
        (_expected_outside_corpus, "not in corpus.files"),
    ],
)
def test_malformed_or_inconsistent_t3a_input_is_rejected(mutate, message: str) -> None:
    artifact = t3a_artifact()
    mutate(artifact)
    with pytest.raises(FusionInputError, match=message):
        parse_t3a(artifact)


def test_unreadable_or_non_json_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FusionInputError, match="cannot read"):
        load_t3a(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(FusionInputError, match="not valid JSON"):
        load_t3a(bad)
    with pytest.raises(FusionInputError, match="JSON object"):
        parse_t3a([])


# --- reports and CLI ---------------------------------------------------------------------------


def test_reports_show_three_systems_and_every_query(tmp_path: Path) -> None:
    result = fused_result(tmp_path)
    markdown = render_markdown(result)
    for case_id, *_ in SCENARIOS:
        assert f"| {case_id} |" in markdown
    assert "| overall | RRF | 8 | 50.0% | 100.0% |" in markdown
    assert "| paraphrase | lexical |" in markdown and "| exact | semantic |" in markdown
    assert "**Lexical-only top-1 wins:** kept 1/3; lost: lex-lost, tie" in markdown
    assert "**Both wrong in T3a:** recovered 1/2 (recovered); not recovered: worse" in markdown
    assert "### tie (paraphrase, T3a lexical_correct_semantic_wrong)" in markdown
    assert "RRF top-1 is an exact score tie broken by path." in markdown
    summary = render_summary(result)
    assert "lexical top1/top3/MRR" in summary and "RRF top1/top3/MRR" in summary
    assert "lexical-only kept 1/3" in summary and "both-wrong recovered 1/2" in summary


def run_cli(*args: str, isolated: bool = False) -> subprocess.CompletedProcess[str]:
    python = [sys.executable, "-I", "-S"] if isolated else [sys.executable]
    return subprocess.run(
        [*python, str(EXPERIMENT_DIR / "run_fusion.py"), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_runs_on_the_standard_library_alone(tmp_path: Path) -> None:
    artifact = t3a_artifact()
    source = tmp_path / "t3a" / "results.json"
    source.parent.mkdir()
    source.write_text(json.dumps(artifact), encoding="utf-8")
    manifest = write_manifest(tmp_path / "m.json", artifact)
    out = tmp_path / "fusion"
    # -I -S: no site-packages at all, so torch, sentence-transformers, numpy,
    # pydantic and ZOMAH itself are unimportable. The run must still succeed.
    completed = run_cli("--input", str(source), "--output-dir", str(out), "--manifest", str(manifest), isolated=True)
    assert completed.returncode == 0, completed.stderr
    assert "RRF top1/top3/MRR" in completed.stdout
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["metrics"]["rrf"]["overall"]["n"] == 8
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# T3b")
    assert json.loads(source.read_text(encoding="utf-8")) == artifact  # input untouched


def test_cli_refuses_to_write_into_the_t3a_run_directory(tmp_path: Path) -> None:
    source = tmp_path / "results.json"
    source.write_text(json.dumps(t3a_artifact()), encoding="utf-8")
    completed = run_cli("--input", str(source), "--output-dir", str(tmp_path))
    assert completed.returncode == 2
    assert "overwrite" in completed.stderr


def test_cli_reports_missing_ranking_data_instead_of_regenerating(tmp_path: Path) -> None:
    artifact = t3a_artifact()
    _drop_semantic_ranking(artifact)
    source = tmp_path / "results.json"
    source.write_text(json.dumps(artifact), encoding="utf-8")
    completed = run_cli("--input", str(source), "--output-dir", str(tmp_path / "out"), isolated=True)
    assert completed.returncode == 2
    assert "semantic ranking is missing" in completed.stderr
    assert "does not regenerate" in completed.stderr
    assert not (tmp_path / "out").exists()


# --- compatibility with real T3a output ----------------------------------------------------------


def test_real_t3a_runner_output_is_accepted_and_reproduced(tmp_path: Path) -> None:
    pytest.importorskip("zomah")
    from embedding_bench.retrieval import HashingEmbedder
    from embedding_bench.runner import run_benchmark

    work = tmp_path / "work"
    work.mkdir()
    t3a = run_benchmark(
        root=EXPERIMENT_DIR.parents[1],
        manifest_path=EXPERIMENT_DIR / "benchmark.json",
        embedder_factory=HashingEmbedder,
        workdir=work,
    )
    t3a = json.loads(json.dumps(t3a))  # exactly what results.json holds
    artifact = parse_t3a(t3a)
    result = run_fusion(artifact, label_check=check_labels(artifact, EXPERIMENT_DIR / "benchmark.json"))
    for side in ("lexical", "semantic"):
        assert result["metrics"][side] == t3a["metrics"][side]
    assert result["input"]["fake_embedder"] is True
    assert len(result["cases"]) == len(t3a["cases"])

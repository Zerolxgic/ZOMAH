"""T3d: frozen judge cases, adapters, and the Jev-vs-Laya comparison, with fake judges only."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from embedding_bench.judge_adapters import JevJudge, LayaJudge
from embedding_bench.judge_compare import compare, load_judgments
from embedding_bench.judge_contract import (
    CASES_SCHEMA,
    NONE_LABEL,
    QUESTION_ID,
    JudgeContractError,
    judgments_document,
    load_cases,
    normalize_answer,
    render_request,
    run_judge,
    sha256_json,
    validate_judge_input,
)
from embedding_bench.judge_report import render_markdown, render_summary

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PATHS = ("a.md", "b.md", "c.md")


# --- synthetic frozen cases ---------------------------------------------------------------------


def judge_input(query: str = "Where is the budget?") -> dict:
    return {
        "query": query,
        "candidates": [
            {
                "id": f"c{i}",
                "path": path,
                "lexical_rank": i if i < 3 else None,
                "semantic_rank": 4 - i,
                "start_line": 10 * i,
                "end_line": 10 * i + 1,
                "excerpt": f"excerpt from {path}",
            }
            for i, path in enumerate(PATHS, start=1)
        ],
    }


def make_case(case_id, case_set, category, expected, expected_is, baselines, *, shows=True, resolved=None) -> dict:
    ji = judge_input(f"query for {case_id}")
    request = render_request(ji)
    ids = {c["path"]: c["id"] for c in ji["candidates"]}
    return {
        "id": case_id,
        "set": case_set,
        "judge_input": ji,
        "judge_input_sha256": sha256_json(ji),
        "request": request,
        "request_sha256": sha256_json(request),
        "evaluation": {
            "expected_path": expected,
            "expected_candidate_id": ids.get(expected),
            "category": category,
            "t3a_agreement": "n/a",
            "t3c_resolution": "agreement" if case_set == "resolved" else "ambiguous",
            "t3c_resolved_path": resolved,
            "expected_is": expected_is,
            "baseline_paths": {
                "lexical_top1": baselines[0],
                "semantic_top1": baselines[1],
                "rrf_top1": baselines[2],
                "t3c_order_first": baselines[0],
                "first_option": "a.md",
            },
            "candidate_ids_by_path": ids,
            "anchor_lines": [1, 1],
            "expected_excerpt_lines": [1, 1],
            "expected_excerpt_shows_anchor": shows,
        },
    }


def cases_doc() -> dict:
    return {
        "schema": CASES_SCHEMA,
        "cases": [
            make_case("A1", "ambiguous", "exact", "a.md", "lexical_top1", ("a.md", "b.md", "a.md"), shows=True),
            make_case("A2", "ambiguous", "paraphrase", "b.md", "semantic_top1", ("a.md", "b.md", "a.md"), shows=False),
            make_case("A3", "ambiguous", "paraphrase", "c.md", "other_candidate", ("a.md", "b.md", "b.md"), shows=True),
            make_case("A4", "ambiguous", "conceptual", "a.md", "semantic_top1", ("b.md", "a.md", "b.md"), shows=False),
            make_case("R1", "resolved", "exact", "a.md", "both_top1", ("a.md", "a.md", "a.md"), resolved="a.md"),
            make_case("R2", "resolved", "exact", "b.md", "both_top1", ("b.md", "b.md", "b.md"), resolved="b.md"),
            make_case("R3", "resolved", "distractor", "c.md", "other_candidate", ("a.md", "a.md", "a.md"), resolved="a.md"),
        ],
    }


def answer(choice: str, probability: float = 0.6) -> dict:
    return {"type": "choice", "choice": choice, "probabilities": {choice: probability}, "confidence": 0.5}


JEV_CHOICES = {"A1": "c1", "A2": "c2", "A3": NONE_LABEL, "A4": "c9", "R1": "c1", "R2": "c3", "R3": "c3"}
LAYA_CHOICES = {"A1": "c2", "A2": "c2", "A3": "c3", "A4": "c1", "R1": "c1", "R2": "c2", "R3": "c1"}


def judgments(doc: dict, name: str, choices: dict, latencies: dict, *, cases_sha256: str, diagnostics=None) -> dict:
    records = []
    for case in doc["cases"]:
        a = answer(choices[case["id"]])
        records.append(
            {
                "id": case["id"],
                "request_sha256": case["request_sha256"],
                "latency_ms": latencies.get(case["id"], 1.0),
                "error": None,
                "answer": a,
                "normalized": normalize_answer(a, [c["id"] for c in case["judge_input"]["candidates"]]).to_dict(),
                "diagnostics": (diagnostics or {}).get(case["id"]),
                "raw": {"answers": {QUESTION_ID: a}},
            }
        )
    return {
        **judgments_document(
            judge=name,
            cases_path="cases.json",
            cases_sha256=cases_sha256,
            config={"fake": True},
            environment={"python": "test"},
            run={"warmup_ms": 1.0, "latency_ms": {}, "records": records},
            created_at="2026-09-27T00:00:00+00:00",
        ),
        "by_id": {r["id"]: r for r in records},
    }


def comparison() -> dict:
    doc = cases_doc()
    budget = {
        "A1": {
            "available": True,
            "options_truncated": ["c1"],
            "state_tokens_kept": 9,
            "state_tokens_full": 9,
        }
    }
    jev = judgments(doc, "jev", JEV_CHOICES, {"A1": 100, "A2": 200, "A3": 300, "A4": 400}, cases_sha256="x")
    laya = judgments(doc, "laya", LAYA_CHOICES, {"A1": 10, "A2": 20, "A3": 30, "A4": 40}, cases_sha256="x", diagnostics=budget)
    return compare(doc, "x", {"jev": jev, "laya": laya})


# --- contract: rendering and normalizing ----------------------------------------------------------


def test_request_is_one_choice_question_with_evidence_first_options() -> None:
    request = render_request(judge_input())
    question = request["questions"][QUESTION_ID]
    assert request["state"] == "Question: Where is the budget?"
    assert question["type"] == "choice" and set(question) == {"type", "instructions", "criteria"}
    assert list(question["criteria"]) == ["c1", "c2", "c3", NONE_LABEL]
    assert question["criteria"]["c3"] == "excerpt from c.md [c.md, lines 30-31; lexical rank absent; semantic rank 1]"


def test_judge_input_shape_is_enforced() -> None:
    validate_judge_input(judge_input())
    extra = judge_input()
    extra["category"] = "exact"
    with pytest.raises(JudgeContractError, match="exactly query and candidates"):
        validate_judge_input(extra)
    leaked = judge_input()
    leaked["candidates"][0]["expected"] = True
    with pytest.raises(JudgeContractError, match="exactly the fields"):
        validate_judge_input(leaked)
    clash = judge_input()
    clash["candidates"][0]["id"] = NONE_LABEL
    with pytest.raises(JudgeContractError, match="unique"):
        validate_judge_input(clash)


def test_normalize_selection_abstention_and_protocol_failures() -> None:
    ids = ["c1", "c2"]
    selected = normalize_answer({"choice": "c2", "probabilities": {"c2": 0.7, "c1": 0.3}, "confidence": 0.4}, ids)
    assert (selected.status, selected.selected_candidate, selected.selected_probability, selected.native_confidence) == (
        "selected",
        "c2",
        0.7,
        0.4,
    )
    abstained = normalize_answer({"choice": NONE_LABEL, "probabilities": {NONE_LABEL: 0.9}}, ids)
    assert abstained.abstained and abstained.selected_candidate is None and abstained.selected_probability == 0.9
    assert normalize_answer({"choice": "c7"}, ids).protocol_error == "choice 'c7' is not a supplied candidate id"
    assert normalize_answer({"choice": "c1"}, ids).selected_probability is None  # no fake confidence
    for malformed in (None, "c1", {"probabilities": {}}, {"choice": 3}, {"choice": "c1", "probabilities": ["x"]},
                      {"choice": "c1", "confidence": "high"}, {"choice": "c1", "probabilities": {"c1": True}}):
        assert normalize_answer(malformed, ids).status == "protocol_failure"
    failed = normalize_answer(None, ids, error="TimeoutError: slow")
    assert failed.status == "protocol_failure" and "TimeoutError" in failed.protocol_error


def test_load_cases_rejects_tampered_requests(tmp_path: Path) -> None:
    doc = cases_doc()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded, digest = load_cases(str(path))
    assert len(loaded["cases"]) == 7 and len(digest) == 64
    doc["cases"][0]["request"]["state"] = "Question: something else"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(JudgeContractError, match="does not match its judge input"):
        load_cases(str(path))
    doc = cases_doc()
    doc["cases"][0]["request_sha256"] = "0" * 64
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(JudgeContractError, match="request_sha256"):
        load_cases(str(path))


# --- adapters: identical packs, selection, failures ------------------------------------------------


class FakeJevClient:
    def __init__(self, choices: dict | None = None, fail: set | None = None) -> None:
        self.requests: list[tuple] = []
        self.choices = choices or {}
        self.fail = fail or set()

    def system_one(self, *, state, questions, model, timeout):
        self.requests.append((state, questions))
        if state in self.fail:
            raise TimeoutError("no response")
        choice = self.choices.get(state, "c1")

        class Response:
            model = "jev-fake"
            request_id = "req-1"

            def model_dump(self, mode):
                return {"model": "jev-fake", "usage": {"input_tokens": 1}, "answers": {QUESTION_ID: answer(choice)}}

        return Response()


class FakeLayaAgent:
    def __init__(self, choices: dict | None = None, answer_key: str = QUESTION_ID) -> None:
        self.requests: list[tuple] = []
        self.choices = choices or {}
        self.answer_key = answer_key

    def system_one(self, state, questions, *, max_len=None, head_max_len=None):
        self.requests.append((state, questions))
        return {"answers": {self.answer_key: answer(self.choices.get(state, "c2"))}, "usage": {"input_tokens": 9}}


def test_both_adapters_receive_the_identical_stored_request_and_nothing_else() -> None:
    doc = cases_doc()
    jev_client, laya_agent = FakeJevClient(), FakeLayaAgent()
    run_judge(doc, JevJudge(jev_client), warmup=False)
    run_judge(doc, LayaJudge(laya_agent), warmup=False)
    stored = [(c["request"]["state"], c["request"]["questions"]) for c in doc["cases"]]
    assert jev_client.requests == laya_agent.requests == stored
    sent = json.dumps(jev_client.requests)
    for case in doc["cases"]:
        assert case["id"] not in sent.replace(f"query for {case['id']}", "")
    for leaked in ("expected", "anchor", "category", "rationale", "gold", "t3a", "t3c", "baseline"):
        assert leaked not in sent


def test_run_judge_records_selection_abstention_invalid_and_errors() -> None:
    doc = cases_doc()
    states = {c["id"]: c["request"]["state"] for c in doc["cases"]}
    client = FakeJevClient(choices={states["A2"]: NONE_LABEL, states["A3"]: "c42"}, fail={states["A4"]})
    ticks = iter(range(0, 1000, 5))
    run = run_judge(doc, JevJudge(client, model="jev-latest"), warmup=True, clock=lambda: next(ticks) / 1000)
    records = {r["id"]: r for r in run["records"]}
    assert run["warmup_ms"] == pytest.approx(5.0)
    assert all(r["latency_ms"] == pytest.approx(5.0) for r in run["records"])
    assert records["A1"]["normalized"]["status"] == "selected"
    assert records["A2"]["normalized"]["abstained"] is True
    assert records["A3"]["normalized"]["protocol_error"] == "choice 'c42' is not a supplied candidate id"
    assert records["A4"]["normalized"]["status"] == "protocol_failure" and "TimeoutError" in records["A4"]["error"]
    assert records["A1"]["diagnostics"]["model"] == "jev-fake"
    assert len(client.requests) == len(doc["cases"]) + 1  # plus the one warm-up request


def test_malformed_provider_result_is_a_protocol_failure_not_a_crash() -> None:
    run = run_judge(cases_doc(), LayaJudge(FakeLayaAgent(answer_key="something_else")), warmup=False)
    assert {r["normalized"]["status"] for r in run["records"]} == {"protocol_failure"}
    assert run["records"][0]["diagnostics"]["available"] is False  # laya not installed here: no crash


# --- comparison --------------------------------------------------------------------------------------


def test_primary_metrics_use_only_ambiguous_cases() -> None:
    result = comparison()
    jev, laya = result["primary"]["jev"], result["primary"]["laya"]
    assert result["counts"] == {"ambiguous": 4, "resolved": 3}
    assert (jev["n"], jev["correct"], jev["selected"], jev["abstained"]) == (4, 2, 2, ["A3"])
    assert jev["protocol_failures"] == [{"id": "A4", "error": "choice 'c9' is not a supplied candidate id"}]
    assert jev["accuracy_when_selecting"] == 1.0
    assert (laya["correct"], laya["selected"], laya["accuracy_when_selecting"]) == (3, 4, 0.75)
    assert jev["latency_ms"] == {"mean": 250.0, "median": 250.0}
    assert laya["latency_ms"]["median"] == 25.0


def test_categories_roles_and_rrf_deltas() -> None:
    result = comparison()
    jev, laya = result["primary"]["jev"], result["primary"]["laya"]
    assert jev["by_category"] == {
        "exact": {"n": 1, "correct": 1},
        "paraphrase": {"n": 2, "correct": 1},
        "conceptual": {"n": 1, "correct": 0},
        "distractor": {"n": 0, "correct": 0},
    }
    assert {role: len(v["correct"]) for role, v in laya["by_expected_role"].items()} == {
        "lexical_top1": 0,
        "semantic_top1": 2,
        "other_candidate": 1,
    }
    assert jev["vs_rrf"] == {"improves": ["A2"], "worsens": []}
    assert laya["vs_rrf"] == {"improves": ["A2", "A3", "A4"], "worsens": ["A1"]}
    assert {name: b["correct"] for name, b in result["baselines"].items()} == {
        "lexical_top1": 1,
        "semantic_top1": 2,
        "rrf_top1": 1,
        "t3c_order_first": 1,
        "first_option": 2,
    }


def test_pairwise_comparison() -> None:
    pw = comparison()["pairwise"]
    assert pw["both_correct"] == ["A2"]
    assert pw["jev_only_correct"] == ["A1"]
    assert pw["laya_only_correct"] == ["A3", "A4"]
    assert pw["both_wrong"] == []
    assert pw["jev_abstains_laya_selects"] == ["A3"]
    assert pw["laya_abstains_jev_selects"] == [] and pw["both_abstain"] == []
    assert pw["same_selection"] == ["A2"]


def test_agreement_diagnostics() -> None:
    diag = comparison()["agreement_diagnostics"]
    assert diag["jev"]["agrees_with_retrievers"] == 1
    assert diag["jev"]["wrong_agreements"] == [{"id": "R3", "judge_correct": True, "judge_status": "selected"}]
    assert diag["jev"]["damaged"] == [{"id": "R2", "status": "selected", "selected_path": "c.md"}]
    assert diag["laya"]["agrees_with_retrievers"] == 3 and diag["laya"]["damaged"] == []
    assert diag["laya"]["wrong_agreements"][0]["judge_correct"] is False


def test_evidence_pack_limits_and_references() -> None:
    result = comparison()
    ev = result["evidence_pack"]
    assert ev["expected_excerpt_misses_anchor"] == ["A2", "A4"]
    assert ev["accuracy_when_anchor_shown"] == {"jev": 0.5, "laya": 0.5}
    assert ev["accuracy_when_anchor_missed"] == {"jev": 0.5, "laya": 1.0}
    assert ev["laya_token_budget"]["expected_option_truncated"] == ["A1"]
    e2e = result["end_to_end_reference"]
    assert (e2e["resolved_correct"], e2e["rrf_top1"]) == (2, 3)
    assert e2e["routing_plus_judge"] == {"jev": 4, "laya": 5}
    assert e2e["judge_everywhere"] == {"jev": 4, "laya": 5}


def test_report_and_summary() -> None:
    result = comparison()
    markdown = render_markdown(result)
    assert "| **jev** | 2/4 | 50.0% | 1 | 1 | 100.0% | 250.0 ms | 250.0 ms |" in markdown
    assert "| RRF #1 | 1/4 | 25.0% |" in markdown
    assert "- jev only correct: 1 — A1" in markdown
    assert "wrong agreement(s): R3: fixed (selected)" in markdown
    assert "bounded Jev or Laya relevance judgment" in markdown
    assert "### A1 (exact; lexical top1)" in markdown
    summary = render_summary(result)
    assert "jev      2/4 correct, 1 abstained, 1 protocol failures" in summary
    assert "pairwise: both 1, jev only 1, laya only 2, neither 0" in summary


def test_judgments_from_other_cases_or_requests_are_rejected(tmp_path: Path) -> None:
    doc = cases_doc()
    path = tmp_path / "judgments.json"
    record = judgments(doc, "jev", JEV_CHOICES, {}, cases_sha256="x")
    record.pop("by_id")
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_judgments(str(path), doc, "x")["judge"] == "jev"
    with pytest.raises(JudgeContractError, match="different cases file"):
        load_judgments(str(path), doc, "y")
    record["records"][0]["request_sha256"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(JudgeContractError, match="different request"):
        load_judgments(str(path), doc, "x")
    record["records"] = record["records"][1:]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(JudgeContractError, match="no judgment"):
        load_judgments(str(path), doc, "x")


# --- command line, standard library only ---------------------------------------------------------------


def test_compare_cli_and_judge_modules_run_on_the_standard_library_alone(tmp_path: Path) -> None:
    doc = cases_doc()
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(doc), encoding="utf-8")
    _, digest = load_cases(str(cases))
    for name, choices in (("jev", JEV_CHOICES), ("laya", LAYA_CHOICES)):
        record = judgments(doc, name, choices, {}, cases_sha256=digest)
        record.pop("by_id")
        (tmp_path / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    out = tmp_path / "comparison"
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(EXPERIMENT_DIR / "compare_judges.py"), "--cases", str(cases),
         "--jev", str(tmp_path / "jev.json"), "--laya", str(tmp_path / "laya.json"), "--output-dir", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "T3d bounded judgment" in completed.stdout
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["primary"]["laya"]["correct"] == 3
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-c",
         f"import sys; sys.path.insert(0, {str(EXPERIMENT_DIR)!r}); "
         "import embedding_bench.judge_adapters, embedding_bench.judge_compare, embedding_bench.judge_runner; "
         "print(sorted(m for m in sys.modules if m.split('.')[0] in "
         "{'torch', 'sentence_transformers', 'transformers', 'laya', 'typesafe_sdk', 'zomah', 'pydantic', 'numpy'}))"],
        capture_output=True, text=True, timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]"


def test_compare_cli_rejects_mismatched_judge_files(tmp_path: Path) -> None:
    doc = cases_doc()
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(doc), encoding="utf-8")
    _, digest = load_cases(str(cases))
    record = judgments(doc, "laya", LAYA_CHOICES, {}, cases_sha256=digest)
    record.pop("by_id")
    (tmp_path / "laya.json").write_text(json.dumps(record), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "compare_judges.py"), "--cases", str(cases),
         "--jev", str(tmp_path / "laya.json"), "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 2 and "not 'jev'" in completed.stderr


def test_runner_writes_judgments_and_refuses_to_overwrite(tmp_path: Path, capsys) -> None:
    from embedding_bench.judge_runner import execute

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(cases_doc()), encoding="utf-8")

    def run() -> int:
        return execute(
            judge_name="laya",
            cases_path=cases,
            output_dir=tmp_path / "laya",
            overwrite=False,
            warmup=False,
            make_judge=lambda: (LayaJudge(FakeLayaAgent()), {"model": "fake"}),
            environment=lambda: {"python": "test"},
        )

    assert run() == 0
    written = json.loads((tmp_path / "laya" / "judgments.json").read_text(encoding="utf-8"))
    assert written["judge"] == "laya" and len(written["records"]) == 7 and written["load_s"] >= 0
    assert "7 cases: 7 selected" in capsys.readouterr().out
    assert run() == 2


# --- preparing cases from a real T3a run (needs ZOMAH) --------------------------------------------------------


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    pytest.importorskip("zomah")
    from embedding_bench.fusion import check_labels, parse_t3a
    from embedding_bench.judge_prepare import prepare_document
    from embedding_bench.retrieval import HashingEmbedder
    from embedding_bench.runner import run_benchmark

    tmp = tmp_path_factory.mktemp("t3d")
    manifest = EXPERIMENT_DIR / "benchmark.json"
    t3a = json.loads(json.dumps(run_benchmark(
        root=EXPERIMENT_DIR.parents[1], manifest_path=manifest, embedder_factory=HashingEmbedder, workdir=tmp
    )))
    artifact = parse_t3a(t3a)
    (tmp / "idx").mkdir()
    doc = prepare_document(
        artifact,
        root=EXPERIMENT_DIR.parents[1],
        manifest_path=manifest,
        label_check=check_labels(artifact, manifest),
        workdir=tmp / "idx",
    )
    return artifact, doc, tmp


def test_prepared_packs_hold_every_candidate_with_its_own_excerpt(prepared) -> None:
    from zomah.knowledge import localize

    from embedding_bench.candidates import build_candidates, classify

    artifact, doc, _ = prepared
    frozen = {case.id: case for case in artifact.cases}
    root = EXPERIMENT_DIR.parents[1]
    for case in doc["cases"]:
        source = frozen[case["id"]]
        candidates = build_candidates(source.lexical, source.semantic)
        pack = case["judge_input"]["candidates"]
        assert [c["path"] for c in pack] == sorted(c.path for c in candidates)
        assert [c["id"] for c in pack] == [f"c{i}" for i in range(1, len(pack) + 1)]
        for c in pack:
            assert c["lexical_rank"] == (source.lexical.index(c["path"]) + 1 if c["path"] in source.lexical else None)
            assert c["semantic_rank"] == source.semantic.index(c["path"]) + 1
            region = localize((root / c["path"]).read_text(encoding="utf-8"), source.query)
            assert (c["start_line"], c["end_line"], c["excerpt"]) == (region.start_line, region.end_line, region.excerpt)
        assert case["set"] == ("ambiguous" if classify(candidates).kind == "ambiguous" else "resolved")
        assert case["request"] == render_request(case["judge_input"])
        assert case["request_sha256"] == sha256_json(case["request"])


def test_prepared_requests_carry_no_labels(prepared) -> None:
    from embedding_bench.manifest import load_manifest

    _, doc, _ = prepared
    rationales = [c.rationale for c in load_manifest(EXPERIMENT_DIR / "benchmark.json").cases if c.rationale]
    for case in doc["cases"]:
        request = case["request"]
        question = request["questions"][QUESTION_ID]
        # Structure only: document excerpts may legitimately contain any word, so no text search.
        assert set(request) == {"state", "questions"} and set(request["questions"]) == {QUESTION_ID}
        assert set(question) == {"type", "instructions", "criteria"}
        assert list(question["criteria"]) == [c["id"] for c in case["judge_input"]["candidates"]] + [NONE_LABEL]
        assert request["state"] == f"Question: {case['judge_input']['query']}"
        assert set(case["judge_input"]) == {"query", "candidates"}
        sent = json.dumps(request)
        assert case["id"] not in sent
        assert not any(rationale in sent for rationale in rationales)


def test_relabelling_never_changes_what_a_judge_receives(prepared) -> None:
    from embedding_bench.judge_prepare import build_cases, load_corpus

    artifact, doc, tmp = prepared
    relabelled = dataclasses.replace(
        artifact,
        cases=tuple(dataclasses.replace(c, category="exact", expected_path="README.md") for c in artifact.cases),
    )
    (tmp / "idx2").mkdir()
    documents = load_corpus(relabelled, EXPERIMENT_DIR.parents[1], tmp / "idx2")
    changed = build_cases(relabelled, documents, {c.id: "no such anchor" for c in artifact.cases})
    assert [c["request"] for c in changed] == [c["request"] for c in doc["cases"]]
    assert [c["evaluation"]["expected_path"] for c in changed] == ["README.md"] * len(changed)


def test_prepare_summary_baselines_and_anchor_lines(prepared) -> None:
    from embedding_bench.judge_prepare import anchor_line_range

    _, doc, _ = prepared
    ambiguous = [c for c in doc["cases"] if c["set"] == "ambiguous"]
    summary = doc["summary"]
    assert summary["ambiguous"] == len(ambiguous) and summary["cases"] == 30
    assert summary["ambiguous_baselines"]["lexical_top1"] == sum(
        1 for c in ambiguous if c["evaluation"]["baseline_paths"]["lexical_top1"] == c["evaluation"]["expected_path"]
    )
    assert anchor_line_range("# T\n\nalpha beta\ngamma delta\n", "beta gamma") == (3, 4)
    assert anchor_line_range("alpha", "missing") is None


def test_prepare_refuses_a_corpus_that_changed_since_t3a(prepared) -> None:
    from embedding_bench.judge_prepare import PrepareError, load_corpus

    artifact, _, tmp = prepared
    data = json.loads(json.dumps(artifact.data))
    data["corpus"]["files"][0]["size_bytes"] += 1
    (tmp / "idx3").mkdir()
    with pytest.raises(PrepareError, match="changed size"):
        load_corpus(dataclasses.replace(artifact, data=data), EXPERIMENT_DIR.parents[1], tmp / "idx3")


def test_prepare_cli_refuses_the_t3a_directory_and_existing_cases(tmp_path: Path) -> None:
    t3a_dir = tmp_path / "run-1"
    t3a_dir.mkdir()
    (t3a_dir / "results.json").write_text("{}", encoding="utf-8")
    script = str(EXPERIMENT_DIR / "prepare_cases.py")
    inside = subprocess.run([sys.executable, script, "--input", str(t3a_dir / "results.json"),
                             "--output", str(t3a_dir / "cases.json")], capture_output=True, text=True, timeout=60)
    assert inside.returncode == 2 and "T3a run's directory" in inside.stderr
    existing = tmp_path / "t3d" / "cases.json"
    existing.parent.mkdir()
    existing.write_text("{}", encoding="utf-8")
    again = subprocess.run([sys.executable, script, "--input", str(t3a_dir / "results.json"),
                            "--output", str(existing)], capture_output=True, text=True, timeout=60)
    assert again.returncode == 2 and "--overwrite" in again.stderr

"""T3e: passage windows, excerpts, semantic ranking, leakage invariants, and evaluation."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from embedding_bench.evidence import anchor_span, pairwise, score_span, span_lines, summarize
from embedding_bench.passages import (
    EXCERPT_BUDGETS,
    EmbeddedDocument,
    Passage,
    bounded_excerpt,
    build_windows,
    embed_document,
    lines_passage,
    rank_passages,
    unit,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]


# --- windows ----------------------------------------------------------------------------------


def test_three_line_windows_with_stride_one_keep_line_numbers_and_spans() -> None:
    text = "alpha\nbeta\ngamma\ndelta\nepsilon"
    windows = build_windows(text)
    assert [(w.start_line, w.end_line) for w in windows] == [(1, 3), (2, 4), (3, 5)]
    assert windows[1].text == "beta\ngamma\ndelta"
    for window in windows:
        assert text[window.char_start : window.char_end] == window.text
    assert [(w.start_line, w.end_line) for w in build_windows("one line")] == [(1, 1)]
    assert [(w.start_line, w.end_line) for w in build_windows("a\nb")] == [(1, 2)]
    # A trailing newline adds an empty last line, exactly like read_file's numbering.
    assert [(w.start_line, w.end_line) for w in build_windows("a\nb\nc\n")] == [(1, 3), (2, 4)]


def test_windows_without_letters_or_digits_are_skipped() -> None:
    text = "x\n\n\n```\n---\n\ny"
    kept = [(w.start_line, w.end_line) for w in build_windows(text)]
    assert kept == [(1, 3), (5, 7)]
    assert build_windows("```\n---\n\n") == ()
    with pytest.raises(ValueError):
        build_windows("a", window=0)


def test_lines_passage_matches_the_read_file_numbering() -> None:
    # Only "\n" breaks lines (U+2028 does not), as in read_file and localize().
    text = "# T\nnot a break: \u2028 still line 2\nline 3\n\nline 5"
    passage = lines_passage(text, 3, 5)
    assert passage.text == "line 3\n\nline 5"
    assert lines_passage(text, 2, 2).text == "not a break: \u2028 still line 2"
    assert text[passage.char_start : passage.char_end] == passage.text


# --- bounded excerpts ---------------------------------------------------------------------------


def passage(text: str, offset: int = 100) -> Passage:
    return Passage(1, 3, text, offset, offset + len(text))


def test_short_passages_are_kept_whole_and_long_ones_cut_at_a_word() -> None:
    short = bounded_excerpt(passage("  Short evidence line.\n"), 80)
    assert (short.text, short.truncated, short.char_start, short.char_end) == ("Short evidence line.", False, 102, 122)
    long = passage("The quick brown fox jumps over the lazy dog while the budget keeps shrinking. " * 3)
    for budget in EXCERPT_BUDGETS:
        excerpt = bounded_excerpt(long, budget)
        body = excerpt.text.removesuffix("…")
        assert excerpt.truncated and excerpt.text.endswith("…")
        assert len(body) <= budget and len(body) >= budget // 2
        assert long.text[excerpt.char_start - long.char_start : excerpt.char_end - long.char_start] == body
        assert not long.text[len(body)].isalnum() or not body[-1].isalnum()  # no word split
    cuts = [bounded_excerpt(long, b).text.removesuffix("…") for b in EXCERPT_BUDGETS]
    assert cuts[1].startswith(cuts[0]) and cuts[2].startswith(cuts[1])
    with pytest.raises(ValueError):
        bounded_excerpt(long, 0)


def test_unbroken_text_is_cut_at_the_budget() -> None:
    excerpt = bounded_excerpt(passage("x" * 300), 80)
    assert excerpt.text == "x" * 80 + "…"


# --- semantic ranking --------------------------------------------------------------------------------


def embedded(scores: dict[int, float]) -> EmbeddedDocument:
    """Windows starting at the given lines whose similarity to query [1, 0] is the given score."""

    passages, vectors = [], []
    for start, score in scores.items():
        passages.append(Passage(start, start + 2, f"window {start}", start * 10, start * 10 + 8))
        vectors.append(unit([score, (1 - score * score) ** 0.5 if abs(score) < 1 else 0.0]))
    return EmbeddedDocument("doc.md", tuple(passages), tuple(vectors), 0.0)


def test_passages_rank_by_cosine_and_ties_go_to_the_earliest_line() -> None:
    ranked = rank_passages([2.0, 0.0], embedded({9: 0.5, 4: 0.9, 7: 0.9, 1: 0.1}))
    assert [s.passage.start_line for s in ranked] == [4, 7, 9, 1]
    assert ranked[0].score == pytest.approx(0.9)
    reordered = rank_passages([2.0, 0.0], embedded({7: 0.9, 1: 0.1, 4: 0.9, 9: 0.5}))
    assert [s.passage.start_line for s in reordered] == [4, 7, 9, 1]


class SpyEmbedder:
    """Deterministic fake that records every text it is asked to embed."""

    def __init__(self) -> None:
        from embedding_bench.retrieval import HashingEmbedder

        self.inner = HashingEmbedder()
        self.texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [*self.inner.embed(text), 1e-3]

    def describe(self) -> dict:
        return {"embedder": "spy"}


def test_embed_document_embeds_each_window_once_as_is() -> None:
    spy = SpyEmbedder()
    doc = embed_document(spy, "doc.md", "a b\nc d\ne f\ng h")
    assert spy.texts == [w.text for w in doc.passages] == ["a b\nc d\ne f", "c d\ne f\ng h"]


# --- evidence scoring ------------------------------------------------------------------------------


def test_anchor_spans_follow_whitespace_normalized_matching() -> None:
    text = "# Title\n\n- email integration\n- calendar integration\n"
    span = anchor_span(text, "- email integration - calendar integration")
    assert text[span[0] : span[1]] == "- email integration\n- calendar integration"
    assert span_lines(text, span) == (3, 4)
    assert anchor_span(text, "not there") is None


def test_score_span_whole_inside_and_coverage() -> None:
    anchor = (100, 150)
    assert score_span((90, 160), anchor) == {"contains_anchor": True, "survives": True, "coverage": 1.0}
    assert score_span((110, 140), anchor) == {"contains_anchor": False, "survives": True, "coverage": 0.6}
    assert score_span((120, 170), anchor) == {"contains_anchor": False, "survives": False, "coverage": 0.6}
    assert score_span((0, 50), anchor)["coverage"] == 0.0


def row(case_id: str, category: str, lexical_hit: bool, semantic_hit: bool) -> dict:
    def evaluation(hit: bool) -> dict:
        score = {"contains_anchor": hit, "survives": hit, "coverage": 1.0 if hit else 0.0}
        return {"line_overlap": hit, "passage": score, "excerpts": {str(b): score for b in EXCERPT_BUDGETS}}

    return {
        "id": case_id,
        "category": category,
        "lexical": {"evaluation": evaluation(lexical_hit)},
        "semantic": {"evaluation": evaluation(semantic_hit)},
    }


def test_category_metrics_and_pairwise_buckets() -> None:
    rows = [row("a", "exact", True, True), row("b", "exact", True, False), row("c", "paraphrase", False, True),
            row("d", "paraphrase", False, False), row("e", "paraphrase", False, True)]
    lexical = summarize(rows, "lexical", ["exact", "paraphrase", "conceptual"])
    semantic = summarize(rows, "semantic", ["exact", "paraphrase", "conceptual"])
    assert lexical["overall"]["hits"]["line_overlap"] == 2
    assert semantic["by_category"]["paraphrase"]["hits"]["excerpt_80_survives"] == 2
    assert semantic["by_category"]["paraphrase"]["mean_coverage"]["excerpt_160"] == pytest.approx(2 / 3)
    assert lexical["by_category"]["conceptual"] == {"n": 0, "hits": {}, "mean_coverage": {}}
    assert pairwise(rows, "line_overlap") == {
        "both_hit": ["a"],
        "lexical_only": ["b"],
        "semantic_only": ["c", "e"],
        "both_miss": ["d"],
    }


# --- the full run on the real corpus (fake embedder) ---------------------------------------------------


def run(tmp_path: Path, *, manifest: Path | None = None, factory=None, reference="default", t3a=None):
    pytest.importorskip("zomah")
    from embedding_bench.localization_runner import LEXICAL_OVERLAP_REFERENCE, run_t3e

    work = tmp_path / f"work-{len(list(tmp_path.iterdir()))}"
    work.mkdir()
    return run_t3e(
        root=REPO_ROOT,
        manifest_path=manifest or EXPERIMENT_DIR / "benchmark.json",
        embedder_factory=factory or SpyEmbedder,
        workdir=work,
        t3a_path=t3a,
        lexical_reference=LEXICAL_OVERLAP_REFERENCE if reference == "default" else reference,
    )


def localization_fingerprint(result: dict) -> list:
    """Everything the localizers chose, with timings and evaluation stripped."""

    out = []
    for r in result["cases"]:
        lex = {k: v for k, v in r["lexical"].items() if k not in ("evaluation", "latency_ms")}
        sem = {k: v for k, v in r["semantic"].items() if k not in ("evaluation", "score_ms", "first_hit_rank")}
        sem["top"] = [{k: v for k, v in w.items() if k != "overlaps_anchor"} for w in sem["top"]]
        out.append((r["id"], lex, sem))
    return out


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return run(tmp_path_factory.mktemp("t3e"))[0]


def test_lexical_baseline_is_recomputed_and_matches_the_reference(baseline) -> None:
    reference = baseline["lexical_reference"]
    assert reference["checked"] and reference["matches"]
    assert reference["recomputed"] == {"exact": [7, 7], "paraphrase": [0, 10], "conceptual": [0, 5], "distractor": [3, 8]}
    assert baseline["metrics"]["lexical"]["overall"]["hits"]["line_overlap"] == 10


def test_lexical_results_are_the_unchanged_localize_output(baseline) -> None:
    from zomah.knowledge import localize

    for r in baseline["cases"]:
        text = (REPO_ROOT / r["expected_path"]).read_text(encoding="utf-8")
        region = localize(text, r["query"])
        lex = r["lexical"]
        assert (lex["passage"]["start_line"], lex["passage"]["end_line"]) == (region.start_line, region.end_line)
        assert lex["native_excerpt"]["text"] == region.excerpt
        core = region.excerpt.removeprefix("…").removesuffix("…")
        assert text[lex["native_excerpt"]["char_start"] : lex["native_excerpt"]["char_end"]] == core


def test_every_selected_span_is_exact_source_text(baseline) -> None:
    for r in baseline["cases"]:
        text = (REPO_ROOT / r["expected_path"]).read_text(encoding="utf-8")
        for side in ("lexical", "semantic"):
            result = r[side]
            p = result["passage"]
            assert text[p["char_start"] : p["char_end"]] == p["text"]
            assert p["text"] == "\n".join(text.split("\n")[p["start_line"] - 1 : p["end_line"]])
            for budget, excerpt in result["excerpts"].items():
                body = excerpt["text"].removesuffix("…")
                assert len(body) <= int(budget)
                assert text[excerpt["char_start"] : excerpt["char_end"]] == body


def test_semantic_choice_is_the_top_ranked_window_and_top3_is_consistent(baseline) -> None:
    for r in baseline["cases"]:
        sem = r["semantic"]
        ranking = sem["ranking"]
        assert [sem["passage"]["start_line"], sem["passage"]["end_line"]] == ranking[0][:2]
        assert [[w["start_line"], w["end_line"]] for w in sem["top"]] == [x[:2] for x in ranking[:3]]
        scores = [x[2] for x in ranking]
        assert scores == sorted(scores, reverse=True)
        lo, hi = r["anchor_lines"]
        hits = [i for i, (s, e, _) in enumerate(ranking, start=1) if s <= hi and lo <= e]
        assert sem["first_hit_rank"] == (hits[0] if hits else None)
        assert [w["overlaps_anchor"] for w in sem["top"]] == [w["start_line"] <= hi and lo <= w["end_line"] for w in sem["top"]]


def test_the_model_only_ever_sees_formatted_queries_and_document_windows(tmp_path: Path) -> None:
    from embedding_bench.manifest import load_manifest
    from embedding_bench.retrieval import QUERY_INSTRUCTION, WARMUP_QUERY, format_query

    spies: list[SpyEmbedder] = []

    def factory() -> SpyEmbedder:
        spies.append(SpyEmbedder())
        return spies[-1]

    run(tmp_path, factory=factory)
    manifest = load_manifest(EXPERIMENT_DIR / "benchmark.json")
    queries = {format_query(c.query, QUERY_INSTRUCTION) for c in manifest.cases} | {format_query(WARMUP_QUERY, QUERY_INSTRUCTION)}
    windows = {w.text for path in {c.expected_path for c in manifest.cases}
               for w in build_windows((REPO_ROOT / path).read_text(encoding="utf-8"))}
    assert spies and all(text in queries or text in windows for text in spies[0].texts)
    categories = set(manifest.categories)
    assert not any(text in categories for text in spies[0].texts)


def test_localization_tasks_carry_no_labels() -> None:
    pytest.importorskip("zomah")
    import inspect

    from embedding_bench.localization_runner import LocalizationTask, lexical_localization, semantic_localization

    assert [f.name for f in dataclasses.fields(LocalizationTask)] == ["id", "query", "path"]
    assert list(inspect.signature(lexical_localization).parameters) == ["task", "text"]
    assert list(inspect.signature(semantic_localization).parameters) == ["task", "document", "query_vector"]


def relabelled_manifest(tmp_path: Path) -> Path:
    """Same queries and documents; different anchors and categories."""

    data = json.loads((EXPERIMENT_DIR / "benchmark.json").read_text(encoding="utf-8"))
    names = list(data["categories"])
    for i, case in enumerate(data["cases"]):
        first = next(line for line in (REPO_ROOT / case["expected_path"]).read_text(encoding="utf-8").split("\n") if line.strip())
        case["evidence_anchor"] = first.strip()
        case["category"] = names[(names.index(case["category"]) + 1 + i) % len(names)]
    path = tmp_path / "relabelled.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_labels_change_evaluation_but_never_localization(tmp_path: Path, baseline) -> None:
    changed = run(tmp_path, manifest=relabelled_manifest(tmp_path), reference=None)[0]
    assert localization_fingerprint(changed) == localization_fingerprint(baseline)
    assert [r["anchor_lines"] for r in changed["cases"]] != [r["anchor_lines"] for r in baseline["cases"]]


def test_mutation_a_localizer_that_reads_the_anchor_would_be_caught(tmp_path: Path, baseline, monkeypatch) -> None:
    """Mutation check: leak the anchor into window selection and the invariance test must fail."""

    from embedding_bench import localization_runner
    from embedding_bench.manifest import load_manifest

    def leaky_runs() -> list:
        prints = []
        for manifest in (EXPERIMENT_DIR / "benchmark.json", relabelled_manifest(tmp_path)):
            anchors = {c.query: c.evidence_anchor for c in load_manifest(manifest).cases}
            honest = localization_runner.semantic_localization

            def leaky(task, document, query_vector, _anchors=anchors, _honest=honest):
                result = _honest(task, document, query_vector)
                for p in document.passages:
                    if " ".join(_anchors[task.query].split())[:20] in " ".join(p.text.split()):
                        result["passage"] = localization_runner._passage_dict(p)
                        break
                return result

            monkeypatch.setattr(localization_runner, "semantic_localization", leaky)
            prints.append(localization_fingerprint(run(tmp_path, manifest=manifest, reference=None)[0]))
            monkeypatch.setattr(localization_runner, "semantic_localization", honest)
        return prints

    original, relabelled = leaky_runs()
    assert original != relabelled  # the leak shows up as label-dependent localization


def test_repeated_runs_are_identical(tmp_path: Path, baseline) -> None:
    again = run(tmp_path)[0]
    assert localization_fingerprint(again) == localization_fingerprint(baseline)
    assert again["metrics"] == baseline["metrics"] and again["pairwise"] == baseline["pairwise"]


def test_reference_drift_stops_before_the_model_loads(tmp_path: Path) -> None:
    from embedding_bench.localization_runner import LexicalReferenceDrift

    def must_not_load():
        raise AssertionError("the model was loaded despite lexical drift")

    with pytest.raises(LexicalReferenceDrift, match="exact"):
        run(tmp_path, factory=must_not_load, reference={"exact": (6, 7), "paraphrase": (0, 10), "conceptual": (0, 5), "distractor": (3, 8)})


def test_anchor_drift_is_reported_and_not_scored(tmp_path: Path) -> None:
    data = json.loads((EXPERIMENT_DIR / "benchmark.json").read_text(encoding="utf-8"))
    data["cases"][0]["evidence_anchor"] = "this sentence is not in the document"
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = run(tmp_path, manifest=path, reference=None)[0]
    assert result["drift"] == [{"id": data["cases"][0]["id"], "reason": "evidence_anchor not found in expected_path"}]
    assert len(result["cases"]) == 29


def t3a_artifact_file(tmp_path: Path) -> Path:
    from embedding_bench.retrieval import HashingEmbedder
    from embedding_bench.runner import run_benchmark

    work = tmp_path / "t3a-work"
    work.mkdir()
    t3a = run_benchmark(root=REPO_ROOT, manifest_path=EXPERIMENT_DIR / "benchmark.json", embedder_factory=HashingEmbedder, workdir=work)
    path = tmp_path / "t3a" / "results.json"
    path.parent.mkdir()
    path.write_text(json.dumps(t3a), encoding="utf-8")
    return path


def test_candidate_packs_hold_one_semantic_passage_per_t3c_candidate(tmp_path: Path) -> None:
    pytest.importorskip("zomah")
    from embedding_bench.candidates import build_candidates
    from embedding_bench.fusion import load_t3a

    t3a = t3a_artifact_file(tmp_path)
    result, packs = run(tmp_path, t3a=t3a)
    artifact = load_t3a(t3a)
    assert len(packs["packs"]) == 30 and packs["t3a_sha256"] == artifact.sha256
    for pack, case in zip(packs["packs"], artifact.cases, strict=True):
        union = sorted(c.path for c in build_candidates(case.lexical, case.semantic))
        assert [c["path"] for c in pack["candidates"]] == union
        assert all(set(c) == {"path", "lexical_rank", "semantic_rank", "passage", "similarity", "excerpts", "top"} for c in pack["candidates"])
    assert result["corpus"]["embedded_documents"] >= len({c.expected_path for c in artifact.cases})


def test_corpus_drift_against_t3a_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("zomah")
    from embedding_bench.judge_prepare import PrepareError

    t3a = t3a_artifact_file(tmp_path)
    data = json.loads(t3a.read_text(encoding="utf-8"))
    data["corpus"]["files"][0]["size_bytes"] += 1
    t3a.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PrepareError, match="changed size"):
        run(tmp_path, t3a=t3a)


# --- command line ---------------------------------------------------------------------------------------


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(EXPERIMENT_DIR / "run_localization.py"), *args],
                          capture_output=True, text=True, timeout=300)


def test_cli_fake_embedder_writes_results_report_and_summary(tmp_path: Path) -> None:
    pytest.importorskip("zomah")
    out = tmp_path / "t3e"
    completed = cli("--fake-embedder", "--output-dir", str(out))
    assert completed.returncode == 0, completed.stderr
    assert "lexical reference: matches" in completed.stdout and "FAKE EMBEDDER" in completed.stdout
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert len(result["cases"]) == 30
    report = (out / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# T3e") and "| evidence survives the 80-char excerpt |" in report
    assert not (out / "candidate_packs.json").exists()


def test_cli_rejects_a_malformed_manifest(tmp_path: Path) -> None:
    pytest.importorskip("zomah")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"categories": {}, "cases": []}), encoding="utf-8")
    completed = cli("--fake-embedder", "--manifest", str(bad), "--output-dir", str(tmp_path / "o"))
    assert completed.returncode == 2 and "cannot run T3e" in completed.stderr


def test_evaluation_and_report_modules_need_only_the_standard_library() -> None:
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-c",
         f"import sys; sys.path.insert(0, {str(EXPERIMENT_DIR)!r}); "
         "import embedding_bench.passages, embedding_bench.evidence, embedding_bench.localization_report; "
         "print(sorted(m for m in sys.modules if m.split('.')[0] in "
         "{'torch', 'sentence_transformers', 'transformers', 'zomah', 'pydantic', 'numpy'}))"],
        capture_output=True, text=True, timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]"

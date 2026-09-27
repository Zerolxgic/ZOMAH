"""T3a experiment logic, tested without torch, numpy, or the real model."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from zomah.access import ReadScope
from zomah.knowledge import KnowledgeIndex

from embedding_bench.corpus import BENCHMARK_IGNORED_DIRECTORIES, Document, build_index, load_documents
from embedding_bench.manifest import (
    BenchmarkCase,
    ManifestError,
    check_anchors,
    load_manifest,
    parse_manifest,
)
from embedding_bench.metrics import (
    AGREEMENT_BUCKETS,
    CaseOutcome,
    agreement_bucket,
    latency_summary,
    rank_of,
    score_case,
    summarize,
    summarize_by_category,
    top_k,
)
from embedding_bench.report import render_markdown, render_summary
from embedding_bench.retrieval import (
    QUERY_INSTRUCTION,
    HashingEmbedder,
    RankedDoc,
    SemanticIndex,
    format_query,
    normalize,
    rank_by_similarity,
    run_lexical,
)
from embedding_bench.runner import anchor_terms_in_query, run_benchmark

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


class SynonymEmbedder:
    """Controlled fake: words in the same concept group share one dimension."""

    CONCEPTS = (
        {"zeppelin", "airship"},
        {"freight", "cargo"},
        {"apples", "orchards", "fruit"},
        {"budgets", "invoices", "money"},
        {"valley"},
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        words = re.findall(r"[^\W_]+", text.casefold())
        vector = [float(sum(word in group for word in words)) for group in self.CONCEPTS]
        return vector + [1e-3]  # never a zero vector

    def token_count(self, text: str) -> int:
        return len(text.split())

    @property
    def max_tokens(self) -> int:
        return 12

    def describe(self) -> dict[str, object]:
        return {"embedder": "synonym-fake"}


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "experiments").mkdir(parents=True)
    (root / ".venv").mkdir()
    (root / "alpha.md").write_text(
        "# Alpha\n\nThe zeppelin carries freight across the valley.\n", encoding="utf-8"
    )
    (root / "beta.md").write_text("# Beta\n\nOrchards grow apples in the valley.\n", encoding="utf-8")
    (root / "gamma.txt").write_text("Gamma notes about budgets and invoices.\n", encoding="utf-8")
    (root / "experiments" / "notes.md").write_text("zeppelin zeppelin freight\n", encoding="utf-8")
    (root / ".venv" / "vendored.md").write_text("zeppelin freight\n", encoding="utf-8")
    (root / "code.py").write_text("zeppelin = 'freight'\n", encoding="utf-8")
    return root


def manifest_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "categories": {"exact": "source terms", "paraphrase": "reworded"},
        "cases": [
            {
                "id": "c-exact",
                "category": "exact",
                "query": "zeppelin freight",
                "expected_path": "alpha.md",
                "evidence_anchor": "The zeppelin carries freight",
            },
            {
                "id": "c-para",
                "category": "paraphrase",
                "query": "airship cargo",
                "expected_path": "alpha.md",
                "evidence_anchor": "carries freight across the valley.",
                "rationale": "reworded c-exact",
            },
            {
                "id": "c-fruit",
                "category": "paraphrase",
                "query": "fruit",
                "expected_path": "beta.md",
                "evidence_anchor": "Orchards grow apples",
            },
            {
                "id": "c-missing-doc",
                "category": "exact",
                "query": "anything",
                "expected_path": "missing.md",
                "evidence_anchor": "whatever",
            },
            {
                "id": "c-missing-anchor",
                "category": "exact",
                "query": "invoices",
                "expected_path": "gamma.txt",
                "evidence_anchor": "receipts and invoices",
            },
        ],
    }
    data.update(overrides)
    return data


def write_manifest(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- manifest validation --------------------------------------------------------------


def test_manifest_parses_cases_in_order() -> None:
    manifest = parse_manifest(manifest_data())
    assert [case.id for case in manifest.cases][:2] == ["c-exact", "c-para"]
    assert manifest.cases[1].rationale == "reworded c-exact"
    assert list(manifest.categories) == ["exact", "paraphrase"]


@pytest.mark.parametrize("field", ["id", "category", "query", "expected_path", "evidence_anchor"])
def test_manifest_rejects_missing_or_blank_required_fields(field: str) -> None:
    for bad in (None, "", "   "):
        data = manifest_data()
        case = dict(data["cases"][0])  # type: ignore[index]
        if bad is None:
            del case[field]
        else:
            case[field] = bad
        data["cases"] = [case]
        with pytest.raises(ManifestError, match=field):
            parse_manifest(data)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(expected_pth="x.md"), "unknown fields"),
        (lambda c: c.update(category="vibes"), "unknown category"),
        (lambda c: c.update(expected_path="/abs/alpha.md"), "relative POSIX"),
        (lambda c: c.update(expected_path="../alpha.md"), "relative POSIX"),
        (lambda c: c.update(expected_path="docs\\alpha.md"), "relative POSIX"),
        (lambda c: c.update(expected_path="./alpha.md"), "relative POSIX"),
        (lambda c: c.update(rationale=""), "rationale"),
    ],
)
def test_manifest_rejects_malformed_cases(mutate, message: str) -> None:
    data = manifest_data()
    case = dict(data["cases"][0])  # type: ignore[index]
    mutate(case)
    data["cases"] = [case]
    with pytest.raises(ManifestError, match=message):
        parse_manifest(data)


def test_manifest_rejects_duplicate_ids_and_bad_shapes() -> None:
    data = manifest_data()
    data["cases"] = [data["cases"][0], data["cases"][0]]  # type: ignore[index]
    with pytest.raises(ManifestError, match="duplicate id"):
        parse_manifest(data)
    with pytest.raises(ManifestError, match="JSON object"):
        parse_manifest([])
    with pytest.raises(ManifestError, match="cases"):
        parse_manifest(manifest_data(cases=[]))
    with pytest.raises(ManifestError, match="categories"):
        parse_manifest(manifest_data(categories={}))


def test_committed_manifest_is_valid_and_covers_every_category() -> None:
    manifest = load_manifest(EXPERIMENT_DIR / "benchmark.json")
    counts = {name: 0 for name in manifest.categories}
    for case in manifest.cases:
        counts[case.category] += 1
    assert set(counts) == {"exact", "paraphrase", "conceptual", "distractor"}
    assert all(count >= 5 for count in counts.values())
    assert 20 <= len(manifest.cases) <= 30


# --- evidence-anchor drift ------------------------------------------------------------


def test_anchor_checks_detect_drift_and_tolerate_rewrapping() -> None:
    cases = parse_manifest(manifest_data()).cases
    documents = {
        # The anchor for c-para is re-wrapped across a line break here.
        "alpha.md": "# Alpha\n\nThe zeppelin carries freight\nacross   the valley.\n",
        "beta.md": "Orchards grow apples.",
        "gamma.txt": "Gamma notes about budgets and invoices.",
    }
    checks = check_anchors(cases, documents)
    assert checks["c-exact"].status == "ok"
    assert checks["c-para"].status == "ok"
    assert checks["c-missing-doc"].status == "drift"
    assert "not in the corpus" in (checks["c-missing-doc"].reason or "")
    assert checks["c-missing-anchor"].status == "drift"
    assert "evidence_anchor" in (checks["c-missing-anchor"].reason or "")


def test_anchor_checks_flag_anchors_shared_with_other_documents() -> None:
    case = BenchmarkCase("c", "exact", "q", "a.md", "shared words")
    checks = check_anchors([case], {"a.md": "some shared words", "b.md": "shared\nwords", "c.md": "no"})
    assert checks["c"].status == "ok"
    assert checks["c"].also_in == ("b.md",)


def test_anchor_wording_changes_count_as_drift() -> None:
    case = BenchmarkCase("c", "exact", "q", "a.md", "not a token estimate.")
    assert check_anchors([case], {"a.md": "not a token estimate, and"})["c"].status == "drift"


# --- ranking metrics and top-k ----------------------------------------------------------


def ranking(*paths: str) -> tuple[RankedDoc, ...]:
    return tuple(RankedDoc(path, float(len(paths) - i)) for i, path in enumerate(paths))


def test_score_case_ranks_and_reciprocal_rank() -> None:
    ranked = ranking("a", "b", "c", "d")
    assert score_case(ranked, "a") == CaseOutcome(rank=1, top1=True, top3=True, reciprocal_rank=1.0)
    assert score_case(ranked, "c") == CaseOutcome(rank=3, top1=False, top3=True, reciprocal_rank=1 / 3)
    assert score_case(ranked, "d") == CaseOutcome(rank=4, top1=False, top3=False, reciprocal_rank=0.25)
    assert score_case(ranked, "z") == CaseOutcome(rank=None, top1=False, top3=False, reciprocal_rank=0.0)
    assert rank_of((), "a") is None


def test_summaries_compute_top1_top3_and_mrr() -> None:
    outcomes = {
        "x": score_case(ranking("a"), "a"),  # rank 1
        "y": score_case(ranking("b", "a"), "a"),  # rank 2
        "z": score_case(ranking("b"), "a"),  # missing
        "w": score_case(ranking("b", "c", "d", "a"), "a"),  # rank 4
    }
    overall = summarize(outcomes.values())
    assert overall["n"] == 4
    assert overall["top1"] == pytest.approx(0.25)
    assert overall["top3"] == pytest.approx(0.5)
    assert overall["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    by_category = summarize_by_category(
        outcomes, {"x": "k1", "y": "k1", "z": "k2", "w": "k2"}, ["k1", "k2", "k3"]
    )
    assert by_category["k1"] == {"n": 2, "top1": 0.5, "top3": 1.0, "mrr": 0.75}
    assert by_category["k2"]["mrr"] == pytest.approx(0.125)
    assert by_category["k3"] == {"n": 0, "top1": None, "top3": None, "mrr": None}


def test_top_k() -> None:
    ranked = ranking("a", "b", "c", "d")
    assert [doc.path for doc in top_k(ranked, 3)] == ["a", "b", "c"]
    assert [doc.path for doc in top_k(ranked[:2], 3)] == ["a", "b"]
    with pytest.raises(ValueError):
        top_k(ranked, 0)


def test_agreement_buckets_use_top1() -> None:
    hit = score_case(ranking("a"), "a")
    second = score_case(ranking("b", "a"), "a")
    assert agreement_bucket(hit, hit) == "both_correct"
    assert agreement_bucket(hit, second) == "lexical_correct_semantic_wrong"
    assert agreement_bucket(second, hit) == "semantic_correct_lexical_wrong"
    assert agreement_bucket(second, second) == "both_wrong"


def test_latency_summary() -> None:
    assert latency_summary([3.0, 1.0, 2.0, 10.0]) == {"n": 4, "median": 2.5, "mean": 4.0, "min": 1.0, "max": 10.0}
    assert latency_summary([])["median"] is None


# --- normalized-vector ranking --------------------------------------------------------


def test_normalize_produces_unit_vectors_and_rejects_degenerate_input() -> None:
    unit = normalize([3.0, 4.0])
    assert unit == pytest.approx((0.6, 0.8))
    assert math.fsum(v * v for v in unit) == pytest.approx(1.0)
    for bad in ([0.0, 0.0], [], [1.0, math.nan], [math.inf, 1.0]):
        with pytest.raises(ValueError):
            normalize(bad)


def test_ranking_is_cosine_and_ignores_vector_magnitude() -> None:
    docs = [("near.md", [100.0, 1.0]), ("far.md", [0.0, 0.01]), ("mid.md", [1.0, 1.0])]
    ranked = rank_by_similarity([2.0, 0.0], docs)
    assert [doc.path for doc in ranked] == ["near.md", "mid.md", "far.md"]
    assert ranked[1].score == pytest.approx(1 / math.sqrt(2))
    assert ranked[2].score == pytest.approx(0.0)


def test_ties_break_by_path_regardless_of_input_order() -> None:
    docs = [("c.md", [1.0, 0.0]), ("a.md", [2.0, 0.0]), ("b.md", [0.0, 1.0]), ("aa.md", [5.0, 0.0])]
    for ordering in (docs, list(reversed(docs))):
        ranked = rank_by_similarity([1.0, 0.0], ordering)
        assert [doc.path for doc in ranked] == ["a.md", "aa.md", "c.md", "b.md"]


# --- corpus and lexical baseline -----------------------------------------------------------


def test_corpus_is_exactly_what_the_knowledge_index_holds(corpus_root: Path, tmp_path: Path) -> None:
    index, refresh = build_index(corpus_root.resolve(), tmp_path / "k.db")
    documents = load_documents(index, corpus_root.resolve())
    assert [doc.path for doc in documents] == ["alpha.md", "beta.md", "gamma.txt"]
    assert refresh.indexed_files == 3
    assert documents[0].title == "Alpha"
    assert documents[0].text == (corpus_root / "alpha.md").read_text(encoding="utf-8")


def test_lexical_results_come_from_the_unchanged_knowledge_index(corpus_root: Path, tmp_path: Path) -> None:
    root = corpus_root.resolve()
    index, _ = build_index(root, tmp_path / "k.db")
    run = run_lexical(index, root, "valley freight", limit=10)
    assert [doc.path for doc in run.ranking] == ["alpha.md", "beta.md"]  # gamma matches no term
    assert run.ranking[0].score > run.ranking[1].score
    assert run.total_ms >= 0 and run.embed_ms is None

    # Same order as calling the real KnowledgeIndex directly with the same exclusions.
    direct = KnowledgeIndex(
        tmp_path / "direct.db",
        ReadScope.from_paths([root]),
        ignored_directories=BENCHMARK_IGNORED_DIRECTORIES,
    )
    direct.refresh()
    assert [Path(hit.path).name for hit in direct.search("valley freight", limit=10)] == [
        doc.path for doc in run.ranking
    ]
    assert run_lexical(index, root, "airship cargo", limit=10).ranking == ()


# --- semantic collection with a fake embedder -------------------------------------------------


def test_format_query_matches_the_documented_qwen_instruction_format() -> None:
    assert format_query("what is x?", "Find it.") == "Instruct: Find it.\nQuery:what is x?"
    assert format_query("q").startswith(f"Instruct: {QUERY_INSTRUCTION}\nQuery:")


def test_semantic_index_embeds_documents_once_and_queries_with_instruction(
    corpus_root: Path, tmp_path: Path
) -> None:
    root = corpus_root.resolve()
    index, _ = build_index(root, tmp_path / "k.db")
    documents = load_documents(index, root)
    embedder = SynonymEmbedder()
    semantic = SemanticIndex.build(embedder, documents)

    assert embedder.calls == [doc.text for doc in documents]  # raw text, no instruction, once each
    assert semantic.dimension == 6
    assert all(doc.embed_s >= 0 and doc.tokens for doc in semantic.documents)

    run = semantic.search("airship cargo")
    assert embedder.calls[-1] == format_query("airship cargo")
    assert len(embedder.calls) == len(documents) + 1
    assert run.ranking[0].path == "alpha.md"
    assert len(run.ranking) == len(documents)
    assert run.embed_ms is not None and run.score_ms is not None
    assert run.total_ms >= run.embed_ms


def test_semantic_index_rejects_mismatched_dimensions() -> None:
    class Broken(SynonymEmbedder):
        def embed(self, text: str) -> list[float]:
            return [1.0, 2.0] if text.startswith("Instruct:") else super().embed(text)

    documents = [Document("a.md", "A", "zeppelin", 8), Document("b.md", "B", "apples", 6)]
    semantic = SemanticIndex.build(Broken(), documents)
    with pytest.raises(ValueError, match="dimension"):
        semantic.search("zeppelin")


def test_hashing_embedder_is_deterministic_across_instances() -> None:
    assert HashingEmbedder().embed("zeppelin freight") == HashingEmbedder().embed("freight zeppelin")
    assert HashingEmbedder().token_count("x") is None


# --- full run and reports --------------------------------------------------------------------


def test_run_benchmark_collects_both_sides_and_skips_drifted_cases(
    corpus_root: Path, tmp_path: Path
) -> None:
    manifest_path = write_manifest(tmp_path, manifest_data())
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = run_benchmark(
        root=corpus_root,
        manifest_path=manifest_path,
        embedder_factory=SynonymEmbedder,
        workdir=workdir,
    )
    json.dumps(result)  # JSON-ready

    assert result["corpus"]["documents"] == 3
    assert result["corpus"]["embedding_dimension"] == 6
    assert (result["manifest"]["cases"], result["manifest"]["scored"], result["manifest"]["drift"]) == (5, 3, 2)
    assert {item["id"] for item in result["drift"]} == {"c-missing-doc", "c-missing-anchor"}
    cases = {case["id"]: case for case in result["cases"]}
    assert set(cases) == {"c-exact", "c-para", "c-fruit"}

    # Lexical cannot match the reworded queries; the synonym embedder can.
    assert cases["c-exact"]["lexical"]["rank"] == 1 and cases["c-exact"]["semantic"]["rank"] == 1
    assert cases["c-para"]["lexical"]["rank"] is None
    assert cases["c-para"]["lexical"]["expected_score"] is None
    assert cases["c-para"]["semantic"]["rank"] == 1
    assert cases["c-para"]["agreement"] == "semantic_correct_lexical_wrong"
    assert cases["c-exact"]["anchor_terms_in_query"] == ["freight", "zeppelin"]
    assert cases["c-para"]["anchor_terms_in_query"] == []

    agreement = result["agreement"]
    assert list(agreement) == list(AGREEMENT_BUCKETS)
    assert sorted(i for ids in agreement.values() for i in ids) == sorted(cases)

    lexical = result["metrics"]["lexical"]
    semantic = result["metrics"]["semantic"]
    assert lexical["overall"]["n"] == 3
    assert lexical["by_category"]["paraphrase"]["top1"] == 0.0
    assert semantic["by_category"]["paraphrase"]["top1"] == 1.0
    assert lexical["by_category"]["exact"] == {"n": 1, "top1": 1.0, "top3": 1.0, "mrr": 1.0}

    timings = result["timings"]
    assert timings["model_load_s"] >= 0 and timings["corpus_embed_s"] >= 0
    assert timings["semantic_query_ms"]["n"] == 3 and timings["lexical_query_ms"]["n"] == 3
    assert result["config"]["embedder"] == {"embedder": "synonym-fake"}
    assert "experiments" in result["config"]["corpus"]["ignored_directories"]
    assert result["corpus"]["over_max_tokens"] == []  # every fixture document is under 12 tokens


def test_reports_include_metrics_disagreements_and_every_query(corpus_root: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = run_benchmark(
        root=corpus_root,
        manifest_path=write_manifest(tmp_path, manifest_data()),
        embedder_factory=SynonymEmbedder,
        workdir=workdir,
    )
    markdown = render_markdown(result)
    for case_id in ("c-exact", "c-para", "c-fruit"):
        assert f"| {case_id} |" in markdown
    assert "## Semantic correct / lexical wrong" in markdown
    assert "### c-para (paraphrase)" in markdown
    assert "Lexical: rank —, expected score not returned" in markdown
    assert "| overall | 3 |" in markdown and "| paraphrase | 2 |" in markdown
    assert "c-missing-doc" in markdown and "not in the corpus" in markdown
    assert "FAKE EMBEDDER" not in markdown

    summary = render_summary(result, json_path="r.json", markdown_path="r.md")
    assert "semantic_correct_lexical_wrong=" in summary
    assert "DRIFT c-missing-anchor" in summary
    assert summary.rstrip().endswith("markdown: r.md")


def test_cli_fake_embedder_writes_json_and_markdown(corpus_root: Path, tmp_path: Path, capsys) -> None:
    import run_benchmark as cli

    out = tmp_path / "out"
    code = cli.main(
        [
            "--fake-embedder",
            "--root",
            str(corpus_root),
            "--manifest",
            str(write_manifest(tmp_path, manifest_data())),
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["config"]["embedder"]["embedder"] == "fake-hashing"
    assert "FAKE EMBEDDER" in (out / "report.md").read_text(encoding="utf-8")
    assert "FAKE EMBEDDER" in capsys.readouterr().out


def test_cli_reports_an_invalid_manifest(corpus_root: Path, tmp_path: Path, capsys) -> None:
    import run_benchmark as cli

    bad = write_manifest(tmp_path, manifest_data(categories={}))
    code = cli.main(["--fake-embedder", "--root", str(corpus_root), "--manifest", str(bad), "--output-dir", str(tmp_path / "o")])
    assert code == 2
    assert "invalid benchmark manifest" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_anchor_terms_diagnostic_ignores_function_words() -> None:
    assert anchor_terms_in_query("Is the cap a token estimate?", "not a token estimate") == ["estimate", "token"]

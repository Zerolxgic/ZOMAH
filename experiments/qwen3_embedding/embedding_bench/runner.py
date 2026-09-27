"""Run the lexical baseline and the semantic experiment over one manifest.

The result is a plain JSON-ready dict; the Markdown and terminal reports are
rendered from it, so a saved ``results.json`` can always be re-rendered.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from zomah.capabilities.search_knowledge import MAX_RESULTS

from embedding_bench.corpus import build_index, corpus_config, load_documents
from embedding_bench.manifest import check_anchors, load_manifest
from embedding_bench.metrics import (
    AGREEMENT_BUCKETS,
    CaseOutcome,
    agreement_bucket,
    latency_summary,
    score_case,
    summarize,
    summarize_by_category,
    top_k,
)
from embedding_bench.retrieval import (
    QUERY_INSTRUCTION,
    WARMUP_QUERY,
    Embedder,
    QueryRun,
    SemanticIndex,
    format_query,
    run_lexical,
)

EXPERIMENT = "T3a: Qwen3-Embedding-0.6B document ranking vs ZOMAH lexical baseline"

# Only used for the "anchor terms in query" diagnostic; never for ranking.
_DIAGNOSTIC_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how if in is it its not of on one or "
    "so that the this to what when which why with".split()
)


def run_benchmark(
    *,
    root: Path,
    manifest_path: Path,
    embedder_factory: Callable[[], Embedder],
    workdir: Path,
    instruction: str = QUERY_INSTRUCTION,
) -> dict[str, Any]:
    started_at = _now()
    root = root.expanduser().resolve(strict=True)
    manifest = load_manifest(manifest_path)
    memory: dict[str, float | None] = {"after_imports": peak_rss_mb()}

    index, refresh = build_index(root, workdir / "knowledge.db")
    documents = load_documents(index, root)
    if not documents:
        raise ValueError(f"no indexable documents under {root}")
    checks = check_anchors(manifest.cases, {doc.path: doc.text for doc in documents})
    scored = [case for case in manifest.cases if checks[case.id].status == "ok"]
    lexical_limit = min(len(documents), MAX_RESULTS)

    lexical_runs = {
        case.id: run_lexical(index, root, case.query, limit=lexical_limit) for case in scored
    }
    memory["after_lexical"] = peak_rss_mb()

    load_start = time.perf_counter()
    embedder = embedder_factory()
    model_load_s = time.perf_counter() - load_start
    memory["after_model_load"] = peak_rss_mb()

    build_start = time.perf_counter()
    semantic_index = SemanticIndex.build(embedder, documents)
    corpus_build_wall_s = time.perf_counter() - build_start
    memory["after_corpus_embedding"] = peak_rss_mb()

    warmup = semantic_index.search(WARMUP_QUERY, instruction)
    semantic_runs = {case.id: semantic_index.search(case.query, instruction) for case in scored}
    memory["end"] = peak_rss_mb()

    categories = {case.id: case.category for case in scored}
    lexical_outcomes = {case.id: score_case(lexical_runs[case.id].ranking, case.expected_path) for case in scored}
    semantic_outcomes = {case.id: score_case(semantic_runs[case.id].ranking, case.expected_path) for case in scored}
    category_order = list(manifest.categories)

    case_rows = []
    agreement: dict[str, list[str]] = {bucket: [] for bucket in AGREEMENT_BUCKETS}
    for case in scored:
        bucket = agreement_bucket(lexical_outcomes[case.id], semantic_outcomes[case.id])
        agreement[bucket].append(case.id)
        case_rows.append(
            {
                "id": case.id,
                "category": case.category,
                "query": case.query,
                "expected_path": case.expected_path,
                "evidence_anchor": case.evidence_anchor,
                "rationale": case.rationale,
                "anchor_also_in": list(checks[case.id].also_in),
                "anchor_terms_in_query": anchor_terms_in_query(case.query, case.evidence_anchor),
                "agreement": bucket,
                "lexical": _side(lexical_runs[case.id], lexical_outcomes[case.id], case.expected_path),
                "semantic": _side(semantic_runs[case.id], semantic_outcomes[case.id], case.expected_path),
            }
        )

    embedded = {doc.path: doc for doc in semantic_index.documents}
    max_tokens = embedder.max_tokens
    corpus_rows = [
        {
            "path": doc.path,
            "title": doc.title,
            "size_bytes": doc.size_bytes,
            "tokens": embedded[doc.path].tokens,
            "embed_s": embedded[doc.path].embed_s,
            "over_max_tokens": (
                None
                if max_tokens is None or embedded[doc.path].tokens is None
                else embedded[doc.path].tokens > max_tokens
            ),
        }
        for doc in documents
    ]

    describe = getattr(embedder, "describe", None)
    return {
        "experiment": EXPERIMENT,
        "started_at": started_at,
        "finished_at": _now(),
        "config": {
            "embedder": describe() if callable(describe) else {"embedder": type(embedder).__name__},
            "query_instruction": instruction,
            "query_format_example": format_query("<query>", instruction),
            "document_instruction": None,
            "similarity": "cosine: dot product of L2-normalized vectors, brute force in memory",
            "tie_break": "path ascending (lexical: ORDER BY bm25, path)",
            "top1_correct_means": "expected_path ranked first",
            "corpus": corpus_config(root),
            "lexical": {
                "implementation": "zomah.capabilities.search_knowledge (unchanged; refreshes before each search)",
                "max_results": lexical_limit,
                "rank_cutoff_applies": len(documents) > MAX_RESULTS,
            },
        },
        "environment": environment_info(root),
        "manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "categories": manifest.categories,
            "cases": len(manifest.cases),
            "scored": len(scored),
            "drift": len(manifest.cases) - len(scored),
        },
        "drift": [
            {
                "id": case.id,
                "category": case.category,
                "expected_path": case.expected_path,
                "reason": checks[case.id].reason,
            }
            for case in manifest.cases
            if checks[case.id].status != "ok"
        ],
        "corpus": {
            "documents": len(documents),
            "total_bytes": sum(doc.size_bytes for doc in documents),
            "refresh": {
                "scanned": refresh.scanned_files,
                "indexed": refresh.indexed_files,
                "skipped": refresh.skipped_files,
            },
            "embedding_dimension": semantic_index.dimension,
            "max_tokens": max_tokens,
            "over_max_tokens": [row["path"] for row in corpus_rows if row["over_max_tokens"]],
            "files": corpus_rows,
        },
        "timings": {
            "model_load_s": model_load_s,
            "corpus_embed_s": sum(doc.embed_s for doc in semantic_index.documents),
            "corpus_build_wall_s": corpus_build_wall_s,
            "warmup_query_ms": warmup.total_ms,
            "semantic_query_ms": latency_summary([semantic_runs[c.id].total_ms for c in scored]),
            "semantic_query_embed_ms": latency_summary([semantic_runs[c.id].embed_ms for c in scored]),
            "semantic_query_score_ms": latency_summary([semantic_runs[c.id].score_ms for c in scored]),
            "lexical_query_ms": latency_summary([lexical_runs[c.id].total_ms for c in scored]),
        },
        "memory_peak_rss_mb": memory,
        "metrics": {
            side: {
                "overall": summarize(outcomes.values()),
                "by_category": summarize_by_category(outcomes, categories, category_order),
            }
            for side, outcomes in (("lexical", lexical_outcomes), ("semantic", semantic_outcomes))
        },
        "agreement": agreement,
        "cases": case_rows,
    }


def anchor_terms_in_query(query: str, anchor: str) -> list[str]:
    """Content words the query shares with its evidence anchor (diagnostic only)."""

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[^\W_]+", text.casefold()) if w not in _DIAGNOSTIC_STOPWORDS}

    return sorted(words(query) & words(anchor))


def peak_rss_mb() -> float | None:
    """Process peak RSS so far (stdlib only; None where unavailable)."""

    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def environment_info(root: Path) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "sqlite": sqlite3.sqlite_version,
        "zomah_commit": _git_commit(root),
    }


def _side(run: QueryRun, outcome: CaseOutcome, expected_path: str) -> dict[str, Any]:
    expected_score = next((doc.score for doc in run.ranking if doc.path == expected_path), None)
    side = {
        "rank": outcome.rank,
        "top1": outcome.top1,
        "top3": outcome.top3,
        "reciprocal_rank": outcome.reciprocal_rank,
        "expected_score": expected_score,
        "top3_hits": [{"path": doc.path, "score": doc.score} for doc in top_k(run.ranking, 3)],
        "ranking": [{"path": doc.path, "score": doc.score} for doc in run.ranking],
        "latency_ms": run.total_ms,
    }
    if run.embed_ms is not None:
        side["embed_ms"] = run.embed_ms
        side["score_ms"] = run.score_ms
    return side


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

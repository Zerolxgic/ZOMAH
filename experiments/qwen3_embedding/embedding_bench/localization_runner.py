"""T3e: lexical vs semantic within-document evidence localization on the frozen benchmark.

Two strictly ordered phases per localizer: localization runs on label-free
tasks (id, query, document), then evaluation reads the benchmark anchors.
The lexical reference is checked before the embedding model is loaded.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from zomah.knowledge import localize

from embedding_bench.candidates import build_candidates
from embedding_bench.corpus import build_index, corpus_config, load_documents
from embedding_bench.evidence import anchor_span, evaluate, pairwise, span_lines, summarize
from embedding_bench.fusion import check_labels, load_t3a
from embedding_bench.judge_prepare import load_corpus
from embedding_bench.manifest import check_anchors, load_manifest
from embedding_bench.passages import (
    EXCERPT_BUDGETS,
    STRIDE,
    WINDOW_LINES,
    EmbeddedDocument,
    Embedder,
    Passage,
    bounded_excerpt,
    embed_document,
    lines_passage,
    rank_passages,
)
from embedding_bench.retrieval import QUERY_INSTRUCTION, WARMUP_QUERY, format_query
from embedding_bench.runner import environment_info, peak_rss_mb

EXPERIMENT = "T3e: semantic within-document evidence localization vs lexical localize()"
# Lexical line-overlap coverage observed before T3e (count, n). A different recomputation means
# the corpus or benchmark drifted, and the run stops before loading the model.
LEXICAL_OVERLAP_REFERENCE = {"exact": (7, 7), "paraphrase": (0, 10), "conceptual": (0, 5), "distractor": (3, 8)}
TOP_K = 3


class LexicalReferenceDrift(RuntimeError):
    """The recomputed lexical baseline differs from the recorded reference."""


@dataclass(frozen=True, slots=True)
class LocalizationTask:
    """Everything a localizer may see. No category, anchor, or correctness label."""

    id: str
    query: str
    path: str


# --- localization (label-free) ----------------------------------------------------------------


def lexical_localization(task: LocalizationTask, text: str) -> dict[str, Any]:
    start = time.perf_counter()
    region = localize(text, task.query)
    latency_ms = (time.perf_counter() - start) * 1000
    passage = lines_passage(text, region.start_line, region.end_line)
    core = region.excerpt.removeprefix("…").removesuffix("…")
    offset = passage.text.find(core)
    if offset < 0:
        raise RuntimeError(f"{task.id}: localize() excerpt is not inside its reported lines")
    return {
        "method": "lexical: zomah.knowledge.localize (unchanged)",
        "path": task.path,
        "passage": _passage_dict(passage),
        "excerpts": _excerpts(passage),
        "native_excerpt": {
            "text": region.excerpt,
            "char_start": passage.char_start + offset,
            "char_end": passage.char_start + offset + len(core),
        },
        "latency_ms": latency_ms,
    }


def semantic_localization(task: LocalizationTask, document: EmbeddedDocument, query_vector: Sequence[float]) -> dict[str, Any]:
    start = time.perf_counter()
    ranked = rank_passages(query_vector, document)
    score_ms = (time.perf_counter() - start) * 1000
    best = ranked[0]
    return {
        "method": f"semantic: Qwen3-Embedding over {WINDOW_LINES}-line windows, stride {STRIDE}",
        "path": task.path,
        "passage": _passage_dict(best.passage),
        "similarity": best.score,
        "excerpts": _excerpts(best.passage),
        "top": [_scored_dict(s) for s in ranked[:TOP_K]],
        "ranking": [[s.passage.start_line, s.passage.end_line, round(s.score, 6)] for s in ranked],
        "windows": len(ranked),
        "score_ms": score_ms,
    }


def _passage_dict(passage: Passage) -> dict[str, Any]:
    return {
        "start_line": passage.start_line,
        "end_line": passage.end_line,
        "text": passage.text,
        "char_start": passage.char_start,
        "char_end": passage.char_end,
    }


def _excerpts(passage: Passage) -> dict[str, Any]:
    out = {}
    for budget in EXCERPT_BUDGETS:
        excerpt = bounded_excerpt(passage, budget)
        out[str(budget)] = {"text": excerpt.text, "char_start": excerpt.char_start, "char_end": excerpt.char_end}
    return out


def _scored_dict(scored) -> dict[str, Any]:
    return {**_passage_dict(scored.passage), "similarity": scored.score}


# --- the run ---------------------------------------------------------------------------------------


def run_t3e(
    *,
    root: Path,
    manifest_path: Path,
    embedder_factory: Callable[[], Embedder],
    workdir: Path,
    t3a_path: Path | None = None,
    lexical_reference: Mapping[str, tuple[int, int]] | None = LEXICAL_OVERLAP_REFERENCE,
    instruction: str = QUERY_INSTRUCTION,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started_at = _now()
    root = root.expanduser().resolve(strict=True)
    manifest = load_manifest(manifest_path)
    memory: dict[str, float | None] = {"after_imports": peak_rss_mb()}

    artifact = None
    if t3a_path is not None:
        artifact = load_t3a(t3a_path)
        check_labels(artifact, manifest_path)
        documents = load_corpus(artifact, root, workdir)  # refuses a corpus that changed since T3a
    else:
        index, _ = build_index(root, workdir / "knowledge.db")
        documents = load_documents(index, root)
    texts = {doc.path: doc.text for doc in documents}
    checks = check_anchors(manifest.cases, texts)
    scored = [case for case in manifest.cases if checks[case.id].status == "ok"]
    categories = list(manifest.categories)
    tasks = [LocalizationTask(case.id, case.query, case.expected_path) for case in scored]

    # Lexical: localize, then evaluate, then check the reference before any model loads.
    lexical = {task.id: lexical_localization(task, texts[task.path]) for task in tasks}
    anchors = {case.id: anchor_span(texts[case.expected_path], case.evidence_anchor) for case in scored}
    for case in scored:
        lexical[case.id]["evaluation"] = evaluate(lexical[case.id], texts[case.expected_path], anchors[case.id])
    reference = check_lexical_reference(scored, lexical, lexical_reference)
    memory["after_lexical"] = peak_rss_mb()

    # Semantic: load, embed every needed document's windows once, then localize each task.
    load_start = time.perf_counter()
    embedder = embedder_factory()
    model_load_s = time.perf_counter() - load_start
    memory["after_model_load"] = peak_rss_mb()

    pack_cases = _pack_cases(artifact) if artifact is not None else []
    needed = sorted({task.path for task in tasks} | {path for _, _, paths in pack_cases for path in paths})
    embedded = {}
    for number, path in enumerate(needed, start=1):
        embedded[path] = embed_document(embedder, path, texts[path])
        if progress is not None:
            progress(
                f"[{number}/{len(needed)}] {path}: {len(embedded[path].passages)} windows in "
                f"{embedded[path].embed_s:.1f} s"
            )
    memory["after_passage_embedding"] = peak_rss_mb()

    warm_start = time.perf_counter()
    embedder.embed(format_query(WARMUP_QUERY, instruction))
    warmup_ms = (time.perf_counter() - warm_start) * 1000

    query_vectors: dict[str, Any] = {}
    query_embed_ms: dict[str, float] = {}
    semantic = {}
    for task in tasks:
        start = time.perf_counter()
        vector = embedder.embed(format_query(task.query, instruction))
        query_embed_ms[task.id] = (time.perf_counter() - start) * 1000
        query_vectors[task.query] = vector
        semantic[task.id] = semantic_localization(task, embedded[task.path], vector)
    memory["end"] = peak_rss_mb()

    # Evaluation of the semantic results: labels are read only from here on.
    rows = []
    for case in scored:
        text, anchor = texts[case.expected_path], anchors[case.id]
        sem = semantic[case.id]
        sem["evaluation"] = evaluate(sem, text, anchor)
        anchor_lines = span_lines(text, anchor)
        for window in sem["top"]:
            window["overlaps_anchor"] = window["start_line"] <= anchor_lines[1] and anchor_lines[0] <= window["end_line"]
        sem["first_hit_rank"] = next(
            (rank for rank, (s, e, _) in enumerate(sem["ranking"], start=1) if s <= anchor_lines[1] and anchor_lines[0] <= e),
            None,
        )
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "query": case.query,
                "expected_path": case.expected_path,
                "anchor_lines": list(anchor_lines),
                "anchor_chars": list(anchor),
                "anchor_length": anchor[1] - anchor[0],
                "lexical": lexical[case.id],
                "semantic": sem,
            }
        )

    lexical_latency = [lexical[t.id]["latency_ms"] for t in tasks]
    warm = [query_embed_ms[t.id] + semantic[t.id]["score_ms"] for t in tasks]
    cold = [embedded[t.path].embed_s * 1000 + query_embed_ms[t.id] + semantic[t.id]["score_ms"] for t in tasks]
    per_doc_ms = [doc.embed_s * 1000 for doc in embedded.values()]
    describe = getattr(embedder, "describe", None)
    result = {
        "experiment": EXPERIMENT,
        "started_at": started_at,
        "finished_at": _now(),
        "config": {
            "embedder": describe() if callable(describe) else {"embedder": type(embedder).__name__},
            "query_instruction": instruction,
            "query_format_example": format_query("<query>", instruction),
            "passage_instruction": None,
            "window_lines": WINDOW_LINES,
            "stride": STRIDE,
            "blank_window_rule": "skip windows without a letter or digit",
            "line_semantics": "split on newline, 1-based, like read_file and localize()",
            "similarity": "cosine: dot product of L2-normalized vectors (as T3a), brute force in memory",
            "tie_break": "highest similarity, then earliest start_line",
            "excerpt_budgets": list(EXCERPT_BUDGETS),
            "excerpt_rule": "leading characters of the selected passage (leading whitespace skipped); a cut "
            "inside a word backs up to whitespace if that keeps half the budget; '…' marks a cut",
            "lexical_native_excerpt": "localize()'s own excerpt (≤160 characters, centred on the first match)",
            "document_under_test": "the benchmark's expected document (evaluation setup: isolates localization from retrieval)",
            "survives_definition": "the excerpt holds the whole anchor, or the anchor is longer and the excerpt is entirely anchor text",
            "corpus": corpus_config(root),
        },
        "environment": environment_info(root),
        "manifest": {"path": str(manifest_path), "cases": len(manifest.cases), "scored": len(scored)},
        "drift": [{"id": c.id, "reason": checks[c.id].reason} for c in manifest.cases if checks[c.id].status != "ok"],
        "lexical_reference": reference,
        "corpus": {
            "documents": len(documents),
            "embedded_documents": len(embedded),
            "passages_total": sum(len(doc.passages) for doc in embedded.values()),
            "passages_by_document": {path: len(doc.passages) for path, doc in embedded.items()},
        },
        "metrics": {
            "lexical": summarize(rows, "lexical", categories),
            "semantic": summarize(rows, "semantic", categories),
        },
        "pairwise": {
            "line_overlap": pairwise(rows, "line_overlap"),
            "excerpt_160_survives": pairwise(rows, "excerpt_160_survives"),
        },
        "top3": {
            "first_hit_rank_histogram": _histogram(row["semantic"]["first_hit_rank"] for row in rows),
            "hit_in_top3": sum(1 for row in rows if any(w["overlaps_anchor"] for w in row["semantic"]["top"])),
        },
        "timings": {
            "model_load_s": model_load_s,
            "passages_total": sum(len(doc.passages) for doc in embedded.values()),
            "passage_embed_total_s": sum(doc.embed_s for doc in embedded.values()),
            "passage_embed_ms_per_document": _stats(per_doc_ms),
            "passage_embed_ms_per_passage": sum(per_doc_ms) / max(1, sum(len(d.passages) for d in embedded.values())),
            "warmup_query_ms": warmup_ms,
            "query_embed_ms": _stats(list(query_embed_ms.values())),
            "warm_localization_ms": _stats(warm),
            "cold_document_localization_ms": _stats(cold),
            "lexical_localization_ms": _stats(lexical_latency),
        },
        "memory_peak_rss_mb": memory,
        "cases": rows,
    }

    packs = None
    if artifact is not None:
        packs = _candidate_packs(artifact, pack_cases, embedded, embedder, query_vectors, instruction)
    return result, packs


def check_lexical_reference(
    scored: Sequence[Any], lexical: Mapping[str, Mapping[str, Any]], reference: Mapping[str, tuple[int, int]] | None
) -> dict[str, Any]:
    recomputed: dict[str, list[int]] = {}
    for case in scored:
        count_n = recomputed.setdefault(case.category, [0, 0])
        count_n[1] += 1
        count_n[0] += bool(lexical[case.id]["evaluation"]["line_overlap"])
    observed = {category: tuple(values) for category, values in recomputed.items()}
    if reference is None:
        return {"checked": False, "recomputed": observed}
    mismatches = {
        category: {"expected": list(reference.get(category, (0, 0))), "recomputed": list(observed.get(category, (0, 0)))}
        for category in sorted(set(reference) | set(observed))
        if tuple(reference.get(category, (0, 0))) != tuple(observed.get(category, (0, 0)))
    }
    if mismatches:
        raise LexicalReferenceDrift(
            "lexical localize() evidence coverage differs from the recorded reference; the corpus or "
            f"benchmark drifted: {mismatches}"
        )
    return {"checked": True, "matches": True, "recomputed": {k: list(v) for k, v in observed.items()}}


# --- candidate-pack simulation (not scored) -----------------------------------------------------------


def _pack_cases(artifact) -> list[tuple[Any, tuple, list[str]]]:
    out = []
    for case in artifact.cases:
        candidates = build_candidates(case.lexical, case.semantic)
        out.append((case, candidates, [c.path for c in candidates]))
    return out


def _candidate_packs(artifact, pack_cases, embedded, embedder, query_vectors, instruction) -> dict[str, Any]:
    packs = []
    for case, candidates, _ in pack_cases:
        vector = query_vectors.get(case.query)
        if vector is None:
            vector = embedder.embed(format_query(case.query, instruction))
        entries = []
        for candidate in sorted(candidates, key=lambda c: c.path):
            task = LocalizationTask(case.id, case.query, candidate.path)
            localized = semantic_localization(task, embedded[candidate.path], vector)
            localized.pop("ranking")
            entries.append(
                {
                    "path": candidate.path,
                    "lexical_rank": candidate.lexical_rank,
                    "semantic_rank": candidate.semantic_rank,
                    **{k: localized[k] for k in ("passage", "similarity", "excerpts", "top")},
                }
            )
        packs.append({"id": case.id, "query": case.query, "candidates": entries})
    return {
        "experiment": "T3e candidate-pack simulation: one semantic passage per T3c candidate (not scored)",
        "created_at": _now(),
        "t3a_results": artifact.source,
        "t3a_sha256": artifact.sha256,
        "note": "label-free evidence packs for a possible later experiment; never sent to a judge here",
        "packs": packs,
    }


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "max": max(values)}


def _histogram(values) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in values:
        key = "none" if value is None else str(value) if value <= TOP_K else f">{TOP_K}"
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

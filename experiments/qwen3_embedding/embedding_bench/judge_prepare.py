"""T3d: freeze bounded-judgment cases from a T3a artifact.

Needs ZOMAH (the unchanged KnowledgeIndex corpus and lexical localization), so
it runs in ZOMAH's environment, once. Judges never run this: they read the
frozen cases file it writes.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from zomah.knowledge import LOCALIZE_WINDOW_LINES, MAX_EXCERPT_CHARS, localize

from embedding_bench.candidates import audit_case, build_candidates, classify, size_stats
from embedding_bench.corpus import Document, build_index, load_documents
from embedding_bench.fusion import T3aArtifact
from embedding_bench.judge_contract import (
    CASES_SCHEMA,
    INSTRUCTIONS,
    NONE_DESCRIPTION,
    NONE_LABEL,
    QUESTION_ID,
    render_request,
    sha256_json,
)
from embedding_bench.manifest import load_manifest, normalize_whitespace

BASELINES = ("lexical_top1", "semantic_top1", "rrf_top1", "t3c_order_first", "first_option")


class PrepareError(ValueError):
    """The T3a artifact, corpus, or T3c cross-check does not support a clean T3d run."""


def load_corpus(artifact: T3aArtifact, root: Path, workdir: Path) -> list[Document]:
    """The current corpus, which must still be the one T3a ranked."""

    index, _ = build_index(root, workdir / "knowledge.db")
    documents = load_documents(index, root)
    recorded = {}
    for entry in artifact.data["corpus"]["files"]:
        recorded[entry["path"]] = entry.get("size_bytes")
    current = {doc.path: doc.size_bytes for doc in documents}
    if set(recorded) != set(current):
        raise PrepareError(
            "corpus changed since T3a: "
            f"missing {sorted(set(recorded) - set(current))}, added {sorted(set(current) - set(recorded))}"
        )
    changed = sorted(path for path in current if recorded[path] != current[path])
    if changed:
        raise PrepareError(f"corpus documents changed size since T3a: {changed}")
    return documents


def anchor_line_range(text: str, anchor: str) -> tuple[int, int] | None:
    """Lines (1-based, split on newline like read_file) spanned by a whitespace-normalized anchor."""

    chars: list[str] = []
    line_of: list[int] = []
    previous_space = True
    for number, line in enumerate(text.split("\n"), start=1):
        for char in line + "\n":
            if char.isspace():
                if not previous_space:
                    chars.append(" ")
                    line_of.append(number)
                previous_space = True
            else:
                chars.append(char)
                line_of.append(number)
                previous_space = False
    target = normalize_whitespace(anchor)
    index = "".join(chars).find(target)
    if not target or index < 0:
        return None
    return line_of[index], line_of[index + len(target) - 1]


def build_cases(
    artifact: T3aArtifact,
    documents: list[Document],
    anchors: Mapping[str, str],
    *,
    t3c_results: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    docs = {doc.path: doc for doc in documents}
    t3c_rows = {row["id"]: row for row in t3c_results["cases"]} if t3c_results else None
    cases = []
    for case in artifact.cases:
        candidates = build_candidates(case.lexical, case.semantic)
        resolution = classify(candidates)
        audit = audit_case(case)
        if t3c_rows is not None:
            _cross_check_t3c(case.id, t3c_rows.get(case.id), resolution.kind, candidates)

        # Judge-facing order is by path, so option position carries no retrieval rank.
        ordered = sorted(candidates, key=lambda c: c.path)
        ids = {c.path: f"c{i}" for i, c in enumerate(ordered, start=1)}
        judge_candidates = []
        for candidate in ordered:
            region = localize(docs[candidate.path].text, case.query)
            judge_candidates.append(
                {
                    "id": ids[candidate.path],
                    "path": candidate.path,
                    "lexical_rank": candidate.lexical_rank,
                    "semantic_rank": candidate.semantic_rank,
                    "start_line": region.start_line,
                    "end_line": region.end_line,
                    "excerpt": region.excerpt,
                }
            )
        judge_input = {"query": case.query, "candidates": judge_candidates}
        request = render_request(judge_input)

        # Everything below is evaluation truth: stored beside the request, never sent.
        expected_id = ids.get(case.expected_path)
        anchor_lines = anchor_line_range(docs[case.expected_path].text, anchors[case.id])
        expected_excerpt = next((c for c in judge_candidates if c["id"] == expected_id), None)
        shows_anchor = (
            None
            if expected_excerpt is None or anchor_lines is None
            else expected_excerpt["start_line"] <= anchor_lines[1] and anchor_lines[0] <= expected_excerpt["end_line"]
        )
        diagnostics = audit["diagnostics"]
        cases.append(
            {
                "id": case.id,
                "set": "ambiguous" if resolution.kind == "ambiguous" else "resolved",
                "judge_input": judge_input,
                "judge_input_sha256": sha256_json(judge_input),
                "request": request,
                "request_sha256": sha256_json(request),
                "evaluation": {
                    "expected_path": case.expected_path,
                    "expected_candidate_id": expected_id,
                    "category": case.category,
                    "t3a_agreement": audit["t3a_agreement"],
                    "t3c_resolution": resolution.kind,
                    "t3c_resolved_path": resolution.path,
                    "expected_is": audit["evaluation"]["expected_is"],
                    "baseline_paths": {
                        "lexical_top1": diagnostics["lexical_top1"],
                        "semantic_top1": diagnostics["semantic_top1"],
                        "rrf_top1": diagnostics["rrf_top1"],
                        "t3c_order_first": candidates[0].path,
                        "first_option": ordered[0].path,
                    },
                    "candidate_ids_by_path": ids,
                    "anchor_lines": list(anchor_lines) if anchor_lines else None,
                    "expected_excerpt_lines": (
                        [expected_excerpt["start_line"], expected_excerpt["end_line"]] if expected_excerpt else None
                    ),
                    "expected_excerpt_shows_anchor": shows_anchor,
                },
            }
        )
        _verify_rank_baselines(cases[-1], diagnostics)
    return cases


def prepare_document(
    artifact: T3aArtifact,
    *,
    root: Path,
    manifest_path: Path,
    label_check: Mapping[str, Any],
    workdir: Path,
    t3c_results: Mapping[str, Any] | None = None,
    t3c_path: str | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    documents = load_corpus(artifact, root, workdir)
    anchors = {case.id: case.evidence_anchor for case in load_manifest(manifest_path).cases}
    cases = build_cases(artifact, documents, anchors, t3c_results=t3c_results)
    ambiguous = [c for c in cases if c["set"] == "ambiguous"]

    def correct(members: list[dict[str, Any]], baseline: str) -> int:
        return sum(
            1 for c in members if c["evaluation"]["baseline_paths"][baseline] == c["evaluation"]["expected_path"]
        )

    baselines = {name: correct(ambiguous, name) for name in BASELINES}
    if t3c_results is not None:
        recorded = t3c_results["ambiguous"]["selector_diagnostics"]
        for name in ("lexical_top1", "semantic_top1", "rrf_top1"):
            if recorded[name]["correct"] != baselines[name] or recorded[name]["n"] != len(ambiguous):
                raise PrepareError(
                    f"{name} on the ambiguous cases is {baselines[name]}/{len(ambiguous)}, "
                    f"but T3c recorded {recorded[name]['correct']}/{recorded[name]['n']}"
                )

    return {
        "schema": CASES_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "t3a_results": artifact.source,
            "t3a_sha256": artifact.sha256,
            "t3a_started_at": artifact.data.get("started_at"),
            "t3c_results": t3c_path,
            "manifest": dict(label_check),
            "corpus_root": str(root),
            "zomah_commit": _git_commit(root),
            "documents": [
                {
                    "path": doc.path,
                    "size_bytes": doc.size_bytes,
                    "sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
                }
                for doc in documents
            ],
        },
        "evidence": {
            "candidate_set": "union(lexical top 3, semantic top 3), deduplicated by path (T3c)",
            "excerpt_algorithm": "zomah.knowledge.localize (unchanged T2f lexical localization)",
            "excerpt_max_chars": MAX_EXCERPT_CHARS,
            "localize_window_lines": LOCALIZE_WINDOW_LINES,
            "candidate_order": "path ascending; ids c1..cN assigned in that order",
            "fields_sent": ["query", "id", "path", "lexical_rank", "semantic_rank", "start_line", "end_line", "excerpt"],
            "never_sent": [
                "case id",
                "category",
                "expected path or candidate",
                "evidence anchor",
                "rationale",
                "T3a/T3c outcomes",
                "RRF",
            ],
        },
        "render": {
            "question_id": QUESTION_ID,
            "question_type": "choice",
            "instructions": INSTRUCTIONS,
            "state_template": "Question: {query}",
            "option_template": "{excerpt} [{path}, lines {start_line}-{end_line}; lexical rank {lexical_rank}; "
            "semantic rank {semantic_rank}]",
            "abstention_label": NONE_LABEL,
            "abstention_description": NONE_DESCRIPTION,
        },
        "summary": {
            "cases": len(cases),
            "ambiguous": len(ambiguous),
            "resolved": len(cases) - len(ambiguous),
            "ambiguous_baselines": baselines,
            "baselines_verified_against_t3c": t3c_results is not None,
            "ambiguous_candidate_size": size_stats(len(c["judge_input"]["candidates"]) for c in ambiguous),
            "ambiguous_expected_excerpt_shows_anchor": sum(
                1 for c in ambiguous if c["evaluation"]["expected_excerpt_shows_anchor"]
            ),
        },
        "cases": cases,
    }


def _verify_rank_baselines(case: dict[str, Any], diagnostics: Mapping[str, Any]) -> None:
    """Lexical/semantic #1 recomputed from the judge input must match T3c's diagnostics."""

    by_rank = {
        side: next((c["path"] for c in case["judge_input"]["candidates"] if c[f"{side}_rank"] == 1), None)
        for side in ("lexical", "semantic")
    }
    for side in ("lexical", "semantic"):
        if by_rank[side] != diagnostics[f"{side}_top1"]:
            raise PrepareError(f"case {case['id']!r}: {side} #1 from the judge input disagrees with T3c")


def _cross_check_t3c(case_id: str, row: Mapping[str, Any] | None, kind: str, candidates) -> None:
    if row is None:
        raise PrepareError(f"case {case_id!r} is missing from the T3c results")
    if row.get("resolution") != kind:
        raise PrepareError(f"case {case_id!r}: resolution {kind!r} differs from T3c's {row.get('resolution')!r}")
    if sorted(c["path"] for c in row.get("candidates", [])) != sorted(c.path for c in candidates):
        raise PrepareError(f"case {case_id!r}: candidate set differs from T3c")


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None

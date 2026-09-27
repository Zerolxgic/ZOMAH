"""Benchmark manifest loading, validation, and evidence-anchor drift checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

REQUIRED_FIELDS = ("id", "category", "query", "expected_path", "evidence_anchor")
OPTIONAL_FIELDS = ("rationale",)


class ManifestError(ValueError):
    """The manifest itself is malformed (as opposed to drifted from the corpus)."""


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    category: str
    query: str
    expected_path: str
    evidence_anchor: str
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class Manifest:
    categories: dict[str, str]
    cases: tuple[BenchmarkCase, ...]


@dataclass(frozen=True, slots=True)
class AnchorCheck:
    """Whether a case can be scored against the current corpus.

    ``also_in`` lists other documents that contain the same anchor text; the
    case is still scored, but its single expected document is then a weaker
    label and the report says so.
    """

    status: str  # "ok" | "drift"
    reason: str | None = None
    also_in: tuple[str, ...] = ()


def load_manifest(path: str | Path) -> Manifest:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    return parse_manifest(data)


def parse_manifest(data: Any) -> Manifest:
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")

    categories = data.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ManifestError("manifest needs a non-empty 'categories' object")
    for name, description in categories.items():
        if not _nonblank(name) or not _nonblank(description):
            raise ManifestError("category names and descriptions must be non-empty strings")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestError("manifest needs a non-empty 'cases' list")

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_cases):
        label = f"case {position}"
        if not isinstance(raw, dict):
            raise ManifestError(f"{label}: must be an object")
        unknown = set(raw) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
        if unknown:
            raise ManifestError(f"{label}: unknown fields {sorted(unknown)}")
        for field in REQUIRED_FIELDS:
            if not _nonblank(raw.get(field)):
                raise ManifestError(f"{label}: '{field}' must be a non-empty string")
        rationale = raw.get("rationale")
        if rationale is not None and not _nonblank(rationale):
            raise ManifestError(f"{label}: 'rationale' must be a non-empty string when present")

        case_id = raw["id"]
        label = f"case {case_id!r}"
        if case_id in seen:
            raise ManifestError(f"{label}: duplicate id")
        seen.add(case_id)
        if raw["category"] not in categories:
            raise ManifestError(f"{label}: unknown category {raw['category']!r}")
        _check_relative_path(raw["expected_path"], label)

        cases.append(
            BenchmarkCase(
                id=case_id,
                category=raw["category"],
                query=raw["query"],
                expected_path=raw["expected_path"],
                evidence_anchor=raw["evidence_anchor"],
                rationale=rationale,
            )
        )

    return Manifest(categories=dict(categories), cases=tuple(cases))


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs so re-wrapped Markdown does not count as drift."""

    return " ".join(text.split())


def check_anchors(
    cases: tuple[BenchmarkCase, ...] | list[BenchmarkCase],
    documents: Mapping[str, str],
) -> dict[str, AnchorCheck]:
    """Verify every case's evidence against the corpus text before scoring.

    ``documents`` maps corpus-relative POSIX paths to the exact text both
    retrievers see. Anchors match as whitespace-normalized substrings; wording
    and punctuation must be unchanged.
    """

    normalized = {path: normalize_whitespace(text) for path, text in documents.items()}
    checks: dict[str, AnchorCheck] = {}
    for case in cases:
        anchor = normalize_whitespace(case.evidence_anchor)
        if case.expected_path not in normalized:
            checks[case.id] = AnchorCheck(
                status="drift", reason="expected_path is not in the corpus"
            )
            continue
        if anchor not in normalized[case.expected_path]:
            checks[case.id] = AnchorCheck(
                status="drift", reason="evidence_anchor not found in expected_path"
            )
            continue
        also_in = tuple(
            sorted(
                path
                for path, text in normalized.items()
                if path != case.expected_path and anchor in text
            )
        )
        checks[case.id] = AnchorCheck(status="ok", also_in=also_in)
    return checks


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_relative_path(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or ".." in path.parts or value != path.as_posix():
        raise ManifestError(
            f"{label}: expected_path must be a normalized relative POSIX path, got {value!r}"
        )

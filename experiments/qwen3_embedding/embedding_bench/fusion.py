"""T3b: deterministic Reciprocal Rank Fusion over frozen T3a rankings.

Standard library only: this reads a T3a ``results.json``, never re-embeds, and
never needs ZOMAH, torch, or a model. Fusion sees rank positions only. Raw
BM25 relevance and cosine similarity are never combined, and labels and
categories are used only to score the fused result, never to produce it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from embedding_bench.manifest import ManifestError, load_manifest
from embedding_bench.metrics import (
    AGREEMENT_BUCKETS,
    CaseOutcome,
    agreement_bucket,
    score_case,
    summarize,
    summarize_by_category,
)

RRF_K = 60  # the standard constant; fixed, never tuned against the benchmark labels
SYSTEMS = ("lexical", "semantic", "rrf")
EXPERIMENT = "T3b: equal-weight Reciprocal Rank Fusion (k=60) over frozen T3a rankings"


class FusionInputError(ValueError):
    """The T3a artifact is missing data, malformed, or inconsistent. Nothing was fused."""


# --- fusion -------------------------------------------------------------------------


class Entry(NamedTuple):
    """One position in an input ranking (metrics only need the path)."""

    path: str


@dataclass(frozen=True, slots=True)
class FusedDoc:
    path: str
    score: Fraction  # exact, so equal rank pairs tie exactly rather than by float noise
    lexical_rank: int | None
    semantic_rank: int | None


def rrf_contribution(rank: int | None, k: int = RRF_K) -> Fraction:
    """1 / (k + rank), or 0 when the document is absent from that ranking."""

    if k < 0:
        raise ValueError("k must not be negative")
    if rank is None:
        return Fraction(0)
    if rank < 1:
        raise ValueError("ranks are 1-based")
    return Fraction(1, k + rank)


def rrf_fuse(lexical: Sequence[str], semantic: Sequence[str], k: int = RRF_K) -> tuple[FusedDoc, ...]:
    """Fuse two rankings (paths, best first) with equal-weight RRF.

    Every document in either ranking is scored; ties go to path ascending.
    """

    lexical_ranks = _positions(lexical, "lexical")
    semantic_ranks = _positions(semantic, "semantic")
    fused = [
        FusedDoc(
            path=path,
            score=rrf_contribution(lexical_ranks.get(path), k) + rrf_contribution(semantic_ranks.get(path), k),
            lexical_rank=lexical_ranks.get(path),
            semantic_rank=semantic_ranks.get(path),
        )
        for path in lexical_ranks.keys() | semantic_ranks.keys()
    ]
    return tuple(sorted(fused, key=lambda doc: (-doc.score, doc.path)))


def _positions(ranking: Sequence[str], name: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for rank, path in enumerate(ranking, start=1):
        if path in positions:
            raise ValueError(f"{name} ranking lists {path!r} more than once")
        positions[path] = rank
    return positions


# --- the frozen T3a input ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrozenCase:
    id: str
    category: str
    query: str
    expected_path: str
    lexical: tuple[str, ...]
    semantic: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class T3aArtifact:
    source: str
    sha256: str | None
    data: dict[str, Any]
    categories: dict[str, str]
    corpus: tuple[str, ...]
    cases: tuple[FrozenCase, ...]
    drifted: tuple[str, ...]


def load_t3a(path: str | Path) -> T3aArtifact:
    path = Path(path).expanduser()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FusionInputError(f"cannot read T3a results {path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FusionInputError(f"T3a results {path} is not valid JSON: {exc}") from exc
    return parse_t3a(data, source=str(path), sha256=hashlib.sha256(raw).hexdigest())


def parse_t3a(data: Any, *, source: str = "<memory>", sha256: str | None = None) -> T3aArtifact:
    """Validate the structure and internal consistency of a T3a result.

    Requires every case's full lexical and semantic rankings. The semantic
    ranking must cover the whole corpus (T3a ranks every document); the
    lexical ranking may omit documents that matched no query term. Stored
    ranks, metrics, and agreement buckets must match what the rankings imply.
    """

    if not isinstance(data, dict):
        raise FusionInputError("T3a results must be a JSON object")
    manifest = _object(data, "manifest")
    categories = manifest.get("categories")
    if not isinstance(categories, dict) or not categories or not all(
        isinstance(name, str) for name in categories
    ):
        raise FusionInputError("T3a results lack manifest.categories")
    corpus_files = _object(data, "corpus").get("files")
    if not isinstance(corpus_files, list) or not corpus_files:
        raise FusionInputError("T3a results lack corpus.files")
    corpus = tuple(_string(item, "path", "corpus.files entry") for item in corpus_files)
    if len(set(corpus)) != len(corpus):
        raise FusionInputError("corpus.files lists a document more than once")
    corpus_set = set(corpus)

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise FusionInputError("T3a results contain no scored cases")

    cases: list[FrozenCase] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise FusionInputError(f"case {position} is not an object")
        case_id = _string(raw, "id", f"case {position}")
        label = f"case {case_id!r}"
        if case_id in seen:
            raise FusionInputError(f"{label} appears more than once")
        seen.add(case_id)
        category = _string(raw, "category", label)
        if category not in categories:
            raise FusionInputError(f"{label} has unknown category {category!r}")
        expected = _string(raw, "expected_path", label)
        if expected not in corpus_set:
            raise FusionInputError(f"{label} expects {expected!r}, which is not in corpus.files")

        rankings: dict[str, tuple[str, ...]] = {}
        for side in ("lexical", "semantic"):
            side_data = raw.get(side)
            ranking = side_data.get("ranking") if isinstance(side_data, dict) else None
            if not isinstance(ranking, list):
                raise FusionInputError(
                    f"{label}: the {side} ranking is missing from the T3a results; "
                    "T3b fuses recorded rankings only and does not regenerate them"
                )
            paths = tuple(_string(entry, "path", f"{label} {side} ranking entry") for entry in ranking)
            if len(set(paths)) != len(paths):
                raise FusionInputError(f"{label}: the {side} ranking lists a document more than once")
            unknown = sorted(set(paths) - corpus_set)
            if unknown:
                raise FusionInputError(f"{label}: the {side} ranking has documents outside the corpus: {unknown}")
            stored_rank = side_data.get("rank")
            actual_rank = paths.index(expected) + 1 if expected in paths else None
            if stored_rank != actual_rank:
                raise FusionInputError(
                    f"{label}: stored {side} rank {stored_rank!r} disagrees with its ranking ({actual_rank!r})"
                )
            rankings[side] = paths
        if set(rankings["semantic"]) != corpus_set:
            raise FusionInputError(
                f"{label}: the semantic ranking does not cover the whole corpus "
                f"({len(rankings['semantic'])} of {len(corpus)} documents)"
            )

        cases.append(
            FrozenCase(
                id=case_id,
                category=category,
                query=_string(raw, "query", label),
                expected_path=expected,
                lexical=rankings["lexical"],
                semantic=rankings["semantic"],
            )
        )

    drift = data.get("drift", [])
    if not isinstance(drift, list):
        raise FusionInputError("T3a drift must be a list")
    drifted = tuple(_string(item, "id", "drift entry") for item in drift)

    artifact = T3aArtifact(
        source=source,
        sha256=sha256,
        data=data,
        categories=dict(categories),
        corpus=corpus,
        cases=tuple(cases),
        drifted=drifted,
    )
    _check_stored_results(artifact)
    return artifact


def _check_stored_results(artifact: T3aArtifact) -> None:
    """The T3a metrics and agreement buckets must follow from its own rankings."""

    stored_metrics = artifact.data.get("metrics")
    stored_agreement = artifact.data.get("agreement")
    if not isinstance(stored_metrics, dict) or not isinstance(stored_agreement, dict):
        raise FusionInputError("T3a results lack metrics or agreement")

    outcomes = _input_outcomes(artifact.cases)
    categories = {case.id: case.category for case in artifact.cases}
    for side in ("lexical", "semantic"):
        recomputed = {
            "overall": summarize(outcomes[side].values()),
            "by_category": summarize_by_category(outcomes[side], categories, list(artifact.categories)),
        }
        stored = stored_metrics.get(side)
        if not isinstance(stored, dict) or not _same_metrics(stored.get("overall"), recomputed["overall"]):
            raise FusionInputError(f"stored {side} metrics disagree with the recorded rankings")
        stored_by_category = stored.get("by_category")
        if not isinstance(stored_by_category, dict) or any(
            not _same_metrics(stored_by_category.get(name), value)
            for name, value in recomputed["by_category"].items()
        ):
            raise FusionInputError(f"stored {side} category metrics disagree with the recorded rankings")

    for bucket in AGREEMENT_BUCKETS:
        expected_ids = sorted(
            case.id
            for case in artifact.cases
            if agreement_bucket(outcomes["lexical"][case.id], outcomes["semantic"][case.id]) == bucket
        )
        stored_ids = stored_agreement.get(bucket)
        if not isinstance(stored_ids, list) or sorted(stored_ids) != expected_ids:
            raise FusionInputError(f"stored agreement bucket {bucket!r} disagrees with the recorded rankings")


def check_labels(artifact: T3aArtifact, manifest_path: str | Path) -> dict[str, Any]:
    """Confirm the T3a cases carry the manifest's frozen labels, unchanged.

    Every manifest case must be either scored in T3a with identical category,
    query, and expected_path, or listed as T3a drift. Rationale and anchor text
    may differ (they never affect scoring).
    """

    manifest_path = Path(manifest_path).expanduser()
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ManifestError) as exc:
        raise FusionInputError(f"cannot load benchmark manifest {manifest_path}: {exc}") from exc

    by_id = {case.id: case for case in manifest.cases}
    for case in artifact.cases:
        frozen = by_id.get(case.id)
        if frozen is None:
            raise FusionInputError(f"case {case.id!r} is not in the benchmark manifest")
        changed = [
            field
            for field in ("category", "query", "expected_path")
            if getattr(frozen, field) != getattr(case, field)
        ]
        if changed:
            raise FusionInputError(f"case {case.id!r}: {', '.join(changed)} differ from the benchmark manifest")
    scored = {case.id for case in artifact.cases}
    unscored = sorted(set(by_id) - scored)
    if unscored != sorted(artifact.drifted):
        raise FusionInputError(
            f"manifest cases missing from the T3a results without being recorded as drift: "
            f"{sorted(set(unscored) - set(artifact.drifted))}"
        )
    if list(manifest.categories) != list(artifact.categories):
        raise FusionInputError("manifest categories differ from the T3a results")

    t3a_manifest = _mapping(artifact.data.get("manifest"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha,
        "t3a_manifest_sha256": t3a_manifest.get("sha256"),
        "sha256_matches": manifest_sha == t3a_manifest.get("sha256"),
        "labels_match": True,
        "cases": len(manifest.cases),
        "not_scored_in_t3a": list(artifact.drifted),
    }


# --- evaluation -------------------------------------------------------------------------


def run_fusion(artifact: T3aArtifact, *, label_check: Mapping[str, Any], k: int = RRF_K) -> dict[str, Any]:
    categories = {case.id: case.category for case in artifact.cases}
    category_order = list(artifact.categories)
    outcomes = _input_outcomes(artifact.cases)
    outcomes["rrf"] = {}

    rows = []
    for case in artifact.cases:
        fused = rrf_fuse(case.lexical, case.semantic, k)
        rrf_outcome = score_case(fused, case.expected_path)
        outcomes["rrf"][case.id] = rrf_outcome
        lexical, semantic = outcomes["lexical"][case.id], outcomes["semantic"][case.id]
        expected = fused[rrf_outcome.rank - 1]
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "query": case.query,
                "expected_path": case.expected_path,
                "t3a_agreement": agreement_bucket(lexical, semantic),
                "ranks": {"lexical": lexical.rank, "semantic": semantic.rank, "rrf": rrf_outcome.rank},
                "top1": {"lexical": lexical.top1, "semantic": semantic.top1, "rrf": rrf_outcome.top1},
                "top3": {"lexical": lexical.top3, "semantic": semantic.top3, "rrf": rrf_outcome.top3},
                "rrf_vs_inputs": compare_to_inputs(rrf_outcome.rank, lexical.rank, semantic.rank),
                "rrf_top1_decided_by_path": len(fused) > 1 and fused[0].score == fused[1].score,
                "expected_rank_decided_by_path": any(
                    doc.score == expected.score and doc.path != expected.path for doc in fused
                ),
                "rrf_expected_score": float(expected.score),
                "rrf_ranking": [
                    {
                        "path": doc.path,
                        "rrf": float(doc.score),
                        "lexical_rank": doc.lexical_rank,
                        "semantic_rank": doc.semantic_rank,
                    }
                    for doc in fused
                ],
            }
        )

    by_bucket = {bucket: [row for row in rows if row["t3a_agreement"] == bucket] for bucket in AGREEMENT_BUCKETS}
    oracle = [row["id"] for row in rows if row["top1"]["lexical"] or row["top1"]["semantic"]]
    embedder = _mapping(_mapping(artifact.data.get("config")).get("embedder"))
    return {
        "experiment": EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "path": artifact.source,
            "sha256": artifact.sha256,
            "t3a_experiment": artifact.data.get("experiment"),
            "t3a_started_at": artifact.data.get("started_at"),
            "t3a_embedder": embedder,
            "t3a_zomah_commit": _mapping(artifact.data.get("environment")).get("zomah_commit"),
            "fake_embedder": embedder.get("embedder") == "fake-hashing",
            "corpus_documents": len(artifact.corpus),
        },
        "manifest_check": dict(label_check),
        "config": {
            "method": "equal-weight Reciprocal Rank Fusion",
            "formula": f"rrf(d) = 1/({k} + lexical_rank(d)) + 1/({k} + semantic_rank(d))",
            "k": k,
            "weights": {"lexical": 1, "semantic": 1},
            "missing_rank_contribution": 0,
            "inputs": "rank positions only; raw BM25 relevance and cosine similarity are not used",
            "arithmetic": "exact rationals (fractions.Fraction); reported scores are rounded to float",
            "tie_break": "path ascending",
            "labels_used_for": "scoring only; never for fusion",
        },
        "metrics": {
            system: {
                "overall": summarize(outcomes[system].values()),
                "by_category": summarize_by_category(outcomes[system], categories, category_order),
            }
            for system in SYSTEMS
        },
        "top1_transitions": {
            bucket: {
                "n": len(members),
                "rrf_correct": [row["id"] for row in members if row["top1"]["rrf"]],
                "rrf_wrong": [row["id"] for row in members if not row["top1"]["rrf"]],
            }
            for bucket, members in by_bucket.items()
        },
        "oracle_top1_union": {
            "correct": len(oracle),
            "n": len(rows),
            "rate": len(oracle) / len(rows),
            "note": "either input ranks the expected document first; a reference ceiling, not a system",
        },
        "rank_vs_inputs": {
            outcome: [row["id"] for row in rows if row["rrf_vs_inputs"] == outcome]
            for outcome in ("better_than_both", "matches_best", "between", "worse_than_both")
        },
        "top3_lost": {
            "vs_lexical": [row["id"] for row in rows if row["top3"]["lexical"] and not row["top3"]["rrf"]],
            "vs_semantic": [row["id"] for row in rows if row["top3"]["semantic"] and not row["top3"]["rrf"]],
        },
        "ties": {
            "rrf_top1_decided_by_path": [row["id"] for row in rows if row["rrf_top1_decided_by_path"]],
            "expected_rank_decided_by_path": [row["id"] for row in rows if row["expected_rank_decided_by_path"]],
        },
        "cases": rows,
    }


def compare_to_inputs(rrf_rank: int | None, lexical_rank: int | None, semantic_rank: int | None) -> str:
    """Where the fused rank of the expected document sits relative to both inputs."""

    fused, lexical, semantic = (math.inf if r is None else r for r in (rrf_rank, lexical_rank, semantic_rank))
    best, worst = min(lexical, semantic), max(lexical, semantic)
    if fused < best:
        return "better_than_both"
    if fused == best:
        return "matches_best"
    if fused > worst:
        return "worse_than_both"
    return "between"


def _input_outcomes(cases: Sequence[FrozenCase]) -> dict[str, dict[str, CaseOutcome]]:
    return {
        side: {
            case.id: score_case([Entry(path) for path in getattr(case, side)], case.expected_path)
            for case in cases
        }
        for side in ("lexical", "semantic")
    }


def _same_metrics(stored: Any, recomputed: Mapping[str, Any]) -> bool:
    if not isinstance(stored, dict) or stored.get("n") != recomputed["n"]:
        return False
    for key in ("top1", "top3", "mrr"):
        a, b = stored.get(key), recomputed[key]
        if a is None or b is None:
            if a is not b:
                return False
        elif not isinstance(a, (int, float)) or not math.isclose(a, b, rel_tol=0, abs_tol=1e-9):
            return False
    return True


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _object(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise FusionInputError(f"T3a results lack {key!r}")
    return value


def _string(data: Any, key: str, label: str) -> str:
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise FusionInputError(f"{label} lacks a non-empty {key!r}")
    return value

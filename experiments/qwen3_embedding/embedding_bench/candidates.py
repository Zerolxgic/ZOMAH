"""T3c: deterministic candidate union and ambiguity audit over frozen T3a rankings.

Standard library only. The candidate set and its resolution class are built
from the two recorded rankings alone; the expected label is read only
afterwards, to evaluate. RRF appears only as a separately labelled diagnostic.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from embedding_bench.fusion import Entry, FrozenCase, T3aArtifact, rrf_fuse
from embedding_bench.metrics import AGREEMENT_BUCKETS, agreement_bucket, score_case

TOP_N = 3
EXPERIMENT = "T3c: union(lexical top-3, semantic top-3) candidate recall and deterministic ambiguity audit"
RESOLUTIONS = ("agreement", "dominance", "ambiguous")
OUTCOMES = (
    "resolved_correct",
    "resolved_wrong_in_union",
    "resolved_retrieval_failure",
    "ambiguous_in_union",
    "ambiguous_retrieval_failure",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    path: str
    lexical_rank: int | None  # rank in the full recorded lexical ranking; None = not returned
    semantic_rank: int | None
    in_lexical_top: bool
    in_semantic_top: bool


@dataclass(frozen=True, slots=True)
class Resolution:
    kind: str  # "agreement" | "dominance" | "ambiguous"
    path: str | None


# --- label-free construction and classification ------------------------------------------


def build_candidates(lexical: Sequence[str], semantic: Sequence[str], n: int = TOP_N) -> tuple[Candidate, ...]:
    """union(lexical top-n, semantic top-n), deduplicated by path.

    Each candidate keeps its rank in both full rankings (None when a ranking
    does not contain it). Order is symmetric in the two retrievers: best rank,
    then worse rank, then path.
    """

    if n < 1:
        raise ValueError("n must be positive")
    lexical_ranks = _ranks(lexical, "lexical")
    semantic_ranks = _ranks(semantic, "semantic")
    paths = dict.fromkeys([*lexical[:n], *semantic[:n]])
    candidates = [
        Candidate(
            path=path,
            lexical_rank=lexical_ranks.get(path),
            semantic_rank=semantic_ranks.get(path),
            in_lexical_top=lexical_ranks.get(path, n + 1) <= n,
            in_semantic_top=semantic_ranks.get(path, n + 1) <= n,
        )
        for path in paths
    ]
    return tuple(sorted(candidates, key=_order_key))


def dominates(a: Candidate, b: Candidate) -> bool:
    """a is ranked at least as well as b on both rankings and strictly better on one.

    A document missing from a ranking counts as worse than any reported rank.
    """

    pairs = ((_value(a.lexical_rank), _value(b.lexical_rank)), (_value(a.semantic_rank), _value(b.semantic_rank)))
    return all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)


def pareto_front(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Candidates no other candidate dominates (a diagnostic, not a resolver)."""

    return tuple(c for c in candidates if not any(dominates(o, c) for o in candidates if o is not c))


def classify(candidates: Sequence[Candidate]) -> Resolution:
    """Resolve a candidate set without labels, or declare it ambiguous.

    AGREEMENT: both rankings put the same document first.
    DOMINANCE: otherwise, exactly one candidate dominates every other candidate.
    AMBIGUOUS: neither rule selects a unique document.
    """

    for candidate in candidates:
        if candidate.lexical_rank == 1 and candidate.semantic_rank == 1:
            return Resolution("agreement", candidate.path)
    dominators = [
        c for c in candidates if all(dominates(c, other) for other in candidates if other is not c)
    ]
    if len(candidates) > 0 and len(dominators) == 1:
        return Resolution("dominance", dominators[0].path)
    return Resolution("ambiguous", None)


# --- evaluation --------------------------------------------------------------------------------


def audit_case(case: FrozenCase, n: int = TOP_N) -> dict[str, Any]:
    candidates = build_candidates(case.lexical, case.semantic, n)
    resolution = classify(candidates)
    front = pareto_front(candidates)

    # Everything below reads the label, for evaluation only.
    expected = next((c for c in candidates if c.path == case.expected_path), None)
    in_union = expected is not None
    if resolution.kind == "ambiguous":
        outcome = "ambiguous_in_union" if in_union else "ambiguous_retrieval_failure"
    elif resolution.path == case.expected_path:
        outcome = "resolved_correct"
    else:
        outcome = "resolved_wrong_in_union" if in_union else "resolved_retrieval_failure"

    lexical_top1 = case.lexical[0] if case.lexical else None
    semantic_top1 = case.semantic[0] if case.semantic else None
    rrf_top1 = rrf_fuse(case.lexical, case.semantic)[0].path
    lexical_outcome = score_case([Entry(p) for p in case.lexical], case.expected_path)
    semantic_outcome = score_case([Entry(p) for p in case.semantic], case.expected_path)
    return {
        "id": case.id,
        "category": case.category,
        "query": case.query,
        "expected_path": case.expected_path,
        "t3a_agreement": agreement_bucket(lexical_outcome, semantic_outcome),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "path": c.path,
                "lexical_rank": c.lexical_rank,
                "semantic_rank": c.semantic_rank,
                "in_lexical_top": c.in_lexical_top,
                "in_semantic_top": c.in_semantic_top,
                "on_pareto_front": c in front,
            }
            for c in candidates
        ],
        "resolution": resolution.kind,
        "resolved_path": resolution.path,
        "pareto_front_size": len(front),
        "evaluation": {
            "outcome": outcome,
            "expected_in_lexical_top": expected is not None and expected.in_lexical_top,
            "expected_in_semantic_top": expected is not None and expected.in_semantic_top,
            "expected_in_union": in_union,
            "expected_on_pareto_front": expected is not None and expected in front,
            "expected_ranks": {"lexical": lexical_outcome.rank, "semantic": semantic_outcome.rank},
            "expected_is": _expected_role(case.expected_path, lexical_top1, semantic_top1, in_union),
        },
        "diagnostics": {
            "lexical_top1": lexical_top1,
            "semantic_top1": semantic_top1,
            "rrf_top1": rrf_top1,
            "lexical_top1_correct": lexical_top1 == case.expected_path,
            "semantic_top1_correct": semantic_top1 == case.expected_path,
            "rrf_top1_correct": rrf_top1 == case.expected_path,
        },
    }


def run_audit(artifact: T3aArtifact, *, label_check: Mapping[str, Any], n: int = TOP_N) -> dict[str, Any]:
    rows = [audit_case(case, n) for case in artifact.cases]
    category_order = list(artifact.categories)
    ambiguous = [r for r in rows if r["resolution"] == "ambiguous"]
    resolved = [r for r in rows if r["resolution"] != "ambiguous"]
    outcome_counts = {name: [r["id"] for r in rows if r["evaluation"]["outcome"] == name] for name in OUTCOMES}
    resolved_correct = len(outcome_counts["resolved_correct"])
    judge_addressable = len(outcome_counts["ambiguous_in_union"])

    def selector_accuracy(members: list[dict[str, Any]], key: str) -> dict[str, Any]:
        correct = [r["id"] for r in members if r["diagnostics"][key]]
        return {"correct": len(correct), "n": len(members), "rate": _rate(len(correct), len(members))}

    return {
        "experiment": EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": _input_block(artifact),
        "manifest_check": dict(label_check),
        "config": {
            "candidate_set": f"union(lexical top-{n}, semantic top-{n}), deduplicated by path",
            "top_n": n,
            "candidate_ranks": "rank in each full recorded ranking; missing = worse than any rank",
            "candidate_order": "best rank, then worse rank, then path (symmetric in the two retrievers)",
            "agreement": "lexical #1 and semantic #1 are the same document",
            "dominance": (
                "exactly one candidate is ranked at least as well as every other candidate on both "
                "rankings and strictly better on at least one"
            ),
            "ambiguous": "neither agreement nor unique dominance",
            "labels_used_for": "evaluation only; never for building or classifying candidate sets",
            "rrf": "k=60 equal-weight RRF, reported as a separate diagnostic, never as the resolver",
        },
        "candidate_recall": {
            "overall": _recall(rows),
            "by_category": {name: _recall([r for r in rows if r["category"] == name]) for name in category_order},
        },
        "candidate_size": {
            **size_stats(r["candidate_count"] for r in rows),
            "by_category": {
                name: size_stats(r["candidate_count"] for r in rows if r["category"] == name)
                for name in category_order
            },
        },
        "resolution": {
            "counts": {kind: [r["id"] for r in rows if r["resolution"] == kind] for kind in RESOLUTIONS},
            "by_category": {
                name: {kind: sum(1 for r in rows if r["category"] == name and r["resolution"] == kind) for kind in RESOLUTIONS}
                for name in category_order
            },
            "resolved_accuracy": {
                "correct": resolved_correct,
                "n": len(resolved),
                "rate": _rate(resolved_correct, len(resolved)),
                "by_kind": {
                    kind: {
                        "correct": sum(
                            1 for r in rows if r["resolution"] == kind and r["evaluation"]["outcome"] == "resolved_correct"
                        ),
                        "n": sum(1 for r in rows if r["resolution"] == kind),
                    }
                    for kind in ("agreement", "dominance")
                },
            },
        },
        "outcomes": outcome_counts,
        "retrieval_failures": [
            r["id"] for r in rows if not r["evaluation"]["expected_in_union"]
        ],
        "ambiguous": {
            "n": len(ambiguous),
            "expected_in_union": judge_addressable,
            "expected_is": {
                role: [r["id"] for r in ambiguous if r["evaluation"]["expected_is"] == role]
                for role in ("lexical_top1", "semantic_top1", "other_candidate", "not_in_union")
            },
            "candidate_size": size_stats(r["candidate_count"] for r in ambiguous),
            "pareto_front_size": size_stats(r["pareto_front_size"] for r in ambiguous),
            "expected_on_pareto_front": sum(1 for r in ambiguous if r["evaluation"]["expected_on_pareto_front"]),
            "selector_diagnostics": {
                "lexical_top1": selector_accuracy(ambiguous, "lexical_top1_correct"),
                "semantic_top1": selector_accuracy(ambiguous, "semantic_top1_correct"),
                "rrf_top1": selector_accuracy(ambiguous, "rrf_top1_correct"),
            },
        },
        "t3a_wins_in_union": {
            bucket: {
                "n": sum(1 for r in rows if r["t3a_agreement"] == bucket),
                "in_union": sum(1 for r in rows if r["t3a_agreement"] == bucket and r["evaluation"]["expected_in_union"]),
                "resolution": {
                    kind: [r["id"] for r in rows if r["t3a_agreement"] == bucket and r["resolution"] == kind]
                    for kind in RESOLUTIONS
                },
            }
            for bucket in AGREEMENT_BUCKETS
        },
        "end_to_end_references": {
            "n": len(rows),
            "deterministic_plus_perfect_judge": resolved_correct + judge_addressable,
            "deterministic_plus_lexical_top1": resolved_correct
            + selector_accuracy(ambiguous, "lexical_top1_correct")["correct"],
            "deterministic_plus_semantic_top1": resolved_correct
            + selector_accuracy(ambiguous, "semantic_top1_correct")["correct"],
            "rrf_top1": sum(1 for r in rows if r["diagnostics"]["rrf_top1_correct"]),
            "note": "reference counts for evaluation; only the deterministic resolutions are a system",
        },
        "cases": rows,
    }


def size_stats(sizes: Iterable[int]) -> dict[str, Any]:
    values = sorted(sizes)
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "histogram": {}}
    histogram: dict[str, int] = {}
    for value in values:
        histogram[str(value)] = histogram.get(str(value), 0) + 1
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "histogram": histogram,
    }


def _recall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = {
        key: sum(1 for r in rows if r["evaluation"][flag])
        for key, flag in (
            ("lexical_top", "expected_in_lexical_top"),
            ("semantic_top", "expected_in_semantic_top"),
            ("union", "expected_in_union"),
        )
    }
    return {"n": n, **{key: {"count": count, "rate": _rate(count, n)} for key, count in counts.items()}}


def _expected_role(expected: str, lexical_top1: str | None, semantic_top1: str | None, in_union: bool) -> str:
    if not in_union:
        return "not_in_union"
    if expected == lexical_top1 and expected == semantic_top1:
        return "both_top1"
    if expected == lexical_top1:
        return "lexical_top1"
    if expected == semantic_top1:
        return "semantic_top1"
    return "other_candidate"


def _input_block(artifact: T3aArtifact) -> dict[str, Any]:
    config = artifact.data.get("config") if isinstance(artifact.data.get("config"), dict) else {}
    embedder = config.get("embedder") if isinstance(config.get("embedder"), dict) else {}
    environment = artifact.data.get("environment") if isinstance(artifact.data.get("environment"), dict) else {}
    return {
        "path": artifact.source,
        "sha256": artifact.sha256,
        "t3a_started_at": artifact.data.get("started_at"),
        "t3a_embedder": embedder,
        "t3a_zomah_commit": environment.get("zomah_commit"),
        "fake_embedder": embedder.get("embedder") == "fake-hashing",
        "corpus_documents": len(artifact.corpus),
    }


def _ranks(ranking: Sequence[str], name: str) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, path in enumerate(ranking, start=1):
        if path in ranks:
            raise ValueError(f"{name} ranking lists {path!r} more than once")
        ranks[path] = rank
    return ranks


def _value(rank: int | None) -> float:
    return math.inf if rank is None else rank


def _order_key(candidate: Candidate) -> tuple[float, float, str]:
    ranks = (_value(candidate.lexical_rank), _value(candidate.semantic_rank))
    return (min(ranks), max(ranks), candidate.path)


def _rate(count: int, n: int) -> float | None:
    return count / n if n else None

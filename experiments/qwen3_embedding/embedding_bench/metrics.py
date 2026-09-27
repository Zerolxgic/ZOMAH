"""Ranking metrics and lexical-vs-semantic agreement."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from embedding_bench.retrieval import RankedDoc

AGREEMENT_BUCKETS = (
    "both_correct",
    "lexical_correct_semantic_wrong",
    "semantic_correct_lexical_wrong",
    "both_wrong",
)


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    rank: int | None  # 1-based; None when the expected document was not returned
    top1: bool
    top3: bool
    reciprocal_rank: float


def top_k(ranking: Sequence[RankedDoc], k: int) -> tuple[RankedDoc, ...]:
    if k < 1:
        raise ValueError("k must be positive")
    return tuple(ranking[:k])


def rank_of(ranking: Sequence[RankedDoc], path: str) -> int | None:
    for position, doc in enumerate(ranking, start=1):
        if doc.path == path:
            return position
    return None


def score_case(ranking: Sequence[RankedDoc], expected_path: str) -> CaseOutcome:
    rank = rank_of(ranking, expected_path)
    return CaseOutcome(
        rank=rank,
        top1=rank == 1,
        top3=rank is not None and rank <= 3,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
    )


def summarize(outcomes: Iterable[CaseOutcome]) -> dict[str, float | int | None]:
    """Top-1 accuracy, top-3 recall, and MRR (a missing document scores 0)."""

    outcomes = list(outcomes)
    n = len(outcomes)
    if n == 0:
        return {"n": 0, "top1": None, "top3": None, "mrr": None}
    return {
        "n": n,
        "top1": sum(o.top1 for o in outcomes) / n,
        "top3": sum(o.top3 for o in outcomes) / n,
        "mrr": sum(o.reciprocal_rank for o in outcomes) / n,
    }


def summarize_by_category(
    outcomes: Mapping[str, CaseOutcome],
    categories: Mapping[str, str],
    category_order: Sequence[str],
) -> dict[str, dict[str, float | int | None]]:
    """``outcomes`` and ``categories`` are keyed by case id."""

    return {
        category: summarize(
            outcome for case_id, outcome in outcomes.items() if categories[case_id] == category
        )
        for category in category_order
    }


def agreement_bucket(lexical: CaseOutcome, semantic: CaseOutcome) -> str:
    """Top-1 agreement: 'correct' means the expected document ranked first."""

    if lexical.top1 and semantic.top1:
        return "both_correct"
    if lexical.top1:
        return "lexical_correct_semantic_wrong"
    if semantic.top1:
        return "semantic_correct_lexical_wrong"
    return "both_wrong"


def latency_summary(values_ms: Sequence[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values_ms),
        "median": statistics.median(values_ms),
        "mean": statistics.fmean(values_ms),
        "min": min(values_ms),
        "max": max(values_ms),
    }

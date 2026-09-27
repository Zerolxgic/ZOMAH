"""Shared builders for synthetic, internally consistent T3a results.json data."""

from __future__ import annotations

import json
from pathlib import Path

from embedding_bench.fusion import Entry
from embedding_bench.metrics import AGREEMENT_BUCKETS, agreement_bucket, score_case, summarize, summarize_by_category

CORPUS = ("a.md", "b.md", "c.md", "d.md", "e.md")
CATEGORIES = {"exact": "source terms", "paraphrase": "reworded"}


def t3a_artifact(scenarios, *, drift=()) -> dict:
    """A minimal T3a results.json whose stored ranks, metrics and buckets are consistent."""

    cases, outcomes = [], {"lexical": {}, "semantic": {}}
    for case_id, category, expected, lexical, semantic in scenarios:
        case = {"id": case_id, "category": category, "query": f"query {case_id}", "expected_path": expected}
        for side, ranking in (("lexical", lexical), ("semantic", semantic)):
            outcome = score_case([Entry(p) for p in ranking], expected)
            outcomes[side][case_id] = outcome
            case[side] = {
                "rank": outcome.rank,
                # Raw scores deliberately on unrelated scales: fusion must ignore them.
                "ranking": [
                    {"path": p, "score": (1000.0 if side == "lexical" else 0.9) - i} for i, p in enumerate(ranking)
                ],
            }
        cases.append(case)
    categories = {c["id"]: c["category"] for c in cases}
    agreement = {bucket: [] for bucket in AGREEMENT_BUCKETS}
    for case in cases:
        bucket = agreement_bucket(outcomes["lexical"][case["id"]], outcomes["semantic"][case["id"]])
        agreement[bucket].append(case["id"])
    return {
        "experiment": "T3a test fixture",
        "started_at": "2026-09-27T00:00:00+00:00",
        "config": {"embedder": {"embedder": "synonym-fake"}},
        "environment": {"zomah_commit": "0123456789abcdef"},
        "manifest": {"sha256": "not-the-manifest", "categories": CATEGORIES, "cases": len(cases) + len(drift)},
        "corpus": {"files": [{"path": p} for p in CORPUS]},
        "drift": [{"id": d} for d in drift],
        "metrics": {
            side: {
                "overall": summarize(outcomes[side].values()),
                "by_category": summarize_by_category(outcomes[side], categories, list(CATEGORIES)),
            }
            for side in ("lexical", "semantic")
        },
        "agreement": agreement,
        "cases": cases,
    }


def write_manifest(path: Path, artifact: dict, *, extra_cases=()) -> Path:
    cases = [
        {
            "id": c["id"],
            "category": c["category"],
            "query": c["query"],
            "expected_path": c["expected_path"],
            "evidence_anchor": "anchor",
        }
        for c in artifact["cases"]
    ]
    path.write_text(json.dumps({"categories": CATEGORIES, "cases": cases + list(extra_cases)}), encoding="utf-8")
    return path

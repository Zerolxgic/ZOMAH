"""T3d: score independent judge runs against frozen cases (standard library only).

Primary metrics use the AMBIGUOUS cases only. RESOLVED (agreement) cases are a
separate diagnostic. Judges are compared side by side; nothing here combines
them into an ensemble.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from embedding_bench.judge_contract import (
    JUDGMENTS_SCHEMA,
    JudgeContractError,
    candidate_ids,
    normalize_answer,
)

BASELINES = ("lexical_top1", "semantic_top1", "rrf_top1", "t3c_order_first", "first_option")
BASELINE_LABELS = {
    "lexical_top1": "lexical #1",
    "semantic_top1": "semantic #1",
    "rrf_top1": "RRF #1",
    "t3c_order_first": "T3c candidate order, first",
    "first_option": "first listed option (c1)",
}
EXPECTED_ROLES = ("lexical_top1", "semantic_top1", "other_candidate")


def load_judgments(path: str, cases_doc: Mapping[str, Any], cases_sha256: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise JudgeContractError(f"cannot read judgments {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != JUDGMENTS_SCHEMA:
        raise JudgeContractError(f"{path} is not a T3d judgments file")
    if doc.get("cases_sha256") != cases_sha256:
        raise JudgeContractError(f"{path} was produced from a different cases file")
    records = doc.get("records")
    if not isinstance(records, list):
        raise JudgeContractError(f"{path} has no records")
    by_id = {}
    for record in records:
        if not isinstance(record, dict) or record.get("id") in by_id:
            raise JudgeContractError(f"{path} has a malformed or duplicate record")
        by_id[record["id"]] = record
    for case in cases_doc["cases"]:
        record = by_id.get(case["id"])
        if record is None:
            raise JudgeContractError(f"{path} has no judgment for case {case['id']!r}")
        if record.get("request_sha256") != case["request_sha256"]:
            raise JudgeContractError(f"{path}: case {case['id']!r} was judged on a different request")
    extra = sorted(set(by_id) - {case["id"] for case in cases_doc["cases"]})
    if extra:
        raise JudgeContractError(f"{path} has judgments for unknown cases {extra}")
    return {**doc, "by_id": by_id}


def compare(
    cases_doc: Mapping[str, Any], cases_sha256: str, judgments: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    names = list(judgments)
    rows = []
    for case in cases_doc["cases"]:
        evaluation = case["evaluation"]
        paths = {c["id"]: c["path"] for c in case["judge_input"]["candidates"]}
        row: dict[str, Any] = {
            "id": case["id"],
            "set": case["set"],
            "category": evaluation["category"],
            "query": case["judge_input"]["query"],
            "expected_path": evaluation["expected_path"],
            "expected_candidate_id": evaluation["expected_candidate_id"],
            "expected_is": evaluation["expected_is"],
            "t3a_agreement": evaluation["t3a_agreement"],
            "t3c_resolved_path": evaluation["t3c_resolved_path"],
            "expected_excerpt_shows_anchor": evaluation["expected_excerpt_shows_anchor"],
            "candidates": case["judge_input"]["candidates"],
            "baselines": {
                name: {
                    "path": evaluation["baseline_paths"][name],
                    "correct": evaluation["baseline_paths"][name] == evaluation["expected_path"],
                }
                for name in BASELINES
            },
            "judges": {},
        }
        for name in names:
            record = judgments[name]["by_id"][case["id"]]
            normalized = normalize_answer(record.get("answer"), candidate_ids(case["request"]), error=record.get("error"))
            selected_path = paths.get(normalized.selected_candidate) if normalized.selected_candidate else None
            row["judges"][name] = {
                **normalized.to_dict(),
                "selected_path": selected_path,
                "correct": selected_path is not None and selected_path == evaluation["expected_path"],
                "latency_ms": record.get("latency_ms"),
                "diagnostics": record.get("diagnostics"),
            }
        rows.append(row)

    ambiguous = [r for r in rows if r["set"] == "ambiguous"]
    resolved = [r for r in rows if r["set"] == "resolved"]
    categories = list(dict.fromkeys(r["category"] for r in rows))
    result: dict[str, Any] = {
        "experiment": "T3d: Jev vs Laya on the bounded retrieval-selection job exposed by T3c",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases_sha256": cases_sha256,
        "source": cases_doc.get("source"),
        "evidence": cases_doc.get("evidence"),
        "render": cases_doc.get("render"),
        "judges": {
            name: {
                "config": judgments[name].get("config"),
                "environment": judgments[name].get("environment"),
                "created_at": judgments[name].get("created_at"),
                "warmup_ms": judgments[name].get("warmup_ms"),
                "load_s": judgments[name].get("load_s"),
            }
            for name in names
        },
        "counts": {"ambiguous": len(ambiguous), "resolved": len(resolved)},
        "baselines": {name: _selector_metrics(ambiguous, lambda r, n=name: r["baselines"][n]["correct"], categories) for name in BASELINES},
        "primary": {name: judge_metrics(ambiguous, name, categories) for name in names},
        "agreement_diagnostics": {name: agreement_diagnostics(resolved, name) for name in names},
        "evidence_pack": evidence_pack(ambiguous, names),
        "end_to_end_reference": end_to_end(rows, names),
        "cases": rows,
    }
    if len(names) == 2:
        result["pairwise"] = pairwise(ambiguous, names[0], names[1])
    return result


def judge_metrics(rows: Sequence[Mapping[str, Any]], name: str, categories: Sequence[str]) -> dict[str, Any]:
    judged = [r["judges"][name] for r in rows]
    selected = [j for j in judged if j["status"] == "selected"]
    correct_ids = [r["id"] for r in rows if r["judges"][name]["correct"]]
    latencies = [j["latency_ms"] for j in judged if isinstance(j["latency_ms"], (int, float))]
    by_role = {
        role: {
            "n": sum(1 for r in rows if r["expected_is"] == role),
            "correct": [r["id"] for r in rows if r["expected_is"] == role and r["judges"][name]["correct"]],
        }
        for role in EXPECTED_ROLES
    }
    return {
        "n": len(rows),
        "correct": len(correct_ids),
        "accuracy": _rate(len(correct_ids), len(rows)),
        "correct_ids": correct_ids,
        "abstained": [r["id"] for r in rows if r["judges"][name]["status"] == "abstained"],
        "protocol_failures": [
            {"id": r["id"], "error": r["judges"][name]["protocol_error"]}
            for r in rows
            if r["judges"][name]["status"] == "protocol_failure"
        ],
        "selected": len(selected),
        "accuracy_when_selecting": _rate(len(correct_ids), len(selected)),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
        },
        "by_category": {
            category: {
                "n": sum(1 for r in rows if r["category"] == category),
                "correct": sum(1 for r in rows if r["category"] == category and r["judges"][name]["correct"]),
            }
            for category in categories
        },
        "by_expected_role": by_role,
        "vs_rrf": {
            "improves": [r["id"] for r in rows if r["judges"][name]["correct"] and not r["baselines"]["rrf_top1"]["correct"]],
            "worsens": [r["id"] for r in rows if not r["judges"][name]["correct"] and r["baselines"]["rrf_top1"]["correct"]],
        },
        "selected_probability": {
            "correct": _mean(j["selected_probability"] for j in selected if j["correct"]),
            "wrong": _mean(j["selected_probability"] for j in selected if not j["correct"]),
        },
    }


def agreement_diagnostics(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    right = [r for r in rows if r["t3c_resolved_path"] == r["expected_path"]]
    wrong = [r for r in rows if r["t3c_resolved_path"] != r["expected_path"]]
    return {
        "n": len(rows),
        "agrees_with_retrievers": sum(1 for r in rows if r["judges"][name]["selected_path"] == r["t3c_resolved_path"]),
        "wrong_agreements": [
            {"id": r["id"], "judge_correct": r["judges"][name]["correct"], "judge_status": r["judges"][name]["status"]}
            for r in wrong
        ],
        "correct_agreements": len(right),
        "damaged": [
            {"id": r["id"], "status": r["judges"][name]["status"], "selected_path": r["judges"][name]["selected_path"]}
            for r in right
            if r["judges"][name]["selected_path"] != r["t3c_resolved_path"]
        ],
    }


def pairwise(rows: Sequence[Mapping[str, Any]], a: str, b: str) -> dict[str, Any]:
    def ids(predicate) -> list[str]:
        return [r["id"] for r in rows if predicate(r["judges"][a], r["judges"][b])]

    return {
        "judges": [a, b],
        "both_correct": ids(lambda x, y: x["correct"] and y["correct"]),
        f"{a}_only_correct": ids(lambda x, y: x["correct"] and not y["correct"]),
        f"{b}_only_correct": ids(lambda x, y: y["correct"] and not x["correct"]),
        "both_wrong": ids(lambda x, y: not x["correct"] and not y["correct"]),
        f"{a}_abstains_{b}_selects": ids(lambda x, y: x["status"] == "abstained" and y["status"] == "selected"),
        f"{b}_abstains_{a}_selects": ids(lambda x, y: y["status"] == "abstained" and x["status"] == "selected"),
        "both_abstain": ids(lambda x, y: x["status"] == "abstained" and y["status"] == "abstained"),
        "same_selection": ids(lambda x, y: x["status"] == "selected" and x["selected_path"] == y["selected_path"]),
    }


def evidence_pack(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    hidden = [r for r in rows if r["expected_excerpt_shows_anchor"] is False]
    shown = [r for r in rows if r["expected_excerpt_shows_anchor"]]
    all_wrong = [r for r in rows if names and all(not r["judges"][n]["correct"] for n in names)]
    result: dict[str, Any] = {
        "expected_excerpt_misses_anchor": [r["id"] for r in hidden],
        "accuracy_when_anchor_shown": {n: _rate(sum(r["judges"][n]["correct"] for r in shown), len(shown)) for n in names},
        "accuracy_when_anchor_missed": {n: _rate(sum(r["judges"][n]["correct"] for r in hidden), len(hidden)) for n in names},
        "all_judges_wrong_anchor_missed": [r["id"] for r in all_wrong if r["expected_excerpt_shows_anchor"] is False],
        "all_judges_wrong_anchor_shown": [r["id"] for r in all_wrong if r["expected_excerpt_shows_anchor"]],
    }
    for name in names:
        truncation = [r["judges"][name]["diagnostics"] for r in rows]
        budgets = [d for d in truncation if isinstance(d, dict) and d.get("available")]
        if budgets:
            expected_cut = []
            for r in rows:
                d = r["judges"][name]["diagnostics"]
                if isinstance(d, dict) and d.get("available") and r["expected_candidate_id"] in d.get("options_truncated", []):
                    expected_cut.append(r["id"])
            result[f"{name}_token_budget"] = {
                "cases_with_truncated_options": sum(1 for d in budgets if d.get("options_truncated")),
                "cases_with_truncated_state": sum(1 for d in budgets if d["state_tokens_kept"] < d["state_tokens_full"]),
                "expected_option_truncated": expected_cut,
                "cases": len(budgets),
            }
    return result


def end_to_end(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    resolved_correct = sum(1 for r in rows if r["set"] == "resolved" and r["t3c_resolved_path"] == r["expected_path"])
    return {
        "n": len(rows),
        "resolved_correct": resolved_correct,
        "rrf_top1": sum(1 for r in rows if r["baselines"]["rrf_top1"]["correct"]),
        "routing_plus_judge": {
            n: resolved_correct + sum(1 for r in rows if r["set"] == "ambiguous" and r["judges"][n]["correct"])
            for n in names
        },
        "judge_everywhere": {n: sum(1 for r in rows if r["judges"][n]["correct"]) for n in names},
        "note": "references only: deterministic agreement routing plus one judge on ambiguous cases",
    }


def _selector_metrics(rows, is_correct, categories) -> dict[str, Any]:
    correct = [r["id"] for r in rows if is_correct(r)]
    return {
        "n": len(rows),
        "correct": len(correct),
        "accuracy": _rate(len(correct), len(rows)),
        "correct_ids": correct,
        "by_category": {
            c: {"n": sum(1 for r in rows if r["category"] == c), "correct": sum(1 for r in rows if r["category"] == c and is_correct(r))}
            for c in categories
        },
    }


def _rate(count: int, n: int) -> float | None:
    return count / n if n else None


def _mean(values) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(values) if values else None

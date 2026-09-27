"""T3d: the provider-neutral bounded-judgment contract shared by every judge.

Standard library only. One judge input per case (query + candidates with
deterministic evidence) is rendered once into one request: a plain-text state
and a single choice question. Jev and Laya both accept exactly this
`{"type": "choice", "instructions", "criteria"}` shape, so each judge receives
byte-identical input. Labels never enter the judge input or the request.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

CASES_SCHEMA = "zomah.t3d.cases/1"
JUDGMENTS_SCHEMA = "zomah.t3d.judgments/1"

QUESTION_ID = "evidence_source"
NONE_LABEL = "none"
# Short on purpose: Laya's stock checkpoint truncates the instruction when many options share
# its option budget, so the whole question must fit in about 16 tokens.
INSTRUCTIONS = "Which candidate contains the evidence that answers the question?"
NONE_DESCRIPTION = "None of the candidates contains the needed evidence."
CANDIDATE_FIELDS = ("id", "path", "lexical_rank", "semantic_rank", "start_line", "end_line", "excerpt")

# A fixed request that is not a benchmark case, used once per run to warm the judge up.
WARMUP_REQUEST = {
    "state": "Question: Which document explains how to water tomato plants?",
    "questions": {
        QUESTION_ID: {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {
                "c1": "Water tomatoes deeply twice a week. [garden.md, lines 3-3; lexical rank 1; semantic rank 1]",
                NONE_LABEL: NONE_DESCRIPTION,
            },
        }
    },
}


class JudgeContractError(ValueError):
    """A cases or judgments file does not follow the T3d contract."""


# --- rendering ------------------------------------------------------------------------


def describe_candidate(candidate: Mapping[str, Any]) -> str:
    """Evidence first, provenance after: a truncating judge keeps the evidence."""

    lexical = candidate["lexical_rank"] if candidate["lexical_rank"] is not None else "absent"
    semantic = candidate["semantic_rank"] if candidate["semantic_rank"] is not None else "absent"
    return (
        f"{candidate['excerpt']} [{candidate['path']}, lines {candidate['start_line']}-{candidate['end_line']}; "
        f"lexical rank {lexical}; semantic rank {semantic}]"
    )


def render_request(judge_input: Mapping[str, Any]) -> dict[str, Any]:
    """The one request every judge receives for a case."""

    validate_judge_input(judge_input)
    criteria = {c["id"]: describe_candidate(c) for c in judge_input["candidates"]}
    criteria[NONE_LABEL] = NONE_DESCRIPTION
    return {
        "state": f"Question: {judge_input['query']}",
        "questions": {QUESTION_ID: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
    }


def validate_judge_input(judge_input: Mapping[str, Any]) -> None:
    if set(judge_input) != {"query", "candidates"}:
        raise JudgeContractError(f"judge input must hold exactly query and candidates, got {sorted(judge_input)}")
    if not isinstance(judge_input["query"], str) or not judge_input["query"].strip():
        raise JudgeContractError("judge input query must be a non-empty string")
    candidates = judge_input["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise JudgeContractError("judge input needs at least one candidate")
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or tuple(candidate) != CANDIDATE_FIELDS:
            raise JudgeContractError(f"candidates must have exactly the fields {CANDIDATE_FIELDS}")
        ids.append(candidate["id"])
    if len(set(ids)) != len(ids) or NONE_LABEL in ids:
        raise JudgeContractError("candidate ids must be unique and must not be the abstention label")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def candidate_ids(request: Mapping[str, Any]) -> list[str]:
    criteria = request["questions"][QUESTION_ID]["criteria"]
    return [label for label in criteria if label != NONE_LABEL]


# --- normalizing one judge answer ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Normalized:
    status: str  # "selected" | "abstained" | "protocol_failure"
    selected_candidate: str | None
    selected_probability: float | None
    native_confidence: float | None
    protocol_error: str | None = None

    @property
    def abstained(self) -> bool:
        return self.status == "abstained"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_candidate": self.selected_candidate,
            "abstained": self.abstained,
            "selected_probability": self.selected_probability,
            "native_confidence": self.native_confidence,
            "protocol_error": self.protocol_error,
        }


def normalize_answer(answer: Any, ids: Sequence[str], *, error: str | None = None) -> Normalized:
    """Map a provider's choice answer onto: a supplied candidate id, abstention, or protocol failure.

    Both providers answer a choice question as {"choice": label, "probabilities": {label: p},
    "confidence": c}. The probability of the chosen label and the provider's own confidence are
    passed through as they are; no confidence scale is imposed.
    """

    if error is not None:
        return _failure(f"judge call failed: {error}")
    if not isinstance(answer, dict):
        return _failure("no answer object for the choice question")
    choice = answer.get("choice")
    if not isinstance(choice, str):
        return _failure("answer has no string 'choice'")
    probabilities = answer.get("probabilities")
    if probabilities is not None and (
        not isinstance(probabilities, dict)
        or not all(isinstance(k, str) and _is_number(v) for k, v in probabilities.items())
    ):
        return _failure("answer 'probabilities' is not a label -> number mapping")
    confidence = answer.get("confidence")
    if confidence is not None and not _is_number(confidence):
        return _failure("answer 'confidence' is not a number")
    probability = float(probabilities[choice]) if probabilities and choice in probabilities else None
    native = float(confidence) if confidence is not None else None
    if choice == NONE_LABEL:
        return Normalized("abstained", None, probability, native)
    if choice not in ids:
        return _failure(f"choice {choice!r} is not a supplied candidate id")
    return Normalized("selected", choice, probability, native)


def _failure(message: str) -> Normalized:
    return Normalized("protocol_failure", None, None, None, message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --- running a judge over frozen cases ------------------------------------------------------


class Judge(Protocol):
    name: str

    def judge(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return {"raw": provider output, "answer": the choice answer or None, "diagnostics": {...}}."""


def load_cases(path: str) -> tuple[dict[str, Any], str]:
    """Load a frozen T3d cases file and verify every stored request against its judge input."""

    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JudgeContractError(f"cases file is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != CASES_SCHEMA:
        raise JudgeContractError(f"not a T3d cases file (schema {CASES_SCHEMA!r} expected)")
    cases = doc.get("cases")
    if not isinstance(cases, list) or not cases:
        raise JudgeContractError("cases file has no cases")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or case_id in seen:
            raise JudgeContractError("every case needs a unique string id")
        seen.add(case_id)
        if case.get("set") not in ("ambiguous", "resolved"):
            raise JudgeContractError(f"case {case_id!r}: set must be 'ambiguous' or 'resolved'")
        missing = [key for key in ("judge_input", "request", "request_sha256", "evaluation") if key not in case]
        if missing:
            raise JudgeContractError(f"case {case_id!r} lacks {missing}")
        if render_request(case["judge_input"]) != case["request"]:
            raise JudgeContractError(f"case {case_id!r}: stored request does not match its judge input")
        if sha256_json(case["request"]) != case["request_sha256"]:
            raise JudgeContractError(f"case {case_id!r}: request_sha256 does not match the request")
    return doc, hashlib.sha256(raw).hexdigest()


def run_judge(
    cases_doc: Mapping[str, Any],
    judge: Judge,
    *,
    warmup: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Send every stored request to one judge, independently, and record what came back.

    The judge sees only the request (state + choice question), never the case id,
    set, or evaluation block.
    """

    warmup_ms = None
    if warmup:
        start = clock()
        try:
            judge.judge(json.loads(json.dumps(WARMUP_REQUEST)))
        except Exception:  # a failed warm-up is recorded as absent; the cases still run
            pass
        warmup_ms = (clock() - start) * 1000

    records = []
    for case in cases_doc["cases"]:
        request = json.loads(json.dumps(case["request"]))  # a private copy per call
        start = clock()
        output: dict[str, Any] = {}
        error = None
        try:
            output = judge.judge(request)
            if not isinstance(output, dict):
                raise TypeError("judge returned a non-dict result")
        except Exception as exc:  # recorded as a protocol failure, never retried here
            error = f"{type(exc).__name__}: {exc}"
            output = {}
        latency_ms = (clock() - start) * 1000
        answer = output.get("answer")
        normalized = normalize_answer(answer, candidate_ids(case["request"]), error=error)
        records.append(
            {
                "id": case["id"],
                "request_sha256": case["request_sha256"],
                "latency_ms": latency_ms,
                "error": error,
                "answer": answer,
                "normalized": normalized.to_dict(),
                "diagnostics": output.get("diagnostics"),
                "raw": output.get("raw"),
            }
        )
    latencies = [r["latency_ms"] for r in records]
    return {
        "warmup_ms": warmup_ms,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
        },
        "records": records,
    }


def judgments_document(
    *,
    judge: str,
    cases_path: str,
    cases_sha256: str,
    config: Mapping[str, Any],
    environment: Mapping[str, Any],
    run: Mapping[str, Any],
    created_at: str,
    load_s: float | None = None,
) -> dict[str, Any]:
    return {
        "schema": JUDGMENTS_SCHEMA,
        "judge": judge,
        "created_at": created_at,
        "cases_path": cases_path,
        "cases_sha256": cases_sha256,
        "config": dict(config),
        "environment": dict(environment),
        "load_s": load_s,
        **run,
    }


def run_summary(run: Mapping[str, Any]) -> str:
    """Counts only; correctness is scored by compare_judges.py, never by a runner."""

    statuses = [record["normalized"]["status"] for record in run["records"]]
    latency = run["latency_ms"]["median"]
    return (
        f"{len(statuses)} cases: {statuses.count('selected')} selected, {statuses.count('abstained')} abstained, "
        f"{statuses.count('protocol_failure')} protocol failures; median latency "
        + ("n/a" if latency is None else f"{latency:.1f} ms")
    )

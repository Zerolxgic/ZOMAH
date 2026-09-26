"""Characterization of shared capability invocation mechanics.

Every behavior is asserted with exact envelopes and trace rows so the
model-facing adapter and the interface-neutral runtime can be held to the
same contract.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

import zomah.capability_runtime as runtime
import zomah.model_boundary as model_boundary
from zomah.capability_runtime import CapabilityEnvelope, invoke_capability
from zomah.model_boundary import invoke_model_capability
from zomah.state import ProjectNotFound
from zomah.tracing import TraceError, TraceOutcome, TraceRecord, TraceStore

Invoker = Callable[..., CapabilityEnvelope]


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str


class LenientRequest(BaseModel):
    project_id: str


class EchoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str


class FailingStartStore(TraceStore):
    def start_run(self, **kwargs: Any) -> str:
        raise TraceError("trace database unavailable")


class FailingFinishStore(TraceStore):
    def finish_run(self, *args: Any, **kwargs: Any) -> None:
        raise TraceError("trace database unavailable")


def model_invoker(identity: str) -> Invoker:
    def invoke(request_type, capability, payload, /, *, trace_store, **dependencies):
        return invoke_model_capability(
            request_type,
            capability,
            payload,
            worker=identity,
            trace_store=trace_store,
            **dependencies,
        )

    return invoke


def runtime_invoker(identity: str) -> Invoker:
    def invoke(request_type, capability, payload, /, *, trace_store, **dependencies):
        return invoke_capability(
            request_type,
            capability,
            payload,
            trace_identity=identity,
            trace_store=trace_store,
            **dependencies,
        )

    return invoke


INVOKERS: dict[str, Callable[[str], Invoker]] = {
    "model": model_invoker,
    "runtime": runtime_invoker,
}


@pytest.fixture(params=sorted(INVOKERS))
def invoke(request: pytest.FixtureRequest) -> Invoker:
    return INVOKERS[request.param]("elyria")


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "trace.db")


def only_trace(store: TraceStore) -> TraceRecord:
    records = store.recent()
    assert len(records) == 1
    return records[0]


def trace_summary(record: TraceRecord) -> tuple[Any, ...]:
    return (
        record.worker,
        record.capability,
        record.target,
        record.outcome,
        record.error_code,
        record.finished_at is not None,
    )


def lookup(request: StrictRequest) -> EchoResponse:
    return EchoResponse(project_id=request.project_id)


def test_success_envelope_and_trace(invoke: Invoker, store: TraceStore) -> None:
    envelope = invoke(StrictRequest, lookup, {"project_id": "zomah"}, trace_store=store)

    assert envelope.model_dump() == {
        "ok": True,
        "result": {"project_id": "zomah"},
        "error": None,
    }
    assert trace_summary(only_trace(store)) == (
        "elyria",
        "lookup",
        "project:zomah",
        TraceOutcome.SUCCEEDED,
        None,
        True,
    )


def test_validation_failure_executes_nothing_and_is_traced(
    invoke: Invoker, store: TraceStore
) -> None:
    calls: list[object] = []

    def handler(request: StrictRequest) -> EchoResponse:
        calls.append(request)
        return EchoResponse(project_id=request.project_id)

    envelope = invoke(StrictRequest, handler, {"project_id": 3, "extra": "x"}, trace_store=store)

    assert calls == []
    assert envelope.model_dump() == {
        "ok": False,
        "result": None,
        "error": {
            "code": "invalid_request",
            "message": "Capability request failed validation.",
            "retryable": False,
            "details": {
                "issues": [
                    {
                        "field": "project_id",
                        "type": "string_type",
                        "message": "Input should be a valid string",
                    },
                    {
                        "field": "extra",
                        "type": "extra_forbidden",
                        "message": "Extra inputs are not permitted",
                    },
                ]
            },
        },
    }
    assert trace_summary(only_trace(store)) == (
        "elyria",
        "handler",
        None,
        TraceOutcome.FAILED,
        "invalid_request",
        True,
    )


def test_trace_is_started_before_execution(invoke: Invoker, store: TraceStore) -> None:
    seen: list[tuple[Any, ...]] = []

    def handler(request: StrictRequest) -> EchoResponse:
        seen.append(trace_summary(only_trace(store)))
        return EchoResponse(project_id=request.project_id)

    invoke(StrictRequest, handler, {"project_id": "zomah"}, trace_store=store)

    assert seen == [
        ("elyria", "handler", "project:zomah", TraceOutcome.STARTED, None, False)
    ]


def test_trace_failure_blocks_execution(invoke: Invoker, tmp_path: Path) -> None:
    calls: list[object] = []

    def handler(request: StrictRequest) -> EchoResponse:
        calls.append(request)
        return EchoResponse(project_id=request.project_id)

    blocked = FailingStartStore(tmp_path / "trace.db")
    for payload in ({"project_id": "zomah"}, {"project_id": 3}):
        envelope = invoke(StrictRequest, handler, payload, trace_store=blocked)
        assert envelope.model_dump() == {
            "ok": False,
            "result": None,
            "error": {
                "code": "trace_unavailable",
                "message": "Mandatory local tracing is unavailable; capability execution was blocked.",
                "retryable": True,
                "details": None,
            },
        }
    assert calls == []


def test_trace_finish_failure_does_not_turn_success_into_failure(
    invoke: Invoker, tmp_path: Path
) -> None:
    store = FailingFinishStore(tmp_path / "trace.db")
    envelope = invoke(StrictRequest, lookup, {"project_id": "zomah"}, trace_store=store)

    assert envelope.ok is True
    assert trace_summary(only_trace(store)) == (
        "elyria",
        "lookup",
        "project:zomah",
        TraceOutcome.STARTED,
        None,
        False,
    )


def test_known_handler_error_is_normalized(invoke: Invoker, store: TraceStore) -> None:
    def missing(request: StrictRequest) -> EchoResponse:
        raise ProjectNotFound(request.project_id)

    envelope = invoke(StrictRequest, missing, {"project_id": "nope"}, trace_store=store)

    assert envelope.model_dump() == {
        "ok": False,
        "result": None,
        "error": {
            "code": "project_not_found",
            "message": "Project was not found: nope",
            "retryable": False,
            "details": None,
        },
    }
    assert trace_summary(only_trace(store)) == (
        "elyria",
        "missing",
        "project:nope",
        TraceOutcome.FAILED,
        "project_not_found",
        True,
    )


def test_unexpected_error_is_redacted(invoke: Invoker, store: TraceStore) -> None:
    secret = "internal-secret-detail"

    def explode(request: StrictRequest) -> EchoResponse:
        raise RuntimeError(secret)

    envelope = invoke(StrictRequest, explode, {"project_id": "zomah"}, trace_store=store)

    assert envelope.model_dump() == {
        "ok": False,
        "result": None,
        "error": {
            "code": "internal_error",
            "message": "Capability failed unexpectedly.",
            "retryable": False,
            "details": None,
        },
    }
    assert secret not in envelope.model_dump_json()
    record = only_trace(store)
    assert record.error_code == "internal_error"
    assert secret not in record.model_dump_json()


def test_unsupported_response_type_is_internal_error(
    invoke: Invoker, store: TraceStore
) -> None:
    def broken(request: StrictRequest) -> Any:
        return {"project_id": request.project_id}

    envelope = invoke(StrictRequest, broken, {"project_id": "zomah"}, trace_store=store)

    assert envelope.model_dump() == {
        "ok": False,
        "result": None,
        "error": {
            "code": "internal_error",
            "message": "Capability returned an unsupported response type.",
            "retryable": False,
            "details": None,
        },
    }
    assert only_trace(store).outcome is TraceOutcome.FAILED


def test_harness_dependencies_cannot_come_from_payload(
    invoke: Invoker, store: TraceStore
) -> None:
    harness_repository = object()
    received: list[object] = []

    def with_repository(request: BaseModel, repository: object) -> EchoResponse:
        received.append(repository)
        return EchoResponse(project_id=request.project_id)  # type: ignore[attr-defined]

    # A strict request rejects the smuggled key before anything runs.
    rejected = invoke(
        StrictRequest,
        with_repository,
        {"project_id": "zomah", "repository": "payload-repository"},
        trace_store=store,
        repository=harness_repository,
    )
    assert rejected.error is not None and rejected.error.code == "invalid_request"
    assert received == []

    # A lenient request drops it; the handler still gets the harness object.
    accepted = invoke(
        LenientRequest,
        with_repository,
        {"project_id": "zomah", "repository": "payload-repository"},
        trace_store=store,
        repository=harness_repository,
    )
    assert accepted.ok is True
    assert received == [harness_repository]


def test_model_boundary_reexports_shared_types() -> None:
    assert model_boundary.CapabilityEnvelope is runtime.CapabilityEnvelope
    assert model_boundary.CapabilityError is runtime.CapabilityError
    assert model_boundary.normalize_capability_error is runtime.normalize_capability_error


def test_model_adapter_and_runtime_produce_identical_envelopes_and_traces(
    tmp_path: Path,
) -> None:
    def missing(request: StrictRequest) -> EchoResponse:
        raise ProjectNotFound(request.project_id)

    def explode(request: StrictRequest) -> EchoResponse:
        raise RuntimeError("secret")

    cases = [
        (lookup, {"project_id": "zomah"}),
        (lookup, {"project_id": 3}),
        (missing, {"project_id": "nope"}),
        (explode, {"project_id": "zomah"}),
    ]
    for index, (handler, payload) in enumerate(cases):
        model_store = TraceStore(tmp_path / f"model-{index}.db")
        runtime_store = TraceStore(tmp_path / f"runtime-{index}.db")
        via_model = model_invoker("elyria")(
            StrictRequest, handler, payload, trace_store=model_store
        )
        via_runtime = runtime_invoker("elyria")(
            StrictRequest, handler, payload, trace_store=runtime_store
        )
        assert via_model == via_runtime
        assert trace_summary(only_trace(model_store)) == trace_summary(
            only_trace(runtime_store)
        )


def test_runtime_records_caller_supplied_trace_identity_verbatim(
    store: TraceStore,
) -> None:
    invoke_capability(
        StrictRequest,
        lookup,
        {"project_id": "zomah"},
        trace_identity="any-caller:identity",
        trace_store=store,
    )
    assert only_trace(store).worker == "any-caller:identity"


def test_runtime_invokes_existing_capability_without_model_or_console_modules(
    tmp_path: Path,
) -> None:
    script = f"""
import sys
from zomah.capabilities import GetProjectStateRequest, get_project_state
from zomah.capability_runtime import invoke_capability
from zomah.state import ProjectState, ProjectStateRepository
from zomah.tracing import TraceStore

repository = ProjectStateRepository({str(tmp_path / "zomah.db")!r})
repository.initialize()
repository.create(ProjectState(
    id="zomah", name="ZOMAH", phase="p", summary="s",
    current_focus="f", next_action="n", updated_by="zerrius",
))
store = TraceStore({str(tmp_path / "trace.db")!r})
envelope = invoke_capability(
    GetProjectStateRequest, get_project_state, {{"project_id": "zomah"}},
    trace_identity="runtime-test", trace_store=store, repository=repository,
)
assert envelope.ok, envelope
assert envelope.result["project"]["id"] == "zomah", envelope.result
record = store.recent()[0]
assert (record.worker, record.capability, record.target) == (
    "runtime-test", "get_project_state", "project:zomah"
), record
loaded = sorted(m for m in sys.modules if m.startswith("zomah."))
assert "zomah.model_boundary" not in loaded, loaded
assert not any(m.startswith("zomah.console") for m in loaded), loaded
print("ok")
"""
    source_root = str(Path(runtime.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        filter(None, [source_root, os.environ.get("PYTHONPATH")])
    )}
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"

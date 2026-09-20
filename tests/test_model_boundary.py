from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from zomah.access import PathOutsideScope, ReadScope
from zomah.capabilities import GetProjectStateRequest, get_project_state
from zomah.execution import ScriptIntegrityError, UnknownScript
from zomah.model_boundary import (
    CapabilityEnvelope,
    invoke_model_capability,
    normalize_capability_error,
)
from zomah.state import ProjectNotFound, ProjectStateRepository, RevisionConflict
from zomah.tracing import TraceStore


class EchoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class EchoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def echo(request: EchoRequest) -> EchoResponse:
    return EchoResponse(text=request.text)


@pytest.fixture
def trace_store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "trace.db")


def test_success_result_is_normalized(trace_store: TraceStore) -> None:
    envelope = invoke_model_capability(
        EchoRequest, echo, {"text": "hello"}, worker="elyria", trace_store=trace_store
    )

    assert envelope == CapabilityEnvelope(ok=True, result={"text": "hello"})


def test_request_validation_is_structured_without_echoing_input(trace_store: TraceStore) -> None:
    secret = "do-not-echo-this"
    envelope = invoke_model_capability(
        EchoRequest,
        echo,
        {"text": "hello", "unexpected": secret},
        worker="elyria",
        trace_store=trace_store,
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "invalid_request"
    assert envelope.error.retryable is False
    serialized = envelope.model_dump_json()
    assert secret not in serialized
    assert "unexpected" in serialized


def test_project_not_found_is_normalized(tmp_path: Path, trace_store: TraceStore) -> None:
    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()

    envelope = invoke_model_capability(
        GetProjectStateRequest,
        get_project_state,
        {"project_id": "missing"},
        worker="elyria",
        trace_store=trace_store,
        repository=repository,
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "project_not_found"
    assert envelope.error.retryable is False


def test_path_scope_error_has_stable_code() -> None:
    error = normalize_capability_error(PathOutsideScope("outside allowed roots"))

    assert error.code == "path_outside_scope"
    assert error.retryable is False


def test_revision_conflict_tells_model_to_refresh() -> None:
    error = normalize_capability_error(RevisionConflict("expected 4, found 5"))

    assert error.code == "revision_conflict"
    assert error.retryable is True
    assert error.details == {
        "recovery": "read current project state and retry with its revision"
    }


def test_unknown_script_is_not_retryable() -> None:
    error = normalize_capability_error(UnknownScript("script is not registered: nope"))

    assert error.code == "unknown_script"
    assert error.retryable is False


def test_script_integrity_error_requires_reregistration() -> None:
    error = normalize_capability_error(
        ScriptIntegrityError("registered script content changed after approval")
    )

    assert error.code == "script_integrity_error"
    assert error.retryable is False
    assert error.details == {
        "recovery": "script must be explicitly reviewed and re-registered"
    }


def test_unexpected_exception_is_redacted(trace_store: TraceStore) -> None:
    secret = "internal-secret-detail"

    def explode(_: EchoRequest) -> EchoResponse:
        raise RuntimeError(secret)

    envelope = invoke_model_capability(
        EchoRequest, explode, {"text": "hello"}, worker="elyria", trace_store=trace_store
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "internal_error"
    assert envelope.error.message == "Capability failed unexpectedly."
    assert secret not in envelope.model_dump_json()


def test_non_model_response_becomes_internal_error(trace_store: TraceStore) -> None:
    def broken(_: EchoRequest):
        return {"text": "not a pydantic response"}

    envelope = invoke_model_capability(
        EchoRequest, broken, {"text": "hello"}, worker="elyria", trace_store=trace_store
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "internal_error"


def test_envelope_rejects_ambiguous_payloads() -> None:
    with pytest.raises(ValueError):
        CapabilityEnvelope(ok=True, result={"ok": True}, error={  # type: ignore[arg-type]
            "code": "bad",
            "message": "bad",
            "retryable": False,
        })

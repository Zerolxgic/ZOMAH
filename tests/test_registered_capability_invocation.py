from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from zomah.capabilities import GetProjectStateRequest, get_project_state
from zomah.capability_invocation import invoke_registered_model_capability
from zomah.capability_registry import (
    CapabilityAuthority,
    CapabilityDefinition,
    CapabilityLifecycle,
    CapabilityRegistry,
    default_capability_registry,
)
from zomah.model_boundary import invoke_model_capability
from zomah.state import ProjectState, ProjectStateRepository
from zomah.tracing import TraceStore


class HiddenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class HiddenResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


def test_registered_get_project_state_matches_existing_model_boundary(
    tmp_path: Path,
) -> None:
    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()
    repository.create(
        ProjectState(
            id="zomah",
            name="ZOMAH",
            phase="operator-console-foundation",
            summary="Verified local control-plane substrate.",
            current_focus="Prove registry-based capability invocation.",
            next_action="Build the Operator Console foundation.",
            updated_by="zerrius",
        )
    )

    payload = {"project_id": "zomah"}

    direct = invoke_model_capability(
        GetProjectStateRequest,
        get_project_state,
        payload,
        worker="elyria",
        trace_store=TraceStore(tmp_path / "direct-trace.db"),
        repository=repository,
    )

    registered = invoke_registered_model_capability(
        default_capability_registry(),
        "get_project_state",
        payload,
        worker="elyria",
        trace_store=TraceStore(tmp_path / "registered-trace.db"),
        repository=repository,
    )

    assert registered == direct


def test_unknown_capability_returns_stable_error_without_execution(
    tmp_path: Path,
) -> None:
    envelope = invoke_registered_model_capability(
        CapabilityRegistry(),
        "missing",
        {},
        worker="elyria",
        trace_store=TraceStore(tmp_path / "trace.db"),
    )

    assert envelope.ok is False
    assert envelope.result is None
    assert envelope.error is not None
    assert envelope.error.code == "unknown_capability"
    assert envelope.error.retryable is False


def test_non_agent_exposed_capability_is_rejected_before_handler_runs(
    tmp_path: Path,
) -> None:
    invoked = False

    def hidden_handler(request: HiddenRequest) -> HiddenResponse:
        nonlocal invoked
        invoked = True
        return HiddenResponse(value=request.value)

    registry = CapabilityRegistry()
    registry.register(
        CapabilityDefinition(
            id="hidden",
            description="User-only test capability.",
            authority=CapabilityAuthority.READ,
            lifecycle=CapabilityLifecycle.VERIFIED,
            user_exposed=True,
            agent_exposed=False,
            request_model=HiddenRequest,
            response_model=HiddenResponse,
            handler=hidden_handler,
        )
    )

    envelope = invoke_registered_model_capability(
        registry,
        "hidden",
        {"value": "should-not-run"},
        worker="elyria",
        trace_store=TraceStore(tmp_path / "trace.db"),
    )

    assert invoked is False
    assert envelope.ok is False
    assert envelope.result is None
    assert envelope.error is not None
    assert envelope.error.code == "capability_not_exposed"
    assert envelope.error.retryable is False

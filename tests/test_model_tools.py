"""Capability → model tool bridge: schema, explicit exposure, executor."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

import zomah.model_tools as model_tools
from zomah.capabilities import GetProjectStateRequest
from zomah.capability_registry import (
    CapabilityAuthority,
    CapabilityDefinition,
    CapabilityLifecycle,
    CapabilityRegistry,
    default_capability_registry,
)
from zomah.model_runtime import ToolCall
from zomah.model_tools import (
    DEFAULT_SESSION_TOOL_IDS,
    CapabilityToolExecutor,
    ToolExposureError,
    build_session_tools,
    select_session_capabilities,
    tool_definition,
)
from zomah.state import ProjectState, ProjectStateRepository
from zomah.tracing import TraceOutcome, TraceStore


class ProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class ProbeResponse(BaseModel):
    value: str


def probe_definition(**overrides: Any) -> CapabilityDefinition:
    fields: dict[str, Any] = {
        "id": "probe",
        "description": "Test-only capability.",
        "authority": CapabilityAuthority.READ,
        "lifecycle": CapabilityLifecycle.VERIFIED,
        "user_exposed": False,
        "agent_exposed": True,
        "request_model": ProbeRequest,
        "response_model": ProbeResponse,
        "handler": lambda request: ProbeResponse(value=request.value),
    }
    fields.update(overrides)
    return CapabilityDefinition(**fields)


def registry_with(*definitions: CapabilityDefinition) -> CapabilityRegistry:
    registry = default_capability_registry()
    for definition in definitions:
        registry.register(definition)
    return registry


@pytest.fixture
def repository(tmp_path: Path) -> ProjectStateRepository:
    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()
    repository.create(
        ProjectState(
            id="zomah",
            name="ZOMAH",
            phase="worker-session",
            summary="s",
            current_focus="f",
            next_action="Expose one tool.",
            updated_by="zerrius",
        )
    )
    return repository


@pytest.fixture
def trace_store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "trace.db")


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_tool_definition_uses_the_request_models_own_schema() -> None:
    definition = default_capability_registry().get("get_project_state")
    tool = tool_definition(definition)
    assert tool.name == "get_project_state"
    assert tool.description == "Retrieve the canonical current state for one project."
    assert tool.parameters == GetProjectStateRequest.model_json_schema()
    assert tool.parameters["required"] == ["project_id"]
    assert tool.parameters["additionalProperties"] is False


def test_default_session_exposes_exactly_get_project_state(
    trace_store: TraceStore, repository: ProjectStateRepository
) -> None:
    registry = registry_with(probe_definition())  # agent-exposed, but not allowlisted
    tools, _ = build_session_tools(
        registry, worker="elyria", trace_store=trace_store, repository=repository
    )
    assert DEFAULT_SESSION_TOOL_IDS == ("get_project_state",)
    assert [tool.name for tool in tools] == ["get_project_state"]


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        (probe_definition(agent_exposed=False, user_exposed=True), "not agent-exposed"),
        (probe_definition(lifecycle=CapabilityLifecycle.TESTED), "not VERIFIED"),
        (probe_definition(authority=CapabilityAuthority.CHANGE), "authority CHANGE"),
        (probe_definition(authority=CapabilityAuthority.EXECUTE), "authority EXECUTE"),
    ],
)
def test_capabilities_outside_session_policy_are_refused(
    definition: CapabilityDefinition, message: str
) -> None:
    with pytest.raises(ToolExposureError, match=message):
        select_session_capabilities(registry_with(definition), ["get_project_state", "probe"])


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ([], "at least one"),
        (["get_project_state", "get_project_state"], "unique"),
        (["missing_capability"], "not registered"),
    ],
)
def test_allowlist_must_be_explicit_and_valid(ids: list[str], message: str) -> None:
    with pytest.raises(ToolExposureError, match=message):
        select_session_capabilities(default_capability_registry(), ids)


def executor(
    trace_store: TraceStore, repository: ProjectStateRepository, registry: Any = None
) -> CapabilityToolExecutor:
    registry = registry or default_capability_registry()
    return build_session_tools(
        registry, worker="elyria", trace_store=trace_store, repository=repository
    )[1]


def test_executor_runs_get_project_state_through_model_boundary_and_traces(
    monkeypatch: pytest.MonkeyPatch,
    trace_store: TraceStore,
    repository: ProjectStateRepository,
) -> None:
    seen: list[tuple[Any, ...]] = []
    threads: list[int] = []
    original = model_tools.invoke_registered_model_capability

    def spy(*args: Any, **kwargs: Any):
        seen.append((args[1], args[2], kwargs["worker"], kwargs["repository"]))
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(model_tools, "invoke_registered_model_capability", spy)
    content = run(
        executor(trace_store, repository).execute(
            ToolCall(id="call_1", name="get_project_state", arguments={"project_id": "zomah"})
        )
    )

    assert seen == [("get_project_state", {"project_id": "zomah"}, "elyria", repository)]
    assert threads[0] != threading.get_ident()
    envelope = json.loads(content)
    assert envelope["ok"] is True
    assert envelope["result"]["project"]["id"] == "zomah"
    assert envelope["result"]["project"]["next_action"] == "Expose one tool."
    assert "error" not in envelope
    assert content == json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)

    (record,) = trace_store.recent()
    assert (record.worker, record.capability, record.target, record.outcome) == (
        "elyria",
        "get_project_state",
        "project:zomah",
        TraceOutcome.SUCCEEDED,
    )


def test_repository_comes_from_the_harness_not_model_arguments(
    trace_store: TraceStore, repository: ProjectStateRepository
) -> None:
    content = run(
        executor(trace_store, repository).execute(
            ToolCall(
                id="call_1",
                name="get_project_state",
                arguments={"project_id": "zomah", "repository": "/tmp/evil.db"},
            )
        )
    )
    envelope = json.loads(content)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_request"
    assert "evil" not in content


@pytest.mark.parametrize(
    ("arguments", "code"),
    [({"project_id": "missing"}, "project_not_found"), ({}, "invalid_request")],
)
def test_capability_errors_become_tool_result_envelopes(
    trace_store: TraceStore,
    repository: ProjectStateRepository,
    arguments: dict[str, Any],
    code: str,
) -> None:
    content = run(
        executor(trace_store, repository).execute(
            ToolCall(id="call_1", name="get_project_state", arguments=arguments)
        )
    )
    envelope = json.loads(content)
    assert (envelope["ok"], envelope["error"]["code"]) == (False, code)
    assert "Traceback" not in content
    (record,) = trace_store.recent()
    assert (record.worker, record.outcome, record.error_code) == (
        "elyria",
        TraceOutcome.FAILED,
        code,
    )


def test_executor_refuses_capabilities_outside_its_session_set(
    trace_store: TraceStore, repository: ProjectStateRepository
) -> None:
    # "probe" is registered and agent-exposed, but not in this session's set.
    registry = registry_with(probe_definition())
    content = run(
        executor(trace_store, repository, registry).execute(
            ToolCall(id="call_1", name="probe", arguments={"value": "x"})
        )
    )
    envelope = json.loads(content)
    assert (envelope["ok"], envelope["error"]["code"]) == (False, "tool_not_available")
    assert trace_store.recent() == []

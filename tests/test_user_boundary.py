from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

import zomah.user_boundary as user_boundary
from zomah.capability_invocation import invoke_registered_model_capability
from zomah.capability_registry import (
    CapabilityAuthority,
    CapabilityDefinition,
    CapabilityLifecycle,
    CapabilityRegistry,
    default_capability_registry,
)
from zomah.state import ProjectState, ProjectStateRepository
from zomah.tracing import TraceError, TraceOutcome, TraceStore
from zomah.user_boundary import (
    invoke_registered_user_capability,
    operator_trace_identity,
)


class ProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class ProbeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class FailingStartStore(TraceStore):
    def start_run(self, **kwargs: Any) -> str:
        raise TraceError("trace database unavailable")


def probe_registry(*, user: bool, agent: bool, calls: list[Any]) -> CapabilityRegistry:
    def probe(request: ProbeRequest, **dependencies: Any) -> ProbeResponse:
        calls.append((request, dependencies))
        return ProbeResponse(value=request.value)

    registry = CapabilityRegistry()
    registry.register(
        CapabilityDefinition(
            id="probe",
            description="Test-only capability.",
            authority=CapabilityAuthority.READ,
            lifecycle=CapabilityLifecycle.TESTED,
            user_exposed=user,
            agent_exposed=agent,
            request_model=ProbeRequest,
            response_model=ProbeResponse,
            handler=probe,
        )
    )
    return registry


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "trace.db")


def test_operator_trace_identity_is_namespaced() -> None:
    assert operator_trace_identity("zerrius") == "operator:zerrius"
    for invalid in ("", " zerrius", "two words", "operator:zerrius"):
        with pytest.raises(ValueError):
            operator_trace_identity(invalid)


def test_unknown_capability_returns_stable_error_without_trace(store: TraceStore) -> None:
    envelope = invoke_registered_user_capability(
        CapabilityRegistry(), "missing", {}, operator_id="zerrius", trace_store=store
    )
    assert envelope.model_dump() == {
        "ok": False,
        "result": None,
        "error": {
            "code": "unknown_capability",
            "message": "Capability is not registered: missing",
            "retryable": False,
            "details": None,
        },
    }
    assert store.recent() == []


def test_agent_only_capability_is_rejected_before_execution(store: TraceStore) -> None:
    calls: list[Any] = []
    registry = probe_registry(user=False, agent=True, calls=calls)

    envelope = invoke_registered_user_capability(
        registry, "probe", {"value": "x"}, operator_id="zerrius", trace_store=store
    )

    assert envelope.model_dump() == {
        "ok": False,
        "result": None,
        "error": {
            "code": "capability_not_exposed",
            "message": "Capability is not exposed to operators: probe",
            "retryable": False,
            "details": None,
        },
    }
    assert calls == []
    assert store.recent() == []

    # The same capability remains available to the model-facing path.
    model = invoke_registered_model_capability(
        registry, "probe", {"value": "x"}, worker="elyria", trace_store=store
    )
    assert model.ok is True


def test_user_only_capability_executes_and_is_traced_as_operator(
    store: TraceStore,
) -> None:
    calls: list[Any] = []
    registry = probe_registry(user=True, agent=False, calls=calls)

    envelope = invoke_registered_user_capability(
        registry, "probe", {"value": "x"}, operator_id="zerrius", trace_store=store
    )

    assert envelope.model_dump() == {"ok": True, "result": {"value": "x"}, "error": None}
    assert len(calls) == 1
    (record,) = store.recent()
    assert (record.worker, record.capability, record.outcome) == (
        "operator:zerrius",
        "probe",
        TraceOutcome.SUCCEEDED,
    )

    # agent_exposed=false still blocks the model-facing path.
    model = invoke_registered_model_capability(
        registry, "probe", {"value": "x"}, worker="elyria", trace_store=store
    )
    assert model.error is not None and model.error.code == "capability_not_exposed"
    assert len(calls) == 1


def test_user_path_delegates_to_shared_runtime(
    monkeypatch: pytest.MonkeyPatch, store: TraceStore
) -> None:
    seen: list[dict[str, Any]] = []
    original = user_boundary.invoke_capability

    def spy(*args: Any, **kwargs: Any):
        seen.append({"args": args, "kwargs": kwargs})
        return original(*args, **kwargs)

    monkeypatch.setattr(user_boundary, "invoke_capability", spy)
    calls: list[Any] = []
    registry = probe_registry(user=True, agent=False, calls=calls)
    definition = registry.get("probe")

    invoke_registered_user_capability(
        registry, "probe", {"value": "x"}, operator_id="zerrius", trace_store=store
    )

    assert len(seen) == 1
    assert seen[0]["args"][:2] == (definition.request_model, definition.handler)
    assert seen[0]["kwargs"]["trace_identity"] == "operator:zerrius"
    assert seen[0]["kwargs"]["trace_store"] is store


def test_trace_failure_blocks_handler(tmp_path: Path) -> None:
    calls: list[Any] = []
    registry = probe_registry(user=True, agent=False, calls=calls)

    envelope = invoke_registered_user_capability(
        registry,
        "probe",
        {"value": "x"},
        operator_id="zerrius",
        trace_store=FailingStartStore(tmp_path / "trace.db"),
    )

    assert envelope.error is not None
    assert envelope.error.code == "trace_unavailable"
    assert calls == []


def test_harness_dependencies_stay_outside_payload(
    tmp_path: Path, store: TraceStore
) -> None:
    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()
    repository.create(
        ProjectState(
            id="zomah",
            name="ZOMAH",
            phase="p",
            summary="s",
            current_focus="f",
            updated_by="zerrius",
        )
    )
    registry = default_capability_registry()

    smuggled = invoke_registered_user_capability(
        registry,
        "get_project_state",
        {"project_id": "zomah", "repository": "payload-repository"},
        operator_id="zerrius",
        trace_store=store,
        repository=repository,
    )
    assert smuggled.error is not None and smuggled.error.code == "invalid_request"

    envelope = invoke_registered_user_capability(
        registry,
        "get_project_state",
        {"project_id": "zomah"},
        operator_id="zerrius",
        trace_store=store,
        repository=repository,
    )
    assert envelope.ok is True
    assert envelope.result is not None
    assert envelope.result["project"]["id"] == "zomah"


def test_user_path_does_not_load_model_boundary(tmp_path: Path) -> None:
    script = f"""
import sys
from zomah.capability_registry import default_capability_registry
from zomah.state import ProjectState, ProjectStateRepository
from zomah.tracing import TraceStore
from zomah.user_boundary import invoke_registered_user_capability

repository = ProjectStateRepository({str(tmp_path / "zomah.db")!r})
repository.initialize()
repository.create(ProjectState(
    id="zomah", name="ZOMAH", phase="p", summary="s",
    current_focus="f", updated_by="zerrius",
))
envelope = invoke_registered_user_capability(
    default_capability_registry(), "get_project_state", {{"project_id": "zomah"}},
    operator_id="zerrius", trace_store=TraceStore({str(tmp_path / "trace.db")!r}),
    repository=repository,
)
assert envelope.ok, envelope
loaded = sorted(m for m in sys.modules if m.startswith("zomah."))
assert "zomah.model_boundary" not in loaded, loaded
assert "zomah.capability_invocation" not in loaded, loaded
print("ok")
"""
    source_root = str(Path(user_boundary.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [source_root, os.environ.get("PYTHONPATH")])
        ),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
    source = Path(user_boundary.__file__).read_text(encoding="utf-8")
    imports = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any(
        "model_boundary" in line or "capability_invocation" in line for line in imports
    )
    assert "invoke_model_capability" not in source

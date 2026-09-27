"""Bridge from ZOMAH capabilities to model-visible tools.

Session tool exposure is explicit: a session starts from an allowlist of
capability ids, and each one must be registered, ``agent_exposed``,
``VERIFIED``, and of an allowed authority (``READ`` only for now). Registry
exposure alone never makes a capability visible to a session.

``CapabilityToolExecutor`` runs model tool calls through the existing
model-facing boundary, ``invoke_registered_model_capability``, so validation,
the registry exposure check, mandatory tracing under the worker's identity,
and error normalization all apply unchanged. Harness dependencies are routed
per capability: a call receives only the entry configured for that capability
id, which must match the names the capability declares exactly. The resulting
``CapabilityEnvelope`` (success or normalized error) is returned to the model
as compact JSON; capability errors are tool results, not turn failures.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from zomah.capability_invocation import invoke_registered_model_capability
from zomah.capability_registry import (
    CapabilityAuthority,
    CapabilityDefinition,
    CapabilityLifecycle,
    CapabilityNotFoundError,
    CapabilityRegistry,
)
from zomah.capability_runtime import CapabilityEnvelope, CapabilityError
from zomah.model_runtime import ToolCall, ToolDefinition
from zomah.tracing import TraceStore

# Tools every live worker session exposes.
DEFAULT_SESSION_TOOL_IDS: tuple[str, ...] = ("get_project_state",)
# Exposed only when the harness configures filesystem read roots.
READ_FILE_TOOL_ID = "read_file"

# Harness-owned keyword arguments per capability id, e.g.
# {"get_project_state": {"repository": repo}, "read_file": {"scope": scope}}.
CapabilityDependencies = Mapping[str, Mapping[str, Any]]
READ_ONLY: frozenset[CapabilityAuthority] = frozenset({CapabilityAuthority.READ})


class ToolExposureError(ValueError):
    """A capability cannot be exposed to a model session as configured."""


def tool_definition(definition: CapabilityDefinition) -> ToolDefinition:
    """Describe one capability as a provider-neutral tool.

    Parameters come from the capability's own request model, never a copy.
    """

    return ToolDefinition(
        name=definition.id,
        description=definition.description,
        parameters=definition.request_model.model_json_schema(),
    )


def select_session_capabilities(
    registry: CapabilityRegistry,
    allowed_ids: Sequence[str],
    *,
    allowed_authorities: Collection[CapabilityAuthority] = READ_ONLY,
) -> tuple[CapabilityDefinition, ...]:
    """Resolve an explicit allowlist, refusing anything outside session policy."""

    if not allowed_ids:
        raise ToolExposureError("a tool session needs at least one capability id")
    if len(set(allowed_ids)) != len(allowed_ids):
        raise ToolExposureError("session capability ids must be unique")
    selected: list[CapabilityDefinition] = []
    for capability_id in allowed_ids:
        try:
            definition = registry.get(capability_id)
        except CapabilityNotFoundError:
            raise ToolExposureError(f"capability is not registered: {capability_id}") from None
        if not definition.agent_exposed:
            raise ToolExposureError(f"capability is not agent-exposed: {capability_id}")
        if definition.lifecycle is not CapabilityLifecycle.VERIFIED:
            raise ToolExposureError(
                f"capability is not VERIFIED ({definition.lifecycle.value}): {capability_id}"
            )
        if definition.authority not in allowed_authorities:
            raise ToolExposureError(
                f"capability authority {definition.authority.value} is not allowed "
                f"for this session: {capability_id}"
            )
        selected.append(definition)
    return tuple(selected)


def route_dependencies(
    capabilities: Sequence[CapabilityDefinition],
    dependencies: CapabilityDependencies,
) -> dict[str, Mapping[str, Any]]:
    """Validate and freeze one dependency entry per session capability.

    Each capability gets exactly the names it declares: a missing name, an
    extra name, or an entry for a capability outside the session fails here,
    at construction, rather than on a model call.
    """

    session_ids = {definition.id for definition in capabilities}
    stray = sorted(set(dependencies) - session_ids)
    if stray:
        raise ToolExposureError(
            f"dependencies configured for capabilities not in this session: {', '.join(stray)}"
        )
    routed: dict[str, Mapping[str, Any]] = {}
    for definition in capabilities:
        entry = dict(dependencies.get(definition.id, {}))
        missing = sorted(definition.dependencies - entry.keys())
        extra = sorted(entry.keys() - definition.dependencies)
        if missing:
            raise ToolExposureError(
                f"{definition.id} requires harness dependencies: {', '.join(missing)}"
            )
        if extra:
            raise ToolExposureError(
                f"{definition.id} does not accept dependencies: {', '.join(extra)}"
            )
        routed[definition.id] = MappingProxyType(entry)
    return routed


class CapabilityToolExecutor:
    """Runs session tool calls through the model-facing capability boundary."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        capabilities: Sequence[CapabilityDefinition],
        *,
        worker: str,
        trace_store: TraceStore,
        dependencies: CapabilityDependencies,
    ) -> None:
        self._registry = registry
        self._worker = worker
        self._trace_store = trace_store
        # Harness-owned objects per capability id (a repository, a read
        # scope), never taken from model arguments.
        self._dependencies = route_dependencies(capabilities, dependencies)
        self._ids = frozenset(self._dependencies)

    async def execute(self, call: ToolCall) -> str:
        if call.name not in self._ids:
            envelope = CapabilityEnvelope(
                ok=False,
                error=CapabilityError(
                    code="tool_not_available",
                    message=f"Tool is not available in this session: {call.name[:64]}",
                    retryable=False,
                ),
            )
        else:
            envelope = await asyncio.to_thread(
                invoke_registered_model_capability,
                self._registry,
                call.name,
                dict(call.arguments),
                worker=self._worker,
                trace_store=self._trace_store,
                **self._dependencies[call.name],
            )
        return envelope.model_dump_json(exclude_none=True)


def build_session_tools(
    registry: CapabilityRegistry,
    *,
    worker: str,
    trace_store: TraceStore,
    dependencies: CapabilityDependencies,
    allowed_ids: Sequence[str] = DEFAULT_SESSION_TOOL_IDS,
) -> tuple[tuple[ToolDefinition, ...], CapabilityToolExecutor]:
    """Tool definitions and their executor for one session, from one allowlist."""

    capabilities = select_session_capabilities(registry, allowed_ids)
    executor = CapabilityToolExecutor(
        registry,
        capabilities,
        worker=worker,
        trace_store=trace_store,
        dependencies=dependencies,
    )
    return tuple(tool_definition(definition) for definition in capabilities), executor

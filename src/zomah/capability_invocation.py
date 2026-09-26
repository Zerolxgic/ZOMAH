from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zomah.capability_registry import (
    CapabilityNotFoundError,
    CapabilityRegistry,
)
from zomah.model_boundary import (
    CapabilityEnvelope,
    CapabilityError,
    invoke_model_capability,
)
from zomah.tracing import TraceStore


def invoke_registered_model_capability(
    registry: CapabilityRegistry,
    capability_id: str,
    payload: Mapping[str, Any],
    /,
    *,
    worker: str,
    trace_store: TraceStore,
    **dependencies: Any,
) -> CapabilityEnvelope:
    """Resolve one agent-visible capability and invoke it through the existing model boundary.

    Registry lookup and exposure checks happen before capability execution.
    Request validation, mandatory tracing, handler invocation, result shaping,
    and capability error normalization remain owned by ``invoke_model_capability``.
    """

    try:
        definition = registry.get(capability_id)
    except CapabilityNotFoundError:
        return CapabilityEnvelope(
            ok=False,
            error=CapabilityError(
                code="unknown_capability",
                message=f"Capability is not registered: {capability_id}",
                retryable=False,
            ),
        )

    if not definition.agent_exposed:
        return CapabilityEnvelope(
            ok=False,
            error=CapabilityError(
                code="capability_not_exposed",
                message=f"Capability is not exposed to the active worker: {capability_id}",
                retryable=False,
            ),
        )

    return invoke_model_capability(
        definition.request_model,
        definition.handler,
        payload,
        worker=worker,
        trace_store=trace_store,
        **dependencies,
    )

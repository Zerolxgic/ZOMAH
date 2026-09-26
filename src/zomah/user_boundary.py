"""Human operator capability boundary.

The second front door into registered capabilities, distinct from the
model-facing ``zomah.capability_invocation`` / ``zomah.model_boundary`` path.
Registry lookup and the ``user_exposed`` check happen here; validation,
mandatory tracing, invocation, and error normalization belong to the shared
``zomah.capability_runtime``. ``agent_exposed`` is irrelevant to this path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zomah.capability_registry import CapabilityNotFoundError, CapabilityRegistry
from zomah.capability_runtime import (
    CapabilityEnvelope,
    CapabilityError,
    invoke_capability,
)
from zomah.tracing import TraceStore

OPERATOR_TRACE_PREFIX = "operator:"


def operator_trace_identity(operator_id: str) -> str:
    """Trace namespace for a human operator: ``operator:<operator_id>``.

    The raw operator id is the future actor/provenance identity; the prefixed
    value is only what the existing trace ``worker`` column records.
    """

    if (
        not operator_id
        or operator_id != operator_id.strip()
        or any(character.isspace() for character in operator_id)
        or ":" in operator_id
    ):
        raise ValueError(f"invalid operator id: {operator_id!r}")
    return f"{OPERATOR_TRACE_PREFIX}{operator_id}"


def invoke_registered_user_capability(
    registry: CapabilityRegistry,
    capability_id: str,
    payload: Mapping[str, Any],
    /,
    *,
    operator_id: str,
    trace_store: TraceStore,
    **dependencies: Any,
) -> CapabilityEnvelope:
    """Resolve one user-exposed capability and invoke it for a human operator.

    Unknown and non-user-exposed capabilities are rejected before anything is
    traced or executed. ``dependencies`` are harness-owned objects passed by
    the caller, never read from ``payload``.
    """

    trace_identity = operator_trace_identity(operator_id)

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

    if not definition.user_exposed:
        return CapabilityEnvelope(
            ok=False,
            error=CapabilityError(
                code="capability_not_exposed",
                message=f"Capability is not exposed to operators: {capability_id}",
                retryable=False,
            ),
        )

    return invoke_capability(
        definition.request_model,
        definition.handler,
        payload,
        trace_identity=trace_identity,
        trace_store=trace_store,
        **dependencies,
    )

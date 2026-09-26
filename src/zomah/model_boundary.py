"""Model-facing capability boundary.

A thin adapter over ``zomah.capability_runtime``: the calling model worker's
identity is recorded as the trace identity, and every other mechanic
(validation, dependency injection, mandatory tracing, invocation, error
normalization and redaction, result shaping) belongs to the shared runtime.

``CapabilityError``, ``CapabilityEnvelope`` and ``normalize_capability_error``
are re-exported so existing imports keep working.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from zomah.capability_runtime import (
    CapabilityEnvelope,
    CapabilityError,
    RequestModel,
    ResponseModel,
    invoke_capability,
    normalize_capability_error,
)
from zomah.tracing import TraceStore

__all__ = [
    "CapabilityEnvelope",
    "CapabilityError",
    "invoke_model_capability",
    "normalize_capability_error",
]


def invoke_model_capability(
    request_type: type[RequestModel],
    capability: Callable[..., ResponseModel],
    payload: Mapping[str, Any],
    /,
    *,
    worker: str,
    trace_store: TraceStore,
    **dependencies: Any,
) -> CapabilityEnvelope:
    """Validate, trace, invoke, and normalize one model-facing capability call.

    Tracing is mandatory at this boundary. A capability never executes if a
    trace cannot be started first. The trace stores compact metadata only;
    model payloads, prompts, results, stdout/stderr, and exception details are
    intentionally excluded.

    `dependencies` are harness-owned objects such as repositories, scopes,
    registries, actor identity, or inspection roots. They are never sourced
    from the model payload by this adapter.
    """

    return invoke_capability(
        request_type,
        capability,
        payload,
        trace_identity=worker,
        trace_store=trace_store,
        **dependencies,
    )

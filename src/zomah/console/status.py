"""Operator Console session state, rendered by the header and ``/status``.

``ConsoleStatus`` is the single owner of transient console session context.
It is not canonical ProjectState: the repository owns that.
"""

from __future__ import annotations

from dataclasses import dataclass

PLACEHOLDER_NOT_SET = "not set"
PLACEHOLDER_NOT_CONNECTED = "not connected"
PLACEHOLDER_UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ConsoleStatus:
    """Transient console session context. ``None`` means no live source.

    ``active_project_id`` is the canonical ProjectState id used for capability
    requests. ``project`` is a human-facing label (a name or folder); it is
    never used as an id, and is only filled from canonical state once that
    state has been read.
    """

    active_project_id: str | None = None
    project: str | None = None
    zomah_state: str | None = None
    model: str | None = None
    tools_available: int | None = None
    context_used: int | None = None
    context_limit: int | None = None


STATUS_LABELS = {
    "project": "Project",
    "state": "State",
    "model": "Model",
    "tools": "Tools",
    "context": "Context",
}


NOT_READ_YET = "not read yet"


def project_display(status: ConsoleStatus) -> str:
    """One deterministic project rendering for the header and ``/status``."""

    label, project_id = status.project, status.active_project_id
    if label and project_id:
        return f"{label} ({project_id})"
    if project_id:
        return f"{project_id} ({NOT_READ_YET})"
    return label or PLACEHOLDER_NOT_SET


def status_values(status: ConsoleStatus) -> dict[str, str]:
    """Display values keyed like ``STATUS_LABELS``, with explicit placeholders."""

    if status.context_used is None and status.context_limit is None:
        context = PLACEHOLDER_UNAVAILABLE
    else:
        used = PLACEHOLDER_UNAVAILABLE if status.context_used is None else status.context_used
        limit = PLACEHOLDER_NOT_SET if status.context_limit is None else status.context_limit
        context = f"{used} / {limit}"
    return {
        "project": project_display(status),
        "state": status.zomah_state or PLACEHOLDER_NOT_CONNECTED,
        "model": status.model or PLACEHOLDER_NOT_CONNECTED,
        "tools": (
            PLACEHOLDER_UNAVAILABLE
            if status.tools_available is None
            else str(status.tools_available)
        ),
        "context": context,
    }

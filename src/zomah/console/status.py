"""Console status view data shared by the header and ``/status``."""

from __future__ import annotations

from dataclasses import dataclass

PLACEHOLDER_NOT_SET = "not set"
PLACEHOLDER_NOT_CONNECTED = "not connected"
PLACEHOLDER_UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ConsoleStatus:
    """Header view data. ``None`` means no live source exists for the field."""

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


def status_values(status: ConsoleStatus) -> dict[str, str]:
    """Display values keyed like ``STATUS_LABELS``, with explicit placeholders."""

    if status.context_used is None or status.context_limit is None:
        context = PLACEHOLDER_UNAVAILABLE
    else:
        context = f"{status.context_used} / {status.context_limit}"
    return {
        "project": status.project or PLACEHOLDER_NOT_SET,
        "state": status.zomah_state or PLACEHOLDER_NOT_CONNECTED,
        "model": status.model or PLACEHOLDER_NOT_CONNECTED,
        "tools": (
            PLACEHOLDER_UNAVAILABLE
            if status.tools_available is None
            else str(status.tools_available)
        ),
        "context": context,
    }

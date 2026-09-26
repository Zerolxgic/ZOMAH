"""Harness-owned dependencies for operator capability calls.

Built once per console session and injected; command views never construct
repositories or trace stores themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from zomah.state import ProjectStateRepository, default_database_path
from zomah.tracing import TraceStore, default_trace_db_path
from zomah.user_boundary import operator_trace_identity


@dataclass(frozen=True, slots=True)
class OperatorAccess:
    """Who the operator is and the long-lived stores their commands may read."""

    operator_id: str
    trace_store: TraceStore
    project_repository: ProjectStateRepository

    def __post_init__(self) -> None:
        operator_trace_identity(self.operator_id)


def build_default_operator_access(operator_id: str) -> OperatorAccess:
    """Open the canonical ProjectState database and trace store.

    The ProjectState schema is not created here; bootstrapping canonical state
    stays with ``examples/bootstrap_state.py``.
    """

    return OperatorAccess(
        operator_id=operator_id,
        trace_store=TraceStore(default_trace_db_path()),
        project_repository=ProjectStateRepository(default_database_path()),
    )

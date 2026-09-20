from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from zomah.state import (
    Decision,
    DecisionStatus,
    ProjectState,
    ProjectStatePatch,
    ProjectStateRepository,
)


ProjectId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class GetProjectStateRequest(BaseModel):
    """Validated input for the get_project_state capability."""

    model_config = ConfigDict(extra="forbid")

    project_id: ProjectId


MAX_MODEL_DECISIONS = 20


class DecisionWindow(BaseModel):
    """Compact summary of canonical decision history visible to the model."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    returned: int = Field(ge=0)
    truncated: bool
    by_status: dict[DecisionStatus, int]


class ModelProjectState(ProjectState):
    """Model-facing projection of canonical project state.

    Current project truth remains complete, while decision history is bounded
    so an old project cannot consume the worker's context window. Canonical
    SQLite state is never truncated.
    """

    decision_window: DecisionWindow


class GetProjectStateResponse(BaseModel):
    """Structured, context-bounded output returned by get_project_state."""

    model_config = ConfigDict(extra="forbid")

    project: ModelProjectState


class UpdateProjectStateRequest(BaseModel):
    """Validated input for the update_project_state capability."""

    model_config = ConfigDict(extra="forbid")

    project_id: ProjectId
    patch: ProjectStatePatch

    @model_validator(mode="after")
    def require_state_change(self) -> "UpdateProjectStateRequest":
        if not _changed_fields(self.patch):
            raise ValueError("project state patch contains no changes")
        return self


class UpdateProjectStateResponse(BaseModel):
    """Structured output returned after a successful state mutation."""

    model_config = ConfigDict(extra="forbid")

    project: ModelProjectState
    previous_revision: int = Field(ge=0)
    new_revision: int = Field(ge=1)
    changed_fields: list[str]


def get_project_state(
    request: GetProjectStateRequest,
    repository: ProjectStateRepository,
) -> GetProjectStateResponse:
    """Return the canonical current state for one project.

    This capability deliberately contains no model/runtime logic. It validates
    the request, delegates canonical state access to the repository, and
    returns a stable structured response. Repository errors such as
    ProjectNotFound remain explicit for a future tool adapter to translate.
    """

    return GetProjectStateResponse(
        project=_model_project_state(repository.get(request.project_id))
    )


def update_project_state(
    request: UpdateProjectStateRequest,
    repository: ProjectStateRepository,
    *,
    actor: str,
) -> UpdateProjectStateResponse:
    """Apply one validated ProjectState patch and return the canonical result.

    Transactionality and optimistic concurrency stay in the repository. Actor
    identity is supplied by the harness call site rather than model input. The
    capability validates that the request actually changes state, delegates the
    mutation, and shapes a stable result for a future tool adapter.
    """

    previous_revision = request.patch.expected_revision
    changed_fields = _changed_fields(request.patch)
    project = repository.apply_patch(request.project_id, request.patch, actor=actor)

    return UpdateProjectStateResponse(
        project=_model_project_state(project),
        previous_revision=previous_revision,
        new_revision=project.revision,
        changed_fields=changed_fields,
    )


def _changed_fields(patch: ProjectStatePatch) -> list[str]:
    fields: list[str] = []

    for field in ("name", "status", "phase", "summary", "current_focus"):
        if getattr(patch, field) is not None:
            fields.append(field)

    for field in ("last_action", "next_action"):
        if field in patch.model_fields_set:
            fields.append(field)

    for field in (
        "add_open_questions",
        "resolve_open_questions",
        "add_blockers",
        "resolve_blockers",
        "add_decisions",
        "add_important_paths",
    ):
        if getattr(patch, field):
            fields.append(field)

    return fields


def _model_project_state(state: ProjectState) -> ModelProjectState:
    """Return a bounded decision window without changing canonical state."""

    by_status = {status: 0 for status in DecisionStatus}
    for decision in state.decisions:
        by_status[decision.status] += 1

    active = [
        decision
        for decision in state.decisions
        if decision.status in {DecisionStatus.PROPOSED, DecisionStatus.ACCEPTED}
    ]
    historical = [
        decision
        for decision in state.decisions
        if decision.status not in {DecisionStatus.PROPOSED, DecisionStatus.ACCEPTED}
    ]

    active.sort(key=lambda item: (item.created_at, item.id), reverse=True)
    historical.sort(key=lambda item: (item.created_at, item.id), reverse=True)

    selected: list[Decision] = active[:MAX_MODEL_DECISIONS]
    if len(selected) < MAX_MODEL_DECISIONS:
        selected.extend(historical[: MAX_MODEL_DECISIONS - len(selected)])

    payload = state.model_dump()
    payload["decisions"] = selected
    payload["decision_window"] = DecisionWindow(
        total=len(state.decisions),
        returned=len(selected),
        truncated=len(selected) < len(state.decisions),
        by_status=by_status,
    )
    return ModelProjectState.model_validate(payload)

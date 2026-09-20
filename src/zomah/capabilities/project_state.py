from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from zomah.state import ProjectState, ProjectStatePatch, ProjectStateRepository


ProjectId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class GetProjectStateRequest(BaseModel):
    """Validated input for the get_project_state capability."""

    model_config = ConfigDict(extra="forbid")

    project_id: ProjectId


class GetProjectStateResponse(BaseModel):
    """Structured output returned by the get_project_state capability."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectState


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

    project: ProjectState
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

    return GetProjectStateResponse(project=repository.get(request.project_id))


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
        project=project,
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

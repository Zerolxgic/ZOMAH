from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from zomah.state import ProjectState, ProjectStateRepository


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

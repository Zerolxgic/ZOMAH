from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.capabilities import GetProjectStateRequest, get_project_state
from zomah.state import ProjectNotFound, ProjectState, ProjectStateRepository


def make_repo(tmp_path: Path) -> ProjectStateRepository:
    repo = ProjectStateRepository(tmp_path / "zomah.db")
    repo.initialize()
    return repo


def seed_project(repo: ProjectStateRepository) -> ProjectState:
    return repo.create(
        ProjectState(
            id="zomah",
            name="ZOMAH",
            phase="state",
            summary="Minimal local agent control plane.",
            current_focus="Expose canonical project state as a capability.",
            next_action="Connect the capability to a future model tool adapter.",
            updated_by="zerrius",
        )
    )


def test_get_project_state_returns_canonical_state(tmp_path: Path):
    repo = make_repo(tmp_path)
    expected = seed_project(repo)

    response = get_project_state(
        GetProjectStateRequest(project_id="zomah"),
        repository=repo,
    )

    assert expected.model_dump() == response.project.model_dump(
        exclude={"decision_window"}
    )
    assert response.project.revision == 0


def test_request_normalizes_project_id_and_forbids_extra_fields():
    request = GetProjectStateRequest(project_id="  zomah  ")
    assert request.project_id == "zomah"

    with pytest.raises(ValidationError):
        GetProjectStateRequest(project_id="zomah", unknown=True)


def test_request_rejects_blank_project_id():
    with pytest.raises(ValidationError):
        GetProjectStateRequest(project_id="   ")


def test_missing_project_remains_explicit(tmp_path: Path):
    repo = make_repo(tmp_path)

    with pytest.raises(ProjectNotFound):
        get_project_state(
            GetProjectStateRequest(project_id="missing"),
            repository=repo,
        )

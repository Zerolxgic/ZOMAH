from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.capabilities import (
    UpdateProjectStateRequest,
    update_project_state,
)
from zomah.state import (
    ProjectNotFound,
    ProjectState,
    ProjectStatePatch,
    ProjectStateRepository,
    RevisionConflict,
)


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
            current_focus="Expose canonical project state as capabilities.",
            last_action="Verified get_project_state.",
            next_action="Implement update_project_state.",
            updated_by="zerrius",
        )
    )


def test_update_project_state_returns_canonical_result_and_revision_metadata(tmp_path: Path):
    repo = make_repo(tmp_path)
    seed_project(repo)

    response = update_project_state(
        UpdateProjectStateRequest(
            project_id="zomah",
            patch=ProjectStatePatch(
                expected_revision=0,
                updated_by="elyria",
                current_focus="Validate the read/write state loop.",
                last_action="Implemented update_project_state.",
                next_action="Choose the next minimal read capability.",
            ),
        ),
        repository=repo,
    )

    assert response.previous_revision == 0
    assert response.new_revision == 1
    assert response.project.revision == 1
    assert response.project.updated_by == "elyria"
    assert response.changed_fields == [
        "current_focus",
        "last_action",
        "next_action",
    ]
    assert repo.get("zomah") == response.project


def test_update_request_rejects_noop_patch():
    with pytest.raises(ValidationError, match="contains no changes"):
        UpdateProjectStateRequest(
            project_id="zomah",
            patch=ProjectStatePatch(
                expected_revision=0,
                updated_by="elyria",
            ),
        )


def test_update_project_state_preserves_explicit_null(tmp_path: Path):
    repo = make_repo(tmp_path)
    seed_project(repo)

    response = update_project_state(
        UpdateProjectStateRequest(
            project_id="zomah",
            patch=ProjectStatePatch(
                expected_revision=0,
                updated_by="zerrius",
                next_action=None,
            ),
        ),
        repository=repo,
    )

    assert response.project.next_action is None
    assert response.changed_fields == ["next_action"]


def test_update_project_state_keeps_revision_conflict_explicit(tmp_path: Path):
    repo = make_repo(tmp_path)
    seed_project(repo)

    repo.apply_patch(
        "zomah",
        ProjectStatePatch(
            expected_revision=0,
            updated_by="zerrius",
            last_action="Another worker updated state first.",
        ),
    )

    with pytest.raises(RevisionConflict):
        update_project_state(
            UpdateProjectStateRequest(
                project_id="zomah",
                patch=ProjectStatePatch(
                    expected_revision=0,
                    updated_by="elyria",
                    last_action="Stale update.",
                ),
            ),
            repository=repo,
        )


def test_update_project_state_keeps_missing_project_explicit(tmp_path: Path):
    repo = make_repo(tmp_path)

    with pytest.raises(ProjectNotFound):
        update_project_state(
            UpdateProjectStateRequest(
                project_id="missing",
                patch=ProjectStatePatch(
                    expected_revision=0,
                    updated_by="elyria",
                    current_focus="This project does not exist.",
                ),
            ),
            repository=repo,
        )

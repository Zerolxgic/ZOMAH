from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.capabilities import (
    UpdateProjectStateRequest,
    update_project_state,
)
from zomah.state import (
    DecisionProposal,
    DecisionStatus,
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
                current_focus="Validate the read/write state loop.",
                last_action="Implemented update_project_state.",
                next_action="Choose the next minimal read capability.",
            ),
        ),
        repository=repo,
        actor="elyria",
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
            patch=ProjectStatePatch(expected_revision=0),
        )


def test_update_project_state_preserves_explicit_null(tmp_path: Path):
    repo = make_repo(tmp_path)
    seed_project(repo)

    response = update_project_state(
        UpdateProjectStateRequest(
            project_id="zomah",
            patch=ProjectStatePatch(
                expected_revision=0,
                next_action=None,
            ),
        ),
        repository=repo,
        actor="zerrius",
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
            last_action="Another worker updated state first.",
        ),
        actor="zerrius",
    )

    with pytest.raises(RevisionConflict):
        update_project_state(
            UpdateProjectStateRequest(
                project_id="zomah",
                patch=ProjectStatePatch(
                    expected_revision=0,
                    last_action="Stale update.",
                ),
            ),
            repository=repo,
            actor="elyria",
        )


def test_update_project_state_keeps_missing_project_explicit(tmp_path: Path):
    repo = make_repo(tmp_path)

    with pytest.raises(ProjectNotFound):
        update_project_state(
            UpdateProjectStateRequest(
                project_id="missing",
                patch=ProjectStatePatch(
                    expected_revision=0,
                    current_focus="This project does not exist.",
                ),
            ),
            repository=repo,
            actor="elyria",
        )


def test_model_patch_cannot_supply_actor_identity():
    with pytest.raises(ValidationError, match="updated_by"):
        ProjectStatePatch.model_validate(
            {
                "expected_revision": 0,
                "updated_by": "zerrius",
                "current_focus": "Pretend this came from the user.",
            }
        )


def test_model_decision_input_cannot_supply_status_or_timestamp():
    with pytest.raises(ValidationError):
        DecisionProposal.model_validate(
            {
                "id": "ADR-003",
                "statement": "A proposed decision.",
                "rationale": "The model may propose but not authorize.",
                "status": "accepted",
                "created_at": "2000-01-01T00:00:00Z",
            }
        )


def test_model_added_decision_is_stored_as_harness_timestamped_proposal(tmp_path: Path):
    repo = make_repo(tmp_path)
    seed_project(repo)
    before = datetime.now(timezone.utc)

    response = update_project_state(
        UpdateProjectStateRequest(
            project_id="zomah",
            patch=ProjectStatePatch(
                expected_revision=0,
                add_decisions=[
                    DecisionProposal(
                        id="ADR-003",
                        statement="Keep authority outside model input.",
                        rationale="Proposal and authorization are different lifecycle states.",
                    )
                ],
            ),
        ),
        repository=repo,
        actor="elyria",
    )

    decision = response.project.decisions[0]
    assert decision.id == "ADR-003"
    assert decision.status == DecisionStatus.PROPOSED
    assert decision.created_at >= before
    assert response.project.updated_by == "elyria"


def test_update_project_state_rejects_blank_injected_actor(tmp_path: Path):
    repo = make_repo(tmp_path)
    seed_project(repo)

    with pytest.raises(ValueError, match="actor must not be blank"):
        update_project_state(
            UpdateProjectStateRequest(
                project_id="zomah",
                patch=ProjectStatePatch(
                    expected_revision=0,
                    current_focus="Should not commit.",
                ),
            ),
            repository=repo,
            actor="   ",
        )

    assert repo.get("zomah").revision == 0

from pathlib import Path

import pytest

from zomah.state import (
    Decision,
    ImportantPath,
    ProjectState,
    ProjectStatePatch,
    ProjectStateRepository,
    RevisionConflict,
)


def make_repo(tmp_path: Path) -> ProjectStateRepository:
    repo = ProjectStateRepository(tmp_path / "zomah.db")
    repo.initialize()
    return repo


def initial_state() -> ProjectState:
    return ProjectState(
        id="zomah",
        name="ZOMAH",
        phase="state",
        summary="Minimal local agent control plane.",
        current_focus="Implement ProjectState persistence.",
        next_action="Validate transactional updates.",
        open_questions=["How should Markdown export be triggered?"],
        decisions=[
            Decision(
                id="ADR-001",
                statement="ZOMAH is a minimal control plane.",
                rationale="Avoid duplicating worker capabilities.",
            )
        ],
        important_paths=[
            ImportantPath(role="repo", path="~/Projects/ZOMAH")
        ],
        updated_by="zerrius",
    )


def test_create_and_read_project_state(tmp_path: Path):
    repo = make_repo(tmp_path)
    created = repo.create(initial_state())

    assert created.id == "zomah"
    assert created.revision == 0
    assert created.open_questions == ["How should Markdown export be triggered?"]
    assert created.decisions[0].id == "ADR-001"


def test_patch_is_transactional_and_increments_revision(tmp_path: Path):
    repo = make_repo(tmp_path)
    repo.create(initial_state())

    updated = repo.apply_patch(
        "zomah",
        ProjectStatePatch(
            expected_revision=0,
            updated_by="elyria",
            last_action="Implemented the first persistence slice.",
            current_focus="Verify state transitions.",
            add_blockers=["Need local integration test."],
            resolve_open_questions=["How should Markdown export be triggered?"],
        ),
    )

    assert updated.revision == 1
    assert updated.updated_by == "elyria"
    assert updated.open_questions == []
    assert updated.blockers == ["Need local integration test."]


def test_stale_revision_is_rejected(tmp_path: Path):
    repo = make_repo(tmp_path)
    repo.create(initial_state())

    repo.apply_patch(
        "zomah",
        ProjectStatePatch(
            expected_revision=0,
            updated_by="elyria",
            last_action="First mutation.",
        ),
    )

    with pytest.raises(RevisionConflict):
        repo.apply_patch(
            "zomah",
            ProjectStatePatch(
                expected_revision=0,
                updated_by="codex",
                last_action="Stale mutation.",
            ),
        )

    assert repo.get("zomah").last_action == "First mutation."


def test_markdown_export_is_one_way_view(tmp_path: Path):
    repo = make_repo(tmp_path)
    repo.create(initial_state())

    output = repo.export_markdown("zomah", tmp_path / "zomah.md")
    text = output.read_text(encoding="utf-8")

    assert "# ZOMAH" in text
    assert "Revision:** 0" in text
    assert "ADR-001" in text


def test_decision_transition_preserves_record(tmp_path: Path):
    from zomah.state import DecisionStatus, DecisionTransition

    repo = make_repo(tmp_path)
    repo.create(initial_state())

    updated = repo.apply_patch(
        "zomah",
        ProjectStatePatch(
            expected_revision=0,
            updated_by="zerrius",
            transition_decisions=[
                DecisionTransition(
                    decision_id="ADR-001",
                    status=DecisionStatus.SUPERSEDED,
                    superseded_by="ADR-002",
                )
            ],
        ),
    )

    assert updated.decisions[0].status == DecisionStatus.SUPERSEDED
    assert updated.decisions[0].superseded_by == "ADR-002"


def test_nullable_action_can_be_explicitly_cleared(tmp_path: Path):
    repo = make_repo(tmp_path)
    state = initial_state()
    state.last_action = "Something happened."
    repo.create(state)

    updated = repo.apply_patch(
        "zomah",
        ProjectStatePatch(
            expected_revision=0,
            updated_by="zerrius",
            last_action=None,
        ),
    )

    assert updated.last_action is None

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.access import ReadScope
from zomah.capabilities import (
    GetProjectStateRequest,
    InspectSystemRequest,
    ListDirectoryRequest,
    ReadFileRequest,
    get_project_state,
    inspect_system,
    list_directory,
    read_file,
)
from zomah.capabilities.inspect_system import MAX_PROCESS_COMMAND_CHARS
from zomah.state import Decision, DecisionStatus, ProjectState, ProjectStateRepository


def test_read_file_defaults_and_hard_limit_are_model_sized(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    target = root / "large.txt"
    target.write_text("".join(f"{i:04d} " + "x" * 180 + "\n" for i in range(200)), encoding="utf-8")

    request = ReadFileRequest(path=str(target))
    response = read_file(request, ReadScope.from_paths([root]))

    assert request.max_lines == 100
    assert len(response.content) <= 16 * 1024
    assert response.truncated is True
    assert response.next_start_line is not None

    with pytest.raises(ValidationError):
        ReadFileRequest(path=str(target), max_lines=251)


def test_list_directory_default_and_hard_limit_are_model_sized(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    for index in range(101):
        (root / f"{index:03d}.txt").write_text("x", encoding="utf-8")

    request = ListDirectoryRequest(path=str(root))
    response = list_directory(request, ReadScope.from_paths([root]))

    assert request.max_entries == 100
    assert response.returned == 100
    assert response.truncated is True
    assert response.next_offset == 100

    with pytest.raises(ValidationError):
        ListDirectoryRequest(path=str(root), max_entries=251)


def test_inspect_system_default_and_hard_limit_are_model_sized() -> None:
    request = InspectSystemRequest(domain="processes")
    assert request.max_entries == 50

    with pytest.raises(ValidationError):
        InspectSystemRequest(domain="processes", max_entries=101)


def test_process_command_is_truncated_before_return(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    process = proc / "10"
    process.mkdir()
    process.joinpath("status").write_text(
        "Name:\tpython\nPid:\t10\nState:\tS (sleeping)\nUid:\t0\t0\t0\t0\nVmRSS:\t1 kB\n",
        encoding="utf-8",
    )
    process.joinpath("cmdline").write_bytes(b"python\x00" + b"x" * 5000 + b"\x00")

    response = inspect_system(InspectSystemRequest(domain="processes"), proc_root=proc)
    command = response.result.items[0].command

    assert command is not None
    assert len(command) == MAX_PROCESS_COMMAND_CHARS
    assert command.endswith("…")


def test_project_state_decision_history_is_bounded_without_truncating_sqlite(
    tmp_path: Path,
) -> None:
    repo = ProjectStateRepository(tmp_path / "zomah.db")
    repo.initialize()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    decisions = [
        Decision(
            id=f"ADR-{index:03d}",
            statement=f"Decision {index}",
            rationale="Test model context budget.",
            status=(
                DecisionStatus.ACCEPTED
                if index < 25
                else DecisionStatus.INVALIDATED
            ),
            created_at=base + timedelta(days=index),
        )
        for index in range(30)
    ]
    repo.create(
        ProjectState(
            id="zomah",
            name="ZOMAH",
            phase="audit",
            summary="Minimal harness.",
            current_focus="Keep model context bounded.",
            decisions=decisions,
            updated_by="zerrius",
        )
    )

    response = get_project_state(
        GetProjectStateRequest(project_id="zomah"),
        repository=repo,
    )

    assert len(response.project.decisions) == 20
    assert response.project.decision_window.total == 30
    assert response.project.decision_window.returned == 20
    assert response.project.decision_window.truncated is True
    assert response.project.decision_window.by_status[DecisionStatus.ACCEPTED] == 25
    assert response.project.decision_window.by_status[DecisionStatus.INVALIDATED] == 5
    assert len(repo.get("zomah").decisions) == 30

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Static

import zomah.console.builtin_commands as builtin_commands
from zomah.console import (
    CommandRegistry,
    CommandResult,
    Composer,
    ConsoleCommand,
    ConsoleStatus,
    OperatorConsole,
)
from zomah.console.operator import OperatorAccess
from zomah.state import Decision, DecisionStatus, ProjectState, ProjectStateRepository
from zomah.tracing import TraceOutcome, TraceStore

Scenario = Callable[[OperatorConsole, Pilot], Awaitable[None]]


@pytest.fixture
def repository(tmp_path: Path) -> ProjectStateRepository:
    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()
    return repository


@pytest.fixture
def access(tmp_path: Path, repository: ProjectStateRepository) -> OperatorAccess:
    return OperatorAccess(
        operator_id="zerrius",
        trace_store=TraceStore(tmp_path / "trace.db"),
        project_repository=repository,
    )


def create_project(repository: ProjectStateRepository, **overrides: Any) -> None:
    fields: dict[str, Any] = {
        "id": "zomah",
        "name": "ZOMAH",
        "phase": "operator-console-foundation",
        "summary": "Verified local control-plane substrate.",
        "current_focus": "Human capability boundary.",
        "next_action": "Accept T0d2.",
        "blockers": ["one blocker"],
        "open_questions": ["q1", "q2"],
        "updated_by": "zerrius",
    }
    fields.update(overrides)
    repository.create(ProjectState(**fields))


def run_console(scenario: Scenario, **kwargs: Any) -> None:
    async def runner() -> None:
        app = OperatorConsole(**kwargs)
        async with app.run_test(size=(100, 40)) as pilot:
            await scenario(app, pilot)

    asyncio.run(runner())


async def submit(pilot: Pilot, text: str) -> None:
    pilot.app.query_one(Composer).load_text(text)
    await pilot.press("escape", "enter")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def results(app: OperatorConsole) -> list[str]:
    return [str(entry.content) for entry in app.query(".entry.result").results(Static)]


def test_no_active_project_id_invokes_nothing(access: OperatorAccess) -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        assert results(app) == [
            "zomah · Project\nNo active canonical project is configured."
        ]
        assert access.trace_store.recent() == []

    run_console(scenario, operator_access=access)


def test_active_project_without_operator_access_is_an_error() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        assert results(app) == [
            "zomah · Project unavailable\n"
            "This console was started without capability access."
        ]
        assert app.query_one(".entry.result").has_class("error")

    run_console(scenario, status=ConsoleStatus(active_project_id="zomah"))


def test_project_reads_canonical_state_through_user_boundary(
    monkeypatch: pytest.MonkeyPatch,
    repository: ProjectStateRepository,
    access: OperatorAccess,
) -> None:
    create_project(repository)
    boundary_calls: list[tuple[Any, ...]] = []
    original = builtin_commands.invoke_registered_user_capability

    def spy(*args: Any, **kwargs: Any):
        boundary_calls.append((args[1], args[2], kwargs["operator_id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(builtin_commands, "invoke_registered_user_capability", spy)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        (output,) = results(app)
        lines = [" ".join(line.split()) for line in output.split("\n")]
        assert lines[:12] == [
            "zomah · Project",
            "ZOMAH (zomah)",
            "Status: active",
            "Phase: operator-console-foundation",
            "Revision: 0",
            "Summary: Verified local control-plane substrate.",
            "Focus: Human capability boundary.",
            "Last action: none recorded",
            "Next action: Accept T0d2.",
            "Blockers: 1",
            "Open questions: 2",
            "Decisions: 0 total",
        ]
        assert lines[12].startswith("Updated: ") and lines[12].endswith(" by zerrius")
        assert not app.query_one(".entry.result").has_class("error")

    run_console(scenario, status=ConsoleStatus(active_project_id="zomah"), operator_access=access)

    assert boundary_calls == [("get_project_state", {"project_id": "zomah"}, "zerrius")]
    (record,) = access.trace_store.recent()
    assert (record.worker, record.capability, record.target, record.outcome) == (
        "operator:zerrius",
        "get_project_state",
        "project:zomah",
        TraceOutcome.SUCCEEDED,
    )


def test_project_output_renders_literally(
    repository: ProjectStateRepository, access: OperatorAccess
) -> None:
    create_project(
        repository,
        name="[bold]ZOMAH[/bold]",
        summary="[red]not markup[/red]\nsecond line",
    )

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        output = results(app)[0]
        assert "[bold]ZOMAH[/bold] (zomah)" in output
        assert "[red]not markup[/red]\nsecond line" in output

    run_console(scenario, status=ConsoleStatus(active_project_id="zomah"), operator_access=access)


def test_missing_project_renders_normalized_error(access: OperatorAccess) -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        assert results(app) == [
            "zomah · Project unavailable\n"
            "Project was not found: nope\n"
            "Code: project_not_found"
        ]
        assert app.query_one(".entry.result").has_class("error")

    run_console(scenario, status=ConsoleStatus(active_project_id="nope"), operator_access=access)

    (record,) = access.trace_store.recent()
    assert (record.worker, record.outcome, record.error_code) == (
        "operator:zerrius",
        TraceOutcome.FAILED,
        "project_not_found",
    )


def test_uninitialized_storage_renders_redacted_retryable_error(tmp_path: Path) -> None:
    access = OperatorAccess(
        operator_id="zerrius",
        trace_store=TraceStore(tmp_path / "trace.db"),
        project_repository=ProjectStateRepository(tmp_path / "empty.db"),
    )

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        assert results(app) == [
            "zomah · Project unavailable\n"
            "Local SQLite storage is temporarily unavailable.\n"
            "Code: storage_unavailable\n"
            "This may succeed if retried."
        ]

    run_console(scenario, status=ConsoleStatus(active_project_id="zomah"), operator_access=access)


def test_truncated_decision_history_is_explained(
    repository: ProjectStateRepository, access: OperatorAccess
) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    create_project(
        repository,
        decisions=[
            Decision(
                id=f"d{index}",
                statement=f"Decision {index}",
                rationale="Because.",
                status=DecisionStatus.ACCEPTED if index % 2 else DecisionStatus.SUPERSEDED,
                superseded_by=None if index % 2 else "d1",
                created_at=start + timedelta(minutes=index),
            )
            for index in range(25)
        ],
    )

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        lines = [" ".join(line.split()) for line in results(app)[0].split("\n")]
        assert "Decisions: 25 total (12 accepted, 13 superseded)" in lines
        assert lines[-1] == (
            "Decision history is truncated: this view received 20 of 25 decisions. "
            "Canonical history is complete in storage."
        )

    run_console(scenario, status=ConsoleStatus(active_project_id="zomah"), operator_access=access)


def test_background_command_does_not_block_ui_and_keeps_transcript_order() -> None:
    release = threading.Event()
    started = threading.Event()

    def slow(context: Any, arguments: str) -> CommandResult:
        started.set()
        assert release.wait(timeout=10)
        return CommandResult("Slow", ("done",))

    registry = CommandRegistry()
    registry.register(ConsoleCommand("/slow", "Blocking I/O stand-in.", slow, background=True))

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        composer.load_text("/slow")
        await pilot.press("escape", "enter")
        await pilot.pause()
        await asyncio.to_thread(started.wait, 10)

        # The handler is still blocked, yet the UI keeps processing input.
        pending = app.query_one(".entry.result.pending", Static)
        assert str(pending.content) == "zomah · /slow running…"
        await pilot.press(*"hi", "enter")
        await pilot.pause()
        operators = [
            str(entry.content) for entry in app.query(".entry.operator").results(Static)
        ]
        assert operators == ["operator\n/slow", "operator\nhi"]

        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        entries = [
            str(entry.content).split("\n")[0]
            for entry in app.query("#transcript .entry").results(Static)
        ][1:]
        assert entries == ["operator", "zomah · Slow", "operator"]
        assert not app.query(".pending")

    try:
        run_console(scenario, commands=registry)
    finally:
        release.set()

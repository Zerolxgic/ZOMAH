"""One session-state owner for the header, /status, and /project."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Static

import zomah.console.builtin_commands as builtin_commands
from zomah.console import Composer, ConsoleStatus, OperatorConsole
from zomah.console.app import build_console
from zomah.console.builtin_commands import default_command_registry, project_view
from zomah.console.commands import CommandContext
from zomah.console.operator import OperatorAccess
from zomah.console.status import STATUS_LABELS, project_display
from zomah.capability_registry import default_capability_registry
from zomah.state import ProjectState, ProjectStateRepository
from zomah.tracing import TraceStore

Scenario = Callable[[OperatorConsole, Pilot], Awaitable[None]]


@pytest.fixture
def access(tmp_path: Path) -> OperatorAccess:
    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()
    repository.create(
        ProjectState(
            id="zomah",
            name="ZOMAH",
            phase="p",
            summary="s",
            current_focus="f",
            updated_by="zerrius",
        )
    )
    return OperatorAccess(
        operator_id="zerrius",
        trace_store=TraceStore(tmp_path / "trace.db"),
        project_repository=repository,
    )


def run_console(scenario: Scenario, **kwargs: Any) -> None:
    async def runner() -> None:
        app = OperatorConsole(**kwargs)
        async with app.run_test(size=(100, 40)) as pilot:
            await scenario(app, pilot)

    asyncio.run(runner())


async def submit(pilot: Pilot, text: str) -> str:
    app = pilot.app
    app.query_one(Composer).load_text(text)
    await pilot.press("escape", "enter")
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()
    return str(list(app.query(".entry.result").results(Static))[-1].content)


def header(app: OperatorConsole) -> dict[str, str]:
    return {
        key: str(app.query_one(f"#field-{key}", Static).content).split(": ", 1)[1]
        for key in STATUS_LABELS
    }


def status_command_values(output: str) -> dict[str, str]:
    rows = output.split("\n")[1:]
    by_label = {row.split(":", 1)[0]: row.split(":", 1)[1].strip() for row in rows}
    return {key: by_label[label] for key, label in STATUS_LABELS.items()}


def test_project_display_is_deterministic() -> None:
    assert project_display(ConsoleStatus()) == "not set"
    assert project_display(ConsoleStatus(active_project_id="zomah")) == (
        "zomah (not read yet)"
    )
    assert project_display(ConsoleStatus(active_project_id="zomah", project="ZOMAH")) == (
        "ZOMAH (zomah)"
    )
    assert project_display(ConsoleStatus(project="~/work/zomah")) == "~/work/zomah"


def test_console_has_no_second_active_project_input() -> None:
    assert "active_project_id" not in inspect.signature(OperatorConsole).parameters
    assert "active_project_id" not in {
        field for field in CommandContext.__dataclass_fields__
    }


def test_header_and_status_command_render_the_same_session_state() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        for status in (
            ConsoleStatus(),
            ConsoleStatus(active_project_id="zomah"),
            ConsoleStatus(
                active_project_id="zomah",
                project="ZOMAH",
                zomah_state="ready",
                model="local-model",
                tools_available=2,
                context_used=10,
                context_limit=100,
            ),
        ):
            app.set_status(status)
            await pilot.pause()
            assert app.status is status
            assert status_command_values(await submit(pilot, "/status")) == header(app)

    run_console(scenario)


def test_unset_session_fields_keep_placeholders() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        expected = {
            "project": "zomah (not read yet)",
            "state": "not connected",
            "model": "not connected",
            "tools": "unavailable",
            "context": "unavailable",
        }
        assert header(app) == expected
        assert status_command_values(await submit(pilot, "/status")) == expected

    run_console(scenario, status=ConsoleStatus(active_project_id="zomah"))


def test_project_option_establishes_session_project_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    configured = build_console(["--project", "zomah", "--operator", "zerrius"])
    assert configured.status == ConsoleStatus(active_project_id="zomah")

    unconfigured = build_console(["--operator", "zerrius"])
    assert unconfigured.status == ConsoleStatus()

    async def scenario() -> None:
        async with configured.run_test(size=(100, 40)) as pilot:
            assert header(configured)["project"] == "zomah (not read yet)"
            await pilot.pause()

    asyncio.run(scenario())


def test_project_command_reads_id_from_session_and_labels_from_canonical_state(
    monkeypatch: pytest.MonkeyPatch, access: OperatorAccess
) -> None:
    boundary_calls: list[tuple[Any, ...]] = []
    original = builtin_commands.invoke_registered_user_capability

    def spy(*args: Any, **kwargs: Any):
        boundary_calls.append((args[1], args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(builtin_commands, "invoke_registered_user_capability", spy)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert "No active canonical project" in await submit(pilot, "/project")
        assert boundary_calls == []

        # Runtime session update is the only way /project learns the id.
        app.set_status(ConsoleStatus(active_project_id="zomah", model="local-model"))
        await pilot.pause()
        assert header(app)["project"] == "zomah (not read yet)"

        output = await submit(pilot, "/project")
        assert output.split("\n")[1] == "ZOMAH (zomah)"
        assert boundary_calls == [("get_project_state", {"project_id": "zomah"})]

        # Display metadata updated from canonical state; id and other fields kept.
        assert app.status == ConsoleStatus(
            active_project_id="zomah", project="ZOMAH", model="local-model"
        )
        assert header(app)["project"] == "ZOMAH (zomah)"
        assert status_command_values(await submit(pilot, "/status")) == header(app)

    run_console(scenario, operator_access=access)

    (record,) = access.trace_store.recent()
    assert (record.worker, record.capability, record.target) == (
        "operator:zerrius",
        "get_project_state",
        "project:zomah",
    )


def test_failed_project_read_leaves_session_state_unchanged(
    access: OperatorAccess,
) -> None:
    initial = ConsoleStatus(active_project_id="missing")

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert "Project was not found: missing" in await submit(pilot, "/project")
        assert app.status is initial
        assert header(app)["project"] == "missing (not read yet)"

    run_console(scenario, status=initial, operator_access=access)


def test_canonical_label_is_not_applied_if_active_project_changed(
    access: OperatorAccess,
) -> None:
    context = CommandContext(
        status=ConsoleStatus(active_project_id="zomah"),
        commands=default_command_registry(),
        capabilities=default_capability_registry(),
        operator_access=access,
    )
    result = project_view(context, "")
    assert result.status_update is not None

    changed = ConsoleStatus(active_project_id="other")
    assert result.status_update(changed) is changed
    assert result.status_update(ConsoleStatus(active_project_id="zomah")) == (
        ConsoleStatus(active_project_id="zomah", project="ZOMAH")
    )


def test_registry_agent_exposure_is_not_live_tool_count(access: OperatorAccess) -> None:
    assert any(d.agent_exposed for d in default_capability_registry().all())

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        await submit(pilot, "/tools")
        assert app.status.tools_available is None
        assert header(app)["tools"] == "unavailable"

    run_console(
        scenario,
        status=ConsoleStatus(active_project_id="zomah"),
        operator_access=access,
    )


def test_background_read_touches_session_and_widgets_only_on_ui_thread(
    monkeypatch: pytest.MonkeyPatch, access: OperatorAccess
) -> None:
    ui_thread = threading.get_ident()
    handler_threads: list[int] = []
    set_status_threads: list[int] = []
    original = builtin_commands.invoke_registered_user_capability

    def spy(*args: Any, **kwargs: Any):
        handler_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(builtin_commands, "invoke_registered_user_capability", spy)
    original_set_status = OperatorConsole.set_status

    def recording_set_status(self: OperatorConsole, status: ConsoleStatus) -> None:
        set_status_threads.append(threading.get_ident())
        original_set_status(self, status)

    monkeypatch.setattr(OperatorConsole, "set_status", recording_set_status)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        assert header(app)["project"] == "ZOMAH (zomah)"

    run_console(
        scenario,
        status=ConsoleStatus(active_project_id="zomah"),
        operator_access=access,
    )

    assert len(handler_threads) == 1 and handler_threads[0] != ui_thread
    assert set_status_threads == [ui_thread]

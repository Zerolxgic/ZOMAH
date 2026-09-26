from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from textual import events
from textual.pilot import Pilot
from textual.widgets import Static

from zomah.console import Composer, ConsoleStatus, OperatorConsole

Scenario = Callable[[OperatorConsole, Pilot], Awaitable[None]]


def run_console(
    scenario: Scenario,
    *,
    size: tuple[int, int] = (80, 24),
    status: ConsoleStatus | None = None,
) -> None:
    async def runner() -> None:
        app = OperatorConsole(status)
        async with app.run_test(size=size) as pilot:
            await scenario(app, pilot)

    asyncio.run(runner())


def field_text(app: OperatorConsole, name: str) -> str:
    return str(app.query_one(f"#field-{name}", Static).content)


def transcript_entries(app: OperatorConsole) -> list[str]:
    return [str(entry.content) for entry in app.query(".entry.operator").results(Static)]


def test_header_renders_explicit_placeholders_without_runtime_state() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert "ZOMAH" in str(app.query_one("#header-identity", Static).content)
        assert field_text(app, "project") == "Project: not set"
        assert field_text(app, "state") == "State: not connected"
        assert field_text(app, "model") == "Model: not connected"
        assert field_text(app, "tools") == "Tools: unavailable"
        assert field_text(app, "context") == "Context: unavailable"

    run_console(scenario)


def test_header_renders_caller_supplied_status() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        app.set_status(
            ConsoleStatus(
                project="zomah",
                zomah_state="ready",
                model="local-model",
                tools_available=3,
                context_used=1200,
                context_limit=32768,
            )
        )
        await pilot.pause()
        assert field_text(app, "project") == "Project: zomah"
        assert field_text(app, "state") == "State: ready"
        assert field_text(app, "model") == "Model: local-model"
        assert field_text(app, "tools") == "Tools: 3"
        assert field_text(app, "context") == "Context: 1200 / 32768"

    run_console(scenario)


def test_composer_is_focused_on_start() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert app.focused is app.query_one(Composer)

    run_console(scenario)


def test_enter_submits_to_transcript_and_composer_stays_usable() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"hello", "enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\nhello"]
        assert composer.text == ""
        assert app.focused is composer

        await pilot.press(*"again", "enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\nhello", "operator\nagain"]
        assert composer.text == ""

    run_console(scenario)


def test_submission_starts_fresh_edit_history() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"sent", "enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\nsent"]
        assert composer.text == ""

        await pilot.press("ctrl+z")
        assert composer.text == ""
        await pilot.press("ctrl+y")
        assert composer.text == ""

        await pilot.press(*"next", "ctrl+z")
        assert composer.text == ""

    run_console(scenario)


def test_shift_enter_inserts_newline_without_submitting() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press("a", "shift+enter", "b")
        assert composer.text == "a\nb"
        assert composer.cursor_location == (1, 1)
        assert transcript_entries(app) == []

        await pilot.press("enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\na\nb"]

    run_console(scenario)


def test_shift_enter_replaces_selection() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"abc", "shift+left", "shift+enter")
        assert composer.text == "ab\n"

    run_console(scenario)


def test_blank_submission_is_ignored() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press("space", "shift+enter", "space", "enter")
        await pilot.pause()
        assert transcript_entries(app) == []
        assert composer.text == " \n "

    run_console(scenario)


def test_ctrl_a_selects_all_and_selection_is_replaced_or_deleted() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"one", "shift+enter", *"two")
        await pilot.press("ctrl+a")
        assert composer.selected_text == "one\ntwo"

        await pilot.press("x")
        assert composer.text == "x"

        await pilot.press(*"yz", "ctrl+a", "backspace")
        assert composer.text == ""

    run_console(scenario)


def test_home_still_moves_to_line_start() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"abc", "home")
        assert composer.cursor_location == (0, 0)

    run_console(scenario)


def test_undo_and_redo() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"draft")
        await pilot.press("ctrl+z")
        assert composer.text == ""
        await pilot.press("ctrl+y")
        assert composer.text == "draft"

    run_console(scenario)


def test_copy_and_paste_use_text_area_behavior() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"copy me", "ctrl+a", "ctrl+c")
        assert app.clipboard == "copy me"

        await pilot.press("end", "space", "ctrl+v")
        assert composer.text == "copy me copy me"

    run_console(scenario)


def test_terminal_paste_inserts_multiline_text_without_submitting() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        # Terminal (bracketed) paste arrives at the App, which forwards it
        # to the focused widget, as the driver does in a real session.
        app.post_message(events.Paste("line one\nline two"))
        await pilot.pause()
        assert composer.text == "line one\nline two"
        assert transcript_entries(app) == []

    run_console(scenario)


def test_submitted_text_is_rendered_literally_not_as_markup() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press(*"[bold]x[/bold]", "enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\n[bold]x[/bold]"]

    run_console(scenario)


def test_layout_keeps_three_regions_ordered_across_resizes() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        for width, height in ((80, 24), (40, 12), (120, 40), (80, 24)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            header = app.query_one("#header").region
            transcript = app.query_one("#transcript").region
            composer = app.query_one(Composer).region

            assert header.y == 0
            assert header.bottom <= transcript.y
            assert transcript.bottom <= composer.y
            assert composer.bottom == height
            assert transcript.height >= 1
            assert composer.height >= 3
            for region in (header, transcript, composer):
                assert region.width == width

    run_console(scenario)

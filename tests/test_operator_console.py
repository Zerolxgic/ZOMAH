from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from textual import events
from textual.pilot import Pilot
from textual.widgets import Static

from zomah.capability_registry import CapabilityRegistry
from zomah.console import (
    CommandRegistry,
    CommandSuggestions,
    Composer,
    CommandResult,
    ConsoleCommand,
    ConsoleStatus,
    OperatorConsole,
)

Scenario = Callable[[OperatorConsole, Pilot], Awaitable[None]]


def run_console(
    scenario: Scenario,
    *,
    size: tuple[int, int] = (80, 24),
    status: ConsoleStatus | None = None,
    commands: CommandRegistry | None = None,
    capabilities: CapabilityRegistry | None = None,
) -> None:
    async def runner() -> None:
        app = OperatorConsole(status, commands, capabilities)
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


def suggestion_names(app: OperatorConsole) -> list[str]:
    suggestions = app.query_one(CommandSuggestions)
    if not suggestions.display:
        return []
    return [
        suggestions.get_option_at_index(index).id
        for index in range(suggestions.option_count)
    ]


def highlighted_name(app: OperatorConsole) -> str | None:
    return app.query_one(CommandSuggestions).highlighted_name


def test_slash_shows_all_commands_alphabetically_with_descriptions() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert suggestion_names(app) == []
        await pilot.press("slash")
        await pilot.pause()
        assert suggestion_names(app) == ["/help", "/project", "/status", "/tools"]
        assert highlighted_name(app) == "/help"
        prompt = str(app.query_one(CommandSuggestions).get_option_at_index(1).prompt)
        assert "/project" in prompt
        assert "active project" in prompt

    run_console(scenario)


def test_prefix_filters_commands_in_alphabetical_order() -> None:
    registry = CommandRegistry()
    for name in ("/stop", "/status", "/help", "/save"):
        registry.register(ConsoleCommand(name, f"{name} description"))

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("slash", "s")
        await pilot.pause()
        assert suggestion_names(app) == ["/save", "/status", "/stop"]
        await pilot.press("t")
        await pilot.pause()
        assert suggestion_names(app) == ["/status", "/stop"]
        await pilot.press("backspace", "backspace")
        await pilot.pause()
        assert suggestion_names(app) == ["/help", "/save", "/status", "/stop"]

    run_console(scenario, commands=registry)


def test_default_prefix_p_shows_only_project() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("slash", "p")
        await pilot.pause()
        assert suggestion_names(app) == ["/project"]

    run_console(scenario)


def test_no_matching_command_hides_suggestions_and_enter_submits() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("slash", "x")
        await pilot.pause()
        assert suggestion_names(app) == []
        await pilot.press("enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\n/x"]

    run_console(scenario)


def test_up_and_down_move_highlight_without_moving_cursor() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press("slash")
        await pilot.pause()
        await pilot.press("down", "down")
        assert highlighted_name(app) == "/status"
        await pilot.press("up")
        assert highlighted_name(app) == "/project"
        assert composer.text == "/"
        assert composer.cursor_location == (0, 1)

    run_console(scenario)


def test_up_and_down_still_move_cursor_without_suggestions() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press("a", "shift+enter", "b", "up")
        assert composer.cursor_location == (0, 1)
        await pilot.press("down")
        assert composer.cursor_location == (1, 1)

    run_console(scenario)


def test_enter_completes_highlighted_command_without_submitting() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press("slash")
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert composer.text == "/project"
        assert composer.cursor_location == (0, len("/project"))
        assert suggestion_names(app) == []
        assert transcript_entries(app) == []

        await pilot.press("enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\n/project"]
        assert composer.text == ""

    run_console(scenario)


def test_fast_typing_then_enter_uses_current_text_not_stale_suggestions() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        # No pause between keys: Enter must see "/t", not an earlier state.
        await pilot.press("slash", "t", "enter")
        await pilot.pause()
        assert composer.text == "/tools"
        assert transcript_entries(app) == []

    run_console(scenario)


def test_escape_dismisses_suggestions_and_preserves_text() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press("slash", "s")
        await pilot.pause()
        assert suggestion_names(app) == ["/status"]
        await pilot.press("escape")
        await pilot.pause()
        assert suggestion_names(app) == []
        assert composer.text == "/s"
        assert app.focused is composer

        await pilot.press("enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\n/s"]

    run_console(scenario)


def test_editing_after_escape_reevaluates_suggestions() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("slash", "escape", "t")
        await pilot.pause()
        assert suggestion_names(app) == ["/tools"]

    run_console(scenario)


def test_suggestions_disappear_when_text_stops_being_a_command_prefix() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("slash", "h")
        await pilot.pause()
        assert suggestion_names(app) == ["/help"]
        await pilot.press("space")
        await pilot.pause()
        assert suggestion_names(app) == []

        await pilot.press("ctrl+a", *"hi /")
        await pilot.pause()
        assert suggestion_names(app) == []

    run_console(scenario)


def test_ctrl_j_inserts_newline_without_submitting() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press("a", "ctrl+j", "b")
        assert composer.text == "a\nb"
        assert composer.cursor_location == (1, 1)
        assert transcript_entries(app) == []

    run_console(scenario)


def test_ctrl_j_replaces_selection() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"abc", "shift+left", "ctrl+j")
        assert composer.text == "ab\n"
        await pilot.press("ctrl+a", "ctrl+j")
        assert composer.text == "\n"

    run_console(scenario)


def test_composer_stays_on_screen_with_suggestions_in_small_terminals() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("slash")
        await pilot.pause()
        for width, height in ((80, 24), (60, 14), (40, 12), (40, 10), (80, 24)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            header = app.query_one("#header").region
            suggestions = app.query_one(CommandSuggestions).region
            composer = app.query_one(Composer).region

            # The list may cover the transcript, never the header or composer.
            assert header.y == 0
            assert header.bottom <= suggestions.y
            assert suggestions.bottom == composer.y
            assert suggestions.height >= 2
            assert composer.bottom == height
            assert composer.height >= 3

        assert suggestion_names(app) == ["/help", "/project", "/status", "/tools"]

    run_console(scenario)


def test_highlighted_suggestion_scrolls_into_view_when_list_is_clipped() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        suggestions = app.query_one(CommandSuggestions)
        await pilot.press("slash")
        await pilot.pause()
        await pilot.press("down", "down", "down")
        await pilot.pause()
        assert highlighted_name(app) == "/tools"
        visible_rows = suggestions.scrollable_content_region.height
        assert visible_rows < 4
        assert suggestions.scroll_y + visible_rows >= 4

    run_console(scenario, size=(40, 12))


def result_entries(app: OperatorConsole) -> list[str]:
    return [str(entry.content) for entry in app.query(".entry.result").results(Static)]


def all_entries(app: OperatorConsole) -> list[tuple[str, str]]:
    entries = []
    for entry in app.query("#transcript .entry").results(Static):
        kind = "operator" if entry.has_class("operator") else (
            "result" if entry.has_class("result") else "notice"
        )
        entries.append((kind, str(entry.content).split("\n")[0]))
    return entries


async def submit(pilot: Pilot, text: str) -> None:
    pilot.app.query_one(Composer).load_text(text)
    await pilot.press("escape", "enter")
    await pilot.pause()


def test_ordinary_text_produces_no_command_result() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "hello there")
        assert transcript_entries(app) == ["operator\nhello there"]
        assert result_entries(app) == []

    run_console(scenario)


def test_command_renders_operator_history_then_distinct_result() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/help")
        assert all_entries(app) == [
            ("notice", "No model connected. Messages are shown here only; type / for commands."),
            ("operator", "operator"),
            ("result", "zomah · Help"),
        ]
        help_text = result_entries(app)[0]
        for name in ("/help", "/project", "/status", "/tools"):
            assert name in help_text
        assert "Ctrl+J" in help_text and "newline fallback" in help_text

    run_console(scenario)


def test_status_command_reads_current_app_status() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/status")
        assert "Model:    not connected" in result_entries(app)[0]
        app.set_status(ConsoleStatus(model="local-model"))
        await submit(pilot, "/status")
        assert "Model:    local-model" in result_entries(app)[1]
        assert field_text(app, "tools") == "Tools: unavailable"

    run_console(scenario)


def test_project_command_without_active_project_id_reports_none_configured() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/project")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert result_entries(app) == [
            "zomah · Project\nNo active canonical project is configured."
        ]

    run_console(scenario, status=ConsoleStatus(project="zomah"))


def test_tools_command_without_session_keeps_header_count_unavailable() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/tools")
        output = result_entries(app)[0]
        assert "get_project_state" in output
        assert "No model session" in output
        assert field_text(app, "tools") == "Tools: unavailable"

    run_console(scenario)


def test_unknown_command_is_an_error_not_ordinary_input() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/whatever")
        assert result_entries(app) == [
            "zomah · Unknown command: /whatever\n"
            "Type / to browse commands, or use /help."
        ]
        assert app.query_one(".entry.result").has_class("error")

    run_console(scenario)


def test_unsupported_arguments_produce_usage_error() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/status now")
        assert result_entries(app) == [
            "zomah · /status does not take arguments.\nUsage: /status"
        ]
        assert app.query_one(".entry.result").has_class("error")

    run_console(scenario)


def test_command_output_renders_literally() -> None:
    registry = CommandRegistry()
    registry.register(
        ConsoleCommand(
            "/probe",
            "[bold]markup[/bold] description",
            lambda ctx, args: CommandResult("[red]title[/red]", ("[link=x]line[/link]",)),
        )
    )

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/probe")
        assert result_entries(app) == [
            "zomah · [red]title[/red]\n[link=x]line[/link]"
        ]

    run_console(scenario, commands=registry)


def test_status_values_render_literally() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await submit(pilot, "/status")
        assert "Project:  [b]p[/b]" in result_entries(app)[0]

    run_console(scenario, status=ConsoleStatus(project="[b]p[/b]"))


def test_completed_command_submits_and_routes() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("slash", "s")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert transcript_entries(app) == ["operator\n/status"]
        assert result_entries(app)[0].startswith("zomah · Status")

    run_console(scenario)

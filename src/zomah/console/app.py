"""Operator Console shell.

The console is an interface into ZOMAH, not a source of ZOMAH state. The
header renders a ``ConsoleStatus`` supplied by its caller; any field that has
no live runtime source yet is rendered as an explicit placeholder rather than
an invented value. Submitted composer text is only echoed into the transcript:
no model is invoked. Submitted slash commands are routed to console views
(``builtin_commands``) that render from supplied state and registry metadata;
no capability handler is invoked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.text import Text
from textual import events
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from zomah import __version__
from zomah.console.commands import (
    CommandRegistry,
    ConsoleCommand,
    is_command_prefix,
)
from zomah.capability_registry import CapabilityRegistry, default_capability_registry
from zomah.console.builtin_commands import default_command_registry
from zomah.console.commands import CommandContext, CommandResult
from zomah.console.routing import CommandSubmission, parse_submission, run_command
from zomah.console.status import STATUS_LABELS, ConsoleStatus, status_values

class CommandSuggestions(OptionList):
    """Slash-command discovery list. Never focused; the composer drives it."""

    can_focus = False

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._names: tuple[str, ...] = ()
        self.border_title = "Commands"
        self.display = False

    @property
    def active(self) -> bool:
        return self.display and bool(self._names)

    @property
    def highlighted_name(self) -> str | None:
        if not self.active or self.highlighted is None:
            return None
        return self._names[self.highlighted]

    def show(self, commands: Sequence[ConsoleCommand]) -> None:
        names = tuple(command.name for command in commands)
        if names != self._names:
            width = max(len(name) for name in names)
            self.set_options(
                Option(
                    Text.assemble(
                        (command.name.ljust(width), "bold"),
                        "  ",
                        (command.description, "dim"),
                    ),
                    id=command.name,
                )
                for command in commands
            )
            self._names = names
            self.highlighted = 0
        self.display = True

    def hide(self) -> None:
        self.display = False
        self._names = ()
        self.clear_options()


class Composer(TextArea):
    """Multiline message editor with slash-command discovery.

    Editing, selection, clipboard, undo, and redo are TextArea's own behavior.
    Composer changes only: Enter submits (or completes a highlighted command),
    Shift+Enter / Ctrl+J insert a newline, Ctrl+A selects all, and Up / Down /
    Esc drive the command suggestions while they are shown.
    """

    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all", show=False),
        Binding("shift+enter,ctrl+j", "newline", "Newline", show=False),
        Binding("escape", "dismiss_suggestions", "Dismiss", show=False),
    ]

    class Submitted(Message):
        """Posted when the operator submits non-blank composer text."""

        def __init__(self, composer: Composer, text: str) -> None:
            super().__init__()
            self.composer = composer
            self.text = text

    def __init__(
        self,
        *,
        commands: CommandRegistry,
        suggestions: CommandSuggestions,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._commands = commands
        self._suggestions = suggestions
        # Text for which the operator dismissed suggestions (Esc or completion).
        self._dismissed_text: str | None = None

    def on_key(self, event: events.Key) -> None:
        # TextArea inserts a newline for Enter before bindings are consulted,
        # so Enter is claimed here; prevent_default skips TextArea's handler.
        if event.key != "enter" or self.read_only:
            return
        event.stop()
        event.prevent_default()
        self.refresh_suggestions()
        name = self._suggestions.highlighted_name
        if name is not None:
            self.complete_command(name)
        else:
            self.submit()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self.refresh_suggestions()

    def refresh_suggestions(self) -> None:
        """Re-derive the suggestion list from the current composer text."""

        text = self.text
        if self._dismissed_text is not None and text != self._dismissed_text:
            self._dismissed_text = None
        matches = (
            self._commands.matching(text)
            if is_command_prefix(text) and self._dismissed_text is None
            else ()
        )
        if matches:
            self._suggestions.show(matches)
        else:
            self._suggestions.hide()

    def complete_command(self, name: str) -> None:
        result = self.replace(
            name, (0, 0), self.document.end, maintain_selection_offset=False
        )
        self.move_cursor(result.end_location)
        self._dismissed_text = name
        self._suggestions.hide()

    def action_newline(self) -> None:
        if self.read_only:
            return
        result = self.replace("\n", *self.selection, maintain_selection_offset=False)
        self.move_cursor(result.end_location)

    def action_dismiss_suggestions(self) -> None:
        self.refresh_suggestions()
        if not self._suggestions.active:
            raise SkipAction()
        self._dismissed_text = self.text
        self._suggestions.hide()

    def action_cursor_up(self, select: bool = False) -> None:
        self.refresh_suggestions()
        if self._suggestions.active and not select:
            self._suggestions.action_cursor_up()
        else:
            super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        self.refresh_suggestions()
        if self._suggestions.active and not select:
            self._suggestions.action_cursor_down()
        else:
            super().action_cursor_down(select)

    def submit(self) -> None:
        text = self.text
        if not text.strip():
            return
        # load_text clears edit history: a sent draft is not undoable.
        self.load_text("")
        self.post_message(self.Submitted(self, text))


class OperatorConsole(App[None]):
    """Three-region ZOMAH console: header, transcript, composer."""

    TITLE = "ZOMAH Operator Console"

    CSS = """
    Screen {
        layout: vertical;
    }
    #header {
        height: auto;
        padding: 0 1;
        border-bottom: solid $primary;
    }
    #header-identity {
        text-style: bold;
    }
    .header-row {
        height: 1;
    }
    .header-row > Static {
        width: 1fr;
        height: 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    .header-row > .right {
        text-align: right;
    }
    #work {
        height: 1fr;
    }
    #transcript {
        height: 1fr;
        min-height: 1;
        padding: 0 1;
    }
    .entry {
        margin-bottom: 1;
    }
    .result {
        border-left: outer $accent;
        padding-left: 1;
    }
    .result.error {
        border-left: outer $error;
    }
    #suggestions {
        height: auto;
        max-height: 100%;
        dock: bottom;
        text-wrap: nowrap;
        text-overflow: ellipsis;
        border: round $primary;
        padding: 0 1;
    }
    #composer {
        height: auto;
        min-height: 3;
        max-height: 12;
    }
    """

    def __init__(
        self,
        status: ConsoleStatus | None = None,
        commands: CommandRegistry | None = None,
        capabilities: CapabilityRegistry | None = None,
    ) -> None:
        super().__init__()
        self._status = status or ConsoleStatus()
        self._commands = commands or default_command_registry()
        # Read for /tools metadata only; the console never invokes handlers.
        self._capabilities = capabilities or default_capability_registry()

    @property
    def status(self) -> ConsoleStatus:
        return self._status

    def compose(self) -> ComposeResult:
        values = status_values(self._status)
        with Vertical(id="header"):
            yield Static(f"ZOMAH {__version__}", id="header-identity")
            for left, right in (("project", "state"), ("model", "tools"), ("context", None)):
                with Horizontal(classes="header-row"):
                    yield self._field(left, values[left])
                    if right is None:
                        yield Static("", classes="right")
                    else:
                        yield self._field(right, values[right], classes="right")
        # Transcript and suggestions share the flexible region, so a shown
        # command list shrinks (and scrolls) before the composer is displaced.
        suggestions = CommandSuggestions(id="suggestions")
        with Vertical(id="work"):
            yield VerticalScroll(
                Static(
                    "No model connected. Messages are shown here only; type / for commands.",
                    classes="entry notice",
                ),
                id="transcript",
            )
            yield suggestions
        yield Composer(
            commands=self._commands,
            suggestions=suggestions,
            id="composer",
            soft_wrap=True,
            show_line_numbers=False,
        )

    @staticmethod
    def _field(name: str, value: str, classes: str = "") -> Static:
        return Static(
            Text.assemble((f"{STATUS_LABELS[name]}: ", "dim"), value),
            id=f"field-{name}",
            classes=classes,
        )

    def set_status(self, status: ConsoleStatus) -> None:
        """Re-render the header from caller-supplied status."""

        self._status = status
        for name, value in status_values(status).items():
            self.query_one(f"#field-{name}", Static).update(
                Text.assemble((f"{STATUS_LABELS[name]}: ", "dim"), value)
            )

    def on_mount(self) -> None:
        self.query_one(Composer).focus()

    async def on_composer_submitted(self, message: Composer.Submitted) -> None:
        # Text (not markup) so operator input and command output render literally.
        entries = [
            Static(
                Text.assemble(("operator\n", "bold"), message.text),
                classes="entry operator",
            )
        ]
        routed = parse_submission(message.text, self._commands)
        if isinstance(routed, CommandSubmission):
            result = run_command(
                routed,
                CommandContext(
                    status=self._status,
                    commands=self._commands,
                    capabilities=self._capabilities,
                ),
            )
            entries.append(self._result_entry(result))
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount_all(entries)
        transcript.scroll_end(animate=False)

    @staticmethod
    def _result_entry(result: CommandResult) -> Static:
        body = "\n".join(result.lines)
        return Static(
            Text.assemble(
                ("zomah · ", "bold"),
                (result.title, "bold"),
                "\n" if body else "",
                body,
            ),
            classes="entry result error" if result.is_error else "entry result",
        )


def main() -> None:
    OperatorConsole().run()

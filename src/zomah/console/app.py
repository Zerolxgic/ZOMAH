"""Operator Console shell.

The console is an interface into ZOMAH, not a source of ZOMAH state. The
header renders a ``ConsoleStatus`` supplied by its caller; any field that has
no live runtime source yet is rendered as an explicit placeholder rather than
an invented value. Submitted composer text is only echoed into the transcript:
no model, capability, or command is invoked from this shell. Slash commands
are offered for discovery and completion only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
    default_command_registry,
    is_command_prefix,
)

PLACEHOLDER_NOT_SET = "not set"
PLACEHOLDER_NOT_CONNECTED = "not connected"
PLACEHOLDER_UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ConsoleStatus:
    """Header view data. ``None`` means no live source exists for the field."""

    project: str | None = None
    zomah_state: str | None = None
    model: str | None = None
    tools_available: int | None = None
    context_used: int | None = None
    context_limit: int | None = None


def _header_values(status: ConsoleStatus) -> dict[str, str]:
    if status.context_used is None or status.context_limit is None:
        context = PLACEHOLDER_UNAVAILABLE
    else:
        context = f"{status.context_used} / {status.context_limit}"
    return {
        "project": status.project or PLACEHOLDER_NOT_SET,
        "state": status.zomah_state or PLACEHOLDER_NOT_CONNECTED,
        "model": status.model or PLACEHOLDER_NOT_CONNECTED,
        "tools": (
            PLACEHOLDER_UNAVAILABLE
            if status.tools_available is None
            else str(status.tools_available)
        ),
        "context": context,
    }


_HEADER_LABELS = {
    "project": "Project",
    "state": "State",
    "model": "Model",
    "tools": "Tools",
    "context": "Context",
}


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
    ) -> None:
        super().__init__()
        self._status = status or ConsoleStatus()
        self._commands = commands or default_command_registry()

    @property
    def status(self) -> ConsoleStatus:
        return self._status

    def compose(self) -> ComposeResult:
        values = _header_values(self._status)
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
                    "No model connected. Submitted text is shown here only.",
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
            Text.assemble((f"{_HEADER_LABELS[name]}: ", "dim"), value),
            id=f"field-{name}",
            classes=classes,
        )

    def set_status(self, status: ConsoleStatus) -> None:
        """Re-render the header from caller-supplied status."""

        self._status = status
        for name, value in _header_values(status).items():
            self.query_one(f"#field-{name}", Static).update(
                Text.assemble((f"{_HEADER_LABELS[name]}: ", "dim"), value)
            )

    def on_mount(self) -> None:
        self.query_one(Composer).focus()

    async def on_composer_submitted(self, message: Composer.Submitted) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        # Text (not markup) so operator input is rendered literally.
        entry = Static(
            Text.assemble(("operator\n", "bold"), message.text),
            classes="entry operator",
        )
        await transcript.mount(entry)
        transcript.scroll_end(animate=False)


def main() -> None:
    OperatorConsole().run()

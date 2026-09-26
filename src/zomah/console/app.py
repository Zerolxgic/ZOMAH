"""Operator Console shell (T0a).

The console is an interface into ZOMAH, not a source of ZOMAH state. The
header renders a ``ConsoleStatus`` supplied by its caller; any field that has
no live runtime source yet is rendered as an explicit placeholder rather than
an invented value. Submitted composer text is only echoed into the transcript:
no model, capability, or command is invoked from this shell.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static, TextArea

from zomah import __version__

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


class Composer(TextArea):
    """Multiline message editor.

    Editing, selection, clipboard, undo, and redo are TextArea's own behavior.
    Only submission, the Shift+Enter newline, and Ctrl+A select-all differ.
    """

    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all", show=False),
    ]

    class Submitted(Message):
        """Posted when the operator submits non-blank composer text."""

        def __init__(self, composer: Composer, text: str) -> None:
            super().__init__()
            self.composer = composer
            self.text = text

    async def _on_key(self, event: events.Key) -> None:
        if self.read_only:
            return await super()._on_key(event)
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.submit()
            return
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            result = self.replace(
                "\n", *self.selection, maintain_selection_offset=False
            )
            self.move_cursor(result.end_location)
            return
        await super()._on_key(event)

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
    #transcript {
        height: 1fr;
        min-height: 1;
        padding: 0 1;
    }
    .entry {
        margin-bottom: 1;
    }
    #composer {
        height: auto;
        min-height: 3;
        max-height: 12;
    }
    """

    def __init__(self, status: ConsoleStatus | None = None) -> None:
        super().__init__()
        self._status = status or ConsoleStatus()

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
        yield VerticalScroll(
            Static(
                "No model connected. Submitted text is shown here only.",
                classes="entry notice",
            ),
            id="transcript",
        )
        yield Composer(id="composer", soft_wrap=True, show_line_numbers=False)

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

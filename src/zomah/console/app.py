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

import argparse
import asyncio
import getpass
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
from zomah.console.attachments import MAX_DRAFT_IMAGES, ImageAttachment
from zomah.console.clipboard import (
    ClipboardImageError,
    ClipboardImageSource,
    WaylandClipboardImageSource,
)
from zomah.console.commands import CommandContext, CommandResult
from zomah.console.operator import OperatorAccess, build_default_operator_access
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


class AttachmentStrip(Static):
    """Draft attachment metadata shown above the composer. The composer drives it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.display = False

    def show(self, attachments: Sequence[ImageAttachment]) -> None:
        self.update(Text("   ".join(attachment.describe() for attachment in attachments)))
        self.display = bool(attachments)


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
        # Ctrl+Shift+V is best effort: it reaches the app only if the terminal
        # passes it through. Alt+V is the explicit fallback for the same action.
        Binding("ctrl+shift+v,alt+v", "paste_image", "Paste image", show=False),
    ]

    class Submitted(Message):
        """Posted when the operator submits non-blank text and/or attachments."""

        def __init__(
            self,
            composer: Composer,
            text: str,
            attachments: tuple[ImageAttachment, ...] = (),
        ) -> None:
            super().__init__()
            self.composer = composer
            self.text = text
            self.attachments = attachments

    class ImagePasteRequested(Message):
        """Posted when the operator asks to attach the clipboard image."""

    def __init__(
        self,
        *,
        commands: CommandRegistry,
        suggestions: CommandSuggestions,
        attachment_strip: AttachmentStrip,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._commands = commands
        self._suggestions = suggestions
        self._attachment_strip = attachment_strip
        # Draft attachments live only here, in memory, until submission.
        self._attachments: list[ImageAttachment] = []
        # Text for which the operator dismissed suggestions (Esc or completion).
        self._dismissed_text: str | None = None

    def on_key(self, event: events.Key) -> None:
        # Legacy terminals send Alt+V as ESC v, a printable key that TextArea
        # would insert as "v" before bindings run; claim it for the same
        # action the binding uses. (Kitty-protocol Alt+V is not printable and
        # reaches the binding.)
        if event.key == "alt+v" and event.is_printable and not self.read_only:
            event.stop()
            event.prevent_default()
            self.action_paste_image()
            return
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

    @property
    def attachments(self) -> tuple[ImageAttachment, ...]:
        return tuple(self._attachments)

    def add_attachment(self, attachment: ImageAttachment) -> None:
        if len(self._attachments) >= MAX_DRAFT_IMAGES:
            raise ValueError(f"a message can carry at most {MAX_DRAFT_IMAGES} images")
        self._attachments.append(attachment)
        self._attachment_strip.show(self._attachments)

    def action_paste_image(self) -> None:
        self.post_message(self.ImagePasteRequested())

    def action_delete_left(self) -> None:
        # Backspace in an empty composer removes the most recent attachment;
        # otherwise it is TextArea's normal deletion.
        if not self.text and self._attachments:
            self._attachments.pop()
            self._attachment_strip.show(self._attachments)
            return
        super().action_delete_left()

    def submit(self) -> None:
        text = self.text
        if not text.strip() and not self._attachments:
            return
        attachments = tuple(self._attachments)
        # load_text clears edit history: a sent draft is not undoable.
        # Attachments are never part of that history.
        self.load_text("")
        self._attachments.clear()
        self._attachment_strip.show(())
        self.post_message(self.Submitted(self, text, attachments))


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
    #attachments {
        height: auto;
        padding: 0 1;
        color: $accent;
        text-wrap: nowrap;
        text-overflow: ellipsis;
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
        *,
        operator_access: OperatorAccess | None = None,
        clipboard: ClipboardImageSource | None = None,
    ) -> None:
        super().__init__()
        self._status = status or ConsoleStatus()
        self._commands = commands or default_command_registry()
        # Capabilities are invoked only through the user boundary, using the
        # injected operator access; /tools reads registry metadata only.
        self._capabilities = capabilities or default_capability_registry()
        self._operator_access = operator_access
        self._clipboard = clipboard or WaylandClipboardImageSource()

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
        attachment_strip = AttachmentStrip(id="attachments")
        yield attachment_strip
        yield Composer(
            commands=self._commands,
            suggestions=suggestions,
            attachment_strip=attachment_strip,
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
        """Replace the session state and re-render the header from it.

        The single write path for session context. Call it on the UI loop.
        """

        self._status = status
        for name, value in status_values(status).items():
            self.query_one(f"#field-{name}", Static).update(
                Text.assemble((f"{STATUS_LABELS[name]}: ", "dim"), value)
            )

    def on_mount(self) -> None:
        self.query_one(Composer).focus()

    async def on_composer_submitted(self, message: Composer.Submitted) -> None:
        # Text (not markup) so operator input and command output render literally.
        # Only attachment metadata reaches the transcript; image bytes are
        # dropped with the message once this handler returns.
        lines = [message.text] if message.text.strip() else []
        lines.extend(attachment.describe() for attachment in message.attachments)
        body = "\n".join(lines)
        entries = [
            Static(
                Text.assemble(("operator\n", "bold"), body),
                classes="entry operator",
            )
        ]
        routed = parse_submission(message.text, self._commands)
        pending: Static | None = None
        if isinstance(routed, CommandSubmission) and message.attachments:
            entries.append(
                self._result_entry(
                    CommandResult(
                        title=f"{routed.name} does not take image attachments.",
                        lines=("The attachments were not sent anywhere.",),
                        is_error=True,
                    )
                )
            )
        elif isinstance(routed, CommandSubmission):
            if routed.command is not None and routed.command.background:
                pending = Static(
                    Text.assemble(("zomah · ", "bold"), (routed.name, "bold"), " running…"),
                    classes="entry result pending",
                )
                entries.append(pending)
            else:
                result = run_command(routed, self._command_context())
                entries.append(self._result_entry(result))
                self._apply_status_update(result)
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount_all(entries)
        transcript.scroll_end(animate=False)
        if pending is not None and isinstance(routed, CommandSubmission) and not message.attachments:
            self.run_worker(
                self._run_in_background(routed, self._command_context(), pending),
                group="commands",
            )

    def on_composer_image_paste_requested(
        self, message: Composer.ImagePasteRequested
    ) -> None:
        composer = self.query_one(Composer)
        if len(composer.attachments) >= MAX_DRAFT_IMAGES:
            self.notify(
                f"A message can carry at most {MAX_DRAFT_IMAGES} images.",
                severity="warning",
            )
            return
        self.run_worker(self._attach_clipboard_image(), group="clipboard")

    async def _attach_clipboard_image(self) -> None:
        """Read the clipboard in a thread; apply the result on the UI loop."""

        try:
            attachment = await asyncio.to_thread(self._clipboard.read_image)
        except ClipboardImageError as error:
            self.notify(str(error), severity="warning")
            return
        except Exception:  # noqa: BLE001 - never surface exception details.
            self.log.error("clipboard image read failed")
            self.notify("Could not read the clipboard image.", severity="warning")
            return
        composer = self.query_one(Composer)
        try:
            composer.add_attachment(attachment)
        except ValueError:
            self.notify(
                f"A message can carry at most {MAX_DRAFT_IMAGES} images.",
                severity="warning",
            )
            return
        self.notify(f"Attached {attachment.describe()}")

    def _command_context(self) -> CommandContext:
        return CommandContext(
            status=self._status,
            commands=self._commands,
            capabilities=self._capabilities,
            operator_access=self._operator_access,
        )

    def _apply_status_update(self, result: CommandResult) -> None:
        if result.status_update is not None:
            self.set_status(result.status_update(self._status))

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

    async def _run_in_background(
        self, routed: CommandSubmission, context: CommandContext, pending: Static
    ) -> None:
        """Run an I/O-bound command in a thread and fill in its pending entry."""

        try:
            result = await asyncio.to_thread(run_command, routed, context)
        except Exception:  # noqa: BLE001 - a view bug must not crash the console.
            self.log.error(f"console command failed: {routed.name}")
            result = CommandResult(
                title=f"{routed.name} failed unexpectedly.", is_error=True
            )
        # Resumed on the UI event loop: only here are widgets and session
        # state touched; the command itself ran in a worker thread.
        entry = self._result_entry(result)
        self._apply_status_update(result)
        pending.update(entry.content)
        pending.set_classes(entry.classes)
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)


def build_console(argv: Sequence[str] | None = None) -> OperatorConsole:
    """Build the console from command-line options without running it."""

    parser = argparse.ArgumentParser(prog="zomah-console")
    parser.add_argument(
        "--project",
        dest="project_id",
        help="canonical ProjectState id for /project (default: none configured)",
    )
    parser.add_argument(
        "--operator",
        default=getpass.getuser(),
        help="operator id recorded as operator:<id> in traces (default: OS user)",
    )
    args = parser.parse_args(argv)
    return OperatorConsole(
        ConsoleStatus(active_project_id=args.project_id),
        operator_access=build_default_operator_access(args.operator),
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_console(argv).run()

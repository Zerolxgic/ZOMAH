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
from dataclasses import replace
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
from zomah.lmstudio import DEFAULT_BASE_URL, LMStudioRuntime
from zomah.model_runtime import ModelRuntimeError
from zomah.worker_session import SessionBusyError, WorkerSession

DEFAULT_WORKER = "elyria"
DISCONNECTED_NOTICE = "No model connected. Messages are shown here only; type / for commands."


def with_session_status(status: ConsoleStatus, session: WorkerSession) -> ConsoleStatus:
    """Copy live session metadata into console status, preserving other fields."""

    return replace(
        status,
        model=session.model,
        tools_available=len(session.tools),
        context_used=session.context_used,
        context_limit=session.context_limit,
    )

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
    .assistant {
        border-left: outer $success;
        padding-left: 1;
    }
    .assistant.error {
        border-left: outer $error;
    }
    .assistant.pending {
        color: $text-muted;
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
        worker_session: WorkerSession | None = None,
    ) -> None:
        super().__init__()
        self._status = status or ConsoleStatus()
        # The session owns the model conversation; the console only renders
        # it and mirrors its metadata into status.
        self._session = worker_session
        self._turn_in_flight = False
        if worker_session is not None:
            self._status = with_session_status(self._status, worker_session)
        self._commands = commands or default_command_registry()
        # Capabilities are invoked only through the user boundary, using the
        # injected operator access; /tools reads registry metadata only.
        self._capabilities = capabilities or default_capability_registry()
        self._operator_access = operator_access
        self._clipboard = clipboard or WaylandClipboardImageSource()

    def _startup_notice(self) -> str:
        if self._session is None:
            return DISCONNECTED_NOTICE
        return f"Model session configured: {self._session.model}. Type / for commands."

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
                    self._startup_notice(),
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
        routed = parse_submission(message.text, self._commands)
        if self._session is not None and not isinstance(routed, CommandSubmission):
            if self._turn_in_flight:
                self._reject_while_busy(message)
                return
            await self._start_model_turn(message)
            return
        # Text (not markup) so operator input and command output render literally.
        entries = [self._operator_entry(message)]
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

    @staticmethod
    def _operator_entry(message: Composer.Submitted) -> Static:
        # Only attachment metadata reaches the transcript; image bytes stay
        # with the message (and the model session, when one is attached).
        lines = [message.text] if message.text.strip() else []
        lines.extend(attachment.describe() for attachment in message.attachments)
        return Static(
            Text.assemble(("operator\n", "bold"), "\n".join(lines)),
            classes="entry operator",
        )

    def _reject_while_busy(self, message: Composer.Submitted) -> None:
        """Refuse a second model turn; put the draft back instead of queueing it."""

        composer = self.query_one(Composer)
        if not composer.text and not composer.attachments:
            composer.load_text(message.text)
            for attachment in message.attachments:
                composer.add_attachment(attachment)
            composer.move_cursor(composer.document.end)
        assert self._session is not None
        self.notify(
            f"{self._session.worker} is still answering. Your message was not sent.",
            severity="warning",
        )

    async def _start_model_turn(self, message: Composer.Submitted) -> None:
        assert self._session is not None
        self._turn_in_flight = True
        worker = self._session.worker
        pending = Static(
            Text.assemble((f"{worker}\n", "bold"), f"{worker} is thinking…"),
            classes="entry assistant pending",
        )
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount_all([self._operator_entry(message), pending])
        transcript.scroll_end(animate=False)
        self.run_worker(
            self._run_model_turn(message.text, message.attachments, pending),
            group="model",
        )

    async def _run_model_turn(
        self, text: str, attachments: tuple[ImageAttachment, ...], pending: Static
    ) -> None:
        """Await one session turn and render its outcome in the pending entry.

        Runs on the UI loop; the runtime keeps blocking I/O off it. Failed
        turns are not committed by the session, so only the transcript shows
        them.
        """

        assert self._session is not None
        worker = self._session.worker
        error: str | None = None
        reply = ""
        try:
            result = await self._session.send(text, attachments)
            reply = result.assistant.text
        except (ModelRuntimeError, SessionBusyError) as exc:
            error = str(exc)
        except Exception:  # noqa: BLE001 - never surface exception details.
            self.log.error("model turn failed unexpectedly")
            error = "The model turn failed unexpectedly."
        finally:
            self._turn_in_flight = False
        if error is None:
            pending.update(Text.assemble((f"{worker}\n", "bold"), reply))
            pending.set_classes("entry assistant")
        else:
            pending.update(Text.assemble((f"{worker}\n", "bold"), error))
            pending.set_classes("entry assistant error")
        self.set_status(with_session_status(self._status, self._session))
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

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
    parser.add_argument(
        "--model",
        help="exact LM Studio model id for a live worker session (default: disconnected)",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        help="configured context window in tokens (shown alongside reported usage)",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        default=DEFAULT_BASE_URL,
        help=f"LM Studio OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args(argv)
    if args.model is None and args.context_limit is not None:
        parser.error("--context-limit requires --model")
    if args.context_limit is not None and args.context_limit <= 0:
        parser.error("--context-limit must be positive")
    session = None
    if args.model is not None:
        session = WorkerSession(
            LMStudioRuntime(base_url=args.lmstudio_base_url),
            worker=DEFAULT_WORKER,
            model=args.model,
            context_limit=args.context_limit,
        )
    return OperatorConsole(
        ConsoleStatus(active_project_id=args.project_id),
        operator_access=build_default_operator_access(args.operator),
        worker_session=session,
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_console(argv).run()

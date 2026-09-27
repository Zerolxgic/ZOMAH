"""Composer image attachments driven by an injected fake clipboard."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from textual import events
from textual.pilot import Pilot
from textual.widgets import Static

from zomah.console import Composer, OperatorConsole
from zomah.console.app import AttachmentStrip
from zomah.console.attachments import MAX_DRAFT_IMAGES, ImageAttachment
from zomah.console.clipboard import ClipboardImageError

Scenario = Callable[[OperatorConsole, Pilot], Awaitable[None]]


def png(size: int) -> ImageAttachment:
    header = b"\x89PNG\r\n\x1a\n"
    return ImageAttachment(mime_type="image/png", data=header + b"\x00" * (size - len(header)))


class FakeClipboard:
    def __init__(self, *results: ImageAttachment | Exception) -> None:
        self._results = list(results)
        self.threads: list[int] = []

    def read_image(self) -> ImageAttachment:
        self.threads.append(threading.get_ident())
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class RecordingConsole(OperatorConsole):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.submitted: list[Composer.Submitted] = []

    def on_composer_submitted(self, message: Composer.Submitted) -> None:
        # Textual also dispatches OperatorConsole's handler for this message.
        self.submitted.append(message)


def run_console(scenario: Scenario, clipboard: FakeClipboard) -> None:
    async def runner() -> None:
        app = RecordingConsole(clipboard=clipboard)
        async with app.run_test(size=(100, 30)) as pilot:
            await scenario(app, pilot)

    asyncio.run(runner())


async def paste_image(pilot: Pilot) -> None:
    await pilot.press("ctrl+shift+v")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def strip_text(app: OperatorConsole) -> str | None:
    strip = app.query_one(AttachmentStrip)
    return str(strip.content) if strip.display else None


def toasts(app: OperatorConsole) -> list[str]:
    return [str(n.message) for n in app._notifications]


def operator_entries(app: OperatorConsole) -> list[str]:
    return [str(e.content) for e in app.query(".entry.operator").results(Static)]


def test_clipboard_image_becomes_attachment_without_text_token() -> None:
    image = png(842 * 1024)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await paste_image(pilot)
        assert composer.attachments == (image,)
        assert composer.attachments[0] is image
        assert composer.text == ""
        assert strip_text(app) == "Image · image/png · 842 KiB"
        assert toasts(app) == ["Attached Image · image/png · 842 KiB"]

    run_console(scenario, FakeClipboard(image))


def test_text_paste_and_ctrl_v_are_unchanged() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        app.post_message(events.Paste("line one\nline two"))
        await pilot.pause()
        assert composer.text == "line one\nline two"
        await pilot.press("ctrl+a", "ctrl+c", "end", "ctrl+v")
        assert composer.text == "line one\nline twoline one\nline two"
        assert composer.attachments == ()
        assert clipboard.threads == []

    clipboard = FakeClipboard()
    run_console(scenario, clipboard)


def test_image_only_draft_submits_and_clears() -> None:
    image = png(2048)

    async def scenario(app: RecordingConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await paste_image(pilot)
        await pilot.press("enter")
        await pilot.pause()
        (message,) = app.submitted
        assert (message.text, message.attachments) == ("", (image,))
        assert message.attachments[0] is image
        assert operator_entries(app) == ["operator\nImage · image/png · 2 KiB"]
        assert composer.attachments == ()
        assert strip_text(app) is None

    run_console(scenario, FakeClipboard(image))


def test_text_and_image_submit_together() -> None:
    image = png(1024)

    async def scenario(app: RecordingConsole, pilot: Pilot) -> None:
        await paste_image(pilot)
        await pilot.press(*"look", "enter")
        await pilot.pause()
        (message,) = app.submitted
        assert (message.text, message.attachments) == ("look", (image,))
        assert operator_entries(app) == ["operator\nlook\nImage · image/png · 1 KiB"]

    run_console(scenario, FakeClipboard(image))


def test_undo_cannot_restore_submitted_draft_and_composer_stays_usable() -> None:
    async def scenario(app: RecordingConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await paste_image(pilot)
        await pilot.press(*"sent", "enter")
        await pilot.pause()
        await pilot.press("ctrl+z", "ctrl+y")
        assert composer.text == ""
        assert composer.attachments == ()
        assert strip_text(app) is None

        await paste_image(pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert [len(m.attachments) for m in app.submitted] == [1, 1]

    run_console(scenario, FakeClipboard(png(1024), png(1024)))


def test_empty_draft_is_still_not_submitted() -> None:
    async def scenario(app: RecordingConsole, pilot: Pilot) -> None:
        await pilot.press("space", "enter")
        await pilot.pause()
        assert app.submitted == []

    run_console(scenario, FakeClipboard())


def test_backspace_on_empty_text_removes_last_attachment() -> None:
    first, second = png(1024), png(2048)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await paste_image(pilot)
        await paste_image(pilot)
        assert composer.attachments == (first, second)
        assert strip_text(app) == (
            "Image · image/png · 1 KiB   Image · image/png · 2 KiB"
        )
        await pilot.press("backspace")
        assert composer.attachments == (first,)
        assert strip_text(app) == "Image · image/png · 1 KiB"
        await pilot.press("backspace", "backspace")
        assert composer.attachments == ()
        assert strip_text(app) is None

    run_console(scenario, FakeClipboard(first, second))


def test_backspace_with_text_keeps_editor_behavior() -> None:
    image = png(1024)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await paste_image(pilot)
        await pilot.press(*"abc", "backspace")
        assert (composer.text, composer.attachments) == ("ab", (image,))
        await pilot.press("ctrl+a", "backspace")
        assert (composer.text, composer.attachments) == ("", (image,))
        await pilot.press(*"x", "ctrl+z")
        await pilot.press("home", "backspace")
        assert composer.attachments == ()

    run_console(scenario, FakeClipboard(image))


def test_attachment_count_is_bounded() -> None:
    images = [png(1024) for _ in range(MAX_DRAFT_IMAGES)]
    clipboard = FakeClipboard(*images)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        for _ in range(MAX_DRAFT_IMAGES + 1):
            await paste_image(pilot)
        assert app.query_one(Composer).attachments == tuple(images)
        assert toasts(app)[-1] == f"A message can carry at most {MAX_DRAFT_IMAGES} images."
        assert len(clipboard.threads) == MAX_DRAFT_IMAGES

    run_console(scenario, clipboard)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (ClipboardImageError("The clipboard has no PNG, JPEG, or WebP image."),
         "The clipboard has no PNG, JPEG, or WebP image."),
        (ClipboardImageError("Image paste needs wl-paste (wl-clipboard)."),
         "Image paste needs wl-paste (wl-clipboard)."),
        (ClipboardImageError("Reading the clipboard image timed out."),
         "Reading the clipboard image timed out."),
        (ClipboardImageError("The clipboard image is larger than 10.0 MiB."),
         "The clipboard image is larger than 10.0 MiB."),
        (RuntimeError("secret internal detail"), "Could not read the clipboard image."),
    ],
)
def test_clipboard_failures_give_feedback_without_attaching(
    failure: Exception, expected: str
) -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"draft")
        await paste_image(pilot)
        assert composer.attachments == ()
        assert composer.text == "draft"
        assert strip_text(app) is None
        assert toasts(app) == [expected]
        assert "secret" not in " ".join(toasts(app))

    run_console(scenario, FakeClipboard(failure))


def test_clipboard_read_is_off_ui_thread_and_attachment_applied_on_ui_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui_thread = threading.get_ident()
    add_threads: list[int] = []
    original = Composer.add_attachment

    def recording_add(self: Composer, attachment: ImageAttachment) -> None:
        add_threads.append(threading.get_ident())
        original(self, attachment)

    monkeypatch.setattr(Composer, "add_attachment", recording_add)
    clipboard = FakeClipboard(png(1024))

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await paste_image(pilot)
        assert len(app.query_one(Composer).attachments) == 1

    run_console(scenario, clipboard)
    assert len(clipboard.threads) == 1 and clipboard.threads[0] != ui_thread
    assert add_threads == [ui_thread]


def test_commands_do_not_take_attachments() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await paste_image(pilot)
        composer.load_text("/status")
        await pilot.press("escape", "enter")
        await pilot.pause()
        results = [str(e.content) for e in app.query(".entry.result").results(Static)]
        assert results == [
            "zomah · /status does not take image attachments.\n"
            "The attachments were not sent anywhere."
        ]
        assert composer.attachments == ()

    run_console(scenario, FakeClipboard(png(1024)))


def test_help_documents_image_paste_and_removal() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        app.query_one(Composer).load_text("/help")
        await pilot.press("escape", "enter")
        await pilot.pause()
        output = str(app.query_one(".entry.result", Static).content)
        lines = [" ".join(line.split()) for line in output.split("\n")]
        assert "Ctrl+Shift+V attach clipboard image when supported by terminal" in lines
        assert "Alt+V attach clipboard image fallback" in lines
        assert "Backspace remove last image when the text is empty" in lines

    run_console(scenario, FakeClipboard())


@pytest.mark.parametrize(
    "key_event",
    [
        events.Key("ctrl+shift+v", None),
        events.Key("alt+v", None),  # kitty keyboard protocol form
        events.Key("alt+v", "v"),  # legacy ESC v form, printable
    ],
    ids=["ctrl+shift+v", "alt+v-kitty", "alt+v-legacy"],
)
def test_image_paste_keys_share_the_paste_image_action(
    monkeypatch: pytest.MonkeyPatch, key_event: events.Key
) -> None:
    image = png(1024)
    actions: list[str] = []
    original = Composer.action_paste_image

    def recording_action(self: Composer) -> None:
        actions.append("paste_image")
        original(self)

    monkeypatch.setattr(Composer, "action_paste_image", recording_action)
    clipboard = FakeClipboard(image)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await pilot.press(*"draft")
        # Posted to the app, as the terminal driver delivers keys.
        app.post_message(key_event)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert actions == ["paste_image"]
        assert composer.attachments == (image,)
        assert composer.text == "draft"
        assert strip_text(app) == "Image · image/png · 1 KiB"

    run_console(scenario, clipboard)
    assert len(clipboard.threads) == 1


def test_alt_v_binding_and_ctrl_shift_v_binding_name_the_same_action() -> None:
    actions = {
        key: binding.action
        for binding in Composer.BINDINGS
        for key in binding.key.split(",")
        if key in {"ctrl+shift+v", "alt+v"}
    }
    assert actions == {"ctrl+shift+v": "paste_image", "alt+v": "paste_image"}


def test_plain_v_is_still_typed() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        app.post_message(events.Key("v", "v"))
        await pilot.pause()
        assert composer.text == "v"
        assert composer.attachments == ()
        assert clipboard.threads == []

    clipboard = FakeClipboard()
    run_console(scenario, clipboard)

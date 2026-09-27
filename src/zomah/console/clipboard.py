"""Clipboard image source for the Operator Console.

Textual's ``Paste`` event carries text only, so images are read from the OS
clipboard directly. The Wayland implementation runs ``wl-paste`` with fixed
arguments (never through a shell), bounds both time and bytes read, and
reports failures as ``ClipboardImageError`` with operator-facing messages.
"""

from __future__ import annotations

import os
import selectors
import shutil
import subprocess
import time
from typing import Protocol

from zomah.console.attachments import (
    MAX_IMAGE_BYTES,
    SUPPORTED_IMAGE_TYPES,
    ImageAttachment,
    format_size,
)

MAX_TYPE_LIST_BYTES = 64 * 1024
DEFAULT_TIMEOUT_SECONDS = 5.0


class ClipboardImageError(Exception):
    """A clipboard image could not be attached. ``str()`` is operator-facing."""


class ClipboardImageSource(Protocol):
    def read_image(self) -> ImageAttachment:
        """Return the clipboard image or raise ``ClipboardImageError``."""
        ...


class _TooLarge(Exception):
    pass


def _run_bounded(args: list[str], *, timeout: float, limit: int) -> tuple[int, bytes]:
    """Run ``args`` without a shell, reading at most ``limit`` stdout bytes."""

    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=False,
    )
    assert process.stdout is not None
    output = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                if not selector.select(remaining):
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                output += chunk
                if len(output) > limit:
                    raise _TooLarge()
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
    return returncode, bytes(output)


class WaylandClipboardImageSource:
    """Reads PNG, JPEG, or WebP images from the Wayland clipboard via wl-paste."""

    def __init__(
        self,
        *,
        executable: str = "wl-paste",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_bytes: int = MAX_IMAGE_BYTES,
    ) -> None:
        self._executable = executable
        self._timeout = timeout
        self._max_bytes = min(max_bytes, MAX_IMAGE_BYTES)

    def read_image(self) -> ImageAttachment:
        if not os.environ.get("WAYLAND_DISPLAY"):
            raise ClipboardImageError("Image paste needs a Wayland session.")
        path = shutil.which(self._executable)
        if path is None:
            raise ClipboardImageError("Image paste needs wl-paste (wl-clipboard).")

        offered = self._run(
            [path, "--list-types"], limit=MAX_TYPE_LIST_BYTES, what="clipboard types"
        )
        types = {line.strip() for line in offered.decode("utf-8", "replace").splitlines()}
        mime_type = next((t for t in SUPPORTED_IMAGE_TYPES if t in types), None)
        if mime_type is None:
            raise ClipboardImageError("The clipboard has no PNG, JPEG, or WebP image.")

        data = self._run(
            [path, "--no-newline", "--type", mime_type],
            limit=self._max_bytes,
            what="clipboard image",
        )
        try:
            return ImageAttachment(mime_type=mime_type, data=data, source="clipboard")
        except ValueError:
            raise ClipboardImageError(
                f"The clipboard image is not valid {mime_type}."
            ) from None

    def _run(self, args: list[str], *, limit: int, what: str) -> bytes:
        try:
            returncode, output = _run_bounded(args, timeout=self._timeout, limit=limit)
        except _TooLarge:
            raise ClipboardImageError(
                f"The clipboard image is larger than {format_size(self._max_bytes)}."
                if what == "clipboard image"
                else "The clipboard offered too many types."
            ) from None
        except subprocess.TimeoutExpired:
            raise ClipboardImageError(f"Reading the {what} timed out.") from None
        except OSError:
            raise ClipboardImageError(f"Could not read the {what}.") from None
        if returncode != 0:
            raise ClipboardImageError(
                "The clipboard is empty or unavailable."
                if what == "clipboard types"
                else f"Could not read the {what}."
            )
        return output

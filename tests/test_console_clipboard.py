"""ImageAttachment validation and the wl-paste clipboard adapter.

The adapter runs against fake ``wl-paste`` scripts, never the live clipboard.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from zomah.console.attachments import MAX_IMAGE_BYTES, ImageAttachment, format_size
from zomah.console.clipboard import ClipboardImageError, WaylandClipboardImageSource

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 64


def test_attachment_keeps_mime_bytes_source_and_size() -> None:
    attachment = ImageAttachment(mime_type="image/png", data=PNG)
    assert attachment.source == "clipboard"
    assert attachment.size_bytes == len(PNG)
    assert attachment.describe() == f"Image · image/png · {len(PNG)} B"
    assert "\\x89PNG" not in repr(attachment)


@pytest.mark.parametrize(
    ("mime_type", "data"),
    [("image/png", PNG), ("image/jpeg", JPEG), ("image/webp", WEBP)],
)
def test_supported_formats_validate_by_signature(mime_type: str, data: bytes) -> None:
    assert ImageAttachment(mime_type=mime_type, data=data).mime_type == mime_type


@pytest.mark.parametrize(
    ("mime_type", "data"),
    [
        ("image/gif", b"GIF89a" + b"\x00" * 8),
        ("image/png", b""),
        ("image/png", JPEG),
        ("image/jpeg", PNG),
        ("image/png", b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_IMAGE_BYTES),
    ],
)
def test_invalid_attachments_are_rejected(mime_type: str, data: bytes) -> None:
    with pytest.raises(ValueError):
        ImageAttachment(mime_type=mime_type, data=data)


def test_size_formatting() -> None:
    assert format_size(512) == "512 B"
    assert format_size(842 * 1024) == "842 KiB"
    assert format_size(3 * 1024 * 1024 + 512 * 1024) == "3.5 MiB"


@pytest.fixture
def wayland(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-test")


def fake_wl_paste(
    directory: Path,
    *,
    types: str = "image/png\n",
    types_exit: int = 0,
    data: bytes = PNG,
    data_exit: int = 0,
    delay: float = 0.0,
) -> str:
    image = directory / "clipboard.bin"
    image.write_bytes(data)
    log = directory / "calls.log"
    script = directory / "wl-paste"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{log}"\n'
        f"sleep {delay}\n"
        'if [ "$1" = "--list-types" ]; then\n'
        f"  printf '{types}'\n"
        f"  exit {types_exit}\n"
        "fi\n"
        f'cat "{image}"\n'
        f"exit {data_exit}\n"
    )
    script.chmod(0o755)
    return str(script)


def calls(directory: Path) -> list[str]:
    return (directory / "calls.log").read_text().splitlines()


def read(executable: str, **kwargs: Any) -> ImageAttachment:
    return WaylandClipboardImageSource(executable=executable, **kwargs).read_image()


def test_reads_png_with_fixed_arguments(tmp_path: Path, wayland: None) -> None:
    attachment = read(fake_wl_paste(tmp_path, types="text/html\\nimage/png\\n"))
    assert (attachment.mime_type, attachment.data, attachment.source) == (
        "image/png",
        PNG,
        "clipboard",
    )
    assert calls(tmp_path) == ["--list-types", "--no-newline --type image/png"]


def test_prefers_png_when_several_image_types_are_offered(
    tmp_path: Path, wayland: None
) -> None:
    read(fake_wl_paste(tmp_path, types="image/webp\\nimage/jpeg\\nimage/png\\n"))
    assert calls(tmp_path)[-1] == "--no-newline --type image/png"


def test_reads_jpeg_when_it_is_the_only_image(tmp_path: Path, wayland: None) -> None:
    attachment = read(fake_wl_paste(tmp_path, types="image/jpeg\\n", data=JPEG))
    assert attachment.mime_type == "image/jpeg"


def test_text_only_clipboard_is_rejected_without_reading_data(
    tmp_path: Path, wayland: None
) -> None:
    with pytest.raises(ClipboardImageError, match="no PNG, JPEG, or WebP image"):
        read(fake_wl_paste(tmp_path, types="text/plain\\nUTF8_STRING\\n"))
    assert calls(tmp_path) == ["--list-types"]


def test_empty_clipboard_is_reported(tmp_path: Path, wayland: None) -> None:
    with pytest.raises(ClipboardImageError, match="empty or unavailable"):
        read(fake_wl_paste(tmp_path, types="", types_exit=1))


def test_missing_wl_paste_is_reported(wayland: None) -> None:
    with pytest.raises(ClipboardImageError, match="needs wl-paste"):
        read("zomah-no-such-wl-paste")


def test_missing_wayland_session_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(ClipboardImageError, match="needs a Wayland session"):
        read(fake_wl_paste(tmp_path))
    assert not (tmp_path / "calls.log").exists()


def test_timeout_is_bounded_and_reported(tmp_path: Path, wayland: None) -> None:
    started = time.monotonic()
    with pytest.raises(ClipboardImageError, match="timed out"):
        read(fake_wl_paste(tmp_path, delay=5), timeout=0.3)
    assert time.monotonic() - started < 3


def test_oversized_image_is_rejected(tmp_path: Path, wayland: None) -> None:
    with pytest.raises(ClipboardImageError, match="larger than 32 B"):
        read(fake_wl_paste(tmp_path, data=PNG), max_bytes=32)


def test_data_that_does_not_match_the_offered_type_is_rejected(
    tmp_path: Path, wayland: None
) -> None:
    with pytest.raises(ClipboardImageError, match="not valid image/png"):
        read(fake_wl_paste(tmp_path, data=b"<html>not an image</html>"))


def test_failed_data_read_is_reported(tmp_path: Path, wayland: None) -> None:
    with pytest.raises(ClipboardImageError, match="Could not read the clipboard image"):
        read(fake_wl_paste(tmp_path, data_exit=1))


def test_never_uses_a_shell(
    tmp_path: Path, wayland: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Any, dict[str, Any]]] = []
    real_popen = subprocess.Popen

    def spy(args: Any, **kwargs: Any):
        seen.append((args, kwargs))
        return real_popen(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    read(fake_wl_paste(tmp_path))
    assert len(seen) == 2
    for args, kwargs in seen:
        assert isinstance(args, list)
        assert kwargs["shell"] is False

"""Composer image attachments.

An ``ImageAttachment`` is draft content held in memory only: it is never
written to disk or kept in transcript history. Validation is limited to a
supported MIME type, a size bound, and the format's leading signature bytes;
images are not decoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Console-side limits, independent of any future model/provider image limit.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_DRAFT_IMAGES = 4

# Preference order when the clipboard offers several supported types.
SUPPORTED_IMAGE_TYPES: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")


def _matches_signature(mime_type: str, data: bytes) -> bool:
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{round(size / 1024)} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    """One validated image in a composer draft."""

    mime_type: str
    data: bytes = field(repr=False)
    source: Literal["clipboard"] = "clipboard"

    def __post_init__(self) -> None:
        if self.mime_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"unsupported image type: {self.mime_type}")
        if not self.data:
            raise ValueError("image data is empty")
        if len(self.data) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds the attachment size limit")
        if not _matches_signature(self.mime_type, self.data):
            raise ValueError(f"image data is not valid {self.mime_type}")

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def describe(self) -> str:
        return f"Image · {self.mime_type} · {format_size(self.size_bytes)}"

"""Compatibility re-export: image attachments live in ``zomah.attachments``."""

from zomah.attachments import (
    MAX_DRAFT_IMAGES,
    MAX_IMAGE_BYTES,
    SUPPORTED_IMAGE_TYPES,
    ImageAttachment,
    format_size,
)

__all__ = [
    "MAX_DRAFT_IMAGES",
    "MAX_IMAGE_BYTES",
    "SUPPORTED_IMAGE_TYPES",
    "ImageAttachment",
    "format_size",
]

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from zomah.access import ReadScope


DirectoryPath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

DEFAULT_MAX_ENTRIES = 200
MAX_ENTRIES = 1000


class DirectoryEntry(BaseModel):
    """One direct child of an approved directory."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    kind: Literal["file", "directory", "symlink", "other"]
    size_bytes: int | None = Field(default=None, ge=0)


class ListDirectoryRequest(BaseModel):
    """Validated model-facing input for the list_directory capability."""

    model_config = ConfigDict(extra="forbid")

    path: DirectoryPath
    offset: int = Field(default=0, ge=0)
    max_entries: int = Field(default=DEFAULT_MAX_ENTRIES, ge=1, le=MAX_ENTRIES)


class ListDirectoryResponse(BaseModel):
    """A deterministic, bounded listing of one approved directory."""

    model_config = ConfigDict(extra="forbid")

    path: str
    entries: list[DirectoryEntry]
    offset: int = Field(ge=0)
    returned: int = Field(ge=0)
    truncated: bool
    next_offset: int | None = Field(default=None, ge=0)


def list_directory(
    request: ListDirectoryRequest,
    scope: ReadScope,
) -> ListDirectoryResponse:
    """List direct children of one directory inside an approved read root.

    The capability is intentionally non-recursive. Elyria can descend into a
    returned directory with another explicit call, which keeps traversal and
    context growth visible and bounded.

    Directory entries are sorted deterministically by name. Symlinks are
    reported as symlinks and are never followed while classifying entries.
    """

    directory = scope.resolve_directory(request.path)

    with os.scandir(directory) as iterator:
        raw_entries = sorted(iterator, key=lambda entry: (entry.name.casefold(), entry.name))

    selected = raw_entries[request.offset : request.offset + request.max_entries + 1]
    truncated = len(selected) > request.max_entries
    visible = selected[: request.max_entries]

    entries = [_to_directory_entry(directory, entry) for entry in visible]
    next_offset = request.offset + len(entries) if truncated else None

    return ListDirectoryResponse(
        path=str(directory),
        entries=entries,
        offset=request.offset,
        returned=len(entries),
        truncated=truncated,
        next_offset=next_offset,
    )


def _to_directory_entry(directory: Path, entry: os.DirEntry[str]) -> DirectoryEntry:
    if entry.is_symlink():
        kind: Literal["file", "directory", "symlink", "other"] = "symlink"
        size_bytes = None
    elif entry.is_dir(follow_symlinks=False):
        kind = "directory"
        size_bytes = None
    elif entry.is_file(follow_symlinks=False):
        kind = "file"
        size_bytes = entry.stat(follow_symlinks=False).st_size
    else:
        kind = "other"
        size_bytes = None

    return DirectoryEntry(
        name=entry.name,
        path=str(directory / entry.name),
        kind=kind,
        size_bytes=size_bytes,
    )

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from zomah.access import InvalidWriteTarget, WriteScope


FilePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class MoveFileRequest(BaseModel):
    """Validated model-facing input for the move_file capability."""

    model_config = ConfigDict(extra="forbid")

    source: FilePath
    destination: FilePath


class MoveFileResponse(BaseModel):
    """Structured result from one approved file move."""

    model_config = ConfigDict(extra="forbid")

    source: str
    destination: str
    size_bytes: int = Field(ge=0)


def move_file(request: MoveFileRequest, scope: WriteScope) -> MoveFileResponse:
    """Move one regular file between approved write locations.

    Both source and destination must be inside configured write roots. The
    destination parent must already exist and the destination itself must not.
    ZOMAH never creates directories, overwrites a destination, copies across
    filesystems, or accepts symlink sources/targets in this capability.
    """

    source = scope.resolve_existing_file(request.source)
    destination = scope.resolve_target(request.destination)

    if source == destination:
        raise InvalidWriteTarget("move source and destination must differ")
    if os.path.lexists(destination):
        raise InvalidWriteTarget(f"move destination already exists: {destination}")

    size_bytes = source.stat().st_size
    _rename_noreplace(source, destination)

    return MoveFileResponse(
        source=str(source),
        destination=str(destination),
        size_bytes=size_bytes,
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Rename without replacement on Linux; never fall back to copy/delete."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except AttributeError:
        _link_unlink_noreplace(source, destination)
        return

    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int

    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise InvalidWriteTarget(
            f"move destination already exists: {destination}"
        )
    if error == errno.EXDEV:
        raise InvalidWriteTarget(
            "cross-filesystem moves are not supported by move_file"
        )
    if error in {errno.ENOSYS, errno.EINVAL}:
        _link_unlink_noreplace(source, destination)
        return

    raise OSError(error, os.strerror(error), str(source), str(destination))


def _link_unlink_noreplace(source: Path, destination: Path) -> None:
    """Safe same-filesystem fallback when renameat2 is unavailable.

    Hard-link creation is no-replace and atomic. The source name is removed
    only after the destination link exists. Cross-filesystem requests fail
    rather than silently becoming copy/delete operations.
    """

    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise InvalidWriteTarget(
            f"move destination already exists: {destination}"
        ) from exc
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise InvalidWriteTarget(
                "cross-filesystem moves are not supported by move_file"
            ) from exc
        raise

    try:
        source.unlink()
    except Exception:
        destination.unlink(missing_ok=True)
        raise

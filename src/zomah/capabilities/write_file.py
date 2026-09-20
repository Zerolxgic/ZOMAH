from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from zomah.access import InvalidWriteTarget, WriteScope


FilePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

WriteMode = Literal["create", "replace", "append"]
MAX_WRITE_BYTES = 64 * 1024


class UnsupportedWriteContent(ValueError):
    """Raised when write_file is asked to produce something other than UTF-8 text."""


class WriteFileRequest(BaseModel):
    """Validated model-facing input for the write_file capability."""

    model_config = ConfigDict(extra="forbid")

    path: FilePath
    content: str
    mode: WriteMode


class WriteFileResponse(BaseModel):
    """Structured result from one approved text-file mutation."""

    model_config = ConfigDict(extra="forbid")

    path: str
    mode: WriteMode
    bytes_written: int = Field(ge=0)
    previous_size_bytes: int | None = Field(default=None, ge=0)
    new_size_bytes: int = Field(ge=0)


def write_file(request: WriteFileRequest, scope: WriteScope) -> WriteFileResponse:
    """Create, replace, or append UTF-8 text inside an approved write root.

    The capability does not create parent directories and never performs file
    deletion. Create and replace publish complete content from a temporary file
    in the destination directory; replace uses `os.replace` for an atomic path
    swap on local filesystems. Append is explicit and preserves the existing
    file.
    """

    path = scope.resolve_target(request.path)
    payload = _encode_text(request.content)

    if request.mode == "create":
        if os.path.lexists(path):
            raise InvalidWriteTarget(f"create target already exists: {path}")
        _atomic_create(path, payload)
        previous_size = None

    elif request.mode == "replace":
        _require_existing_text_file(path, mode="replace")
        previous_size = path.stat().st_size
        _atomic_replace(path, payload)

    else:  # append
        _require_existing_text_file(path, mode="append")
        previous_size = path.stat().st_size
        _append_bytes(path, payload)

    new_size = path.stat().st_size
    return WriteFileResponse(
        path=str(path),
        mode=request.mode,
        bytes_written=len(payload),
        previous_size_bytes=previous_size,
        new_size_bytes=new_size,
    )


def _encode_text(content: str) -> bytes:
    if "\x00" in content:
        raise UnsupportedWriteContent("write content may not contain NUL bytes")

    payload = content.encode("utf-8")
    if len(payload) > MAX_WRITE_BYTES:
        raise UnsupportedWriteContent(
            f"write content exceeds the {MAX_WRITE_BYTES}-byte limit"
        )
    return payload


def _require_existing_text_file(path: Path, *, mode: WriteMode) -> None:
    if not os.path.lexists(path):
        raise InvalidWriteTarget(f"{mode} target does not exist: {path}")
    if path.is_symlink():
        raise InvalidWriteTarget(f"symlink write targets are not allowed: {path}")
    if not path.is_file():
        raise InvalidWriteTarget(f"{mode} target is not a regular file: {path}")

    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                if "\x00" in chunk:
                    raise UnsupportedWriteContent(
                        f"existing target contains NUL bytes: {path}"
                    )
    except UnicodeDecodeError as exc:
        raise UnsupportedWriteContent(
            f"existing target is not valid UTF-8 text: {path}"
        ) from exc


def _write_temp(path: Path, payload: bytes, *, mode_bits: int) -> Path:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.zomah-",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, mode_bits)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_create(path: Path, payload: bytes) -> None:
    temp_path = _write_temp(path, payload, mode_bits=0o600)
    try:
        os.link(temp_path, path)
    except FileExistsError as exc:
        raise InvalidWriteTarget(f"create target already exists: {path}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_replace(path: Path, payload: bytes) -> None:
    mode_bits = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    temp_path = _write_temp(path, payload, mode_bits=mode_bits)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _append_bytes(path: Path, payload: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from zomah.access import ReadScope


FilePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

DEFAULT_MAX_LINES = 200
MAX_LINES = 1000
MAX_OUTPUT_CHARS = 64 * 1024
MAX_LINE_CHARS = 16 * 1024


class UnsupportedTextFile(ValueError):
    """Raised when read_file cannot safely treat the target as UTF-8 text."""


class ReadFileRequest(BaseModel):
    """Validated model-facing input for the read_file capability."""

    model_config = ConfigDict(extra="forbid")

    path: FilePath
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=DEFAULT_MAX_LINES, ge=1, le=MAX_LINES)


class ReadFileResponse(BaseModel):
    """Bounded UTF-8 text returned from one approved filesystem path."""

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    start_line: int = Field(ge=1)
    end_line: int | None = Field(default=None, ge=1)
    truncated: bool
    next_start_line: int | None = Field(default=None, ge=1)
    encoding: Literal["utf-8"] = "utf-8"


def read_file(request: ReadFileRequest, scope: ReadScope) -> ReadFileResponse:
    """Read a bounded UTF-8 text slice from one path inside an approved root.

    ZOMAH resolves the requested path before the scope check. That prevents
    `..` traversal and symlink escapes from bypassing configured roots.

    The capability intentionally does not guess encodings or expose binary
    reads. Those can be added later only if real use demonstrates a need.
    """

    path = scope.resolve_file(request.path)
    lines: list[str] = []
    chars = 0
    current_line = 0
    next_start_line: int | None = None

    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                current_line += 1

                if current_line < request.start_line:
                    continue

                if "\x00" in line:
                    raise UnsupportedTextFile(f"file contains NUL bytes: {path}")

                if len(line) > MAX_LINE_CHARS:
                    raise UnsupportedTextFile(
                        f"file contains a line larger than {MAX_LINE_CHARS} characters: {path}"
                    )

                if len(lines) >= request.max_lines:
                    next_start_line = current_line
                    break

                if chars + len(line) > MAX_OUTPUT_CHARS:
                    next_start_line = current_line
                    break

                lines.append(line)
                chars += len(line)
    except UnicodeDecodeError as exc:
        raise UnsupportedTextFile(f"file is not valid UTF-8 text: {path}") from exc

    end_line = None
    if lines:
        end_line = request.start_line + len(lines) - 1

    return ReadFileResponse(
        path=str(path),
        content="".join(lines),
        start_line=request.start_line,
        end_line=end_line,
        truncated=next_start_line is not None,
        next_start_line=next_start_line,
    )

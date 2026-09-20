from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.access import InvalidReadTarget, PathOutsideScope, ReadScope
from zomah.capabilities import ReadFileRequest, read_file
from zomah.capabilities.read_file import UnsupportedTextFile


def test_read_file_reads_bounded_text_inside_scope(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    target = root / "notes.md"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    response = read_file(
        ReadFileRequest(path=str(target), start_line=2, max_lines=2),
        ReadScope.from_paths([root]),
    )

    assert response.path == str(target.resolve())
    assert response.content == "two\nthree\n"
    assert response.start_line == 2
    assert response.end_line == 3
    assert response.truncated is True
    assert response.next_start_line == 4


def test_read_file_reports_end_of_file_without_truncation(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    target = root / "notes.md"
    target.write_text("one\ntwo\n", encoding="utf-8")

    response = read_file(
        ReadFileRequest(path=str(target), start_line=2),
        ReadScope.from_paths([root]),
    )

    assert response.content == "two\n"
    assert response.end_line == 2
    assert response.truncated is False
    assert response.next_start_line is None


def test_read_file_rejects_path_outside_scope(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")

    with pytest.raises(PathOutsideScope):
        read_file(
            ReadFileRequest(path=str(outside)),
            ReadScope.from_paths([root]),
        )


def test_read_file_rejects_parent_traversal_outside_scope(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    traversed = root / ".." / "outside.txt"

    with pytest.raises(PathOutsideScope):
        read_file(
            ReadFileRequest(path=str(traversed)),
            ReadScope.from_paths([root]),
        )


def test_read_file_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    link = root / "escape.txt"
    link.symlink_to(outside)

    with pytest.raises(PathOutsideScope):
        read_file(
            ReadFileRequest(path=str(link)),
            ReadScope.from_paths([root]),
        )


def test_read_file_rejects_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()

    with pytest.raises(InvalidReadTarget, match="absolute path"):
        read_file(
            ReadFileRequest(path="notes.md"),
            ReadScope.from_paths([root]),
        )


def test_read_file_rejects_invalid_utf8(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    target = root / "binary-ish.dat"
    target.write_bytes(b"valid-prefix\xff\xfe")

    with pytest.raises(UnsupportedTextFile, match="valid UTF-8"):
        read_file(
            ReadFileRequest(path=str(target)),
            ReadScope.from_paths([root]),
        )


def test_read_file_request_caps_model_requested_line_count() -> None:
    with pytest.raises(ValidationError):
        ReadFileRequest(path="/tmp/file", max_lines=1001)

from pathlib import Path

import pytest

from zomah.access import InvalidWriteTarget, PathOutsideScope, WriteScope
from zomah.capabilities.write_file import (
    MAX_WRITE_BYTES,
    UnsupportedWriteContent,
    WriteFileRequest,
    write_file,
)


def _scope(root: Path) -> WriteScope:
    return WriteScope.from_paths([root])


def test_create_file_inside_write_root(tmp_path: Path) -> None:
    target = tmp_path / "note.md"

    response = write_file(
        WriteFileRequest(path=str(target), content="hello\n", mode="create"),
        _scope(tmp_path),
    )

    assert target.read_text() == "hello\n"
    assert response.previous_size_bytes is None
    assert response.bytes_written == 6
    assert response.new_size_bytes == 6


def test_create_refuses_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("existing")

    with pytest.raises(InvalidWriteTarget, match="already exists"):
        write_file(
            WriteFileRequest(path=str(target), content="new", mode="create"),
            _scope(tmp_path),
        )

    assert target.read_text() == "existing"


def test_replace_existing_text_file_atomically(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("old")
    target.chmod(0o640)

    response = write_file(
        WriteFileRequest(path=str(target), content="replacement", mode="replace"),
        _scope(tmp_path),
    )

    assert target.read_text() == "replacement"
    assert response.previous_size_bytes == 3
    assert response.new_size_bytes == len(b"replacement")
    assert target.stat().st_mode & 0o777 == 0o640


def test_replace_refuses_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "missing.md"

    with pytest.raises(InvalidWriteTarget, match="does not exist"):
        write_file(
            WriteFileRequest(path=str(target), content="new", mode="replace"),
            _scope(tmp_path),
        )


def test_append_requires_existing_text_file(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("one\n")

    response = write_file(
        WriteFileRequest(path=str(target), content="two\n", mode="append"),
        _scope(tmp_path),
    )

    assert target.read_text() == "one\ntwo\n"
    assert response.previous_size_bytes == 4
    assert response.new_size_bytes == 8

    with pytest.raises(InvalidWriteTarget, match="does not exist"):
        write_file(
            WriteFileRequest(
                path=str(tmp_path / "missing.md"), content="x", mode="append"
            ),
            _scope(tmp_path),
        )


def test_write_rejects_outside_root_and_parent_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()

    scope = _scope(root)

    with pytest.raises(PathOutsideScope):
        write_file(
            WriteFileRequest(
                path=str(outside / "nope.md"), content="x", mode="create"
            ),
            scope,
        )

    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathOutsideScope):
        write_file(
            WriteFileRequest(
                path=str(root / "escape" / "nope.md"),
                content="x",
                mode="create",
            ),
            scope,
        )


def test_write_rejects_existing_symlink_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "link.md"
    link.symlink_to(outside)

    with pytest.raises(InvalidWriteTarget, match="symlink"):
        write_file(
            WriteFileRequest(path=str(link), content="changed", mode="replace"),
            _scope(root),
        )

    assert outside.read_text() == "outside"


def test_write_requires_existing_parent_and_absolute_path(tmp_path: Path) -> None:
    scope = _scope(tmp_path)

    with pytest.raises(InvalidWriteTarget, match="absolute path"):
        write_file(
            WriteFileRequest(path="relative.md", content="x", mode="create"),
            scope,
        )

    with pytest.raises(InvalidWriteTarget, match="parent directory does not exist"):
        write_file(
            WriteFileRequest(
                path=str(tmp_path / "missing" / "note.md"),
                content="x",
                mode="create",
            ),
            scope,
        )

    assert not (tmp_path / "missing").exists()


def test_write_enforces_utf8_text_boundary_and_payload_limit(tmp_path: Path) -> None:
    scope = _scope(tmp_path)

    with pytest.raises(UnsupportedWriteContent, match="NUL"):
        write_file(
            WriteFileRequest(
                path=str(tmp_path / "nul.txt"),
                content="bad\x00text",
                mode="create",
            ),
            scope,
        )

    with pytest.raises(UnsupportedWriteContent, match="limit"):
        write_file(
            WriteFileRequest(
                path=str(tmp_path / "large.txt"),
                content="é" * MAX_WRITE_BYTES,
                mode="create",
            ),
            scope,
        )

    binary = tmp_path / "binary.dat"
    binary.write_bytes(b"abc\x00def")
    with pytest.raises(UnsupportedWriteContent, match="NUL"):
        write_file(
            WriteFileRequest(path=str(binary), content="text", mode="replace"),
            scope,
        )

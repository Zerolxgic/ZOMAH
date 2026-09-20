from pathlib import Path

import pytest

from zomah.access import InvalidWriteTarget, PathOutsideScope, WriteScope
from zomah.capabilities.move_file import MoveFileRequest, move_file


def _scope(*roots: Path) -> WriteScope:
    return WriteScope.from_paths(roots)


def test_move_file_inside_write_root(tmp_path: Path) -> None:
    source = tmp_path / "inbox" / "note.md"
    destination = tmp_path / "archive" / "note.md"
    source.parent.mkdir()
    destination.parent.mkdir()
    source.write_text("hello\n")

    response = move_file(
        MoveFileRequest(source=str(source), destination=str(destination)),
        _scope(tmp_path),
    )

    assert not source.exists()
    assert destination.read_text() == "hello\n"
    assert response.source == str(source)
    assert response.destination == str(destination)
    assert response.size_bytes == 6


def test_move_between_two_approved_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    source = first / "note.txt"
    destination = second / "note.txt"
    source.write_text("data")

    move_file(
        MoveFileRequest(source=str(source), destination=str(destination)),
        _scope(first, second),
    )

    assert not source.exists()
    assert destination.read_text() == "data"


def test_move_rejects_source_outside_write_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = outside / "note.md"
    source.write_text("keep")

    with pytest.raises(PathOutsideScope):
        move_file(
            MoveFileRequest(
                source=str(source), destination=str(root / "note.md")
            ),
            _scope(root),
        )

    assert source.read_text() == "keep"


def test_move_rejects_destination_outside_write_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = root / "note.md"
    source.write_text("keep")

    with pytest.raises(PathOutsideScope):
        move_file(
            MoveFileRequest(
                source=str(source), destination=str(outside / "note.md")
            ),
            _scope(root),
        )

    assert source.read_text() == "keep"
    assert not (outside / "note.md").exists()


def test_move_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(InvalidWriteTarget, match="does not exist"):
        move_file(
            MoveFileRequest(
                source=str(tmp_path / "missing.md"),
                destination=str(tmp_path / "new.md"),
            ),
            _scope(tmp_path),
        )


def test_move_rejects_symlink_source(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text("keep")
    link = tmp_path / "link.md"
    link.symlink_to(target)

    with pytest.raises(InvalidWriteTarget, match="symlink"):
        move_file(
            MoveFileRequest(
                source=str(link), destination=str(tmp_path / "moved.md")
            ),
            _scope(tmp_path),
        )

    assert target.read_text() == "keep"
    assert link.is_symlink()


def test_move_rejects_existing_destination_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    source.write_text("source")
    destination.write_text("destination")

    with pytest.raises(InvalidWriteTarget, match="already exists"):
        move_file(
            MoveFileRequest(source=str(source), destination=str(destination)),
            _scope(tmp_path),
        )

    assert source.read_text() == "source"
    assert destination.read_text() == "destination"


def test_move_rejects_missing_destination_parent(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("keep")

    with pytest.raises(InvalidWriteTarget, match="parent directory does not exist"):
        move_file(
            MoveFileRequest(
                source=str(source),
                destination=str(tmp_path / "missing" / "destination.md"),
            ),
            _scope(tmp_path),
        )

    assert source.read_text() == "keep"
    assert not (tmp_path / "missing").exists()


def test_move_rejects_directory_source(tmp_path: Path) -> None:
    source = tmp_path / "folder"
    source.mkdir()

    with pytest.raises(InvalidWriteTarget, match="not a regular file"):
        move_file(
            MoveFileRequest(
                source=str(source), destination=str(tmp_path / "moved")
            ),
            _scope(tmp_path),
        )

    assert source.is_dir()


def test_move_rejects_same_source_and_destination(tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_text("keep")

    with pytest.raises(InvalidWriteTarget, match="must differ|already exists"):
        move_file(
            MoveFileRequest(source=str(source), destination=str(source)),
            _scope(tmp_path),
        )

    assert source.read_text() == "keep"

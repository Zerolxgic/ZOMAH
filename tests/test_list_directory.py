from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.access import InvalidDirectoryTarget, PathOutsideScope, ReadScope
from zomah.capabilities import ListDirectoryRequest, list_directory


def test_list_directory_returns_sorted_structured_children(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    (root / "Beta").mkdir()
    (root / "alpha.txt").write_text("abc", encoding="utf-8")
    (root / "zeta.txt").write_text("hello", encoding="utf-8")

    response = list_directory(
        ListDirectoryRequest(path=str(root)),
        ReadScope.from_paths([root]),
    )

    assert response.path == str(root.resolve())
    assert [entry.name for entry in response.entries] == ["alpha.txt", "Beta", "zeta.txt"]
    assert [entry.kind for entry in response.entries] == ["file", "directory", "file"]
    assert response.entries[0].size_bytes == 3
    assert response.entries[1].size_bytes is None
    assert response.returned == 3
    assert response.truncated is False
    assert response.next_offset is None


def test_list_directory_is_non_recursive(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "inside.txt").write_text("inside", encoding="utf-8")

    response = list_directory(
        ListDirectoryRequest(path=str(root)),
        ReadScope.from_paths([root]),
    )

    assert [entry.name for entry in response.entries] == ["nested"]
    assert response.entries[0].kind == "directory"


def test_list_directory_paginates_large_directories(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    for name in ["a", "b", "c"]:
        (root / name).write_text(name, encoding="utf-8")

    first = list_directory(
        ListDirectoryRequest(path=str(root), max_entries=2),
        ReadScope.from_paths([root]),
    )
    second = list_directory(
        ListDirectoryRequest(path=str(root), offset=first.next_offset or 0, max_entries=2),
        ReadScope.from_paths([root]),
    )

    assert [entry.name for entry in first.entries] == ["a", "b"]
    assert first.truncated is True
    assert first.next_offset == 2
    assert [entry.name for entry in second.entries] == ["c"]
    assert second.truncated is False
    assert second.next_offset is None


def test_list_directory_reports_symlinks_without_following_them(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "elsewhere"
    link.symlink_to(outside, target_is_directory=True)

    response = list_directory(
        ListDirectoryRequest(path=str(root)),
        ReadScope.from_paths([root]),
    )

    assert response.entries[0].name == "elsewhere"
    assert response.entries[0].kind == "symlink"
    assert response.entries[0].size_bytes is None


def test_list_directory_rejects_directory_outside_scope(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(PathOutsideScope):
        list_directory(
            ListDirectoryRequest(path=str(outside)),
            ReadScope.from_paths([root]),
        )


def test_list_directory_rejects_symlink_escape_as_target(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "escape"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(PathOutsideScope):
        list_directory(
            ListDirectoryRequest(path=str(link)),
            ReadScope.from_paths([root]),
        )


def test_list_directory_rejects_file_target(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    target = root / "file.txt"
    target.write_text("text", encoding="utf-8")

    with pytest.raises(InvalidDirectoryTarget, match="not a directory"):
        list_directory(
            ListDirectoryRequest(path=str(target)),
            ReadScope.from_paths([root]),
        )


def test_list_directory_rejects_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()

    with pytest.raises(InvalidDirectoryTarget, match="absolute path"):
        list_directory(
            ListDirectoryRequest(path="."),
            ReadScope.from_paths([root]),
        )


def test_list_directory_request_caps_model_requested_entry_count() -> None:
    with pytest.raises(ValidationError):
        ListDirectoryRequest(path="/tmp", max_entries=1001)

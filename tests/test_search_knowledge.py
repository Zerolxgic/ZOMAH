from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.access import ReadScope
from zomah.capabilities import SearchKnowledgeRequest, search_knowledge
from zomah.knowledge import KnowledgeIndex


def make_index(tmp_path: Path, root: Path) -> KnowledgeIndex:
    return KnowledgeIndex(tmp_path / "knowledge.db", ReadScope.from_paths([root]))


def test_search_indexes_and_ranks_matching_documents(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "jev.md").write_text(
        "# Jev Evaluation\n\nJev evaluates model behavior and agent runs.\n",
        encoding="utf-8",
    )
    (root / "other.md").write_text(
        "# Grocery Notes\n\nRemember coffee beans.\n",
        encoding="utf-8",
    )
    index = make_index(tmp_path, root)

    report = index.refresh()
    response = search_knowledge(SearchKnowledgeRequest(query="Jev evaluation"), index)

    assert report.indexed_files == 2
    assert response.returned == 1
    assert response.results[0].rank == 1
    assert response.results[0].title == "Jev Evaluation"
    assert response.results[0].path == str((root / "jev.md").resolve())
    assert "Jev" in response.results[0].excerpt


def test_search_uses_any_sanitized_query_term(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "one.md").write_text("# One\n\nProjectState continuity.\n", encoding="utf-8")
    (root / "two.md").write_text("# Two\n\nOpenClaw adapter.\n", encoding="utf-8")
    index = make_index(tmp_path, root)
    index.refresh()

    response = search_knowledge(
        SearchKnowledgeRequest(query='ProjectState OR "OpenClaw"'),
        index,
    )

    assert response.returned == 2
    assert {result.title for result in response.results} == {"One", "Two"}


def test_refresh_updates_changed_files_and_removes_deleted_files(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Note\n\nalpha\n", encoding="utf-8")
    index = make_index(tmp_path, root)
    first = index.refresh()
    assert first.indexed_files == 1

    note.write_text("# Note\n\nbeta with more text\n", encoding="utf-8")
    second = index.refresh()
    assert second.indexed_files == 1
    assert search_knowledge(SearchKnowledgeRequest(query="beta"), index).returned == 1
    assert search_knowledge(SearchKnowledgeRequest(query="alpha"), index).returned == 0

    note.unlink()
    third = index.refresh()
    assert third.removed_files == 1
    assert search_knowledge(SearchKnowledgeRequest(query="beta"), index).returned == 0


def test_refresh_skips_symlinks_unsupported_and_binary_text(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()

    (root / "good.md").write_text("# Good\n\nsearchable anchor\n", encoding="utf-8")
    (root / "ignored.py").write_text("searchable anchor", encoding="utf-8")
    (root / "binary.md").write_bytes(b"abc\x00def")
    target = outside / "secret.md"
    target.write_text("# Secret\n\nsearchable anchor\n", encoding="utf-8")
    (root / "escape.md").symlink_to(target)

    index = make_index(tmp_path, root)
    report = index.refresh()
    response = search_knowledge(SearchKnowledgeRequest(query="searchable"), index)

    assert report.scanned_files == 2
    assert report.indexed_files == 1
    assert report.skipped_files == 1
    assert response.returned == 1
    assert response.results[0].title == "Good"


def test_refresh_ignores_tooling_directories(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "good.md").write_text(
        "# Good\n\nproject knowledge anchor\n", encoding="utf-8"
    )

    ignored = [".git", ".venv", "__pycache__", ".pytest_cache"]
    for directory_name in ignored:
        directory = root / directory_name
        directory.mkdir()
        (directory / "noise.md").write_text(
            "# Noise\n\nproject knowledge anchor\n", encoding="utf-8"
        )

    index = make_index(tmp_path, root)
    report = index.refresh()
    response = search_knowledge(
        SearchKnowledgeRequest(query="project knowledge anchor"), index
    )

    assert report.scanned_files == 1
    assert report.indexed_files == 1
    assert response.returned == 1
    assert response.results[0].title == "Good"


def test_refresh_is_incremental_for_unchanged_documents(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\n\ncontinuity\n", encoding="utf-8")
    index = make_index(tmp_path, root)
    index.refresh()

    report = index.refresh()

    assert report.indexed_files == 0
    assert report.unchanged_files == 1


def test_max_results_bounds_output(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    for number in range(5):
        (root / f"note-{number}.md").write_text(
            f"# Note {number}\n\nsharedterm\n", encoding="utf-8"
        )
    index = make_index(tmp_path, root)
    index.refresh()

    response = search_knowledge(
        SearchKnowledgeRequest(query="sharedterm", max_results=2), index
    )

    assert response.returned == 2
    assert [result.rank for result in response.results] == [1, 2]


def test_request_rejects_empty_query_and_large_limit() -> None:
    with pytest.raises(ValidationError):
        SearchKnowledgeRequest(query="   ")
    with pytest.raises(ValidationError):
        SearchKnowledgeRequest(query="state", max_results=51)


def test_query_without_searchable_terms_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    index = make_index(tmp_path, root)
    index.refresh()

    with pytest.raises(ValueError, match="searchable term"):
        search_knowledge(SearchKnowledgeRequest(query="___"), index)

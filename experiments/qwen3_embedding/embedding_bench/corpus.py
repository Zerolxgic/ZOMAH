"""The benchmark corpus: exactly the documents ZOMAH's real KnowledgeIndex holds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zomah.access import ReadScope
from zomah.knowledge import (
    DEFAULT_IGNORED_DIRECTORIES,
    DEFAULT_TEXT_EXTENSIONS,
    MAX_INDEX_FILE_BYTES,
    KnowledgeIndex,
    KnowledgeRefreshReport,
)

# KnowledgeIndex's own exclusions, plus this experiment's directory: its README
# describes the benchmark queries and would otherwise become a retrieval target.
BENCHMARK_IGNORED_DIRECTORIES = DEFAULT_IGNORED_DIRECTORIES | {"experiments"}


@dataclass(frozen=True, slots=True)
class Document:
    path: str  # relative POSIX path under the corpus root
    title: str
    text: str
    size_bytes: int


def build_index(root: Path, db_path: Path) -> tuple[KnowledgeIndex, KnowledgeRefreshReport]:
    """Build a fresh, unmodified KnowledgeIndex over ``root`` and refresh it once."""

    index = KnowledgeIndex(
        db_path,
        ReadScope.from_paths([root]),
        extensions=DEFAULT_TEXT_EXTENSIONS,
        ignored_directories=BENCHMARK_IGNORED_DIRECTORIES,
        max_file_bytes=MAX_INDEX_FILE_BYTES,
    )
    return index, index.refresh()


def load_documents(index: KnowledgeIndex, root: Path) -> list[Document]:
    """Read the indexed documents back out of the lexical index.

    Taking the corpus from the index (rather than walking the tree again) makes
    the semantic side embed exactly the files and text FTS5 ranks: same
    extension, size, UTF-8, NUL, symlink, and directory rules.
    """

    with index.connect() as conn:
        rows = conn.execute(
            """
            SELECT d.path, d.title, d.size_bytes, f.body
            FROM knowledge_documents AS d
            JOIN knowledge_fts AS f ON f.path = d.path
            ORDER BY d.path
            """
        ).fetchall()
    documents = [
        Document(
            path=relative_path(row["path"], root),
            title=row["title"],
            text=row["body"],
            size_bytes=row["size_bytes"],
        )
        for row in rows
    ]
    documents.sort(key=lambda doc: doc.path)
    return documents


def relative_path(path: str, root: Path) -> str:
    return Path(path).relative_to(root).as_posix()


def corpus_config(root: Path) -> dict[str, object]:
    return {
        "root": str(root),
        "extensions": sorted(DEFAULT_TEXT_EXTENSIONS),
        "ignored_directories": sorted(BENCHMARK_IGNORED_DIRECTORIES),
        "max_file_bytes": MAX_INDEX_FILE_BYTES,
        "source": "zomah.knowledge.KnowledgeIndex (fresh temporary index)",
    }

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from zomah.access import ReadScope


DEFAULT_TEXT_EXTENSIONS = frozenset({".md", ".txt", ".rst"})
DEFAULT_IGNORED_DIRECTORIES = frozenset(
    {".git", ".venv", "__pycache__", ".pytest_cache"}
)
MAX_INDEX_FILE_BYTES = 2 * 1024 * 1024


class KnowledgeIndexError(RuntimeError):
    """Base error for ZOMAH's derived knowledge index."""


class KnowledgeIndexUnavailable(KnowledgeIndexError):
    """Raised when the local SQLite build does not provide FTS5."""


@dataclass(frozen=True, slots=True)
class KnowledgeRefreshReport:
    scanned_files: int
    indexed_files: int
    unchanged_files: int
    removed_files: int
    skipped_files: int


def default_knowledge_db_path() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "zomah" / "knowledge.db"


class KnowledgeIndex:
    """Derived SQLite FTS5 index over approved UTF-8 text roots.

    The index is intentionally rebuildable and separate from canonical project
    state. `zomah.db` remains authoritative state; `knowledge.db` is only a
    retrieval accelerator over files that already exist on disk.
    """

    def __init__(
        self,
        db_path: str | Path,
        scope: ReadScope,
        *,
        extensions: frozenset[str] = DEFAULT_TEXT_EXTENSIONS,
        ignored_directories: frozenset[str] = DEFAULT_IGNORED_DIRECTORIES,
        max_file_bytes: int = MAX_INDEX_FILE_BYTES,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.scope = scope
        self.extensions = frozenset(ext.casefold() for ext in extensions)
        self.ignored_directories = frozenset(ignored_directories)
        self.max_file_bytes = max_file_bytes

        if not self.extensions:
            raise ValueError("at least one knowledge text extension is required")
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def refresh(self) -> KnowledgeRefreshReport:
        """Incrementally synchronize approved text files into the FTS index."""

        seen_paths: set[str] = set()
        scanned = 0
        indexed = 0
        unchanged = 0
        skipped = 0
        now = datetime.now(timezone.utc).isoformat()

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")

            metadata = {
                row["path"]: (row["modified_ns"], row["size_bytes"])
                for row in conn.execute(
                    "SELECT path, modified_ns, size_bytes FROM knowledge_documents"
                )
            }

            for path in self._iter_candidate_files():
                scanned += 1
                canonical_path = str(path)
                seen_paths.add(canonical_path)

                try:
                    stat = path.stat()
                except OSError:
                    skipped += 1
                    continue

                if stat.st_size > self.max_file_bytes:
                    skipped += 1
                    self._delete_document(conn, canonical_path)
                    continue

                fingerprint = (stat.st_mtime_ns, stat.st_size)
                if metadata.get(canonical_path) == fingerprint:
                    unchanged += 1
                    continue

                try:
                    text = path.read_text(encoding="utf-8", errors="strict")
                except (OSError, UnicodeDecodeError):
                    skipped += 1
                    self._delete_document(conn, canonical_path)
                    continue

                if "\x00" in text:
                    skipped += 1
                    self._delete_document(conn, canonical_path)
                    continue

                title = _document_title(path, text)
                self._upsert_document(
                    conn,
                    path=canonical_path,
                    title=title,
                    body=text,
                    modified_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                    indexed_at=now,
                )
                indexed += 1

            known_paths = {
                row["path"]
                for row in conn.execute("SELECT path FROM knowledge_documents")
            }
            stale_paths = known_paths - seen_paths
            for path in stale_paths:
                self._delete_document(conn, path)

        return KnowledgeRefreshReport(
            scanned_files=scanned,
            indexed_files=indexed,
            unchanged_files=unchanged,
            removed_files=len(stale_paths),
            skipped_files=skipped,
        )

    def search(self, query: str, *, limit: int) -> list[sqlite3.Row]:
        fts_query = _fts_query(query)

        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT
                        d.path,
                        d.title,
                        d.modified_ns,
                        d.size_bytes,
                        snippet(knowledge_fts, 2, '', '', ' … ', 24) AS excerpt,
                        -bm25(knowledge_fts, 0.0, 5.0, 1.0) AS relevance
                    FROM knowledge_fts
                    JOIN knowledge_documents AS d ON d.path = knowledge_fts.path
                    WHERE knowledge_fts MATCH ?
                    ORDER BY bm25(knowledge_fts, 0.0, 5.0, 1.0), d.path
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
            )

    def _initialize(self) -> None:
        try:
            with self.connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_documents (
                        path TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        modified_ns INTEGER NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        indexed_at TEXT NOT NULL
                    );

                    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                        path UNINDEXED,
                        title,
                        body,
                        tokenize = 'unicode61 remove_diacritics 2'
                    );
                    """
                )
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).casefold():
                raise KnowledgeIndexUnavailable(
                    "SQLite FTS5 is required for search_knowledge"
                ) from exc
            raise

    def _iter_candidate_files(self):
        for root in self.scope.roots:
            for directory, dirnames, filenames in os.walk(root, followlinks=False):
                current = Path(directory)

                # Be explicit: never descend into symlink directories even if
                # platform traversal behavior changes.
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name not in self.ignored_directories
                    and not (current / name).is_symlink()
                ]

                for filename in filenames:
                    path = current / filename
                    if path.is_symlink():
                        continue
                    if path.suffix.casefold() not in self.extensions:
                        continue
                    if not path.is_file():
                        continue
                    yield path

    @staticmethod
    def _delete_document(conn: sqlite3.Connection, path: str) -> None:
        conn.execute("DELETE FROM knowledge_fts WHERE path = ?", (path,))
        conn.execute("DELETE FROM knowledge_documents WHERE path = ?", (path,))

    @staticmethod
    def _upsert_document(
        conn: sqlite3.Connection,
        *,
        path: str,
        title: str,
        body: str,
        modified_ns: int,
        size_bytes: int,
        indexed_at: str,
    ) -> None:
        conn.execute("DELETE FROM knowledge_fts WHERE path = ?", (path,))
        conn.execute(
            """
            INSERT INTO knowledge_documents(path, title, modified_ns, size_bytes, indexed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                title = excluded.title,
                modified_ns = excluded.modified_ns,
                size_bytes = excluded.size_bytes,
                indexed_at = excluded.indexed_at
            """,
            (path, title, modified_ns, size_bytes, indexed_at),
        )
        conn.execute(
            "INSERT INTO knowledge_fts(path, title, body) VALUES (?, ?, ?)",
            (path, title, body),
        )


def _document_title(path: Path, text: str) -> str:
    if path.suffix.casefold() == ".md":
        for line in text.splitlines()[:40]:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                if title:
                    return title
    return path.stem


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[^\W_]+(?:[-_.][^\W_]+)*", query, flags=re.UNICODE)
    if not tokens:
        raise ValueError("query must contain at least one searchable term")

    # Quote every token so model-provided text cannot become FTS operators.
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{token}"' for token in escaped)

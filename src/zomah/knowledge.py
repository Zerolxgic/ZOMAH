from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from zomah.access import ReadScope


DEFAULT_TEXT_EXTENSIONS = frozenset({".md", ".txt", ".rst"})
DEFAULT_IGNORED_DIRECTORIES = frozenset(
    {".git", ".venv", "__pycache__", ".pytest_cache"}
)
MAX_INDEX_FILE_BYTES = 2 * 1024 * 1024

# Localization: lines considered together when scoring a region, and the
# largest excerpt returned for one result (characters, excluding ellipses).
LOCALIZE_WINDOW_LINES = 3
MAX_EXCERPT_CHARS = 160
_EXCERPT_LEAD_CHARS = 40


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


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    """One ranked document plus its strongest lexically matching region.

    ``excerpt`` is cut from exactly lines ``start_line``..``end_line`` (1-based,
    numbered like ``read_file``) of the indexed document text.
    """

    path: str
    title: str
    size_bytes: int
    relevance: float
    start_line: int
    end_line: int
    excerpt: str


@dataclass(frozen=True, slots=True)
class LocalizedRegion:
    start_line: int
    end_line: int
    excerpt: str


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

    def search(self, query: str, *, limit: int) -> list[KnowledgeHit]:
        """Rank documents with FTS5/BM25, then localize the match in each one.

        Ranking and order come only from FTS5. Localization runs on the stored
        document text (current as of the last refresh) and never reorders.
        """

        fts_query = _fts_query(query)

        with self.connect() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT
                        d.path,
                        d.title,
                        d.size_bytes,
                        knowledge_fts.body AS body,
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

        hits: list[KnowledgeHit] = []
        for row in rows:
            region = localize(row["body"], query)
            hits.append(
                KnowledgeHit(
                    path=row["path"],
                    title=row["title"],
                    size_bytes=row["size_bytes"],
                    relevance=row["relevance"],
                    start_line=region.start_line,
                    end_line=region.end_line,
                    excerpt=region.excerpt,
                )
            )
        return hits

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


_QUERY_TOKEN = re.compile(r"[^\W_]+(?:[-_.][^\W_]+)*", flags=re.UNICODE)
_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _normalize_word(word: str) -> str:
    """Case- and diacritic-insensitive form, mirroring FTS5 unicode61."""

    decomposed = unicodedata.normalize("NFKD", word.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _query_phrases(query: str) -> tuple[tuple[str, ...], ...]:
    """The FTS query terms as word sequences (``tool-result`` → tool, result)."""

    phrases: list[tuple[str, ...]] = []
    for token in _QUERY_TOKEN.findall(query):
        words = tuple(_normalize_word(word) for word in _WORD.findall(token))
        if words and words not in phrases:
            phrases.append(words)
    return tuple(phrases)


def _line_matches(line: str, phrases: tuple[tuple[str, ...], ...]) -> list[tuple[int, int]]:
    """(phrase index, character offset) for every query phrase in one line."""

    words = [(_normalize_word(m.group()), m.start()) for m in _WORD.finditer(line)]
    normalized = [word for word, _ in words]
    matches: list[tuple[int, int]] = []
    for position, (word, offset) in enumerate(words):
        for index, phrase in enumerate(phrases):
            if phrase[0] == word and tuple(normalized[position : position + len(phrase)]) == phrase:
                matches.append((index, offset))
    return matches


def localize(body: str, query: str) -> LocalizedRegion:
    """Find the strongest lexical region of ``body`` for ``query``.

    Lines are split on ``\n`` only, matching ``read_file``'s numbering (both
    read text with universal newlines). Every window of
    ``LOCALIZE_WINDOW_LINES`` lines is scored by distinct query terms, then
    total term occurrences; ties go to the earliest window. The chosen window
    is trimmed to its first and last matching lines, and the excerpt is cut
    from that region (centred on the first match when the region is longer
    than ``MAX_EXCERPT_CHARS``). The reported range is the lines the excerpt
    actually covers. Terms spanning a line break are not matched. Without any
    body match (e.g. a title-only hit) the first non-blank line is returned.
    """

    lines = body.split("\n")
    phrases = _query_phrases(query)
    matches = [_line_matches(line, phrases) for line in lines]

    best: tuple[tuple[int, int, int], int] | None = None
    for start in range(len(lines)):
        window = [m for line_matches in matches[start : start + LOCALIZE_WINDOW_LINES] for m in line_matches]
        if not window:
            continue
        key = (-len({index for index, _ in window}), -len(window), start)
        if best is None or key < best[0]:
            best = (key, start)

    if best is None:
        first = last = next((i for i, line in enumerate(lines) if line.strip()), 0)
        anchor = 0
    else:
        start = best[1]
        matched = [i for i in range(start, min(start + LOCALIZE_WINDOW_LINES, len(lines))) if matches[i]]
        first, last = matched[0], matched[-1]
        anchor = min(offset for _, offset in matches[first])

    region = "\n".join(lines[first : last + 1])
    if len(region) <= MAX_EXCERPT_CHARS:
        cut_start, cut_end = 0, len(region)
    else:
        cut_start = max(0, anchor - _EXCERPT_LEAD_CHARS)
        cut_end = min(len(region), cut_start + MAX_EXCERPT_CHARS)
        cut_start = max(0, cut_end - MAX_EXCERPT_CHARS)

    start_line = first + region.count("\n", 0, cut_start) + 1
    end_line = first + region.count("\n", 0, max(cut_start, cut_end - 1)) + 1
    excerpt = (
        ("…" if cut_start > 0 else "")
        + region[cut_start:cut_end]
        + ("…" if cut_end < len(region) else "")
    )
    return LocalizedRegion(start_line=start_line, end_line=end_line, excerpt=excerpt)


def _fts_query(query: str) -> str:
    tokens = _QUERY_TOKEN.findall(query)
    if not tokens:
        raise ValueError("query must contain at least one searchable term")

    # Quote every token so model-provided text cannot become FTS operators.
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{token}"' for token in escaped)

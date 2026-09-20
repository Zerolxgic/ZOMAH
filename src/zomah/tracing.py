from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


MAX_WORKER_CHARS = 128
MAX_CAPABILITY_CHARS = 128
MAX_TARGET_CHARS = 1024


class TraceError(RuntimeError):
    """Raised when mandatory local trace state cannot be recorded safely."""


class TraceOutcome(str, Enum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TraceRecord(BaseModel):
    """Compact local record of one model-boundary capability invocation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    worker: str
    capability: str
    started_at: datetime
    finished_at: datetime | None = None
    outcome: TraceOutcome
    error_code: str | None = None
    target: str | None = None


def default_trace_db_path() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "zomah" / "trace.db"


class TraceStore:
    """Minimal SQLite trace store for model-boundary capability calls.

    Traces intentionally contain no prompts, raw request payloads, capability
    results, stdout/stderr, or exception details. They record only enough
    metadata to reconstruct what capability was attempted, by which worker,
    against which compact target/reference, and how the call ended.
    """

    def __init__(self, database: str | Path):
        self.database = Path(database).expanduser()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        try:
            with self.connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS capability_traces (
                        run_id TEXT PRIMARY KEY,
                        worker TEXT NOT NULL,
                        capability TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        outcome TEXT NOT NULL CHECK (
                            outcome IN ('started', 'succeeded', 'failed')
                        ),
                        error_code TEXT,
                        target TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_capability_traces_started_at
                    ON capability_traces(started_at DESC);

                    CREATE INDEX IF NOT EXISTS idx_capability_traces_worker_capability
                    ON capability_traces(worker, capability, started_at DESC);
                    """
                )
        except sqlite3.Error as exc:
            raise TraceError("failed to initialize local trace storage") from exc

    def start_run(
        self,
        *,
        worker: str,
        capability: str,
        target: str | None = None,
    ) -> str:
        worker = _bounded_required(worker, field="worker", maximum=MAX_WORKER_CHARS)
        capability = _bounded_required(
            capability,
            field="capability",
            maximum=MAX_CAPABILITY_CHARS,
        )
        target = _bounded_optional(target, maximum=MAX_TARGET_CHARS)
        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            with self.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO capability_traces (
                        run_id, worker, capability, started_at,
                        finished_at, outcome, error_code, target
                    ) VALUES (?, ?, ?, ?, NULL, ?, NULL, ?)
                    """,
                    (
                        run_id,
                        worker,
                        capability,
                        started_at,
                        TraceOutcome.STARTED.value,
                        target,
                    ),
                )
        except sqlite3.Error as exc:
            raise TraceError("failed to start capability trace") from exc

        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        outcome: TraceOutcome,
        error_code: str | None = None,
    ) -> None:
        if outcome is TraceOutcome.STARTED:
            raise ValueError("finished trace outcome cannot remain started")
        finished_at = datetime.now(timezone.utc).isoformat()
        error_code = _bounded_optional(error_code, maximum=128)

        try:
            with self.connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE capability_traces
                    SET finished_at = ?, outcome = ?, error_code = ?
                    WHERE run_id = ?
                    """,
                    (finished_at, outcome.value, error_code, run_id),
                )
                if cursor.rowcount != 1:
                    raise TraceError(f"trace run was not found: {run_id}")
        except TraceError:
            raise
        except sqlite3.Error as exc:
            raise TraceError("failed to finish capability trace") from exc

    def get(self, run_id: str) -> TraceRecord:
        try:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM capability_traces WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise TraceError("failed to read capability trace") from exc

        if row is None:
            raise TraceError(f"trace run was not found: {run_id}")
        return _record(row)

    def recent(self, *, limit: int = 20) -> list[TraceRecord]:
        limit = max(1, min(limit, 100))
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM capability_traces
                    ORDER BY started_at DESC, run_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise TraceError("failed to read recent capability traces") from exc
        return [_record(row) for row in rows]


def _record(row: sqlite3.Row) -> TraceRecord:
    return TraceRecord(
        run_id=row["run_id"],
        worker=row["worker"],
        capability=row["capability"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        outcome=row["outcome"],
        error_code=row["error_code"],
        target=row["target"],
    )


def _bounded_required(value: str, *, field: str, maximum: int) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise TraceError(f"trace {field} must not be empty")
    if len(cleaned) > maximum:
        raise TraceError(f"trace {field} exceeds {maximum} characters")
    return cleaned


def _bounded_optional(value: str | None, *, maximum: int) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > maximum:
        return cleaned[: maximum - 1] + "…"
    return cleaned

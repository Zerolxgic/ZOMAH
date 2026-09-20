from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    Decision,
    DecisionProposal,
    DecisionStatus,
    DecisionTransition,
    ImportantPath,
    ProjectState,
    ProjectStatePatch,
)


class StateError(RuntimeError):
    pass


class ProjectNotFound(StateError):
    pass


class RevisionConflict(StateError):
    pass


@dataclass(frozen=True)
class Paths:
    database: Path
    markdown_root: Path | None = None


def default_database_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return root / "zomah" / "zomah.db"


class ProjectStateRepository:
    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(schema)

    def create(self, state: ProjectState) -> ProjectState:
        now = _iso(state.updated_at)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, name, status, phase, summary, current_focus,
                    last_action, next_action, revision, updated_at, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.id,
                    state.name,
                    state.status.value,
                    state.phase,
                    state.summary,
                    state.current_focus,
                    state.last_action,
                    state.next_action,
                    state.revision,
                    now,
                    state.updated_by,
                ),
            )
            _insert_questions(conn, state.id, state.open_questions, now)
            _insert_blockers(conn, state.id, state.blockers, now)
            _insert_decisions(conn, state.id, state.decisions)
            _insert_paths(conn, state.id, state.important_paths)
        return self.get(state.id)

    def get(self, project_id: str) -> ProjectState:
        with self.connect() as conn:
            project = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise ProjectNotFound(project_id)

            questions = [
                row["text"]
                for row in conn.execute(
                    """
                    SELECT text FROM project_open_questions
                    WHERE project_id = ? AND is_resolved = 0
                    ORDER BY id
                    """,
                    (project_id,),
                )
            ]
            blockers = [
                row["text"]
                for row in conn.execute(
                    """
                    SELECT text FROM project_blockers
                    WHERE project_id = ? AND is_resolved = 0
                    ORDER BY id
                    """,
                    (project_id,),
                )
            ]
            decisions = [
                Decision(
                    id=row["decision_id"],
                    statement=row["statement"],
                    rationale=row["rationale"],
                    status=row["status"],
                    created_at=row["created_at"],
                    superseded_by=row["superseded_by"],
                )
                for row in conn.execute(
                    """
                    SELECT * FROM project_decisions
                    WHERE project_id = ? ORDER BY created_at, decision_id
                    """,
                    (project_id,),
                )
            ]
            paths = [
                ImportantPath(
                    role=row["role"], path=row["path"], description=row["description"]
                )
                for row in conn.execute(
                    "SELECT role, path, description FROM project_paths WHERE project_id = ? ORDER BY id",
                    (project_id,),
                )
            ]

        return ProjectState(
            id=project["id"],
            name=project["name"],
            status=project["status"],
            phase=project["phase"],
            summary=project["summary"],
            current_focus=project["current_focus"],
            last_action=project["last_action"],
            next_action=project["next_action"],
            open_questions=questions,
            decisions=decisions,
            important_paths=paths,
            blockers=blockers,
            revision=project["revision"],
            updated_at=project["updated_at"],
            updated_by=project["updated_by"],
        )

    def apply_patch(
        self,
        project_id: str,
        patch: ProjectStatePatch,
        *,
        actor: str,
    ) -> ProjectState:
        actor = _actor(actor)
        now = datetime.now(timezone.utc)
        now_iso = _iso(now)

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT revision FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if current is None:
                raise ProjectNotFound(project_id)
            if current["revision"] != patch.expected_revision:
                raise RevisionConflict(
                    f"expected revision {patch.expected_revision}, current revision {current['revision']}"
                )

            updates: dict[str, object] = {
                "updated_at": now_iso,
                "updated_by": actor,
                "revision": patch.expected_revision + 1,
            }
            for field in ("name", "status", "phase", "summary", "current_focus"):
                value = getattr(patch, field)
                if value is not None:
                    updates[field] = value.value if hasattr(value, "value") else value

            # Nullable scalars distinguish omission from an explicit null via
            # Pydantic's model_fields_set.
            for field in ("last_action", "next_action"):
                if field in patch.model_fields_set:
                    updates[field] = getattr(patch, field)

            assignments = ", ".join(f"{key} = ?" for key in updates)
            conn.execute(
                f"UPDATE projects SET {assignments} WHERE id = ?",
                (*updates.values(), project_id),
            )

            _insert_questions(conn, project_id, patch.add_open_questions, now_iso)
            _resolve_items(
                conn,
                "project_open_questions",
                project_id,
                patch.resolve_open_questions,
                now_iso,
            )
            _insert_blockers(conn, project_id, patch.add_blockers, now_iso)
            _resolve_items(
                conn, "project_blockers", project_id, patch.resolve_blockers, now_iso
            )
            _insert_decision_proposals(conn, project_id, patch.add_decisions, now_iso)
            _insert_paths(conn, project_id, patch.add_important_paths)

        return self.get(project_id)

    def transition_decision(
        self,
        project_id: str,
        transition: DecisionTransition,
        *,
        expected_revision: int,
        actor: str,
    ) -> ProjectState:
        """Apply an authorized decision lifecycle transition.

        This method is intentionally separate from the model-facing project
        state patch. A future harness/admin adapter may call it only after the
        relevant authorization decision has been made outside the model input.
        """

        actor = _actor(actor)
        now_iso = _iso(datetime.now(timezone.utc))

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT revision FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if current is None:
                raise ProjectNotFound(project_id)
            if current["revision"] != expected_revision:
                raise RevisionConflict(
                    f"expected revision {expected_revision}, current revision {current['revision']}"
                )

            decision = conn.execute(
                """
                SELECT status FROM project_decisions
                WHERE decision_id = ? AND project_id = ?
                """,
                (transition.decision_id, project_id),
            ).fetchone()
            if decision is None:
                raise StateError(f"decision not found: {transition.decision_id}")

            _validate_decision_transition(
                DecisionStatus(decision["status"]), transition.status
            )

            conn.execute(
                """
                UPDATE project_decisions
                SET status = ?, superseded_by = ?
                WHERE decision_id = ? AND project_id = ?
                """,
                (
                    transition.status.value,
                    transition.superseded_by,
                    transition.decision_id,
                    project_id,
                ),
            )
            conn.execute(
                """
                UPDATE projects
                SET revision = ?, updated_at = ?, updated_by = ?
                WHERE id = ?
                """,
                (expected_revision + 1, now_iso, actor, project_id),
            )

        return self.get(project_id)

    def export_markdown(self, project_id: str, destination: str | Path) -> Path:
        state = self.get(project_id)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(state), encoding="utf-8")
        return path


def render_markdown(state: ProjectState) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None"

    decisions = (
        "\n\n".join(
            f"### {d.id}\n\n"
            f"**Status:** {d.status.value}\n\n"
            f"{d.statement}\n\n"
            f"**Rationale:** {d.rationale}"
            + (f"\n\n**Superseded by:** {d.superseded_by}" if d.superseded_by else "")
            for d in state.decisions
        )
        if state.decisions
        else "None"
    )

    paths = (
        "\n".join(
            f"- **{p.role}:** `{p.path}`" + (f" — {p.description}" if p.description else "")
            for p in state.important_paths
        )
        if state.important_paths
        else "- None"
    )

    return f"""# {state.name}\n\n**Project ID:** `{state.id}`  \n**Status:** {state.status.value}  \n**Phase:** {state.phase}  \n**Revision:** {state.revision}  \n**Updated:** {state.updated_at.isoformat()}  \n**Updated by:** {state.updated_by}\n\n## Summary\n\n{state.summary}\n\n## Current Focus\n\n{state.current_focus}\n\n## Last Action\n\n{state.last_action or 'None'}\n\n## Next Action\n\n{state.next_action or 'None'}\n\n## Open Questions\n\n{bullets(state.open_questions)}\n\n## Blockers\n\n{bullets(state.blockers)}\n\n## Important Paths\n\n{paths}\n\n## Decisions\n\n{decisions}\n"""


def _insert_questions(conn: sqlite3.Connection, project_id: str, items: list[str], now: str) -> None:
    for item in items:
        conn.execute(
            """
            INSERT INTO project_open_questions(project_id, text, is_resolved, created_at)
            VALUES (?, ?, 0, ?)
            ON CONFLICT(project_id, text) DO UPDATE SET
                is_resolved = 0,
                resolved_at = NULL
            """,
            (project_id, item, now),
        )


def _insert_blockers(conn: sqlite3.Connection, project_id: str, items: list[str], now: str) -> None:
    for item in items:
        conn.execute(
            """
            INSERT INTO project_blockers(project_id, text, is_resolved, created_at)
            VALUES (?, ?, 0, ?)
            ON CONFLICT(project_id, text) DO UPDATE SET
                is_resolved = 0,
                resolved_at = NULL
            """,
            (project_id, item, now),
        )


def _resolve_items(
    conn: sqlite3.Connection,
    table: str,
    project_id: str,
    items: list[str],
    now: str,
) -> None:
    if table not in {"project_open_questions", "project_blockers"}:
        raise ValueError("invalid resolution table")
    for item in items:
        conn.execute(
            f"""
            UPDATE {table}
            SET is_resolved = 1, resolved_at = ?
            WHERE project_id = ? AND text = ? AND is_resolved = 0
            """,
            (now, project_id, item),
        )


def _insert_decisions(
    conn: sqlite3.Connection, project_id: str, items: list[Decision]
) -> None:
    """Insert complete decisions from trusted bootstrap/admin state."""

    for item in items:
        conn.execute(
            """
            INSERT INTO project_decisions(
                decision_id, project_id, statement, rationale, status, created_at, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                project_id,
                item.statement,
                item.rationale,
                item.status.value,
                _iso(item.created_at),
                item.superseded_by,
            ),
        )


def _insert_decision_proposals(
    conn: sqlite3.Connection,
    project_id: str,
    items: list[DecisionProposal],
    created_at: str,
) -> None:
    for item in items:
        conn.execute(
            """
            INSERT INTO project_decisions(
                decision_id, project_id, statement, rationale, status, created_at, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                item.id,
                project_id,
                item.statement,
                item.rationale,
                DecisionStatus.PROPOSED.value,
                created_at,
            ),
        )


def _validate_decision_transition(
    current: DecisionStatus, target: DecisionStatus
) -> None:
    allowed = {
        DecisionStatus.PROPOSED: {
            DecisionStatus.ACCEPTED,
            DecisionStatus.INVALIDATED,
            DecisionStatus.CANCELLED,
        },
        DecisionStatus.ACCEPTED: {
            DecisionStatus.SUPERSEDED,
            DecisionStatus.INVALIDATED,
            DecisionStatus.CANCELLED,
        },
    }
    if target not in allowed.get(current, set()):
        raise StateError(
            f"invalid decision transition: {current.value} -> {target.value}"
        )


def _insert_paths(conn: sqlite3.Connection, project_id: str, items: list[ImportantPath]) -> None:
    for item in items:
        conn.execute(
            """
            INSERT OR IGNORE INTO project_paths(project_id, role, path, description)
            VALUES (?, ?, ?, ?)
            """,
            (project_id, item.role, item.path, item.description),
        )


def _actor(value: str) -> str:
    actor = value.strip()
    if not actor:
        raise ValueError("actor must not be blank")
    return actor


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()

from pathlib import Path

from zomah.state import (
    Decision,
    ImportantPath,
    ProjectState,
    ProjectStateRepository,
    default_database_path,
)


def main() -> None:
    database = default_database_path()
    repo = ProjectStateRepository(database)
    repo.initialize()

    state = ProjectState(
        id="zomah",
        name="ZOMAH",
        phase="state",
        summary="Minimal local agent control plane for Elyria and future workers.",
        current_focus="Implement and validate ProjectState persistence.",
        last_action="Established the Z1 ProjectState implementation scaffold.",
        next_action="Run the state slice locally, inspect the database, and choose the next Z1 test.",
        decisions=[
            Decision(
                id="ADR-001",
                statement="ZOMAH is a minimal control plane, not a full agent framework.",
                rationale="Existing workers and runtimes should retain capabilities they already provide well.",
            ),
            Decision(
                id="ADR-002",
                statement="SQLite is canonical project state; Markdown is a generated human-readable view.",
                rationale="Structured validated state needs one authoritative local source of truth.",
            ),
        ],
        important_paths=[
            ImportantPath(
                role="repo",
                path=str(Path.cwd()),
                description="Local ZOMAH implementation workspace.",
            )
        ],
        updated_by="zerrius",
    )

    try:
        created = repo.create(state)
        action = "created"
    except Exception as exc:
        # Keep the example intentionally explicit: a duplicate bootstrap should
        # never silently overwrite existing state.
        if "UNIQUE constraint failed: projects.id" not in str(exc):
            raise
        created = repo.get("zomah")
        action = "already exists"

    mirror = database.parent / "exports" / "zomah.md"
    repo.export_markdown("zomah", mirror)

    print(f"ProjectState {action}: {created.id} revision={created.revision}")
    print(f"SQLite: {database}")
    print(f"Markdown mirror: {mirror}")


if __name__ == "__main__":
    main()

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT NOT NULL,
    summary TEXT NOT NULL,
    current_focus TEXT NOT NULL,
    last_action TEXT,
    next_action TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS project_open_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    is_resolved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE(project_id, text)
);

CREATE TABLE IF NOT EXISTS project_blockers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    is_resolved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE(project_id, text)
);

CREATE TABLE IF NOT EXISTS project_decisions (
    decision_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    superseded_by TEXT
);

CREATE TABLE IF NOT EXISTS project_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    description TEXT,
    UNIQUE(project_id, role, path)
);

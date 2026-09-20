# ProjectState Persistence Contract

## Status

**Version:** v0  
**State:** Accepted for initial implementation  
**Canonical store:** SQLite  
**Human-readable mirror:** Generated Markdown

## Purpose

This contract defines how ZOMAH stores, updates, and exports `ProjectState`.

`ProjectState` answers one question:

> Where does this project stand right now?

The persistence layer must preserve that current truth without requiring the worker to directly manipulate database rows, rewrite complete state documents, or treat generated Markdown as authoritative.

## 1. Canonical storage

SQLite is the sole canonical state store in v0.

Default database location on Linux:

```text
$XDG_DATA_HOME/zomah/zomah.db
```

When `XDG_DATA_HOME` is unset:

```text
~/.local/share/zomah/zomah.db
```

The location may be overridden by ZOMAH configuration, but the database should not live inside the source repository by default.

Generated Markdown is a view of SQLite state. It is never a second source of truth.

## 2. SQLite operating rules

ZOMAH should open the database with:

```text
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
```

State-changing operations execute inside a transaction.

A successful `update_project_state` call either commits the complete patch or changes nothing.

Schema versioning should use SQLite's built-in:

```text
PRAGMA user_version;
```

A separate migration framework is not required for v0.

## 3. Canonical tables

### `projects`

Stores the scalar portion of current project state.

```text
id               TEXT PRIMARY KEY
name             TEXT NOT NULL
status           TEXT NOT NULL
phase            TEXT
summary          TEXT
current_focus    TEXT
last_action      TEXT
next_action      TEXT
revision         INTEGER NOT NULL DEFAULT 0
created_at       TEXT NOT NULL
updated_at       TEXT NOT NULL
updated_by       TEXT NOT NULL
```

Allowed project status values:

```text
active
paused
blocked
completed
archived
```

`phase` remains free-form and short. ZOMAH should not create a universal phase taxonomy in v0.

`revision` is internal harness metadata. It does not need to be exposed as part of Elyria's ordinary conversational project summary, but it is returned with state operations so updates can use optimistic concurrency.

### `project_open_questions`

```text
id               TEXT PRIMARY KEY
project_id       TEXT NOT NULL REFERENCES projects(id)
question         TEXT NOT NULL
status           TEXT NOT NULL DEFAULT 'open'
created_at       TEXT NOT NULL
resolved_at      TEXT
resolution       TEXT
```

Allowed status values:

```text
open
resolved
```

`get_project_state` returns only open questions by default.

Resolved questions remain stored so they are not silently erased from system history.

### `project_blockers`

```text
id               TEXT PRIMARY KEY
project_id       TEXT NOT NULL REFERENCES projects(id)
blocker          TEXT NOT NULL
status           TEXT NOT NULL DEFAULT 'open'
created_at       TEXT NOT NULL
resolved_at      TEXT
resolution       TEXT
```

Allowed status values:

```text
open
resolved
```

`get_project_state` returns only open blockers by default.

### `project_paths`

```text
id               TEXT PRIMARY KEY
project_id       TEXT NOT NULL REFERENCES projects(id)
role             TEXT NOT NULL
path             TEXT NOT NULL
description      TEXT
created_at       TEXT NOT NULL
updated_at       TEXT NOT NULL
```

Important paths are anchors, not a filesystem inventory.

ZOMAH should prevent accidental duplicate entries for the same project, role, and path.

### `project_decisions`

```text
id               TEXT PRIMARY KEY
project_id       TEXT NOT NULL REFERENCES projects(id)
statement        TEXT NOT NULL
rationale        TEXT
decision_date    TEXT NOT NULL
status           TEXT NOT NULL
superseded_by    TEXT REFERENCES project_decisions(id)
created_by       TEXT NOT NULL
created_at       TEXT NOT NULL
updated_at       TEXT NOT NULL
```

Allowed decision states:

```text
proposed
accepted
superseded
invalidated
cancelled
```

Decision rows are never rewritten to erase their historical meaning.

When an accepted decision is replaced, the old decision becomes `superseded` and `superseded_by` points to the replacement.

## 4. ProjectState read model

`get_project_state(project)` assembles the canonical read model from the normalized tables.

Conceptually:

```text
ProjectState
├─ id
├─ name
├─ status
├─ phase
├─ summary
├─ current_focus
├─ last_action
├─ next_action
├─ open_questions[]
├─ decisions[]
├─ important_paths[]
├─ blockers[]
├─ updated_at
├─ updated_by
└─ revision
```

`revision` is harness metadata used for safe updates.

The worker does not need to know how the underlying tables are joined.

## 5. Update semantics

Elyria does not submit a replacement `ProjectState` document.

She submits a typed patch.

Conceptually:

```text
ProjectStatePatch
├─ expected_revision?
├─ scalars?
│  ├─ name?
│  ├─ status?
│  ├─ phase?
│  ├─ summary?
│  ├─ current_focus?
│  ├─ last_action?
│  └─ next_action?
├─ questions?
│  ├─ add[]
│  └─ resolve[]
├─ blockers?
│  ├─ add[]
│  └─ resolve[]
├─ paths?
│  ├─ upsert[]
│  └─ remove[]
└─ decisions?
   ├─ add[]
   └─ transition[]
```

### Patch rules

- Omitted fields mean **no change**.
- A worker cannot accidentally erase an omitted field.
- Clearing a nullable scalar must be explicit.
- Collection updates are operations, not replacement arrays.
- Decision lifecycle transitions are validated by ZOMAH.
- `updated_at`, `updated_by`, and the new `revision` are set by ZOMAH, not trusted from model input.
- The complete patch is validated before mutation begins.
- The complete patch is committed in one transaction.

### Optimistic concurrency

If `expected_revision` is supplied and does not match the project's current revision, ZOMAH rejects the patch as stale.

A successful state-changing transaction increments `revision` exactly once.

This prevents two workers or sessions from silently overwriting each other's view of current state.

## 6. Mutation provenance

Every successful update returns a structured result similar to:

```text
project_id
previous_revision
new_revision
updated_at
updated_by
changed_fields
operation_id
```

Detailed historical reconstruction belongs to ZOMAH tracing rather than the `ProjectState` tables themselves.

The state store records enough local lifecycle information to preserve decisions and resolved questions/blockers, but it is not intended to become a general event store.

## 7. Markdown mirror

Markdown export is one-way:

```text
SQLite -> Markdown
```

Not:

```text
Markdown -> SQLite
```

ZOMAH v0 does not ingest edits from generated Markdown back into state.

This avoids split-brain state and ambiguous conflict resolution.

### Default export path

A configurable default may be:

```text
exports/project-state/<project-id>.md
```

The user may later configure an Obsidian/Vault destination without changing the persistence contract.

### Export contents

The generated document should include:

- project name and ID
- lifecycle status
- phase
- summary
- current focus
- last action
- next action
- open questions
- open blockers
- important paths
- decisions with lifecycle status
- last update metadata
- state revision

The document should clearly identify itself as generated.

Example header:

```text
> Generated by ZOMAH from canonical SQLite state.
> Manual edits are not imported.
```

### Export failure behavior

SQLite remains authoritative.

If a database transaction succeeds but Markdown export fails:

1. the database commit remains valid;
2. the state operation returns success for the canonical update;
3. the export failure is reported separately and traced;
4. ZOMAH may regenerate the mirror later.

The Markdown mirror must never be able to roll back canonical state.

## 8. Deletion policy

ZOMAH v0 does not require hard deletion of project state records.

Projects transition to `archived` rather than being deleted.

Decisions are lifecycle-transitioned rather than erased.

Questions and blockers transition to `resolved` rather than disappearing from storage.

Path entries may be removed from current state when they cease to be valid; detailed history of that operation belongs to tracing.

## 9. Explicit non-goals

The ProjectState persistence layer is not:

- a general knowledge database
- a conversation-memory store
- a document index
- a task manager
- a Git replacement
- an event-sourcing framework
- a vector database

Those concerns remain separate unless demonstrated needs justify integration later.

## 10. v0 implementation boundary

The first implementation slice only needs to prove:

1. create a project;
2. read a complete `ProjectState`;
3. apply a validated patch transactionally;
4. reject a stale revision;
5. preserve decision lifecycle history;
6. preserve resolved questions and blockers;
7. generate a Markdown mirror from canonical state.

Once these work reliably, ZOMAH has a real durable state substrate and can move on to model-visible read tools.

# ADR-002: SQLite Is the Canonical ProjectState Store

- **Status:** Accepted
- **Date:** 2026-09-19
- **Decision ID:** ADR-002

## Context

ZOMAH needs durable project state that survives model sessions and does not depend on conversation history.

The state must be structured enough for validation and safe partial updates, simple enough to inspect and operate locally, and independent of any one model or inference runtime.

Human-readable project state is also valuable for inspection, archiving, and future knowledge-vault integration.

Using both a database and editable Markdown as authoritative stores would create conflict and synchronization ambiguity.

## Decision

ZOMAH v0 will use **SQLite as the sole canonical ProjectState store**.

A human-readable Markdown representation may be generated from SQLite, but the mirror is read-only from the persistence system's point of view.

The direction of authority is:

```text
SQLite -> Markdown
```

Never:

```text
Markdown -> SQLite
```

in v0.

Project state updates will be patch-based and transactional rather than full-document replacement.

Each project has an internal monotonically increasing `revision` used for optimistic concurrency.

## Consequences

### Positive

- one authoritative source of current state;
- transactional updates;
- schema constraints and referential integrity;
- easy local deployment with no database service;
- straightforward backups and inspection;
- safe partial updates from model-generated tool calls;
- future workers can share state without silent last-write-wins overwrites;
- Markdown remains available for humans and knowledge tools without becoming a second authority.

### Negative

- Markdown edits do not update canonical state in v0;
- schema changes require explicit migrations;
- normalized child tables add slightly more implementation than storing one JSON blob.

These costs are accepted because state integrity is more important than minimizing the schema to a single row/document.

## Alternatives considered

### Markdown as canonical state

Rejected.

Markdown is excellent for human inspection but weak as the sole authority for concurrent structured mutation, schema validation, and lifecycle relationships.

### JSON/YAML files as canonical state

Rejected for v0.

They would be simple initially but require ZOMAH to build its own locking, transactional update behavior, indexing, and relationship handling as the system grows.

### SQLite with JSON arrays in one project row

Rejected.

It reduces table count but makes targeted updates, lifecycle preservation, constraints, and future querying less reliable.

### Full event sourcing

Rejected for v0.

ZOMAH needs current truth plus useful provenance, not a complete event-sourced architecture. Detailed operation history belongs to the tracing layer.

## Operational rules

- foreign keys enabled;
- WAL journal mode;
- state changes are transactional;
- omitted patch fields mean no change;
- collection mutations use explicit add/resolve/upsert/remove operations;
- successful state mutations increment project revision once;
- stale expected revisions are rejected;
- generated Markdown can fail independently without invalidating a committed SQLite update.

## Governing principle

The persistence layer should be boring, local, explicit, and difficult for a model to corrupt accidentally.

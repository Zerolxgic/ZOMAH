# ADR-007: Knowledge Index Freshness Is a Harness Responsibility

- **Status:** Accepted
- **Date:** 2026-09-20
- **Decision ID:** ADR-007

## Context

`search_knowledge` uses a rebuildable SQLite FTS5 index over approved text roots. The canonical source remains the files on disk.

Without automatic synchronization, successful `write_file` or `move_file` operations—and changes made outside ZOMAH—can leave `knowledge.db` stale until some separate refresh operation occurs. Exposing an indexing tool to the worker would add capability surface and require the model to reason about harness maintenance.

## Decision

ZOMAH will refresh the derived knowledge index internally immediately before every `search_knowledge` query.

The refresh is incremental. Each approved candidate file is fingerprinted using modification time and size; unchanged documents are not re-read or re-indexed. New, changed, moved, and deleted documents are reconciled before the search executes.

No model-facing `refresh_index`, `index_document`, or equivalent tool will be added.

## Consequences

### Positive

- Search observes current filesystem knowledge without requiring worker bookkeeping.
- Files changed outside ZOMAH are also discovered.
- `write_file` and `move_file` stay independent from the retrieval implementation.
- The model-facing capability surface does not grow.
- `knowledge.db` remains disposable derived state.

### Tradeoff

Each search walks the configured knowledge roots to compare file fingerprints. This is intentionally accepted for v0 because it is simple and correct. If real vault size makes that scan materially expensive, ZOMAH may later add filesystem notifications, dirty-path tracking, or another internal optimization without changing the model-facing contract.

## Governing principle

Index maintenance is harness mechanics. The worker asks for knowledge; ZOMAH is responsible for making the derived index current enough to answer.

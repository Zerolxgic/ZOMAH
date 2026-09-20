# ADR-006: Model-Facing Outputs Use Local-Model Budgets

- **Status:** Accepted
- **Date:** 2026-09-20
- **Decision ID:** ADR-006

## Context

ZOMAH's capability implementations were initially bounded for host safety, but several limits were still machine-sized rather than model-sized. A single successful tool call could return enough text or structured entries to consume a large fraction of Elyria's practical Qwen3.5-9B context window.

The model should be able to request more information deliberately through existing pagination, filtering, and follow-up calls instead of receiving the largest response the host can safely produce.

## Decision

Model-facing capability outputs will use conservative budgets appropriate for the local worker.

Initial v0 budgets are:

```text
read_file
  default lines: 100
  maximum lines: 250
  maximum returned text: 16 KiB characters

list_directory
  default entries: 100
  maximum entries: 250

inspect_system
  default entries: 50
  maximum entries: 100
  long process/service text fields are individually bounded

run_script
  stdout: 16 KiB
  stderr: 8 KiB

project state
  current project fields remain present
  at most 20 decision records are returned to the model
  total decision counts by lifecycle status remain visible
  canonical SQLite history is never truncated
```

For project decisions, currently active lifecycle states (`proposed` and `accepted`) are prioritized before terminal historical states when constructing the model-facing window.

## Consequences

### Positive

- One tool call is less likely to crowd out reasoning context.
- Elyria is encouraged to retrieve information incrementally.
- Canonical state and full source files remain unchanged.
- Existing tool names and responsibilities do not change.
- Truncation remains explicit through existing metadata or decision-window metadata.

### Negative

- Some tasks require an additional paginated or filtered call.
- A project with more than 20 active decisions cannot expose every full decision record in one model response.

These are accepted tradeoffs for a local 9B worker with a practical 16K context target.

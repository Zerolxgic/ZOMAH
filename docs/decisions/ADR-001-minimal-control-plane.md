# ADR-001: ZOMAH Will Be a Minimal Control Plane

- **Status:** Accepted
- **Date:** 2026-09-19
- **Decision ID:** ADR-001

## Context

ZOMAH is being created primarily to support Elyria, a local AI worker whose main responsibilities are machine stewardship, project continuity, state tracking, knowledge retrieval, filesystem organization, and narrowly controlled automation.

Qwen3.5-9B and the surrounding inference runtime already provide reasoning and tool-call behavior. Other workers, such as Codex or OpenClaw, may also provide substantial native capabilities.

A conventional agent framework could add planners, routers, memory frameworks, shell environments, delegation systems, reflection loops, and other abstractions. Doing so would duplicate capabilities already available elsewhere and would enlarge the worker's decision surface.

The project therefore needs a clear boundary for what belongs inside ZOMAH.

## Decision

ZOMAH will be implemented as a **thin agent control plane**, not as a general-purpose agent framework.

The worker remains responsible for reasoning and choosing among its available capabilities.

ZOMAH will initially provide only shared substrate that capable workers do not reliably provide on their own:

- durable state
- contextual retrieval
- controlled machine access
- permissions
- validation
- tracing
- verification

ZOMAH will not add a capability merely because it is conventional in agent frameworks.

New capabilities must be justified by an observed requirement or failure.

## Initial model-visible capabilities

The v0 surface is limited to:

### Read

- `search_knowledge`
- `read_file`
- `list_directory`
- `inspect_system`
- `get_project_state`

### Change

- `write_file`
- `move_file`
- `update_project_state`

### Execute

- `run_script`

`run_script` is limited to registered or approved scripts.

Arbitrary shell execution is excluded from v0.

## Consequences

### Positive

- The complete execution model remains understandable.
- The local model has fewer overlapping tools to choose among.
- Failures are easier to attribute to the model, context, tool, or harness.
- Existing model/runtime capabilities remain reusable rather than duplicated.
- ZOMAH can support multiple workers without becoming coupled to any one of them.
- Security boundaries remain easier to reason about.

### Negative

- Some capabilities common in larger frameworks will require explicit future additions.
- Early versions may expose missing primitives during real-world use.
- ZOMAH will rely on external runtimes and workers for capabilities intentionally kept outside the harness.

These are accepted tradeoffs.

The project prefers discovering a missing primitive through use over preemptively building abstractions that may never be needed.

## Alternatives considered

### Build a complete agent framework

Rejected for v0.

This would duplicate model/runtime capabilities and increase orchestration complexity before there is evidence those abstractions are necessary.

### Use OpenClaw as the core runtime

Rejected as the architectural center.

OpenClaw may remain an adapter or worker-facing integration, but ZOMAH's durable state and control boundary should not depend on OpenClaw's continued presence.

### Give Elyria unrestricted shell access

Rejected for v0.

Elyria may create Bash scripts when useful, but execution authority is separate. Initial execution is limited to registered or approved scripts.

### Expose generic MCP access directly

Rejected for v0.

Specific capabilities may later be backed by MCP internally, but the model-visible surface should remain narrow and intentional.

## Governing principle

> If an existing component already does something well, ZOMAH should not reimplement it.

A proposed addition that makes ZOMAH less obviously minimal must demonstrate why the current primitives are insufficient.

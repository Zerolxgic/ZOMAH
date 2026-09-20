# ADR-008: Model-Boundary Capability Tracing Is Mandatory

- **Status:** Accepted
- **Date:** 2026-09-20
- **Decision ID:** ADR-008

## Context

ZOMAH's v0 capability audit requires every model-driven capability action to be reconstructable before Elyria is allowed to drive CHANGE or EXECUTE operations.

Tracing must not become a second memory system, prompt archive, or large telemetry dependency.

## Decision

Every invocation through ZOMAH's model-facing capability boundary must create a local SQLite trace row before the capability executes.

If the trace cannot be started, capability execution is blocked.

The initial trace stores only:

```text
run_id
worker
capability
started_at
finished_at
outcome
error_code
target
```

ZOMAH deliberately does not store prompts, raw model payloads, capability results, file contents, stdout/stderr, or Python exception details in the trace database.

Targets are compact references only. Knowledge-search queries and script arguments are not retained.

If final trace completion fails after a capability has already executed, the durable `started` row remains as an incomplete trace. ZOMAH does not report an already-completed mutation as failed merely because final trace bookkeeping failed, because doing so could invite an unsafe retry.

## Consequences

- model-driven capabilities cannot execute completely untraced;
- successful, rejected, and failed calls share one small audit trail;
- tracing remains independent from canonical ProjectState and derived knowledge state;
- sensitive model content is not duplicated into trace storage;
- incomplete trace rows are visible evidence of finalization failure rather than being silently deleted;
- no external observability or telemetry platform is required for v0.

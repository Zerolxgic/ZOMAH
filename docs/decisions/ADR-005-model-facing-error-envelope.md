# ADR-005: Normalize Errors at the Model Boundary

- **Status:** Accepted
- **Date:** 2026-09-20
- **Decision ID:** ADR-005

## Context

ZOMAH capabilities deliberately use ordinary Python exceptions internally. This keeps direct tests and local development explicit, but a local 9B model should not have to infer recovery behavior from heterogeneous exception text.

The v0 capability audit therefore requires a small, stable error contract before Qwen integration.

## Decision

ZOMAH will preserve native internal exceptions and normalize them only at the model-facing runtime boundary.

Every model-facing invocation returns exactly one of:

```text
ok: true
result: {...}
```

or:

```text
ok: false
error:
  code: stable_machine_code
  message: concise_human_readable_message
  retryable: true | false
  details: optional_small_structured_context
```

Request validation is performed at the same boundary. Validation details identify fields and error types but do not echo rejected input values.

Unexpected exceptions are redacted as `internal_error`; raw exception details belong in developer observability/tracing, not in model context.

## Consequences

- Capability implementations remain simple and directly testable.
- Elyria receives consistent failure semantics across all nine tools.
- Recovery behavior can be taught using stable error codes instead of Python exception classes.
- Future tracing can retain richer developer diagnostics without exposing them to the model.
- This adapter is not a tool registry and does not choose capabilities; it only validates one already-selected invocation and normalizes its result.

# ADR-004: Registered Script Approval Binds to Exact Bytes

- **Status:** Accepted
- **Date:** 2026-09-20
- **Decision ID:** ADR-004

## Context

ZOMAH exposes `run_script` as its only v0 EXECUTE capability. The model may select only a pre-registered script ID and validated arguments; it cannot provide an executable path or arbitrary shell command.

Path-only registration is still insufficient. If an approved script can be modified after registration, the same approved name and path could execute different code. This is especially dangerous if a registered executable is placed inside a filesystem root the model can mutate with `write_file`.

## Decision

Approval for a registered script binds to the **exact executable bytes** that were present when the harness owner registered it.

ZOMAH will:

- require `ScriptRegistry` to know the configured model `WriteScope`;
- reject registration of any executable located inside a write root;
- record the script's SHA-256 digest at approval time;
- before every execution, verify that the registered path still resolves to the same regular file path, is not a symlink, remains executable, and still matches the approved SHA-256 digest;
- reject execution if any integrity check fails;
- require explicit re-registration after an intentional script change.

The working directory may still be writable when a registered automation is intentionally designed to operate there. The protected object is the approved executable itself.

## Consequences

### Positive

- A worker cannot rewrite an approved script with `write_file` and then execute the modified code under the old approval.
- Human or external edits to an approved executable are detected before execution.
- Script approval remains explicit and auditable.
- The model-facing `run_script` contract does not become larger or more complex.

### Negative

- Legitimate edits invalidate an existing registration and require re-approval.
- Harness configuration must provide a `WriteScope` when constructing the script registry.

These costs are accepted because EXECUTE authority should be narrower than filesystem CHANGE authority.

## Rejected alternatives

### Trust the registered pathname

Rejected. A pathname can continue to exist while its contents change.

### Allow registered scripts inside model-writable roots

Rejected. That would let CHANGE authority rewrite EXECUTE authority.

### Give the model a way to approve a new digest

Rejected. Approval is a harness-owner authorization action, not a model capability.

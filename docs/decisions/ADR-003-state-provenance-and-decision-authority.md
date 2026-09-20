# ADR-003: State Provenance and Decision Authority Are Harness-Owned

- **Status:** Accepted
- **Date:** 2026-09-20
- **Decision ID:** ADR-003

## Context

`update_project_state` is intended to be model-facing. Its first implementation allowed the patch payload to supply `updated_by`, complete decision records, and decision lifecycle transitions.

That collapsed important boundaries:

- a worker could claim that an update came from another actor;
- a worker could choose its own decision timestamp;
- a model-originated decision could be created directly as `accepted`;
- proposal and authorization could occur through the same model-facing operation.

ZOMAH's project-state history is only useful if provenance and lifecycle states are trustworthy.

## Decision

Actor identity and authorization metadata are owned by ZOMAH, not model input.

For model-facing project-state updates:

- `updated_by` is removed from `ProjectStatePatch`;
- the harness injects the actor identity when applying the patch;
- model-originated decision additions use `DecisionProposal`;
- proposals contain only an ID, statement, and rationale;
- ZOMAH stamps proposal creation time;
- ZOMAH stores model-originated decisions with status `proposed`;
- decision lifecycle transitions are removed from the model-facing patch.

Decision transitions remain available only through a separate internal/admin repository path that requires an explicit actor and expected project revision.

Trusted bootstrap or administrative code may still construct complete canonical `Decision` records when establishing existing accepted state.

## Consequences

### Positive

- workers cannot impersonate another actor through tool arguments;
- timestamps represent harness observation rather than model claims;
- proposal remains distinct from authorization;
- decision history retains meaningful lifecycle semantics;
- optimistic revision checks apply to authorized decision transitions as well as ordinary state mutation.

### Negative

- future UI or owner workflows will need a separate authorization path to accept or otherwise transition proposed decisions;
- existing direct repository callers must provide actor identity outside the patch payload.

These costs are intentional. ZOMAH should preserve authority boundaries rather than optimize for a single convenient mutation object.

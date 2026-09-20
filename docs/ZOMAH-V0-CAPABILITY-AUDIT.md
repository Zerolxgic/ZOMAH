# ZOMAH v0 Capability Audit

**Status:** Audit complete — remediation required before model integration  
**Date:** 2026-09-20  
**Test baseline:** 83 passing tests  
**Audit target:** Nine model-visible v0 capabilities

## Executive finding

The v0 model-visible capability surface is sufficiently small and functionally coherent for Elyria's intended steward role. No additional model-facing capability is justified before Qwen integration.

However, the audit identified six internal gaps that should be resolved before a model is allowed to drive the tools. These are not requests for more agent features. They are boundary-hardening and observability work around the existing surface.

## Audited capability surface

### READ

- `get_project_state`
- `list_directory`
- `read_file`
- `search_knowledge`
- `inspect_system`

### CHANGE

- `update_project_state`
- `write_file`
- `move_file`

### EXECUTE

- `run_script`

## Coverage

**Result: PASS for the intended single-project steward loop.**

The existing tools let Elyria:

- recover canonical project state;
- navigate approved filesystem locations;
- locate knowledge without knowing exact paths;
- read bounded source content;
- inspect relevant live machine state;
- update project state;
- create, replace, append, and organize approved files;
- execute explicitly registered automation.

No new model-visible capability is required before model integration.

### Watch item: project discovery

`get_project_state` requires a known project ID. This is sufficient if the harness supplies the current project identity in context.

If real use later demonstrates that Elyria must independently answer questions such as "which projects are active?", project discovery may become justified. It is not added during this audit.

## Authority

**Result: STRONG overall, with two required fixes.**

The current design correctly separates `ReadScope` from `WriteScope`, blocks arbitrary shell execution, excludes deletion, rejects path escape, and restricts execution to registered scripts.

### Required fix A — Project-state provenance and decision authority

The current model-facing `ProjectStatePatch` contains `updated_by`, so a model could claim an arbitrary actor identity.

New `Decision` objects also expose `created_at` and currently default to `accepted`, allowing a model-driven update to create an already-accepted decision without an independently enforced authorization boundary.

Before model integration:

- actor identity must be injected by the harness, not supplied by the model;
- decision timestamps must be harness-owned;
- model-originated decisions must not silently become accepted decisions;
- acceptance/transition authority must preserve the distinction between proposal and authorization.

### Required fix B — Registered-script integrity

`run_script` resolves an approved executable at registration time, but the executable can later be modified on disk.

If a registered script were located inside a model-writable root, a worker could modify that approved executable and then invoke it, effectively expanding execution authority.

Before model integration:

- registered scripts must be outside model `WriteScope`; and
- ZOMAH should verify executable integrity before each run, preferably using a pinned content digest captured when the script is registered.

## Overlap and tool-choice clarity

**Result: PASS.**

The nine tools have distinct responsibilities:

- project state is separate from filesystem state;
- discovery is separate from content reading;
- observation is separate from mutation;
- mutation is separate from execution;
- knowledge retrieval returns references rather than duplicating `read_file`;
- `run_script` is registered automation, not a second shell/filesystem interface.

No tool should be merged or removed based on the current evidence.

The request schemas are also reasonably small. The nine current Pydantic request schemas serialize to roughly 6.9 KB total. `update_project_state` is by far the largest schema, while the remaining eight are individually small.

## Failure behavior

**Result: SAFE at the capability layer, but incomplete for a model-facing runtime.**

Individual capabilities generally fail conservatively: scope violations reject, revision conflicts reject, moves do not overwrite, registered scripts time out, and no capability silently escalates into a broader operation.

### Required fix C — Structured error envelope

Capabilities currently expose heterogeneous Python exceptions.

Before model integration, the adapter must translate failures into a small stable model-facing error structure, for example:

```text
code
message
retryable
details?
```

Important errors should be distinguishable without requiring a 9B model to interpret Python exception text.

Examples include:

- path outside scope;
- file not found;
- destination exists;
- revision conflict;
- unsupported text;
- unknown script;
- invalid script argument;
- timeout;
- unavailable system inspection.

## Context cost

**Result: REQUEST SCHEMAS PASS; OUTPUT BUDGETS NEED TIGHTENING.**

Tool definitions themselves are compact enough for Qwen3.5-9B.

Several allowed outputs are too large for a local model with a practical context around 16K tokens:

- `read_file` can return up to 64 KiB;
- `run_script` can return up to 64 KiB for stdout and another 64 KiB for stderr;
- `list_directory` allows up to 1000 entries;
- `inspect_system` allows up to 500 process/service/mount entries;
- `get_project_state` can grow without a bound as decision history accumulates.

### Required fix D — Model-facing output budgets

Before Qwen integration, establish conservative output ceilings designed for the local model rather than the host machine.

The first pass should reduce maximum result sizes while preserving pagination/filtering so the model can request more deliberately.

`ProjectState` also needs to preserve historical decisions in storage without allowing unlimited historical material to inflate the default current-state response.

## Retrieval correctness

**Result: FUNCTIONAL, with one lifecycle gap.**

SQLite FTS5 lexical retrieval is working well and is sufficient for v0. Embeddings are not justified.

### Required fix E — Knowledge-index freshness

`knowledge.db` is derived state, but current file mutations do not automatically refresh it.

A file created, replaced, appended, or moved through ZOMAH can therefore leave search results stale until an explicit internal refresh occurs.

Before model integration, index maintenance must become an automatic harness responsibility. This should remain invisible to the model.

A minimal first implementation may refresh incrementally before `search_knowledge` and be optimized later only if real vault size makes that too expensive.

## Observability

**Result: BLOCKED.**

The project charter declares tracing mandatory, but no trace/run implementation exists yet.

### Required fix F — Z5 trace before Qwen

Before a model drives CHANGE or EXECUTE capabilities, ZOMAH must record enough information to reconstruct meaningful actions.

The initial trace does not need a telemetry platform. A small local SQLite trace is sufficient if it records at least:

```text
run_id
worker
capability
started_at
finished_at
outcome
error_code
state-changing target/reference
```

Inputs and outputs should be recorded carefully so tracing does not become a second uncontrolled store of secrets or huge content.

## Non-blocking watch items

These remain intentionally deferred until real use demonstrates a need:

- project discovery/listing;
- user-level systemd service inspection;
- additional knowledge file formats;
- network inspection;
- binary file reads/writes;
- file deletion;
- cross-filesystem moves;
- arbitrary shell execution;
- embeddings/vector retrieval;
- multi-agent delegation.

## Audit decision

**Do not add another model-visible capability.**

The v0 surface is sufficient to begin model testing after the six internal remediation items are complete:

1. harness-owned provenance and decision authorization;
2. registered-script integrity;
3. structured model-facing errors;
4. local-model output budgets;
5. automatic knowledge-index freshness;
6. Z5 tracing.

After those are verified, connect Qwen3.5-9B through LM Studio and evaluate real tool selection and recovery behavior.

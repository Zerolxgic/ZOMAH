# ZOMAH

**Zerrius's Obviously Minimal Agent Harness**

## Status

**Phase:** Inception / Architecture  
**Project state:** Active  
**Implementation state:** Not started

## Purpose

ZOMAH is a minimal local agent control plane designed primarily for Elyria and other local AI workers.

Its purpose is not to replace the capabilities already provided by models, inference runtimes, agent systems, or external tools.

ZOMAH exists to provide the small set of capabilities those systems do not reliably provide on their own:

- durable state
- contextual retrieval
- controlled machine access
- permissions
- validation
- tracing
- verification

The guiding principle is simple:

> If an existing component already does something well, ZOMAH should not reimplement it.

## Primary Use Case

The first ZOMAH worker is **Elyria**, running primarily on Qwen3.5-9B through a local inference runtime.

Elyria's primary responsibilities are expected to include:

- tracking system state
- tracking project state
- retrieving project and knowledge-vault information
- reading and writing files
- organizing and moving files
- inspecting the local machine
- maintaining continuity across work sessions
- executing narrowly approved automation
- generating small Bash scripts when useful

Elyria is not intended to function primarily as a coding agent.

Specialized coding systems such as Codex remain separate workers and retain their own native capabilities.

## Architectural Principle

ZOMAH is a **control plane, not an agent framework**.

The model remains responsible for reasoning.

The worker remains responsible for deciding which available capability to use.

ZOMAH provides the controlled interface between the worker and the environment.

```text
                Worker / Model
                     │
                 tool calls
                     │
             ┌───────▼───────┐
             │     ZOMAH     │
             │               │
             │ State         │
             │ Retrieval     │
             │ Permissions   │
             │ Validation    │
             │ Tracing       │
             │ Verification  │
             └───────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Files       System     Knowledge
```

## Design Rules

### 1. Minimal by evidence

A capability is added only when an observed requirement or failure demonstrates that it is needed.

Convenience alone is not sufficient justification.

### 2. Do not duplicate worker capabilities

If Qwen, LM Studio, Codex, OpenClaw, MCP, or another worker/runtime already provides a capability appropriately, ZOMAH should integrate with it rather than reproduce it.

### 3. State belongs to ZOMAH

Workers may read and request updates to state.

ZOMAH owns persistence, schema validation, provenance, and history.

### 4. Current state and history are different

Current project state contains the compressed present truth.

Historical events, conversations, traces, and previous decisions remain separate.

### 5. Authority is explicit

Reading information, modifying information, and executing actions are separate permission classes.

No worker receives unrestricted machine authority by default.

### 6. Tracing is mandatory

Every meaningful action must be reconstructable.

ZOMAH should always be able to answer:

> What happened, who initiated it, what information was used, and what changed?

## ZOMAH v0 Model-Visible Tools

### Read

1. `search_knowledge`
2. `read_file`
3. `list_directory`
4. `inspect_system`
5. `get_project_state`

### Change

6. `write_file`
7. `move_file`
8. `update_project_state`

### Execute

9. `run_script`

`run_script` executes registered or approved scripts.

Arbitrary shell execution is explicitly outside the initial scope.

## Permission Classes

### READ

Inspection and retrieval operations.

Generally allowed automatically inside configured scopes.

### CHANGE

Filesystem and state mutation.

Restricted to configured roots and validated operations.

### EXECUTE

Execution of machine actions.

Initially restricted to explicitly registered scripts.

## ZOMAH Internal Responsibilities

These are harness responsibilities and should not appear as tools the model must consciously invoke:

- permission checks
- schema validation
- tracing
- run-state persistence
- knowledge indexing
- operation provenance
- system snapshot handling
- tool-result normalization

## Core Domain Objects

ZOMAH v0 begins with six conceptual primitives:

- `Task`
- `Context`
- `Worker`
- `RunState`
- `Trace`
- `Verification`

The first domain-specific state object is:

- `ProjectState`

## ProjectState v0

A ProjectState answers:

> Where does this project stand right now?

Initial fields:

```text
id
name
status
phase
summary
current_focus
last_action
next_action
open_questions[]
decisions[]
important_paths[]
blockers[]
updated_at
updated_by
```

ProjectState stores conclusions, not activity history.

Historical information belongs in traces, events, documentation, source control, or other archival systems.

## Decision Lifecycle

Project decisions must preserve history.

Initial decision states:

```text
proposed
accepted
superseded
invalidated
cancelled
```

A superseded decision remains available and points toward the decision that replaced it.

ZOMAH should never silently rewrite project history.

## Explicit Non-Goals for v0

ZOMAH will not initially provide:

- arbitrary shell access
- file deletion
- `sudo`
- package management
- Git automation
- browser automation
- a coding environment
- autonomous multi-agent delegation
- email integration
- calendar integration
- unrestricted MCP access
- dynamic tool creation
- a planner
- a reflection agent
- a generic memory framework
- a large vector-database architecture

Any of these may be reconsidered if actual usage demonstrates the need.

## Security Position

Elyria should have enough authority to maintain the system without receiving unnecessary authority over the system.

The initial security boundary is therefore:

```text
READ     → broad within configured roots
CHANGE   → scoped and validated
EXECUTE  → explicitly approved
```

Destructive operations require stronger justification than reversible operations.

## Initial Success Criteria

ZOMAH v0 is viable when Elyria can reliably:

1. determine the current state of an active project
2. locate relevant documentation in the knowledge system
3. inspect relevant machine state
4. read and update files within approved locations
5. move and organize files within approved locations
6. update project state without corrupting existing state
7. execute an approved automation script
8. leave a useful trace of every meaningful operation
9. resume work later without depending on conversation history alone

## Development Philosophy

ZOMAH should remain understandable enough that its complete execution model can be reasoned about without relying on hidden framework behavior.

Complexity must earn its place.

When choosing between:

```text
more abstraction
```

and:

```text
a small explicit implementation
```

ZOMAH should prefer the explicit implementation until evidence says otherwise.

## Initial Milestones

### Z0 — Foundation

Define:

- project charter
- terminology
- architectural boundaries
- decision process
- repository structure

### Z1 — State

Implement and validate:

- `ProjectState`
- persistence
- updates
- provenance

### Z2 — Read

Implement:

- `read_file`
- `list_directory`
- `search_knowledge`
- `inspect_system`
- `get_project_state`

### Z3 — Change

Implement:

- `write_file`
- `move_file`
- `update_project_state`

with scoped permissions and validation.

### Z4 — Execution

Implement approved-script registration and:

- `run_script`

### Z5 — Trace

Establish:

- run identity
- action records
- tool inputs/results
- errors
- state changes
- verification results

### Z6 — Elyria Integration

Connect Qwen3.5-9B through the selected local inference runtime and validate complete tool loops against real workstation tasks.

## Current Decision

ZOMAH will begin as the smallest viable local control plane for Elyria.

We will expand it only in response to demonstrated needs rather than anticipated ones.
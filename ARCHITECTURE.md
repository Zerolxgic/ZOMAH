# ZOMAH Architecture

## 1. Purpose

ZOMAH is a thin local control plane for capable AI workers.

Its first worker is Elyria, primarily backed by Qwen3.5-9B through a local inference runtime. ZOMAH does not own the model's reasoning or reproduce capabilities already provided by the model or runtime.

ZOMAH owns the boundary between the worker and durable machine state.

## 2. Architectural boundary

```text
                    Worker / Model
                         |
                    tool request
                         |
                +--------v--------+
                |      ZOMAH      |
                |-----------------|
                | permission gate |
                | input validation|
                | state access    |
                | retrieval       |
                | trace capture   |
                | result shaping  |
                +--------+--------+
                         |
            +------------+------------+
            |            |            |
          Files        System       Knowledge
            |            |            |
       read/write     inspect      search/read
       move           scripts      project docs
                         |
                    ProjectState
```

### Worker owns

- reasoning
- language
- persona
- deciding whether a tool is needed
- selecting among available model-visible tools
- interpreting returned information

### ZOMAH owns

- durable state
- state schema validation
- permissions
- path and scope validation
- action provenance
- tracing
- knowledge indexing hooks
- normalized tool results
- verification hooks

### External components own

- model inference
- native tool-call generation/parsing where provided by the runtime
- specialized coding-agent capabilities
- external systems such as OpenClaw, MCP servers, or future services

## 3. Design constraints

### Minimal by evidence

No capability is added because agent frameworks commonly include it. A new primitive must solve an observed requirement or failure.

### No capability duplication

ZOMAH integrates with existing capabilities rather than recreating them.

Examples:

- Qwen chooses tools; ZOMAH does not add a separate planner.
- The inference runtime performs inference; ZOMAH does not become a model server.
- Codex retains its own coding environment; ZOMAH does not recreate one.

### Explicit authority

Machine authority is divided into three classes:

```text
READ
CHANGE
EXECUTE
```

A worker receives only the authority required for the operation being attempted.

### Current truth is not history

`ProjectState` is a compressed statement of the project's current truth.

Historical activity belongs in traces, events, documentation, source control, or other archival stores.

### Trace by default

Tracing is infrastructure, not a model-visible action. The worker should never have to remember to trace itself.

## 4. Model-visible tool surface

### READ

#### `search_knowledge(query, scope?, limit?)`

Search indexed project and knowledge material.

Returns structured results containing identifiers or paths, excerpts, relevance information, and useful metadata.

It should not inject entire documents into model context by default.

#### `read_file(path, start?, end?)`

Read a known file or bounded portion of a file.

#### `list_directory(path, depth?)`

Inspect filesystem structure within approved roots.

#### `inspect_system(domain, filter?)`

Return structured system information.

Initial domains may include:

- processes
- services
- storage
- memory
- gpu
- mounts
- network

This is intentionally preferred over unrestricted shell inspection.

#### `get_project_state(project)`

Retrieve the current canonical `ProjectState`.

### CHANGE

#### `write_file(path, content, mode)`

Initial modes:

- `create`
- `replace`
- `append`

Writes are restricted by configured roots and validation policy.

#### `move_file(source, destination)`

Move or rename files inside approved scopes.

Deletion is not part of v0.

#### `update_project_state(project, patch)`

Apply a validated partial update to project state.

The worker does not rewrite the entire state record. Actor identity is injected by the harness and is not part of model input. Decision additions are proposals only; the model-facing patch cannot assign decision status, timestamps, or lifecycle transitions.

### EXECUTE

#### `run_script(script, arguments?)`

Execute a registered or explicitly approved script. Approval binds to the exact registered executable bytes, not only to a pathname. Registered executables must live outside every configured model `WriteScope`; ZOMAH stores a SHA-256 digest when the script is approved and verifies path resolution, executability, and the digest before every run. Any change requires explicit re-registration.

Arbitrary shell execution is outside v0.

## 5. ProjectState v0

`ProjectState` answers:

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

### Lifecycle status

Initial project status values:

```text
active
paused
blocked
completed
archived
```

### Decisions

Decision records preserve lifecycle history.

Initial decision states:

```text
proposed
accepted
superseded
invalidated
cancelled
```

Superseded decisions remain available and may reference the decision that replaced them.

Model-originated decisions enter canonical state as `proposed`. ZOMAH owns their creation timestamps. Acceptance, supersession, invalidation, and cancellation are separate authorized transitions and are intentionally not exposed through `update_project_state`.

### Important paths

Important paths are anchors, not a complete filesystem inventory.

Each entry should identify a role and a path, with an optional description.

## 6. Trace model

Every meaningful operation should be reconstructable.

A minimal trace should eventually answer:

```text
run_id
task
worker
model
operation
inputs
permission_result
result
state_change
error
started_at
finished_at
verification
```

The exact schema is deferred until the tracing slice.

## 7. Security boundary

Initial policy:

```text
READ     -> broad inside configured roots
CHANGE   -> scoped and validated
EXECUTE  -> explicitly approved
```

Not available in v0:

- arbitrary shell
- file deletion
- sudo
- package installation
- unrestricted MCP
- dynamic tool creation

Generating a Bash script is not equivalent to permission to execute it.

Creation and execution remain separate authority levels.

## 8. Integration model

ZOMAH should be able to sit underneath different workers without becoming coupled to one runtime.

Conceptually:

```text
Elyria --------+
Codex ---------+--> ZOMAH --> state / retrieval / controlled machine access
OpenClaw ------+
future worker -+
```

A worker may retain capabilities that ZOMAH does not expose.

ZOMAH only supplies missing shared substrate.

## 9. v0 success condition

The architecture is viable when Elyria can:

1. recover current project state without depending on conversation history;
2. find and read relevant documentation;
3. inspect useful machine state;
4. safely write or organize approved files;
5. update project state without corrupting prior state;
6. execute approved automation;
7. leave enough trace information to reconstruct meaningful actions.

Anything beyond this remains outside the initial architecture until usage demonstrates a need.

## 10. Model-facing output budgets

ZOMAH budgets capability results for the local model rather than for host memory capacity. Large files, listings, system snapshots, script output, and decision history are intentionally bounded. Workers use pagination, filtering, or follow-up reads when more information is required.

Canonical data is not truncated: these limits apply only to the model-facing projection of a capability result. Project decision history remains complete in SQLite while `get_project_state` and `update_project_state` return a bounded decision window plus lifecycle counts.

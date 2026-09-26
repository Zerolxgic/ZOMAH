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
### Capability and interface model

ZOMAH capabilities are implementation units owned by the control plane.

Human-facing commands and model-facing tools are interfaces into those capabilities rather than separate implementations.

```text
                         Human Operator
                              |
                       Operator Console
                              |
                              v
                     +-------------------+
                     | Command Registry  |
                     +---------+---------+
                               |
                               |
Worker / Model                 |
     |                         |
agent adapter                  |
     |                         |
     +-------------+-----------+
                   |
                   v
          +--------------------+
          | Capability Registry|
          +---------+----------+
                    |
             invocation boundary
                    |
                    v
               +---------+
               |  ZOMAH  |
               +----+----+
                    |
        +-----------+-----------+
        |           |           |
      State     Retrieval     Tools
```

The Capability Registry describes what ZOMAH can expose.

The Command Registry describes how the Operator Console exposes user-facing operations.

A slash command may invoke a registered capability, but not every slash command is a capability. Interface-only commands such as help, console status, or session management remain Operator Console concerns.

Likewise, not every registered capability must be visible to every worker or session.

Capability exposure may depend on:

- lifecycle state;
- authority class;
- active project;
- worker identity;
- session policy;
- runtime availability;
- dependency availability.

The number of tools shown as available to a model therefore means:

> capabilities currently exposed to that model in the active project and session

rather than every capability installed in ZOMAH.

The Operator Console and model adapters must not bypass ZOMAH validation, permissions, tracing, provenance, lifecycle rules, or authorization boundaries.

### Operator Console

The Operator Console is ZOMAH's planned primary human interface.

The first implementation is planned as an interactive terminal UI using Textual.

Its accepted layout contains three persistent regions:

```text
+------------------------------------------------------+
| ZOMAH                                                |
| Project / folder                     ZOMAH state     |
| Model                                available tools |
| Context usage / context limit                        |
+------------------------------------------------------+

+------------------------------------------------------+
| Session / transcript                                 |
|                                                      |
| conversation, results, tool activity, status         |
|                                                      |
+------------------------------------------------------+

+------------------------------------------------------+
| Composer                                             |
| multiline text / slash commands / attachments        |
+------------------------------------------------------+
```

The persistent header is a view over session state. It is not itself the source of project, model, tool, or context truth.

The composer should behave like a modern multiline message editor.

Required first-pass behavior includes:

- Enter submits;
- Shift+Enter inserts a newline;
- normal cursor movement and text selection;
- select-all and bulk deletion or replacement;
- copy and paste;
- undo and redo;
- clipboard image attachment;
- slash-command discovery;
- alphabetical prefix filtering;
- arrow-key command navigation;
- keyboard shortcuts as optional accelerators.

Slash commands should be discoverable without requiring the operator to memorize keybindings.

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
- capability registration and exposure policy
- interface-independent capability invocation

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

Search project and knowledge material through a derived SQLite FTS5 index. ZOMAH refreshes that index incrementally immediately before the query, keeping index maintenance invisible to the worker.

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

### Retrieval implementation boundary

`search_knowledge` is the stable worker-facing retrieval capability.

The implementation underneath it may evolve without expanding the model-visible tool surface.

Current retrieval direction:

```text
search_knowledge
       |
       v
retrieval subsystem
       |
       +-- lexical retrieval
       |
       +-- semantic retrieval experiment
               |
               +-- Qwen3-Embedding-0.6B
```

`Qwen3-Embedding-0.6B` is the selected model for the first isolated semantic-retrieval experiment.

The embedding model is not intended to appear as a direct worker-facing tool. A worker asks ZOMAH to search knowledge; ZOMAH determines which retrieval mechanisms participate.

Semantic retrieval must be evaluated before becoming part of the active retrieval path.

A reranker or relevance judge is not assumed to be necessary. It should be introduced only if evaluation demonstrates a retrieval failure that it has a clear job to solve.

Retrieval components may judge semantic similarity or relevance, but they do not replace deterministic evidence, provenance, lifecycle validation, policy enforcement, or machine-state observation.

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

Every model-boundary capability invocation is traced automatically before execution. The v0 local SQLite trace is intentionally small:

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

Tracing stores no prompts, raw request payloads, capability results, file contents, stdout/stderr, or exception details. Targets are compact references only. Knowledge-search query text and script arguments are deliberately excluded.

A capability does not execute if its trace row cannot be started. If final trace completion fails after a capability has already executed, the existing `started` row remains as visible evidence of an incomplete trace rather than turning a completed mutation into a model-facing failure that could invite a retry.

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

### Deterministic tooling integration

ZOMAH may expose separately developed deterministic tools through capability adapters.

Integration should preserve each tool's existing contract, evidence boundary, lifecycle state, and failure behavior.

ZOMAH should prefer wrapping an existing verified tool over rewriting its implementation.

Conceptually:

```text
Capability Registry
        |
        v
    Tool Adapter
        |
        +-- native ZOMAH Python capability
        |
        +-- external deterministic process
```

This allows existing ZOMAH capabilities and standalone deterministic tools to share one invocation architecture without requiring them to share one implementation language or codebase.

Some capabilities may be available to both the human operator and the worker. Others may remain internal infrastructure.

Policy-enforcement capabilities such as action gating may be mandatory parts of the invocation path rather than optional tools a worker chooses whether to call.

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

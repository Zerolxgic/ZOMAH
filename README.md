# ZOMAH

**Zerrius's Orchestrated Minimal Agent Harness**

ZOMAH is a deliberately small local control plane for capable AI workers, starting with Elyria.

## Current implementation

The original ZOMAH v0 substrate is implemented and verified. The current capability surface includes:

- Pydantic v2 contracts
- stdlib `sqlite3`
- normalized SQLite schema
- transactional patch updates
- optimistic revision checks
- one-way Markdown export
- pytest coverage
- model-facing `get_project_state` capability
- model-facing `update_project_state` capability with harness-owned actor provenance
- scoped, bounded `read_file` capability
- scoped, bounded `list_directory` capability
- structured `inspect_system` capability for explicit Linux system domains
- lexical `search_knowledge` capability backed by SQLite FTS5 with automatic incremental freshness
- scoped UTF-8 `write_file` capability with separate write roots
- scoped no-overwrite `move_file` capability for regular files
- registry-only `run_script` execution with validated arguments and bounded results
- local-model-sized output budgets across file reads, listings, system inspection, script output, and ProjectState decision history
- mandatory local SQLite tracing at the model-facing capability boundary

No agent loop, generic tool registry, model client, or framework has been added yet.

### Current direction

The next architectural slice is a shared Capability Registry followed by the ZOMAH Operator Console.

Accepted next-phase work includes:

- a shared user/agent capability registry;
- an interactive terminal Operator Console;
- integration of the verified deterministic tooling workspace;
- an isolated `Qwen3-Embedding-0.6B` semantic-retrieval experiment before retrieval integration;
- later connection of Qwen3.5-9B through the model-facing capability boundary.

Project-state mutation keeps authorization outside model input: workers may propose decisions, but ZOMAH stamps actor/time provenance and stores model-originated decisions as `proposed`. Decision acceptance and lifecycle transitions use a separate internal/admin repository path.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Structure

```text
src/zomah/state/models.py      ProjectState contracts
src/zomah/state/schema.sql     SQLite schema
src/zomah/state/repository.py  persistence + patch behavior
tests/test_project_state.py    Z1 persistence contract tests
tests/test_get_project_state.py read capability tests
tests/test_update_project_state.py change capability tests
src/zomah/access.py             read path scope enforcement
src/zomah/capabilities/read_file.py bounded text read capability
src/zomah/capabilities/list_directory.py bounded directory listing capability
src/zomah/capabilities/inspect_system.py structured Linux system inspection
src/zomah/capabilities/search_knowledge.py lexical knowledge retrieval
src/zomah/capabilities/write_file.py scoped text mutation capability
src/zomah/capabilities/move_file.py scoped file organization capability
src/zomah/execution.py             trusted script registry + argument contracts
src/zomah/capabilities/run_script.py registered execution capability
tests/test_read_file.py         file-read boundary tests
tests/test_list_directory.py    directory-list boundary tests
tests/test_inspect_system.py      system-inspection contract tests
tests/test_search_knowledge.py     retrieval boundary tests
tests/test_write_file.py           file-write boundary tests
tests/test_move_file.py            file-move boundary tests
tests/test_run_script.py             execution-boundary tests
src/zomah/capability_runtime.py      interface-neutral validate/trace/invoke/normalize core
src/zomah/model_boundary.py          model-facing adapter over the capability runtime
src/zomah/user_boundary.py           human operator adapter over the capability runtime
src/zomah/console/operator.py        operator identity + injected capability dependencies
tests/test_user_boundary.py          operator boundary tests
tests/test_console_project.py        /project capability read path tests
tests/test_console_session.py        console session-state ownership tests
src/zomah/attachments.py             in-memory ImageAttachment + limits (shared)
src/zomah/model_runtime.py           provider-neutral model runtime contract
src/zomah/worker_session.py          ephemeral WorkerSession over a ModelRuntime
tests/test_worker_session.py         worker session contract tests (fake runtime)
src/zomah/console/clipboard.py       wl-paste clipboard image source
tests/test_console_clipboard.py      attachment validation + wl-paste adapter tests
tests/test_console_attachments.py    composer attachment behavior tests
tests/test_capability_runtime.py     shared invocation contract tests
src/zomah/tracing.py                 minimal local capability trace store
tests/test_tracing.py                automatic tracing boundary tests
src/zomah/console/app.py             Operator Console shell (Textual)
src/zomah/console/commands.py        console slash-command registry
src/zomah/console/routing.py         submission → message/command routing
src/zomah/console/builtin_commands.py built-in /help /project /status /tools views
src/zomah/console/status.py          ConsoleStatus view data
tests/test_console_commands.py       command registry tests
tests/test_console_routing.py        routing and built-in view tests
tests/test_operator_console.py       console layout and composer tests
```

## First local run

After installing the project:

```bash
python examples/bootstrap_state.py
```

This initializes the canonical SQLite database at `$XDG_DATA_HOME/zomah/zomah.db` or `~/.local/share/zomah/zomah.db`, creates the initial ZOMAH `ProjectState` if it does not already exist, and writes a generated Markdown mirror beside the database under `exports/`.

## Operator Console (T0a shell)

```bash
zomah-console                                  # or: python -m zomah.console
zomah-console --project zomah --operator zerrius
```

The console currently provides the header, transcript, and multiline composer only. Header fields without a live runtime source show explicit placeholders, and submitted text is echoed to the transcript without invoking any model or capability. Enter submits, Shift+Enter inserts a newline (requires a terminal that supports the kitty keyboard protocol; Ctrl+J is the compatibility fallback), Ctrl+A selects all, and Ctrl+Q quits.

Typing `/` lists the registered console commands (`/help`, `/project`, `/status`, `/tools`) alphabetically, filtered by prefix. Up/Down move the highlight, Enter completes the highlighted command into the composer, and Esc dismisses the list. The console command registry (`src/zomah/console/commands.py`) is separate from the capability registry and grants no machine authority.

Any submission starting with `/` is routed as a command (`src/zomah/console/routing.py`); unknown commands and unsupported arguments return an error result and never fall through as ordinary input. The built-in views live in `src/zomah/console/builtin_commands.py`: `/help` lists commands and composer keys, `/status` shows the supplied `ConsoleStatus`, and `/tools` shows `CapabilityRegistry` metadata (registry agent exposure is not live model availability). `/project` reads canonical ProjectState for the configured `--project` id through the human capability boundary; without `--project` it reports that no active canonical project is configured and invokes nothing.

`ConsoleStatus` (`src/zomah/console/status.py`) is the single owner of transient console session state: the active canonical project id, the human-facing project label, ZOMAH state, model, live tool count, and context usage. The header and `/status` both render it, and `/project` takes its project id from it. `--project` sets only the id, so the header shows `zomah (not read yet)` until a successful `/project` read labels it from canonical state (`ZOMAH (zomah)`). Session updates are applied on the UI loop through `OperatorConsole.set_status`; it is not canonical ProjectState.

### Clipboard images

Alt+V attaches the clipboard image (PNG, JPEG, or WebP) to the composer draft. Ctrl+Shift+V triggers the same action on a best-effort basis, only if the terminal passes the key through for an image-only clipboard; on the verified Ghostty 1.3.1 setup it does not, so Alt+V is the dependable key. Text paste is unchanged. Images are read from the Wayland clipboard with `wl-paste` (wl-clipboard), off the UI loop, with fixed arguments, a 5 s timeout, and a 10 MiB per-image limit; a draft holds at most 4 images. Attached images appear in a strip above the composer; Backspace in an empty composer removes the most recent one. Images live only in the draft: submission sends them with the text to the transcript as metadata only, and they are never written to disk. No model receives them yet.

## Worker session (T1a contracts)

`zomah.model_runtime` defines a provider-neutral, async completion contract: `SystemMessage`, `UserMessage` (text plus in-memory `ImageAttachment`s, never pre-encoded), `AssistantMessage`, `ModelRequest`, `ModelResponse`, runtime-reported `TokenUsage`, the `ModelRuntime` protocol, and one operator-safe `ModelRuntimeError`. `zomah.worker_session.WorkerSession` holds ephemeral, in-memory session state over an injected runtime: committed history, worker/model identity, an optional configured context limit, the latest reported usage, and an (empty) tool snapshot. `send()` commits the user turn and reply only on success; a runtime failure commits nothing; a concurrent `send()` raises `SessionBusyError`. Context usage comes only from runtime reports and is `None` when unreported. No model runtime adapter, tools, persistence, or console wiring exists yet.

## Operator capability boundary

`zomah.user_boundary.invoke_registered_user_capability` is the human front door to registered capabilities, separate from the model-facing `invoke_registered_model_capability`. It rejects unknown and non-`user_exposed` capabilities before anything runs (`agent_exposed` is irrelevant to it), then calls the shared `invoke_capability` runtime with the registered request model and handler. Calls are traced as `operator:<operator_id>` in the existing trace `worker` column. The console builds its trace store and ProjectState repository once at startup (`OperatorAccess`) and runs capability-backed commands off the UI event loop.

## Knowledge retrieval

`search_knowledge` uses a derived SQLite FTS5 index over approved UTF-8 text roots. The index is rebuildable and separate from canonical ProjectState storage. ZOMAH refreshes the index incrementally before each search so new, changed, moved, or deleted documents become visible without a model-facing indexing tool.

## Write boundary

`write_file` and `move_file` use a separate `WriteScope`; read authority never implies write authority. `write_file` supports explicit `create`, `replace`, and `append` modes only. `move_file` moves regular files only, never overwrites a destination, never creates directories, and never falls back to cross-filesystem copy/delete behavior.

## Execute boundary

`run_script` can execute only scripts pre-registered by the harness owner. The model selects a script ID and values allowed by that script's explicit argument contract; it never supplies an executable path, shell command, working directory, timeout, or inherited environment. Execution always uses `shell=False`, bounded returned output, and a fixed timeout. Registered executables must live outside every model `WriteScope`, and registration records the script's SHA-256 digest. ZOMAH revalidates the executable path, execute bit, and exact digest before every run; modified scripts require explicit re-registration.

### Model-facing error boundary

Capability implementations keep native Python exceptions for development. Before results reach a model, `zomah.model_boundary.invoke_model_capability` validates the request and returns a stable `ok/result/error` envelope. Unexpected exception details are redacted from model context. The validation, tracing, invocation, and normalization mechanics live in the interface-neutral `zomah.capability_runtime.invoke_capability`; the model boundary is a thin adapter that supplies the worker as the trace identity.

## Tracing

Model-driven capability calls are traced automatically in `$XDG_DATA_HOME/zomah/trace.db` or `~/.local/share/zomah/trace.db`. Traces contain compact metadata only: worker, capability, timestamps, outcome, error code, and a non-content target/reference. Raw model payloads, prompts, capability results, stdout/stderr, and exception details are intentionally not stored. If tracing cannot start, ZOMAH blocks capability execution.

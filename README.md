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
src/zomah/model_boundary.py          model request/error/trace adapter
src/zomah/tracing.py                 minimal local capability trace store
tests/test_tracing.py                automatic tracing boundary tests
```

## First local run

After installing the project:

```bash
python examples/bootstrap_state.py
```

This initializes the canonical SQLite database at `$XDG_DATA_HOME/zomah/zomah.db` or `~/.local/share/zomah/zomah.db`, creates the initial ZOMAH `ProjectState` if it does not already exist, and writes a generated Markdown mirror beside the database under `exports/`.

## Knowledge retrieval

`search_knowledge` uses a derived SQLite FTS5 index over approved UTF-8 text roots. The index is rebuildable and separate from canonical ProjectState storage. ZOMAH refreshes the index incrementally before each search so new, changed, moved, or deleted documents become visible without a model-facing indexing tool.

## Write boundary

`write_file` and `move_file` use a separate `WriteScope`; read authority never implies write authority. `write_file` supports explicit `create`, `replace`, and `append` modes only. `move_file` moves regular files only, never overwrites a destination, never creates directories, and never falls back to cross-filesystem copy/delete behavior.

## Execute boundary

`run_script` can execute only scripts pre-registered by the harness owner. The model selects a script ID and values allowed by that script's explicit argument contract; it never supplies an executable path, shell command, working directory, timeout, or inherited environment. Execution always uses `shell=False`, bounded returned output, and a fixed timeout. Registered executables must live outside every model `WriteScope`, and registration records the script's SHA-256 digest. ZOMAH revalidates the executable path, execute bit, and exact digest before every run; modified scripts require explicit re-registration.

### Model-facing error boundary

Capability implementations keep native Python exceptions for development. Before results reach a model, `zomah.model_boundary.invoke_model_capability` validates the request and returns a stable `ok/result/error` envelope. Unexpected exception details are redacted from model context.

## Tracing

Model-driven capability calls are traced automatically in `$XDG_DATA_HOME/zomah/trace.db` or `~/.local/share/zomah/trace.db`. Traces contain compact metadata only: worker, capability, timestamps, outcome, error code, and a non-content target/reference. Raw model payloads, prompts, capability results, stdout/stderr, and exception details are intentionally not stored. If tracing cannot start, ZOMAH blocks capability execution.

# ZOMAH

**Zerrius's Obviously Minimal Agent Harness**

ZOMAH is a deliberately small local control plane for capable AI workers, starting with Elyria.

## Current implementation

Z1 has begun with the `ProjectState` persistence slice:

- Pydantic v2 contracts
- stdlib `sqlite3`
- normalized SQLite schema
- transactional patch updates
- optimistic revision checks
- one-way Markdown export
- pytest coverage
- model-facing `get_project_state` capability
- model-facing `update_project_state` capability
- scoped, bounded `read_file` capability
- scoped, bounded `list_directory` capability
- structured `inspect_system` capability for explicit Linux system domains
- lexical `search_knowledge` capability backed by SQLite FTS5
- scoped UTF-8 `write_file` capability with separate write roots
- scoped no-overwrite `move_file` capability for regular files

No agent loop, generic tool registry, model client, or framework has been added yet.

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
tests/test_read_file.py         file-read boundary tests
tests/test_list_directory.py    directory-list boundary tests
tests/test_inspect_system.py      system-inspection contract tests
tests/test_search_knowledge.py     retrieval boundary tests
tests/test_write_file.py           file-write boundary tests
tests/test_move_file.py            file-move boundary tests
```

## First local run

After installing the project:

```bash
python examples/bootstrap_state.py
```

This initializes the canonical SQLite database at `$XDG_DATA_HOME/zomah/zomah.db` or `~/.local/share/zomah/zomah.db`, creates the initial ZOMAH `ProjectState` if it does not already exist, and writes a generated Markdown mirror beside the database under `exports/`.

## Knowledge retrieval

`search_knowledge` uses a derived SQLite FTS5 index over approved UTF-8 text roots. The index is rebuildable and separate from canonical ProjectState storage.

## Write boundary

`write_file` and `move_file` use a separate `WriteScope`; read authority never implies write authority. `write_file` supports explicit `create`, `replace`, and `append` modes only. `move_file` moves regular files only, never overwrites a destination, never creates directories, and never falls back to cross-filesystem copy/delete behavior.

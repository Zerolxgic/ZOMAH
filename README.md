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
tests/test_read_file.py         file-read boundary tests
tests/test_list_directory.py    directory-list boundary tests
tests/test_inspect_system.py      system-inspection contract tests
```

## First local run

After installing the project:

```bash
python examples/bootstrap_state.py
```

This initializes the canonical SQLite database at `$XDG_DATA_HOME/zomah/zomah.db` or `~/.local/share/zomah/zomah.db`, creates the initial ZOMAH `ProjectState` if it does not already exist, and writes a generated Markdown mirror beside the database under `exports/`.

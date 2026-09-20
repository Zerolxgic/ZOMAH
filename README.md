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

No agent loop, tool router, model client, or framework has been added yet.

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
tests/test_project_state.py    Z1 contract tests
```

## First local run

After installing the project:

```bash
python examples/bootstrap_state.py
```

This initializes the canonical SQLite database at `$XDG_DATA_HOME/zomah/zomah.db` or `~/.local/share/zomah/zomah.db`, creates the initial ZOMAH `ProjectState` if it does not already exist, and writes a generated Markdown mirror beside the database under `exports/`.

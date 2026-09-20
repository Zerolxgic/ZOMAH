from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from zomah.capabilities import SearchKnowledgeRequest
from zomah.model_boundary import invoke_model_capability
from zomah.tracing import (
    TraceError,
    TraceOutcome,
    TraceStore,
    default_trace_db_path,
)


class EchoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class EchoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def echo(request: EchoRequest) -> EchoResponse:
    return EchoResponse(text=request.text)


def test_trace_store_records_started_and_finished_run(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "trace.db")
    run_id = store.start_run(
        worker="elyria",
        capability="read_file",
        target="/approved/note.md",
    )
    store.finish_run(run_id, outcome=TraceOutcome.SUCCEEDED)

    record = store.get(run_id)
    assert record.worker == "elyria"
    assert record.capability == "read_file"
    assert record.target == "/approved/note.md"
    assert record.outcome is TraceOutcome.SUCCEEDED
    assert record.finished_at is not None
    assert record.error_code is None


def test_model_boundary_success_is_traced_automatically(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "trace.db")

    envelope = invoke_model_capability(
        EchoRequest,
        echo,
        {"text": "hello"},
        worker="elyria",
        trace_store=store,
    )

    assert envelope.ok is True
    records = store.recent()
    assert len(records) == 1
    assert records[0].worker == "elyria"
    assert records[0].capability == "echo"
    assert records[0].outcome is TraceOutcome.SUCCEEDED


def test_validation_failure_is_traced_without_payload(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "trace.db")
    secret = "do-not-store-this"

    envelope = invoke_model_capability(
        EchoRequest,
        echo,
        {"text": "hello", "unexpected": secret},
        worker="elyria",
        trace_store=store,
    )

    assert envelope.ok is False
    record = store.recent()[0]
    assert record.outcome is TraceOutcome.FAILED
    assert record.error_code == "invalid_request"
    assert record.target is None

    with sqlite3.connect(store.database) as conn:
        serialized = "\n".join(str(value) for row in conn.execute(
            "SELECT * FROM capability_traces"
        ) for value in row if value is not None)
    assert secret not in serialized


def test_capability_failure_records_normalized_error_code(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "trace.db")

    def fail(_: EchoRequest) -> EchoResponse:
        raise FileNotFoundError("missing")

    envelope = invoke_model_capability(
        EchoRequest,
        fail,
        {"text": "hello"},
        worker="elyria",
        trace_store=store,
    )

    assert envelope.ok is False
    record = store.recent()[0]
    assert record.outcome is TraceOutcome.FAILED
    assert record.error_code == "resource_not_found"


def test_trace_start_failure_blocks_capability_execution() -> None:
    called = False

    class BrokenTraceStore:
        def start_run(self, **_: object) -> str:
            raise TraceError("offline")

    def capability(_: EchoRequest) -> EchoResponse:
        nonlocal called
        called = True
        return EchoResponse(text="should not run")

    envelope = invoke_model_capability(
        EchoRequest,
        capability,
        {"text": "hello"},
        worker="elyria",
        trace_store=BrokenTraceStore(),  # type: ignore[arg-type]
    )

    assert called is False
    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "trace_unavailable"
    assert envelope.error.retryable is True


def test_search_query_text_is_not_used_as_trace_target(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "trace.db")
    secret_query = "private project codename"

    class SearchResponse(BaseModel):
        model_config = ConfigDict(extra="forbid")
        ok: bool

    def fake_search(_: SearchKnowledgeRequest) -> SearchResponse:
        return SearchResponse(ok=True)

    envelope = invoke_model_capability(
        SearchKnowledgeRequest,
        fake_search,
        {"query": secret_query},
        worker="elyria",
        trace_store=store,
    )

    assert envelope.ok is True
    record = store.recent()[0]
    assert record.target == "knowledge-search"
    assert secret_query not in record.model_dump_json()


def test_recent_is_bounded_and_newest_first(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "trace.db")
    ids = []
    for index in range(3):
        run_id = store.start_run(worker="elyria", capability=f"cap-{index}")
        store.finish_run(run_id, outcome=TraceOutcome.SUCCEEDED)
        ids.append(run_id)

    recent = store.recent(limit=2)
    assert len(recent) == 2
    assert recent[0].run_id == ids[-1]
    assert recent[1].run_id == ids[-2]


def test_default_trace_path_uses_xdg_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_trace_db_path() == tmp_path / "zomah" / "trace.db"

"""search_knowledge as the third model tool, sharing read_file's scope."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Static

import zomah.model_tools as model_tools
from zomah.access import ReadScope
from zomah.capabilities import (
    SearchKnowledgeRequest,
    SearchKnowledgeResponse,
    search_knowledge,
)
from zomah.capability_registry import (
    CapabilityAuthority,
    CapabilityLifecycle,
    default_capability_registry,
)
from zomah.console.app import build_console
from zomah.knowledge import KnowledgeIndex, default_knowledge_db_path
from zomah.model_runtime import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ModelRuntimeError,
    ToolCall,
    ToolResultMessage,
)
from zomah.model_tools import build_session_tools
from zomah.state import ProjectState, ProjectStateRepository
from zomah.tracing import TraceOutcome, TraceStore
from zomah.worker_session import WorkerSession

BUDGET_DOC = (
    "# Tool result budget\n\n"
    "The per-turn tool-result budget defaults to 32768 characters. It is a "
    "deterministic character bound, not a token estimate.\n"
)
LONG_TAIL = "Unrelated filler paragraph about gardening and weather.\n" * 400


@pytest.fixture
def corpus(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "README.md").write_text(BUDGET_DOC + "\n" + LONG_TAIL, encoding="utf-8")
    (root / "notes.txt").write_text("Gardening notes: tomatoes and basil.\n", encoding="utf-8")
    for ignored in (".git", ".venv", "__pycache__", ".pytest_cache"):
        (root / ignored).mkdir()
        (root / ignored / "hidden.md").write_text("budget budget budget\n", encoding="utf-8")
    (root / "config.toml").write_text('budget = "32768"\n', encoding="utf-8")
    (root / "module.py").write_text("BUDGET = 32768  # budget\n", encoding="utf-8")
    (root / "data.json").write_text('{"budget": 32768}\n', encoding="utf-8")
    (outside / "linked.md").write_text("budget from outside the root\n", encoding="utf-8")
    (outside / "subdir").mkdir()
    (outside / "subdir" / "deep.md").write_text("budget inside linked dir\n", encoding="utf-8")
    (root / "link.md").symlink_to(outside / "linked.md")
    (root / "linkdir").symlink_to(outside / "subdir", target_is_directory=True)
    return {"root": root, "outside": outside, "tmp": tmp_path}


@pytest.fixture
def trace_store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "trace.db")


@pytest.fixture
def repository(tmp_path: Path) -> ProjectStateRepository:
    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()
    repository.create(
        ProjectState(
            id="zomah", name="ZOMAH", phase="search", summary="s",
            current_focus="f", updated_by="zerrius",
        )
    )
    return repository


def three_tools(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> tuple[Any, Any, ReadScope]:
    scope = ReadScope.from_paths([corpus["root"]])
    index = KnowledgeIndex(corpus["tmp"] / "knowledge.db", scope)
    tools, executor = build_session_tools(
        default_capability_registry(),
        worker="elyria",
        trace_store=trace_store,
        allowed_ids=("get_project_state", "read_file", "search_knowledge"),
        dependencies={
            "get_project_state": {"repository": repository},
            "read_file": {"scope": scope},
            "search_knowledge": {"index": index},
        },
    )
    return tools, executor, scope


def execute(executor: Any, name: str, **arguments: Any) -> str:
    return asyncio.run(executor.execute(ToolCall(id="c", name=name, arguments=arguments)))


def search(executor: Any, query: str, **kwargs: Any) -> dict[str, Any]:
    return json.loads(execute(executor, "search_knowledge", query=query, **kwargs))


def paths(envelope: dict[str, Any]) -> list[str]:
    return [result["path"] for result in envelope["result"]["results"]]


# --- registry and session exposure ---------------------------------------------------


def test_search_knowledge_registry_metadata() -> None:
    registry = default_capability_registry()
    assert [d.id for d in registry.all()] == ["get_project_state", "read_file", "search_knowledge"]
    definition = registry.get("search_knowledge")
    assert (definition.authority, definition.lifecycle) == (
        CapabilityAuthority.READ,
        CapabilityLifecycle.VERIFIED,
    )
    assert (definition.user_exposed, definition.agent_exposed) == (True, True)
    assert definition.dependencies == frozenset({"index"})
    assert (definition.request_model, definition.response_model, definition.handler) == (
        SearchKnowledgeRequest,
        SearchKnowledgeResponse,
        search_knowledge,
    )


def test_search_tool_schema_comes_from_the_request_model(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    tools, _, _ = three_tools(corpus, repository, trace_store)
    assert [tool.name for tool in tools] == ["get_project_state", "read_file", "search_knowledge"]
    assert tools[2].parameters == SearchKnowledgeRequest.model_json_schema()


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], ["get_project_state"]),
        (["--read-root", "ROOT"], ["get_project_state", "read_file"]),
        (
            ["--read-root", "ROOT", "--enable-knowledge-search"],
            ["get_project_state", "read_file", "search_knowledge"],
        ),
    ],
    ids=["model-only", "read-root", "read-root+search"],
)
def test_cli_session_tool_sets(
    monkeypatch: pytest.MonkeyPatch,
    corpus: dict[str, Path],
    extra: list[str],
    expected: list[str],
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(corpus["tmp"] / "data"))
    args = [str(corpus["root"]) if a == "ROOT" else a for a in extra]
    app = build_console(["--operator", "zerrius", "--model", "qwen/qwen3.5-9b", *args])
    session = app._session
    assert session is not None
    assert [tool.name for tool in session.tools] == expected
    assert app.status.tools_available == len(expected)

    async def runner() -> str:
        async with app.run_test(size=(110, 40)):
            return str(app.query_one("#field-tools", Static).content)

    assert asyncio.run(runner()) == f"Tools: {len(expected)}"


def test_cli_search_index_shares_read_file_scope_and_default_db(
    monkeypatch: pytest.MonkeyPatch, corpus: dict[str, Path]
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(corpus["tmp"] / "data"))
    app = build_console(
        [
            "--operator", "zerrius", "--model", "m",
            "--read-root", str(corpus["root"]), "--enable-knowledge-search",
        ]
    )
    routed = app._session._executor._dependencies  # type: ignore[union-attr]
    index = routed["search_knowledge"]["index"]
    assert set(routed["search_knowledge"]) == {"index"}
    assert index.scope is routed["read_file"]["scope"]
    assert index.db_path == default_knowledge_db_path()
    assert index.db_path.is_relative_to(corpus["tmp"] / "data")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--read-root", "ROOT", "--enable-knowledge-search"],
         "--enable-knowledge-search requires --model"),
        (["--model", "m", "--enable-knowledge-search"],
         "--enable-knowledge-search requires at least one --read-root"),
    ],
)
def test_cli_search_flag_requirements(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    corpus: dict[str, Path],
    args: list[str],
    message: str,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(corpus["tmp"] / "data"))
    with pytest.raises(SystemExit):
        build_console(
            ["--operator", "zerrius", *[str(corpus["root"]) if a == "ROOT" else a for a in args]]
        )
    assert message in capsys.readouterr().err


# --- search through the model boundary ----------------------------------------------


def test_only_index_is_routed_to_search_knowledge(
    monkeypatch: pytest.MonkeyPatch,
    corpus: dict[str, Path],
    repository: ProjectStateRepository,
    trace_store: TraceStore,
) -> None:
    seen: dict[str, set[str]] = {}
    original = model_tools.invoke_registered_model_capability

    def spy(registry: Any, capability_id: str, payload: Any, /, **kwargs: Any):
        seen[capability_id] = set(kwargs) - {"worker", "trace_store"}
        return original(registry, capability_id, payload, **kwargs)

    monkeypatch.setattr(model_tools, "invoke_registered_model_capability", spy)
    _, executor, _ = three_tools(corpus, repository, trace_store)
    assert search(executor, "budget")["ok"] is True
    assert seen == {"search_knowledge": {"index"}}


def test_indexed_markdown_is_found_with_excerpts_not_full_content(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    _, executor, _ = three_tools(corpus, repository, trace_store)
    envelope = search(executor, "tool-result budget characters")
    readme = str((corpus["root"] / "README.md").resolve())
    assert paths(envelope)[0] == readme
    top = envelope["result"]["results"][0]
    assert top["rank"] == 1 and top["title"] == "Tool result budget"
    assert "budget" in top["excerpt"].lower()
    assert len(top["excerpt"]) < 400
    # 400 filler lines exist in the source; at most a snippet's worth comes back.
    assert json.dumps(envelope).count("Unrelated filler") <= 1
    assert top["size_bytes"] > 20_000


def test_refresh_sees_changes_and_deletions_before_each_search(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    _, executor, _ = three_tools(corpus, repository, trace_store)
    new_doc = corpus["root"] / "later.md"
    assert paths(search(executor, "zeppelin")) == []

    new_doc.write_text("# Later\n\nA zeppelin appears.\n", encoding="utf-8")
    assert paths(search(executor, "zeppelin")) == [str(new_doc.resolve())]

    new_doc.write_text("# Later\n\nNow about submarines.\n", encoding="utf-8")
    os.utime(new_doc, ns=(1, 1))  # force a distinct fingerprint
    assert paths(search(executor, "zeppelin")) == []
    assert paths(search(executor, "submarines")) == [str(new_doc.resolve())]

    new_doc.unlink()
    assert paths(search(executor, "submarines")) == []


def test_ignored_dirs_unsupported_extensions_and_symlinks_stay_unindexed(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    _, executor, _ = three_tools(corpus, repository, trace_store)
    found = paths(search(executor, "budget", max_results=50))
    assert found == [str((corpus["root"] / "README.md").resolve())]
    for name in ("config.toml", "module.py", "data.json", "link.md"):
        assert not any(path.endswith(name) for path in found)
    assert not any(part in path for path in found for part in (".git", ".venv", "linkdir"))


def test_every_search_result_is_readable_by_read_file(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    _, executor, scope = three_tools(corpus, repository, trace_store)
    for path in paths(search(executor, "gardening budget tomatoes", max_results=50)):
        assert any(Path(path).is_relative_to(root) for root in scope.roots)
        envelope = json.loads(execute(executor, "read_file", path=path, max_lines=1))
        assert envelope["ok"] is True


def test_trace_target_hides_the_query(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    _, executor, _ = three_tools(corpus, repository, trace_store)
    search(executor, "secretive-query-term budget")
    (record,) = trace_store.recent()
    assert (record.worker, record.capability, record.outcome, record.target) == (
        "elyria", "search_knowledge", TraceOutcome.SUCCEEDED, "knowledge-search",
    )
    rows = trace_store.connect().execute("select * from capability_traces").fetchall()
    dumped = json.dumps([list(row) for row in rows])
    assert "secretive" not in dumped and "deterministic character bound" not in dumped


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"query": "!!! ???"}, "invalid_request"),
        ({"query": ""}, "invalid_request"),
        ({"query": "budget", "max_results": 51}, "invalid_request"),
        ({"query": "budget", "index": "/tmp/other.db"}, "invalid_request"),
    ],
)
def test_search_errors_are_tool_result_envelopes(
    corpus: dict[str, Path],
    repository: ProjectStateRepository,
    trace_store: TraceStore,
    arguments: dict[str, Any],
    code: str,
) -> None:
    _, executor, _ = three_tools(corpus, repository, trace_store)
    envelope = json.loads(execute(executor, "search_knowledge", **arguments))
    assert (envelope["ok"], envelope["error"]["code"]) == (False, code)


# --- session tool loop ---------------------------------------------------------------


class ScriptedRuntime:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        next_response = self.responses.pop(0)
        return next_response(request) if callable(next_response) else next_response


def tool_reply(call: ToolCall) -> ModelResponse:
    return ModelResponse(message=AssistantMessage("", (call,)))


def read_top_result(request: ModelRequest) -> ModelResponse:
    """The fake model picks the top search path itself and asks to read it."""

    envelope = json.loads(request.messages[-1].content)  # type: ignore[union-attr]
    top = envelope["result"]["results"][0]["path"]
    return tool_reply(ToolCall(id="read_1", name="read_file", arguments={"path": top}))


def answer_from_file(request: ModelRequest) -> ModelResponse:
    envelope = json.loads(request.messages[-1].content)  # type: ignore[union-attr]
    first_lines = envelope["result"]["content"].splitlines()[:3]
    return ModelResponse(message=AssistantMessage("Answer: " + " ".join(first_lines)))


def test_model_searches_then_chooses_to_read_then_answers(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    tools, executor, _ = three_tools(corpus, repository, trace_store)
    runtime = ScriptedRuntime(
        tool_reply(ToolCall(id="search_1", name="search_knowledge",
                            arguments={"query": "per-turn tool-result budget"})),
        read_top_result,
        answer_from_file,
    )
    worker = WorkerSession(runtime, worker="elyria", model="m", tools=tools, tool_executor=executor)
    result = asyncio.run(worker.send("Where is the budget documented?"))

    assert "32768 characters" in result.assistant.text
    # After the search, only the search result went back to the model: ZOMAH
    # did not read anything on its own.
    after_search = [m for m in runtime.requests[1].messages if isinstance(m, ToolResultMessage)]
    assert [m.call_id for m in after_search] == ["search_1"]
    assert [m.role for m in worker.history] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    readme = str((corpus["root"] / "README.md").resolve())
    assert [(r.capability, r.outcome, r.target) for r in reversed(trace_store.recent())] == [
        ("search_knowledge", TraceOutcome.SUCCEEDED, "knowledge-search"),
        ("read_file", TraceOutcome.SUCCEEDED, readme),
    ]


def test_zomah_never_reads_unless_the_model_asks(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    tools, executor, _ = three_tools(corpus, repository, trace_store)
    runtime = ScriptedRuntime(
        tool_reply(ToolCall(id="s1", name="search_knowledge", arguments={"query": "budget"})),
        ModelResponse(message=AssistantMessage("I found README.md.")),
    )
    worker = WorkerSession(runtime, worker="elyria", model="m", tools=tools, tool_executor=executor)
    asyncio.run(worker.send("search only"))
    assert [r.capability for r in trace_store.recent()] == ["search_knowledge"]


def test_search_results_count_toward_the_turn_budget(
    corpus: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    tools, executor, _ = three_tools(corpus, repository, trace_store)
    envelope_len = len(execute(executor, "search_knowledge", query="budget"))
    trace_store.connect().execute("delete from capability_traces").connection.commit()

    runtime = ScriptedRuntime(
        tool_reply(ToolCall(id="s1", name="search_knowledge", arguments={"query": "budget"})),
        ModelResponse(message=AssistantMessage("unreachable")),
    )
    worker = WorkerSession(
        runtime, worker="elyria", model="m", tools=tools, tool_executor=executor,
        tool_result_budget_chars=envelope_len - 1,
    )
    with pytest.raises(ModelRuntimeError, match="per-turn budget"):
        asyncio.run(worker.send("search"))
    assert len(runtime.requests) == 1
    assert worker.history == ()
    (record,) = trace_store.recent()
    assert (record.capability, record.outcome) == ("search_knowledge", TraceOutcome.SUCCEEDED)

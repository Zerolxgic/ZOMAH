"""read_file as a second model tool, with per-capability dependency routing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from textual.widgets import Static

import zomah.model_tools as model_tools
from zomah.access import ReadScope
from zomah.capabilities import ReadFileRequest, ReadFileResponse, read_file
from zomah.capability_registry import (
    CapabilityAuthority,
    CapabilityDefinition,
    CapabilityLifecycle,
    default_capability_registry,
)
from zomah.console import Composer, OperatorConsole
from zomah.console.app import build_console
from zomah.model_runtime import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultMessage,
)
from zomah.model_tools import (
    DEFAULT_SESSION_TOOL_IDS,
    READ_FILE_TOOL_ID,
    ToolExposureError,
    build_session_tools,
    tool_definition,
)
from zomah.state import ProjectState, ProjectStateRepository
from zomah.tracing import TraceOutcome, TraceStore
from zomah.worker_session import WorkerSession

SECRET = "TOP-SECRET-OUTSIDE-ROOT"


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "zomah"\n'
        "description = \"Zerrius's Orchestrated Minimal Agent Harness\"\n"
        'requires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    (root / "long.txt").write_text(
        "".join(f"line {i}\n" for i in range(1, 301)), encoding="utf-8"
    )
    (root / "binary.bin").write_bytes(b"\xff\xfe\x00\x01not utf-8")
    (outside / "secret.txt").write_text(SECRET + "\n", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside / "secret.txt")
    return {"root": root, "outside": outside, "tmp": tmp_path}


@pytest.fixture
def repository(tmp_path: Path) -> ProjectStateRepository:
    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()
    repository.create(
        ProjectState(
            id="zomah", name="ZOMAH", phase="tools", summary="s",
            current_focus="f", next_action="Expose read_file.", updated_by="zerrius",
        )
    )
    return repository


@pytest.fixture
def trace_store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "trace.db")


def full_session_tools(
    layout: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> tuple[Any, Any]:
    return build_session_tools(
        default_capability_registry(),
        worker="elyria",
        trace_store=trace_store,
        allowed_ids=(*DEFAULT_SESSION_TOOL_IDS, READ_FILE_TOOL_ID),
        dependencies={
            "get_project_state": {"repository": repository},
            "read_file": {"scope": ReadScope.from_paths([layout["root"]])},
        },
    )


def execute(executor: Any, name: str, **arguments: Any) -> dict[str, Any]:
    content = asyncio.run(executor.execute(ToolCall(id="call_1", name=name, arguments=arguments)))
    return json.loads(content)


# --- registry -----------------------------------------------------------------


def test_read_file_is_registered_as_verified_read_for_users_and_agents() -> None:
    registry = default_capability_registry()
    assert [d.id for d in registry.all()] == [
        "get_project_state",
        "read_file",
        "search_knowledge",
    ]
    definition = registry.get("read_file")
    assert (definition.authority, definition.lifecycle) == (
        CapabilityAuthority.READ,
        CapabilityLifecycle.VERIFIED,
    )
    assert (definition.user_exposed, definition.agent_exposed) == (True, True)
    assert (definition.request_model, definition.response_model, definition.handler) == (
        ReadFileRequest,
        ReadFileResponse,
        read_file,
    )
    assert definition.dependencies == frozenset({"scope"})
    assert registry.get("get_project_state").dependencies == frozenset({"repository"})
    assert definition.description == (
        "Read a bounded UTF-8 text slice from an absolute path inside configured read roots."
    )


class _Req(BaseModel):
    value: str


@pytest.mark.parametrize("names", [frozenset({"worker"}), frozenset({"trace_store"}),
                                   frozenset({"not an identifier"}), {"scope"}])
def test_dependency_declarations_are_validated(names: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        CapabilityDefinition(
            id="probe", description="d", authority=CapabilityAuthority.READ,
            lifecycle=CapabilityLifecycle.VERIFIED, user_exposed=False, agent_exposed=True,
            request_model=_Req, response_model=_Req, handler=lambda r: r, dependencies=names,
        )


# --- session exposure -----------------------------------------------------------


def test_default_session_exposes_only_get_project_state_despite_registry(
    repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    registry = default_capability_registry()
    assert len(registry.agent_exposed()) == 3
    tools, _ = build_session_tools(
        registry,
        worker="elyria",
        trace_store=trace_store,
        dependencies={"get_project_state": {"repository": repository}},
    )
    assert [tool.name for tool in tools] == ["get_project_state"]


def test_configured_read_scope_exposes_exactly_two_tools(
    layout: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    tools, _ = full_session_tools(layout, repository, trace_store)
    assert [tool.name for tool in tools] == ["get_project_state", "read_file"]
    read_tool = tools[1]
    assert read_tool == tool_definition(default_capability_registry().get("read_file"))
    assert read_tool.parameters == ReadFileRequest.model_json_schema()


# --- dependency routing ----------------------------------------------------------


def test_each_capability_receives_only_its_own_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    layout: dict[str, Path],
    repository: ProjectStateRepository,
    trace_store: TraceStore,
) -> None:
    seen: dict[str, set[str]] = {}
    original = model_tools.invoke_registered_model_capability

    def spy(registry: Any, capability_id: str, payload: Any, /, **kwargs: Any):
        seen[capability_id] = set(kwargs) - {"worker", "trace_store"}
        return original(registry, capability_id, payload, **kwargs)

    monkeypatch.setattr(model_tools, "invoke_registered_model_capability", spy)
    _, executor = full_session_tools(layout, repository, trace_store)
    assert execute(executor, "get_project_state", project_id="zomah")["ok"] is True
    assert execute(executor, "read_file", path=str(layout["root"] / "pyproject.toml"))["ok"]
    assert seen == {"get_project_state": {"repository"}, "read_file": {"scope"}}


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_file", {"path": "/etc/hostname", "scope": "/"}),
        ("read_file", {"path": "/etc/hostname", "repository": "x"}),
        ("get_project_state", {"project_id": "zomah", "scope": "/"}),
    ],
)
def test_model_arguments_cannot_supply_harness_dependencies(
    layout: dict[str, Path],
    repository: ProjectStateRepository,
    trace_store: TraceStore,
    name: str,
    arguments: dict[str, Any],
) -> None:
    _, executor = full_session_tools(layout, repository, trace_store)
    envelope = execute(executor, name, **arguments)
    assert (envelope["ok"], envelope["error"]["code"]) == (False, "invalid_request")


@pytest.mark.parametrize(
    ("dependencies", "message"),
    [
        ({"get_project_state": {"repository": "r"}}, "read_file requires harness dependencies: scope"),
        ({"get_project_state": {"repository": "r"}, "read_file": {"repository": "r"}},
         "read_file requires harness dependencies: scope"),
        ({"get_project_state": {"repository": "r"}, "read_file": {"scope": "s", "repository": "r"}},
         "read_file does not accept dependencies: repository"),
        ({"get_project_state": {"repository": "r", "scope": "s"}, "read_file": {"scope": "s"}},
         "get_project_state does not accept dependencies: scope"),
    ],
)
def test_misconfigured_dependencies_fail_at_construction(
    trace_store: TraceStore, dependencies: dict[str, Any], message: str
) -> None:
    with pytest.raises(ToolExposureError, match=message):
        build_session_tools(
            default_capability_registry(),
            worker="elyria",
            trace_store=trace_store,
            allowed_ids=("get_project_state", "read_file"),
            dependencies=dependencies,
        )


def test_dependencies_for_capabilities_outside_the_session_are_refused(
    trace_store: TraceStore,
) -> None:
    with pytest.raises(ToolExposureError, match="not in this session: read_file"):
        build_session_tools(
            default_capability_registry(),
            worker="elyria",
            trace_store=trace_store,
            dependencies={"get_project_state": {"repository": "r"}, "read_file": {"scope": "s"}},
        )


def test_capability_without_dependencies_needs_no_entry(trace_store: TraceStore) -> None:
    registry = default_capability_registry()
    registry.register(
        CapabilityDefinition(
            id="echo", description="Echo a value.", authority=CapabilityAuthority.READ,
            lifecycle=CapabilityLifecycle.VERIFIED, user_exposed=False, agent_exposed=True,
            request_model=_Req, response_model=_Req, handler=lambda request: request,
        )
    )
    _, executor = build_session_tools(
        registry, worker="elyria", trace_store=trace_store,
        allowed_ids=("echo",), dependencies={},
    )
    assert execute(executor, "echo", value="hi") == {"ok": True, "result": {"value": "hi"}}


# --- read_file through the model boundary -----------------------------------------


def test_in_root_read_succeeds_and_traces_path_without_content(
    layout: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    _, executor = full_session_tools(layout, repository, trace_store)
    target = str(layout["root"] / "pyproject.toml")
    envelope = execute(executor, "read_file", path=target)
    assert envelope["ok"] is True
    assert "Zerrius's Orchestrated Minimal Agent Harness" in envelope["result"]["content"]
    assert ">=3.11" in envelope["result"]["content"]

    (record,) = trace_store.recent()
    assert (record.worker, record.capability, record.outcome, record.target) == (
        "elyria", "read_file", TraceOutcome.SUCCEEDED, target,
    )
    raw_rows = trace_store.connect().execute("select * from capability_traces").fetchall()
    assert "Orchestrated" not in json.dumps([list(row) for row in raw_rows])


@pytest.mark.parametrize(
    ("path_key", "code"),
    [
        ("outside", "path_outside_scope"),
        ("symlink", "path_outside_scope"),
        ("relative", "invalid_read_target"),
        ("missing", "invalid_read_target"),
        ("directory", "invalid_read_target"),
        ("binary", "unsupported_text_file"),
    ],
)
def test_scope_and_file_errors_are_envelopes_without_content(
    layout: dict[str, Path],
    repository: ProjectStateRepository,
    trace_store: TraceStore,
    path_key: str,
    code: str,
) -> None:
    paths = {
        "outside": str(layout["outside"] / "secret.txt"),
        "symlink": str(layout["root"] / "escape.txt"),
        "relative": "pyproject.toml",
        "missing": str(layout["root"] / "nope.txt"),
        "directory": str(layout["root"]),
        "binary": str(layout["root"] / "binary.bin"),
    }
    _, executor = full_session_tools(layout, repository, trace_store)
    envelope = execute(executor, "read_file", path=paths[path_key])
    assert (envelope["ok"], envelope["error"]["code"]) == (False, code)
    assert SECRET not in json.dumps(envelope)
    (record,) = trace_store.recent()
    assert (record.worker, record.outcome, record.error_code) == (
        "elyria", TraceOutcome.FAILED, code,
    )


def test_bounded_reads_keep_next_start_line_continuation(
    layout: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    _, executor = full_session_tools(layout, repository, trace_store)
    target = str(layout["root"] / "long.txt")
    first = execute(executor, "read_file", path=target, max_lines=5)["result"]
    assert (first["truncated"], first["next_start_line"], first["end_line"]) == (True, 6, 5)
    second = execute(executor, "read_file", path=target, start_line=6, max_lines=5)["result"]
    assert second["content"].startswith("line 6\n")
    assert second["next_start_line"] == 11


# --- tool loop -----------------------------------------------------------------


class ScriptedRuntime:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def test_model_recovers_from_a_read_error_and_uses_both_tools(
    layout: dict[str, Path], repository: ProjectStateRepository, trace_store: TraceStore
) -> None:
    tools, executor = full_session_tools(layout, repository, trace_store)
    target = str(layout["root"] / "pyproject.toml")
    bad = ToolCall(id="c1", name="read_file", arguments={"path": "pyproject.toml"})
    good = ToolCall(id="c2", name="read_file", arguments={"path": target})
    state = ToolCall(id="c3", name="get_project_state", arguments={"project_id": "zomah"})
    runtime = ScriptedRuntime(
        ModelResponse(message=AssistantMessage("", (bad,))),
        ModelResponse(message=AssistantMessage("", (good, state))),
        ModelResponse(message=AssistantMessage("Description and phase found.")),
    )
    worker = WorkerSession(
        runtime, worker="elyria", model="m", tools=tools, tool_executor=executor
    )
    result = asyncio.run(worker.send("Read pyproject.toml and the project state."))

    assert result.assistant.text == "Description and phase found."
    assert [tool.name for tool in runtime.requests[0].tools] == ["get_project_state", "read_file"]
    first_result = runtime.requests[1].messages[-1]
    assert isinstance(first_result, ToolResultMessage) and first_result.call_id == "c1"
    assert json.loads(first_result.content)["error"]["code"] == "invalid_read_target"
    results = [m for m in runtime.requests[2].messages if isinstance(m, ToolResultMessage)]
    assert [m.call_id for m in results] == ["c1", "c2", "c3"]
    assert json.loads(results[1].content)["result"]["path"] == target
    assert json.loads(results[2].content)["result"]["project"]["id"] == "zomah"
    assert [m.role for m in worker.history] == [
        "user", "assistant", "tool", "assistant", "tool", "tool", "assistant",
    ]
    assert [(r.capability, r.outcome) for r in reversed(trace_store.recent())] == [
        ("read_file", TraceOutcome.FAILED),
        ("read_file", TraceOutcome.SUCCEEDED),
        ("get_project_state", TraceOutcome.SUCCEEDED),
    ]


# --- console / CLI ----------------------------------------------------------------


def header_tools(app: OperatorConsole) -> str:
    return str(app.query_one("#field-tools", Static).content)


def run_commands(app: OperatorConsole, *commands: str) -> list[str]:
    outputs: list[str] = []

    async def runner() -> None:
        async with app.run_test(size=(110, 40)) as pilot:
            outputs.append(header_tools(app))
            for command in commands:
                app.query_one(Composer).load_text(command)
                await pilot.press("escape", "enter")
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                outputs.append(str(list(app.query(".entry.result").results(Static))[-1].content))

    asyncio.run(runner())
    return outputs


def test_cli_without_read_root_keeps_one_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    app = build_console(["--operator", "zerrius", "--model", "qwen/qwen3.5-9b"])
    assert [tool.name for tool in app._session.tools] == ["get_project_state"]  # type: ignore[union-attr]
    tools_header, status_output, tools_output = run_commands(app, "/status", "/tools")
    assert tools_header == "Tools: 1"
    assert " ".join(status_output.split()).count("Tools: 1") == 1
    assert "read_file" in tools_output  # registered metadata...
    assert "Model qwen/qwen3.5-9b: 1 tool available this session." in tools_output


def test_cli_with_read_roots_exposes_two_tools_and_keeps_project_human(
    monkeypatch: pytest.MonkeyPatch, layout: dict[str, Path]
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(layout["tmp"] / "data"))
    second_root = layout["tmp"] / "second"
    second_root.mkdir()
    app = build_console(
        [
            "--operator", "zerrius", "--project", "zomah", "--model", "qwen/qwen3.5-9b",
            "--read-root", str(layout["root"]), "--read-root", str(second_root),
        ]
    )
    session = app._session
    assert session is not None
    assert [tool.name for tool in session.tools] == ["get_project_state", "read_file"]
    scope = session._executor._dependencies["read_file"]["scope"]  # type: ignore[union-attr]
    assert scope.roots == (layout["root"].resolve(), second_root.resolve())

    tools_header, status_output, tools_output, project_output = run_commands(
        app, "/status", "/tools", "/project"
    )
    assert tools_header == "Tools: 2"
    assert "Tools: 2" in " ".join(status_output.split())
    assert "Model qwen/qwen3.5-9b: 2 tools available this session." in tools_output
    # /project reads through the human boundary (canonical state absent here).
    assert project_output.startswith("zomah · Project")
    records = app._operator_access.trace_store.recent()
    assert [(r.worker, r.capability) for r in records] == [("operator:zerrius", "get_project_state")]


@pytest.mark.parametrize("case", ["missing", "not-a-directory", "without-model"])
def test_bad_read_root_configuration_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    layout: dict[str, Path],
    case: str,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(layout["tmp"] / "data"))
    args, message = {
        "missing": (
            ["--model", "m", "--read-root", "/definitely/not/a/dir"],
            "--read-root must name existing directories: /definitely/not/a/dir",
        ),
        "not-a-directory": (
            ["--model", "m", "--read-root", str(layout["root"] / "pyproject.toml")],
            "--read-root must name existing directories",
        ),
        "without-model": (
            ["--read-root", str(layout["root"])],
            "--read-root requires --model",
        ),
    }[case]
    with pytest.raises(SystemExit):
        build_console(["--operator", "zerrius", *args])
    assert message in capsys.readouterr().err

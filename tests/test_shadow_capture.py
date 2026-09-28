"""Experimental shadow observation of model-requested tool calls.

The observer is best-effort local telemetry. It sees each normalized tool call
immediately before WorkerSession's normal execution path and must never change
what runs, whether it runs, or how the turn ends.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from zomah.console.app import build_console
from zomah.model_runtime import AssistantMessage, ModelRequest, ModelResponse, ModelRuntimeError, ToolCall, ToolDefinition
from zomah.shadow_capture import JsonlToolCallObserver
from zomah.worker_session import MAX_TOOL_CALLS_PER_RESPONSE, WorkerSession

SRC = Path(__file__).resolve().parents[1] / "src" / "zomah"
GOAL = "What phase is zomah in?"
PROJECT_TOOL = ToolDefinition(
    name="get_project_state",
    description="Retrieve the canonical current state for one project.",
    parameters={"type": "object", "properties": {"project_id": {"type": "string"}}},
)


# --- fakes ------------------------------------------------------------------------------------


class ScriptedRuntime:
    def __init__(self, *outcomes: ModelResponse) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outcomes.pop(0)


class RecordingExecutor:
    def __init__(self, log: list | None = None, result: str | None = None) -> None:
        self.calls: list[ToolCall] = []
        self.log = log if log is not None else []
        self.result = result

    async def execute(self, call: ToolCall) -> str:
        self.calls.append(call)
        self.log.append(("execute", call.id, json.loads(json.dumps(call.arguments))))
        return self.result if self.result is not None else json.dumps({"ok": True, "result": {"echo": call.arguments}})


class RecordingObserver:
    def __init__(self, log: list | None = None) -> None:
        self.seen: list[dict[str, Any]] = []
        self.log = log if log is not None else []

    def observe(self, *, worker: str, model: str, goal: str, call: ToolCall) -> None:
        self.seen.append({"worker": worker, "model": model, "goal": goal, "call": call})
        self.log.append(("observe", call.id))


class FailingObserver:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.attempts = 0

    def observe(self, **_: Any) -> None:
        self.attempts += 1
        raise self.error


class MutatingObserver:
    """A misbehaving observer that edits the call it was shown."""

    def observe(self, *, call: ToolCall, **_: Any) -> None:
        call.arguments["project_id"] = "someone-else"
        call.arguments.setdefault("nested", {})["injected"] = True


def call(call_id: str, name: str = "get_project_state", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments or {"project_id": "zomah"})


def tools_reply(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(message=AssistantMessage("Checking.", calls))


def final(text: str = "Phase is X.") -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text))


def two_round_script() -> tuple[ModelResponse, ...]:
    return (
        tools_reply(call("call_1"), call("call_2", project_id="other")),
        tools_reply(call("call_3", project_id="third")),
        final(),
    )


def session(runtime: ScriptedRuntime, executor: Any, observer: Any = None, **kwargs: Any) -> WorkerSession:
    return WorkerSession(
        runtime,
        worker="elyria",
        model="qwen/qwen3.5-9b",
        tools=kwargs.pop("tools", (PROJECT_TOOL,)),
        tool_executor=executor,
        tool_call_observer=observer,
        **kwargs,
    )


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def turn(observer: Any, script=two_round_script) -> tuple[Any, WorkerSession, ScriptedRuntime, RecordingExecutor]:
    runtime, executor = ScriptedRuntime(*script()), RecordingExecutor()
    worker = session(runtime, executor, observer)
    return run(worker.send(GOAL)), worker, runtime, executor


# --- WorkerSession ----------------------------------------------------------------------------


def test_without_an_observer_the_session_behaves_exactly_as_before() -> None:
    plain_result, plain, plain_runtime, plain_executor = turn(None)
    observed_result, observed, observed_runtime, observed_executor = turn(RecordingObserver())
    assert plain._tool_call_observer is None
    assert plain_result == observed_result
    assert plain.history == observed.history
    assert plain_runtime.requests == observed_runtime.requests
    assert plain_executor.calls == observed_executor.calls == [call("call_1"), call("call_2", project_id="other"), call("call_3", project_id="third")]


def test_observer_sees_each_normalized_call_with_worker_model_and_goal_before_it_runs() -> None:
    log: list = []
    observer = RecordingObserver(log)
    runtime, executor = ScriptedRuntime(*two_round_script()), RecordingExecutor(log)
    run(session(runtime, executor, observer).send(GOAL))

    assert [entry[:2] for entry in log] == [
        ("observe", "call_1"), ("execute", "call_1"),
        ("observe", "call_2"), ("execute", "call_2"),
        ("observe", "call_3"), ("execute", "call_3"),
    ]
    assert [seen["call"] for seen in observer.seen] == executor.calls
    assert all(isinstance(seen["call"], ToolCall) for seen in observer.seen)
    assert {(s["worker"], s["model"], s["goal"]) for s in observer.seen} == {("elyria", "qwen/qwen3.5-9b", GOAL)}


@pytest.mark.parametrize("error", [RuntimeError("disk full"), ValueError("bad"), OSError("read-only"), TypeError("x")])
def test_observer_failure_never_changes_execution_or_the_turn(error: Exception) -> None:
    baseline_result, baseline, baseline_runtime, baseline_executor = turn(None)
    failing = FailingObserver(error)
    result, worker, runtime, executor = turn(failing)
    assert failing.attempts == 3
    assert result == baseline_result
    assert worker.history == baseline.history
    assert runtime.requests == baseline_runtime.requests
    assert executor.calls == baseline_executor.calls
    assert worker.busy is False


def test_observer_cannot_change_what_executes() -> None:
    log: list = []
    runtime, executor = ScriptedRuntime(tools_reply(call("call_1")), final()), RecordingExecutor(log)
    worker = session(runtime, executor, MutatingObserver())
    run(worker.send(GOAL))
    assert log == [("execute", "call_1", {"project_id": "zomah"})]
    assert worker.history[1] == AssistantMessage("Checking.", (call("call_1"),))


def test_unavailable_tools_are_observed_as_requested_but_never_executed() -> None:
    # Documented semantics: the observer records what the model asked for, including
    # calls WorkerSession then answers with tool_not_available and never executes.
    observer = RecordingObserver()
    runtime, executor = ScriptedRuntime(tools_reply(call("h1", name="delete_everything", path="/")), final()), RecordingExecutor()
    run(session(runtime, executor, observer).send(GOAL))
    assert [s["call"].name for s in observer.seen] == ["delete_everything"]
    assert executor.calls == []
    tool_result = json.loads(runtime.requests[1].messages[-1].content)
    assert tool_result["error"]["code"] == "tool_not_available"

    toolless_observer = RecordingObserver()
    toolless = WorkerSession(ScriptedRuntime(tools_reply(call("h2")), final()), worker="elyria", model="m",
                             tool_call_observer=toolless_observer)
    run(toolless.send(GOAL))
    assert [s["call"].id for s in toolless_observer.seen] == ["h2"]


def test_calls_stopped_by_loop_bounds_or_budget_are_not_observed() -> None:
    too_many = tuple(call(f"c{i}") for i in range(MAX_TOOL_CALLS_PER_RESPONSE + 1))
    observer = RecordingObserver()
    runtime, executor = ScriptedRuntime(tools_reply(*too_many)), RecordingExecutor()
    with pytest.raises(ModelRuntimeError, match="too many tools at once"):
        run(session(runtime, executor, observer).send(GOAL))
    assert observer.seen == [] and executor.calls == []

    budget_observer = RecordingObserver()
    runtime, executor = ScriptedRuntime(tools_reply(call("c1"), call("c2"))), RecordingExecutor(result="x" * 60)
    with pytest.raises(ModelRuntimeError, match="budget"):
        run(session(runtime, executor, budget_observer, tool_result_budget_chars=50).send(GOAL))
    assert [s["call"].id for s in budget_observer.seen] == ["c1"]


# --- JSONL capture ------------------------------------------------------------------------------

EVENT_KEYS = {
    "event_id", "timestamp", "source", "stage", "worker", "model", "goal",
    "tool_call_id", "planned_action", "arguments",
}


def read_events(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_jsonl_writes_one_valid_object_per_event_with_identity(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "shadow.jsonl"
    observer = JsonlToolCallObserver(path)
    for i in range(3):
        observer.observe(worker="elyria", model="qwen/qwen3.5-9b", goal=GOAL, call=call(f"call_{i}", project_id=f"p{i}"))

    raw_lines = path.read_text(encoding="utf-8").split("\n")
    assert raw_lines[-1] == "" and len(raw_lines) == 4  # newline-terminated, one line per event
    events = read_events(path)
    assert [set(event) for event in events] == [EVENT_KEYS] * 3
    assert len({event["event_id"] for event in events}) == 3
    for i, event in enumerate(events):
        assert event["event_id"].startswith("zomah-") and len(event["event_id"]) == len("zomah-") + 16
        stamp = datetime.fromisoformat(event["timestamp"])
        assert stamp.utcoffset() == timedelta(0)
        assert event["source"] == "zomah"
        assert event["stage"] == "model_requested"
        assert (event["worker"], event["model"], event["goal"]) == ("elyria", "qwen/qwen3.5-9b", GOAL)
        assert (event["tool_call_id"], event["planned_action"]) == (f"call_{i}", "get_project_state")
        assert event["arguments"] == {"project_id": f"p{i}"}

    observer.observe(worker="elyria", model="m", goal="again", call=call("call_9"))
    assert len(read_events(path)) == 4  # appends, never truncates


def observed_arguments(tmp_path: Path, **arguments: Any) -> dict[str, Any]:
    path = tmp_path / "shadow.jsonl"
    JsonlToolCallObserver(path).observe(worker="w", model="m", goal="g", call=ToolCall("id", "tool", arguments))
    return read_events(path)[-1]["arguments"]


def test_secret_fields_are_redacted_at_any_depth(tmp_path: Path) -> None:
    arguments = observed_arguments(
        tmp_path,
        authorization="Basic dXNlcjpwYXNz",
        password="hunter2",
        token="t0ken-value",
        api_key="k-123",
        private_key="-----BEGIN PRIVATE KEY-----abc",
        cookie="session=abc",
        credential="cred",
        path="/home/zerrius/notes.md",
        headers={"Authorization": "Bearer abc", "X-Api-Key": "k", "Accept": "text/plain"},
        items=[{"refresh_token": "r"}, {"nested": {"clientSecret": "s", "keep": 1}}, "plain"],
        privateKey="-----BEGIN RSA PRIVATE KEY-----xyz",
        accessKey="AKIAEXAMPLE",
    )
    for key in ("authorization", "password", "token", "api_key", "private_key", "cookie", "credential", "privateKey", "accessKey"):
        assert arguments[key] == "[REDACTED]", key
    assert arguments["headers"] == {"Authorization": "[REDACTED]", "X-Api-Key": "[REDACTED]", "Accept": "text/plain"}
    assert arguments["items"] == [{"refresh_token": "[REDACTED]"}, {"nested": {"clientSecret": "[REDACTED]", "keep": 1}}, "plain"]
    assert arguments["path"] == "/home/zerrius/notes.md"
    raw = (tmp_path / "shadow.jsonl").read_text(encoding="utf-8")
    for secret in ("hunter2", "t0ken-value", "k-123", "BEGIN", "session=abc", "AKIAEXAMPLE", "dXNlcjpwYXNz"):
        assert secret not in raw, secret


def test_payload_fields_are_omitted(tmp_path: Path) -> None:
    arguments = observed_arguments(
        tmp_path,
        path="/tmp/a.md",
        mode="replace",
        content="# full file body",
        contents="x",
        body={"large": "object"},
        data=["a", "b"],
        payload="p",
        file_content="f",
        bytes="b",
        blob="bl",
        file={"Content": "nested body", "name": "a.md"},
    )
    for key in ("content", "contents", "body", "data", "payload", "file_content", "bytes", "blob"):
        assert arguments[key] == "[OMITTED_PAYLOAD]", key
    assert arguments["file"] == {"Content": "[OMITTED_PAYLOAD]", "name": "a.md"}
    assert (arguments["path"], arguments["mode"]) == ("/tmp/a.md", "replace")


def test_text_is_sanitized_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    secrets_text = (
        "use Bearer abc.def-ghi and sk-proj-ABCDEFGH12345 and ghp_ABCDEFGHIJ123 and "
        "github_pat_ABCDEFGHIJ_123 and xoxb-1234567890-abc and MY_API_KEY=supersecret"
    )
    JsonlToolCallObserver(path).observe(
        worker="w", model="m", goal="deploy with GITHUB_TOKEN=ghs_zzz " + "g" * 2000,
        call=ToolCall("id", "tool", {"query": secrets_text, "note": "n" * 700, "limit": 5, "flag": True, "none": None,
                                     "tags": ["xoxp-1234567890-abc", "ok"]}),
    )
    event = read_events(path)[0]
    args = event["arguments"]
    for leaked in ("abc.def-ghi", "ABCDEFGH12345", "ghp_ABCDEFGHIJ123", "github_pat_ABCDEFGHIJ", "xoxb-1234567890", "supersecret"):
        assert leaked not in args["query"], leaked
    assert "MY_API_KEY=[REDACTED]" in args["query"] and args["query"].count("[REDACTED]") == 6
    assert args["tags"] == ["[REDACTED]", "ok"]
    assert (args["limit"], args["flag"], args["none"]) == (5, True, None)
    assert args["note"] == "n" * 512 + "…[truncated]"
    assert event["goal"].startswith("deploy with GITHUB_TOKEN=[REDACTED] ")
    assert len(event["goal"]) == 1024 + len("…[truncated]") and "ghs_zzz" not in event["goal"]


def test_redaction_happens_before_truncation(tmp_path: Path) -> None:
    # A secret straddling the length bound must not leak its first characters.
    arguments = observed_arguments(tmp_path, query="x" * 505 + " sk-ABCDEFGHIJKLMNOP")
    assert "sk-ABC" not in arguments["query"] and arguments["query"].startswith("x" * 505 + " [REDAC")


def test_observer_file_is_only_created_when_a_call_is_observed(tmp_path: Path) -> None:
    path = tmp_path / "later" / "shadow.jsonl"
    JsonlToolCallObserver(path)
    assert not path.exists() and not path.parent.exists()


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")


@posix_only
@pytest.mark.parametrize("umask", [0o000, 0o022, 0o077])
def test_new_shadow_log_is_created_owner_only_whatever_the_umask(tmp_path: Path, umask: int) -> None:
    path = tmp_path / "shadow.jsonl"
    previous = os.umask(umask)
    try:
        JsonlToolCallObserver(path).observe(worker="w", model="m", goal="g", call=call("call_1"))
    finally:
        os.umask(previous)
    assert file_mode(path) == 0o600
    assert len(read_events(path)) == 1


@posix_only
@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644])
def test_existing_shadow_log_keeps_its_mode_and_is_appended(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "shadow.jsonl"
    path.write_text('{"existing":true}\n', encoding="utf-8")
    path.chmod(mode)
    previous = os.umask(0o000)
    try:
        JsonlToolCallObserver(path).observe(worker="w", model="m", goal="g", call=call("call_1"))
    finally:
        os.umask(previous)
    assert file_mode(path) == mode  # never broadened (nor narrowed) by the writer
    assert [set(event) for event in read_events(path)] == [{"existing"}, EVENT_KEYS]


def test_unwritable_shadow_log_never_changes_the_turn(tmp_path: Path) -> None:
    # A real JSONL observer that cannot open its file stays best-effort.
    blocked = tmp_path / "shadow.jsonl"
    blocked.mkdir()
    baseline_result, baseline, baseline_runtime, baseline_executor = turn(None)
    result, worker, runtime, executor = turn(JsonlToolCallObserver(blocked))
    assert result == baseline_result
    assert worker.history == baseline.history
    assert runtime.requests == baseline_runtime.requests
    assert executor.calls == baseline_executor.calls
    assert blocked.is_dir() and not any(blocked.iterdir())


# --- console opt-in -------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
def test_console_has_no_observer_unless_shadow_log_is_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    if value is None:
        monkeypatch.delenv("ZOMAH_SHADOW_TOOL_LOG", raising=False)
    else:
        monkeypatch.setenv("ZOMAH_SHADOW_TOOL_LOG", value)
    app = build_console(["--operator", "zerrius", "--model", "qwen/qwen3.5-9b"])
    assert app._session is not None and app._session._tool_call_observer is None
    assert [tool.name for tool in app._session.tools] == ["get_project_state"]
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_console_builds_a_jsonl_observer_only_when_opted_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    log = tmp_path / "shadow" / "calls.jsonl"
    monkeypatch.setenv("ZOMAH_SHADOW_TOOL_LOG", str(log))
    app = build_console(["--operator", "zerrius", "--model", "qwen/qwen3.5-9b"])
    observer = app._session._tool_call_observer
    assert isinstance(observer, JsonlToolCallObserver) and observer._path == log
    assert not log.exists()  # nothing is written until the model requests a tool
    assert app._session._executor is not observer
    assert build_console(["--operator", "zerrius"])._session is None  # no model, no session, no observer


# --- isolation ------------------------------------------------------------------------------------


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_shadow_capture_depends_only_on_the_standard_library_and_the_toolcall_type() -> None:
    imports = imported_modules(SRC / "shadow_capture.py")
    assert imports == {"__future__", "json", "os", "re", "uuid", "collections.abc", "datetime", "pathlib", "typing", "zomah.model_runtime"}
    assert "zomah.shadow_capture" not in imported_modules(SRC / "worker_session.py")


def test_no_jev_or_laya_anywhere_in_zomah() -> None:
    for path in SRC.rglob("*.py"):
        for module in imported_modules(path):
            root = module.split(".")[0].casefold()
            assert root not in {"jev", "laya", "typesafe_sdk"}, (path, module)


def test_importing_shadow_capture_loads_no_capability_runtime_tracing_or_model_client() -> None:
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, zomah.shadow_capture; "
         "print(sorted(m for m in sys.modules if m.startswith('zomah')))"],
        capture_output=True, text=True, timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout.replace("'", '"')) == ["zomah", "zomah.attachments", "zomah.model_runtime", "zomah.shadow_capture"]

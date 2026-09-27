"""Per-turn aggregate tool-result budget in WorkerSession."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from zomah.access import ReadScope
from zomah.capability_registry import default_capability_registry
from zomah.console.app import build_console
from zomah.model_runtime import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ModelRuntimeError,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)
from zomah.model_tools import build_session_tools
from zomah.tracing import TraceOutcome, TraceStore
from zomah.worker_session import (
    DEFAULT_TOOL_RESULT_BUDGET_CHARS,
    MAX_TOOL_CALLS_PER_RESPONSE,
    MAX_TOOL_ROUNDS,
    WorkerSession,
)

BUDGET_MESSAGE = "Tool results exceeded this session's per-turn budget; the turn was stopped."
TOOL = ToolDefinition(name="fetch", description="Return a sized result.", parameters={"type": "object"})


class ScriptedRuntime:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class SizedExecutor:
    """Returns ``arguments["size"]`` characters, or a fixed string."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    async def execute(self, call: ToolCall) -> str:
        self.calls.append(call)
        return "r" * int(call.arguments["size"])


def fetch(call_id: str, size: int, name: str = "fetch") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"size": size})


def tools_reply(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(message=AssistantMessage("", calls))


def final(text: str = "done") -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text))


def session(runtime: ScriptedRuntime, executor: Any = None, **kwargs: Any) -> WorkerSession:
    return WorkerSession(
        runtime,
        worker="elyria",
        model="m",
        tools=(TOOL,),
        tool_executor=executor or SizedExecutor(),
        **kwargs,
    )


def send(worker: WorkerSession, text: str = "go") -> Any:
    return asyncio.run(worker.send(text))


def results(request: ModelRequest) -> list[ToolResultMessage]:
    return [m for m in request.messages if isinstance(m, ToolResultMessage)]


def test_default_and_custom_budget() -> None:
    assert DEFAULT_TOOL_RESULT_BUDGET_CHARS == 32768
    assert session(ScriptedRuntime()).tool_result_budget_chars == 32768
    assert session(ScriptedRuntime(), tool_result_budget_chars=500).tool_result_budget_chars == 500
    # A session without tools still has a (unused) budget.
    plain = WorkerSession(ScriptedRuntime(), worker="w", model="m")
    assert plain.tool_result_budget_chars == 32768


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_non_positive_or_non_integer_budget_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="positive number of characters"):
        session(ScriptedRuntime(), tool_result_budget_chars=value)


def test_result_under_budget_passes_unchanged() -> None:
    runtime = ScriptedRuntime(tools_reply(fetch("c1", 40)), final())
    worker = session(runtime, tool_result_budget_chars=100)
    send(worker)
    assert results(runtime.requests[1]) == [ToolResultMessage(call_id="c1", content="r" * 40)]


def test_results_exactly_filling_the_budget_succeed() -> None:
    runtime = ScriptedRuntime(tools_reply(fetch("c1", 60), fetch("c2", 40)), final())
    worker = session(runtime, tool_result_budget_chars=100)
    assert send(worker).assistant.text == "done"
    assert [len(m.content) for m in results(runtime.requests[1])] == [60, 40]


def test_one_character_over_stops_turn_without_partial_result_or_completion() -> None:
    executor = SizedExecutor()
    runtime = ScriptedRuntime(tools_reply(fetch("c1", 60), fetch("c2", 41)), final())
    worker = session(runtime, executor, tool_result_budget_chars=100)

    with pytest.raises(ModelRuntimeError) as raised:
        send(worker)
    assert str(raised.value) == BUDGET_MESSAGE
    assert raised.value.retryable is False
    assert len(executor.calls) == 2  # the second tool ran; its result was refused
    assert len(runtime.requests) == 1  # no further completion
    assert worker.history == ()
    assert worker.busy is False


def test_counter_is_shared_across_rounds_within_a_turn() -> None:
    runtime = ScriptedRuntime(
        tools_reply(fetch("c1", 50)),
        tools_reply(fetch("c2", 30)),
        tools_reply(fetch("c3", 21)),
        final(),
    )
    worker = session(runtime, tool_result_budget_chars=100)
    with pytest.raises(ModelRuntimeError, match="per-turn budget"):
        send(worker)
    assert len(runtime.requests) == 3
    assert [m.call_id for m in results(runtime.requests[2])] == ["c1", "c2"]


def test_counter_resets_for_each_operator_turn_and_prior_history_is_kept() -> None:
    runtime = ScriptedRuntime(
        tools_reply(fetch("a", 90)),
        final("first"),
        tools_reply(fetch("b", 90)),
        final("second"),
        tools_reply(fetch("c", 101)),
    )
    worker = session(runtime, tool_result_budget_chars=100)
    send(worker, "one")
    send(worker, "two")  # 90 more is fine: a new turn starts at zero
    committed = worker.history
    assert [m.role for m in committed] == ["user", "assistant", "tool", "assistant"] * 2

    with pytest.raises(ModelRuntimeError, match="per-turn budget"):
        send(worker, "three")
    assert worker.history == committed
    assert UserMessage("three") not in worker.history


ERROR_ENVELOPE = json.dumps({"ok": False, "error": {"code": "x", "message": "e" * 50}})


class ErrorExecutor:
    async def execute(self, call: ToolCall) -> str:
        return ERROR_ENVELOPE


def test_error_envelopes_and_unavailable_tool_results_count() -> None:
    unavailable = ToolCall(id="u1", name="not_a_session_tool", arguments={})
    probe_runtime = ScriptedRuntime(tools_reply(unavailable), final())
    send(session(probe_runtime))
    unavailable_len = len(results(probe_runtime.requests[1])[0].content)
    both = len(ERROR_ENVELOPE) + unavailable_len

    def run_with(budget: int) -> Any:
        runtime = ScriptedRuntime(tools_reply(fetch("c1", 0), unavailable), final())
        return send(session(runtime, ErrorExecutor(), tool_result_budget_chars=budget))

    assert run_with(both).assistant.text == "done"
    with pytest.raises(ModelRuntimeError, match="per-turn budget"):
        run_with(both - 1)


def test_existing_loop_bounds_still_apply_with_a_large_budget() -> None:
    endless = ScriptedRuntime(*(tools_reply(fetch(f"c{i}", 1)) for i in range(MAX_TOOL_ROUNDS + 1)))
    with pytest.raises(ModelRuntimeError, match="kept requesting tools"):
        send(session(endless, tool_result_budget_chars=10**9))

    burst = ScriptedRuntime(
        tools_reply(*(fetch(f"c{i}", 1) for i in range(MAX_TOOL_CALLS_PER_RESPONSE + 1)))
    )
    with pytest.raises(ModelRuntimeError, match="too many tools at once"):
        send(session(burst, tool_result_budget_chars=10**9))


def test_read_capability_is_traced_even_when_its_result_exceeds_budget(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "notes.txt"
    target.write_text("x" * 400 + "\n", encoding="utf-8")
    trace_store = TraceStore(tmp_path / "trace.db")
    tools, executor = build_session_tools(
        default_capability_registry(),
        worker="elyria",
        trace_store=trace_store,
        allowed_ids=("read_file",),
        dependencies={"read_file": {"scope": ReadScope.from_paths([root])}},
    )
    runtime = ScriptedRuntime(
        tools_reply(ToolCall(id="r1", name="read_file", arguments={"path": str(target)})),
        final(),
    )
    worker = WorkerSession(
        runtime,
        worker="elyria",
        model="m",
        tools=tools,
        tool_executor=executor,
        tool_result_budget_chars=256,
    )
    with pytest.raises(ModelRuntimeError, match="per-turn budget"):
        asyncio.run(worker.send("read notes"))
    (record,) = trace_store.recent()
    assert (record.worker, record.capability, record.outcome, record.target) == (
        "elyria",
        "read_file",
        TraceOutcome.SUCCEEDED,
        str(target),
    )
    assert len(runtime.requests) == 1
    assert worker.history == ()


def test_cli_budget_default_override_and_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    base = ["--operator", "zerrius"]

    default = build_console([*base, "--model", "m"])._session
    assert default is not None and default.tool_result_budget_chars == 32768

    custom = build_console([*base, "--model", "m", "--tool-result-budget-chars", "65536"])._session
    assert custom is not None and custom.tool_result_budget_chars == 65536

    for args, message in (
        (["--model", "m", "--tool-result-budget-chars", "0"], "must be positive"),
        (["--model", "m", "--tool-result-budget-chars", "-5"], "must be positive"),
        (["--model", "m", "--tool-result-budget-chars", "lots"], "invalid int value"),
        (["--tool-result-budget-chars", "4096"], "--tool-result-budget-chars requires --model"),
    ):
        with pytest.raises(SystemExit):
            build_console([*base, *args])
        assert message in capsys.readouterr().err

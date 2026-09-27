"""WorkerSession native tool loop with a scripted runtime and fake executor."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from zomah.model_runtime import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ModelRuntimeError,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)
from zomah.worker_session import (
    MAX_TOOL_CALLS_PER_RESPONSE,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TOOL_ROUNDS,
    WorkerSession,
)

PROJECT_TOOL = ToolDefinition(
    name="get_project_state",
    description="Retrieve the canonical current state for one project.",
    parameters={"type": "object", "properties": {"project_id": {"type": "string"}}},
)


class ScriptedRuntime:
    def __init__(self, *outcomes: ModelResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingExecutor:
    def __init__(self, fault: Exception | None = None, result: Any = None) -> None:
        self.calls: list[ToolCall] = []
        self.fault = fault
        self.result = result
        self.gate: asyncio.Event | None = None

    async def execute(self, call: ToolCall) -> str:
        self.calls.append(call)
        if self.gate is not None:
            await self.gate.wait()
        if self.fault is not None:
            raise self.fault
        if self.result is not None:
            return self.result
        return json.dumps({"ok": True, "result": {"echo": call.arguments}})


def call(call_id: str, name: str = "get_project_state", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments or {"project_id": "zomah"})


def tools_reply(*calls: ToolCall, text: str = "", usage: TokenUsage | None = None) -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text, calls), usage=usage)


def final(text: str, usage: TokenUsage | None = None) -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text), usage=usage)


def session(runtime: ScriptedRuntime, executor: Any = None, **kwargs: Any) -> WorkerSession:
    executor = executor if executor is not None else RecordingExecutor()
    return WorkerSession(
        runtime,
        worker="elyria",
        model="qwen/qwen3.5-9b",
        tools=(PROJECT_TOOL,),
        tool_executor=executor,
        **kwargs,
    )


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_session_advertises_exactly_its_tools() -> None:
    runtime = ScriptedRuntime(final("hi"))
    worker = session(runtime)
    assert worker.tools == (PROJECT_TOOL,)
    run(worker.send("hello"))
    assert runtime.requests[0].tools == (PROJECT_TOOL,)


def test_session_without_tools_sends_no_tools() -> None:
    runtime = ScriptedRuntime(final("hi"))
    worker = WorkerSession(runtime, worker="elyria", model="m")
    run(worker.send("hello"))
    assert worker.tools == ()
    assert runtime.requests[0].tools == ()


def test_tool_call_runs_and_result_reaches_second_completion() -> None:
    executor = RecordingExecutor()
    first_call = call("call_1")
    runtime = ScriptedRuntime(tools_reply(first_call, text="Checking."), final("Phase is X."))
    worker = session(runtime, executor)

    result = run(worker.send("What phase is zomah in?"))

    assert executor.calls == [first_call]
    expected_result = ToolResultMessage(
        call_id="call_1", content='{"ok": true, "result": {"echo": {"project_id": "zomah"}}}'
    )
    assert runtime.requests[1].messages == (
        UserMessage("What phase is zomah in?"),
        AssistantMessage("Checking.", (first_call,)),
        expected_result,
    )
    assert runtime.requests[1].tools == (PROJECT_TOOL,)
    assert result.assistant == AssistantMessage("Phase is X.")
    assert worker.history == (
        UserMessage("What phase is zomah in?"),
        AssistantMessage("Checking.", (first_call,)),
        expected_result,
        AssistantMessage("Phase is X."),
    )


def test_next_turn_sends_committed_tool_history() -> None:
    runtime = ScriptedRuntime(tools_reply(call("call_1")), final("One."), final("Two."))
    worker = session(runtime)
    run(worker.send("first"))
    run(worker.send("second"))
    roles = [message.role for message in runtime.requests[2].messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]


def test_multiple_calls_execute_sequentially_in_order() -> None:
    executor = RecordingExecutor()
    calls = (call("call_a", project_id="a"), call("call_b", project_id="b"))
    runtime = ScriptedRuntime(tools_reply(*calls), final("Both."))
    worker = session(runtime, executor)
    run(worker.send("compare"))

    assert executor.calls == list(calls)
    results = runtime.requests[1].messages[2:]
    assert [message.call_id for message in results] == ["call_a", "call_b"]  # type: ignore[union-attr]


def test_tool_outside_session_snapshot_does_not_execute() -> None:
    executor = RecordingExecutor()
    runtime = ScriptedRuntime(
        tools_reply(call("call_1", "read_file", path="/etc/passwd")), final("Sorry.")
    )
    worker = session(runtime, executor)
    run(worker.send("read a file"))

    assert executor.calls == []
    tool_result = runtime.requests[1].messages[-1]
    assert isinstance(tool_result, ToolResultMessage)
    assert json.loads(tool_result.content) == {
        "ok": False,
        "error": {
            "code": "tool_not_available",
            "message": "Tool is not available in this session: read_file",
            "retryable": False,
        },
    }
    assert worker.history[-1] == AssistantMessage("Sorry.")


def test_toolless_session_answers_hallucinated_calls_without_executing() -> None:
    runtime = ScriptedRuntime(tools_reply(call("call_1")), final("OK."))
    worker = WorkerSession(runtime, worker="elyria", model="m")
    run(worker.send("hi"))
    assert "tool_not_available" in runtime.requests[1].messages[-1].content  # type: ignore[union-attr]


def test_tool_round_limit_stops_an_endless_loop() -> None:
    executor = RecordingExecutor()
    runtime = ScriptedRuntime(
        *(tools_reply(call(f"call_{i}")) for i in range(MAX_TOOL_ROUNDS + 1))
    )
    worker = session(runtime, executor)
    with pytest.raises(ModelRuntimeError, match="kept requesting tools") as raised:
        run(worker.send("loop"))
    assert raised.value.retryable is False
    assert len(runtime.requests) == MAX_TOOL_ROUNDS + 1
    assert len(executor.calls) == MAX_TOOL_ROUNDS
    assert worker.history == ()
    assert worker.busy is False


def test_too_many_calls_in_one_response_execute_nothing() -> None:
    executor = RecordingExecutor()
    calls = [call(f"call_{i}") for i in range(MAX_TOOL_CALLS_PER_RESPONSE + 1)]
    worker = session(ScriptedRuntime(tools_reply(*calls)), executor)
    with pytest.raises(ModelRuntimeError, match="too many tools at once"):
        run(worker.send("burst"))
    assert executor.calls == []
    assert worker.history == ()


def test_too_many_calls_in_one_turn_stop_before_executing_the_excess() -> None:
    executor = RecordingExecutor()
    per_round = 3
    rounds = [
        tools_reply(*(call(f"call_{r}_{i}") for i in range(per_round))) for r in range(3)
    ]
    worker = session(ScriptedRuntime(*rounds), executor)
    with pytest.raises(ModelRuntimeError, match="too many tools in one turn"):
        run(worker.send("many"))
    assert per_round * 3 > MAX_TOOL_CALLS_PER_TURN >= per_round * 2
    assert len(executor.calls) == per_round * 2
    assert worker.history == ()


def test_runtime_failure_after_a_tool_ran_commits_nothing() -> None:
    executor = RecordingExecutor()
    runtime = ScriptedRuntime(
        final("Earlier.", TokenUsage(10, 2)),
        tools_reply(call("call_1")),
        ModelRuntimeError("LM Studio did not respond in time.", retryable=True),
        final("Later."),
    )
    worker = session(runtime, executor)
    run(worker.send("earlier"))
    before = worker.history

    with pytest.raises(ModelRuntimeError, match="did not respond in time"):
        run(worker.send("check state"))
    assert len(executor.calls) == 1  # the READ tool really ran
    assert worker.history == before
    assert worker.last_usage == TokenUsage(10, 2)

    run(worker.send("again"))
    assert [m.role for m in runtime.requests[3].messages] == ["user", "assistant", "user"]


@pytest.mark.parametrize(
    "executor",
    [RecordingExecutor(fault=RuntimeError("secret trace detail")), RecordingExecutor(result=5)],
    ids=["raises", "non-string"],
)
def test_executor_fault_stops_turn_safely(executor: RecordingExecutor) -> None:
    worker = session(ScriptedRuntime(tools_reply(call("call_1"))), executor)
    with pytest.raises(ModelRuntimeError) as raised:
        run(worker.send("check"))
    assert str(raised.value) == "A tool failed unexpectedly; the turn was stopped."
    assert worker.history == ()


def test_context_used_comes_from_the_final_completion_only() -> None:
    runtime = ScriptedRuntime(
        tools_reply(call("call_1"), usage=TokenUsage(100, 10)),
        final("Done.", TokenUsage(150, 20)),
    )
    worker = session(runtime, context_limit=16384)
    result = run(worker.send("check"))
    assert result.usage == TokenUsage(150, 20)
    assert worker.last_usage == TokenUsage(150, 20)
    assert worker.context_used == 170


def test_cancellation_during_tool_execution_commits_nothing() -> None:
    async def scenario() -> None:
        executor = RecordingExecutor()
        executor.gate = asyncio.Event()
        worker = session(ScriptedRuntime(tools_reply(call("call_1")), final("x")), executor)
        task = asyncio.create_task(worker.send("check"))
        while not executor.calls:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker.history == ()
        assert worker.busy is False

    run(scenario())


def test_tool_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="needs a tool executor"):
        WorkerSession(ScriptedRuntime(), worker="w", model="m", tools=(PROJECT_TOOL,))
    with pytest.raises(ValueError, match="unique"):
        WorkerSession(
            ScriptedRuntime(),
            worker="w",
            model="m",
            tools=(PROJECT_TOOL, PROJECT_TOOL),
            tool_executor=RecordingExecutor(),
        )

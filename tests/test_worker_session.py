"""WorkerSession and ModelRuntime contracts against a scripted fake runtime."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

import zomah.worker_session as worker_session_module
from zomah.attachments import ImageAttachment
from zomah.model_runtime import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ModelRuntimeError,
    SystemMessage,
    TokenUsage,
    UserMessage,
)
from zomah.worker_session import SessionBusyError, TurnResult, WorkerSession

PNG = ImageAttachment(mime_type="image/png", data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


class ScriptedRuntime:
    """Fake ModelRuntime: records requests and replays scripted outcomes."""

    def __init__(self, *outcomes: ModelResponse | Exception) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[ModelRequest] = []
        self.release: asyncio.Event | None = None
        self.entered: asyncio.Event | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def reply(text: str, usage: TokenUsage | None = None) -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text), usage=usage)


def session(runtime: ScriptedRuntime, **kwargs: object) -> WorkerSession:
    return WorkerSession(runtime, worker="elyria", model="test-model", **kwargs)


def run(coroutine):
    return asyncio.run(coroutine)


def test_system_message_precedes_user_history() -> None:
    runtime = ScriptedRuntime(reply("hi"))
    worker = session(runtime, system_prompt="You are Elyria.")
    run(worker.send("hello"))

    (request,) = runtime.requests
    assert request.model == "test-model"
    assert request.messages == (SystemMessage("You are Elyria."), UserMessage("hello"))
    assert worker.history == (UserMessage("hello"), AssistantMessage("hi"))


def test_without_system_prompt_only_the_user_turn_is_sent() -> None:
    runtime = ScriptedRuntime(reply("hi"))
    worker = session(runtime)
    run(worker.send("hello"))
    assert runtime.requests[0].messages == (UserMessage("hello"),)
    assert worker.system_message is None


def test_attachments_pass_through_as_the_same_objects() -> None:
    runtime = ScriptedRuntime(reply("I see a PNG."))
    worker = session(runtime)
    result = run(worker.send("what is this?", [PNG]))

    sent = runtime.requests[0].messages[-1]
    assert isinstance(sent, UserMessage)
    assert sent.attachments == (PNG,)
    assert sent.attachments[0] is PNG
    assert result.user.attachments[0] is PNG
    assert worker.history[0].attachments[0] is PNG  # type: ignore[union-attr]


def test_image_only_turn_is_allowed_and_empty_turn_is_rejected() -> None:
    runtime = ScriptedRuntime(reply("ok"))
    worker = session(runtime)
    run(worker.send("", [PNG]))
    with pytest.raises(ValueError):
        run(worker.send("   "))
    assert len(runtime.requests) == 1


def test_success_commits_user_then_assistant_and_returns_result() -> None:
    usage = TokenUsage(input_tokens=12, output_tokens=5)
    runtime = ScriptedRuntime(reply("hi there", usage))
    worker = session(runtime)
    result = run(worker.send("hello"))

    assert result == TurnResult(
        user=UserMessage("hello"), assistant=AssistantMessage("hi there"), usage=usage
    )
    assert [message.role for message in worker.history] == ["user", "assistant"]
    assert worker.history == (UserMessage("hello"), AssistantMessage("hi there"))


def test_second_turn_receives_committed_history() -> None:
    runtime = ScriptedRuntime(reply("one"), reply("two"))
    worker = session(runtime, system_prompt="sys")
    run(worker.send("first"))
    run(worker.send("second"))

    assert runtime.requests[1].messages == (
        SystemMessage("sys"),
        UserMessage("first"),
        AssistantMessage("one"),
        UserMessage("second"),
    )
    assert worker.history == (
        UserMessage("first"),
        AssistantMessage("one"),
        UserMessage("second"),
        AssistantMessage("two"),
    )


def test_reported_usage_is_retained_and_context_used_derived_from_it() -> None:
    runtime = ScriptedRuntime(reply("a", TokenUsage(input_tokens=100, output_tokens=20)))
    worker = session(runtime, context_limit=32768)
    run(worker.send("hello"))

    assert worker.last_usage == TokenUsage(input_tokens=100, output_tokens=20)
    assert worker.context_used == 120
    assert worker.context_limit == 32768


def test_absent_or_partial_usage_is_unavailable() -> None:
    runtime = ScriptedRuntime(
        reply("a", TokenUsage(input_tokens=100, output_tokens=20)),
        reply("b"),
        reply("c", TokenUsage(input_tokens=150)),
    )
    worker = session(runtime)
    assert (worker.last_usage, worker.context_used) == (None, None)

    run(worker.send("one"))
    assert worker.context_used == 120

    # A later turn without usage must not keep reporting stale figures.
    run(worker.send("two"))
    assert (worker.last_usage, worker.context_used) == (None, None)

    run(worker.send("three"))
    assert worker.last_usage == TokenUsage(input_tokens=150)
    assert worker.context_used is None


def test_context_limit_is_independent_of_usage() -> None:
    runtime = ScriptedRuntime(reply("a", TokenUsage(input_tokens=10, output_tokens=2)))
    unlimited = session(runtime)
    run(unlimited.send("x"))
    assert unlimited.context_limit is None
    assert unlimited.context_used == 12

    configured = session(ScriptedRuntime(), context_limit=8192)
    assert configured.context_limit == 8192
    assert configured.context_used is None

    with pytest.raises(ValueError):
        session(ScriptedRuntime(), context_limit=0)


@pytest.mark.parametrize(
    "failure",
    [
        ModelRuntimeError("The model runtime is unavailable.", retryable=True),
        RuntimeError("connection reset by peer at 127.0.0.1:1234"),
    ],
    ids=["runtime-error", "unexpected-exception"],
)
def test_runtime_failure_commits_nothing(failure: Exception) -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=2)
    runtime = ScriptedRuntime(reply("one", usage), failure, reply("two"))
    worker = session(runtime)
    run(worker.send("first"))
    before = worker.history

    with pytest.raises(ModelRuntimeError) as raised:
        run(worker.send("second", [PNG]))
    assert "127.0.0.1" not in str(raised.value)
    assert worker.history == before
    assert worker.last_usage == usage
    assert worker.busy is False

    # The failed turn never becomes history for the next request.
    run(worker.send("third"))
    assert runtime.requests[2].messages == (
        UserMessage("first"),
        AssistantMessage("one"),
        UserMessage("third"),
    )


def test_unexpected_failure_message_is_safe_and_cause_is_kept() -> None:
    worker = session(ScriptedRuntime(RuntimeError("secret transport detail")))
    with pytest.raises(ModelRuntimeError) as raised:
        run(worker.send("hello"))
    assert str(raised.value) == "The model runtime failed unexpectedly."
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert raised.value.retryable is False


def test_invalid_runtime_response_commits_nothing() -> None:
    worker = session(ScriptedRuntime("not a response"))  # type: ignore[arg-type]
    with pytest.raises(ModelRuntimeError, match="invalid response"):
        run(worker.send("hello"))
    assert worker.history == ()


def test_history_snapshots_cannot_mutate_session() -> None:
    worker = session(ScriptedRuntime(reply("hi")))
    run(worker.send("hello"))
    snapshot = worker.history

    assert isinstance(snapshot, tuple)
    with pytest.raises(AttributeError):
        snapshot.append(UserMessage("injected"))  # type: ignore[attr-defined]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot[0].text = "rewritten"  # type: ignore[misc]
    assert worker.history == (UserMessage("hello"), AssistantMessage("hi"))
    assert worker.history is not snapshot


def test_messages_cannot_be_constructed_with_another_role() -> None:
    with pytest.raises(TypeError):
        UserMessage("hi", role="assistant")  # type: ignore[call-arg]
    assert AssistantMessage("x").role == "assistant"
    with pytest.raises(ValueError):
        SystemMessage("  ")


def test_concurrent_send_is_rejected_without_corrupting_history() -> None:
    async def scenario() -> None:
        runtime = ScriptedRuntime(reply("first reply"))
        runtime.release = asyncio.Event()
        runtime.entered = asyncio.Event()
        worker = session(runtime)

        first = asyncio.create_task(worker.send("first"))
        await runtime.entered.wait()
        assert worker.busy is True
        with pytest.raises(SessionBusyError):
            await worker.send("second")

        runtime.release.set()
        result = await first
        assert result.assistant == AssistantMessage("first reply")
        assert worker.history == (UserMessage("first"), AssistantMessage("first reply"))
        assert len(runtime.requests) == 1
        assert worker.busy is False

    run(scenario())


def test_cancelled_turn_commits_nothing_and_releases_session() -> None:
    async def scenario() -> None:
        runtime = ScriptedRuntime(reply("never"), reply("later"))
        runtime.release = asyncio.Event()
        runtime.entered = asyncio.Event()
        worker = session(runtime)

        task = asyncio.create_task(worker.send("first"))
        await runtime.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker.history == ()
        assert worker.busy is False

    run(scenario())


def test_tool_snapshot_is_empty() -> None:
    worker = session(ScriptedRuntime())
    assert worker.tools == ()


def test_session_layer_imports_no_textual_console_or_capability_boundary() -> None:
    script = """
import asyncio, sys
from zomah.model_runtime import AssistantMessage, ModelResponse
from zomah.worker_session import WorkerSession

class Runtime:
    async def complete(self, request):
        return ModelResponse(message=AssistantMessage("ok"))

session = WorkerSession(Runtime(), worker="elyria", model="m")
asyncio.run(session.send("hello"))
loaded = set(sys.modules)
forbidden = {
    "textual",
    "zomah.console",
    "zomah.capability_registry",
    "zomah.capability_runtime",
    "zomah.capability_invocation",
    "zomah.model_boundary",
    "zomah.user_boundary",
    "zomah.tracing",
}
assert not (loaded & forbidden), sorted(loaded & forbidden)
assert not any(name.startswith("textual.") for name in loaded)
print("ok")
"""
    source_root = str(Path(worker_session_module.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [source_root, os.environ.get("PYTHONPATH")])
        ),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_console_attachment_import_is_the_shared_type() -> None:
    import zomah.attachments
    import zomah.console.attachments

    assert zomah.console.attachments.ImageAttachment is zomah.attachments.ImageAttachment

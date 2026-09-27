"""Operator Console wired to a live WorkerSession, with a gated fake runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Static

from zomah.attachments import ImageAttachment
from zomah.capability_registry import default_capability_registry
from zomah.console import Composer, ConsoleStatus, OperatorConsole
from zomah.console.app import DISCONNECTED_NOTICE, build_console
from zomah.console.status import STATUS_LABELS
from zomah.lmstudio import LMStudioRuntime
from zomah.model_runtime import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ModelRuntimeError,
    TokenUsage,
    UserMessage,
)
from zomah.worker_session import WorkerSession

Scenario = Callable[[OperatorConsole, Pilot], Awaitable[None]]
PNG = ImageAttachment(mime_type="image/png", data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 2040)


class GatedRuntime:
    """Fake ModelRuntime. Holds each call until released when ``gated``."""

    def __init__(self, *outcomes: ModelResponse | Exception, gated: bool = False) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ModelRequest] = []
        self.gate = asyncio.Event() if gated else None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClipboard:
    def __init__(self, image: ImageAttachment) -> None:
        self.image = image

    def read_image(self) -> ImageAttachment:
        return self.image


def reply(text: str, usage: TokenUsage | None = None) -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text), usage=usage)


def make_session(runtime: Any, **kwargs: Any) -> WorkerSession:
    options: dict[str, Any] = {"worker": "elyria", "model": "qwen/qwen3.5-9b"}
    options.update(kwargs)
    return WorkerSession(runtime, **options)


def run_console(scenario: Scenario, **kwargs: Any) -> None:
    async def runner() -> None:
        app = OperatorConsole(**kwargs)
        async with app.run_test(size=(100, 40)) as pilot:
            await scenario(app, pilot)

    asyncio.run(runner())


async def type_and_submit(pilot: Pilot, text: str) -> None:
    composer = pilot.app.query_one(Composer)
    composer.load_text(text)
    await pilot.press("escape", "enter")
    await pilot.pause()


async def settle(pilot: Pilot) -> None:
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def header(app: OperatorConsole) -> dict[str, str]:
    return {
        key: str(app.query_one(f"#field-{key}", Static).content).split(": ", 1)[1]
        for key in STATUS_LABELS
    }


def entries(app: OperatorConsole) -> list[tuple[str, str]]:
    found = []
    for entry in app.query("#transcript .entry").results(Static):
        kind = next(
            (k for k in ("operator", "assistant", "result", "notice") if entry.has_class(k)),
            "other",
        )
        found.append((kind, str(entry.content)))
    return found


def notice(app: OperatorConsole) -> str:
    return str(app.query_one("#transcript .entry", Static).content)


def test_disconnected_console_keeps_local_echo() -> None:
    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert notice(app) == DISCONNECTED_NOTICE
        await type_and_submit(pilot, "hello")
        await settle(pilot)
        assert entries(app)[1:] == [("operator", "operator\nhello")]
        assert header(app)["model"] == "not connected"
        assert header(app)["tools"] == "unavailable"
        assert header(app)["context"] == "unavailable"

    run_console(scenario)


def test_connected_startup_reflects_session_and_preserves_project_fields() -> None:
    session = make_session(GatedRuntime(), context_limit=16384)
    status = ConsoleStatus(active_project_id="zomah", project="ZOMAH")

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert notice(app) == "Model session configured: qwen/qwen3.5-9b. Type / for commands."
        assert header(app) == {
            "project": "ZOMAH (zomah)",
            "state": "not connected",
            "model": "qwen/qwen3.5-9b",
            "tools": "0",
            "context": "unavailable / 16384",
        }
        assert app.status == ConsoleStatus(
            active_project_id="zomah",
            project="ZOMAH",
            model="qwen/qwen3.5-9b",
            tools_available=0,
            context_used=None,
            context_limit=16384,
        )

    run_console(scenario, status=status, worker_session=session)


def test_ordinary_turn_shows_operator_then_pending_then_reply() -> None:
    runtime = GatedRuntime(reply("Hello!", TokenUsage(18, 7)), gated=True)
    session = make_session(runtime, context_limit=16384)
    status = ConsoleStatus(active_project_id="zomah")

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await type_and_submit(pilot, "Say hello.")
        # In flight: operator entry and pending entry are both visible.
        assert entries(app)[1:] == [
            ("operator", "operator\nSay hello."),
            ("assistant", "elyria\nelyria is thinking…"),
        ]
        assert app.query_one(".entry.assistant").has_class("pending")
        assert [r.messages[-1] for r in runtime.requests] == [UserMessage("Say hello.")]
        assert header(app)["context"] == "unavailable / 16384"

        assert runtime.gate is not None
        runtime.gate.set()
        await settle(pilot)
        assert entries(app)[1:] == [
            ("operator", "operator\nSay hello."),
            ("assistant", "elyria\nHello!"),
        ]
        assert not app.query(".pending")
        assert header(app)["context"] == "25 / 16384"
        assert app.status.context_used == 25
        assert app.status.active_project_id == "zomah"
        assert session.history == (UserMessage("Say hello."), AssistantMessage("Hello!"))

    run_console(scenario, status=status, worker_session=session)


def test_assistant_label_comes_from_session_worker() -> None:
    session = make_session(GatedRuntime(reply("Hi.")), worker="nova")

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await type_and_submit(pilot, "hello")
        await settle(pilot)
        assert entries(app)[-1] == ("assistant", "nova\nHi.")

    run_console(scenario, worker_session=session)


def test_second_turn_carries_committed_history_and_status_matches_header() -> None:
    runtime = GatedRuntime(reply("One.", TokenUsage(10, 2)), reply("Two.", TokenUsage(20, 3)))
    session = make_session(runtime, context_limit=16384)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await type_and_submit(pilot, "first")
        await settle(pilot)
        await type_and_submit(pilot, "second")
        await settle(pilot)
        assert runtime.requests[1].messages == (
            UserMessage("first"),
            AssistantMessage("One."),
            UserMessage("second"),
        )
        assert header(app)["context"] == "23 / 16384"
        await type_and_submit(pilot, "/status")
        status_output = entries(app)[-1][1]
        assert "qwen/qwen3.5-9b" in status_output
        assert "23 / 16384" in status_output
        assert len(runtime.requests) == 2

    run_console(scenario, worker_session=session)


def test_busy_rejects_second_message_without_queueing_and_keeps_draft() -> None:
    runtime = GatedRuntime(reply("First."), reply("Later."), gated=True)
    session = make_session(runtime)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        composer = app.query_one(Composer)
        await type_and_submit(pilot, "first")
        await type_and_submit(pilot, "second")

        assert [kind for kind, _ in entries(app)[1:]] == ["operator", "assistant"]
        assert composer.text == "second"
        assert [str(n.message) for n in app._notifications] == [
            "elyria is still answering. Your message was not sent."
        ]
        assert len(runtime.requests) == 1

        # Local commands still work while the turn is in flight.
        await type_and_submit(pilot, "/help")
        assert entries(app)[-1][0] == "result"
        assert len(runtime.requests) == 1

        assert runtime.gate is not None
        runtime.gate.set()
        await settle(pilot)
        assert session.history == (UserMessage("first"), AssistantMessage("First."))

        # The /help above replaced the restored draft; send "second" now.
        await type_and_submit(pilot, "second")
        await settle(pilot)
        assert session.history[-2:] == (UserMessage("second"), AssistantMessage("Later."))
        assert len(runtime.requests) == 2

    run_console(scenario, worker_session=session)


def test_runtime_failure_is_rendered_safely_and_not_committed() -> None:
    runtime = GatedRuntime(
        reply("One.", TokenUsage(10, 2)),
        ModelRuntimeError("LM Studio is not reachable.", retryable=True),
        reply("Three."),
    )
    session = make_session(runtime, context_limit=16384)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await type_and_submit(pilot, "first")
        await settle(pilot)
        await type_and_submit(pilot, "this fails")
        await settle(pilot)

        # The transcript keeps the operator's submission and a safe error...
        assert entries(app)[-2:] == [
            ("operator", "operator\nthis fails"),
            ("assistant", "elyria\nLM Studio is not reachable."),
        ]
        assert app.query(".entry.assistant")[-1].has_class("error")
        # ...but the model conversation never committed the failed turn.
        assert session.history == (UserMessage("first"), AssistantMessage("One."))
        assert header(app)["context"] == "12 / 16384"

        await type_and_submit(pilot, "third")
        await settle(pilot)
        assert runtime.requests[2].messages == (
            UserMessage("first"),
            AssistantMessage("One."),
            UserMessage("third"),
        )

    run_console(scenario, worker_session=session)


class ExplodingSession(WorkerSession):
    async def send(self, text: str, attachments: Any = ()) -> Any:
        raise KeyError("secret internal detail")


def test_unexpected_session_error_is_redacted() -> None:
    session = ExplodingSession(GatedRuntime(), worker="elyria", model="m")

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await type_and_submit(pilot, "hello")
        await settle(pilot)
        assert entries(app)[-1] == ("assistant", "elyria\nThe model turn failed unexpectedly.")
        assert "secret" not in " ".join(text for _, text in entries(app))
        # The console recovered: another turn can start.
        await type_and_submit(pilot, "again")
        await settle(pilot)
        assert entries(app)[-1] == ("assistant", "elyria\nThe model turn failed unexpectedly.")

    run_console(scenario, worker_session=session)


def test_slash_commands_never_reach_the_session() -> None:
    runtime = GatedRuntime()
    session = make_session(runtime)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        for command in ("/help", "/status", "/tools", "/project", "/nope", "/help extra"):
            await type_and_submit(pilot, command)
            await settle(pilot)
        assert runtime.requests == []
        assert session.history == ()
        assert all(kind in {"notice", "operator", "result"} for kind, _ in entries(app))

    run_console(scenario, worker_session=session)


def test_attachments_are_forwarded_unchanged() -> None:
    runtime = GatedRuntime(reply("A PNG."))
    session = make_session(runtime)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("alt+v")
        await settle(pilot)
        await type_and_submit(pilot, "what is this?")
        await settle(pilot)

        sent = runtime.requests[0].messages[-1]
        assert isinstance(sent, UserMessage)
        assert sent.attachments[0] is PNG
        assert entries(app)[1:] == [
            ("operator", "operator\nwhat is this?\nImage · image/png · 2 KiB"),
            ("assistant", "elyria\nA PNG."),
        ]
        assert app.query_one(Composer).attachments == ()

    run_console(scenario, worker_session=session, clipboard=FakeClipboard(PNG))


def test_image_only_message_reaches_the_session() -> None:
    runtime = GatedRuntime(reply("An image."))
    session = make_session(runtime)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        await pilot.press("alt+v")
        await settle(pilot)
        await pilot.press("enter")
        await pilot.pause()
        await settle(pilot)
        assert runtime.requests[0].messages[-1] == UserMessage("", (PNG,))
        assert entries(app)[1] == ("operator", "operator\nImage · image/png · 2 KiB")

    run_console(scenario, worker_session=session, clipboard=FakeClipboard(PNG))


class ToolSnapshotSession(WorkerSession):
    @property
    def tools(self) -> tuple[str, ...]:
        return ("get_project_state", "read_file")


def test_tool_count_is_the_session_snapshot_not_registry_exposure() -> None:
    assert any(d.agent_exposed for d in default_capability_registry().all())

    async def empty(app: OperatorConsole, pilot: Pilot) -> None:
        assert header(app)["tools"] == "0"

    async def two(app: OperatorConsole, pilot: Pilot) -> None:
        assert header(app)["tools"] == "2"

    run_console(empty, worker_session=make_session(GatedRuntime()))
    run_console(
        two,
        worker_session=ToolSnapshotSession(GatedRuntime(), worker="elyria", model="m"),
    )


def test_cli_builds_session_only_when_model_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    connected = build_console(
        [
            "--operator", "zerrius",
            "--model", "qwen/qwen3.5-9b",
            "--context-limit", "16384",
            "--lmstudio-base-url", "http://127.0.0.1:4321/v1",
        ]
    )
    session = connected._session
    assert session is not None
    assert (session.worker, session.model, session.context_limit) == (
        "elyria",
        "qwen/qwen3.5-9b",
        16384,
    )
    assert isinstance(session._runtime, LMStudioRuntime)
    assert session._runtime.endpoint == "http://127.0.0.1:4321/v1/chat/completions"
    assert connected.status.model == "qwen/qwen3.5-9b"

    disconnected = build_console(["--operator", "zerrius"])
    assert disconnected._session is None
    assert disconnected.status.model is None

    for bad in (["--context-limit", "16384"], ["--model", "m", "--context-limit", "0"]):
        with pytest.raises(SystemExit):
            build_console(["--operator", "zerrius", *bad])


class StepRuntime:
    """Replays outcomes; completions listed in ``hold`` wait for ``release``."""

    def __init__(self, *outcomes: ModelResponse, hold: set[int] = frozenset()) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ModelRequest] = []
        self.hold = hold
        self.release = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        index = len(self.requests)
        self.requests.append(request)
        if index in self.hold:
            await self.release.wait()
        return self.outcomes.pop(0)


def tool_stack(tmp_path: Path) -> tuple[Any, Any, Any]:
    from zomah.console.operator import OperatorAccess
    from zomah.model_tools import build_session_tools
    from zomah.state import ProjectState, ProjectStateRepository
    from zomah.tracing import TraceStore

    repository = ProjectStateRepository(tmp_path / "zomah.db")
    repository.initialize()
    repository.create(
        ProjectState(
            id="zomah", name="ZOMAH", phase="worker-session", summary="s",
            current_focus="f", next_action="Expose one tool.", updated_by="zerrius",
        )
    )
    access = OperatorAccess(
        operator_id="zerrius",
        trace_store=TraceStore(tmp_path / "trace.db"),
        project_repository=repository,
    )
    registry = default_capability_registry()
    tools, executor = build_session_tools(
        registry,
        worker="elyria",
        trace_store=access.trace_store,
        dependencies={"get_project_state": {"repository": repository}},
    )
    return registry, access, (tools, executor)


def test_cli_session_exposes_one_tool_backed_by_the_shared_stores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    app = build_console(["--operator", "zerrius", "--model", "qwen/qwen3.5-9b"])
    session = app._session
    assert session is not None
    assert [tool.name for tool in session.tools] == ["get_project_state"]
    assert app.status.tools_available == 1
    executor = session._executor
    assert executor._trace_store is app._operator_access.trace_store
    assert executor._dependencies == {
        "get_project_state": {"repository": app._operator_access.project_repository}
    }
    assert executor._worker == session.worker == "elyria"
    assert app._capabilities is executor._registry

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert header(app)["tools"] == "1"

    async def runner() -> None:
        async with app.run_test(size=(100, 40)) as pilot:
            await scenario(app, pilot)

    asyncio.run(runner())


def test_console_turn_with_tool_call_keeps_pending_until_final_answer(tmp_path: Path) -> None:
    from zomah.model_runtime import ToolCall
    from zomah.tracing import TraceOutcome

    registry, access, (tools, executor) = tool_stack(tmp_path)
    call = ToolCall(id="call_1", name="get_project_state", arguments={"project_id": "zomah"})
    runtime = StepRuntime(
        ModelResponse(message=AssistantMessage("", (call,)), usage=TokenUsage(90, 9)),
        reply("Phase: worker-session. Next: Expose one tool.", TokenUsage(260, 14)),
        hold={1},
    )
    session = make_session(
        runtime, context_limit=16384, tools=tools, tool_executor=executor
    )

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        assert header(app)["tools"] == "1"
        await type_and_submit(pilot, "What phase is zomah in? Use your tool.")
        while len(runtime.requests) < 2:
            await asyncio.sleep(0.01)
        # The tool already ran, but the turn is not finished: still pending.
        (record,) = access.trace_store.recent()
        assert (record.worker, record.capability, record.outcome) == (
            "elyria", "get_project_state", TraceOutcome.SUCCEEDED,
        )
        assert entries(app)[-1] == ("assistant", "elyria\nelyria is thinking…")
        assert header(app)["context"] == "unavailable / 16384"

        runtime.release.set()
        await settle(pilot)
        assert entries(app)[-1] == (
            "assistant", "elyria\nPhase: worker-session. Next: Expose one tool."
        )
        assert header(app)["context"] == "274 / 16384"
        assert [m.role for m in session.history] == ["user", "assistant", "tool", "assistant"]

    run_console(
        scenario, worker_session=session, capabilities=registry, operator_access=access
    )


def test_slash_commands_run_no_model_capability_and_project_uses_human_boundary(
    tmp_path: Path,
) -> None:
    registry, access, (tools, executor) = tool_stack(tmp_path)
    runtime = GatedRuntime()
    session = make_session(runtime, tools=tools, tool_executor=executor)

    async def scenario(app: OperatorConsole, pilot: Pilot) -> None:
        for command in ("/help", "/status", "/tools", "/project"):
            await type_and_submit(pilot, command)
            await settle(pilot)
        assert runtime.requests == []
        assert "ZOMAH (zomah)" in entries(app)[-1][1]
        (record,) = access.trace_store.recent()
        assert (record.worker, record.capability) == ("operator:zerrius", "get_project_state")

    run_console(
        scenario,
        status=ConsoleStatus(active_project_id="zomah"),
        worker_session=session,
        capabilities=registry,
        operator_access=access,
    )

"""Ephemeral worker session: conversation state between interfaces and a model.

``WorkerSession`` coordinates one conversation with an injected
``ModelRuntime``. It holds only current-session state in memory: committed
messages, worker and model identity, an optional configured context limit, the
most recent runtime-reported usage, and the exact session-visible tool
snapshot. It does not plan, reason, retry, summarize, trim, or persist
anything; inference belongs to the runtime and tool choice to the model.

Tool calls are deterministic protocol continuation: when the model answers
with tool calls, each call whose name is in the session snapshot is passed to
the injected ``ToolExecutor``, the results are appended to the candidate
conversation, and the model is asked again, until it gives a final answer.
The loop is bounded by ``MAX_TOOL_ROUNDS`` and the per-response / per-turn
tool-call limits.

Concurrency: one turn at a time. A ``send`` issued while another is in flight
raises ``SessionBusyError`` immediately instead of queueing. The session is
meant for use from a single asyncio event loop.

Failure: a turn is committed only once the model gives a final answer. If any
completion fails, a bound is exceeded, or the turn is cancelled, nothing from
the candidate turn (user message, tool calls, tool results) is committed and
previously reported usage is left unchanged. Tools that already ran are not
undone; this is safe only while session tools are read-only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from zomah.attachments import ImageAttachment
from zomah.model_runtime import (
    AssistantMessage,
    ConversationMessage,
    ModelRequest,
    ModelResponse,
    ModelRuntime,
    ModelRuntimeError,
    SystemMessage,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)

# Model responses carrying tool calls allowed in one operator turn.
MAX_TOOL_ROUNDS = 4
# Tool calls accepted from a single model response, and across one turn.
MAX_TOOL_CALLS_PER_RESPONSE = 4
MAX_TOOL_CALLS_PER_TURN = 8

SessionMessage = UserMessage | AssistantMessage | ToolResultMessage


class ToolExecutor(Protocol):
    async def execute(self, call: ToolCall) -> str:
        """Run one advertised tool call and return its result content.

        Expected failures (bad arguments, missing data) belong in the returned
        content for the model to interpret; raising is reserved for faults.
        """
        ...


class SessionBusyError(RuntimeError):
    """A turn was sent while another turn in the same session was in flight."""

    def __init__(self) -> None:
        super().__init__("The worker is still answering the previous message.")


@dataclass(frozen=True, slots=True)
class TurnResult:
    """The committed user message, final reply, and the final completion's usage."""

    user: UserMessage
    assistant: AssistantMessage
    usage: TokenUsage | None


def _unavailable_tool_result(name: str) -> str:
    shown = name if len(name) <= 64 else name[:64] + "…"
    return json.dumps(
        {
            "ok": False,
            "error": {
                "code": "tool_not_available",
                "message": f"Tool is not available in this session: {shown}",
                "retryable": False,
            },
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


class WorkerSession:
    def __init__(
        self,
        runtime: ModelRuntime,
        *,
        worker: str,
        model: str,
        system_prompt: str | None = None,
        context_limit: int | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        if not worker.strip():
            raise ValueError("worker identity must be non-empty")
        if not model.strip():
            raise ValueError("model identity must be non-empty")
        if context_limit is not None and context_limit <= 0:
            raise ValueError("context limit must be positive")
        tools = tuple(tools)
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("session tool names must be unique")
        if tools and tool_executor is None:
            raise ValueError("a session with tools needs a tool executor")
        self._runtime = runtime
        self._worker = worker
        self._model = model
        self._system = SystemMessage(system_prompt) if system_prompt is not None else None
        self._context_limit = context_limit
        self._tools = tools
        self._tool_names = frozenset(names)
        self._executor = tool_executor
        self._history: list[SessionMessage] = []
        self._usage: TokenUsage | None = None
        self._busy = False

    @property
    def worker(self) -> str:
        return self._worker

    @property
    def model(self) -> str:
        return self._model

    @property
    def system_message(self) -> SystemMessage | None:
        return self._system

    @property
    def context_limit(self) -> int | None:
        """Configured limit; independent of reported usage."""

        return self._context_limit

    @property
    def last_usage(self) -> TokenUsage | None:
        """Usage reported for the final completion of the most recent turn.

        ``None`` before any turn and whenever that completion reported no
        usage: an earlier turn's figures would understate the grown context.
        """

        return self._usage

    @property
    def context_used(self) -> int | None:
        """Context tokens from runtime-reported usage only; never estimated."""

        return None if self._usage is None else self._usage.context_tokens

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        """The exact tools advertised to the model in this session."""

        return self._tools

    @property
    def history(self) -> tuple[SessionMessage, ...]:
        """Committed conversation, oldest first (a snapshot)."""

        return tuple(self._history)

    @property
    def busy(self) -> bool:
        return self._busy

    async def send(
        self, text: str, attachments: Sequence[ImageAttachment] = ()
    ) -> TurnResult:
        """Send one operator turn and commit it with the final reply on success.

        Raises ``SessionBusyError`` if a turn is already in flight,
        ``ValueError`` for an empty turn, and ``ModelRuntimeError`` if a
        completion fails, a tool faults, or a loop bound is exceeded; in every
        failure case nothing from the turn is committed.
        """

        if self._busy:
            raise SessionBusyError()
        user = UserMessage(text=text, attachments=tuple(attachments))
        self._busy = True
        try:
            candidate: list[SessionMessage] = [user]
            rounds = 0
            calls_this_turn = 0
            while True:
                response = await self._complete(candidate)
                message = response.message
                if not message.tool_calls:
                    candidate.append(message)
                    self._history.extend(candidate)
                    self._usage = response.usage
                    return TurnResult(user=user, assistant=message, usage=response.usage)

                rounds += 1
                calls_this_turn += len(message.tool_calls)
                if rounds > MAX_TOOL_ROUNDS:
                    raise ModelRuntimeError(
                        "The model kept requesting tools without answering; "
                        "the turn was stopped."
                    )
                if len(message.tool_calls) > MAX_TOOL_CALLS_PER_RESPONSE:
                    raise ModelRuntimeError(
                        "The model requested too many tools at once; the turn was stopped."
                    )
                if calls_this_turn > MAX_TOOL_CALLS_PER_TURN:
                    raise ModelRuntimeError(
                        "The model requested too many tools in one turn; "
                        "the turn was stopped."
                    )

                candidate.append(message)
                for call in message.tool_calls:
                    content = await self._run_tool(call)
                    candidate.append(ToolResultMessage(call_id=call.id, content=content))
        finally:
            self._busy = False

    async def _complete(self, candidate: Sequence[SessionMessage]) -> ModelResponse:
        prefix: tuple[ConversationMessage, ...] = (self._system,) if self._system else ()
        request = ModelRequest(
            model=self._model,
            messages=(*prefix, *self._history, *candidate),
            tools=self._tools,
        )
        try:
            response = await self._runtime.complete(request)
        except ModelRuntimeError:
            raise
        except Exception as exc:
            raise ModelRuntimeError("The model runtime failed unexpectedly.") from exc
        if not isinstance(response, ModelResponse) or not isinstance(
            response.message, AssistantMessage
        ):
            raise ModelRuntimeError("The model runtime returned an invalid response.")
        return response

    async def _run_tool(self, call: ToolCall) -> str:
        # Never trust the response: only advertised tools may execute.
        if call.name not in self._tool_names or self._executor is None:
            return _unavailable_tool_result(call.name)
        try:
            content = await self._executor.execute(call)
        except Exception as exc:
            raise ModelRuntimeError("A tool failed unexpectedly; the turn was stopped.") from exc
        if not isinstance(content, str):
            raise ModelRuntimeError("A tool failed unexpectedly; the turn was stopped.")
        return content

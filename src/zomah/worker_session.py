"""Ephemeral worker session: conversation state between interfaces and a model.

``WorkerSession`` coordinates one conversation with an injected
``ModelRuntime``. It holds only current-session state in memory: committed
messages, worker and model identity, an optional configured context limit, the
most recent runtime-reported usage, and the session-visible tool snapshot
(empty until tools are introduced). It does not plan, reason, retry,
summarize, trim, or persist anything; inference belongs to the runtime.

Concurrency: one turn at a time. A ``send`` issued while another is in flight
raises ``SessionBusyError`` immediately instead of queueing. The session is
meant for use from a single asyncio event loop.

Failure: a turn is committed only if the runtime returns a reply. If the
runtime fails, neither the candidate user message nor any assistant message is
committed, and previously reported usage is left unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
    UserMessage,
)


class SessionBusyError(RuntimeError):
    """A turn was sent while another turn in the same session was in flight."""

    def __init__(self) -> None:
        super().__init__("The worker is still answering the previous message.")


@dataclass(frozen=True, slots=True)
class TurnResult:
    """The committed user/assistant pair and the usage reported for the turn."""

    user: UserMessage
    assistant: AssistantMessage
    usage: TokenUsage | None


class WorkerSession:
    def __init__(
        self,
        runtime: ModelRuntime,
        *,
        worker: str,
        model: str,
        system_prompt: str | None = None,
        context_limit: int | None = None,
    ) -> None:
        if not worker.strip():
            raise ValueError("worker identity must be non-empty")
        if not model.strip():
            raise ValueError("model identity must be non-empty")
        if context_limit is not None and context_limit <= 0:
            raise ValueError("context limit must be positive")
        self._runtime = runtime
        self._worker = worker
        self._model = model
        self._system = SystemMessage(system_prompt) if system_prompt is not None else None
        self._context_limit = context_limit
        self._history: list[UserMessage | AssistantMessage] = []
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
        """Usage reported for the most recent successful turn.

        ``None`` before any turn and whenever the latest turn reported no
        usage: an earlier turn's figures would understate the grown context.
        """

        return self._usage

    @property
    def context_used(self) -> int | None:
        """Context tokens from runtime-reported usage only; never estimated."""

        return None if self._usage is None else self._usage.context_tokens

    @property
    def tools(self) -> tuple[str, ...]:
        """Session-visible tool snapshot. Always empty until tools are introduced."""

        return ()

    @property
    def history(self) -> tuple[UserMessage | AssistantMessage, ...]:
        """Committed user/assistant messages, oldest first (a snapshot)."""

        return tuple(self._history)

    @property
    def busy(self) -> bool:
        return self._busy

    async def send(
        self, text: str, attachments: Sequence[ImageAttachment] = ()
    ) -> TurnResult:
        """Send one operator turn and commit it with the reply on success.

        Raises ``SessionBusyError`` if a turn is already in flight,
        ``ValueError`` for an empty turn, and ``ModelRuntimeError`` if the
        runtime fails; in every failure case nothing is committed.
        """

        if self._busy:
            raise SessionBusyError()
        user = UserMessage(text=text, attachments=tuple(attachments))
        self._busy = True
        try:
            request = ModelRequest(model=self._model, messages=self._candidate(user))
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
            self._history.extend((user, response.message))
            self._usage = response.usage
            return TurnResult(user=user, assistant=response.message, usage=response.usage)
        finally:
            self._busy = False

    def _candidate(self, user: UserMessage) -> tuple[ConversationMessage, ...]:
        prefix: tuple[ConversationMessage, ...] = (self._system,) if self._system else ()
        return (*prefix, *self._history, user)

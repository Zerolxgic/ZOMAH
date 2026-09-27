"""Provider-neutral contract for one model conversation completion.

A ``ModelRuntime`` is the adapter to an external inference runtime. It owns
inference and any provider protocol details, including how images are encoded
for the wire. ZOMAH hands it an ordered conversation and receives one
assistant reply plus whatever token usage the runtime reports.

Only system, user, and assistant messages exist here; tool calls and tool
results are deliberately absent until they are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from zomah.attachments import ImageAttachment


@dataclass(frozen=True, slots=True)
class SystemMessage:
    text: str
    role: Literal["system"] = field(default="system", init=False)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("system message text must be non-empty")


@dataclass(frozen=True, slots=True)
class UserMessage:
    """An operator turn: text, image attachments, or both."""

    text: str
    attachments: tuple[ImageAttachment, ...] = ()
    role: Literal["user"] = field(default="user", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.attachments, tuple):
            raise TypeError("attachments must be a tuple")
        if not self.text.strip() and not self.attachments:
            raise ValueError("a user message needs text or at least one attachment")


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    text: str
    role: Literal["assistant"] = field(default="assistant", init=False)


ConversationMessage = SystemMessage | UserMessage | AssistantMessage


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts as reported by the runtime. ``None`` means not reported."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and value < 0:
                raise ValueError("token counts must be non-negative")

    @property
    def context_tokens(self) -> int | None:
        """Tokens occupying the context after this completion, if fully reported."""

        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One completion request: the model to use and the ordered conversation."""

    model: str
    messages: tuple[ConversationMessage, ...]

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model identity must be non-empty")
        if not self.messages:
            raise ValueError("a request needs at least one message")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """The assistant reply and any runtime-reported usage (``None`` if absent)."""

    message: AssistantMessage
    usage: TokenUsage | None = None


class ModelRuntimeError(Exception):
    """A completion failed. ``str()`` is safe to show an operator.

    Adapters translate transport and provider failures into this type without
    carrying internal details in the message.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ModelRuntime(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one assistant reply or raise ``ModelRuntimeError``."""
        ...

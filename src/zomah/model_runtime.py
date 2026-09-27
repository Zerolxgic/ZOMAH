"""Provider-neutral contract for one model conversation completion.

A ``ModelRuntime`` is the adapter to an external inference runtime. It owns
inference and any provider protocol details, including how images are encoded
for the wire. ZOMAH hands it an ordered conversation and receives one
assistant reply plus whatever token usage the runtime reports.

Tool calling is native: a request may advertise ``ToolDefinition``s, an
assistant message may carry ``ToolCall``s, and ``ToolResultMessage``s answer
them by call id. These types say nothing about what a tool does or who may
run it; that belongs to whoever executes the calls.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

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


def _require_name(value: str, what: str) -> None:
    if not value or value != value.strip() or any(c.isspace() for c in value):
        raise ValueError(f"{what} must be non-empty without whitespace")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool the model may call: name, description, JSON-schema parameters."""

    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_name(self.name, "tool name")
        if not self.description.strip():
            raise ValueError("tool description must be non-empty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a JSON-schema object")
        object.__setattr__(self, "parameters", copy.deepcopy(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call requested by the model, with parsed JSON-object arguments."""

    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("tool call id must be non-empty")
        if not self.name.strip():
            raise ValueError("tool call name must be non-empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be a JSON object")
        object.__setattr__(self, "arguments", copy.deepcopy(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """Assistant text, and any tool calls the model made instead of answering."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    role: Literal["assistant"] = field(default="assistant", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple")


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """The result of one tool call, associated by the provider's call id."""

    call_id: str
    content: str
    role: Literal["tool"] = field(default="tool", init=False)

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("tool result call id must be non-empty")


ConversationMessage = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


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
    """One completion request: model, ordered conversation, advertised tools."""

    model: str
    messages: tuple[ConversationMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model identity must be non-empty")
        if not self.messages:
            raise ValueError("a request needs at least one message")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """The assistant message and any runtime-reported usage (``None`` if absent).

    If ``message.tool_calls`` is non-empty the model is asking for tools to be
    run rather than giving a final answer.
    """

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

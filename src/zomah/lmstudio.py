"""LM Studio model runtime adapter.

Implements ``ModelRuntime`` over LM Studio's OpenAI-compatible, stateless
``POST {base_url}/chat/completions`` endpoint. ZOMAH's ``WorkerSession`` owns
the conversation and sends the full ordered history on every completion;
LM Studio never holds session state.

Only the wire protocol lives here: message, image, and tool encoding,
response, tool-call, and usage parsing, and translating transport/provider
failures into operator-safe ``ModelRuntimeError``s. Tools are advertised with
the OpenAI-compatible ``tools`` field; the model decides whether to call them
(no ``tool_choice`` is sent). No model discovery, loading, generation
settings, streaming, or retries.
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import json
import socket
import urllib.error
import urllib.request
from typing import Any

from zomah.model_runtime import (
    AssistantMessage,
    ConversationMessage,
    ModelRequest,
    ModelResponse,
    ModelRuntimeError,
    SystemMessage,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# HTTP statuses worth retrying by the caller; other 4xx are request problems.
_RETRYABLE_CLIENT_STATUSES = {408, 429}


def encode_message(message: ConversationMessage) -> dict[str, Any]:
    """Map one provider-neutral message to an OpenAI-compatible chat message."""

    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.text}
    if isinstance(message, AssistantMessage):
        if not message.tool_calls:
            return {"role": "assistant", "content": message.text}
        return {
            "role": "assistant",
            "content": message.text,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ],
        }
    if isinstance(message, ToolResultMessage):
        return {"role": "tool", "tool_call_id": message.call_id, "content": message.content}
    if isinstance(message, UserMessage):
        if not message.attachments:
            return {"role": "user", "content": message.text}
        parts: list[dict[str, Any]] = []
        if message.text.strip():
            parts.append({"type": "text", "text": message.text})
        for attachment in message.attachments:
            encoded = base64.b64encode(attachment.data).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{attachment.mime_type};base64,{encoded}"},
                }
            )
        return {"role": "user", "content": parts}
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def encode_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        },
    }


def encode_request(request: ModelRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [encode_message(message) for message in request.messages],
        "stream": False,
    }
    if request.tools:
        body["tools"] = [encode_tool(tool) for tool in request.tools]
    return body


def _malformed() -> ModelRuntimeError:
    return ModelRuntimeError("LM Studio returned an invalid response.")


def _token_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _invalid_tool_call() -> ModelRuntimeError:
    return ModelRuntimeError("LM Studio returned an invalid tool call.")


def _decode_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw_calls, list):
        raise _invalid_tool_call()
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict) or raw.get("type") != "function":
            raise _invalid_tool_call()
        call_id, function = raw.get("id"), raw.get("function")
        if not isinstance(call_id, str) or not call_id.strip() or not isinstance(function, dict):
            raise _invalid_tool_call()
        name, arguments = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
            raise _invalid_tool_call()
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise _invalid_tool_call() from exc
        if not isinstance(decoded, dict):
            raise _invalid_tool_call()
        calls.append(ToolCall(id=call_id, name=name, arguments=decoded))
    if len({call.id for call in calls}) != len(calls):
        raise _invalid_tool_call()
    return tuple(calls)


def decode_response(payload: Any) -> ModelResponse:
    """Parse the first choice of a chat completion into a ``ModelResponse``."""

    if not isinstance(payload, dict):
        raise _malformed()
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _malformed()
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise _malformed()
    content = message.get("content")
    raw_calls = message.get("tool_calls")
    if raw_calls:
        tool_calls = _decode_tool_calls(raw_calls)
        text = content if isinstance(content, str) else ""
    else:
        tool_calls = ()
        if not isinstance(content, str) or not content.strip():
            raise ModelRuntimeError("The model returned no text reply.")
        text = content

    usage = None
    reported = payload.get("usage")
    if isinstance(reported, dict):
        input_tokens = _token_count(reported.get("prompt_tokens"))
        output_tokens = _token_count(reported.get("completion_tokens"))
        if input_tokens is not None or output_tokens is not None:
            usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return ModelResponse(message=AssistantMessage(text, tool_calls), usage=usage)


def _is_timeout(error: BaseException | None) -> bool:
    return isinstance(error, (TimeoutError, socket.timeout))


class LMStudioRuntime:
    """``ModelRuntime`` for LM Studio's OpenAI-compatible chat completions.

    ``timeout`` bounds each blocking socket operation (connect, send, each
    read) of the single HTTP request. The request runs in a worker thread so
    the event loop stays free. Proxy environment variables are ignored: the
    server is addressed directly.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        api_key: str | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._timeout = timeout
        self._api_key = api_key
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def __repr__(self) -> str:
        return f"LMStudioRuntime(endpoint={self._endpoint!r}, timeout={self._timeout!r})"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        body = json.dumps(encode_request(request)).encode("utf-8")
        raw = await asyncio.to_thread(self._post, body)
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _malformed() from exc
        return decode_response(payload)

    def _post(self, body: bytes) -> bytes:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        http_request = urllib.request.Request(
            self._endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with self._opener.open(http_request, timeout=self._timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                declared = response.headers.get("Content-Length", "")
            # A bounded read() returns a short body instead of raising when the
            # connection drops mid-transfer; that is a transport failure.
            if declared.isdigit() and len(raw) < int(declared) <= MAX_RESPONSE_BYTES:
                raise http.client.IncompleteRead(raw, int(declared) - len(raw))
        except urllib.error.HTTPError as exc:
            exc.close()
            status = exc.code
            if status >= 500:
                raise ModelRuntimeError(
                    f"LM Studio reported a server error (HTTP {status}).", retryable=True
                ) from exc
            raise ModelRuntimeError(
                f"LM Studio rejected the request (HTTP {status}).",
                retryable=status in _RETRYABLE_CLIENT_STATUSES,
            ) from exc
        except urllib.error.URLError as exc:
            if _is_timeout(exc.reason):
                raise ModelRuntimeError(
                    "LM Studio did not respond in time.", retryable=True
                ) from exc
            raise ModelRuntimeError("LM Studio is not reachable.", retryable=True) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ModelRuntimeError(
                "LM Studio did not respond in time.", retryable=True
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise ModelRuntimeError(
                "The connection to LM Studio failed.", retryable=True
            ) from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise _malformed()
        return raw

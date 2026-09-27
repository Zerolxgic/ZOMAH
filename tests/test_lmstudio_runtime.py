"""LMStudioRuntime against a deterministic local fake HTTP server."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import zomah.lmstudio as lmstudio
from zomah.attachments import ImageAttachment
from zomah.lmstudio import LMStudioRuntime
from zomah.model_runtime import (
    AssistantMessage,
    ModelRequest,
    ModelRuntimeError,
    SystemMessage,
    TokenUsage,
    UserMessage,
)
from zomah.worker_session import WorkerSession

PNG = ImageAttachment(mime_type="image/png", data=b"\x89PNG\r\n\x1a\n" + b"\x01" * 8)
JPEG = ImageAttachment(mime_type="image/jpeg", data=b"\xff\xd8\xff\xe0" + b"\x02" * 8)
WEBP = ImageAttachment(mime_type="image/webp", data=b"RIFF\x00\x00\x00\x00WEBP" + b"\x03" * 8)


def completion(content: Any = "Hello from Qwen.", usage: Any = "default") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage == "default":
        payload["usage"] = {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}
    elif usage is not None:
        payload["usage"] = usage
    return payload


@dataclass
class Reply:
    status: int = 200
    body: bytes | dict[str, Any] = field(default_factory=completion)
    delay: float = 0.0
    declared_length: int | None = None  # promise more bytes than sent


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


class FakeLMStudio:
    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.requests: list[Recorded] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                fake.requests.append(
                    Recorded("POST", self.path, dict(self.headers), json.loads(raw))
                )
                reply = fake.replies.pop(0) if fake.replies else Reply()
                if reply.delay:
                    threading.Event().wait(reply.delay)
                body = reply.body if isinstance(reply.body, bytes) else json.dumps(reply.body).encode()
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header(
                        "Content-Length", str(reply.declared_length or len(body))
                    )
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self) -> None:  # noqa: N802
                fake.requests.append(Recorded("GET", self.path, dict(self.headers), {}))
                self.send_response(405)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"


@pytest.fixture
def server() -> Iterator[FakeLMStudio]:
    fake = FakeLMStudio()
    fake.thread.start()
    try:
        yield fake
    finally:
        fake.server.shutdown()
        fake.server.server_close()


def complete(runtime: LMStudioRuntime, request: ModelRequest):
    return asyncio.run(runtime.complete(request))


def request(*messages: Any, model: str = "qwen/qwen3.5-9b") -> ModelRequest:
    return ModelRequest(model=model, messages=messages or (UserMessage("hi"),))


def test_posts_non_streaming_completion_with_model_and_ordered_messages(
    server: FakeLMStudio,
) -> None:
    runtime = LMStudioRuntime(base_url=server.base_url + "/")
    response = complete(
        runtime,
        request(
            SystemMessage("You are Elyria."),
            UserMessage("first"),
            AssistantMessage("reply"),
            UserMessage("second"),
            model="qwen/qwen3.5-9b@q4_k_m",
        ),
    )

    (sent,) = server.requests
    assert (sent.method, sent.path) == ("POST", "/v1/chat/completions")
    assert sent.headers["Content-Type"] == "application/json"
    assert "Authorization" not in sent.headers
    assert sent.body == {
        "model": "qwen/qwen3.5-9b@q4_k_m",
        "messages": [
            {"role": "system", "content": "You are Elyria."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ],
        "stream": False,
    }
    assert response.message == AssistantMessage("Hello from Qwen.")
    assert response.usage == TokenUsage(input_tokens=42, output_tokens=7)


def test_images_become_data_url_parts_only_on_the_wire(server: FakeLMStudio) -> None:
    message = UserMessage("compare these", (PNG, JPEG, WEBP))
    complete(LMStudioRuntime(base_url=server.base_url), request(message))

    content = server.requests[0].body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "compare these"}
    for part, attachment in zip(content[1:], (PNG, JPEG, WEBP), strict=True):
        assert part["type"] == "image_url"
        prefix = f"data:{attachment.mime_type};base64,"
        url = part["image_url"]["url"]
        assert url.startswith(prefix)
        assert base64.b64decode(url[len(prefix):]) == attachment.data
    assert [p["image_url"]["url"].split(";")[0] for p in content[1:]] == [
        "data:image/png",
        "data:image/jpeg",
        "data:image/webp",
    ]
    # The provider-neutral objects are untouched.
    assert message.attachments == (PNG, JPEG, WEBP)
    assert message.attachments[0].data.startswith(b"\x89PNG")


def test_image_only_user_message_has_no_empty_text_part(server: FakeLMStudio) -> None:
    complete(LMStudioRuntime(base_url=server.base_url), request(UserMessage("", (PNG,))))
    content = server.requests[0].body["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url"]


def test_optional_bearer_token_is_sent_when_configured(server: FakeLMStudio) -> None:
    runtime = LMStudioRuntime(base_url=server.base_url, api_key="local-token")
    complete(runtime, request())
    assert server.requests[0].headers["Authorization"] == "Bearer local-token"
    assert "local-token" not in repr(runtime)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, None),
        ({}, None),
        ({"prompt_tokens": 10}, TokenUsage(input_tokens=10)),
        ({"prompt_tokens": 10, "completion_tokens": True}, TokenUsage(input_tokens=10)),
        ({"prompt_tokens": -1, "completion_tokens": "5"}, None),
        ("not-a-dict", None),
    ],
)
def test_usage_is_mapped_only_from_reported_counts(
    server: FakeLMStudio, usage: Any, expected: TokenUsage | None
) -> None:
    server.replies.append(Reply(body=completion(usage=usage)))
    response = complete(LMStudioRuntime(base_url=server.base_url), request())
    assert response.usage == expected


def refused_base_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def assert_safe(error: ModelRuntimeError, *, message: str, retryable: bool) -> None:
    assert str(error) == message
    assert error.retryable is retryable
    for leak in ("127.0.0.1", "Traceback", "Errno", "secret-provider-detail", "/home"):
        assert leak not in str(error)


def test_unreachable_server_is_safe_and_retryable() -> None:
    runtime = LMStudioRuntime(base_url=refused_base_url())
    with pytest.raises(ModelRuntimeError) as raised:
        complete(runtime, request())
    assert_safe(raised.value, message="LM Studio is not reachable.", retryable=True)
    assert raised.value.__cause__ is not None


def test_timeout_is_safe_and_retryable(server: FakeLMStudio) -> None:
    server.replies.append(Reply(delay=2.0))
    runtime = LMStudioRuntime(base_url=server.base_url, timeout=0.3)
    with pytest.raises(ModelRuntimeError) as raised:
        complete(runtime, request())
    assert_safe(raised.value, message="LM Studio did not respond in time.", retryable=True)
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("status", "message", "retryable"),
    [
        (400, "LM Studio rejected the request (HTTP 400).", False),
        (404, "LM Studio rejected the request (HTTP 404).", False),
        (408, "LM Studio rejected the request (HTTP 408).", True),
        (429, "LM Studio rejected the request (HTTP 429).", True),
        (500, "LM Studio reported a server error (HTTP 500).", True),
        (503, "LM Studio reported a server error (HTTP 503).", True),
    ],
)
def test_http_errors_are_normalized_without_body_and_not_retried(
    server: FakeLMStudio, status: int, message: str, retryable: bool
) -> None:
    server.replies.append(
        Reply(status=status, body={"error": {"message": "secret-provider-detail /home/user"}})
    )
    with pytest.raises(ModelRuntimeError) as raised:
        complete(LMStudioRuntime(base_url=server.base_url), request())
    assert_safe(raised.value, message=message, retryable=retryable)
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"not json at all",
        b"\xff\xfe\x00",
        json.dumps([1, 2]).encode(),
        json.dumps({"choices": []}).encode(),
        json.dumps({"choices": ["x"]}).encode(),
        json.dumps({"choices": [{"message": "text"}]}).encode(),
    ],
)
def test_malformed_responses_are_rejected_safely(server: FakeLMStudio, body: bytes) -> None:
    server.replies.append(Reply(body=body))
    with pytest.raises(ModelRuntimeError) as raised:
        complete(LMStudioRuntime(base_url=server.base_url), request())
    assert_safe(raised.value, message="LM Studio returned an invalid response.", retryable=False)


@pytest.mark.parametrize("content", [None, "", "   ", ["part"], 5])
def test_missing_assistant_text_is_rejected(server: FakeLMStudio, content: Any) -> None:
    server.replies.append(Reply(body=completion(content=content)))
    with pytest.raises(ModelRuntimeError) as raised:
        complete(LMStudioRuntime(base_url=server.base_url), request())
    assert_safe(raised.value, message="The model returned no text reply.", retryable=False)


def test_tool_call_response_is_not_treated_as_text(server: FakeLMStudio) -> None:
    body = completion(content="partial text")
    body["choices"][0]["message"]["tool_calls"] = [
        {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    ]
    body["choices"][0]["finish_reason"] = "tool_calls"
    server.replies.append(Reply(body=body))
    with pytest.raises(ModelRuntimeError) as raised:
        complete(LMStudioRuntime(base_url=server.base_url), request())
    assert str(raised.value) == (
        "The model requested a tool call, which ZOMAH does not support yet."
    )
    assert raised.value.retryable is False


def test_truncated_response_is_a_retryable_transport_failure(
    server: FakeLMStudio,
) -> None:
    body = json.dumps(completion()).encode()
    server.replies.append(Reply(body=body[:20], declared_length=len(body)))
    with pytest.raises(ModelRuntimeError) as raised:
        complete(LMStudioRuntime(base_url=server.base_url), request())
    assert_safe(raised.value, message="The connection to LM Studio failed.", retryable=True)
    assert len(server.requests) == 1


def test_oversized_response_is_rejected(
    server: FakeLMStudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lmstudio, "MAX_RESPONSE_BYTES", 64)
    with pytest.raises(ModelRuntimeError, match="invalid response"):
        complete(LMStudioRuntime(base_url=server.base_url), request())


def test_proxy_environment_is_ignored(
    server: FakeLMStudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("http_proxy", refused_base_url())
    monkeypatch.setenv("HTTP_PROXY", refused_base_url())
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    response = complete(LMStudioRuntime(base_url=server.base_url), request())
    assert response.message.text == "Hello from Qwen."


def test_request_runs_off_the_event_loop(server: FakeLMStudio) -> None:
    server.replies.append(Reply(delay=0.5))

    async def scenario() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        task = asyncio.create_task(ticker())
        await LMStudioRuntime(base_url=server.base_url).complete(request())
        task.cancel()
        return ticks

    assert asyncio.run(scenario()) >= 5


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        LMStudioRuntime(base_url="127.0.0.1:1234/v1")
    with pytest.raises(ValueError):
        LMStudioRuntime(timeout=0)
    assert LMStudioRuntime().endpoint == "http://127.0.0.1:1234/v1/chat/completions"


def test_worker_session_runs_unchanged_on_lm_studio(server: FakeLMStudio) -> None:
    server.replies.extend(
        [
            Reply(body=completion("First answer.")),
            Reply(status=503),
            Reply(body=completion("Second answer.", usage=None)),
        ]
    )
    worker = WorkerSession(
        LMStudioRuntime(base_url=server.base_url),
        worker="elyria",
        model="qwen/qwen3.5-9b",
        system_prompt="You are Elyria.",
        context_limit=32768,
    )

    async def scenario() -> None:
        first = await worker.send("hello", [PNG])
        assert first.assistant == AssistantMessage("First answer.")
        assert worker.context_used == 49

        with pytest.raises(ModelRuntimeError) as raised:
            await worker.send("this one fails")
        assert raised.value.retryable is True
        assert worker.history == (UserMessage("hello", (PNG,)), AssistantMessage("First answer."))

        await worker.send("again")
        assert worker.context_used is None
        assert worker.context_limit == 32768

    asyncio.run(scenario())

    roles = [[m["role"] for m in r.body["messages"]] for r in server.requests]
    assert roles == [
        ["system", "user"],
        ["system", "user", "assistant", "user"],
        ["system", "user", "assistant", "user"],
    ]
    # The failed turn never reached a later request.
    assert server.requests[2].body["messages"][-1] == {"role": "user", "content": "again"}
    assert len(server.requests) == 3


def test_adapter_imports_no_textual_console_or_capability_modules() -> None:
    script = """
import sys
import zomah.lmstudio, zomah.worker_session
loaded = set(sys.modules)
forbidden = {
    "textual", "zomah.console", "zomah.capability_registry",
    "zomah.capability_runtime", "zomah.capability_invocation",
    "zomah.model_boundary", "zomah.user_boundary", "zomah.tracing",
}
assert not (loaded & forbidden), sorted(loaded & forbidden)
assert not any(name.startswith("textual.") for name in loaded)
print("ok")
"""
    source_root = str(Path(lmstudio.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")])),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=env
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"

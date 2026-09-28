"""Best-effort JSONL capture of model-requested tool calls.

This module observes tool plans only. It does not execute, approve, reject,
route, or otherwise influence tool execution.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zomah.model_runtime import ToolCall


_REDACTED = "[REDACTED]"
_OMITTED_PAYLOAD = "[OMITTED_PAYLOAD]"

_SECRET_KEY_FRAGMENTS = (
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private_key",
    "access_key",
)

_PAYLOAD_KEYS = frozenset(
    {
        "blob",
        "body",
        "bytes",
        "content",
        "contents",
        "data",
        "file_content",
        "payload",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
)

_ENV_SECRET_PATTERN = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY)"
    r"[A-Z0-9_]*)=([^\s]+)"
)


def _sanitize_text(value: str, *, limit: int = 512) -> str:
    value = _ENV_SECRET_PATTERN.sub(r"\1=[REDACTED]", value)
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub(_REDACTED, value)
    if len(value) > limit:
        return value[:limit] + "…[truncated]"
    return value


def _sanitize_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None:
        normalized = key.casefold().replace("-", "_")
        if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS):
            return _REDACTED
        if normalized in _PAYLOAD_KEYS:
            return _OMITTED_PAYLOAD

    if isinstance(value, Mapping):
        return {
            str(child_key): _sanitize_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


class JsonlToolCallObserver:
    """Append sanitized tool-plan observations to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()

    def observe(
        self,
        *,
        worker: str,
        model: str,
        goal: str,
        call: ToolCall,
    ) -> None:
        record = {
            "event_id": f"zomah-{uuid.uuid4().hex[:16]}",
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "source": "zomah",
            "worker": worker,
            "model": model,
            "goal": _sanitize_text(goal, limit=1024),
            "tool_call_id": call.id,
            "executor": call.name,
            "planned_action": call.name,
            "arguments": _sanitize_value(dict(call.arguments)),
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

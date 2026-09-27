"""T3d: Jev and Laya as alternative implementations of one bounded-judgment role.

Each adapter sends the stored request unchanged (a state and one choice
question) and returns the provider's raw output plus the choice answer. They
never retrieve, never see labels, and never see each other's output. Provider
libraries are imported only by the `create_*` factories, so tests use fakes.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from typing import Any

from embedding_bench.judge_contract import QUESTION_ID

JEV_DEFAULT_MODEL = "jev-latest"
LAYA_DEFAULT_MODEL = "convaiinnovations/laya"


class JevJudge:
    """TypeSafe Jev System One through `typesafe_sdk`, with one native Choice question.

    The `jev` decorator package wraps the same SDK but discards the choice
    probabilities, so the SDK is used directly to keep them.
    """

    name = "jev"

    def __init__(self, client: Any, *, model: str | None = None, timeout: float | None = None) -> None:
        self.client = client
        self.model = model
        self.timeout = timeout

    def judge(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self.client.system_one(
            state=request["state"],
            questions=request["questions"],
            model=self.model,
            timeout=self.timeout,
        )
        raw = _jev_raw(response)
        answers = raw.get("answers") if isinstance(raw, dict) else None
        return {
            "raw": raw,
            "answer": answers.get(QUESTION_ID) if isinstance(answers, dict) else None,
            "diagnostics": {
                "model": getattr(response, "model", None),
                "request_id": _safe_attr(response, "request_id"),
                "usage": raw.get("usage") if isinstance(raw, dict) else None,
            },
        }


class LayaJudge:
    """A stock Laya checkpoint answering the same choice question in one forward pass."""

    name = "laya"

    def __init__(self, agent: Any, *, max_len: int | None = None, head_max_len: int | None = None) -> None:
        self.agent = agent
        self.max_len = max_len
        self.head_max_len = head_max_len

    def judge(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self.agent.system_one(
            request["state"], request["questions"], max_len=self.max_len, head_max_len=self.head_max_len
        )
        raw = json.loads(json.dumps(result, default=float))
        answers = raw.get("answers") if isinstance(raw, dict) else None
        return {
            "raw": raw,
            "answer": answers.get(QUESTION_ID) if isinstance(answers, dict) else None,
            "diagnostics": self.token_budget(request),
        }

    def token_budget(self, request: dict[str, Any]) -> dict[str, Any]:
        """How much of each option and of the state Laya actually read (it truncates silently)."""

        try:
            from laya.common import build_sequence, render_options, serialize_state

            tok, cfg = self.agent.tok, self.agent.cfg
            max_len = self.max_len or cfg.get("max_len", 512)
            head_max_len = self.head_max_len or cfg.get("head_max_len", 192)
            question = self.agent._to_internal(request["questions"][QUESTION_ID])
            labels = list(question["crit"])
            options = render_options(question)
            full = [len(tok(" " + text, add_special_tokens=False)["input_ids"]) for text in options]
            sequence, markers = build_sequence(tok, request["state"], question, max_len, head_max_len)
            ends = [*markers[1:], sequence.index(tok.sep_token_id, markers[-1])]
            kept = [end - start - 1 for start, end in zip(markers, ends)]
            state_full = len(tok(serialize_state(request["state"]), add_special_tokens=False)["input_ids"])
            state_start = ends[-1] + 1
            state_kept = max(0, len(sequence) - state_start - 1)
        except Exception as exc:  # diagnostics must never turn a judgment into a failure
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "available": True,
            "max_len": max_len,
            "head_max_len": head_max_len,
            "sequence_tokens": len(sequence),
            "option_tokens_full": dict(zip(labels, full)),
            "option_tokens_kept": dict(zip(labels, kept)),
            "options_truncated": [label for label, f, k in zip(labels, full, kept) if k < f],
            "state_tokens_full": state_full,
            "state_tokens_kept": state_kept,
        }


# --- factories and environment records (these import the providers) --------------------------------


def create_jev_judge(*, model: str | None, timeout: float | None) -> tuple[JevJudge, dict[str, Any]]:
    from typesafe_sdk import TypeSafeClient

    client = TypeSafeClient()  # reads TYPESAFE_API_KEY; the key is never recorded
    judge = JevJudge(client, model=model, timeout=timeout)
    return judge, {
        "provider": "TypeSafe Jev (System One)",
        "client": "typesafe_sdk.TypeSafeClient.system_one",
        "model_requested": model or f"SDK default ({JEV_DEFAULT_MODEL})",
        "question": "one native Choice question; abstention is the 'none' option",
        "timeout_s": timeout if timeout is not None else "SDK default",
        "retries": "SDK default retry policy",
    }


def create_laya_judge(
    *, model: str, subfolder: str | None, device: str | None, max_len: int | None, head_max_len: int | None
) -> tuple[LayaJudge, dict[str, Any]]:
    import os

    import laya

    local = os.path.isdir(os.path.expanduser(model))
    snapshot_before = None if local else hub_snapshot(model, "rl_agent_config.json")
    agent = laya.load(model, device=device, subfolder=subfolder)
    judge = LayaJudge(agent, max_len=max_len, head_max_len=head_max_len)
    return judge, {
        "provider": "Laya",
        "model": model,
        "subfolder": subfolder,
        "snapshot": hub_snapshot(model, "rl_agent_config.json"),
        "cached_before_load": None if local else snapshot_before is not None,
        "local_directory": local,
        "device": str(getattr(agent, "device", device)),
        "dtype": str(getattr(agent, "dtype", None)),
        "max_len": max_len if max_len is not None else f"checkpoint default ({agent.cfg.get('max_len', 512)})",
        "head_max_len": (
            head_max_len if head_max_len is not None else f"checkpoint default ({agent.cfg.get('head_max_len', 192)})"
        ),
        "call": "Agent.system_one(state, questions) — one stock forward pass, no fine-tuning, no chains",
    }


def environment(packages: tuple[str, ...]) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def hub_snapshot(repo_id: str, filename: str) -> str | None:
    try:
        from pathlib import Path

        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo_id, filename)
    except Exception:
        return None
    if not isinstance(path, str):
        return None
    parts = Path(path).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def _jev_raw(response: Any) -> Any:
    """The provider's own JSON body when available, else the SDK model's JSON form."""

    try:
        return json.loads(response.raw_http_response.content)
    except Exception:
        pass
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return response


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    return value if isinstance(value, (str, int, float)) else None

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
)

from zomah.execution import ScriptRegistry


ScriptName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
ScriptArgumentValue = StrictStr | StrictInt | StrictBool
MAX_STDOUT_BYTES = 16 * 1024
MAX_STDERR_BYTES = 8 * 1024


class RunScriptRequest(BaseModel):
    """Validated model-facing input for a registered script execution."""

    model_config = ConfigDict(extra="forbid")

    script: ScriptName
    arguments: dict[str, ScriptArgumentValue] = Field(default_factory=dict)


class RunScriptResponse(BaseModel):
    """Structured result from one registered script execution."""

    model_config = ConfigDict(extra="forbid")

    script: str
    exit_code: int | None
    timed_out: bool
    duration_ms: int = Field(ge=0)
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool


def run_script(
    request: RunScriptRequest,
    registry: ScriptRegistry,
) -> RunScriptResponse:
    """Execute one pre-registered script without a shell.

    Model input can select only a registered script ID and values allowed by
    that script's explicit argument contract. Executable path, working
    directory, timeout, and environment are all harness-owned.
    """

    registered = registry.resolve(request.script)
    argv = registered.build_argv(request.arguments)
    environment = _minimal_environment()

    started = time.monotonic()
    timed_out = False

    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        process = subprocess.Popen(
            argv,
            cwd=registered.working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )

        try:
            process.wait(timeout=registered.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process)
            process.wait()

        duration_ms = max(0, round((time.monotonic() - started) * 1000))
        stdout, stdout_truncated = _read_bounded_text(stdout_file, MAX_STDOUT_BYTES)
        stderr, stderr_truncated = _read_bounded_text(stderr_file, MAX_STDERR_BYTES)

    return RunScriptResponse(
        script=registered.name,
        exit_code=None if timed_out else process.returncode,
        timed_out=timed_out,
        duration_ms=duration_ms,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def _minimal_environment() -> dict[str, str]:
    """Return a small predictable environment without inherited secrets."""

    return {
        "HOME": str(Path.home()),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_bounded_text(handle: object, max_bytes: int) -> tuple[str, bool]:
    # TemporaryFile has seek/read but its concrete type is intentionally not
    # part of the public typing surface.
    handle.seek(0)  # type: ignore[attr-defined]
    payload = handle.read(max_bytes + 1)  # type: ignore[attr-defined]
    truncated = len(payload) > max_bytes
    payload = payload[:max_bytes]
    return payload.decode("utf-8", errors="replace"), truncated

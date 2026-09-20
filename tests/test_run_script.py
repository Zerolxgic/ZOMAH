from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.capabilities.run_script import RunScriptRequest, run_script
from zomah.execution import (
    InvalidScriptArguments,
    RegisteredScript,
    ScriptArgumentSpec,
    ScriptRegistry,
    ScriptRegistryError,
    UnknownScript,
)


def _make_script(tmp_path: Path, body: str, *, name: str = "script.sh") -> Path:
    path = tmp_path / name
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o700)
    return path


def test_run_registered_script_with_validated_arguments(tmp_path: Path) -> None:
    script = _make_script(
        tmp_path,
        'name="world"\nloud=0\n'
        'while [[ $# -gt 0 ]]; do\n'
        '  case "$1" in\n'
        '    --name) name="$2"; shift 2 ;;\n'
        '    --loud) loud=1; shift ;;\n'
        '  esac\n'
        'done\n'
        'if [[ "$loud" -eq 1 ]]; then name="${name^^}"; fi\n'
        'printf "hello %s\\n" "$name"\n',
    )
    registry = ScriptRegistry(
        (
            RegisteredScript(
                name="greet",
                path=script,
                arguments=(
                    ScriptArgumentSpec(
                        name="name",
                        flag="--name",
                        kind="string",
                        pattern=r"[A-Za-z]{1,20}",
                    ),
                    ScriptArgumentSpec(
                        name="loud", flag="--loud", kind="boolean"
                    ),
                ),
            ),
        )
    )

    response = run_script(
        RunScriptRequest(
            script="greet", arguments={"name": "Elyria", "loud": True}
        ),
        registry,
    )

    assert response.exit_code == 0
    assert response.timed_out is False
    assert response.stdout == "hello ELYRIA\n"
    assert response.stderr == ""
    assert response.stdout_truncated is False


def test_unknown_script_is_rejected(tmp_path: Path) -> None:
    registry = ScriptRegistry()
    with pytest.raises(UnknownScript, match="not registered"):
        run_script(RunScriptRequest(script="missing"), registry)


def test_unknown_argument_is_rejected(tmp_path: Path) -> None:
    script = _make_script(tmp_path, 'printf "ok\\n"\n')
    registry = ScriptRegistry((RegisteredScript(name="safe", path=script),))

    with pytest.raises(InvalidScriptArguments, match="unknown"):
        run_script(
            RunScriptRequest(script="safe", arguments={"surprise": "value"}),
            registry,
        )


def test_missing_required_argument_is_rejected(tmp_path: Path) -> None:
    script = _make_script(tmp_path, 'printf "%s\\n" "$1"\n')
    registry = ScriptRegistry(
        (
            RegisteredScript(
                name="required",
                path=script,
                arguments=(
                    ScriptArgumentSpec(
                        name="mode",
                        flag=None,
                        kind="string",
                        required=True,
                        choices=("status", "summary"),
                    ),
                ),
            ),
        )
    )

    with pytest.raises(InvalidScriptArguments, match="missing required"):
        run_script(RunScriptRequest(script="required"), registry)


def test_string_and_integer_argument_constraints_are_enforced(tmp_path: Path) -> None:
    script = _make_script(tmp_path, 'printf "ok\\n"\n')
    registered = RegisteredScript(
        name="bounded",
        path=script,
        arguments=(
            ScriptArgumentSpec(
                name="mode",
                flag="--mode",
                kind="string",
                choices=("quick", "full"),
            ),
            ScriptArgumentSpec(
                name="count",
                flag="--count",
                kind="integer",
                minimum=1,
                maximum=5,
            ),
        ),
    )
    registry = ScriptRegistry((registered,))

    with pytest.raises(InvalidScriptArguments, match="must be one of"):
        run_script(
            RunScriptRequest(script="bounded", arguments={"mode": "other"}),
            registry,
        )
    with pytest.raises(InvalidScriptArguments, match="between 1 and 5"):
        run_script(
            RunScriptRequest(script="bounded", arguments={"count": 99}),
            registry,
        )


def test_request_does_not_accept_executable_path() -> None:
    with pytest.raises(ValidationError):
        RunScriptRequest.model_validate(
            {"script": "safe", "path": "/bin/bash", "arguments": {}}
        )


def test_registration_rejects_symlink_and_non_executable(tmp_path: Path) -> None:
    script = tmp_path / "script.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n")

    with pytest.raises(ScriptRegistryError, match="not executable"):
        RegisteredScript(name="plain", path=script)

    script.chmod(0o700)
    link = tmp_path / "link.sh"
    link.symlink_to(script)
    with pytest.raises(ScriptRegistryError, match="symlink"):
        RegisteredScript(name="linked", path=link)


def test_timeout_is_enforced(tmp_path: Path) -> None:
    script = _make_script(tmp_path, 'sleep 2\nprintf "too late\\n"\n')
    registry = ScriptRegistry(
        (RegisteredScript(name="slow", path=script, timeout_seconds=0.05),)
    )

    response = run_script(RunScriptRequest(script="slow"), registry)

    assert response.timed_out is True
    assert response.exit_code is None
    assert response.duration_ms < 1500
    assert "too late" not in response.stdout


def test_nonzero_exit_is_reported_with_stderr(tmp_path: Path) -> None:
    script = _make_script(tmp_path, 'printf "problem\\n" >&2\nexit 7\n')
    registry = ScriptRegistry((RegisteredScript(name="fails", path=script),))

    response = run_script(RunScriptRequest(script="fails"), registry)

    assert response.timed_out is False
    assert response.exit_code == 7
    assert response.stderr == "problem\n"


def test_output_is_bounded_and_marked_truncated(tmp_path: Path) -> None:
    script = _make_script(
        tmp_path,
        "python - <<'PY'\nprint('x' * 70000, end='')\nPY\n",
    )
    registry = ScriptRegistry((RegisteredScript(name="chatty", path=script),))

    response = run_script(RunScriptRequest(script="chatty"), registry)

    assert response.exit_code == 0
    assert len(response.stdout.encode("utf-8")) == 64 * 1024
    assert response.stdout_truncated is True


def test_environment_does_not_inherit_arbitrary_parent_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _make_script(tmp_path, 'printf "%s" "${ZOMAH_TEST_SECRET-}"\n')
    monkeypatch.setenv("ZOMAH_TEST_SECRET", "do-not-leak")
    registry = ScriptRegistry((RegisteredScript(name="env", path=script),))

    response = run_script(RunScriptRequest(script="env"), registry)

    assert response.exit_code == 0
    assert response.stdout == ""


def test_registered_working_directory_is_fixed(tmp_path: Path) -> None:
    script_dir = tmp_path / "scripts"
    work_dir = tmp_path / "work"
    script_dir.mkdir()
    work_dir.mkdir()
    script = _make_script(script_dir, 'pwd\n')
    registry = ScriptRegistry(
        (
            RegisteredScript(
                name="pwd",
                path=script,
                working_directory=work_dir,
            ),
        )
    )

    response = run_script(RunScriptRequest(script="pwd"), registry)

    assert response.stdout.strip() == str(work_dir)


def test_sudo_binary_cannot_be_registered() -> None:
    sudo = Path("/usr/bin/sudo")
    if not sudo.exists():
        pytest.skip("sudo is not installed")
    with pytest.raises(ScriptRegistryError, match="sudo"):
        RegisteredScript(name="sudo", path=sudo)

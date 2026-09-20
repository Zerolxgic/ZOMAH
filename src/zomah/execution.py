from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping


SCRIPT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
FLAG_RE = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
MAX_ARGUMENT_LENGTH = 1024
MAX_TIMEOUT_SECONDS = 60.0

ArgumentKind = Literal["string", "integer", "boolean"]
ArgumentValue = str | int | bool


class ScriptRegistryError(ValueError):
    """Base error for invalid or unavailable registered scripts."""


class UnknownScript(ScriptRegistryError):
    """Raised when the model requests a script name that is not registered."""


class InvalidScriptArguments(ScriptRegistryError):
    """Raised when model-provided arguments violate a registered contract."""


@dataclass(frozen=True, slots=True)
class ScriptArgumentSpec:
    """One model-facing argument accepted by a registered script.

    String arguments must be constrained by explicit choices, a full-match
    regular expression, or both. Integer arguments require an inclusive range.
    Boolean arguments become presence/absence flags. These constraints keep the
    model-facing argument surface explicit even though the registered script is
    trusted automation.
    """

    name: str
    flag: str | None
    kind: ArgumentKind
    required: bool = False
    choices: tuple[str, ...] = ()
    pattern: str | None = None
    minimum: int | None = None
    maximum: int | None = None
    max_length: int = MAX_ARGUMENT_LENGTH

    def __post_init__(self) -> None:
        if not SCRIPT_NAME_RE.fullmatch(self.name):
            raise ScriptRegistryError(f"invalid script argument name: {self.name!r}")
        if self.flag is not None and not FLAG_RE.fullmatch(self.flag):
            raise ScriptRegistryError(f"invalid script argument flag: {self.flag!r}")
        if not 1 <= self.max_length <= MAX_ARGUMENT_LENGTH:
            raise ScriptRegistryError(
                f"max_length must be between 1 and {MAX_ARGUMENT_LENGTH}"
            )

        if self.kind == "string":
            if not self.choices and self.pattern is None:
                raise ScriptRegistryError(
                    f"string argument {self.name!r} requires choices or pattern"
                )
            if self.pattern is not None:
                try:
                    re.compile(self.pattern)
                except re.error as exc:
                    raise ScriptRegistryError(
                        f"invalid regex for argument {self.name!r}: {exc}"
                    ) from exc
            if self.minimum is not None or self.maximum is not None:
                raise ScriptRegistryError(
                    f"string argument {self.name!r} may not define integer bounds"
                )

        elif self.kind == "integer":
            if self.minimum is None or self.maximum is None:
                raise ScriptRegistryError(
                    f"integer argument {self.name!r} requires minimum and maximum"
                )
            if self.minimum > self.maximum:
                raise ScriptRegistryError(
                    f"integer argument {self.name!r} has inverted bounds"
                )
            if self.choices or self.pattern is not None:
                raise ScriptRegistryError(
                    f"integer argument {self.name!r} may not define string constraints"
                )

        elif self.kind == "boolean":
            if self.flag is None:
                raise ScriptRegistryError(
                    f"boolean argument {self.name!r} requires a flag"
                )
            if (
                self.choices
                or self.pattern is not None
                or self.minimum is not None
                or self.maximum is not None
            ):
                raise ScriptRegistryError(
                    f"boolean argument {self.name!r} may not define value constraints"
                )

    def render(self, value: ArgumentValue) -> list[str]:
        """Validate one value and render it into argv components."""

        if self.kind == "boolean":
            if type(value) is not bool:
                raise InvalidScriptArguments(
                    f"argument {self.name!r} must be a boolean"
                )
            return [self.flag] if value else []  # type: ignore[list-item]

        if self.kind == "integer":
            if type(value) is not int:
                raise InvalidScriptArguments(
                    f"argument {self.name!r} must be an integer"
                )
            assert self.minimum is not None and self.maximum is not None
            if not self.minimum <= value <= self.maximum:
                raise InvalidScriptArguments(
                    f"argument {self.name!r} must be between "
                    f"{self.minimum} and {self.maximum}"
                )
            rendered = str(value)

        else:
            if type(value) is not str:
                raise InvalidScriptArguments(
                    f"argument {self.name!r} must be a string"
                )
            if len(value) > self.max_length:
                raise InvalidScriptArguments(
                    f"argument {self.name!r} exceeds {self.max_length} characters"
                )
            if self.choices and value not in self.choices:
                raise InvalidScriptArguments(
                    f"argument {self.name!r} must be one of: "
                    + ", ".join(self.choices)
                )
            if self.pattern is not None and re.fullmatch(self.pattern, value) is None:
                raise InvalidScriptArguments(
                    f"argument {self.name!r} does not match its allowed pattern"
                )
            rendered = value

        if self.flag is None:
            return [rendered]
        return [self.flag, rendered]


@dataclass(frozen=True, slots=True)
class RegisteredScript:
    """A trusted executable and the exact model-facing contract around it."""

    name: str
    path: Path
    arguments: tuple[ScriptArgumentSpec, ...] = ()
    timeout_seconds: float = 15.0
    working_directory: Path | None = None

    def __post_init__(self) -> None:
        if not SCRIPT_NAME_RE.fullmatch(self.name):
            raise ScriptRegistryError(f"invalid script name: {self.name!r}")

        raw_path = Path(self.path).expanduser()
        if not raw_path.is_absolute():
            raise ScriptRegistryError("registered script path must be absolute")
        if raw_path.is_symlink():
            raise ScriptRegistryError("registered script path may not be a symlink")
        try:
            canonical_path = raw_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ScriptRegistryError(
                f"registered script does not exist: {raw_path}"
            ) from exc
        if not canonical_path.is_file():
            raise ScriptRegistryError(
                f"registered script is not a regular file: {canonical_path}"
            )
        if not os.access(canonical_path, os.X_OK):
            raise ScriptRegistryError(
                f"registered script is not executable: {canonical_path}"
            )
        if canonical_path.name == "sudo":
            raise ScriptRegistryError("sudo may not be registered as a ZOMAH script")

        if not 0 < self.timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise ScriptRegistryError(
                f"timeout_seconds must be > 0 and <= {MAX_TIMEOUT_SECONDS:g}"
            )

        cwd_raw = (
            canonical_path.parent
            if self.working_directory is None
            else Path(self.working_directory).expanduser()
        )
        if not cwd_raw.is_absolute():
            raise ScriptRegistryError("script working directory must be absolute")
        try:
            canonical_cwd = cwd_raw.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ScriptRegistryError(
                f"script working directory does not exist: {cwd_raw}"
            ) from exc
        if not canonical_cwd.is_dir():
            raise ScriptRegistryError(
                f"script working directory is not a directory: {canonical_cwd}"
            )

        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise ScriptRegistryError(
                f"registered script {self.name!r} has duplicate argument names"
            )
        flags = [argument.flag for argument in self.arguments if argument.flag]
        if len(flags) != len(set(flags)):
            raise ScriptRegistryError(
                f"registered script {self.name!r} has duplicate argument flags"
            )

        object.__setattr__(self, "path", canonical_path)
        object.__setattr__(self, "working_directory", canonical_cwd)

    def build_argv(self, provided: Mapping[str, ArgumentValue]) -> list[str]:
        specs = {argument.name: argument for argument in self.arguments}
        unknown = sorted(set(provided) - set(specs))
        if unknown:
            raise InvalidScriptArguments(
                "unknown script argument(s): " + ", ".join(unknown)
            )

        missing = [
            argument.name
            for argument in self.arguments
            if argument.required and argument.name not in provided
        ]
        if missing:
            raise InvalidScriptArguments(
                "missing required script argument(s): " + ", ".join(missing)
            )

        argv = [str(self.path)]
        for argument in self.arguments:
            if argument.name in provided:
                argv.extend(argument.render(provided[argument.name]))
        return argv


class ScriptRegistry:
    """Explicit registry of trusted scripts available to run_script.

    This is not a generic tool registry. It only maps model-visible script IDs
    to trusted, prevalidated executable contracts created by the harness owner.
    """

    def __init__(self, scripts: tuple[RegisteredScript, ...] = ()) -> None:
        self._scripts: dict[str, RegisteredScript] = {}
        for script in scripts:
            self.register(script)

    def register(self, script: RegisteredScript) -> None:
        if script.name in self._scripts:
            raise ScriptRegistryError(f"script is already registered: {script.name}")
        self._scripts[script.name] = script

    def resolve(self, name: str) -> RegisteredScript:
        try:
            return self._scripts[name]
        except KeyError as exc:
            raise UnknownScript(f"script is not registered: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._scripts))

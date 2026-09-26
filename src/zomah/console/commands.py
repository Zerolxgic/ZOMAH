"""Operator Console command registry.

Console commands describe what the operator can type as ``/name``. This is an
interface registry, separate from ``CapabilityRegistry``: a command may later
invoke a registered capability, but registering a command grants no authority
and executes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConsoleCommand:
    """One slash command offered by the Operator Console."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if (
            not self.name.startswith("/")
            or len(self.name) < 2
            or self.name != self.name.lower()
            or any(character.isspace() for character in self.name)
            or "/" in self.name[1:]
        ):
            raise ValueError(f"invalid console command name: {self.name!r}")
        if not self.description.strip():
            raise ValueError("console command description must be non-empty")


class DuplicateCommandError(ValueError):
    """Raised when the same command name is registered more than once."""


def is_command_prefix(text: str) -> bool:
    """True when ``text`` is a lone, still-being-typed slash-command token."""

    return text.startswith("/") and not any(character.isspace() for character in text)


class CommandRegistry:
    """Small explicit registry of console commands."""

    def __init__(self) -> None:
        self._commands: dict[str, ConsoleCommand] = {}

    def register(self, command: ConsoleCommand) -> None:
        if command.name in self._commands:
            raise DuplicateCommandError(f"command already registered: {command.name}")
        self._commands[command.name] = command

    def all(self) -> tuple[ConsoleCommand, ...]:
        return tuple(sorted(self._commands.values(), key=lambda command: command.name))

    def matching(self, prefix: str) -> tuple[ConsoleCommand, ...]:
        """Commands whose name starts with ``prefix``, alphabetically."""

        return tuple(command for command in self.all() if command.name.startswith(prefix))


def default_command_registry() -> CommandRegistry:
    """Build the explicit initial console command set.

    T0b registers these for discovery only; none of them executes yet.
    """

    registry = CommandRegistry()
    for command in (
        ConsoleCommand("/help", "Show available commands and composer keys."),
        ConsoleCommand("/project", "Show or select the active project."),
        ConsoleCommand("/status", "Show ZOMAH and session status."),
        ConsoleCommand("/tools", "List tools available to the active model."),
    ):
        registry.register(command)
    return registry

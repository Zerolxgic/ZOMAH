"""Operator Console command registry.

Console commands describe what the operator can type as ``/name``. This is an
interface registry, separate from ``CapabilityRegistry``: a command's handler
renders a console view from the ``CommandContext`` it is given, and
registering a command grants no machine authority. A command that needs a
capability goes through the user boundary with the operator access it is
given, like any other operator call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zomah.capability_registry import CapabilityRegistry
from zomah.console.operator import OperatorAccess
from zomah.console.status import ConsoleStatus


StatusUpdate = Callable[[ConsoleStatus], ConsoleStatus]


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Structured console output for one command, rendered as plain text.

    ``status_update`` is an optional pure function the app applies to the
    current session status on the UI loop after rendering; views never
    mutate session state themselves.
    """

    title: str
    lines: tuple[str, ...] = ()
    is_error: bool = False
    status_update: StatusUpdate | None = None


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Inputs a console command may render from.

    ``status`` is a snapshot of session state taken at submission; the
    active canonical project id comes from it. ``operator_access`` is
    ``None`` when the console was started without capability access.
    """

    status: ConsoleStatus
    commands: CommandRegistry
    capabilities: CapabilityRegistry
    operator_access: OperatorAccess | None = None


CommandHandler = Callable[[CommandContext, str], CommandResult]


@dataclass(frozen=True, slots=True)
class ConsoleCommand:
    """One slash command offered by the Operator Console.

    ``handler`` receives the context and the argument text (empty when none
    was given). Arguments are rejected before the handler runs unless
    ``accepts_arguments`` is set. ``background`` handlers perform I/O and are
    run off the UI event loop; the others run inline.
    """

    name: str
    description: str
    handler: CommandHandler | None = None
    accepts_arguments: bool = False
    background: bool = False

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

    def get(self, name: str) -> ConsoleCommand | None:
        return self._commands.get(name)

    def all(self) -> tuple[ConsoleCommand, ...]:
        return tuple(sorted(self._commands.values(), key=lambda command: command.name))

    def matching(self, prefix: str) -> tuple[ConsoleCommand, ...]:
        """Commands whose name starts with ``prefix``, alphabetically."""

        return tuple(command for command in self.all() if command.name.startswith(prefix))

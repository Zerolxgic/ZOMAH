"""ZOMAH Operator Console: the interactive terminal interface into ZOMAH."""

from zomah.console.app import Composer, CommandSuggestions, ConsoleStatus, OperatorConsole
from zomah.console.builtin_commands import default_command_registry
from zomah.console.commands import CommandRegistry, CommandResult, ConsoleCommand

__all__ = [
    "CommandRegistry",
    "CommandResult",
    "CommandSuggestions",
    "Composer",
    "ConsoleCommand",
    "ConsoleStatus",
    "OperatorConsole",
    "default_command_registry",
]

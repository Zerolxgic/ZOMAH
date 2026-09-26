"""Route submitted composer text to operator messages or console commands.

Any submission whose first character is ``/`` is a command submission: its
first whitespace-delimited token is the command name and the rest (stripped)
is its argument text. Unknown commands produce an error result; they never
fall through as ordinary operator messages.
"""

from __future__ import annotations

from dataclasses import dataclass

from zomah.console.commands import (
    CommandContext,
    CommandRegistry,
    CommandResult,
    ConsoleCommand,
)


@dataclass(frozen=True, slots=True)
class OperatorMessage:
    """Ordinary operator text. Echoed locally; nothing is invoked."""

    text: str


@dataclass(frozen=True, slots=True)
class CommandSubmission:
    """A submitted slash command. ``command`` is ``None`` when unregistered."""

    name: str
    arguments: str
    command: ConsoleCommand | None


def parse_submission(
    text: str, registry: CommandRegistry
) -> OperatorMessage | CommandSubmission:
    if not text.startswith("/"):
        return OperatorMessage(text)
    token, *rest = text.split(maxsplit=1)
    return CommandSubmission(
        name=token,
        arguments=rest[0].strip() if rest else "",
        command=registry.get(token),
    )


def run_command(submission: CommandSubmission, context: CommandContext) -> CommandResult:
    command = submission.command
    if command is None:
        return CommandResult(
            title=f"Unknown command: {submission.name}",
            lines=("Type / to browse commands, or use /help.",),
            is_error=True,
        )
    if submission.arguments and not command.accepts_arguments:
        return CommandResult(
            title=f"{command.name} does not take arguments.",
            lines=(f"Usage: {command.name}",),
            is_error=True,
        )
    if command.handler is None:
        return CommandResult(
            title=f"{command.name} has no console view.",
            is_error=True,
        )
    return command.handler(context, submission.arguments)

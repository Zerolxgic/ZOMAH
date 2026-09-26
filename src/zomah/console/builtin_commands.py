"""Built-in Operator Console views: /help, /project, /status, /tools.

Each view renders only from its ``CommandContext``. None of them invokes a
capability handler, queries ProjectState, or reaches a model.
"""

from __future__ import annotations

from zomah.console.commands import (
    CommandContext,
    CommandRegistry,
    CommandResult,
    ConsoleCommand,
)
from zomah.console.status import STATUS_LABELS, status_values

COMPOSER_CONTROLS: tuple[tuple[str, str], ...] = (
    ("Enter", "submit"),
    ("Shift+Enter", "newline"),
    ("Ctrl+J", "newline fallback"),
    ("Ctrl+A", "select all"),
    ("Ctrl+C / Ctrl+V", "copy / paste"),
    ("Ctrl+Z / Ctrl+Y", "undo / redo"),
    ("Esc", "dismiss suggestions"),
    ("Ctrl+Q", "quit"),
)


def _columns(rows: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not rows:
        return ()
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return tuple(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    )


def help_view(context: CommandContext, arguments: str) -> CommandResult:
    commands = [(command.name, command.description) for command in context.commands.all()]
    return CommandResult(
        title="Help",
        lines=(
            "Commands",
            *(f"  {line}" for line in _columns(commands)),
            "",
            "Composer keys",
            *(f"  {line}" for line in _columns(list(COMPOSER_CONTROLS))),
        ),
    )


def status_view(context: CommandContext, arguments: str) -> CommandResult:
    values = status_values(context.status)
    return CommandResult(
        title="Status",
        lines=_columns([(f"{label}:", values[key]) for key, label in STATUS_LABELS.items()]),
    )


def project_view(context: CommandContext, arguments: str) -> CommandResult:
    project = context.status.project
    if project is None:
        return CommandResult(title="Project", lines=("No active project is set.",))
    return CommandResult(title="Project", lines=(f"Active project/folder: {project}",))


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def tools_view(context: CommandContext, arguments: str) -> CommandResult:
    definitions = context.capabilities.all()
    rows = [("ID", "AUTHORITY", "LIFECYCLE", "USER", "AGENT")]
    rows.extend(
        (
            definition.id,
            definition.authority.value,
            definition.lifecycle.value,
            _yes_no(definition.user_exposed),
            _yes_no(definition.agent_exposed),
        )
        for definition in definitions
    )
    status = context.status
    if status.model is None:
        session = "No model session: no tools are available to a model right now."
    elif status.tools_available is None:
        session = f"Model {status.model}: tool availability not reported by the session."
    else:
        session = f"Model {status.model}: {status.tools_available} tools available this session."
    return CommandResult(
        title="Tools",
        lines=(
            f"Capability registry: {len(definitions)} registered",
            *(_columns(rows) if definitions else ("(no capabilities registered)",)),
            "",
            "USER/AGENT are registry exposure policy, not live availability.",
            session,
        ),
    )


def default_command_registry() -> CommandRegistry:
    """Build the explicit initial console command set."""

    registry = CommandRegistry()
    for command in (
        ConsoleCommand("/help", "Show available commands and composer keys.", help_view),
        ConsoleCommand("/project", "Show the active project/folder, if one is set.", project_view),
        ConsoleCommand("/status", "Show ZOMAH and session status.", status_view),
        ConsoleCommand("/tools", "Show registered capabilities and model tool availability.", tools_view),
    ):
        registry.register(command)
    return registry

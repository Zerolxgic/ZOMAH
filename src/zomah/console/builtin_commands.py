"""Built-in Operator Console views: /help, /project, /status, /tools.

Each view renders only from its ``CommandContext``. ``/project`` reads
canonical ProjectState through the user capability boundary; the other views
render supplied state and registry metadata. No view reaches a model.
"""

from __future__ import annotations

from zomah.console.commands import (
    CommandContext,
    CommandRegistry,
    CommandResult,
    ConsoleCommand,
)
from zomah.capabilities import GetProjectStateResponse
from zomah.capability_runtime import CapabilityError
from zomah.console.status import STATUS_LABELS, status_values
from zomah.user_boundary import invoke_registered_user_capability

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
    project_id = context.active_project_id
    if project_id is None:
        return CommandResult(
            title="Project",
            lines=("No active canonical project is configured.",),
        )
    access = context.operator_access
    if access is None:
        return CommandResult(
            title="Project unavailable",
            lines=("This console was started without capability access.",),
            is_error=True,
        )
    envelope = invoke_registered_user_capability(
        context.capabilities,
        "get_project_state",
        {"project_id": project_id},
        operator_id=access.operator_id,
        trace_store=access.trace_store,
        repository=access.project_repository,
    )
    # The envelope carries exactly one of error / result.
    if envelope.error is not None:
        return _capability_error_result(envelope.error)
    return _project_result(GetProjectStateResponse.model_validate(envelope.result))


def _capability_error_result(error: CapabilityError) -> CommandResult:
    lines = [error.message, f"Code: {error.code}"]
    if error.retryable:
        lines.append("This may succeed if retried.")
    return CommandResult(title="Project unavailable", lines=tuple(lines), is_error=True)


def _project_result(response: GetProjectStateResponse) -> CommandResult:
    project = response.project
    window = project.decision_window
    counts = ", ".join(
        f"{count} {status.value}" for status, count in window.by_status.items() if count
    )
    decisions = f"{window.total} total" + (f" ({counts})" if counts else "")
    lines = [
        f"{project.name} ({project.id})",
        *_columns(
            [
                ("Status:", project.status.value),
                ("Phase:", project.phase),
                ("Revision:", str(project.revision)),
                ("Summary:", project.summary),
                ("Focus:", project.current_focus),
                ("Last action:", project.last_action or "none recorded"),
                ("Next action:", project.next_action or "none recorded"),
                ("Blockers:", str(len(project.blockers))),
                ("Open questions:", str(len(project.open_questions))),
                ("Decisions:", decisions),
                ("Updated:", f"{project.updated_at.isoformat()} by {project.updated_by}"),
            ]
        ),
    ]
    if window.truncated:
        lines.append(
            f"Decision history is truncated: this view received {window.returned} of "
            f"{window.total} decisions. Canonical history is complete in storage."
        )
    return CommandResult(title="Project", lines=tuple(lines))


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
        ConsoleCommand(
            "/project",
            "Show canonical state for the active project.",
            project_view,
            background=True,
        ),
        ConsoleCommand("/status", "Show ZOMAH and session status.", status_view),
        ConsoleCommand("/tools", "Show registered capabilities and model tool availability.", tools_view),
    ):
        registry.register(command)
    return registry

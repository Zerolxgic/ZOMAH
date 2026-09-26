from __future__ import annotations

from pydantic import BaseModel

from zomah.capability_registry import (
    CapabilityAuthority,
    CapabilityDefinition,
    CapabilityLifecycle,
    CapabilityRegistry,
    default_capability_registry,
)
from zomah.console.builtin_commands import (
    COMPOSER_CONTROLS,
    default_command_registry,
    help_view,
    project_view,
    status_view,
    tools_view,
)
from zomah.console.commands import CommandContext, CommandRegistry, ConsoleCommand
from zomah.console.routing import (
    CommandSubmission,
    OperatorMessage,
    parse_submission,
    run_command,
)
from zomah.console.status import ConsoleStatus


class _Request(BaseModel):
    value: str


class _Response(BaseModel):
    value: str


def context(
    status: ConsoleStatus | None = None,
    capabilities: CapabilityRegistry | None = None,
) -> CommandContext:
    return CommandContext(
        status=status or ConsoleStatus(),
        commands=default_command_registry(),
        capabilities=capabilities or default_capability_registry(),
    )


def route(text: str, ctx: CommandContext | None = None):
    ctx = ctx or context()
    routed = parse_submission(text, ctx.commands)
    assert isinstance(routed, CommandSubmission)
    return run_command(routed, ctx)


def test_ordinary_text_is_an_operator_message() -> None:
    registry = default_command_registry()
    assert parse_submission("hello /help", registry) == OperatorMessage("hello /help")
    assert parse_submission(" /help", registry) == OperatorMessage(" /help")


def test_command_token_is_parsed_separately_from_arguments() -> None:
    registry = default_command_registry()
    routed = parse_submission("/help  one two\nthree ", registry)
    assert isinstance(routed, CommandSubmission)
    assert routed.name == "/help"
    assert routed.arguments == "one two\nthree"
    assert routed.command is registry.get("/help")

    bare = parse_submission("/status", registry)
    assert isinstance(bare, CommandSubmission)
    assert (bare.name, bare.arguments) == ("/status", "")

    newline_args = parse_submission("/tools\nx", registry)
    assert isinstance(newline_args, CommandSubmission)
    assert (newline_args.name, newline_args.arguments) == ("/tools", "x")


def test_unknown_and_malformed_slash_input_is_an_unknown_command() -> None:
    registry = default_command_registry()
    for text, name in (("/whatever", "/whatever"), ("/", "/"), ("/HELP", "/HELP"), ("/usr/bin x", "/usr/bin")):
        routed = parse_submission(text, registry)
        assert isinstance(routed, CommandSubmission)
        assert routed.name == name
        assert routed.command is None

    result = route("/whatever")
    assert result.is_error
    assert result.title == "Unknown command: /whatever"
    assert "Type / to browse commands, or use /help." in result.lines


def test_arguments_to_argumentless_command_are_rejected_before_handler() -> None:
    calls: list[str] = []

    def handler(ctx: CommandContext, arguments: str):
        calls.append(arguments)
        raise AssertionError("handler must not run")

    registry = CommandRegistry()
    registry.register(ConsoleCommand("/plain", "No arguments.", handler))
    routed = parse_submission("/plain extra", registry)
    assert isinstance(routed, CommandSubmission)
    result = run_command(routed, context())
    assert result.is_error
    assert result.title == "/plain does not take arguments."
    assert result.lines == ("Usage: /plain",)
    assert calls == []

    for name in ("/help", "/project", "/status", "/tools"):
        assert route(f"{name} now").is_error


def test_command_without_handler_reports_no_view() -> None:
    registry = CommandRegistry()
    registry.register(ConsoleCommand("/bare", "Registered without a view."))
    routed = parse_submission("/bare", registry)
    assert isinstance(routed, CommandSubmission)
    result = run_command(routed, context())
    assert result.is_error
    assert result.title == "/bare has no console view."


def test_every_default_command_has_a_view_and_takes_no_arguments() -> None:
    for command in default_command_registry().all():
        assert command.handler is not None
        assert command.accepts_arguments is False


def test_help_lists_commands_alphabetically_and_composer_keys() -> None:
    result = help_view(context(), "")
    lines = result.lines
    command_lines = [line.strip() for line in lines[1:5]]
    assert [line.split()[0] for line in command_lines] == [
        "/help",
        "/project",
        "/status",
        "/tools",
    ]
    for command in default_command_registry().all():
        assert any(
            line.startswith(command.name) and line.endswith(command.description)
            for line in command_lines
        )
    for key, action in COMPOSER_CONTROLS:
        assert any(
            line.strip().startswith(key) and line.endswith(action) for line in lines
        )
    assert {key for key, _ in COMPOSER_CONTROLS} >= {
        "Enter",
        "Shift+Enter",
        "Ctrl+J",
        "Ctrl+A",
        "Ctrl+C / Ctrl+V",
        "Ctrl+Z / Ctrl+Y",
        "Esc",
        "Ctrl+Q",
    }


def test_status_reflects_supplied_status() -> None:
    status = ConsoleStatus(
        project="zomah",
        zomah_state="ready",
        model="local-model",
        tools_available=2,
        context_used=10,
        context_limit=100,
    )
    lines = [" ".join(line.split()) for line in status_view(context(status), "").lines]
    assert lines == [
        "Project: zomah",
        "State: ready",
        "Model: local-model",
        "Tools: 2",
        "Context: 10 / 100",
    ]


def test_status_preserves_placeholders() -> None:
    lines = [" ".join(line.split()) for line in status_view(context(), "").lines]
    assert lines == [
        "Project: not set",
        "State: not connected",
        "Model: not connected",
        "Tools: unavailable",
        "Context: unavailable",
    ]


def test_project_view_without_active_project_id_ignores_header_project() -> None:
    # ConsoleStatus.project is display text, never a canonical ProjectState id.
    for status in (ConsoleStatus(), ConsoleStatus(project="zomah")):
        result = project_view(context(status), "")
        assert result.lines == ("No active canonical project is configured.",)
        assert result.is_error is False


def test_tools_reads_registry_metadata_without_invoking_handlers() -> None:
    calls: list[object] = []

    def handler(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("capability handler must not run")

    registry = CapabilityRegistry()
    registry.register(
        CapabilityDefinition(
            id="probe_capability",
            description="Test-only capability.",
            authority=CapabilityAuthority.CHANGE,
            lifecycle=CapabilityLifecycle.TESTED,
            user_exposed=False,
            agent_exposed=True,
            request_model=_Request,
            response_model=_Response,
            handler=handler,
        )
    )
    lines = tools_view(context(capabilities=registry), "").lines
    rows = [line.split() for line in lines]
    assert ["ID", "AUTHORITY", "LIFECYCLE", "USER", "AGENT"] in rows
    assert ["probe_capability", "CHANGE", "TESTED", "no", "yes"] in rows
    assert calls == []


def test_tools_distinguishes_exposure_from_live_model_availability() -> None:
    no_session = tools_view(context(), "").lines
    assert "Capability registry: 1 registered" in no_session
    assert ["get_project_state", "READ", "VERIFIED", "yes", "yes"] in [
        line.split() for line in no_session
    ]
    assert "USER/AGENT are registry exposure policy, not live availability." in no_session
    assert "No model session: no tools are available to a model right now." in no_session

    unreported = tools_view(context(ConsoleStatus(model="m")), "").lines
    assert "Model m: tool availability not reported by the session." in unreported

    reported = tools_view(context(ConsoleStatus(model="m", tools_available=0)), "").lines
    assert "Model m: 0 tools available this session." in reported


def test_views_handle_empty_registries() -> None:
    empty = CommandContext(
        status=ConsoleStatus(),
        commands=CommandRegistry(),
        capabilities=CapabilityRegistry(),
    )
    assert help_view(empty, "").lines[0] == "Commands"
    tools = tools_view(empty, "").lines
    assert tools[:2] == ("Capability registry: 0 registered", "(no capabilities registered)")

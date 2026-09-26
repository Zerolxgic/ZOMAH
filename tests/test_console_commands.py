from __future__ import annotations

import pytest

from zomah.console.commands import (
    CommandRegistry,
    ConsoleCommand,
    DuplicateCommandError,
    default_command_registry,
    is_command_prefix,
)


def test_default_registry_seeds_console_commands_alphabetically() -> None:
    commands = default_command_registry().all()
    assert [command.name for command in commands] == [
        "/help",
        "/project",
        "/status",
        "/tools",
    ]
    assert all(command.description for command in commands)


def test_matching_is_prefix_only_and_alphabetical() -> None:
    registry = default_command_registry()
    assert [c.name for c in registry.matching("/")] == [
        "/help",
        "/project",
        "/status",
        "/tools",
    ]
    assert [c.name for c in registry.matching("/p")] == ["/project"]
    assert registry.matching("/roj") == ()
    assert registry.matching("/x") == ()


def test_duplicate_command_is_rejected() -> None:
    registry = CommandRegistry()
    registry.register(ConsoleCommand("/help", "Help."))
    with pytest.raises(DuplicateCommandError):
        registry.register(ConsoleCommand("/help", "Other help."))


@pytest.mark.parametrize("name", ["help", "/", "/Help", "/two words", "/a/b", " /help"])
def test_invalid_command_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        ConsoleCommand(name, "Description.")


def test_blank_description_is_rejected() -> None:
    with pytest.raises(ValueError):
        ConsoleCommand("/help", "  ")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/", True),
        ("/pro", True),
        ("", False),
        ("p", False),
        ("/project ", False),
        ("/a\nb", False),
        (" /help", False),
    ],
)
def test_is_command_prefix(text: str, expected: bool) -> None:
    assert is_command_prefix(text) is expected

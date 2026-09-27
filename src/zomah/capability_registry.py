from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from zomah.capabilities import (
    GetProjectStateRequest,
    GetProjectStateResponse,
    ReadFileRequest,
    ReadFileResponse,
    SearchKnowledgeRequest,
    SearchKnowledgeResponse,
    get_project_state,
    read_file,
    search_knowledge,
)


class CapabilityAuthority(StrEnum):
    """Machine authority class required by a registered capability."""

    READ = "READ"
    CHANGE = "CHANGE"
    EXECUTE = "EXECUTE"
    INTERNAL = "INTERNAL"


class CapabilityLifecycle(StrEnum):
    """Lifecycle state used for registry metadata and future exposure policy."""

    DESIGNED = "DESIGNED"
    PROTOTYPE_BUILT = "PROTOTYPE_BUILT"
    TESTING_PENDING = "TESTING_PENDING"
    TESTED = "TESTED"
    VERIFIED = "VERIFIED"


CapabilityHandler = Callable[..., Any]

# Keyword names the invocation boundaries use themselves; never dependencies.
RESERVED_INVOCATION_NAMES = frozenset({"worker", "operator_id", "trace_store"})
ModelType = type[BaseModel]


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    """Explicit description of one ZOMAH capability.

    The registry stores metadata and references the existing implementation.
    It does not wrap, duplicate, or reinterpret capability behavior.

    ``dependencies`` names the harness-owned keyword arguments the handler
    requires (for example ``repository`` or ``scope``). Callers supply exactly
    these; they are never taken from a request payload.
    """

    id: str
    description: str
    authority: CapabilityAuthority
    lifecycle: CapabilityLifecycle
    user_exposed: bool
    agent_exposed: bool
    request_model: ModelType
    response_model: ModelType
    handler: CapabilityHandler
    dependencies: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.id or self.id.strip() != self.id:
            raise ValueError("capability id must be non-empty and already normalized")
        if not self.description.strip():
            raise ValueError("capability description must be non-empty")
        if not isinstance(self.dependencies, frozenset):
            raise TypeError("capability dependencies must be a frozenset of names")
        for name in self.dependencies:
            if not name.isidentifier() or name in RESERVED_INVOCATION_NAMES:
                raise ValueError(f"invalid capability dependency name: {name!r}")


class DuplicateCapabilityError(ValueError):
    """Raised when the same capability id is registered more than once."""


class CapabilityNotFoundError(KeyError):
    """Raised when a requested capability id is not registered."""


class CapabilityRegistry:
    """Small explicit registry for ZOMAH capability metadata and handlers."""

    def __init__(self) -> None:
        self._definitions: dict[str, CapabilityDefinition] = {}

    def register(self, definition: CapabilityDefinition) -> None:
        if definition.id in self._definitions:
            raise DuplicateCapabilityError(
                f"capability already registered: {definition.id}"
            )
        self._definitions[definition.id] = definition

    def get(self, capability_id: str) -> CapabilityDefinition:
        try:
            return self._definitions[capability_id]
        except KeyError as exc:
            raise CapabilityNotFoundError(capability_id) from exc

    def all(self) -> tuple[CapabilityDefinition, ...]:
        return self._ordered(self._definitions.values())

    def user_exposed(self) -> tuple[CapabilityDefinition, ...]:
        return self._ordered(
            definition
            for definition in self._definitions.values()
            if definition.user_exposed
        )

    def agent_exposed(self) -> tuple[CapabilityDefinition, ...]:
        return self._ordered(
            definition
            for definition in self._definitions.values()
            if definition.agent_exposed
        )

    @staticmethod
    def _ordered(
        definitions: Any,
    ) -> tuple[CapabilityDefinition, ...]:
        return tuple(sorted(definitions, key=lambda definition: definition.id))


def default_capability_registry() -> CapabilityRegistry:
    """Build the explicit initial ZOMAH capability registry.

    Registration describes capabilities; it does not make any of them
    available to a model session (session tool sets are explicit allowlists).
    """

    registry = CapabilityRegistry()
    registry.register(
        CapabilityDefinition(
            id="get_project_state",
            description="Retrieve the canonical current state for one project.",
            authority=CapabilityAuthority.READ,
            lifecycle=CapabilityLifecycle.VERIFIED,
            user_exposed=True,
            agent_exposed=True,
            request_model=GetProjectStateRequest,
            response_model=GetProjectStateResponse,
            handler=get_project_state,
            dependencies=frozenset({"repository"}),
        )
    )
    registry.register(
        CapabilityDefinition(
            id="read_file",
            description=(
                "Read a bounded UTF-8 text slice from an absolute path inside "
                "configured read roots."
            ),
            authority=CapabilityAuthority.READ,
            lifecycle=CapabilityLifecycle.VERIFIED,
            user_exposed=True,
            agent_exposed=True,
            request_model=ReadFileRequest,
            response_model=ReadFileResponse,
            handler=read_file,
            dependencies=frozenset({"scope"}),
        )
    )
    registry.register(
        CapabilityDefinition(
            id="search_knowledge",
            description=(
                "Search indexed UTF-8 knowledge files inside configured read roots "
                "and return ranked references and excerpts."
            ),
            authority=CapabilityAuthority.READ,
            lifecycle=CapabilityLifecycle.VERIFIED,
            user_exposed=True,
            agent_exposed=True,
            request_model=SearchKnowledgeRequest,
            response_model=SearchKnowledgeResponse,
            handler=search_knowledge,
            dependencies=frozenset({"index"}),
        )
    )
    return registry

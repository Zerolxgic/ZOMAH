import pytest
from pydantic import BaseModel

from zomah.capabilities import (
    GetProjectStateRequest,
    GetProjectStateResponse,
    get_project_state,
)
from zomah.capability_registry import (
    CapabilityAuthority,
    CapabilityDefinition,
    CapabilityLifecycle,
    CapabilityNotFoundError,
    CapabilityRegistry,
    DuplicateCapabilityError,
    default_capability_registry,
)


class ExampleRequest(BaseModel):
    value: str


class ExampleResponse(BaseModel):
    value: str


def example_handler(request: ExampleRequest) -> ExampleResponse:
    return ExampleResponse(value=request.value)


def make_definition(
    capability_id: str,
    *,
    user_exposed: bool,
    agent_exposed: bool,
) -> CapabilityDefinition:
    return CapabilityDefinition(
        id=capability_id,
        description=f"Example capability {capability_id}.",
        authority=CapabilityAuthority.READ,
        lifecycle=CapabilityLifecycle.VERIFIED,
        user_exposed=user_exposed,
        agent_exposed=agent_exposed,
        request_model=ExampleRequest,
        response_model=ExampleResponse,
        handler=example_handler,
    )


def test_default_registry_registers_get_project_state():
    registry = default_capability_registry()

    definition = registry.get("get_project_state")

    assert definition.id == "get_project_state"
    assert definition.authority is CapabilityAuthority.READ
    assert definition.lifecycle is CapabilityLifecycle.VERIFIED
    assert definition.user_exposed is True
    assert definition.agent_exposed is True
    assert definition.request_model is GetProjectStateRequest
    assert definition.response_model is GetProjectStateResponse
    assert definition.handler is get_project_state


def test_registry_rejects_duplicate_ids():
    registry = CapabilityRegistry()
    definition = make_definition(
        "example",
        user_exposed=True,
        agent_exposed=True,
    )

    registry.register(definition)

    with pytest.raises(DuplicateCapabilityError):
        registry.register(definition)


def test_registry_missing_lookup_is_explicit():
    registry = CapabilityRegistry()

    with pytest.raises(CapabilityNotFoundError):
        registry.get("missing")


def test_registry_filters_user_and_agent_exposure_and_sorts_by_id():
    registry = CapabilityRegistry()
    registry.register(
        make_definition(
            "zeta-user",
            user_exposed=True,
            agent_exposed=False,
        )
    )
    registry.register(
        make_definition(
            "alpha-agent",
            user_exposed=False,
            agent_exposed=True,
        )
    )
    registry.register(
        make_definition(
            "middle-both",
            user_exposed=True,
            agent_exposed=True,
        )
    )

    assert [item.id for item in registry.all()] == [
        "alpha-agent",
        "middle-both",
        "zeta-user",
    ]
    assert [item.id for item in registry.user_exposed()] == [
        "middle-both",
        "zeta-user",
    ]
    assert [item.id for item in registry.agent_exposed()] == [
        "alpha-agent",
        "middle-both",
    ]


@pytest.mark.parametrize(
    "capability_id",
    ["", " leading", "trailing ", "  "],
)
def test_capability_definition_rejects_invalid_ids(capability_id: str):
    with pytest.raises(ValueError):
        make_definition(
            capability_id,
            user_exposed=True,
            agent_exposed=True,
        )


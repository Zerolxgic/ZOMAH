from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class DecisionStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"


NonEmptyStr = Annotated[str, Field(min_length=1)]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    statement: NonEmptyStr
    rationale: NonEmptyStr
    status: DecisionStatus = DecisionStatus.ACCEPTED
    created_at: datetime = Field(default_factory=utc_now)
    superseded_by: str | None = None


class DecisionTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: NonEmptyStr
    status: DecisionStatus
    superseded_by: str | None = None


class ImportantPath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: NonEmptyStr
    path: NonEmptyStr
    description: str | None = None


class ProjectState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    name: NonEmptyStr
    status: ProjectStatus = ProjectStatus.ACTIVE
    phase: NonEmptyStr
    summary: NonEmptyStr
    current_focus: NonEmptyStr
    last_action: str | None = None
    next_action: str | None = None
    open_questions: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    important_paths: list[ImportantPath] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)
    updated_by: NonEmptyStr


class ProjectStatePatch(BaseModel):
    """A partial, validated state mutation.

    Required scalar fields may be omitted but not blanked. `last_action` and
    `next_action` may be explicitly set to null. Collection fields use
    add/resolve operations so workers never replace whole historical lists.
    """

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)
    updated_by: NonEmptyStr

    name: NonEmptyStr | None = None
    status: ProjectStatus | None = None
    phase: NonEmptyStr | None = None
    summary: NonEmptyStr | None = None
    current_focus: NonEmptyStr | None = None
    last_action: str | None = None
    next_action: str | None = None

    add_open_questions: list[NonEmptyStr] = Field(default_factory=list)
    resolve_open_questions: list[NonEmptyStr] = Field(default_factory=list)
    add_blockers: list[NonEmptyStr] = Field(default_factory=list)
    resolve_blockers: list[NonEmptyStr] = Field(default_factory=list)
    add_decisions: list[Decision] = Field(default_factory=list)
    transition_decisions: list[DecisionTransition] = Field(default_factory=list)
    add_important_paths: list[ImportantPath] = Field(default_factory=list)

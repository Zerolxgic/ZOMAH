from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """Canonical stored decision record.

    This model is returned to workers, but it is not the model-facing input for
    creating a decision. Trusted bootstrap/admin code may still construct a
    complete Decision directly.
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    statement: NonEmptyStr
    rationale: NonEmptyStr
    status: DecisionStatus = DecisionStatus.ACCEPTED
    created_at: datetime = Field(default_factory=utc_now)
    superseded_by: str | None = None


class DecisionProposal(BaseModel):
    """Model-facing decision proposal.

    Status and timestamps are intentionally absent. ZOMAH stores proposals as
    `proposed` and stamps their creation time itself.
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    statement: NonEmptyStr
    rationale: NonEmptyStr


class DecisionTransition(BaseModel):
    """Internal/admin transition for an existing decision."""

    model_config = ConfigDict(extra="forbid")

    decision_id: NonEmptyStr
    status: DecisionStatus
    superseded_by: str | None = None

    @model_validator(mode="after")
    def validate_supersession(self) -> "DecisionTransition":
        if self.status == DecisionStatus.SUPERSEDED and not self.superseded_by:
            raise ValueError("superseded decisions require superseded_by")
        if self.status != DecisionStatus.SUPERSEDED and self.superseded_by is not None:
            raise ValueError("superseded_by is only valid for superseded decisions")
        return self


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
    """A partial, validated model-facing state mutation.

    Actor identity and timestamps are deliberately absent. The harness injects
    actor provenance when applying the patch. Decision additions are proposals
    only; authorization transitions are not model-facing.
    """

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)

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
    add_decisions: list[DecisionProposal] = Field(default_factory=list)
    add_important_paths: list[ImportantPath] = Field(default_factory=list)

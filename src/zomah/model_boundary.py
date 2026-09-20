from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from zomah.access import (
    InvalidDirectoryTarget,
    InvalidReadTarget,
    InvalidWriteTarget,
    PathOutsideScope,
)
from zomah.capabilities.read_file import UnsupportedTextFile
from zomah.capabilities.write_file import UnsupportedWriteContent
from zomah.execution import (
    InvalidScriptArguments,
    ScriptIntegrityError,
    ScriptRegistryError,
    UnknownScript,
)
from zomah.knowledge import KnowledgeIndexError, KnowledgeIndexUnavailable
from zomah.state import ProjectNotFound, RevisionConflict, StateError


RequestModel = TypeVar("RequestModel", bound=BaseModel)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class CapabilityError(BaseModel):
    """Small stable error shape intended for model-facing tool results."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool
    details: dict[str, Any] | None = None


class CapabilityEnvelope(BaseModel):
    """Normalized success/error envelope returned at the model boundary."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    result: dict[str, Any] | None = None
    error: CapabilityError | None = None

    @model_validator(mode="after")
    def require_exactly_one_payload(self) -> "CapabilityEnvelope":
        if self.ok:
            if self.result is None or self.error is not None:
                raise ValueError("successful capability envelopes require result only")
        else:
            if self.result is not None or self.error is None:
                raise ValueError("failed capability envelopes require error only")
        return self


def invoke_model_capability(
    request_type: type[RequestModel],
    capability: Callable[..., ResponseModel],
    payload: Mapping[str, Any],
    /,
    **dependencies: Any,
) -> CapabilityEnvelope:
    """Validate one model request and normalize the capability result.

    Capability implementations keep their native Python exceptions for local
    development and direct tests. Only this model-facing boundary converts
    failures into a compact stable envelope.

    `dependencies` are harness-owned objects such as repositories, scopes,
    registries, actor identity, or inspection roots. They are never sourced
    from the model payload by this adapter.
    """

    try:
        request = request_type.model_validate(dict(payload))
    except ValidationError as exc:
        return _failure(_validation_error(exc))

    try:
        response = capability(request, **dependencies)
    except Exception as exc:  # noqa: BLE001 - boundary intentionally contains failures.
        return _failure(normalize_capability_error(exc))

    if not isinstance(response, BaseModel):
        return _failure(
            CapabilityError(
                code="internal_error",
                message="Capability returned an unsupported response type.",
                retryable=False,
            )
        )

    return CapabilityEnvelope(
        ok=True,
        result=response.model_dump(mode="json"),
        error=None,
    )


def normalize_capability_error(exc: Exception) -> CapabilityError:
    """Translate a known internal exception into a stable model-facing error.

    The mapping is intentionally explicit. Unexpected exceptions are redacted
    rather than exposing Python internals to the model. A later trace layer can
    retain developer-facing exception details separately.
    """

    if isinstance(exc, PathOutsideScope):
        return _error("path_outside_scope", str(exc), retryable=False)
    if isinstance(exc, InvalidReadTarget):
        return _error("invalid_read_target", str(exc), retryable=False)
    if isinstance(exc, InvalidDirectoryTarget):
        return _error("invalid_directory_target", str(exc), retryable=False)
    if isinstance(exc, InvalidWriteTarget):
        return _error("invalid_write_target", str(exc), retryable=False)

    if isinstance(exc, UnsupportedTextFile):
        return _error("unsupported_text_file", str(exc), retryable=False)
    if isinstance(exc, UnsupportedWriteContent):
        return _error("unsupported_write_content", str(exc), retryable=False)

    if isinstance(exc, ProjectNotFound):
        return _error(
            "project_not_found",
            f"Project was not found: {exc}",
            retryable=False,
        )
    if isinstance(exc, RevisionConflict):
        return _error(
            "revision_conflict",
            str(exc),
            retryable=True,
            details={"recovery": "read current project state and retry with its revision"},
        )
    if isinstance(exc, StateError):
        return _error("state_error", str(exc), retryable=False)

    if isinstance(exc, UnknownScript):
        return _error("unknown_script", str(exc), retryable=False)
    if isinstance(exc, InvalidScriptArguments):
        return _error("invalid_script_arguments", str(exc), retryable=False)
    if isinstance(exc, ScriptIntegrityError):
        return _error(
            "script_integrity_error",
            str(exc),
            retryable=False,
            details={"recovery": "script must be explicitly reviewed and re-registered"},
        )
    if isinstance(exc, ScriptRegistryError):
        return _error("script_registry_error", str(exc), retryable=False)

    if isinstance(exc, KnowledgeIndexUnavailable):
        return _error("knowledge_index_unavailable", str(exc), retryable=False)
    if isinstance(exc, KnowledgeIndexError):
        return _error("knowledge_index_error", str(exc), retryable=True)

    if isinstance(exc, sqlite3.OperationalError):
        return _error(
            "storage_unavailable",
            "Local SQLite storage is temporarily unavailable.",
            retryable=True,
        )
    if isinstance(exc, FileNotFoundError):
        return _error("resource_not_found", str(exc), retryable=False)
    if isinstance(exc, PermissionError):
        return _error("permission_denied", str(exc), retryable=False)
    if isinstance(exc, OSError):
        return _error(
            "filesystem_error",
            _safe_os_error_message(exc),
            retryable=True,
        )
    if isinstance(exc, ValueError):
        return _error("invalid_request", str(exc), retryable=False)

    return CapabilityError(
        code="internal_error",
        message="Capability failed unexpectedly.",
        retryable=False,
    )


def _validation_error(exc: ValidationError) -> CapabilityError:
    issues: list[dict[str, str]] = []
    for item in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "request"
        issues.append(
            {
                "field": location,
                "type": str(item["type"]),
                "message": str(item["msg"]),
            }
        )

    return CapabilityError(
        code="invalid_request",
        message="Capability request failed validation.",
        retryable=False,
        details={"issues": issues},
    )


def _safe_os_error_message(exc: OSError) -> str:
    if exc.errno is not None:
        return os.strerror(exc.errno)
    return "Filesystem operation failed."


def _error(
    code: str,
    message: str,
    *,
    retryable: bool,
    details: dict[str, Any] | None = None,
) -> CapabilityError:
    return CapabilityError(
        code=code,
        message=message,
        retryable=retryable,
        details=details,
    )


def _failure(error: CapabilityError) -> CapabilityEnvelope:
    return CapabilityEnvelope(ok=False, result=None, error=error)

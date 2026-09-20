from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class PathScopeError(PermissionError):
    """Base error for filesystem paths rejected by ZOMAH's configured scope."""


class PathOutsideScope(PathScopeError):
    """Raised when a requested path resolves outside every approved root."""


class InvalidReadTarget(PathScopeError):
    """Raised when a requested read target is not an existing regular file."""


class InvalidDirectoryTarget(PathScopeError):
    """Raised when a requested directory target is not an existing directory."""


class InvalidWriteTarget(PathScopeError):
    """Raised when a requested write target violates the configured write boundary."""


@dataclass(frozen=True, slots=True)
class ReadScope:
    """A minimal allowlist of canonical filesystem roots for read operations.

    Roots are resolved when the scope is created. Requested paths are resolved
    before containment checks, so `..` traversal and symlink escapes are
    rejected by the same boundary.
    """

    roots: tuple[Path, ...]

    @classmethod
    def from_paths(cls, roots: Iterable[str | Path]) -> "ReadScope":
        canonical_roots: list[Path] = []

        for root in roots:
            canonical = Path(root).expanduser().resolve(strict=True)
            if not canonical.is_dir():
                raise ValueError(f"read root is not a directory: {canonical}")
            canonical_roots.append(canonical)

        if not canonical_roots:
            raise ValueError("at least one read root is required")

        return cls(roots=tuple(dict.fromkeys(canonical_roots)))

    def resolve_file(self, requested_path: str | Path) -> Path:
        canonical = self._resolve_existing(
            requested_path,
            absolute_error="read_file requires an absolute path",
            missing_label="file",
            invalid_error=InvalidReadTarget,
        )

        if not canonical.is_file():
            raise InvalidReadTarget(f"read target is not a regular file: {canonical}")

        return canonical

    def resolve_directory(self, requested_path: str | Path) -> Path:
        canonical = self._resolve_existing(
            requested_path,
            absolute_error="list_directory requires an absolute path",
            missing_label="directory",
            invalid_error=InvalidDirectoryTarget,
        )

        if not canonical.is_dir():
            raise InvalidDirectoryTarget(
                f"directory target is not a directory: {canonical}"
            )

        return canonical

    def _resolve_existing(
        self,
        requested_path: str | Path,
        *,
        absolute_error: str,
        missing_label: str,
        invalid_error: type[PathScopeError],
    ) -> Path:
        requested = Path(requested_path).expanduser()
        if not requested.is_absolute():
            raise invalid_error(absolute_error)

        try:
            canonical = requested.resolve(strict=True)
        except FileNotFoundError as exc:
            raise invalid_error(
                f"{missing_label} does not exist: {requested}"
            ) from exc

        if not any(canonical.is_relative_to(root) for root in self.roots):
            raise PathOutsideScope(f"path is outside configured read roots: {canonical}")

        return canonical


@dataclass(frozen=True, slots=True)
class WriteScope:
    """A minimal allowlist of canonical filesystem roots for write operations.

    Write authority is intentionally separate from read authority. Parent
    directories must already exist, and their canonical location is checked
    before a target path is returned. Existing symlink targets are rejected so
    a write cannot acquire authority indirectly through a link.
    """

    roots: tuple[Path, ...]

    @classmethod
    def from_paths(cls, roots: Iterable[str | Path]) -> "WriteScope":
        canonical_roots: list[Path] = []

        for root in roots:
            canonical = Path(root).expanduser().resolve(strict=True)
            if not canonical.is_dir():
                raise ValueError(f"write root is not a directory: {canonical}")
            canonical_roots.append(canonical)

        if not canonical_roots:
            raise ValueError("at least one write root is required")

        return cls(roots=tuple(dict.fromkeys(canonical_roots)))

    def resolve_existing_file(self, requested_path: str | Path) -> Path:
        requested = Path(requested_path).expanduser()
        if not requested.is_absolute():
            raise InvalidWriteTarget("move_file source requires an absolute path")

        if not os.path.lexists(requested):
            raise InvalidWriteTarget(f"move source does not exist: {requested}")
        if requested.is_symlink():
            raise InvalidWriteTarget(f"symlink move sources are not allowed: {requested}")

        try:
            canonical = requested.resolve(strict=True)
        except FileNotFoundError as exc:
            raise InvalidWriteTarget(f"move source does not exist: {requested}") from exc

        if not any(canonical.is_relative_to(root) for root in self.roots):
            raise PathOutsideScope(
                f"path is outside configured write roots: {canonical}"
            )
        if not canonical.is_file():
            raise InvalidWriteTarget(
                f"move source is not a regular file: {canonical}"
            )

        return canonical

    def resolve_target(self, requested_path: str | Path) -> Path:
        requested = Path(requested_path).expanduser()
        if not requested.is_absolute():
            raise InvalidWriteTarget("write_file requires an absolute path")

        if not requested.name or requested.name in {".", ".."}:
            raise InvalidWriteTarget(f"invalid write target: {requested}")

        try:
            parent = requested.parent.resolve(strict=True)
        except FileNotFoundError as exc:
            raise InvalidWriteTarget(
                f"parent directory does not exist: {requested.parent}"
            ) from exc

        if not parent.is_dir():
            raise InvalidWriteTarget(f"parent is not a directory: {parent}")

        if not any(parent.is_relative_to(root) for root in self.roots):
            raise PathOutsideScope(
                f"path is outside configured write roots: {parent / requested.name}"
            )

        target = parent / requested.name
        if os.path.lexists(target) and target.is_symlink():
            raise InvalidWriteTarget(f"symlink write targets are not allowed: {target}")

        return target

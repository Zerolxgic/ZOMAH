from __future__ import annotations

import os
import pwd
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


SystemDomain = Literal[
    "processes",
    "services",
    "storage",
    "memory",
    "gpu",
    "mounts",
]

Query = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]

DEFAULT_MAX_ENTRIES = 100
MAX_ENTRIES = 500


class ProcessInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pid: int = Field(ge=1)
    name: str
    state: str | None = None
    user: str | None = None
    rss_bytes: int | None = Field(default=None, ge=0)
    command: str | None = None


class ServiceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    load: str
    active: str
    sub: str
    description: str


class StorageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["root", "home"]
    path: str
    total_bytes: int = Field(ge=0)
    used_bytes: int = Field(ge=0)
    free_bytes: int = Field(ge=0)
    percent_used: float = Field(ge=0, le=100)


class MemoryInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    used_bytes: int = Field(ge=0)
    percent_used: float = Field(ge=0, le=100)
    swap_total_bytes: int = Field(ge=0)
    swap_free_bytes: int = Field(ge=0)


class GpuInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card: str
    vendor: str
    vendor_id: str | None = None
    device_id: str | None = None
    driver: str | None = None
    vram_total_bytes: int | None = Field(default=None, ge=0)
    vram_used_bytes: int | None = Field(default=None, ge=0)


class MountInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mount_point: str
    fs_type: str
    source: str
    read_only: bool


class ProcessSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ProcessInfo]
    total_matches: int = Field(ge=0)
    returned: int = Field(ge=0)
    truncated: bool


class ServiceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    items: list[ServiceInfo]
    total_matches: int = Field(ge=0)
    returned: int = Field(ge=0)
    truncated: bool
    message: str | None = None


class StorageSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[StorageInfo]


class GpuSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[GpuInfo]


class MountSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[MountInfo]
    total_matches: int = Field(ge=0)
    returned: int = Field(ge=0)
    truncated: bool


SystemResult = (
    ProcessSnapshot
    | ServiceSnapshot
    | StorageSnapshot
    | MemoryInfo
    | GpuSnapshot
    | MountSnapshot
)


class InspectSystemRequest(BaseModel):
    """Validated model-facing input for structured system inspection."""

    model_config = ConfigDict(extra="forbid")

    domain: SystemDomain
    query: Query | None = None
    max_entries: int = Field(default=DEFAULT_MAX_ENTRIES, ge=1, le=MAX_ENTRIES)


class InspectSystemResponse(BaseModel):
    """Structured facts from one explicitly supported system domain."""

    model_config = ConfigDict(extra="forbid")

    domain: SystemDomain
    collected_at: datetime
    result: SystemResult


class CommandRunner(Protocol):
    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]: ...


def inspect_system(
    request: InspectSystemRequest,
    *,
    proc_root: Path = Path("/proc"),
    sys_root: Path = Path("/sys"),
    runner: CommandRunner | None = None,
) -> InspectSystemResponse:
    """Inspect one supported Linux system domain without arbitrary shell access.

    Every collector is explicit. The only external command currently used is a
    fixed `systemctl list-units` invocation for service state; user input never
    becomes executable command text.
    """

    runner = runner or _run_command

    if request.domain == "processes":
        result: SystemResult = _inspect_processes(request, proc_root)
    elif request.domain == "services":
        result = _inspect_services(request, runner)
    elif request.domain == "storage":
        result = _inspect_storage()
    elif request.domain == "memory":
        result = _inspect_memory(proc_root)
    elif request.domain == "gpu":
        result = _inspect_gpu(sys_root)
    elif request.domain == "mounts":
        result = _inspect_mounts(request, proc_root)
    else:  # pragma: no cover - Pydantic rejects unknown domains first.
        raise ValueError(f"unsupported system domain: {request.domain}")

    return InspectSystemResponse(
        domain=request.domain,
        collected_at=datetime.now(UTC),
        result=result,
    )


def _inspect_processes(request: InspectSystemRequest, proc_root: Path) -> ProcessSnapshot:
    query = request.query.casefold() if request.query else None
    matches: list[ProcessInfo] = []
    inspector_pid = os.getpid()

    for process_dir in sorted(
        (path for path in proc_root.iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        if int(process_dir.name) == inspector_pid:
            continue

        info = _read_process(process_dir)
        if info is None:
            continue

        searchable = " ".join(
            part for part in [info.name, info.user, info.command] if part
        ).casefold()
        if query and query not in searchable:
            continue

        matches.append(info)

    visible = matches[: request.max_entries]
    return ProcessSnapshot(
        items=visible,
        total_matches=len(matches),
        returned=len(visible),
        truncated=len(matches) > len(visible),
    )


def _read_process(process_dir: Path) -> ProcessInfo | None:
    try:
        status_text = (process_dir / "status").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None

    fields: dict[str, str] = {}
    for line in status_text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()

    try:
        pid = int(fields.get("Pid", process_dir.name))
    except ValueError:
        return None

    uid_text = fields.get("Uid", "").split()
    user: str | None = None
    if uid_text:
        try:
            user = pwd.getpwuid(int(uid_text[0])).pw_name
        except (KeyError, ValueError):
            user = uid_text[0]

    rss_bytes: int | None = None
    rss = fields.get("VmRSS")
    if rss:
        parts = rss.split()
        if parts and parts[0].isdigit():
            rss_bytes = int(parts[0]) * 1024

    command: str | None = None
    try:
        raw = (process_dir / "cmdline").read_bytes()
        if raw:
            command = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        pass

    return ProcessInfo(
        pid=pid,
        name=fields.get("Name", process_dir.name),
        state=fields.get("State"),
        user=user,
        rss_bytes=rss_bytes,
        command=command,
    )


def _inspect_services(
    request: InspectSystemRequest,
    runner: CommandRunner,
) -> ServiceSnapshot:
    argv = [
        "systemctl",
        "list-units",
        "--type=service",
        "--all",
        "--no-pager",
        "--no-legend",
        "--plain",
    ]

    try:
        completed = runner(argv)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return ServiceSnapshot(
            available=False,
            items=[],
            total_matches=0,
            returned=0,
            truncated=False,
            message=f"service inspection unavailable: {exc}",
        )

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "systemctl failed").strip()
        return ServiceSnapshot(
            available=False,
            items=[],
            total_matches=0,
            returned=0,
            truncated=False,
            message=message,
        )

    query = request.query.casefold() if request.query else None
    matches: list[ServiceInfo] = []

    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue

        item = ServiceInfo(
            name=parts[0],
            load=parts[1],
            active=parts[2],
            sub=parts[3],
            description=parts[4],
        )
        searchable = f"{item.name} {item.active} {item.sub} {item.description}".casefold()
        if query and query not in searchable:
            continue
        matches.append(item)

    visible = matches[: request.max_entries]
    return ServiceSnapshot(
        available=True,
        items=visible,
        total_matches=len(matches),
        returned=len(visible),
        truncated=len(matches) > len(visible),
    )


def _inspect_storage() -> StorageSnapshot:
    targets = [("root", Path("/")), ("home", Path.home())]
    items: list[StorageInfo] = []

    for label, path in targets:
        usage = shutil.disk_usage(path)
        percent_used = 0.0 if usage.total == 0 else (usage.used / usage.total) * 100
        items.append(
            StorageInfo(
                label=label,
                path=str(path),
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                percent_used=round(percent_used, 2),
            )
        )

    return StorageSnapshot(items=items)


def _inspect_memory(proc_root: Path) -> MemoryInfo:
    values: dict[str, int] = {}
    for line in (proc_root / "meminfo").read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        multiplier = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
        values[key] = value * multiplier

    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = max(total - available, 0)
    percent_used = 0.0 if total == 0 else (used / total) * 100

    return MemoryInfo(
        total_bytes=total,
        available_bytes=available,
        used_bytes=used,
        percent_used=round(percent_used, 2),
        swap_total_bytes=values.get("SwapTotal", 0),
        swap_free_bytes=values.get("SwapFree", 0),
    )


def _inspect_gpu(sys_root: Path) -> GpuSnapshot:
    drm_root = sys_root / "class" / "drm"
    if not drm_root.exists():
        return GpuSnapshot(items=[])

    items: list[GpuInfo] = []
    for card_path in sorted(drm_root.iterdir(), key=lambda path: path.name):
        if not re.fullmatch(r"card\d+", card_path.name):
            continue

        device = card_path / "device"
        if not device.exists():
            continue

        vendor_id = _read_optional_text(device / "vendor")
        device_id = _read_optional_text(device / "device")
        driver = None
        try:
            driver = (device / "driver").resolve(strict=True).name
        except (FileNotFoundError, OSError):
            driver = _uevent_value(device / "uevent", "DRIVER")

        items.append(
            GpuInfo(
                card=card_path.name,
                vendor=_vendor_name(vendor_id),
                vendor_id=vendor_id,
                device_id=device_id,
                driver=driver,
                vram_total_bytes=_read_optional_int(device / "mem_info_vram_total"),
                vram_used_bytes=_read_optional_int(device / "mem_info_vram_used"),
            )
        )

    return GpuSnapshot(items=items)


def _inspect_mounts(request: InspectSystemRequest, proc_root: Path) -> MountSnapshot:
    query = request.query.casefold() if request.query else None
    matches: list[MountInfo] = []

    for line in (proc_root / "self" / "mountinfo").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        try:
            separator = parts.index("-")
        except ValueError:
            continue
        if len(parts) <= separator + 2 or len(parts) < 6:
            continue

        mount_point = _decode_mount_field(parts[4])
        options = parts[5].split(",")
        fs_type = parts[separator + 1]
        source = _decode_mount_field(parts[separator + 2])

        item = MountInfo(
            mount_point=mount_point,
            fs_type=fs_type,
            source=source,
            read_only="ro" in options,
        )
        searchable = f"{item.mount_point} {item.fs_type} {item.source}".casefold()
        if query and query not in searchable:
            continue
        matches.append(item)

    matches.sort(key=lambda item: (item.mount_point.casefold(), item.mount_point))
    visible = matches[: request.max_entries]
    return MountSnapshot(
        items=visible,
        total_matches=len(matches),
        returned=len(visible),
        truncated=len(matches) > len(visible),
    )


def _run_command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_optional_int(path: Path) -> int | None:
    text = _read_optional_text(path)
    if text is None:
        return None
    try:
        value = int(text, 0)
    except ValueError:
        return None
    return value if value >= 0 else None


def _uevent_value(path: Path, key: str) -> str | None:
    text = _read_optional_text(path)
    if not text:
        return None
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip() or None
    return None


def _vendor_name(vendor_id: str | None) -> str:
    return {
        "0x1002": "AMD",
        "0x10de": "NVIDIA",
        "0x8086": "Intel",
    }.get((vendor_id or "").casefold(), "Unknown")


def _decode_mount_field(value: str) -> str:
    replacements = {
        r"\040": " ",
        r"\011": "\t",
        r"\012": "\n",
        r"\134": "\\",
    }
    for encoded, decoded in replacements.items():
        value = value.replace(encoded, decoded)
    return value

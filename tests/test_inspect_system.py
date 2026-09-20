from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from zomah.capabilities import InspectSystemRequest, inspect_system
from zomah.capabilities.inspect_system import (
    GpuSnapshot,
    MemoryInfo,
    MountSnapshot,
    ProcessSnapshot,
    ServiceSnapshot,
    StorageSnapshot,
)


def test_memory_inspection_returns_structured_bytes(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(
        "MemTotal:       1000 kB\n"
        "MemAvailable:    250 kB\n"
        "SwapTotal:       100 kB\n"
        "SwapFree:         40 kB\n",
        encoding="utf-8",
    )

    response = inspect_system(InspectSystemRequest(domain="memory"), proc_root=proc)

    assert isinstance(response.result, MemoryInfo)
    assert response.result.total_bytes == 1_024_000
    assert response.result.available_bytes == 256_000
    assert response.result.used_bytes == 768_000
    assert response.result.percent_used == 75.0
    assert response.result.swap_total_bytes == 102_400


def test_process_inspection_filters_and_bounds_results(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()

    for pid, name, command in [
        (10, "python", "python worker.py"),
        (20, "bash", "bash script.sh"),
        (30, "python", "python server.py"),
    ]:
        process = proc / str(pid)
        process.mkdir()
        process.joinpath("status").write_text(
            f"Name:\t{name}\nPid:\t{pid}\nState:\tS (sleeping)\nUid:\t0\t0\t0\t0\nVmRSS:\t{pid} kB\n",
            encoding="utf-8",
        )
        process.joinpath("cmdline").write_bytes(command.replace(" ", "\x00").encode() + b"\x00")

    response = inspect_system(
        InspectSystemRequest(domain="processes", query="python", max_entries=1),
        proc_root=proc,
    )

    assert isinstance(response.result, ProcessSnapshot)
    assert response.result.total_matches == 2
    assert response.result.returned == 1
    assert response.result.truncated is True
    assert response.result.items[0].pid == 10
    assert response.result.items[0].rss_bytes == 10 * 1024


def test_service_inspection_uses_fixed_systemctl_command_and_filters() -> None:
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "ssh.service loaded active running OpenSSH server daemon\n"
                "cups.service loaded inactive dead CUPS Scheduler\n"
            ),
            stderr="",
        )

    response = inspect_system(
        InspectSystemRequest(domain="services", query="ssh"),
        runner=runner,
    )

    assert seen == [[
        "systemctl",
        "list-units",
        "--type=service",
        "--all",
        "--no-pager",
        "--no-legend",
        "--plain",
    ]]
    assert isinstance(response.result, ServiceSnapshot)
    assert response.result.available is True
    assert [item.name for item in response.result.items] == ["ssh.service"]


def test_service_inspection_returns_structured_unavailable_state() -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="systemd unavailable")

    response = inspect_system(InspectSystemRequest(domain="services"), runner=runner)

    assert isinstance(response.result, ServiceSnapshot)
    assert response.result.available is False
    assert response.result.message == "systemd unavailable"
    assert response.result.items == []


def test_mount_inspection_decodes_fields_and_filters(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    mount_dir = proc / "self"
    mount_dir.mkdir(parents=True)
    mount_dir.joinpath("mountinfo").write_text(
        "36 25 0:32 / / rw,relatime - ext4 /dev/nvme0n1p2 rw\n"
        "37 25 0:33 / /mnt/My\\040Drive ro,relatime - fuseblk /dev/sdb1 ro\n",
        encoding="utf-8",
    )

    response = inspect_system(
        InspectSystemRequest(domain="mounts", query="My Drive"),
        proc_root=proc,
    )

    assert isinstance(response.result, MountSnapshot)
    assert response.result.total_matches == 1
    assert response.result.items[0].mount_point == "/mnt/My Drive"
    assert response.result.items[0].read_only is True


def test_gpu_inspection_reads_drm_sysfs(tmp_path: Path) -> None:
    sys_root = tmp_path / "sys"
    device = sys_root / "class" / "drm" / "card0" / "device"
    device.mkdir(parents=True)
    device.joinpath("vendor").write_text("0x1002\n", encoding="utf-8")
    device.joinpath("device").write_text("0x747e\n", encoding="utf-8")
    device.joinpath("mem_info_vram_total").write_text("17179869184\n", encoding="utf-8")
    device.joinpath("mem_info_vram_used").write_text("4294967296\n", encoding="utf-8")
    device.joinpath("uevent").write_text("DRIVER=amdgpu\n", encoding="utf-8")

    response = inspect_system(InspectSystemRequest(domain="gpu"), sys_root=sys_root)

    assert isinstance(response.result, GpuSnapshot)
    assert len(response.result.items) == 1
    gpu = response.result.items[0]
    assert gpu.vendor == "AMD"
    assert gpu.driver == "amdgpu"
    assert gpu.vram_total_bytes == 17_179_869_184
    assert gpu.vram_used_bytes == 4_294_967_296


def test_storage_inspection_returns_root_and_home() -> None:
    response = inspect_system(InspectSystemRequest(domain="storage"))

    assert isinstance(response.result, StorageSnapshot)
    assert [item.label for item in response.result.items] == ["root", "home"]
    assert all(item.total_bytes >= item.used_bytes for item in response.result.items)


def test_inspect_system_rejects_unknown_domain() -> None:
    with pytest.raises(ValidationError):
        InspectSystemRequest(domain="shell")  # type: ignore[arg-type]


def test_inspect_system_caps_entry_count() -> None:
    with pytest.raises(ValidationError):
        InspectSystemRequest(domain="processes", max_entries=501)

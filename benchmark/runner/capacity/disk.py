"""Disk preflight: is there room for the uncached images plus a safety reserve?"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Final, Protocol

from benchmark.runner.capacity._exec import Runner, subprocess_runner
from benchmark.runner.capacity.sizes import (
    Fetch,
    LocalFn,
    RegistryFn,
    SizeResolver,
    local_size,
    registry_size,
)
from benchmark.runner.capacity.types import GIB, CapacityVerdict, ResourceConfig


class StatVfsResult(Protocol):
    """The subset of ``os.statvfs_result`` the free-space math needs."""

    @property
    def f_bavail(self) -> int: ...

    @property
    def f_frsize(self) -> int: ...


StatVfs = Callable[[str], StatVfsResult]


def _statvfs(path: str) -> StatVfsResult:
    """Default statvfs adapter, keeping the injected protocol and ``os`` in sync."""
    return os.statvfs(path)


_RETENTION_REMEDY: Final[str] = (
    "Drop per-challenge images as soon as their challenge finishes: "
    "run with `--image-retention per-challenge`."
)
_PREPULL_REMEDY: Final[str] = (
    "Pre-stage the images before the run (enable prefetch, or pull them ahead of time with "
    "`scripts/benchmark/prepull_swebench_images.py`)."
)
_VOLUME_REMEDY: Final[str] = (
    "Grow the EC2 root volume (infra root_block_device volume_size) and re-apply Terraform."
)
_OVERRIDE_REMEDY: Final[str] = (
    "Proceed anyway with `--allow-insufficient-disk` only if you accept an out-of-space run."
)


def docker_root(*, runner: Runner = subprocess_runner) -> str:
    """The docker daemon's root directory (where images live)."""
    code, out = runner(["docker", "info", "--format", "{{.DockerRootDir}}"])
    text = out.strip()
    return text if code == 0 and text else "/var/lib/docker"


def free_bytes(path: str, *, statvfs: StatVfs = _statvfs) -> int:
    """Free (non-reserved) bytes on the filesystem holding ``path``."""
    try:
        stats = statvfs(path or "/")
    except OSError:
        return 0
    return stats.f_bavail * stats.f_frsize


def _remedies(shortfall_bytes: int, needed_bytes: int) -> tuple[str, ...]:
    """Human-readable steps to close a shortfall (empty when there is none)."""
    if shortfall_bytes <= 0:
        return ()
    return (_RETENTION_REMEDY, _PREPULL_REMEDY, _VOLUME_REMEDY, _OVERRIDE_REMEDY)


def preflight(
    refs: Iterable[str],
    *,
    config: ResourceConfig,
    fetch: Fetch | None = None,
    docker_root_fn: Callable[[], str] = docker_root,
    free_fn: Callable[[str], int] = free_bytes,
    registry_fn: RegistryFn = registry_size,
    local_fn: LocalFn = local_size,
) -> CapacityVerdict:
    """Compare the uncached image footprint plus a reserve against free disk.

    Cached images count as zero (they are already on disk). ``ok`` is either the
    disk check passing or ``config.allow_insufficient_disk``. All system access is
    injected so tests never touch docker or the registry.
    """
    root = docker_root_fn()
    free = free_fn(root)
    resolver = SizeResolver(config=config, fetch=fetch, local_fn=local_fn, registry_fn=registry_fn)
    sizes = resolver.resolve(refs)
    uncached = sum(s.uncompressed_estimate_bytes for s in sizes if not s.cached)
    reserve = int(config.disk_reserve_gib * GIB)
    needed = uncached + reserve
    shortfall = max(0, needed - free)
    ok = free >= needed or config.allow_insufficient_disk
    return CapacityVerdict(
        ok=ok,
        needed_bytes=needed,
        free_bytes=free,
        shortfall_bytes=shortfall,
        per_image=sizes,
        docker_root=root,
        remedies=_remedies(shortfall, needed),
    )

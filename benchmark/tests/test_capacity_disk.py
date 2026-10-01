"""Offline tests for the disk preflight arithmetic and remedies."""

from __future__ import annotations

from benchmark.runner.capacity.disk import docker_root, free_bytes, preflight
from benchmark.runner.capacity.types import GIB, ResourceConfig


def _no_local(ref: str) -> int | None:
    return None


def _registry_of(size: int):
    def registry(ref: str, *, fetch: object = None) -> int | None:
        return size

    return registry


def _preflight(
    *,
    config: ResourceConfig,
    free: int,
    registry_size: int = 0,
    local=None,
    refs: list[str] | None = None,
):
    return preflight(
        refs or ["a"],
        config=config,
        docker_root_fn=lambda: "/docker",
        free_fn=lambda path: free,
        registry_fn=_registry_of(registry_size),
        local_fn=local or _no_local,
    )


def test_preflight_ok_when_free_exceeds_needed() -> None:
    config = ResourceConfig(disk_reserve_gib=1.0, disk_safety_factor=1.0)
    verdict = _preflight(config=config, free=100 * GIB, registry_size=2 * GIB, refs=["a", "b"])
    assert verdict.ok is True
    assert verdict.needed_bytes == 4 * GIB + GIB
    assert verdict.shortfall_bytes == 0
    assert verdict.docker_root == "/docker"
    assert verdict.remedies == ()


def test_preflight_reports_shortfall_and_remedies() -> None:
    config = ResourceConfig(disk_reserve_gib=10.0, disk_safety_factor=1.0)
    verdict = _preflight(config=config, free=GIB, registry_size=5 * GIB)
    assert verdict.ok is False
    assert verdict.needed_bytes == 15 * GIB
    assert verdict.shortfall_bytes == 14 * GIB
    assert any("--image-retention per-challenge" in r for r in verdict.remedies)
    assert any("allow-insufficient-disk" in r for r in verdict.remedies)


def test_preflight_counts_cached_images_as_zero() -> None:
    config = ResourceConfig(disk_reserve_gib=1.0, disk_safety_factor=1.0)
    verdict = _preflight(
        config=config,
        free=100 * GIB,
        registry_size=5 * GIB,
        local=lambda ref: 9 * GIB if ref == "a" else None,
        refs=["a", "b"],
    )
    assert verdict.per_image[0].cached is True
    assert verdict.needed_bytes == 5 * GIB + GIB


def test_allow_insufficient_disk_overrides_ok_but_keeps_shortfall() -> None:
    config = ResourceConfig(disk_reserve_gib=10.0, allow_insufficient_disk=True)
    verdict = _preflight(config=config, free=0, registry_size=5 * GIB)
    assert verdict.ok is True
    assert verdict.shortfall_bytes > 0
    assert verdict.remedies


def test_docker_root_and_free_bytes_are_injectable() -> None:
    assert docker_root(runner=lambda argv: (0, "/srv/docker\n")) == "/srv/docker"
    assert docker_root(runner=lambda argv: (1, "")) == "/var/lib/docker"

    class Stats:
        f_bavail = 3
        f_frsize = 1_000

    def raise_oserror(path: str) -> Stats:
        raise OSError

    assert free_bytes("/x", statvfs=lambda path: Stats()) == 3_000
    assert free_bytes("/x", statvfs=raise_oserror) == 0

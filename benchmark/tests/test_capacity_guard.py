"""Offline tests for the capacity_manager context manager."""

from __future__ import annotations

import pytest

from benchmark.runner.capacity.guard import InsufficientDiskError, capacity_manager
from benchmark.runner.capacity.types import GIB, ResourceConfig, RetentionPolicy


def _registry_of(size: int):
    def registry(ref: str, *, fetch: object = None) -> int | None:
        return size

    return registry


def test_capacity_manager_raises_on_shortfall() -> None:
    config = ResourceConfig(disk_reserve_gib=10.0)
    with (
        pytest.raises(InsufficientDiskError) as excinfo,
        capacity_manager(
            ["a"],
            config=config,
            docker_root_fn=lambda: "/docker",
            free_fn=lambda path: 0,
            registry_fn=_registry_of(5 * GIB),
            local_fn=lambda ref: None,
        ),
    ):
        pass
    assert excinfo.value.verdict.shortfall_bytes > 0
    assert "--image-retention per-challenge" in str(excinfo.value)


def test_capacity_manager_allows_override_and_drains_per_challenge() -> None:
    removed: list[str] = []

    def remove(ref: str) -> int:
        removed.append(ref)
        return 123

    config = ResourceConfig(
        retention=RetentionPolicy.PER_CHALLENGE,
        allow_insufficient_disk=True,
        disk_reserve_gib=10.0,
    )
    with capacity_manager(
        ["a"],
        config=config,
        docker_root_fn=lambda: "/docker",
        free_fn=lambda path: 0,
        registry_fn=_registry_of(5 * GIB),
        local_fn=lambda ref: None,
        remove_fn=remove,
    ) as cap:
        cap.retention.acquire("a")
        cap.retention.release("a")
    assert removed == ["a"]


def test_capacity_manager_prefetch_off_unless_enabled() -> None:
    config = ResourceConfig(disk_reserve_gib=0.0)
    with capacity_manager(
        ["a"],
        config=config,
        docker_root_fn=lambda: "/docker",
        free_fn=lambda path: 100 * GIB,
        registry_fn=_registry_of(1 * GIB),
        local_fn=lambda ref: 1 * GIB,
    ) as cap:
        assert cap.prefetch is None
        assert cap.verdict.ok is True

"""Offline tests for reference-counted, targeted image retention."""

from __future__ import annotations

from benchmark.runner.capacity.retention import ImageRetention
from benchmark.runner.capacity.types import ResourceConfig, RetentionPolicy


class _Remover:
    """Records removals and reports a fixed freed size."""

    def __init__(self, freed: int = 100) -> None:
        self.freed = freed
        self.removed: list[str] = []

    def __call__(self, ref: str) -> int:
        self.removed.append(ref)
        return self.freed


def test_ref_held_by_two_cells_is_removed_only_after_both_release() -> None:
    remover = _Remover()
    config = ResourceConfig(retention=RetentionPolicy.PER_CHALLENGE)
    retention = ImageRetention(config, remove_fn=remover, refs_for_ids=lambda ids: ["img-a"])
    retention.acquire("img-a")
    retention.acquire("img-a")
    retention.release("img-a")
    assert retention.release_challenge(["c1"]) == 0
    assert remover.removed == []
    retention.release("img-a")
    assert retention.release_challenge(["c1"]) == 100
    assert remover.removed == ["img-a"]
    assert retention.freed_bytes == 100


def test_per_challenge_removes_only_its_own_refs() -> None:
    remover = _Remover(freed=50)
    config = ResourceConfig(retention=RetentionPolicy.PER_CHALLENGE)
    mapping = {"c1": ["img-a"], "c2": ["img-b"]}
    retention = ImageRetention(
        config, remove_fn=remover, refs_for_ids=lambda ids: [r for i in ids for r in mapping[i]]
    )
    for ref in ("img-a", "img-b"):
        retention.acquire(ref)
        retention.release(ref)
    assert retention.release_challenge(["c1"]) == 50
    assert remover.removed == ["img-a"]
    assert retention.release_challenge(["c2"]) == 50
    assert remover.removed == ["img-a", "img-b"]


def test_keep_never_removes() -> None:
    remover = _Remover()
    config = ResourceConfig(retention=RetentionPolicy.KEEP)
    retention = ImageRetention(config, remove_fn=remover, refs_for_ids=lambda ids: ["img-a"])
    retention.acquire("img-a")
    retention.release("img-a")
    assert retention.release_challenge(["c1"]) == 0
    assert retention.drain() == 0
    assert remover.removed == []


def test_drain_removes_unheld_per_challenge_images() -> None:
    remover = _Remover(freed=7)
    config = ResourceConfig(retention=RetentionPolicy.PER_CHALLENGE)
    retention = ImageRetention(config, remove_fn=remover, refs_for_ids=lambda ids: [])
    retention.acquire("img-a")
    retention.release("img-a")
    assert retention.drain() == 7
    assert remover.removed == ["img-a"]
    assert retention.freed_bytes == 7

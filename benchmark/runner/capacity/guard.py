"""One-call context manager wrapping a benchmark run with capacity management."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from benchmark.runner.capacity.disk import docker_root, free_bytes, preflight
from benchmark.runner.capacity.prefetch import Prefetcher, PullFn
from benchmark.runner.capacity.pull import pull_image
from benchmark.runner.capacity.retention import ImageRetention, RefsForIdsFn, RemoveFn
from benchmark.runner.capacity.sizes import (
    Fetch,
    LocalFn,
    RegistryFn,
    image_refs_for_ids,
    local_size,
    registry_size,
)
from benchmark.runner.capacity.types import GIB, CapacityVerdict, ResourceConfig


class InsufficientDiskError(RuntimeError):
    """Raised when the planned image set cannot fit on the docker root."""

    def __init__(self, verdict: CapacityVerdict) -> None:
        super().__init__(_message(verdict))
        self.verdict = verdict


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def _message(verdict: CapacityVerdict) -> str:
    """A human-readable shortfall summary including the recommended remedies."""
    lines = [
        f"insufficient disk at {verdict.docker_root}: need {_gib(verdict.needed_bytes)}, "
        f"have {_gib(verdict.free_bytes)} (short {_gib(verdict.shortfall_bytes)})."
    ]
    lines.extend(f"- {remedy}" for remedy in verdict.remedies)
    return "\n".join(lines)


@dataclass(frozen=True)
class CapacityHandle:
    """What a wrapped run may use: retention control, the prefetcher and the verdict."""

    retention: ImageRetention
    prefetch: Prefetcher | None
    verdict: CapacityVerdict


@contextmanager
def capacity_manager(
    refs: Iterable[str],
    *,
    config: ResourceConfig,
    pull_fn: PullFn | None = None,
    fetch: Fetch | None = None,
    docker_root_fn: Callable[[], str] = docker_root,
    free_fn: Callable[[str], int] = free_bytes,
    registry_fn: RegistryFn = registry_size,
    local_fn: LocalFn = local_size,
    remove_fn: RemoveFn | None = None,
    refs_for_ids: RefsForIdsFn = image_refs_for_ids,
) -> Iterator[CapacityHandle]:
    """Preflight, optionally prefetch, then drain/delete on exit.

    Raises :class:`InsufficientDiskError` unless the disk check passes or
    ``config.allow_insufficient_disk`` is set. Under PER_CHALLENGE retention the
    run's leftover images are removed on exit.
    """
    ref_list = list(refs)
    verdict = preflight(
        ref_list,
        config=config,
        fetch=fetch,
        docker_root_fn=docker_root_fn,
        free_fn=free_fn,
        registry_fn=registry_fn,
        local_fn=local_fn,
    )
    if not verdict.ok:
        raise InsufficientDiskError(verdict)
    retention = ImageRetention(config, remove_fn=remove_fn, refs_for_ids=refs_for_ids)
    prefetch = _build_prefetch(
        ref_list,
        config=config,
        verdict=verdict,
        pull_fn=pull_fn,
        fetch=fetch,
        docker_root_fn=docker_root_fn,
        free_fn=free_fn,
        local_fn=local_fn,
    )
    try:
        yield CapacityHandle(retention=retention, prefetch=prefetch, verdict=verdict)
    finally:
        if prefetch is not None:
            prefetch.close()
        retention.drain()


def _build_prefetch(
    refs: list[str],
    *,
    config: ResourceConfig,
    verdict: CapacityVerdict,
    pull_fn: PullFn | None,
    fetch: Fetch | None,
    docker_root_fn: Callable[[], str],
    free_fn: Callable[[str], int],
    local_fn: LocalFn,
) -> Prefetcher | None:
    """Construct and start a prefetcher, or None when prefetch is disabled."""
    if not config.prefetch_enabled:
        return None

    def default_pull(ref: str) -> bool:
        return pull_image(ref, fetch=fetch, local_fn=local_fn)

    def disk_free() -> int:
        return free_fn(docker_root_fn())

    prefetch = Prefetcher(
        refs,
        config=config,
        sizes=verdict.per_image,
        pull_fn=pull_fn or default_pull,
        free_fn=disk_free,
    )
    prefetch.start()
    return prefetch

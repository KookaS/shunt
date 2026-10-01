"""Pull one image, GHCR-first then Docker Hub, with 429 backoff."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

from benchmark.runner.capacity._exec import Runner, subprocess_runner
from benchmark.runner.capacity.sizes import (
    Fetch,
    ghcr_ref,
    instance_id_from_ref,
    local_size,
)

LocalFn = Callable[[str], int | None]
SleepFn = Callable[[float], None]

_RATE_LIMIT_MARKERS: Final[tuple[str, ...]] = ("429", "too many requests", "toomanyrequests")


def _sleep_seconds(attempt: int, delay: float, backoff_base: float) -> float:
    """Exponential backoff seeded by ``delay`` (falling back to the cap) and capped."""
    first = delay if delay > 0 else backoff_base
    return min(backoff_base, first * (2**attempt))


def _is_rate_limited(output: str) -> bool:
    """True when a failed pull's output looks like a registry throttle."""
    lowered = output.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def _pull_candidate(
    candidate: str,
    *,
    run: Runner,
    sleep: SleepFn,
    delay: float,
    backoff_base: float,
    max_429_retries: int,
) -> bool:
    """Pull one ref, retrying only on registry throttling."""
    attempt = 0
    while True:
        code, out = run(["docker", "pull", candidate])
        if code == 0:
            return True
        if not _is_rate_limited(out) or attempt >= max_429_retries:
            return False
        sleep(_sleep_seconds(attempt, delay, backoff_base))
        attempt += 1


def pull_image(
    ref: str,
    *,
    instance_id: str | None = None,
    fetch: Fetch | None = None,
    delay: float = 0.0,
    max_429_retries: int = 6,
    backoff_base: float = 300.0,
    local_fn: LocalFn = local_size,
    run: Runner = subprocess_runner,
    sleep: SleepFn = time.sleep,
) -> bool:
    """Ensure ``ref`` is local; try its GHCR mirror first, then Docker Hub.

    A successful GHCR pull is tagged back to ``ref`` so the SWE-bench harness finds
    the image under the name it expects. ``fetch`` is accepted as a uniform seam for
    callers that already hold one but is unused: pulls go through the docker CLI, not
    the registry API. Returns True when the image is present.
    """
    if local_fn(ref) is not None:
        return True
    runner_refs = ghcr_ref(instance_id or instance_id_from_ref(ref))
    for candidate in (runner_refs, ref):
        if not _pull_candidate(
            candidate,
            run=run,
            sleep=sleep,
            delay=delay,
            backoff_base=backoff_base,
            max_429_retries=max_429_retries,
        ):
            continue
        if candidate != ref:
            run(["docker", "tag", candidate, ref])
        return True
    return False

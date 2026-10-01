"""RAM-aware worker cap: refuse to launch containers a host cannot hold.

OOM is the FATAL direction. An over-provisioned worker pool gets a container
``SIGKILL``-ed (exit 137), which costs the whole trajectory — not merely wall-clock — and
the kill is invisible to every status-based retry. Under-provisioning workers only slows a
run down. The bound is therefore a HIGH percentile (p90) of the observed per-container
peaks, inflated by a safety margin, never the mean: the mean hides the heavy tail that
actually triggers the OOM, and a cluster of small containers must not license one large
one.

The guard is self-contained (no ``capacity/`` dependency) so an entrypoint can adopt it on
its own; wiring is the integrator's job. ``cap_workers`` is pure arithmetic over injected
inputs, and ``MemoryGuard`` owns the live seam: it reads ``MemAvailable``, records the
peaks of finished containers, and refuses — before any container starts — when even one
cannot fit.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_BYTES_PER_GIB: Final[int] = 1024**3
_MEMINFO_PATH: Final[Path] = Path("/proc/meminfo")


class InsufficientMemoryError(RuntimeError):
    """The host cannot hold even one container, so the run must refuse to start.

    The refusal is HARD: the caller must abort before launching anything rather than clamp
    to zero (or, worse, launch the one container the arithmetic says will not fit).
    """

    def __init__(
        self, *, requested: int, effective: int, available_bytes: int, bound_gib: float
    ) -> None:
        self.requested = requested
        self.effective = effective
        self.available_bytes = available_bytes
        self.bound_gib = bound_gib
        super().__init__(
            f"insufficient memory: requested={requested} workers, effective={effective}, "
            f"available={available_bytes} bytes ({available_bytes / _BYTES_PER_GIB:.2f} GiB), "
            f"per-container bound={bound_gib:.3f} GiB. Cannot fit one container; refusing "
            "before any container starts."
        )


@dataclass(frozen=True)
class MemoryConfig:
    """Per-container memory assumptions and the sampling/margin policy."""

    container_gib: float = 1.5
    reserve_gib: float = 2.0
    sample_containers: int = 3
    percentile: float = 0.9
    safety_margin: float = 0.2


def read_mem_available_bytes(
    path: str | Path = _MEMINFO_PATH,
    *,
    read_text: Callable[[Path], str] | None = None,
) -> int:
    """``MemAvailable`` from ``/proc/meminfo`` in bytes.

    ``path`` and ``read_text`` are injectable so tests never touch the real procfs. Raises
    ``ValueError`` when the kernel publishes no ``MemAvailable`` line — a missing number
    must refuse rather than be guessed.
    """
    reader = Path.read_text if read_text is None else read_text
    for line in reader(Path(path)).splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            break
        return int(fields[1]) * 1024
    raise ValueError(f"no MemAvailable line in {path}")


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile (numpy's default method) of *values*."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def _bound_gib(config: MemoryConfig, observed_peak_gib: Sequence[float]) -> float:
    """GiB one container is assumed to need: observed high percentile + margin, else default.

    Only the FIRST ``sample_containers`` observations shape the bound, so a long run's
    bound stabilises after the opening wave instead of creeping with every ``record``.
    """
    sample = tuple(observed_peak_gib)[: max(0, config.sample_containers)]
    if not sample:
        return config.container_gib
    return _percentile(sample, config.percentile) * (1.0 + config.safety_margin)


def cap_workers(
    requested: int,
    *,
    config: MemoryConfig,
    available_bytes: int | None = None,
    observed_peak_gib: Sequence[float] = (),
) -> int:
    """The worker count to actually run: ``requested`` capped by what memory can hold.

    Raises ``InsufficientMemoryError`` when even one container cannot fit, rather than
    clamping to zero or launching the doomed container.
    """
    available = read_mem_available_bytes() if available_bytes is None else available_bytes
    bound_gib = _bound_gib(config, observed_peak_gib)
    headroom = available - int(config.reserve_gib * _BYTES_PER_GIB)
    fits = math.floor(headroom / (bound_gib * _BYTES_PER_GIB))
    if fits < 1:
        raise InsufficientMemoryError(
            requested=requested, effective=0, available_bytes=available, bound_gib=bound_gib
        )
    return min(requested, max(1, fits))


class MemoryGuard:
    """Stateful memory seam: reads availability and refines the bound from observations."""

    def __init__(
        self,
        *,
        config: MemoryConfig,
        requested_workers: int,
        available_bytes_fn: Callable[[], int] = read_mem_available_bytes,
    ) -> None:
        self._config = config
        self._requested = requested_workers
        self._available_bytes_fn = available_bytes_fn
        self._lock = threading.Lock()
        self._observed: list[float] = []

    @property
    def observed_peak_gib(self) -> tuple[float, ...]:
        """Snapshot of recorded per-container peaks, taken under the lock."""
        with self._lock:
            return tuple(self._observed)

    def record(self, peak_gib: float) -> None:
        """Append one finished container's peak, thread-safe (workers record concurrently)."""
        with self._lock:
            self._observed.append(float(peak_gib))

    def check_or_raise(self) -> int:
        """``cap_workers`` for this guard's request; raises rather than starting a doomed run."""
        return cap_workers(
            self._requested,
            config=self._config,
            available_bytes=self._available_bytes_fn(),
            observed_peak_gib=self.observed_peak_gib,
        )

    @property
    def effective_workers(self) -> int:
        """The capped worker count, validated against current memory and observations."""
        return self.check_or_raise()


__all__ = [
    "InsufficientMemoryError",
    "MemoryConfig",
    "MemoryGuard",
    "cap_workers",
    "read_mem_available_bytes",
]

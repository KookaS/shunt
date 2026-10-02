"""RAM-aware worker cap: refuse to launch containers a host cannot hold.

OOM is the FATAL direction. An over-provisioned worker pool gets a container
``SIGKILL``-ed (exit 137), which costs the whole trajectory — not merely wall-clock — and
the kill is invisible to every status-based retry. Under-provisioning workers only slows a
run down. The bound is therefore a conservative per-container size from
``resources.container_memory_gib`` (never the mean of a few observed containers): a
cluster of small containers must not license one large one.

The guard is self-contained (no ``capacity/`` dependency) so an entrypoint can adopt it on
its own. ``cap_workers`` is pure arithmetic over injected inputs, and ``MemoryGuard`` owns
the live seam: it reads ``MemAvailable`` and refuses — before any container starts — when
even one cannot fit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

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
    """Per-container memory assumptions for the worker cap."""

    container_gib: float = 1.5
    reserve_gib: float = 2.0

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None = None, *, base: MemoryConfig | None = None
    ) -> MemoryConfig:
        """Overlay the ``resources:`` memory keys; raise on a non-numeric or negative value."""
        cfg = base or cls()
        if not data:
            return cfg
        return cls(
            container_gib=_as_gib(
                data.get("container_memory_gib"), cfg.container_gib, "container_memory_gib"
            ),
            reserve_gib=_as_gib(
                data.get("memory_reserve_gib"), cfg.reserve_gib, "memory_reserve_gib"
            ),
        )


def _as_gib(value: object, default: float, name: str) -> float:
    """Coerce a resources memory field to a non-negative float, preserving ``default`` on None."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a number, got {value!r}")
    try:
        result = float(cast("float | int | str", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a number, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{name}: must be >= 0, got {result}")
    return result


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


def cap_workers(
    requested: int,
    *,
    config: MemoryConfig,
    available_bytes: int | None = None,
) -> int:
    """The worker count to actually run: ``requested`` capped by what memory can hold.

    Raises ``InsufficientMemoryError`` when even one container cannot fit, rather than
    clamping to zero or launching the doomed container.
    """
    available = read_mem_available_bytes() if available_bytes is None else available_bytes
    bound_gib = config.container_gib
    headroom = available - int(config.reserve_gib * _BYTES_PER_GIB)
    fits = math.floor(headroom / (bound_gib * _BYTES_PER_GIB))
    if fits < 1:
        raise InsufficientMemoryError(
            requested=requested, effective=0, available_bytes=available, bound_gib=bound_gib
        )
    return min(requested, max(1, fits))


class MemoryGuard:
    """Stateful memory seam: reads availability and reports the capped worker count."""

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

    def check_or_raise(self) -> int:
        """``cap_workers`` for this guard's request; raises rather than starting a doomed run."""
        return cap_workers(
            self._requested,
            config=self._config,
            available_bytes=self._available_bytes_fn(),
        )

    @property
    def effective_workers(self) -> int:
        """The capped worker count, validated against current memory."""
        return self.check_or_raise()


__all__ = [
    "InsufficientMemoryError",
    "MemoryConfig",
    "MemoryGuard",
    "cap_workers",
    "read_mem_available_bytes",
]

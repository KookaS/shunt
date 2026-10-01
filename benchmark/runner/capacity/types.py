"""Frozen value types for the container/image capacity manager."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, cast

GIB: Final[int] = 1024**3

# Where an image's uncompressed footprint came from: a real local inspect, a
# registry manifest scaled by the calibration factor, or the configured fallback.
ImageSource = Literal["local", "registry", "assumed"]


class RetentionPolicy(Enum):
    """How long a pulled image is kept on disk after a challenge finishes."""

    KEEP = "keep"
    PER_CHALLENGE = "per-challenge"

    @classmethod
    def parse(cls, value: object) -> RetentionPolicy:
        """Coerce a policy name (or member) to a member; anything unknown means KEEP."""
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("_", "-")
        for member in cls:
            if member.value == text:
                return member
        return cls.KEEP


@dataclass(frozen=True)
class ImageSize:
    """One image's on-disk footprint: what we know, and how we know it."""

    ref: str
    cached: bool
    compressed_bytes: int | None
    uncompressed_estimate_bytes: int
    source: ImageSource


@dataclass(frozen=True)
class CapacityVerdict:
    """The disk preflight result for a planned image set."""

    ok: bool
    needed_bytes: int
    free_bytes: int
    shortfall_bytes: int
    per_image: tuple[ImageSize, ...]
    docker_root: str
    remedies: tuple[str, ...]


@dataclass(frozen=True)
class ResourceConfig:
    """Tunables for disk preflight, prefetch and per-challenge retention."""

    retention: RetentionPolicy = RetentionPolicy.KEEP
    allow_insufficient_disk: bool = False
    disk_safety_factor: float = 1.5
    disk_reserve_gib: float = 10.0
    assumed_image_gib: float = 3.0
    prefetch_window: int = 5
    prefetch_workers: int = 2
    prefetch_enabled: bool = False

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None = None, *, base: ResourceConfig | None = None
    ) -> ResourceConfig:
        """Overlay a config mapping onto ``base`` (or the defaults)."""
        cfg = base or cls()
        if not data:
            return cfg
        retention = data.get("retention", None)
        return cls(
            retention=(
                RetentionPolicy.parse(retention) if retention is not None else cfg.retention
            ),
            allow_insufficient_disk=_as_bool(
                data.get("allow_insufficient_disk"), cfg.allow_insufficient_disk
            ),
            disk_safety_factor=_as_float(data.get("disk_safety_factor"), cfg.disk_safety_factor),
            disk_reserve_gib=_as_float(data.get("disk_reserve_gib"), cfg.disk_reserve_gib),
            assumed_image_gib=_as_float(data.get("assumed_image_gib"), cfg.assumed_image_gib),
            prefetch_window=_as_int(data.get("prefetch_window"), cfg.prefetch_window),
            prefetch_workers=_as_int(data.get("prefetch_workers"), cfg.prefetch_workers),
            prefetch_enabled=_as_bool(data.get("prefetch_enabled"), cfg.prefetch_enabled),
        )

    @classmethod
    def from_args(
        cls,
        args: object,
        *,
        mapping: Mapping[str, object] | None = None,
        base: ResourceConfig | None = None,
    ) -> ResourceConfig:
        """Config mapping first, then non-None CLI flags over it.

        Entrypoints should declare tri-state flags with ``default=None`` so an
        absent flag does not overwrite the configured value.
        """
        cfg = cls.from_mapping(mapping, base=base)
        retention = getattr(args, "retention", None)
        return cls(
            retention=(
                RetentionPolicy.parse(retention) if retention is not None else cfg.retention
            ),
            allow_insufficient_disk=_as_bool(
                getattr(args, "allow_insufficient_disk", None), cfg.allow_insufficient_disk
            ),
            disk_safety_factor=_as_float(
                getattr(args, "disk_safety_factor", None), cfg.disk_safety_factor
            ),
            disk_reserve_gib=_as_float(
                getattr(args, "disk_reserve_gib", None), cfg.disk_reserve_gib
            ),
            assumed_image_gib=_as_float(
                getattr(args, "assumed_image_gib", None), cfg.assumed_image_gib
            ),
            prefetch_window=_as_int(getattr(args, "prefetch_window", None), cfg.prefetch_window),
            prefetch_workers=_as_int(getattr(args, "prefetch_workers", None), cfg.prefetch_workers),
            prefetch_enabled=_as_bool(
                getattr(args, "prefetch_enabled", None), cfg.prefetch_enabled
            ),
        )


def _as_bool(value: object, default: bool) -> bool:
    """Coerce a config/CLI value to bool, preserving ``default`` on None."""
    return default if value is None else bool(value)


def _as_float(value: object, default: float) -> float:
    """Coerce a config/CLI value to float, preserving ``default`` on None."""
    if value is None:
        return default
    return float(cast("float | int | str", value))


def _as_int(value: object, default: int) -> int:
    """Coerce a config/CLI value to int, preserving ``default`` on None."""
    if value is None:
        return default
    return int(cast("float | int | str", value))

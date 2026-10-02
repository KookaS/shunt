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

# Every key the `resources:` block may carry, across ResourceConfig and the memory
# keys MemoryConfig consumes. The union lives here so `ResourceConfig.from_mapping`
# can reject a typo in ANY of them without importing the memory-guard module.
_RESOURCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "retention",
        "allow_insufficient_disk",
        "disk_safety_factor",
        "disk_reserve_gib",
        "assumed_image_gib",
        "prefetch",
        "prefetch_enabled",
        "prefetch_workers",
        "prefetch_wait_timeout_s",
        "container_memory_gib",
        "memory_reserve_gib",
    }
)


class RetentionPolicy(Enum):
    """How long a pulled image is kept on disk after a challenge finishes."""

    KEEP = "keep"
    PER_CHALLENGE = "per-challenge"

    @classmethod
    def parse(cls, value: object) -> RetentionPolicy:
        """Coerce a policy name (or member) to a member; raise on anything unknown."""
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("_", "-")
        for member in cls:
            if member.value == text:
                return member
        valid = ", ".join(member.value for member in cls)
        raise ValueError(f"unknown retention {value!r}; expected one of: {valid}")


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
    prefetch_workers: int = 2
    prefetch_enabled: bool = False
    prefetch_wait_timeout_s: float = 600.0

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None = None, *, base: ResourceConfig | None = None
    ) -> ResourceConfig:
        """Overlay a config mapping onto ``base`` (or the defaults).

        STRICT: an unknown key, an unknown ``retention`` value, or a non-numeric /
        negative numeric value raises ``ValueError`` so a typo cannot silently fall
        back to a default. The accepted key set spans the whole ``resources:`` block
        (including the memory keys ``MemoryConfig`` reads) so this can validate it.
        """
        cfg = base or cls()
        if not data:
            return cfg
        unknown = set(data) - _RESOURCE_KEYS
        if unknown:
            raise ValueError(f"unknown resources key(s): {', '.join(sorted(unknown))}")
        retention_raw = data.get("retention")
        prefetch_raw = data.get("prefetch_enabled", data.get("prefetch"))
        return cls(
            retention=(
                cfg.retention if retention_raw is None else RetentionPolicy.parse(retention_raw)
            ),
            allow_insufficient_disk=_as_bool(
                data.get("allow_insufficient_disk"),
                cfg.allow_insufficient_disk,
                name="allow_insufficient_disk",
            ),
            disk_safety_factor=_as_float(
                data.get("disk_safety_factor"), cfg.disk_safety_factor, name="disk_safety_factor"
            ),
            disk_reserve_gib=_as_float(
                data.get("disk_reserve_gib"), cfg.disk_reserve_gib, name="disk_reserve_gib"
            ),
            assumed_image_gib=_as_float(
                data.get("assumed_image_gib"), cfg.assumed_image_gib, name="assumed_image_gib"
            ),
            prefetch_workers=_as_int(
                data.get("prefetch_workers"), cfg.prefetch_workers, name="prefetch_workers"
            ),
            prefetch_enabled=_as_bool(prefetch_raw, cfg.prefetch_enabled, name="prefetch"),
            prefetch_wait_timeout_s=_as_float(
                data.get("prefetch_wait_timeout_s"),
                cfg.prefetch_wait_timeout_s,
                name="prefetch_wait_timeout_s",
            ),
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
                getattr(args, "allow_insufficient_disk", None),
                cfg.allow_insufficient_disk,
                name="allow_insufficient_disk",
            ),
            disk_safety_factor=_as_float(
                getattr(args, "disk_safety_factor", None),
                cfg.disk_safety_factor,
                name="disk_safety_factor",
            ),
            disk_reserve_gib=_as_float(
                getattr(args, "disk_reserve_gib", None),
                cfg.disk_reserve_gib,
                name="disk_reserve_gib",
            ),
            assumed_image_gib=_as_float(
                getattr(args, "assumed_image_gib", None),
                cfg.assumed_image_gib,
                name="assumed_image_gib",
            ),
            prefetch_workers=_as_int(
                getattr(args, "prefetch_workers", None),
                cfg.prefetch_workers,
                name="prefetch_workers",
            ),
            prefetch_enabled=_as_bool(
                getattr(args, "prefetch_enabled", None), cfg.prefetch_enabled, name="prefetch"
            ),
            prefetch_wait_timeout_s=_as_float(
                getattr(args, "prefetch_wait_timeout_s", None),
                cfg.prefetch_wait_timeout_s,
                name="prefetch_wait_timeout_s",
            ),
        )


def _as_bool(value: object, default: bool, *, name: str) -> bool:
    """Coerce a config/CLI value to bool, preserving ``default`` on None; else STRICT."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name}: expected true/false, got {value!r}")
    return value


def _as_float(value: object, default: float, *, name: str) -> float:
    """Coerce a config/CLI value to a non-negative float, preserving ``default`` on None."""
    if value is None:
        return default
    if isinstance(value, bool):  # bool is an int subclass; a bool here is a typo, not a number
        raise ValueError(f"{name}: expected a number, got {value!r}")
    try:
        result = float(cast("float | int | str", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a number, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{name}: must be >= 0, got {result}")
    return result


def _as_int(value: object, default: int, *, name: str) -> int:
    """Coerce a config/CLI value to a non-negative int, preserving ``default`` on None."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    try:
        number = float(cast("float | int | str", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected an integer, got {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    result = int(number)
    if result < 0:
        raise ValueError(f"{name}: must be >= 0, got {result}")
    return result

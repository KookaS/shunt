"""Container/image capacity manager: preflight, prefetch, disk-aware retention.

Entrypoints wrap a run with :func:`capacity_manager`; everything is injectable so
the package's tests run offline (no docker, no network).
"""

from __future__ import annotations

from benchmark.runner.capacity.disk import docker_root, free_bytes, preflight
from benchmark.runner.capacity.guard import (
    CapacityHandle,
    InsufficientDiskError,
    capacity_manager,
)
from benchmark.runner.capacity.prefetch import Prefetcher
from benchmark.runner.capacity.pull import pull_image
from benchmark.runner.capacity.retention import ImageRetention, docker_rmi
from benchmark.runner.capacity.sizes import (
    SizeResolver,
    ghcr_ref,
    image_refs_for_ids,
    instance_id_from_ref,
    local_size,
    registry_size,
    resolve_sizes,
)
from benchmark.runner.capacity.types import (
    GIB,
    CapacityVerdict,
    ImageSize,
    ResourceConfig,
    RetentionPolicy,
)

__all__ = [
    "GIB",
    "CapacityHandle",
    "CapacityVerdict",
    "ImageRetention",
    "ImageSize",
    "InsufficientDiskError",
    "Prefetcher",
    "ResourceConfig",
    "RetentionPolicy",
    "SizeResolver",
    "capacity_manager",
    "docker_rmi",
    "docker_root",
    "free_bytes",
    "ghcr_ref",
    "image_refs_for_ids",
    "instance_id_from_ref",
    "local_size",
    "preflight",
    "pull_image",
    "registry_size",
    "resolve_sizes",
]

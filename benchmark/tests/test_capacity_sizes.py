"""Offline tests for capacity size resolution and registry manifest parsing."""

from __future__ import annotations

import json
from collections.abc import Mapping

from benchmark.runner.capacity.sizes import (
    SizeResolver,
    ghcr_ref,
    image_refs_for_ids,
    instance_id_from_ref,
    local_size,
    registry_size,
    resolve_sizes,
)
from benchmark.runner.capacity.types import GIB, ResourceConfig

TOKEN = json.dumps({"token": "anon-token"}).encode()
INDEX = json.dumps(
    {
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {"digest": "sha256:amd", "platform": {"os": "linux", "architecture": "amd64"}},
            {"digest": "sha256:arm", "platform": {"os": "linux", "architecture": "arm64"}},
        ],
    }
).encode()
MANIFEST = json.dumps(
    {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"size": 1_000},
        "layers": [{"size": 10_000}, {"size": 20_000}],
    }
).encode()
EMPTY_INDEX = json.dumps(
    {
        "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
        "manifests": [
            {"digest": "sha256:arm", "platform": {"os": "linux", "architecture": "arm64"}}
        ],
    }
).encode()

HUB_REF = "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
GHCR_REF = ghcr_ref("django__django-11099")


def _fetch(url: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
    """A deterministic registry: Docker Hub index, one amd64 child manifest."""
    if "auth.docker.io/token" in url:
        return 200, TOKEN
    if url.endswith("/manifests/latest"):
        return 200, INDEX
    if url.endswith("/manifests/sha256:amd"):
        return 200, MANIFEST
    return 404, b""


def test_image_refs_and_ghcr_names() -> None:
    refs = image_refs_for_ids(["django__django-11099"])
    assert refs == ["swebench/sweb.eval.x86_64.django_1776_django-11099:latest"]
    assert GHCR_REF == "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-11099:latest"
    assert instance_id_from_ref(HUB_REF) == "django__django-11099"
    assert instance_id_from_ref(GHCR_REF) == "django__django-11099"


def test_registry_parses_index_to_amd64_and_sums_layers() -> None:
    assert registry_size(HUB_REF, fetch=_fetch) == 31_000
    assert registry_size(GHCR_REF, fetch=_fetch) == 31_000


def test_registry_returns_none_when_no_amd64_entry() -> None:
    def no_amd64(url: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
        if "auth.docker.io/token" in url:
            return 200, TOKEN
        if url.endswith("/manifests/latest"):
            return 200, EMPTY_INDEX
        return 404, b""

    assert registry_size(HUB_REF, fetch=no_amd64) is None


def test_registry_returns_none_on_unreachable_registry() -> None:
    assert registry_size(HUB_REF, fetch=lambda url, headers: (0, b"")) is None


def test_registry_uses_token_for_docker_hub_only() -> None:
    seen: list[str] = []

    def tracking(url: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
        seen.append(url)
        return _fetch(url, headers)

    registry_size(HUB_REF, fetch=tracking)
    assert any("auth.docker.io/token" in url for url in seen)
    seen.clear()
    registry_size(GHCR_REF, fetch=tracking)
    assert not any("auth.docker.io/token" in url for url in seen)


def test_local_size_parses_inspect_output() -> None:
    def runner(argv: list[str]) -> tuple[int, str]:
        assert argv[:3] == ["docker", "image", "inspect"]
        return 0, "12345\n"

    assert local_size("img", runner=runner) == 12345
    assert local_size("img", runner=lambda argv: (1, "")) is None


def test_resolve_sizes_cached_local_and_registry_scaled() -> None:
    config = ResourceConfig(disk_safety_factor=2.0, assumed_image_gib=3.0)

    def local(ref: str) -> int | None:
        return 5_000 if ref == "cached" else None

    def registry(ref: str, *, fetch: object = None) -> int | None:
        return 1_000

    sizes = SizeResolver(config=config, local_fn=local, registry_fn=registry).resolve(
        ["cached", "uncached"]
    )
    cached, uncached = sizes
    assert (cached.cached, cached.source, cached.uncompressed_estimate_bytes) == (
        True,
        "local",
        5_000,
    )
    assert (uncached.cached, uncached.source, uncached.compressed_bytes) == (
        False,
        "registry",
        1_000,
    )
    assert uncached.uncompressed_estimate_bytes == 2_000


def test_resolve_sizes_assumed_fallback_when_registry_unavailable() -> None:
    config = ResourceConfig(assumed_image_gib=4.0)
    sizes = resolve_sizes(["img"], config=config, fetch=lambda url, headers: (0, b""))
    assert sizes[0].source == "assumed"
    assert sizes[0].uncompressed_estimate_bytes == int(4.0 * GIB)


def test_online_calibration_updates_the_factor() -> None:
    config = ResourceConfig(disk_safety_factor=1.5)
    resolver = SizeResolver(
        config=config, local_fn=lambda ref: None, registry_fn=lambda ref, *, fetch=None: 1_000
    )
    assert resolver.factor == 1.5
    resolver.observe(1_000, 3_000)
    assert resolver.factor == 3.0
    resolver.observe(1_000, 5_000)
    assert resolver.factor == 4.0
    resolved = resolver.resolve(["img"])
    assert resolved[0].uncompressed_estimate_bytes == 4_000

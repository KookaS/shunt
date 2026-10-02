"""Resolve image references to on-disk footprints (local, registry, or assumed).

CPU/disk only: an injectable ``fetch`` keeps every path offline-testable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from benchmark.runner import swebench_specs
from benchmark.runner.capacity._exec import Runner, subprocess_runner
from benchmark.runner.capacity.types import GIB, ImageSize, ResourceConfig

_TIMEOUT_S: Final[float] = 30.0
_DOCKER_HUBS: Final[tuple[str, ...]] = ("docker.io", "index.docker.io", "registry-1.docker.io")

# An HTTP fetch: url + headers -> (status, body). Injectable; no network in tests.
Fetch = Callable[[str, Mapping[str, str]], tuple[int, bytes]]
LocalFn = Callable[[str], int | None]
RegistryFn = Callable[..., int | None]

_ACCEPT: Final[str] = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_INDEX_MEDIA: Final[tuple[str, ...]] = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)


def image_refs_for_ids(instance_ids: Iterable[str]) -> list[str]:
    """Map instance ids to the prebuilt image refs the harness pulls."""
    return [swebench_specs.image_ref(iid) for iid in instance_ids]


def ghcr_ref(instance_id: str) -> str:
    """The Epoch Research GHCR mirror of a SWE-bench instance image."""
    return f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id.lower()}:latest"


def instance_id_from_ref(ref: str) -> str:
    """Recover the SWE-bench instance id from a runner image ref."""
    tail = ref.rsplit("/", 1)[-1].split(":", 1)[0]
    for prefix in ("sweb.eval.x86_64.", "swe-bench.eval.x86_64."):
        if tail.startswith(prefix):
            tail = tail[len(prefix) :]
            break
    return tail.replace("_1776_", "__")


def _urllib_fetch(url: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
    """Default fetch: one bounded urllib request, HTTP errors returned, not raised."""
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError, ValueError):
        return 0, b""


def local_size(ref: str, *, runner: Runner = subprocess_runner) -> int | None:
    """Uncompressed local image size in bytes, or None when the image is absent."""
    code, out = runner(["docker", "image", "inspect", "--format", "{{.Size}}", ref])
    if code != 0:
        return None
    text = out.strip().splitlines()[0] if out.strip() else ""
    return int(text) if text.isdigit() else None


@dataclass(frozen=True)
class _RefParts:
    """A parsed image reference: registry host, repository name and tag."""

    registry: str
    name: str
    tag: str


def _parse_ref(ref: str) -> _RefParts:
    """Split ``[host/]name[:tag]`` into registry, repository and tag."""
    digest_parts = ref.split("@", 1)[0]
    if ":" in digest_parts.rsplit("/", 1)[-1]:
        name, tag = digest_parts.rsplit(":", 1)
    else:
        name, tag = digest_parts, "latest"
    head, _, rest = name.partition("/")
    if rest and ("." in head or ":" in head or head == "localhost"):
        return _RefParts(head, rest, tag)
    return _RefParts("docker.io", name, tag)


def _dockerhub_token(parts: _RefParts, fetch: Fetch) -> str | None:
    """Anonymous pull token for Docker Hub, or None when the endpoint refuses."""
    scope = f"repository:{parts.name}:pull"
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope={scope}"
    status, body = fetch(url, {})
    if status != 200:
        return None
    data = _load_json(body)
    if not isinstance(data, dict):
        return None
    token = data.get("token") or data.get("access_token")
    return str(token) if token else None


def _load_json(body: bytes) -> object | None:
    """Parse a JSON body, returning None on any decode error."""
    try:
        return json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _is_index(manifest: Mapping[str, object]) -> bool:
    """True when a manifest is an OCI index / Docker manifest list."""
    return str(manifest.get("mediaType") or "") in _INDEX_MEDIA


def _amd64_digest(index: Mapping[str, object]) -> str | None:
    """The manifest digest of a linux/amd64 entry in an image index."""
    entries = index.get("manifests")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        platform = entry.get("platform") or {}
        if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
            digest = entry.get("digest")
            if digest:
                return str(digest)
    return None


def _manifest_size(manifest: Mapping[str, object]) -> int | None:
    """Sum layer sizes plus the config blob size of a concrete manifest."""
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        return None
    total = 0
    for layer in layers:
        if isinstance(layer, dict) and layer.get("size") is not None:
            total += int(layer["size"])
    config = manifest.get("config")
    if isinstance(config, dict) and config.get("size") is not None:
        total += int(config["size"])
    return total


def _registry_base(parts: _RefParts) -> str:
    """The v2 API base for a registry host."""
    if parts.registry in _DOCKER_HUBS:
        return "https://registry-1.docker.io"
    return f"https://{parts.registry}"


def registry_size(ref: str, *, fetch: Fetch | None = None) -> int | None:
    """Compressed image size from the registry, or None when unavailable.

    Resolves an OCI/Docker image index down to its linux/amd64 manifest and sums
    the layer blobs plus the config. Docker Hub is queried with an anonymous bearer
    token; GHCR's registry endpoint refuses anonymous manifest requests (HTTP 401),
    so those refs fall through to the local inspect or the assumed fallback.
    """
    http = fetch or _urllib_fetch
    parts = _parse_ref(ref)
    headers = {"Accept": _ACCEPT}
    if parts.registry in _DOCKER_HUBS:
        token = _dockerhub_token(parts, http)
        if token is None:
            return None
        headers["Authorization"] = f"Bearer {token}"
    base = _registry_base(parts)
    status, body = http(f"{base}/v2/{parts.name}/manifests/{parts.tag}", headers)
    if status != 200:
        return None
    manifest = _load_json(body)
    if not isinstance(manifest, dict):
        return None
    if _is_index(manifest):
        digest = _amd64_digest(manifest)
        if digest is None:
            return None
        status, body = http(f"{base}/v2/{parts.name}/manifests/{digest}", headers)
        if status != 200:
            return None
        manifest = _load_json(body)
        if not isinstance(manifest, dict):
            return None
    return _manifest_size(manifest)


class SizeResolver:
    """Resolve image sizes from the local store, the registry, or the assumed fallback."""

    def __init__(
        self,
        *,
        config: ResourceConfig,
        fetch: Fetch | None = None,
        local_fn: LocalFn = local_size,
        registry_fn: RegistryFn = registry_size,
    ) -> None:
        self._config = config
        self._fetch = fetch
        self._local = local_fn
        self._registry = registry_fn
        self._factor = config.disk_safety_factor

    def resolve(self, refs: Iterable[str]) -> tuple[ImageSize, ...]:
        """Classify every ref as cached-local, registry-scaled, or assumed."""
        return tuple(self._resolve_one(ref) for ref in refs)

    def _resolve_one(self, ref: str) -> ImageSize:
        cached_size = self._local(ref)
        if cached_size is not None:
            return ImageSize(ref, True, None, cached_size, "local")
        compressed = self._registry(ref, fetch=self._fetch)
        if compressed is not None:
            return ImageSize(ref, False, compressed, int(compressed * self._factor), "registry")
        assumed = int(self._config.assumed_image_gib * GIB)
        return ImageSize(ref, False, None, assumed, "assumed")


def resolve_sizes(
    refs: Iterable[str], *, config: ResourceConfig, fetch: Fetch | None = None
) -> tuple[ImageSize, ...]:
    """Resolve a ref set with a fresh resolver seeded from ``config``."""
    return SizeResolver(config=config, fetch=fetch).resolve(refs)

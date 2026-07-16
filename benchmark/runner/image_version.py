"""Resolve + canonicalize SWE-bench image manifest digests for cache anchoring.

The MANIFEST digest (never the config digest) is a cell's immutable image anchor;
resolution failure NEVER invalidates — callers fall back to the stored digest.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

DIGEST_PREFIX = "sha256:"
_INSPECT_TIMEOUT = 60

# A runner takes an argv and returns (returncode, stdout); injectable for tests
# so unit tests never require a live docker daemon or the registry.
Runner = Callable[[list[str]], tuple[int, str]]


def _subprocess_runner(argv: list[str]) -> tuple[int, str]:
    """Default runner: shell out, capture stdout, never raise (missing docker → rc=1)."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_INSPECT_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return proc.returncode, proc.stdout


def canonical_digest(raw: str) -> str:
    """Canonicalize a digest to a bare ``sha256:...`` (strips any ``name@`` prefix)."""
    text = raw.strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    if "@" in text:
        text = text.rsplit("@", 1)[1].strip()
    return text if text.startswith(DIGEST_PREFIX) else ""


def resolve_manifest_digest(image_ref: str, runner: Runner = _subprocess_runner) -> str | None:
    """Resolve the registry MANIFEST digest for an image ref (no pull).

    Returns a bare ``sha256:...`` or None on ANY failure — a None must never mark a
    cell stale (offline / unreachable / yanked ⇒ recompute-forever otherwise).
    """
    argv = [
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        "--format",
        "{{.Manifest.Digest}}",
        image_ref,
    ]
    code, out = runner(argv)
    if code != 0:
        return None
    return canonical_digest(out) or None


def used_image_digest(image_ref: str, runner: Runner = _subprocess_runner) -> str | None:
    """Digest of the image the harness ACTUALLY used (post-run ``docker inspect``).

    swebench pulls by namespace+tag, so we record the local image's RepoDigest
    rather than forcing a digest-pin through its pull. None if undeterminable.
    """
    argv = ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", image_ref]
    code, out = runner(argv)
    if code != 0:
        return None
    return canonical_digest(out) or None


def resolve_spec_digests(
    image_refs: dict[str, str], runner: Runner = _subprocess_runner
) -> dict[str, str | None]:
    """Map each challenge id to its resolved manifest digest (None on failure)."""
    return {cid: resolve_manifest_digest(ref, runner) for cid, ref in image_refs.items()}

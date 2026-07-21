"""Shunt — tool-agnostic, cache-safe LLM router."""

from importlib.metadata import PackageNotFoundError, version

# Read the version from installed package metadata rather than a second literal.
# The release workflow rewrites pyproject's version from the git tag but never
# touched this file, so `shunt version` printed 0.0.0 in every published
# release. One source of truth; the fallback covers a source checkout that was
# never installed.
try:
    __version__ = version("shunt-router")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0+unknown"

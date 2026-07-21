"""Resolve and apply the process log level (`--log-level` > `SHUNT_LOG_LEVEL` > INFO)."""

from __future__ import annotations

import logging
import os
from typing import Final

DEFAULT_LEVEL: Final[str] = "INFO"
LEVELS: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL_ENV: Final[str] = "SHUNT_LOG_LEVEL"

# Shunt holds provider API keys in the request path, and these libraries log full
# request headers — including `Authorization` — at their own DEBUG level. Turning on
# Shunt's debug logs must never turn those on: it would write live credentials to the
# log file, defeating the redaction on every other path out of this process.
_CREDENTIAL_LOGGING_LIBRARIES: Final[tuple[str, ...]] = (
    "httpx",
    "httpcore",
    "openai",
    "litellm",
    "urllib3",
    "requests",
)
_LIBRARY_CEILING: Final[int] = logging.INFO

# Not a security matter, a legibility one: these emit hundreds of lines per model load
# (filelock alone logs every lock acquire/release), which buries the routing trace that
# debug mode exists to show.
_NOISY_LIBRARIES: Final[tuple[str, ...]] = (
    "filelock",
    "huggingface_hub",
    "fsspec",
    "asyncio",
    "matplotlib",
    "PIL",
)
_NOISY_CEILING: Final[int] = logging.WARNING

LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def resolve_log_level(override: str | None = None) -> str:
    """Resolve the effective level, loudest source first. Rejects an unknown name."""
    raw = override or os.environ.get(LOG_LEVEL_ENV) or DEFAULT_LEVEL
    level = raw.strip().upper()
    if level not in LEVELS:
        # Fail loud: a typo'd level that silently fell back to INFO would look like
        # "debug logging is broken" rather than "you misspelled it".
        raise ValueError(f"invalid log level {raw!r}; expected one of {', '.join(LEVELS)}")
    return level


def configure_logging(override: str | None = None) -> str:
    """Apply the resolved level to the root logger and return it."""
    level = resolve_log_level(override)
    logging.basicConfig(level=getattr(logging, level), format=LOG_FORMAT, force=True)
    numeric = getattr(logging, level)
    for name in _CREDENTIAL_LOGGING_LIBRARIES:
        logging.getLogger(name).setLevel(max(numeric, _LIBRARY_CEILING))
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(max(numeric, _NOISY_CEILING))
    return level

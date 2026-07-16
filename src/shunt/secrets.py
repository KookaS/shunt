"""Load a local ``.env`` file (KEY=value lines) into the environment."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILENAME = ".env"


def _resolve(path: str | Path | None) -> Path | None:
    """Pick the .env path: explicit arg, else $SHUNT_ENV_FILE, else ./.env."""
    if path is not None:
        return Path(path)
    env = os.environ.get("SHUNT_ENV_FILE")
    return Path(env) if env else Path.cwd() / DEFAULT_ENV_FILENAME


def load_dotenv_file(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Parse ``KEY=value`` lines from a .env file into ``os.environ``.

    Existing env vars are preserved unless ``override`` is set; a missing file is a
    no-op. Returns the keys applied (comments/blank lines ignored).
    """
    resolved = _resolve(path)
    if resolved is None or not resolved.is_file():
        return {}
    # Parse first (last occurrence wins, matching dotenv/shell `source`), then apply
    # with env-precedence: a pre-existing env var is never overwritten unless override.
    parsed: dict[str, str] = {}
    for raw in resolved.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):  # tolerate `export KEY=value` lines
            key = key[len("export ") :].strip()
        if not key:
            continue
        parsed[key] = value.strip().strip('"').strip("'")
    # Record only keys actually written — an env var that already exists wins and is
    # NOT in the returned map (so "applied" never misreports what reached the env).
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied

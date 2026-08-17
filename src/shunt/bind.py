"""The address the proxy binds — one definition, read by the server and by `shunt doctor`."""

# Deliberately dependency-free (stdlib only, no FastAPI) so the CLI's diagnosis path can import
# it without paying for the server stack. It exists because the defaults were previously inline
# literals in `proxy/server.py` and a second copy in the diagnostics module: if those drift,
# `shunt doctor` confidently reports the wrong port as free, which is worse than not checking.

from __future__ import annotations

import os
from typing import Final

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8080
HOST_ENV: Final = "SHUNT_HOST"
PORT_ENV: Final = "SHUNT_PORT"

_MAX_PORT: Final = 65535


def resolve_bind() -> tuple[str, int]:
    """The (host, port) `shunt start` will bind. Raises ValueError with an actionable message."""
    host = os.environ.get(HOST_ENV, DEFAULT_HOST)
    raw = os.environ.get(PORT_ENV, str(DEFAULT_PORT))
    try:
        port = int(raw)
    except ValueError:
        raise ValueError(f"{PORT_ENV} is not an integer: {raw!r}") from None
    # Range is checked here, not left to the socket layer: `connect_ex` raises OverflowError for
    # an out-of-range port, which surfaced as a crash rather than as a diagnosis.
    if not 0 <= port <= _MAX_PORT:
        raise ValueError(f"{PORT_ENV} must be between 0 and {_MAX_PORT}, got {raw!r}")
    return host, port

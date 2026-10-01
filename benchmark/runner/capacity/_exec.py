"""Shared subprocess seam for the capacity manager (injectable for offline tests)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Final

_TIMEOUT_S: Final[int] = 300

# An argv -> (returncode, stdout). Injectable so no test needs a live docker daemon.
Runner = Callable[[list[str]], tuple[int, str]]


def subprocess_runner(argv: list[str]) -> tuple[int, str]:
    """Shell out, capture stdout, never raise (missing docker -> rc=1, empty stdout)."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return proc.returncode, proc.stdout

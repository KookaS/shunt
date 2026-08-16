"""The encrypted local trajectory plane: redact (defense in depth) -> encrypt -> append."""

# Writes ONLY encrypted bytes to a LOCAL, gitignored directory outside the repo. Uses real
# authenticated encryption (Fernet/AES) from the optional `cryptography` extra — never a
# hand-rolled cipher — imported lazily so the core wheel stays lean and a deployment that never
# enables capture never needs it. The key comes from the environment / OS keyring, never the
# wire and never a committed file.

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from shunt.capture.trajectory import redact_record

if TYPE_CHECKING:
    from shunt.capture.trajectory import StepRecord

logger = logging.getLogger(__name__)


class _Cipher(Protocol):
    """The subset of Fernet the sink uses (keeps mypy strict without cryptography stubs)."""

    def encrypt(self, data: bytes) -> bytes: ...


_KEY_ENV = "SHUNT_ESCALATION_KEY"
_ENC_SUFFIX = ".traj.enc"


def resolve_live_dir(configured: str | None) -> Path:
    """The local plane dir: the configured path, else $SHUNT_HOME/trajectories, else ~/.shunt/…

    Always OUTSIDE the repo tree and gitignored; the recorder never writes full content in-repo.
    """
    if configured:
        return Path(configured).expanduser()
    home = os.environ.get("SHUNT_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".shunt"
    return base / "trajectories"


def load_key() -> bytes:
    """Read the Fernet key from the environment. Raises if capture is enabled without one."""
    key = os.environ.get(_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"full-content capture is enabled but {_KEY_ENV} is unset; set a Fernet key "
            "(never commit it) or disable capture.full_content"
        )
    return key.encode("utf-8")


class LiveTrajectorySink:
    """Encrypt redacted step records and append them to the local gitignored plane."""

    def __init__(self, live_dir: Path, key: bytes) -> None:
        self._dir = live_dir
        self._fernet: _Cipher = _make_fernet(key)

    def write(self, records: list[StepRecord]) -> None:
        """Redact (defense in depth) -> JSON -> encrypt -> append one ciphertext line per step."""
        if not records:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        _restrict(self._dir, 0o700)  # private plane: owner-only dir + files (may hold private code)
        path = self._dir / f"session{_ENC_SUFFIX}"
        with path.open("ab") as handle:
            _restrict(path, 0o600)
            for record in records:
                payload = json.dumps(asdict(redact_record(record)), sort_keys=True).encode("utf-8")
                handle.write(self._fernet.encrypt(payload) + b"\n")


def _restrict(path: Path, mode: int) -> None:
    """Tighten permissions on the local encrypted plane (best-effort; POSIX-only)."""
    try:
        path.chmod(mode)
    except OSError:  # pragma: no cover - non-POSIX filesystem / unusual mount
        logger.debug("could not chmod %s to %o", path, mode)


def _make_fernet(key: bytes) -> _Cipher:
    """Build a Fernet cipher from the optional `cryptography` extra (lazy import)."""
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "full-content capture needs the 'capture' extra: pip install 'shunt-router[capture]'"
        ) from exc
    # cryptography ships no type stubs, so Fernet(...) is Any without the extra installed;
    # cast to the Protocol keeps strict mypy green whether or not `capture` is installed.
    return cast(_Cipher, Fernet(key))


__all__ = ["LiveTrajectorySink", "load_key", "resolve_live_dir"]

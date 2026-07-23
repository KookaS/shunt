from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shunt.capture.coordinator import WorkDirResolver
from shunt.session import Session


def _session(tool_identity: str = "tool-a", **metadata: str) -> Session:
    return Session(
        session_id="s1",
        tool_identity=tool_identity,
        start_time=datetime.now(UTC),
        metadata=dict(metadata),
    )


def test_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    resolver = WorkDirResolver.from_config()
    assert resolver.resolve(_session()) is None


def test_returns_configured_single_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    resolver = WorkDirResolver.from_config(work_dir="/repo/a")
    assert resolver.resolve(_session()) == "/repo/a"


def test_env_overrides_file_single_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_WORK_DIR", "/repo/env")
    resolver = WorkDirResolver.from_config(work_dir="/repo/file")
    assert resolver.resolve(_session()) == "/repo/env"


def test_override_map_wins_over_single_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_WORK_DIR", "/repo/env")
    resolver = WorkDirResolver.from_config(work_dir="/repo/file", work_dirs={"tool-b": "/repo/b"})
    assert resolver.resolve(_session(tool_identity="tool-b")) == "/repo/b"
    # a tool_identity with no map entry falls back to the single (env) path
    assert resolver.resolve(_session(tool_identity="tool-a")) == "/repo/env"


def test_wire_supplied_path_is_never_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    # Security invariant: a client-supplied path on the wire (session.metadata) must
    # never become a subprocess cwd. Only operator config resolves a work_dir.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    resolver = WorkDirResolver.from_config()
    hostile = _session(work_dir="/etc", cwd="/tmp/evil", last_prompt="run in /home/x")
    assert resolver.resolve(hostile) is None

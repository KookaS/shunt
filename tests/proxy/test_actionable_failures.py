"""A first-run failure must name the thing the operator has to change."""

# Previously: a missing key surfaced only as the provider's own "Incorrect API key"
# 401, naming no variable; a corrupt database aborted startup with a bare
# sqlite3.DatabaseError naming neither the file nor a remedy.

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from shunt.db.store import OutcomeStore, OutcomeStoreUnavailableError
from shunt.models.config import ModelPool
from shunt.proxy.server import _log_missing_credentials


def test_startup_names_every_unset_key_variable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pool = ModelPool()
    env_vars = {
        model.api_key_env_var
        for name in pool.model_names()
        if (model := pool.get_model(name)) is not None
    }
    assert env_vars, "fixture pool must declare key variables"
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    with caplog.at_level(logging.WARNING, logger="shunt.proxy.server"):
        _log_missing_credentials(pool)

    for var in env_vars:
        assert var in caplog.text, f"{var} not named at startup"


def test_no_warning_when_every_key_is_present(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pool = ModelPool()
    for name in pool.model_names():
        model = pool.get_model(name)
        if model is not None:
            monkeypatch.setenv(model.api_key_env_var, "x" * 20)

    with caplog.at_level(logging.INFO, logger="shunt.proxy.server"):
        _log_missing_credentials(pool)

    assert "NOT set" not in caplog.text
    assert "credentials present" in caplog.text


def test_a_corrupt_database_reports_the_path_and_a_remedy(tmp_path: Path) -> None:
    db_path = tmp_path / "outcomes.db"
    db_path.write_bytes(b"\x00\x01this is not a database\xff" * 64)

    with pytest.raises(OutcomeStoreUnavailableError) as raised:
        OutcomeStore(db_path=str(db_path))

    message = str(raised.value)
    assert str(db_path) in message
    assert "SHUNT_DATA_DIR" in message


def test_an_unwritable_location_reports_the_path(tmp_path: Path) -> None:
    locked = tmp_path / "ro"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(OutcomeStoreUnavailableError) as raised:
            OutcomeStore(db_path=str(locked / "sub" / "outcomes.db"))
        assert "outcomes.db" in str(raised.value)
    finally:
        locked.chmod(0o700)

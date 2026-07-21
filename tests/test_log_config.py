"""Log-level resolution, and the guard that keeps debug logging from leaking keys."""

from __future__ import annotations

import logging

import pytest

from shunt.log_config import (
    _CREDENTIAL_LOGGING_LIBRARIES,
    LOG_LEVEL_ENV,
    configure_logging,
    resolve_log_level,
)


def test_default_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    assert resolve_log_level() == "INFO"


def test_env_var_selects_the_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
    assert resolve_log_level() == "DEBUG"


def test_explicit_override_beats_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "error")
    assert resolve_log_level("debug") == "DEBUG"


def test_an_unknown_level_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    # Silently falling back to INFO would read as "debug logging is broken".
    monkeypatch.setenv(LOG_LEVEL_ENV, "verbose")
    with pytest.raises(ValueError, match="invalid log level"):
        resolve_log_level()


def test_debug_never_enables_libraries_that_log_auth_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The decisive one: httpx/openai DEBUG dumps request headers, Authorization
    # included. Shunt holds real provider keys, so turning on our debug logs must not
    # turn on theirs.
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    configure_logging("debug")
    try:
        assert logging.getLogger().level == logging.DEBUG
        for name in _CREDENTIAL_LOGGING_LIBRARIES:
            assert logging.getLogger(name).level >= logging.INFO, name
    finally:
        configure_logging("info")


def test_configure_returns_the_applied_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    try:
        assert configure_logging("warning") == "WARNING"
    finally:
        configure_logging("info")

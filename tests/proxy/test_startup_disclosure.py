"""The startup disclosure must not promise spending that cannot happen."""

from __future__ import annotations

import logging

import pytest

from shunt.proxy.server import _log_exploration_disclosure
from shunt.router.policy import CapturePolicy, ExplorationPolicy, RouterPolicy


def _policy(
    *, enabled: bool = True, strategy: str = "knn", work_dir: str | None = None
) -> RouterPolicy:
    return RouterPolicy(
        strategy=strategy,
        exploration=ExplorationPolicy(enabled=enabled),
        capture=CapturePolicy(work_dir=work_dir),
    )


def test_enabled_but_still_cold_starting_discloses_inert(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # While cold-start is active the engine returns before exploring, so a rig with SOME
    # outcomes but not enough is still inert. Announcing a "~1.4x envelope" there is a
    # false operational disclosure — observed live after the first flagged session.
    with caplog.at_level(logging.WARNING):
        _log_exploration_disclosure(_policy(), cold_start_active=True)

    message = caplog.text
    assert "INERT" in message
    assert "costs nothing extra" in message
    assert "1.4x" not in message


def test_enabled_past_cold_start_discloses_the_cost_envelope(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _log_exploration_disclosure(_policy(), cold_start_active=False)

    assert "1.4x" in caplog.text
    assert "INERT" not in caplog.text


def test_disabled_exploration_says_so_regardless_of_cold_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        _log_exploration_disclosure(_policy(enabled=False), cold_start_active=False)

    assert "exploration is OFF" in caplog.text


def test_fixed_strategy_never_claims_exploration(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        _log_exploration_disclosure(_policy(strategy="always_cheap"), cold_start_active=False)

    assert "exploration is OFF" in caplog.text


def test_manual_only_says_only_upward_exploration_can_fire(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With no work_dir the only outcome-write path (`shunt flag`) is a separate CLI
    # process, so the in-process gate never gets slack and downshift exploration cannot
    # fire — reporting conservative_alpha without saying so reads as a live safety valve.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    with caplog.at_level(logging.WARNING):
        _log_exploration_disclosure(_policy(), cold_start_active=False)

    assert "only explore UPWARD" in caplog.text
    assert "cheaper model" in caplog.text


def test_configured_work_dir_arms_the_downshift_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # With auto-capture configured, verified downshift outcomes feed the in-process gate
    # at session close, so it CAN open — the "cannot open" disclosure must not appear.
    with caplog.at_level(logging.WARNING):
        _log_exploration_disclosure(_policy(work_dir="/repo"), cold_start_active=False)

    assert "ARMED" in caplog.text
    assert "cannot open" not in caplog.text

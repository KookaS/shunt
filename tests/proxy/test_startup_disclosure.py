"""The startup disclosure must not promise spending that cannot happen."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from shunt.capture import WorkDirResolver
from shunt.proxy.server import (
    _build_work_dir_resolver,
    _log_capture_disclosure,
    _log_exploration_disclosure,
)
from shunt.router.policy import CapturePolicy, EscalationPolicy, ExplorationPolicy, RouterPolicy


def _policy(
    *,
    enabled: bool = True,
    strategy: str | None = None,
    work_dir: str | None = None,
    escalation_off: bool = False,
) -> RouterPolicy:
    # The shipped default is a cascade preset, and naming one explicitly with the ladder off is
    # a load error — so an escalation-off case has to name a NON-cascade strategy. That is not a
    # workaround: `always_cheap` is what an operator who turns the ladder off actually runs.
    if strategy is None:
        strategy = "always_cheap" if escalation_off else "knn_semantic_cascade"
    return RouterPolicy(
        strategy=strategy,
        exploration=ExplorationPolicy(enabled=enabled),
        escalation=EscalationPolicy(enabled=not escalation_off),
        capture=CapturePolicy(work_dir=work_dir),
    )


def _resolver(policy: RouterPolicy, launch_dir: str = "") -> WorkDirResolver:
    """The resolver the server would build — no launch dir unless a test supplies one."""
    return _build_work_dir_resolver(policy, launch_dir)


def test_enabled_but_still_cold_starting_discloses_inert(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # While cold-start is active the engine returns before exploring, so a rig with SOME
    # outcomes but not enough is still inert. Announcing a "~1.4x envelope" there is a
    # false operational disclosure — observed live after the first flagged session.
    with caplog.at_level(logging.WARNING):
        _log_exploration_disclosure(_policy(), _resolver(_policy()), cold_start_active=True)

    message = caplog.text
    assert "INERT" in message
    assert "costs nothing extra" in message
    assert "1.4x" not in message


def test_enabled_past_cold_start_discloses_the_cost_envelope(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _log_exploration_disclosure(_policy(), _resolver(_policy()), cold_start_active=False)

    assert "1.4x" in caplog.text
    assert "INERT" not in caplog.text


def test_disabled_exploration_says_so_regardless_of_cold_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        _log_exploration_disclosure(
            _policy(enabled=False), _resolver(_policy(enabled=False)), cold_start_active=False
        )

    assert "exploration is OFF" in caplog.text


def test_fixed_strategy_never_claims_exploration(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        _log_exploration_disclosure(
            _policy(strategy="always_cheap"),
            _resolver(_policy(strategy="always_cheap")),
            cold_start_active=False,
        )

    assert "exploration is OFF" in caplog.text


def test_manual_only_says_only_upward_exploration_can_fire(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With no work_dir the only outcome-write path (`shunt flag`) is a separate CLI
    # process, so the in-process gate never gets slack and downshift exploration cannot
    # fire — reporting conservative_alpha without saying so reads as a live safety valve.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    with caplog.at_level(logging.WARNING):
        _log_exploration_disclosure(_policy(), _resolver(_policy()), cold_start_active=False)

    assert "only explore UPWARD" in caplog.text
    assert "cheaper model" in caplog.text


def test_configured_work_dir_arms_the_downshift_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # With auto-capture configured, verified downshift outcomes feed the in-process gate
    # at session close, so it CAN open — the "cannot open" disclosure must not appear.
    with caplog.at_level(logging.WARNING):
        _log_exploration_disclosure(
            _policy(work_dir="/repo"), _resolver(_policy(work_dir="/repo")), cold_start_active=False
        )

    assert "ARMED" in caplog.text
    assert "cannot open" not in caplog.text


def test_escalation_enabled_without_work_dir_warns_not_armed(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Escalation ships ON, but its only signal is the repo's tests re-run off the wire. Without a
    # work_dir it is INERT — the guarantee is "never silently inert", so the boot disclosure must
    # say it is enabled but not armed (it is NOT a load error: enabled-without-a-work_dir is the
    # common default state).
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    with caplog.at_level(logging.WARNING):
        _log_capture_disclosure(_policy(), _resolver(_policy()))

    assert "Auto-escalation is ENABLED" in caplog.text
    assert "will NOT fire" in caplog.text
    assert "work_dir" in caplog.text


def test_escalation_enabled_with_work_dir_is_armed_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _log_capture_disclosure(_policy(work_dir="/repo"), _resolver(_policy(work_dir="/repo")))

    assert "Auto-escalation is ENABLED" not in caplog.text


def test_escalation_disabled_without_work_dir_does_not_warn(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    with caplog.at_level(logging.WARNING):
        _log_capture_disclosure(
            _policy(escalation_off=True), _resolver(_policy(escalation_off=True))
        )

    assert "Auto-escalation is ENABLED" not in caplog.text


# ── The launch-directory layer must be disclosed by NAME, and must fail closed ────


def _launch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    return repo


def test_capture_armed_by_the_launch_dir_says_which_layer_armed_it(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # "capture is ON" alone cannot be trusted by an operator who did not configure a repo:
    # the disclosure must name the directory whose test code is about to be executed.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _launch_repo(tmp_path)
    policy = RouterPolicy(capture=CapturePolicy())
    with caplog.at_level(logging.INFO):
        _log_capture_disclosure(policy, _build_work_dir_resolver(policy, str(repo)))

    assert "capture is ON via launch directory" in caplog.text
    assert str(repo) in caplog.text
    assert "MANUAL-ONLY" not in caplog.text
    # And escalation must NOT be reported as unarmed once the layer resolved a repo.
    assert "Auto-escalation is ENABLED but" not in caplog.text


def test_a_workdir_less_container_still_discloses_manual_only(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The shipped container's WORKDIR is an empty /app: no repo, no git, no test toolchain.
    # The layer must fail CLOSED there — an image that auto-captured from whatever
    # directory it booted in would run code the operator never pointed it at. The refusing
    # check here is the git-root walk-up: /app is in no repository.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    app = tmp_path / "app"
    app.mkdir()
    policy = RouterPolicy(capture=CapturePolicy())
    with caplog.at_level(logging.INFO):
        _log_capture_disclosure(policy, _build_work_dir_resolver(policy, str(app)))

    assert "MANUAL-ONLY" in caplog.text
    assert "capture is ON" not in caplog.text


def test_the_container_policy_stays_manual_only_even_on_a_mounted_repo(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The container ships `trust_launch_dir: false`, and that refusal must stand on its own:
    # mount a real, verifiable repo at the WORKDIR and capture is still MANUAL-ONLY. The
    # previous test would pass on an unarmed layer too (its /app has no `.git`), so it
    # cannot tell a stated policy from an accident of the image layout — this one can.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _launch_repo(tmp_path)
    policy = RouterPolicy(capture=CapturePolicy(trust_launch_dir=False))
    with caplog.at_level(logging.INFO):
        _log_capture_disclosure(policy, _build_work_dir_resolver(policy, str(repo)))

    assert "MANUAL-ONLY" in caplog.text
    assert "capture is ON" not in caplog.text

"""Preflight API health check: one real $0 completion at ``--live`` proves the key works.

All stubbed (no live call): covers the classifier fork (auth/no-balance refuses, transient
does not) and the live-only wiring (simulated skips it; a refuse aborts before any cell).
"""

from __future__ import annotations

from typing import Any, Final

import litellm
import pytest

from benchmark.config import CapabilityRank, RankedModel
from benchmark.runner import infer, ladder_collect, run_matrix

RANK: Final[CapabilityRank] = CapabilityRank(
    ordered=[RankedModel("c0", "a", 0, "measured")], evidence={}
)


def _stub_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass registry/key resolution so the probe path is exercised without real config."""
    monkeypatch.setattr(infer, "_cheapest_enabled_model", lambda: "c0")
    monkeypatch.setattr(infer, "litellm_model_target", lambda _m: ("deepseek/c0", {}))


def _stub_completion(monkeypatch: pytest.MonkeyPatch, raises: BaseException | None) -> None:
    """Make ``litellm.completion`` raise ``raises`` (or return a dummy response when None)."""

    def fake_completion(**_kw: Any) -> object:
        if raises is not None:
            raise raises
        return object()

    monkeypatch.setattr(litellm, "completion", fake_completion)


def _auth() -> litellm.exceptions.AuthenticationError:
    return litellm.exceptions.AuthenticationError(
        message="invalid api key", model="c0", llm_provider="deepseek"
    )


def _rate_limit() -> litellm.exceptions.RateLimitError:
    return litellm.exceptions.RateLimitError(message="429", model="c0", llm_provider="deepseek")


# --- preflight_api_check: the classifier fork -----------------------------------------


def test_preflight_passes_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_target(monkeypatch)
    _stub_completion(monkeypatch, raises=None)
    assert infer.preflight_api_check() is True  # healthy key → proceed


def test_preflight_auth_error_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_target(monkeypatch)
    _stub_completion(monkeypatch, raises=_auth())
    with pytest.raises(infer.ApiUnusableError, match="preflight health check failed"):
        infer.preflight_api_check()


def test_preflight_no_balance_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_target(monkeypatch)
    _stub_completion(monkeypatch, raises=Exception("Error 402: Insufficient Balance"))
    with pytest.raises(infer.ApiUnusableError):
        infer.preflight_api_check()


def test_preflight_transient_does_not_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rate-limit / 5xx blip is inconclusive: preflight returns True, never refuses over it.
    _stub_target(monkeypatch)
    _stub_completion(monkeypatch, raises=_rate_limit())
    assert infer.preflight_api_check() is True


# --- preflight_refuses: the live-only gate --------------------------------------------


def test_preflight_refuses_skips_when_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def _boom() -> bool:
        called["n"] += 1
        raise AssertionError("preflight must never run in simulated mode")

    monkeypatch.setattr(infer, "preflight_api_check", _boom)
    assert run_matrix.preflight_refuses(live=False) is False
    assert called["n"] == 0  # simulated is free/offline → probe never fires


def test_preflight_refuses_true_on_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> bool:
        raise infer.ApiUnusableError("no balance")

    monkeypatch.setattr(infer, "preflight_api_check", _raise)
    assert run_matrix.preflight_refuses(live=True) is True


def test_preflight_refuses_false_on_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(infer, "preflight_api_check", lambda: True)
    assert run_matrix.preflight_refuses(live=True) is False


# --- wiring: a live run refuses (exit 2) before any cell runs --------------------------


def test_run_ladder_refuses_before_any_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark import config

    monkeypatch.setattr(ladder_collect, "_has_keys", lambda: True)
    monkeypatch.setattr(config, "capability_rank", lambda *a, **k: RANK)
    monkeypatch.setattr(config, "collect_config", lambda: {"constants_pinned": True})
    monkeypatch.setattr(ladder_collect, "_refuse_live", lambda *a, **k: False)
    # Uncapped-live confirm now gates before preflight; accept it so preflight is still reached.
    monkeypatch.setattr(ladder_collect, "_confirm_uncapped_live", lambda: True)

    def _dead() -> bool:
        raise infer.ApiUnusableError("insufficient balance")

    monkeypatch.setattr(infer, "preflight_api_check", _dead)

    def _no_challenges(*_a: object, **_k: object) -> None:
        raise AssertionError("no challenge may run once preflight refuses")

    monkeypatch.setattr(ladder_collect, "_run_challenges", _no_challenges)
    monkeypatch.setattr(ladder_collect, "_sampled_tasks", lambda: ["t1"])

    assert ladder_collect.run_ladder(live=True) == 2  # refused, no cells started

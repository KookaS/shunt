"""Preflight API health check: one real $0 completion at ``--live`` proves the key works.

All stubbed (no live call): covers the classifier fork (auth/no-balance refuses, transient
does not) and the live-only wiring (simulated skips it; a refuse aborts before any cell).
"""

from __future__ import annotations

from typing import Any, Final

import litellm
import pytest

from benchmark.config import CapabilityRank, RankedModel
from benchmark.runner import infer, ladder_collect, lane_scheduler, run_matrix

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


# --- preflight_api_probe: the five-way classification ---------------------------------


def _probe(monkeypatch: pytest.MonkeyPatch, exc: BaseException | None) -> infer.PreflightOutcome:
    _stub_target(monkeypatch)
    _stub_completion(monkeypatch, raises=exc)
    return infer.preflight_api_probe()


def test_probe_classifies_success_as_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _probe(monkeypatch, None).kind == "ok"


def test_probe_classifies_auth_as_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = _probe(monkeypatch, _auth())
    assert outcome.kind == "unusable"
    assert outcome.disables


def test_probe_classifies_429_as_rate_limited_and_keeps_a_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = _probe(monkeypatch, _rate_limit())
    assert outcome.kind == "rate_limited"
    assert outcome.quarantines
    assert outcome.usable  # a throttle is not a disable

    with_header = _rate_limit()
    with_header.headers = {"Retry-After": "45"}
    assert _probe(monkeypatch, with_header).retry_after == 45.0


def test_probe_classifies_a_404_as_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = litellm.exceptions.NotFoundError(
        message="model not found: please check the model", model="c0", llm_provider="deepseek"
    )
    outcome = _probe(monkeypatch, exc)
    assert outcome.kind == "unavailable"
    assert outcome.disables


def test_probe_classifies_a_plan_gated_403_as_unavailable_not_a_dead_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    response = httpx.Response(403, request=httpx.Request("POST", "https://example.test/v1"))
    exc = litellm.exceptions.PermissionDeniedError(
        message="Model is not available on the Workers Free plan",
        model="c0",
        llm_provider="cloudflare",
        response=response,
    )
    outcome = _probe(monkeypatch, exc)
    assert outcome.kind == "unavailable", "a plan gate is a model gap, not invalid credentials"


def test_probe_leaves_a_transient_blip_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = _probe(monkeypatch, Exception("502 bad gateway"))
    assert outcome.kind == "transient"
    assert outcome.usable
    assert not outcome.disables and not outcome.quarantines


# --- preflight_free_lanes: per-lane fault tolerance -----------------------------------


def _ok(_lane: str) -> infer.PreflightOutcome:
    return infer.PreflightOutcome("ok", "healthy")


def test_a_dead_lane_does_not_kill_a_sibling_good_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fleet-wide abort bug: provider A's dead lane must not take provider A's good lane
    # (or any other lane) down with it.
    def probe(lane: str) -> infer.PreflightOutcome:
        return infer.PreflightOutcome("unusable", "dead key") if lane == "dead" else _ok(lane)

    report = run_matrix.preflight_free_lanes(True, ["dead", "good"], probe=probe)
    assert not report.refused
    assert report.usable == ["good"]
    assert "dead" in report.disabled
    assert "unusable" in report.disabled["dead"]


def test_the_fleet_aborts_only_when_no_lane_remains_usable() -> None:
    report = run_matrix.preflight_free_lanes(
        True, ["a", "b"], probe=lambda _lane: infer.PreflightOutcome("unavailable", "retired")
    )
    assert report.refused
    assert report.disabled.keys() == {"a", "b"}


def test_a_429_quarantines_only_that_lane() -> None:
    lanes = {lane: lane_scheduler.LaneLimits(rpm=10, rpd=100) for lane in ("throttled", "fine")}
    sched = lane_scheduler.LaneScheduler(limits=lanes)

    def probe(lane: str) -> infer.PreflightOutcome:
        if lane == "throttled":
            return infer.PreflightOutcome("rate_limited", "429", retry_after=60.0)
        return _ok(lane)

    report = run_matrix.preflight_free_lanes(
        True, ["throttled", "fine"], scheduler=sched, probe=probe
    )
    assert not report.refused
    assert set(report.usable) == {"throttled", "fine"}  # a throttle still leaves a lane
    assert "throttled" in report.quarantined
    assert sched.lane_state("throttled").quarantined_until is not None
    assert sched.lane_state("fine").disabled_reason is None


def test_a_disabled_lane_is_recorded_on_the_scheduler_for_persistence() -> None:
    sched = lane_scheduler.LaneScheduler(limits={"dead": lane_scheduler.LaneLimits()})
    report = run_matrix.preflight_free_lanes(
        True,
        ["dead"],
        scheduler=sched,
        probe=lambda _lane: infer.PreflightOutcome("unusable", "invalid api key"),
    )
    assert report.refused
    state = sched.lane_state("dead")
    assert state.disabled_reason is not None
    assert "unusable" in state.disabled_reason
    assert "invalid api key" in state.disabled_reason


def test_a_persisted_disabled_lane_is_never_re_probed() -> None:
    sched = lane_scheduler.LaneScheduler(
        limits={lane: lane_scheduler.LaneLimits() for lane in ("known-dead", "good")}
    )
    sched.disable("known-dead", "unusable: dead key")
    probed: list[str] = []

    def probe(lane: str) -> infer.PreflightOutcome:
        probed.append(lane)
        return _ok(lane)

    report = run_matrix.preflight_free_lanes(
        True, ["known-dead", "good"], scheduler=sched, probe=probe
    )
    assert probed == ["good"]  # the known-dead lane costs no call
    assert report.usable == ["good"]
    assert report.disabled["known-dead"] == "unusable: dead key"


def test_a_non_live_sweep_uses_the_static_admission_without_probing() -> None:
    probed: list[str] = []

    def probe(lane: str) -> infer.PreflightOutcome:
        probed.append(lane)
        return _ok(lane)

    report = run_matrix.preflight_free_lanes(False, ["a", "b"], probe=probe)
    assert report.usable == ["a", "b"]
    assert probed == []

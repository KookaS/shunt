"""Loop-health telemetry: label-coverage, propensity-support, and a routing-collapse
alarm that keys on the model-choice distribution ALONE (never on reward)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shunt.db.loop_health import (
    LoopHealthSnapshot,
    LoopHealthThresholds,
    compute_loop_health,
)
from shunt.db.store import OutcomeEvent, OutcomeStore, SessionProvenance


def _emb(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


@pytest.fixture
def store(tmp_path: Path) -> OutcomeStore:
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _snapshot(**overrides: object) -> LoopHealthSnapshot:
    base: dict[str, object] = {
        "total_sessions": 0,
        "eligible_sessions": 0,
        "verified_labeled": 0,
        "any_labeled": 0,
        "model_propensities": [],
        "recent_choices": [],
        "cost_by_model": [],
    }
    base.update(overrides)
    return LoopHealthSnapshot(**base)  # type: ignore[arg-type]


# ── Label coverage ─────────────────────────────────────────────────────────────
def test_label_coverage_is_verified_over_eligible() -> None:
    snap = _snapshot(total_sessions=10, eligible_sessions=8, verified_labeled=2, any_labeled=5)
    health = compute_loop_health(snap, frontier_models=set(), candidate_models={"a", "b", "c", "d"})
    cov = health.label_coverage
    assert cov.verified_coverage == pytest.approx(2 / 8)
    assert cov.labeled_coverage == pytest.approx(5 / 8)


def test_label_coverage_zero_sessions_no_div_by_zero() -> None:
    health = compute_loop_health(
        _snapshot(), frontier_models=set(), candidate_models={"a", "b", "c", "d"}
    )
    assert health.label_coverage.verified_coverage == 0.0
    assert health.label_coverage.labeled_coverage == 0.0


# ── Propensity support (ε-clip) ─────────────────────────────────────────────────
def test_collapsed_arm_is_support_deficient_and_epsilon_boundary() -> None:
    eps = 0.01
    thresholds = LoopHealthThresholds(propensity_epsilon=eps)
    snap = _snapshot(
        model_propensities=[
            ("well-explored", 20, 0.4, 0.1),
            ("boundary", 5, eps, eps),  # exactly at ε → NOT deficient (strict <)
            ("collapsed", 5, eps / 2, 0.0),  # below ε → deficient
        ],
    )
    health = compute_loop_health(
        snap,
        frontier_models=set(),
        candidate_models={"well-explored", "boundary", "collapsed"},
        thresholds=thresholds,
    )
    deficient = set(health.support_deficient_models)
    assert deficient == {"collapsed"}
    assert "well-explored" not in deficient
    assert "boundary" not in deficient  # ε-clip is a strict boundary


def test_never_selected_arm_is_flagged_deficient() -> None:
    # The strongest MNAR case: an arm the router stopped selecting ENTIRELY has zero
    # observed sessions. It must still surface as support-deficient, not vanish.
    snap = _snapshot(model_propensities=[("always-picked", 50, 0.9, 0.8)])
    health = compute_loop_health(
        snap,
        frontier_models=set(),
        candidate_models={"always-picked", "never-picked"},
    )
    deficient = set(health.support_deficient_models)
    assert deficient == {"never-picked"}
    never = next(p for p in health.propensity_support if p.model == "never-picked")
    assert never.n == 0
    assert never.mean_propensity == 0.0


# ── Routing-collapse alarm — reward-independent ─────────────────────────────────
def test_collapse_alarm_fires_on_choice_distribution_alone() -> None:
    # Every recent decision went to one frontier model: entropy 0, frontier-share 1.
    snap = _snapshot(recent_choices=["front-a"] * 40)
    health = compute_loop_health(
        snap, frontier_models={"front-a"}, candidate_models={"front-a", "m2", "m3", "m4"}
    )
    rc = health.routing_collapse
    assert rc.frontier_share == pytest.approx(1.0)
    assert rc.choice_entropy == pytest.approx(0.0)
    assert rc.alarm is True
    assert rc.frontier_share_alarm is True
    assert rc.entropy_collapse_alarm is True


def test_healthy_spread_does_not_alarm() -> None:
    snap = _snapshot(recent_choices=["cheap-a", "cheap-b", "mid-c", "front-d"] * 10)
    health = compute_loop_health(
        snap,
        frontier_models={"front-d"},
        candidate_models={"cheap-a", "cheap-b", "mid-c", "front-d"},
    )
    rc = health.routing_collapse
    assert rc.frontier_share == pytest.approx(0.25)
    assert rc.choice_entropy == pytest.approx(1.0)
    assert rc.alarm is False


def test_alarm_is_blind_to_reward_still_fires_when_outcomes_look_good(
    store: OutcomeStore,
) -> None:
    # A degenerate loop: the router ossified onto one frontier model, yet every one of
    # its outcomes is a verified SUCCESS. A reward-based detector would see nothing wrong.
    # The collapse alarm must still fire because it reads the choice distribution only.
    for i in range(30):
        store.store_session(
            f"s{i}",
            "prompt",
            _emb(),
            "front-only",
            1.0,
            {},
            1.0,
            provenance=SessionProvenance(selection_propensity=0.99),
        )
        store.append_outcome_event(
            OutcomeEvent(f"s{i}", 2, "auto_tier2", "success", 1.0, f"run-{i}")
        )

    snap = store.loop_health_snapshot(recent_window=100)
    health = compute_loop_health(
        snap, frontier_models={"front-only"}, candidate_models={"front-only", "m2", "m3", "m4"}
    )
    # Reward is uniformly good...
    assert snap.verified_labeled == 30
    # ...and the alarm fires anyway, purely on the collapsed choice distribution.
    assert health.routing_collapse.alarm is True
    assert health.routing_collapse.frontier_share == pytest.approx(1.0)


# ── Store snapshot integration ──────────────────────────────────────────────────
def test_snapshot_reads_coverage_and_propensity_from_store(store: OutcomeStore) -> None:
    # Two verified, one only-Tier-1-ish (any-labeled), one bare session (no outcome).
    for i in range(2):
        store.store_session(
            f"v{i}",
            "p",
            _emb(),
            "m",
            1.0,
            {},
            1.0,
            provenance=SessionProvenance(selection_propensity=0.5),
        )
        store.append_outcome_event(OutcomeEvent(f"v{i}", 2, "auto_tier2", "success", 1.0, f"r{i}"))
    store.store_session(
        "bare",
        "p",
        _emb(),
        "m",
        1.0,
        {},
        1.0,
        provenance=SessionProvenance(selection_propensity=0.001),
    )

    snap = store.loop_health_snapshot()
    assert snap.total_sessions == 3
    assert snap.eligible_sessions == 3  # all embedded
    assert snap.verified_labeled == 2
    # Propensity aggregates by model; here all share model "m".
    models = {m for m, *_ in snap.model_propensities}
    assert models == {"m"}

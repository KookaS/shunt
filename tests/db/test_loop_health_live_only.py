"""The loop-health snapshot reads LIVE rows only: seeded (`bench:`) rows are a replayed
benchmark corpus, and letting them into cost or the recency window publishes the benchmark
matrix's economics and model distribution as this router's own."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from shunt.db.loop_health import compute_loop_health
from shunt.db.store import OutcomeStore, SessionProvenance


def _emb(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[OutcomeStore]:
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _seeded_then_live(store: OutcomeStore) -> None:
    """The rig's shape: a big seeded burst imported AFTER the live traffic it dwarfs."""
    for i in range(6):
        store.store_session(
            f"live-{i}",
            "p",
            _emb(),
            "deepseek-v4-flash",
            0.01,
            {},
            1.0,
            timestamp=f"2026-01-01T00:00:0{i}+00:00",
        )
    # Seeded last, so a naive `ORDER BY timestamp DESC LIMIT n` sees nothing else.
    for i in range(50):
        store.store_session(
            f"bench:aaaaaaaaaaaa:kimi-k3-{i}",
            "p",
            _emb(),
            "kimi-k3",
            2.0,
            {},
            1.0,
            timestamp=f"2026-08-01T00:00:{i:02d}+00:00",
        )


def test_recent_choices_excludes_seeded_rows(store: OutcomeStore) -> None:
    _seeded_then_live(store)
    snap = store.loop_health_snapshot(recent_window=100)
    assert snap.recent_choices == ["deepseek-v4-flash"] * 6
    assert "kimi-k3" not in snap.recent_choices


def test_collapse_alarm_reads_live_behaviour_not_the_seed_corpus(store: OutcomeStore) -> None:
    # Every live decision went to one cheap arm — a genuine collapse. The seeded corpus is
    # spread over six models, so an origin-blind window would show healthy entropy and the
    # alarm could not fire while a seeded corpus is present.
    _seeded_then_live(store)
    health = compute_loop_health(
        store.loop_health_snapshot(),
        frontier_models={"kimi-k3"},
        candidate_models={"deepseek-v4-flash", "kimi-k3", "gpt-5-mini", "zai-glm-5.2"},
    )
    assert health.routing_collapse.window_size == 6
    assert health.routing_collapse.distinct_models == 1
    assert health.routing_collapse.entropy_collapse_alarm is True
    assert health.routing_collapse.frontier_share == 0.0


def test_cost_by_model_excludes_seeded_spend(store: OutcomeStore) -> None:
    _seeded_then_live(store)
    snap = store.loop_health_snapshot()
    assert {m for m, _, _ in snap.cost_by_model} == {"deepseek-v4-flash"}
    assert sum(total for _, _, total in snap.cost_by_model) == pytest.approx(0.06)
    health = compute_loop_health(
        snap, frontier_models={"kimi-k3"}, candidate_models={"deepseek-v4-flash", "kimi-k3"}
    )
    # $100 of seeded spend exists in the store and reaches neither router_cost nor n_total.
    assert health.cost.router_cost == pytest.approx(0.06)
    assert health.cost.n_total == 6


def test_cost_unknown_is_reported_per_model_and_totalled(store: OutcomeStore) -> None:
    _seeded_then_live(store)
    for i in range(2):
        store.store_session(
            f"live-unknown-{i}",
            "p",
            _emb(),
            "gpt-5-mini",
            0.0,
            {},
            1.0,
            timestamp="2026-01-02T00:00:00+00:00",
            provenance=SessionProvenance(cost_known=False),
        )
    snap = store.loop_health_snapshot()
    assert snap.cost_unknown_by_model == [("gpt-5-mini", 2)]
    # Not zero-filled into cost_by_model: gpt-5-mini contributes no cost row at all.
    assert {m for m, _, _ in snap.cost_by_model} == {"deepseek-v4-flash"}
    health = compute_loop_health(
        snap, frontier_models={"kimi-k3"}, candidate_models={"deepseek-v4-flash", "gpt-5-mini"}
    )
    assert health.cost.n_cost_unknown == 2
    assert health.cost.n_total == 6


def test_routing_ope_rows_are_live_only_and_carry_the_ope_shape(store: OutcomeStore) -> None:
    from shunt.analysis.ope import rows_from_records
    from shunt.db.store import OutcomeEvent

    store.store_session(
        "live-1",
        "p",
        _emb(),
        "deepseek-v4-flash",
        0.01,
        {},
        1.0,
        decision_provenance={"candidate_model_scores": {"deepseek-v4-flash": 0.7, "kimi-k3": 0.3}},
        provenance=SessionProvenance(selection_propensity=0.7),
    )
    store.append_outcome_event(OutcomeEvent("live-1", 2, "auto_tier2", "success", 1.0, "r1"))
    # A seeded row that (defensively) carries a propensity must still be excluded.
    store.store_session(
        "bench:aaaaaaaaaaaa:kimi-k3",
        "p",
        _emb(),
        "kimi-k3",
        2.0,
        {},
        1.0,
        provenance=SessionProvenance(selection_propensity=0.9),
    )
    # A live row with no propensity carries no counterfactual information — excluded.
    store.store_session("live-2", "p", _emb(), "kimi-k3", 0.02, {}, 1.0)

    rows = store.routing_ope_rows()
    assert [r["session_id"] for r in rows] == ["live-1"]
    assert rows[0]["model_chosen"] == "deepseek-v4-flash"
    assert rows[0]["selection_propensity"] == pytest.approx(0.7)
    assert rows[0]["outcome"] == "success"
    assert rows[0]["candidate_model_scores"] == {"deepseek-v4-flash": 0.7, "kimi-k3": 0.3}
    assert "timestamp" in rows[0]

    # The shape the estimator reads: `rows_from_records` must route these to the routing arm.
    log = rows_from_records(rows)
    assert log[0].action == "deepseek-v4-flash"
    assert log[0].propensity == pytest.approx(0.7)
    assert log[0].reward == 1.0
    assert log[0].session_id == "live-1"


def test_routing_ope_rows_keeps_an_unlabelled_decision_with_outcome_none(
    store: OutcomeStore,
) -> None:
    store.store_session(
        "live-1",
        "p",
        _emb(),
        "m",
        0.01,
        {},
        1.0,
        provenance=SessionProvenance(selection_propensity=0.5),
    )
    rows = store.routing_ope_rows()
    # None, not a fabricated failure: the estimator must be able to EXCLUDE the row.
    assert rows[0]["outcome"] is None


def test_model_propensities_exclude_seeded_rows(store: OutcomeStore) -> None:
    # Today's seeder writes no propensity, so `IS NOT NULL` excludes seeded rows by accident.
    # This row is the future state the live clause defends against: a seeded row that DOES
    # carry one. Without `_LIVE_CLAUSE` it lands in F5 panel C and /admin/loop-health.
    store.store_session(
        "live-1",
        "p",
        _emb(),
        "deepseek-v4-flash",
        0.01,
        {},
        1.0,
        provenance=SessionProvenance(selection_propensity=0.7),
    )
    store.store_session(
        "bench:aaaaaaaaaaaa:kimi-k3",
        "p",
        _emb(),
        "kimi-k3",
        2.0,
        {},
        1.0,
        provenance=SessionProvenance(selection_propensity=0.05),
    )
    snap = store.loop_health_snapshot()
    assert snap.model_propensities == [
        ("deepseek-v4-flash", 1, pytest.approx(0.7), pytest.approx(0.7))
    ]

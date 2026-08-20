"""Live-only cost accounting: seeded (`bench:`) spend is replayed benchmark cost, not
inference cost, and an unreported cost is UNKNOWN rather than a free session."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from shunt.db.store import OutcomeStore, SessionProvenance


def _emb(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


def _iso(days_ago: float) -> str:
    now = dt.datetime.now(dt.timezone.utc)  # noqa: UP017
    return (now - dt.timedelta(days=days_ago)).isoformat()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[OutcomeStore]:
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _mixed_corpus(store: OutcomeStore) -> None:
    """Seeded + live rows, including two live rows whose provider reported no cost.

    Seeded: $100.00 + $18.6751. Live cost_known: $0.20 + $0.0492. Live unknown: 2 rows.
    """
    rows = [
        ("bench:aaaaaaaaaaaa:kimi-k3", "kimi-k3", 100.0, True, 0.5),
        ("bench:bbbbbbbbbbbb:gpt-5-mini", "gpt-5-mini", 18.6751, True, 0.5),
        ("live-1", "deepseek-v4-flash", 0.20, True, 0.5),
        ("live-2", "gpt-5-mini", 0.0492, True, 40.0),
        # cost_known=0 — the provider reported no usage.cost. NEVER zero-filled.
        ("live-3", "deepseek-v4-flash", 0.0, False, 0.5),
        ("live-4", "kimi-k3", 0.0, False, 40.0),
    ]
    for sid, model, cost, known, days_ago in rows:
        store.store_session(
            sid,
            "prompt",
            _emb(),
            model,
            cost,
            {},
            1.0,
            timestamp=_iso(days_ago),
            provenance=SessionProvenance(cost_known=known),
        )


def test_live_cost_excludes_seeded_spend(store: OutcomeStore) -> None:
    _mixed_corpus(store)
    agg = store.live_cost_aggregates()
    assert agg.total == pytest.approx(0.2492)
    assert agg.window_days is None
    assert dict((m, n) for m, n, _ in agg.by_model) == {"deepseek-v4-flash": 1, "gpt-5-mini": 1}
    assert "kimi-k3" not in {m for m, _, _ in agg.by_model}


def test_unknown_cost_is_counted_and_never_zero_filled(store: OutcomeStore) -> None:
    _mixed_corpus(store)
    agg = store.live_cost_aggregates()
    assert agg.n_cost_unknown == 2
    assert agg.n_cost_known == 2
    # The unknown rows contribute no row to by_model and no dollar to the total: a
    # zero-filled unknown would leave n_cost_known at 4 and the total unchanged, which is
    # exactly the silent under-count this asserts against.
    assert sum(n for _, n, _ in agg.by_model) == 2
    assert agg.total == pytest.approx(0.2492)


def test_window_filters_on_session_timestamp(store: OutcomeStore) -> None:
    _mixed_corpus(store)
    week = store.live_cost_aggregates(window_days=7)
    assert week.window_days == 7
    assert week.total == pytest.approx(0.20)  # the 40-days-ago live row drops out
    assert week.n_cost_known == 1
    assert week.n_cost_unknown == 1
    month = store.live_cost_aggregates(window_days=30)
    assert month.total == pytest.approx(0.20)
    all_time = store.live_cost_aggregates(window_days=365)
    assert all_time.total == pytest.approx(0.2492)
    assert all_time.n_cost_unknown == 2


def test_empty_store_reports_zero_not_none(store: OutcomeStore) -> None:
    agg = store.live_cost_aggregates(window_days=7)
    assert (agg.total, agg.n_cost_known, agg.n_cost_unknown, agg.by_model) == (0.0, 0, 0, [])


def test_get_stats_keeps_whole_store_total_and_splits_it(store: OutcomeStore) -> None:
    _mixed_corpus(store)
    stats = store.get_stats()
    # No silent behaviour change for existing callers: total_cost is still the whole store.
    assert stats["total_cost"] == pytest.approx(118.9243)
    assert stats["live_total_cost"] == pytest.approx(0.2492)
    assert stats["seeded_total_cost"] == pytest.approx(118.6751)
    assert stats["live_total_cost"] + stats["seeded_total_cost"] == pytest.approx(
        stats["total_cost"]
    )
    assert stats["cost_unknown_count"] == 2


def test_stratum_census_funnels_seeded_and_live_separately(store: OutcomeStore) -> None:
    from shunt.db.store import OutcomeEvent

    _mixed_corpus(store)
    store.append_outcome_event(
        OutcomeEvent("bench:aaaaaaaaaaaa:kimi-k3", 2, "benchmark_seed", "success", 1.0, "r1")
    )
    store.append_outcome_event(OutcomeEvent("live-1", 1, "wire_tier1", "success", 0.5, "r2"))
    store.append_outcome_event(OutcomeEvent("live-2", 2, "auto_tier2", "failure", 1.0, "r3"))

    census = store.stratum_census()
    assert (census.seeded.stratum, census.live.stratum) == ("seeded", "live")
    assert census.seeded.stored == 2
    assert census.live.stored == 4
    assert census.seeded.embedded == 2
    assert (census.seeded.labeled, census.seeded.tier2, census.seeded.indexed) == (1, 1, 1)
    # live-1 is Tier-1 only: it counts as labeled and then DROPS OUT — never materialized,
    # never a kNN index member. That drop is what the funnel exists to show.
    assert (census.live.labeled, census.live.tier2, census.live.indexed) == (2, 1, 1)


def test_stratum_census_on_a_live_only_store(store: OutcomeStore) -> None:
    store.store_session("live-1", "p", _emb(), "m", 0.1, {}, 1.0)
    census = store.stratum_census()
    assert census.seeded.stored == 0
    assert census.seeded.indexed == 0
    assert census.live.stored == 1

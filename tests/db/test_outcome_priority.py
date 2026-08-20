"""Outcome-source priority: a live event overrides a seeded one in the materialized view.

benchmark_seed events are synthetic prior-only data with zero priority, so any real
Tier-2 event (auto_tier2 or human) for the same session must win the view row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shunt.db.store import OutcomeEvent, OutcomeStore


@pytest.fixture
def store(tmp_path: Path) -> OutcomeStore:
    s = OutcomeStore(db_path=str(tmp_path / "priority.db"))
    yield s
    s.close()


def _seed_measured_cell(store: OutcomeStore) -> None:
    """One seeded session: a real measured cell plus its benchmark_seed Tier-2 event."""
    store.store_session(
        "swebench_abc",
        "implement the feature",
        np.zeros(64, dtype=np.float32),
        "qwen3.7-plus",
        0.5,
        {},
        1.0,
    )
    store.append_outcome_event(
        OutcomeEvent(
            session_id="swebench_abc",
            tier=2,
            source="benchmark_seed",
            outcome="success",
            confidence=1.0,
            run_signature="bench:digest:qwen3.7-plus:content",
        )
    )


def test_human_tier2_beats_benchmark_seed(store: OutcomeStore) -> None:
    _seed_measured_cell(store)
    store.append_outcome_event(
        OutcomeEvent(
            session_id="swebench_abc",
            tier=2,
            source="human",
            outcome="failure",
            confidence=1.0,
            run_signature="human-1",
        )
    )
    o = store.get_outcome("swebench_abc")
    assert o is not None
    assert o["tier2_outcome"] == "failure"
    assert o["outcome_source"] == "human"


def test_auto_tier2_beats_benchmark_seed(store: OutcomeStore) -> None:
    _seed_measured_cell(store)
    store.append_outcome_event(
        OutcomeEvent(
            session_id="swebench_abc",
            tier=2,
            source="auto_tier2",
            outcome="failure",
            confidence=0.9,
            run_signature="auto-1",
        )
    )
    o = store.get_outcome("swebench_abc")
    assert o is not None
    assert o["tier2_outcome"] == "failure"
    assert o["outcome_source"] == "auto_tier2"

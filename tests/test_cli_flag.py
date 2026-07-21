"""`shunt flag` is the router's outcome write-back path — it must actually write."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from shunt.cli import _flag
from shunt.db.outcome_index import OutcomeIndexAdapter
from shunt.db.store import OutcomeStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OutcomeStore:
    created = OutcomeStore(db_path=str(tmp_path / "sessions.db"))
    monkeypatch.setattr("shunt.db.store.OutcomeStore", lambda *a, **k: created)
    return created


def _session(store: OutcomeStore, session_id: str = "s1") -> None:
    store.store_session(
        session_id=session_id,
        prompt_text="a task",
        embedding=np.ones(384, dtype=np.float32),
        model_chosen="model-a",
        cost=0.01,
        cache_stats={},
        duration=1.0,
    )


def test_flagging_good_records_a_verified_outcome(store: OutcomeStore) -> None:
    # Without this the outcome table stays empty, the engine never leaves cold-start, and
    # kNN routing plus exploration are configured-on but unreachable.
    _session(store)
    assert OutcomeIndexAdapter(store).count_labeled() == 0

    _flag(argparse.Namespace(session_id="s1", rating="good"))

    assert OutcomeIndexAdapter(store).count_labeled() == 1
    outcome = store.get_outcome("s1")
    assert outcome is not None
    assert outcome["tier2_outcome"] == "success"
    assert outcome["human_label"] == "good"


def test_flagging_bad_records_a_failure(store: OutcomeStore) -> None:
    _session(store)

    _flag(argparse.Namespace(session_id="s1", rating="bad"))

    outcome = store.get_outcome("s1")
    assert outcome is not None
    assert outcome["tier2_outcome"] == "failure"


def test_flagging_an_unknown_session_fails_loudly(store: OutcomeStore) -> None:
    with pytest.raises(SystemExit) as exc:
        _flag(argparse.Namespace(session_id="nope", rating="good"))

    assert exc.value.code == 1
    assert OutcomeIndexAdapter(store).count_labeled() == 0


def test_a_flagged_session_becomes_a_routable_neighbour(store: OutcomeStore) -> None:
    # The whole point: a labelled session must show up as kNN evidence, not just a DB row.
    _session(store)
    _flag(argparse.Namespace(session_id="s1", rating="good"))

    neighbors = OutcomeIndexAdapter(store).query(np.ones(384, dtype=np.float32), k=5)
    assert [n.model for n in neighbors] == ["model-a"]
    assert neighbors[0].outcome is True

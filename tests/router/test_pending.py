from __future__ import annotations

import pytest

from shunt.router.pending import PendingDecision, PendingOutcomes


def _decision(key: str, model: str = "cheap") -> PendingDecision:
    return PendingDecision(key=key, model=model, exploratory=False, propensity=1.0)


def test_record_and_resolve_roundtrip() -> None:
    q = PendingOutcomes()
    q.record(_decision("s1"))
    assert "s1" in q
    resolved = q.resolve("s1", success=True)
    assert resolved is not None
    assert resolved.decision.key == "s1"
    assert resolved.success is True
    assert "s1" not in q  # resolved decisions leave the queue


def test_resolve_unknown_key_is_noop() -> None:
    q = PendingOutcomes()
    assert q.resolve("never-seen", success=False) is None


def test_unlabeled_is_never_a_failure_only_forgotten() -> None:
    # Eviction drops the oldest unlabeled decision; it is never resolved as a failure.
    q = PendingOutcomes(max_pending=2)
    q.record(_decision("old"))
    q.record(_decision("mid"))
    q.record(_decision("new"))  # evicts "old"
    assert "old" not in q
    assert q.resolve("old", success=True) is None  # gone, not turned into an outcome
    assert len(q) == 2


def test_max_pending_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_pending"):
        PendingOutcomes(max_pending=0)


def test_re_recording_key_refreshes_position() -> None:
    q = PendingOutcomes(max_pending=2)
    q.record(_decision("a"))
    q.record(_decision("b"))
    q.record(_decision("a"))  # refresh a → b is now oldest
    q.record(_decision("c"))  # evicts b, not a
    assert "a" in q
    assert "b" not in q

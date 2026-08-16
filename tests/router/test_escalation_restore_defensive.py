"""F4: a malformed / legacy / forward-schema escalation snapshot must not abort router boot."""

# `restore_escalation_state` is called unguarded at startup (proxy/server.py). A bare
# `FailureEvent(**e)` on sqlite-read dicts raised TypeError on an unknown or missing field —
# taking the whole router down — and silently accepted a legacy dict with no `blocking`, which
# downgrades real capability failures to non-counting. It now degrades to an empty window.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from shunt.models.config import ModelConfig
from shunt.router.engine import RouterEngine, task_state_key
from shunt.router.escalation import EscalationConfig


def _cfg(name: str) -> ModelConfig:
    return ModelConfig(name=name, provider="p", base_url="http://x", api_key_env_var="K")


class _Pool:
    def __init__(self) -> None:
        self._ranked = [_cfg("qwen"), _cfg("glm")]

    def get_model(self, name: str) -> ModelConfig | None:
        return next((m for m in self._ranked if m.name == name), None)

    def ranked_models(self) -> list[ModelConfig]:
        return list(self._ranked)

    def rank_of(self, name: str) -> int | None:
        return next((i for i, m in enumerate(self._ranked) if m.name == name), None)

    def models_from_rank(self, i: int) -> list[ModelConfig]:
        return self._ranked[max(i, 0) :]

    def is_healthy(self, name: str) -> bool:
        return True


@dataclass
class _Session:
    tool_identity: str


class _SessionManager:
    def get_session(self, session_id: str) -> _Session:
        return _Session(tool_identity="toolA")


class _Index:
    def count_labeled(self) -> int:
        return 100

    def count_total_labeled(self) -> int:
        return 100

    def effective_labeled(self) -> float:
        return 100.0

    def effective_tier2(self) -> float:
        return 100.0

    def model_priors(self) -> dict[str, tuple[float, float]]:
        return {}

    def query(self, embedding: np.ndarray, k: int = 20) -> list:  # type: ignore[type-arg]
        return []


class _Embedder:
    def embed(self, text: str) -> np.ndarray:  # type: ignore[type-arg]
        return np.zeros(8, dtype=np.float32)


def _engine() -> RouterEngine:
    return RouterEngine(
        model_pool=_Pool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True),
        task_key_resolver=lambda _s: "repoA",
    )


def _event(**overrides: Any) -> dict[str, Any]:
    base = {
        "decision_index": 0,
        "dedup_key": "t::a",
        "exit_code": 1,
        "success": False,
        "confirmed": True,
        "blocking": True,
    }
    base.update(overrides)
    return base


# A persisted blob is keyed by `task_state_key(work_dir)`, never by the raw work_dir — the raw key
# is an operator path that can embed a credential, and this blob lands in plaintext sqlite. These
# fixtures fabricate a REAL snapshot, so they key it the way the engine does.
_KEY = task_state_key("repoA")


def _state(*events: dict[str, Any]) -> dict[str, Any]:
    return {"failure_log": {_KEY: list(events)}, "decision_index": {}, "effort_arm": {}}


def _log(eng: RouterEngine) -> list[Any]:
    return eng.snapshot_escalation_state()["failure_log"].get(_KEY, [])


def test_wellformed_payload_restores() -> None:
    eng = _engine()
    eng.restore_escalation_state(_state(_event(), _event(decision_index=1)))
    assert len(_log(eng)) == 2


def test_forward_schema_extra_field_does_not_abort_boot() -> None:
    # An unknown field means a NEWER shunt wrote the snapshot. Ignore it; never crash startup.
    eng = _engine()
    eng.restore_escalation_state(_state(_event(future="whatever")))
    assert len(_log(eng)) == 1


def test_legacy_payload_missing_blocking_degrades_to_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DOCUMENTED CHOICE: `blocking` is the gate `counts_as_failure` reads and the log holds BOTH
    # blocking and non-blocking events, so a payload predating the field cannot be reconstructed.
    # Defaulting False silently downgrades real capability failures; defaulting True over-counts
    # infra. The window is dropped instead — it only delays an escalation and self-heals.
    payload = _event()
    del payload["blocking"]
    eng = _engine()
    with caplog.at_level(logging.WARNING, logger="shunt.router.engine"):
        eng.restore_escalation_state(_state(payload))
    assert _log(eng) == []
    assert any("unreadable" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "payload",
    [
        _event(exit_code=None),
        _event(decision_index="not-an-int"),
        _event(exit_code=[1]),
        "not-a-dict",
    ],
)
def test_malformed_payload_degrades_instead_of_raising(payload: Any) -> None:
    eng = _engine()
    eng.restore_escalation_state(_state(payload))  # must not raise
    assert _log(eng) == []


def test_non_list_failure_log_entry_degrades() -> None:
    eng = _engine()
    eng.restore_escalation_state({"failure_log": {_KEY: "oops"}})
    assert _log(eng) == []


def test_decision_index_is_coerced_and_uncoercible_entries_dropped() -> None:
    # The sibling paths already coerce (int(v)/str(v)); a string index used to survive restore and
    # blow up later inside decide() with `unsupported operand type(s) for -: 'int' and 'str'`.
    eng = _engine()
    eng.restore_escalation_state({"decision_index": {_KEY: "7", task_state_key("repoB"): "nope"}})
    indexes = eng.snapshot_escalation_state()["decision_index"]
    assert indexes == {_KEY: 7}
    eng.decide("s1", "task")  # the arithmetic that used to raise


def test_failure_event_decision_index_is_coerced() -> None:
    eng = _engine()
    eng.restore_escalation_state(_state(_event(decision_index="3")))
    assert _log(eng)[0]["decision_index"] == 3

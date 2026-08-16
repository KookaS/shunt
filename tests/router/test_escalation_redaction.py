"""F3: a secret-bearing failing-check id is scrubbed at every sink it can reach."""

# A parametrized test id can embed a credential (`test_login[sk-...]`), and that id IS the
# escalation dedup key. `ExplorationRecord.persistable()` was the only pinned sink; the escalation
# snapshot (plaintext sqlite `router_state`), the trajectory redactor, and the rerun-flake log
# line each leaked it verbatim. One test per sink, so a regression names its own sink.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from shunt.capture.trajectory import StepRecord, redact_record
from shunt.db.store import OutcomeStore
from shunt.models.config import ModelConfig
from shunt.proxy.redaction import redact_secrets
from shunt.router.engine import RouterEngine, task_state_key
from shunt.router.escalation import (
    EscalationAction,
    EscalationConfig,
    ExplorationRecord,
    FailureEvent,
)
from shunt.verifiers.base import VerifierResult
from shunt.verifiers.rerun import RerunConfirmingVerifier

# Assembled at runtime from halves so the fixture cannot be mistaken for a live credential by a
# secret scanner — gitleaks flags the literal form, which is the scanner behaving correctly.
# Same construction as `tests/proxy/test_redaction.py`, which pins that convention.
_FAKE: Final[str] = "A" * 8 + "0123456789bcdef"
_SECRET: Final[str] = f"sk-{_FAKE}"
_CHECK_ID = f"tests/test_auth.py::test_login[{_SECRET}]"
# The escalation task key is the resolved work_dir — an OPERATOR PATH, and an operator path can
# embed a credential. It is a dict key, so `persistable()` (which scrubs values) never saw it.
_SECRET_WORK_DIR = f"/home/alice/work/{_SECRET}/repo"


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


def _engine(task_key: str = "repoA") -> RouterEngine:
    return RouterEngine(
        model_pool=_Pool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True),
        task_key_resolver=lambda _s: task_key,
    )


def _engine_with_secret_failure(task_key: str = "repoA") -> RouterEngine:
    eng = _engine(task_key)
    eng.decide("s1", "task")
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key=task_key,
        dedup_key=_CHECK_ID,
        exit_code=1,
        is_infra_failure=False,
        confirmed=True,
    )
    return eng


def test_the_fake_credential_is_still_secret_shaped() -> None:
    # Positive control for every test below: they all assert a secret is ABSENT, which a fixture
    # defused into a non-credential shape would satisfy VACUOUSLY. The fake is assembled at
    # runtime to keep it out of the secret scanner, so this pins that it still trips the
    # redaction primitive — otherwise this whole file could go green while the sinks leak.
    assert redact_secrets(_SECRET) == "<redacted>"
    assert _SECRET in _CHECK_ID
    assert _SECRET in _SECRET_WORK_DIR


def test_sink1_exploration_record_persistable_is_clean() -> None:
    record = ExplorationRecord(
        checkpoint_id=_CHECK_ID,
        decision_index=0,
        action=EscalationAction.HOLD,
        policy_action=EscalationAction.HOLD,
        propensity=1.0,
        epsilon=0.0,
        seed=None,
        randomized=False,
    )
    assert _SECRET not in json.dumps(record.persistable())


def test_sink2_escalation_snapshot_is_clean() -> None:
    state = _engine_with_secret_failure().snapshot_escalation_state()
    assert state["failure_log"][task_state_key("repoA")]  # the event IS there — the key is scrubbed
    assert _SECRET not in json.dumps(state)


def test_sink2_survives_the_sqlite_round_trip(tmp_path: Path) -> None:
    # The snapshot lands in PLAINTEXT `router_state`; assert against the raw column, not the API.
    db = tmp_path / "o.db"
    store = OutcomeStore(db_path=str(db))
    store.save_escalation_state(_engine_with_secret_failure().snapshot_escalation_state())
    raw = db.read_bytes()
    assert _SECRET.encode() not in raw
    loaded = store.load_escalation_state()
    assert loaded is not None
    assert _SECRET not in json.dumps(loaded)


def test_a_secret_bearing_key_still_groups_across_a_restart() -> None:
    # F2: the dedup key is redacted at the SINGLE construction seam, so the in-memory log and the
    # plaintext snapshot hold the SAME string. Before that seam, the restored log carried the
    # redacted key while a fresh capture appended the raw one — the two halves never grouped, so
    # recurrence silently stopped for exactly the secret-bearing ids the redaction exists to
    # protect ("state survives restart" held only for keys that did not redact).
    original = _engine_with_secret_failure()  # 1 verified failure; key redacted in-memory
    state = original.snapshot_escalation_state()
    fresh = _engine()
    fresh.restore_escalation_state(state)
    # Same task, same secret-bearing check id, a SECOND verified failure AFTER the restart.
    fresh.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key=_CHECK_ID,
        exit_code=1,
        is_infra_failure=False,
        confirmed=True,
    )
    _model, reason, _prov = fresh.decide("s2", "task")
    assert reason == "auto_escalation"  # two same-key failures, one on each side of the restart


def test_failure_event_persistable_redacts_the_dedup_key() -> None:
    event = FailureEvent(
        decision_index=0, dedup_key=_CHECK_ID, exit_code=1, success=False, confirmed=True
    )
    out = event.persistable()
    assert _SECRET not in str(out["dedup_key"])
    assert out["confirmed"] is True  # the rest of the projection is untouched


def test_sink5_snapshot_keys_never_carry_the_task_key_secret() -> None:
    state = _engine_with_secret_failure(_SECRET_WORK_DIR).snapshot_escalation_state()
    blob = json.dumps(state)
    assert _SECRET not in blob
    assert _SECRET_WORK_DIR not in blob
    # The accrued window IS still there — keyed by the digest, not dropped.
    key = task_state_key(_SECRET_WORK_DIR)
    assert list(state["failure_log"]) == [key]
    assert list(state["decision_index"]) == [key]


def test_sink5_the_digest_key_round_trips_state_a_redacted_key_would_lose() -> None:
    # Guards the naive fix. `redact_secrets(key)` changes the key's IDENTITY, so the restored
    # entry is unreachable from the live `_task_key()` and the accrued window silently vanishes
    # on every restart. The digest is stable across processes, so a restart still finds it.
    state = _engine_with_secret_failure(_SECRET_WORK_DIR).snapshot_escalation_state()
    fresh = _engine(_SECRET_WORK_DIR)
    fresh.restore_escalation_state(state)
    _model, _reason, provenance = fresh.decide("s2", "task")
    # The counter kept climbing from the restored value, so decide() DID find the restored entry.
    assert provenance["decision_index"] == state["decision_index"][task_state_key(_SECRET_WORK_DIR)]
    assert fresh.snapshot_escalation_state()["failure_log"] == state["failure_log"]


def test_sink5_task_keys_that_redact_alike_stay_distinct() -> None:
    # Guards the OTHER naive fix. The shape net fires on ordinary repo names, so two unrelated
    # repos redact to the SAME string — redacted keys would merge their escalation windows and
    # escalate one repo on the other's failures. Worse than the leak it would have fixed.
    a, b = "/srv/repos/api-gateway-service-v2", "/srv/repos/api-gateway-service-v3"
    assert redact_secrets(a) == redact_secrets(b)  # the collision is real, not hypothetical
    keys = [
        next(iter(_engine_with_secret_failure(k).snapshot_escalation_state()["failure_log"]))
        for k in (a, b)
    ]
    assert len(set(keys)) == 2  # ...and the snapshot keys do not collide


def test_sink5_a_pre_digest_snapshot_is_dropped_not_laundered_forward() -> None:
    # Upgrade path: a snapshot written before the fix is keyed by the raw work_dir. Restoring it
    # verbatim would re-persist that path — secret and all — on the NEXT snapshot, so the leak
    # would outlive the fix. Such an entry is unreachable anyway; it is dropped.
    eng = _engine(_SECRET_WORK_DIR)
    eng.restore_escalation_state({"decision_index": {_SECRET_WORK_DIR: 7}, "effort_arm": {}})
    assert _SECRET not in json.dumps(eng.snapshot_escalation_state())


def test_sink5_secret_task_key_survives_the_sqlite_round_trip(tmp_path: Path) -> None:
    # `close()` first: journal_mode=WAL leaves a fresh write in `o.db-wal`, so reading `o.db`
    # alone would pass on the UNFIXED code. Scan every file the store left behind.
    store = OutcomeStore(db_path=str(tmp_path / "o.db"))
    store.save_escalation_state(
        _engine_with_secret_failure(_SECRET_WORK_DIR).snapshot_escalation_state()
    )
    store.close()
    on_disk = b"".join(p.read_bytes() for p in sorted(tmp_path.iterdir()) if p.is_file())
    assert _SECRET.encode() not in on_disk
    assert _SECRET_WORK_DIR.encode() not in on_disk


def test_sink3_redact_record_scrubs_failing_check_id() -> None:
    record = StepRecord(
        step_index=0,
        decision_index=0,
        metadata={},
        observation="",
        action="",
        args=None,
        result="",
        failing_check_id=_CHECK_ID,
        exit_code=1,
        blocking=True,
        is_infra_failure=False,
        confirmed=True,
        success=False,
    )
    redacted = redact_record(record)
    assert redacted.failing_check_id is not None
    assert _SECRET not in redacted.failing_check_id
    assert _SECRET not in json.dumps(redacted.committable())


class _FlakyVerifier:
    """Fails once with a secret-bearing check id, then passes — the flake path that logs it."""

    def __init__(self) -> None:
        self._calls = 0

    def verify(self, text: str = "", work_dir: str | None = None) -> VerifierResult:
        self._calls += 1
        if self._calls == 1:
            return VerifierResult(
                outcome="failure", confidence=0.7, exit_code=1, failing_check_id=_CHECK_ID
            )
        return VerifierResult(outcome="success", confidence=0.8, exit_code=0)


def test_sink4_rerun_flake_log_is_clean(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="shunt.verifiers.rerun"):
        result = RerunConfirmingVerifier(_FlakyVerifier(), reruns=1).verify()
    assert result.outcome == "unknown"
    assert caplog.records, "the flake path must log"
    assert all(_SECRET not in record.getMessage() for record in caplog.records)

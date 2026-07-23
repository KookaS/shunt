"""B1 + B9 regression walls: drive the REAL AutoDetectVerifier subprocess end-to-end."""

# B1 was the critical bug no unit test caught (every escalation test hardcoded exit_code=2);
# this runs a genuine failing pytest (exit 1) through verify → capture → record_outcome →
# decide_escalation. B9 proves a collection/import error is withheld. Offline local pytest only.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from shunt.capture.coordinator import CaptureCoordinator, WorkDirResolver
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig
from shunt.session import Session
from shunt.verifiers.rerun import RerunConfirmingVerifier
from shunt.verifiers.tier2 import AutoDetectVerifier

_PYPROJECT = "[tool.pytest.ini_options]\naddopts = ''\n"


def _make_repo(tmp_path: Path, test_body: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(_PYPROJECT)
    (repo / "test_x.py").write_text(test_body)
    return repo


# ── engine fakes (cheap→mid→high, empty neighbourhood → base pick is qwen) ──────
@dataclass
class _M:
    name: str


class _TieredPool:
    def __init__(self) -> None:
        self._tiers = {"cheap": [_M("qwen")], "mid": [_M("glm")], "high": [], "frontier": []}

    def get_tier_models(self, tier: str) -> list[_M]:
        return self._tiers.get(tier, [])

    def is_healthy(self, name: str) -> bool:
        return True


class _SessionManager:
    def get_session(self, session_id: str) -> object:
        return object()


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


class _Store:
    """Minimal OutcomeStore surface CaptureCoordinator._append_tier2 touches (fresh insert)."""

    def get_session(self, session_id: str) -> dict | None:  # type: ignore[type-arg]
        return {"model_fingerprint": None, "decision_provenance": None}

    def append_outcome_event(self, event: object) -> bool:
        return True

    def persist_index(self) -> None:
        return None


def _engine(work_dir: str) -> RouterEngine:
    return RouterEngine(
        model_pool=_TieredPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2),
        task_key_resolver=lambda _s: work_dir,
    )


def _coord(eng: RouterEngine, work_dir: str) -> CaptureCoordinator:
    return CaptureCoordinator(
        resolver=WorkDirResolver(work_dir=work_dir),
        # reruns=1: confirm the red reproduces (sets `confirmed`) while keeping the run count low.
        verifier=RerunConfirmingVerifier(AutoDetectVerifier(), reruns=1),
        store=_Store(),  # type: ignore[arg-type]
        record_outcome_callback=eng.record_outcome,
    )


def _closed(sid: str) -> Session:
    now = datetime.now(UTC)
    s = Session(session_id=sid, tool_identity="toolA", start_time=now)
    s.end_time = now
    return s


def test_real_failing_pytest_escalates_after_two(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "def test_x():\n    assert False\n")
    work_dir = str(repo)
    eng = _engine(work_dir)
    coord = _coord(eng, work_dir)

    m1, _, _ = eng.decide("s1", "task")
    assert m1 == "qwen"
    coord.capture(_closed("s1"))  # 1st real verified red (subprocess exit 1)
    eng.decide("s2", "task")
    coord.capture(_closed("s2"))  # 2nd real verified red, same node id

    m3, r3, _ = eng.decide("s3", "task")
    assert m3 == "glm"  # escalated on genuine exit-1 pytest failures (the B1 regression wall)
    assert r3 == "auto_escalation"


def test_real_import_error_does_not_escalate(tmp_path: Path) -> None:
    # A module-level import of a missing package → pytest collection error (env-cause, exit 2).
    repo = _make_repo(tmp_path, "import definitely_missing_pkg_xyz\n\ndef test_x():\n    pass\n")
    work_dir = str(repo)
    eng = _engine(work_dir)
    coord = _coord(eng, work_dir)

    eng.decide("s1", "task")
    coord.capture(_closed("s1"))
    eng.decide("s2", "task")
    coord.capture(_closed("s2"))

    m3, r3, _ = eng.decide("s3", "task")
    assert m3 == "qwen"  # env-cause red is withheld from escalation (B9)
    assert r3 != "auto_escalation"


def test_real_verifier_classifies_import_error_as_infra(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "import definitely_missing_pkg_xyz\n\ndef test_x():\n    pass\n")
    result = AutoDetectVerifier().verify(work_dir=str(repo))
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True


def test_real_verifier_reports_genuine_failure(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "def test_x():\n    assert False\n")
    result = AutoDetectVerifier().verify(work_dir=str(repo))
    assert result.outcome == "failure"
    assert result.is_infra_failure is False
    assert result.exit_code == 1

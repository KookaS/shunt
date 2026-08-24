"""`shunt escalate` on REAL persisted state produced by the REAL engine."""

# Nothing here hand-builds an escalation snapshot: the state under inspection is written by
# RouterEngine.snapshot_escalation_state() into a real OutcomeStore, so the CLI is proven
# against the shape production actually persists rather than a fixture's idea of it.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from shunt.cli import _escalate
from shunt.db.store import OutcomeStore
from shunt.models import ModelPool
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig
from tests.router.escalation_fakes import EchoSessionManager, Embedder, Index

_REPO = "/repo-under-test"


def _engine(pool: ModelPool, ladder: str = "effort_then_rank") -> RouterEngine:
    return RouterEngine(
        model_pool=pool,
        session_manager=EchoSessionManager(),
        outcome_index=Index(),
        embedder=Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder=ladder),
        task_key_resolver=lambda _s: _REPO,
    )


def _red(engine: RouterEngine, dedup_key: str = "tests/test_a.py::test_x") -> None:
    engine.record_outcome(
        downshift=False,
        success=False,
        task_key=_REPO,
        dedup_key=dedup_key,
        exit_code=1,
        is_infra_failure=False,
        confirmed=True,
    )


def _persist(engine: RouterEngine) -> None:
    store = OutcomeStore()
    store.save_escalation_state(engine.snapshot_escalation_state())
    store.close()


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hermetic data dir + the PACKAGED registry/policy (no dev-machine ~/.config/shunt)."""
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("SHUNT_MODEL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    return tmp_path


def _run(work_dir: str | None = _REPO, as_json: bool = False) -> None:
    _escalate(argparse.Namespace(work_dir=work_dir, as_json=as_json))


def test_reports_the_live_window_and_the_next_directive(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two verified reds on the same key with no decision in between: the window is DUE,
    # so the next decision escalates — and the CLI must say so off the persisted state.
    pool = ModelPool()
    engine = _engine(pool)
    engine.decide("s0", "task")
    _red(engine)
    _red(engine)
    _persist(engine)

    _run()

    out = capsys.readouterr().out
    assert "tests/test_a.py::test_x: 2 counting / 2 event(s)  ← DUE" in out
    # The directive comes from the real decide_escalation, not a CLI-local reimplementation.
    assert "Action: raise_effort" in out
    assert "Reason: same_verified_failure_x2" in out
    assert "Escalation suppressed:  no" in out


def test_names_why_a_non_counting_event_does_not_count(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = ModelPool()
    engine = _engine(pool)
    engine.decide("s0", "task")
    engine.record_outcome(
        downshift=False,
        success=False,
        task_key=_REPO,
        dedup_key="ruff",
        exit_code=1,
        is_infra_failure=True,  # infra ⇒ non-blocking ⇒ never counts
        confirmed=True,
    )
    _persist(engine)

    _run()

    out = capsys.readouterr().out
    assert "ruff: 0 counting / 1 event(s)" in out
    assert "ignored: non-blocking (lint / infra failure)" in out
    assert "Action: hold" in out
    assert "Reason: no_recurring_failure" in out


def test_reports_the_climbed_rank_floor_and_its_model(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # rank_only: the two reds step a model RANK, which persists as the task's rank floor —
    # the state a user cannot otherwise see without reading the sqlite blob.
    pool = ModelPool()
    engine = _engine(pool, ladder="rank_only")
    engine.decide("s0", "task")
    _red(engine)
    _red(engine)
    model, reason, _prov = engine.decide("s1", "task")
    assert reason == "auto_escalation"
    _persist(engine)

    _run()

    out = capsys.readouterr().out
    floor = pool.rank_of(model)
    assert f"Rank floor:     {floor}" in out
    assert f"Model:          {model}" in out
    assert "escalation rank floor" in out


def test_json_report_carries_the_same_verdict(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = _engine(ModelPool())
    engine.decide("s0", "task")
    _red(engine)
    _red(engine)
    _persist(engine)

    _run(as_json=True)

    report = json.loads(capsys.readouterr().out)
    assert report["next_action"] == "raise_effort"
    assert report["keys"][0]["due"] is True
    assert report["enabled"] is True
    assert [c["key"] for c in report["config"]] == [
        "enabled",
        "escalate_after_n",
        "stale_window",
        "ladder",
        "rank_shortlist",
        "exploration_epsilon",
    ]


def test_config_values_name_the_file_that_set_them(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_dir = cli_env / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    policy = config_dir / "router.yaml"
    policy.write_text("router:\n  escalation:\n    enabled: true\n    escalate_after_n: 3\n")

    _run()

    out = capsys.readouterr().out
    assert f"escalate_after_n     3                ({policy})" in out
    # A knob the file left alone is reported as the default, not as file-sourced.
    assert "stale_window         10               (built-in default)" in out


def test_an_old_config_without_an_escalation_block_reports_why_it_is_off(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `parse_router_policy` treats an absent block as "never opted in" — the CLI must say
    # that, or the operator hunts for a knob that is not what turned escalation off.
    config_dir = cli_env / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "router.yaml").write_text("router:\n  strategy: always_cheap\n")

    _run()

    out = capsys.readouterr().out
    assert "Escalation:     DISABLED" in out
    assert "no `escalation:` block in the file" in out
    assert "Reason: disabled" in out


def test_a_pre_rename_knn_config_without_an_escalation_block_stays_enabled(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The non-brick rule: `strategy: knn` is the pre-rename spelling of the escalating
    # default, so the absent-block escape hatch must NOT turn the ladder off under it —
    # that would silently change what every pre-rename install does.
    config_dir = cli_env / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "router.yaml").write_text("router:\n  strategy: knn\n")

    _run()

    out = capsys.readouterr().out
    assert "Escalation:     enabled" in out


def test_unresolved_work_dir_reports_escalation_as_inert(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_repo = cli_env / "not-a-repo"
    not_a_repo.mkdir()
    monkeypatch.chdir(not_a_repo)

    _run(work_dir=None)

    out = capsys.readouterr().out
    assert "Work dir:       (unresolved)" in out
    assert "INERT" in out


def _plant_collapsed_corpus(store: OutcomeStore, model: str, n: int = 40) -> None:
    """n verified sessions all on ONE model: cold start ends and the collapse alarm fires."""
    import numpy as np

    for i in range(n):
        sid = f"collapse-{i}"
        store.store_session(
            session_id=sid,
            prompt_text="x",
            # Embedded on purpose: the effective-sample-size gate that ends cold start counts
            # only sessions the kNN index can see.
            embedding=np.zeros(8, dtype=np.float32),
            model_chosen=model,
            cost=1.0,
            cache_stats={},
            duration=1.0,
        )
        store.store_outcome(
            session_id=sid,
            tier1_outcome="success",
            tier1_confidence=1.0,
            tier2_outcome="success",
            tier2_confidence=1.0,
            aggregated_confidence=1.0,
        )


def test_collapse_guard_is_reported_as_suppressing_escalation(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The recorded failure mode: escalation silently does nothing while the routing-collapse
    # guard holds. A due window that still HOLDs must show WHY.
    pool = ModelPool()
    frontier = pool.ranked_models()[-1].name
    engine = _engine(pool)
    engine.decide("s0", "task")
    _red(engine)
    _red(engine)
    store = OutcomeStore()
    _plant_collapsed_corpus(store, frontier)
    store.save_escalation_state(engine.snapshot_escalation_state())
    store.close()

    _run()

    out = capsys.readouterr().out
    assert "Routing-collapse alarm: YES" in out
    assert "Cold start active:      no" in out
    assert "Escalation suppressed:  YES" in out
    assert "Reason: collapse_suppressed" in out

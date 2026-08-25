"""Hyperparameter e2e harness — capture/refit knobs + the budget skip."""


# Knob → case matrix:
#   capture.verify_timeout_seconds      null vs 0.001
#   capture.rerun_confirmations         1 vs 2  (0 is unexpressible: schema ge=1)
#   router.refit.every_n_outcomes       7 vs 100 (config-read wiring proof)
#   router.budget.max_spend_usd         SKIPPED — covered by Phase 2

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.e2e.helpers import (
    chat_body,
    close_session,
    failure_log_len,
    make_repo,
    parse_decision,
    post_completion,
    wait_capture_idle,
)
from tests.e2e.hyperparameter_helpers import TASK_A, refit_cadence, seed_outcome, write_policy

# A test that fails on the first two runs and passes from the third (a run counter
# marker). Rerun-confirmation re-runs a failing suite on unchanged state, so the
# outcome depends on how many reruns the verifier is configured to make.
_FLAKY_TWICE = (
    "import os\n"
    "import pathlib\n"
    "\n"
    "MARKER = pathlib.Path(os.path.join(os.path.dirname(__file__), 'flake2.marker'))\n"
    "def test_x():\n"
    "    n = int(MARKER.read_text()) if MARKER.exists() else 0\n"
    "    MARKER.write_text(str(n + 1))\n"
    "    assert n >= 2\n"
)


def _capture_one_session(client: TestClient) -> str:
    """POST a completion, close the session, drain the capture queue, return the sid."""
    resp = post_completion(client, chat_body())
    sid = resp.headers["X-Shunt-Session-Id"]
    close_session(client, sid)
    wait_capture_idle(client)
    return sid


def _boot_green(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> tuple[TestClient, Path]:
    """Boot the served app over a green repo with a mutated capture policy."""
    write_policy(tmp_path / "config", mutate)
    repo = make_repo(tmp_path / "repo", kind="green")
    return app_factory(repo=repo), repo


# ── 11. verify_timeout_seconds: a too-tiny budget makes capture a silent no-op ─


@pytest.mark.parametrize(
    ("timeout", "expected_tier2"),
    [
        # null = the verifier's built-in 1800s floor: the green suite completes and
        # a verified Tier-2 success is recorded.
        pytest.param(None, "success", id="verify_timeout=null"),
        # 1ms < any real suite: subprocess.run times out, the run records NOTHING and
        # the capture loop is a documented no-op (outcome 'unknown' is never a label).
        pytest.param(0.001, None, id="verify_timeout=tiny"),
    ],
)
def test_verify_timeout_seconds_gates_whether_capture_records(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    timeout: float | None,
    expected_tier2: str | None,
) -> None:
    client, _repo = _boot_green(
        app_factory,
        tmp_path,
        mutate=lambda r: r["capture"].update(verify_timeout_seconds=timeout),
    )
    with client:
        sid = _capture_one_session(client)
        row = client.app.state.outcome_store.get_outcome(sid)
        assert (row or {}).get("tier2_outcome") == expected_tier2


# ── 12. rerun_confirmations: reruns are what make a flaky red abstain ────────


@pytest.mark.parametrize(
    ("reruns", "expected_tier2", "expected_failure_log"),
    [
        # One rerun reproduces the second red → confirmed failure, recorded + counted.
        pytest.param(1, "failure", 1, id="rerun_confirmations=1"),
        # Two reruns hit the third-run pass → flake, abstained: nothing recorded.
        pytest.param(2, None, 0, id="rerun_confirmations=2"),
    ],
)
def test_rerun_confirmations_determine_a_flaky_repo_s_outcome(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    reruns: int,
    expected_tier2: str | None,
    expected_failure_log: int,
) -> None:
    client, repo = _boot_green(
        app_factory,
        tmp_path,
        mutate=lambda r: r["capture"].update(rerun_confirmations=reruns),
    )
    (repo / "test_flaky2.py").write_text(_FLAKY_TWICE)
    with client:
        sid = _capture_one_session(client)
        row = client.app.state.outcome_store.get_outcome(sid)
        assert (row or {}).get("tier2_outcome") == expected_tier2
        # The confirmed-failure path feeds the escalation log; the flake path must not.
        assert failure_log_len(client, str(repo)) == expected_failure_log


# ── 13. SHUNT_COLD_START_THRESHOLD_TIER2: the tier-2 bar ends cold start ──────


@pytest.mark.parametrize(
    ("tier2_threshold", "expected_reason"),
    [
        # 0 verified Tier-2 outcomes needed → the router is warm with a 5-outcome corpus.
        pytest.param("0", "cheapest_above_threshold", id="cold_start_tier2=0"),
        # A bar above the corpus keeps cold-start routing active with the same 5 outcomes.
        pytest.param("100", "cold_start", id="cold_start_tier2=100"),
    ],
)
def test_cold_start_tier2_threshold_gates_warm_versus_cold(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier2_threshold: str,
    expected_reason: str,
) -> None:
    monkeypatch.delenv("SHUNT_COLD_START_THRESHOLD_TIER1", raising=False)
    write_policy(
        tmp_path / "config",
        mutate=lambda r: r["exploration"].update(enabled=False),
    )
    client = app_factory(repo=None)
    monkeypatch.setenv("SHUNT_COLD_START_THRESHOLD_TIER2", tier2_threshold)
    with client:
        store = client.app.state.outcome_store
        for i in range(5):
            seed_outcome(
                store,
                session_id=f"cs-{i}",
                text=TASK_A,
                model="deepseek-v4-flash",
                cost=0.01,
                outcome="success",
            )
        resp = post_completion(client, chat_body(content=TASK_A))
        model, reason = parse_decision(resp.headers["X-Shunt-Decision"])
        assert reason == expected_reason
        assert model == "deepseek-v4-flash"  # same cheap model either way; the REASON differs


# ── 14. refit.every_n_outcomes: the served app wires the configured cadence ───


@pytest.mark.parametrize(
    ("every_n", "expected_cadence"),
    [
        pytest.param(7, 7, id="every_n_outcomes=7"),
        pytest.param(100, 100, id="every_n_outcomes=100"),
    ],
)
def test_every_n_outcomes_config_reaches_the_served_refit_scheduler(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    every_n: int,
    expected_cadence: int,
) -> None:
    # The scheduler's rebuild cadence is driven by every_n (the Phase-2 suite proves
    # the rebuild fires AT that cadence); this proves the SERVED app reads the config
    # key — two non-default values must land on the scheduler unchanged.
    write_policy(
        tmp_path / "config",
        mutate=lambda r: r["refit"].update(every_n_outcomes=every_n),
    )
    with app_factory() as client:
        assert refit_cadence(client) == expected_cadence


# ── 15. budget.max_spend_usd: covered by Phase 2 — deliberately not duplicated ─


@pytest.mark.skip(
    reason="category (a) covered-by-construction: tests/e2e/test_budget_hard_stop.py "
    "already proves max_spend_usd enforcement e2e (402 refusal naming the cap on the "
    "REAL recorded-cost path); duplicating it here would double-bill the same knob."
)
def test_max_spend_usd_hard_stop_covered_by_phase_2() -> None:
    raise AssertionError("this case is deliberately skipped (see the mark reason)")

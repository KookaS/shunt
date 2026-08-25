"""Hyperparameter e2e harness — auto-escalation knobs (driven via the real capture path)."""


# Knob → case matrix:
#   escalation.escalate_after_n     2 vs 100
#   escalation.stale_window         2 vs 10
#   escalation.rank_shortlist       1 vs 3

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.e2e.helpers import (
    chat_body,
    close_session,
    make_repo,
    parse_decision,
    post_completion,
    wait_failure_count,
)
from tests.e2e.hyperparameter_helpers import ranked_model, write_policy


def _drive_session(client: TestClient, work_dir: str, expected_failures: int) -> tuple[str, str]:
    """POST a completion, close the session, wait for *expected_failures* verified reds."""
    resp = post_completion(client, chat_body())
    model, reason = parse_decision(resp.headers["X-Shunt-Decision"])
    close_session(client, resp.headers["X-Shunt-Session-Id"])
    wait_failure_count(client, work_dir, expected_failures)
    return model, reason


def _boot_red(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> tuple[TestClient, str]:
    """Boot the served app over a deterministically-red repo with a mutated policy."""
    write_policy(tmp_path / "config", mutate)
    repo = make_repo(tmp_path / "repo", kind="red")
    client = app_factory(repo=repo)
    return client, str(repo)


# ── 8. escalate_after_n: the Nth same verified failure is the step trigger ────


@pytest.mark.parametrize(
    ("n", "escalates_on_session"),
    [
        # Ships at 2: the second same-key verified red steps a rung at the next boundary.
        pytest.param(2, 3, id="escalate_after_n=2"),
        # A threshold no sequence this test drives reaches: nothing ever escalates.
        pytest.param(100, None, id="escalate_after_n=100"),
    ],
)
def test_escalate_after_n_fires_at_the_nth_verified_failure(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    n: int,
    escalates_on_session: int | None,
) -> None:
    client, work_dir = _boot_red(
        app_factory, tmp_path, mutate=lambda r: r["escalation"].update(escalate_after_n=n)
    )
    with client:
        # Failure-log counts per session: 1,2 then (when an escalation fires at session
        # 3 the window is RETIRED so its own red restarts at 1) 1,2. For n=100 the log
        # just grows 1,2,3,4 and no escalation can ever fire.
        expected = [1, 2, (1, 2, 1, 2)[2] if n == 2 else 3, (1, 2, 1, 2)[3] if n == 2 else 4]
        decisions: list[tuple[str, str]] = []
        for i, expected_count in enumerate(expected):
            decisions.append(_drive_session(client, work_dir, expected_count))
            if escalates_on_session is not None:
                # Never before the Nth boundary; the Nth boundary does.
                assert (decisions[i][1] == "auto_escalation") == (i + 1 == escalates_on_session)
        if escalates_on_session is None:
            assert all(reason != "auto_escalation" for _, reason in decisions)


# ── 9. stale_window: a failure recurring outside the window is retired ───────


@pytest.mark.parametrize(
    ("window", "escalates"),
    [
        # With stale_window=2 the FIRST failure (decision 0) is already outside the
        # window when decision 2 is routed, so only one countable failure remains.
        pytest.param(2, False, id="stale_window=2"),
        # With the shipped window both failures recur within it → a step fires.
        pytest.param(10, True, id="stale_window=10"),
    ],
)
def test_stale_window_retires_failures_outside_it(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    window: int,
    escalates: bool,
) -> None:
    client, work_dir = _boot_red(
        app_factory, tmp_path, mutate=lambda r: r["escalation"].update(stale_window=window)
    )
    with client:
        # Two reds land at decision indices 0 and 1; the escalation check runs at
        # decision 2. When the window is large both recur; when it is 2 the first is
        # retired ((2 - 0) >= 2) and no recurrence reaches escalate_after_n=2.
        decisions = [_drive_session(client, work_dir, count) for count in (1, 2, 1)]
        escalated = any(reason == "auto_escalation" for _, reason in decisions)
        assert escalated is escalates
        if escalates:
            # Only the boundary AFTER the second verified red steps (effort rung: same
            # model, higher reasoning arm — cache-safe).
            assert decisions[2][1] == "auto_escalation"
            assert decisions[2][0] == ranked_model(client, 0)  # deepseek-v4-flash (cheapest)


# ── 10. rank_shortlist: the shortlist shape sets the rank-step target ─────────


@pytest.mark.parametrize(
    ("shortlist", "expected_rank_index"),
    [
        # 1 = jump straight past the mid-tier to the top rank (the whole point of the knob).
        pytest.param(1, -1, id="rank_shortlist=1"),
        # 3 (shipped) = walk the 3 cheapest ranks one at a time: from deepseek(0) the
        # rank step's first target is zai-glm-5.2(1).
        pytest.param(3, 1, id="rank_shortlist=3"),
    ],
)
def test_rank_shortlist_sets_the_rank_step_target(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    shortlist: int,
    expected_rank_index: int,
) -> None:
    client, work_dir = _boot_red(
        app_factory, tmp_path, mutate=lambda r: r["escalation"].update(rank_shortlist=shortlist)
    )
    with client:
        # Sessions 1-2 accrue the two reds that step the EFFORT rung at session 3
        # (same model, cache-safe); sessions 4-5 accrue the two more that step RANK at
        # session 5. The rank target is what the shortlist determines.
        decisions = [_drive_session(client, work_dir, count) for count in (1, 2, 1, 2, 1)]
        # The effort rung fired first (same model, still deepseek).
        assert decisions[2][1] == "auto_escalation"
        assert decisions[2][0] == ranked_model(client, 0)
        # The rank rung lands on the shortlist-defined target — never the intermediate ranks.
        assert decisions[4][1] == "auto_escalation"
        assert decisions[4][0] == ranked_model(client, expected_rank_index)

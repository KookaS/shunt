"""End-to-end routing + auto-escalation through the REAL served app with a FAKE upstream."""

# The loop under test: HTTP request -> engine decide (cold-start routing, escalation at
# the boundary) -> canned upstream response -> session close -> real pytest subprocess
# -> Tier-2 outcome -> engine failure log -> NEXT session's decision escalates.
#
# Only the upstream (`_acompletion`) and the embedder are faked; the proxy, capture
# worker/coordinator, verifier subprocess, store and escalation engine are all real.
#
# Scope: cold-start routing + auto-escalation + capture + fixed strategies. LEARNED
# kNN routing (post-cold-start, >=20 verified outcomes) is deliberately out of e2e
# scope — driving that many verified sessions over HTTP is not worth the runtime, and
# it is covered by the router unit suites.

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest
from fastapi.testclient import TestClient

from tests.e2e.helpers import (
    CHAT_PATH,
    MESSAGES_PATH,
    chat_body,
    close_session,
    failure_log_len,
    make_repo,
    parse_decision,
    post_completion,
    wait_capture_idle,
    wait_failure_count,
    wait_outcome,
)

_ANTHROPIC_BODY: Final[dict[str, Any]] = {
    "model": "claude-opus-4-6",
    "messages": [{"role": "user", "content": "Fix it"}],
    "max_tokens": 100,
    "stream": False,
}


def _drive_session(client: TestClient, *, path: str = CHAT_PATH) -> tuple[Any, str]:
    """POST a chat body (same tool identity), return ``(response, session_id)``."""
    resp = post_completion(client, chat_body(), path=path)
    return resp, resp.headers["X-Shunt-Session-Id"]


# ── The two user scenarios ────────────────────────────────────────────────────


@pytest.mark.parametrize("routed_app", ["green"], indirect=True)
def test_green_repo_routes_and_never_escalates(
    routed_app: tuple[TestClient, Path],
) -> None:
    """A fake model that outputs predefined text and does not fail: routing works,
    the text passes through, the outcome is verified success, and nothing escalates."""
    client, repo = routed_app
    work_dir = str(repo)

    r1, sid1 = _drive_session(client)
    model1, reason1 = parse_decision(r1.headers["X-Shunt-Decision"])
    assert model1 == "deepseek-v4-flash"
    assert reason1 == "cold_start"
    assert r1.json()["choices"][0]["message"]["content"] == "Hello back"
    close_session(client, sid1)
    wait_outcome(client, sid1, "success")

    r2, sid2 = _drive_session(client)
    close_session(client, sid2)
    wait_outcome(client, sid2, "success")

    r3, _sid3 = _drive_session(client)
    _, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 != "auto_escalation"
    assert failure_log_len(client, work_dir) == 0


@pytest.mark.parametrize("routed_app", ["red"], indirect=True)
def test_same_failure_twice_escalates_next_session(
    routed_app: tuple[TestClient, Path], mock_acompletion: Any
) -> None:
    """A fake model that keeps producing the same verified failure: after two
    same-check reds, the NEXT session escalates the reasoning arm (cache-safe)."""
    client, repo = routed_app
    work_dir = str(repo)

    r1, sid1 = _drive_session(client)
    model1, reason1 = parse_decision(r1.headers["X-Shunt-Decision"])
    # Precondition, not an assertion of taste: reason=cold_start proves the corpus is
    # fresh and trusted, so the escalate path (not _decide_stale_space) is what runs.
    assert model1 == "deepseek-v4-flash"
    assert reason1 == "cold_start"
    close_session(client, sid1)
    wait_failure_count(client, work_dir, 1)

    r2, sid2 = _drive_session(client)
    _, reason2 = parse_decision(r2.headers["X-Shunt-Decision"])
    assert reason2 == "cold_start"
    close_session(client, sid2)
    wait_failure_count(client, work_dir, 2)

    r3, _sid3 = _drive_session(client)
    model3, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 == "auto_escalation"
    assert model3 == "deepseek-v4-flash"  # effort rung: same model, higher reasoning arm
    # The escalated arm reaches the WIRE. `thinking` is not an OpenAI SDK kwarg, so it
    # travels in extra_body; `reasoning_effort` is SDK-native and travels as a kwarg.
    assert mock_acompletion.call_args_list[2].kwargs.get("reasoning_effort") == "max"
    assert mock_acompletion.call_args_list[2].kwargs.get("extra_body") == {
        "thinking": {"type": "enabled"}
    }


# ── Escalation semantics ──────────────────────────────────────────────────────


@pytest.mark.parametrize("routed_app", ["red"], indirect=True)
def test_fail_then_fix_clears_the_log(
    routed_app: tuple[TestClient, Path],
) -> None:
    """A verified pass retires the failure window — the ladder is not sticky."""
    client, repo = routed_app
    work_dir = str(repo)

    r1, sid1 = _drive_session(client)
    close_session(client, sid1)
    wait_failure_count(client, work_dir, 1)

    make_repo(repo, kind="green")  # the agent "fixes" the build

    r2, sid2 = _drive_session(client)
    close_session(client, sid2)
    wait_outcome(client, sid2, "success")
    # The success capture popped the failure window (poll the log, not the DB — the
    # DB commit precedes the engine's pop, so an immediate read can race it).
    wait_failure_count(client, work_dir, 0)

    r3, _sid3 = _drive_session(client)
    _, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 != "auto_escalation"
    assert reason3 == "cold_start"


@pytest.mark.parametrize("routed_app", ["red"], indirect=True)
def test_distinct_failures_do_not_aggregate(
    routed_app: tuple[TestClient, Path],
) -> None:
    """Two DIFFERENT failing checks are two hard tasks, not one recurring failure."""
    client, repo = routed_app
    work_dir = str(repo)

    make_repo(repo, kind="red_a")  # fails test_a.py::test_a
    r1, sid1 = _drive_session(client)
    close_session(client, sid1)
    wait_failure_count(client, work_dir, 1)

    make_repo(repo, kind="red_b")  # now fails test_b.py::test_b — different key
    r2, sid2 = _drive_session(client)
    close_session(client, sid2)
    wait_failure_count(client, work_dir, 2)

    r3, _sid3 = _drive_session(client)
    _, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 != "auto_escalation"


@pytest.mark.parametrize("routed_app", ["red"], indirect=True)
def test_second_recurrence_reaches_the_rank_rung(
    routed_app: tuple[TestClient, Path], mock_acompletion: Any
) -> None:
    """The ladder climbs effort first, then a model RANK on a second recurrence."""
    client, repo = routed_app
    work_dir = str(repo)

    r1, sid1 = _drive_session(client)
    close_session(client, sid1)
    wait_failure_count(client, work_dir, 1)

    r2, sid2 = _drive_session(client)
    close_session(client, sid2)
    wait_failure_count(client, work_dir, 2)

    r3, sid3 = _drive_session(client)  # 1st escalation: effort arm on the SAME model
    model3, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 == "auto_escalation"
    assert model3 == "deepseek-v4-flash"
    assert mock_acompletion.call_args_list[2].kwargs.get("reasoning_effort") == "max"
    close_session(client, sid3)
    # The escalation RETIRED the window at decide-time, so s3's own red restarts the
    # count at 1 — two MORE same-key failures are needed for the next rung.
    wait_failure_count(client, work_dir, 1)

    r4, sid4 = _drive_session(client)  # window was retired -> no re-fire
    _, reason4 = parse_decision(r4.headers["X-Shunt-Decision"])
    assert reason4 == "cold_start"
    close_session(client, sid4)
    wait_failure_count(client, work_dir, 2)

    r5, sid5 = _drive_session(client)  # 2nd recurrence -> RANK step
    model5, reason5 = parse_decision(r5.headers["X-Shunt-Decision"])
    assert reason5 == "auto_escalation"
    assert model5 != "deepseek-v4-flash"

    r6 = post_completion(client, chat_body())  # 2nd turn on the rank-escalated session
    assert r6.headers["X-Shunt-Session-Id"] == sid5
    model6, reason6 = parse_decision(r6.headers["X-Shunt-Decision"])
    assert model6 == model5  # the RANK step also locks the new model for the session
    assert reason6 == "auto_escalation"
    close_session(client, sid5)
    wait_capture_idle(client)


# ── Capture semantics ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("routed_app", ["flaky"], indirect=True)
def test_flaky_test_never_escalates(
    routed_app: tuple[TestClient, Path],
) -> None:
    """A fail-then-pass flake is abstained by rerun-confirmation: no label, no step."""
    client, repo = routed_app
    work_dir = str(repo)
    marker = repo / "flake.marker"

    r1, sid1 = _drive_session(client)
    close_session(client, sid1)
    wait_capture_idle(client)
    assert client.app.state.outcome_store.get_outcome(sid1) is None
    assert failure_log_len(client, work_dir) == 0
    marker.unlink(missing_ok=True)

    r2, sid2 = _drive_session(client)
    close_session(client, sid2)
    wait_capture_idle(client)
    assert client.app.state.outcome_store.get_outcome(sid2) is None
    assert failure_log_len(client, work_dir) == 0
    marker.unlink(missing_ok=True)

    r3, _sid3 = _drive_session(client)
    _, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 != "auto_escalation"


@pytest.mark.parametrize("routed_app", ["infra"], indirect=True)
def test_infra_failure_never_escalates(
    routed_app: tuple[TestClient, Path],
) -> None:
    """An env-cause red (import error) is non-blocking and withheld from escalation."""
    client, repo = routed_app
    work_dir = str(repo)

    r1, sid1 = _drive_session(client)
    close_session(client, sid1)
    wait_capture_idle(client)
    assert failure_log_len(client, work_dir) == 0

    r2, sid2 = _drive_session(client)
    close_session(client, sid2)
    wait_capture_idle(client)
    assert failure_log_len(client, work_dir) == 0

    r3, _sid3 = _drive_session(client)
    _, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 != "auto_escalation"


def test_manual_only_without_work_dir_never_escalates(
    app_factory: Any, mock_acompletion: Any
) -> None:
    """Escalation ships ON but is INERT without a capture.work_dir: no verified-failure
    signal can exist, so no session may ever escalate. The precondition guard, live."""
    with app_factory(repo=None) as client:
        r1, sid1 = _drive_session(client)
        _, reason1 = parse_decision(r1.headers["X-Shunt-Decision"])
        assert reason1 == "cold_start"
        close_session(client, sid1)
        wait_capture_idle(client)
        assert client.app.state.outcome_store.get_outcome(sid1) is None

        r2, sid2 = _drive_session(client)
        _, reason2 = parse_decision(r2.headers["X-Shunt-Decision"])
        assert reason2 == "cold_start"
        close_session(client, sid2)
        wait_capture_idle(client)

        r3, _sid3 = _drive_session(client)
        _, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
        assert reason3 != "auto_escalation"


# ── Wire semantics ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("routed_app", ["red"], indirect=True)
def test_escalated_arm_persists_across_turns_same_session(
    routed_app: tuple[TestClient, Path], mock_acompletion: Any
) -> None:
    """Cache-safety spine: within the escalated session the model is LOCKED and the
    raised arm is re-applied on every later turn — never re-routed mid-cached-turn."""
    client, repo = routed_app
    work_dir = str(repo)

    r1, sid1 = _drive_session(client)
    close_session(client, sid1)
    wait_failure_count(client, work_dir, 1)

    r2, sid2 = _drive_session(client)
    close_session(client, sid2)
    wait_failure_count(client, work_dir, 2)

    r3, sid3 = _drive_session(client)  # escalated, left OPEN
    model3, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 == "auto_escalation"
    assert mock_acompletion.call_args_list[2].kwargs.get("reasoning_effort") == "max"

    r4 = post_completion(client, chat_body())  # 2nd turn, same session, no close
    assert r4.headers["X-Shunt-Session-Id"] == sid3
    model4, reason4 = parse_decision(r4.headers["X-Shunt-Decision"])
    assert model4 == model3
    assert reason4 == "auto_escalation"
    assert mock_acompletion.call_args_list[3].kwargs.get("reasoning_effort") == "max"
    close_session(client, sid3)
    wait_capture_idle(client)


@pytest.mark.parametrize("routed_app", ["red"], indirect=True)
def test_streaming_escalation(routed_app: tuple[TestClient, Path], mock_acompletion: Any) -> None:
    """The same loop on the SSE wire: the escalated request streams AND carries the arm."""
    client, repo = routed_app
    work_dir = str(repo)

    r1, sid1 = _drive_session(client)
    close_session(client, sid1)
    wait_failure_count(client, work_dir, 1)

    r2, sid2 = _drive_session(client)
    close_session(client, sid2)
    wait_failure_count(client, work_dir, 2)

    r3 = post_completion(client, chat_body(stream=True))
    model3, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 == "auto_escalation"
    assert model3 == "deepseek-v4-flash"
    assert "Hello" in r3.text
    assert "data: [DONE]" in r3.text
    kwargs3 = mock_acompletion.call_args_list[2].kwargs
    assert kwargs3.get("stream") is True
    assert kwargs3.get("reasoning_effort") == "max"
    close_session(client, r3.headers["X-Shunt-Session-Id"])
    wait_capture_idle(client)


@pytest.mark.parametrize("routed_app", ["red"], indirect=True)
def test_anthropic_messages_escalation(
    routed_app: tuple[TestClient, Path], mock_acompletion: Any
) -> None:
    """The same loop on the Anthropic /v1/messages wire (Claude Code's path)."""
    client, repo = routed_app
    work_dir = str(repo)

    r1 = post_completion(client, _ANTHROPIC_BODY, path=MESSAGES_PATH)
    sid1 = r1.headers["X-Shunt-Session-Id"]
    close_session(client, sid1)
    wait_failure_count(client, work_dir, 1)

    r2 = post_completion(client, _ANTHROPIC_BODY, path=MESSAGES_PATH)
    sid2 = r2.headers["X-Shunt-Session-Id"]
    close_session(client, sid2)
    wait_failure_count(client, work_dir, 2)

    r3 = post_completion(client, _ANTHROPIC_BODY, path=MESSAGES_PATH)
    model3, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
    assert reason3 == "auto_escalation"
    assert model3 == "deepseek-v4-flash"
    assert mock_acompletion.call_args_list[2].kwargs.get("reasoning_effort") == "max"
    assert r3.json()["content"][0]["text"] == "Hello back"


# ── Fixed-strategy routing allocation ─────────────────────────────────────────


def test_always_frontier_allocates_a_frontier_model(
    app_factory: Any, mock_acompletion: Any
) -> None:
    """Routing allocated to the frontier: the cheap default is never picked, the fake
    model's predefined text still reaches the client, and nothing escalates."""
    with app_factory(repo=None, strategy="always_frontier") as client:
        pool = client.app.state.model_pool
        top_model = pool.ranked_models()[-1].name

        r1, _sid1 = _drive_session(client)
        model1, reason1 = parse_decision(r1.headers["X-Shunt-Decision"])
        assert model1 == top_model
        assert reason1 == "always_frontier"
        assert r1.json()["choices"][0]["message"]["content"] == "Hello back"

        r2, _sid2 = _drive_session(client)
        _, reason2 = parse_decision(r2.headers["X-Shunt-Decision"])
        assert reason2 == "always_frontier"  # fixed strategy never escalates


def test_session_cascade_starts_cheap_then_climbs(
    app_factory: Any, tmp_path: Path, mock_acompletion: Any
) -> None:
    """The `session_cascade` preset, end to end: cheapest model first, then the ladder."""
    # The LIVE spelling of the benchmark's Session-Cascade row — the cache-safe operating point
    # the blocked Price-Cascade / kNN-cascade rows approximate at one decision per attempt.
    # `always_cheap` above proves the base rule and that a pinned control never escalates; this
    # asserts the opposite for the preset, which is the whole difference between the two ids.
    repo = make_repo(tmp_path / "cascade-repo", kind="red")
    with app_factory(repo=repo, strategy="session_cascade") as client:
        work_dir = str(repo)
        cheapest = client.app.state.model_pool.ranked_models()[0].name

        r1, sid1 = _drive_session(client)
        model1, reason1 = parse_decision(r1.headers["X-Shunt-Decision"])
        assert model1 == cheapest
        # Its own reason token, not always_cheap's: the two pick the same model but differ
        # on whether a verified failure may move it, and an analysis filtering on
        # `always_cheap` to find CONTROL sessions must not scoop up cascade sessions.
        assert reason1 == "session_cascade"
        close_session(client, sid1)
        wait_failure_count(client, work_dir, 1)

        r2, sid2 = _drive_session(client)
        _, reason2 = parse_decision(r2.headers["X-Shunt-Decision"])
        assert reason2 == "session_cascade"  # one red is fail-then-fix, not a rung
        close_session(client, sid2)
        wait_failure_count(client, work_dir, 2)

        r3, _sid3 = _drive_session(client)
        model3, reason3 = parse_decision(r3.headers["X-Shunt-Decision"])
        assert reason3 == "auto_escalation"
        # Effort rung first: the served model is unchanged, so the cache namespace is too.
        # A rank step here would be the cache-safety spine broken, not a better cascade.
        assert model3 == cheapest


def test_session_cascade_verified_outcome_is_visible_over_http(
    app_factory: Any, tmp_path: Path, mock_acompletion: Any
) -> None:
    """The default strategy's verified outcomes must be observable on /admin/loop-health."""
    # The handshake harness's escalation drivers are bare HTTP clients: they cannot reach the
    # engine's failure log, so they pace their prompts by polling this endpoint for a landed
    # verified outcome. They used to poll `label_coverage.verified_labeled`, which joins on
    # `sessions.embedding_blob IS NOT NULL` — a kNN-corpus question — and `session_cascade`
    # never embeds, so that counter is 0 forever and all three legs timed out. Guard the
    # strategy-independent counter here rather than in the Docker harness alone.
    repo = make_repo(tmp_path / "visible-repo", kind="red")
    with app_factory(repo=repo, strategy="session_cascade") as client:
        _r1, sid1 = _drive_session(client)
        close_session(client, sid1)
        wait_failure_count(client, str(repo), 1)

        health = client.get("/admin/loop-health").json()
        assert health["label_coverage"]["verified_labeled"] == 0  # embedding-gated, by design
        assert health["verification"]["verified_outcomes"] >= 1


def test_always_cheap_allocates_the_cheapest_model(app_factory: Any, mock_acompletion: Any) -> None:
    """Routing allocated to the cheap end of the pool on a fixed strategy."""
    with app_factory(repo=None, strategy="always_cheap") as client:
        pool = client.app.state.model_pool
        cheapest = pool.ranked_models()[0].name

        r1, _sid1 = _drive_session(client)
        model1, reason1 = parse_decision(r1.headers["X-Shunt-Decision"])
        assert model1 == cheapest
        assert reason1 == "always_cheap"

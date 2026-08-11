# The scheduler itself is unit-tested in tests/capture/test_refit.py; this file proves
# the full wiring the units cannot: HTTP request -> session close -> real pytest
# subprocess -> verified Tier-2 outcome -> RefitScheduler.note_capture (server.py:477,
# coordinator.py:258) -> OutcomeStore.rebuild_index -> the next request routes on the
# learned neighbourhood. Only the upstream and the embedder are faked (as in the rest of
# the e2e suite); the proxy, capture worker/coordinator, verifier subprocess, store and
# engine are all real.
"""End-to-end refit cadence through the SERVED app: a captured outcome must trigger a
kNN-index rebuild at exactly ``router.refit.every_n_outcomes``, never before, and the
rebuilt index must change the routing decision (cold-start -> learned kNN)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from shunt.models.config import strict_yaml_load
from shunt.router.policy import packaged_policy_path
from tests.e2e.helpers import (
    chat_body,
    close_session,
    make_repo,
    parse_decision,
    post_completion,
    wait_capture_idle,
    wait_outcome,
)

# A small cadence so the test drives few sessions; the packaged default is 50.
EVERY_N_OUTCOMES = 3
# End cold-start at the SAME number of verified outcomes as the refit cadence, so the
# decision that follows the Nth capture is provably routing on the learned index.
COLD_START_THRESHOLD_TIER2 = EVERY_N_OUTCOMES


def _write_refit_config(config_dir: Path, *, every_n: int) -> None:
    """Write the packaged router.yaml with the tiny refit cadence into the hermetic
    $SHUNT_CONFIG_DIR before the app boots; exploration is disabled so the post-cold-start
    decision is the deterministic selection rule, not a TS sample."""
    # SHUNT_CONFIG_DIR is otherwise an empty dir that falls back to the packaged policy.
    data = strict_yaml_load(packaged_policy_path().read_text())
    router = data["router"]
    router["refit"]["every_n_outcomes"] = every_n
    router["exploration"]["enabled"] = False
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "router.yaml").write_text(yaml.safe_dump(data))


def _count_rebuilds(store: Any) -> dict[str, int]:
    """Wrap the store's public ``rebuild_index`` so a test can count scheduler-fired refits.

    Only the RefitScheduler calls it in the served path; boot-time rebuilds go through the
    private ``_rebuild_index``, so every increment IS a cadence-triggered refit.
    """
    real = store.rebuild_index
    counter: dict[str, int] = {"n": 0}

    def _counted() -> None:
        counter["n"] += 1
        real()

    store.rebuild_index = _counted
    return counter


def _drive_and_capture(client: TestClient, counter: dict[str, int]) -> str:
    """POST a completion, close the session, wait for the verified Tier-2 capture (which
    runs the repo's real pytest suite off the wire), and return the session id."""
    resp = post_completion(client, chat_body())
    sid = resp.headers["X-Shunt-Session-Id"]
    close_session(client, sid)
    wait_outcome(client, sid, "success")
    wait_capture_idle(client)
    return sid


@pytest.fixture
def refit_client(
    app_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, dict[str, int]]]:
    """Boot the served app over a green repo with every_n_outcomes=3 and cold-start
    ending at 3 verified outcomes; yields (client, rebuild-counter)."""
    _write_refit_config(tmp_path / "config", every_n=EVERY_N_OUTCOMES)
    monkeypatch.setenv("SHUNT_COLD_START_THRESHOLD_TIER2", str(COLD_START_THRESHOLD_TIER2))
    repo = make_repo(tmp_path / "repo", kind="green")
    with app_factory(repo=repo) as client:
        counter = _count_rebuilds(client.app.state.outcome_store)
        yield client, counter


def test_index_rebuilt_exactly_at_cadence(refit_client: Any) -> None:
    client, counter = refit_client

    captured: list[str] = []
    for i in range(6):
        captured.append(_drive_and_capture(client, counter))
        # The refit fires on the 3rd and 6th capture only: never before, never after.
        assert counter["n"] == (i + 1) // EVERY_N_OUTCOMES

    store = client.app.state.outcome_store
    indexed = {sid for sid, _ in store.get_labeled_embeddings()}
    assert indexed == set(captured)
    assert store._index.count == len(indexed) == len(captured)


def test_decision_routes_on_learned_outcome_after_refit(refit_client: Any) -> None:
    client, counter = refit_client

    # Pre-learned: every decision is cold-start, and the index holds nothing yet.
    for _ in range(3):
        resp = post_completion(client, chat_body())
        model, reason = parse_decision(resp.headers["X-Shunt-Decision"])
        assert model == "qwen3.7-plus"
        assert reason == "cold_start"
        close_session(client, resp.headers["X-Shunt-Session-Id"])
        wait_outcome(client, resp.headers["X-Shunt-Session-Id"], "success")
        wait_capture_idle(client)

    # The 3rd captured outcome fired the refit, and the index now holds the learned
    # sessions (the truthful labeled log re-projected into HNSW).
    assert counter["n"] == 1
    store = client.app.state.outcome_store
    assert store._index.count == len(store.get_labeled_embeddings()) == 3

    # The next request routes on the learned kNN neighbourhood, not cold-start.
    resp = post_completion(client, chat_body())
    model, reason = parse_decision(resp.headers["X-Shunt-Decision"])
    assert model == "qwen3.7-plus"
    assert reason == "cheapest_above_threshold"

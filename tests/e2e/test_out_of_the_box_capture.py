"""`cd myrepo && shunt start` must capture a verified outcome with no configuration."""

# The zero-configuration path is the one a first-time user takes, so it gets an end-to-end
# test rather than a unit test of the resolver: boot the real served app with its working
# directory inside a real repo, no ``SHUNT_WORK_DIR``, no ``capture.work_dir``, and require
# that a verified Tier-2 outcome lands in the store.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from shunt.proxy import server as server_module
from shunt.proxy.server import app
from tests.e2e.helpers import chat_body, close_session, post_completion, wait_outcome
from tests.fake_embedder import FakeEmbedder

_CONTROLLED_ENV: tuple[str, ...] = (
    "SHUNT_ROUTER_STRATEGY",
    "SHUNT_WORK_DIR",
    "SHUNT_EXPLORATION_ENABLED",
    "SHUNT_EXPLORE_BUDGET_FRAC",
    "SHUNT_MODEL_CONFIG_PATH",
)


# The launch-directory layer confines the repo to no root set, so the repo below can sit
# anywhere; $HOME is still moved into tmp_path so nothing a boot path derives from it
# (data dirs, caches) can touch the developer's real home or assume CI's is writable.
_PYPROJECTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "pep621": '[project]\nname = "demo"\noptional-dependencies = { dev = ["pytest"] }\n',
        # Poetry declares the dev dependency with the package name as the KEY, in a table no
        # other tool uses. It is the shape that most often defeats a structured detector.
        "poetry": '[tool.poetry.group.dev.dependencies]\npytest = "^8.0"\n',
    }
)


def _make_repo(root: Path, flavour: str) -> Path:
    repo = root / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(_PYPROJECTS[flavour])
    (repo / "test_x.py").write_text("def test_x():\n    assert True\n")
    return repo


@pytest.fixture(autouse=True)
def _mock_upstream() -> Iterator[AsyncMock]:
    """No real provider call: the canned upstream answers every request at $0."""
    from tests.e2e.helpers import make_fake_acompletion

    with patch("shunt.proxy.router._acompletion", new_callable=AsyncMock) as mock:
        mock.side_effect = make_fake_acompletion()
        yield mock


@pytest.mark.parametrize("flavour", sorted(_PYPROJECTS))
def test_launching_inside_a_repo_captures_a_verified_outcome_with_no_config(
    flavour: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _CONTROLLED_ENV:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    repo = _make_repo(home, flavour)
    monkeypatch.setenv("HOME", str(home))  # keep every derived path off the real home
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path / "config"))  # ⇒ packaged policy
    monkeypatch.setattr(server_module, "_MODEL_CONFIG_PATH", None)
    monkeypatch.setattr(server_module, "Embedder", FakeEmbedder)
    monkeypatch.chdir(repo)  # the whole configuration: `cd myrepo`

    with TestClient(app) as client:
        response = post_completion(client, chat_body())
        session_id = response.headers["X-Shunt-Session-Id"]
        close_session(client, session_id)
        # The green suite in the launch repo is re-run off the wire and labelled.
        wait_outcome(client, session_id, "success")
        row = client.app.state.outcome_store.get_outcome(session_id)

    assert row is not None
    # Tier-2, from the off-wire re-run of the launch repo's own suite — not a weak wire
    # prior, and never the model's own claim that its tests passed.
    assert row["tier2_outcome"] == "success"
    assert row["outcome_source"] == "auto_tier2"

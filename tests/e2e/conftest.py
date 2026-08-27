"""E2E fixtures: the real served app booted with a faked embedder + a mocked
upstream, hermetic fresh env per test (no real provider call, no ONNX)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from shunt.proxy import server as server_module
from shunt.proxy.server import app
from tests.e2e.helpers import make_fake_acompletion, make_repo
from tests.fake_embedder import FakeEmbedder

# Router env vars the suite controls; any leftover dev-machine value would make a
# test non-hermetic, so the factory deletes them all before setting its own.
_CONTROLLED_ENV: tuple[str, ...] = (
    "SHUNT_ROUTER_STRATEGY",
    "SHUNT_WORK_DIR",
    "SHUNT_EXPLORATION_ENABLED",
    "SHUNT_EXPLORE_BUDGET_FRAC",
    "SHUNT_MODEL_CONFIG_PATH",
)


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., TestClient]:
    """Boot the real served app under a fresh, hermetic environment."""

    # `knn_cascade`, not the packaged default: this suite's routing assertions were written
    # against the kNN base pick and its reason vocabulary (`cold_start`,
    # `cheapest_above_threshold`, `exploration_untested`), which the shipped `session_cascade`
    # default does not produce — it never consults the neighbourhood. The default is not left
    # untested by this: `test_routing_escalation.py` boots `session_cascade` explicitly, and
    # every test that cares about another strategy names it here.
    default_strategy = "knn_semantic_cascade"

    def _boot(*, repo: Path | None = None, strategy: str | None = default_strategy) -> TestClient:
        for name in _CONTROLLED_ENV:
            monkeypatch.delenv(name, raising=False)
        # SHUNT_CONFIG_DIR pointing at an empty tmp dir forces the PACKAGED router.yaml
        # + models.yaml (no dev-machine ~/.config/shunt overrides); SHUNT_DATA_DIR pins
        # a fresh store per test so the escalation counter never bleeds across tests.
        monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path / "config"))
        # The registry path is frozen at import time; reset it so a dev-machine
        # SHUNT_MODEL_CONFIG_PATH can't swap in a foreign pool mid-suite.
        monkeypatch.setattr(server_module, "_MODEL_CONFIG_PATH", None)
        if repo is not None:
            monkeypatch.setenv("SHUNT_WORK_DIR", str(repo))
        else:
            # `repo=None` must mean MANUAL-ONLY, and the launch-directory layer resolves
            # from the process cwd — which under pytest is this git repo, with a suite.
            # Boot from a hermetic non-repo dir so the refusal comes from the layer's own
            # git-root check rather than from wherever the suite happened to be started.
            manual_only = tmp_path / "not-a-repo"
            manual_only.mkdir(exist_ok=True)
            monkeypatch.chdir(manual_only)
        if strategy is not None:
            monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", strategy)
        monkeypatch.setattr(server_module, "Embedder", FakeEmbedder)
        return TestClient(app)

    return _boot


@pytest.fixture
def routed_app(
    app_factory: Callable[..., TestClient], tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[tuple[TestClient, Path]]:
    """The app booted over a repo of ``request.param`` kind (default ``red``)."""
    kind = getattr(request, "param", "red")
    repo = make_repo(tmp_path / "repo", kind=kind)
    with app_factory(repo=repo) as client:
        yield client, repo


# Autouse on purpose — a real provider call from any test would be a defect (the
# suite is defined as fake-LLM), and this machine's live keys would silently pass a
# broken assertion by returning real content.
@pytest.fixture(autouse=True)
def mock_acompletion() -> Iterator[AsyncMock]:
    """The fake upstream: every request returns the canned predefined text, $0."""
    with patch("shunt.proxy.router._acompletion", new_callable=AsyncMock) as mock:
        mock.side_effect = make_fake_acompletion()
        yield mock

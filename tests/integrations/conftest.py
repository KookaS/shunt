"""Shared hermetic-handshake harness: Shunt in-process, against a live fake upstream.

One factory, so a test can boot Shunt on ANY router.yaml (a strategy template, say)
without re-deriving the port-rewritten registry and the ONNX block.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import shunt.proxy.server as server
from tests.integrations.fake_upstream import FakeUpstream

_REGISTRY = Path(__file__).parent / "fake_registry.yaml"
_DOCKER_BASE_URL = "http://fake-upstream:9099/v1"

# The escape hatch every hermetic boot needs: a router.yaml naming the SHIPPED registry's
# models cannot boot against the 3-model fake registry, because `restrict_to_live` (rightly)
# refuses a live name the registry does not define. An empty list routes over whatever
# registry is active — what a custom-registry deployment wants, and what this harness is.
FAKE_ROUTER_YAML = "router:\n  models: []\n"


class FakeEmbedder:
    """A fixed-vector embedder so an in-process boot never loads real ONNX. Counts calls."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed(self, text: str) -> object:
        import numpy as np

        self.embed_calls += 1
        return np.full(768, 0.1, dtype=np.float32)

    def fingerprint(self) -> dict[str, object]:
        return {"repo": "fake", "dim": 768, "max_chars": 4000, "revision": None}

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def max_chars(self) -> int:
        return 4000

    def warm(self) -> None:
        return None


# `make(router_yaml)` -> a context manager over the booted trio.
HandshakeFactory = Callable[
    [str], AbstractContextManager[tuple[TestClient, FakeUpstream, "FakeEmbedder"]]
]


@pytest.fixture
def handshake_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[HandshakeFactory]:
    """Yield `make(router_yaml)` → a context manager over (client, upstream, embedder)."""

    @contextmanager
    def _make(router_yaml: str) -> Iterator[tuple[TestClient, FakeUpstream, FakeEmbedder]]:
        # One shared instance so a test can assert whether the request path embedded at all —
        # the observable difference between the two shipped strategies.
        embedder = FakeEmbedder()
        # Real ONNX loading is (correctly) blocked in the unit suite; inject a fake so the
        # routing path stays hermetic and never downloads the ~600MB model.
        monkeypatch.setattr(server, "Embedder", lambda *a, **k: embedder)
        with FakeUpstream() as upstream:
            registry = _REGISTRY.read_text().replace(_DOCKER_BASE_URL, f"{upstream.base_url}/v1")
            registry_path = tmp_path / "fake_registry.yaml"
            registry_path.write_text(registry)
            monkeypatch.setattr(server, "_MODEL_CONFIG_PATH", str(registry_path))
            monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
            (tmp_path / "router.yaml").write_text(router_yaml)
            with TestClient(server.app) as client:
                yield client, upstream, embedder

    yield _make

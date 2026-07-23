"""The read-only loop-health admin endpoint: aggregate metrics only, no raw prompts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shunt.proxy.server import app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Isolate the store on a tmp dir so the endpoint reads only test-written data.
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    with TestClient(app) as c:
        yield c


def _seed(client: TestClient) -> None:
    store = client.app.state.outcome_store
    for i in range(3):
        store.store_session(
            f"s{i}",
            "SECRET-PROMPT-TEXT",  # PII-bearing raw prompt — must never surface in the payload
            np.random.randn(64).astype(np.float32),
            "some-model",
            1.0,
            {},
            1.0,
        )


def test_endpoint_returns_aggregate_metrics(client: TestClient) -> None:
    _seed(client)
    resp = client.get("/admin/loop-health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {
        "label_coverage",
        "propensity_support",
        "routing_collapse",
        "cost",
        "support_deficient_models",
    }
    assert body["label_coverage"]["total_sessions"] == 3


def test_endpoint_never_leaks_raw_prompt(client: TestClient) -> None:
    _seed(client)
    resp = client.get("/admin/loop-health")
    raw = json.dumps(resp.json())
    assert "prompt_text" not in raw
    assert "SECRET-PROMPT-TEXT" not in raw


def test_endpoint_is_read_only(client: TestClient) -> None:
    # A mutating verb on a read-only endpoint must be rejected, not silently accepted.
    assert client.post("/admin/loop-health").status_code == 405

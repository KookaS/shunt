"""Each shipped strategy template actually drives a request end to end.

Boots Shunt on the template itself (real config load, real routing, `_acompletion` not
mocked) against the same hermetic fake upstream the tool handshakes use, over both wire
formats. Proves the wiring — the template loads, the strategy it names is the strategy
that decides, and the header says so. It proves nothing about model quality: no provider
is called and nothing is billed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import yaml

from shunt.session.manager import SessionManager
from tests.integrations.conftest import HandshakeFactory

_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = _ROOT / "examples" / "strategies"

# What each template's decision must SAY on a fresh install, and whether the request path
# embedded. The reason token is the observable difference between the two strategies:
# `session_cascade` routes from the pool alone, while the kNN cascade embeds the first turn
# and (with an empty outcome store) reports the cold-start pick.
_EXPECTED: Final[dict[str, tuple[str, bool]]] = {
    "session-cascade": ("session_cascade", False),
    "knn-semantic-cascade": ("cold_start", True),
}

# The fake registry's cheapest row, and the cold-start default — both templates open here.
_EXPECTED_MODEL = "deepseek-v4-flash"


def _template_paths() -> list[Path]:
    return sorted(_TEMPLATES.glob("*.yaml"))


def _bootable(path: Path) -> str:
    """The template verbatim, with ONE key changed: the live-model pool.

    The template names the SHIPPED registry's models; against the 3-model fake registry
    `restrict_to_live` would rightly refuse to boot. An empty list routes over whatever
    registry is active. Nothing else is touched, and the assertion below proves it.
    """
    raw = yaml.safe_load(path.read_text())
    before = dict(raw["router"])
    raw["router"]["models"] = []
    assert {k: v for k, v in raw["router"].items() if k != "models"} == {
        k: v for k, v in before.items() if k != "models"
    }
    return str(yaml.safe_dump(raw))


def test_templates_exist() -> None:
    # Guards the parametrized tests below against silently passing on an empty glob.
    assert len(_template_paths()) == 2


@pytest.mark.parametrize("path", _template_paths(), ids=lambda p: p.stem)
def test_template_routes_over_the_openai_wire(
    path: Path, handshake_factory: HandshakeFactory
) -> None:
    reason, embeds = _EXPECTED[path.stem]
    with handshake_factory(_bootable(path)) as (client, upstream, embedder):
        body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        resp = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer dummy"}
        )
        assert resp.status_code == 200
        decision = resp.headers["X-Shunt-Decision"]
        assert decision == f"{_EXPECTED_MODEL}; reason={reason}", decision
        assert resp.json()["choices"][0]["message"]["content"] == "ok"
        assert any("/chat/completions" in hit for hit in upstream.received)
        # The dependency claim in the template's own header, asserted: only the kNN
        # cascade loads an embedder in the request path.
        assert (embedder.embed_calls > 0) is embeds


@pytest.mark.parametrize("path", _template_paths(), ids=lambda p: p.stem)
def test_template_routes_over_the_anthropic_wire(
    path: Path, handshake_factory: HandshakeFactory
) -> None:
    reason, _embeds = _EXPECTED[path.stem]
    with handshake_factory(_bootable(path)) as (client, upstream, _embedder):
        body = {
            "model": "auto",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        resp = client.post("/v1/messages", json=body, headers={"x-api-key": "dummy"})
        assert resp.status_code == 200
        assert resp.headers["X-Shunt-Decision"] == f"{_EXPECTED_MODEL}; reason={reason}"
        assert any("/chat/completions" in hit for hit in upstream.received)


@pytest.mark.parametrize("path", _template_paths(), ids=lambda p: p.stem)
def test_decision_provenance_names_the_template_strategy(
    path: Path, handshake_factory: HandshakeFactory
) -> None:
    """The routed session's provenance carries the id THIS FILE named, not the default's."""
    named = str(yaml.safe_load(path.read_text())["router"]["strategy"])
    agent = f"shunt-strategy-handshake/{path.stem}"
    with handshake_factory(_bootable(path)) as (client, _upstream, _embedder):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer dummy", "User-Agent": agent},
        )
        assert resp.status_code == 200
        manager: SessionManager = client.app.state.session_manager
        identity = SessionManager.compute_tool_identity("testclient", agent)
        session = manager.get_session_by_identity(identity)
        assert session is not None, "the request created no session to read provenance from"
        assert (session.decision_provenance or {}).get("strategy_id") == named

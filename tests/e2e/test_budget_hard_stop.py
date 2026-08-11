# The live tier's documented spend cap is `router.budget.max_spend_usd`. This proves the
# enforcement against the REAL recorded cost path: the mocked upstream reports
# `usage.cost` (the same seam `session.total_cost` accumulates in the shipped router), and
# once a session's cumulative reported spend reaches the cap the next request is REFUSED —
# a clean 402 naming the cap (a status no client SDK retries), never a fabricated success,
# with the refusal carried in the decision header and an explicit `x-should-retry: false`.
"""Budget hard-stop end to end, hermetically (fake embedder, mocked upstream)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from tests.e2e.helpers import (
    CHAT_PATH,
    MESSAGES_PATH,
    canned_chat_response,
    chat_body,
    user_agent_headers,
)

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"

_CONFIG = """\
router:
  strategy: knn
  models: []
  budget:
    max_spend_usd: {cap}
"""


def _write_policy(cfg_dir: Path, *, cap: str) -> None:
    """Write the budget-tuned router.yaml into the SHUNT_CONFIG_DIR app_factory uses."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "router.yaml").write_text(_CONFIG.format(cap=cap))


def _cost_response(cost: float) -> MagicMock:
    """A canned upstream completion that REPORTS a given charge on usage.cost."""
    resp = canned_chat_response()
    resp.usage.cost = cost
    return resp


def test_request_under_cap_routes_normally(
    app_factory: Callable[..., TestClient], tmp_path: Path
) -> None:
    """(a) A request under the cap routes normally and accumulates real cost."""
    _write_policy(tmp_path / "config", cap="0.01")
    with app_factory() as client, patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock:
        mock.return_value = _cost_response(0.0004)
        first = client.post(
            CHAT_PATH, json=chat_body(content="under cap"), headers=user_agent_headers()
        )
        sid = first.headers["X-Shunt-Session-Id"]
        second = client.post(
            CHAT_PATH, json=chat_body(content="still under"), headers=user_agent_headers()
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] == "Hello back"
    # The provider-reported charges were accumulated onto the session — the REAL cost
    # the cap is enforced against, never a locally derived price.
    session = client.app.state.session_manager.get_session(sid)
    assert session is not None
    assert session.total_cost == 0.0008
    assert mock.await_count == 2


def test_cumulative_spend_crossing_cap_hard_stops(
    app_factory: Callable[..., TestClient], tmp_path: Path
) -> None:
    """(b)+(c)+(d) Spend reaching the cap hard-stops the NEXT request with a clean
    refusal that names the cap and is recorded in the decision header — and the refusal
    happens BEFORE any upstream call, so nothing is fabricated."""
    _write_policy(tmp_path / "config", cap="0.001")
    with app_factory() as client, patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock:
        mock.return_value = _cost_response(0.001)
        under = client.post(
            CHAT_PATH, json=chat_body(content="first"), headers=user_agent_headers()
        )
        sid = under.headers["X-Shunt-Session-Id"]
        refused = client.post(
            CHAT_PATH, json=chat_body(content="second"), headers=user_agent_headers()
        )

    # The first request routed (0.0 < cap) and recorded $0.001 of reported spend.
    assert under.status_code == 200
    session = client.app.state.session_manager.get_session(sid)
    assert session is not None
    assert session.total_cost == 0.001

    # The second request: cumulative spend == cap => hard-stop, not a fabricated success.
    # 402 Payment Required: the spend cap is permanent, so no client SDK retries it; the
    # explicit `x-should-retry: false` header is the belt-and-suspenders on top.
    assert refused.status_code == 402
    assert "X-Shunt-Session-Id" in refused.headers
    assert refused.headers.get("x-should-retry") == "false"
    assert mock.await_count == 1  # the refusal never reached the upstream
    payload = refused.json()
    assert payload.get("error") is not None
    assert payload.get("choices") is None  # no fake completion body
    # (d) the error message names the cap.
    message = payload["error"]["message"]
    assert "max_spend_usd" in message
    assert "$0.001000" in message
    # (c) the refusal is recorded on the wire: the decision header says so.
    decision = refused.headers.get("X-Shunt-Decision", "")
    assert "refusing to route" in decision
    assert "max_spend_usd" in decision


def test_refusal_applies_to_the_anthropic_surface(
    app_factory: Callable[..., TestClient], tmp_path: Path
) -> None:
    """The hard-stop is in the shared routing seam, so /v1/messages refuses too."""
    _write_policy(tmp_path / "config", cap="0.0005")
    with app_factory() as client, patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock:
        mock.return_value = _cost_response(0.001)
        under = client.post(
            CHAT_PATH, json=chat_body(content="chat first"), headers=user_agent_headers()
        )
        assert under.status_code == 200  # 0.0 < 0.0005; $0.001 recorded after
        refused = client.post(
            MESSAGES_PATH,
            json=chat_body(content="messages now"),
            headers=user_agent_headers(),
        )

    assert refused.status_code == 402
    assert refused.headers.get("x-should-retry") == "false"
    assert "max_spend_usd" in refused.json()["error"]["message"]
    assert mock.await_count == 1

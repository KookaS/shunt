"""The routing decision must not block the event loop while it embeds."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from shunt.models.config import ModelPool
from shunt.proxy.router import ProxyRouter
from shunt.session import SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"


@pytest.mark.asyncio
async def test_a_slow_decision_does_not_stall_the_event_loop() -> None:
    """Drives the real _route_with_fallback, not a stand-in for it."""
    # Inline, the decision ran ONNX inference and took a blocking lock ON the loop, so
    # concurrent first turns serialized and every in-flight SSE stream stalled with them.
    router = ProxyRouter(
        model_pool=ModelPool(),
        session_manager=SessionManager(inactivity_timeout=900, grace_period=120),
        retry_count=1,
    )
    decide_delay = 0.25

    def slow_decide(session: object, prompt_text: str = "") -> str:
        time.sleep(decide_delay)  # blocking, exactly like the real embed
        return "model-a"

    stalls: list[float] = []

    async def heartbeat() -> None:
        while True:
            before = time.perf_counter()
            await asyncio.sleep(0.01)
            stalls.append(time.perf_counter() - before)

    beat = asyncio.create_task(heartbeat())
    try:
        with (
            patch.object(router, "_get_or_lock_model", side_effect=slow_decide),
            patch(_ACOMPLETION_PATCH, new=AsyncMock(return_value=MagicResponse())),
        ):
            sessions = [
                router._sessions.create_session(f"tool-{i}")  # type: ignore[attr-defined]
                for i in range(3)
            ]
            # The decision is made before the upstream chain is walked, so an empty
            # pool exhausting afterwards is irrelevant here — the timing is the assert.
            await asyncio.gather(
                *(
                    router._route_with_fallback({"messages": []}, s)  # type: ignore[arg-type]
                    for s in sessions
                ),
                return_exceptions=True,
            )
    finally:
        beat.cancel()

    # A stalled loop shows up as a heartbeat tick that could not fire for ~the
    # decision time. Off-loop, no single tick comes close to one decision.
    assert stalls, "heartbeat never ran"
    assert max(stalls) < decide_delay, f"event loop stalled for {max(stalls):.3f}s"


class MagicResponse:
    """Minimal stand-in for an OpenAI completion response."""

    def __init__(self) -> None:
        self.choices: list[object] = []
        self.usage = None

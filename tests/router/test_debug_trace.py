"""The debug trace must explain a routing decision without a debugger attached.

Cold start short-circuits before the kNN query, so on a fresh rig the interesting half
of the trace never appears. These drive the warm path directly.
"""

from __future__ import annotations

import logging

import pytest

from shunt.models.config import ModelPool
from shunt.router.selection import NeighborResult, SelectionRule


def _neighbors(model: str, n: int, *, ok: bool) -> list[NeighborResult]:
    return [
        NeighborResult(
            model=model,
            outcome=ok,
            cost=0.001,
            verification_confidence=0.9,
            distance=0.1,
            session_id=f"{model}-{i}",
        )
        for i in range(n)
    ]


def test_it_says_why_each_model_passed_or_failed_the_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rule = SelectionRule(min_success_rate=0.6, min_samples=3)
    neighbors = _neighbors("qwen3.7-plus", 4, ok=True) + _neighbors("kimi-k3", 4, ok=False)

    with caplog.at_level(logging.DEBUG, logger="shunt.router.selection"):
        rule.select(neighbors, ModelPool(), cold_start_active=False)

    assert "model=qwen3.7-plus samples=4/3 success=1.000/0.600" in caplog.text
    assert "ELIGIBLE" in caplog.text
    # The failing model must be visible too — "why did it NOT pick X" is the question.
    assert "model=kimi-k3 samples=4/3 success=0.000/0.600" in caplog.text
    assert "rejected" in caplog.text


def test_an_empty_neighbourhood_announces_the_escalation_fall_through(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # This path returns the CHEAPEST model under a learned-sounding reason, so it must
    # be distinguishable in the log from a genuine threshold-clearing choice.
    rule = SelectionRule(min_success_rate=0.6, min_samples=3)
    with caplog.at_level(logging.DEBUG, logger="shunt.router.selection"):
        _, reason = rule.select([], ModelPool(), cold_start_active=False)

    assert reason == "exploration_untested"
    assert "no model cleared the threshold" in caplog.text


def test_too_few_samples_is_reported_as_such(caplog: pytest.LogCaptureFixture) -> None:
    rule = SelectionRule(min_success_rate=0.6, min_samples=3)
    with caplog.at_level(logging.DEBUG, logger="shunt.router.selection"):
        rule.select(_neighbors("qwen3.7-plus", 2, ok=True), ModelPool(), cold_start_active=False)

    assert "samples=2/3" in caplog.text
    assert "rejected" in caplog.text

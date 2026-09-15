from __future__ import annotations

from dataclasses import dataclass

import pytest

from shunt.proxy.session_drift import (
    COUNTER_NAMES,
    MAX_WINDOW,
    MIN_WINDOW,
    WINDOW_GRID,
    SessionCounters,
    drift_level,
    summarize,
    window_features,
    wire_errors_from_metadata,
)
from shunt.proxy.wire_signals import WIRE_TOOL_ERROR_COUNT


@dataclass
class _Step:
    is_revert: bool = False
    retry_count: int = 0
    loop_signal: bool = False


def _counters(
    *, revert: int = 0, retry: int = 0, loop: int = 0, wire: int = 0, steps: int = 10
) -> SessionCounters:
    return SessionCounters(
        is_reverts=revert,
        retry_total=retry,
        loop_signals=loop,
        wire_tool_errors=wire,
        n_steps=steps,
    )


def test_window_grid_spans_three_to_ten() -> None:
    assert MIN_WINDOW == 3
    assert MAX_WINDOW == 10
    assert WINDOW_GRID == (3, 4, 5, 6, 7, 8, 9, 10)
    assert COUNTER_NAMES == ("is_revert", "retry_count", "loop_signal", "wire_tool_error_count")


def test_summarize_counts_each_per_step_counter() -> None:
    steps = [
        _Step(is_revert=True, retry_count=2, loop_signal=False),
        _Step(is_revert=False, retry_count=0, loop_signal=True),
        _Step(is_revert=True, retry_count=3, loop_signal=True),
    ]
    summary = summarize(steps)
    assert summary.is_reverts == 2
    assert summary.retry_total == 5
    assert summary.loop_signals == 2
    assert summary.n_steps == 3


def test_summarize_reads_the_session_wire_peak_not_the_step_list() -> None:
    summary = summarize([_Step()], wire_tool_errors=4)
    assert summary.wire_tool_errors == 4
    assert summarize([_Step()]).wire_tool_errors == 0


def test_wire_errors_from_metadata_reads_the_live_key() -> None:
    assert wire_errors_from_metadata({WIRE_TOOL_ERROR_COUNT: 2}) == 2
    assert wire_errors_from_metadata({}) == 0
    assert wire_errors_from_metadata({WIRE_TOOL_ERROR_COUNT: "bad"}) == 0


def test_rates_use_step_count_and_never_divide_by_zero() -> None:
    summary = _counters(revert=2, retry=4, loop=1, wire=3, steps=10)
    assert summary.rates() == (0.2, 0.4, 0.1, 0.3)
    empty = _counters(revert=1, retry=1, loop=1, wire=1, steps=0)
    assert empty.rates() == (1.0, 1.0, 1.0, 1.0)


def test_window_features_are_means_with_a_last_minus_first_slope() -> None:
    first = _counters(revert=0, retry=0, loop=0, wire=0, steps=10)
    last = _counters(revert=5, retry=5, loop=5, wire=5, steps=10)
    features = window_features([first, first, last])
    assert features.window == 3
    assert features.mean_rates == (pytest.approx(1 / 6),) * 4
    assert features.drift_level == pytest.approx(4 * (0.5 / 3))
    assert features.drift_slope == pytest.approx(4 * 0.5)
    assert features.to_dict()["window"] == 3


def test_stable_window_has_zero_slope_and_empty_window_raises() -> None:
    steady = _counters(revert=1, retry=1, loop=1, wire=1, steps=10)
    assert window_features([steady, steady, steady]).drift_slope == pytest.approx(0.0)
    assert drift_level([steady, steady]) == pytest.approx(4 * 0.1)
    with pytest.raises(ValueError, match="must not be empty"):
        window_features([])

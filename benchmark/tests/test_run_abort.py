"""Run-level abort: an unusable API (dead key / no balance) or N consecutive failures abort
the WHOLE run instead of marching through hundreds of cells writing garbage.

All stubbed (no live/paid/Docker): ``_run_one_cell`` is monkeypatched to yield sentinels.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from benchmark.runner import lane_scheduler, run_matrix

SF: Final = run_matrix._START_FAILURE
_OK: Final[dict[str, Any]] = {"real_cost": 0.0}


def _unusable(reason: str = "no balance") -> run_matrix._ApiUnusable:
    return run_matrix._ApiUnusable(reason)


def _inject(monkeypatch, results: list[Any]) -> list[tuple[str, str, str]]:
    """Make ``_run_one_cell`` yield the next injected result; return the per-cell call log."""
    calls: list[tuple[str, str, str]] = []
    seq = iter(results)

    def fake_run_one_cell(cell: tuple[str, str, str], _ctx: Any) -> Any:
        calls.append(cell)
        return next(seq)

    monkeypatch.setattr(run_matrix, "_run_one_cell", fake_run_one_cell)
    return calls


def _cells(n: int) -> list[tuple[str, str, str]]:
    """One-cell-per-challenge list (per-cell == per-challenge here)."""
    return [(f"repo__task-{i}", "m", "default") for i in range(1, n + 1)]


# --- immediate abort on the first unambiguous API-unusable cell -----------------------


def test_api_unusable_aborts_immediately_serial(monkeypatch) -> None:
    # First cell is API-unusable → abort at once; cells 2..10 never start (retrying a dead key
    # is pointless). The abort message names the cause.
    calls = _inject(monkeypatch, [_unusable("insufficient balance")] + [_OK] * 9)
    with pytest.raises(run_matrix.RunAbortError, match="insufficient balance"):
        run_matrix._run_cells_serial(_cells(10), ctx=None, max_cost=None)  # type: ignore[arg-type]
    assert len(calls) == 1  # aborted on the very first unusable cell


def test_api_unusable_is_never_recorded_as_a_row(monkeypatch) -> None:
    # A successful cell, THEN an unusable one: the good row is kept (checkpointed), the unusable
    # cell aborts — it is never turned into a fake pass=False row.
    calls = _inject(monkeypatch, [{"real_cost": 1.0}, _unusable(), _OK])
    with pytest.raises(run_matrix.RunAbortError):
        run_matrix._run_cells_serial(_cells(3), ctx=None, max_cost=None)  # type: ignore[arg-type]
    assert len(calls) == 2  # ran the good cell, hit the unusable one, aborted


def test_api_unusable_aborts_parallel(monkeypatch) -> None:
    calls = _inject(monkeypatch, [_unusable()] * 10)
    with pytest.raises(run_matrix.RunAbortError):
        run_matrix._run_cells_parallel(
            _cells(10),
            ctx=None,  # type: ignore[arg-type]
            workers=2,
            max_cost=None,
        )
    assert len(calls) >= 1


# --- consecutive-any-failure catch-all ------------------------------------------------


def test_consecutive_any_failures_abort(monkeypatch) -> None:
    # 5 consecutive None-skips (rate-limit / dep error) with a catch-all cap of 5 → abort at the
    # 5th; the dedicated container-start counter is untouched (these aren't start failures).
    calls = _inject(monkeypatch, [None] * 10)
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        run_matrix._run_cells_serial(
            _cells(10),
            ctx=None,  # type: ignore[arg-type]
            max_cost=None,
            max_start_failures=None,
            max_consecutive_failures=5,
        )
    assert len(calls) == 5


def test_success_resets_consecutive_counter(monkeypatch) -> None:
    # Failures interspersed with a success never reach 5 CONSECUTIVE → no abort.
    results = [None] * 4 + [_OK] + [None] * 4 + [_OK] + [None] * 4
    calls = _inject(monkeypatch, results)
    rows = run_matrix._run_cells_serial(
        _cells(len(results)),
        ctx=None,  # type: ignore[arg-type]
        max_cost=None,
        max_consecutive_failures=5,
    )
    assert len(calls) == len(results)  # ran every cell, never aborted
    assert len(rows) == 2  # the two successful cells


def test_start_failures_count_toward_consecutive_catch_all(monkeypatch) -> None:
    # Container-start failures also count toward the catch-all: with no start-failure cap but a
    # consecutive cap of 3, three start-failures in a row still abort the run.
    calls = _inject(monkeypatch, [SF] * 6)
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        run_matrix._run_cells_serial(
            _cells(6),
            ctx=None,  # type: ignore[arg-type]
            max_cost=None,
            max_start_failures=None,
            max_consecutive_failures=3,
        )
    assert len(calls) == 3


def test_no_consecutive_cap_never_aborts_on_skips(monkeypatch) -> None:
    # Default (None) restores pure skip-and-continue: 10 None-skips, no abort.
    calls = _inject(monkeypatch, [None] * 10)
    rows = run_matrix._run_cells_serial(_cells(10), ctx=None, max_cost=None)  # type: ignore[arg-type]
    assert len(calls) == 10 and rows == []


def test_consecutive_abort_parallel(monkeypatch) -> None:
    calls = _inject(monkeypatch, [None] * 10)
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        run_matrix._run_cells_parallel(
            _cells(10),
            ctx=None,  # type: ignore[arg-type]
            workers=2,
            max_cost=None,
            max_start_failures=None,
            max_consecutive_failures=3,
        )
    assert len(calls) >= 3


# --- rate limiting: a per-lane 429 is a MISSING skip, never a row or an abort ----------


def test_is_rate_limited_classifies_429_signatures_and_not_balance_errors() -> None:
    from benchmark.runner import infer

    assert infer.is_rate_limited(RuntimeError("Error code: 429 - rate limit exceeded"))
    assert infer.is_rate_limited(RuntimeError("Too Many Requests"))
    assert not infer.is_rate_limited(RuntimeError("insufficient balance"))
    assert not infer.is_rate_limited(RuntimeError("some unrelated failure"))


def test_retry_after_seconds_reads_headers_and_messages() -> None:
    from types import SimpleNamespace

    from benchmark.runner import infer

    # A structured header (litellm carries an httpx.Response under `.response`, or a plain
    # `headers` mapping) wins over the message.
    exc = RuntimeError("429 rate limit")
    exc.headers = {"Retry-After": "25"}  # type: ignore[attr-defined]
    assert infer.retry_after_seconds(exc) == 25.0
    nested = RuntimeError("429 rate limit")
    nested.response = SimpleNamespace(headers={"retry-after": "42"})  # type: ignore[attr-defined]
    assert infer.retry_after_seconds(nested) == 42.0
    # A `Retry-After: <n>` fragment in the message is parsed when no header is present.
    assert infer.retry_after_seconds(RuntimeError("429 retry after 17 seconds")) == 17.0
    # No Retry-After anywhere, or a non-numeric HTTP-date, is None — never invented.
    assert infer.retry_after_seconds(RuntimeError("429 rate limit")) is None
    assert (
        infer.retry_after_seconds(RuntimeError("retry-after: Wed, 21 Oct 2015 07:28:00 GMT"))
        is None
    )


def test_run_one_cell_returns_a_rate_limited_sentinel(monkeypatch) -> None:
    from types import SimpleNamespace

    from benchmark.runner import infer

    def fake_run_live_cell(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setattr(infer, "run_live_cell", fake_run_live_cell)
    ctx = SimpleNamespace(work_dir=".", timeout=1, step_limit=1)
    result = run_matrix._run_one_cell(("c1", "m", "default"), ctx)  # type: ignore[arg-type]
    assert isinstance(result, run_matrix._RateLimited)


def test_rate_limited_cell_is_skipped_not_recorded_serial(monkeypatch) -> None:
    calls = _inject(monkeypatch, [run_matrix._RateLimited("429"), _OK])
    rows = run_matrix._run_cells_serial(_cells(2), ctx=None, max_cost=None)  # type: ignore[arg-type]
    assert len(calls) == 2
    assert rows == [_OK]  # the throttled cell wrote no row


def test_rate_limited_cell_is_skipped_not_recorded_parallel(monkeypatch) -> None:
    calls = _inject(monkeypatch, [run_matrix._RateLimited("429"), _OK])
    rows = run_matrix._run_cells_parallel(
        _cells(2),
        ctx=None,  # type: ignore[arg-type]
        workers=2,
        max_cost=None,
    )
    assert len(calls) == 2
    assert rows == [_OK]


# --- permanent pre-call model error: skip, never a poison row, run continues ----------


def _live_ctx() -> Any:
    """Minimal _LiveContext stand-in for the real _run_one_cell path (no container)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        hashes={"repo__task-1": "h", "repo__task-2": "h"},
        versions={"m": "v"},
        pricing={},
        digests=None,
        arm_hash_map=None,
        work_dir=".",
        timeout=1,
        step_limit=150,
        free_collection_models=frozenset(),
    )


def test_permanent_pre_call_error_yields_no_invalid_row_and_run_proceeds(monkeypatch) -> None:
    # A provider 400 before any call makes run_live_cell return
    # pass=False/stop_reason=unsolved/calls=0. As a row that is the UNSOLVED_NOT_RUN poison
    # invariant — it must be classified as a skip (cell MISSING), never crash the campaign.
    from benchmark.routing import censoring, integrity
    from benchmark.runner import infer

    permanent = infer._errored_outcome(
        "repo__task-1", "m", stop_reason=censoring.UNSOLVED, step_limit=150, cost_limit=0.0
    )
    good: dict[str, Any] = {
        "pass": True,
        "in_tok": 10,
        "out_tok": 5,
        "calls": 3,
        "real_cost": 0.0,
        "stop_reason": censoring.SOLVED,
        "computed_at": "2026-09-10T00:00:00+00:00",
        "step_limit": "150",
        "cost_limit": "0.0",
    }
    seen: list[str] = []

    def fake_run_live_cell(cid: str, model: str, **_kw: Any) -> dict[str, object]:
        seen.append(cid)
        return permanent if cid.endswith("task-1") else good

    monkeypatch.setattr(infer, "run_live_cell", fake_run_live_cell)
    # Not materialised in the test checkout; the spec hash is an anchor, not the subject.
    monkeypatch.setattr(integrity, "swebench_spec_hash", lambda _cid: "h")

    cells = [("repo__task-1", "m", "default"), ("repo__task-2", "m", "default")]
    ctx = _live_ctx()

    # The poison row is converted to the never-ran sentinel, not raised.
    assert isinstance(run_matrix._run_one_cell(cells[0], ctx), run_matrix._PermanentPreCallFailure)

    # ...and the batch keeps going: only the good cell produces a row (the other stays MISSING).
    seen.clear()
    rows = run_matrix._run_cells_serial(cells, ctx, max_cost=None)
    assert [r["challenge_id"] for r in rows] == ["repo__task-2"]
    assert seen == ["repo__task-1", "repo__task-2"]


def test_model_unavailable_disables_the_lane_and_writes_no_row(monkeypatch) -> None:
    # A model the provider does not serve is a LANE fault: disable (model, provider) and never
    # record a fake pass=False. The regression this pins: the live taxonomy never classified it,
    # so requesty-laguna-xs.2 was retried 3x per cell and never disabled.
    lanes = lane_scheduler.LaneScheduler(
        limits={"m": lane_scheduler.LaneLimits(rpm=1000, rpd=100000)},
        stall_timeout_s=0.01,
    )
    _inject(monkeypatch, [run_matrix._ModelUnavailable("404: please check the model")])
    tracker = run_matrix._FailureTracker(None, None)
    rows, _spent, _stopped = run_matrix._run_scheduled_batch(
        [("c1", "m", "default")],
        ctx=None,  # type: ignore[arg-type]
        lanes=lanes,
        tracker=tracker,
        checkpoint=None,
        spent=0.0,
        hard=None,
        max_cost=None,
        overshoot_note="",
    )
    assert rows == []
    reason = lanes.lane_state("m").disabled_reason or ""
    assert "model unavailable" in reason


def test_permanent_pre_call_failure_counts_toward_consecutive_catch_all(monkeypatch) -> None:
    # The skip is still a failure: enough permanent pre-call failures in a row trip the
    # catch-all abort, so a systematically broken provider/model pair cannot march through
    # the whole matrix. Asserted through the real serial loop, so the wiring is exercised.
    sentinel = run_matrix._PERMANENT_PRE_CALL_FAILURE
    calls = _inject(monkeypatch, [sentinel] * 10)
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        run_matrix._run_cells_serial(
            _cells(10),
            ctx=None,  # type: ignore[arg-type]
            max_cost=None,
            max_start_failures=None,
            max_consecutive_failures=2,
        )
    assert len(calls) == 2

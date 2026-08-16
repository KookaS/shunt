"""Unit tests for the stop_reason vocabulary and CENSORED-data helpers."""

from __future__ import annotations

from benchmark.routing import censoring


class TestStopReasonFromRun:
    """The mini-swe-agent exit_status → stop_reason mapping on the success path."""

    def test_resolved_is_solved(self) -> None:
        assert censoring.stop_reason_from_run(resolved=True, exit_status="Submitted") == "solved"

    def test_resolved_wins_over_limit_exit(self) -> None:
        # A limit exit_status is irrelevant once the harness resolved the instance.
        assert censoring.stop_reason_from_run(resolved=True, exit_status="TimeExceeded") == "solved"

    def test_finished_unsolved(self) -> None:
        assert censoring.stop_reason_from_run(resolved=False, exit_status="Submitted") == "unsolved"

    def test_limits_exceeded_maps_to_step_limit(self) -> None:
        assert (
            censoring.stop_reason_from_run(resolved=False, exit_status="LimitsExceeded")
            == "step_limit"
        )

    def test_time_exceeded_maps_to_wall_limit(self) -> None:
        assert (
            censoring.stop_reason_from_run(resolved=False, exit_status="TimeExceeded")
            == "wall_limit"
        )

    def test_unknown_exit_status_degrades_to_unsolved(self) -> None:
        # An unrecognised exit signal is NOT silently censored — it is a genuine unsolved.
        assert (
            censoring.stop_reason_from_run(resolved=False, exit_status="RepeatedFormatError")
            == "unsolved"
        )
        assert censoring.stop_reason_from_run(resolved=False, exit_status="") == "unsolved"


class TestCensoredClassification:
    def test_censored_reasons(self) -> None:
        for reason in ("step_limit", "wall_limit", "abandoned"):
            assert censoring.is_censored_reason(reason)
        for reason in ("solved", "unsolved"):
            assert not censoring.is_censored_reason(reason)

    def test_timeout_flag_only_for_wall_and_abandon(self) -> None:
        assert censoring.timeout_flag_for("wall_limit") is True
        assert censoring.timeout_flag_for("abandoned") is True
        # step_limit is a GRACEFUL censored stop — not a timeout.
        assert censoring.timeout_flag_for("step_limit") is False
        assert censoring.timeout_flag_for("unsolved") is False
        assert censoring.timeout_flag_for("solved") is False


class TestDeriveStopReason:
    """Legacy rows (no stop_reason column) derive one from pass + timeout_flag."""

    def test_explicit_value_wins(self) -> None:
        assert (
            censoring.derive_stop_reason(passed=False, timeout_flag=True, stop_reason="step_limit")
            == "step_limit"
        )

    def test_legacy_pass_is_solved(self) -> None:
        assert censoring.derive_stop_reason(passed=True, timeout_flag=False) == "solved"

    def test_legacy_timeout_is_wall_limit(self) -> None:
        # A legacy fail with timeout_flag set was a censored wall/abandon stop.
        assert censoring.derive_stop_reason(passed=False, timeout_flag=True) == "wall_limit"

    def test_legacy_plain_fail_is_unsolved(self) -> None:
        assert censoring.derive_stop_reason(passed=False, timeout_flag=False) == "unsolved"


class TestIsCensoredRow:
    def test_new_row_with_stop_reason(self) -> None:
        assert censoring.is_censored({"pass": False, "stop_reason": "step_limit"}) is True
        assert censoring.is_censored({"pass": True, "stop_reason": "solved"}) is False
        assert censoring.is_censored({"pass": False, "stop_reason": "unsolved"}) is False

    def test_legacy_row_without_stop_reason(self) -> None:
        # No stop_reason column: a timeout fail is censored; a plain fail is not.
        assert censoring.is_censored({"pass": False, "timeout_flag": True}) is True
        assert censoring.is_censored({"pass": False, "timeout_flag": False}) is False
        assert censoring.is_censored({"pass": True}) is False

    def test_empty_cell_is_not_censored(self) -> None:
        assert censoring.is_censored({}) is False

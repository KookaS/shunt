"""Effective-sample-size cold→warm gate: same thresholds, measured in nₑ not raw count."""

from __future__ import annotations

from shunt.router.cold_start import ColdStartStrategy


def _strategy() -> ColdStartStrategy:
    return ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)


class TestEffectiveGateBackwardCompat:
    def test_uniform_weights_match_raw_count_gate(self) -> None:
        # With nₑ == raw count (uniform confidences), the effective gate must decide exactly
        # as the raw-count gate does at the same values.
        s = _strategy()
        assert s.is_active_effective(50.0, 0.0) == s.is_active(50, 0)  # both inactive
        assert s.is_active_effective(19.0, 19.0) == s.is_active(19, 19)  # both active
        assert s.is_active_effective(30.0, 20.0) == s.is_active(30, 20)  # tier2 boundary


class TestEffectiveGateBoundaries:
    def test_tier2_boundary_ends_cold_start(self) -> None:
        assert _strategy().is_active_effective(0.0, 20.0) is False

    def test_tier1_boundary_ends_cold_start(self) -> None:
        assert _strategy().is_active_effective(50.0, 0.0) is False

    def test_just_below_both_stays_cold(self) -> None:
        assert _strategy().is_active_effective(49.9, 19.9) is True


class TestEffectivePersistsLongerThanRawCount:
    def test_low_confidence_outcomes_keep_cold_start_active(self) -> None:
        # 25 verified rows would end cold-start under a raw-count gate (25 >= 20). But if
        # each is low-confidence, nₑ falls under 20 → cold-start correctly persists.
        raw_count = 25
        assert _strategy().is_active(raw_count, raw_count) is False  # raw-count gate: warm
        ne_low_conf = 12.0  # what 25 barely-confident rows collapse to
        assert _strategy().is_active_effective(ne_low_conf, ne_low_conf) is True  # nₑ: still cold

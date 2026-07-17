"""Tests for the deterministic calibration-holdout sampler.

Pins the two stability properties the design depends on: adding challenges does not
reassign existing ones, and raising the threshold only adds members (never removes).
"""

from __future__ import annotations

from benchmark.runner import calibration as cal


class TestDeterminism:
    def test_score_is_stable_and_in_unit_interval(self):
        s1 = cal.holdout_score("astropy__astropy-7166")
        s2 = cal.holdout_score("astropy__astropy-7166")
        assert s1 == s2
        assert 0.0 <= s1 < 1.0

    def test_salt_changes_the_assignment(self):
        # A new salt re-rolls the whole holdout (deliberate re-roll escape hatch).
        a = cal.holdout_score("x", salt="calib-v1")
        b = cal.holdout_score("x", salt="calib-v2")
        assert a != b


class TestStabilityUnderGrowth:
    def test_adding_challenges_does_not_reassign_existing(self):
        base = [f"repo__task-{i}" for i in range(200)]
        before = cal.calibration_holdout(base, fraction=0.18)
        grown = base + [f"repo__new-{i}" for i in range(200)]
        after = cal.calibration_holdout(grown, fraction=0.18)
        # Every id that was in the holdout is STILL in it after adding 200 more.
        assert set(before) <= set(after)
        # And no original id flipped OUT because of the new ids.
        originals_after = {i for i in after if i in set(base)}
        assert originals_after == set(before)


class TestMonotoneThreshold:
    def test_raising_fraction_only_adds_members(self):
        ids = [f"repo__task-{i}" for i in range(500)]
        small = set(cal.calibration_holdout(ids, fraction=0.15))
        large = set(cal.calibration_holdout(ids, fraction=0.20))
        assert small <= large  # 15% holdout is a subset of the 20% holdout
        assert len(large) >= len(small)

    def test_zero_fraction_selects_nothing_and_one_selects_all(self):
        ids = [f"t{i}" for i in range(50)]
        assert cal.calibration_holdout(ids, fraction=0.0) == []
        assert set(cal.calibration_holdout(ids, fraction=1.0)) == set(ids)

    def test_invalid_fraction_raises(self):
        import pytest

        with pytest.raises(ValueError):
            cal.in_calibration_holdout("x", fraction=1.5)


class TestApproximateFraction:
    def test_selects_roughly_the_target_fraction(self):
        ids = [f"repo__task-{i}" for i in range(5000)]
        picked = cal.calibration_holdout(ids, fraction=0.18)
        # Binomial(5000, 0.18): expect ~900, allow a generous band for hash noise.
        assert 800 <= len(picked) <= 1000

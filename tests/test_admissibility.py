"""Unit tests for the numeric instrument-validity adjudicator (benchmark/admissibility.py)."""

# The load-bearing case is the last one: a gate that passes everything is worse than no gate,
# because it reads as coverage. Both rejection directions are asserted, and the local copy is
# pinned bit-for-bit against the canonical adjudicator it mirrors.

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from benchmark.admissibility import admissibility_verdict, run_gate

# The canonical adjudicator lives OUTSIDE this repo, so a public clone does not have it. The
# parity test reads its path from the SHUNT_SHARED_ADJUDICATOR env var (set by the build
# environment that owns the shared module) and skips when the variable is unset — naming the
# shared module's internal path here would leak a tree this repo does not ship.
_SHARED_GATE = Path(os.environ.get("SHUNT_SHARED_ADJUDICATOR", ""))


class TestTheThreeCanonicalShapes:
    def test_recovers_a_planted_signal_and_collapses_on_shuffled_labels(self) -> None:
        result = admissibility_verdict(0.92, 0.04, chance_level=0.0, chance_band=0.15)
        assert result.admissible
        assert result.positive_passed and result.null_at_chance

    def test_an_instrument_that_cannot_learn_is_rejected(self) -> None:
        result = admissibility_verdict(0.05, 0.03, chance_level=0.0, chance_band=0.15)
        assert not result.admissible
        assert not result.positive_passed
        assert "coverage-gap, not a falsification" in result.reason

    def test_an_instrument_that_scores_high_on_destroyed_signal_is_rejected(self) -> None:
        # The killer shape: the positive control looks great and means nothing, because the same
        # pipeline scores nearly as well with the signal removed.
        result = admissibility_verdict(0.88, 0.81, chance_level=0.0, chance_band=0.15)
        assert not result.admissible
        assert result.positive_passed and not result.null_at_chance
        assert "leakage" in result.reason

    def test_both_failures_at_once_report_a_broken_instrument(self) -> None:
        result = admissibility_verdict(0.05, 0.40, chance_level=0.0, chance_band=0.15)
        assert not result.admissible
        assert "broken" in result.reason


class TestBoundaries:
    def test_the_positive_leg_is_strict_at_the_band_edge(self) -> None:
        # Exactly AT chance+band is not clearing it. A >= here would admit an instrument whose
        # positive control is indistinguishable from its own null.
        assert not admissibility_verdict(
            0.15, 0.0, chance_level=0.0, chance_band=0.15
        ).positive_passed
        assert admissibility_verdict(
            0.1501, 0.0, chance_level=0.0, chance_band=0.15
        ).positive_passed

    def test_the_null_leg_is_two_sided(self) -> None:
        # A destroyed-signal score far BELOW chance is as much a defect as one above it: it means
        # the pipeline is anti-correlated with its own labels, which no honest null does.
        assert not admissibility_verdict(
            0.9, -0.6, chance_level=0.0, chance_band=0.15
        ).null_at_chance

    def test_a_nonzero_chance_level_shifts_both_legs(self) -> None:
        # Accuracy-shaped metrics sit at 0.5, not 0. A gate hardcoded to 0 would pass every
        # coin-flip classifier's positive control.
        assert admissibility_verdict(0.8, 0.5, chance_level=0.5, chance_band=0.1).admissible
        assert not admissibility_verdict(0.55, 0.5, chance_level=0.5, chance_band=0.1).admissible


class TestRunGate:
    def test_executes_both_controls_then_adjudicates(self) -> None:
        calls: list[str] = []

        def positive() -> float:
            calls.append("positive")
            return 0.9

        def shuffled() -> float:
            calls.append("shuffled")
            return 0.02

        result = run_gate(positive, shuffled, chance_level=0.0, chance_band=0.1)
        assert calls == ["positive", "shuffled"]
        assert result.admissible


class TestReporting:
    def test_headline_names_the_state_and_both_scores(self) -> None:
        result = admissibility_verdict(0.9, 0.02, chance_level=0.0, chance_band=0.1)
        assert result.headline.startswith("INSTRUMENT ADMISSIBLE")
        assert "+0.9000" in result.headline and "+0.0200" in result.headline

    def test_to_dict_carries_the_numbers_a_figure_footer_needs(self) -> None:
        result = admissibility_verdict(0.9, 0.02, chance_level=0.0, chance_band=0.1)
        payload = result.to_dict()
        assert payload["admissible"] is True
        assert payload["chance_upper"] == pytest.approx(0.1)


def _load_shared_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_shared_admissibility_gate", _SHARED_GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations through sys.modules[cls.__module__],
    # which raises on a module that was never registered (same recipe as conftest's loader).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not _SHARED_GATE.is_file(),
    reason="shared adjudicator not in this checkout (set SHUNT_SHARED_ADJUDICATOR to its path)",
)
class TestParityWithTheSharedGate:
    """Duplicating a gate is how the first copy goes stale. This pins the copy instead."""

    # Every case the local adjudicator can face: both legs pass, each fails alone, both fail, and
    # the two edges. If the shared adjudicator's semantics ever move, this fails on the next run
    # in the build environment that owns it rather than drifting silently for months.
    CASES = (
        (0.92, 0.04, 0.0, 0.15),
        (0.05, 0.03, 0.0, 0.15),
        (0.88, 0.81, 0.0, 0.15),
        (0.05, 0.40, 0.0, 0.15),
        (0.15, 0.00, 0.0, 0.15),
        (0.80, 0.50, 0.5, 0.10),
        (1.00, 0.52, 0.5, 0.10),
    )

    def test_verdicts_match_case_for_case(self) -> None:
        shared = _load_shared_gate()
        for positive, shuffled, level, band in self.CASES:
            mine = admissibility_verdict(positive, shuffled, chance_level=level, chance_band=band)
            theirs = shared.admissibility_verdict(
                positive, shuffled, chance_level=level, chance_band=band
            )
            assert (mine.admissible, mine.positive_passed, mine.null_at_chance) == (
                theirs.admissible,
                theirs.positive_passed,
                theirs.null_at_chance,
            ), f"divergence at {(positive, shuffled, level, band)}"
            assert mine.numbers == theirs.numbers

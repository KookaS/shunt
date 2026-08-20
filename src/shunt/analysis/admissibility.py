"""Numeric instrument-validity adjudicator: positive control ABOVE chance, destroyed-signal AT it.

Verdict about the INSTRUMENT, never about the hypothesis. Controls are supplied by the caller.
"""

# WHY THIS EXISTS. Any artifact that emits a verdict about whether a signal EXISTS — an eval
# harness, a detector score, a benchmark that can return "no effect" — has to answer a question
# that is prior to "is this computed correctly": is it computing anything about the thing it names?
# The two-sided answer is a POSITIVE CONTROL (plant a known-learnable signal; the ASSEMBLED
# pipeline must recover it) and a DESTROYED-SIGNAL NULL (the same pipeline, signal removed, must
# collapse to chance). A negative result on an instrument never shown to produce a positive is a
# coverage-gap, not a falsification — you cannot falsify a thesis on an instrument that has never
# been shown to detect anything.
#
# WHY IT SHIPS HERE. `src/shunt/` may not import `benchmark/` (SH006), and a shipped figure now
# has to state the instrument verdict beside every off-policy number it draws — so the adjudicator
# moved into the wheel and `benchmark/admissibility.py` became a re-export shim. Mirroring it
# instead would let a drifted copy silently change a verdict, which is the exact failure the next
# paragraph pins against.
#
# WHY IT IS A LOCAL COPY, AND WHAT KEEPS IT FROM GOING STALE. The canonical adjudicator lives in
# a shared adjudicator module that is not shipped in this repository. This file ships in the
# public repo, so it cannot import it — a clean clone would not have it, exactly as
# `benchmark/runner/replay_admissibility.py` documents for its own leg. Duplicating a gate is how
# the first copy goes stale, so the duplication is PINNED rather than trusted:
# `tests/test_admissibility.py::TestParityWithTheSharedGate` imports the shared module by the
# `SHUNT_SHARED_ADJUDICATOR` path when that variable is set and asserts this adjudicator returns
# bit-identical verdicts over a table of cases, and skips in a clone where the path does not
# exist. Semantics here are the shared module's, unchanged; only the module home differs.
#
# THE OTHER IN-REPO LEG IS DELIBERATELY NOT THIS ONE. `replay_admissibility` is discrete — its
# destroyed-signal control must land on the DEFINITE OPPOSITE verdict (FAILURE), not at chance —
# so it encodes the same discipline for its own modality. This module is for the continuous case:
# a score, a chance level, and an empirical band.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdmissibilityResult:
    """Outcome of the two-sided instrument-validity gate. ``admissible`` is the hard guardrail.

    ``admissible=False`` means the INSTRUMENT is not trusted, so no verdict read off it is
    quotable. It is never itself a verdict on the experiment's hypothesis.
    """

    admissible: bool
    reason: str
    positive_passed: bool
    null_at_chance: bool
    positive_score: float
    shuffled_score: float
    chance_level: float
    chance_band: float
    numbers: dict[str, Any] = field(default_factory=dict)

    @property
    def headline(self) -> str:
        """One line an emitted verdict can carry verbatim."""
        state = "ADMISSIBLE" if self.admissible else "INADMISSIBLE"
        return (
            f"INSTRUMENT {state}: positive control {self.positive_score:+.4f}, "
            f"destroyed-signal null {self.shuffled_score:+.4f}, chance "
            f"{self.chance_level:+.4f}±{self.chance_band:.4f}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "admissible": self.admissible,
            "reason": self.reason,
            "positive_passed": self.positive_passed,
            "null_at_chance": self.null_at_chance,
            **self.numbers,
        }


def admissibility_verdict(
    positive_score: float,
    shuffled_score: float,
    *,
    chance_level: float,
    chance_band: float,
) -> AdmissibilityResult:
    """Adjudicate two control scores against an empirical chance band (higher = more signal).

    Admissible iff the positive control clears ``chance_level + chance_band`` AND the
    destroyed-signal score sits within ``chance_band`` of ``chance_level``.
    """
    # `chance_band` is the HALF-WIDTH of the empirical chance interval, never 0: a finite-sample
    # null does not sit exactly at chance, and a zero band rejects every honest instrument.
    #
    # Both failure directions are caught, and they mean different things:
    #   positive at/below chance -> the assembled instrument cannot recover a signal that IS
    #                               there (frozen / mis-wired / degenerate front end).
    #   shuffled ABOVE chance    -> the instrument manufactures signal from noise (leakage,
    #                               fakery, capacity overfit).
    upper = chance_level + chance_band
    positive_passed = positive_score > upper
    null_at_chance = abs(shuffled_score - chance_level) <= chance_band

    if positive_passed and null_at_chance:
        reason = (
            f"ADMISSIBLE: positive control {positive_score:+.4f} clears chance band "
            f"(>{upper:+.4f}) AND destroyed-signal null {shuffled_score:+.4f} is at chance "
            f"({chance_level:+.4f}±{chance_band:.4f}) — the instrument recovers a real signal "
            f"and does not manufacture one from noise."
        )
    elif not positive_passed and not null_at_chance:
        reason = (
            f"INADMISSIBLE: positive control {positive_score:+.4f} did NOT clear chance "
            f"(>{upper:+.4f}) — the instrument cannot recover a planted signal — AND the "
            f"destroyed-signal null {shuffled_score:+.4f} is OFF chance "
            f"({chance_level:+.4f}±{chance_band:.4f}) — it also manufactures signal from noise. "
            f"The instrument is broken."
        )
    elif not positive_passed:
        reason = (
            f"INADMISSIBLE: positive control {positive_score:+.4f} did NOT clear the chance "
            f"band (>{upper:+.4f}) — the assembled instrument cannot recover a signal that IS "
            f"there. A negative result on it is a coverage-gap, not a falsification."
        )
    else:
        reason = (
            f"INADMISSIBLE: destroyed-signal null {shuffled_score:+.4f} is OFF chance "
            f"({chance_level:+.4f}±{chance_band:.4f}) — the instrument scores high even with the "
            f"signal destroyed (leakage / fakery / capacity overfit). The positive score "
            f"{positive_score:+.4f} is not trustworthy: it may be manufactured, not recovered."
        )

    return AdmissibilityResult(
        admissible=positive_passed and null_at_chance,
        reason=reason,
        positive_passed=positive_passed,
        null_at_chance=null_at_chance,
        positive_score=positive_score,
        shuffled_score=shuffled_score,
        chance_level=chance_level,
        chance_band=chance_band,
        numbers={
            "positive_score": positive_score,
            "shuffled_score": shuffled_score,
            "chance_level": chance_level,
            "chance_band": chance_band,
            "chance_upper": upper,
        },
    )


def run_gate(
    positive_control: Callable[[], float],
    shuffled_control: Callable[[], float],
    *,
    chance_level: float,
    chance_band: float,
) -> AdmissibilityResult:
    """Execute the two caller-supplied controls, then adjudicate.

    The two must share the instrument, its parameters and its seed, so the ONLY difference
    between them is signal-present vs signal-destroyed.
    """
    return admissibility_verdict(
        float(positive_control()),
        float(shuffled_control()),
        chance_level=chance_level,
        chance_band=chance_band,
    )


__all__ = ["AdmissibilityResult", "admissibility_verdict", "run_gate"]

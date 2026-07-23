"""RerunConfirmingVerifier (B): a failure is trusted only if it reproduces on rerun."""

from __future__ import annotations

from shunt.verifiers.base import Verifier, VerifierResult
from shunt.verifiers.rerun import RerunConfirmingVerifier


class _ScriptedVerifier(Verifier):
    """Returns a pre-scripted sequence of outcomes, one per verify() call."""

    def __init__(self, *outcomes: str) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def verify(self, text: str = "", work_dir: str | None = None) -> VerifierResult:
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        conf = 0.8 if outcome == "success" else 0.7
        return VerifierResult(
            outcome=outcome,
            confidence=conf,
            exit_code=0 if outcome == "success" else 2,
            failing_check_id="t::a" if outcome == "failure" else None,
        )


def test_reproduced_failure_is_confirmed() -> None:
    inner = _ScriptedVerifier("failure", "failure", "failure")
    result = RerunConfirmingVerifier(inner, reruns=2).verify(work_dir="/x")
    assert result.outcome == "failure"
    assert result.failing_check_id == "t::a"
    assert result.confirmed is True  # B10: the flag is stamped structurally on a reproduced red
    assert inner.calls == 3  # first run + 2 reruns, all failed → confirmed


def test_bare_verifier_failure_is_not_confirmed() -> None:
    # B10: a single-run red from a non-confirming verifier must NOT be trusted as confirmed —
    # the escalation log gates on this flag, so a bare AutoDetectVerifier can't trip it.
    assert VerifierResult(outcome="failure", confidence=0.7).confirmed is False


def test_zero_reruns_failure_is_not_confirmed() -> None:
    inner = _ScriptedVerifier("failure")
    result = RerunConfirmingVerifier(inner, reruns=0).verify(work_dir="/x")
    assert result.confirmed is False  # no rerun happened → not confirmed


def test_fail_then_pass_is_a_flake_abstained() -> None:
    inner = _ScriptedVerifier("failure", "success")
    result = RerunConfirmingVerifier(inner, reruns=2).verify(work_dir="/x")
    assert result.outcome == "unknown"  # abstain — never written as a negative label
    assert inner.calls == 2


def test_success_is_not_rerun() -> None:
    inner = _ScriptedVerifier("success")
    result = RerunConfirmingVerifier(inner, reruns=2).verify(work_dir="/x")
    assert result.outcome == "success"
    assert inner.calls == 1  # a pass is never rerun


def test_zero_reruns_passes_failure_through() -> None:
    inner = _ScriptedVerifier("failure")
    result = RerunConfirmingVerifier(inner, reruns=0).verify(work_dir="/x")
    assert result.outcome == "failure"
    assert inner.calls == 1

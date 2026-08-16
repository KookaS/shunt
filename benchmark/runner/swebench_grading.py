"""Grader-parity adjudication: classify a replay run the way SWE-bench's own grader does."""

# WHY THIS EXISTS. The replay runs SWE-bench's own test DIRECTIVES, which are whole test FILES, and
# it used to read the verdict off the whole-file EXIT CODE. The grader does not: it parses per-test
# statuses out of the log and checks only FAIL_TO_PASS ∪ PASS_TO_PASS. A test in the patched file
# that SWE-bench put in NEITHER list cannot change the grade, but it does set the exit code — so
# the two quantities disagree on real instances. Measured: matplotlib-20676's gold-patch run is
# `1 failed, 34 passed`, the failure being `test_widgets.py::test_rectangle_selector`, in neither
# list; per-test it is 2/2 F2P and 32/32 P2P, which is exactly RESOLVED. The exit code called it
# red, so the admissibility gate's positive control failed on an instance the grader resolves.
#
# ONE ADJUDICATOR FOR THE CONTROLS AND THE STEPS. This classifier is passed to `replay_step`, so
# the two gate legs and every replayed step share it. That is not tidiness, it is what makes the
# gate a control at all: adjudicating GOLD per-test while steps stayed on the exit code would
# certify an instrument nothing uses, and would admit matplotlib-20676 only to stamp all 320 of
# its steps a blocking red for ever (the unrelated test fails at every step) — turning a false
# rejection into a false STAMPING, which is worse, because a rejection at least clears.
#
# EVERYTHING IS DELEGATED, NOT RE-DERIVED. swebench's `MAP_REPO_TO_PARSER` extracts the statuses,
# `get_eval_tests_report` splits them into F2P/P2P success+failure, `get_resolution_status` decides
# FULL. The one function NOT reused is `get_logs_eval`: it takes a FILE PATH and slices between the
# harness's own `START/END_TEST_OUTPUT` markers, which a `docker exec` of the bare test command
# never emits. It already falls back to parsing the whole log when the markers yield nothing, which
# is precisely what this does with the combined output — so the wrapper is the marker handling
# only, and the parse itself is swebench's.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from shunt.verifiers.parse import parse_test_outcome

if TYPE_CHECKING:
    from shunt.verifiers.base import VerifierResult


def status_map(repo: str, combined: str) -> dict[str, str]:
    """Per-test statuses swebench's own log parser extracts from a run's combined output."""
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER  # noqa: PLC0415

    # The second argument is swebench's `TestSpec`. Every parser reachable from this corpus's
    # repos ignores it (pinned by `tests/test_swebench_grading.py`), and building a real TestSpec
    # would need the full HF row plus environment resolution to feed a value no parser reads.
    parsed = MAP_REPO_TO_PARSER[repo](combined, None)
    return {str(name): str(state) for name, state in parsed.items()}


# A status that decides a test one way or the other. SWE-bench's `test_passed` accepts PASSED and
# XFAIL; its `test_failed` accepts FAILED, ERROR, and ABSENT-from-the-map. XFAIL counting as a pass
# is swebench's own semantics and is kept: an agent that marks an F2P test xfail scores a green
# here exactly as it would on the leaderboard.
_DECISIVE: Final = frozenset({"PASSED", "XFAIL", "FAILED", "ERROR"})
_SKIPPED: Final = "SKIPPED"


def unobserved_signal(statuses: dict[str, str], fail_to_pass: tuple[str, ...]) -> tuple[str, ...]:
    """FAIL_TO_PASS tests whose status proves the run never measured the planted signal."""
    # SKIPPED and ABSENT are NOT the same thing here, and conflating them costs real data.
    #
    # SKIPPED is unambiguous: the runner saw the test and declined to run it. swebench drops it
    # from BOTH the numerator and the denominator, so an all-skipped run divides 0 by 0 and
    # `get_resolution_status` returns FULL — a confident green for a run that executed nothing.
    # Defensible when grading one submitted patch; fabrication as a per-step label. Any skipped
    # F2P test therefore demotes. Measured on 48 real gold logs: zero occurrences, so this costs
    # nothing today and only fires on the step-dependent case (an agent edit that starts skipping).
    #
    # ABSENT is ambiguous, because for some runners absence IS how a failure is reported.
    # django-11099's base leg is the measured example: its two F2P tests fail via `subTest`, so the
    # `... FAIL` suffix is never printed and the only record is `FAIL: test_ascii_validator
    # (auth_tests...) (invalid=…)`, which `parse_log_django` keys by the BARE name — the full
    # `test_x (module.Class)` id the spec uses is absent. swebench's own grader lands on the right
    # answer through "missing ⇒ failed". Demoting on absence alone would therefore have rejected
    # essentially every django instance, whose base leg is failing F2P tests by construction.
    # So absence only counts as non-measurement when NO F2P test was decided at all — the shape of
    # a run in which nothing relevant executed (e.g. `-rA`'s bare `SKIPPED [1] path:12:` line,
    # which parses to the single junk key `[1]` and leaves every F2P id missing at rc=0).
    if not fail_to_pass:
        return ()
    skipped = tuple(t for t in fail_to_pass if statuses.get(t) == _SKIPPED)
    if skipped:
        return skipped
    undecided = tuple(t for t in fail_to_pass if statuses.get(t, "") not in _DECISIVE)
    return undecided if len(undecided) == len(fail_to_pass) else ()


def _bad_run_marker(combined: str) -> str | None:
    """The swebench 'this run did not complete' marker present in *combined*, if any."""
    from swebench.harness.constants import (  # noqa: PLC0415
        APPLY_PATCH_FAIL,
        RESET_FAILED,
        TESTS_ERROR,
        TESTS_TIMEOUT,
    )

    # Taken from `get_logs_eval`'s bad-code list rather than restated: these are the exact strings
    # swebench refuses to grade on. `replay_step` already guards apply/reset structurally, but the
    # timeout/errored pair has no equivalent here.
    for marker in (TESTS_TIMEOUT, TESTS_ERROR, APPLY_PATCH_FAIL, RESET_FAILED):
        if marker in combined:
            return str(marker)
    return None


def timeout_marker() -> str:
    """SWE-bench's own 'this run timed out' string — what a killed exec must emit to be demoted."""
    # The replay's exec timeout writes this into the combined output so `_bad_run_marker` above
    # refuses to grade the truncated fragment. Reading it from swebench rather than restating it is
    # what keeps emitter and detector from drifting apart into a silently-graded timeout.
    from swebench.harness.constants import TESTS_TIMEOUT  # noqa: PLC0415

    return str(TESTS_TIMEOUT)


@dataclass(frozen=True)
class Grade:
    """What SWE-bench's grading functions say about one run, restricted to F2P ∪ P2P."""

    resolved: bool
    failures: tuple[str, ...]
    f2p_passed: int
    p2p_passed: int


def grade(
    repo: str,
    statuses: dict[str, str],
    fail_to_pass: tuple[str, ...],
    pass_to_pass: tuple[str, ...],
) -> Grade:
    """Run swebench's own report + resolution functions over a parsed status map."""
    from swebench.harness.constants import (  # noqa: PLC0415
        FAIL_ONLY_REPOS,
        FAIL_TO_PASS,
        PASS_TO_PASS,
        EvalType,
        ResolvedStatus,
    )
    from swebench.harness.grading import (  # noqa: PLC0415
        get_eval_tests_report,
        get_resolution_status,
    )

    gold_results = {FAIL_TO_PASS: list(fail_to_pass), PASS_TO_PASS: list(pass_to_pass)}
    eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    report = get_eval_tests_report(statuses, gold_results, eval_type=eval_type)
    return Grade(
        resolved=get_resolution_status(report) == ResolvedStatus.FULL.value,
        # F2P failures FIRST, and the order matters beyond readability: the admissibility gate
        # reads `failures[0] in fail_to_pass` to decide whether the destroyed-signal leg failed on
        # the bug itself rather than on some unrelated test (pinned by test_swebench_grading.py).
        failures=(*report[FAIL_TO_PASS]["failure"], *report[PASS_TO_PASS]["failure"]),
        f2p_passed=len(report[FAIL_TO_PASS]["success"]),
        p2p_passed=len(report[PASS_TO_PASS]["success"]),
    )


@dataclass(frozen=True)
class GraderParity:
    """Adjudicate one instance's replay runs on F2P ∪ P2P statuses, as the SWE-bench grader does."""

    repo: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]

    def _unobserved_detail(self, unobserved: tuple[str, ...], parsed: int) -> str:
        """Why this run never measured the planted signal — the two causes read differently."""
        # An empty FAIL_TO_PASS list is non-measurement for a different reason than an unobserved
        # one: nothing was declared, so there is no ratio to report. Formatting both through the
        # same f-string printed "0/0 FAIL_TO_PASS tests were never measured", a count that invites
        # the reader to hunt for the two tests it names in the same breath as saying there are none.
        if not self.fail_to_pass:
            return (
                f"grader parity: the spec declares no FAIL_TO_PASS tests, so there is no planted "
                f"signal to recover ({parsed} statuses parsed)"
            )
        return (
            f"grader parity: {len(unobserved)}/{len(self.fail_to_pass)} FAIL_TO_PASS tests were "
            f"never measured (skipped, or no F2P test decided at all among {parsed} parsed "
            f"statuses) — {', '.join(unobserved[:3])}"
        )

    def __call__(self, combined: str, returncode: int) -> VerifierResult:
        """Classify a run: infra stays infra, otherwise the grader's resolved/not-resolved."""
        result = parse_test_outcome(combined, returncode)
        # The one thing `get_logs_eval` does besides the marker split that IS worth carrying over:
        # it refuses to grade a log carrying swebench's own "this run did not complete" markers.
        # A timeout-truncated log is a fragment, and its surviving prefix would otherwise be graded
        # as if the tests that were killed had failed.
        #
        # This runs BEFORE the infra short-circuit deliberately, and the cost of scanning an
        # already-infra log is four substring searches — immaterial next to the container run that
        # produced it. The marker is specific evidence from swebench's own harness and must win
        # over the generic parser's guess, so a killed run reports "did not complete" instead of
        # "environment/collection error (rc=137)". Reordering these two would trade a real
        # diagnostic for no measurable saving.
        bad = _bad_run_marker(combined)
        if bad:
            return replace(
                result,
                outcome="unknown",
                confidence=0.0,
                detail=f"grader parity: the run did not complete ({bad!r})",
                is_infra_failure=True,
                failing_check_id=None,
            )
        if result.is_infra_failure:
            # The shared parser still owns the "no verdict exists" cases — usage/collection/signal
            # exits and a clean run that selected nothing. Adjudicating those per-test would read
            # an empty status map as "every F2P missing" and fabricate a red out of a non-run.
            return result
        statuses = status_map(self.repo, combined)
        # THE PLANTED SIGNAL MUST ACTUALLY HAVE BEEN OBSERVED. Everything the grade turns on lives
        # in FAIL_TO_PASS, and swebench scores an unobserved test as failed (missing) or drops it
        # entirely (skipped) — so without this a run that executed nothing relevant lands on a
        # CONFIDENT verdict in EITHER direction. Demoting to infra is the real-only answer:
        # absence of measurement, not an outcome. `unobserved_signal` above documents exactly why
        # skipped and absent are treated differently. An empty FAIL_TO_PASS list means the spec
        # declares no signal at all, which is the same non-measurement.
        unobserved = unobserved_signal(statuses, self.fail_to_pass)
        if unobserved or not self.fail_to_pass:
            return replace(
                result,
                outcome="unknown",
                confidence=0.0,
                detail=self._unobserved_detail(unobserved, len(statuses)),
                is_infra_failure=True,
                failing_check_id=None,
            )
        graded = grade(self.repo, statuses, self.fail_to_pass, self.pass_to_pass)
        # THE MIRROR HOLE, ON THE P2P SIDE. A whole PASS_TO_PASS module skipped at import prints
        # `SKIPPED [2] b.py:1: could not import 'optional_dep'`, which parses to the junk key `[2]`
        # — so the P2P ids are ABSENT, take "missing ⇒ failed", and manufacture a confident red
        # with a stable dedup key out of a run that never executed them. It cannot be closed by
        # treating absent P2P like absent F2P: that collides with django-11099, whose base leg
        # legitimately reports failures under bare names. The precise shape is narrower — every
        # F2P green, and NOT ONE P2P test decided either way — so that is what demotes.
        #
        # `f2p_passed == len(fail_to_pass)` IS the "every F2P green" half, and testing
        # `graded.failures` instead does not encode it: that tuple leads with the F2P failures, so
        # a decisively FAILED F2P test satisfied it too. That mis-fires exactly where it costs the
        # most — the BASE leg is the F2P-red state by construction, so a base run whose P2P module
        # skipped at import was demoted to `unknown`, the instance was ruled inadmissible, and
        # `clear_rejected` wiped its real stamps. A measured red on the planted signal is the one
        # thing this guard must never discard.
        if (
            self.pass_to_pass
            and graded.f2p_passed == len(self.fail_to_pass)
            and graded.failures
            and not any(statuses.get(t, "") in _DECISIVE for t in self.pass_to_pass)
        ):
            return replace(
                result,
                outcome="unknown",
                confidence=0.0,
                detail=(
                    f"grader parity: F2P {graded.f2p_passed}/{len(self.fail_to_pass)} all passed, "
                    f"but not one of the {len(self.pass_to_pass)} PASS_TO_PASS tests was decided "
                    f"among {len(statuses)} parsed statuses — the maintenance set never ran"
                ),
                is_infra_failure=True,
                failing_check_id=None,
            )
        tally = (
            f"F2P {graded.f2p_passed}/{len(self.fail_to_pass)}, "
            f"P2P {graded.p2p_passed}/{len(self.pass_to_pass)} "
            f"(rc={returncode}, {len(statuses)} statuses parsed)"
        )
        if graded.resolved:
            return replace(
                result,
                outcome="success",
                confidence=0.8,
                detail=f"grader: RESOLVED — {tally}",
                failing_check_id=None,
            )
        return replace(
            result,
            outcome="failure",
            confidence=0.7,
            detail=f"grader: NOT resolved — {tally}",
            # The dedup key must name a failure that actually COUNTS. The exit-code path took the
            # first node id in the output, which on matplotlib-20676 is the excluded
            # `test_rectangle_selector` — a constant, grade-irrelevant key at every step.
            failing_check_id=graded.failures[0] if graded.failures else result.failing_check_id,
        )


__all__ = [
    "Grade",
    "GraderParity",
    "grade",
    "status_map",
    "timeout_marker",
    "unobserved_signal",
]

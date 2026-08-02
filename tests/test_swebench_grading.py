"""Grader-parity adjudication: the replay's verdict is the per-test grade over F2P ∪ P2P, not the
whole-file exit code. Real swebench parsers + real grading functions throughout (no Docker)."""

from __future__ import annotations

import inspect

import pytest

from benchmark.runner.swebench_grading import (
    GraderParity,
    grade,
    status_map,
    unobserved_signal,
)

pytest.importorskip("swebench")

# The measured shape this whole change exists for. matplotlib-20676's GOLD run really is
# `1 failed, 34 passed` with the failure `test_widgets.py::test_rectangle_selector`, a test
# SWE-bench excluded from BOTH lists — so the grader RESOLVES the instance while the exit code
# (1) reads red. Trimmed to the lines the parser keys on, verbatim in form.
_MPL_F2P = (
    "lib/matplotlib/tests/test_widgets.py::test_span_selector_bound[horizontal]",
    "lib/matplotlib/tests/test_widgets.py::test_span_selector_bound[vertical]",
)
_MPL_P2P = ("lib/matplotlib/tests/test_widgets.py::test_rectangle_minspan[0-10-0-10]",)
_MPL_EXCLUDED = "lib/matplotlib/tests/test_widgets.py::test_rectangle_selector"

_MPL_GOLD_LOG = f"""
PASSED {_MPL_F2P[0]}
PASSED {_MPL_F2P[1]}
PASSED {_MPL_P2P[0]}
FAILED {_MPL_EXCLUDED} - AssertionError
1 failed, 34 passed in 12.34s
"""

# The excluded test's FAILED line comes FIRST on purpose: `parse_test_outcome` keys the dedup id
# off the first node id it sees, so this ordering is what separates "the first failure in the
# output" from "the first failure that counts toward the grade".
_MPL_BASE_LOG = f"""
FAILED {_MPL_EXCLUDED} - AssertionError
FAILED {_MPL_F2P[0]} - AssertionError
FAILED {_MPL_F2P[1]} - AssertionError
PASSED {_MPL_P2P[0]}
3 failed, 32 passed in 12.34s
"""


def _mpl() -> GraderParity:
    return GraderParity(repo="matplotlib/matplotlib", fail_to_pass=_MPL_F2P, pass_to_pass=_MPL_P2P)


def test_a_failing_test_outside_f2p_and_p2p_does_not_make_the_run_a_failure() -> None:
    # REVERT PROBE: adjudicate on the exit code instead and this is `failure` — the exact
    # over-rejection that destroyed matplotlib-20676 (7 trajectories / 320 steps).
    result = _mpl()(_MPL_GOLD_LOG, 1)
    assert result.outcome == "success"
    assert result.exit_code == 1  # the real rc is still recorded; it is just not the verdict
    assert result.failing_check_id is None


def test_a_failing_f2p_test_is_a_failure() -> None:
    result = _mpl()(_MPL_BASE_LOG, 1)
    assert result.outcome == "failure"
    # The dedup key names a test that actually COUNTS toward the grade. Reading the first node id
    # in the output instead would pick the EXCLUDED test — which fails at every step of
    # matplotlib-20676, so the recurrence trigger would see one constant, grade-irrelevant key.
    assert result.failing_check_id == _MPL_F2P[0]
    assert result.failing_check_id != _MPL_EXCLUDED


def test_a_failing_p2p_test_is_a_failure_even_when_every_f2p_passes() -> None:
    log = f"PASSED {_MPL_F2P[0]}\nPASSED {_MPL_F2P[1]}\nFAILED {_MPL_P2P[0]} - boom\n1 failed\n"
    result = _mpl()(log, 1)
    assert result.outcome == "failure"
    assert result.failing_check_id == _MPL_P2P[0]


def test_the_base_leg_shape_that_only_per_test_adjudication_can_reject() -> None:
    # The BASE-leg hole: the F2P tests PASS without the fix, and only the EXCLUDED test fails.
    # On the exit code that reads FAILURE (rc=1) and the instance is admitted, even though the
    # replay cannot tell fixed from unfixed. Per-test it is RESOLVED, so the base leg reports
    # success and the gate rejects it.
    passing = "\n".join(f"PASSED {t}" for t in (*_MPL_F2P, *_MPL_P2P))
    assert _mpl()(f"{passing}\nFAILED {_MPL_EXCLUDED}\n1 failed\n", 1).outcome == "success"


def test_an_f2p_test_absent_while_another_was_decided_is_a_failure() -> None:
    # Absence alone is NOT treated as non-measurement, because for some runners absence IS how a
    # failure is reported (see the django case below). As long as at least one F2P test was
    # decided, the run executed and swebench's "missing ⇒ failed" is the right reading.
    log = f"PASSED {_MPL_F2P[0]}\nPASSED {_MPL_P2P[0]}\n1 passed\n"
    result = _mpl()(log, 0)
    assert result.outcome == "failure"
    assert result.failing_check_id == _MPL_F2P[1]


def test_djangos_bare_name_failure_line_is_still_a_failure_not_a_non_measurement() -> None:
    # MEASURED on django-11099's base leg. A `subTest` failure never prints the `... FAIL` suffix,
    # so the only record is the `FAIL:` line, which `parse_log_django` keys by the BARE test name
    # while the spec uses `test_x (module.Class)`. Reading that absence as "not measured" would
    # have rejected essentially every django instance, whose base leg fails F2P by construction.
    f2p = (
        "test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)",
        "test_unicode_validator (auth_tests.test_validators.UsernameValidatorsTests)",
        "test_help_text (auth_tests.test_validators.UsernameValidatorsTests)",
    )
    log = (
        "test_help_text (auth_tests.test_validators.UsernameValidatorsTests) ... ok\n"
        "FAIL: test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests) "
        "(invalid='trailingnewline\\n')\n"
        "FAIL: test_unicode_validator (auth_tests.test_validators.UsernameValidatorsTests) "
        "(invalid='trailingnewline\\n')\n"
        "Ran 22 tests in 0.053s\n\nFAILED (failures=2)\n"
    )
    result = GraderParity(repo="django/django", fail_to_pass=f2p, pass_to_pass=())(log, 1)
    assert result.outcome == "failure"
    assert result.is_infra_failure is False
    assert result.failing_check_id == f2p[0]


# ---------------------------------------------------------------------------
# Infra still wins: a run that produced no verdict must never be graded into one.
# ---------------------------------------------------------------------------


def test_a_usage_error_stays_infra_and_is_never_graded() -> None:
    result = _mpl()("ERROR: not found: /testbed/a.py::test_x", 4)
    assert result.is_infra_failure is True
    assert result.outcome == "unknown"


def test_a_clean_run_that_selected_nothing_stays_infra() -> None:
    result = _mpl()("collected 0 items\nno tests ran in 0.01s", 0)
    assert result.is_infra_failure is True


def test_an_unparseable_log_is_infra_not_a_fabricated_red() -> None:
    # Every F2P would read as "missing" — the grader scores missing as failed, which would turn a
    # parser mismatch into a confident capability red on data nothing measured.
    result = _mpl()("some output with 3 failed but no per-test status lines", 1)
    assert result.is_infra_failure is True
    assert result.outcome == "unknown"
    assert result.failing_check_id is None


# ---------------------------------------------------------------------------
# Delegation pins: the statuses and the resolution come from swebench, not from us.
# ---------------------------------------------------------------------------

_CORPUS_REPOS = (
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
)


@pytest.mark.parametrize("repo", _CORPUS_REPOS)
def test_every_corpus_repo_parser_ignores_the_test_spec_argument(repo: str) -> None:
    # `status_map` passes None for swebench's TestSpec because no parser this corpus reaches reads
    # it. If swebench ever changes one, this fails LOUDLY instead of silently feeding it None.
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    body = inspect.getsource(MAP_REPO_TO_PARSER[repo]).split("\n", 1)[1]
    assert "test_spec" not in body


def test_status_map_uses_the_repos_own_parser_dialect() -> None:
    # django's runner does not print pytest node ids; the django parser reads `... ok` lines.
    statuses = status_map("django/django", "test_x (auth_tests.test_basic.TestGetUser) ... ok\n")
    assert statuses == {"test_x (auth_tests.test_basic.TestGetUser)": "PASSED"}


def test_grade_counts_only_f2p_and_p2p() -> None:
    statuses = {
        _MPL_F2P[0]: "PASSED",
        _MPL_F2P[1]: "PASSED",
        _MPL_P2P[0]: "PASSED",
        _MPL_EXCLUDED: "FAILED",
    }
    graded = grade("matplotlib/matplotlib", statuses, _MPL_F2P, _MPL_P2P)
    assert graded.resolved is True
    assert graded.failures == ()
    assert (graded.f2p_passed, graded.p2p_passed) == (2, 1)


# ---------------------------------------------------------------------------
# The planted signal must actually have been OBSERVED. swebench scores an unseen F2P test as
# failed (missing) or drops it entirely (skipped), so without this guard a run that executed
# nothing relevant lands on a confident verdict in EITHER direction.
# ---------------------------------------------------------------------------


def test_a_run_in_which_every_f2p_test_was_skipped_is_not_a_pass() -> None:
    # `get_resolution_status` returns FULL on 0/0 because skipped tests leave both the numerator
    # and the denominator empty. That is a fabricated green — zero graded tests executed.
    log = (
        f"SKIPPED {_MPL_F2P[0]}\nSKIPPED {_MPL_F2P[1]}\nPASSED {_MPL_P2P[0]}\n1 passed, 2 skipped\n"
    )
    result = _mpl()(log, 0)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True
    assert result.failing_check_id is None


def test_one_skipped_f2p_test_is_not_a_pass_on_the_remaining_subset() -> None:
    log = (
        f"PASSED {_MPL_F2P[0]}\nSKIPPED {_MPL_F2P[1]}\nPASSED {_MPL_P2P[0]}\n2 passed, 1 skipped\n"
    )
    assert _mpl()(log, 0).is_infra_failure is True


def test_a_junk_status_key_does_not_turn_an_all_skipped_run_into_a_red() -> None:
    # pytest `-rA` prints bare `SKIPPED [1] path:12: reason` lines, which the parser keys as
    # "[1]" — a non-empty status map in which no F2P id appears at all. Scoring those as
    # "missing therefore failed" emits a blocking red, with a real-looking node id as its dedup
    # key, for a run in which nothing executed.
    # P2P is empty here so the FAIL_TO_PASS guard is the only thing standing between this and a
    # fabricated red; the P2P-side shape has its own case below.
    parity = GraderParity(repo="matplotlib/matplotlib", fail_to_pass=_MPL_F2P, pass_to_pass=())
    result = parity(
        "SKIPPED [1] lib/matplotlib/tests/test_widgets.py:12: no display\n36 skipped\n", 0
    )
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True
    assert result.failing_check_id is None


def test_a_spec_with_no_f2p_tests_declares_no_signal_to_recover() -> None:
    empty = GraderParity(repo="matplotlib/matplotlib", fail_to_pass=(), pass_to_pass=_MPL_P2P)
    assert empty(f"PASSED {_MPL_P2P[0]}\n1 passed\n", 0).is_infra_failure is True


def test_f2p_failures_are_ordered_before_p2p_failures() -> None:
    # The admissibility gate reads `failures[0] in fail_to_pass` to decide whether the base leg
    # failed on the bug itself, so this ordering is load-bearing, not cosmetic.
    log = f"FAILED {_MPL_P2P[0]}\nFAILED {_MPL_F2P[0]}\nPASSED {_MPL_F2P[1]}\n2 failed\n"
    graded = grade(
        "matplotlib/matplotlib", status_map("matplotlib/matplotlib", log), _MPL_F2P, _MPL_P2P
    )
    assert graded.failures[0] == _MPL_F2P[0]


def test_an_entirely_unrun_p2p_module_is_not_a_manufactured_red() -> None:
    # The mirror of the F2P hole. A PASS_TO_PASS module skipped at import prints
    # `SKIPPED [2] b.py:1: could not import 'optional_dep'`, which parses to the junk key "[2]" —
    # so the P2P ids are ABSENT, take "missing implies failed", and manufacture a confident red
    # with a stable dedup key for tests that never ran. Every F2P is green here.
    p2p = ("b.py::test_other_1", "b.py::test_other_2")
    parity = GraderParity(repo="matplotlib/matplotlib", fail_to_pass=_MPL_F2P, pass_to_pass=p2p)
    log = (
        f"PASSED {_MPL_F2P[0]}\nPASSED {_MPL_F2P[1]}\n"
        "SKIPPED [2] b.py:1: could not import 'optional_dep'\n2 passed, 2 skipped\n"
    )
    result = parity(log, 0)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True
    assert result.failing_check_id is None


def test_a_measured_f2p_red_survives_the_unrun_p2p_guard() -> None:
    # REVERT PROBE: drop the `f2p_passed == len(fail_to_pass)` condition (test `graded.failures`
    # alone, as the guard first did) and this comes back `unknown` / infra=True — because
    # `graded.failures` LEADS with the F2P failures, so a decisively FAILED F2P test satisfied it.
    # The BASE leg is the F2P-red state by construction, so this shape lands there, the instance is
    # ruled inadmissible, and `clear_rejected` wipes real measurements.
    p2p = ("b.py::test_other_1", "b.py::test_other_2")
    parity = GraderParity(repo="matplotlib/matplotlib", fail_to_pass=_MPL_F2P, pass_to_pass=p2p)
    log = (
        f"FAILED {_MPL_F2P[0]} - AssertionError\nFAILED {_MPL_F2P[1]} - AssertionError\n"
        "SKIPPED [2] b.py:1: could not import 'optional_dep'\n2 failed, 2 skipped\n"
    )
    result = parity(log, 1)
    assert result.outcome == "failure"
    assert result.is_infra_failure is False
    assert result.failing_check_id == _MPL_F2P[0]


def test_the_unrun_p2p_detail_cannot_claim_a_pass_the_data_contradicts() -> None:
    # The detail is the only record a human reads back, so it must cite the counts it rests on
    # rather than assert "every FAIL_TO_PASS test passed" on faith.
    p2p = ("b.py::test_other_1",)
    parity = GraderParity(repo="matplotlib/matplotlib", fail_to_pass=_MPL_F2P, pass_to_pass=p2p)
    log = (
        f"PASSED {_MPL_F2P[0]}\nPASSED {_MPL_F2P[1]}\n"
        "SKIPPED [1] b.py:1: could not import 'optional_dep'\n2 passed, 1 skipped\n"
    )
    result = parity(log, 0)
    assert result.is_infra_failure is True
    assert f"F2P {len(_MPL_F2P)}/{len(_MPL_F2P)} all passed" in result.detail


def test_the_no_signal_detail_never_prints_a_zero_of_zero_count() -> None:
    # An empty FAIL_TO_PASS list has no ratio to report; the shared f-string rendered
    # "0/0 FAIL_TO_PASS tests were never measured", a count that contradicts its own sentence.
    empty = GraderParity(repo="matplotlib/matplotlib", fail_to_pass=(), pass_to_pass=_MPL_P2P)
    detail = empty(f"PASSED {_MPL_P2P[0]}\n1 passed\n", 0).detail
    assert "0/0" not in detail
    assert "declares no FAIL_TO_PASS tests" in detail


def test_the_timeout_marker_is_exactly_what_the_bad_run_scan_refuses_to_grade() -> None:
    # The replay's exec timeout writes this string into the combined output to force a demotion.
    # If emitter and detector ever drift, a killed run is graded as if its dead tests had failed.
    from benchmark.runner.swebench_grading import timeout_marker

    result = _mpl()(f"PASSED {_MPL_F2P[0]}\n{timeout_marker()}\n", 124)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True


def test_one_decided_p2p_test_is_enough_to_keep_grading() -> None:
    # The narrow shape only: a single P2P test absent among many decided ones is still a failure,
    # which is what keeps astropy-7606's unmatched spec id a genuine rejection rather than infra.
    p2p = ("b.py::test_other_1", "b.py::test_other_2")
    parity = GraderParity(repo="matplotlib/matplotlib", fail_to_pass=_MPL_F2P, pass_to_pass=p2p)
    log = f"PASSED {_MPL_F2P[0]}\nPASSED {_MPL_F2P[1]}\nPASSED {p2p[0]}\n3 passed\n"
    assert parity(log, 0).outcome == "failure"


def test_a_log_carrying_swebenchs_timeout_marker_is_never_graded() -> None:
    # `get_logs_eval` refuses to grade a log containing this marker; the truncated prefix would
    # otherwise be scored as if the tests that were killed had failed.
    log = f"PASSED {_MPL_F2P[0]}\n>>>>> Tests Timed Out\n"
    result = _mpl()(log, 1)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True
    assert "did not complete" in result.detail


# ---------------------------------------------------------------------------
# The two rejections recorded as KNOWN_ARTIFACTS. These pin the parse MECHANISM each record
# claims, so the prose in `replay_admissibility.KNOWN_ARTIFACTS` cannot quietly become a lie: if
# swebench's parser or the collection-marker guard is ever changed, the record is re-opened by a
# failing test rather than by someone noticing.
# ---------------------------------------------------------------------------

_DJ_F2P = (
    "test_get_language_from_path_real (i18n.tests.MiscTests)",
    "test_get_supported_language_variant_null (i18n.tests.MiscTests)",
)
_DJ_P2P = (
    "test_get_language_from_path_null (i18n.tests.MiscTests)",
    "test_get_supported_language_variant_real (i18n.tests.MiscTests)",
)
# Trimmed verbatim from django-15098's real BASE-leg container log. Line 2 is the artifact and is
# exactly as emitted: the first test failed only via `subTest`, so nothing terminated its `... `
# and the next test's header was appended to the same physical line.
_DJ_BASE_LOG = (
    f"{_DJ_P2P[0]} ... ok\n"
    f"{_DJ_F2P[0]} ... {_DJ_F2P[1]} ... ok\n"
    f"{_DJ_P2P[1]} ... ok\n"
    "\n"
    "======================================================================\n"
    f"FAIL: {_DJ_F2P[0]} (path='/en-latn-us/')\n"
    "----------------------------------------------------------------------\n"
    "AssertionError: None != 'en-latn-us'\n"
    "\n"
    "======================================================================\n"
    f"FAIL: {_DJ_F2P[0]} (path='/de-ch-1901/')\n"
    "----------------------------------------------------------------------\n"
    "AssertionError: None != 'de-ch-1901'\n"
    "\n"
    "Ran 92 tests in 0.204s\n\nFAILED (failures=2)\n"
)


def test_django_15098s_two_parse_artifacts_are_exactly_what_its_record_claims() -> None:
    statuses = status_map("django/django", _DJ_BASE_LOG)
    # Artifact 1: the subTest failure is keyed by the BARE name — no `(module.Class)` qualifier,
    # no subTest parameter — so the spec's id cannot match it.
    assert statuses["test_get_language_from_path_real"] == "FAILED"
    # Artifact 2: the SECOND F2P test's `... ok` was keyed together with the FIRST test's name,
    # as one concatenated key. Its own id therefore never appears.
    assert statuses[f"{_DJ_F2P[0]} ... {_DJ_F2P[1]}"] == "PASSED"
    assert _DJ_F2P[0] not in statuses
    assert _DJ_F2P[1] not in statuses
    # Both, and only both together, produce "no F2P test was decided at all".
    assert unobserved_signal(statuses, _DJ_F2P) == _DJ_F2P


def test_django_15098s_demotion_is_our_guard_being_stricter_than_swebenchs_grader() -> None:
    # SWE-bench's own grader reaches the CORRECT base verdict on this map through
    # missing-implies-failed. The `unknown` is `unobserved_signal` deliberately refusing to trust
    # an absent test — the conservatism the record documents, not a grading disagreement.
    statuses = status_map("django/django", _DJ_BASE_LOG)
    graded = grade("django/django", statuses, _DJ_F2P, _DJ_P2P)
    assert graded.resolved is False
    assert (graded.f2p_passed, graded.p2p_passed) == (0, len(_DJ_P2P))

    parity = GraderParity(repo="django/django", fail_to_pass=_DJ_F2P, pass_to_pass=_DJ_P2P)
    result = parity(_DJ_BASE_LOG, 1)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True


_RQ_F2P = (
    "tests/test_requests.py::TestRequests::test_invalid_url[InvalidURL-http://.example.com]",
)
_RQ_P2P = ("tests/test_requests.py::TestRequests::test_entry_points",)
# requests-5414's GOLD leg: every graded test passes, and 158 UNRELATED tests error on a conftest
# fixture importing an absent optional dep. Nothing failed, so no `FAILED <nodeid>` line exists.
_RQ_GOLD_LOG = (
    f"PASSED {_RQ_F2P[0]}\n"
    f"PASSED {_RQ_P2P[0]}\n"
    "ERROR tests/test_requests.py::TestRequests::test_https_warnings\n"
    "E       ModuleNotFoundError: No module named 'trustme'\n"
    "tests/conftest.py:42: ModuleNotFoundError\n"
    "================= 131 passed, 1 xfailed, 158 errors in 10.83s ==================\n"
)


def test_requests_5414s_gold_leg_short_circuits_before_grading_a_resolved_log() -> None:
    # Per-test this log IS resolved — which is why the rejection is an instrument artifact.
    statuses = status_map("psf/requests", _RQ_GOLD_LOG)
    assert grade("psf/requests", statuses, _RQ_F2P, _RQ_P2P).resolved is True
    # But the collection marker fires first (nothing failed ⇒ no proof tests ran and failed), so
    # the run never reaches the grader. This guard is what stands between a broken container and
    # a fabricated red; the record documents the cost of keeping it.
    parity = GraderParity(repo="psf/requests", fail_to_pass=_RQ_F2P, pass_to_pass=_RQ_P2P)
    result = parity(_RQ_GOLD_LOG, 1)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True


def test_requests_5414s_base_leg_is_unaffected_because_something_actually_failed() -> None:
    # The blind spot is specifically the would-be-RESOLVED run: add one real failure and the
    # short-circuit stops firing, which is why only the gold leg is lost.
    base_log = _RQ_GOLD_LOG.replace(f"PASSED {_RQ_F2P[0]}", f"FAILED {_RQ_F2P[0]}").replace(
        "131 passed, 1 xfailed", "1 failed, 130 passed, 1 xfailed"
    )
    parity = GraderParity(repo="psf/requests", fail_to_pass=_RQ_F2P, pass_to_pass=_RQ_P2P)
    result = parity(base_log, 1)
    assert result.outcome == "failure"
    assert result.is_infra_failure is False

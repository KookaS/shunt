"""The dedup key must survive a re-run of an IDENTICAL state — it is the escalation trigger."""

# Every fixture below is a verbatim excerpt of real container output captured by replaying one
# SWE-bench state repeatedly (`benchmark/runner/offline_replay.replay_step`); only the run-varying
# fragments differ between the two variants of each pair.
#

# WHY THIS FILE EXISTS. `_failing_check_id` hashes the whole output when no test node id is
# present, so any per-run byte in it makes the key random. A random key does not merely add
# noise: the escalation policy triggers on dedup-key RECURRENCE, so a key that never repeats
# silently disables escalation for that slice while still looking like a populated field.
# Replaying one identical sympy state four times produced four different keys before this was
# normalized (16% of the corpus), and one identical sphinx state three times produced three.

from __future__ import annotations

from shunt.verifiers.parse import parse_test_outcome


def _key(combined: str) -> str | None:
    return parse_test_outcome(combined, 1).failing_check_id


# sympy's `bin/test -C --verbose` header. Both the seed and PYTHONHASHSEED are redrawn per run.
_SYMPY = """============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
random seed:        {seed}
hash randomization: on (PYTHONHASHSEED={hashseed})

sympy/functions/elementary/tests/test_hyperbolic.py[45]
test_coth E
NameError: name 'cotm' is not defined
=========== tests finished: 44 passed, 1 exceptions, in {secs} seconds ===========
"""


def test_sympy_run_seeds_do_not_change_the_key() -> None:
    first = _SYMPY.format(seed="63896174", hashseed="3309299595", secs="4.72")
    second = _SYMPY.format(seed="62785731", hashseed="2286833452", secs="4.58")
    assert first != second
    assert _key(first) == _key(second)
    assert _key(first) is not None


def test_the_sympy_key_still_separates_two_genuinely_different_failures() -> None:
    # Normalization must not flatten everything to one key — then recurrence would fire always.
    same_shape = _SYMPY.format(seed="1", hashseed="2", secs="1.0")
    other_failure = same_shape.replace("name 'cotm' is not defined", "name 'zoo' is not defined")
    assert _key(same_shape) != _key(other_failure)


# sphinx's `tox --current-env -epy39 -v --` output. The `--durations` block is sorted BY time, so
# run-to-run jitter permutes the LINES — normalizing the numbers alone leaves the key random.
_SPHINX = """py39: commands[0]> python -X dev -m pytest --durations 25 tests/test_directive_code.py
base tempdir: /tmp/pytest-of-root/pytest-{tmp}
=================================== FAILURES ===================================
___________ test_LiteralIncludeReader_dedent_and_append_and_prepend ____________
tests/test_directive_code.py:260: AssertionError
============================= slowest 25 durations =============================
{durations}
========================= 1 failed, 40 passed in {total}s =========================
py39: exit 1 ({elapsed} seconds) /testbed> python -X dev -m pytest pid={pid}
  py39: FAIL code 1 ({total_tox}=setup[0.01]+cmd[{elapsed}] seconds)
"""

_DURATIONS_A = """0.62s setup    tests/test_directive_code.py::test_LiteralIncludeReader_lines2
0.34s call     tests/test_directive_code.py::test_force_option
0.13s setup    tests/test_directive_code.py::test_code_block_caption_latex"""

# Same three entries, different times — so a different ORDER. This is what defeats a
# numbers-only normalizer.
_DURATIONS_B = """0.15s call     tests/test_directive_code.py::test_force_option
0.14s setup    tests/test_directive_code.py::test_LiteralIncludeReader_lines2
0.12s setup    tests/test_directive_code.py::test_code_block_caption_latex"""


def test_sphinx_duration_block_reordering_and_pid_do_not_change_the_key() -> None:
    first = _SPHINX.format(
        tmp="0", durations=_DURATIONS_A, total="2.29", elapsed="3.54", pid="54", total_tox="3.55"
    )
    second = _SPHINX.format(
        tmp="1", durations=_DURATIONS_B, total="1.50", elapsed="1.85", pid="153", total_tox="1.86"
    )
    assert first != second
    assert _key(first) == _key(second)
    assert _key(first) is not None


def test_the_sphinx_key_still_separates_two_genuinely_different_failures() -> None:
    base = _SPHINX.format(
        tmp="0", durations=_DURATIONS_A, total="1.0", elapsed="1.0", pid="1", total_tox="1.0"
    )
    other = base.replace("test_LiteralIncludeReader_dedent", "test_LiteralIncludeReader_prepend")
    assert _key(base) != _key(other)


def test_a_node_id_bearing_failure_keys_off_the_node_id_not_the_hash() -> None:
    # The seven `pytest -rA` families never reach the hash fallback — proving that keeps the
    # normalizer honest about which slice of the corpus it is actually protecting.
    combined = "FAILED lib/matplotlib/tests/test_widgets.py::test_rectangle_selector\nassert 0\n"
    assert _key(combined) == "lib/matplotlib/tests/test_widgets.py::test_rectangle_selector"


# ---------------------------------------------------------------------------
# A captured INNER test session must not overrule the outer session's own verdict. Verbatim from
# `pytest-dev__pytest-10051`'s gold-patch leg, which reports `16 passed` at exit 0 while a
# `pytester` sub-session inside one of its tests prints `collected 0 items` / `no tests ran`.
# With the admissibility gate now CLEARING what it rejects, this misclassification would have
# wiped the whole pytest-dev family's per-step outcomes.
# ---------------------------------------------------------------------------

_NESTED = """============================= test session starts ==============================
collected 16 items

testing/logging/test_fixture.py ................                         [100%]

=================================== FAILURES ===================================
______________________________ test_fixture_help _______________________________
----------------------------- Captured stdout call -----------------------------
============================= test session starts ==============================
rootdir: /tmp/pytest-of-root/pytest-1/test_fixture_help0
collected 0 items

============================ no tests ran in 0.00s =============================
============================== 16 passed in 0.16s ==============================
"""


def test_a_captured_inner_session_does_not_demote_the_outer_pass() -> None:
    result = parse_test_outcome(_NESTED, 0)
    assert result.outcome == "success"
    assert result.is_infra_failure is False


def test_a_run_that_really_selected_nothing_is_still_not_a_pass() -> None:
    # The A1 defence must survive the veto: sympy's `bin/test` given a bare unittest name exits 0
    # printing `0 passed`, and stamping that green fabricated 4 064 committed steps.
    result = parse_test_outcome("tests finished: 0 passed, in 0.00 seconds", 0)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True
    assert parse_test_outcome("collected 0 items\nno tests ran in 0.01s", 0).outcome == "unknown"
    assert parse_test_outcome("Ran 0 tests in 0.001s\n\nOK\n", 0).outcome == "unknown"

"""A repo that is red, stays red, and fails the SAME way every time."""

# Determinism is the whole point. The escalation ladder counts *same-key* verified
# failures, so two reds with different failing-check ids never reach the threshold.
# The fake upstream answers "ok" and edits nothing, so nothing can turn this green
# mid-run — which is exactly what a hand-driven terminal recipe cannot promise.
#
# Named ``red_check.py``, not ``test_*.py``, so Shunt's OWN suite never collects it
# (``testpaths = ["tests", ...]`` walks this directory). The sibling pyproject's
# ``python_files`` opts it back in for the verifier, whose rootdir is /repo.


def test_demo_stays_red() -> None:
    # `raise`, not `assert False`: ruff B011 (and `python -O`) would strip a bare assert,
    # and a check that can be optimised away is not a deterministic red.
    raise AssertionError("deterministic red: the harness needs a stable failing check")

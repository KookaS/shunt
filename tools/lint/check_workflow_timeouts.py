#!/usr/bin/env python3
"""SH013: every workflow job declares a bounded, sane `timeout-minutes`."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from _shared import Finding

_CODE = "SH013"

# GitHub's default when a job omits `timeout-minutes` is 360 — six hours of runner time
# that nothing in the repo asked for. That default is not theoretical: on 2026-08-15 three
# `integration-handshake` legs looped against a stateless fake upstream and had to be
# cancelled BY HAND at 2h16m. `continue-on-error` did not help — it stops a failing leg
# from turning the run red, it does not stop a hang. A job that legitimately needs longer
# says so out loud here, and the number is the claim that gets reviewed.
_CEILING_MINUTES = 60

_WORKFLOW_DIR = Path(".github/workflows")

# `on:` parses as the YAML boolean True (the Norway problem), so a workflow's own keys are
# matched by identity, not by assuming the string survived. Only `jobs` is read here, but
# the note matters to anyone extending this gate.
_JOBS_KEY = "jobs"


def _jobs(document: object) -> dict[str, object]:
    """The workflow's job mapping, or empty when the file is not a workflow."""
    if not isinstance(document, dict):
        return {}
    jobs = document.get(_JOBS_KEY)
    if not isinstance(jobs, dict):
        return {}
    return {str(name): body for name, body in jobs.items()}


def _job_line(source: str, job_name: str) -> int:
    """The 1-based line the job is declared on, for a clickable finding."""
    needle = f"  {job_name}:"
    for number, line in enumerate(source.splitlines(), start=1):
        if line.rstrip() == needle:
            return number
    return 1


def _check_job(path: str, source: str, name: str, body: object) -> Finding | None:
    """One job's verdict: missing bound, unusable bound, or over the ceiling."""
    line = _job_line(source, name)
    # A reusable-workflow call (`uses:`) runs the CALLED workflow's jobs, which carry
    # their own bounds; `timeout-minutes` is not even a valid key there.
    if isinstance(body, dict) and "uses" in body:
        return None
    declared = body.get("timeout-minutes") if isinstance(body, dict) else None
    if declared is None:
        return Finding(
            path,
            line,
            2,
            f"job '{name}' has no timeout-minutes — it inherits GitHub's 360-minute "
            f"default and can burn six hours before anyone notices",
        )
    if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
        return Finding(
            path, line, 2, f"job '{name}' has a non-positive timeout-minutes: {declared!r}"
        )
    if declared > _CEILING_MINUTES:
        return Finding(
            path,
            line,
            2,
            f"job '{name}' declares timeout-minutes: {declared}, over the "
            f"{_CEILING_MINUTES}-minute ceiling — justify it and raise _CEILING_MINUTES, "
            f"or split the job",
        )
    return None


def check(path: str) -> list[Finding]:
    """Every job in one workflow file that is missing or misdeclaring its bound."""
    try:
        source = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        document = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        return [Finding(path, 1, 0, f"workflow does not parse as YAML: {exc}")]
    findings = [_check_job(path, source, name, body) for name, body in _jobs(document).items()]
    return [f for f in findings if f is not None]


def _default_paths() -> list[str]:
    """Every workflow — a whole-tree scan, since the failure is an ADDED job."""
    return sorted(str(p) for p in _WORKFLOW_DIR.glob("*.yml")) + sorted(
        str(p) for p in _WORKFLOW_DIR.glob("*.yaml")
    )


def main(argv: list[str]) -> int:
    """Report every finding and exit 1 when any job is unbounded."""
    paths = [a for a in argv if not a.startswith("-")] or _default_paths()
    findings = [f for path in paths for f in check(path)]
    for finding in findings:
        print(  # noqa: T201 - a lint gate reports on stderr
            f"{finding.path}:{finding.line}:{finding.col}: [{_CODE} ERROR] {finding.message}",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

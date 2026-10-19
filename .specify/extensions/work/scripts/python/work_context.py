#!/usr/bin/env python3
"""Context for the /work command, as one JSON object on stdout.

    work_context.py [--json] [--agent claude|codex|opencode]

Keys:

  PROJECT_DIR      the directory holding `.specify/`
  AGENT            the integration the command was rendered for ("" when not given)
  SUBSTRATE        the substrate's root ("" when not found), and its tools, each "" when
                   absent: CONTRACT_CLI (scripts/contract.py), CONTRACT_RUN
                   (scripts/contract_run.py), WORK_GATE (scripts/work_gate.py), SPEC_CHARTS
                   (scripts/spec_charts.py), SPEC_LINT (scripts/check_spec_commit.py),
                   CHANGESET_CLI (scripts/spec_changeset.py), DIFF_CARD (scripts/spec_diff_card.py)
  BACKLOG_DIR      the backlog directory from `.agentic-kit.json` ("" = no backlog)
  BACKLOG_CLI      true when the `backlog` CLI is on PATH
  CATALOG          `<BACKLOG_DIR>/metrics.yaml`, the project's metric catalogue ("" = none)
  APPROVAL         the marker's approval level: `auto` (v6 default), `warn` or `required`
  ACTIVE_FEATURE   the feature `.specify/feature.json` points at ("" = none)
  STEPS            {step: absolute path of that step's installed command file for AGENT,
                   "" when not installed} for specify, clarify, plan, contract, checklist,
                   tasks, analyze, implement, converge
  MISSING_STEPS    the steps with no installed file

The substrate is found like the contract extension finds its CLI: `$AGENTIC_CONTRACT_CLI`
(its scripts directory); then walking up from the project, the first `agentic/` holding
`scripts/contract.py` (a substrate mounted at a repository root); then a directory that IS
the substrate (`scripts/contract.py` beside `spec/contract.py`). Stdlib only. Exit 1 with a
message on stderr outside a spec-kit project.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

#: The flow /work runs, in order, and each step's spec-kit command name.
STEPS = (
    ("specify", "speckit.specify"),
    ("clarify", "speckit.clarify"),
    ("plan", "speckit.plan"),
    ("contract", "speckit.contract.feature"),
    ("contract-root", "speckit.contract.root"),
    ("checklist", "speckit.checklist"),
    ("tasks", "speckit.tasks"),
    ("analyze", "speckit.analyze"),
    ("implement", "speckit.implement"),
    ("converge", "speckit.converge"),
)
#: Where each integration's renderer puts a command (spec-kit 1.2.0).
LAYOUTS = {
    "claude": lambda c: Path(".claude/skills") / c.replace(".", "-") / "SKILL.md",
    "codex": lambda c: Path(".agents/skills") / c.replace(".", "-") / "SKILL.md",
    "opencode": lambda c: Path(".opencode/commands") / f"{c}.md",
}
TOOLS = (
    ("CONTRACT_CLI", "contract.py"),
    ("CONTRACT_RUN", "contract_run.py"),
    ("WORK_GATE", "work_gate.py"),
    ("SPEC_CHARTS", "spec_charts.py"),
    ("SPEC_LINT", "check_spec_commit.py"),
    ("CHANGESET_CLI", "spec_changeset.py"),
    ("DIFF_CARD", "spec_diff_card.py"),
)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def project_dir(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / ".specify").is_dir():
            return d
    fail(f"no .specify/ at or above {start}: run this from a spec-kit project")
    raise AssertionError  # unreachable


def substrate(project: Path) -> Path | None:
    env = os.environ.get("AGENTIC_CONTRACT_CLI", "")
    if env and Path(env).is_file():
        return Path(env).resolve().parents[1]
    for d in (project, *project.parents):
        if (d / "agentic" / "scripts" / "contract.py").is_file():
            return (d / "agentic").resolve()
        if (d / "scripts" / "contract.py").is_file() and (d / "spec" / "contract.py").is_file():
            return d.resolve()
    return None


def step_file(project: Path, agent: str, command: str) -> str:
    layouts = [LAYOUTS[agent]] if agent in LAYOUTS else list(LAYOUTS.values())
    for layout in layouts:
        path = project / layout(command)
        if path.is_file():
            return str(path)
    return ""


def main(argv: list[str]) -> int:
    agent = ""
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--json":
            continue
        if a == "--agent":
            if not args:
                fail("--agent needs a value")
            agent = args.pop(0)
        elif a.startswith("--agent="):
            agent = a.split("=", 1)[1]
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            fail(f"unknown option {a!r}")
    project = project_dir(Path.cwd().resolve())
    try:
        marker = json.loads((project / ".agentic-kit.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        marker = {}
    marker = marker if isinstance(marker, dict) else {}
    backlog = marker.get("backlog_dir") if isinstance(marker.get("backlog_dir"), str) else ""
    catalog = project / backlog / "metrics.yaml" if backlog else None
    try:
        feature = json.loads((project / ".specify" / "feature.json").read_text(encoding="utf-8"))
        active = feature.get("feature_directory") or "" if isinstance(feature, dict) else ""
    except (OSError, ValueError):
        active = ""
    sub = substrate(project)
    out: dict = {"PROJECT_DIR": str(project), "AGENT": agent, "SUBSTRATE": str(sub) if sub else ""}
    for key, name in TOOLS:
        path = sub / "scripts" / name if sub else None
        out[key] = str(path) if path and path.is_file() else ""
    steps = {step: step_file(project, agent, cmd) for step, cmd in STEPS}
    out.update(
        BACKLOG_DIR=str(project / backlog) if backlog else "",
        BACKLOG_CLI=shutil.which("backlog") is not None,
        CATALOG=str(catalog) if catalog else "",
        APPROVAL=marker.get("approval")
        if marker.get("approval") in ("auto", "warn", "required")
        else "warn",
        ACTIVE_FEATURE=str(project / active) if active else "",
        STEPS=steps,
        MISSING_STEPS=[s for s, f in steps.items() if not f],
    )
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

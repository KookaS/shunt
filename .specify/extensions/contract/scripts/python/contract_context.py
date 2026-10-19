#!/usr/bin/env python3
"""Context for the task-contract commands, as one JSON object on stdout.

    contract_context.py [--json]          the active feature (feature mode)
    contract_context.py [--json] --root   the backlog's contracts directory (root mode)

Keys: PROJECT_DIR (the directory holding `.specify/`), CONTRACT_CLI (absolute path of the
substrate's `scripts/contract.py`, "" when not found), and per mode —

  feature: FEATURE_DIR, UNIT_ID (`specs/NNN-x`, the id the contract CLI takes),
           FEATURE_SPEC, IMPL_PLAN, TASKS, CONTRACT_FILE, EVIDENCE_FILE, HAS_CONTRACT
  root:    BACKLOG_DIR, CONTRACTS_DIR

The contract CLI is found, in order: `$AGENTIC_CONTRACT_CLI`; then walking up from the
project, the first `agentic/scripts/contract.py` (a substrate mounted at a repository
root); then a directory that IS the substrate (`scripts/contract.py` beside
`spec/contract.py`). Stdlib only. Exit 1 with a message on stderr when the mode's
prerequisites are missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def project_dir(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / ".specify").is_dir():
            return d
    fail(f"no .specify/ at or above {start}: run this from a spec-kit project")
    raise AssertionError  # unreachable


def contract_cli(project: Path) -> str:
    env = os.environ.get("AGENTIC_CONTRACT_CLI", "")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    for d in (project, *project.parents):
        cand = d / "agentic" / "scripts" / "contract.py"
        if cand.is_file():
            return str(cand.resolve())
        if (d / "scripts" / "contract.py").is_file() and (d / "spec" / "contract.py").is_file():
            return str((d / "scripts" / "contract.py").resolve())
    return ""


def feature_paths(project: Path) -> dict:
    script = project / ".specify" / "scripts" / "bash" / "check-prerequisites.sh"
    if not script.is_file():
        fail(f"{script} is missing: spec-kit's shared scripts are not installed here")
    proc = subprocess.run(["bash", str(script), "--json", "--paths-only"], cwd=project,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        fail(proc.stderr.strip() or "check-prerequisites.sh failed")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        fail(f"check-prerequisites.sh printed no JSON: {proc.stdout!r}")
        raise AssertionError from None  # unreachable


def main(argv: list[str]) -> int:
    root_mode = False
    for a in argv:
        if a == "--root":
            root_mode = True
        elif a in ("--json",):
            pass
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            fail(f"unknown option {a!r}")
    project = project_dir(Path.cwd().resolve())
    out = {"PROJECT_DIR": str(project), "CONTRACT_CLI": contract_cli(project)}
    if root_mode:
        try:
            marker = json.loads((project / ".agentic-kit.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {}
        backlog = marker.get("backlog_dir") if isinstance(marker, dict) else None
        if not backlog:
            fail("this project has no backlog (no backlog_dir in .agentic-kit.json): root and "
                 "branch contracts belong to backlog items")
        out.update(BACKLOG_DIR=str(project / backlog),
                   CONTRACTS_DIR=str(project / backlog / "contracts"))
    else:
        paths = feature_paths(project)
        fdir = Path(paths.get("FEATURE_DIR") or "")
        if not fdir.is_dir():
            fail(f"no active feature directory ({fdir}): run the specify step first")
        for key, name in (("FEATURE_SPEC", "spec.md"), ("IMPL_PLAN", "plan.md")):
            if not Path(paths.get(key) or fdir / name).is_file():
                fail(f"{fdir.name} has no {name}: the contract is written after the plan")
        try:
            unit = fdir.resolve().relative_to(project).as_posix()
        except ValueError:
            fail(f"the active feature {fdir} is outside the project {project}")
            raise AssertionError from None  # unreachable
        out.update(FEATURE_DIR=str(fdir), UNIT_ID=unit,
                   FEATURE_SPEC=paths.get("FEATURE_SPEC") or str(fdir / "spec.md"),
                   IMPL_PLAN=paths.get("IMPL_PLAN") or str(fdir / "plan.md"),
                   TASKS=paths.get("TASKS") or str(fdir / "tasks.md"),
                   CONTRACT_FILE=str(fdir / "contract.yaml"),
                   EVIDENCE_FILE=str(fdir / "evidence.jsonl"),
                   HAS_CONTRACT=(fdir / "contract.yaml").is_file())
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

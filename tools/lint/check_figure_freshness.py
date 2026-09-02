#!/usr/bin/env python3
"""SH017: committed figures' freshness record must match the tree they ship in."""

# THE FAILURE THIS EXISTS FOR. Every committed PNG under docs/assets/figures/<half>/ carries a
# freshness record in benchmark/routing/figure_inputs.json: per figure job, a digest of its
# data inputs plus its producer's whole transitive first-party import closure, and a SHA-256 of
# each committed output. An edit to almost any module under benchmark/ (or to the pricing /
# registry data a job prices from) re-stales that record, and until this gate nothing ran at
# commit time — the stale figure_inputs.json repeatedly sailed to a red `benchmark-integrity`
# CI job and a red figure-freshness pytest before anyone noticed. The check is the SAME
# `pipeline --check-figures` CI runs, which is seconds (it hashes files; it draws nothing), so
# it belongs where the drift is born: the commit that ships the edit.
#
# Whole-tree gate (the SH015/SH016 shape): `pass_filenames: false` + `always_run: true`. The
# failure is an EDIT to code/data that outlives the manifest — no staged-file list names the
# file it invalidated. Repair by re-running the certifying pipeline stage(s), e.g.
# `uv run --extra benchmark python -m benchmark.pipeline --from evaluate`; never by hand-editing
# benchmark/routing/figure_inputs.json.

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap() -> None:
    """Prepend the repo root and `src/` so `benchmark` and `shunt` resolve as a script."""
    # `benchmark` is deliberately NOT installed (AGENTS.md) and the editable install that
    # provides `shunt` lives in whichever interpreter pre-commit happens to use — under a bare
    # `python tools/lint/x.py`, sys.path[0] is tools/lint/ and neither resolves. This is the
    # sanctioned mutation (see check_model_triage.py).
    for path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
        if path not in sys.path:  # noqa: TID251 (banned-api read; tooling bootstrap)
            sys.path.insert(0, path)  # noqa: TID251, SH003 (tooling bootstrap; packages not installed)


def main(argv: list[str] | None = None) -> int:
    """Figure-freshness gate: exit 1 on a stale/drifted/unproduced committed figure set."""
    del argv  # whole-tree gate: no file args (the SH004/SH005 shape)
    _bootstrap()
    from benchmark import config, pipeline  # noqa: PLC0415 (deferred until _bootstrap ran)

    config.load("benchmark/benchmark.yaml")
    # `check_figures` prints each STALE/MISSING/DRIFTED/UNCERTIFIED/UNPRODUCED job with the
    # sanctioned repair command, and returns the exit code; report nothing further.
    return pipeline.check_figures()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""SH015: live-pool models must clear the triage frontier (routine or escalation)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from _shared import Finding

if TYPE_CHECKING:
    from benchmark.routing import triage as _triage

_CODE = "SH015"
# Every live-pool DROP points at the shipped policy's models: list as the fix location.
_POLICY_PATH = "src/shunt/config/router.yaml"
# The packaged repo root — where `benchmark/` (deliberately not installed) lives.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap() -> None:
    """Prepend the repo root and `src/` so `benchmark` and `shunt` resolve as a script."""
    # `benchmark` is deliberately NOT installed (AGENTS.md), and the editable install that
    # provides `shunt` lives in whichever interpreter pre-commit happens to use — under a bare
    # `python tools/lint/x.py`, sys.path[0] is tools/lint/ and neither resolves unless the hook
    # env happens to have the venv on PATH. Prepend both repo root and `src/` so the gate is
    # hermetic over the checkout, install-independent. This is the sanctioned mutation.
    for path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
        if path not in sys.path:  # noqa: TID251 (banned-api read; tooling bootstrap)
            sys.path.insert(0, path)  # noqa: TID251, SH003 (tooling bootstrap; packages not installed)


def _findings(rows: Sequence[_triage.TriageRow]) -> list[Finding]:
    """Every live-pool row whose verdict is DROP, as an SH015 finding at the policy file."""
    from benchmark.routing import triage  # noqa: PLC0415 (deferred until _bootstrap ran)
    from benchmark.routing._live_pool import packaged_live_pool  # noqa: PLC0415

    return [
        Finding(
            _POLICY_PATH,
            0,
            0,
            f"live-pool model {r.model} is DROP: routine {r.routine_status}, "
            f"escalation {r.esc_verdict or 'n/a'} (n={r.n}, rate {r.marginal_rate:.3f}); "
            f"remove it from router.yaml models: or collect evidence that clears the frontier",
        )
        for r in triage.violations(packaged_live_pool(), rows)
    ]


def _cold_start_constant_violations(live_pool: Sequence[str]) -> list[tuple[str, str]]:
    """(constant-name, model) for every cold-start constant not a member of the live pool."""
    from shunt.proxy.router import _DEFAULT_MODEL  # noqa: PLC0415
    from shunt.router.cold_start import _COLD_START_MODEL, _DEFAULT_FALLBACK_MODELS  # noqa: PLC0415
    from shunt.router.selection import _DEFAULT_COLD_START_MODEL  # noqa: PLC0415

    pool = set(live_pool)
    violations: list[tuple[str, str]] = []
    for name, value in (
        ("_COLD_START_MODEL", _COLD_START_MODEL),
        ("_DEFAULT_COLD_START_MODEL", _DEFAULT_COLD_START_MODEL),
        ("_DEFAULT_MODEL", _DEFAULT_MODEL),
    ):
        if value not in pool:
            violations.append((name, value))
    for fallback in _DEFAULT_FALLBACK_MODELS:
        if fallback not in pool:
            violations.append(("_DEFAULT_FALLBACK_MODELS", fallback))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Triage gate: exit 1 on a live-pool DROP or a cold-start constant off the pool."""
    del argv  # whole-tree gate: no file args (the SH004/SH005 shape)
    _bootstrap()
    from benchmark.routing import triage  # noqa: PLC0415
    from benchmark.routing._live_pool import packaged_live_pool  # noqa: PLC0415

    live_pool = packaged_live_pool()
    rows = triage.triage_default()
    findings = _findings(rows)
    constant_violations = _cold_start_constant_violations(live_pool)
    for f in findings:
        print(f"{f.path}:{f.line}:{f.col}: [{_CODE} ERROR] {f.message}", file=sys.stderr)
    for r in triage.exceptioned(live_pool, rows):
        print(
            f"{_POLICY_PATH}:0:0: [{_CODE} ADVISORY] live-pool model {r.model} kept under "
            f"named exception: {r.exception_note}",
            file=sys.stderr,
        )
    for name, model in constant_violations:
        print(
            f"{_POLICY_PATH}:0:0: [{_CODE} ERROR] cold-start constant {name}={model!r} is "
            f"not a member of the live pool; a default that names a dominated model would "
            f"defeat the router's measured pool",
            file=sys.stderr,
        )
    if not any(r.n > 0 for r in rows):
        print(
            f"{_POLICY_PATH}:0:0: [{_CODE} ADVISORY] no measured corpus: every model is "
            f"INSUFFICIENT-DATA, so the live-pool DROP check is vacuous here",
            file=sys.stderr,
        )
    for model in triage.load_exceptions():
        if model not in set(live_pool):
            print(
                f"{_POLICY_PATH}:0:0: [{_CODE} ADVISORY] triage-exception key {model!r} is "
                f"not a member of the live pool; the exception is dead config",
                file=sys.stderr,
            )
    return 1 if findings or constant_violations else 0


if __name__ == "__main__":
    sys.exit(main())

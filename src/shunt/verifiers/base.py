from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class VerifierResult:
    outcome: str
    confidence: float
    detail: str = ""
    is_infra_failure: bool = False
    matched_pattern: str | None = None
    # The process exit code of the off-wire check (None when not run via subprocess). Carried
    # for provenance/telemetry only — the escalation trigger keys on `outcome`/`confirmed`, not
    # this: a real pytest/jest/go failure is exit 1 (cargo 101), never a fixed "blocking" code.
    exit_code: int | None = None
    # A stable identity for WHAT failed (e.g. a pytest node id), so a recurrence of the SAME
    # failure can be deduped across attempts. None on success / when nothing parseable.
    failing_check_id: str | None = None
    # A failure that a confirming verifier (RerunConfirmingVerifier) re-ran and reproduced on
    # unchanged state — the flake guard. Off by default so a bare single-run red is never
    # trusted as confirmed; only a verifier that actually re-runs sets it.
    confirmed: bool = False


class Verifier(abc.ABC):
    @abc.abstractmethod
    def verify(self, text: str, work_dir: str | None = None) -> VerifierResult: ...

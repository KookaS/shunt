"""Rerun-confirm wrapper: a failure is only trusted if it reproduces on unchanged state."""

# ~83% of pass→fail transitions are flakes, not real regressions. The state-of-practice rule
# is fail-then-pass-on-the-same-version = flake, ignore. This wrapper reruns a failing inner
# verification; if any rerun stops failing, the failure is treated as a flake and abstained
# (outcome "unknown" → the capture path writes NOTHING), so a flaky red never poisons the
# store or trips escalation. A failure that reproduces every time is passed through unchanged.

from __future__ import annotations

import logging
from dataclasses import replace

from .base import Verifier, VerifierResult

logger = logging.getLogger(__name__)


class RerunConfirmingVerifier(Verifier):
    """Wrap a verifier so a ``failure`` is confirmed by rerun before it is trusted."""

    def __init__(self, inner: Verifier, reruns: int = 2) -> None:
        self._inner = inner
        self._reruns = max(0, reruns)

    def verify(self, text: str = "", work_dir: str | None = None) -> VerifierResult:
        first = self._inner.verify(text, work_dir)
        if first.outcome != "failure" or self._reruns == 0:
            return first
        for attempt in range(1, self._reruns + 1):
            again = self._inner.verify(text, work_dir)
            if again.outcome != "failure":
                logger.info(
                    "verify: failure did not reproduce on rerun %d/%d (was %r, now %r) — "
                    "treating as flake, abstaining",
                    attempt,
                    self._reruns,
                    first.failing_check_id,
                    again.outcome,
                )
                return VerifierResult(
                    outcome="unknown",
                    confidence=0.0,
                    detail=f"flake: first run failed, rerun {attempt} did not reproduce",
                    exit_code=again.exit_code,
                )
        # Reproduced on every rerun → a real failure. Stamp `confirmed` so the escalation log
        # trusts it structurally; a bare (non-confirming) verifier never sets this flag.
        return replace(first, confirmed=True)

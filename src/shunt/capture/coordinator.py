"""Off-wire verified-outcome capture: resolve a repo, re-run its tests, label.

Pure orchestration (no threads) so resolve→label is unit-testable without a server;
only off-wire re-execution writes a verified Tier-2 label (never model self-narration).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Protocol

from shunt.capture.refit import RefitScheduler
from shunt.db.store import OutcomeEvent, OutcomeStore
from shunt.proxy.wire_signals import derive_wire_tier1_outcome
from shunt.session import Session
from shunt.verifiers.base import VerifierResult

logger = logging.getLogger(__name__)

# Outcomes the capture path may persist. `unknown` is deliberately absent — an
# unverifiable session writes NOTHING; a label is never synthesized from it.
_LABELLABLE: frozenset[str] = frozenset({"success", "weak_success", "failure"})
# The verified labels that count as the chosen model proving capable (mirrors the
# read-back seam's `_SUCCESS_LABELS`); `failure` is the only not-capable outcome.
_SUCCESS_LABELS: frozenset[str] = frozenset({"success", "weak_success"})


class OffWireVerifier(Protocol):
    """The off-wire re-executor seam (AutoDetectVerifier in production)."""

    def verify(self, text: str = "", work_dir: str | None = None) -> VerifierResult: ...


class RecordOutcomeCallback(Protocol):
    """The engine's ``record_outcome`` seam — fed only verified Tier-2 outcomes.

    Injected (not a hard engine dependency) so the coordinator stays unit-testable and a
    no-engine deployment simply passes ``None`` and records nothing into the gate.
    """

    def __call__(self, *, downshift: bool, success: bool) -> None: ...


class WorkDirResolver:
    """Resolve a session to a repo root from **operator config only**.

    Precedence: per-``tool_identity`` override map, then single ``work_dir`` (env
    ``SHUNT_WORK_DIR`` beats file); none ⇒ ``None`` (manual-only). Never a wire path (RCE).
    """

    def __init__(
        self, work_dir: str | None = None, work_dirs: dict[str, str] | None = None
    ) -> None:
        self._work_dir = work_dir
        self._work_dirs = work_dirs or {}

    @classmethod
    def from_config(
        cls, work_dir: str | None = None, work_dirs: dict[str, str] | None = None
    ) -> WorkDirResolver:
        """Build a resolver, letting env ``SHUNT_WORK_DIR`` override the file's single path."""
        env = os.environ.get("SHUNT_WORK_DIR")
        return cls(work_dir=env or work_dir, work_dirs=work_dirs)

    def resolve(self, session: Session) -> str | None:
        mapped = self._work_dirs.get(session.tool_identity)
        if mapped:
            return mapped
        return self._work_dir or None


def _run_signature(work_dir: str) -> str:
    """Deterministic per-repo token: a worker retry of the same close re-runs the same
    verification (same work_dir) → same idempotency_key → deduped write."""
    return hashlib.sha256(work_dir.encode()).hexdigest()[:16]


class CaptureCoordinator:
    """Orchestrate one closed session's capture: resolve → verify → append → index."""

    def __init__(
        self,
        resolver: WorkDirResolver,
        verifier: OffWireVerifier,
        store: OutcomeStore,
        record_outcome_callback: RecordOutcomeCallback | None = None,
        refit_scheduler: RefitScheduler | None = None,
    ) -> None:
        self._resolver = resolver
        self._verifier = verifier
        self._store = store
        self._record_outcome_callback = record_outcome_callback
        self._refit_scheduler = refit_scheduler

    def capture(self, session: Session) -> None:
        """Capture a **closed** session's verified outcome, or write nothing."""
        work_dir = self._resolver.resolve(session)
        if work_dir is None:
            # No operator-configured repo → no off-wire run, no fabricated Tier-2. A weak,
            # quarantined Tier-1 wire prior is recorded only if a genuine structured signal
            # was observed on this session; otherwise nothing when there is no verifiable signal.
            logger.debug("capture: no work_dir for session %s — manual-only", session.session_id)
            self._maybe_append_wire_tier1(session)
            return

        result = self._verifier.verify(work_dir=work_dir)
        if result.outcome not in _LABELLABLE:
            # `unknown` (no framework / infra failure) is never a Tier-2 label. Fall back to
            # the weak, quarantined Tier-1 wire prior if a structured signal exists.
            logger.debug(
                "capture: session %s outcome=%s not labellable — checking wire prior",
                session.session_id,
                result.outcome,
            )
            self._maybe_append_wire_tier1(session)
            return

        self._append_tier2(session, work_dir, result)

    def _maybe_append_wire_tier1(self, session: Session) -> None:
        """Record a weak, quarantined Tier-1 prior from structured wire signals, or nothing.

        Written only on a genuine structured signal; the store quarantines Tier-1,
        and a ``wire_tier1`` key lets it coexist with a later ``auto_tier2`` event.
        """
        derived = derive_wire_tier1_outcome(session.metadata)
        if derived is None:
            return
        outcome, confidence = derived
        self._store.append_outcome_event(
            OutcomeEvent(
                session_id=session.session_id,
                tier=1,
                source="wire_tier1",
                outcome=outcome,
                confidence=confidence,
                run_signature="wire",
                model_fingerprint=self._session_fingerprint(session.session_id),
            )
        )
        logger.info(
            "capture: session %s recorded weak Tier-1 wire prior %s (quarantined)",
            session.session_id,
            outcome,
        )

    def _append_tier2(self, session: Session, work_dir: str, result: VerifierResult) -> None:
        row = self._store.get_session(session.session_id)
        inserted = self._store.append_outcome_event(
            OutcomeEvent(
                session_id=session.session_id,
                tier=2,
                source="auto_tier2",
                outcome=result.outcome,
                confidence=result.confidence,
                run_signature=_run_signature(work_dir),
                model_fingerprint=_fingerprint_of(row),
            )
        )
        if inserted:
            # Only a fresh insert reaches here: a duplicate capture (worker retry, double
            # sweep) dedups on the idempotency_key → inserted=False → no double-record.
            self._store.persist_index()
            self._record_outcome(row, result)
            if self._refit_scheduler is not None:
                self._refit_scheduler.note_capture()
            logger.info(
                "capture: session %s labelled %s (auto_tier2)", session.session_id, result.outcome
            )

    def _record_outcome(self, row: dict[str, Any] | None, result: VerifierResult) -> None:
        """Feed the verified Tier-2 outcome to the engine's gate, attributed to this
        session's own routed decision (durable on the session row)."""
        if self._record_outcome_callback is None:
            return
        self._record_outcome_callback(
            downshift=_downshift_of(row),
            success=result.outcome in _SUCCESS_LABELS,
        )

    def _session_fingerprint(self, session_id: str) -> str | None:
        return _fingerprint_of(self._store.get_session(session_id))


def _fingerprint_of(row: dict[str, Any] | None) -> str | None:
    """The resolved model-version fingerprint copied onto the session row, or None."""
    if row is None:
        return None
    fingerprint = row.get("model_fingerprint")
    return fingerprint if isinstance(fingerprint, str) else None


def _downshift_of(row: dict[str, Any] | None) -> bool:
    """Whether the session's routed decision was an exploratory downshift (from provenance)."""
    if row is None:
        return False
    raw = row.get("decision_provenance")
    if not isinstance(raw, str):
        return False
    try:
        provenance = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return bool(provenance.get("downshift", False))

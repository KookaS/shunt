"""Opt-in full-content per-step trajectory capture — the recorder + its redaction."""

# Secure-by-default: OFF unless an operator sets it; every free-text field is redacted before it
# can be persisted. The recorder is observe-only and boundary-only — it is handed a finalized
# turn at a decision/session boundary, never on the wire and never mid-cached-turn, and it never
# alters or triggers a routing decision or an upstream call.
#
# Only the behaviour-only fields (no prose, no args, no observation text) may ever cross into a
# committable file; that whitelist is `COMMITTABLE_FIELDS` here and is kept at parity with the
# offline evaluator's whitelist by a test.

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Final, Protocol

from shunt.proxy.redaction import redact_secrets

if TYPE_CHECKING:
    from collections.abc import Callable

# Behaviour-only fields safe to commit: the verified-outcome core + numeric behaviour signals.
# No free-text (metadata/observation/action/args/result) is here — those never leave the
# encrypted local plane. Kept in parity with the offline evaluator's schema whitelist by a test.
COMMITTABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "decision_index",
        "failing_check_id",
        "exit_code",
        "blocking",
        "confirmed",
        "success",
        "step_index",
        "is_infra_failure",
        "rank_index",
        "effort_index",
        "is_revert",
        "retry_count",
        "loop_signal",
    }
)


@dataclass(frozen=True)
class StepRecord:
    """One captured per-step trajectory record (full content + behaviour signals)."""

    step_index: int
    decision_index: int
    # free text — redacted before persistence, never committable
    metadata: dict[str, str]
    observation: str
    action: str
    args: str | None
    result: str
    # Behaviour-only fields below — the committable subset, no prose
    failing_check_id: str | None
    exit_code: int | None
    blocking: bool
    is_infra_failure: bool
    confirmed: bool
    success: bool
    is_revert: bool = False
    retry_count: int = 0
    loop_signal: bool = False
    rank_index: int | None = None
    effort_index: int | None = None

    def committable(self) -> dict[str, object]:
        """The behaviour-only projection — the ONLY subset that may cross to a committed file."""
        # `failing_check_id` is the one committable string that can carry arbitrary text (a
        # parametrized test id can embed a secret), so it is scrubbed on the way out like every
        # other free-text field — the committable projection must never ship an unredacted string.
        out: dict[str, object] = {}
        for k, v in asdict(self).items():
            if k not in COMMITTABLE_FIELDS:
                continue
            out[k] = redact_secrets(v) if k == "failing_check_id" and isinstance(v, str) else v
        return out


def redact_record(record: StepRecord) -> StepRecord:
    """Redact every free-text field (and metadata values) of a captured step."""
    # `failing_check_id` is included: it is nominally a behaviour field, but a parametrized test
    # id carries arbitrary text and can embed a secret, so "every free-text field" must mean it
    # too. `committable()` redacts it independently — this is the defense-in-depth copy.
    metadata = {k: redact_secrets(v) for k, v in record.metadata.items()}
    return replace(
        record,
        metadata=metadata,
        observation=redact_secrets(record.observation),
        action=redact_secrets(record.action),
        args=redact_secrets(record.args) if record.args is not None else None,
        result=redact_secrets(record.result),
        failing_check_id=(
            redact_secrets(record.failing_check_id) if record.failing_check_id is not None else None
        ),
    )


class TrajectorySink(Protocol):
    """A destination the recorder appends redacted records to (e.g. the encrypted local plane)."""

    def write(self, records: list[StepRecord]) -> None: ...


class TrajectoryRecorder:
    """Off-wire, boundary-only recorder. Inert unless `enabled`; redacts before it persists."""

    def __init__(
        self,
        sink: TrajectorySink,
        *,
        enabled: bool = False,
        redactor: Callable[[StepRecord], StepRecord] = redact_record,
    ) -> None:
        self._sink = sink
        self._enabled = enabled
        self._redactor = redactor

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, records: list[StepRecord]) -> None:
        """Redact then persist a finalized turn's steps. No-op when disabled (default)."""
        if not self._enabled or not records:
            return
        self._sink.write([self._redactor(record) for record in records])


__all__ = [
    "COMMITTABLE_FIELDS",
    "StepRecord",
    "TrajectoryRecorder",
    "TrajectorySink",
    "redact_record",
]

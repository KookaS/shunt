"""Canonical trajectory format: one frozen schema shared by normalizers, replay,
metrics, authenticity, and both data planes. Serialized as JSON Lines — a header
record first, then one StepView per line.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING, Final

from shunt.proxy.redaction import redact_secrets
from shunt.router.escalation import derive_blocking

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION: Final[str] = "1"

# Free-text committable fields are scrubbed before they can enter a committed projection.
# `failing_check_id` is the ONE committable string that can carry arbitrary text (a parametrized
# test id can embed a secret), so it passes through the same redactor as every other free-text
# field on the way out of the live plane.
_REDACTED_COMMITTABLE_FIELDS: Final[frozenset[str]] = frozenset({"failing_check_id"})

# The behaviour-only field whitelist (the ONLY fields that may cross from the live
# plane into a committable file). It is the verified-outcome core plus the numeric
# behaviour signals — every field that carries no prose, no args, no observation
# text, hence no possible secret. The free-text fields (metadata/observation/action/
# args/result) are deliberately absent: they stay in the encrypted live plane and
# never reach a committed file.
COMMITTABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        # verified-outcome core
        "decision_index",
        "failing_check_id",
        "exit_code",
        "blocking",
        "confirmed",
        "success",
        # numeric behaviour signals
        "step_index",
        "is_infra_failure",
        "tier_index",
        "effort_index",
        "is_revert",
        "retry_count",
        "loop_signal",
    }
)


@dataclass(frozen=True)
class StepView:
    """One per-decision record: PrefixGuard-7 core + verified outcome + drift signals."""

    # identity / order
    step_index: int
    decision_index: int
    parent_step_index: int | None
    # PrefixGuard-7 (free text — never committable off the live plane)
    metadata: dict[str, str]
    observation: str
    action: str
    tool: str
    args: str | None
    result: str
    status: str
    # verified outcome (the escalation signal)
    test_passed: int | None
    test_total: int | None
    failing_check_id: str | None
    exit_code: int | None
    blocking: bool
    is_infra_failure: bool
    confirmed: bool
    success: bool
    # behaviour signals (future drift-detector fuel)
    is_revert: bool
    retry_count: int
    loop_signal: bool
    subgoal_progress: float | None
    dedup_key: str | None
    # cost / routing
    model: str | None
    reasoning_effort: str | None
    tier_index: int | None
    effort_index: int | None
    real_cost: float | None


@dataclass(frozen=True)
class TrajectoryHeader:
    """Provenance + integrity header; the JSONL's first record."""

    schema_version: str
    trajectory_id: str
    dataset: str
    plane: str
    framework: str
    terminal_resolved: bool
    instance_id: str | None
    license: str | None
    dataset_revision: str | None
    redacted: bool
    content_sha256: str
    n_steps: int


@dataclass(frozen=True)
class Trajectory:
    """A header plus its ordered per-decision StepViews."""

    header: TrajectoryHeader
    steps: list[StepView]


def normalize_dedup_key(failing_check_id: str | None) -> str | None:
    """The dedup key from a failing-check id — normalize-at-source identity (see below)."""
    # The live verifier (`shunt.verifiers.tier2`) already emits the stable node-id/hash, so a
    # StepView's `failing_check_id` IS the normalized key and this is identity. The single
    # definition, so parsers and the authenticity recompute cannot drift into a false mismatch.
    #
    # CONTRACT (Layer-1 authenticity depends on it): every parser stores
    # `dedup_key = normalize_dedup_key(failing_check_id)`, and this MUST stay idempotent
    # (`normalize(normalize(x)) == normalize(x)`). Then `recompute_dedup_key` can never raise a
    # false `dedup_key.mismatch` on legitimate committed data. A future non-identity normalizer
    # must preserve idempotency or the pinned contract test fails.
    return failing_check_id


def recompute_dedup_key(step: StepView) -> str | None:
    """Recompute the dedup key from `failing_check_id` via the shared normalizer."""
    return normalize_dedup_key(step.failing_check_id)


def recompute_blocking(step: StepView) -> bool:
    """A confirmed, non-infra capability failure — the ONE derivation, shared with shunt."""
    return derive_blocking(step.success, step.is_infra_failure)


def committable_projection(step: StepView) -> dict[str, object]:
    """The behaviour-only subset of a StepView that may be committed (free text redacted)."""
    out: dict[str, object] = {}
    for name in _STEP_FIELD_NAMES:
        if name not in COMMITTABLE_FIELDS:
            continue
        value = getattr(step, name)
        if name in _REDACTED_COMMITTABLE_FIELDS and isinstance(value, str):
            value = redact_secrets(value)
        out[name] = value
    return out


def content_sha256(steps: list[StepView]) -> str:
    """SHA-256 of the ordered StepView payload (canonical JSON, sorted keys)."""
    payload = json.dumps(
        [asdict(s) for s in steps], sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scrub_free_text(step: StepView) -> StepView:
    """A copy of *step* with every free-text field run through the secret redactor."""
    # Structural (not caller-trusted) defence in depth on the committable write path: a future live
    # collect must not be able to land an unredacted secret in a committed dump. On clean public
    # data the redactor is a no-op; when it does fire, dump_jsonl re-hashes over the scrubbed bytes
    # so the persisted trajectory still passes its own Layer-1 check.
    return replace(
        step,
        metadata={k: redact_secrets(v) for k, v in step.metadata.items()},
        observation=redact_secrets(step.observation),
        action=redact_secrets(step.action),
        args=redact_secrets(step.args) if step.args is not None else None,
        result=redact_secrets(step.result),
    )


def dump_jsonl(traj: Trajectory, path: Path) -> None:
    """Write the header then one StepView per line (JSONL), free-text scrubbed.

    The header hash commits to the *scrubbed* payload actually written, so a trajectory
    whose secrets were redacted on the way to disk still passes its own Layer-1 check.
    """
    steps = [_scrub_free_text(s) for s in traj.steps]
    scrubbed_hash = content_sha256(steps)
    header = traj.header
    if scrubbed_hash != header.content_sha256:
        # Redaction changed the bytes: re-hash over what is written and record that it happened.
        header = replace(header, content_sha256=scrubbed_hash, redacted=True)
    lines = [json.dumps(asdict(header), sort_keys=True, ensure_ascii=True)]
    lines.extend(json.dumps(asdict(s), sort_keys=True, ensure_ascii=True) for s in steps)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> Trajectory:
    """Read a header + StepView JSONL back into a Trajectory (inverse of dump_jsonl)."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    header = TrajectoryHeader(**records[0])
    steps = [StepView(**record) for record in records[1:]]
    return Trajectory(header=header, steps=steps)


_STEP_FIELD_NAMES: Final[tuple[str, ...]] = tuple(f.name for f in fields(StepView))


__all__ = [
    "COMMITTABLE_FIELDS",
    "SCHEMA_VERSION",
    "StepView",
    "Trajectory",
    "TrajectoryHeader",
    "committable_projection",
    "content_sha256",
    "dump_jsonl",
    "load_jsonl",
    "normalize_dedup_key",
    "recompute_blocking",
    "recompute_dedup_key",
]

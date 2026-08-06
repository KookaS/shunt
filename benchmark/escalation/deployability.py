"""What a production escalation decision actually sees, and whether a result rests only on that."""

# THE SEAM THIS EXTENDS, rather than duplicating. `shunt.router.escalation.decide_escalation` is
# the shipped decision, and its inputs ARE the production decision context: a windowed list of
# `FailureEvent` plus an `EscalationContext`. `_checkpoint_features` already reduces exactly those
# to the vector an off-policy model may condition on. So the context is NOT re-declared here — the
# field set is READ off those two dataclasses, and a field added or removed there changes this gate
# on the next import. A hand-copied list is how the first copy goes stale.
#
# THE TWO MISMATCHES THIS MAKES MACHINE-READABLE.
#
#   FIELDS. `OFFLINE_SOURCE` maps each production field to the StepView attribute a replay record
#   carries it in. A production field absent from that map has NO offline counterpart at all: the
#   ladder ceilings (`max_rank_index`, `max_effort_index`) are resolved from the live model pool at
#   decision time and were never recorded. Conversely a StepView attribute absent from its VALUES
#   is not in the production context — `action`, `tool`, `observation`, `status` and `step_index`
#   are offline-only, and `is_infra_failure` is folded into `blocking` by `derive_blocking` before
#   the event is built, so production never retains it as its own field.
#
#   COUNTING. The eval replays the recurrence counter in two modes. `as_shipped` counts every
#   same-key verified failure, which is exactly what `escalation.counts_as_failure` does live.
#   `edit_gated` additionally skips every failure before the agent's first edit-like action, which
#   it decides by regex over `StepView.action` (`replay.is_edit_action`) — and `action` is one of
#   the offline-only attributes named above. There is no counting knob in `EscalationPolicy` and no
#   `count_from_first_edit` anywhere in `src/`, so `edit_gated` is not a configuration the product
#   can be asked for. It is declared here rather than restated in prose because that restatement is
#   what went stale: the fact lived in seven comments and doc sentences and in no data structure, so
#   the label a figure carries could not know about it.
#
#   CADENCE. `_maybe_escalate` is reached only through `_finalize_decision` <- `RouterEngine.decide`
#   <- `RouterProxy._decide_once`, and `_get_or_lock_model` returns the already-locked model on
#   every later turn. Production therefore makes ONE escalation decision per session. Failure events
#   are OBSERVED per verified outcome, but the decision is not re-made — so a score read off a
#   prefix at step depth d describes a decision point production never reaches.
#
# A verdict from this gate is a scope statement, not a statistic: it changes no number, it says
# which question the number answers.

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from shunt.router.escalation import EscalationContext, FailureEvent

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from benchmark.escalation.schema import StepView

# Every field `decide_escalation` receives, read off the shipped dataclasses so it cannot drift.
CONTEXT_FIELDS: Final[frozenset[str]] = frozenset(
    f.name for f in (*fields(FailureEvent), *fields(EscalationContext))
)

# production context field -> the StepView attribute an offline replay record carries it in.
OFFLINE_SOURCE: Final[dict[str, str]] = {
    "decision_index": "decision_index",
    "dedup_key": "failing_check_id",
    "exit_code": "exit_code",
    "success": "success",
    "confirmed": "confirmed",
    "blocking": "blocking",
    "current_rank_index": "rank_index",
    "current_effort_index": "effort_index",
    "loop_health_alarm": "loop_signal",
}

# Offline feature -> the StepView attributes `features._features` reads to build it. Kept in
# lockstep with `features.FEATURE_NAMES` by a test: an undeclared feature is treated as
# unsupported, so the gate fails closed rather than passing a column nobody mapped.
FEATURE_SOURCES: Final[dict[str, tuple[str, ...]]] = {
    "fail_rate": ("success",),
    "infra_rate": ("is_infra_failure",),
    "max_action_repeat_rate": ("action",),
}

_IN_CONTEXT: Final[frozenset[str]] = frozenset(OFFLINE_SOURCE.values())


class Cadence(StrEnum):
    """The unit a decision is made at — production's, or the one an eval scored."""

    SESSION = "session"
    STEP = "step"


PRODUCTION_CADENCE: Final[Cadence] = Cadence.SESSION


class CountingMode(StrEnum):
    """How the replayed recurrence counter treats the reproduction phase."""

    AS_SHIPPED = "as_shipped"
    EDIT_GATED = "edit_gated"


PRODUCTION_COUNTING: Final[CountingMode] = CountingMode.AS_SHIPPED

# Counting mode -> the StepView attributes deciding it reads. Scored through the SAME `_IN_CONTEXT`
# subset test the features use, so the verdict is derived from `OFFLINE_SOURCE` rather than from a
# second opinion about what production holds. An unlisted mode hits the `<undeclared>` sentinel in
# `assess` and is unsupported until someone maps it.
COUNTING_SOURCES: Final[dict[str, tuple[str, ...]]] = {
    CountingMode.AS_SHIPPED: (),
    CountingMode.EDIT_GATED: ("action",),
}


@dataclass(frozen=True)
class ProjectedContext:
    """One offline record on the production context; `missing` IS the field mismatch."""

    values: dict[str, object]
    missing: frozenset[str]


def project_step(step: StepView) -> ProjectedContext:
    """Project an offline replay record onto what production holds at a decision boundary."""
    # `is None`, never falsiness: `success=False` and `decision_index=0` are filled values, and
    # treating them as missing would report a mismatch that is not there.
    values: dict[str, object] = {}
    missing: set[str] = set()
    for name in sorted(CONTEXT_FIELDS):
        source = OFFLINE_SOURCE.get(name)
        value = getattr(step, source, None) if source is not None else None
        if value is None:
            missing.add(name)
        else:
            values[name] = value
    return ProjectedContext(values=values, missing=frozenset(missing))


def unfilled_context_fields(steps: Iterable[StepView]) -> frozenset[str]:
    """Context fields NO record fills — the corpus-level mismatch, measured rather than assumed."""
    filled: set[str] = set()
    for step in steps:
        filled |= project_step(step).values.keys()
    return CONTEXT_FIELDS - filled


@dataclass(frozen=True)
class Deployability:
    """Whether an evaluated feature set at a cadence yields a deployable estimate."""

    cadence: Cadence
    supported: tuple[str, ...]
    unsupported: tuple[str, ...]
    starved: tuple[str, ...]
    # Which recurrence counter produced the number, and the StepView attributes that counter reads
    # which a production decision never holds. Defaulted to the SHIPPED counter (which reads
    # nothing extra), so an existing caller keeps its exact verdict and only a caller that opts
    # into an eval-only mode pays for it.
    counting: CountingMode = PRODUCTION_COUNTING
    counting_unsupported: tuple[str, ...] = ()

    @property
    def deployable(self) -> bool:
        """Deployable only when every feature AND the counter project, at production's cadence."""
        return (
            not self.unsupported
            and not self.starved
            and not self.counting_unsupported
            and self.cadence is PRODUCTION_CADENCE
        )

    @property
    def label(self) -> str:
        """The label a reported number MUST carry — this is the distinction being enforced."""
        return "DEPLOYABLE ESTIMATE" if self.deployable else "OFFLINE-ONLY UPPER BOUND"

    @property
    def reason(self) -> str:
        """Why, naming the offending features and cadence — never a bare boolean."""
        if self.deployable:
            return (
                f"every scored feature projects onto the production decision context, the counter "
                f"is the shipped one ({self.counting}), and the cadence is production's "
                f"({self.cadence})"
            )
        parts = []
        if self.counting_unsupported:
            parts.append(
                f"the '{self.counting}' counter reads step fields absent from the production "
                f"decision context ({', '.join(self.counting_unsupported)}), and the product has "
                f"no such counting mode"
            )
        if self.unsupported:
            parts.append(
                f"{len(self.unsupported)} feature(s) read fields absent from the production "
                f"decision context ({', '.join(self.unsupported)})"
            )
        if self.starved:
            parts.append(
                f"{len(self.starved)} feature(s) read production fields no corpus record fills "
                f"({', '.join(self.starved)})"
            )
        if self.cadence is not PRODUCTION_CADENCE:
            parts.append(
                f"scored at cadence '{self.cadence}' while production decides once per "
                f"'{PRODUCTION_CADENCE}'"
            )
        return "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "deployable": self.deployable,
            "reason": self.reason,
            "cadence": str(self.cadence),
            "production_cadence": str(PRODUCTION_CADENCE),
            "counting": str(self.counting),
            "production_counting": str(PRODUCTION_COUNTING),
            "counting_unsupported_fields": list(self.counting_unsupported),
            "supported_features": list(self.supported),
            "unsupported_features": list(self.unsupported),
            "starved_features": list(self.starved),
        }


def assess(
    feature_names: Sequence[str],
    cadence: Cadence,
    *,
    unfilled: frozenset[str] = frozenset(),
    counting: CountingMode = PRODUCTION_COUNTING,
) -> Deployability:
    """Gate a result: do this feature set, counter and cadence describe a policy production runs?"""
    # An undeclared feature maps to a sentinel that is never in context, so a new column added to
    # `features.FEATURE_NAMES` without a source declaration is UNSUPPORTED until someone maps it.
    starved_sources = {OFFLINE_SOURCE[f] for f in unfilled if f in OFFLINE_SOURCE}
    unsupported = tuple(
        n
        for n in feature_names
        if not set(FEATURE_SOURCES.get(n, ("<undeclared>",))) <= _IN_CONTEXT
    )
    starved = tuple(
        n
        for n in feature_names
        if n not in unsupported and set(FEATURE_SOURCES.get(n, ())) & starved_sources
    )
    supported = tuple(n for n in feature_names if n not in unsupported and n not in starved)
    # Same subset test, same `_IN_CONTEXT` reference: the counter is judged by exactly the rule
    # that judges a feature, so the two verdicts cannot disagree about what production holds.
    counting_unsupported = tuple(
        s for s in COUNTING_SOURCES.get(counting, ("<undeclared>",)) if s not in _IN_CONTEXT
    )
    return Deployability(
        cadence=cadence,
        supported=supported,
        unsupported=unsupported,
        starved=starved,
        counting=counting,
        counting_unsupported=counting_unsupported,
    )


__all__ = [
    "CONTEXT_FIELDS",
    "COUNTING_SOURCES",
    "FEATURE_SOURCES",
    "OFFLINE_SOURCE",
    "PRODUCTION_CADENCE",
    "PRODUCTION_COUNTING",
    "Cadence",
    "CountingMode",
    "Deployability",
    "ProjectedContext",
    "assess",
    "project_step",
    "unfilled_context_fields",
]

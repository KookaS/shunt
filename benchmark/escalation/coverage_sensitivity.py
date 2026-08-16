"""Re-score one swept cell on coverage-restricted model strata — is its separation an artifact?"""

# WHY THIS EXISTS. Per-step stamping coverage on this corpus is MODEL-CORRELATED: the stamped
# share runs from ~0.52 to ~0.94 by model, and stamping tracks capture date which tracks model.
# Every per-step number is therefore measured on a population whose model composition differs from
# the corpus's. A pre-registered falsifier asks the obvious question — if the poorly-covered models
# are what produce the separation, the separation is a coverage artifact — and no committed
# entrypoint answered it, so it was adjudicated only by a within-model/within-challenge proxy.
#
# WHY A THRESHOLD LADDER RATHER THAN ONE "FULLY COVERED" SUBSET. No model on this corpus reaches a
# stamped share of 1.0, so "the fully-stamped models" names an EMPTY set and any single cut point
# ("share >= 0.88", say) is a researcher degree of freedom chosen after seeing the data. The ladder
# removes the choice: the cut points ARE the observed per-model shares, so the strata are nested
# from all-models down to the single best-covered model, and the reader sees the whole trajectory
# instead of one hand-picked rung. If the separation is a coverage artifact it must DECAY along
# that ladder; if it survives to the strictest rung it is not one.
#
# WHAT IS RE-RUN, AND WHY THE WHOLE FAMILY. Each stratum re-runs the FULL swept grid, not the one
# cell, because the cell's gate is the family-wise (max-over-cells) null. Re-scoring the cell alone
# would compare a restricted observation against the unrestricted family's null — a different
# reference — and the point of the exercise is to judge the restricted cell inside ITS OWN null.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.escalation import features, policy_eval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.escalation import metrics, replay
    from benchmark.escalation.schema import Trajectory


@dataclass(frozen=True)
class CoverageStratum:
    """One nested model subset, and the swept cell re-scored inside it."""

    # The cut point: models whose per-step stamped share is at or above this are retained.
    min_stamped_share: float
    models: tuple[str, ...]
    n_stamped: int
    # Share of the all-models stratum's rows this one keeps — the power cost, stated rather than
    # left to be inferred from two counts.
    retained_share: float
    auroc: float
    # What a pure run-length predictor scores at this stratum's flag count. The same disclosure the
    # unrestricted cell carries: a restricted AUROC that only matches this is length selection.
    length_baseline_auroc: float | None
    null: metrics.NullResult
    null_familywise: metrics.NullResult | None
    null_length_stratified: metrics.NullResult | None
    # AUROC minus the all-models stratum's AUROC. 0.0 on the all-models row by construction.
    delta_auroc: float

    @property
    def clears_null(self) -> bool:
        """Whether the restricted cell still sits above the null the gate reads for it."""
        gate = self.null if self.null_familywise is None else self.null_familywise
        return gate.beats_null

    def to_dict(self) -> dict[str, object]:
        return {
            "min_stamped_share": round(self.min_stamped_share, 4),
            "models": list(self.models),
            "n_stamped": self.n_stamped,
            "retained_share": round(self.retained_share, 4),
            "auroc": round(self.auroc, 4),
            "delta_auroc_vs_all_models": round(self.delta_auroc, 4),
            "length_baseline_auroc": (
                None if self.length_baseline_auroc is None else round(self.length_baseline_auroc, 4)
            ),
            "null_auroc": self.null.to_dict(),
            "null_auroc_familywise": (
                None if self.null_familywise is None else self.null_familywise.to_dict()
            ),
            "null_auroc_length_stratified": (
                None
                if self.null_length_stratified is None
                else self.null_length_stratified.to_dict()
            ),
            "clears_null": self.clears_null,
        }


@dataclass(frozen=True)
class CoverageSensitivity:
    """The swept cell's AUROC along the nested coverage ladder, top row = every model."""

    counting: str
    escalate_after_n: int
    stale_window: int
    ladder: str
    stamped_share_by_model: dict[str, float]
    strata: tuple[CoverageStratum, ...]

    @property
    def strictest(self) -> CoverageStratum | None:
        """The smallest, best-covered stratum — the rung the falsifier is adjudicated on."""
        return self.strata[-1] if self.strata else None

    @property
    def survives_restriction(self) -> bool:
        """Whether EVERY rung of the ladder still clears its own null."""
        return bool(self.strata) and all(s.clears_null for s in self.strata)

    def to_dict(self) -> dict[str, object]:
        return {
            "counting": self.counting,
            "cell": {
                "escalate_after_n": self.escalate_after_n,
                "stale_window": self.stale_window,
                "ladder": self.ladder,
            },
            "stamped_share_by_model": {
                model: round(share, 4)
                for model, share in sorted(self.stamped_share_by_model.items())
            },
            "strata": [s.to_dict() for s in self.strata],
            "survives_restriction": self.survives_restriction,
        }


def stamped_share_by_model(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    """Per-model share of trajectories that carry per-step verified outcomes."""
    # Derived from the corpus, never a literal list: a rebuild that changes coverage moves the
    # strata with it instead of leaving a hardcoded model set describing an older corpus.
    return {
        c.model: (c.n_stamped / c.n_trajectories if c.n_trajectories else 0.0)
        for c in features.model_coverage(trajectories)
    }


def _cut_points(shares: dict[str, float]) -> tuple[float, ...]:
    """The nested ladder's cut points: every distinct observed share, loosest first."""
    return tuple(sorted({share for share in shares.values() if share > 0.0}))


def evaluate(
    trajectories: Sequence[Trajectory],
    grid: Sequence[replay.GridPoint],
    point: replay.GridPoint,
    *,
    n_permutations: int,
    count_from_first_edit: bool = True,
) -> CoverageSensitivity | None:
    """Re-score ``point`` on each nested coverage stratum; None when the corpus has no strata.

    Takes the WHOLE corpus: the shares that define the strata are a property of what was dropped,
    and the scoring then runs on the stamped rows exactly as the unrestricted cell does.
    """
    shares = stamped_share_by_model(trajectories)
    stamped = [t for t in trajectories if features.is_stamped(t)]
    cuts = _cut_points(shares)
    if not cuts or not stamped:
        return None
    rows: list[CoverageStratum] = []
    baseline: float | None = None
    for cut in cuts:
        models = tuple(sorted(m for m, share in shares.items() if share >= cut))
        subset = [t for t in stamped if features.model_of(t) in models]
        cell = _restricted_cell(subset, grid, point, n_permutations, count_from_first_edit)
        if cell is None:
            continue
        if baseline is None:
            baseline = cell.null_auroc.observed
        rows.append(_stratum(cut, models, subset, cell, len(stamped), baseline))
    return CoverageSensitivity(
        counting="edit_gated" if count_from_first_edit else "as_shipped",
        escalate_after_n=point.escalate_after_n,
        stale_window=point.stale_window,
        ladder=point.ladder,
        stamped_share_by_model=shares,
        strata=tuple(rows),
    )


def _restricted_cell(
    subset: Sequence[Trajectory],
    grid: Sequence[replay.GridPoint],
    point: replay.GridPoint,
    n_permutations: int,
    count_from_first_edit: bool,
) -> policy_eval.PolicyCell | None:
    """Replay the whole family on one stratum and return its cell for ``point``."""
    # A stratum with one outcome class carries no AUROC at all (every null draw is the same
    # degenerate number), so it is dropped rather than published as a 0.5 that means nothing.
    labels = {t.header.terminal_resolved for t in subset}
    if len(labels) < 2:
        return None
    cells = policy_eval.evaluate(
        subset, grid, n_permutations=n_permutations, count_from_first_edit=count_from_first_edit
    )
    return next(
        (
            c
            for c in cells
            if c.escalate_after_n == point.escalate_after_n
            and c.stale_window == point.stale_window
            and c.ladder == point.ladder
        ),
        None,
    )


def _stratum(
    cut: float,
    models: tuple[str, ...],
    subset: Sequence[Trajectory],
    cell: policy_eval.PolicyCell,
    n_all: int,
    baseline: float,
) -> CoverageStratum:
    """One published ladder row."""
    return CoverageStratum(
        min_stamped_share=cut,
        models=models,
        n_stamped=len(subset),
        retained_share=len(subset) / n_all if n_all else 0.0,
        auroc=cell.null_auroc.observed,
        length_baseline_auroc=cell.length_baseline_auroc,
        null=cell.null_auroc,
        null_familywise=cell.null_auroc_family,
        null_length_stratified=cell.null_auroc_length,
        delta_auroc=cell.null_auroc.observed - baseline,
    )


__all__ = [
    "CoverageSensitivity",
    "CoverageStratum",
    "evaluate",
    "stamped_share_by_model",
]

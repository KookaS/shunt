"""Guard against quoting a number scored on a coverage-selected subset as if it were the sample.

Answers one question for any scored subset: were these tasks chosen by coverage, and if so is
the leftover measurably easier or harder than what survived?
"""

# THE TRAP THIS EXISTS FOR. The collector is ADAPTIVE (`benchmark.runner.collect.derive_strata`):
# it runs the expensive tier only on the DISCRIMINATING set. So "tasks measured on every model"
# is not a random sample of the suite — it is the slice the cheap models disagreed on, which is
# the hard slice. Any strategy whose scorable set is gated on coverage is therefore scored on a
# difficulty-biased sample, and its cost and pass rate are not comparable to a strategy scored on
# the full sample. Nothing in the summary said so, and a verdict was published off that gap.
#
# WHY IT IS A GUARD RATHER THAN A CORRECTION. The bias cannot be divided out — the sampling
# probability of an unrun cell is not recorded per strategy. What CAN be done is refuse to let the
# number travel alone: every row carries whether it was subset-scored and, when it was, the
# measured difficulty gap between what it scored on and what it dropped.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Below this gap the two halves are not distinguishable enough to call the subset biased; the row
# still reports that it was subset-scored. In percentage points of the reference model's pass rate.
BIAS_THRESHOLD_PP: float = 5.0


@dataclass(frozen=True)
class SubsetSelection:
    """One strategy's scored set measured against the sample it was drawn from."""

    n_scored: int
    n_sample: int
    reference_model: str
    pass_rate_scored: float
    pass_rate_dropped: float
    n_dropped_measured: int

    @property
    def is_subset(self) -> bool:
        """True when the row was scored on fewer tasks than the sample it claims to describe."""
        return self.n_scored < self.n_sample

    @property
    def bias_pp(self) -> float:
        """Reference-model pass-rate gap (scored minus dropped) in percentage points."""
        return 100.0 * (self.pass_rate_scored - self.pass_rate_dropped)

    @property
    def biased(self) -> bool:
        """True when the dropped tasks are measurably easier or harder than the scored ones."""
        return (
            self.is_subset
            and self.n_dropped_measured > 0
            and abs(self.bias_pp) >= BIAS_THRESHOLD_PP
        )

    @property
    def note(self) -> str:
        """One line naming the selection, or empty when the row is scored on the full sample."""
        if not self.is_subset:
            return ""
        head = f"scored on {self.n_scored}/{self.n_sample} tasks selected by coverage"
        if self.n_dropped_measured == 0:
            return f"{head}; the dropped tasks carry no {self.reference_model} cell to compare"
        gap = (
            f"{self.reference_model} passes {self.pass_rate_scored * 100:.1f}% here vs "
            f"{self.pass_rate_dropped * 100:.1f}% on the {self.n_dropped_measured} dropped "
            f"({self.bias_pp:+.1f}pp)"
        )
        verdict = "difficulty-biased, not a random sample" if self.biased else "no measured gap"
        return f"{head}; {gap} — {verdict}"


def _pass_rate(
    task_ids: Iterable[str], results: Mapping[str, Mapping[str, dict]], model: str
) -> tuple[float, int]:
    """(pass rate, n measured) for ``model`` over the tasks that actually have its cell."""
    cells = [results.get(tid, {}).get(model) for tid in task_ids]
    measured = [c for c in cells if c]
    if not measured:
        return (0.0, 0)
    return (sum(1 for c in measured if c.get("pass", False)) / len(measured), len(measured))


def assess(
    scored: Iterable[str],
    sample: Iterable[str],
    matrix: Mapping[str, object],
    reference_model: str,
) -> SubsetSelection:
    """Measure a scored set against its sample using ``reference_model`` as the difficulty probe."""
    # The cheapest model is the probe because its pass rate IS the corpus's difficulty axis: the
    # collector's discriminating set is defined by cheap-model disagreement, so if selection
    # tracked difficulty it shows up here and nowhere cheaper to compute.
    scored_ids = list(scored)
    sample_ids = list(sample)
    dropped = [tid for tid in sample_ids if tid not in set(scored_ids)]
    raw = matrix.get("results")
    results: Mapping[str, Mapping[str, dict]] = raw if isinstance(raw, dict) else {}
    rate_in, _ = _pass_rate(scored_ids, results, reference_model)
    rate_out, n_out = _pass_rate(dropped, results, reference_model)
    return SubsetSelection(
        n_scored=len(scored_ids),
        n_sample=len(sample_ids),
        reference_model=reference_model,
        pass_rate_scored=rate_in,
        pass_rate_dropped=rate_out,
        n_dropped_measured=n_out,
    )


def reference_model(matrix: Mapping[str, object]) -> str:
    """The difficulty probe: the cheapest model the matrix prices, or '' when it prices none."""
    models = matrix.get("models", {})
    if not isinstance(models, dict) or not models:
        return ""
    return min(
        models,
        key=lambda m: (
            float(models[m].get("input_price", 0.0)) + float(models[m].get("output_price", 0.0)),
            m,
        ),
    )


def footer(notes: Mapping[str, str]) -> list[str]:
    """The block printed under a strategy table: one line per subset-scored strategy."""
    flagged = {name: note for name, note in notes.items() if note}
    if not flagged:
        return []
    lines = [
        "",
        "SUBSET-SELECTION WARNING — these rows are NOT scored on the full sample.",
        "  The collector runs the expensive tier only on the discriminating set, so a task's",
        "  coverage tracks its difficulty. A row below is not comparable to a full-sample row.",
    ]
    lines.extend(f"  {name:25} {note}" for name, note in sorted(flagged.items()))
    return lines


def rows_footer(rows: Iterable[Mapping[str, object]]) -> list[str]:
    """``footer`` over summary rows, read off the ``subset_note`` each row carries."""
    return footer({str(r.get("strategy", "?")): str(r.get("subset_note") or "") for r in rows})

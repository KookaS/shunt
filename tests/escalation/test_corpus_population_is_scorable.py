"""Every model in the analysed corpus must contribute at least one scorable run."""

# The defect this gate exists for: `data/live/` is BOTH the archive and the population every
# escalation statistic is computed over (`corpus.census` counts the files in it). A trajectory
# captured but never put through the stamp stage carries no confirmed steps, so it adds to every
# denominator and to no numerator. 41 such runs once entered the corpus and moved the published
# stamped share from 723/822 to 723/863 while the stamped count never changed — and the
# corpus-and-coverage figure's header read "7 models" while only 6 could be plotted.
#
# `features.is_stamped` already names the harm in its own docstring: an unstamped run hands the
# model a collection-date proxy, "a data artifact, not escalation signal". The existing census
# tripwire fires on any size change and reads as "update the docs to match", which invites
# bumping the number rather than asking whether the data belongs. This gate asks that question.
# Archive-only collections belong in a sibling directory (`data/probe/README.md`).

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from benchmark.escalation import features
from benchmark.escalation.corpus import LIVE_DIR
from benchmark.escalation.schema import load_jsonl


def _model_of(trajectory_id: str) -> str:
    """`<instance>__<model>__<arm>` — the model is the second-to-last field."""
    parts = trajectory_id.split("__")
    return parts[-2] if len(parts) >= 3 else trajectory_id


def _stamped_by_model(root: Path) -> tuple[dict[str, int], dict[str, int]]:
    """(trajectories, stamped trajectories) per model over a corpus directory."""
    totals: dict[str, int] = defaultdict(int)
    stamped: dict[str, int] = defaultdict(int)
    for path in sorted(root.glob("*.jsonl")):
        model = _model_of(path.stem)
        totals[model] += 1
        if features.is_stamped(load_jsonl(path)):
            stamped[model] += 1
    return dict(totals), dict(stamped)


def test_every_model_in_the_live_corpus_has_a_scorable_run() -> None:
    totals, stamped = _stamped_by_model(LIVE_DIR)
    assert totals, "the live corpus must hold at least one trajectory"
    # Non-vacuity: the predicate must actually discriminate, or an always-True `is_stamped`
    # would make this gate pass on the very corpus it exists to reject.
    assert any(stamped.get(m, 0) for m in totals), "no model is stamped — the predicate is broken"

    dead = sorted(m for m in totals if stamped.get(m, 0) == 0)
    assert not dead, (
        "these models contribute trajectories to the analysed corpus but ZERO scorable runs, "
        f"so they inflate every denominator and no numerator: {[(m, totals[m]) for m in dead]}. "
        "Either run the stamp stage on them (`python -m benchmark.pipeline --from stamp` — "
        "offline, no API spend, needs Docker and the instance images), or move them to an "
        "archive directory outside the analysed population, as data/probe/ does."
    )

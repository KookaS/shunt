"""Offline scoring of the degenerate-repetition trigger over stored trajectories.

Reads the committed escalation corpus, scores each trajectory by the longest run of
identical ``(action, args)`` pairs (the proxy's per-action content and tool-call
arguments), and reports AUROC, a length-stratified shuffled-label null, and per-N
depth/precision/volume operating points. Also builds the synthetic positive control
and its shuffled-label null so the instrument can be adjudicated before any verdict.
Pre-registered in ``benchmark/escalation/prereg/degenerate-repetition-detector.md``.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from benchmark.admissibility import admissibility_verdict
from benchmark.escalation import features, metrics, policy_eval
from benchmark.escalation.normalize.base import build_trajectory, make_step
from benchmark.escalation.schema import load_jsonl
from shunt.proxy.repetition import action_key, first_fire_index, longest_run

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from benchmark.escalation.schema import StepView, Trajectory

# Pre-registered decision constants (frozen before any scoring).
N_GRID: Final[tuple[int, ...]] = (2, 3, 4, 5, 6, 8, 10, 15, 20)
DEPTH_CUTOFF: Final[float] = 0.20
PRECISION_BAR: Final[float] = 0.63
VOLUME_FLOOR: Final[float] = 0.05
PRIMARY_MODEL: Final[str] = "deepseek-v4-flash"


def trajectory_keys(traj: Trajectory) -> list[str]:
    """The action key of every step, in order — the sequence the trigger counts runs over."""
    return [action_key(step.action, step.args) for step in traj.steps]


def repetition_score(traj: Trajectory) -> int:
    """The longest unbroken run of identical action keys on a trajectory."""
    return longest_run(trajectory_keys(traj))


def first_fire_depth(traj: Trajectory, n: int) -> float | None:
    """Where a threshold-N fire first lands, as ``(index + 1) / n_steps``, or None."""
    index = first_fire_index(trajectory_keys(traj), n)
    if index is None:
        return None
    return (index + 1) / traj.header.n_steps


def load_population(live_dir: Path, *, model: str = PRIMARY_MODEL) -> list[Trajectory]:
    """Stamped trajectories for one model from a directory of JSONL trajectories."""
    trajs = [load_jsonl(path) for path in sorted(live_dir.glob("*.jsonl"))]
    return [
        traj for traj in trajs if features.is_stamped(traj) and model in traj.header.trajectory_id
    ]


@dataclass(frozen=True)
class OperatingPoint:
    """One threshold N's fire count, depth distribution and shallow-depth precision."""

    n: int
    fires: int
    volume: float
    precision: float | None
    shallow_fires: int
    shallow_volume: float
    shallow_precision: float | None
    depth_min: float | None
    depth_median: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "fires": self.fires,
            "volume": _rounded(self.volume),
            "precision": _rounded(self.precision),
            "shallow_fires": self.shallow_fires,
            "shallow_volume": _rounded(self.shallow_volume),
            "shallow_precision": _rounded(self.shallow_precision),
            "depth_min": _rounded(self.depth_min),
            "depth_median": _rounded(self.depth_median),
        }


def operating_point(trajs: Sequence[Trajectory], n: int) -> OperatingPoint:
    """Score threshold *n* over a population: fires, overall precision, shallow slice."""
    fired = [
        (depth, not traj.header.terminal_resolved)
        for traj in trajs
        if (depth := first_fire_depth(traj, n)) is not None
    ]
    shallow = [doomed for depth, doomed in fired if depth <= DEPTH_CUTOFF]
    depths = [depth for depth, _ in fired]
    size = len(trajs)
    return OperatingPoint(
        n=n,
        fires=len(fired),
        volume=len(fired) / size if size else 0.0,
        precision=_precision(fired),
        shallow_fires=len(shallow),
        shallow_volume=len(shallow) / size if size else 0.0,
        shallow_precision=sum(shallow) / len(shallow) if shallow else None,
        depth_min=min(depths) if depths else None,
        depth_median=statistics.median(depths) if depths else None,
    )


def _precision(fired: Sequence[tuple[float, bool]]) -> float | None:
    return sum(doomed for _, doomed in fired) / len(fired) if fired else None


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


@dataclass(frozen=True)
class RepetitionCensus:
    """The repetition a population actually contains, per pre-registered N.

    Evidence for whether the N grid is testable at all: the per-trajectory longest-run
    histogram and the count of trajectories whose longest run reaches each N. An N no
    trajectory reaches cannot be exercised by the corpus, so its operating point is a
    coverage gap rather than a negative.
    """

    n: int
    longest_run_histogram: dict[int, int]
    reaching_n: dict[int, int]
    max_longest_run: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "longest_run_histogram": {
                str(run): count for run, count in sorted(self.longest_run_histogram.items())
            },
            "reaching_n": {str(n): count for n, count in self.reaching_n.items()},
            "max_longest_run": self.max_longest_run,
        }


def repetition_census(
    trajs: Sequence[Trajectory], n_grid: Sequence[int] = N_GRID
) -> RepetitionCensus:
    """Longest-run histogram and the count of trajectories reaching each pre-registered N."""
    runs = [repetition_score(traj) for traj in trajs]
    histogram: dict[int, int] = {}
    for run in runs:
        histogram[run] = histogram.get(run, 0) + 1
    return RepetitionCensus(
        n=len(runs),
        longest_run_histogram=histogram,
        reaching_n={n: sum(1 for run in runs if run >= n) for n in n_grid},
        max_longest_run=max(runs, default=0),
    )


def score_auroc_and_null(
    trajs: Sequence[Trajectory], *, n_permutations: int = 2000, seed: int = 0
) -> dict[str, object]:
    """Observed AUROC against its length-stratified null, plus the operating-point census."""
    scores = [float(repetition_score(traj)) for traj in trajs]
    labels = [not traj.header.terminal_resolved for traj in trajs]
    lengths = [traj.header.n_steps for traj in trajs]
    observed = metrics.auroc(scores, labels)
    null = policy_eval._length_stratified_null(
        scores, labels, lengths, n_permutations=n_permutations, seed=seed
    )
    return {
        "n": len(trajs),
        "base_failure_rate": _rounded(metrics.prevalence(labels)),
        "auroc": _rounded(observed),
        "null": null.to_dict(),
        "beats_length_null": null.beats_null,
        "census": repetition_census(trajs).to_dict(),
    }


def _step(index: int, action: str) -> StepView:
    return make_step(
        step_index=index,
        observation="",
        action=action,
        tool="bash",
        args=action,
        result="",
        metadata={},
    )


def _planted_run(challenge: str, arm: str, *, resolved: bool, loop_len: int) -> Trajectory:
    """A failed run repeats one action ``loop_len`` times; a resolved run never repeats."""
    actions = [f"cmd-{challenge}-{arm}-{i}" for i in range(4)]
    if not resolved:
        actions.extend([f"loop-{challenge}-{arm}"] * loop_len)
    else:
        actions.extend([f"done-{challenge}-{arm}-{i}" for i in range(2)])
    steps = [_step(i, action) for i, action in enumerate(actions)]
    return build_trajectory(
        steps,
        {"trajectory_id": arm, "terminal_resolved": resolved, "instance_id": challenge},
        "planted",
    )


def planted_corpus(*, n_challenges: int = 40, loop_len: int = 6) -> list[Trajectory]:
    """A synthetic corpus with a known action-repetition signal the detector must recover.

    Deterministic by construction: the signal is the repeated-action loop itself, so no
    RNG is needed. ``1 + (c % 3)`` fail arms and ``1 + ((c + 1) % 3)`` resolved arms per
    challenge keep the corpus non-degenerate while staying reproducible.
    """
    out: list[Trajectory] = []
    for c in range(n_challenges):
        challenge = f"challenge-{c}"
        for arm in range(1 + c % 3):
            out.append(
                _planted_run(
                    challenge, f"{challenge}-fail-{arm}", resolved=False, loop_len=loop_len
                )
            )
        for arm in range(1 + (c + 1) % 3):
            out.append(
                _planted_run(challenge, f"{challenge}-ok-{arm}", resolved=True, loop_len=loop_len)
            )
    return out


def shuffle_labels_within_challenge(
    corpus: Iterable[Trajectory], *, seed: int = 0
) -> list[Trajectory]:
    """The same trajectories with terminal labels permuted within each challenge."""
    by_group: dict[str, list[Trajectory]] = {}
    for traj in corpus:
        by_group.setdefault(traj.header.instance_id or traj.header.trajectory_id, []).append(traj)
    rng = random.Random(seed)
    out: list[Trajectory] = []
    for runs in by_group.values():
        labels = [run.header.terminal_resolved for run in runs]
        rng.shuffle(labels)
        for run, label in zip(runs, labels, strict=True):
            out.append(replace(run, header=replace(run.header, terminal_resolved=label)))
    return out


def control_report(
    corpus: Sequence[Trajectory], *, n_permutations: int = 2000, seed: int = 0
) -> dict[str, object]:
    """Positive control and shuffled-label null for the assembled detector on *corpus*."""
    scores = [float(repetition_score(traj)) for traj in corpus]
    labels = [not traj.header.terminal_resolved for traj in corpus]
    observed = metrics.auroc(scores, labels)
    groups = [traj.header.instance_id or traj.header.trajectory_id for traj in corpus]
    rng = random.Random(seed)
    draws = [
        metrics.auroc(
            scores,
            _shuffled_labels(labels, groups, rng),
        )
        for _ in range(n_permutations)
    ]
    null = metrics.permutation_null(observed, draws)
    chance_band = null.ci_high - 0.5
    verdict = admissibility_verdict(observed, null.mean, chance_level=0.5, chance_band=chance_band)
    return {
        "positive_score": _rounded(observed),
        "shuffled_score": _rounded(null.mean),
        "chance_level": 0.5,
        "chance_band": _rounded(chance_band),
        "null": null.to_dict(),
        "admissible": verdict.admissible,
        "positive_passed": verdict.positive_passed,
        "null_at_chance": verdict.null_at_chance,
        "reason": verdict.reason,
    }


def _shuffled_labels(
    labels: Sequence[bool], groups: Sequence[str], rng: random.Random
) -> list[bool]:
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    out = list(labels)
    for indices in by_group.values():
        block = [labels[i] for i in indices]
        rng.shuffle(block)
        for index, value in zip(indices, block, strict=True):
            out[index] = value
    return out


def operating_points(
    trajs: Sequence[Trajectory], n_grid: Sequence[int] = N_GRID
) -> list[OperatingPoint]:
    """One OperatingPoint per threshold in *n_grid*."""
    return [operating_point(trajs, n) for n in n_grid]


def target_met(points: Sequence[OperatingPoint]) -> OperatingPoint | None:
    """The first pre-registered N meeting precision and volume at depth <= cutoff, or None."""
    for point in points:
        if (
            point.shallow_precision is not None
            and point.shallow_precision >= PRECISION_BAR
            and point.shallow_volume >= VOLUME_FLOOR
        ):
            return point
    return None


__all__ = [
    "DEPTH_CUTOFF",
    "N_GRID",
    "PRECISION_BAR",
    "PRIMARY_MODEL",
    "VOLUME_FLOOR",
    "OperatingPoint",
    "RepetitionCensus",
    "control_report",
    "first_fire_depth",
    "load_population",
    "operating_point",
    "operating_points",
    "planted_corpus",
    "repetition_census",
    "repetition_score",
    "score_auroc_and_null",
    "shuffle_labels_within_challenge",
    "target_met",
    "trajectory_keys",
]

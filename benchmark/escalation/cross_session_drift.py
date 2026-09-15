"""Offline scoring of the cross-session drift detector over the committed corpus.

Reads the escalation live plane, builds one session-close summary per trajectory, and scores
the K-window drift level of the sessions preceding each next session against that session's
terminal failure. Reports AUROC against a session-count-stratified shuffled-label null, the
break-even fire volume and precision, and — before any verdict — the planted positive control
and its shuffled-label null so the instrument can be adjudicated by the shared gate.
Pre-registered in ``benchmark/escalation/prereg/cross-session-drift-detector.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from benchmark.admissibility import admissibility_verdict
from benchmark.escalation import features, metrics
from shunt.proxy.session_drift import (
    MAX_WINDOW,
    MIN_WINDOW,
    WINDOW_GRID,
    DriftFeatures,
    SessionCounters,
    summarize,
    window_features,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.escalation.schema import Trajectory

# Pre-registered decision constants (frozen before any scoring).
AUROC_BAR: Final[float] = 0.60
PRECISION_FLOOR: Final[float] = 0.50
VOLUME_FLOOR: Final[float] = 0.05
N_PERMUTATIONS: Final[int] = 2000
SEED: Final[int] = 0
PRIMARY_AXIS: Final[str] = "drift_level"
SECONDARY_AXIS: Final[str] = "drift_slope"
_LENGTH_BINS: Final[int] = 10

# The planted control's per-session counter level for a "drifting" session, and the fixed
# step count every synthetic session carries so rates are comparable.
_PLANTED_STEPS: Final[int] = 10
# Priors per planted episode: far longer than MAX_WINDOW so almost every scored window is
# drawn from a single episode (only the K windows straddling the episode seam are mixed).
_PLANTED_PRIORS: Final[int] = 40
_PLANTED_DRIFT: Final[SessionCounters] = SessionCounters(
    is_reverts=3, retry_total=3, loop_signals=3, wire_tool_errors=3, n_steps=_PLANTED_STEPS
)
_PLANTED_CALM: Final[SessionCounters] = SessionCounters(
    is_reverts=0, retry_total=0, loop_signals=0, wire_tool_errors=0, n_steps=_PLANTED_STEPS
)


@dataclass(frozen=True)
class SessionSample:
    """One closed session's repo, identity, counter summary and terminal-failure label."""

    repo: str
    session_id: str
    counters: SessionCounters
    failed: bool


@dataclass(frozen=True)
class ScoredWindow:
    """One next-session decision: its same-repo window, the drift features and the label."""

    repo: str
    session_id: str
    window: int
    n_prior: int
    features: DriftFeatures
    failed: bool


def repo_of(traj: Trajectory) -> str:
    """The upstream repo key: the ``__``-prefix of the instance id, else of the trajectory id."""
    raw = traj.header.instance_id or traj.header.trajectory_id
    return raw.split("__", 1)[0]


def session_counters(traj: Trajectory, *, companion: bool = False) -> SessionCounters:
    """The four whitelisted counters for one trajectory; ``companion`` maps tool errors.

    The default reads the live ``WIRE_TOOL_ERROR_COUNT`` (absent offline, so 0). The labelled
    companion substitutes the real per-step ``status == "error"`` count instead — see the
    pre-registration §3c; it is never part of the primary measurement.
    """
    wire_errors = sum(1 for step in traj.steps if step.status == "error") if companion else 0
    return summarize(traj.steps, wire_tool_errors=wire_errors)


def samples_from_corpus(
    trajs: Sequence[Trajectory], *, companion: bool = False
) -> list[SessionSample]:
    """One ``SessionSample`` per trajectory, in corpus order."""
    return [
        SessionSample(
            repo=repo_of(traj),
            session_id=traj.header.trajectory_id,
            counters=session_counters(traj, companion=companion),
            failed=not traj.header.terminal_resolved,
        )
        for traj in trajs
    ]


def load_population(live_dir: Path) -> list[Trajectory]:
    """Every stamped trajectory under *live_dir* (the population the drift features score)."""
    from benchmark.escalation.schema import load_jsonl  # noqa: PLC0415

    trajs = [load_jsonl(path) for path in sorted(live_dir.glob("*.jsonl"))]
    return [traj for traj in trajs if features.is_stamped(traj)]


def scored_windows(samples: Sequence[SessionSample], k: int) -> list[ScoredWindow]:
    """Every next-session decision with at least *k* prior same-repo sessions (rolling)."""
    _check_k(k)
    by_repo: dict[str, list[SessionSample]] = {}
    for sample in samples:
        by_repo.setdefault(sample.repo, []).append(sample)
    out: list[ScoredWindow] = []
    for repo, sessions in by_repo.items():
        ordered = sorted(sessions, key=lambda sample: sample.session_id)
        for index in range(k, len(ordered)):
            window = ordered[index - k : index]
            out.append(
                ScoredWindow(
                    repo=repo,
                    session_id=ordered[index].session_id,
                    window=k,
                    n_prior=index,
                    features=window_features([sample.counters for sample in window]),
                    failed=ordered[index].failed,
                )
            )
    return out


def _check_k(k: int) -> None:
    if k < MIN_WINDOW or k > MAX_WINDOW:
        raise ValueError(f"k must be in [{MIN_WINDOW}, {MAX_WINDOW}], got {k}")


def axis_scores(windows: Sequence[ScoredWindow], axis: str) -> list[float]:
    """The chosen drift feature as a score vector, one entry per scored window."""
    if axis not in (PRIMARY_AXIS, SECONDARY_AXIS):
        raise ValueError(f"unknown axis {axis!r}")
    return [float(getattr(window.features, axis)) for window in windows]


def _count_cells(windows: Sequence[ScoredWindow]) -> list[tuple[str, int]]:
    """The stratification cell of each row: repo × session-count bin (10 bins)."""
    return [(window.repo, min(window.n_prior, _LENGTH_BINS)) for window in windows]


def _permute_within_cells(
    labels: Sequence[bool], cells: Sequence[tuple[str, int]], rng: random.Random
) -> list[bool]:
    """Shuffle labels inside each equal-count cell, so every cell's failure rate is fixed."""
    by_cell: dict[tuple[str, int], list[int]] = {}
    for index, cell in enumerate(cells):
        by_cell.setdefault(cell, []).append(index)
    out = list(labels)
    for indices in by_cell.values():
        block = [labels[i] for i in indices]
        rng.shuffle(block)
        for index, value in zip(indices, block, strict=True):
            out[index] = value
    return out


def stratified_null(
    windows: Sequence[ScoredWindow],
    axis: str,
    *,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> metrics.NullResult:
    """The AUROC null under within-(repo, session-count) label shuffles."""
    scores = axis_scores(windows, axis)
    labels = [window.failed for window in windows]
    cells = _count_cells(windows)
    rng = random.Random(seed)
    draws = [
        metrics.auroc(scores, _permute_within_cells(labels, cells, rng))
        for _ in range(n_permutations)
    ]
    return metrics.permutation_null(metrics.auroc(scores, labels), draws)


def fire_report(windows: Sequence[ScoredWindow], axis: str) -> dict[str, object]:
    """Break-even flag volume and precision for one drift axis."""
    scores = axis_scores(windows, axis)
    labels = [window.failed for window in windows]
    base = metrics.prevalence(labels)
    budget = max(1, min(len(scores), round(base * len(scores))))
    cut = sorted(scores, reverse=True)[budget - 1]
    fired = [index for index, score in enumerate(scores) if score >= cut]
    precision = sum(labels[i] for i in fired) / len(fired) if fired else None
    return {
        "axis": axis,
        "n": len(windows),
        "base_failure_rate": round(base, 6),
        "flag_budget": budget,
        "cut": round(cut, 6),
        "fires": len(fired),
        "volume": round(len(fired) / len(scores), 6) if scores else 0.0,
        "precision": None if precision is None else round(precision, 6),
    }


def score_report(windows: Sequence[ScoredWindow], axis: str) -> dict[str, object]:
    """Observed AUROC with its stratified null plus the break-even fire row."""
    null = stratified_null(windows, axis)
    return {
        "n": len(windows),
        "auroc": round(null.observed, 6),
        "null": null.to_dict(),
        "beats_null": null.beats_null,
        "fire": fire_report(windows, axis),
    }


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def target_met(report: dict[str, object]) -> bool:
    """Whether a report clears the pre-registered AUROC, null, volume and precision bars."""
    fire = report.get("fire")
    if not isinstance(fire, dict):
        return False
    precision = fire.get("precision")
    return (
        _as_float(report.get("auroc")) >= AUROC_BAR
        and report.get("beats_null") is True
        and precision is not None
        and _as_float(precision) >= PRECISION_FLOOR
        and _as_float(fire.get("volume")) >= VOLUME_FLOOR
    )


def _planted_repo_seed(index: int, seed: int) -> int:
    return seed * 1_000_003 + index


def planted_corpus(*, n_repos: int = 120, seed: int = SEED) -> list[SessionSample]:
    """A synthetic corpus with a known between-repo drift signal the detector must recover.

    Each repo holds two episodes in random order: a drifting episode (``_PLANTED_PRIORS``
    high-counter sessions and a closing session, all labelled failed) and a calm episode
    (flat-zero sessions, all labelled resolved). An episode is longer than any K in the
    grid, so almost every scored window is drawn from one episode's level.
    """
    out: list[SessionSample] = []
    for repo_index in range(n_repos):
        rng = random.Random(_planted_repo_seed(repo_index, seed))
        labels = [True, False]
        rng.shuffle(labels)
        repo = f"planted-{repo_index:04d}"
        for episode, failed in enumerate(labels):
            level = _PLANTED_DRIFT if failed else _PLANTED_CALM
            for prior in range(_PLANTED_PRIORS):
                out.append(
                    SessionSample(repo, f"{repo}-{episode}-p{prior:02d}", level, failed=failed)
                )
            out.append(SessionSample(repo, f"{repo}-{episode}-target", level, failed=failed))
    return out


def shuffle_labels_within_repo(
    samples: Sequence[SessionSample], *, seed: int = SEED
) -> list[SessionSample]:
    """The same samples with terminal labels permuted within each repo."""
    by_repo: dict[str, list[SessionSample]] = {}
    for sample in samples:
        by_repo.setdefault(sample.repo, []).append(sample)
    rng = random.Random(seed)
    out: list[SessionSample] = []
    for sessions in by_repo.values():
        labels = [sample.failed for sample in sessions]
        rng.shuffle(labels)
        for sample, label in zip(sessions, labels, strict=True):
            out.append(replace(sample, failed=label))
    return out


def control_report(
    samples: Sequence[SessionSample],
    *,
    k: int = 5,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> dict[str, object]:
    """Positive control and shuffled-label null for the assembled pipeline on *samples*."""
    windows = scored_windows(samples, k)
    scores = axis_scores(windows, PRIMARY_AXIS)
    labels = [window.failed for window in windows]
    observed = metrics.auroc(scores, labels)
    rng = random.Random(seed)
    draws = [
        metrics.auroc(
            scores,
            _permute_within_repo_labels(labels, [w.repo for w in windows], rng),
        )
        for _ in range(n_permutations)
    ]
    null = metrics.permutation_null(observed, draws)
    chance_band = null.ci_high - 0.5
    verdict = admissibility_verdict(observed, null.mean, chance_level=0.5, chance_band=chance_band)
    return {
        "k": k,
        "positive_score": round(observed, 6),
        "shuffled_score": round(null.mean, 6),
        "chance_level": 0.5,
        "chance_band": round(chance_band, 6),
        "null": null.to_dict(),
        "admissible": verdict.admissible,
        "positive_passed": verdict.positive_passed,
        "null_at_chance": verdict.null_at_chance,
        "reason": verdict.reason,
    }


def _permute_within_repo_labels(
    labels: Sequence[bool], repos: Sequence[str], rng: random.Random
) -> list[bool]:
    by_repo: dict[str, list[int]] = {}
    for index, repo in enumerate(repos):
        by_repo.setdefault(repo, []).append(index)
    out = list(labels)
    for indices in by_repo.values():
        block = [labels[i] for i in indices]
        rng.shuffle(block)
        for index, value in zip(indices, block, strict=True):
            out[index] = value
    return out


def k_grid_reports(
    samples: Sequence[SessionSample], axis: str = PRIMARY_AXIS
) -> list[dict[str, object]]:
    """One score report per K in the pre-registered grid."""
    return [score_report(scored_windows(samples, k), axis) for k in WINDOW_GRID]


def companion_report(trajs: Sequence[Trajectory], axis: str = PRIMARY_AXIS) -> dict[str, object]:
    """The pre-registered coverage-companion (section 3c) over the K grid, with its null.

    The primary whitelisted counters are dead on the committed corpus, so the same
    assembled pipeline is re-scored with the real per-step ``status == "error"`` count
    substituted for the missing wire counter. It can never satisfy the primary reach
    target; it exists so the absence of a usable signal is measured, not inferred.
    """
    samples = samples_from_corpus(trajs, companion=True)
    return {
        "axis": axis,
        "substitution": "wire_tool_error_count <- count(status == 'error')",
        "per_k": [{"k": k, **score_report(scored_windows(samples, k), axis)} for k in WINDOW_GRID],
    }


__all__ = [
    "AUROC_BAR",
    "N_PERMUTATIONS",
    "PRIMARY_AXIS",
    "PRECISION_FLOOR",
    "SECONDARY_AXIS",
    "SEED",
    "VOLUME_FLOOR",
    "ScoredWindow",
    "SessionSample",
    "axis_scores",
    "companion_report",
    "control_report",
    "fire_report",
    "k_grid_reports",
    "load_population",
    "planted_corpus",
    "repo_of",
    "samples_from_corpus",
    "score_report",
    "scored_windows",
    "session_counters",
    "shuffle_labels_within_repo",
    "stratified_null",
    "target_met",
]

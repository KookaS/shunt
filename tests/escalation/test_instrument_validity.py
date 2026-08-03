"""The R0 instrument-validity gate for the escalation policy sweep.

Tripwire for evals whose headline is quoted without a positive control or a
shuffled-label null; the reference is the pipeline's own empirical null band.
"""

from __future__ import annotations

import dataclasses
import random

from benchmark.escalation import datasets, metrics, policy_eval
from benchmark.escalation.schema import Trajectory
from tests.escalation.factories import make_step, make_trajectory

_PERMUTATIONS = metrics.MIN_PERMUTATIONS


def _recurrent(
    tid: str,
    *,
    resolved: bool,
    key: str | None,
    length: int = 12,
    challenge: str | None = None,
) -> Trajectory:
    """A run that repeats one failing check id every step, so the recurrence trigger fires."""
    steps = [
        make_step(step_index=i, decision_index=i, success=False, failing_check_id=key)
        for i in range(length - 1)
    ]
    steps.append(
        make_step(
            step_index=length - 1,
            decision_index=length - 1,
            success=resolved,
            action="submit",
        )
    )
    traj = make_trajectory(steps, trajectory_id=tid, terminal_resolved=resolved)
    if challenge is not None:
        header = dataclasses.replace(traj.header, instance_id=challenge)
        traj = dataclasses.replace(traj, header=header)
    return traj


def _recurrent_noisy(
    tid: str,
    *,
    resolved: bool,
    carrier: bool,
    key: str,
    length: int = 12,
    challenge: str | None = None,
) -> Trajectory:
    """A carrier run repeats the key on ~70% of steps; a non-carrier carries no key."""
    if carrier:
        carry = round(0.7 * (length - 1))
        # The residual steps are unkeyed FAILURES, never verified passes: a pass clears the
        # escalation log, which would un-fire a run meant to recur.
        steps = [
            make_step(
                step_index=i,
                decision_index=i,
                success=False,
                failing_check_id=key if i < carry else None,
            )
            for i in range(length - 1)
        ]
    else:
        steps = [make_step(step_index=i, decision_index=i) for i in range(length - 1)]
    steps.append(
        make_step(
            step_index=length - 1,
            decision_index=length - 1,
            success=resolved,
            action="submit",
        )
    )
    traj = make_trajectory(steps, trajectory_id=tid, terminal_resolved=resolved)
    if challenge is not None:
        header = dataclasses.replace(traj.header, instance_id=challenge)
        traj = dataclasses.replace(traj, header=header)
    return traj


def _planted_corpus() -> list[Trajectory]:
    """A balanced corpus with a KNOWN-learnable escalation signal.

    Failed runs repeat one per-challenge key every step; resolved runs carry none.
    """
    rng = random.Random(0)
    keys = ["pkg::a", "pkg::b", "pkg::c", "pkg::d", "pkg::e"]
    out: list[Trajectory] = []
    for c in range(24):
        n_fail = rng.randint(1, 3)
        n_ok = rng.randint(1, 3)
        key = keys[c % len(keys)]
        for arm in range(n_fail):
            out.append(
                _recurrent(
                    f"c{c}__fail{arm}",
                    resolved=False,
                    key=key,
                    challenge=f"challenge{c}",
                )
            )
        for arm in range(n_ok):
            out.append(
                _recurrent(
                    f"c{c}__ok{arm}",
                    resolved=True,
                    key=None,
                    challenge=f"challenge{c}",
                )
            )
    return out


def _noisy_corpus() -> list[Trajectory]:
    """A corpus whose fired<->failed link is real but imperfect — not a perfect 1.0.

    Failed runs are stuck on their per-challenge key with probability 0.70 and resolved runs
    spuriously carry it with probability 0.40, so the best cell's AUROC lands near 0.662.
    """
    rng = random.Random(0)
    keys = ["pkg::a", "pkg::b", "pkg::c", "pkg::d", "pkg::e"]
    out: list[Trajectory] = []
    for c in range(80):
        n_fail = rng.randint(1, 3)
        n_ok = rng.randint(1, 3)
        key = keys[c % len(keys)]
        for arm in range(n_fail):
            out.append(
                _recurrent_noisy(
                    f"c{c}__fail{arm}",
                    resolved=False,
                    carrier=rng.random() < 0.70,
                    key=key,
                    challenge=f"challenge{c}",
                )
            )
        for arm in range(n_ok):
            out.append(
                _recurrent_noisy(
                    f"c{c}__ok{arm}",
                    resolved=True,
                    carrier=rng.random() < 0.40,
                    key=key,
                    challenge=f"challenge{c}",
                )
            )
    return out


def _shuffled(corpus: list[Trajectory]) -> list[Trajectory]:
    """The SAME trajectories byte-for-byte, labels permuted within each challenge."""
    rng = random.Random(0)
    by_group: dict[str, list[Trajectory]] = {}
    for traj in corpus:
        by_group.setdefault(traj.header.instance_id, []).append(traj)
    out: list[Trajectory] = []
    for runs in by_group.values():
        labels = [r.header.terminal_resolved for r in runs]
        rng.shuffle(labels)
        for run, lab in zip(runs, labels, strict=True):
            header = dataclasses.replace(run.header, terminal_resolved=lab)
            out.append(dataclasses.replace(run, header=header))
    return out


def test_policy_sweep_clears_the_admissibility_gate() -> None:
    """The assembled sweep recovers a planted signal and collapses on shuffled labels."""
    # Positive control: the planted corpus's best fired cell must clear the pipeline's
    # OWN empirical chance band — the shuffled corpus's family-wise null 97.5th pct —
    # by the same strict `>` the admissibility gate encodes (never a fixed tolerance).
    positive_cells = policy_eval.evaluate(
        _planted_corpus(), datasets.DEFAULT_GRID, n_permutations=_PERMUTATIONS
    )
    scored = [c for c in positive_cells if c.precision is not None]
    best = max(scored, key=lambda c: c.null_auroc.observed)
    positive_score = best.null_auroc.observed

    # Shuffled-label null: labels permuted within each challenge destroy the
    # fired<->outcome link, so the max observed AUROC must sit inside the chance band.
    null_cells = policy_eval.evaluate(
        _shuffled(_planted_corpus()), datasets.DEFAULT_GRID, n_permutations=_PERMUTATIONS
    )
    scored_null = [c for c in null_cells if c.precision is not None]
    shuffled_score = max((c.null_auroc.observed for c in scored_null), default=0.5)
    null_band = max((c.gate_null.ci_high for c in null_cells), default=0.5)

    # The family-wise null on the planted corpus must itself clear — the gate the
    # report actually reads, not just a raw AUROC comparison.
    assert best.gate_null.beats_null
    # Positive control clears the empirical chance band (the shuffled corpus's own
    # family-wise null 97.5th pct), mirroring the shared gate's `> chance + band`.
    assert positive_score > null_band
    # Shuffled-label null collapses to chance: inside [1 - band, band] of 0.5.
    assert shuffled_score <= null_band
    assert shuffled_score >= 1.0 - null_band


def test_policy_sweep_has_power_at_the_production_effect_size() -> None:
    """The R0 positive control proves POWER at AUROC ~0.65, not only detectability at 1.0."""
    # The perfect-signal control above proves the instrument CAN detect a planted signal; it does
    # not prove power at the strength the production claim rests on (AUROC 0.662). Here the link is
    # imperfect — 70% of failed runs fire, 40% of resolved runs fire spuriously — so the recovered
    # AUROC must land in the [0.55, 0.80] band, not at 1.0, while still clearing the pipeline's own
    # empirical chance band.
    noisy = _noisy_corpus()
    cells = policy_eval.evaluate(noisy, datasets.DEFAULT_GRID, n_permutations=_PERMUTATIONS)
    scored = [c for c in cells if c.precision is not None]
    best = max(scored, key=lambda c: c.null_auroc.observed)
    recovered = best.null_auroc.observed

    null_cells = policy_eval.evaluate(
        _shuffled(noisy), datasets.DEFAULT_GRID, n_permutations=_PERMUTATIONS
    )
    scored_null = [c for c in null_cells if c.precision is not None]
    shuffled_score = max((c.null_auroc.observed for c in scored_null), default=0.5)
    null_band = max((c.gate_null.ci_high for c in null_cells), default=0.5)

    # Power at the realistic effect size: the best cell clears the family-wise null computed on its
    # OWN corpus, and the recovered AUROC sits in the production claim's band, not at 1.0.
    assert best.gate_null.beats_null
    assert recovered > null_band
    assert 0.55 <= recovered <= 0.80
    # Labels permuted within each challenge destroy the imperfect link too: no cell may clear.
    assert shuffled_score <= null_band
    assert shuffled_score >= 1.0 - null_band
    assert not any(c.gate_null.beats_null for c in scored_null)

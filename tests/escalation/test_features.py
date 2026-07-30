"""The two structural anti-leak walls: no future length, no terminal step.

Both encode defects that were shipped and measured. They are regression walls, not coverage: if
either assertion can be deleted without another test failing, the defect class can return.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from benchmark.escalation import features, prefix_eval
from benchmark.escalation.schema import Trajectory, load_jsonl
from tests.escalation.factories import make_step, make_trajectory

_LIVE_DIR = Path(__file__).resolve().parents[2] / "benchmark/escalation/data/live"


@lru_cache(maxsize=1)
def _live_corpus() -> tuple[Trajectory, ...]:
    """The committed live trajectories — the only corpus the real dependencies show up in."""
    return tuple(load_jsonl(p) for p in sorted(_LIVE_DIR.glob("*.jsonl")))


def _run(n_steps: int, *, resolved: bool, action: str = "act", tid: str = "i__m__e"):  # type: ignore[no-untyped-def]
    """A trajectory whose non-terminal steps are all identical, terminal stamped with the grade."""
    steps = [make_step(step_index=i, decision_index=i, action=action) for i in range(n_steps - 1)]
    terminal = make_step(
        step_index=n_steps - 1, decision_index=n_steps - 1, success=resolved, action="submit"
    )
    return make_trajectory([*steps, terminal], trajectory_id=tid, terminal_resolved=resolved)


def test_features_are_independent_of_future_trajectory_length() -> None:
    # WALL 1. The old label was positional, so a content-free clock (score = t/n) scored AUROC
    # 0.970 while a perfect task oracle capped at 0.757. Features read a FIXED absolute depth, so
    # two runs sharing a prefix are indistinguishable however long they eventually get. No learner
    # on these features can implement a clock, because the clock's input is not in them.
    short = _run(8, resolved=True)
    long = _run(60, resolved=True)
    assert features.extract_features(short, 5) == features.extract_features(long, 5)


def test_features_never_read_the_label_stamped_terminal_step() -> None:
    # WALL 2. `_stamp_terminal` writes the harness verdict onto the last step: AUROC(last step
    # failed -> terminal failure) is 1.00 including it and 0.56 excluding it. A prefix that could
    # reach step n-1 would read the answer verbatim.
    failed = _run(6, resolved=False)
    resolved = _run(6, resolved=True)
    assert failed.steps[-1].success != resolved.steps[-1].success  # the leak exists in the data
    assert features.extract_features(failed, 5) == features.extract_features(resolved, 5)


def test_prefix_stops_short_of_the_terminal_step() -> None:
    traj = _run(6, resolved=False)
    assert len(features.scorable_steps(traj)) == len(traj.steps) - 1
    # depth 5 needs 5 non-terminal steps; a 6-step run has exactly 5, a 5-step run has 4.
    assert features.extract_features(traj, 5) is not None
    assert features.extract_features(_run(5, resolved=False), 5) is None


def test_features_move_when_the_prefix_itself_differs() -> None:
    # The walls must not be a constant function: real prefix evidence still has to register.
    quiet = make_trajectory(
        [make_step(step_index=i, decision_index=i) for i in range(6)], terminal_resolved=True
    )
    thrashing = make_trajectory(
        [
            make_step(step_index=i, decision_index=i, success=False, failing_check_id="pkg::t")
            for i in range(6)
        ],
        terminal_resolved=True,
    )
    assert features.extract_features(quiet, 5) != features.extract_features(thrashing, 5)


@pytest.mark.parametrize("depth", features.DEFAULT_DEPTHS)
def test_the_design_matrix_has_full_column_rank_on_the_real_corpus(depth: int) -> None:
    # THE GUARD THIS REPLACED asserted `columns[a] != columns[b]` — exact vector inequality — which
    # only catches bit-identical duplication. Affine dependence passes it by construction, and two
    # affine dependencies were shipped behind it: `nonzero_exit_rate == fail_rate + infra_rate`
    # (exact over every depth-5 and depth-10 prefix step) and
    # `max_action_repeat_rate + distinct_action_rate == 1.2` at depth 5 (every row). The design
    # ranked 7 of 8 at depths 5 and 10 while the module claimed eight independent columns.
    #
    # RANK, not pairwise inequality, is the property that matters: a column that adds no rank adds
    # no information to the fit while still splitting the L2 logistic's weight. It is taken over
    # [features | intercept] because the model fits an intercept, so a column that is constant, or
    # an affine (not merely linear) combination of others, is redundant to THAT design.
    #
    # Run on the REAL corpus, not a synthetic fixture: both dependencies are properties of how the
    # stamping stage populates fields, and a 12-trajectory hand-built corpus reproduces neither —
    # which is exactly how they survived. Loading it costs under a second.
    trajectories = [t for t in _live_corpus() if features.is_stamped(t)]
    rows = features.build_rows(trajectories, depth)
    assert len(rows) >= prefix_eval.MIN_ROWS, f"depth {depth} is not estimable on this corpus"
    design = np.column_stack(
        [np.asarray([r.features for r in rows], dtype=float), np.ones(len(rows))]
    )
    rank = int(np.linalg.matrix_rank(design))
    assert rank == design.shape[1], (
        f"depth {depth}: design matrix ranks {rank} of {design.shape[1]} columns "
        f"({', '.join(features.FEATURE_NAMES)} + intercept) — some feature is an affine "
        "combination of the others and carries no information the fit can use"
    )


def test_the_aliased_and_constant_features_are_gone() -> None:
    # Named explicitly so re-adding one is a deliberate act, not an accident: the first two are
    # aliases of fail_rate, the third tracked stamping coverage rather than agent behaviour, and
    # the last two were affine combinations of columns that are kept (see the census in features).
    for dropped in (
        "blocking_rate",
        "check_id_rate",
        "missing_exit_rate",
        "nonzero_exit_rate",
        "distinct_action_rate",
    ):
        assert dropped not in features.FEATURE_NAMES


def test_unstamped_trajectories_are_identified() -> None:
    # A run the stamping stage never touched carries parser defaults, not evidence. Including it
    # hands the model a collection-date proxy, so it must be detectable and excludable.
    stamped = make_trajectory(
        [make_step(step_index=i, decision_index=i, confirmed=True) for i in range(4)],
        terminal_resolved=True,
    )
    unstamped = make_trajectory(
        [make_step(step_index=i, decision_index=i, confirmed=False) for i in range(3)]
        + [make_step(step_index=3, decision_index=3, confirmed=True)],
        terminal_resolved=True,
    )
    assert features.is_stamped(stamped)
    assert not features.is_stamped(unstamped)  # only the terminal step is confirmed


def test_model_and_group_are_read_from_identity() -> None:
    traj = _run(6, resolved=True, tid="astropy__astropy-12907__deepseek-v4-flash__high")
    assert features.model_of(traj) == "deepseek-v4-flash"
    assert features.group_of(traj) == "astropy__astropy-12907__deepseek-v4-flash__high"


def test_model_coverage_reports_zero_capture_models() -> None:
    covered = make_trajectory(
        [
            make_step(step_index=i, decision_index=i, success=False, failing_check_id="pkg::t")
            for i in range(4)
        ],
        trajectory_id="inst__good-model__high",
        terminal_resolved=False,
    )
    blind = make_trajectory(
        [make_step(step_index=i, decision_index=i, confirmed=False) for i in range(4)],
        trajectory_id="inst__blind-model__high",
        terminal_resolved=False,
    )
    coverage = {c.model: c for c in features.model_coverage([covered, blind])}
    assert coverage["blind-model"].n_stamped == 0
    assert coverage["blind-model"].capture_rate == 0.0
    assert coverage["good-model"].capture_rate == 1.0

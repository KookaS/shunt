"""The causal prefix eval: the clock scores chance, challenges never split, nulls gate the claim."""

# `test_content_free_clock_scores_chance_under_the_task_label` is the reason this module exists. The
# old harness graded a positional label on which a clock — a score that knows nothing but how far
# into the run it is — reached AUROC 0.970 while a perfect task-level oracle capped at 0.757. That
# test reproduces the defect against the OLD label in-line, then proves the new pipeline is immune.

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from benchmark.escalation import features, metrics, prefix_eval
from tests.escalation.factories import make_step, make_trajectory

pytest.importorskip("sklearn")

_PERMUTATIONS = metrics.MIN_PERMUTATIONS


def _noisy_run(n_steps: int, *, resolved: bool, tid: str, rng: random.Random, fail_rate: float):
    """A run with label-independent per-step noise; `n_steps` and `fail_rate` are the only knobs."""
    steps = [
        make_step(
            step_index=i,
            decision_index=i,
            action=f"cmd{rng.randrange(4)}",
            success=rng.random() >= fail_rate,
            failing_check_id="pkg::t" if rng.random() < fail_rate else None,
        )
        for i in range(n_steps - 1)
    ]
    terminal = make_step(step_index=n_steps - 1, decision_index=n_steps - 1, success=resolved)
    return make_trajectory([*steps, terminal], trajectory_id=tid, terminal_resolved=resolved)


def _length_confounded_corpus(n_pairs: int = 24):  # type: ignore[no-untyped-def]
    """Failed runs are LONG, resolved runs are SHORT; per-step behaviour is identically noisy.

    So the corpus's only class signal is trajectory length — exactly what a clock reads and
    exactly what a fixed-depth prefix cannot see.
    """
    rng = random.Random(7)
    out = []
    for i in range(n_pairs):
        out.append(
            _regroup(
                _noisy_run(40, resolved=False, tid=f"c{i}__m__long", rng=rng, fail_rate=0.4),
                f"c{i}",
            )
        )
        out.append(
            _regroup(
                _noisy_run(12, resolved=True, tid=f"c{i}__m__short", rng=rng, fail_rate=0.4),
                f"c{i}",
            )
        )
    return out


def _regroup(traj, instance: str):  # type: ignore[no-untyped-def]
    """Re-header a trajectory onto a shared challenge id so grouped CV has real groups."""
    from dataclasses import replace

    return replace(traj, header=replace(traj.header, instance_id=instance))


def test_content_free_clock_scores_chance_under_the_task_label() -> None:
    corpus = _length_confounded_corpus()

    # (a) Reproduce the defect. Under the OLD positional label ("within H of the end of a failed
    #     run") a clock that only knows t/n separates the classes almost perfectly.
    clock: list[float] = []
    positional: list[bool] = []
    for traj in corpus:
        n = len(traj.steps)
        failed = not traj.header.terminal_resolved
        for t in range(n):
            clock.append(t / n)
            positional.append(failed and t >= n - 3)
    assert metrics.auroc(clock, positional) > 0.9, "the old label really is won by a clock"

    # (b) The new pipeline on the SAME corpus. Length is the only signal present, and the features
    #     cannot see it, so out-of-fold discrimination collapses to chance and the null is not
    #     cleared — the corpus's one 'signal' is correctly reported as nothing.
    report = prefix_eval.evaluate_depth(corpus, 5, n_permutations=_PERMUTATIONS)
    assert report is not None
    assert report.auroc_prefix == pytest.approx(0.5, abs=0.12)
    assert not report.null_prefix.beats_null
    assert not report.has_skill


def test_grouped_cv_never_splits_a_challenge() -> None:
    # Row-level CV leaks task identity, which alone predicts the outcome at AUROC ~0.86. Every
    # fold boundary must fall between challenges, never inside one.
    from sklearn.model_selection import GroupKFold

    groups = [f"c{i // 4}" for i in range(80)]
    matrix = np.random.default_rng(0).normal(size=(80, 3))
    labels = [i % 2 == 0 for i in range(80)]
    for train, test in GroupKFold(n_splits=prefix_eval.N_SPLITS).split(matrix, labels, groups):
        train_groups = {groups[i] for i in train}
        test_groups = {groups[i] for i in test}
        assert not (train_groups & test_groups)


def test_leaked_prior_is_leave_one_out_not_self_scoring() -> None:
    rows = [
        features.EvalRow(f"t{i}", group="c0", model="m", failed=i < 3, features=(0.0,))
        for i in range(4)
    ]
    labels = [r.failed for r in rows]
    prior = prefix_eval.leaked_task_prior(rows, labels)
    # Row 0 failed; the other three in its challenge are 2 failed / 1 resolved → 2/3.
    assert prior[0] == pytest.approx(2 / 3)
    # Row 3 resolved; the other three all failed → 1.0. A self-scoring prior would give 0.0 here.
    assert prior[3] == pytest.approx(1.0)


def test_leaked_prior_falls_back_to_the_global_rate_for_a_singleton_challenge() -> None:
    rows = [
        features.EvalRow("a", group="c0", model="m", failed=True, features=(0.0,)),
        features.EvalRow("b", group="c1", model="m", failed=True, features=(0.0,)),
        features.EvalRow("c", group="c2", model="m", failed=False, features=(0.0,)),
    ]
    prior = prefix_eval.leaked_task_prior(rows, [r.failed for r in rows])
    assert prior[0] == pytest.approx(0.5)  # (2 failed - self) / (3 - 1)


def _grouped_rows(n_groups: int = 40):  # type: ignore[no-untyped-def]
    """Two rows per challenge, and the whole outcome is decided by the challenge."""
    rows = []
    for g in range(n_groups):
        for k in range(2):
            rows.append(
                features.EvalRow(
                    f"t{g}_{k}", group=f"c{g}", model="m", failed=g % 2 == 0, features=(0.0,)
                )
            )
    return rows


def test_the_headline_prior_never_reads_a_test_row_s_own_challenge() -> None:
    # THE LEAK THIS FIX CLOSES. On a corpus where the outcome is a pure function of the challenge,
    # the leave-one-out prior scores a perfect 1.0 by reading its sibling's label out of its OWN
    # test fold. The deployable prior cannot: GroupKFold puts every sibling in train-or-test
    # together, so an unseen challenge falls back to the train base rate and the prior is chance.
    rows = _grouped_rows()
    labels = [r.failed for r in rows]
    groups = [r.group for r in rows]
    assert metrics.auroc(prefix_eval.leaked_task_prior(rows, labels), labels) == 1.0
    honest = prefix_eval.grouped_task_prior(labels, groups)
    assert metrics.auroc(honest, labels) < 0.65
    # And concretely: every test row was scored with its fold's TRAIN base rate — never with a
    # value derived from its own challenge, which is what "unachievable in deployment" means.
    y = [int(lab) for lab in labels]
    for train, test in prefix_eval.grouped_splits(labels, groups):
        assert not ({groups[i] for i in train} & {groups[i] for i in test})
        base = sum(y[i] for i in train) / len(train)
        # Compared elementwise: `pytest.approx` is unhashable, so it cannot go inside a set.
        assert all(honest[i] == pytest.approx(base) for i in test)


def test_the_prior_uses_a_challenge_train_rate_when_that_challenge_is_in_training() -> None:
    # The fallback is not a hardcoded constant. GroupKFold makes the "seen" case unreachable, so
    # the branch is proved on a hand-made split rather than through the grouped path.
    labels = [True, False, True, True]
    groups = ["c0", "c1", "c0", "c2"]
    split = [(np.array([0, 1]), np.array([2, 3]))]
    prior = prefix_eval.prior_from_splits(labels, groups, split)
    assert prior[2] == pytest.approx(1.0)  # c0 IS in train, and its one train row failed
    assert prior[3] == pytest.approx(0.5)  # c2 is unseen → the train base rate, 1 of 2


def test_the_prior_and_the_risk_model_share_one_partition() -> None:
    # A prior fitted on a DIFFERENT split than the model would re-open the leak by the back door.
    labels = [i % 3 == 0 for i in range(60)]
    groups = [f"c{i // 3}" for i in range(60)]
    splits = prefix_eval.grouped_splits(labels, groups)
    assert len(splits) == prefix_eval.N_SPLITS
    assert sorted(i for _, test in splits for i in test) == list(range(60))


@pytest.fixture(scope="module")
def signal_report():  # type: ignore[no-untyped-def]
    """The pipeline is deterministic; refitting it per test only multiplies permutation cost."""
    return prefix_eval.evaluate_depth(_signal_corpus(), 5, n_permutations=_PERMUTATIONS)


def test_scores_are_continuous_not_binary(signal_report) -> None:  # type: ignore[no-untyped-def]
    # The old detector score had support {0.0, 1.0}, which is why ROC/PR had two real operating
    # points and a calibration curve was not buildable at all.
    assert signal_report is not None
    assert len(set(signal_report.scores)) > 10
    assert all(0.0 <= s <= 1.0 for s in signal_report.scores)


def _signal_corpus(n_pairs: int = 24):  # type: ignore[no-untyped-def]
    """Failed runs thrash more than resolved ones, at the SAME length. Real, noisy prefix signal."""
    rng = random.Random(3)
    out = []
    for i in range(n_pairs):
        out.append(
            _regroup(
                _noisy_run(10, resolved=False, tid=f"c{i}__m__a", rng=rng, fail_rate=0.85), f"c{i}"
            )
        )
        out.append(
            _regroup(
                _noisy_run(10, resolved=True, tid=f"c{i}__m__b", rng=rng, fail_rate=0.15), f"c{i}"
            )
        )
    return out


def test_real_prefix_signal_is_detected_and_clears_the_null(signal_report) -> None:  # type: ignore[no-untyped-def]
    # The negative controls above would also pass if the pipeline were broken and always returned
    # chance. This is the positive control: genuine prefix evidence must be found.
    assert signal_report is not None
    assert signal_report.auroc_prefix > 0.8
    assert signal_report.null_prefix.beats_null


def test_the_incremental_is_measured_against_a_prior_floored_at_chance() -> None:
    # THE DEFECT THIS PINS. The module header has always said the comparator is `max(prior, 0.5)`,
    # and `prior_comparator` implemented it — with zero consumers. The shipped headline was
    # `combined - prior`, unfloored, against a prior that is anti-predictive on the real corpus
    # (0.4175 at depth 5), so ~57% of the published +0.144 was the broken baseline's deficit
    # re-labelled as detector skill. Floored, the same fit reports +0.061.
    anti = prefix_eval._Fit(
        prior=[],
        prefix=[],
        combined=[],
        auroc_prior=0.40,
        auroc_prior_leaked=0.0,
        auroc_prefix=0.55,
        auroc_combined=0.60,
    )
    assert anti.prior_comparator == prefix_eval.CHANCE
    assert anti.incremental == pytest.approx(0.10)  # 0.60 - 0.50, NOT 0.60 - 0.40
    # A prior that beats chance is its own comparator — the floor binds one way only.
    good = replace(anti, auroc_prior=0.70)
    assert good.prior_comparator == pytest.approx(0.70)
    assert good.incremental == pytest.approx(-0.10)


def test_the_null_permutes_within_a_challenge_not_globally() -> None:
    # THE DEFECT THIS PINS. The header declares the null permutes inside each challenge so every
    # challenge's outcome multiset — and therefore the deployable prior — is identical under the
    # null and the observation. `_null_draws` called a flat `rng.shuffle` over the whole label
    # list: a probe found 10/10 groups had their multiset changed, putting the two arms in
    # different headroom regimes and leaving the gate with no power.
    groups = [f"c{i // 4}" for i in range(40)]
    labels = [(i % 4) < (i // 4) % 3 for i in range(40)]
    shuffled = prefix_eval.permute_within_groups(labels, groups, random.Random(0))
    assert shuffled != labels  # it really permutes
    for group in set(groups):
        picked = [i for i, g in enumerate(groups) if g == group]
        assert sorted(labels[i] for i in picked) == sorted(shuffled[i] for i in picked), (
            f"challenge {group}'s outcome multiset changed — that is a global shuffle"
        )


def test_depth_is_skipped_when_the_corpus_cannot_support_it() -> None:
    tiny = _signal_corpus(n_pairs=2)
    assert prefix_eval.evaluate_depth(tiny, 5, n_permutations=_PERMUTATIONS) is None
    # And a depth no run reaches yields nothing rather than an imputed row.
    assert prefix_eval.evaluate_depth(_signal_corpus(), 500, n_permutations=_PERMUTATIONS) is None

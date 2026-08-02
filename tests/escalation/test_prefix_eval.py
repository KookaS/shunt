"""The causal prefix eval: the clock scores chance, challenges never split, nulls gate the claim."""

# `test_content_free_clock_scores_chance_under_the_task_label` is the reason this module exists. The
# old harness graded a positional label on which a clock — a score that knows nothing but how far
# into the run it is — reached AUROC 0.970 while a perfect task-level oracle capped at 0.757. That
# test reproduces the defect against the OLD label in-line, then proves the new pipeline is immune.

from __future__ import annotations

import inspect
import math
import random
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from benchmark.escalation import features, metrics, prefix_eval
from benchmark.escalation.schema import Trajectory, load_jsonl
from tests.escalation import frozen_corpus
from tests.escalation.factories import make_null, make_step, make_trajectory

pytest.importorskip("sklearn")

_PERMUTATIONS = metrics.MIN_PERMUTATIONS
_LIVE_DIR = Path(__file__).resolve().parents[2] / "benchmark/escalation/data/live"
# The anti-leak margin as a LITERAL, never `features.MIN_WITHHELD` — the census recount below is
# only an independent check of the module while its boundary comes from somewhere else. One copy,
# shared with `test_features.py`; the rationale is at its definition.
_MEASURED_MARGIN = frozen_corpus.MEASURED_MARGIN


@lru_cache(maxsize=1)
def _live_corpus() -> tuple[Trajectory, ...]:
    """The committed live trajectories — the corpus every published prefix number is measured on."""
    return tuple(load_jsonl(p) for p in sorted(_LIVE_DIR.glob("*.jsonl")))


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
    # Failed runs are LONG, resolved runs are SHORT; per-step behaviour is identically noisy, so
    # the corpus's only class signal is trajectory length — what a clock reads and a fixed-depth
    # prefix cannot see. Both lengths clear `features.MIN_WITHHELD` steps beyond the depth under
    # test, or `build_rows` admits nothing and the fixture is silently empty, not loudly failing.
    rng = random.Random(7)
    out = []
    for i in range(n_pairs):
        out.append(
            _regroup(
                _noisy_run(
                    _long_enough(60), resolved=False, tid=f"c{i}__m__long", rng=rng, fail_rate=0.4
                ),
                f"c{i}",
            )
        )
        out.append(
            _regroup(
                _noisy_run(
                    _long_enough(30), resolved=True, tid=f"c{i}__m__short", rng=rng, fail_rate=0.4
                ),
                f"c{i}",
            )
        )
    return out


def _long_enough(n_steps: int, depth: int = 5) -> int:
    """A run length that survives the admission gate at `depth`, whatever the withheld tail is."""
    # `features.MIN_WITHHELD` is the anti-leak tail these fixtures must clear; hardcoding a length
    # here would make every fixture in this module go silently empty the next time it moves.
    return max(n_steps, depth + features.MIN_WITHHELD + 1)


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
    # fold boundary must fall between challenges, never inside one. Asserted on the shipped
    # `grouped_splits` rather than on a splitter constructed here, so it tracks the real partition.
    groups = [f"c{i // 4}" for i in range(80)]
    labels = [i % 2 == 0 for i in range(80)]
    for train, test in prefix_eval.grouped_splits(labels, groups):
        train_groups = {groups[i] for i in train}
        test_groups = {groups[i] for i in test}
        assert not (train_groups & test_groups)


def _imbalanced_groups(rng: random.Random, n_groups: int = 166):  # type: ignore[no-untyped-def]
    """Challenge structure shaped like the live corpus: 1-11 runs each, outcome clustered by task.

    Group SIZE is deliberately correlated with the group's outcome, which is what lets a
    size-balancing splitter produce wildly different fold prevalences.
    """
    groups: list[str] = []
    labels: list[bool] = []
    for g in range(n_groups):
        # Bigger challenges skew failed — the size/label coupling GroupKFold cannot see.
        size = rng.randint(1, 11)
        p_fail = 0.15 + 0.06 * size
        for _ in range(size):
            groups.append(f"c{g}")
            labels.append(rng.random() < p_fail)
    return labels, groups


def test_the_cv_balances_folds_by_label_not_only_by_size() -> None:
    # THE DEFECT THIS PINS, at its source. Plain GroupKFold balances folds by SIZE and never looks
    # at the label; on the live corpus its fold test base rates spanned 0.365-0.581 at depth 5.
    # That spread is what turns the prior column into a fold-id proxy (see
    # `test_a_pure_noise_corpus_reports_no_incremental_skill`), so the partition itself must be
    # stratified. The bound is on the SPREAD, not on any one fold, because that is the quantity
    # the artifact is proportional to.
    labels, groups = _imbalanced_groups(random.Random(0))
    y = np.asarray(labels, dtype=int)
    splits = prefix_eval.grouped_splits(labels, groups)
    rates = [float(y[test].mean()) for _train, test in splits]
    assert max(rates) - min(rates) < 0.08, f"fold prevalence spread is too wide: {rates}"
    # Still grouped: stratifying must not buy balance by splitting a challenge.
    for train, test in splits:
        assert not ({groups[i] for i in train} & {groups[i] for i in test})


def test_the_shipped_partition_is_the_stratified_one_not_a_degraded_groupkfold() -> None:
    # WHAT THE DELETED FALLBACK WOULD HAVE HIDDEN. `grouped_splits` used to wrap the stratified
    # splitter in `except ValueError: -> GroupKFold(...)` with no log, no warning and no report
    # field. GroupKFold IS the artifact stratification exists to remove (see
    # `test_the_cv_balances_folds_by_label_not_only_by_size`), so a silent degrade would have
    # restored the bug it was written to survive — invisibly, inside a published number. The
    # branch was also unreachable: on the installed sklearn neither condition its comment named
    # raises (see the test below), and `evaluate_depth` refuses below MIN_ROWS anyway.
    #
    # In its place, the positive claim: the shipped partition IS StratifiedGroupKFold's output, and
    # on live-shaped group structure it is NOT GroupKFold's — so "which splitter ran" is asserted
    # rather than assumed.
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    labels, groups = _imbalanced_groups(random.Random(2))
    placeholder = np.zeros((len(labels), 1))
    y = np.asarray(labels, dtype=int)
    folds = [test.tolist() for _train, test in prefix_eval.grouped_splits(labels, groups)]
    stratified = StratifiedGroupKFold(n_splits=prefix_eval.N_SPLITS)
    unstratified = GroupKFold(n_splits=prefix_eval.N_SPLITS)
    assert folds == [test.tolist() for _train, test in stratified.split(placeholder, y, groups)]
    assert folds != [test.tolist() for _train, test in unstratified.split(placeholder, y, groups)]


def test_the_degenerate_label_vectors_the_deleted_fallback_named_still_partition() -> None:
    # The two cases the removed `except ValueError` claimed to catch — "a single-class vector, or
    # fewer members of the minority class than folds". Neither raises on the installed sklearn
    # (1.9.0): both return a full grouped partition, the second with a UserWarning. That is why the
    # fallback was dead code. Pinned here so a future sklearn that DOES raise on them surfaces as a
    # loud failure rather than as a silently unstratified published number.
    groups = [f"c{i // 2}" for i in range(40)]
    assert len(prefix_eval.grouped_splits([True] * 40, groups)) == prefix_eval.N_SPLITS
    one_positive = [i == 0 for i in range(40)]  # ONE minority member against 5 folds
    splits = prefix_eval.grouped_splits(one_positive, groups)
    assert len(splits) == prefix_eval.N_SPLITS
    assert sorted(i for _train, test in splits for i in test) == list(range(40))
    for train, test in splits:
        assert not ({groups[i] for i in train} & {groups[i] for i in test})
    # Too few groups to fold at all remains "no partition", not a crash.
    assert prefix_eval.grouped_splits([True, False], ["c0", "c0"]) == []


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
    # test fold. The deployable prior cannot: the grouped partition puts every sibling in
    # train-or-test together, so an unseen challenge falls back to the train base rate.
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
    # This branch is UNREACHABLE through `grouped_task_prior` — a grouped splitter never puts a
    # challenge on both sides, measured group_overlap = 0 in every fold at every depth — and the
    # module says so at the call site. It is kept, and pinned here, because `prior_from_splits`
    # takes `splits` as an ARGUMENT and is exported: a caller supplying a non-grouped partition
    # (as this test does) must get that challenge's train rate, not a corpus-wide constant.
    # Testing it therefore means constructing the split by hand; that is the point, not a
    # workaround. The companion assertion below is the one that proves it stays dead in the
    # shipped path.
    labels = [True, False, True, True]
    groups = ["c0", "c1", "c0", "c2"]
    split = [(np.array([0, 1]), np.array([2, 3]))]
    prior = prefix_eval.prior_from_splits(labels, groups, split)
    assert prior[2] == pytest.approx(1.0)  # c0 IS in train, and its one train row failed
    assert prior[3] == pytest.approx(0.5)  # c2 is unseen → the train base rate, 1 of 2


def test_the_seen_challenge_branch_stays_dead_under_the_shipped_partition() -> None:
    # The other half of the reconciliation above: no fold of the real partition ever hands
    # `prior_from_splits` a test challenge it saw in training, so every test row gets its fold's
    # train base rate and nothing else. If this ever fails, the prior has re-acquired a per-row
    # component and the "unreachable" comment in the module is a lie.
    labels, groups = _imbalanced_groups(random.Random(1))
    y = [int(lab) for lab in labels]
    prior = prefix_eval.grouped_task_prior(labels, groups)
    for train, test in prefix_eval.grouped_splits(labels, groups):
        assert not ({groups[i] for i in train} & {groups[i] for i in test})
        base = sum(y[i] for i in train) / len(train)
        assert all(prior[i] == pytest.approx(base) for i in test)


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
    length = _long_enough(30)
    out = []
    for i in range(n_pairs):
        out.append(
            _regroup(
                _noisy_run(length, resolved=False, tid=f"c{i}__m__a", rng=rng, fail_rate=0.85),
                f"c{i}",
            )
        )
        out.append(
            _regroup(
                _noisy_run(length, resolved=True, tid=f"c{i}__m__b", rng=rng, fail_rate=0.15),
                f"c{i}",
            )
        )
    return out


def test_the_auprc_null_is_measured_rather_than_left_to_prevalence(signal_report) -> None:  # type: ignore[no-untyped-def]
    # THE DEFECT THIS CLOSES. The PR figure drew PREVALENCE as its no-skill line — the no-skill
    # average precision for EXCHANGEABLE rows. These rows cluster by challenge and the whole
    # pipeline is refit per shuffle, so the no-information AUPRC is an empirical quantity, and
    # nothing computed it. A figure cannot read a statistic that the pipeline never produced, and
    # hardcoding one into the plot is the defect class this harness exists to remove.
    assert signal_report is not None
    null = signal_report.null_auprc
    assert null is not None
    assert null.n_permutations == _PERMUTATIONS
    # Drawn from the SAME shuffles as the AUROC nulls beside it, so the two references on one
    # canvas set describe one randomization rather than two.
    assert len(null.draws) == len(signal_report.null_prefix.draws)
    assert null.observed == pytest.approx(signal_report.auprc_prefix)
    assert signal_report.to_dict()["null_auprc_prefix"] is not None


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
    # (0.4175 at depth 5 as the harness then stood), so ~57% of the published +0.144 was the broken
    # baseline's deficit re-labelled as detector skill. The corpus and the CV have both moved since
    # — the prior now measures 0.4873-0.4959 across depths, so the deficit the floor absorbs is
    # small — but the floor is not conditional on the deficit's size, which is what this pins.
    anti = prefix_eval._Fit(
        prior=[],
        prefix=[],
        combined=[],
        splits=[],
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
    # list: a probe found 10/10 groups had their multiset changed, so the shuffle nulled the whole
    # grouped partition rather than the prefix's contribution to it (measured at depth 5, the prior
    # is invariant under this permutation — sd 0.0000 over 200 draws — against sd 0.0026 globally).
    groups = [f"c{i // 4}" for i in range(40)]
    labels = [(i % 4) < (i // 4) % 3 for i in range(40)]
    shuffled = prefix_eval.permute_within_groups(labels, groups, random.Random(0))
    assert shuffled != labels  # it really permutes
    for group in set(groups):
        picked = [i for i, g in enumerate(groups) if g == group]
        assert sorted(labels[i] for i in picked) == sorted(shuffled[i] for i in picked), (
            f"challenge {group}'s outcome multiset changed — that is a global shuffle"
        )


def _null_fit(seed: int):  # type: ignore[no-untyped-def]
    """One end-to-end fit on PURE NOISE over the live corpus's group structure."""
    labels, groups = _imbalanced_groups(random.Random(seed))
    matrix = np.random.default_rng(seed).normal(size=(len(labels), 6))
    rows = [
        features.EvalRow(f"t{i}", group=g, model="m", failed=lab, features=tuple(matrix[i]))
        for i, (g, lab) in enumerate(zip(groups, labels, strict=True))
    ]
    return prefix_eval._fit_once(rows, matrix, labels, groups), labels, groups


# 8 seeds: at the measured null sd of `combined - prefix` (0.0011) the mean's standard error is
# ~0.0004, so the 0.005 band below is >10 sigma wide — stable — while the pre-fix mean of +0.026
# sits 5x outside it. Runtime ~3s total on this machine, which is what keeps it a CI test rather
# than a benchmark. Raising the count buys resolution this assertion does not need.
_NULL_SEEDS = 8


def test_a_pure_noise_corpus_reports_no_incremental_skill() -> None:
    # THE MISSING REGRESSION TEST — the one whose absence let the fold-prevalence artifact ship.
    #
    # The positive control `_signal_corpus` builds exactly one failed and one resolved run per
    # challenge, so EVERY fold base rate is 0.5 and the fixture is structurally immune to a defect
    # that lives in fold-prevalence imbalance. This corpus is the opposite by construction:
    # challenge sizes vary 1-11 like the live corpus, size is coupled to outcome, and the features
    # are pure Gaussian noise carrying no information about the label whatsoever.
    #
    # On such a corpus every arm must land at chance. Under the old size-only GroupKFold it did
    # not: `prior_from_splits` gives each test row its TRAIN-fold base rate, the exact arithmetic
    # complement of its own test fold's rate (measured corr -1.0000), so the prior column entering
    # the combined model was a fold-id proxy. It bought +0.026 of POOLED AUROC here (+0.13 on the
    # live corpus) while adding zero within-fold rank — noise sold as incremental skill.
    fits = [_null_fit(seed)[0] for seed in range(_NULL_SEEDS)]

    # (a) The direct measurement of the artifact: what does ADDING the prior column buy? On noise,
    #     nothing. This is the high-power assertion — its null sd is 20x tighter than the
    #     incremental's, because both arms share the same noisy prefix model and it cancels.
    bought = [fit.auroc_combined - fit.auroc_prefix for fit in fits]
    mean_bought = sum(bought) / len(bought)
    assert abs(mean_bought) < 0.005, f"the prior column bought pooled AUROC on noise: {bought}"

    # (b) The headline itself, stated as the brief demands: E[incremental] ~= 0. Its band is wider
    #     (null sd ~0.021, driven by the prefix model's own noise on ~1000 rows, which (a) cancels)
    #     so 0.02 is ~1 sd of the mean's own spread over these seeds — tight enough to catch a
    #     systematic offset, loose enough not to flake on sampling noise that is genuinely there.
    incrementals = [fit.incremental for fit in fits]
    mean_incremental = sum(incrementals) / len(incrementals)
    assert abs(mean_incremental) < 0.02, f"noise produced incremental skill: {incrementals}"

    # (c) And the prefix arm alone, the simplest sanity check that the corpus really is null.
    prefixes = [fit.auroc_prefix for fit in fits]
    assert sum(prefixes) / len(prefixes) == pytest.approx(prefix_eval.CHANCE, abs=0.02)


def _nested_corpus():  # type: ignore[no-untyped-def]
    """A corpus whose depth-5/10/20 row sets are strictly NESTED, like the real one's."""
    # Run length decides admission (`depth + MIN_WITHHELD` scorable steps), so three length tiers
    # give three genuinely different row sets: 80 rows at depth 5, 64 at depth 10, 48 at depth 20 —
    # every one over `MIN_ROWS`. A single-length corpus would make all three row sets identical and
    # the shared-shuffle assertions below would hold vacuously.
    rng = random.Random(9)
    out = []
    for tier, (length, n_pairs) in enumerate(((45, 24), (35, 8), (31, 8))):
        for i in range(n_pairs):
            for resolved, suffix, rate in ((False, "a", 0.85), (True, "b", 0.15)):
                run = _noisy_run(
                    length,
                    resolved=resolved,
                    tid=f"n{tier}_{i}__m__{suffix}",
                    rng=rng,
                    fail_rate=rate,
                )
                out.append(_regroup(run, f"n{tier}_{i}"))
    return out


@pytest.fixture(scope="module")
def family_reports():  # type: ignore[no-untyped-def]
    """The nested corpus scored at THREE depths at once — the family the verdict is read from."""
    # Fitted once for the module: this is 3 depths x `_PERMUTATIONS` pipeline re-fits.
    return prefix_eval.evaluate(_nested_corpus(), (5, 10, 20), n_permutations=_PERMUTATIONS)


def test_the_nested_corpus_really_nests(family_reports) -> None:  # type: ignore[no-untyped-def]
    # The fixture's premise, asserted rather than assumed: if the three depths ever scored the same
    # rows, every shared-shuffle assertion below would pass without testing anything.
    sizes = [r.n_rows for r in family_reports]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == len(sizes)
    assert min(sizes) >= prefix_eval.MIN_ROWS


def test_one_shuffle_is_scored_at_every_depth_rather_than_drawn_per_depth(family_reports) -> None:  # type: ignore[no-untyped-def]
    # THE MECHANISM THE CORRECTION RESTS ON. A max over three independently-drawn nulls is not a
    # family-wise reference for anything: the three depths' row sets are strictly nested and their
    # statistics are strongly dependent, so the max distribution is only the right band when each
    # replicate is ONE shuffle scored at all three depths. That every depth's family null carries
    # the IDENTICAL draw vector is what proves the shuffle was shared, not repeated.
    assert [r.depth for r in family_reports] == [5, 10, 20]
    families = {r.gate_null_prefix.draws for r in family_reports}
    assert len(families) == 1, "the depths were gated against different draw sets"
    assert len({r.gate_null_incremental.draws for r in family_reports}) == 1
    # ...and it is genuinely the MAX across the depths, not one depth's null reused: every draw
    # dominates the corresponding per-depth draw, replicate by replicate.
    shared = next(iter(families))
    assert len(shared) == _PERMUTATIONS
    for report in family_reports:
        assert len(report.null_prefix.draws) == _PERMUTATIONS
        assert all(a >= b for a, b in zip(shared, report.null_prefix.draws, strict=True))
    assert any(shared != r.null_prefix.draws for r in family_reports)


def test_the_family_wise_band_is_never_looser_than_the_per_depth_one(family_reports) -> None:  # type: ignore[no-untyped-def]
    # The correction has to cost something or it is not a correction. A max over three dependent
    # statistics stochastically dominates each of them, so at every depth the family-wise 97.5th
    # percentile sits at or above that depth's own, and the adjusted p is at or above the raw p.
    for report in family_reports:
        assert report.gate_null_prefix.ci_high >= report.null_prefix.ci_high
        assert report.gate_null_incremental.ci_high >= report.null_incremental.ci_high
        assert report.gate_null_prefix.p_value >= report.null_prefix.p_value
        assert report.gate_null_incremental.p_value >= report.null_incremental.p_value
    # And it really is stricter somewhere, or the assertions above are vacuous equalities.
    assert any(r.gate_null_prefix.ci_high > r.null_prefix.ci_high for r in family_reports)


def test_a_single_depth_is_a_family_of_one_and_is_not_penalised(signal_report) -> None:  # type: ignore[no-untyped-def]
    # A caller asking about ONE depth runs one test, and the max over a family of one is that test.
    # Correcting it anyway would be a tax on a multiplicity that does not exist — and it would move
    # every single-depth number in this suite, including the instrument-validity controls.
    assert signal_report is not None
    assert signal_report.gate_null_prefix is signal_report.null_prefix_family
    assert signal_report.gate_null_prefix.draws == signal_report.null_prefix.draws
    assert signal_report.gate_null_incremental.draws == signal_report.null_incremental.draws


def test_the_skill_gate_reads_the_family_wise_null(signal_report) -> None:  # type: ignore[no-untyped-def]
    # The gate must consult `gate_null_*`, not the uncorrected pair. Proven by moving ONLY the
    # family-wise band on a report that clears every per-depth null: if `has_skill` still says True,
    # the correction is computed and ignored — the defect class this harness keeps re-learning.
    assert signal_report is not None
    assert signal_report.has_skill
    corrected = replace(
        signal_report,
        null_prefix_family=make_null(signal_report.auroc_prefix, signal_report.auroc_prefix + 0.05),
    )
    assert corrected.null_prefix.beats_null
    assert not corrected.has_skill
    published = corrected.to_dict()
    assert published["null_auroc_prefix_familywise"] == corrected.gate_null_prefix.to_dict()
    assert published["null_auroc_prefix"] == corrected.null_prefix.to_dict()


def test_the_bootstrap_discloses_that_it_is_conditional_on_one_fit() -> None:
    # NOT COSMETIC. `evaluate_depth` fits the pipeline ONCE and both bootstraps then resample
    # indices into those fitted scores, while the permutation null beside them re-runs the whole
    # pipeline per shuffle and the module header says so. A reader carries the null's contract
    # across to the interval unless the interval says otherwise. Refitting properly widens the
    # prefix interval 1.19-1.44x and the incremental one 1.05-1.13x, and since `ci_incremental[0]
    # > 0.0` is a skill condition, too narrow biases toward DECLARING skill. The disclosure is
    # therefore load-bearing, and so is the warning against the naive refit that reintroduces the
    # very leak `grouped_splits` exists to prevent.
    source = inspect.getsource(prefix_eval._incremental_ci)
    for claim in ("CONDITIONAL", "1.19-1.44x", "1.05-1.13x", "DECLARING skill", "ORIGINAL"):
        assert claim in source, f"the bootstrap's limitation no longer states: {claim}"
    # The published `ci_prefix` is built by `grouped_bootstrap_ci` and carries the same limitation,
    # so its call site must point at the same paragraph rather than leave it undisclosed.
    assembled = inspect.getsource(prefix_eval._depth_report)
    assert "CONDITIONAL ON THIS ONE FIT" in assembled
    assert "_incremental_ci" in assembled


def test_the_combined_arm_reports_a_fold_honest_auroc_beside_its_pooled_one() -> None:
    # `auroc_combined_folded` is the diagnostic that was MISSING. `auroc_prefix_folded` existed,
    # but the prefix arm is the clean one — the prior column, and so the artifact, enters only the
    # COMBINED arm, which had no fold-honest twin to be contradicted by. The pooled/folded gap is
    # what a reader checks; with the CV stratified there is nothing left for the pooled figure to
    # carry, so on noise the two must agree. Under the old partition this gap ran to -0.018 here.
    fit, labels, groups = _null_fit(0)
    folded = prefix_eval.folded_auroc(fit.combined, labels, groups, fit.splits)
    assert folded == pytest.approx(fit.auroc_combined, abs=0.015)


def test_the_report_publishes_the_combined_folded_auroc(signal_report) -> None:  # type: ignore[no-untyped-def]
    # A number computed and not published is the same defect class as `auroc_prefix_folded` and
    # `mde_auroc` before it: the docs then quote a figure with no machine-readable source.
    assert signal_report is not None
    assert not math.isnan(signal_report.auroc_combined_folded)
    published = signal_report.to_dict()
    assert published["auroc_combined_folded"] == round(signal_report.auroc_combined_folded, 4)


def test_depth_is_skipped_when_the_corpus_cannot_support_it() -> None:
    tiny = _signal_corpus(n_pairs=2)
    assert prefix_eval.evaluate_depth(tiny, 5, n_permutations=_PERMUTATIONS) is None
    # And a depth no run reaches yields nothing rather than an imputed row.
    assert prefix_eval.evaluate_depth(_signal_corpus(), 500, n_permutations=_PERMUTATIONS) is None


def _blind_runs(n: int, n_steps: int = 30):  # type: ignore[no-untyped-def]
    """Runs the per-step stamping stage never touched — long enough to be admitted at depth 5."""
    # `confirmed=False` on every non-terminal step is exactly what `is_stamped` reads. They are
    # deliberately LONG (>= depth + MIN_WITHHELD), so the corpus gate — not the margin — is the
    # only thing that can exclude them; a short fixture would pass vacuously.
    return [
        _regroup(
            make_trajectory(
                [
                    make_step(step_index=j, decision_index=j, confirmed=False)
                    for j in range(_long_enough(n_steps))
                ],
                trajectory_id=f"z{i}__blind__x",
                terminal_resolved=i % 2 == 0,
            ),
            f"z{i}",
        )
        for i in range(n)
    ]


def test_evaluate_depth_gates_its_own_corpus_instead_of_trusting_the_caller(signal_report) -> None:  # type: ignore[no-untyped-def]
    # THE TRAP THIS CLOSES. `run_eval.evaluate` filtered on `is_stamped`; `evaluate_depth` did not,
    # so an ad-hoc probe calling it directly scored a DIFFERENT corpus than the sanctioned pipeline,
    # silently. It manufactured a false "significant" result on the real corpus — 241 rows / 84
    # groups at depth 20 against the sanctioned 239 / 83, the two extra rows moving the paired
    # bootstrap CI off zero. Prose ("remember to filter") is not a defence; the gate is in the
    # function. Contaminating a clean corpus must now change NOTHING except the exclusion count.
    blind = _blind_runs(20)
    contaminated = prefix_eval.evaluate_depth(
        [*_signal_corpus(), *blind], 5, n_permutations=_PERMUTATIONS
    )
    assert signal_report is not None
    assert contaminated is not None
    assert (contaminated.n_rows, contaminated.n_groups) == (
        signal_report.n_rows,
        signal_report.n_groups,
    )
    assert contaminated.incremental_auroc == signal_report.incremental_auroc
    assert contaminated.ci_incremental == signal_report.ci_incremental
    assert contaminated.n_excluded_unstamped == len(blind)
    assert signal_report.n_excluded_unstamped == 0


def test_a_wholly_unstamped_corpus_is_refused_rather_than_scored() -> None:
    # The same gate at its limit: 60 unstamped runs, every one long enough to be admitted and both
    # outcome classes present, is ample for a fit — and must still yield no report at all, because
    # an unstamped run's fields are parser defaults rather than evidence. Before the gate this
    # returned a full DepthReport with 60 rows.
    blind = _blind_runs(60)
    assert len(blind) > prefix_eval.MIN_ROWS
    assert prefix_eval.evaluate_depth(blind, 5, n_permutations=_PERMUTATIONS) is None


def test_the_census_names_all_three_reasons_a_run_is_not_scored() -> None:
    # F6'S DEFECT, at its source. One `n_excluded_short` counted two unrelated exclusions under a
    # name that means only one of them. Each bucket is populated by a fixture that can ONLY land in
    # that bucket, so a mis-attribution cannot hide inside a correct total.
    scored = _signal_corpus()  # stamped, long: admitted
    blind = _blind_runs(9)  # stamped-stage never ran: the corpus gate
    rng = random.Random(11)
    short = [  # 3 scorable steps — never reaches depth 5
        _regroup(
            _noisy_run(4, resolved=i % 2 == 0, tid=f"s{i}__m__x", rng=rng, fail_rate=0.5), f"s{i}"
        )
        for i in range(7)
    ]
    margin = [  # reaches depth 5 with only MIN_WITHHELD - 1 steps left unread: the anti-leak margin
        _regroup(
            _noisy_run(
                5 + features.MIN_WITHHELD,
                resolved=i % 2 == 0,
                tid=f"m{i}__m__x",
                rng=rng,
                fail_rate=0.5,
            ),
            f"m{i}",
        )
        for i in range(5)
    ]
    census = prefix_eval.corpus_census([*scored, *blind, *short, *margin], 5)
    assert len(census.rows) == len(scored)
    assert census.n_unstamped == len(blind)
    assert census.n_too_short == len(short)
    assert census.n_by_margin == len(margin)


@pytest.mark.parametrize("depth", features.DEFAULT_DEPTHS)
def test_the_real_corpus_census_separates_margin_cuts_from_runs_that_never_reached_the_depth(
    depth: int,
) -> None:
    # WHAT THIS USED TO ASSERT, AND WHY IT NO LONGER DOES. It pinned `n_unstamped == 8` plus a
    # per-depth tuple — (452, 124, 0, 339) / (372, 110, 10, 409) / (239, 83, 207, 345) — measured on
    # the 2026-07-30 corpus. Every one of those is a property of the DATA, so correcting the corpus
    # (to 799 runs, 727 stamped, 414/344/228 rows) turned six tests red while nothing in
    # `corpus_census` had changed. A test whose only failure mode is "the data improved" is a false
    # wall, and re-pinning the new numbers is the same wall with a fresher number on it.
    #
    # Two legs survive a corpus correction and are kept:
    #
    # (a) a per-bucket RECOUNT, reaching the same three buckets by a different route — from the
    #     measured margin as a LITERAL rather than from `features.MIN_WITHHELD`. That is not the
    #     tautology "the census equals itself": a `corpus_census` that restated the margin, folded
    #     the two length exclusions back into one number, or dropped the stamped gate fails here,
    #     and so does a loosened `MIN_WITHHELD` — at any corpus size.
    # (b) the framework floors, which are what `evaluate_depth` actually requires of a depth.
    #
    # The exact counts moved to `frozen_corpus` (next test), and the live census stays published to
    # the report JSON, where it is reviewed rather than asserted.
    corpus = _live_corpus()
    census = prefix_eval.corpus_census(corpus, depth)
    stamped = [t for t in corpus if features.is_stamped(t)]
    scorable = [len(features.scorable_steps(t)) for t in stamped]
    assert census.n_unstamped == len(corpus) - len(stamped)
    assert census.n_too_short == sum(n < depth for n in scorable)
    assert census.n_by_margin == sum(depth <= n < depth + _MEASURED_MARGIN for n in scorable)
    assert len(census.rows) == sum(n >= depth + _MEASURED_MARGIN for n in scorable)
    # The old conflated figure, reconstructed — and still shown to be the sum of two different
    # things, so no future rewrite can quietly report one number where two are meant.
    assert census.n_too_short + census.n_by_margin == (
        len(corpus) - census.n_unstamped - len(census.rows)
    )
    assert len(census.rows) >= prefix_eval.MIN_ROWS
    assert len({r.group for r in census.rows}) >= prefix_eval.N_SPLITS


@pytest.mark.parametrize("depth", features.DEFAULT_DEPTHS)
def test_the_frozen_corpus_census_is_exactly_the_one_its_run_lengths_imply(depth: int) -> None:
    # The sharp half, on eleven runs that never move. `frozen_corpus.EXPECTED` is checked in as
    # literals — the sort of exact assertion the live corpus cannot carry — and is re-derived here
    # from the SPECS table so a typo in either fails loudly instead of quietly re-pinning a wrong
    # number. The derivation is an independent oracle, not a call into the code under test.
    census = prefix_eval.corpus_census(frozen_corpus.build(), depth)
    measured = frozen_corpus.ExpectedCensus(
        n_rows=len(census.rows),
        n_groups=len({r.group for r in census.rows}),
        n_unstamped=census.n_unstamped,
        n_too_short=census.n_too_short,
        n_by_margin=census.n_by_margin,
    )
    stamped = [s for s in frozen_corpus.SPECS if s.stamped]
    implied = frozen_corpus.ExpectedCensus(
        n_rows=sum(s.n_scorable >= depth + _MEASURED_MARGIN for s in stamped),
        n_groups=len(
            {s.challenge for s in stamped if s.n_scorable >= depth + _MEASURED_MARGIN},
        ),
        n_unstamped=len(frozen_corpus.SPECS) - len(stamped),
        n_too_short=sum(s.n_scorable < depth for s in stamped),
        n_by_margin=sum(depth <= s.n_scorable < depth + _MEASURED_MARGIN for s in stamped),
    )
    assert measured == frozen_corpus.EXPECTED[depth]
    assert implied == frozen_corpus.EXPECTED[depth]


def test_the_frozen_census_moves_when_the_frozen_corpus_does() -> None:
    # THE REVERT PROBE for the assertion above: an exact-count assertion that cannot fail is
    # coverage theatre, so each half is shown to bind on a different mutation. Dropping one admitted
    # run moves the row count; shortening the run that sits EXACTLY on the depth-5 bar by one step
    # moves it out of `rows` and into `n_by_margin` without changing anything else.
    corpus = frozen_corpus.build()
    base = prefix_eval.corpus_census(corpus, 5)
    at_the_bar = "inst-b__frontier__max"  # 27 scorable steps = 5 + MIN_WITHHELD, admitted
    assert at_the_bar in frozen_corpus.admitted(5)
    dropped = prefix_eval.corpus_census(
        [t for t in corpus if t.header.trajectory_id != at_the_bar], 5
    )
    assert len(dropped.rows) == len(base.rows) - 1
    trimmed = [
        replace(t, steps=[*t.steps[:-2], t.steps[-1]])
        if t.header.trajectory_id == at_the_bar
        else t
        for t in corpus
    ]
    shortened = prefix_eval.corpus_census(trimmed, 5)
    assert len(shortened.rows) == len(base.rows) - 1
    assert shortened.n_by_margin == base.n_by_margin + 1
    assert shortened.n_too_short == base.n_too_short


def test_the_framework_floors_really_refuse_a_starved_corpus() -> None:
    # The live-corpus leg asserts only `MIN_ROWS` / `N_SPLITS`, so those floors have to be shown to
    # refuse something. The frozen corpus at depth 20 is two rows in two challenges — a corpus the
    # eval cannot estimate — and the same two assertions reject it.
    census = prefix_eval.corpus_census(frozen_corpus.build(), 20)
    assert len(census.rows) < prefix_eval.MIN_ROWS
    assert len({r.group for r in census.rows}) < prefix_eval.N_SPLITS
    assert prefix_eval.evaluate_depth(frozen_corpus.build(), 20) is None


def test_the_report_publishes_the_corpus_census(signal_report) -> None:  # type: ignore[no-untyped-def]
    # A count computed and not published is the defect class this harness keeps re-learning. The
    # census must reach the JSON, and the two length exclusions must be SEPARATE keys — the single
    # `n_excluded_short` they replace is gone, so a consumer reading it fails loudly rather than
    # silently reading a number whose meaning changed.
    assert signal_report is not None
    published = signal_report.to_dict()
    keys = ("n_excluded_unstamped", "n_excluded_too_short", "n_excluded_by_margin")
    assert "n_excluded_short" not in published
    for key in keys:
        assert published[key] == getattr(signal_report, key)
    assert published["n_rows"] + sum(published[k] for k in keys) == len(_signal_corpus())

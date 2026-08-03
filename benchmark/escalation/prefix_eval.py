"""The causal escalation eval: predict the TASK outcome from a prefix, conditional on the prior."""

# WHAT THIS REPLACES. The old harness graded a binary {0,1} policy flag against a POSITIONAL label
# ("this step is one of the last H of a run that failed"). Three audits proved that metric cannot
# see a detector: a perfect task-level oracle capped at AUROC 0.757, a content-free clock scored
# 0.970, and AUROC was pinned at ~0.503 for every H in {1,3,5,10}.
#
# WHAT THIS DOES INSTEAD. One row per trajectory. Label = "this attempt ultimately failed" — the
# question escalation exists to answer. Score = a calibrated probability from prefix-only features
# at a fixed decision depth (see features.py for the two structural anti-leak rules). Skill is
# reported THREE ways because only the third is honest:
#
#   auroc_prior     — the router's own t=0 knowledge, estimated the way a DEPLOYED router must:
#                     fold-wise, from TRAIN folds only. Under the grouped partition a test
#                     challenge is by construction absent from training, so this prior falls back
#                     to the train base rate — which is exactly the point, and is what a router
#                     facing an unseen instance actually has.
#   auroc_prefix    — grouped-CV out-of-fold discrimination from the prefix alone. Grouped by
#                     challenge, so a challenge never appears in both train and test.
#   incremental     — auroc(prior + prefix) - auroc(prior). THE number. Everything else overstates.
#
# WHY THE PRIOR IS NOT LEAVE-ONE-OUT. It used to be: each row's prior was the mean of its OWN
# instance's other labels, over the whole corpus. Grouped by instance, those siblings sit in the
# row's own TEST fold, so the baseline had read the test labels — it scored AUROC ~0.885 where the
# honest estimate is ~0.43, and the headline `incremental` was asking the detector to beat a
# leaked oracle. `leaked_task_prior` is retained purely as a DIAGNOSTIC (reported alongside as
# `auroc_prior_leaked`) because the gap between the two is itself the finding.
#
# Every headline carries a label-permutation null (>= 200 shuffles, the whole pipeline re-run per
# shuffle) and a challenge-level bootstrap CI. "Beats chance" means "above the null's 97.5th
# percentile", never "above the point estimate".
#
# THE NULL IS FAMILY-WISE ACROSS DEPTHS, because the verdict is. `run_eval._status` prints OK if
# ANY reported depth clears and `_no_skill_reason` reports the max-incremental depth, so the
# requested `DEFAULT_DEPTHS` — (10,) since 2026-08-02, when depths 5 and 20 left the ladder
# (depth 5: rank-deficient design; depth 20: exceeded its own selection-bias tolerance;
# features.py records the decision) — are a max over
# that many tests, and that max carried no correction at all.
# Bonferroni would be the wrong instrument here: the depths' row sets are strictly NESTED (a run
# admitted at a deeper depth is admitted at every shallower one, so a deeper depth's rows are a
# subset of the shallower's), the reported statistics are therefore strongly positively
# dependent, and a threshold priced for independence would be conservative by an unknown amount.
# `evaluate` instead draws ONE within-challenge shuffle per replicate, scores it at EVERY requested
# depth, and keeps the max; each depth's observed statistic is then gated against that max
# distribution. That is the single-step maxT construction — exact under arbitrary dependence, and
# the resulting p-value IS the family-wise adjusted p-value for that depth. `evaluate_depth` is a
# family of ONE, where the max over the family is the marginal null, so a single-depth call is
# unchanged to the bit. The uncorrected per-depth nulls are still reported beside the family-wise
# ones, so the size of the correction is visible rather than folded away.
#
# ONE SHUFFLE, MANY ROW SETS. The shared draw permutes labels within each challenge over the UNION
# of the depths' rows, and every depth then reads its own rows out of that one permuted vector. A
# given depth's subset therefore does NOT keep its own challenge multiset exactly — labels can move
# between a challenge's long and short runs. That is the price of one shared randomization, and it
# is paid in the conservative direction: at the shallowest depth the union IS that depth's row set,
# so its null is unchanged, and at deeper depths the extra freedom makes the null WIDER (harder to
# clear), never narrower. What it must not do is null anything else, and it does not: the
# permutation is still within-challenge, so every challenge's corpus-level outcome multiset holds.
#
# THE BOOTSTRAP INTERVALS ARE CONDITIONAL ON ONE FIT, unlike the permutation null beside them — see
# `_incremental_ci` for the mechanism, the measured cost, and which way the bias points.
#
# THE NULL PERMUTES WITHIN A CHALLENGE, NOT GLOBALLY, because that is what nulls the prefix and
# only the prefix. Permuting inside each challenge preserves every challenge's outcome multiset, so
# the fold base rates — and with them the deployable prior — are IDENTICAL under the null and the
# observation: measured at depth 5 over 200 draws, the prior is 0.4938 in EVERY within-challenge
# draw (sd 0.0000), exactly the observed value. It also matches the paired grouped bootstrap, which
# is the independent check.
#
# THE ARITHMETIC ARGUMENT THAT USED TO STAND HERE IS RETRACTED, and is kept written down because a
# justification that quietly outlives its numbers is the failure this module keeps re-learning. It
# said a global shuffle put the two arms in different HEADROOM regimes — the prior collapsing to
# chance under the shuffle while the real prior capped the observed incremental near +0.117, against
# a measured global null ci_high of +0.117 — so the gate was unpassable for any detector however
# good. Both figures were computed on the LEAKED leave-one-out prior (AUROC ~0.883, hence
# 1 - 0.883 = 0.117) that is no longer the baseline. Re-measured at depth 5 on the sanctioned corpus
# (452 rows / 124 challenges, 200 draws): the deployable prior is 0.4938 and is floored to 0.5, and
# a global shuffle leaves it at 0.4930 (sd 0.0026), floored to the same 0.5. The prior cannot
# collapse because it is already at chance, so there is no headroom asymmetry left, and the global
# null's ci_high is +0.0429 — a threshold a detector could clear.
#
# THE PRICE, stated rather than hidden: a challenge whose runs all share an outcome is invariant
# under a within-challenge permutation, so its rows never move. At depth 5, 42/124 challenges are
# heterogeneous (33.9%) and 238/452 rows can move (52.7%) — this null is estimated off roughly half
# the corpus, and it is the NARROWER of the two (prefix-AUROC sd 0.0048 against the global
# shuffle's 0.0305). Narrower is not more conservative: a tighter null has a lower 97.5th
# percentile, so this is the EASIER gate to clear (incremental ci_high -0.0576 against the global
# +0.0429). It is chosen for nulling the right quantity, not for strictness — the global shuffle's
# extra spread sits almost entirely in the PREFIX arm (sd 0.0305 against the prior's 0.0026), i.e.
# it is challenge-level clustering being destroyed, not detector headroom being revealed. No
# published claim currently turns on the choice: the observed depth-5 incremental (-0.0576) clears
# neither null (p = 0.841 within-challenge, 0.925 global).
#
# THE CV IS STRATIFIED ON THE LABEL, NOT JUST GROUPED. Plain GroupKFold balances folds by SIZE
# only, and the resulting fold-prevalence spread turned the prior column into a fold-id proxy that
# inflated the pooled combined AUROC without adding a single within-fold rank — the full derivation
# is in `grouped_splits`. Every number this module publishes is therefore measured under
# StratifiedGroupKFold, and `auroc_prefix_folded` / `auroc_combined_folded` are reported next to
# their pooled twins precisely so that class of between-fold accounting is visible rather than
# hidden inside the headline.
#
# THE COMPARATOR IS FLOORED AT CHANCE. The honest prior scores BELOW 0.5 on this corpus, and
# `combined - prior` against an anti-predictive baseline manufactures positive skill out of a broken
# comparator. `incremental` therefore measures against `max(prior, 0.5)`: beat the better of the
# router's prior and no-information, or report nothing.

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark.escalation import features, metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.escalation.features import EvalRow
    from benchmark.escalation.schema import Trajectory

N_SPLITS: Final[int] = 5
# Below this many rows or groups the grouped CV is not estimable and the depth is skipped.
MIN_ROWS: Final[int] = 40
_MIN_CLASSES: Final[int] = 2
# No-information AUROC. The prior comparator is floored here so an anti-predictive baseline can
# never be "beaten" into an apparent finding.
CHANCE: Final[float] = 0.5


@dataclass(frozen=True)
class CorpusCensus:
    """The rows `evaluate_depth` may score at one depth, plus why every other run is not one."""

    rows: list[EvalRow]
    n_unstamped: int
    n_too_short: int
    n_by_margin: int


def corpus_census(trajectories: Sequence[Trajectory], depth: int) -> CorpusCensus:
    """The sanctioned corpus at `depth`, with every excluded trajectory accounted for by reason."""
    # THE CORPUS GATE lives here so it cannot be skipped by a caller. `run_eval.evaluate` filtered
    # on `is_stamped` and `evaluate_depth` did not, so any ad-hoc probe calling `evaluate_depth`
    # directly scored a DIFFERENT corpus than the sanctioned pipeline, silently. That is not
    # hypothetical: it manufactured a false "significant" result — 241 rows / 84 groups at depth 20
    # against the sanctioned 239 / 83, and those two unstamped rows moved the paired bootstrap CI
    # off zero. An unstamped run's fields are parser defaults, so it feeds the model a
    # collection-date proxy rather than escalation evidence (`features.is_stamped`).
    #
    # The two length exclusions are counted apart because ONE number could not name both. Admission
    # needs `depth + MIN_WITHHELD` scorable steps, and the single `n_excluded_short` reported their
    # sum under a name that means only the first: at depth 5 it read 339 of 791 while ZERO of those
    # 339 were short of depth 5 — every one reached it and was cut by the anti-leak margin, so a
    # reader diagnosing coverage blamed run length for a threshold this harness chose.
    stamped = [t for t in trajectories if features.is_stamped(t)]
    rows = features.build_rows(stamped, depth)
    # ONE predicate — "did this run reach `depth` scorable steps" — with the margin count taken as
    # the residual, so `features.MIN_WITHHELD` is not restated here to drift from the admission
    # rule in `extract_features` that actually enforces it.
    reached = sum(len(features.scorable_steps(t)) >= depth for t in stamped)
    return CorpusCensus(
        rows=rows,
        n_unstamped=len(trajectories) - len(stamped),
        n_too_short=len(stamped) - reached,
        n_by_margin=reached - len(rows),
    )


@dataclass(frozen=True)
class DepthReport:
    """One decision depth's full result: skill, prior-conditional skill, null, and intervals."""

    depth: int
    n_rows: int
    # `corpus_census`, published: these three plus `n_rows` sum to the trajectories `evaluate_depth`
    # was handed, so a reader can reconstruct why every unscored run is unscored — and a probe that
    # fed it an unfiltered corpus sees the gate fire instead of only the gated number.
    n_excluded_unstamped: int  # the per-step stamping stage never ran on it (see `is_stamped`)
    n_excluded_too_short: int  # fewer than `depth` scorable steps — never reached the decision
    n_excluded_by_margin: int  # reached `depth`, but < MIN_WITHHELD steps left unread after it
    n_groups: int
    base_rate: float
    auroc_prior: float
    # The old leaked leave-one-out prior, kept as a diagnostic contrast only — never the baseline
    # `incremental_auroc` is measured against.
    auroc_prior_leaked: float
    auroc_prefix: float
    # Diagnostic: the n-weighted mean of the PER-FOLD AUROCs. `auroc_prefix` pools every fold's
    # probabilities into one ranking, which carries a between-fold calibration offset unrelated to
    # discrimination. The pooled number stays the headline because it is what the drawn ROC
    # integrates to (a title that disagrees with its own curve is a defect this harness already
    # fixed once); the folded number says how much of it is that offset.
    auroc_prefix_folded: float
    auroc_combined: float
    incremental_auroc: float
    auprc_prefix: float
    # UNCORRECTED, and reported as such: this depth's own permutation null, gating nothing. It is
    # kept beside the family-wise pair below so the size of the multiplicity correction is readable.
    null_prefix: metrics.NullResult
    null_incremental: metrics.NullResult
    ci_prefix: tuple[float, float]
    ci_incremental: tuple[float, float]
    scores: tuple[float, ...]
    labels: tuple[bool, ...]
    # The fold-honest contrast for the COMBINED model — the arm the prior column enters, and so
    # the only arm where a between-fold offset can masquerade as incremental skill. The prefix had
    # `auroc_prefix_folded` and this did not, which is exactly where the damage was: the
    # pooled/folded gap is the diagnostic for the fold-prevalence artifact, and it was only ever
    # computed on the clean arm. It sits last with a NaN default so it stays purely additive for
    # callers that build a report field-by-field; NaN rather than the pooled value because "not
    # computed" must not read as "no gap". `evaluate_depth` always supplies it.
    auroc_combined_folded: float = float("nan")
    # THE GATE'S ACTUAL NULLS: the max-over-depths distribution this depth's statistic is measured
    # against (module header, "the null is family-wise across depths"). None means the family was
    # this depth alone — a hand-built report, or `evaluate` over a single depth — and the max over a
    # family of one IS the marginal null above, which the gate then reads instead. Defaulted rather
    # than required so callers that build a report field-by-field keep working.
    null_prefix_family: metrics.NullResult | None = None
    null_incremental_family: metrics.NullResult | None = None
    # THE MEASURED NULL FOR AUPRC, drawn from the same shuffles as the two above. It exists because
    # the PR figure needs a no-skill reference and PREVALENCE IS NOT IT on this corpus: prevalence
    # is the theoretical no-skill average precision for exchangeable rows, while these rows cluster
    # by challenge and the whole pipeline is refit per shuffle, so the pipeline's own no-information
    # AUPRC is an empirical quantity, not an arithmetic one. Deliberately NOT family-wise: the
    # multiplicity correction exists because `run_eval._status` takes a max over depths of the two
    # AUROC statistics, and AUPRC gates nothing — a maxT band here would be a correction for a test
    # nobody runs. None means the report was hand-built, and the figure then draws no baseline at
    # all rather than falling back to prevalence.
    null_auprc: metrics.NullResult | None = None
    # Whether this depth's design matrix [features | intercept] has full column rank on the rows it
    # actually admits. A depth whose design is rank-deficient CANNOT be evaluated — its AUROC is
    # arithmetic, not evidence — and the eval must not headline it. Depth 5 on the rebuilt corpus
    # is the measured case: 412 of 414 rows carry one identical vector, so the design ranks 3 of 4
    # and its score is chance by construction (see `features.py` and the strict xfail in
    # `test_features.py`). Defaulted True so hand-built reports stay construction-compatible; a
    # real `evaluate` always supplies the measured value.
    design_full_rank: bool = True

    @property
    def gate_null_prefix(self) -> metrics.NullResult:
        """The prefix-AUROC null the admissibility gate reads — family-wise in a family."""
        return self.null_prefix if self.null_prefix_family is None else self.null_prefix_family

    @property
    def gate_null_incremental(self) -> metrics.NullResult:
        """The incremental-AUROC null the admissibility gate reads — family-wise in a family."""
        if self.null_incremental_family is None:
            return self.null_incremental
        return self.null_incremental_family

    @property
    def prior_comparator(self) -> float:
        """The baseline `incremental_auroc` is measured against: the prior, floored at chance."""
        return max(self.auroc_prior, CHANCE)

    @property
    def mde_auroc(self) -> float:
        """Minimum detectable prefix AUROC: chance plus this corpus's own 95% half-width.

        Clustering by challenge inflates the variance, so a point estimate below this is
        unresolvable here — "not detected" must not be read as "not there".
        """
        low, high = self.ci_prefix
        if math.isnan(low) or math.isnan(high):
            return float("nan")
        return CHANCE + (high - low) / 2.0

    @property
    def has_skill(self) -> bool:
        """Skill means the nulls AND the paired bootstrap all exclude no-information."""
        # A constant score is excluded up front: its AUROC is the 0.5 tie average by definition, so
        # it would clear a band while discriminating nothing. Non-constancy is the precondition for
        # the comparison to mean anything. The bootstrap term is not redundant with the null — the
        # null asks "could shuffled labels do this", the interval asks "would another sample of
        # challenges do this" — and it was computed and then ignored until now.
        # Both nulls are the FAMILY-WISE ones: `run_eval._status` says OK if any reported depth
        # clears, so a per-depth threshold would let the best of the reported depths pass at a
        # nominal 2.5%
        # each. `gate_null_*` is the marginal null exactly when the family really is one depth.
        if len(set(self.scores)) <= 1:
            return False
        return (
            self.gate_null_prefix.beats_null
            and self.gate_null_incremental.beats_null
            and self.ci_incremental[0] > 0.0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "n_rows": self.n_rows,
            # Three separately meaningful exclusions, not one bucket: a corpus gate, a depth the
            # run never reached, and the anti-leak margin. They sum with `n_rows` to the corpus
            # `evaluate_depth` was handed.
            "n_excluded_unstamped": self.n_excluded_unstamped,
            "n_excluded_too_short": self.n_excluded_too_short,
            "n_excluded_by_margin": self.n_excluded_by_margin,
            "n_groups": self.n_groups,
            "base_rate": round(self.base_rate, 4),
            "auroc_prior": round(self.auroc_prior, 4),
            "auroc_prior_leaked": round(self.auroc_prior_leaked, 4),
            "auroc_prefix": round(self.auroc_prefix, 4),
            # Fold-honest contrast to the pooled headline: the gap between them is the
            # between-fold calibration offset, not discrimination. Computed since this harness
            # was written and reported here so a reader can see both.
            "auroc_prefix_folded": round(self.auroc_prefix_folded, 4),
            "auroc_combined": round(self.auroc_combined, 4),
            # The pooled/folded gap on the arm that carries the prior. A large gap here with a
            # small one on the prefix is the signature of the fold-prevalence artifact.
            "auroc_combined_folded": round(self.auroc_combined_folded, 4),
            # The floored baseline `incremental_auroc` is actually measured against, so a reader
            # can tell when the raw prior was anti-predictive and the floor bound instead.
            "prior_comparator": round(self.prior_comparator, 4),
            "incremental_auroc": round(self.incremental_auroc, 4),
            "auprc_prefix": round(self.auprc_prefix, 4),
            # The MEASURED no-skill AUPRC — what the PR figure draws its baseline from. Published
            # so the number on the canvas has a machine-readable source, and so the gap between it
            # and `base_rate` (the theoretical no-skill point) is readable rather than assumed away.
            "null_auprc_prefix": None if self.null_auprc is None else self.null_auprc.to_dict(),
            # The published "this eval can only resolve a detector at AUROC >= X" figure. It was
            # computed and never reported, so the number in the docs had no machine-readable
            # source — the same defect class as `auroc_prefix_folded` above.
            "mde_auroc": round(self.mde_auroc, 4),
            "ci95_auroc_prefix": [round(v, 4) for v in self.ci_prefix],
            "ci95_incremental": [round(v, 4) for v in self.ci_incremental],
            # The UNCORRECTED per-depth nulls, published so the correction's size is readable...
            "null_auroc_prefix": self.null_prefix.to_dict(),
            "null_incremental": self.null_incremental.to_dict(),
            # ...and the family-wise max-statistic nulls the gate actually reads. Their `p_value` is
            # the family-wise ADJUSTED p for this depth; identical to the pair above exactly when
            # the family is one depth. `beats_null` here, not above, is what `has_skill` consults.
            "null_auroc_prefix_familywise": self.gate_null_prefix.to_dict(),
            "null_incremental_familywise": self.gate_null_incremental.to_dict(),
            "has_skill": self.has_skill,
            # Whether this depth could actually be evaluated — see the field's own comment for why
            # a rank-deficient design is arithmetic rather than evidence.
            "design_full_rank": self.design_full_rank,
        }


def leaked_task_prior(rows: Sequence[EvalRow], labels: Sequence[bool]) -> list[float]:
    """DIAGNOSTIC ONLY: leave-one-out per-challenge failure rate over the WHOLE corpus."""
    # Leaked, not conservative: grouped CV puts a row's same-challenge siblings in its own TEST
    # fold, so this reads labels no deployed router can see. Reported only as the contrast that
    # shows how much of the old headline was leakage. `grouped_task_prior` is the headline prior.
    total = len(labels)
    n_failed = sum(labels)
    by_group: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_group.setdefault(row.group, []).append(index)
    out = [0.0] * total
    for indices in by_group.values():
        group_failed = sum(labels[i] for i in indices)
        for i in indices:
            if len(indices) > 1:
                out[i] = (group_failed - labels[i]) / (len(indices) - 1)
            elif total > 1:
                out[i] = (n_failed - labels[i]) / (total - 1)
    return out


def grouped_splits(
    labels: Sequence[bool], groups: Sequence[str]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ONE grouped, LABEL-STRATIFIED partition the prior and the risk model share."""
    # Shared so the prior can never be fitted on rows the model treats as test: a prior estimated
    # on a different partition would re-introduce exactly the leak this module now walls off.
    #
    # STRATIFIED, and that is not cosmetic — it is the fix for a confirmed artifact. Plain
    # GroupKFold balances folds by SIZE and never looks at the label, so on this corpus its fold
    # test base rates spanned 0.4239-0.5652 at depth 5 — a spread of 0.141 (0.153 at depth 10,
    # 0.112 at depth 20). `prior_from_splits` hands every test row its TRAIN-fold base rate, which
    # is the exact arithmetic complement of its own test-fold rate (measured Spearman -1.0000 at
    # every depth, and still -1.0000 stratified — that part is arithmetic, not a defect). The prior
    # column fed into the combined model was therefore a FOLD-ID PROXY carrying the test fold's own
    # prevalence with a minus sign: it added zero within-fold discrimination (within-fold
    # Spearman(prefix, combined) = 1.000000) while moving the POOLED AUROC by +0.13 — pure
    # between-fold accounting sold as incremental skill. Stratifying on the label collapses the
    # spread to 0.011 at depth 5 (0.012 / 0.030 at depths 10 / 20) and removes the artifact at
    # source; what survives is small enough that the folded/pooled contrast below can police it.
    #
    # THERE IS NO FALLBACK, deliberately. An `except ValueError` used to degrade to plain
    # GroupKFold here, on the stated grounds that StratifiedGroupKFold "refuses where GroupKFold did
    # not — a single-class vector, or fewer members of the minority class than folds". Both claims
    # are false of the installed sklearn (1.9.0): a single-class vector returns a partition
    # silently, and one minority member against 5 folds returns one with a UserWarning. It DOES
    # raise, but on a different and far smaller condition — "n_splits=N cannot be greater than the
    # number of members in each class", i.e. a corpus with fewer rows than folds. Measured: an
    # exhaustive sweep of every group partition x every label vector for n <= 7 (126 948
    # configurations) raises on 13 728 of them, all at 2-7 rows; 200 000 randomized trials at
    # n >= MIN_ROWS (40), base rates 0 to 1, raised none; and an instrumented run over the committed
    # corpus at depths 5/10/20 counted 603 stratified splits and 0 raises. The branch was
    # therefore unreachable through `evaluate_depth`, which returns None below MIN_ROWS = 40 — and
    # it was invisible if ever taken: no log, no warning, no report field. A
    # silent degrade to GroupKFold is strictly worse than a raised exception, because GroupKFold IS
    # the artifact the stratification exists to remove, so the fallback would have restored the bug
    # it was written to survive, unannounced. A genuine ValueError propagates.
    y = np.asarray(labels, dtype=int)
    n_splits = min(N_SPLITS, len(set(groups)))
    if n_splits < _MIN_CLASSES:
        return []
    splitter = StratifiedGroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros((len(y), 1)), y, list(groups)))


def prior_from_splits(
    labels: Sequence[bool],
    groups: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    """Per-challenge failure rate estimated from TRAIN folds only, fold by fold."""
    # A challenge unseen in training gets the train-fold base rate. Under the shared grouped
    # partition that is EVERY test row, which is the honest point: a router meeting a new instance
    # has no history on it. The `observed is not None` branch below is therefore UNREACHABLE from
    # `grouped_task_prior` — measured group_overlap = 0 in every fold at every depth, by
    # construction, since a grouped splitter never puts a challenge on both sides. It is retained
    # because `splits` is a caller-supplied argument: this function is exported and is also the
    # place a non-grouped partition would be plugged in, and silently scoring such a row with a
    # corpus-wide base rate would be a worse failure than the branch being dead here.
    y = np.asarray(labels, dtype=int)
    out = np.full(len(y), float(y.mean()))
    for train, test in splits:
        base = float(y[train].mean())
        seen: dict[str, list[int]] = {}
        for i in train:
            seen.setdefault(groups[i], []).append(int(y[i]))
        for i in test:
            observed = seen.get(groups[i])
            out[i] = base if observed is None else sum(observed) / len(observed)
    return out.tolist()


def grouped_task_prior(labels: Sequence[bool], groups: Sequence[str]) -> list[float]:
    """The deployable t=0 prior: `prior_from_splits` over the one shared grouped partition."""
    return prior_from_splits(labels, groups, grouped_splits(labels, groups))


def oof_scores(
    matrix: np.ndarray,
    labels: Sequence[bool],
    groups: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[float]:
    """Out-of-fold P(fail) from a grouped 5-fold logistic model — the continuous risk score.

    Logistic regression on standardized features is already a calibrated-probability model
    (its loss IS the log score), so no post-hoc Platt layer is stacked on top of it.
    """
    # `splits` is an optimisation, never a second partition: it defaults to the one
    # `grouped_splits` would build here anyway. Stratifying made that call ~36x dearer than plain
    # GroupKFold (0.14s against 0.004s on this corpus), and the permutation null re-fits the whole
    # pipeline 200 times per depth, so recomputing the identical partition three times per fit was
    # minutes of pure waste. Callers that pass it MUST pass the shared one — see `_fit_once`.
    y = np.asarray(labels, dtype=int)
    out = np.full(len(y), float(y.mean()))
    for train, test in grouped_splits(labels, groups) if splits is None else splits:
        if len(np.unique(y[train])) < _MIN_CLASSES:
            out[test] = float(y[train].mean())
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(matrix[train], y[train])
        out[test] = model.predict_proba(matrix[test])[:, 1]
    return out.tolist()


def folded_auroc(
    scores: Sequence[float],
    labels: Sequence[bool],
    groups: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    """n-weighted mean of the PER-FOLD AUROCs over the same grouped split `oof_scores` uses."""
    # Pooling every fold's probabilities into one ranking mixes in a between-fold calibration
    # offset, so a row can out-rank another purely for sitting in a higher-prevalence fold.
    # Scoring each fold on its own removes that offset, and the gap between this and the pooled
    # figure is how much of the pooled number was accounting rather than discrimination.
    #
    # Stratifying the CV (see `grouped_splits`) shrank the room that offset has: on this corpus the
    # fold test base rates now span 0.5054-0.5165 at depth 5 (0.5270-0.5395 at depth 10,
    # 0.5417-0.5714 at depth 20) against 0.4239-0.5652 / 0.4474-0.6000 / 0.5000-0.6122 under the
    # old size-only GroupKFold. Small is not zero, which is why both the prefix and the combined
    # arm report this contrast.
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    total = 0.0
    weight = 0
    for _train, test in grouped_splits(labels, groups) if splits is None else splits:
        if len(np.unique(y[test])) < _MIN_CLASSES:
            continue  # a single-class fold has no ROC to integrate; it is skipped, not scored 0.5
        total += metrics.auroc(s[test].tolist(), [bool(v) for v in y[test]]) * len(test)
        weight += len(test)
    return total / weight if weight else CHANCE


@dataclass(frozen=True)
class _Fit:
    """One end-to-end pass of the pipeline over a single label vector."""

    prior: list[float]
    prefix: list[float]
    combined: list[float]
    # The one partition every arm above was estimated on, carried so downstream fold-wise
    # statistics (`folded_auroc`) score on exactly it rather than rebuilding an identical copy.
    splits: list[tuple[np.ndarray, np.ndarray]]
    auroc_prior: float
    auroc_prior_leaked: float
    auroc_prefix: float
    auroc_combined: float

    @property
    def prior_comparator(self) -> float:
        """The baseline the increment is measured against: the prior, floored at chance."""
        return max(self.auroc_prior, CHANCE)

    @property
    def incremental(self) -> float:
        # Floored, per the module header: an anti-predictive prior (this corpus's is 0.4873-0.4959
        # across depths) would otherwise donate `0.5 - auroc_prior` of manufactured skill to every
        # increment. The deficit is small now that the CV is stratified — it was ~0.08 at depth 5
        # under the old partition — but the floor is what keeps it from being counted at all.
        return self.auroc_combined - self.prior_comparator


def _fit_once(
    rows: Sequence[EvalRow], base: np.ndarray, labels: Sequence[bool], groups: Sequence[str]
) -> _Fit:
    """Fit the prior, the prefix model and the combined model once for this label vector."""
    # ONE partition, built once and shared by all three arms — the "one partition" guarantee the
    # module rests on, now enforced by construction rather than by three calls agreeing.
    splits = grouped_splits(labels, groups)
    prior = prior_from_splits(labels, groups, splits)
    prefix = oof_scores(base, labels, groups, splits)
    combined = oof_scores(np.column_stack([base, np.asarray(prior)]), labels, groups, splits)
    return _Fit(
        prior=prior,
        prefix=prefix,
        combined=combined,
        splits=splits,
        auroc_prior=metrics.auroc(prior, labels),
        auroc_prior_leaked=metrics.auroc(leaked_task_prior(rows, labels), labels),
        auroc_prefix=metrics.auroc(prefix, labels),
        auroc_combined=metrics.auroc(combined, labels),
    )


@dataclass(frozen=True)
class _Prepared:
    """One depth's sanctioned rows and its observed fit — everything the nulls are drawn around."""

    depth: int
    census: CorpusCensus
    labels: list[bool]
    groups: list[str]
    base: np.ndarray
    fit: _Fit
    design_full_rank: bool


def _prepare(trajectories: Sequence[Trajectory], depth: int) -> _Prepared | None:
    """Gate the corpus at `depth` and fit it once, or None when the data cannot support it."""
    # The corpus this scores is decided by `corpus_census`, never by the caller — see the gate's own
    # comment for the false "significant" result an ungated direct call already produced.
    census = corpus_census(trajectories, depth)
    rows = census.rows
    labels = [r.failed for r in rows]
    if len(rows) < MIN_ROWS or len(set(labels)) < _MIN_CLASSES:
        return None
    groups = [r.group for r in rows]
    base = np.asarray([r.features for r in rows], dtype=float)
    # The same rank the `test_features.py` guard checks: [features | intercept] must span its own
    # column space, or a depth with 3 distinct rows (depth 5 today) reports an arithmetic AUROC.
    design = np.column_stack([base, np.ones(len(base))])
    full_rank = int(np.linalg.matrix_rank(design)) == design.shape[1]
    return _Prepared(
        depth, census, labels, groups, base, _fit_once(rows, base, labels, groups), full_rank
    )


def evaluate_depth(
    trajectories: Sequence[Trajectory],
    depth: int,
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
) -> DepthReport | None:
    """One decision depth, scored as a family of ONE — no multiplicity correction to apply."""
    # A family of one is the honest description, not a shortcut: a caller asking about a single
    # depth runs a single test, and the max over one statistic IS that statistic. Routed through
    # `evaluate` so there is one implementation of the pipeline rather than two that must agree.
    reports = evaluate(trajectories, [depth], n_permutations=n_permutations, seed=seed)
    return reports[0] if reports else None


@dataclass(frozen=True)
class _SharedNull:
    """One family of label shuffles: each depth's own draws, plus the max across depths."""

    # Per depth, one (prefix AUROC, incremental AUROC, prefix AUPRC) triple per shuffle. AUPRC
    # rides along on the SAME shuffles rather than being drawn separately: the fits are what cost
    # anything here, and a second set of shuffles would put the PR figure's baseline on a different
    # randomization than the ROC figure's band on the same canvas set.
    per_depth: dict[int, list[tuple[float, float, float]]]
    max_prefix: list[float]
    max_incremental: list[float]


@dataclass(frozen=True)
class _Union:
    """The depths' rows merged into one label vector, with each depth's slice back into it."""

    labels: list[bool]
    groups: list[str]
    index: dict[int, list[int]]


def _union_index(prepared: Sequence[_Prepared]) -> _Union:
    """Merge every depth's rows into one vector, keeping each depth's row order as an index list."""
    # The row sets are nested (admission needs `depth + MIN_WITHHELD` scorable steps, monotone in
    # depth), so with depths given shallowest-first the union IS the shallowest depth's row set in
    # its own order. Built from the actual rows rather than assuming that, because `depths` is
    # caller-supplied and may arrive in any order.
    position: dict[str, int] = {}
    union = _Union(labels=[], groups=[], index={})
    for prep in prepared:
        picked: list[int] = []
        for row in prep.census.rows:
            if row.trajectory_id not in position:
                position[row.trajectory_id] = len(union.labels)
                union.labels.append(row.failed)
                union.groups.append(row.group)
            picked.append(position[row.trajectory_id])
        union.index[prep.depth] = picked
    return union


def _shared_null_draws(
    prepared: Sequence[_Prepared], union: _Union, *, n_permutations: int, seed: int
) -> _SharedNull:
    """Draw each shuffle ONCE, score it at every depth, and keep both the per-depth and max draws.

    The WHOLE pipeline is re-fit per shuffle — prior included — so the null absorbs every
    optimism the fitting itself introduces, not just the ranking step.
    """
    # ONE shuffle per replicate, shared across depths: that is what makes the max distribution the
    # exact family-wise reference (module header). Drawing independently per depth would leave the
    # depths' nulls uncoupled and the max of them meaningless as a joint statistic.
    rng = random.Random(seed)
    shared = _SharedNull({p.depth: [] for p in prepared}, [], [])
    while len(shared.max_prefix) < n_permutations:
        shuffled = permute_within_groups(union.labels, union.groups, rng)
        drawn = [[shuffled[i] for i in union.index[p.depth]] for p in prepared]
        if any(len(set(labels)) < _MIN_CLASSES for labels in drawn):
            continue
        fits = [
            _fit_once(p.census.rows, p.base, labels, p.groups)
            for p, labels in zip(prepared, drawn, strict=True)
        ]
        for prep, fit, drawn_labels in zip(prepared, fits, drawn, strict=True):
            shared.per_depth[prep.depth].append(
                (fit.auroc_prefix, fit.incremental, metrics.auprc(fit.prefix, drawn_labels))
            )
        shared.max_prefix.append(max(f.auroc_prefix for f in fits))
        shared.max_incremental.append(max(f.incremental for f in fits))
    return shared


def _depth_report(prep: _Prepared, shared: _SharedNull, *, seed: int) -> DepthReport:
    """Assemble one depth's report from its observed fit and the shared null family."""
    fit = prep.fit
    labels = prep.labels
    groups = prep.groups
    draws = shared.per_depth[prep.depth]
    auprc_prefix = metrics.auprc(fit.prefix, labels)
    return DepthReport(
        depth=prep.depth,
        n_rows=len(labels),
        n_excluded_unstamped=prep.census.n_unstamped,
        n_excluded_too_short=prep.census.n_too_short,
        n_excluded_by_margin=prep.census.n_by_margin,
        n_groups=len(set(groups)),
        base_rate=metrics.prevalence(labels),
        auroc_prior=fit.auroc_prior,
        auroc_prior_leaked=fit.auroc_prior_leaked,
        auroc_prefix=fit.auroc_prefix,
        auroc_prefix_folded=folded_auroc(fit.prefix, labels, groups, fit.splits),
        auroc_combined=fit.auroc_combined,
        auroc_combined_folded=folded_auroc(fit.combined, labels, groups, fit.splits),
        incremental_auroc=fit.incremental,
        auprc_prefix=auprc_prefix,
        null_prefix=metrics.permutation_null(fit.auroc_prefix, [d[0] for d in draws]),
        null_incremental=metrics.permutation_null(fit.incremental, [d[1] for d in draws]),
        null_prefix_family=metrics.permutation_null(fit.auroc_prefix, shared.max_prefix),
        null_incremental_family=metrics.permutation_null(fit.incremental, shared.max_incremental),
        null_auprc=metrics.permutation_null(auprc_prefix, [d[2] for d in draws]),
        # CONDITIONAL ON THIS ONE FIT, exactly as `ci_incremental` is: `grouped_bootstrap_ci`
        # resamples indices into `fit.prefix`, which was fitted once on the full corpus. Refitting
        # per replicate widens this interval by a measured 1.19-1.44x — see `_incremental_ci` for
        # the mechanism, the rest of the measurements, and the trap in implementing the refit.
        ci_prefix=metrics.grouped_bootstrap_ci(
            fit.prefix, labels, groups, metrics.auroc, seed=seed
        ),
        ci_incremental=_incremental_ci(fit, labels, groups, seed),
        scores=tuple(fit.prefix),
        labels=tuple(labels),
        design_full_rank=prep.design_full_rank,
    )


def permute_within_groups(
    labels: Sequence[bool], groups: Sequence[str], rng: random.Random
) -> list[bool]:
    """Shuffle labels INSIDE each challenge, so every group's outcome multiset is preserved."""
    # A global shuffle destroys the challenge-level clustering of outcomes, so the null's fold base
    # rates — and with them the deployable prior — stop matching the observation's and the shuffle
    # nulls the whole partition rather than the prefix's contribution to it. Measured at depth 5:
    # the prior is invariant here (sd 0.0000 over 200 draws) against sd 0.0026 globally, while the
    # PREFIX arm's null sd is 0.0048 here against 0.0305 globally — that gap is the clustering, not
    # detector headroom. The retracted headroom-asymmetry argument is in the module header.
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


def _incremental_ci(
    fit: _Fit, labels: Sequence[bool], groups: Sequence[str], seed: int
) -> tuple[float, float]:
    """Challenge-level PAIRED bootstrap CI for the incremental AUROC, CONDITIONAL on this one fit.

    The resample draws row INDICES into scores `_fit_once` produced once, so it does not re-fit;
    that narrows the interval, and by how much — and in which direction — is stated below.
    """
    # Paired matters: both arms are scored on the SAME resampled rows, so the interval reflects the
    # variance of the DIFFERENCE rather than the sum of two independent variances. It must also
    # bracket the same quantity the point estimate reports — combined minus the FLOORED prior, not
    # prefix minus prior — or the interval and the headline would be describing different
    # statistics.
    #
    # WHAT THIS INTERVAL IS, AND WHAT IT IS NOT — the same disclosure applies to the `ci_prefix`
    # built by `grouped_bootstrap_ci` in `_depth_report`, which resamples the same way.
    # `evaluate_depth` calls `_fit_once` exactly ONCE; both bootstraps then resample indices into
    # the already-fitted `fit.combined` / `fit.prior` / `fit.prefix` vectors, B = 1000 times. The
    # permutation null beside them does the opposite — `_shared_null_draws` re-runs the whole
    # pipeline per shuffle, and the module header advertises exactly that — so a reader is entitled
    # to assume the bootstrap resamples the procedure too. It does not. It brackets the sampling
    # variability of the STATISTIC GIVEN THIS FIT, not of the train-and-evaluate procedure as a
    # whole, and the model-fitting variance is therefore missing from it.
    #
    # HOW MUCH IS MISSING, measured on the committed corpus (2026-07-31, pre-rebuild) against a
    # group-id-preserving refit-per-replicate variant: the prefix-AUROC interval is 1.19-1.44x wider
    # when the fit is resampled with it, the incremental interval 1.05-1.13x wider, and `mde_auroc`
    # moves 0.582 -> 0.612 at depth 5, 0.594 -> 0.613 at depth 10 and 0.602 -> 0.647 at depth 20.
    # It is verdict-neutral on THIS corpus — `ci_incremental[0] > 0.0` stays false at every depth
    # either way — which is why the refit is disclosed here rather than shipped in a hurry.
    #
    # WHICH WAY THE BIAS POINTS, and why it is the direction that matters: `ci_incremental[0] > 0.0`
    # is one of the three `has_skill` conditions, so an interval that is too NARROW is biased toward
    # DECLARING skill. This understates a false-positive risk on the one gate the escalation thesis
    # turns on; it does not understate a false negative.
    #
    # MITIGATION, stated accurately rather than as an excuse: what is being resampled is genuine
    # out-of-fold StratifiedGroupKFold prediction, so the POINT ESTIMATE carries no in-sample
    # optimism, and this is a legitimate conditional-on-this-fit interval. It is simply not the
    # interval for the whole train-and-evaluate procedure, and the report does not say which it is.
    #
    # ⚠ IF YOU IMPLEMENT THE REFIT, READ THIS FIRST — the obvious route ships a worse defect than
    # the one it fixes. Do NOT re-fit by treating bootstrap-duplicated challenges as if they were
    # distinct challenges (e.g. by relabelling each draw's copies `c0#1`, `c0#2`, ...). That hands
    # `grouped_splits` a group key it thinks is new, so the SAME challenge lands in train and test
    # inside one replicate — precisely the leak the grouped partition exists to prevent. Measured
    # on this corpus: prefix AUROC inflates by +0.07 to +0.10, the incremental FLIPS POSITIVE
    # (-0.0576 -> +0.0170 at depth 5, -0.0651 -> +0.0245 at depth 20), and the interval it produces
    # is NARROWER than the defect it was meant to fix (0.81-0.83x) — it would manufacture apparent
    # skill and then report it with tighter error bars. A correct refit must keep each row's
    # ORIGINAL challenge id so all duplicates of a challenge stay ONE cluster and stay on one side
    # of every fold boundary.
    draws: list[float] = []
    for picked in metrics.grouped_resamples(groups, seed=seed):
        sample_labels = [labels[i] for i in picked]
        if not any(sample_labels) or all(sample_labels):
            continue
        combined = [fit.combined[i] for i in picked]
        prior = [fit.prior[i] for i in picked]
        # Floored per resample, exactly as the point estimate is: an interval built on the
        # unfloored difference would bracket a different statistic than the headline.
        comparator = max(metrics.auroc(prior, sample_labels), CHANCE)
        draws.append(metrics.auroc(combined, sample_labels) - comparator)
    return metrics.bootstrap_ci(draws)


def evaluate(
    trajectories: Sequence[Trajectory],
    depths: Sequence[int] = features.DEFAULT_DEPTHS,
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
) -> list[DepthReport]:
    """Every requested depth the corpus supports, gated against ONE family-wise permutation null."""
    # The depths are a FAMILY, because `run_eval._status` reads them as one: OK if any of them
    # clears. So the shuffles are drawn once and scored at all of them, and the max over depths is
    # the reference each depth's observed statistic is gated against (module header). The cost is
    # unchanged — `n_permutations` shuffles x len(depths) fits, exactly as one independent null per
    # depth cost — so this is a correction, not a budget increase.
    prepared = [p for p in (_prepare(trajectories, d) for d in depths) if p is not None]
    if not prepared:
        return []
    union = _union_index(prepared)
    shared = _shared_null_draws(prepared, union, n_permutations=n_permutations, seed=seed)
    return [_depth_report(p, shared, seed=seed) for p in prepared]


__all__ = [
    "CHANCE",
    "MIN_ROWS",
    "N_SPLITS",
    "CorpusCensus",
    "DepthReport",
    "corpus_census",
    "evaluate",
    "evaluate_depth",
    "grouped_splits",
    "grouped_task_prior",
    "leaked_task_prior",
    "oof_scores",
    "permute_within_groups",
    "prior_from_splits",
]

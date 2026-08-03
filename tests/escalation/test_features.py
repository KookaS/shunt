"""The two structural anti-leak walls: no future length, no terminal step (nor approach to it)."""

# Both walls encode defects that were shipped and measured. They are regression walls, not
# coverage: if either assertion can be deleted without another test failing, the defect class can
# return.
#
# Wall 2 is walled at TWO points, because it was shipped half-built: `scorable_steps` clips the one
# label-stamped step, and `MIN_WITHHELD` keeps the prefix away from it. The clip alone passed every
# test here while depth 40 scored AUROC 0.770 off terminal proximity, so a test that only pins the
# clip is not a test of wall 2 — `_LONG` below exists so these fixtures cannot pass vacuously.

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from benchmark.escalation import features, prefix_eval
from benchmark.escalation.schema import Trajectory, load_jsonl
from tests.escalation import frozen_corpus
from tests.escalation.factories import make_step, make_trajectory

_LIVE_DIR = Path(__file__).resolve().parents[2] / "benchmark/escalation/data/live"

# The margin as MEASURED, still a literal and still not `features.MIN_WITHHELD`. Every fixture
# below is sized from this rather than from the constant, because a test that derives its own
# boundary from the constant it is testing moves with the bug: setting `MIN_WITHHELD = 0` keeps
# such a test green while re-admitting every leaked row. That mutant was run and did survive the
# self-referential version of these tests, which is why the literal exists. It now lives in
# `frozen_corpus`, the one copy, so the frozen fixture's expected census and these walls cannot
# drift apart.
_MEASURED_MARGIN = frozen_corpus.MEASURED_MARGIN

# Total steps that comfortably clear depth 5 + the margin + the terminal step. Every fixture below
# that means to PRODUCE a feature vector must be at least this long, or the margin excludes it and
# the assertion becomes `None == None` — which is how a vacuous test would pass.
_LONG = 5 + _MEASURED_MARGIN + 1


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
    short = _run(_LONG, resolved=True)
    long = _run(_LONG * 3, resolved=True)
    assert features.extract_features(short, 5) is not None  # not vacuously equal via None
    assert features.extract_features(short, 5) == features.extract_features(long, 5)


def test_features_never_read_the_label_stamped_terminal_step() -> None:
    # WALL 2. `_stamp_terminal` writes the harness verdict onto the last step: AUROC(last step
    # failed -> terminal failure) is 1.00 including it and 0.56 excluding it. A prefix that could
    # reach step n-1 would read the answer verbatim.
    failed = _run(_LONG, resolved=False)
    resolved = _run(_LONG, resolved=True)
    assert failed.steps[-1].success != resolved.steps[-1].success  # the leak exists in the data
    assert features.extract_features(failed, 5) is not None  # not vacuously equal via None
    assert features.extract_features(failed, 5) == features.extract_features(resolved, 5)


def test_prefix_stops_short_of_the_terminal_step() -> None:
    traj = _run(_LONG, resolved=False)
    assert len(features.scorable_steps(traj)) == len(traj.steps) - 1
    assert features.extract_features(traj, 5) is not None


@pytest.mark.parametrize("depth", features.DEFAULT_DEPTHS)
def test_a_prefix_that_nearly_reaches_the_terminal_step_is_excluded(depth: int) -> None:
    # THE WALL THIS FILE WAS MISSING. Clipping the one stamped step bounds the leak only while the
    # prefix is short relative to the run; the admission test was `len(steps) >= depth`, so a run
    # of exactly `depth + 1` steps was admitted with its prefix ending one step from the verdict.
    # At depth 40 that scored grouped-OOF AUROC 0.770, falling to 0.608 once 20 steps were withheld
    # — the model was reading the outcome, not predicting it. A trajectory qualifies only when
    # MIN_WITHHELD non-terminal steps remain UNREAD after the prefix.
    #
    # The boundary is pinned EXACTLY, from both sides, because an off-by-one here is the whole bug:
    # one step fewer than the margin is still leakage.
    just_short = _run(depth + _MEASURED_MARGIN, resolved=False)  # margin - 1 withheld
    exact = _run(depth + _MEASURED_MARGIN + 1, resolved=False)  # exactly margin withheld
    assert len(features.scorable_steps(just_short)) == depth + _MEASURED_MARGIN - 1
    assert len(features.scorable_steps(exact)) == depth + _MEASURED_MARGIN
    assert features.extract_features(just_short, depth) is None
    assert features.extract_features(exact, depth) is not None
    # `build_rows` must inherit the rule rather than re-implement the admission test.
    assert features.build_rows([just_short], depth) == []
    assert len(features.build_rows([exact], depth)) == 1


def test_the_withheld_margin_is_absolute_not_proportional() -> None:
    # A proportional margin (withhold a FRACTION of the run) would be a function of `n_steps`, so
    # it would re-admit future length through the admission test and break wall 1. Absolute means
    # the SAME number of steps is withheld whatever the run length: a long run and a short one that
    # both clear the bar are both admitted, and clearing it is decided by a difference, not a ratio.
    at_bar = _run(20 + _MEASURED_MARGIN + 1, resolved=False)
    far_past = _run((20 + _MEASURED_MARGIN + 1) * 5, resolved=False)
    assert features.extract_features(at_bar, 20) is not None
    assert features.extract_features(far_past, 20) is not None
    # A 10% proportional rule would reject `at_bar` at depth 20 while accepting `far_past`; an
    # absolute one accepts both, and their prefixes are identical (wall 1).
    assert features.extract_features(at_bar, 20) == features.extract_features(far_past, 20)


def test_the_margin_is_pinned_to_its_measured_value() -> None:
    # Loosening the margin is the mutation that re-opens the leak, and NO fixture-shaped test can
    # catch it: fixtures sized from the constant move with it, and the corpus-floor test below only
    # catches the margin getting TIGHTER (more rows always clears a floor). So the value is pinned
    # directly, and changing it must mean re-running the measurement rather than editing a number.
    #
    # WHERE 22 COMES FROM: on the 216 stamped runs with >= 45 scorable steps (a fixed population,
    # so the decay is not selection), the AUROC of a single step's `success` against the final
    # grade is 0.77-0.79 for every step within 10 of the end, 0.652 at distance 18, 0.607 at 21,
    # and first becomes statistically indistinguishable from that population's own far-field floor
    # (0.554, the mean over distances 30-42) at distance 23 — challenge-level bootstrap, 400
    # resamples, excess CI first including 0 at 23 and staying flat out to 42. The last step a
    # prefix may read sits at distance `withheld + 1`, so 22 withheld puts it at 23.
    assert features.MIN_WITHHELD == _MEASURED_MARGIN


@pytest.mark.parametrize("depth", features.DEFAULT_DEPTHS)
def test_every_admitted_row_really_withholds_the_tail_on_the_real_corpus(depth: int) -> None:
    # The end-to-end invariant, asserted against the literal rather than the constant: no admitted
    # trajectory may have fewer than the measured margin of non-terminal steps left unread. This is
    # what fails when the margin is loosened, and it is corpus-growth-safe — adding trajectories
    # cannot break it, only a broken admission rule can.
    admitted = [
        t
        for t in _live_corpus()
        if features.is_stamped(t) and features.extract_features(t, depth) is not None
    ]
    assert admitted, f"depth {depth} admitted nothing"
    worst = min(len(features.scorable_steps(t)) - depth for t in admitted)
    assert worst >= _MEASURED_MARGIN, (
        f"depth {depth}: a row was admitted with only {worst} non-terminal steps withheld, "
        f"below the measured {_MEASURED_MARGIN} — the prefix is reading terminal proximity"
    )


def test_the_margin_cannot_be_overridden_from_the_call_site() -> None:
    # It is a module constant, not a parameter with a default, so no caller can widen the admission
    # test back to `len(steps) >= depth`. If this ever becomes a keyword argument, a caller can
    # silently re-admit every leaked row and every other wall in this file still passes.
    assert isinstance(features.MIN_WITHHELD, int)
    params = inspect.signature(features.extract_features).parameters
    assert list(params) == ["traj", "depth"]
    assert list(inspect.signature(features.build_rows).parameters) == ["trajectories", "depth"]


def test_features_move_when_the_prefix_itself_differs() -> None:
    # The walls must not be a constant function: real prefix evidence still has to register.
    quiet = make_trajectory(
        [make_step(step_index=i, decision_index=i) for i in range(_LONG)], terminal_resolved=True
    )
    thrashing = make_trajectory(
        [
            make_step(step_index=i, decision_index=i, success=False, failing_check_id="pkg::t")
            for i in range(_LONG)
        ],
        terminal_resolved=True,
    )
    assert features.extract_features(quiet, 5) is not None
    assert features.extract_features(quiet, 5) != features.extract_features(thrashing, 5)


# DEPTH 5's DESIGN IS DEGENERATE ON THE REBUILT CORPUS, and that is a fact about the DATA rather
# than a redundant column. The two columns that used to share the blame — `max_key_repeat_rate`
# (== `fail_rate` on 414/414 rows) and `distinct_check_id_rate` (constant) — have been REMOVED
# from `FEATURE_NAMES` (see features.py), and the design STILL cannot rank full. Measured
# 2026-08-02 over the 414 admitted depth-5 rows: 412 carry the IDENTICAL vector (1.0, 0.0, 0.2),
# leaving three distinct rows in the whole design. The mechanism is not a coding error — in the
# first five replayed steps the agent has not yet changed the workspace, so the selector fails
# every time on the same test id and `fail_rate` collapses to 1.0 while the other two rates
# collapse to their own constants. With three distinct rows the design cannot rank above 3 of 4
# whatever its columns are, so no column edit repairs it, and restating the assertion as "as
# full-rank as the rows allow" would only hide it.
#
# DEPTH 5 LEFT `DEFAULT_DEPTHS` on 2026-08-02 (features.py records the decision): a reported
# depth must be measurable, and this one is not. The finding is NOT dropped with the depth — it
# stays pinned here as its own strict xfail, so the day depth 5 regains rank (a corpus change, an
# admission change) the XPASS turns the suite red and forces the ladder to be reconsidered.
# `strict=True` is the ratchet: an xfail that would have XPASSed had it stayed on the ladder is
# exactly the same guard, moved to where it still bites.
_DEGENERATE_AT_DEPTH_5 = pytest.mark.xfail(
    strict=True,
    reason="412 of 414 depth-5 rows carry one identical feature vector on the rebuilt corpus, so "
    "the design cannot rank above 3 of 4 — a corpus property, not a redundant column; depth 5 "
    "left DEFAULT_DEPTHS for this reason and this xfail keeps the finding from returning unseen",
)


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


@_DEGENERATE_AT_DEPTH_5
def test_the_dropped_depth_5_remains_rank_deficient_on_the_real_corpus() -> None:
    # Depth 5 left `DEFAULT_DEPTHS` because its design cannot rank full — the corpus property this
    # xfail pins. Kept as its OWN test (not parametrized over the ladder) so the reason it is not
    # reported cannot silently come back as a green full-rank row.
    trajectories = [t for t in _live_corpus() if features.is_stamped(t)]
    rows = features.build_rows(trajectories, 5)
    assert len(rows) >= prefix_eval.MIN_ROWS
    design = np.column_stack(
        [np.asarray([r.features for r in rows], dtype=float), np.ones(len(rows))]
    )
    assert int(np.linalg.matrix_rank(design)) == design.shape[1]


@pytest.mark.parametrize("depth", features.DEFAULT_DEPTHS)
def test_the_margin_does_not_empty_the_eval_on_the_real_corpus(depth: int) -> None:
    # The withheld-tail rule is a FILTER on real rows, so the failure mode it introduces is the
    # opposite of the leak: silently starving a depth until the eval reports on nothing.
    #
    # THIS USED TO PIN THE MEASURED SURVIVOR COUNTS as floors — 452/372/239, taken from a corpus
    # that has since been rebuilt to 414/344/228. Those are live-corpus numbers, so CORRECTING the
    # corpus turned the suite red while the margin was doing exactly its job, which is the failure
    # mode backwards. What "does not empty the eval" means is framework-defined and does not move
    # with the data: `evaluate_depth` refuses a depth below `MIN_ROWS` rows or `N_SPLITS` groups,
    # so those two constants ARE the floor, and a corpus that starves a depth still fails here.
    # The exact survivor set is pinned on `frozen_corpus` (next test) and the live census is
    # published to the report JSON, where it is reviewed rather than asserted.
    rows = features.build_rows([t for t in _live_corpus() if features.is_stamped(t)], depth)
    assert len(rows) >= prefix_eval.MIN_ROWS, f"depth {depth}: {len(rows)} rows starves the eval"
    assert len({r.group for r in rows}) >= prefix_eval.N_SPLITS


@pytest.mark.parametrize("depth", features.DEFAULT_DEPTHS)
def test_the_margin_admits_exactly_the_runs_with_room_to_spare_on_the_frozen_corpus(
    depth: int,
) -> None:
    # The sharp half, on a corpus that cannot move: `build_rows` must admit exactly the runs whose
    # scorable length clears `depth + 22`, named individually rather than counted. `frozen_corpus`
    # straddles every bar from both sides (26/27, 31/32, 41/42), so an off-by-one in the admission
    # rule renames a member of this set instead of merely shifting a total — and a set comparison
    # says WHICH run moved, which a count never could.
    stamped = [t for t in frozen_corpus.build() if features.is_stamped(t)]
    admitted = {r.trajectory_id for r in features.build_rows(stamped, depth)}
    assert admitted == frozen_corpus.admitted(depth)
    assert admitted, f"the margin emptied the frozen corpus at depth {depth}"


# THE SELECTION-BIAS BOUND ON `DEFAULT_DEPTHS`, in admitted-base-rate points against the corpus's
# own base rate. This is what replaced `assert features.DEFAULT_DEPTHS == (5, 10, 20)` — a value-pin
# that justified nothing about WHICH depths and would have passed identically on (30, 40, 45), where
# the harness publishes a positive incremental at every depth for reasons that have nothing to do
# with prefix evidence.
#
# WHY A BASE-RATE BOUND IS THE RIGHT PIN. Admission at depth d needs `depth + MIN_WITHHELD` scorable
# steps, and run length is outcome-correlated, so a deeper depth does not look further into the same
# runs — it swaps in a population enriched for failures. That enrichment is directly measurable as
# the gap between the admitted rows' base rate and the corpus's, so the magic number becomes a
# checkable bound, and it is the rule that stays correct after the corpus is rebuilt (unlike any
# specific depth, count or AUROC).
#
# WHY 0.12, measured 2026-07-31 on the pre-rebuild corpus. The reported depths sit at +0.049 /
# +0.072 / +0.092 against a corpus base rate of 0.460 over 791 stamped runs, so the worst of them
# uses 77% of the bound — genuine margin, and enough that the scheduled corpus rebuild moving these
# figures does not turn the test into a flake. Upward, the bound first refuses depth 32 (+0.131),
# one depth past where the incremental first turns positive on this corpus (depth 31), and it
# refuses everything deeper: +0.15 at depth 35, +0.20 at 40, +0.23 at 45. So it admits the whole
# window where the incremental is negative and refuses the window where selection drives it
# positive, with one depth of slack — stated rather than tuned away, because tightening it to catch
# depth 31 exactly would be fitting the tolerance to a corpus that is about to be regenerated.
# 0.12 on a 0.460 base rate is also a 26% relative shift in the admitted population's outcome mix,
# which is where "the depth is choosing the runs" stops being a caveat and becomes the finding.
#
# WHAT THE REBUILD ACTUALLY DID (re-measured 2026-08-02, after the state-capture marking). The
# prediction in the paragraph above — that the rebuild moving these figures would not turn this into
# a flake — did NOT hold. The stamped corpus is now 727 runs at a 0.4209 base rate (marking a
# lost-capture step unmeasured takes it out of `is_stamped`), and the reported depths sat at
# +0.0525 / +0.0820 / +0.1186. Depth 20 therefore used 98.8% of the old absolute bound, and the
# first depth the bound refused moved from 32 down to 26 — the "one depth of slack" is spent. The
# direction is the mechanism, not noise: admission at depth 20 still conditions on run length, and
# the corrected corpus has fewer stamped runs to dilute it.
# THE BOUND IS NOT WIDENED FOR THIS — it is REEXPRESSED AS THE RELATIVE NUMBER IT ALWAYS WAS, and
# the ladder is cut to match. 0.12 was the ABSOLUTE encoding of a 26% RELATIVE shift on a 0.460
# base rate; on 0.4209 that same 26% is 0.109, so the old absolute number was looser than its own
# justification and depth 20 (+0.1186 = 28.2% relative) was already past the criterion the comment
# stated. Writing the bound as the relative 26% it claimed to encode closes that gap by
# construction — the bound moves with the corpus instead of silently drifting from its meaning —
# and depth 20, which the relative bound now refuses, has LEFT `DEFAULT_DEPTHS` (features.py's
# ladder comment records the decision). Depths 5 and 10 sit at +12.5% and +19.5% relative, inside
# the 26%, which is why the ladder ends where it does. Raising 0.26 would be fitting the bound to
# the data it exists to police; cutting the ladder is not.
_BASE_RATE_TOLERANCE_REL = 0.26


def _stamped_corpus():  # type: ignore[no-untyped-def]
    """The corpus every prefix number is measured on — the population the depths sample from."""
    return [t for t in _live_corpus() if features.is_stamped(t)]


def _corpus_base_rate(trajectories) -> float:  # type: ignore[no-untyped-def]
    """Share of the WHOLE stamped corpus that failed — the reference no depth can move."""
    # Absolute, never "relative to the shallowest reported depth": a reference read off the tuple
    # under test moves with it, so (30, 32, 34) would score a drift near zero and pass. The corpus
    # is the population the eval claims to describe, and it does not depend on the depths at all.
    return sum(not t.header.terminal_resolved for t in trajectories) / len(trajectories)


def _admitted_base_rate(trajectories, depth: int) -> float | None:  # type: ignore[no-untyped-def]
    """Failure rate among the rows `depth` actually admits, or None when it admits nothing."""
    rows = features.build_rows(trajectories, depth)
    return sum(r.failed for r in rows) / len(rows) if rows else None


@pytest.mark.parametrize("depth", features.DEFAULT_DEPTHS)
def test_every_reported_depth_admits_a_population_close_to_the_corpus_base_rate(depth: int) -> None:
    stamped = _stamped_corpus()
    corpus = _corpus_base_rate(stamped)
    admitted = _admitted_base_rate(stamped, depth)
    assert admitted is not None, f"depth {depth} admitted nothing"
    bound = _BASE_RATE_TOLERANCE_REL * corpus
    assert abs(admitted - corpus) <= bound, (
        f"depth {depth} admits a base rate of {admitted:.4f} against the corpus's {corpus:.4f} "
        f"— a drift of {admitted - corpus:+.4f} "
        f"({abs(admitted - corpus) / corpus:.1%} relative), past the "
        f"{_BASE_RATE_TOLERANCE_REL:.0%} relative selection-bias bound ({bound:.4f} absolute on "
        "this base rate). At that drift the depth is selecting a different population rather than "
        "reading a longer prefix, and its incremental AUROC measures the selection"
    )


def test_the_base_rate_bound_actually_refuses_a_selection_driven_depth() -> None:
    # Without this, the bound above could be satisfied by a tolerance so loose it forbids nothing —
    # the exact failure of the value-pin it replaces. The depths past ~30 are where the sweep found
    # the incremental climbing to +0.18, and the bound must refuse them.
    stamped = _stamped_corpus()
    corpus = _corpus_base_rate(stamped)
    deep = {
        depth: rate
        for depth in range(30, 46)
        if (rate := _admitted_base_rate(stamped, depth)) is not None
    }
    assert deep, "no deep depth is measurable on this corpus — the bound cannot be shown to bite"
    assert max(rate - corpus for rate in deep.values()) > _BASE_RATE_TOLERANCE_REL * corpus
    # ...and in the direction the mechanism predicts: admission conditions on run length, run length
    # is outcome-correlated, so deeper means MORE failures, not merely different ones. This half is
    # the claim that survives the corpus rebuild even when every figure above moves.
    shallow = _admitted_base_rate(stamped, min(features.DEFAULT_DEPTHS))
    assert shallow is not None
    assert max(deep.values()) > shallow


def test_the_depth_ladder_carries_its_justification_in_the_source() -> None:
    # `DEFAULT_DEPTHS` decides the SIGN of the published verdict and used to carry a five-word
    # comment. The three things a reader needs are the ladder's rationale, the fact that the depths
    # were not picked to maximise a result, and — the load-bearing one — why the ladder has a
    # ceiling: raising it measures admission selection rather than prefix evidence.
    head = inspect.getsource(features).split("DEFAULT_DEPTHS: Final")[0].lower()
    for claim in ("x2 ladder", "not chosen by outcome", "the ceiling", "selection channel"):
        assert claim in head, f"the depth ladder's justification no longer states: {claim}"


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
        # Dropped by wall 2, not by arithmetic: it is the last-3-steps window, so it is the column
        # nearest the verdict and carried the proximity leak (+0.1368 dAUROC at depth 40 against
        # +0.0122 for fail_rate). At depth 5 it is EXACTLY fail_rate on 787/791 trajectories, and
        # the only 4 rows that differed — the 4 shortest runs in the corpus, prefixes ending 2-6
        # steps from the verdict — were the sole reason the depth-5 design ranked full. Once
        # MIN_WITHHELD excludes those, the equality is 452/452 and the design is rank-deficient.
        "recent_fail_rate",
        # Dropped 2026-08-02 for non-independence on the REBUILT corpus (features.py has the
        # measurements). `max_key_repeat_rate` is `fail_rate` on 414/414 depth-5 rows, 342/344 at
        # depth 10 and 226/228 at depth 20 — the rank guard was green on two outlier rows apiece.
        # `distinct_check_id_rate` is constant at depth 5 and binary at 10/20. Both were aliases
        # of the same constant-key mechanism that degrades the shallow design.
        "max_key_repeat_rate",
        "distinct_check_id_rate",
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

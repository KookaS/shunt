"""The multi-dimensional kill gate's two-sided validity control — run BEFORE quoting its verdict.

Plants routers whose win or loss is true by construction, asks the ASSEMBLED gate to recover
which is which, then destroys the labels and asks the same gate to collapse to chance.
"""

# WHAT THE INSTRUMENT IS, AND WHERE THE CONTROL ENTERS. The gate is
#
#     strategy_summary.csv + kill_gate_verdict.json
#         ->  load_arm_rows / coverage_blockers / admissibility_blockers
#         ->  compare_to_baseline (quality gate + 4 axes)  ->  BaselineComparison.dominates
#         ->  adjudicate  ->  verdict_payload / write_verdict  ->  PASS / FAIL / UNTESTED
#
# and the verdict it emits is read off the last stage. This control enters at the FILES — it
# writes a planted `strategy_summary.csv` and a planted offline verdict per scenario and calls
# `gate_dimensions.rederive_verdict`, the same assembly `main` runs — so the CSV parse, the
# non-finite refusal, the coverage precondition, the admissibility precondition, the axis
# arithmetic, the tolerance, the conjunction, the serialised record and the CLI's exit code are
# all inside what is scored.
#
# IT DID NOT ALWAYS. An earlier version entered one stage later, at the in-memory arm rows, and
# its header claimed the front of the chain anyway. Traced, it never executed `load_arm_rows`,
# `_num`, `coverage_blockers`, `admissibility_blockers`, `verdict_payload`, `write_verdict` or
# `main` — and the shipped verdict's word, UNTESTED, comes from the blockers branch that the
# control never entered. So the clearance certified a PASS/FAIL discriminator while the quoted
# verdict came from an uncertified branch. `_assert_the_blocked_branch_is_untested` and
# `_assert_the_cli_agrees` close that: the branch that produced the shipped word is now scored.
#
# WHY THE STATISTIC IS BALANCED ACCURACY AND NOT A PASS RATE. The gate emits a verdict about
# whether a signal (a genuine multi-dimensional advantage) EXISTS, so the thing that must be
# validated is its ability to tell present from absent. A gate frozen at FAIL has a perfect
# "no-false-PASS" record and is worthless; balanced accuracy scores it at 0.5, which is the
# whole point. Chance is 0.5 by construction, so the empirical band is narrow and honest.
#
# AND WHY BALANCED ACCURACY ALONE IS NOT ENOUGH. Measured by mutation, it is not: an instrument
# with the strict-win requirement deleted, with the tolerance sign inverted, or with the quality
# gate hardcoded open each still scored 0.8375-0.9000 here — "clearly above chance", and
# therefore ADMISSIBLE. So the R0 score is joined by a hard refusal,
# `_assert_every_planted_scenario_is_recovered`: on THIS corpus the only correct score is
# exactly 1.0, because every label follows from the registered rule with nothing left to judge.
#
# WHAT THE LABELS ARE, AND THE ONE INDEPENDENCE CLAIM THIS DOES NOT MAKE. The trap a gate's own
# control falls into is computing the ground truth by running the gate, which would certify any
# rule at all. Nothing here consults `adjudicate`: a scenario is a WIN because the router was
# drawn strictly below BOTH baselines on one axis and no worse on the others, and a LOSS because
# it was drawn strictly above, or inside the tolerance on the worse side, or tied everywhere, or
# below the quality margin. What that buys is a check that the IMPLEMENTATION realises the
# registered criterion. It is NOT a check that the criterion is the right one: `tie_only` and
# `quality` are losses because the registered rule says ties earn nothing and quality is a hard
# gate, and an instrument cannot validate its own premise. Those two premises are argued in the
# decision record, not here.
#
# THE NULL, AND WHY THE OLD ONE WAS VACUOUS. It used to compute the gate's predictions ONCE and
# then permute labels around that frozen vector. Measured: a maximally-inverted gate, a gate
# frozen at PASS, a gate frozen at FAIL and pure coinflip noise ALL reported NULL_AT_CHANCE. A
# null that cannot fail is not a null — permuting labels against a fixed prediction vector is at
# chance by arithmetic, whatever produced the vector.
#
# The null now DESTROYS THE SIGNAL IN THE DATA and re-runs the whole chain on it: each
# scenario's router row is redrawn independently of the planted mode, so the scenario population
# and the baselines are untouched and only the per-scenario alignment is gone. Three conditions
# must then hold, and each kills a degenerate instrument the old null waved through:
#   (1) AT CHANCE — the re-run scores within the empirical band of 0.5. Leakage fails it.
#   (2) RESPONSIVE — destroying the signal must CHANGE some verdicts. A frozen gate changes
#       none, so "at chance" for it means deaf, not honest, and it fails here.
#   (3) SIGNAL-DEPENDENT — the positive score must exceed the destroyed score by more than the
#       band, i.e. the score has to be attributable to the planted signal. An inverted gate
#       (0.0 against 0.5) and coinflip noise (0.5 against 0.5) both fail it.
# `tests/test_gate_dimensions.py::TestTheNullCanFail` is the acceptance test: all four
# degenerate gates above are REJECTED by the null leg, and the real gate is not.
#
# A FAILURE HERE IMPLICATES: the axis comparison, the relative tolerance, the quality margin,
# the dominance conjunction and the two-baseline requirement. It does NOT implicate the
# strategy table's coverage, its imputation share, or whether the real corpus HAS a routing
# signal — those are upstream and are what the coverage precondition refuses on.

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np

from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.routing import gate_dimensions as gd
from benchmark.routing.gate_dimensions import (
    OPERATIONAL_AXES,
    OPERATIONAL_TOLERANCE,
    PASS,
    QUALITY_COLUMN,
    QUALITY_MARGIN_PP,
    UNTESTED,
    GateAxis,
)
from benchmark.routing.scripts import knn_nulls

#: Balanced accuracy under no signal.
CHANCE_LEVEL: Final[float] = 0.5

#: The planted advantage/deficit, as a relative fraction. Far above `OPERATIONAL_TOLERANCE`,
#: so a coarse scenario is unambiguous.
PLANTED_DELTA: Final[float] = 0.25

#: A SECOND, BOUNDARY-ADJACENT magnitude, and the reason it exists. A corpus planted only at
#: 5x the tolerance certifies coarse separability and nothing else: measured by mutation, an
#: instrument with the strict-win requirement deleted, with the tolerance sign inverted, or with
#: the quality gate hardcoded open still scored a perfect 1.0000 on such a corpus, because no
#: scenario ever sat near the edge those clauses decide. `NEAR_DELTA` is just OUTSIDE the
#: tolerance (a real win) and `WITHIN_DELTA` just INSIDE it (a tie, therefore not a win), so the
#: boundary itself is now part of what the positive leg has to get right.
NEAR_DELTA: Final[float] = 0.08
WITHIN_DELTA: Final[float] = 0.02

#: The planted quality deficit, in percentage points beyond the margin.
PLANTED_QUALITY_DELTA_PP: Final[float] = 5.0

_ROUTER: Final[str] = "router"
_FRONTIER: Final[str] = "frontier"
_CONSTANT: Final[str] = "constant"
_BASELINES: Final[tuple[str, ...]] = (_FRONTIER, _CONSTANT)

#: How far below the full gate a blinded mutant must score before the corpus is accepted as
#: exercising the thing that mutant is blind to. Chosen once, above the corpus's own sampling
#: noise at the default size and below the arithmetic separation each mutant implies (0.375 for
#: the cost-only mutant, 0.25 for the single-baseline mutant).
_MUTANT_MARGIN: Final[float] = 0.15

_COLUMNS: Final[tuple[str, ...]] = tuple(g.axis.column for g in OPERATIONAL_AXES)

#: Win modes. `win_near` plants an advantage only just outside the tolerance, so an instrument
#: that reads the tolerance backwards cannot score it by accident.
_WIN_MODES: Final[tuple[str, ...]] = ("win", "win", "win_near")

#: Loss modes, in draw order. Each one is the mutant-killer for a different clause of the
#: criterion, and every label is the GENERATOR'S intent under the pre-registered rule, never a
#: verdict read back off the gate:
#:   quality        — the router genuinely DOMINATES operationally and is inferior on quality.
#:                    The operational win is essential: without it `dominates` already fails on
#:                    its own and the quality conjunct is never the clause under test.
#:   both_baselines — worse than both, on one axis.
#:   constant_only  — better than the frontier, worse than the constant policy. The only mode a
#:                    single-baseline gate gets wrong; Scrouting's ablation in miniature.
#:   tie_only       — identical to the better baseline on every axis. A router that ties a policy
#:                    doing no routing has earned nothing, so this is a loss BY THE REGISTERED
#:                    RULE — and it is the only scenario the strict-win requirement decides.
#:   tie_within     — inside the tolerance but on the WORSE side of it. A tie, therefore not a
#:                    win; an inverted tolerance sign reads it as a win.
_LOSS_MODES: Final[tuple[str, ...]] = (
    "quality",
    "both_baselines",
    "constant_only",
    "constant_only",
    "tie_only",
    "tie_within",
)


def _scenario(
    rng: np.random.Generator, planted_win: bool
) -> tuple[dict[str, dict[str, float]], str]:
    """One planted arm table, plus the mode it was planted in. Never calls the gate."""
    quality = 90.0
    axis = str(rng.choice(_COLUMNS))
    mode = str(rng.choice(_WIN_MODES if planted_win else _LOSS_MODES))

    # Baselines: independent positive draws, with the frontier deliberately the pricier arm on
    # the chosen axis so a `constant_only` deficit has room to sit between the two.
    scale = {c: float(rng.uniform(1.0, 10.0)) for c in _COLUMNS}  # noqa: SH008 (planted control)
    frontier = {
        c: scale[c] * 4.0 if c == axis else scale[c] * float(rng.uniform(1.0, 2.0))  # noqa: SH008 (planted control)
        for c in _COLUMNS
    }
    constant = dict(scale)
    floor = {c: min(frontier[c], constant[c]) for c in _COLUMNS}

    # Every non-planted axis is EXACTLY the elementwise floor: a tie against the better
    # baseline, so the scenario's verdict turns on the planted axis alone.
    router = dict(floor)
    router_quality = quality

    if mode == "win":
        router[axis] = floor[axis] * (1.0 - PLANTED_DELTA)
    elif mode == "win_near":
        router[axis] = floor[axis] * (1.0 - NEAR_DELTA)
    elif mode == "quality":
        # A genuine operational win, so the ONLY thing that can make this a loss is the
        # quality gate. Without this line the mode is confounded and tests nothing.
        router[axis] = floor[axis] * (1.0 - PLANTED_DELTA)
        router_quality = quality - QUALITY_MARGIN_PP - PLANTED_QUALITY_DELTA_PP
    elif mode == "both_baselines":
        router[axis] = max(frontier[axis], constant[axis]) * (1.0 + PLANTED_DELTA)
    elif mode == "constant_only":  # better than the frontier, worse than the constant policy
        router[axis] = constant[axis] * (1.0 + PLANTED_DELTA)
    elif mode == "tie_within":
        router[axis] = floor[axis] * (1.0 + WITHIN_DELTA)
    # tie_only leaves `router` at the elementwise floor: every axis an exact tie.

    rows = {
        _ROUTER: {**router, QUALITY_COLUMN: router_quality},
        _FRONTIER: {**frontier, QUALITY_COLUMN: quality},
        _CONSTANT: {**constant, QUALITY_COLUMN: quality},
    }
    return rows, mode


def build_corpus(
    n: int, seed: int
) -> tuple[list[dict[str, dict[str, float]]], np.ndarray, list[str]]:
    """``n`` planted scenarios, half wins, with their ground-truth labels and modes."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 == 0 for i in range(n)])
    corpus: list[dict[str, dict[str, float]]] = []
    modes: list[str] = []
    for planted_win in labels:
        rows, mode = _scenario(rng, bool(planted_win))
        corpus.append(rows)
        modes.append(mode)
    return corpus, labels, modes


#: The planted offline-gate census the chain's coverage precondition is re-derived from.
#: `_OFFLINE_CLEAR` sits above the 0.9 floor and `_OFFLINE_TRIPPED` below it, so the blocked
#: branch is entered by the DATA rather than by a flag anyone set.
_OFFLINE_TASKS: Final[int] = 200
_OFFLINE_FLOOR: Final[float] = 0.9
_OFFLINE_CLEAR: Final[int] = 195
_OFFLINE_TRIPPED: Final[int] = 100

#: How far the destroyed-signal null redraws the router row around the baselines' floor. Wide
#: enough that the gate's PASS rate stays non-degenerate, so "at chance" is a measurement and
#: not an arithmetic identity.
NULL_SPREAD: Final[float] = 0.25

Predictor = Callable[[Sequence[dict[str, dict[str, float]]]], np.ndarray]


def write_summary(
    path: Path, rows: dict[str, dict[str, float]], *, admissible: bool = True
) -> Path:
    """Write one planted scenario as the tracked strategy table's own CSV shape."""
    columns = (QUALITY_COLUMN, *_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("strategy", *columns, "instrument_admissible"))
        for arm, row in rows.items():
            # `repr` round-trips a float exactly, so the CSV parse the chain performs is scored
            # on the planted number rather than on a rounding of it.
            writer.writerow((arm, *(repr(float(row[c])) for c in columns), str(admissible)))
    return path


def write_offline_verdict(path: Path, *, tripped: bool) -> Path:
    """Write an offline-gate verdict whose coverage census implies ``tripped``, never asserts it."""
    measured = _OFFLINE_TRIPPED if tripped else _OFFLINE_CLEAR
    path.write_text(
        json.dumps(
            {
                "n": _OFFLINE_TASKS,
                "coverage": {
                    "floor": _OFFLINE_FLOOR,
                    "control_measured": measured,
                    "router_measured": measured,
                    "control_coverage": measured / _OFFLINE_TASKS,
                    "router_coverage": measured / _OFFLINE_TASKS,
                    "tripped": tripped,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def chain_verdict(
    rows: dict[str, dict[str, float]],
    root: Path,
    *,
    axes: tuple[GateAxis, ...] = OPERATIONAL_AXES,
    baselines: tuple[str, ...] = _BASELINES,
    coverage_tripped: bool = False,
    admissible: bool = True,
) -> gd.GateVerdict:
    """One scenario through the WHOLE gate, entered at the files ``main`` reads."""
    summary = write_summary(root / "strategy_summary.csv", rows, admissible=admissible)
    offline = write_offline_verdict(root / "kill_gate_verdict.json", tripped=coverage_tripped)
    return gd.rederive_verdict(summary, offline, router=_ROUTER, baselines=baselines, axes=axes)


def gate_predictions(
    corpus: Sequence[dict[str, dict[str, float]]],
    *,
    axes: tuple[GateAxis, ...] = OPERATIONAL_AXES,
    baselines: tuple[str, ...] = _BASELINES,
) -> np.ndarray:
    """Run the ASSEMBLED gate, entered at its input files, over the corpus; True where PASS."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        return np.array(
            [
                chain_verdict(rows, root, axes=axes, baselines=baselines).verdict == PASS
                for rows in corpus
            ]
        )


def balanced_accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    """Mean of sensitivity and specificity — 0.5 for any constant predictor."""
    pos, neg = labels, ~labels
    sensitivity = float(predicted[pos].mean()) if pos.any() else CHANCE_LEVEL
    specificity = float((~predicted[neg]).mean()) if neg.any() else CHANCE_LEVEL
    return (sensitivity + specificity) / 2.0


def _assert_every_planted_scenario_is_recovered(
    predicted: np.ndarray, labels: np.ndarray, modes: list[str]
) -> None:
    """Fail loudly on ANY misclassified scenario — this corpus has no ambiguous cases."""
    # WHY THIS IS A REFUSAL AND NOT A SCORE. The R0 adjudicator asks "is the positive leg clearly
    # above chance", which is the right question for a noisy estimator and the WRONG one for a
    # decision rule read over a corpus where every scenario was planted unambiguously. Measured:
    # deleting the strict-win requirement, inverting the tolerance sign, and hardcoding the
    # quality gate open each dropped the positive leg only to 0.8375-0.9000 — comfortably
    # "clearly above chance", and still ADMISSIBLE. Balanced accuracy alone therefore certifies
    # coarse separability, not the individual clauses of the criterion.
    #
    # On THIS corpus the correct score is exactly 1.0, because every scenario's label follows
    # from the registered rule with no judgement left over. So anything less is a defective
    # clause, and it is raised — naming the modes that failed, which is the diagnosis — rather
    # than averaged into a number that still reads as a pass. It is the same shape as the
    # sibling session-cascade control's path assertion: prove the mechanism, not just the score.
    wrong = predicted != labels
    if not wrong.any():
        return
    failed = sorted({modes[i] for i in np.flatnonzero(wrong)})
    raise RuntimeError(
        f"the multi-dimensional gate misclassified {int(wrong.sum())} of {labels.size} "
        f"UNAMBIGUOUSLY planted scenarios, in modes {failed}. Every label here follows from the "
        f"registered criterion with nothing left to judge, so a miss is a defective clause of "
        f"that criterion — no verdict from this gate may be trusted until it is fixed."
    )


def _assert_axes_are_load_bearing(
    corpus: list[dict[str, dict[str, float]]], labels: np.ndarray, full: float
) -> None:
    """Fail loudly unless blinding the gate measurably degrades it on THIS corpus."""
    # A score is only evidence about the parts of the instrument the corpus exercises. Two
    # mutants are therefore run on the same planted corpus, and each must score materially
    # WORSE than the full gate: a cost-only gate (the OLD criterion, blind to sessions, tail
    # and variance) and a single-baseline gate (blind to the constant-policy arm). If either
    # matches the full gate, the corpus never exercised what it is blind to and the positive
    # leg certifies less than it appears to.
    cost_only = balanced_accuracy(gate_predictions(corpus, axes=OPERATIONAL_AXES[:1]), labels)
    frontier_only = balanced_accuracy(gate_predictions(corpus, baselines=(_FRONTIER,)), labels)
    for name, score in (("cost-only", cost_only), ("frontier-only", frontier_only)):
        if score > full - _MUTANT_MARGIN:
            raise RuntimeError(
                f"the multi-dimensional gate control did not exercise what it claims: the "
                f"{name} mutant scored {score:.4f} against the full gate's {full:.4f} "
                f"(margin {_MUTANT_MARGIN}). The planted corpus does not separate them, so no "
                f"verdict from this gate may be trusted on the strength of the positive leg."
            )


def destroyed_corpus(
    corpus: Sequence[dict[str, dict[str, float]]], rng: np.random.Generator
) -> list[dict[str, dict[str, float]]]:
    """The same scenarios with the planted signal removed: the router row redrawn at random."""
    # The baselines, the axis scales and the scenario count are untouched — only the router's
    # relation to them is destroyed. So the gate still SEES a corpus of the same shape, and the
    # only thing missing is the thing the positive leg claims to detect.
    out: list[dict[str, dict[str, float]]] = []
    for rows in corpus:
        frontier, constant = rows[_FRONTIER], rows[_CONSTANT]
        floor = {c: min(frontier[c], constant[c]) for c in _COLUMNS}
        router = {
            c: floor[c] * float(rng.uniform(1.0 - NULL_SPREAD, 1.0 + NULL_SPREAD))  # noqa: SH008 (planted control)
            for c in _COLUMNS
        }
        out.append(
            {
                _ROUTER: {
                    **router,
                    QUALITY_COLUMN: frontier[QUALITY_COLUMN] + float(rng.uniform(-1.0, 1.0)),  # noqa: SH008 (planted control)
                },
                _FRONTIER: dict(frontier),
                _CONSTANT: dict(constant),
            }
        )
    return out


def _assert_the_blocked_branch_is_untested(rows: dict[str, dict[str, float]], root: Path) -> None:
    """Fail loudly unless the branch that produced the SHIPPED word is the one being scored."""
    # The shipped verdict is UNTESTED, and UNTESTED comes from `adjudicate`'s blockers branch —
    # which the previous control never entered, so its clearance certified a PASS/FAIL
    # discriminator while the quoted word came from uncertified code. Every leg below is a
    # planted WIN: the operational verdict is held constant so the only thing under test is
    # whether a precondition, DERIVED from planted census data, overrides it.
    clear = chain_verdict(rows, root)
    if clear.verdict != PASS or clear.blockers:
        raise RuntimeError(
            f"the chain does not reach PASS on an unambiguously planted win with every "
            f"precondition clear: got {clear.verdict} with blockers {clear.blockers}."
        )
    for label, kwargs in (
        ("coverage floor", {"coverage_tripped": True}),
        ("instrument admissibility", {"admissible": False}),
    ):
        blocked = chain_verdict(rows, root, **kwargs)  # type: ignore[arg-type]
        if blocked.verdict != UNTESTED or not blocked.blockers:
            raise RuntimeError(
                f"the {label} precondition did not force UNTESTED on a planted win: got "
                f"{blocked.verdict} with blockers {blocked.blockers}. The shipped verdict is "
                f"read off this branch, so it may not be quoted until it is scored."
            )
        if blocked.provisional != PASS:
            raise RuntimeError(
                f"the {label} precondition dropped the provisional reading on a planted win: "
                f"got {blocked.provisional!r}, expected {PASS!r}."
            )
        payload = gd.verdict_payload(blocked)
        if payload["verdict"] != UNTESTED or payload["provisional_verdict"] != PASS:
            raise RuntimeError(
                f"the serialised record contradicts the {label} verdict it came from: "
                f"{payload['verdict']!r} / {payload['provisional_verdict']!r}."
            )


def _assert_the_cli_agrees(
    win: dict[str, dict[str, float]], loss: dict[str, dict[str, float]], root: Path
) -> None:
    """Fail loudly unless ``main`` writes the same record and exits 0 only on PASS."""
    # An automated consumer reads the CLI's exit code, not this module's return value, and
    # `main`/`write_verdict` were outside everything the control executed. Both are scored here.
    for rows, expected_code, expected in ((win, 0, PASS), (loss, 1, "FAIL")):
        summary = write_summary(root / "cli_summary.csv", rows)
        offline = write_offline_verdict(root / "cli_offline.json", tripped=False)
        out = root / "cli_verdict.json"
        argv = [
            "gate_dimensions",
            "--summary",
            str(summary),
            "--router",
            _ROUTER,
            "--offline-verdict",
            str(offline),
            "--verdict",
            str(out),
        ]
        for baseline in _BASELINES:
            argv += ["--baseline", baseline]
        code = _run_cli(argv)
        payload = json.loads(out.read_text(encoding="utf-8"))
        if code != expected_code or payload["verdict"] != expected:
            raise RuntimeError(
                f"the gate's CLI disagrees with its own criterion on a planted {expected}: "
                f"exit {code} (expected {expected_code}), artifact says {payload['verdict']!r}."
            )


def _run_cli(argv: list[str]) -> int:
    """``gate_dimensions.main`` under a patched argv, returning the exit code it raises."""
    saved = sys.argv
    sys.argv = argv
    try:
        # The CLI's human report is the thing under test only through its exit code and its
        # artifact, so it is swallowed rather than interleaved into the control's own output.
        with contextlib.redirect_stdout(io.StringIO()):
            gd.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    return 0


def run_control(
    *,
    n_scenarios: int = 240,
    n_perm: int = knn_nulls.DEFAULT_PERMUTATIONS,
    seed: int = 0,
    predict: Predictor | None = None,
) -> AdmissibilityResult:
    """Run both legs through the assembled gate and adjudicate with the shared R0 gate."""
    # `predict` exists ONLY so the null leg can be shown to fail: the acceptance test injects a
    # frozen, an inverted and a coinflip gate and asserts each is rejected. When it is supplied
    # the two hard refusals below are skipped — they diagnose defective clauses of the REAL
    # criterion, and an injected stand-in has no clauses to diagnose.
    predictor: Predictor = gate_predictions if predict is None else predict
    corpus, labels, modes = build_corpus(n_scenarios, seed)
    predicted = predictor(corpus)
    positive = balanced_accuracy(predicted, labels)
    if predict is None:
        _assert_every_planted_scenario_is_recovered(predicted, labels, modes)
        _assert_axes_are_load_bearing(corpus, labels, positive)
        with tempfile.TemporaryDirectory() as tmp:
            win, _ = _scenario(np.random.default_rng(seed + 3), True)
            loss, _ = _scenario(np.random.default_rng(seed + 4), False)
            _assert_the_blocked_branch_is_untested(win, Path(tmp))
            _assert_the_cli_agrees(win, loss, Path(tmp))

    # THE NULL. The signal is destroyed in the DATA and the whole chain is re-run on it — not,
    # as it once was, permuted around a frozen prediction vector, which is at chance by
    # arithmetic no matter what produced the vector. The band is still the label-permutation
    # spread of the SAME statistic, which is the right sampling reference for "how far from 0.5
    # does balanced accuracy land when nothing aligns".
    null_predicted = predictor(destroyed_corpus(corpus, np.random.default_rng(seed + 2)))
    null_score = balanced_accuracy(null_predicted, labels)
    rng = np.random.default_rng(seed + 1)
    draws = np.array(
        [balanced_accuracy(null_predicted, rng.permutation(labels)) for _ in range(n_perm)]
    )
    band = knn_nulls.band_of(draws)
    half_width = (band.hi - band.lo) / 2.0

    # Two conditions the band alone cannot express. RESPONSIVE: destroying the signal must
    # change some verdicts, or "at chance" means deaf rather than honest — this is what rejects
    # a frozen gate. SIGNAL-DEPENDENT: the positive score must beat the destroyed score by more
    # than the band, or the score was not attributable to the planted signal — this is what
    # rejects an inverted gate and coinflip noise.
    responsive = bool(np.any(np.asarray(predicted) != np.asarray(null_predicted)))
    signal_dependent = (positive - null_score) > half_width

    verdict = admissibility_verdict(
        positive, null_score, chance_level=CHANCE_LEVEL, chance_band=half_width
    )
    null_ok = verdict.null_at_chance and responsive and signal_dependent
    reason = verdict.reason
    if verdict.null_at_chance and not null_ok:
        broke = "not responsive to the data" if not responsive else "not signal-dependent"
        reason = (
            f"INADMISSIBLE: the destroyed-signal null scores {null_score:+.4f} at chance, but "
            f"the instrument is {broke} — positive {positive:+.4f} against destroyed "
            f"{null_score:+.4f} (band {half_width:.4f}), verdicts changed on "
            f"{float(np.mean(np.asarray(predicted) != np.asarray(null_predicted))):.1%} of "
            f"scenarios. A score that survives destroying the signal was never measuring it."
        )
    return replace(
        verdict,
        admissible=verdict.positive_passed and null_ok,
        null_at_chance=null_ok,
        reason=reason,
        numbers={
            **verdict.numbers,
            "n_scenarios": int(labels.size),
            "n_perm": n_perm,
            "axes": list(_COLUMNS),
            "baselines": list(_BASELINES),
            "tolerance": OPERATIONAL_TOLERANCE,
            "quality_margin_pp": QUALITY_MARGIN_PP,
            "planted_delta": PLANTED_DELTA,
            "modes_planted": sorted(set(modes)),
            "n_by_mode": {m: modes.count(m) for m in sorted(set(modes))},
            "cost_only_mutant": balanced_accuracy(
                gate_predictions(corpus, axes=OPERATIONAL_AXES[:1]), labels
            )
            if predict is None
            else None,
            "frontier_only_mutant": balanced_accuracy(
                gate_predictions(corpus, baselines=(_FRONTIER,)), labels
            )
            if predict is None
            else None,
            "null_responsive": responsive,
            "null_signal_dependent": bool(signal_dependent),
            "null_verdicts_changed": float(
                np.mean(np.asarray(predicted) != np.asarray(null_predicted))
            ),
            "null_mean": band.mean,
            "null_sd": band.sd,
            "null_lo": band.lo,
            "null_hi": band.hi,
        },
    )


def main() -> None:
    """CLI: run the control and print the R0 admissibility verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=int, default=240)
    ap.add_argument("--permutations", type=int, default=knn_nulls.DEFAULT_PERMUTATIONS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = run_control(n_scenarios=args.scenarios, n_perm=args.permutations, seed=args.seed)
    print(result.reason)
    for key, value in sorted(result.numbers.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

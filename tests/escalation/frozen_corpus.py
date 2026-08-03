"""A frozen 11-run corpus whose prefix census is known by construction, not by measurement."""

# WHY THIS MODULE EXISTS. Six tests used to assert HARDCODED counts against the LIVE corpus —
# `n_unstamped == 8`, row floors of 452/372/239, per-depth census tuples measured on 2026-07-30.
# Every one of them went red the moment the corpus was CORRECTED, which is exactly backwards: the
# data improving must not read as the code regressing. Nothing in `corpus_census` or `build_rows`
# had changed; the corpus had (799 runs, 727 stamped, 414/344/228 admitted rows at depths 5/10/20).
#
# Re-pinning the new numbers is the same trap with a fresher number, so exact counts moved HERE, to
# a corpus that never moves. Each run is declared by the only three things the census reads — which
# challenge it belongs to, how many scorable steps it has, and whether the per-step stamping stage
# ran on it — so every count below is a property of THIS FILE and changes only when someone edits
# it deliberately.
#
# The live corpus keeps the legs that survive a correction: the framework floors
# (`prefix_eval.MIN_ROWS`, `N_SPLITS`, which are what `evaluate_depth` actually requires) and a
# per-bucket recount derived from the measured margin below. Its exact census stays PUBLISHED to
# the report JSON — reviewed, not pinned.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from benchmark.escalation import schema
from benchmark.escalation.schema import StepView, Trajectory, TrajectoryHeader

# The anti-leak margin as MEASURED, written as a literal on purpose — the ONE copy, imported by
# `test_features.py` and `test_prefix_eval.py` so the fixture's expected census and the walls that
# guard the margin cannot drift apart. It is deliberately NOT `features.MIN_WITHHELD`: a test that
# derives its own boundary from the constant it is testing moves with the bug, and setting
# `MIN_WITHHELD = 0` would keep such a test green while re-admitting every leaked row. That mutant
# was run and did survive the self-referential version of these tests, which is why 22 is a
# literal here. Where the 22 comes from is documented at `features.MIN_WITHHELD`.
MEASURED_MARGIN: Final[int] = 22


@dataclass(frozen=True)
class RunSpec:
    """One frozen run, declared by the only three things the prefix census reads."""

    trajectory_id: str
    challenge: str
    n_scorable: int
    stamped: bool
    resolved: bool


# Admission at `depth` needs `depth + MEASURED_MARGIN` scorable steps, so the reported ladder
# (5, 10, 20) has its bars at 27, 32 and 42. The lengths below straddle every one of those bars
# from BOTH sides — 26/27, 31/32, 41/42 — so an off-by-one in the admission rule moves a named run
# between buckets rather than merely shifting a total, and none of the census's three exclusion
# buckets is empty at any reported depth.
SPECS: Final[tuple[RunSpec, ...]] = (
    RunSpec("inst-a__cheap__nothink", "inst-a", 4, stamped=True, resolved=True),
    RunSpec("inst-a__cheap__high", "inst-a", 10, stamped=True, resolved=False),
    RunSpec("inst-b__cheap__nothink", "inst-b", 26, stamped=True, resolved=False),
    RunSpec("inst-b__frontier__max", "inst-b", 27, stamped=True, resolved=True),
    RunSpec("inst-c__cheap__high", "inst-c", 31, stamped=True, resolved=False),
    RunSpec("inst-c__frontier__max", "inst-c", 32, stamped=True, resolved=True),
    RunSpec("inst-d__cheap__high", "inst-d", 41, stamped=True, resolved=False),
    RunSpec("inst-e__frontier__max", "inst-e", 42, stamped=True, resolved=True),
    RunSpec("inst-f__cheap__high", "inst-f", 50, stamped=True, resolved=False),
    RunSpec("inst-g__cheap__nothink", "inst-g", 30, stamped=False, resolved=False),
    RunSpec("inst-g__frontier__max", "inst-g", 45, stamped=False, resolved=True),
)


@dataclass(frozen=True)
class ExpectedCensus:
    """The exact census `corpus_census` must produce on this corpus at one depth."""

    n_rows: int
    n_groups: int
    n_unstamped: int
    n_too_short: int
    n_by_margin: int


# Checked in as literals, so a reader can verify them against SPECS by hand: a run is admitted iff
# it is stamped AND `n_scorable >= depth + MEASURED_MARGIN`, too-short iff `n_scorable < depth`,
# and margin-cut in between. `test_prefix_eval` re-derives the whole table from SPECS and refuses a
# mismatch, so a typo here fails loudly instead of quietly re-pinning a wrong number. A depth added
# to `features.DEFAULT_DEPTHS` and not added here raises KeyError rather than skipping silently.
# The 20 entry is intentionally still here: depth 20 left `DEFAULT_DEPTHS` on 2026-08-02 (its
# admitted population drifted past the ladder's own selection-bias bound — features.py records the
# decision), and this row is the fixture record of what it WOULD admit, referenced from that
# comment so the removed depth does not vanish from every representation at once.
EXPECTED: Final[dict[int, ExpectedCensus]] = {
    5: ExpectedCensus(n_rows=6, n_groups=5, n_unstamped=2, n_too_short=1, n_by_margin=2),
    10: ExpectedCensus(n_rows=4, n_groups=4, n_unstamped=2, n_too_short=1, n_by_margin=4),
    20: ExpectedCensus(n_rows=2, n_groups=2, n_unstamped=2, n_too_short=2, n_by_margin=5),
}


def admitted(depth: int) -> frozenset[str]:
    """The trajectory ids the eval must admit at `depth`, read off SPECS rather than measured."""
    return frozenset(
        spec.trajectory_id
        for spec in SPECS
        if spec.stamped and spec.n_scorable >= depth + MEASURED_MARGIN
    )


def build() -> list[Trajectory]:
    """The frozen corpus, rebuilt from SPECS — deterministic, in memory, no I/O."""
    return [_run(spec) for spec in SPECS]


def _run(spec: RunSpec) -> Trajectory:
    """One run: `n_scorable` readable steps plus the label-stamped terminal step."""
    steps = [_step(spec, index, terminal=False) for index in range(spec.n_scorable)]
    steps.append(_step(spec, spec.n_scorable, terminal=True))
    header = TrajectoryHeader(
        schema_version=schema.SCHEMA_VERSION,
        trajectory_id=spec.trajectory_id,
        dataset="frozen",
        plane="committable",
        framework="mini_swe_agent",
        terminal_resolved=spec.resolved,
        instance_id=spec.challenge,
        license="MIT",
        dataset_revision="frozen",
        redacted=False,
        content_sha256=schema.content_sha256(steps),
        n_steps=len(steps),
        snapshot_steps=None,
    )
    return Trajectory(header=header, steps=steps)


def _step(spec: RunSpec, index: int, *, terminal: bool) -> StepView:
    """A deterministic step; unstamped runs carry `confirmed` on the terminal step only."""
    # Varied, not constant: the failure stride is a function of the RUN as well as the step, so
    # different runs produce different feature vectors rather than one repeated row. `confirmed`
    # mirrors the live corpus's unstamped runs, where the per-step replay never ran but the
    # terminal step still carries the harness grade — exactly what `features.is_stamped` reads.
    stride = 2 + spec.n_scorable % 3
    success = spec.resolved if terminal else index % stride != 0
    check = None if success else f"pkg::t{index % 2}"
    return StepView(
        step_index=index,
        decision_index=index,
        parent_step_index=None,
        metadata={},
        observation="",
        action=f"cmd{index % 4}",
        tool="bash",
        args=None,
        result="",
        status="ok" if success else "error",
        test_passed=None,
        test_total=None,
        failing_check_id=check,
        exit_code=0 if success else 1,
        blocking=not success,
        is_infra_failure=False,
        confirmed=terminal or spec.stamped,
        success=success,
        is_revert=False,
        retry_count=0,
        loop_signal=False,
        subgoal_progress=None,
        dedup_key=check,
        model=None,
        reasoning_effort=None,
        rank_index=None,
        effort_index=None,
        real_cost=None,
    )

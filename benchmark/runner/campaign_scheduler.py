"""Priority-first, per-model phase campaign scheduler for the free-tier collector.

The campaign's cell set is STATIC — every (challenge, model, arm) is enumerated up front —
but WHICH model gets a lane at any instant is decided live from the owner's model-value
ranking. This module builds that order and enforces it over the lane admission scheduler:

* :func:`build_plan` ranks the runnable overlay models by
  ``collection_priority.priority``, drops duplicate and not-worth lanes with the priority
  API's own named reason, and retires a model whose ``runnable_benchmarks`` set is empty
  (text complete, pure-text) so its workers free for the rest.
* :class:`PriorityLaneScheduler` is the pull loop's ordering rule: the highest-priority
  admissible model is admitted first, and a lower-priority model runs only while every
  higher one is blocked, quarantined or unavailable — idle-capacity fill, never a stall.
  Within one model, a pending text cell gates that model's multimodal cells (text first).
* :func:`run_cells` reuses ``run_matrix``'s cell execution, sentinel handling and
  checkpoint merge verbatim; only the ordering is new. Resume is the existing
  ``classify_cells`` MISSING/STALE set — there is no scheduler ledger to drift.

Planning and selection are pure (no model call, no I/O), so the whole schedule is testable
offline. Run the planner: ``uv run python -m benchmark.runner.campaign_scheduler --help``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from benchmark.routing import collection_priority as cp
from benchmark.runner import lane_scheduler

if TYPE_CHECKING:
    from benchmark.runner import run_matrix

TEXT_BENCHMARK = cp.TEXT_BENCHMARK
MULTIMODAL_BENCHMARK = cp.MULTIMODAL_BENCHMARK
Cell = lane_scheduler.Cell

_TEXT_RANK = 0
_MULTIMODAL_RANK = 1


def effective_workers(requested: int, runnable_models: int) -> int:
    """The worker count to use: never more workers than there are models to run.

    ``min(requested, runnable_models)`` floored at zero: an empty run needs no worker, and
    asking for eight workers over three models would idle five.
    """
    return max(0, min(int(requested), int(runnable_models)))


@dataclass(frozen=True)
class ModelPhase:
    """One model's schedulable phase: which model, which benchmark, at what value."""

    model: str
    benchmark: str
    priority: float

    @property
    def phase_rank(self) -> int:
        """Text sorts before multimodal within one model (0 before 1)."""
        return _MULTIMODAL_RANK if self.benchmark == MULTIMODAL_BENCHMARK else _TEXT_RANK


@dataclass
class CampaignPlan:
    """The static, resumable schedule: ordered phases plus every named exclusion."""

    phases: list[ModelPhase] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    retired: dict[str, str] = field(default_factory=dict)
    duplicates: dict[str, str] = field(default_factory=dict)
    requested_workers: int = 1
    effective_workers: int = 0

    @property
    def models(self) -> list[str]:
        """The runnable model ids, in first-phase (highest-priority) order."""
        return list(dict.fromkeys(phase.model for phase in self.phases))

    def phase_of(self, model: str) -> ModelPhase | None:
        """The model's first (text-first) phase, or ``None`` when it is not scheduled."""
        return next((phase for phase in self.phases if phase.model == model), None)


class PrioritySource(Protocol):
    """The ``collection_priority`` surface the planner consumes (injectable for tests)."""

    def priority(self, identity: str) -> float: ...

    def worth_collecting(self, identity: str) -> tuple[bool, str]: ...

    def runnable_benchmarks(self, identity: str) -> list[str]: ...

    def duplicate_of(self, identity: str) -> str | None: ...


class _ModulePriority:
    """The live priority model: the module-level API over the committed corpus and YAML."""

    def priority(self, identity: str) -> float:
        return cp.priority(identity)

    def worth_collecting(self, identity: str) -> tuple[bool, str]:
        return cp.worth_collecting(identity)

    def runnable_benchmarks(self, identity: str) -> list[str]:
        return cp.runnable_benchmarks(identity)

    def duplicate_of(self, identity: str) -> str | None:
        return cp.duplicate_of(identity)


def _subset_exempt(exempt: Iterable[str] | None) -> set[str]:
    """The channels exempt from duplicate-dropping: the caller's set, else the named subset."""
    if exempt is not None:
        return set(exempt)
    from benchmark import config

    return set(config.concordance_subset_models())


def build_plan(
    models: Sequence[str],
    *,
    requested_workers: int = 1,
    engine: PrioritySource | None = None,
    exempt_duplicates: Iterable[str] | None = None,
) -> CampaignPlan:
    """Build the ordered, named schedule for ``models`` from the priority API.

    A duplicate model is dropped first (another channel already serves its identity), then
    a model with no benchmark left is RETIRED with its reason, then a not-worth model is
    SKIPPED with its reason. Every surviving (model, benchmark) phase is sorted by
    descending priority, then model name, then text before multimodal — a total order, so
    the same committed corpus yields the same plan on every run.

    ``exempt_duplicates`` is the one dedupe exception: the named cross-provider concordance
    subset must survive plan-level dedupe so one identity is measured on several providers.
    Absent an explicit set, the committed subset is read here, which makes ``build_plan`` the
    SINGLE owner of the exemption — a caller that wraps the priority engine to fake it is
    redundant, and a caller that forgets the wrapper no longer silently drops the subset.
    Coverage-based dedupe still applies at ``classify_cells``' fan-out cap.
    """
    source = engine if engine is not None else _ModulePriority()
    exempt = _subset_exempt(exempt_duplicates)
    plan = CampaignPlan(requested_workers=max(0, int(requested_workers)))
    phases: list[ModelPhase] = []
    for model in dict.fromkeys(models):
        if model not in exempt:
            duplicate = source.duplicate_of(model)
            if duplicate is not None:
                plan.duplicates[model] = duplicate
                continue
        keep, reason = source.worth_collecting(model)
        runnable = source.runnable_benchmarks(model)
        if not runnable:
            plan.retired[model] = reason
            continue
        if not keep:
            plan.skipped[model] = reason
            continue
        value = float(source.priority(model))
        phases.extend(ModelPhase(model, benchmark, value) for benchmark in runnable)
    phases.sort(key=lambda phase: (-phase.priority, phase.model, phase.phase_rank))
    plan.phases = phases
    plan.effective_workers = effective_workers(plan.requested_workers, len(plan.models))
    return plan


def format_plan(plan: CampaignPlan) -> str:
    """Render the schedule; each exclusion's named reason appears exactly once."""
    lines = [
        f"campaign plan: {len(plan.models)} model(s), {len(plan.phases)} phase(s), "
        f"workers {plan.effective_workers}/{plan.requested_workers}"
    ]
    for phase in plan.phases:
        lines.append(f"  {phase.model:<40} {phase.benchmark:<20} priority={phase.priority:.4g}")
    for model, duplicate in sorted(plan.duplicates.items()):
        lines.append(f"  duplicate: {model} -> {duplicate} (identity already served)")
    for model, reason in sorted(plan.retired.items()):
        lines.append(f"  retired:   {model} — {reason}")
    for model, reason in sorted(plan.skipped.items()):
        lines.append(f"  skipped:   {model} — {reason}")
    return "\n".join(lines)


@dataclass
class PriorityLaneScheduler(lane_scheduler.LaneScheduler):
    """The lane pull loop ordered by model value, with per-model text-before-multimodal.

    Admission, quarantine, day buckets and the pull loop itself stay in
    :class:`lane_scheduler.LaneScheduler`; this subclass overrides only the choice of
    WHICH admissible cell runs next. A blocked high-priority lane drops out of the
    candidate set, so the next priority model's cell is admitted — idle-capacity fill.
    """

    priority_of: dict[str, float] = field(default_factory=dict)
    text_pending: dict[str, int] = field(default_factory=dict)
    multimodal_cells: frozenset[Cell] = frozenset()

    @classmethod
    def from_plan(
        cls,
        plan: CampaignPlan,
        planned: Mapping[str, Sequence[Cell]],
        *,
        lanes: lane_scheduler.LaneScheduler,
    ) -> PriorityLaneScheduler:
        """Wrap ``lanes`` with the plan's priority order and its per-model phase backlog."""
        priority_of = {phase.model: phase.priority for phase in plan.phases}
        text_pending: dict[str, int] = {}
        multimodal: set[Cell] = set()
        for phase in plan.phases:
            for cell in planned.get(phase.benchmark, ()):
                if cell[1] != phase.model:
                    continue
                if phase.benchmark == TEXT_BENCHMARK:
                    text_pending[phase.model] = text_pending.get(phase.model, 0) + 1
                else:
                    multimodal.add(cell)
        return cls(
            limits=lanes.limits,
            state=lanes.state,
            reserves=lanes.reserves,
            stall_timeout_s=lanes.stall_timeout_s,
            served=lanes.served,
            priority_of=priority_of,
            text_pending=text_pending,
            multimodal_cells=frozenset(multimodal),
        )

    def text_remaining(self, model: str) -> int:
        """How many text cells of ``model`` are still waiting (0 retires the gate)."""
        return self.text_pending.get(model, 0)

    def select_next(self, pending: Sequence[Cell], now: float) -> Cell | lane_scheduler.Stalled:
        """The highest-priority admissible cell; ties keep plan order. STALLED if none.

        A multimodal cell whose model still has text pending is skipped, so a model that
        has not finished text never jumps the gate even when it is the highest priority.
        """
        best: tuple[tuple[float, int], Cell] | None = None
        for index, cell in enumerate(pending):
            model = cell[1]
            if cell in self.multimodal_cells and self.text_remaining(model) > 0:
                continue
            if not self.can_admit(model, now):
                continue
            key = (-self.priority_of.get(model, 0.0), index)
            if best is None or key < best[0]:
                best = (key, cell)
        return lane_scheduler.STALLED if best is None else best[1]

    def next_ready(self, pending: Sequence[Cell], now: float) -> Cell | lane_scheduler.Stalled:
        """Priority pull in place of the base least-recently-served fairness."""
        return self.select_next(pending, now)

    def admit(self, cell: Cell, now: float, *, tokens: int | None = None) -> bool:
        """Admit through the base windows, consuming one unit of a text cell's backlog."""
        admitted = super().admit(cell, now, tokens=tokens)
        if admitted and cell not in self.multimodal_cells:
            model = cell[1]
            if model in self.text_pending:
                self.text_pending[model] = max(0, self.text_pending[model] - 1)
        return admitted


@dataclass(frozen=True)
class CampaignRun:
    """A built campaign: the ordered plan, its MISSING-cell set, and the priority scheduler."""

    plan: CampaignPlan
    cells: Mapping[str, Sequence[Cell]]
    scheduler: PriorityLaneScheduler


def build_campaign(
    models: Sequence[str],
    cells: Sequence[Cell],
    *,
    benchmark: str,
    lanes: lane_scheduler.LaneScheduler,
    requested_workers: int = 1,
    engine: PrioritySource | None = None,
) -> CampaignRun:
    """Build the plan and keep ONLY the cells of its runnable models.

    A caller hands every cell that needs computing (``classify_cells.to_run`` — MISSING plus
    STALE); models the plan excludes (duplicate, not-worth, retired) have their cells dropped
    here, so the executor can never run work the value model refused. The base ``lanes``
    scheduler supplies admission, quotas and persisted state; the priority subclass overrides
    only the choice of which admissible cell is pulled next.
    """
    plan = build_plan(models, requested_workers=requested_workers, engine=engine)
    runnable = set(plan.models)
    planned = {benchmark: [cell for cell in cells if cell[1] in runnable]}
    scheduler = PriorityLaneScheduler.from_plan(plan, planned, lanes=lanes)
    return CampaignRun(plan=plan, cells=planned, scheduler=scheduler)


def run_cells(
    run: CampaignRun,
    ctx: run_matrix._LiveContext,
    *,
    tracker: run_matrix._FailureTracker | None = None,
    checkpoint: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run a built campaign priority-first, reusing the runner's pull loop.

    ``run_matrix._run_scheduled_batch`` owns execution, sentinel classification,
    rate-limit quarantine, stall timeout and the checkpoint merge; this function only
    supplies the priority-ordered scheduler and the flattened MISSING-cell set. Sources
    (text, then multimodal) are run by the caller because the live executor resolves its
    spec module from the configured manifest; a mixed pool would grade against one split.
    """
    from benchmark.runner import run_matrix  # lazy: keep the pure planner import light

    failures = tracker if tracker is not None else run_matrix._FailureTracker(None, None)
    cells = [cell for bench in run.cells for cell in run.cells[bench]]
    rows, _spent, _stopped = run_matrix._run_scheduled_batch(
        cells, ctx, run.scheduler, failures, checkpoint, 0.0, None, None, ""
    )
    return rows


def _model_ids(args: argparse.Namespace) -> list[str]:
    """The overlay model list from ``--extra-models``, else every configured overlay row."""
    from benchmark import config

    if args.extra_models:
        return [name.strip() for name in args.extra_models.split(",") if name.strip()]
    return sorted(config.free_registry_ids())


def main(argv: list[str] | None = None) -> int:
    """CLI: load the overlay, build the plan, print it ordered and named. No model call.

    Execution is the documented :func:`run_cells` function (or the existing runner with
    ``lanes=PriorityLaneScheduler.from_plan(...)``); this CLI is the dry-run an operator
    checks before spending host-hours.
    """
    from benchmark import config

    parser = argparse.ArgumentParser(
        prog="benchmark.runner.campaign_scheduler",
        description="Priority-first, per-model phase scheduler for the free-tier campaign.",
    )
    parser.add_argument("--config", default="configs/free-tier/benchmark.yaml")
    parser.add_argument("--free-registry", default=None)
    parser.add_argument("--extra-models", default=None, help="comma-separated overlay ids")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    config.load(args.config)
    if args.free_registry:
        config.set_free_registry(args.free_registry)
    models = _model_ids(args)
    if not models:
        print("no overlay models: pass --extra-models or --free-registry", file=sys.stderr)
        return 2
    print(format_plan(build_plan(models, requested_workers=args.workers)))  # noqa: T201 - CLI
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

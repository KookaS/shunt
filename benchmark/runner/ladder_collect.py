#!/usr/bin/env python3
"""Ladder collection: cheap-first, escalate a task until its first passing tier."""

# Under the monotonicity axiom this observes each task's crossover tier tau EXACTLY with
# ZERO UNKNOWN gaps at minimum cost — everything above the first pass is imputed-pass, below
# is real-fail. It needs no audit draw:
# there is no non-discriminating remainder to sample. Reuses `collect_phase` ONE ladder rung
# per call — classify/run/merge, --max-cost, checkpointing, and the container-start /
# overshoot machinery are inherited verbatim. Simulated by default; --live delegates to the
# real harness exactly as collect.py does. State lives entirely in results.csv, so a killed
# run resumes: a rung skips any task whose cached rung already passed.

from __future__ import annotations

import argparse
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from benchmark import config
from benchmark.config import RankedModel
from benchmark.routing import integrity
from benchmark.runner import infer
from benchmark.runner.collect import _refuse_live, _resolve_digests
from benchmark.runner.run_matrix import (
    _FailureTracker,
    _has_keys,
    collect_phase,
    preflight_refuses,
)
from shunt.secrets import load_dotenv_file


def _confirm_uncapped_live() -> bool:
    """Interactive gate for `ladder --live` with no --max-cost; abort unless the user types y."""
    # Mirrors run_matrix._confirm_uncapped_live: a fully uncapped ladder --live run can spend
    # unbounded real budget, so it MUST be confirmed and MUST refuse on non-interactive stdin.
    from benchmark.runner.run_matrix import _prompt_confirm  # noqa: PLC0415

    answer = _prompt_confirm(
        "LADDER --live with NO --max-cost cap can spend unbounded real money. Proceed? [y/N] "
    )
    if answer is None:
        print("  aborted: non-interactive stdin, refusing uncapped live spend.")
        return False
    choice = answer.strip().lower()
    if choice == "y":
        return True
    reason = "by user" if choice == "n" else f"invalid input {answer!r} (expected y or n)"
    print(f"  aborted: {reason}.")
    return False


def _sampled_tasks() -> list[str]:
    """The pinned sampled challenge set (same seed/sample as every other run mode)."""
    hashes = integrity.all_hashes()
    seed = config.benchmark_params().get("seed", 42)
    return config.sample_tasks(sorted(hashes.keys()), seed=seed)


def _solved_matrix() -> dict:
    """Default-arm-flattened current cache (challenge -> model -> outcome row)."""
    return config.flatten_default_arm(config.load_results())


def _task_solved(matrix: dict, task: str, rung_models: list[str]) -> bool:
    """True iff the task has a cached PASS at any rung (already crossed — skip on resume)."""
    cells = matrix.get(task, {})
    return any(bool(cells.get(m, {}).get("pass")) for m in rung_models)


def _untested_tiers(matrix: dict, task: str, rung_models: list[str]) -> list[str]:
    """Rungs with NO cached cell for this task — the tiers still to run to complete it.

    A cached FAIL below and a cached PASS above are both already observed; only tiers with
    no row remain. Makes the loop resumable: a prior interrupted run's cached rungs are reused.
    """
    cells = matrix.get(task, {})
    return [m for m in rung_models if m not in cells]


def _prepare(tasks: list[str]) -> tuple[list[str], dict, dict, dict | None, Path]:
    """Resolve the shared collect inputs once (hashes, versions, digests, results path)."""
    hashes = integrity.all_hashes()
    versions = integrity.model_versions()
    check_images = bool(config.collect_config().get("check_images", False))
    digests = _resolve_digests(tasks, check_images)
    return tasks, hashes, versions, digests, config.results_csv_path()


def run_ladder(
    config_path: str = "benchmark/benchmark.yaml",
    *,
    tasks: list[str] | None = None,
    live: bool = False,
    timeout: int = infer._AGENT_WALL_LIMIT_S,
    workers: int = 1,
    max_cost: float | None = None,
    max_cost_overshoot: float = 0.0,
    max_start_failures: int | None = None,
    max_consecutive_failures: int | None = None,
    check_images: bool = False,  # noqa: ARG001 (parity with sibling collectors; digests via config)
    step_limit: int | None = None,
) -> int:
    """Drive the ladder; returns a process exit code (0 ok, 2 refused).

    ``tasks`` overrides the sampled set for a TARGETED run (still-unknown crossovers only).
    """
    load_dotenv_file()
    config.load(config_path)
    step_limit = config.live_step_limit() if step_limit is None else step_limit
    ladder = config.capability_rank().ordered
    if not ladder:
        print("  REFUSING: empty capability rank (no enabled models to collect).")
        return 2

    tasks = tasks if tasks is not None else _sampled_tasks()
    live = live and _has_keys()
    ladder_models = [rung.model for rung in ladder]
    if live and not bool(config.collect_config().get("constants_pinned", False)):
        print("  REFUSING --live: collect.constants_pinned is false — pin the sizing first.")
        return 2
    # Uncapped live spend must be confirmed (refused on non-interactive stdin), like full matrix.
    if live and max_cost is None and not _confirm_uncapped_live():
        return 3
    if _refuse_live(live, ladder_models, []):
        return 2
    # Preflight: one real $0 completion proves the key works BEFORE any container starts.
    if preflight_refuses(live):
        return 2

    _, hashes, versions, digests, results_path = _prepare(tasks)
    _run_challenges(
        ladder,
        tasks,
        hashes,
        versions,
        digests,
        results_path,
        live=live,
        timeout=timeout,
        workers=workers,
        max_cost=max_cost,
        max_cost_overshoot=max_cost_overshoot,
        max_start_failures=max_start_failures,
        max_consecutive_failures=max_consecutive_failures,
        step_limit=step_limit,
    )
    if not live:
        print("  simulated: rungs classified only; leaving cells uncached (no fabrication).")
    return 0


def _total_real_cost(results_path: Path) -> float:
    """Sum real_cost across every cached cell — the on-disk source of truth for spend."""
    import csv  # noqa: PLC0415

    if not results_path.exists():
        return 0.0
    total = 0.0
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                total += float(row.get("real_cost") or 0.0)
            except ValueError:
                continue
    return total


def _tier_cost_ceilings(results_path: Path) -> dict[str, float]:
    """Conservative per-model reservation cost: the MAX observed ``real_cost`` (an upper bound)."""
    # The median is exceeded ~50% of the time, so it is NOT a worst case; the max is a true-ish
    # upper bound on a tier's remaining cost, which is what the reservation needs to be safe.
    import csv  # noqa: PLC0415

    per_model: dict[str, list[float]] = {}
    if results_path.exists():
        with open(results_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    cost = float(row.get("real_cost") or 0.0)
                except ValueError:
                    continue
                per_model.setdefault(str(row.get("model") or ""), []).append(cost)
    return {m: max(v) for m, v in per_model.items() if v}


def _worst_case_to_complete(
    untested: list[str], ceilings: dict[str, float], cold_start_cost: float
) -> float:
    """Conservative UPPER-BOUND cost to finish a challenge: sum of untested-tier ceilings."""
    # An UNMEASURED tier reserves ``cold_start_cost`` (or the priciest known ceiling if that is
    # higher) — NEVER 0, so a fresh run cannot admit an unbounded concurrent first wave.
    fallback = max(cold_start_cost, max(ceilings.values(), default=0.0))
    return sum(ceilings.get(m, fallback) for m in untested)


@dataclass
class _LadderRun:
    """One ladder run's shared state: the immutable job plus the thread-safe coordinator."""

    # The escalation ACROSS challenges is concurrent (up to ``workers``); WITHIN a challenge it
    # stays serial cheap->strong. ``reservations`` (guarded by ``lock``) holds each in-flight
    # challenge's CONSERVATIVE worst-case remaining cost (a true-ish upper bound: max observed
    # real_cost per tier, cold_start_tier_cost for still-unmeasured tiers). The lock only guards
    # single-threaded mutation of that state; the OVERSPEND guarantee comes from the arithmetic —
    # admitting a challenge only when spent + all reservations + its worst case fits ``max_cost``,
    # so N concurrent challenges can never jointly cross the budget.
    ladder: list[RankedModel]
    ladder_models: list[str]
    tasks: list[str]
    hashes: dict
    versions: dict
    digests: dict | None
    results_path: Path
    live: bool
    timeout: int
    workers: int
    max_cost: float | None
    max_cost_overshoot: float
    baseline: float
    failures: _FailureTracker
    cold_start_tier_cost: float
    step_limit: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    write_lock: threading.Lock = field(default_factory=threading.Lock)
    abort: threading.Event = field(default_factory=threading.Event)
    reservations: dict[str, float] = field(default_factory=dict)
    abort_error: BaseException | None = None


def _latch_abort(run: _LadderRun, exc: BaseException) -> None:
    """Latch a run-level abort so healthy siblings stop dispatching at once, not only at harvest."""
    with run.lock:
        if run.abort_error is None:
            run.abort_error = exc
    run.abort.set()


def _escalate_task(run: _LadderRun, task: str) -> None:
    """Run ONE challenge cheap->strong until its first pass (SERIAL within the challenge)."""
    # A failing worker latches the abort BEFORE its future is harvested, so a healthy sibling
    # completing first cannot slip more dispatch past a dead API between failure and harvest.
    try:
        _escalate_tiers(run, task)
    except BaseException as exc:
        _latch_abort(run, exc)
        raise


def _escalate_tiers(run: _LadderRun, task: str) -> None:
    """Walk the ladder cheap->strong for one challenge; stop at the first observed pass."""
    # Only DIFFERENT challenges run concurrently; tiers of the same challenge never do — tier N+1
    # runs only if tier N failed. Cached rungs are reused (resume); no ``max_cost`` cap WITHIN a
    # challenge (the reservation already deemed it affordable). Bails between rungs on shared abort.
    for rung in run.ladder:
        if run.abort.is_set():
            return  # a sibling worker latched a run-level abort — stop between rungs
        matrix = _solved_matrix()
        cell = matrix.get(task, {}).get(rung.model)
        if cell is not None:
            if cell.get("pass"):
                return  # already passed here (cached) — stop; higher tiers impute pass
            continue  # cached fail — escalate to the next tier
        collect_phase(
            [task],
            [rung.model],
            config.load_results(),
            run.hashes,
            run.versions,
            run.digests,
            live=run.live,
            timeout=run.timeout,
            workers=1,  # concurrency is ACROSS challenges; each rung is a single cell
            max_cost=None,
            results_path=run.results_path,
            max_cost_overshoot=run.max_cost_overshoot,
            max_start_failures=None,  # counted via the shared ``failures`` tracker instead
            max_consecutive_failures=None,
            write_lock=run.write_lock,
            failures=run.failures,
            step_limit=run.step_limit,
        )
        cell = _solved_matrix().get(task, {}).get(rung.model)
        if cell is not None and cell.get("pass"):
            return  # first pass observed — stop escalating (above is imputed pass)
    # Fell through every tier without a pass: all tiers observed FAIL → complete + unsolvable.


def _reserve(run: _LadderRun, task: str) -> str:
    """Decide a pending challenge under the shared lock: 'skip', 'reserved', or 'blocked'."""
    # Completion-predictive RESERVATION: dispatch only if committed spend + every in-flight
    # reservation + this challenge's conservative worst case fits ``max_cost``. The reservation is
    # an UPPER BOUND (max observed real_cost per tier; cold_start for unmeasured tiers), so N
    # concurrent challenges can never jointly cross the budget and a started challenge — which runs
    # uncapped (max_cost=None) — is safe because admission proved its worst case already fit.
    matrix = _solved_matrix()
    if _task_solved(matrix, task, run.ladder_models):
        return "skip"  # already crossed (resume) — nothing to collect
    untested = _untested_tiers(matrix, task, run.ladder_models)
    if not untested:
        return "skip"  # every tier already observed (all fail) — complete + unsolvable
    with run.lock:
        if run.max_cost is None:
            return "reserved"
        spent = _total_real_cost(run.results_path) - run.baseline
        if spent >= run.max_cost:
            return "blocked"  # FIX 3 hard stop: committed spend already reached the cap
        reserved = sum(run.reservations.values())
        ceilings = _tier_cost_ceilings(run.results_path)
        worst = _worst_case_to_complete(untested, ceilings, run.cold_start_tier_cost)
        if spent + reserved + worst > run.max_cost:
            return "blocked"
        run.reservations[task] = worst
        return "reserved"


def _dispatch_ready(
    run: _LadderRun, pool: ThreadPoolExecutor, in_flight: dict[Future[None], str], next_idx: int
) -> int:
    """Submit in-order challenges that fit the budget and a free worker; return the new cursor.

    Stops advancing at the first challenge the budget can't fit RIGHT NOW (challenge-atomic, in
    canonical order) so a drain frees headroom and it is retried — the prefix serial dispatches.
    """
    while next_idx < len(run.tasks) and len(in_flight) < run.workers and not run.abort.is_set():
        task = run.tasks[next_idx]
        decision = _reserve(run, task)
        if decision == "blocked":
            break
        next_idx += 1
        if decision == "reserved":
            print(f"Escalating {task}")
            in_flight[pool.submit(_escalate_task, run, task)] = task
    return next_idx


def _harvest(run: _LadderRun, in_flight: dict[Future[None], str]) -> None:
    """Wait for >=1 challenge to finish, release its reservation, and latch any abort."""
    done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
    for fut in done:
        task = in_flight.pop(fut)
        with run.lock:
            run.reservations.pop(task, None)
        exc = fut.exception()
        if exc is not None:
            _latch_abort(run, exc)  # API-unusable / consecutive-failure abort → halt new dispatch


def _run_challenges(
    ladder: list[RankedModel],
    tasks: list[str],
    hashes: dict,
    versions: dict,
    digests: dict | None,
    results_path: Path,
    *,
    live: bool,
    timeout: int,
    workers: int,
    max_cost: float | None,
    max_cost_overshoot: float,
    max_start_failures: int | None,
    max_consecutive_failures: int | None,
    step_limit: int = infer._DEFAULT_STEP_LIMIT,
) -> None:
    """Escalate up to ``workers`` challenges CONCURRENTLY (each cheap->strong serially within)."""
    # Same fan-out mechanism and defaults as cost_optimal/full. Challenge-atomic budgeting holds
    # under concurrency via ``_reserve``'s worst-case reservation; a started challenge always
    # finishes. A run-level abort halts new dispatch and drains the in-flight challenges.
    run = _LadderRun(
        ladder=ladder,
        ladder_models=[r.model for r in ladder],
        tasks=tasks,
        hashes=hashes,
        versions=versions,
        digests=digests,
        results_path=results_path,
        live=live,
        timeout=timeout,
        workers=max(1, workers),
        max_cost=max_cost,
        max_cost_overshoot=max_cost_overshoot,
        baseline=_total_real_cost(results_path) if max_cost is not None else 0.0,
        failures=_FailureTracker(max_start_failures, max_consecutive_failures),
        cold_start_tier_cost=config.cold_start_tier_cost(),
        step_limit=step_limit,
    )
    next_idx = 0
    in_flight: dict[Future[None], str] = {}
    with ThreadPoolExecutor(max_workers=run.workers) as pool:
        while (next_idx < len(run.tasks) or in_flight) and not run.abort.is_set():
            next_idx = _dispatch_ready(run, pool, in_flight, next_idx)
            if not in_flight:
                if next_idx < len(run.tasks) and run.max_cost is not None:
                    print(
                        f"  budget guard: ${run.max_cost:g} cap reached — stopping before the "
                        f"remaining {len(run.tasks) - next_idx} challenge(s) (challenge-atomic)."
                    )
                break
            _harvest(run, in_flight)
    if run.abort_error is not None:
        raise run.abort_error


def _add_args(ap: argparse.ArgumentParser, config_path: str) -> None:
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--live", action="store_true", help="Run uncached cells for real")
    ap.add_argument(
        "--timeout",
        type=int,
        default=infer._AGENT_WALL_LIMIT_S,
        help="Generous graceful per-cell wall-clock backstop (s); PRIMARY bound is --step-limit.",
    )
    ap.add_argument(
        "--step-limit",
        type=int,
        default=None,
        help="Primary model-speed-agnostic per-cell bound (max agent steps). "
        "Default: benchmark.yaml live.step_limit.",
    )
    ap.add_argument(
        "--workers", type=int, default=1, help="Challenges to escalate concurrently (1 = serial)"
    )
    ap.add_argument("--max-cost", type=float, default=None, help="Abort once real_cost crosses USD")


def main(config_path: str = "benchmark/benchmark.yaml") -> int:
    ap = argparse.ArgumentParser(
        description="Shunt ladder collection (cheap-first; simulated by default)."
    )
    _add_args(ap, config_path)
    args = ap.parse_args()
    return run_ladder(
        args.config,
        live=args.live,
        timeout=args.timeout,
        workers=args.workers,
        max_cost=args.max_cost,
        step_limit=args.step_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())

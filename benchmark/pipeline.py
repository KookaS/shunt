"""Unified benchmark pipeline: collect -> stamp -> evaluate -> report -> figures -> summary."""

# One entrypoint that composes the existing stage modules (run_matrix, offline_replay,
# escalation.run_eval, routing.report) so a single command regenerates the CORE artifacts:
# results.csv, the stamped live trajectories, the escalation metrics + plots, and the
# routing.report figures (pareto_scatter, cost_savings, cost_quality_equal, cumulative_regret,
# the heatmap, capability_evidence.json, and the coverage/summary CSVs).
# The standalone plots under benchmark/routing/scripts/ run in the FIGURES stage (see
# STANDALONE_FIGURES): they are heavy — several load the real fastembed embedder — so they
# are not part of a --live collection run, but `--from figures` refreshes all of them and
# `--check-figures` proves the committed PNGs are not stale without regenerating anything.
# Each stage shells out to its module unchanged; this file only orchestrates them.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from benchmark import config, corpus_lock
from benchmark.escalation import features, schema
from benchmark.escalation.live_capture import LIVE_DIR
from benchmark.runner import replay_admissibility

logger = logging.getLogger(__name__)

COLLECT = "collect"
STAMP = "stamp"
EVALUATE = "evaluate"
REPORT = "report"
FIGURES = "figures"
STAGE_ORDER = (COLLECT, STAMP, EVALUATE, REPORT, FIGURES)

RUN_MATRIX = "benchmark.runner.run_matrix"
OFFLINE_REPLAY = "benchmark.runner.offline_replay"
ESCALATION_EVAL = "benchmark.escalation.run_eval"
ROUTING_REPORT = "benchmark.routing.report"
ROUTING_EVAL = "benchmark.routing.run_eval"

# One measured sympy trajectory took 1 271 s to replay, so the old 120 s ceiling silently
# dropped exactly the slowest (and largest) trajectories — the stage logged `skipping straggler`
# and reported success. An hour is above every trajectory measured so far; a run that still
# exceeds it is now a LOUD stage failure rather than a skip (see `stage_stamp`).
DEFAULT_REPLAY_TIMEOUT = 3600.0

# Per-trajectory record of which replay instrument last completed it. Two jobs, one file:
# it is the resume point for an interrupted rebuild, and it is what tells a trajectory the
# admissibility gate CLEARED apart from one the stamping stage never reached — both look
# identical on disk (`confirmed=False`, `exit_code=None`) by design.
STAMP_LEDGER_NAME = "stamp_ledger.json"
_ROUTING_REPORTS_DIR = Path("benchmark/routing/reports")
_ESCALATION_PLOTS_DIR = Path("benchmark/escalation/reports")

_RAN = "ran"
_FAILED = "failed"
_SKIPPED = "skipped"


class StageError(RuntimeError):
    """A stage's underlying module exited non-zero — caught so downstream stages still run."""


@dataclass
class PipelineState:
    """Cross-stage scratch: the evaluate stage's captured stdout feeds the summary."""

    evaluate_stdout: str = ""


@dataclass
class PipelineResult:
    """The pipeline's exit code plus a per-stage ran/failed/skipped ledger for the summary."""

    returncode: int
    outcomes: dict[str, str] = field(default_factory=dict)


def _banner(stage: str) -> None:
    """Loud per-stage marker so a supervising monitor can tell collection from reporting."""
    print(f"=== [pipeline] stage: {stage} ===", flush=True)  # noqa: T201


def run_module(
    module: str, argv: list[str], *, timeout: float | None = None, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    """Invoke an existing benchmark module as `python -m <module> <argv>` (the single seam).

    capture=True buffers combined output for the summary; otherwise stdout/stderr stream live.
    """
    cmd = [sys.executable, "-m", module, *argv]
    if capture:
        return subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, check=False)
    return subprocess.run(cmd, timeout=timeout, text=True, check=False)


def _collect_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--strategy",
        args.strategy,
        "--config",
        args.config,
        "--timeout",
        str(args.timeout),
        "--workers",
        str(args.workers),
        "--max-cost-overshoot",
        str(args.max_cost_overshoot),
        "--max-start-failures",
        str(args.max_start_failures),
    ]
    if args.live:
        argv.append("--live")
    if args.max_cost is not None:
        argv += ["--max-cost", str(args.max_cost)]
    if args.check_images:
        argv.append("--check-images")
    return argv


def stage_collect(args: argparse.Namespace, _state: PipelineState) -> None:
    """Run the outcome matrix (run_matrix) with the passed strategy/live/budget flags."""
    result = run_module(RUN_MATRIX, _collect_argv(args))
    if result.returncode != 0:
        raise StageError(f"{RUN_MATRIX} exited {result.returncode}")


def load_stamp_ledger(live_dir: Path = LIVE_DIR) -> dict[str, str]:
    """trajectory_id -> the instrument digest that last completed it (empty when absent/bad)."""
    path = live_dir / STAMP_LEDGER_NAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("stamp: unreadable %s — treating every trajectory as pending", path)
        return {}
    return {str(k): str(v) for k, v in payload.items()}


def record_stamped(live_dir: Path, trajectory_id: str, digest: str) -> None:
    """Mark *trajectory_id* done under *digest* so an interrupted rebuild resumes where it left."""
    # Read-modify-write on the resume point, called from every worker thread as its replay lands.
    # Under the lock and through an atomic write, so a lost update cannot silently re-queue an
    # already-completed trajectory and a kill mid-write cannot truncate the ledger (which
    # `load_stamp_ledger` would then read as "nothing is done" — a full 137 h restart).
    with corpus_lock.corpus_lock(live_dir):
        ledger = load_stamp_ledger(live_dir)
        ledger[trajectory_id] = digest
        corpus_lock.atomic_write_text(
            live_dir / STAMP_LEDGER_NAME, json.dumps(ledger, indent=2, sort_keys=True) + "\n"
        )


def pending_trajectories(
    live_dir: Path = LIVE_DIR, *, restamp: bool = False
) -> list[tuple[str, str, Path]]:
    """Live trajectories the stamping stage still owes, under the CURRENT replay instrument."""
    # Two predicates, and both are needed.
    #
    # `features.is_stamped` is the eval's own test, so the default queue matches what the eval
    # will actually score. (It used to be `any(step.failing_check_id)`, which diverged: a fully
    # replayed run in which no step ever failed carries no check id, so the eval scored it while
    # this queued it forever.) But it is one-way: a trajectory the admissibility gate CLEARED is
    # unstamped by construction, so on its own `is_stamped` would re-queue every rejected
    # instance on every run, for ever.
    #
    # The ledger closes that, and is also the only thing that can drive a full rebuild:
    # `--restamp` deliberately ignores `is_stamped` (every committed trajectory is already
    # stamped, which is why `--from stamp` was a no-op on the corpus) and leans on the ledger
    # alone, so an interrupted 20–90 h rebuild resumes instead of restarting.
    pending: list[tuple[str, str, Path]] = []
    if not live_dir.exists():
        return pending
    ledger = load_stamp_ledger(live_dir)
    digest = replay_admissibility.instrument_digest()
    for path in sorted(live_dir.glob("*.jsonl")):
        try:
            traj = schema.load_jsonl(path)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("stamp: skipping unreadable %s (%s)", path, exc)
            continue
        trajectory_id = traj.header.trajectory_id
        if ledger.get(trajectory_id) == digest:
            continue  # this instrument already finished it (stamped, or cleared as inadmissible)
        if not restamp and features.is_stamped(traj):
            continue
        instance_id = traj.header.instance_id
        if not instance_id:
            logger.warning("stamp: skipping %s (no instance_id in header)", path)
            continue
        pending.append((trajectory_id, instance_id, path))
    return pending


def _reap_replay_container(trajectory_id: str) -> None:
    """Kill the container a timed-out replay left behind (the child dies, the container lives)."""
    subprocess.run(
        ["docker", "rm", "-f", f"shunt-replay-{trajectory_id}"], capture_output=True, check=False
    )


# HOW MANY REPLAYS RUN AT ONCE. Each one is a container executing an instance's test file, which
# pins roughly one core, so the ceiling is CPU, not I/O — and oversubscribing costs about 2x, so
# a too-high number is not merely neutral. Two cores are held back for the parent, the docker
# daemon and the host.
#
# THE CAP IS MEMORY, NOT CPU, and that is why it is well below the core count. The heavy instance
# families (sympy, matplotlib, sphinx, django) run whole test FILES, in the hundreds-of-MB-to-GB
# range each, against ~9.6 GB available on a 15.9 GB host — so the core-derived number (14 here)
# would OOM long before it saturated the CPU, and an OOM-killed container is an infra failure that
# costs the trajectory rather than merely slowing it. Six leaves clear headroom while still
# turning a measured ~137 h serial rebuild into roughly a day. Raise it with `--stamp-workers` on
# a machine with more RAM; that flag exists precisely because this ceiling is host-specific.
_MAX_STAMP_WORKERS: Final = 6
_RESERVED_CORES: Final = 2
# How often the stage says where it is. Long enough to stay quiet next to the per-trajectory
# completion lines, short enough that "hung" is visible within a minute rather than at the end.
_HEARTBEAT_S: Final = 60.0


def default_stamp_workers() -> int:
    """Parallel replays to run by default: host cores minus the host's share, capped by memory."""
    return max(1, min(_MAX_STAMP_WORKERS, (os.cpu_count() or 1) - _RESERVED_CORES))


def group_by_instance(
    pending: list[tuple[str, str, Path]],
) -> list[list[tuple[str, str, Path]]]:
    """One parallel work unit per INSTANCE; its own trajectories replay serially inside it."""
    # THIS GROUPING IS A CORRECTNESS BOUNDARY, not a scheduling nicety. The admissibility gate is
    # a property of the INSTANCE and costs two full container test runs; its verdict is cached in
    # `admissibility.json`, and in the serial loop the first trajectory of an instance measures it
    # and the other 4.8 on average hit the cache. Hand two trajectories of ONE instance to two
    # workers and both miss the cache and both run the gate — up to 11x redundant container work
    # on the largest instance, and, because the legs are real test runs, two runs CAN disagree
    # (one flaky test in FAIL_TO_PASS u PASS_TO_PASS is enough). A disagreement leaves some of an
    # instance's trajectories stamped and the rest actively CLEARED, which the serial loop can
    # never produce. Keeping an instance whole in one worker removes that race by construction
    # rather than by locking it.
    groups: dict[str, list[tuple[str, str, Path]]] = {}
    for item in pending:
        groups.setdefault(item[1], []).append(item)
    # Longest instance first: an 11-trajectory unit scheduled last is a tail that leaves every
    # other worker idle while it drains.
    return sorted(groups.values(), key=len, reverse=True)


def _rejected_instances(live_dir: Path) -> int:
    """How many instances the gate has rejected so far, read from the committed verdict file."""
    verdicts = replay_admissibility.load_verdicts(live_dir)
    return sum(1 for v in verdicts.values() if not v.admissible)


def _hms(seconds: float) -> str:
    return f"{int(seconds) // 3600:d}:{int(seconds) // 60 % 60:02d}:{int(seconds) % 60:02d}"


@dataclass
class _StampProgress:
    """Live counters + the single output gate, so N workers' lines never interleave mid-line."""

    total: int
    live_dir: Path
    started: float = field(default_factory=time.monotonic)
    done: int = 0
    failures: list[str] = field(default_factory=list)
    in_flight: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, text: str) -> None:
        """Print one block atomically with respect to every other writer in this process."""
        with self.lock:
            print(text, flush=True)  # noqa: T201

    def begin(self, trajectory_id: str) -> None:
        with self.lock:
            self.in_flight[trajectory_id] = time.monotonic()

    def finish(self, trajectory_id: str, failure: str | None) -> str:
        with self.lock:
            self.in_flight.pop(trajectory_id, None)
            self.done += 1
            if failure:
                self.failures.append(failure)
            elapsed = time.monotonic() - self.started
            eta = (elapsed / self.done) * (self.total - self.done)
            state = f"FAILED {failure}" if failure else "ok"
            return (
                f"  stamp [{self.done}/{self.total}] {trajectory_id}: {state} "
                f"| elapsed {_hms(elapsed)} eta {_hms(eta)}"
            )

    def heartbeat(self) -> str:
        """One line saying whether the stage is progressing, stalled, or repeating an error."""
        # `in flight` with a per-trajectory age is what separates SLOW from HUNG: a replay that
        # has been running 40 minutes is named, so a supervisor never has to infer a stall from
        # the absence of output.
        with self.lock:
            now = time.monotonic()
            ages = sorted(((now - t, tid) for tid, t in self.in_flight.items()), reverse=True)
            elapsed = now - self.started
            eta = (elapsed / self.done) * (self.total - self.done) if self.done else 0.0
            last = self.failures[-1] if self.failures else "none"
            flight = ", ".join(f"{tid} {age:.0f}s" for age, tid in ages[:8]) or "none"
            return (
                f"  stamp HEARTBEAT {self.done}/{self.total} done, {self.total - self.done} left "
                f"| {len(self.in_flight)} in flight | {len(self.failures)} failed "
                f"| {_rejected_instances(self.live_dir)} instances rejected "
                f"| elapsed {_hms(elapsed)} eta {_hms(eta)}\n"
                f"  stamp HEARTBEAT in flight: {flight}\n"
                f"  stamp HEARTBEAT last failure: {last}"
            )


def _text(stream: bytes | str | None) -> str:
    """Whatever a captured stream holds, as text (a TimeoutExpired hands back bytes/None)."""
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode("utf-8", "replace")


def _child_block(trajectory_id: str, stdout: bytes | str | None, stderr: bytes | str | None) -> str:
    """One replay's captured output, every line tagged with the trajectory it came from."""
    # Captured and re-emitted rather than streamed: N children sharing one pipe interleave
    # mid-line, and a rebuild log in which a gate verdict cannot be attributed to a trajectory is
    # not diagnosable. Volume is low (a verdict line, maybe a clearing warning, a final line).
    lines = [ln for ln in f"{_text(stderr)}\n{_text(stdout)}".splitlines() if ln.strip()]
    return "".join(f"  [{trajectory_id}] {ln}\n" for ln in lines).rstrip("\n")


def _replay_one(
    item: tuple[str, str, Path],
    *,
    digest: str,
    replay_timeout: float,
    progress: _StampProgress,
) -> None:
    """Replay ONE trajectory in a subprocess and record it in the ledger only if it completed."""
    trajectory_id, instance_id, path = item
    progress.begin(trajectory_id)
    argv = [trajectory_id, "--instance-id", instance_id, "--jsonl", str(path)]
    failure: str | None = None
    block = ""
    try:
        result = run_module(OFFLINE_REPLAY, argv, timeout=replay_timeout, capture=True)
    except subprocess.TimeoutExpired as expired:
        _reap_replay_container(trajectory_id)
        block = _child_block(trajectory_id, expired.stdout, expired.stderr)
        failure = f"{trajectory_id}: exceeded --replay-timeout {replay_timeout}s"
    except Exception as exc:  # noqa: BLE001 — one bad trajectory never aborts the other workers
        failure = f"{trajectory_id}: failed to launch ({exc})"
    else:
        block = _child_block(trajectory_id, result.stdout, result.stderr)
        if result.returncode != 0:
            failure = f"{trajectory_id}: exited {result.returncode}"
        else:
            record_stamped(path.parent, trajectory_id, digest)
    line = progress.finish(trajectory_id, failure)
    progress.emit(f"{block}\n{line}" if block else line)


def _replay_instance(
    group: list[tuple[str, str, Path]],
    *,
    digest: str,
    replay_timeout: float,
    progress: _StampProgress,
) -> None:
    """Replay one instance's trajectories in sequence — the gate measures once, the rest cache."""
    for item in group:
        _replay_one(item, digest=digest, replay_timeout=replay_timeout, progress=progress)


def _beat(progress: _StampProgress, stop: threading.Event, interval: float) -> None:
    while not stop.wait(interval):
        progress.emit(progress.heartbeat())


def run_stamp_stage(
    pending: list[tuple[str, str, Path]],
    *,
    live_dir: Path,
    workers: int,
    replay_timeout: float,
    heartbeat_s: float = _HEARTBEAT_S,
) -> list[str]:
    """Replay every owed trajectory across *workers*, instance by instance; return the failures."""
    digest = replay_admissibility.instrument_digest()
    groups = group_by_instance(pending)
    workers = max(1, min(workers, len(groups)))
    progress = _StampProgress(total=len(pending), live_dir=live_dir)
    progress.emit(
        f"  stamp: {len(pending)} trajectories across {len(groups)} instances, "
        f"{workers} workers, replay-timeout {replay_timeout:.0f}s, digest {digest[:12]}"
    )
    stop = threading.Event()
    beater = threading.Thread(target=_beat, args=(progress, stop, heartbeat_s), daemon=True)
    beater.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _replay_instance,
                    group,
                    digest=digest,
                    replay_timeout=replay_timeout,
                    progress=progress,
                )
                for group in groups
            ]
        for future in futures:
            future.result()  # re-raise anything the per-item guard did not catch
    finally:
        stop.set()
        beater.join(timeout=2.0)
    progress.emit(progress.heartbeat())
    return progress.failures


def stage_stamp(args: argparse.Namespace, _state: PipelineState) -> None:
    """Replay every owed trajectory; a timeout or non-zero exit FAILS the stage, never skips."""
    pending = pending_trajectories(restamp=args.restamp)
    if not pending:
        print("  stamp: no trajectories owed under the current replay instrument")  # noqa: T201
        return
    failures = run_stamp_stage(
        pending,
        live_dir=LIVE_DIR,
        workers=args.stamp_workers,
        replay_timeout=args.replay_timeout,
    )
    if failures:
        # LOUD, not a warning buried in a green run: an unreplayed trajectory keeps whatever it
        # was stamped with before, so a silent skip is how fabricated outcomes survive a rebuild.
        raise StageError(f"{len(failures)}/{len(pending)} replays did not complete: {failures}")


def stage_evaluate(_args: argparse.Namespace, state: PipelineState) -> None:
    """Score the escalation detector (metrics + plots); keep stdout for the summary status."""
    argv = ["--plots-dir", str(_ESCALATION_PLOTS_DIR)]
    result = run_module(ESCALATION_EVAL, argv, capture=True)
    if result.stdout:
        print(result.stdout, end="")  # noqa: T201
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)  # noqa: T201
    state.evaluate_stdout = result.stdout or ""
    if result.returncode != 0:
        raise StageError(f"{ESCALATION_EVAL} exited {result.returncode}")


def stage_report(_args: argparse.Namespace, _state: PipelineState) -> None:
    """Regenerate the routing plots + capability_evidence.json + coverage/summary CSVs."""
    result = run_module(ROUTING_REPORT, ["--out-dir", str(_ROUTING_REPORTS_DIR)])
    if result.returncode != 0:
        raise StageError(f"{ROUTING_REPORT} exited {result.returncode}")


# ---------------------------------------------------------------------------
# The standalone figures + their staleness gate.
#
# Every committed PNG under benchmark/routing/reports/ that report.py does NOT write is
# produced by one of the modules below. They used to sit on no refresh path at all, which
# is how timing_comparison.png shipped for a release cycle without the Price-Cascade bar
# and with a 57%-wrong denominator: the strategy was added, nothing re-ran the producer,
# and no check could tell.
#
# `inputs` is what the figure is ABOUT — the outcome data, the strategy set, and the
# producing script(s). Their combined digest is recorded in FIGURE_MANIFEST when the
# figures stage regenerates; `stale_figures()` recomputes it and reports any drift. That
# check is seconds (it hashes files, it does not draw), so it can run in the test suite
# while the regeneration itself stays a deliberate `make benchmark-figures`.
#
# Deliberately NOT in the digest: the shared analysis modules (summary, impute, metrics).
# They are exercised by their own tests and by the report stage; folding them in would
# turn every unrelated refactor into a 15-minute figure rebuild, and a gate people
# routinely override is not a gate.
# ---------------------------------------------------------------------------

# Repo-root-anchored so a digest is identical wherever the checkout lives (a CWD-relative
# path would hash its own string and make the gate machine-dependent).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROUTING = _REPO_ROOT / "benchmark" / "routing"
_SCRIPTS = _ROUTING / "scripts"
_STRATEGIES = _ROUTING / "strategies"
FIGURE_MANIFEST = _ROUTING / "figure_inputs.json"


@dataclass(frozen=True)
class FigureJob:
    """One standalone figure producer: the module to run, what it writes, what it reads."""

    module: str
    outputs: tuple[str, ...]
    inputs: tuple[Path, ...]

    @property
    def name(self) -> str:
        return self.module.rsplit(".", 1)[-1]


def _data_inputs() -> tuple[Path, ...]:
    """The measured outcomes + task set every routing figure is derived from."""
    return (
        config.results_csv_path(),
        config.challenges_path(),
        _REPO_ROOT / "benchmark" / "benchmark.yaml",
    )


STANDALONE_FIGURES: Final[tuple[FigureJob, ...]] = (
    FigureJob(
        "benchmark.routing.scripts.viz_knn",
        (
            "knn_cost_comparison.png",
            "knn_pca_scatter.png",
            "model_allocation.png",
            "model_performance_descriptive.png",
            "neighborhood_purity.png",
        ),
        (_SCRIPTS / "viz_knn.py", _SCRIPTS / "knn_nulls.py", _ROUTING / "plot_style.py"),
    ),
    FigureJob(
        "benchmark.routing.scripts.plot_knn_nulls",
        ("knn_cross_repo_transfer.png", "knn_transfer_curve.png"),
        (_SCRIPTS / "plot_knn_nulls.py", _SCRIPTS / "knn_nulls.py", _SCRIPTS / "viz_knn.py"),
    ),
    FigureJob(
        "benchmark.routing.scripts.threshold_sweep",
        ("threshold_sweep_heatmap.png",),
        (_SCRIPTS / "threshold_sweep.py",),
    ),
    FigureJob(
        "benchmark.routing.scripts.plot_exploration",
        ("exploration_replay.png",),
        (_SCRIPTS / "plot_exploration.py", _ROUTING / "exploration_replay.py"),
    ),
    FigureJob(
        "benchmark.routing.scripts.embedding_compare",
        ("embedding_compare.png",),
        (_SCRIPTS / "embedding_compare.py",),
    ),
    FigureJob(
        "benchmark.routing.scripts.plot_strategies",
        ("strategy_comparison.png",),
        (_SCRIPTS / "plot_strategies.py", _STRATEGIES),
    ),
    FigureJob(
        "benchmark.routing.scripts.plot_timing",
        ("timing_comparison.png",),
        (_SCRIPTS / "plot_timing.py", _STRATEGIES),
    ),
)


def _digest(paths: tuple[Path, ...]) -> str:
    """SHA-256 over the named files, directories expanded to their sorted *.py."""
    sha = hashlib.sha256()
    for path in paths:
        members = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for member in members:
            sha.update(str(member.resolve().relative_to(_REPO_ROOT)).encode())
            sha.update(member.read_bytes() if member.exists() else b"<missing>")
    return sha.hexdigest()


def figure_digests() -> dict[str, str]:
    """Current input digest per standalone figure job."""
    data = _data_inputs()
    return {job.name: _digest((*data, *job.inputs)) for job in STANDALONE_FIGURES}


def write_figure_manifest(path: Path = FIGURE_MANIFEST) -> Path:
    """Record the digests the committed PNGs were last regenerated from."""
    path.write_text(json.dumps(figure_digests(), indent=2, sort_keys=True) + "\n")
    return path


def stale_figures(path: Path = FIGURE_MANIFEST) -> list[str]:
    """Figure jobs whose inputs changed since the committed PNGs were produced."""
    if not path.exists():
        return [job.name for job in STANDALONE_FIGURES]
    try:
        recorded = json.loads(path.read_text())
    except ValueError:
        return [job.name for job in STANDALONE_FIGURES]
    return [name for name, digest in figure_digests().items() if recorded.get(name) != digest]


def missing_figures(reports_dir: Path = _REPO_ROOT / _ROUTING_REPORTS_DIR) -> list[str]:
    """Declared outputs that are not on disk — a producer that 'succeeded' writing nothing."""
    return [
        f"{job.name}:{out}"
        for job in STANDALONE_FIGURES
        for out in job.outputs
        if not (reports_dir / out).exists()
    ]


def stage_figures(_args: argparse.Namespace, _state: PipelineState) -> None:
    """Regenerate every standalone figure, then re-record the input manifest."""
    failed: list[str] = []
    for job in STANDALONE_FIGURES:
        print(f"  figures: {job.name}", flush=True)  # noqa: T201
        result = run_module(job.module, [])
        if result.returncode != 0:
            failed.append(f"{job.module} exited {result.returncode}")
    absent = missing_figures()
    if absent:
        failed.append(f"declared figures never written: {', '.join(absent)}")
    if failed:
        raise StageError("; ".join(failed))
    print(f"  figures: manifest -> {write_figure_manifest()}")  # noqa: T201


_StageFunc = Callable[[argparse.Namespace, "PipelineState"], None]
_STAGE_FUNCS: Final[dict[str, _StageFunc]] = {
    COLLECT: stage_collect,
    STAMP: stage_stamp,
    EVALUATE: stage_evaluate,
    REPORT: stage_report,
    FIGURES: stage_figures,
}


def _selected_stages(args: argparse.Namespace) -> list[str]:
    """Stages to run given --no-report / --from, minus stamp on a simulated (non-live) run."""
    if args.no_report:
        return [COLLECT]
    start = STAGE_ORDER.index(args.start_from)
    stages = list(STAGE_ORDER[start:])
    # Stamping is a CONTAINER replay of already-collected trajectories — it costs Docker
    # time, never API budget — so it is dropped only when it was implied rather than asked
    # for. `--from stamp` asks for it explicitly; silently removing it there stranded 332
    # trajectories unstamped from 2026-07-26 (three model families fell out of the scored
    # escalation corpus entirely, and the only route to the replay looked like `--live`).
    if not args.live and args.start_from != STAMP and STAMP in stages:
        stages.remove(STAMP)  # no new live trajectories to stamp on a simulated run
    return stages


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    """Drive the selected stages with per-stage failure isolation, then the summary."""
    outcomes = dict.fromkeys(STAGE_ORDER, _SKIPPED)
    state = PipelineState()
    selected = _selected_stages(args)
    for stage in selected:
        _banner(stage)
        try:
            _STAGE_FUNCS[stage](args, state)
            outcomes[stage] = _RAN
        except StageError as exc:
            outcomes[stage] = _FAILED
            logger.error("stage %s failed: %s", stage, exc)
        except Exception as exc:  # noqa: BLE001 — isolation: one stage never aborts the rest
            outcomes[stage] = _FAILED
            logger.error("stage %s crashed: %s", stage, exc)
    # A figures-only refresh does not re-derive the kill-gate/escalation numbers, so the
    # consolidated summary would just re-run both evaluations to restate stale text.
    if not args.no_report and REPORT in selected:
        run_summary(args, state, outcomes)
    rc = 1 if any(v == _FAILED for v in outcomes.values()) else 0
    return PipelineResult(returncode=rc, outcomes=outcomes)


def _routing_kill_gate_line() -> str:
    """The paired router-vs-frontier kill-gate line, captured from routing.run_eval."""
    try:
        result = run_module(ROUTING_EVAL, [], capture=True)
    except OSError as exc:
        return f"n/a ({exc})"
    for line in (result.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("Paired contrast"):
            return stripped
    return "n/a (no paired contrast emitted)"


def _first_json_object(text: str) -> dict[str, object] | None:
    """Extract and decode the first top-level {...} object from mixed stdout (else None)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except ValueError:
                    return None
    return None


def _escalation_status(state: PipelineState) -> tuple[str, str]:
    """(status, reason) — reuse the evaluate stage stdout, else capture run_eval directly."""
    text = state.evaluate_stdout
    if not text:
        try:
            text = (
                run_module(
                    ESCALATION_EVAL, ["--plots-dir", str(_ESCALATION_PLOTS_DIR)], capture=True
                ).stdout
                or ""
            )
        except OSError:
            text = ""
    payload = _first_json_object(text)
    if payload and "status" in payload:
        return str(payload["status"]), str(payload.get("reason", ""))
    match = re.search(r"status:\s*([A-Z_]+)", text)
    return (match.group(1), "") if match else ("n/a", "")


def _capability_lines() -> tuple[str, int, str]:
    """(rank line weakest->strongest, band count, strongest==control verdict) from the JSON."""
    path = _ROUTING_REPORTS_DIR / "capability_evidence.json"
    if not path.exists():
        return "n/a", 0, "n/a"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return "n/a", 0, "n/a"
    models = [str(m["model"]) for m in data.get("models", [])]
    control = data.get("control_model")
    bands = len(data.get("bands", []))
    if not models:
        return "n/a", bands, "n/a"
    rank_line = " < ".join(models)
    strongest = models[-1]
    if not control:
        verdict = "n/a (no control)"
    elif strongest == control:
        verdict = f"OK (strongest {strongest} == control)"
    else:
        verdict = f"MISMATCH (strongest {strongest} != control {control})"
    return rank_line, bands, verdict


def _real_cost() -> float | None:
    """Total measured real_cost (USD) across results.csv — read-only, never written here."""
    path = config.results_csv_path()
    if not path.exists():
        return None
    total = 0.0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total += float(row.get("real_cost") or row.get("cost") or 0.0)
    return total


def _skill_label(status: str) -> str:
    return "SKILL" if status == "OK" else "NO_SKILL"


def run_summary(args: argparse.Namespace, state: PipelineState, outcomes: dict[str, str]) -> None:
    """Print the one consolidated block: kill-gate, escalation, capability, cost, stage ledger."""
    _banner("summary")
    config.load(args.config)
    routing_line = _routing_kill_gate_line()
    status, reason = _escalation_status(state)
    rank_line, bands, control_verdict = _capability_lines()
    cost = _real_cost()
    cost_str = "n/a" if cost is None else f"${cost:.4f}"
    esc = status if status == "n/a" else f"{status} ({_skill_label(status)})"
    if reason:
        esc += f" — {reason}"
    ledger = " ".join(f"{s}={outcomes[s]}" for s in STAGE_ORDER)
    print("Consolidated benchmark summary")  # noqa: T201
    print(f"  routing kill-gate : {routing_line}")  # noqa: T201
    print(f"  escalation        : {esc}")  # noqa: T201
    print(f"  capability rank   : {rank_line}")  # noqa: T201
    print(f"  strongest==control: {control_verdict}")  # noqa: T201
    print(f"  capability bands  : {bands}")  # noqa: T201
    print(f"  real cost         : {cost_str}")  # noqa: T201
    print(f"  stages            : {ledger}")  # noqa: T201


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Benchmark pipeline: collect -> stamp -> evaluate -> report -> figures."
    )
    ap.add_argument(
        "--strategy", choices=("cost_optimal", "full", "ladder"), default="cost_optimal"
    )
    ap.add_argument("--config", default="benchmark/benchmark.yaml")
    ap.add_argument("--live", action="store_true", help="Collect uncached cells for real")
    ap.add_argument("--max-cost", type=float, default=None)
    ap.add_argument("--max-cost-overshoot", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-start-failures", type=int, default=5)
    ap.add_argument("--check-images", action="store_true")
    ap.add_argument(
        "--no-report", action="store_true", help="Run only collect (skip stamp/evaluate/report)"
    )
    ap.add_argument(
        "--from",
        dest="start_from",
        choices=STAGE_ORDER,
        default=COLLECT,
        help="Start at a later stage (e.g. --from report re-derives artifacts, no re-collection)",
    )
    ap.add_argument("--replay-timeout", type=float, default=DEFAULT_REPLAY_TIMEOUT)
    ap.add_argument(
        "--stamp-workers",
        type=int,
        default=default_stamp_workers(),
        help=(
            "Instances replayed in parallel during the stamp stage (default: host cores minus "
            f"{_RESERVED_CORES}, capped at {_MAX_STAMP_WORKERS}). One instance is never split "
            "across workers. Container replays pin ~1 core each — oversubscribing costs ~2x."
        ),
    )
    ap.add_argument(
        "--restamp",
        action="store_true",
        help=(
            "Re-replay ALREADY-stamped trajectories too (the full-corpus rebuild). Resumes from "
            f"{STAMP_LEDGER_NAME}; without it --from stamp only replays unstamped trajectories."
        ),
    )
    ap.add_argument(
        "--check-figures",
        action="store_true",
        help="Only verify the committed standalone figures are current, then exit (no drawing)",
    )
    return ap


def check_figures() -> int:
    """Report stale/missing standalone figures; 0 when the committed set is current."""
    stale, absent = stale_figures(), missing_figures()
    for name in stale:
        print(f"STALE: {name} — its inputs changed since the committed PNGs were drawn")  # noqa: T201
    for name in absent:
        print(f"MISSING: {name}")  # noqa: T201
    if not stale and not absent:
        print(f"Figures current: {len(STANDALONE_FIGURES)} standalone jobs.")  # noqa: T201
        return 0
    print(  # noqa: T201
        "Regenerate with: make benchmark-figures "
        "(or: uv run --extra benchmark python -m benchmark.pipeline --from figures)",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: parse flags and drive the pipeline; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if args.check_figures:
        config.load(args.config)
        return check_figures()
    return run_pipeline(args).returncode


if __name__ == "__main__":
    raise SystemExit(main())

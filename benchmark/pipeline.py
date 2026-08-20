"""Unified benchmark pipeline: collect -> stamp -> evaluate -> report -> figures -> summary."""

# One entrypoint that composes the existing stage modules (run_matrix, offline_replay,
# escalation.run_eval, routing.report) so a single command regenerates the CORE artifacts:
# results.csv, the stamped live trajectories, the escalation metrics + figures, and the
# routing.report figures (the kill gate, the cost-quality frontier, the evidence basis, the
# oracle gap, cache economics, task difficulty, arm manipulation) plus
# capability_evidence.json and the coverage/summary CSVs.
# The standalone plots under benchmark/routing/scripts/ run in the FIGURES stage (see
# STANDALONE_FIGURES): they are heavy — several load the real fastembed embedder — so they
# are not part of a --live collection run, but `--from figures` refreshes all of them and
# `--check-figures` proves the committed PNGs are not stale without regenerating anything.
# FIGURE_JOBS is the wider set the freshness gate covers: the standalone jobs PLUS the report
# and escalation figures, which are drawn by their own stages and previously had no gate at
# all — 22 of the 34 committed PNGs could outlive the data they were drawn from.
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
# The PNGs live inside the published docs tree so `docs/*.md` can link them relatively —
# mkdocs only copies what is under `docs_dir`, and `mkdocs build --strict` fails a link it
# cannot resolve. One subdirectory PER HALF, so the directory itself says which half owns a
# figure instead of leaving that to figures.json alone. The reports dirs keep the NON-image
# artifacts (routing's CSV/JSON, escalation's metrics.json).
_FIGURES_DIR = Path("docs/assets/figures")
_ROUTING_FIGURES_DIR = _FIGURES_DIR / "routing"
_ESCALATION_FIGURES_DIR = _FIGURES_DIR / "escalation"
_INFERENCE_FIGURES_DIR = _FIGURES_DIR / "inference"
_ROUTING_REPORTS_DIR = Path("benchmark/routing/reports")
_ESCALATION_PLOTS_DIR = Path("benchmark/escalation/reports")

_RAN = "ran"
_FAILED = "failed"
_SKIPPED = "skipped"


class StageError(RuntimeError):
    """A stage's underlying module exited non-zero — caught so the loop finishes and halts the
    downstream stages that consume this stage's outputs."""


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


def stage_evaluate(
    _args: argparse.Namespace,
    state: PipelineState,
    *,
    manifest: Path | None = None,
) -> None:
    """Score the escalation detector (metrics + plots); keep stdout for the summary status."""
    manifest = manifest or FIGURE_MANIFEST
    argv = [
        "--plots-dir",
        str(_ESCALATION_FIGURES_DIR),
        "--metrics-dir",
        str(_ESCALATION_PLOTS_DIR),
    ]
    result = run_module(ESCALATION_EVAL, argv, capture=True)
    if result.stdout:
        print(result.stdout, end="")  # noqa: T201
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)  # noqa: T201
    state.evaluate_stdout = result.stdout or ""
    if result.returncode != 0:
        # A stage that failed certifies nothing: the escalation figures may be partial or from
        # earlier inputs, so their digest is RETRACTED — the gate reports them stale, never
        # fresh. (The eval certifies them itself on success, inside its own draw path.)
        write_figure_manifest(manifest, jobs=(), drop=tuple(job.name for job in ESCALATION_FIGURES))
        raise StageError(f"{ESCALATION_EVAL} exited {result.returncode}")


def stage_report(
    _args: argparse.Namespace, _state: PipelineState, *, manifest: Path | None = None
) -> None:
    """Regenerate the routing plots + capability_evidence.json + coverage/summary CSVs."""
    manifest = manifest or FIGURE_MANIFEST
    result = run_module(
        ROUTING_REPORT,
        ["--out-dir", str(_ROUTING_REPORTS_DIR), "--figures-dir", str(_ROUTING_FIGURES_DIR)],
    )
    if result.returncode != 0:
        write_figure_manifest(manifest, jobs=(), drop=tuple(job.name for job in REPORT_FIGURES))
        raise StageError(f"{ROUTING_REPORT} exited {result.returncode}")
    # The report just redrew the routing figures — that is what certifies them current, and it
    # certifies ONLY the report job, never the standalone/escalation jobs it did not draw.
    write_figure_manifest(manifest, jobs=REPORT_FIGURES)


# ---------------------------------------------------------------------------
# The standalone figures + their staleness gate.
#
# Every committed PNG under docs/assets/figures/<half>/ that report.py does NOT write is
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
# Every figure job's `inputs` ALSO covers the routing ANALYSIS layer the producing scripts
# import (`summary`, `config`, `impute`, `metrics`, `plot_style`, the strategies, …). It used
# to omit those — viz_knn imported them while its tuple named only its own file — so a change
# to e.g. `summary.py` left `stale_figures()==[]` certifying a figure drawn from changed
# analysis code. The closure is pinned by the test suite, so the tuple cannot silently shrink
# again. Deliberately OUT: the collection/eval machinery (`benchmark.runner.*`,
# `benchmark.escalation.*`, `corpus_lock`) — its OUTPUTS (results.csv, the escalation corpus)
# are already fingerprinted as data, so digesting the machinery would double-count the same
# bytes and mark every figure stale on edits that cannot change them.
# ---------------------------------------------------------------------------

# Repo-root-anchored so a digest is identical wherever the checkout lives (a CWD-relative
# path would hash its own string and make the gate machine-dependent).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROUTING = _REPO_ROOT / "benchmark" / "routing"
_SCRIPTS = _ROUTING / "scripts"
_STRATEGIES = _ROUTING / "strategies"
_ESCALATION = _REPO_ROOT / "benchmark" / "escalation"
FIGURE_MANIFEST = _ROUTING / "figure_inputs.json"


def _module_file(name: str) -> Path:
    """The source file for an in-repo module name (a package resolves to its __init__.py)."""
    from importlib.util import find_spec

    spec = find_spec(name)
    if spec is None or spec.origin is None or spec.origin == "namespace":
        raise ValueError(f"no importable source for {name!r}")
    path = Path(spec.origin).resolve()
    try:
        path.relative_to(_REPO_ROOT.resolve())
    except ValueError:
        raise ValueError(f"{name!r} resolves outside the repo ({path})") from None
    return path


# The analysis layer every standalone routing figure derives from. Kept as module names (one
# source per row, packages included) and resolved to files by `_module_file`; `_figure_inputs`
# appends them to each job's tuple. `benchmark.routing.plot_style` is in the layer rather than
# listed per job so no job can silently omit it again.
_ROUTING_ANALYSIS: Final[tuple[str, ...]] = (
    "benchmark.admissibility",  # a shim over `shunt.analysis.admissibility`, digested just below
    "benchmark.config",
    "benchmark.plot_frame",
    "benchmark.routing",
    "benchmark.routing.cache_cost",
    "benchmark.routing.censoring",
    "benchmark.routing.figures",
    "benchmark.routing.figures.context",
    "benchmark.routing.impute",
    "benchmark.routing.instrument_control",
    "benchmark.routing.metrics",
    "benchmark.routing.plot_style",
    "benchmark.routing.scripts",
    "benchmark.routing.scripts.knn_nulls",
    "benchmark.routing.selection_guard",
    "benchmark.routing.strategies",
    "benchmark.routing.strategies.knn",
    "benchmark.routing.strategies.oracle",
    "benchmark.routing.summary",
    # The frame, the shared style helpers and the instrument adjudicator ship in the wheel;
    # benchmark/plot_frame.py, benchmark/routing/plot_style.py and benchmark/admissibility.py are
    # re-export shims, so digesting only those would certify a figure drawn by changed code.
    "shunt.analysis.admissibility",
    "shunt.inspect.plot_frame",
    "shunt.inspect.plot_style",
)
# plot_timing additionally derives from the report/summary machinery, whose own closure reaches
# the run_eval/validate path — those modules join its digest too.
_TIMING_ANALYSIS: Final[tuple[str, ...]] = (
    *_ROUTING_ANALYSIS,
    "benchmark.routing.authenticity",
    "benchmark.routing.coverage",
    "benchmark.routing.frontier_estimate",
    "benchmark.routing.integrity",
    "benchmark.routing.report",
    "benchmark.routing.run_eval",
    "benchmark.routing.validate",
)


def _figure_inputs(*paths: Path, analysis: tuple[str, ...]) -> tuple[Path, ...]:
    """A figure job's input set: its producing files plus its analysis-layer module sources."""
    return (*paths, *(_module_file(name) for name in analysis))


@dataclass(frozen=True)
class FigureJob:
    """One figure producer: the module to run, what it writes, what it reads."""

    module: str
    outputs: tuple[str, ...]
    inputs: tuple[Path, ...]
    # Which pipeline stage regenerates this job. Only FIGURES jobs are re-run by
    # `stage_figures`; the report and escalation figures are drawn by their own stages.
    # They are still DIGESTED here, which is the point — 22 of the 34 committed figures
    # previously had no freshness gate at all.
    stage: str = FIGURES
    # Two homes, because the outputs are two kinds of thing: PNGs are published assets and
    # live under `docs/assets/figures/<half>/`, everything else is a derived data artifact and
    # stays in `reports_dir`. `output_dir` is the one place that decides which.
    figures_dir: Path = _REPO_ROOT / _ROUTING_FIGURES_DIR
    reports_dir: Path = _REPO_ROOT / _ROUTING_REPORTS_DIR
    half: str = "routing"
    # Outputs the producer legitimately skips when the data does not support them.
    optional_outputs: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.module.rsplit(".", 1)[-1]

    @property
    def required_outputs(self) -> tuple[str, ...]:
        return tuple(o for o in self.outputs if o not in self.optional_outputs)

    def output_dir(self, out: str) -> Path:
        """Where one declared output lands: images ship with the docs, data stays put."""
        return self.figures_dir if out.endswith(".png") else self.reports_dir


def _data_inputs(half: str = "routing") -> tuple[Path, ...]:
    """The measured outcomes + task set a half's figures are derived from."""
    if half == "inference":
        return (
            # The seed bundle's manifest, not the .npz: the manifest names the bundle file
            # (content-keyed) plus its results/challenges digests, so it moves whenever the
            # corpus does at the cost of one small file read.
            _ROUTING / "data" / "seed" / "manifest.json",
            # `build_live_pool()` (seed_live.py:431) INTERSECTS the registry with the router
            # policy to decide which model rows survive seeding, so renaming or dropping a
            # live model changes the corpus row count and therefore the committed figures.
            # Both are digest inputs for that reason — and the first time someone renames a
            # model, `make check-inference-figures` goes red for a reason that reads as
            # entirely unrelated to the rename. This is that reason.
            _REPO_ROOT / "src" / "shunt" / "config" / "models.yaml",
            _REPO_ROOT / "src" / "shunt" / "config" / "router.yaml",
        )
    if half == "escalation":
        return (
            # The corpus manifest, not the trajectory directory: `_digest` expands a
            # directory to its *.py, so pointing it at the corpus's .jsonl files would
            # hash nothing. manifest.json already carries a content_sha256 per trajectory, so it
            # moves whenever the corpus does and costs one file read instead of 88MB.
            _ESCALATION / "data" / "live" / "manifest.json",
            _REPO_ROOT / "benchmark" / "benchmark.yaml",
            # `_shipped_escalation()` reads the packaged router config, so a knob change
            # (escalate_after_n 2 -> 3) silently invalidates every escalation figure.
            _REPO_ROOT / "src" / "shunt" / "config" / "router.yaml",
        )
    return (
        config.results_csv_path(),
        config.challenges_path(),
        _REPO_ROOT / "benchmark" / "benchmark.yaml",
        # The routing strategies and coverage read each model's canonical DEFAULT-ARM row
        # (`flatten_default_arm`), and that default is declared in the model registry — a
        # `default_arm` change (think -> nothink) re-picks which cached arm is the strategy
        # input and moved every strategy number (and the complete-challenge census) without
        # any results.csv edit. Fingerprinted here so the committed figures can never outlive
        # the registry they were drawn from again.
        _REPO_ROOT / "src" / "shunt" / "config" / "models.yaml",
    )


_STANDALONE: Final[tuple[FigureJob, ...]] = (
    FigureJob(
        "benchmark.routing.scripts.viz_knn",
        ("knn_calibration.png",),
        _figure_inputs(
            _SCRIPTS / "viz_knn.py",
            _SCRIPTS / "knn_nulls.py",
            analysis=_ROUTING_ANALYSIS,
        ),
    ),
    FigureJob(
        "benchmark.routing.scripts.plot_knn_nulls",
        ("embedding_signal.png",),
        _figure_inputs(
            _SCRIPTS / "plot_knn_nulls.py",
            _SCRIPTS / "knn_nulls.py",
            _SCRIPTS / "viz_knn.py",
            analysis=_ROUTING_ANALYSIS,
        ),
    ),
    FigureJob(
        "benchmark.routing.scripts.threshold_sweep",
        ("sweep_regimes.png",),
        _figure_inputs(_SCRIPTS / "threshold_sweep.py", analysis=_ROUTING_ANALYSIS),
    ),
    FigureJob(
        "benchmark.routing.scripts.plot_exploration",
        ("exploration_cost.png",),
        _figure_inputs(
            _SCRIPTS / "plot_exploration.py",
            _ROUTING / "exploration_replay.py",
            analysis=_ROUTING_ANALYSIS,
        ),
    ),
)

# The report and escalation figures are drawn by `stage_report` / `stage_evaluate`, so they
# are not re-run by `stage_figures` — but they ARE digested, which is new. Until this landed,
# 22 of the 34 committed PNGs sat outside every freshness and missing-output check, and a
# report figure could outlive the data it was drawn from with nothing to say so.
_REPORT_JOB: Final[FigureJob] = FigureJob(
    "benchmark.routing.report",
    (
        "arm_manipulation.png",
        "cache_economics.png",
        "complementarity.png",
        "cost_quality_frontier.png",
        "evidence_basis.png",
        "kill_gate.png",
        "ladder_rungs.png",
        "live_gap.png",
        "oracle_gap.png",
        # Drawn here, not by viz_knn, because it audits the SHIPPED strategies' picks.
        # viz_knn could only publish its kNN proxy's picks, which is how the report set
        # ended up quoting two different (cost, pass) pairs for one strategy name.
        "routing_decision_audit.png",
        "task_difficulty.png",
    ),
    _figure_inputs(
        _ROUTING / "report.py",
        # The per-figure draw modules. A directory, so `_digest` expands it to its sorted
        # *.py — a new figure module joins the digest without anyone remembering to list it.
        _ROUTING / "figures",
        # ladder_rungs.png recomputes its rows through this module, so a change to the pairing
        # rule or the null must age the figure exactly as a change to its own draw code does.
        _SCRIPTS / "ladder_evidence.py",
        _REPO_ROOT / "benchmark" / "runner" / "kill_gate.py",
        _STRATEGIES,
        analysis=_TIMING_ANALYSIS,
    ),
    stage=REPORT,
)

_ESCALATION_ANALYSIS: Final[tuple[str, ...]] = (
    "benchmark.admissibility",  # a shim over `shunt.analysis.admissibility`, digested just below
    "benchmark.config",
    "benchmark.plot_contract",
    "benchmark.plot_frame",
    "benchmark.escalation",
    "benchmark.escalation.datasets",
    "benchmark.escalation.deployability",
    "benchmark.escalation.features",
    "benchmark.escalation.metrics",
    "benchmark.escalation.ope",  # a shim over `shunt.analysis.ope`, digested just below
    "benchmark.escalation.plots",
    "benchmark.escalation.policy_eval",
    "benchmark.escalation.prefix_eval",
    "benchmark.escalation.replay",
    "benchmark.escalation.run_eval",
    "benchmark.escalation.schema",
    "benchmark.escalation.session_eval",
    # See the note in `_ROUTING_ANALYSIS`: the shims delegate, so digest the implementations.
    "shunt.analysis.admissibility",
    "shunt.analysis.ope",
    "shunt.inspect.plot_contract",
    "shunt.inspect.plot_frame",
    "shunt.inspect.plot_style",
)

_ESCALATION_JOB: Final[FigureJob] = FigureJob(
    "benchmark.escalation.run_eval",
    (
        "corpus_and_coverage.png",
        "escalation_budget.png",
        "escalation_decision.png",
        "metrics.json",
        "operating_point.png",
        "policy_sweep.png",
        "session_value.png",
    ),
    _figure_inputs(
        _REPO_ROOT / "src" / "shunt" / "router" / "escalation.py",
        analysis=_ESCALATION_ANALYSIS,
    ),
    stage=EVALUATE,
    figures_dir=_REPO_ROOT / _ESCALATION_FIGURES_DIR,
    reports_dir=_REPO_ROOT / _ESCALATION_PLOTS_DIR,
    half="escalation",
    # Skipped when no instance has both a cheap-retry and a frontier session to compare.
    optional_outputs=("session_value.png",),
)

_INFERENCE_ANALYSIS: Final[tuple[str, ...]] = (
    # The corpus builder's own reach: `docs_corpus` seeds through `seed_live`, which pulls the
    # censoring rules and the strategy set into what the seeded rows look like.
    "benchmark.config",
    "benchmark.routing",
    "benchmark.routing.censoring",
    "benchmark.routing.docs_corpus",
    "benchmark.routing.seed_live",
    # The drawing is all SHIPPED code — the same modules the rig container renders from, where
    # no `benchmark/` exists. `shunt.inspect.inference` itself is digested as a directory (see
    # the job's `inputs`) so a new module in the package joins without anyone listing it.
    "shunt.analysis.admissibility",
    "shunt.analysis.ope",
    "shunt.db.store",
    "shunt.inspect.plot_contract",
    "shunt.inspect.plot_frame",
    "shunt.inspect.plot_style",
)

# The seven inference figures: the live router's own account, drawn from a seed-only store built
# from committed data. Its producer is benchmark-side (only the committed mode's inputs live
# here); the drawing is shipped code. stage=FIGURES, so `--from figures` redraws it and records
# its digest exactly as it does the standalone routing jobs.
_INFERENCE_FIGURES: Final[FigureJob] = FigureJob(
    "benchmark.routing.render_inference_figures",
    (
        "inference_cost.png",
        "inference_escalation.png",
        "inference_neighbourhood.png",
        "inference_ope.png",
        "inference_policy.png",
        "inference_strata.png",
        "inference_unit_economics.png",
    ),
    _figure_inputs(
        _ROUTING / "render_inference_figures.py",
        # A directory, so `_digest` expands it to its sorted *.py: the whole shipped drawing
        # package (data, estimators, figures, specs, the container entrypoint).
        _REPO_ROOT / "src" / "shunt" / "inspect" / "inference",
        _STRATEGIES,
        analysis=_INFERENCE_ANALYSIS,
    ),
    figures_dir=_REPO_ROOT / _INFERENCE_FIGURES_DIR,
    # Every declared output is a PNG, so `reports_dir` is never consulted; it points at the
    # figures dir rather than inheriting routing's, which would be a lie about where this
    # job's artifacts live.
    reports_dir=_REPO_ROOT / _INFERENCE_FIGURES_DIR,
    half="inference",
)

FIGURE_JOBS: Final[tuple[FigureJob, ...]] = (
    *_STANDALONE,
    _REPORT_JOB,
    _ESCALATION_JOB,
    _INFERENCE_FIGURES,
)
# Every half a figure job declares — the `--half` filter's choices, so registering a new half
# extends the flag without an argparse edit.
FIGURE_HALVES: Final[tuple[str, ...]] = tuple(sorted({job.half for job in FIGURE_JOBS}))
# Kept as the name `stage_figures` and benchmark/tests/test_pipeline.py already use: the
# subset this pipeline stage actually regenerates.
STANDALONE_FIGURES: Final[tuple[FigureJob, ...]] = tuple(
    job for job in FIGURE_JOBS if job.stage == FIGURES
)
# The report/escalation figures are drawn by their own stages (REPORT/EVALUATE); a stage
# certifies only the jobs IT regenerated, so each stage names its own job set to the manifest.
REPORT_FIGURES: Final[tuple[FigureJob, ...]] = tuple(
    job for job in FIGURE_JOBS if job.stage == REPORT
)
ESCALATION_FIGURES: Final[tuple[FigureJob, ...]] = tuple(
    job for job in FIGURE_JOBS if job.stage == EVALUATE
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


def _file_digest(path: Path) -> str:
    """SHA-256 of one file's bytes — the committed-output half of the freshness record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _ManifestEntry:
    """One job's freshness record: its input digest and the output bytes certified with it."""

    inputs: str
    # None = the entry predates output digesting: the job's committed bytes were never
    # certified, which is reported as UNCERTIFIED rather than silently accepted as fresh.
    outputs: dict[str, str] | None


def figure_digests(jobs: tuple[FigureJob, ...] = FIGURE_JOBS) -> dict[str, str]:
    """Current input digest per figure job — every committed figure, not only standalone ones."""
    return {job.name: _digest((*_data_inputs(job.half), *job.inputs)) for job in jobs}


def figure_output_digests(jobs: tuple[FigureJob, ...] = FIGURE_JOBS) -> dict[str, dict[str, str]]:
    """Current SHA-256 of each declared output that is on disk, per figure job."""
    # The INPUT digest answers "were the figures drawn from today's data and code"; it cannot
    # answer "are the committed bytes still what that code draws". Recording the OUTPUT bytes at
    # certification time is what closes that: two figures drifted this session (a PNG whose bytes
    # moved against an unmodified producer, and one that does not reproduce from committed data)
    # and the input-only gate reported both green.
    return {
        job.name: {
            out: _file_digest(job.output_dir(out) / out)
            for out in job.outputs
            if (job.output_dir(out) / out).exists()
        }
        for job in jobs
    }


def _manifest_entries(path: Path) -> dict[str, _ManifestEntry]:
    """Parse the manifest; a legacy bare-string value means inputs recorded, outputs never were."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    entries: dict[str, _ManifestEntry] = {}
    for name, value in payload.items():
        if isinstance(value, dict):
            outputs = value.get("outputs")
            entries[str(name)] = _ManifestEntry(
                inputs=str(value.get("inputs", "")),
                outputs=(
                    {str(k): str(v) for k, v in outputs.items()}
                    if isinstance(outputs, dict)
                    else None
                ),
            )
        else:
            entries[str(name)] = _ManifestEntry(inputs=str(value), outputs=None)
    return entries


def write_figure_manifest(
    path: Path = FIGURE_MANIFEST,
    *,
    jobs: tuple[FigureJob, ...] | None = None,
    drop: tuple[str, ...] = (),
) -> Path:
    """Upsert current digests for `jobs` (None = all) as freshly rendered; delete `drop`."""
    # The manifest is the freshness gate's memory — an entry means "the committed figures were
    # drawn from inputs with this digest". A render that FAILED must keep no entry (the gate
    # reports it stale), never the digest from an earlier run; and a stage may only certify the
    # jobs it actually regenerated. Entries named by neither set are preserved: this write only
    # records what the calling stage produced. The old whole-file overwrite is what let a
    # crashed stage's digest survive as fresh.
    entries = _manifest_entries(path)
    for name in drop:
        entries.pop(name, None)
    certified = jobs if jobs is not None else FIGURE_JOBS
    inputs, outputs = figure_digests(certified), figure_output_digests(certified)
    for name, digest in inputs.items():
        entries[name] = _ManifestEntry(inputs=digest, outputs=outputs[name])
    payload = {
        name: {"inputs": entry.inputs, "outputs": entry.outputs}
        if entry.outputs is not None
        else entry.inputs
        for name, entry in entries.items()
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def stale_figures(
    path: Path = FIGURE_MANIFEST, jobs: tuple[FigureJob, ...] = FIGURE_JOBS
) -> list[str]:
    """Figure jobs whose inputs changed since the committed PNGs were produced."""
    entries = _manifest_entries(path)
    return [
        name
        for name, digest in figure_digests(jobs).items()
        if name not in entries or entries[name].inputs != digest
    ]


def drifted_figures(
    path: Path = FIGURE_MANIFEST, jobs: tuple[FigureJob, ...] = FIGURE_JOBS
) -> list[str]:
    """Committed outputs whose bytes no longer match what was recorded at certification."""
    entries, current = _manifest_entries(path), figure_output_digests(jobs)
    drifted: list[str] = []
    for job in jobs:
        recorded = entries.get(job.name)
        if recorded is None or recorded.outputs is None:
            continue
        for out, digest in recorded.outputs.items():
            if current[job.name].get(out) != digest:
                drifted.append(f"{job.name}:{out}")
    return drifted


def uncertified_figures(
    path: Path = FIGURE_MANIFEST, jobs: tuple[FigureJob, ...] = FIGURE_JOBS
) -> list[str]:
    """Outputs of jobs recorded with an input digest only — their bytes were never certified."""
    # Named per OUTPUT, not per job: the two figures known to have drifted this session
    # (sweep_regimes.png, inference_neighbourhood.png) have to be readable in the report, and a
    # job name alone hides which of its 8 PNGs the reader should go look at.
    entries = _manifest_entries(path)
    return [
        f"{job.name}:{out}"
        for job in jobs
        if job.name in entries and entries[job.name].outputs is None
        for out in job.outputs
    ]


def missing_figures(jobs: tuple[FigureJob, ...] = FIGURE_JOBS) -> list[str]:
    """Declared outputs that are not on disk — a producer that 'succeeded' writing nothing."""
    return [
        f"{job.name}:{out}"
        for job in jobs
        for out in job.required_outputs
        if not (job.output_dir(out) / out).exists()
    ]


def stage_figures(
    _args: argparse.Namespace, _state: PipelineState, *, manifest: Path = FIGURE_MANIFEST
) -> None:
    """Regenerate every standalone figure, then re-record the input manifest."""
    # The manifest records ONE entry per job that actually rendered this run, and DROPS the
    # entry of any job whose render failed — a crashed render reports STALE (its old digest is
    # never retained as fresh). A job that exits non-zero, or that "succeeds" writing none of
    # its declared outputs, is a failure and is dropped from the manifest.
    failed: list[str] = []
    regenerated: list[FigureJob] = []
    for job in STANDALONE_FIGURES:
        print(f"  figures: {job.name}", flush=True)  # noqa: T201
        try:
            result = run_module(job.module, [])
        except Exception as exc:  # noqa: BLE001 — one crashed producer fails only its job
            failed.append(f"{job.module} crashed ({exc})")
            continue
        if result.returncode != 0:
            failed.append(f"{job.module} exited {result.returncode}")
            continue
        regenerated.append(job)
    absent = missing_figures(STANDALONE_FIGURES)
    absent_jobs = {entry.split(":", 1)[0] for entry in absent}
    if absent:
        failed.append(f"declared figures never written: {', '.join(absent)}")
    refreshed = tuple(job for job in regenerated if job.name not in absent_jobs)
    dropped = tuple(
        job.name for job in STANDALONE_FIGURES if job.name in absent_jobs or job not in regenerated
    )
    write_figure_manifest(manifest, jobs=refreshed, drop=dropped)
    if failed:
        raise StageError("; ".join(failed))
    print(f"  figures: manifest -> {manifest}")  # noqa: T201


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
    # STAGE_ORDER is the dependency chain (collect -> stamp -> evaluate -> report -> figures):
    # a failed stage halts every selected stage after it that consumes its outputs, so a
    # downstream stage never runs on a corpus its producer failed to complete. They are
    # recorded _SKIPPED, not run; the loop still finishes and the pipeline returns the failure
    # code.
    outcomes = dict.fromkeys(STAGE_ORDER, _SKIPPED)
    state = PipelineState()
    selected = _selected_stages(args)
    blocked = False
    for stage in selected:
        if blocked:
            outcomes[stage] = _SKIPPED
            logger.warning("stage %s skipped: an upstream stage it depends on failed", stage)
            continue
        _banner(stage)
        try:
            _STAGE_FUNCS[stage](args, state)
            outcomes[stage] = _RAN
        except StageError as exc:
            outcomes[stage] = _FAILED
            blocked = True
            logger.error("stage %s failed: %s", stage, exc)
        except Exception as exc:  # noqa: BLE001 — isolation: one stage never aborts the rest
            outcomes[stage] = _FAILED
            blocked = True
            logger.error("stage %s crashed: %s", stage, exc)
    # The consolidated summary re-runs evaluation modules when their stage's stdout was never
    # captured, so it only follows a REPORT that actually produced its artifacts — never one that
    # was skipped because the corpus upstream of it failed.
    if outcomes.get(REPORT) == _RAN:
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
                    ESCALATION_EVAL,
                    [
                        "--plots-dir",
                        str(_ESCALATION_FIGURES_DIR),
                        "--metrics-dir",
                        str(_ESCALATION_PLOTS_DIR),
                    ],
                    capture=True,
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
    # `OK_OFFLINE_ONLY` is still skill — the signal exists, but it was measured at a cadence
    # production does not run. Mapping it to NO_SKILL would mis-report the presence of the signal.
    return "SKILL" if status in ("OK", "OK_OFFLINE_ONLY") else "NO_SKILL"


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
    ap.add_argument(
        "--half",
        choices=FIGURE_HALVES,
        default=None,
        help=(
            "Restrict --check-figures to one figure half (default: all). Scoped to the CHECK "
            "deliberately: the stages regenerate whatever their own job set names, and no "
            "caller needs a half-scoped redraw — `make check-inference-figures` needs a "
            "half-scoped report of a half whose render lives outside this pipeline's stages."
        ),
    )
    return ap


def figure_jobs_for(half: str | None) -> tuple[FigureJob, ...]:
    """Every figure job, or just one half's — the `--half` filter's one definition."""
    return FIGURE_JOBS if half is None else tuple(j for j in FIGURE_JOBS if j.half == half)


def check_figures(half: str | None = None) -> int:
    """Report stale/missing figures for one half (None = all); 0 when the set is current."""
    jobs = figure_jobs_for(half)
    stale, absent = stale_figures(jobs=jobs), missing_figures(jobs)
    drifted, uncertified = drifted_figures(jobs=jobs), uncertified_figures(jobs=jobs)
    for name in stale:
        print(f"STALE: {name} — its inputs changed since the committed PNGs were drawn")  # noqa: T201
    for name in absent:
        print(f"MISSING: {name}")  # noqa: T201
    for name in drifted:
        print(  # noqa: T201
            f"DRIFTED: {name} — the committed bytes are not the bytes certified for this job"
        )
    for name in uncertified:
        print(f"UNCERTIFIED: {name} — no certified bytes on record for this output")  # noqa: T201
    if not (stale or absent or drifted or uncertified):
        pngs = sum(len(job.outputs) for job in jobs)
        scope = "" if half is None else f" ({half})"
        print(f"Figures current{scope}: {len(jobs)} jobs, {pngs} outputs.")  # noqa: T201
        return 0
    # `--from evaluate` and not `--from report`: the report stage spawns run_eval only as a
    # STATUS PROBE, which never calls write_figure_manifest, so it cannot certify that job.
    print(  # noqa: T201
        "Repair with: uv run --extra benchmark python -m benchmark.pipeline --from evaluate "
        "(redraws the evaluate/report/figures jobs and re-records both digests). "
        "Never hand-edit benchmark/routing/figure_inputs.json.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: parse flags and drive the pipeline; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if args.check_figures:
        config.load(args.config)
        return check_figures(args.half)
    if args.half is not None:
        # Loud rather than silently ignored: --half only filters the check.
        _build_parser().error("--half applies to --check-figures only")
    return run_pipeline(args).returncode


if __name__ == "__main__":
    raise SystemExit(main())

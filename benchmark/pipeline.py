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
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.escalation import schema
from benchmark.escalation.live_capture import LIVE_DIR

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

DEFAULT_REPLAY_TIMEOUT = 120.0
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


def _unstamped_trajectories(live_dir: Path = LIVE_DIR) -> list[tuple[str, str, Path]]:
    """Live trajectories with no per-step failing_check_id yet — the ones stamping still owes."""
    pending: list[tuple[str, str, Path]] = []
    if not live_dir.exists():
        return pending
    for path in sorted(live_dir.glob("*.jsonl")):
        try:
            traj = schema.load_jsonl(path)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("stamp: skipping unreadable %s (%s)", path, exc)
            continue
        if any(step.failing_check_id for step in traj.steps):
            continue
        instance_id = traj.header.instance_id
        if not instance_id:
            logger.warning("stamp: skipping %s (no instance_id in header)", path)
            continue
        pending.append((traj.header.trajectory_id, instance_id, path))
    return pending


def stage_stamp(args: argparse.Namespace, _state: PipelineState) -> None:
    """Restamp only the still-unstamped live trajectories, timeout-bounded per trajectory."""
    pending = _unstamped_trajectories()
    if not pending:
        print("  stamp: no unstamped trajectories to replay")  # noqa: T201
        return
    for trajectory_id, instance_id, path in pending:
        argv = [trajectory_id, "--instance-id", instance_id, "--jsonl", str(path)]
        try:
            result = run_module(OFFLINE_REPLAY, argv, timeout=args.replay_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "stamp: %s exceeded %ss — skipping straggler", trajectory_id, args.replay_timeout
            )
            continue
        except OSError as exc:
            logger.warning("stamp: %s failed to launch (%s) — skipping", trajectory_id, exc)
            continue
        if result.returncode != 0:
            logger.warning("stamp: %s exited %s — skipping", trajectory_id, result.returncode)


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
    if not args.live and STAMP in stages:
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

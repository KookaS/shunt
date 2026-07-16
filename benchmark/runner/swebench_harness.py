"""Thin wrapper over the official ``swebench`` evaluation harness.

Shells out to ``run_evaluation`` on a predictions.jsonl and parses its report
into per-instance ``resolved`` bools — reusing the harness, pulling prebuilt images.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark.runner import swebench_specs

DEFAULT_DATASET = swebench_specs.DATASET_NAME
DEFAULT_SPLIT = swebench_specs.DATASET_SPLIT
DEFAULT_NAMESPACE = swebench_specs.DEFAULT_NAMESPACE


@dataclass(frozen=True)
class HarnessResult:
    """Outcome of one harness run: per-instance resolved + the parsed report."""

    resolved: dict[str, bool]
    report_path: Path | None
    report: dict[str, object]
    returncode: int


def read_predictions(predictions_path: Path) -> list[dict[str, object]]:
    """Read a predictions.jsonl (one prediction object per line)."""
    lines = predictions_path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _model_names(predictions: list[dict[str, object]]) -> list[str]:
    seen: list[str] = []
    for pred in predictions:
        name = str(pred["model_name_or_path"])
        if name not in seen:
            seen.append(name)
    return seen


def build_command(
    predictions_path: Path,
    run_id: str,
    instance_ids: list[str],
    namespace: str = DEFAULT_NAMESPACE,
    max_workers: int = 4,
    cache_level: str = "env",
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    timeout: int = 1800,
) -> list[str]:
    """Assemble the ``run_evaluation`` argv — pulls prebuilt images, ephemeral runs."""
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id,
        "--namespace",
        namespace,
        "--max_workers",
        str(max_workers),
        "--cache_level",
        cache_level,
        "--timeout",
        str(timeout),
    ]
    if instance_ids:
        cmd.append("--instance_ids")
        cmd.extend(instance_ids)
    return cmd


def parse_report(report: dict[str, object], submitted_ids: list[str]) -> dict[str, bool]:
    """Map each submitted instance to resolved=True iff it is in ``resolved_ids``."""
    raw = report.get("resolved_ids") or []
    resolved_ids = set(raw) if isinstance(raw, list) else set()
    return {iid: iid in resolved_ids for iid in submitted_ids}


def _find_report(work_dir: Path, model_name: str, run_id: str) -> Path | None:
    expected = work_dir / f"{model_name.replace('/', '__')}.{run_id}.json"
    if expected.exists():
        return expected
    matches = sorted(work_dir.glob(f"*.{run_id}.json"))
    return matches[0] if matches else None


def run_harness(
    predictions_path: Path,
    run_id: str,
    work_dir: Path,
    namespace: str = DEFAULT_NAMESPACE,
    max_workers: int = 4,
    cache_level: str = "env",
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    timeout: int = 1800,
    verbose: bool = False,
) -> HarnessResult:
    """Run the harness in ``work_dir`` and return the parsed per-instance result."""
    predictions_path = predictions_path.resolve()
    predictions = read_predictions(predictions_path)
    submitted = [str(p["instance_id"]) for p in predictions]
    models = _model_names(predictions)
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        predictions_path=predictions_path,
        run_id=run_id,
        instance_ids=submitted,
        namespace=namespace,
        max_workers=max_workers,
        cache_level=cache_level,
        dataset_name=dataset_name,
        split=split,
        timeout=timeout,
    )
    if verbose:
        print(f"harness: {' '.join(cmd)} (cwd={work_dir})", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=work_dir, check=False)
    report_path = _find_report(work_dir, models[0], run_id) if models else None
    if report_path is None:
        return HarnessResult({iid: False for iid in submitted}, None, {}, proc.returncode)
    report = json.loads(report_path.read_text())
    return HarnessResult(parse_report(report, submitted), report_path, report, proc.returncode)


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run the swebench harness on a predictions.jsonl.")
    ap.add_argument("predictions", type=Path, help="Path to predictions.jsonl")
    ap.add_argument("--run-id", required=True, help="Harness run id")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path("benchmark/runner/artifacts"),
        help="Working dir for harness logs + report (gitignored)",
    )
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Prebuilt-image namespace")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--cache-level", default="env", choices=["none", "base", "env", "instance"])
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    result = run_harness(
        predictions_path=args.predictions,
        run_id=args.run_id,
        work_dir=args.work_dir,
        namespace=args.namespace,
        max_workers=args.max_workers,
        cache_level=args.cache_level,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    n_resolved = sum(1 for v in result.resolved.values() if v)
    print(f"resolved {n_resolved}/{len(result.resolved)}")
    for iid, ok in sorted(result.resolved.items()):
        print(f"  {'RESOLVED' if ok else 'UNRESOLVED'}  {iid}")
    return 0 if n_resolved == len(result.resolved) else 1


if __name__ == "__main__":
    raise SystemExit(_main())

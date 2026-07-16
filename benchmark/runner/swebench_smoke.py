"""$0 gold-patch smoke: prove the SWE-bench harness loop end-to-end.

Scores gold-patch predictions through the real harness and writes a separate
gitignored report (``artifacts/smoke_gold.json``) — gold is NOT routing data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.runner import infer, swebench_harness, swebench_specs


def artifacts_dir() -> Path:
    """Gitignored working dir for harness logs, predictions, and the smoke report."""
    return Path(__file__).resolve().parent / "artifacts"


def run_smoke(
    instance_ids: list[str],
    work_dir: Path,
    run_id: str,
    namespace: str = swebench_harness.DEFAULT_NAMESPACE,
    max_workers: int = 4,
    cache_level: str = "env",
    timeout: int = 1800,
    verbose: bool = False,
) -> dict[str, object]:
    """Run the gold smoke and return the report dict (also written to disk)."""
    predictions = infer.build_gold_predictions(instance_ids)
    preds_path = infer.write_predictions(predictions, work_dir / f"predictions_{run_id}.jsonl")
    result = swebench_harness.run_harness(
        predictions_path=preds_path,
        run_id=run_id,
        work_dir=work_dir,
        namespace=namespace,
        max_workers=max_workers,
        cache_level=cache_level,
        timeout=timeout,
        verbose=verbose,
    )
    n_resolved = sum(1 for ok in result.resolved.values() if ok)
    report = {
        "mode": "gold",
        "run_id": run_id,
        "namespace": namespace,
        "dataset": swebench_specs.DATASET_NAME,
        "n_total": len(instance_ids),
        "n_resolved": n_resolved,
        "all_resolved": n_resolved == len(instance_ids),
        "resolved": dict(sorted(result.resolved.items())),
        "harness_returncode": result.returncode,
        "harness_report": str(result.report_path) if result.report_path else None,
    }
    smoke_path = work_dir / "smoke_gold.json"
    smoke_path.write_text(json.dumps(report, indent=2) + "\n")
    report["smoke_report"] = str(smoke_path)
    return report


def _default_instance_ids() -> list[str]:
    return [spec.instance_id for spec in swebench_specs.all_specs()]


def main() -> int:
    ap = argparse.ArgumentParser(description="$0 gold-patch SWE-bench Verified smoke.")
    ap.add_argument("--instance-ids", nargs="+", default=None, help="Defaults to all specs")
    ap.add_argument("--run-id", default="gold-smoke", help="Harness run id")
    ap.add_argument("--namespace", default=swebench_harness.DEFAULT_NAMESPACE)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--cache-level", default="env", choices=["none", "base", "env", "instance"])
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    instance_ids = args.instance_ids or _default_instance_ids()
    if not instance_ids:
        print("No materialised specs — run benchmark.runner.swebench_specs <ids> first.")
        return 2

    report = run_smoke(
        instance_ids=instance_ids,
        work_dir=artifacts_dir(),
        run_id=args.run_id,
        namespace=args.namespace,
        max_workers=args.max_workers,
        cache_level=args.cache_level,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    print(f"\nGOLD SMOKE: resolved {report['n_resolved']}/{report['n_total']}")
    resolved = report["resolved"]
    if isinstance(resolved, dict):
        for iid, ok in resolved.items():
            print(f"  {'RESOLVED' if ok else 'UNRESOLVED'}  {iid}")
    print(f"Report: {report['smoke_report']}")
    return 0 if report["all_resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

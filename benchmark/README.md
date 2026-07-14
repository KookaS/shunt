# Routing Benchmark

Offline evaluation of routing strategies — which model should handle a given task, given the task's prompt and known model performance on similar tasks.

## Layout

```
benchmark/
  routing/                     # Active: routing strategy benchmark
    matrices/                  # Task×model performance matrices
      pricing.json             # Model pricing table (cost per token)
      coderouterbench100.json  # 100 OOD176 tasks (or minimal default)
    strategies/                # Routing strategies (one file per strategy)
      __init__.py
      oracle.py                # Perfect-information upper bound
      fixed.py                 # Fixed-model baselines
    run_eval.py                # Evaluate all strategies against a matrix
    metrics.py                 # Metric definitions (cost, quality, trade-offs)
    report.py                  # PLANNED: comparison tables and plots
    results.csv                # COMMITTED per-strategy results (header only for now)
  .gitignore
  .runs/                       # Gitignored run data (external runner output)
  README.md                    # This file
```

Prior model-capability benchmarks (`v1/`) and the 66 hidden-test tasks were removed — they belonged to an earlier project phase and the code no longer exists.

## Run

```sh
# Default (minimal embedded matrix)
python3 routing/run_eval.py

# Specific performance matrix
python3 routing/run_eval.py --matrix routing/matrices/coderouterbench100.json
```

Results append to `routing/results.csv`. Re-runs skip already‑computed (strategy, task) pairs.

## Pre‑alpha

The directory structure is in place; actual strategy implementations and metric calculations are being built. `results.csv` has a header row only. Expect breakage and missing features.

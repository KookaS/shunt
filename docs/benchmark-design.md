---
title: Benchmark design
description: Two-part benchmark structure — model run logging and offline routing strategy evaluation.
---

# Benchmark design

The benchmark has two parts. `.runs/` collects raw model results from eval sessions. `routing/` evaluates routing strategies offline against those results.

| Tree | Question | Output |
|---|---|---|
| `.runs/` | Which models solve which tasks? | Per-model pass/fail on tasks from real sessions |
| `routing/` | Which routing strategy maximizes reward? | Per-strategy metrics across a task × model matrix |

`.runs/` is the empirical source. `routing` consumes its output matrix — no dependency on runner infrastructure.

## Why split them

`.runs/` answers a model-selection question: given N models, which is the cheapest that solves each task? This is the discrimination test. If every model passes everything, routing is pointless.

`routing/` answers a strategy-selection question: given a known task × model matrix, which algorithm (kNN, cascade, bandit, fixed) maximizes pass rate minus cost?

They share a `benchmark/` root because both evaluate model-decision capability. They stay separate because they have different runners, metrics, and output formats.

## Structure

```
benchmark/
  README.md                               Model-capability benchmark overview

  .runs/                                  Per-model run data from evals
    <model>__<capability>__<task-id>/     One directory per run

  routing/                                Routing strategy evaluation
    results.csv                           THE committed source of truth (per-cell outcomes)
    data/                                 Curated read-only inputs
      challenges.json                     Challenge index + task metadata
    reports/                              Gitignored — derived strategy_summary.csv + plots
    strategies/
      __init__.py                         Strategy protocol
      oracle.py                           Best per-task (upper bound)
      fixed.py                            Always-cheap, always-frontier, random
    run_eval.py                           Evaluate all strategies × tasks
    metrics.py                            Reward, regret, efficiency
```

## Strategy interface

```python
class Strategy(ABC):
    @property
    def name(self) -> str: ...

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        """Return the model name to route this task to."""
```

The evaluator iterates tasks, calls `select()` per strategy, looks up the outcome, and accumulates metrics.

## Metrics

| Metric | Formula | Meaning |
|---|---|---|
| AvgPerf% | `pass_count / total_tasks × 100` | % of tasks solved |
| TotalCost | `sum(cost of chosen model per task)` | Raw dollar cost |
| Reward | `1.0 × pass_rate − γ × total_cost` | Cost-aware utility |
| CumReg | `sum(oracle_reward − strategy_reward)` | Regret vs oracle |
| rAcc | fraction of tasks routed to the oracle's model | Routing accuracy |

γ defaults to 0.1, matching the `agent-as-a-router` cost-weight baseline.

Cost is recorded from actual model API responses: litellm's computed cost for direct routes (deepseek), and provider-returned `usage.cost` for Requesty-routed models (including cache-aware rates). For offline eval, costs come from the cached `results.csv` (recorded during live benchmark matrix runs). Recording per-request API cost on the live proxy path is roadmap, not a current feature.

## Baselines

| Strategy | Behavior |
|---|---|
| **Oracle** | Cheapest model that passes each task. Upper bound. |
| **Always-Cheap** | Always cheapest model (derived from pricing matrix). Lower bound — if a router can't beat this, it is pointless. |
| **Always-Frontier** | Always most expensive model (derived from pricing matrix). Maximum cost baseline. |
| **Random** | Random model per task (mean over N seeds). Null baseline. |

Additional strategies: kNN and kNN-cascade, both implemented in `strategies/`.

## Relationship to src/shunt/

The strategies in `benchmark/routing/strategies/` are evaluation copies — they consume a known matrix and compute metrics offline. They are separate from `src/shunt/router/`, the decision module that is built and unit-tested but not yet wired into the live proxy (which currently forwards to a cheap default). The offline kNN strategy is designed to mirror that module's algorithm, so that if the module clears the kill gate and is wired in, live behavior matches what the benchmark scored.

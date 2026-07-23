---
title: Benchmark design
description: Two-part benchmark structure — live model execution and offline routing strategy evaluation.
---

# Benchmark design

The benchmark has two parts. `runner/` executes models against SWE-bench Verified instances and records the outcomes. `routing/` evaluates routing strategies offline against those recorded outcomes.

| Tree | Question | Output |
|---|---|---|
| `runner/` | Which models solve which tasks? | Verified per-cell pass/fail, cost, and tokens, written to `routing/results.csv` |
| `routing/` | Which routing strategy maximizes reward? | Per-strategy metrics across a task × model matrix |

`runner/` is the empirical source; `routing/results.csv` is the committed record it produces. `routing/` consumes that file — no dependency on runner infrastructure at eval time.

## Why split them

`runner/` answers a model-selection question: given N models, which is the cheapest that solves each task? This is the discrimination test. If every model passes everything, routing is pointless.

`routing/` answers a strategy-selection question: given a known task × model matrix, which algorithm (kNN, cascade, bandit, fixed) maximizes pass rate minus cost?

They share a `benchmark/` root because both evaluate model-decision capability. They stay separate because they have different runners, metrics, and output formats.

## Structure

```
benchmark/
  README.md                               Model-capability benchmark overview
  benchmark.yaml                          Enabled models, strategies, and run settings

  challenges/swebench_verified/           Instance specs (the sole challenge source)

  runner/                                 Live execution against the SWE-bench harness
    run_matrix.py                         Runs the (challenge x model x arm) matrix, upserts rows
    collect.py                            Adaptive collection (phase A + frontier tail)
    check_integrity.py                    Anchor/authenticity audit of the committed rows

  routing/                                Routing strategy evaluation
    results.csv                           THE committed source of truth (per-cell outcomes)
    data/                                 Curated read-only inputs
      challenges.json                     Challenge index + task metadata
    reports/                              Gitignored — derived strategy_summary.csv + plots
    strategies/
      __init__.py                         Strategy protocol
      oracle.py                           Best per-task (upper bound)
      fixed.py                            Always-cheap, always-frontier, random
      knn.py                              Embed task → retrieve neighbours → cheapest capable
      knn_cascade.py                      kNN-informed try-verify-escalate
      knn_blended.py                      kNN over our runs + down-weighted external neighbours
      external_prior.py                   SWE-bench leaderboard difficulty prior
      _template.py                        Skeleton for a new strategy
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

Cost is recorded from actual model API responses: the provider-returned cache-aware `usage.cost` when present (e.g. Requesty-routed models, including cache-aware rates), falling back to litellm's computed cost otherwise (e.g. direct routes litellm can price, such as deepseek). For offline eval, costs come from the cached `results.csv` (recorded during live benchmark matrix runs). Recording per-request API cost on the live proxy path is roadmap, not a current feature.

## Baselines

| Strategy | Behavior |
|---|---|
| **Oracle** | Cheapest model that passes each task. Upper bound. |
| **Always-Cheap** | Always cheapest model (derived from pricing matrix). Lower bound — if a router can't beat this, it is pointless. |
| **Always-Frontier** | Always most expensive model (derived from pricing matrix). Maximum cost baseline. |
| **Random** | Random model per task (mean over N seeds). Null baseline. |

Additional strategies in `strategies/`: kNN, kNN-cascade, kNN-blended, and External-Prior.

## Relationship to src/shunt/

The strategies in `benchmark/routing/strategies/` are evaluation copies — they consume a known matrix and compute metrics offline. They are separate from `src/shunt/router/`, the decision module that is now called on the first turn by the live proxy and learns from verified outcomes recorded at session close. The offline kNN strategy is designed to mirror that module's algorithm, so that live behavior matches what the benchmark scored.

# Routing Benchmark

Evaluates routing strategies against a task×model performance matrix to answer: *which routing algorithm maximizes reward (performance − λ·cost)?*

## Layout

```
routing/
  matrices/
    pricing.json              # Model cost table (USD per 1M tokens)
    coderouterbench100.json   # 100 OOD176 tasks from CodeRouterBench
  strategies/
    __init__.py               # Strategy protocol
    oracle.py                 # Upper bound: perfect per-task selection
    fixed.py                  # Always-cheap, always-frontier, random
    knn.py                    # PLANNED: kNN retrieval (shunt's approach)
    cascade.py                # PLANNED: ACRouter-style verify-and-escalate
    bandit.py                 # PLANNED: LinUCB / LinTS
  run_eval.py                 # Evaluate all strategies
  metrics.py                  # Metric definitions
  report.py                   # PLANNED: Comparison tables
  results.csv                 # COMMITTED per-strategy metrics
```

## Usage

```bash
# Evaluate all strategies
python3 run_eval.py

# Or against a specific matrix
python3 run_eval.py --matrix matrices/coderouterbench100.json
```

## Metric definitions

| Metric | Meaning |
|--------|---------|
| AvgPerf% | Tasks solved correctly |
| AvgPerf_ci_lower / AvgPerf_ci_upper | 95% bootstrap CI on AvgPerf% |
| TotalCost | Total backend model cost (USD) |
| Reward | `Σ(1.0 × passed − γ × cost)` per task (γ=0.1 default) |
| CumReg | `total(oracle_reward) − total(strategy_reward)` |
| CumReg_ci_lower / CumReg_ci_upper | 95% bootstrap CI on CumReg |
| rAcc | Fraction of tasks where strategy picked same model as oracle |
| Pareto | True if strategy is on the Pareto frontier (no other strategy has higher AvgPerf% AND lower TotalCost) |

## Baselines

| Strategy | Description |
|----------|-------------|
| Oracle | Upper bound: cheapest pass-per-task |
| Always-Cheap | Route all to cheapest model (derived from pricing matrix) |
| Always-Frontier | Route all to most expensive model (derived from pricing matrix) |
| Random | Uniform random (mean over seeds) |
| kNN | Embed task → retrieve similar → cheapest capable |
| Cascade | Try cheap models, escalate on failure |

## Source

Design informed by [agent-as-a-router](https://github.com/LanceZPF/agent-as-a-router) (CodeRouterBench).
Reward formulation matches their `REWARD_COST_WEIGHT = 0.1`.

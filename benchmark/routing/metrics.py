from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean

GAMMA = 0.1


def _reward(passed: bool, cost: float, gamma: float | None = None) -> float:
    g = GAMMA if gamma is None else gamma
    return 1.0 - g * cost if passed else 0.0 - g * cost


def compute_metrics(
    decisions: list[tuple[str, str, bool, float]], gamma: float | None = None
) -> dict:
    if not decisions:
        return {}

    n = len(decisions)
    passes = sum(1 for _, _, p, _ in decisions if p)
    total_cost = sum(c for _, _, _, c in decisions)
    pass_rate = passes / n
    rewards = [_reward(p, c, gamma) for _, _, p, c in decisions]

    return {
        "n_tasks": n,
        "n_pass": passes,
        "AvgPerf%": round(pass_rate * 100, 2),
        "TotalCost": round(total_cost, 4),
        "AvgCost": round(total_cost / n, 6),
        "Reward": round(sum(rewards), 4),
        "AvgReward": round(mean(rewards), 6) if rewards else 0.0,
    }


def compare_to_oracle(
    decisions: list[tuple[str, str, bool, float]],
    oracle_decisions: list[tuple[str, str, bool, float]],
    gamma: float | None = None,
) -> dict:
    if not decisions or not oracle_decisions:
        return {"CumReg": 0.0, "rAcc": 0.0}

    strategy_rewards = [_reward(p, c, gamma) for _, _, p, c in decisions]
    oracle_rewards = [_reward(p, c, gamma) for _, _, p, c in oracle_decisions]
    cumreg = sum(oracle_rewards) - sum(strategy_rewards)
    n = len(decisions)
    correct = sum(1 for sd, od in zip(decisions, oracle_decisions, strict=True) if sd[1] == od[1])

    return {
        "CumReg": round(cumreg, 4),
        "rAcc": round(correct / n, 4),
    }


def bootstrap_ci(
    strategy_decisions: list[tuple[str, str, bool, float]],
    oracle_decisions: list[tuple[str, str, bool, float]],
    n_bootstrap: int = 1000,
    gamma: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    task_groups: dict[str, list] = defaultdict(list)
    for d in strategy_decisions:
        task_groups[d[0]].append(d)
    oracle_groups: dict[str, list] = defaultdict(list)
    for d in oracle_decisions:
        oracle_groups[d[0]].append(d)

    task_ids = list(task_groups.keys())
    if not task_ids:
        return ((0.0, 0.0), (0.0, 0.0))

    boot_avgperf: list[float] = []
    boot_cumreg: list[float] = []

    for _ in range(n_bootstrap):
        sample_ids = random.choices(task_ids, k=len(task_ids))
        sample_decisions: list[tuple[str, str, bool, float]] = []
        sample_oracle: list[tuple[str, str, bool, float]] = []
        for tid in sample_ids:
            sample_decisions.extend(task_groups[tid])
            sample_oracle.extend(oracle_groups[tid])

        metrics = compute_metrics(sample_decisions, gamma=gamma)
        comparison = compare_to_oracle(sample_decisions, sample_oracle, gamma=gamma)
        boot_avgperf.append(metrics["AvgPerf%"])
        boot_cumreg.append(comparison["CumReg"])

    alpha = int(0.025 * n_bootstrap)
    boot_avgperf.sort()
    boot_cumreg.sort()
    avgperf_ci = (round(boot_avgperf[alpha], 2), round(boot_avgperf[n_bootstrap - 1 - alpha], 2))
    cumreg_ci = (round(boot_cumreg[alpha], 4), round(boot_cumreg[n_bootstrap - 1 - alpha], 4))

    return avgperf_ci, cumreg_ci


def compute_pareto(strategies_metrics: dict[str, dict]) -> dict[str, bool]:
    names = list(strategies_metrics.keys())
    pareto = {name: True for name in names}

    for i, name_i in enumerate(names):
        mi = strategies_metrics[name_i]
        for j, name_j in enumerate(names):
            if i == j:
                continue
            mj = strategies_metrics[name_j]
            if (
                mj["AvgPerf%"] >= mi["AvgPerf%"]
                and mj["TotalCost"] <= mi["TotalCost"]
                and (mj["AvgPerf%"] > mi["AvgPerf%"] or mj["TotalCost"] < mi["TotalCost"])
            ):
                pareto[name_i] = False
                break

    return pareto

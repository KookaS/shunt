from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean


def _reward(passed: bool, cost: float, gamma: float = 0.1) -> float:
    return 1.0 - gamma * cost if passed else 0.0 - gamma * cost


def _default_arm_passes(
    per_model: dict[str, dict[str, dict]],
    models: list[str],
    default_arms: dict[str, str],
) -> list[bool] | None:
    """Default-arm pass outcome per model, or None if any model's cell is absent."""
    passes: list[bool] = []
    for model in models:
        cell = per_model.get(model, {}).get(default_arms.get(model, ""))
        if cell is None:
            return None
        passes.append(bool(cell.get("pass", False)))
    return passes


def task_signal(
    per_model: dict[str, dict[str, dict]],
    models: list[str],
    default_arms: dict[str, str],
) -> str:
    """One task's routing-signal class from default-arm outcomes.

    ``uncovered`` if any model's default-arm cell is absent; else ``all_pass`` /
    ``all_fail`` / ``discriminating`` (default-arm passes mixed).
    """
    passes = _default_arm_passes(per_model, models, default_arms) if models else None
    if passes is None:
        return "uncovered"
    if all(passes):
        return "all_pass"
    if not any(passes):
        return "all_fail"
    return "discriminating"


def discriminating_set(
    results: dict[str, dict[str, dict[str, dict]]],
    tasks: list[str],
    models: list[str],
    default_arms: dict[str, str],
) -> tuple[set[str], set[str]]:
    """(D, U) membership: D = tiers disagree; U = fully-covered but all-pass|all-fail.

    Pure function of the cache — the shared predicate ``discriminating_stats`` counts
    and the ``collect`` mode strata-splits over, so counts and membership never diverge.
    """
    discriminating: set[str] = set()
    uncontested: set[str] = set()
    for tid in tasks:
        signal = task_signal(results.get(tid, {}), models, default_arms)
        if signal == "discriminating":
            discriminating.add(tid)
        elif signal in ("all_pass", "all_fail"):
            uncontested.add(tid)
    return discriminating, uncontested


def discriminating_stats(
    results: dict[str, dict[str, dict[str, dict]]],
    tasks: list[str],
    models: list[str],
    default_arms: dict[str, str],
) -> dict[str, int]:
    """Routing-signal breakdown of a task set: how many tasks actually discriminate.

    Fully-covered = every model has a default-arm outcome; among those, discriminating
    = default-arm passes are mixed (all-pass / all-fail carry no routing signal).
    """
    counts = {"all_pass": 0, "all_fail": 0, "discriminating": 0}
    n_tasks = 0
    n_fully_covered = 0
    for tid in tasks:
        per_model = results.get(tid, {})
        if any(model in per_model for model in models):
            n_tasks += 1
        signal = task_signal(per_model, models, default_arms)
        if signal == "uncovered":
            continue
        n_fully_covered += 1
        counts[signal] += 1
    return {
        "n_tasks": n_tasks,
        "n_fully_covered": n_fully_covered,
        "n_all_pass": counts["all_pass"],
        "n_all_fail": counts["all_fail"],
        "n_discriminating": counts["discriminating"],
    }


def compute_metrics(decisions: list[tuple[str, str, bool, float]], gamma: float = 0.1) -> dict:
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
    gamma: float = 0.1,
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
    gamma: float = 0.1,
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


def compute_cost_decomposition(
    control_decisions: list[tuple[str, str, bool, float, int, int, int, bool]],
    test_decisions: list[tuple[str, str, bool, float, int, int, int, bool]],
) -> dict:
    """Oaxaca-Blinder decomposition of cost savings into price, volume, and
    interaction effects (summing to frontier_cost - shunt_cost). Only tasks
    where both arms pass are included (equal-quality comparison).
    """
    total_price_saving = 0.0
    total_volume_saving = 0.0
    total_interaction = 0.0
    total_direct_saving = 0.0
    n_eq_pass = 0

    for cd, td in zip(control_decisions, test_decisions, strict=True):
        if not cd[2] or not td[2]:
            continue

        f_cost = cd[3]
        s_cost = td[3]
        f_tok = cd[4] + cd[5]
        s_tok = td[4] + td[5]

        if f_tok <= 0 or s_tok <= 0:
            continue

        f_price = f_cost / f_tok
        s_price = s_cost / s_tok

        tok_diff = f_tok - s_tok
        price_diff = f_price - s_price

        p_save = price_diff * s_tok
        v_save = tok_diff * s_price
        ixn = price_diff * tok_diff

        total_price_saving += p_save
        total_volume_saving += v_save
        total_interaction += ixn
        total_direct_saving += f_cost - s_cost
        n_eq_pass += 1

    if n_eq_pass == 0:
        return {
            "n_eq_pass": 0,
            "total_direct_saving": 0.0,
            "price_savings": 0.0,
            "volume_savings": 0.0,
            "interaction": 0.0,
            "price_pct": 0.0,
            "volume_pct": 0.0,
            "interaction_pct": 0.0,
        }

    decomposition = total_price_saving + total_volume_saving + total_interaction
    price_pct = total_price_saving / decomposition * 100 if decomposition else 0.0
    volume_pct = total_volume_saving / decomposition * 100 if decomposition else 0.0
    ixn_pct = total_interaction / decomposition * 100 if decomposition else 0.0

    return {
        "n_eq_pass": n_eq_pass,
        "total_direct_saving": round(total_direct_saving, 6),
        "price_savings": round(total_price_saving, 6),
        "volume_savings": round(total_volume_saving, 6),
        "interaction": round(total_interaction, 6),
        "price_pct": round(price_pct, 2),
        "volume_pct": round(volume_pct, 2),
        "interaction_pct": round(ixn_pct, 2),
    }


def compute_timing(decisions: list[tuple]) -> dict:
    """Compute timing metrics from decision tuples with calls field.

    Decision tuple: (task_id, model, passed, cost, in_tok, out_tok, calls)
    """
    if not decisions:
        return {}
    total_calls = sum(d[6] for d in decisions)
    avg_calls = total_calls / len(decisions)
    return {
        "total_calls": total_calls,
        "avg_calls_per_task": round(avg_calls, 2),
    }


def compute_timing_per_model(
    strategies_decisions: dict[str, list[tuple]],
) -> dict:
    """Compute avg calls per model across all strategies.

    Each decision tuple has calls at index 6.
    """
    model_calls: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for _strategy_name, decisions in strategies_decisions.items():
        for d in decisions:
            model = d[1]
            calls = d[6]
            model_calls[model] = model_calls.get(model, 0) + calls
            model_counts[model] = model_counts.get(model, 0) + 1
    return {m: round(model_calls[m] / model_counts[m], 1) for m in model_calls}


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

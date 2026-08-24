from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

from benchmark.routing.cache_cost import CachePrice, cache_aware_total
from benchmark.routing.strategies import BilledAttempt


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p for ``b``/``c`` discordant pairs (1.0 when none)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


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
        # Zero-evidence, but SHAPED: every consumer (summary rows, the CSV writer,
        # report.py's plots) indexes these keys. Returning {} used to emit a row with
        # blank TotalCost/AvgPerf% that still got Pareto-flagged, then KeyError'd at
        # plot time. n_tasks == 0 is the explicit "no scorable task" signal.
        return {
            "n_tasks": 0,
            "n_pass": 0,
            "AvgPerf%": 0.0,
            "TotalCost": 0.0,
            "AvgCost": 0.0,
            "Reward": 0.0,
            "AvgReward": 0.0,
        }

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


@dataclass(frozen=True)
class BootstrapCIs:
    """Percentile CIs from one task-resampling bootstrap: quality, regret and COST."""

    # Cost gets a CI for the same reason quality always had one, only more so: the per-task cost
    # distribution here is heavy-tailed (a handful of tasks carry most of the gap between two
    # arms), so a point total is fragile in a way a pass rate is not. The resample is by TASK, the
    # same unit the other two use, so all three CIs describe one uncertainty.
    avgperf: tuple[float, float]
    cumreg: tuple[float, float]
    total_cost: tuple[float, float]
    avg_cost: tuple[float, float]
    # Cache-aware cost is resampled the SAME way, because cache_cost is scoped per task: a
    # whole-task resample keeps each task's own attempt sequence intact, so the within-task
    # adjacency a discount needs survives and the resampled total is the statistic it is named
    # after. Defaulting to zero keeps the old four-field contract for callers with no price map.
    total_cost_cacheaware: tuple[float, float] = (0.0, 0.0)
    avg_cost_cacheaware: tuple[float, float] = (0.0, 0.0)


def _assert_cache_cost_scoping(
    task_ids: Sequence[str],
    attempts: Mapping[str, list[BilledAttempt]],
    prices: Mapping[str, CachePrice],
) -> None:
    """Refuse to bootstrap cache-aware cost unless each task's cost is its own.

    Compute every task's cost forward and backward through the set; a cross-task cache
    model leaks adjacency through hidden state, so the passes disagree and the refusal re-fires.
    """
    forward = [cache_aware_total(attempts.get(tid, []), prices) for tid in task_ids]
    backward = [cache_aware_total(attempts.get(tid, []), prices) for tid in reversed(task_ids)]
    if forward != list(reversed(backward)):
        raise RuntimeError(
            "refusing to bootstrap cache-aware cost: cache cost is not scoped per task "
            "(a task's cost changed with the set/order of the other tasks, so a whole-task "
            "resample does not preserve the statistic)"
        )


def bootstrap_ci(
    strategy_decisions: list[tuple[str, str, bool, float]],
    oracle_decisions: list[tuple[str, str, bool, float]],
    n_bootstrap: int = 1000,
    gamma: float = 0.1,
    seed: int = 42,
    attempts: Mapping[str, list[BilledAttempt]] | None = None,
    prices: Mapping[str, CachePrice] | None = None,
) -> BootstrapCIs:
    """Task-resampled 95% percentile CIs on AvgPerf%, CumReg, TotalCost and AvgCost."""
    # Cost gets a cache-aware CI for the same reason the naive total does — but only because
    # `cache_cost.cache_aware_total` is SCOPED PER TASK (one task = one session, so a discount
    # fires only on a within-task repeat). A whole-task resample keeps each task's own attempt
    # sequence intact, so within-task adjacency survives and the resampled cache-aware total is
    # the statistic it is named after. The old cross-run cache model made one task's cost depend
    # on the tasks before it, which resampling genuinely shredded; that is why this is a guard,
    # not a comment — `_assert_cache_cost_scoping` re-fires the refusal if scoping is ever broken.
    if (attempts is None) != (prices is None):
        raise ValueError(
            "a cache-aware bootstrap CI requires BOTH the per-task attempts and the price map"
        )

    task_groups: dict[str, list] = defaultdict(list)
    for d in strategy_decisions:
        task_groups[d[0]].append(d)
    oracle_groups: dict[str, list] = {}
    for d in oracle_decisions:
        oracle_groups.setdefault(d[0], []).append(d)

    task_ids = list(task_groups.keys())
    zero = (0.0, 0.0)
    if not task_ids:
        return BootstrapCIs(avgperf=zero, cumreg=zero, total_cost=zero, avg_cost=zero)

    missing = [tid for tid in task_ids if tid not in oracle_groups]
    if missing:
        raise ValueError(
            "oracle coverage gap: no oracle row for strategy task(s) "
            f"{sorted(missing)} — a missing oracle row is a coverage gap, not a zero"
        )

    cache_scope: tuple[Mapping[str, list[BilledAttempt]], Mapping[str, CachePrice]] | None = None
    if attempts is not None and prices is not None:
        _assert_cache_cost_scoping(task_ids, attempts, prices)
        cache_scope = (attempts, prices)

    boot_avgperf: list[float] = []
    boot_cumreg: list[float] = []
    boot_total_cost: list[float] = []
    boot_avg_cost: list[float] = []
    boot_total_cache: list[float] = []
    boot_avg_cache: list[float] = []

    rng = random.Random(seed)
    for _ in range(n_bootstrap):
        sample_ids = rng.choices(task_ids, k=len(task_ids))
        sample_decisions: list[tuple[str, str, bool, float]] = []
        sample_oracle: list[tuple[str, str, bool, float]] = []
        for tid in sample_ids:
            sample_decisions.extend(task_groups[tid])
            sample_oracle.extend(oracle_groups[tid])

        metrics = compute_metrics(sample_decisions, gamma=gamma)
        comparison = compare_to_oracle(sample_decisions, sample_oracle, gamma=gamma)
        boot_avgperf.append(metrics["AvgPerf%"])
        boot_cumreg.append(comparison["CumReg"])
        boot_total_cost.append(metrics["TotalCost"])
        boot_avg_cost.append(metrics["AvgCost"])

        if cache_scope is not None:
            scope_attempts, scope_prices = cache_scope
            sample_cache_total = sum(
                cache_aware_total(scope_attempts.get(tid, []), scope_prices) for tid in sample_ids
            )
            boot_total_cache.append(sample_cache_total)
            boot_avg_cache.append(sample_cache_total / len(sample_ids))

    alpha = int(0.025 * n_bootstrap)

    def _pct(draws: list[float], digits: int) -> tuple[float, float]:
        draws.sort()
        return (round(draws[alpha], digits), round(draws[n_bootstrap - 1 - alpha], digits))

    return BootstrapCIs(
        avgperf=_pct(boot_avgperf, 2),
        cumreg=_pct(boot_cumreg, 4),
        total_cost=_pct(boot_total_cost, 4),
        avg_cost=_pct(boot_avg_cost, 6),
        total_cost_cacheaware=_pct(boot_total_cache, 4) if cache_scope is not None else zero,
        avg_cost_cacheaware=_pct(boot_avg_cache, 6) if cache_scope is not None else zero,
    )


def _scorable_pair(
    control: tuple[str, str, bool, float, int, int, int, bool],
    test: tuple[str, str, bool, float, int, int, int, bool],
) -> bool:
    """A task pair enters the equal-quality comparison only when BOTH arms are measured."""
    # ``scorable`` (index 7) is False when a cell was never measured, censored, or imputed —
    # a coverage-gap fill, never a real observation. The decomposition and every pairing in
    # the kill gate share this predicate, so one definition cannot drift from the other.
    return bool(control[7]) and bool(test[7])


def compute_cost_decomposition(
    control_decisions: list[tuple[str, str, bool, float, int, int, int, bool]],
    test_decisions: list[tuple[str, str, bool, float, int, int, int, bool]],
) -> dict:
    """Oaxaca-Blinder decomposition of cost savings into price, volume, and
    interaction effects (summing to frontier_cost - shunt_cost).
    """
    # Only tasks where BOTH arms land on a measured cell and both pass are included
    # (equal-quality comparison on the scorable subset).
    total_price_saving = 0.0
    total_volume_saving = 0.0
    total_interaction = 0.0
    total_direct_saving = 0.0
    n_eq_pass = 0

    for cd, td in zip(control_decisions, test_decisions, strict=True):
        if not _scorable_pair(cd, td):
            continue
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


def cheapest_sufficient(
    per_model: dict[str, dict],
    models_by_price: list[str],
) -> str | None:
    """The cheapest model that actually solved this task — the router's correct answer.

    ``None`` when no measured model passed, so the task has no correct answer to miss.
    """
    # The denominator of the routing error budget. Comparing the chosen model against
    # this, rather than against "did the chosen model pass", separates the two errors a
    # router can make: over-provisioning (a cheaper model would also have solved it) and
    # under-provisioning (the pick failed where a dearer one would have worked).
    for model in models_by_price:
        cell = per_model.get(model)
        if cell and cell.get("pass"):
            return model
    return None


# Below this many co-measured task pairs a reasoning-effort contrast is a rumour, not a
# measurement — the arm figure greys the row rather than reporting its ratio.
MIN_ARM_PAIRS: int = 10
# A reasoning knob has demonstrably FIRED when the high arm spends materially more output
# tokens than the low arm on the same tasks. 1.15 is well outside the +/-5% wobble the four
# never-fired knobs sit inside (0.99-1.02) and well below the one that did (2.92).
ARM_FIRED_RATIO: float = 1.15


def arm_manipulation(
    raw: dict[str, dict[str, dict[str, dict]]],
    pairs: list[tuple[str, str, str]],
) -> list[dict]:
    """Per (model, low arm, high arm): did the reasoning knob change the model's behaviour?

    ``fired`` is the MANIPULATION CHECK — without it a flat pass-rate contrast is not a
    null result, it is a knob that was never turned.
    """
    # Paired on co-measured tasks only, and on OUTPUT tokens rather than pass rate: output
    # tokens are what a reasoning-effort setting mechanically controls, so they answer
    # "was the treatment applied?" before any figure asks "did the treatment help?".
    rows: list[dict] = []
    for model, low, high in pairs:
        lo_tok: list[int] = []
        hi_tok: list[int] = []
        for per_model in raw.values():
            arms = per_model.get(model, {})
            lo_cell, hi_cell = arms.get(low), arms.get(high)
            if not lo_cell or not hi_cell:
                continue
            lo_tok.append(int(lo_cell.get("out_tok", 0) or 0))
            hi_tok.append(int(hi_cell.get("out_tok", 0) or 0))
        n = len(lo_tok)
        lo_mean = mean(lo_tok) if n else 0.0
        hi_mean = mean(hi_tok) if n else 0.0
        ratio = hi_mean / lo_mean if lo_mean > 0 else float("nan")
        rows.append(
            {
                "model": model,
                "low_arm": low,
                "high_arm": high,
                "n_pairs": n,
                "low_out_tok": lo_mean,
                "high_out_tok": hi_mean,
                "out_tok_ratio": ratio,
                "fired": bool(n >= MIN_ARM_PAIRS and ratio == ratio and ratio >= ARM_FIRED_RATIO),
            }
        )
    return rows


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

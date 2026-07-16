"""Out-of-sample generalization test of the external difficulty signal (~490 held-out)."""

from __future__ import annotations

# The 10-task 'External-Prior matches Oracle' result is IN-SAMPLE (it reads each task's
# own rate). This module asks: does difficulty *cluster in embedding space* well enough
# for a neighbour to predict it? The decisive metric is THRESHOLD-FREE — corr / rank-AUC
# between an instance's own p_solve and its embedding neighbours' mean p_solve (LOO). It
# is ~0 here (difficulty doesn't cluster). The tier-routing table is a cost-model-
# dependent illustration with two traps: (1) at k=20/thr=0.5 the neighbour mean can't
# cross 0.5 (p_solve median 0.86) so it escalates nothing — a non-test, not a match;
# (2) "Oracle-tier-acc" maximizes tier accuracy, not reward, and escalates blindly, so at
# γ=0.1 (5.4× cost ratio) it loses reward. The true upper bound is Reward-Oracle (per-task
# max of stay/escalate), which does edge out always-cheap — headroom exists but is tiny.
from dataclasses import dataclass

import numpy as np

from benchmark import config
from benchmark.routing.strategies.knn_blended import _build_hnsw, _default_external_provider


@dataclass(frozen=True)
class RouterRow:
    strategy: str
    accuracy: float  # fraction routed to the tier-accuracy-oracle's tier
    avg_reward: float


@dataclass(frozen=True)
class HeldoutReport:
    rows: list[RouterRow]
    corr: float  # corr(neighbour-mean p_solve, own p_solve) — the generalization signal
    auc: float  # rank-AUC: does a low neighbour-mean flag an own-hard instance? (0.5 = none)
    n: int
    n_hard: int  # instances with own p_solve < threshold
    neighbour_escalations: int  # how often the neighbour predictor actually fired


def _tier_costs() -> tuple[float, float]:
    """(cheap_cost, escalated_cost) — cheapest cheap-tier vs cheapest non-cheap model."""
    pricing = config.load_pricing()

    def total(info: dict) -> float:
        return float(info.get("input_cost_per_1m", 0)) + float(info.get("output_cost_per_1m", 0))

    def tiered(want_cheap: bool) -> list[float]:
        return [
            total(i)
            for m, i in pricing.items()
            if isinstance(i, dict)
            and not m.startswith("_")
            and (i.get("tier") == "cheap") == want_cheap
            and i.get("tier") is not None
        ]

    cheap, other = tiered(True), tiered(False)
    return (min(cheap) if cheap else 1.0, min(other) if other else 10.0)


def _reward(resolve: float, cost: float, gamma: float) -> float:
    return resolve - gamma * cost


def _neighbour_mean(index, emb: np.ndarray, i: int, k: int, vals: np.ndarray) -> float:
    """Mean neighbour value for row i (self excluded). Neutral fallback = GLOBAL mean.

    Never fall back to ``vals[i]`` — that is the leave-one-out target itself and
    would silently leak. Global mean is tier-neutral when no neighbour survives.
    """
    labels, dists = index.knn_query(emb[i : i + 1], min(k + 1, len(emb)))
    kept = [vals[lab] for lab, d in zip(labels[0], dists[0], strict=True) if float(d) >= 0.001]
    return float(np.mean(kept[:k])) if kept else float(np.mean(vals))


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC that a LOW score flags label==1 (here: low neighbour-mean ⇒ own-hard)."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)  # descending: low scores get high rank
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    auc = (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _routers(
    own_cheap_ok: bool, pred_cheap_ok: bool, reward_cheap: float, reward_esc: float
) -> dict:
    """Tier chosen by each router for one instance."""
    return {
        "Always-Cheap": "cheap",
        "Neighbour": "cheap" if pred_cheap_ok else "escalate",
        "Oracle-tier-acc": "cheap" if own_cheap_ok else "escalate",
        "Reward-Oracle": "cheap" if reward_cheap >= reward_esc else "escalate",
    }


def evaluate_heldout(
    threshold: float = 0.5, k: int = 20, gamma: float | None = None
) -> HeldoutReport:
    """Leave-one-out over the ~490 held-out external instances."""
    g = config.gamma() if gamma is None else gamma
    our = {p.stem for p in config.challenge_dir("swebench_verified").glob("*.json")}
    ext = _default_external_provider(exclude=our)
    n = len(ext.iids)
    psolve = np.array([ext.p_cheap[i] for i in ext.iids])  # cheap-tier signal := p_solve
    pfront = np.array([ext.p_frontier[i] for i in ext.iids])
    cheap_cost, esc_cost = _tier_costs()
    index = _build_hnsw(ext.embeddings)

    nbr_mean = np.array([_neighbour_mean(index, ext.embeddings, i, k, psolve) for i in range(n)])
    names = ("Always-Cheap", "Neighbour", "Oracle-tier-acc", "Reward-Oracle")
    hits = dict.fromkeys(names, 0)
    rew = dict.fromkeys(names, 0.0)
    escalations = 0
    for i in range(n):
        oracle_tier = "cheap" if psolve[i] >= threshold else "escalate"
        r_cheap = _reward(psolve[i], cheap_cost, g)
        r_esc = _reward(pfront[i], esc_cost, g)
        routers = _routers(psolve[i] >= threshold, nbr_mean[i] >= threshold, r_cheap, r_esc)
        escalations += routers["Neighbour"] == "escalate"
        for name, tier in routers.items():
            hits[name] += tier == oracle_tier
            rew[name] += r_cheap if tier == "cheap" else r_esc
    rows = [RouterRow(nm, hits[nm] / n, rew[nm] / n) for nm in names]
    hard = (psolve < threshold).astype(int)
    return HeldoutReport(
        rows=rows,
        corr=float(np.corrcoef(nbr_mean, psolve)[0, 1]),
        auc=_rank_auc(nbr_mean, hard),
        n=n,
        n_hard=int(hard.sum()),
        neighbour_escalations=escalations,
    )


def main() -> int:
    config.load()
    rep = evaluate_heldout()
    print(f"Held-out generalization over {rep.n} external instances (leave-one-out, k=20):\n")
    print(
        f"  GENERALIZATION SIGNAL (threshold-free): corr(neighbour p_solve, own) = {rep.corr:.3f}, "
        f"rank-AUC(neighbour flags own-hard) = {rep.auc:.3f}  [0.5 = no signal]"
    )
    print(
        f"  {rep.n_hard}/{rep.n} instances are own-hard; the neighbour predictor fired on "
        f"{rep.neighbour_escalations} (k=20/thr=0.5 mean cannot cross 0.5 — a non-test)\n"
    )
    print(f"  {'router':18} {'tier-accuracy':>14} {'avg-reward (γ=0.1)':>20}")
    for r in sorted(rep.rows, key=lambda x: -x.avg_reward):
        print(f"  {r.strategy:18} {r.accuracy:>13.1%} {r.avg_reward:>20.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

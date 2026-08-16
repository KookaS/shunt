"""Unit tests for the ladder collector (challenge-major, completion-predictive budget)."""

# Dispatch is STUBBED throughout — no live/paid harness path runs. The collector's job is
# scheduling (challenge-major: fully escalate one challenge cheap-first until its first pass,
# skipping cached rungs, before starting the next) plus a completion-predictive budget guard
# that never starts a challenge it cannot afford to FINISH. These tests prove that scheduling
# and the zero-UNKNOWN imputation property it creates.

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Final

import pytest

from benchmark import config
from benchmark.config import CapabilityRank, RankedModel
from benchmark.routing.impute import complete_matrix
from benchmark.runner import ladder_collect, run_matrix

# A 4-model synthetic capability rank; the collector escalates weakest -> strongest.
_ORDER: Final[list[str]] = ["c0", "m0", "h0", "f0"]
RANK: Final[CapabilityRank] = CapabilityRank(
    ordered=[RankedModel(m, "a", i, "measured") for i, m in enumerate(_ORDER)], evidence={}
)


class _Recorder:
    """Stubs collect_phase + the cache read: records each (task, model) call and lets a task
    'pass' at a preset true crossover model, so escalation shrinks the frontier like live."""

    def __init__(self, tau: dict[str, str]) -> None:
        self.tau = tau  # task -> the ranked model at which it first passes
        self.calls: list[tuple[str, str]] = []
        self._observed: dict[str, dict[str, bool]] = {}  # task -> {model: passed?}

    def collect_phase(self, tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        model = models[0]
        for t in tasks:  # challenge-major → one task per call, but honour the list form
            self.calls.append((t, model))
            self._observed.setdefault(t, {})[model] = self.tau.get(t) == model

    def solved_matrix(self) -> dict:
        return {t: {m: {"pass": p} for m, p in obs.items()} for t, obs in self._observed.items()}

    def tiers(self, task: str) -> list[str]:
        return [m for t, m in self.calls if t == task]


def _wire(monkeypatch: pytest.MonkeyPatch, rec: _Recorder) -> None:
    monkeypatch.setattr(config, "capability_rank", lambda *a, **k: RANK)
    monkeypatch.setattr(ladder_collect, "collect_phase", rec.collect_phase)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", rec.solved_matrix)
    monkeypatch.setattr(ladder_collect, "_prepare", lambda tasks: (tasks, {}, {}, None, None))


def _run(monkeypatch: pytest.MonkeyPatch, rec: _Recorder, tasks: list[str]) -> int:
    _wire(monkeypatch, rec)
    monkeypatch.setattr(ladder_collect, "_sampled_tasks", lambda: list(tasks))
    return ladder_collect.run_ladder(live=False)


# ------------------------------------------------------------------ scheduling behaviour


# --------------------------------------------------- FIX E: uncapped --live confirmation


def test_confirm_uncapped_live_refuses_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_matrix, "_prompt_confirm", lambda _p: None)  # non-TTY / EOF
    assert ladder_collect._confirm_uncapped_live() is False


def test_confirm_uncapped_live_accepts_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_matrix, "_prompt_confirm", lambda _p: "  Y  ")
    assert ladder_collect._confirm_uncapped_live() is True


def test_confirm_uncapped_live_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_matrix, "_prompt_confirm", lambda _p: "maybe")
    assert ladder_collect._confirm_uncapped_live() is False


def test_run_ladder_refuses_uncapped_live(monkeypatch: pytest.MonkeyPatch) -> None:
    # An uncapped (max_cost=None) --live ladder run must refuse unless confirmed — mirrors the
    # full-matrix guard. Confirm stubbed False (as a non-interactive stdin would) → exit code 3.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setattr(config, "capability_rank", lambda *a, **k: RANK)
    monkeypatch.setattr(config, "collect_config", lambda: {"constants_pinned": True})
    monkeypatch.setattr(ladder_collect, "_confirm_uncapped_live", lambda: False)
    monkeypatch.setattr(ladder_collect, "_sampled_tasks", lambda: ["t1"])
    assert ladder_collect.run_ladder(live=True, max_cost=None) == 3


def test_cheap_first_escalation_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(tau={"t1": "h0"})
    _run(monkeypatch, rec, ["t1"])
    assert rec.tiers("t1") == ["c0", "m0", "h0"]  # weakest->up, stop at h0 (f0 never runs)


def test_stop_on_first_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(tau={"t1": "c0"})
    _run(monkeypatch, rec, ["t1"])
    assert rec.tiers("t1") == ["c0"]  # passed at weakest -> no higher rung runs


def test_escalate_on_fail_per_task(monkeypatch: pytest.MonkeyPatch) -> None:
    # Challenge-major: each challenge is fully escalated (cheap-first, stop at its pass)
    # before the next starts. t1@c0, t2@m0, t3@h0.
    rec = _Recorder(tau={"t1": "c0", "t2": "m0", "t3": "h0"})
    _run(monkeypatch, rec, ["t1", "t2", "t3"])
    assert rec.tiers("t1") == ["c0"]  # solved at weakest
    assert rec.tiers("t2") == ["c0", "m0"]  # fail c0, pass m0
    assert rec.tiers("t3") == ["c0", "m0", "h0"]  # fail c0/m0, pass h0
    assert "f0" not in {m for _, m in rec.calls}  # everything solved by h0 -> f0 never runs
    # Challenge-major ORDER: t1 fully done before t2 starts, t2 before t3.
    assert rec.calls == [
        ("t1", "c0"),
        ("t2", "c0"),
        ("t2", "m0"),
        ("t3", "c0"),
        ("t3", "m0"),
        ("t3", "h0"),
    ]


def test_resume_from_partial_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # t1 already cached as a c0 PASS before the run: it is skipped ENTIRELY (crossed).
    rec = _Recorder(tau={"t1": "c0", "t2": "m0"})
    rec._observed["t1"] = {"c0": True}  # pre-seed the cache (a prior interrupted run)
    _run(monkeypatch, rec, ["t1", "t2"])
    assert rec.tiers("t1") == []  # never re-requested (cached pass reused)
    assert rec.tiers("t2") == ["c0", "m0"]


def test_early_stop_when_all_solved(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(tau={"t1": "c0", "t2": "c0"})
    rc = _run(monkeypatch, rec, ["t1", "t2"])
    assert rc == 0
    assert rec.calls == [("t1", "c0"), ("t2", "c0")]  # each solved at the weakest tier


# --------------------------------------------- completion-predictive (challenge-atomic) budget


def test_challenge_atomic_budget_never_starts_unaffordable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each tier costs $1 and always FAILS, so completing one fresh challenge runs all 4 tiers
    # ($4 worst-case). Cap $9: t1 completes (spent→4), t2 completes (spent→8); t3 would need
    # up to $4 more → 8+4=12 > 9, so it is NEVER started. No partial/wasted challenge.
    spend = {"v": 0.0}
    calls: list[tuple[str, str]] = []

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        calls.append((tasks[0], models[0]))
        spend["v"] += 1.0  # $1 per tier; the cell stays a FAIL (solved_matrix never shows a pass)

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(
        ladder_collect, "_solved_matrix", lambda: {}
    )  # nothing cached / ever passes
    monkeypatch.setattr(ladder_collect, "_total_real_cost", lambda _p: spend["v"])
    monkeypatch.setattr(ladder_collect, "_tier_cost_ceilings", lambda _p: {m: 1.0 for m in _ORDER})

    ladder_collect._run_challenges(
        list(RANK.ordered),
        ["t1", "t2", "t3"],
        {},
        {},
        None,
        Path("nonexistent.csv"),
        live=True,
        timeout=1,
        workers=1,
        max_cost=9.0,
        max_cost_overshoot=0.0,
        max_start_failures=None,
        max_consecutive_failures=None,
    )
    started = {t for t, _ in calls}
    assert started == {"t1", "t2"}  # t3 unaffordable → never started (challenge-atomic)
    # Every STARTED challenge is FULLY completed (all 4 tiers), never left partial.
    assert [m for t, m in calls if t == "t1"] == _ORDER
    assert [m for t, m in calls if t == "t2"] == _ORDER
    assert spend["v"] == 8.0  # spent only on fully-completed challenges; nothing wasted on t3


def test_challenge_atomic_budget_starts_last_affordable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cap $8 (exactly two challenges' worth): both t1 and t2 complete; t2 is the LAST fully
    # affordable challenge and it finishes. t3 (would push to 12) is not started.
    spend = {"v": 0.0}
    calls: list[tuple[str, str]] = []

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        calls.append((tasks[0], models[0]))
        spend["v"] += 1.0

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", lambda: {})
    monkeypatch.setattr(ladder_collect, "_total_real_cost", lambda _p: spend["v"])
    monkeypatch.setattr(ladder_collect, "_tier_cost_ceilings", lambda _p: {m: 1.0 for m in _ORDER})

    ladder_collect._run_challenges(
        list(RANK.ordered),
        ["t1", "t2", "t3"],
        {},
        {},
        None,
        Path("nonexistent.csv"),
        live=True,
        timeout=1,
        workers=1,
        max_cost=8.0,
        max_cost_overshoot=0.0,
        max_start_failures=None,
        max_consecutive_failures=None,
    )
    assert {t for t, _ in calls} == {"t1", "t2"}
    assert [m for t, m in calls if t == "t2"] == _ORDER  # the last affordable one completed
    assert spend["v"] == 8.0


def test_tier_cost_ceilings_uses_max_not_median(tmp_path: Path) -> None:
    # FIX 2: the per-model reservation cost is the MAX observed real_cost (a true upper bound),
    # not the median (which is exceeded ~half the time and so under-reserves).
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("model,real_cost\nc0,0.10\nc0,0.10\nc0,5.00\nm0,0.20\nbad,notafloat\n")
    ceilings = ladder_collect._tier_cost_ceilings(csv_path)
    assert ceilings["c0"] == 5.00  # max of {0.10, 0.10, 5.00}, NOT the 0.10 median
    assert ceilings["m0"] == 0.20
    # A path with no data yields no ceilings (the caller falls back to cold_start).
    assert ladder_collect._tier_cost_ceilings(tmp_path / "absent.csv") == {}


def test_worst_case_is_conservative_upper_bound() -> None:
    # FIX 1 + FIX 2: sum the per-tier CEILINGS over untested tiers; an unmeasured tier reserves
    # max(cold_start, priciest known ceiling) — never 0.
    ceilings = {"c0": 5.0, "m0": 5.0}
    # median-reservation would sum ~1.0/tier and admit; the max-based ceiling sums the true worst.
    assert ladder_collect._worst_case_to_complete(["c0", "m0"], ceilings, 1.0) == 10.0
    # h0 is UNMEASURED → borrows max(cold_start=1.0, max ceiling=5.0) = 5.0 (never 0).
    assert ladder_collect._worst_case_to_complete(["c0", "m0", "h0"], ceilings, 1.0) == 15.0
    # FIX 1: NO data at all → each tier reserves the conservative cold_start floor, never 0.
    assert ladder_collect._worst_case_to_complete(["c0", "m0"], {}, 1.0) == 2.0


def test_explicit_tasks_override_sampled_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # A targeted run (e.g. only the still-unknown-crossover challenges) passes tasks explicitly;
    # they must override _sampled_tasks so the ladder collects only those.
    rec = _Recorder(tau={})  # nothing passes -> every tier runs on the given task
    _wire(monkeypatch, rec)
    monkeypatch.setattr(ladder_collect, "_sampled_tasks", lambda: ["SHOULD_NOT_BE_USED"])
    ladder_collect.run_ladder(tasks=["only-this-one"], live=False)
    ran = {t for t, _ in rec.calls}
    assert ran == {"only-this-one"}  # never touched the sampled set


# ------------------------------------------------------------------ strategy dispatch


def test_strategy_ladder_dispatches_to_run_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run_ladder(config_path: str = "", **kwargs: object) -> int:
        seen["called"] = True
        seen["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(ladder_collect, "run_ladder", fake_run_ladder)
    args = argparse.Namespace(
        strategy="ladder",
        config="benchmark/benchmark.yaml",
        live=False,
        timeout=600,
        workers=1,
        max_cost=None,
        max_cost_overshoot=0.0,
        max_start_failures=5,
        max_consecutive_failures=5,
        check_images=False,
    )
    assert run_matrix._dispatch(args) == 0
    assert seen["called"] is True
    # the flags that gate spend/threading must thread through unchanged
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["max_cost_overshoot"] == 0.0
    assert kwargs["max_start_failures"] == 5
    assert kwargs["max_consecutive_failures"] == 5


def test_strategy_ladder_is_a_choice() -> None:
    ap = argparse.ArgumentParser()
    run_matrix._add_args(ap, "benchmark/benchmark.yaml")
    parsed = ap.parse_args(["--strategy", "ladder"])
    assert parsed.strategy == "ladder"


# ---------------------------------------------- the property that makes coverage equal


def _regime_a_cache(tau: dict[str, str]) -> dict:
    """A ladder-collection cache: real cells weakest..tau only (fail below, pass at tau)."""
    cache: dict[str, dict] = {}
    for task, crossover in tau.items():
        cut = _ORDER.index(crossover)
        cells = {}
        for i, model in enumerate(_ORDER[: cut + 1]):
            passed = i == cut
            cells[model] = {"pass": passed, "real_cost": 0.1 * (i + 1), "cost": 0.1 * (i + 1)}
        cache[task] = cells
    return cache


def test_regime_a_yields_zero_unknown() -> None:
    # A fully ladder-collected matrix imputes with NO residual UNKNOWN band -> equal coverage,
    # and every task is COMPLETE (an established crossover).
    tau = {"t1": "c0", "t2": "m0", "t3": "h0", "t4": "f0"}
    cache = _regime_a_cache(tau)
    im = complete_matrix(cache, RANK)
    assert im.n_unknown == 0
    assert im.tau == tau  # crossover recovered exactly for every task
    assert im.complete == frozenset(tau)  # all tasks complete; none excluded
    n_models = len(_ORDER)
    for cells in im.matrix.values():
        assert len(cells) == n_models  # every strategy scorable on every task


# ============================================================ CROSS-CHALLENGE PARALLELISM
#
# The ladder escalates up to ``--workers`` DIFFERENT challenges concurrently (same fan-out
# mechanism/default as cost_optimal/full), while each challenge still escalates cheap->strong
# SERIALLY, stopping at its first pass. Budget stays challenge-atomic under concurrency via a
# worst-case reservation. All dispatch is STUBBED — no live/paid harness runs.


class _ConcurrentRecorder:
    """Thread-safe recorder: per-(task,model) calls, peak concurrency, and pass crossovers."""

    def __init__(self, tau: dict[str, str], *, delay: float = 0.0) -> None:
        self.tau = tau
        self.delay = delay
        self.calls: list[tuple[str, str]] = []
        self._observed: dict[str, dict[str, bool]] = {}
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def collect_phase(self, tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        task, model = tasks[0], models[0]
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.calls.append((task, model))
            self._observed.setdefault(task, {})[model] = self.tau.get(task) == model
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self._active -= 1

    def solved_matrix(self) -> dict:
        with self._lock:
            return {
                t: {m: {"pass": p} for m, p in obs.items()} for t, obs in self._observed.items()
            }

    def tiers(self, task: str) -> list[str]:
        with self._lock:
            return [m for t, m in self.calls if t == task]


def _wire_concurrent(monkeypatch: pytest.MonkeyPatch, rec: _ConcurrentRecorder) -> None:
    monkeypatch.setattr(config, "capability_rank", lambda *a, **k: RANK)
    monkeypatch.setattr(ladder_collect, "collect_phase", rec.collect_phase)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", rec.solved_matrix)
    monkeypatch.setattr(ladder_collect, "_prepare", lambda tasks: (tasks, {}, {}, None, None))


def _run_concurrent(
    monkeypatch: pytest.MonkeyPatch, rec: _ConcurrentRecorder, tasks: list[str], workers: int
) -> int:
    _wire_concurrent(monkeypatch, rec)
    monkeypatch.setattr(ladder_collect, "_sampled_tasks", lambda: list(tasks))
    return ladder_collect.run_ladder(live=False, workers=workers)


def test_challenges_escalate_concurrently_bounded_by_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 8 one-rung challenges, workers=4: the pool genuinely overlaps DIFFERENT challenges (peak >1)
    # but never exceeds ``workers``. Every challenge is still collected.
    rec = _ConcurrentRecorder(tau={f"t{i}": "c0" for i in range(8)}, delay=0.05)
    _run_concurrent(monkeypatch, rec, [f"t{i}" for i in range(8)], workers=4)
    assert 1 < rec.max_active <= 4  # concurrency happened, capped at --workers
    assert {t for t, _ in rec.calls} == {f"t{i}" for i in range(8)}  # all collected


def test_within_challenge_escalation_stays_serial_stop_first_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Under concurrency each challenge still escalates cheap->strong and STOPS at its first pass:
    # tier N+1 runs only if tier N failed. t2 crosses at h0 → exactly [c0, m0, h0], never f0.
    rec = _ConcurrentRecorder(tau={"t1": "c0", "t2": "h0", "t3": "m0"}, delay=0.02)
    _run_concurrent(monkeypatch, rec, ["t1", "t2", "t3"], workers=3)
    assert rec.tiers("t1") == ["c0"]
    assert rec.tiers("t2") == ["c0", "m0", "h0"]  # serial, in order, stop at the pass
    assert rec.tiers("t3") == ["c0", "m0"]
    assert "f0" not in {m for _, m in rec.calls}


def test_worker_count_invariance_same_cell_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SET of collected cells is IDENTICAL for workers=1 and workers=4 — only order differs.
    tau = {"t1": "c0", "t2": "m0", "t3": "h0", "t4": "f0", "t5": "m0", "t6": "c0"}
    tasks = list(tau)
    rec1 = _ConcurrentRecorder(tau=tau)
    _run_concurrent(monkeypatch, rec1, tasks, workers=1)
    rec4 = _ConcurrentRecorder(tau=tau, delay=0.01)
    _run_concurrent(monkeypatch, rec4, tasks, workers=4)
    assert frozenset(rec1.calls) == frozenset(rec4.calls)


def _budget_stubs(
    monkeypatch: pytest.MonkeyPatch, spend: dict[str, float], calls: list[tuple[str, str]]
) -> threading.Lock:
    """Wire the budget stubs (nothing ever passes; $1/rung; costs read from ``spend``)."""
    lock = threading.Lock()

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        with lock:
            calls.append((tasks[0], models[0]))
            spend["v"] += 1.0
        time.sleep(0.02)  # force real overlap so a broken reservation WOULD overspend

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", lambda: {})
    monkeypatch.setattr(ladder_collect, "_total_real_cost", lambda _p: spend["v"])
    monkeypatch.setattr(ladder_collect, "_tier_cost_ceilings", lambda _p: {m: 1.0 for m in _ORDER})
    return lock


def test_budget_reservation_prevents_concurrent_overspend(monkeypatch: pytest.MonkeyPatch) -> None:
    # THE CRITICAL ONE. 5 challenges, each worst-case $4 (4 tiers × $1, all fail), cap $9,
    # workers=4. If concurrent challenges checked the budget INDEPENDENTLY, all 4 would each see
    # spent≈0 and start → $16 ≫ $9. The worst-case reservation instead admits only the {t1,t2}
    # prefix ($8 ≤ $9); t3+ never start. Spend never crosses the cap.
    spend = {"v": 0.0}
    calls: list[tuple[str, str]] = []
    _budget_stubs(monkeypatch, spend, calls)
    ladder_collect._run_challenges(
        list(RANK.ordered),
        ["t1", "t2", "t3", "t4", "t5"],
        {},
        {},
        None,
        Path("nonexistent.csv"),
        live=True,
        timeout=1,
        workers=4,
        max_cost=9.0,
        max_cost_overshoot=0.0,
        max_start_failures=None,
        max_consecutive_failures=None,
    )
    started = {t for t, _ in calls}
    assert started == {"t1", "t2"}  # only the affordable prefix started
    assert spend["v"] == 8.0 <= 9.0  # never overspent, even with 4 workers racing
    # Every STARTED challenge is FULLY completed (all 4 tiers) — never left partial.
    for task in started:
        assert [m for t, m in calls if t == task] == _ORDER


def test_budget_reservation_matches_serial_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    # Worker-count invariance for the BUDGET path: the admitted set is the same at workers=1 and 4.
    def run(workers: int) -> set[str]:
        spend = {"v": 0.0}
        calls: list[tuple[str, str]] = []
        _budget_stubs(monkeypatch, spend, calls)
        ladder_collect._run_challenges(
            list(RANK.ordered),
            ["t1", "t2", "t3", "t4"],
            {},
            {},
            None,
            Path("nonexistent.csv"),
            live=True,
            timeout=1,
            workers=workers,
            max_cost=9.0,
            max_cost_overshoot=0.0,
            max_start_failures=None,
            max_consecutive_failures=None,
        )
        return {t for t, _ in calls}

    assert run(1) == run(4) == {"t1", "t2"}


def test_cold_start_empty_csv_does_not_admit_unbounded_first_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FIX 1: fresh run — empty results.csv → no measured ceilings. Each of the 4 untested tiers
    # reserves the cold_start default ($1.0), so a challenge's worst case is $4. Cap $9, workers=4:
    # if unmeasured tiers reserved 0 (the OLD bug), all 4 would start uncapped. With the cold-start
    # floor only the {t1,t2} prefix (2×$4=$8 ≤ $9) is admitted; t3+ never start.
    spend = {"v": 0.0}
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        with lock:
            calls.append((tasks[0], models[0]))
            spend["v"] += 1.0
        time.sleep(0.02)  # force overlap so a zero reservation WOULD admit an unbounded wave

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", lambda: {})
    monkeypatch.setattr(ladder_collect, "_total_real_cost", lambda _p: spend["v"])
    # NB: _tier_cost_ceilings is NOT stubbed — it reads the nonexistent CSV → {} (no measured cost),
    # exercising the real cold-start fallback. cold_start_tier_cost=1.0 comes from benchmark.yaml.
    ladder_collect._run_challenges(
        list(RANK.ordered),
        ["t1", "t2", "t3", "t4"],
        {},
        {},
        None,
        Path("nonexistent.csv"),
        live=True,
        timeout=1,
        workers=4,
        max_cost=9.0,
        max_cost_overshoot=0.0,
        max_start_failures=None,
        max_consecutive_failures=None,
    )
    started = {t for t, _ in calls}
    assert started == {"t1", "t2"}  # cold-start reservation blocked the unbounded 4-wide first wave
    assert spend["v"] <= 9.0  # never overspent on a fresh run


def _nonuniform_stubs(
    monkeypatch: pytest.MonkeyPatch, spend: dict[str, float], calls: list[tuple[str, str]]
) -> None:
    """Per-tier costs DIFFER (c0<m0<h0<f0); each cell spends its own tier's real cost."""
    costs = {"c0": 0.5, "m0": 1.0, "h0": 2.0, "f0": 3.0}  # worst case to finish one challenge = 6.5
    lock = threading.Lock()

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        with lock:
            calls.append((tasks[0], models[0]))
            spend["v"] += costs[models[0]]
        time.sleep(0.01)  # force overlap so a broken reservation WOULD overspend

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", lambda: {})
    monkeypatch.setattr(ladder_collect, "_total_real_cost", lambda _p: spend["v"])
    monkeypatch.setattr(ladder_collect, "_tier_cost_ceilings", lambda _p: dict(costs))


def test_nonuniform_costs_workers_subset_and_never_overspend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FIX 2 + FIX 3 SAFETY under NON-UNIFORM per-tier costs (worst case per challenge = $6.5).
    # Cap $13, so exactly the {t1,t2} prefix fits (2×$6.5=$13). The admitted set with --workers>1
    # must be a SUBSET of (never a superset of) the serial workers=1 set, and spend never exceeds
    # the cap regardless of worker count.
    def run(workers: int) -> tuple[set[str], float]:
        spend = {"v": 0.0}
        calls: list[tuple[str, str]] = []
        _nonuniform_stubs(monkeypatch, spend, calls)
        ladder_collect._run_challenges(
            list(RANK.ordered),
            ["t1", "t2", "t3", "t4", "t5"],
            {},
            {},
            None,
            Path("nonexistent.csv"),
            live=True,
            timeout=1,
            workers=workers,
            max_cost=13.0,
            max_cost_overshoot=0.0,
            max_start_failures=None,
            max_consecutive_failures=None,
        )
        return {t for t, _ in calls}, spend["v"]

    serial, serial_spend = run(1)
    concurrent, concurrent_spend = run(4)
    assert serial == {"t1", "t2"}
    assert concurrent <= serial  # subset, never a superset — more workers never admit MORE
    assert serial_spend <= 13.0 and concurrent_spend <= 13.0  # no overspend at any worker count


def test_abort_halts_dispatch_within_pool_slack(monkeypatch: pytest.MonkeyPatch) -> None:
    # FIX 4: a worker that hits ApiUnusableError latches the abort BEFORE its future is harvested,
    # so the dispatcher admits NO further challenges. With workers=3 at most the initial pool-full
    # wave can have started before the first failure latches — dispatch stops there, not at 20.
    started: list[str] = []
    lock = threading.Lock()

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        with lock:
            started.append(tasks[0])
        raise run_matrix.RunAbortError("aborting: API unusable (dead key)")

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", lambda: {})
    with pytest.raises(run_matrix.RunAbortError, match="API unusable"):
        ladder_collect._run_challenges(
            list(RANK.ordered),
            [f"t{i}" for i in range(20)],
            {},
            {},
            None,
            Path("nonexistent.csv"),
            live=True,
            timeout=1,
            workers=3,
            max_cost=None,
            max_cost_overshoot=0.0,
            max_start_failures=None,
            max_consecutive_failures=None,
        )
    assert 1 <= len(set(started)) <= 3  # stopped at the failure point, within pool-size slack


def test_api_unusable_aborts_all_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    # A systemic API-unusable failure (collect_phase raises RunAbortError, exactly as the real
    # phase does after infer.ApiUnusableError) aborts the WHOLE run: it propagates, and NOT all
    # challenges get dispatched — new dispatch halts once the abort is latched.
    started: list[str] = []
    lock = threading.Lock()

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        with lock:
            started.append(tasks[0])
        time.sleep(0.02)
        raise run_matrix.RunAbortError("aborting: API unusable (dead key)")

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", lambda: {})
    with pytest.raises(run_matrix.RunAbortError, match="API unusable"):
        ladder_collect._run_challenges(
            list(RANK.ordered),
            [f"t{i}" for i in range(12)],
            {},
            {},
            None,
            Path("nonexistent.csv"),
            live=True,
            timeout=1,
            workers=3,
            max_cost=None,
            max_cost_overshoot=0.0,
            max_start_failures=None,
            max_consecutive_failures=None,
        )
    assert len(set(started)) < 12  # aborted early — not every challenge was dispatched


def test_consecutive_failure_catch_all_across_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    # The catch-all consecutive-failure abort surfaces from a worker thread and halts the run.
    # (Stub raises the same RunAbortError the shared _FailureTracker raises once the threshold
    # is crossed by no-row cells accumulating across concurrent challenges.)
    started: list[str] = []
    lock = threading.Lock()
    n = {"v": 0}

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **_k: object) -> None:
        with lock:
            started.append(tasks[0])
            n["v"] += 1
            crossed = n["v"] >= 3
        time.sleep(0.01)
        if crossed:
            raise run_matrix.RunAbortError("aborting: 3 consecutive cell failures")

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", lambda: {})
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        ladder_collect._run_challenges(
            list(RANK.ordered),
            [f"t{i}" for i in range(20)],
            {},
            {},
            None,
            Path("nonexistent.csv"),
            live=True,
            timeout=1,
            workers=4,
            max_cost=None,
            max_cost_overshoot=0.0,
            max_start_failures=None,
            max_consecutive_failures=3,
        )
    assert len(set(started)) < 20  # halted, did not march through every challenge


def test_shared_failure_tracker_is_threaded_into_collect_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Wiring proof: every concurrent challenge worker hands collect_phase the SAME shared
    # _FailureTracker (and a shared write_lock), so the catch-all counter and CSV writes are
    # genuinely cross-thread — not a fresh per-call counter that could never accumulate.
    seen_failures: list[object] = []
    seen_locks: list[object] = []
    lock = threading.Lock()

    def fake_collect(tasks: list[str], models: list[str], *_a: object, **kw: object) -> None:
        with lock:
            seen_failures.append(kw.get("failures"))
            seen_locks.append(kw.get("write_lock"))

    monkeypatch.setattr(ladder_collect, "collect_phase", fake_collect)
    monkeypatch.setattr(ladder_collect, "_solved_matrix", lambda: {})
    ladder_collect._run_challenges(
        list(RANK.ordered),
        ["t1", "t2", "t3"],
        {},
        {},
        None,
        Path("nonexistent.csv"),
        live=True,
        timeout=1,
        workers=3,
        max_cost=None,
        max_cost_overshoot=0.0,
        max_start_failures=5,
        max_consecutive_failures=5,
    )
    assert seen_failures and all(f is seen_failures[0] for f in seen_failures)
    assert isinstance(seen_failures[0], run_matrix._FailureTracker)
    assert seen_locks and all(lk is seen_locks[0] for lk in seen_locks)


def test_resume_skips_cached_rungs_under_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    # A killed run resumes: challenges already cached as a PASS are skipped entirely, and only the
    # untested rungs of the rest run — same as serial, but concurrently.
    rec = _ConcurrentRecorder(tau={"t1": "c0", "t2": "m0", "t3": "c0"}, delay=0.01)
    rec._observed["t1"] = {"c0": True}  # pre-seed a prior interrupted run's cached PASS
    _run_concurrent(monkeypatch, rec, ["t1", "t2", "t3"], workers=3)
    assert rec.tiers("t1") == []  # cached pass reused — never re-requested
    assert rec.tiers("t2") == ["c0", "m0"]
    assert rec.tiers("t3") == ["c0"]


def test_failure_tracker_thread_safe_consecutive_abort() -> None:
    # Unit-level: the shared _FailureTracker aborts on N consecutive soft failures and a success
    # resets the counter — the thread-safe primitive the ladder shares across challenge workers.
    tracker = run_matrix._FailureTracker(max_start_failures=None, max_consecutive_failures=3)
    tracker.record_soft_failure()
    tracker.record_soft_failure()
    tracker.record_success()  # resets — the next two failures must NOT abort
    tracker.record_soft_failure()
    tracker.record_soft_failure()
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        tracker.record_soft_failure()  # third consecutive → abort


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_cold_start_tier_cost_rejects_nonpositive(
    bad: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 0/negative reservation would defeat the overspend guard (unmeasured tiers reserve $0),
    # so the accessor MUST reject it loudly instead of silently disarming the guard.
    monkeypatch.setattr(config, "ladder_config", lambda: {"cold_start_tier_cost": bad})
    with pytest.raises(ValueError, match="cold_start_tier_cost must be > 0"):
        config.cold_start_tier_cost()


def test_cold_start_tier_cost_accepts_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ladder_config", lambda: {"cold_start_tier_cost": 2.5})
    assert config.cold_start_tier_cost() == 2.5

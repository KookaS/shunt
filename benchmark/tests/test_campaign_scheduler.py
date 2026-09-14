"""Priority-first phase scheduler: cap, order, idle fill, phase gate, resume — no model calls.

Every case drives the pure planner or the pull loop against a hand-made plan and a
monkeypatched cell executor, so the whole schedule is proven offline.
"""

from __future__ import annotations

from typing import Any

from benchmark.runner import campaign_scheduler as cs
from benchmark.runner import lane_scheduler as ls

TEXT = cs.TEXT_BENCHMARK
MM = cs.MULTIMODAL_BENCHMARK


class _Engine:
    """A scripted ``collection_priority`` surface: no corpus, no YAML, no model call."""

    def __init__(
        self,
        importance: dict[str, float],
        *,
        worth: dict[str, tuple[bool, str]] | None = None,
        runnable: dict[str, list[str]] | None = None,
        duplicate: dict[str, str] | None = None,
    ) -> None:
        self.importance = importance
        self.worth = worth or {}
        self.runnable = runnable or {}
        self.duplicate = duplicate or {}

    def priority(self, identity: str) -> float:
        return self.importance.get(identity, 0.1)

    def worth_collecting(self, identity: str) -> tuple[bool, str]:
        return self.worth.get(identity, (True, "keep"))

    def runnable_benchmarks(self, identity: str) -> list[str]:
        return self.runnable.get(identity, [TEXT])

    def duplicate_of(self, identity: str) -> str | None:
        return self.duplicate.get(identity)


# ── worker cap ───────────────────────────────────────────────────────────────


def test_effective_workers_is_capped_by_runnable_models() -> None:
    assert cs.effective_workers(8, 3) == 3
    assert cs.effective_workers(2, 5) == 2
    assert cs.effective_workers(4, 4) == 4
    assert cs.effective_workers(4, 0) == 0


def test_plan_exposes_the_effective_worker_cap() -> None:
    plan = cs.build_plan(["a", "b", "c"], requested_workers=8, engine=_Engine({}))
    assert plan.requested_workers == 8
    assert plan.effective_workers == 3


# ── priority order ───────────────────────────────────────────────────────────


def test_plan_orders_models_by_descending_priority() -> None:
    engine = _Engine({"high": 10.0, "mid": 5.0, "low": 1.0})
    plan = cs.build_plan(["low", "high", "mid"], requested_workers=3, engine=engine)
    assert plan.models == ["high", "mid", "low"]
    assert [phase.priority for phase in plan.phases] == [10.0, 5.0, 1.0]


def test_plan_is_deterministic_for_the_same_engine() -> None:
    engine = _Engine({"b": 5.0, "a": 5.0, "c": 5.0})
    first = cs.build_plan(["c", "a", "b"], requested_workers=2, engine=engine)
    second = cs.build_plan(["c", "a", "b"], requested_workers=2, engine=engine)
    assert first == second
    assert first.models == ["a", "b", "c"]  # equal priority breaks ties by name


# ── named exclusions: duplicate, not-worth, retired ──────────────────────────


def test_a_duplicate_channel_is_dropped() -> None:
    engine = _Engine({"orig": 5.0}, duplicate={"twin": "orig"})
    plan = cs.build_plan(["orig", "twin"], engine=engine)
    assert plan.duplicates == {"twin": "orig"}
    assert plan.models == ["orig"]


def test_build_plan_queues_every_concordance_subset_channel(monkeypatch) -> None:
    """build_plan itself exempts the committed subset, so all 3 channels per identity queue.

    Regression: ``duplicate_of`` dropped the 2nd/3rd channel of each identity, so the promised
    3x3 concordance subset never queued unless a caller remembered to wrap the engine. The
    exemption now lives in ``build_plan`` (the single owner), which reads the committed subset.
    """
    from pathlib import Path

    from benchmark import config
    from benchmark.routing import collection_priority as cp

    root = Path(__file__).resolve().parents[2]
    overlay = root / "configs/free-tier/overlay.yaml"
    monkeypatch.setattr(config, "_config", config._config)
    monkeypatch.setattr(config, "_pricing", None)
    monkeypatch.setattr(config, "_free_registry", None)
    monkeypatch.setattr(config, "_free_registry_path_override", str(overlay))
    config.load(str(root / "configs/free-tier/benchmark.yaml"))
    cp.clear_cache()

    subset = config.concordance_subset_models()
    assert len(subset) == 9
    plan = cs.build_plan(sorted(subset), requested_workers=9)

    assert plan.duplicates == {}
    assert set(plan.models) == subset
    cp.clear_cache()


def test_a_not_worth_model_is_skipped_with_its_reason_printed_once() -> None:
    engine = _Engine(
        {"bad": 9.0, "good": 1.0},
        worth={"bad": (False, "denylisted: buys nothing")},
    )
    plan = cs.build_plan(["bad", "good"], engine=engine)
    assert plan.skipped == {"bad": "denylisted: buys nothing"}
    assert plan.models == ["good"]
    rendered = cs.format_plan(plan)
    assert rendered.count("denylisted: buys nothing") == 1


def test_a_model_with_no_benchmark_left_is_retired_with_its_reason() -> None:
    engine = _Engine(
        {"pure": 9.0, "texty": 1.0},
        worth={"pure": (False, "text coverage complete and the model is pure-text")},
        runnable={"pure": []},
    )
    plan = cs.build_plan(["pure", "texty"], engine=engine)
    assert plan.retired == {"pure": "text coverage complete and the model is pure-text"}
    assert "pure" not in plan.models
    assert plan.models == ["texty"]


def test_build_plan_excludes_a_withdrawn_channel_via_the_priority_engine() -> None:
    """The overlay retains a withdrawn row; the priority engine must still quiesce its lane."""
    from benchmark.routing import collection_priority as cp

    engine = cp.CollectionPriority(
        channels={"gone": cp.Channel("gone", "gone")},
        covered={},
        verified_total=10,
        withdrawn=frozenset({"gone"}),
    )
    plan = cs.build_plan(["gone"], engine=engine)
    assert plan.models == []
    assert "withdrawn" in plan.skipped["gone"]


# ── priority allocation and idle-capacity fill ───────────────────────────────


def _scheduler(
    priorities: dict[str, float],
    limits: dict[str, ls.LaneLimits],
    planned: dict[str, list[ls.Cell]],
) -> cs.PriorityLaneScheduler:
    base = ls.LaneScheduler(limits=limits, reserves={lane: 1 for lane in limits})
    plan = cs.CampaignPlan(phases=[cs.ModelPhase(m, TEXT, p) for m, p in priorities.items()])
    return cs.PriorityLaneScheduler.from_plan(plan, planned, lanes=base)


def test_high_priority_is_admitted_before_low_priority() -> None:
    sched = _scheduler(
        {"high": 10.0, "low": 1.0},
        {"high": ls.LaneLimits(rpm=10, rpd=1000), "low": ls.LaneLimits(rpm=10, rpd=1000)},
        {TEXT: [("c1", "low", "default"), ("c2", "high", "default")]},
    )
    pending = [("c1", "low", "default"), ("c2", "high", "default")]
    assert sched.select_next(pending, now=0.0) == ("c2", "high", "default")


def test_a_blocked_high_lane_fills_with_the_next_priority_and_recovers() -> None:
    sched = _scheduler(
        {"high": 10.0, "low": 1.0},
        {"high": ls.LaneLimits(rpm=0, rpd=1000), "low": ls.LaneLimits(rpm=10, rpd=1000)},
        {TEXT: [("c1", "high", "default"), ("c2", "low", "default")]},
    )
    pending = [("c1", "high", "default"), ("c2", "low", "default")]
    # The high lane is rate-limited: its idle capacity is handed to the low lane.
    assert sched.select_next(pending, now=0.0) == ("c2", "low", "default")
    # The high lane recovers: it takes the next pull back.
    sched.limits["high"] = ls.LaneLimits(rpm=10, rpd=1000)
    assert sched.select_next(pending, now=1.0) == ("c1", "high", "default")


def test_a_disabled_high_lane_is_skipped_for_the_next_priority_lane() -> None:
    # A preflight-disabled lane is persisted on the scheduler (disabled_reason); the priority
    # pull loop must never admit it, and must hand its capacity to the next model.
    sched = _scheduler(
        {"high": 10.0, "low": 1.0},
        {"high": ls.LaneLimits(rpm=10, rpd=1000), "low": ls.LaneLimits(rpm=10, rpd=1000)},
        {TEXT: [("c1", "high", "default"), ("c2", "low", "default")]},
    )
    sched.disable("high", "unusable: dead key")
    pending = [("c1", "high", "default"), ("c2", "low", "default")]
    assert sched.select_next(pending, now=0.0) == ("c2", "low", "default")


def test_a_quarantined_high_lane_fills_with_low_and_recovers_after_backoff() -> None:
    sched = _scheduler(
        {"high": 10.0, "low": 1.0},
        {"high": ls.LaneLimits(rpm=10, rpd=1000), "low": ls.LaneLimits(rpm=10, rpd=1000)},
        {TEXT: [("c1", "high", "default"), ("c2", "low", "default")]},
    )
    sched.record_rate_limit("high", now=0.0, rng=__import__("random").Random(0))
    pending = [("c1", "high", "default"), ("c2", "low", "default")]
    assert sched.select_next(pending, now=0.0) == ("c2", "low", "default")
    until = sched.lane_state("high").quarantined_until
    assert until is not None
    assert sched.select_next(pending, now=until) == ("c1", "high", "default")


# ── per-model phase progression ──────────────────────────────────────────────


def test_multimodal_is_gated_until_the_models_text_is_done() -> None:
    vision_text = ("vt", "vision", "default")
    vision_mm = ("vm", "vision", "default")
    other_text = ("tt", "texty", "default")
    base = ls.LaneScheduler(
        limits={
            "vision": ls.LaneLimits(rpm=10, rpd=1000),
            "texty": ls.LaneLimits(rpm=10, rpd=1000),
        },
        reserves={"vision": 1, "texty": 1},
    )
    plan = cs.CampaignPlan(
        phases=[
            cs.ModelPhase("vision", TEXT, 9.0),
            cs.ModelPhase("vision", MM, 9.0),
            cs.ModelPhase("texty", TEXT, 5.0),
        ]
    )
    sched = cs.PriorityLaneScheduler.from_plan(
        plan, {TEXT: [vision_text, other_text], MM: [vision_mm]}, lanes=base
    )
    # vision outranks texty, but its multimodal cell is gated while its text is pending.
    assert sched.select_next([vision_mm, other_text], now=0.0) == other_text
    assert sched.admit(vision_text, now=0.0)
    assert sched.text_remaining("vision") == 0
    # text done: vision proceeds to multimodal even though texty still has text pending.
    assert sched.select_next([vision_mm, other_text], now=1.0) == vision_mm


def test_a_text_complete_model_schedules_multimodal_ahead_of_a_lower_text() -> None:
    engine = _Engine(
        {"vision": 9.0, "texty": 5.0},
        runnable={"vision": [MM], "texty": [TEXT]},
    )
    plan = cs.build_plan(["texty", "vision"], engine=engine)
    assert plan.phases[0] == cs.ModelPhase("vision", MM, 9.0)
    assert plan.models == ["vision", "texty"]


# ── execution reuse: run_cells drives the runner's pull loop priority-first ───


def _row(cell: ls.Cell) -> dict[str, Any]:
    return {
        "challenge_id": cell[0],
        "model": cell[1],
        "reasoning": cell[2],
        "pass": True,
        "real_cost": 0.0,
        "calls": 1,
        "in_tok": 1,
        "out_tok": 1,
    }


def test_run_cells_executes_and_checkpoints_in_priority_order(monkeypatch) -> None:
    from benchmark.runner import run_matrix

    executed: list[ls.Cell] = []
    checkpointed: list[ls.Cell] = []

    def _fake_run(cell: ls.Cell, _ctx: object) -> dict[str, Any]:
        executed.append(cell)
        return _row(cell)

    monkeypatch.setattr(run_matrix, "_run_one_cell", _fake_run)

    base = ls.LaneScheduler(
        limits={
            "high": ls.LaneLimits(rpm=10, rpd=1000),
            "low": ls.LaneLimits(rpm=10, rpd=1000),
        },
        reserves={"high": 1, "low": 1},
    )
    run = cs.build_campaign(
        ["high", "low"],
        [("c-low", "low", "default"), ("c-high", "high", "default")],
        benchmark=TEXT,
        lanes=base,
        engine=_Engine({"high": 10.0, "low": 1.0}),
    )
    assert isinstance(run.scheduler, cs.PriorityLaneScheduler)
    ctx = run_matrix._LiveContext.__new__(run_matrix._LiveContext)
    rows = cs.run_cells(
        run,
        ctx,
        checkpoint=lambda row: checkpointed.append(
            (row["challenge_id"], row["model"], row["reasoning"])
        ),
    )
    assert [cell[1] for cell in executed] == ["high", "low"]
    assert checkpointed == [("c-high", "high", "default"), ("c-low", "low", "default")]
    assert len(rows) == 2
    assert base.served == {"high": 1, "low": 1}


def test_build_campaign_drops_cells_for_excluded_models() -> None:
    """Only the plan's runnable models reach the executor: a not-worth model's cells vanish."""
    engine = _Engine(
        {"keep": 5.0, "bad": 9.0},
        worth={"bad": (False, "denylisted: buys nothing")},
    )
    base = ls.LaneScheduler(limits={}, reserves={})
    run = cs.build_campaign(
        ["keep", "bad"],
        [("c1", "keep", "default"), ("c2", "bad", "default")],
        benchmark=TEXT,
        lanes=base,
        engine=engine,
    )
    assert run.plan.models == ["keep"]
    assert run.plan.skipped == {"bad": "denylisted: buys nothing"}
    assert run.cells == {TEXT: [("c1", "keep", "default")]}


def test_build_campaign_uses_the_effective_worker_cap() -> None:
    base = ls.LaneScheduler(limits={}, reserves={})
    run = cs.build_campaign(
        ["a", "b", "c"],
        [("c1", "a", "default")],
        benchmark=TEXT,
        lanes=base,
        requested_workers=8,
        engine=_Engine({}),
    )
    assert run.plan.requested_workers == 8
    assert run.plan.effective_workers == 3


# ── resume: same plan, only MISSING cells collected ──────────────────────────


def test_resume_reuses_classify_cells_and_collects_only_missing() -> None:
    from benchmark.runner import run_matrix

    tasks = ["c1", "c2"]
    models = ["m"]
    hashes = {"c1": "h1", "c2": "h2"}
    versions = {"m": "v1"}
    selected = {("c1", "m"): ["default"], ("c2", "m"): ["default"]}

    first = run_matrix.classify_cells(tasks, models, {}, hashes, versions, None, selected, {})
    assert set(first.to_run) == {("c1", "m", "default"), ("c2", "m", "default")}

    present = {
        "calls": "1",
        "real_cost": "0.0",
        "version_hash": "h1",
        "model_version": "v1",
    }
    cache = {"c1": {"m": {"default": present}}}
    second = run_matrix.classify_cells(tasks, models, cache, hashes, versions, None, selected, {})
    assert second.to_run == [("c2", "m", "default")]


def test_build_plan_recomputed_from_the_same_corpus_is_identical() -> None:
    engine = _Engine({"a": 4.0, "b": 2.0}, runnable={"b": [TEXT, MM]})
    first = cs.build_plan(["a", "b"], engine=engine)
    second = cs.build_plan(["a", "b"], engine=engine)
    assert first == second
    # b outranks nothing (a=4 > b=2) but its text phase still precedes its multimodal one.
    assert [phase.benchmark for phase in first.phases if phase.model == "b"] == [TEXT, MM]


# ── live-path integration: _run_full builds and uses the priority campaign ─────


def _run_full_args(**over: object):
    import argparse

    base: dict[str, object] = {
        "live": True,
        "extra_models": "high,low,bad",
        "check_images": False,
        "step_limit": 10,
        "cells": None,
        "timeout": 1,
        "workers": 4,
        "max_cost": None,
        "max_cost_overshoot": 0.0,
        "max_start_failures": 5,
        "max_consecutive_failures": 5,
        "require_zero_cost": False,
        "no_summary": True,
        "no_plots": True,
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_live_free_path_builds_and_uses_a_priority_campaign(monkeypatch) -> None:
    """The live $0 collector must use the priority subclass, exclusion, cap and MISSING set.

    This is the integration guard the audit's coverage gap called for: before this, the live
    path built a plain ``LaneScheduler`` and referenced neither ``collection_priority`` nor
    ``build_plan``, so the priority order, domination stop, worker cap and phase gate were all
    inert. Every effect here is monkeypatched — no model call, no network.
    """
    from pathlib import Path

    from benchmark import config
    from benchmark.routing import collection_priority as cp
    from benchmark.routing import integrity
    from benchmark.runner import run_matrix, swebench_specs

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        config,
        "lanes_config",
        lambda: {"unknown_limits": {"rpm": 10, "rpd": 100}, "stall_timeout_s": 900, "limits": {}},
    )
    monkeypatch.setattr(ls, "load_lane_state", lambda *a, **k: {})
    monkeypatch.setattr(ls, "save_lane_state", lambda *a, **k: None)
    monkeypatch.setattr(config, "challenges_path", lambda: Path("challenges.json"))
    monkeypatch.setattr(config, "load_matrix", lambda *a, **k: {})
    monkeypatch.setattr(config, "enabled_models", lambda: [])
    monkeypatch.setattr(config, "register_collection_models", lambda names: None)
    monkeypatch.setattr(config, "load_results", lambda: {})
    monkeypatch.setattr(config, "sample_tasks", lambda tasks, seed=42: ["c1", "c2"])
    monkeypatch.setattr(config, "concordance_subset_models", lambda: set())
    monkeypatch.setattr(config, "models_missing_cache", lambda models: [])
    monkeypatch.setattr(config, "results_csv_path", lambda: Path("results_free.csv"))
    monkeypatch.setattr(config, "free_registry", lambda: {})
    monkeypatch.setattr(swebench_specs, "manifest_source", lambda: "swebench_verified")
    monkeypatch.setattr(swebench_specs, "spec_module_for", lambda source: object())
    monkeypatch.setattr(integrity, "all_hashes", lambda source: {"c1": "h", "c2": "h"})
    monkeypatch.setattr(integrity, "model_versions", lambda: {"high": "h", "low": "l", "bad": "b"})
    monkeypatch.setattr(integrity, "scaffold_prompt_hash", lambda: "")
    monkeypatch.setattr(integrity, "sampling_hash_map", lambda models: {})
    monkeypatch.setattr(run_matrix, "_apply_multimodal_gate", lambda source, models: (models, {}))
    monkeypatch.setattr(run_matrix, "_arm_context", lambda tasks, models: ({}, {}))
    monkeypatch.setattr(run_matrix, "_extra_models", lambda raw: ["high", "low", "bad"])
    # The priority engine: high outranks low; "bad" is not worth collecting.
    monkeypatch.setattr(
        cp, "priority", lambda m: {"high": 10.0, "low": 1.0, "bad": 9.0}.get(m, 0.1)
    )
    monkeypatch.setattr(
        cp,
        "worth_collecting",
        lambda m: (False, "denylisted: buys nothing") if m == "bad" else (True, "keep"),
    )
    monkeypatch.setattr(cp, "runnable_benchmarks", lambda m: [cp.TEXT_BENCHMARK])
    monkeypatch.setattr(cp, "duplicate_of", lambda m: None)
    monkeypatch.setattr(
        run_matrix,
        "classify_cells",
        lambda *a, **k: run_matrix.CellStatus(
            missing=[("c1", "high", "default"), ("c2", "low", "default")],
            stale=[("c2", "bad", "default")],
        ),
    )
    monkeypatch.setattr(run_matrix, "_print_status", lambda *a, **k: None)
    monkeypatch.setattr(run_matrix, "_has_keys", lambda: True)
    monkeypatch.setattr(run_matrix, "preflight_refuses", lambda *a, **k: False)
    monkeypatch.setattr(run_matrix, "_run_and_merge", lambda *a, **k: (captured.update(k), 1)[1])
    monkeypatch.setattr(run_matrix, "_report_coverage", lambda *a, **k: None)
    monkeypatch.setattr(run_matrix, "refresh_summary", lambda *a, **k: None)
    monkeypatch.setattr(run_matrix, "regenerate_plots", lambda: None)

    assert run_matrix._run_full(_run_full_args()) == 0
    campaign = captured.get("campaign")
    assert isinstance(campaign, cs.CampaignRun)
    assert isinstance(campaign.scheduler, cs.PriorityLaneScheduler)
    # Priority order with the not-worth model excluded by name.
    assert campaign.plan.models == ["high", "low"]
    assert campaign.plan.skipped == {"bad": "denylisted: buys nothing"}
    # Worker cap: min(requested=4, runnable models=2).
    assert campaign.plan.effective_workers == 2
    # Only the cells of runnable models are collected; the stale "bad" cell is dropped.
    assert campaign.cells[cs.TEXT_BENCHMARK] == [
        ("c1", "high", "default"),
        ("c2", "low", "default"),
    ]

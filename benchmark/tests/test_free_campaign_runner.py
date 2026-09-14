"""Long-running free-campaign driver: rescan, queue, lane gate, retry, phase, resume, stop.

Every case drives the driver through its injected seams (refresh, planner, executor, clock)
so no network, no container and no model call runs. The reuses — classify_cells, build_plan,
lane_scheduler — are exercised through their real implementations where they are pure.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from benchmark.runner import campaign_scheduler as cs
from benchmark.runner import free_campaign_runner as fcr
from benchmark.runner import lane_scheduler as ls

TEXT = fcr.TEXT_BENCHMARK
MM = fcr.MULTIMODAL_BENCHMARK


class Clock:
    """A deterministic monotonic clock; ``sleep`` advances it instead of blocking."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _discovery(lanes: list[tuple[str, str]], *, scan: str = "2026-09-11") -> fcr.Discovery:
    return fcr.Discovery(
        scan_as_of=scan,
        lanes={
            channel: fcr.LaneSpec(
                channel=channel,
                provider=provider,
                version=f"id/{channel}",
                priority=float(len(lanes) - index),
            )
            for index, (channel, provider) in enumerate(lanes)
        },
    )


def _entry(
    cell: ls.Cell,
    provider: str,
    *,
    version: str = "v",
    benchmark: str = TEXT,
    priority: float = 0.0,
) -> fcr.QueueEntry:
    return fcr.QueueEntry(
        cell=cell,
        provider=provider,
        version=version,
        benchmark=benchmark,
        priority=priority,
    )


def _row(cell: ls.Cell, *, real_cost: str = "0.0") -> dict[str, Any]:
    return {
        "challenge_id": cell[0],
        "model": cell[1],
        "reasoning": cell[2],
        "pass": True,
        "real_cost": real_cost,
        "calls": 1,
        "in_tok": 1,
        "out_tok": 1,
    }


def _runner(
    *,
    clock: Clock,
    planner,
    execute,
    refresh=None,
    discovery: fcr.Discovery | None = None,
    retry_cap: int = 2,
    workers: int = 1,
) -> fcr.FreeCampaignRunner:
    disc = discovery or _discovery([("a", "pa"), ("b", "pb")])
    return fcr.FreeCampaignRunner(
        workers=workers,
        rescan_hours=12.0,
        retry_cap=retry_cap,
        poll_seconds=1.0,
        refresh_fn=refresh or (lambda write: disc),
        planner_fn=planner,
        execute_fn=execute,
        limits_fn=lambda d: {channel: ls.LaneLimits(rpm=1000, rpd=100000) for channel in d.lanes},
        lane_state={},
        persist=False,
        single_instance=False,
        clock=clock,
        sleep=clock.sleep,
    )


class _Engine:
    """A scripted collection_priority surface (no corpus, no YAML, no model call)."""

    def __init__(
        self,
        importance: dict[str, float] | None = None,
        *,
        duplicate: dict[str, str] | None = None,
        runnable: dict[str, list[str]] | None = None,
        worth: dict[str, tuple[bool, str]] | None = None,
    ) -> None:
        self.importance = importance or {}
        self.duplicate = duplicate or {}
        self.runnable = runnable or {}
        self.worth = worth or {}

    def priority(self, identity: str) -> float:
        return self.importance.get(identity, 1.0)

    def duplicate_of(self, identity: str) -> str | None:
        return self.duplicate.get(identity)

    def runnable_benchmarks(self, identity: str) -> list[str]:
        return self.runnable.get(identity, [TEXT])

    def worth_collecting(self, identity: str) -> tuple[bool, str]:
        return self.worth.get(identity, (True, "keep"))


# ── 1. rescan loop ────────────────────────────────────────────────────────────


def test_rescan_schedules_the_next_pass_and_repeats_when_due() -> None:
    clock = Clock()
    scans: list[float] = []

    def refresh(write: bool) -> fcr.Discovery:
        scans.append(clock.now)
        return _discovery([("a", "pa")], scan=f"scan-{len(scans)}")

    runner = _runner(
        clock=clock,
        planner=lambda d: [],
        execute=lambda e: fcr.ExecOutcome(),
        refresh=refresh,
    )
    assert runner.rescan() is True
    assert runner.summary.rescans == 1
    assert runner.summary.scan_as_of == "scan-1"
    assert runner._next_scan_at == clock.now + 12 * 3600
    assert runner._due(clock.now) is False

    clock.advance(12 * 3600)
    assert runner._due(clock.now) is True
    assert runner.run_once() is False  # rescans, no work queued
    assert len(scans) == 2
    assert runner.summary.rescans == 2
    assert runner.summary.scan_as_of == "scan-2"


def test_run_rescans_on_each_interval() -> None:
    clock = Clock()
    scans: list[float] = []

    def refresh(write: bool) -> fcr.Discovery:
        scans.append(clock.now)
        return fcr.Discovery(scan_as_of=f"scan-{len(scans)}")

    runner = _runner(
        clock=clock, planner=lambda d: [], execute=lambda e: fcr.ExecOutcome(), refresh=refresh
    )
    summary = runner.run(max_iterations=3)
    assert len(scans) == 3
    assert summary.rescans == 3


# ── 2. billed lane gate ────────────────────────────────────────────────────────


def test_billed_lane_is_disabled_and_its_queued_cells_are_dropped() -> None:
    clock = Clock()
    seen: list[str] = []

    def execute(entry: fcr.QueueEntry) -> fcr.ExecOutcome:
        seen.append(entry.lane)
        if entry.lane == "a":
            return fcr.ExecOutcome(rows=(_row(entry.cell, real_cost="0.01"),))
        return fcr.ExecOutcome(rows=(_row(entry.cell),))

    entries = [
        _entry(("c1", "a", "default"), "pa", priority=3.0),
        _entry(("c2", "a", "default"), "pa", priority=2.0),
        _entry(("c3", "b", "default"), "pb", priority=1.0),
    ]
    runner = _runner(clock=clock, planner=lambda d: list(entries), execute=execute)
    runner.rescan()
    runner.run_pass()

    assert seen == ["a", "b"]  # a's second cell never ran
    assert "a@pa" in runner.summary.lanes_disabled
    assert all(entry.lane != "a" for entry in runner.queue)
    assert runner.summary.cells_dropped == 1
    assert runner.lanes.lane_state("a").disabled_reason is not None


def test_a_malformed_real_cost_row_does_not_crash_the_loop() -> None:
    clock = Clock()
    entries = [_entry(("c1", "a", "default"), "pa")]

    def execute(entry: fcr.QueueEntry) -> fcr.ExecOutcome:
        return fcr.ExecOutcome(rows=({**_row(entry.cell), "real_cost": "not-a-number"},))

    runner = _runner(clock=clock, planner=lambda d: list(entries), execute=execute)
    runner.rescan()
    runner.run_pass()
    assert runner.summary.cells_run == 1
    assert runner.queue == []
    assert "a@pa" not in runner.summary.lanes_disabled


def test_a_poisoned_lane_is_disabled_and_dropped_not_aborted() -> None:
    clock = Clock()
    entries = [
        _entry(("c1", "a", "default"), "pa", priority=2.0),
        _entry(("c2", "a", "default"), "pa", priority=1.0),
    ]

    def execute(entry: fcr.QueueEntry) -> fcr.ExecOutcome:
        return fcr.ExecOutcome(poison="DataIntegrityError: FREE_LANE_BILLED")

    runner = _runner(clock=clock, planner=lambda d: list(entries), execute=execute)
    runner.rescan()
    runner.run_pass()
    assert "a@pa" in runner.summary.lanes_disabled
    assert runner.queue == []


def test_listing_expiry_refuses_an_expired_lane_with_the_named_reason(monkeypatch) -> None:
    """The listing's expiration_date reaches LaneLimits.expires_at and the lane is refused.

    Regression: ``expires_at`` existed on LaneLimits and had an ``_EXPIRED`` limitation, but
    nothing ever populated it, so a lapsed time-boxed free promo stayed schedulable forever.
    """
    from benchmark import config

    monkeypatch.setattr(config, "lane_unknown_limits", lambda: {"rpm": 10, "rpd": 100})
    monkeypatch.setattr(config, "lane_limits_config", lambda: {})
    monkeypatch.setattr(config, "lane_limits_from_registry", lambda channel, provider: {})

    clock = Clock()
    discovery = fcr.Discovery(
        lanes={
            "a": fcr.LaneSpec(
                channel="a",
                provider="pa",
                version="v",
                expires_at="1970-01-01T00:00:00+00:00",
            )
        }
    )
    runner = fcr.FreeCampaignRunner(
        workers=1,
        refresh_fn=lambda write: discovery,
        planner_fn=lambda d: [],
        execute_fn=lambda e: fcr.ExecOutcome(),
        lane_state={},
        persist=False,
        clock=clock,
        sleep=clock.sleep,
    )
    runner._apply_discovery(discovery)

    assert runner.lanes.limits["a"].expires_at == "1970-01-01T00:00:00+00:00"
    assert not runner.lanes.can_admit("a", clock.now)
    reason = runner.lanes.refusal_reason("a", clock.now)
    assert reason is not None and "expired" in reason


def test_a_deterministic_poison_cell_is_tombstoned_and_runs_once_across_passes() -> None:
    """A non-FREE_LANE_BILLED DataIntegrityError tombstones the CELL, not the lane.

    Regression: the deterministic poison row was never persisted, so every rescan re-planned
    the cell as MISSING and re-ran (and re-billed) it. Now it is quarantined by cell key.
    """
    clock = Clock()
    calls = {"n": 0}

    def execute(entry: fcr.QueueEntry) -> fcr.ExecOutcome:
        calls["n"] += 1
        return fcr.ExecOutcome(
            poison="DataIntegrityError: stop_reason='solved' with pass=False",
            poison_code="MALFORMED_NUMERIC",
        )

    entries = [_entry(("c1", "a", "default"), "pa")]
    runner = _runner(clock=clock, planner=lambda d: list(entries), execute=execute)
    runner.rescan()
    runner.run_pass()

    assert calls["n"] == 1
    assert "a@pa" not in runner.summary.lanes_disabled  # a cell fault, not a lane fault
    assert runner.summary.cells_quarantined == 1

    # A second pass re-plans the same still-MISSING cell, but the tombstone keeps it out.
    runner.rescan()
    runner.run_pass()
    assert calls["n"] == 1
    assert runner.queue == []


def test_poison_tombstones_round_trip_and_filter_a_rebuilt_queue(tmp_path) -> None:
    path = tmp_path / "free-tier" / "poisoned_cells.json"
    assert fcr.load_poisoned_cells(path) == {}
    fcr.save_poisoned_cells({"c1:a:default": "MALFORMED_NUMERIC: bad row"}, path)
    assert fcr.load_poisoned_cells(path) == {"c1:a:default": "MALFORMED_NUMERIC: bad row"}

    clock = Clock()
    entry = _entry(("c1", "a", "default"), "pa")
    runner = fcr.FreeCampaignRunner(
        workers=1,
        refresh_fn=lambda write: _discovery([("a", "pa")]),
        planner_fn=lambda d: [entry],
        execute_fn=lambda e: fcr.ExecOutcome(),
        lane_state={},
        poisoned_cells={"c1:a:default": "MALFORMED_NUMERIC"},
        persist=False,
        clock=clock,
        sleep=clock.sleep,
    )
    runner.rescan()
    assert runner.queue == []


# ── 3. retry semantics ─────────────────────────────────────────────────────────


def test_failed_cell_requeues_and_succeeds_on_retry() -> None:
    clock = Clock()
    calls = {"n": 0}

    def execute(entry: fcr.QueueEntry) -> fcr.ExecOutcome:
        calls["n"] += 1
        if calls["n"] == 1:
            return fcr.ExecOutcome()  # transient: no row
        return fcr.ExecOutcome(rows=(_row(entry.cell),))

    entries = [_entry(("c1", "a", "default"), "pa")]
    runner = _runner(clock=clock, planner=lambda d: list(entries), execute=execute)
    runner.rescan()
    runner.run_pass()

    assert calls["n"] == 1
    assert runner.summary.cells_retried == 1
    assert len(runner.queue) == 1 and runner.queue[0].attempt == 1
    assert runner._has_ready(clock.now) is False  # backed off

    clock.advance(fcr.RETRY_BACKOFF_S + 1)
    assert runner._has_ready(clock.now) is True
    runner.run_pass()
    assert calls["n"] == 2
    assert runner.summary.cells_run == 1
    assert runner.queue == []


def test_transient_cell_drops_after_the_retry_cap_without_disabling_the_lane() -> None:
    clock = Clock()
    entries = [_entry(("c1", "a", "default"), "pa")]
    runner = _runner(
        clock=clock,
        planner=lambda d: list(entries),
        execute=lambda e: fcr.ExecOutcome(),
        retry_cap=1,
    )
    runner.rescan()
    for _ in range(4):
        if runner.queue:
            clock.advance(fcr.RETRY_BACKOFF_S * 4)
            runner.run_pass()
    assert runner.queue == []
    assert runner.summary.cells_dropped == 1
    assert "a@pa" not in runner.summary.lanes_disabled


def test_rate_limited_lane_defers_its_cell_until_quarantine_expires() -> None:
    clock = Clock()
    calls = {"n": 0}

    def execute(entry: fcr.QueueEntry) -> fcr.ExecOutcome:
        calls["n"] += 1
        if calls["n"] == 1:
            runner.lanes.record_rate_limit(entry.lane, clock.now, rng=random.Random(0))
            return fcr.ExecOutcome()  # _run_scheduled_batch records, writes no row
        return fcr.ExecOutcome(rows=(_row(entry.cell),))

    entries = [_entry(("c1", "a", "default"), "pa")]
    runner = _runner(clock=clock, planner=lambda d: list(entries), execute=execute)
    runner.rescan()
    runner.run_pass()
    assert runner.summary.cells_deferred == 1
    until = runner.lanes.lane_state("a").quarantined_until
    assert until is not None and runner.queue[0].not_before == until
    clock.advance((until - clock.now) + 1)
    runner.run_pass()
    assert calls["n"] == 2
    assert runner.summary.cells_run == 1


# ── 4. identity dedupe with the concordance exemption ──────────────────────────


def test_identity_dedupe_drops_a_second_provider_unless_it_is_a_concordance_channel() -> None:
    engine = _Engine(duplicate={"chanB": "chanA"})
    plain = cs.build_plan(["chanA", "chanB"], engine=engine, exempt_duplicates=set())
    assert plain.models == ["chanA"]
    assert plain.duplicates == {"chanB": "chanA"}

    exempt = cs.build_plan(["chanA", "chanB"], engine=engine, exempt_duplicates={"chanB"})
    assert exempt.models == ["chanA", "chanB"]
    assert exempt.duplicates == {}


# ── 5. per-model text -> multimodal phase ──────────────────────────────────────


def test_phase_gate_runs_text_before_unlocking_multimodal() -> None:
    text_cell = ("c1", "vision", "default")
    mm_cell = ("c2", "vision", "default")

    text_plan = cs.build_plan(["vision"], engine=_Engine(runnable={"vision": [TEXT]}))
    assert fcr.phase_entries(text_plan, [text_cell], TEXT) == [text_cell]
    assert fcr.phase_entries(text_plan, [mm_cell], MM) == []

    unlocked = cs.build_plan(["vision"], engine=_Engine(runnable={"vision": [MM]}))
    assert fcr.phase_entries(unlocked, [mm_cell], MM) == [mm_cell]
    assert fcr.phase_entries(unlocked, [text_cell], TEXT) == []


def test_a_pure_text_model_with_no_benchmark_left_is_retired() -> None:
    engine = _Engine(runnable={"pure": []}, worth={"pure": (False, "text complete and pure-text")})
    plan = cs.build_plan(["pure"], engine=engine)
    assert plan.models == []
    assert "pure" in plan.retired


# ── 6. SIGTERM / resume ────────────────────────────────────────────────────────


def test_stop_requested_halts_before_any_cell_runs() -> None:
    clock = Clock()
    ran: list[ls.Cell] = []

    def execute(entry: fcr.QueueEntry) -> fcr.ExecOutcome:
        ran.append(entry.cell)
        return fcr.ExecOutcome(rows=(_row(entry.cell),))

    runner = _runner(
        clock=clock,
        planner=lambda d: [_entry(("c1", "a", "default"), "pa")],
        execute=execute,
    )
    runner.rescan()
    runner.request_stop()
    runner.run_pass()
    assert ran == []
    assert runner.run(max_iterations=1).stopped is True


def test_resume_queues_only_the_missing_cells_from_the_corpus() -> None:
    from benchmark.runner import run_matrix

    clock = Clock()
    tasks = ["c1", "c2"]
    cell_models = ["a"]
    hashes = {"c1": "h1", "c2": "h2"}
    versions = {"a": "va"}
    selected = {("c1", "a"): ["default"], ("c2", "a"): ["default"]}
    present = {"calls": "1", "real_cost": "0.0", "version_hash": "h1", "model_version": "va"}
    cache = {"c1": {"a": {"default": present}}}

    def planner(discovery: fcr.Discovery) -> list[fcr.QueueEntry]:
        status = run_matrix.classify_cells(
            tasks, cell_models, cache, hashes, versions, None, selected, {}
        )
        return [_entry(cell, "pa") for cell in status.to_run]

    runner = _runner(
        clock=clock,
        planner=planner,
        execute=lambda e: fcr.ExecOutcome(),
        discovery=_discovery([("a", "pa")]),
    )
    runner.rescan()
    # c1 is present, so only c2 — the MISSING cell — is queued: a mid-cell kill re-runs it.
    assert [entry.cell for entry in runner.queue] == [("c2", "a", "default")]


# ── 7. lane exception isolation ────────────────────────────────────────────────


def test_a_lane_exception_requeues_and_does_not_kill_the_loop() -> None:
    clock = Clock()
    seen: list[str] = []

    def execute(entry: fcr.QueueEntry) -> fcr.ExecOutcome:
        seen.append(entry.lane)
        if entry.lane == "a" and seen.count("a") == 1:
            raise RuntimeError("docker daemon is not running")
        return fcr.ExecOutcome(rows=(_row(entry.cell),))

    entries = [
        _entry(("c1", "a", "default"), "pa", priority=2.0),
        _entry(("c2", "b", "default"), "pb", priority=1.0),
    ]
    runner = _runner(clock=clock, planner=lambda d: list(entries), execute=execute)
    runner.rescan()
    runner.run_pass()

    assert "b" in seen  # the fleet continued past a's exception
    assert runner.summary.cells_run == 1
    assert any(entry.lane == "a" and entry.attempt == 1 for entry in runner.queue)
    assert runner.summary.stopped is False
    assert "a@pa" not in runner.summary.lanes_disabled  # transient, not disabled


# ── 8. empty / unreachable scans ───────────────────────────────────────────────


def test_empty_scan_does_not_spin_or_crash() -> None:
    clock = Clock()
    runner = _runner(
        clock=clock,
        planner=lambda d: [],
        execute=lambda e: fcr.ExecOutcome(),
        discovery=fcr.Discovery(scan_as_of="empty"),
    )
    summary = runner.run(max_iterations=3)
    assert summary.rescans == 3
    assert summary.cells_run == 0
    assert summary.stopped is False


def test_scan_failure_backs_off_without_crashing() -> None:
    clock = Clock()

    def refresh(write: bool) -> fcr.Discovery:
        raise OSError("network unreachable")

    runner = _runner(
        clock=clock, planner=lambda d: [], execute=lambda e: fcr.ExecOutcome(), refresh=refresh
    )
    assert runner.rescan() is False
    assert runner._next_scan_at > clock.now
    summary = runner.run(max_iterations=2)
    assert summary.rescans == 0
    assert summary.stopped is False


# ── conversion seam + provider carry-through ───────────────────────────────────


def test_discovery_from_refresh_carries_provider_and_free_marker() -> None:
    class _Lane:
        name = "chanA"
        provider = "groq"
        version = "id/x"
        priority = 3.5

    class _Result:
        snapshot = {"scan_as_of": "2026-09-11"}
        runnable = [_Lane()]
        excluded = {"other": "no-free-lane"}

    discovery = fcr.discovery_from_refresh(_Result(), access={"groq": None})
    assert discovery.scan_as_of == "2026-09-11"
    assert discovery.lanes["chanA"].provider == "groq"
    assert discovery.lanes["chanA"].free is True
    assert discovery.excluded == {"other": "no-free-lane"}


def test_rescan_reenables_a_lane_that_is_free_again() -> None:
    clock = Clock()
    runner = _runner(
        clock=clock,
        planner=lambda d: [],
        execute=lambda e: fcr.ExecOutcome(),
        discovery=_discovery([("a", "pa")]),
    )
    runner.rescan()
    runner.lanes.disable("a", "billed yesterday")
    runner._disabled[("a", "pa")] = "billed yesterday"
    runner.rescan()
    assert runner.lanes.lane_state("a").disabled_reason is None
    assert ("a", "pa") not in runner._disabled


def test_rescan_keeps_a_billed_lane_disabled() -> None:
    """A FREE_LANE_BILLED disable is a money interlock, not a transient state.

    Regression: ``_apply_discovery`` re-enabled ANY lane whose scan still lists it as free, so
    a lane proven to bill was re-enabled next rescan and re-billed before the interlock fired.
    """
    clock = Clock()
    runner = _runner(
        clock=clock,
        planner=lambda d: [],
        execute=lambda e: fcr.ExecOutcome(),
        discovery=_discovery([("a", "pa")]),
    )
    runner.rescan()
    billed = f"{fcr._free_lane_billed_code()}: a returned real_cost>0 while admitted free"
    runner.lanes.disable("a", billed)
    runner._disabled[("a", "pa")] = billed
    runner.rescan()
    assert runner.lanes.lane_state("a").disabled_reason == billed
    assert ("a", "pa") in runner._disabled


# ── 9. autonomy interlocks: $0 breaker, crash-safe persistence, single instance ─────────


def test_runtime_build_arms_the_process_wide_cost_breaker(monkeypatch) -> None:
    """The collector arms the layer-3 $0 breaker; it was only armed by run_matrix's own CLI.

    A SIGKILL cannot be caught, so the crash-safe path is the layer-3 interlock, and it must be
    armed before any cell can call a provider — not only by a CLI the collector never invokes.
    """
    from benchmark.routing import integrity
    from benchmark.runner import run_matrix, swebench_specs

    armed: list[int] = []
    monkeypatch.setattr(fcr, "_arm_cost_breaker", lambda: armed.append(1))
    monkeypatch.setattr(swebench_specs, "manifest_source", lambda: "verified")
    monkeypatch.setattr(integrity, "all_hashes", lambda source: {"c1": "h1"})
    monkeypatch.setattr(integrity, "model_versions", lambda: {"a": "va"})
    monkeypatch.setattr(run_matrix, "_arm_context", lambda tasks, models: ({}, {}))
    monkeypatch.setattr(run_matrix, "_is_free_lane", lambda model: True)
    monkeypatch.setattr(run_matrix, "_live_context", lambda *a, **k: object())

    clock = Clock()
    runner = _runner(clock=clock, planner=lambda d: [], execute=lambda e: fcr.ExecOutcome())
    runner._discovery = _discovery([("a", "pa")])
    runner._ensure_runtime()
    assert armed == [1]
    assert runner._runtime_ready is True


def test_disabling_a_billed_lane_persists_the_interlock_immediately(tmp_path) -> None:
    """The money interlock is flushed at the moment it is set, not at a clean shutdown."""
    clock = Clock()
    path = tmp_path / "free-tier" / "lane_state.json"
    runner = fcr.FreeCampaignRunner(
        workers=1,
        refresh_fn=lambda write: _discovery([("a", "pa")]),
        planner_fn=lambda d: [],
        execute_fn=lambda e: fcr.ExecOutcome(),
        lane_state={},
        lane_state_path=path,
        persist=True,
        single_instance=False,
        clock=clock,
        sleep=clock.sleep,
    )
    runner.rescan()
    runner._disable_lane(
        _entry(("c1", "a", "default"), "pa"),
        f"{fcr._free_lane_billed_code()}: a returned real_cost>0 while admitted free",
    )
    assert path.exists()
    reason = ls.load_lane_state(path)["a"].disabled_reason
    assert reason is not None and fcr._free_lane_billed_code() in reason


def test_a_restart_keeps_a_billed_lane_disabled(tmp_path) -> None:
    """Crash-resume: the persisted interlock survives a fresh process rescanning the lane free.

    Regression: lane state was saved only at a clean exit, so a SIGKILL lost the disable and
    the restart re-enabled (and re-billed) a still-advertised free lane after the 12h rescan.
    """
    clock = Clock()
    path = tmp_path / "free-tier" / "lane_state.json"

    def build(**over: Any) -> fcr.FreeCampaignRunner:
        return fcr.FreeCampaignRunner(
            workers=1,
            refresh_fn=lambda write: _discovery([("a", "pa")]),
            planner_fn=lambda d: [],
            execute_fn=lambda e: fcr.ExecOutcome(),
            lane_state_path=path,
            persist=True,
            single_instance=False,
            clock=clock,
            sleep=clock.sleep,
            **over,
        )

    first = build(lane_state={})
    first.rescan()
    first._disable_lane(
        _entry(("c1", "a", "default"), "pa"),
        f"{fcr._free_lane_billed_code()}: a returned real_cost>0 while admitted free",
    )

    # A fresh process loads the persisted state; the lane is still admitted free by the scan,
    # but the billing interlock must keep it disabled.
    second = build(lane_state=None)
    second.rescan()
    assert second.lanes.lane_state("a").disabled_reason is not None
    assert fcr._is_billing_disable(second.lanes.lane_state("a").disabled_reason)


def test_a_second_collector_refuses_the_singleton_lock(tmp_path) -> None:
    """One collector at a time: a second process must not share the corpus and clobber it."""
    lock = tmp_path / "collector.lock"

    def build() -> fcr.FreeCampaignRunner:
        return fcr.FreeCampaignRunner(
            workers=1,
            refresh_fn=lambda write: _discovery([("a", "pa")]),
            planner_fn=lambda d: [],
            execute_fn=lambda e: fcr.ExecOutcome(),
            lane_state={},
            persist=False,
            single_instance=True,
            lock_path=lock,
            clock=Clock(),
            sleep=Clock().sleep,
        )

    first = build()
    second = build()
    assert first._acquire_singleton() is True
    try:
        assert second._acquire_singleton() is False
    finally:
        first._release_singleton()
    assert second._acquire_singleton() is True
    second._release_singleton()


def test_run_refuses_to_start_when_another_collector_holds_the_lock(tmp_path) -> None:
    lock = tmp_path / "collector.lock"
    clock = Clock()
    holder = fcr.FreeCampaignRunner(
        workers=1,
        refresh_fn=lambda write: fcr.Discovery(),
        planner_fn=lambda d: [],
        execute_fn=lambda e: fcr.ExecOutcome(),
        lane_state={},
        persist=False,
        single_instance=True,
        lock_path=lock,
        clock=clock,
        sleep=clock.sleep,
    )
    assert holder._acquire_singleton() is True
    try:
        other = fcr.FreeCampaignRunner(
            workers=1,
            refresh_fn=lambda write: fcr.Discovery(),
            planner_fn=lambda d: [],
            execute_fn=lambda e: fcr.ExecOutcome(),
            lane_state={},
            persist=False,
            single_instance=True,
            lock_path=lock,
            clock=clock,
            sleep=clock.sleep,
        )
        try:
            other.run(max_iterations=1)
        except fcr.CollectorAlreadyRunningError:
            pass
        else:  # pragma: no cover - the lock must refuse
            raise AssertionError("a second collector was allowed to start")
    finally:
        holder._release_singleton()


def test_zero_rescan_hours_is_clamped_so_discovery_cannot_busy_scan() -> None:
    """`--rescan-hours 0` must not make every loop iteration a full provider rescan."""
    clock = Clock()
    runner = fcr.FreeCampaignRunner(
        workers=1,
        rescan_hours=0.0,
        poll_seconds=1.0,
        refresh_fn=lambda write: fcr.Discovery(scan_as_of="empty"),
        planner_fn=lambda d: [],
        execute_fn=lambda e: fcr.ExecOutcome(),
        lane_state={},
        persist=False,
        single_instance=False,
        clock=clock,
        sleep=clock.sleep,
    )
    runner.rescan()
    assert runner._next_scan_at - clock.now >= fcr.MIN_RESCAN_S


def test_structurally_refused_cells_are_dropped_not_polled_forever() -> None:
    """A lane that can never admit (request > TPM) must leave the queue idle, not spin.

    Regression: such cells stayed queued with ``not_before == 0``, so ``_has_ready`` was
    permanently False while the idle loop woke every fraction of the poll interval.
    """
    clock = Clock()
    entry = _entry(("c1", "a", "default"), "pa")
    runner = fcr.FreeCampaignRunner(
        workers=1,
        rescan_hours=12.0,
        poll_seconds=30.0,
        refresh_fn=lambda write: _discovery([("a", "pa")]),
        planner_fn=lambda d: [entry],
        execute_fn=lambda e: fcr.ExecOutcome(),
        limits_fn=lambda d: {"a": ls.LaneLimits(rpm=10, rpd=1000, tpm=10, max_request_tokens=1000)},
        lane_state={},
        persist=False,
        single_instance=False,
        clock=clock,
        sleep=clock.sleep,
    )
    runner.rescan()
    assert runner.queue == []
    # With an empty queue the loop sleeps to the next rescan, not a short poll.
    assert runner._next_delay() >= 12 * 3600


def test_repeated_scan_failures_abort_instead_of_retrying_forever() -> None:
    """A deterministic scan failure must stop loudly, not back off forever unattended."""
    clock = Clock()

    def refresh(write: bool) -> fcr.Discovery:
        raise ValueError("malformed overlay: mapping values are not allowed here")

    runner = fcr.FreeCampaignRunner(
        workers=1,
        rescan_hours=12.0,
        poll_seconds=1.0,
        scan_failure_cap=3,
        refresh_fn=refresh,
        planner_fn=lambda d: [],
        execute_fn=lambda e: fcr.ExecOutcome(),
        lane_state={},
        persist=False,
        single_instance=False,
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(fcr.ScanFailureAbortError):
        runner.run()
    assert runner.summary.rescans == 0


def test_a_transient_scan_failure_resets_after_a_success() -> None:
    """The consecutive-failure counter must reset on a good scan, not accumulate across days."""
    clock = Clock()
    outcomes: list[Exception | fcr.Discovery] = [OSError("blip"), fcr.Discovery(scan_as_of="ok")]
    calls = {"n": 0}

    def refresh(write: bool) -> fcr.Discovery:
        calls["n"] += 1
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    runner = fcr.FreeCampaignRunner(
        workers=1,
        rescan_hours=12.0,
        scan_failure_cap=2,
        refresh_fn=refresh,
        planner_fn=lambda d: [],
        execute_fn=lambda e: fcr.ExecOutcome(),
        lane_state={},
        persist=False,
        single_instance=False,
        clock=clock,
        sleep=clock.sleep,
    )
    assert runner.rescan() is False
    assert runner._scan_failures == 1
    assert runner.rescan() is True
    assert runner._scan_failures == 0
    assert calls["n"] == 2

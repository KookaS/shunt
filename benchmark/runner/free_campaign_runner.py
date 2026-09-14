"""Long-running free-tier collection driver: rescan, queue, per-lane gate, crash-safe resume.

The single process an operator leaves running for days. It sits above the machinery that
already exists and adds only orchestration:

* :func:`default_refresh` wraps ``refresh_free_campaign.refresh`` — the discovery half
  (scan every provider, admit the free + schedulable set). The driver re-runs it at start and
  every ``--rescan-hours`` and rebuilds the work set, logging each ``scan_as_of``.
* :func:`phase_entries` + ``campaign_scheduler.build_plan`` reuse ``campaign_scheduler`` for the
  priority order, the retire/duplicate/skip exclusions and the per-model text-before-multimodal
  gate; ``build_plan`` itself owns the concordance-subset dedupe exemption.
* :class:`FreeCampaignRunner` owns the shared cell queue (``workers`` pull concurrently), the
  ``(model, provider)`` free gate, retry/defer/disable policy, and stop handling. A completed
  cell is persisted through ``run_matrix._run_scheduled_batch`` + ``merge_rows``; there is no
  scheduler ledger — resume is the existing ``classify_cells`` MISSING/STALE set.

Usage: ``uv run python -m benchmark.runner.free_campaign_runner --config ... --workers 4``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from benchmark import corpus_lock
from benchmark.runner import campaign_scheduler as cs
from benchmark.runner import lane_scheduler as ls

logger = logging.getLogger(__name__)

# Deterministically poisonous cells (a produced row that fails the write-time data-integrity
# wall with a code OTHER than FREE_LANE_BILLED) are tombstoned here: once observed, the cell is
# never queued again, so a pass does not re-run (and a billed lane does not re-bill) it. A
# transient failure never lands here — only a deterministic poison does.
POISON_PATH: Final[Path] = Path("benchmark/runner/artifacts/free-tier/poisoned_cells.json")


def load_poisoned_cells(path: Path = POISON_PATH) -> dict[str, str]:
    """Read the persisted poison tombstones (``cell_key -> named reason``); ``{}`` when absent."""
    if not path.exists():
        return {}
    with corpus_lock.corpus_lock(path.parent):
        raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    return {str(key): str(reason) for key, reason in raw.items()}


def save_poisoned_cells(cells: Mapping[str, str], path: Path = POISON_PATH) -> None:
    """Atomically persist the poison tombstones under ``corpus_lock`` (crash-safe)."""
    with corpus_lock.corpus_lock(path.parent):
        corpus_lock.atomic_write_text(
            path, json.dumps(dict(cells), indent=2, sort_keys=True) + "\n"
        )


TEXT_BENCHMARK: str = cs.TEXT_BENCHMARK
MULTIMODAL_BENCHMARK: str = cs.MULTIMODAL_BENCHMARK
Cell = ls.Cell
LaneKey = tuple[str, str]
# Small, positive per-attempt backoff for a transient pre-admission failure (seconds).
RETRY_BACKOFF_S: float = 30.0
# Scan failures retry sooner than the full rescan interval so a transient network blip clears
# without waiting half a day; capped below the interval.
SCAN_BACKOFF_S: float = 300.0
# The floor on the rescan interval. A collector left running unattended must never spin on
# discovery: `--rescan-hours 0` (or a negative value) would otherwise make every iteration due
# and re-GET every provider catalogue as fast as the poll loop allows.
MIN_RESCAN_S: float = 60.0
# Consecutive scan failures before the collector stops. A transient network blip backs off and
# retries; a deterministic failure (malformed overlay, broken corpus) would otherwise retry
# every SCAN_BACKOFF_S forever, which is a slow spin an unattended process cannot signal.
SCAN_FAILURE_CAP: Final[int] = 12
# The single-instance lock an unattended collector holds for its whole lifetime. Two drivers
# sharing the free corpus would clobber each other's lane accounting, poison set and CSV upsert,
# so a second process refuses to start rather than corrupt a multi-day collection.
COLLECTOR_LOCK_PATH: Final[Path] = Path("benchmark/runner/artifacts/free-tier/collector.lock")


class CollectorAlreadyRunningError(RuntimeError):
    """A second collector process tried to start while one already holds the corpus lock."""


class ScanFailureAbortError(RuntimeError):
    """Discovery or work-set construction failed ``SCAN_FAILURE_CAP`` times consecutively."""


@dataclass(frozen=True)
class LaneSpec:
    """One admitted free lane: its canonical identity, provider and value rank."""

    channel: str
    provider: str
    version: str
    priority: float = 0.0
    free: bool = True
    # The listing's `expiration_date`, threaded to `LaneLimits.expires_at` so an expired
    # time-boxed promo is refused (LANE_EXPIRED) instead of scheduled forever.
    expires_at: str | None = None


@dataclass
class Discovery:
    """One rescan's result: the admitted free set and the named scheduling exclusions."""

    scan_as_of: str = ""
    lanes: dict[str, LaneSpec] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)


@dataclass
class QueueEntry:
    """One queued cell plus the ``(model, provider)`` lane identity it must run on."""

    cell: Cell
    provider: str
    version: str
    benchmark: str
    priority: float = 0.0
    attempt: int = 0
    not_before: float = 0.0

    @property
    def lane(self) -> str:
        """The channel id (lane key) the cell runs on."""
        return self.cell[1]

    @property
    def lane_key(self) -> LaneKey:
        """The free gate's key: ``(model, provider)`` — the unit disabled when billed."""
        return (self.cell[1], self.provider)


@dataclass(frozen=True)
class ExecOutcome:
    """One cell attempt's result: produced rows, or the reason it produced none."""

    rows: tuple[dict[str, Any], ...] = ()
    aborted: str = ""
    poison: str = ""
    # The violation code(s) behind a deterministic poison, so a tombstone records WHY. A
    # transient failure carries none and is retried (see ``error``).
    poison_code: str = ""
    error: str = ""


@dataclass
class RunnerSummary:
    """The driver's observable outcome; no hidden ledger backs it."""

    scan_as_of: str = ""
    rescans: int = 0
    cells_queued: int = 0
    cells_run: int = 0
    cells_retried: int = 0
    cells_deferred: int = 0
    cells_dropped: int = 0
    cells_quarantined: int = 0
    lanes_disabled: dict[str, str] = field(default_factory=dict)
    stopped: bool = False


def discovery_from_refresh(
    result: Any, *, access: Mapping[str, str | None] | None = None
) -> Discovery:
    """Convert a ``refresh_free_campaign.RefreshResult`` into the driver's :class:`Discovery`.

    ``runnable_lanes`` has already applied the per-provider free gate (declared free,
    schedulable, not withdrawn, not dominated), so every returned lane is admitted. ``free``
    is the declared-provider marker carried separately so a later rescan can tell a lane that
    was billed apart from one whose provider never offered a free tier.
    """
    snapshot = getattr(result, "snapshot", {}) or {}
    declared = access or {}
    lanes: dict[str, LaneSpec] = {}
    for lane in getattr(result, "runnable", ()) or ():
        provider = str(getattr(lane, "provider", "") or "")
        lanes[str(lane.name)] = LaneSpec(
            channel=str(lane.name),
            provider=provider,
            version=str(getattr(lane, "version", lane.name) or lane.name),
            priority=float(getattr(lane, "priority", 0.0) or 0.0),
            free=declared.get(provider) is None,
            expires_at=str(getattr(lane, "expires_at", "") or "") or None,
        )
    return Discovery(
        scan_as_of=str(snapshot.get("scan_as_of") or ""),
        lanes=lanes,
        excluded={str(k): str(v) for k, v in (getattr(result, "excluded", {}) or {}).items()},
    )


def default_refresh(write: bool) -> Discovery:
    """The live discovery seam: scan every provider and admit the free schedulable set."""
    from benchmark import config
    from benchmark.routing.scripts import refresh_free_campaign as refresh

    return discovery_from_refresh(
        refresh.refresh(write=write), access=config.free_provider_access()
    )


def phase_entries(plan: cs.CampaignPlan, cells: Sequence[Cell], benchmark: str) -> list[Cell]:
    """The cells whose model is currently PHASED to ``benchmark``.

    ``campaign_scheduler.build_plan`` assigns every model its single current benchmark from
    ``collection_priority.runnable_benchmarks``: text until Verified coverage is complete, then
    multimodal for a vision model, empty for a pure-text model that is done. Filtering the
    work set through the plan is therefore what enforces text-before-multimodal per model.
    """
    wanted = {phase.model for phase in plan.phases if phase.benchmark == benchmark}
    return [cell for cell in cells if cell[1] in wanted]


class FreeCampaignRunner:
    """Rescan → rebuild work set → drain the priority queue through the lane scheduler."""

    def __init__(
        self,
        *,
        workers: int = 1,
        rescan_hours: float = 12.0,
        retry_cap: int = 2,
        poll_seconds: float = 30.0,
        scan_failure_cap: int = SCAN_FAILURE_CAP,
        write_scan: bool = False,
        seed: int | None = None,
        results_path: Path | None = None,
        lane_state: dict[str, ls.LaneState] | None = None,
        lane_state_path: Path | None = None,
        poisoned_cells: dict[str, str] | None = None,
        poison_path: Path | None = None,
        persist: bool = True,
        single_instance: bool = True,
        lock_path: Path | None = None,
        refresh_fn: Callable[[bool], Discovery] | None = None,
        planner_fn: Callable[[Discovery], list[QueueEntry]] | None = None,
        execute_fn: Callable[[QueueEntry], ExecOutcome] | None = None,
        limits_fn: Callable[[Discovery], dict[str, ls.LaneLimits]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.workers = max(1, int(workers))
        self.rescan_seconds = max(MIN_RESCAN_S, float(rescan_hours) * 3600.0)
        self.retry_cap = max(0, int(retry_cap))
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.scan_failure_cap = max(1, int(scan_failure_cap))
        self._scan_failures = 0
        self.write_scan = write_scan
        self.seed = seed
        self.results_path = results_path
        self._persist = persist
        self._lane_state_path = lane_state_path or ls.STATE_PATH
        self._single_instance = single_instance
        self._lock_path = lock_path or COLLECTOR_LOCK_PATH
        self._lock_handle: Any = None
        self._poison_path = poison_path or POISON_PATH
        self._poisoned: dict[str, str] = dict(
            poisoned_cells
            if poisoned_cells is not None
            else (load_poisoned_cells(self._poison_path) if persist else {})
        )
        self._refresh_fn = refresh_fn or default_refresh
        self._planner_fn = planner_fn
        self._execute_fn = execute_fn
        self._limits_fn = limits_fn
        self._clock = clock
        self._sleep = sleep

        self._lanes = ls.LaneScheduler(
            limits={},
            state=(
                ls.load_lane_state(self._lane_state_path)
                if lane_state is None and persist
                else (lane_state if lane_state is not None else {})
            ),
            stall_timeout_s=1.0,
        )
        self._lane_locks: dict[str, threading.Lock] = {}
        self._disabled: dict[LaneKey, str] = {}
        self._queue: list[QueueEntry] = []
        self._in_flight: dict[Future[ExecOutcome], QueueEntry] = {}
        self._stop = threading.Event()
        self._next_scan_at = 0.0
        self._scan_seconds = 0.0
        self._discovery = Discovery()
        self._summary = RunnerSummary()
        self._runtime_ready = False
        self._ctx: Any = None
        self._tracker: Any = None
        self._checkpoint: Callable[[dict[str, Any]], None] | None = None
        self._write_lock = threading.Lock()

    # ── reads / test seams ────────────────────────────────────────────────────
    @property
    def queue(self) -> list[QueueEntry]:
        """A snapshot copy of the pending queue (test/operator introspection)."""
        return list(self._queue)

    @property
    def summary(self) -> RunnerSummary:
        """The live run summary (mutated in place; returned by :meth:`run`)."""
        return self._summary

    @property
    def lanes(self) -> ls.LaneScheduler:
        """The shared lane scheduler (admission, quarantine, persisted state)."""
        return self._lanes

    def _now(self) -> float:
        return float(self._clock())

    def _due(self, now: float) -> bool:
        return now >= self._next_scan_at

    def request_stop(self) -> None:
        """Ask the loop to stop after the current checkpoint (SIGTERM/SIGINT handler body)."""
        self._stop.set()

    def install_signal_handlers(self) -> None:
        """Route SIGTERM/SIGINT to :meth:`request_stop`; a non-main thread is left alone."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):  # not the main thread, or unsupported platform
                return

    def _on_signal(self, signum: int, _frame: Any) -> None:
        logger.info("signal %s: stopping after the current checkpoint", signum)
        self.request_stop()

    # ── rescan / plan ─────────────────────────────────────────────────────────
    def rescan(self) -> bool:
        """Run one discovery refresh and rebuild the work set; back off when it fails.

        The refresh and the work-set rebuild are ONE recoverable unit: a deterministic failure
        in either (malformed overlay, broken corpus) must not escape the loop and crash the
        process for a supervisor to restart into the same failure. A transient blip backs off;
        ``SCAN_FAILURE_CAP`` consecutive failures stop the collector loudly instead of retrying
        forever.
        """
        try:
            discovery = self._refresh_fn(self.write_scan)
            self._apply_discovery(discovery)
            self._rebuild_queue(discovery)
        except ScanFailureAbortError:
            raise
        except Exception as exc:  # noqa: BLE001 - a scan failure must never kill the loop
            self._scan_failures += 1
            if self._scan_failures >= self.scan_failure_cap:
                raise ScanFailureAbortError(
                    f"discovery/work-set failed {self._scan_failures} consecutive times; "
                    f"last error: {type(exc).__name__}: {exc}"
                ) from exc
            logger.warning(
                "discovery/work-set failed (%s); backing off (failure %d/%d)",
                exc,
                self._scan_failures,
                self.scan_failure_cap,
            )
            self._next_scan_at = self._now() + min(self.rescan_seconds, SCAN_BACKOFF_S)
            return False
        self._scan_failures = 0
        self._next_scan_at = self._now() + self.rescan_seconds
        self._scan_seconds = self.rescan_seconds
        self._summary.rescans += 1
        self._summary.scan_as_of = discovery.scan_as_of
        logger.info(
            "scan_as_of=%s: %d admitted lane(s), %d queued cell(s)",
            discovery.scan_as_of or "unknown",
            len(discovery.lanes),
            len(self._queue),
        )
        return True

    def _apply_discovery(self, discovery: Discovery) -> None:
        """Rebuild lane limits, re-enable newly free lanes, pre-create per-lane locks."""
        self._discovery = discovery
        limits = self._lane_limits(discovery)
        self._lanes.limits = limits
        for channel, spec in discovery.lanes.items():
            state = self._lanes.lane_state(channel)  # pre-create: no insertion under workers
            self._lane_locks.setdefault(channel, threading.Lock())
            if not (spec.free and state.disabled_reason):
                continue
            if _is_billing_disable(state.disabled_reason):
                # The provider billed a cell on a lane it still advertises as free. Re-enabling
                # on the next rescan would re-bill it before the interlock fired again, so the
                # lane stays disabled until an operator clears it.
                logger.warning(
                    "rescan: %s still admitted free but was billed (%s); keeping it disabled",
                    channel,
                    state.disabled_reason,
                )
                continue
            logger.info("rescan: %s free again; re-enabling lane", channel)
            self._lanes.enable(channel)
            self._disabled.pop((channel, spec.provider), None)

    def _lane_limits(self, discovery: Discovery) -> dict[str, ls.LaneLimits]:
        """Resolve every lane's limits via the injected seam (config by default)."""
        if self._limits_fn is not None:
            return self._limits_fn(discovery)
        from benchmark import config

        unknown = config.lane_unknown_limits()
        per_lane = config.lane_limits_config()
        return {
            channel: ls.LaneLimits.from_mapping(
                {
                    **unknown,
                    **config.lane_limits_from_registry(channel, spec.provider),
                    # The scanned listing's own expiry; the listing is the authority, but a
                    # `lanes.limits` override below can still pin it deliberately.
                    **({"expires_at": spec.expires_at} if spec.expires_at else {}),
                    **per_lane.get(channel, {}),
                }
            )
            for channel, spec in discovery.lanes.items()
        }

    def _rebuild_queue(self, discovery: Discovery) -> None:
        """Rebuild the priority queue, dropping disabled, structurally-refused and poisoned cells.

        A structurally-refused lane (no free access, or a request larger than the whole TPM) can
        never admit a cell, so its cells are dropped here exactly as ``_run_scheduled_batch``
        skips them. Leaving them queued would make ``_has_ready`` permanently False while the
        queue stayed non-empty, so the idle loop woke on every poll interval forever.
        """
        entries = self._plan(discovery)
        refused = set(self._lanes.structural_refusals())
        self._queue = [
            entry
            for entry in entries
            if entry.lane not in refused
            and not self._is_disabled(entry)
            and ls.cell_key(entry.cell) not in self._poisoned
        ]
        dropped = len(entries) - len(self._queue)
        if dropped:
            logger.info("dropped %d cell(s) from disabled/refused/tombstoned lanes", dropped)
        self._summary.cells_queued += len(self._queue)

    def _plan(self, discovery: Discovery) -> list[QueueEntry]:
        if self._planner_fn is not None:
            return list(self._planner_fn(discovery))
        return self._default_plan(discovery)

    def _default_plan(self, discovery: Discovery) -> list[QueueEntry]:
        """Classify MISSING/STALE + priority-plan the work set from the live corpus."""
        from benchmark import config
        from benchmark.routing import collection_priority, integrity
        from benchmark.runner import run_matrix, swebench_multimodal_specs, swebench_specs

        collection_priority.clear_cache()  # a rescan may have rewritten the overlay
        models = list(discovery.lanes)
        if not models:
            return []
        config.register_collection_models(models)
        source = swebench_specs.manifest_source()
        benchmark = (
            MULTIMODAL_BENCHMARK if source == swebench_multimodal_specs.SOURCE else TEXT_BENCHMARK
        )
        cache = config.load_results()
        hashes = integrity.all_hashes(source)
        versions = integrity.model_versions()
        tasks = config.sample_tasks(sorted(hashes), seed=self.seed if self.seed is not None else 42)
        selected, arm_hash_map = run_matrix._arm_context(tasks, models)
        subset = config.concordance_subset_models()
        if subset:
            selected = run_matrix._restrict_concordance_tasks(
                selected, subset, set(config.concordance_subset_challenges())
            )
        status = run_matrix.classify_cells(
            tasks,
            models,
            cache,
            hashes,
            versions,
            None,
            selected,
            arm_hash_map,
            step_limit=config.live_step_limit(),
            prompt_hash=integrity.scaffold_prompt_hash(),
            sampling_hash_map=integrity.sampling_hash_map(models),
            identity_skip_models=set(models) - subset,
            identity_fanout_cap=config.concordance_fanout_cap(),
            identity_fanout_models=subset or None,
        )
        # build_plan owns the concordance-subset dedupe exemption; no engine wrapper is needed.
        plan = cs.build_plan(models, requested_workers=self.workers, exempt_duplicates=subset)
        logger.info("%s", cs.format_plan(plan))
        return [
            self._entry(cell, discovery, plan)
            for cell in phase_entries(plan, status.to_run, benchmark)
        ]

    def _entry(self, cell: Cell, discovery: Discovery, plan: cs.CampaignPlan) -> QueueEntry:
        spec = discovery.lanes.get(cell[1])
        phase = plan.phase_of(cell[1])
        return QueueEntry(
            cell=cell,
            provider=spec.provider if spec else "",
            version=spec.version if spec else cell[1],
            benchmark=phase.benchmark if phase else TEXT_BENCHMARK,
            priority=spec.priority if spec else 0.0,
        )

    # ── queue control ─────────────────────────────────────────────────────────
    def _ordered(self) -> list[QueueEntry]:
        return sorted(
            self._queue,
            key=lambda e: (
                -e.priority,
                0 if e.benchmark == TEXT_BENCHMARK else 1,
                e.cell,
            ),
        )

    def _is_disabled(self, entry: QueueEntry) -> bool:
        if entry.lane_key in self._disabled:
            return True
        return self._lanes.lane_state(entry.lane).disabled_reason is not None

    def _ready(self, entry: QueueEntry, now: float) -> bool:
        return (
            not self._is_disabled(entry)
            and entry.not_before <= now
            and self._lanes.can_admit(entry.lane, now)
        )

    def _has_ready(self, now: float) -> bool:
        return any(self._ready(entry, now) for entry in self._queue)

    def _pop_ready(self, now: float) -> tuple[QueueEntry, threading.Lock] | None:
        """The highest-priority admissible entry, or ``None`` (reserving its lane lock)."""
        for entry in self._ordered():
            if self._is_disabled(entry) or entry.not_before > now:
                continue
            lock = self._lane_locks.setdefault(entry.lane, threading.Lock())
            if not lock.acquire(blocking=False):
                continue  # a same-lane cell is already in flight; try the next lane
            if not self._lanes.can_admit(entry.lane, now):
                lock.release()
                self._defer(entry, now)
                continue
            self._queue.remove(entry)
            return entry, lock
        return None

    # ── execution ─────────────────────────────────────────────────────────────
    def run_pass(self) -> None:
        """Drain the ready queue with ``workers`` concurrent lane-serialized cells."""
        if not self._queue:
            return
        if self._execute_fn is None:
            self._ensure_runtime()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while not self._stop.is_set() and not self._due(self._now()):
                self._fill(pool)
                if not self._in_flight:
                    break
                self._drain(block=True)
        self._drain(block=False)  # reap anything that finished while shutting down

    def _fill(self, pool: ThreadPoolExecutor) -> None:
        now = self._now()
        while len(self._in_flight) < self.workers:
            item = self._pop_ready(now)
            if item is None:
                break
            entry, lock = item
            future = pool.submit(self._execute_locked, entry, lock)
            self._in_flight[future] = entry

    def _execute_locked(self, entry: QueueEntry, lock: threading.Lock) -> ExecOutcome:
        """Run one cell, release its lane lock, and never let an exception escape."""
        try:
            return self._execute(entry)
        except Exception as exc:  # noqa: BLE001 - a lane failure disables/requeues, never aborts
            logger.warning("lane exception %s: %s", entry.lane, exc)
            return ExecOutcome(error=f"{type(exc).__name__}: {exc}")
        finally:
            lock.release()

    def _execute(self, entry: QueueEntry) -> ExecOutcome:
        if self._execute_fn is not None:
            return self._execute_fn(entry)
        return self._default_execute(entry)

    def _default_execute(self, entry: QueueEntry) -> ExecOutcome:
        """Reuse ``_run_scheduled_batch`` (execution) + the wired checkpoint (merge_rows)."""
        from benchmark.routing import validate
        from benchmark.runner import run_matrix

        try:
            rows, _spent, _stopped = run_matrix._run_scheduled_batch(
                [entry.cell],
                self._ctx,
                self._lanes,
                self._tracker,
                self._checkpoint,
                0.0,
                None,
                None,
                "",
            )
        except run_matrix.RunAbortError as exc:
            return ExecOutcome(aborted=str(exc))
        except validate.DataIntegrityError as exc:
            # A produced row that fails the write-time wall. FREE_LANE_BILLED is a lane fault
            # and is already handled inside ``_run_scheduled_batch``; anything reaching here is
            # deterministic — the same inputs produce the same poison — so the CELL is the
            # fault. Tombstone it (without disabling the lane) instead of re-running it forever.
            codes = ",".join(sorted({str(v.code) for v in exc.violations}))
            return ExecOutcome(
                poison=f"{type(exc).__name__}: {exc}",
                poison_code=codes or type(exc).__name__,
            )
        except Exception as exc:  # noqa: BLE001 - a transient fault must retry, never tombstone
            return ExecOutcome(error=f"{type(exc).__name__}: {exc}")
        return ExecOutcome(rows=tuple(rows))

    def _ensure_runtime(self) -> None:
        """Build the live executor's context/tracker/checkpoint once, lazily."""
        if self._runtime_ready:
            return
        # The process-wide $0 breaker is the collector's last-resort spend interlock (the $0
        # interlock, layer 3): a free lane should record real_cost==0, so any accumulated
        # positive scaffold cost is a leak. run_matrix arms it only for its own
        # ``--require-zero-cost`` CLI; an unattended collector is inherently a $0 run, so it
        # arms the same breaker here before a single cell can call a provider.
        _arm_cost_breaker()
        from benchmark import config
        from benchmark.routing import integrity
        from benchmark.runner import infer, run_matrix, swebench_specs

        source = swebench_specs.manifest_source()
        hashes = integrity.all_hashes(source)
        versions = integrity.model_versions()
        models = list(self._discovery.lanes)
        _selected, arm_hash_map = run_matrix._arm_context(sorted(hashes), models)
        free_models = frozenset(m for m in models if run_matrix._is_free_lane(m))
        self._ctx = run_matrix._live_context(
            hashes,
            versions,
            None,
            arm_hash_map,
            infer._AGENT_WALL_LIMIT_S,
            config.live_step_limit(),
            free_models,
        )
        self._tracker = run_matrix._FailureTracker(None, None)
        path = self.results_path or config.results_csv_path()
        self._checkpoint = lambda row, _p=path: run_matrix._checkpoint_row(
            row, path=_p, lock=self._write_lock
        )
        self._runtime_ready = True

    def _drain(self, *, block: bool) -> None:
        """Reap completed futures and apply the per-cell/per-lane policy to each outcome."""
        if not self._in_flight:
            return
        if block:
            wait(
                list(self._in_flight),
                return_when=FIRST_COMPLETED,
                timeout=self.poll_seconds,
            )
        for future in [f for f in list(self._in_flight) if f.done()]:
            entry = self._in_flight.pop(future)
            try:
                outcome = future.result()
            except Exception as exc:  # noqa: BLE001 - a worker crash disables one lane only
                outcome = ExecOutcome(error=f"{type(exc).__name__}: {exc}")
            try:
                self._handle_outcome(entry, outcome)
            except Exception as exc:  # noqa: BLE001 - policy failure disables one lane only
                logger.warning("outcome handling failed for %s: %s", entry.cell, exc)
                self._disable_lane(entry, f"handler error: {exc}")

    # ── policy: disable / defer / retry ───────────────────────────────────────
    def _handle_outcome(self, entry: QueueEntry, outcome: ExecOutcome) -> None:
        now = self._now()
        if outcome.aborted:
            self._disable_lane(entry, outcome.aborted)
            return
        if outcome.poison:
            if _free_lane_billed_code() in outcome.poison:
                self._disable_lane(entry, outcome.poison)  # a lane fault: stop the whole lane
            else:
                self._quarantine_cell(entry, outcome)  # a deterministic cell fault: tombstone
            return
        if outcome.rows:
            if self._billed(entry, outcome):
                return
            self._summary.cells_run += len(outcome.rows)
            # A completed cell trued up its lane's trailing-day/token buckets inside
            # ``_run_scheduled_batch``. Persist now, not only at a clean shutdown: a SIGKILL
            # between cells must not reset the day's RPD and let the restart re-burn it.
            self._persist_lane_state()
            return
        state = self._lanes.lane_state(entry.lane)
        if state.disabled_reason:
            self._disable_lane(entry, state.disabled_reason)
            return
        if state.quarantined_until is not None and state.quarantined_until > now:
            self._defer(entry, now)
            return
        if outcome.error:
            logger.info("transient failure on %s: %s", entry.lane, outcome.error)
        if self._requeue(entry, now) is None:
            self._summary.cells_dropped += 1

    def _billed(self, entry: QueueEntry, outcome: ExecOutcome) -> bool:
        """A free lane that returned a positive real_cost is billed: disable and never re-run."""
        billed = [r for r in outcome.rows if _safe_float(r.get("real_cost")) > 0.0]
        if not billed:
            return False
        self._disable_lane(
            entry,
            f"{_free_lane_billed_code()}: {entry.lane} returned real_cost>0 while admitted free",
        )
        return True

    def _quarantine_cell(self, entry: QueueEntry, outcome: ExecOutcome) -> None:
        """Tombstone ONE deterministically poisonous cell; the lane keeps running.

        A deterministic poison (a produced row failing the data-integrity wall for a reason
        other than FREE_LANE_BILLED) reproduces on every attempt, so re-running it only wastes
        host time and re-bills a billed lane. The tombstone is keyed by the cell and persisted,
        so neither this pass nor a restarted driver queues it again. A transient failure never
        reaches here — it stays MISSING and is retried through :meth:`_requeue`.
        """
        key = ls.cell_key(entry.cell)
        reason = f"{outcome.poison_code or 'poison'}: {outcome.poison}"
        self._poisoned[key] = reason
        self._summary.cells_quarantined += 1
        logger.warning("quarantined poison cell %s: %s", entry.cell, reason)
        self._persist_poisoned()

    def _disable_lane(self, entry: QueueEntry, reason: str) -> None:
        """Disable ``(model, provider)`` and DROP every queued cell for that lane."""
        self._lanes.disable(entry.lane, reason)
        self._disabled[entry.lane_key] = reason
        self._summary.lanes_disabled[f"{entry.lane}@{entry.provider}"] = reason
        # Persist the disable BEFORE doing anything else. This is a money interlock: a SIGKILL
        # in the window between detecting a billed lane and a clean shutdown must not lose it,
        # because the next process would rescan the still-advertised-free lane, re-enable it and
        # re-bill. Immediate persistence bounds the loss to this decision, not the whole run.
        self._persist_lane_state()
        dropped = [e for e in self._queue if e.lane_key == entry.lane_key]
        for queued in dropped:
            self._queue.remove(queued)
        self._summary.cells_dropped += len(dropped)
        logger.warning(
            "disabled lane %s@%s (%s); dropped %d queued cell(s)",
            entry.lane,
            entry.provider,
            reason,
            len(dropped),
        )

    def _defer(self, entry: QueueEntry, now: float) -> None:
        """Hold a rate-limited lane's cell until its quarantine expires (no re-run now)."""
        until = self._lanes.lane_state(entry.lane).quarantined_until
        entry.not_before = until if until is not None and until > now else now + self.poll_seconds
        if not any(queued is entry for queued in self._queue):
            self._queue.append(entry)
        self._summary.cells_deferred += 1
        logger.info("deferring %s on %s until %.0f", entry.cell, entry.lane, entry.not_before)

    def _requeue(self, entry: QueueEntry, now: float) -> QueueEntry | None:
        """Attempt-cap a transient failure: requeue with backoff, else drop it (stays MISSING)."""
        entry.attempt += 1
        if entry.attempt > self.retry_cap:
            logger.warning(
                "retry cap reached for %s on %s; leaving it MISSING", entry.cell, entry.lane
            )
            return None
        entry.not_before = now + RETRY_BACKOFF_S * entry.attempt
        self._queue.append(entry)
        self._summary.cells_retried += 1
        logger.info("requeue %s on %s (attempt %d)", entry.cell, entry.lane, entry.attempt)
        return entry

    # ── outer loop ────────────────────────────────────────────────────────────
    def run_once(self) -> bool:
        """One outer iteration: rescan when due, then drain the ready queue."""
        if self._due(self._now()):
            self.rescan()
        if self._has_ready(self._now()):
            self.run_pass()
            return True
        return False

    def _acquire_singleton(self) -> bool:
        """Take the process-lifetime collector lock; False when another collector holds it.

        ``flock(LOCK_EX|LOCK_NB)`` is released automatically when the process dies (including
        SIGKILL), so a crashed collector can never leave a stale lock behind.
        """
        if not self._single_instance:
            return True
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._lock_handle = handle
        return True

    def _release_singleton(self) -> None:
        """Release the collector lock (no-op when it was not held or locking is disabled)."""
        handle = self._lock_handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._lock_handle = None

    def run(self, *, max_iterations: int | None = None) -> RunnerSummary:
        """The long-running loop; returns when stopped (or after ``max_iterations``)."""
        if not self._acquire_singleton():
            raise CollectorAlreadyRunningError(
                f"another free-campaign collector already holds {self._lock_path}"
            )
        try:
            self.install_signal_handlers()
            iterations = 0
            while not self._stop.is_set():
                if max_iterations is not None and iterations >= max_iterations:
                    break
                iterations += 1
                self.run_once()
                if self._stop.is_set():
                    break
                if self._has_ready(self._now()):
                    continue
                self._wait(self._next_delay())
        finally:
            self._summary.stopped = self._stop.is_set()
            self._persist_lane_state()
            self._persist_poisoned()
            self._release_singleton()
        return self._summary

    def _wait(self, delay: float) -> None:
        """Sleep between passes; the default wait is woken early by SIGTERM/SIGINT.

        An injected ``sleep`` (tests, simulation) is used verbatim; the real ``time.sleep``
        is replaced by :meth:`threading.Event.wait` so a signal sets ``_stop`` and the loop
        exits promptly instead of sleeping out a multi-hour rescan interval.
        """
        if self._sleep is time.sleep:
            self._stop.wait(timeout=delay)
        else:
            self._sleep(delay)

    def _next_delay(self) -> float:
        """Seconds until the next rescan, or the soonest deferred cell becomes ready.

        The floor is ``poll_seconds``: every time-based wait the driver sets (retry backoff,
        quarantine, defer) is at least that long, so a delay of zero only means the blocked
        cells are blocked for a NON-time reason — polling faster cannot help and would busy-loop
        an unattended process. Structurally-refused cells are dropped at rebuild, so this floor
        is the backstop rather than the only defence.
        """
        now = self._now()
        delay = max(0.0, self._next_scan_at - now)
        if self._queue:
            delay = min(delay, max(0.0, min(e.not_before for e in self._queue) - now))
        return max(self.poll_seconds, delay)

    def _persist_lane_state(self) -> None:
        if not self._persist:
            return
        try:
            ls.save_lane_state(self._lanes.state, self._lane_state_path)
        except OSError as exc:  # persistence failure must not crash the driver
            logger.warning("could not persist lane state: %s", exc)

    def _persist_poisoned(self) -> None:
        """Persist the poison tombstones; a write failure must not crash the driver."""
        if not self._persist:
            return
        try:
            save_poisoned_cells(self._poisoned, self._poison_path)
        except OSError as exc:
            logger.warning("could not persist poison tombstones: %s", exc)


def _safe_float(value: Any) -> float:
    """A float from a row cell, or ``0.0`` when blank/unparseable (never an exception)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _arm_cost_breaker() -> None:
    """Arm the process-wide mini-swe-agent cost breaker for this $0 collection process.

    Delegates to the shipped layer-3 interlock so there is one implementation:
    ``run_matrix._arm_free_lane_cost_breaker`` sets ``MSWEA_GLOBAL_COST_LIMIT`` and re-applies
    it to the already-constructed global stats. Imported lazily so this module stays light for
    the pure tests that never build a live runtime.
    """
    from benchmark.runner import run_matrix

    run_matrix._arm_free_lane_cost_breaker()


def _free_lane_billed_code() -> str:
    """The shipped FREE_LANE_BILLED code, imported lazily to keep this module light."""
    from benchmark.routing import validate

    return validate.FREE_LANE_BILLED


def _is_billing_disable(reason: str | None) -> bool:
    """True iff the lane was disabled because a billed cell proved it is not $0.

    Such a disable is a money-safety interlock, not a transient condition: a rescan must NOT
    clear it just because the provider still advertises the lane as free.
    """
    return reason is not None and reason.startswith(_free_lane_billed_code())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.runner.free_campaign_runner",
        description="Long-running free-tier collection driver (rescan + queue + lane gate).",
    )
    parser.add_argument("--config", default="configs/free-tier/benchmark.yaml")
    parser.add_argument("--free-registry", default="configs/free-tier/overlay.yaml")
    parser.add_argument("--workers", type=int, default=1, help="concurrent lane slots")
    parser.add_argument("--rescan-hours", type=float, default=12.0)
    parser.add_argument("--retry-cap", type=int, default=2, help="transient attempts per cell")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--write-scan",
        action="store_true",
        help="persist the discovery refresh (default: dry-run, writes no tracked file)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: configure the overlay, build the driver, run it to completion."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    from benchmark import config

    config.load(args.config)
    if args.free_registry:
        config.set_free_registry(args.free_registry)
    # Arm the $0 process breaker before any live runtime can be built, so a restart cannot
    # begin collecting for the few seconds before ``_ensure_runtime`` re-arms it.
    _arm_cost_breaker()
    runner = FreeCampaignRunner(
        workers=args.workers,
        rescan_hours=args.rescan_hours,
        retry_cap=args.retry_cap,
        poll_seconds=args.poll_seconds,
        write_scan=args.write_scan,
        seed=args.seed,
    )
    try:
        summary = runner.run()
    except CollectorAlreadyRunningError as exc:
        logger.error("%s", exc)
        return 2
    except ScanFailureAbortError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "stopped=%s rescans=%d queued=%d run=%d retried=%d deferred=%d dropped=%d",
        summary.stopped,
        summary.rescans,
        summary.cells_queued,
        summary.cells_run,
        summary.cells_retried,
        summary.cells_deferred,
        summary.cells_dropped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

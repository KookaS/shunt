"""Parallel live runner, reasoning-arm request wiring, and cost-ceiling guard.

Covers distinct-arm-distinct-request, serial==parallel determinism, and the
cost ceiling. No live/paid calls: ``infer.run_live_cell`` is always monkeypatched.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any, Final

import pytest

from benchmark import config
from benchmark.runner import infer, run_matrix

_HASHES: Final[dict[str, str]] = {f"repo__task-{i}": "h" for i in range(1, 9)}
_VERSIONS: Final[dict[str, str]] = {"m": "v"}


def _cells(n: int) -> list[tuple[str, str, str]]:
    return [(f"repo__task-{i}", "m", "default") for i in range(1, n + 1)]


def _grid_cells(n_challenges: int, models: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Challenge-major grid: every model (× one arm) for challenge 1, then challenge 2, …

    Mirrors how ``classify_cells`` emits real cells — the unit that challenge-major
    completion must keep together.
    """
    return [(f"repo__task-{c}", m, "default") for c in range(1, n_challenges + 1) for m in models]


def _outcome(cost: float = 0.01, passed: bool = True) -> dict[str, Any]:
    return {"pass": passed, "in_tok": 10, "out_tok": 5, "calls": 1, "real_cost": cost}


def _fixed_outcome(cost: float = 0.01, passed: bool = True) -> dict[str, Any]:
    """Outcome with a pinned ``computed_at`` so rows are byte-stable across runs.

    ``_build_row`` would otherwise stamp ``_now_iso()``, defeating exact byte-identity.
    """
    return {**_outcome(cost, passed), "computed_at": "2026-01-01T00:00:00+00:00"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class TestArmRequestWiring:
    """A reasoning arm must change the live request — else sampling bills duplicates."""

    def _multi_arm_model(self) -> str:
        for name, cfg in config.reasoning_configs().items():
            if cfg and len(cfg.arms) > 1 and config.model_has_cache(name):
                return name
        raise AssertionError("expected a multi-arm registry model to exercise arm wiring")

    def test_distinct_arms_yield_distinct_model_kwargs(self):
        config.load("benchmark/config.yaml")
        model = self._multi_arm_model()
        cfg = config.reasoning_configs()[model]
        assert cfg is not None
        arm_ids = [a.id for a in cfg.arms]
        kwargs = {a: infer._scaffold_model_kwargs(model, a, {}, {}) for a in arm_ids}
        # The whole point: no two arms collapse to the same billed request.
        distinct = {tuple(sorted(k.items(), key=repr)) for k in kwargs.values()}
        assert len(distinct) == len(arm_ids), f"arms aliased to identical kwargs: {kwargs}"

    def test_arm_params_overlay_target_and_base(self):
        config.load("benchmark/config.yaml")
        model = self._multi_arm_model()
        cfg = config.reasoning_configs()[model]
        assert cfg is not None
        arm = cfg.arms[0].id
        merged = infer._scaffold_model_kwargs(
            model, arm, {"base": 1}, {"api_base": "x", "api_key": "k"}
        )
        assert merged["base"] == 1 and merged["api_base"] == "x"
        for key, value in config.arm_api_params(model, arm).items():
            assert merged[key] == value

    def test_arm_params_cannot_clobber_auth_keys(self, monkeypatch):
        # A future registry arm carrying api_base/api_key/model_name must fail loud,
        # not silently break the routing target's auth on a paid call.
        monkeypatch.setattr(config, "arm_api_params", lambda m, a: {"api_key": "evil"})
        import pytest

        with pytest.raises(ValueError, match="reserved request key"):
            infer._scaffold_model_kwargs("m", "x", {}, {"api_key": "real"})

    def test_generate_patch_live_threads_arm_to_scaffold(self, monkeypatch):
        captured: dict[str, str] = {}

        def fake_invoke(spec, model, scaffold, arm):
            captured["arm"] = arm
            return infer.AgentPatch(patch="p", in_tok=1, out_tok=1, calls=1, cost=0.0)

        monkeypatch.setattr(infer, "_invoke_scaffold", fake_invoke)
        monkeypatch.setattr(infer, "has_api_keys", lambda env=None: True)

        class _Spec:
            instance_id = "repo__task-1"

        infer.generate_patch_live(_Spec(), "m", arm="high")
        assert captured["arm"] == "high"


class TestParallelDeterminism:
    """Parallel output must be byte-identical to serial for a fixed cell set."""

    def _run(self, monkeypatch, workers: int, cells):
        def fake(cid, model, **kw):
            return _outcome(cost=0.01)

        monkeypatch.setattr(infer, "run_live_cell", fake)
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])
        return run_matrix.run_live_cells(
            cells, {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=workers
        )

    def test_parallel_rows_identical_to_serial(self, monkeypatch):
        cells = _cells(8)
        serial = self._run(monkeypatch, 1, cells)
        parallel = self._run(monkeypatch, 4, cells)

        # Same rows, same ORDER — the CSV write is order-independent, but we still
        # pin ordering so a determinism regression can't hide behind the sort.
        def strip(rows):
            return [{k: r[k] for k in r if k != "computed_at"} for r in rows]

        assert strip(serial) == strip(parallel)

    def test_parallel_actually_concurrent(self, monkeypatch):
        # A crude proof the pool overlaps work WITHIN a challenge: one challenge with 4
        # model cells that each sleep 100ms finishes in well under the 400ms a serial
        # loop would take. (Concurrency is intra-challenge under challenge-major batching.)
        barrier_hits: list[float] = []
        lock = threading.Lock()

        def fake(cid, model, **kw):
            time.sleep(0.1)
            with lock:
                barrier_hits.append(time.monotonic())
            return _outcome()

        monkeypatch.setattr(infer, "run_live_cell", fake)
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])
        cells = _grid_cells(1, ("m0", "m1", "m2", "m3"))
        start = time.monotonic()
        run_matrix.run_live_cells(
            cells, {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=4
        )
        assert time.monotonic() - start < 0.35

    def test_parallel_preserves_per_cell_isolation(self, monkeypatch):
        def fake(cid, model, **kw):
            if cid == "repo__task-3":
                raise infer.HarnessInfraError("docker down")
            return _outcome()

        monkeypatch.setattr(infer, "run_live_cell", fake)
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])
        rows = run_matrix.run_live_cells(
            _cells(6), {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=4
        )
        got = sorted(r["challenge_id"] for r in rows)
        assert "repo__task-3" not in got
        assert len(got) == 5


class TestCostCeiling:
    """--max-cost aborts remaining cells cleanly, keeping completed rows."""

    def test_ceiling_stops_and_keeps_completed(self, monkeypatch):
        def fake(cid, model, **kw):
            return _outcome(cost=1.0)

        monkeypatch.setattr(infer, "run_live_cell", fake)
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])
        # 8 cells at $1 each, ceiling $3 → stop after crossing, far short of all 8.
        rows = run_matrix.run_live_cells(
            _cells(8), {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=1, max_cost=3.0
        )
        assert 0 < len(rows) < 8
        # Every returned row is a real completed cell (nothing fabricated).
        assert all(float(r["real_cost"]) == 1.0 for r in rows)

    def test_no_ceiling_runs_all(self, monkeypatch):
        def fake(cid, model, **kw):
            return _outcome(cost=1.0)

        monkeypatch.setattr(infer, "run_live_cell", fake)
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])
        rows = run_matrix.run_live_cells(
            _cells(8), {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=1, max_cost=None
        )
        assert len(rows) == 8


class TestIncrementalCheckpoint:
    """Each completed cell is persisted to results.csv the instant it finishes."""

    def _patch(self, monkeypatch, fake) -> None:
        monkeypatch.setattr(infer, "run_live_cell", fake)
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])

    def test_no_path_means_no_persistence(self, monkeypatch, tmp_path):
        # Backward compat: default results_path=None keeps the old in-memory-only path.
        calls: list[int] = []
        monkeypatch.setattr(run_matrix, "merge_rows", lambda rows, *a, **k: calls.append(len(rows)))
        self._patch(monkeypatch, lambda cid, model, **kw: _outcome())
        run_matrix.run_live_cells(
            _cells(4), {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=1
        )
        assert calls == []
        assert not (tmp_path / "results.csv").exists()

    def test_each_cell_persisted_before_next_starts_serial(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"
        seen_before: list[int] = []

        def fake(cid, model, **kw):
            seen_before.append(len(_read_csv_rows(out)))
            return _outcome()

        self._patch(monkeypatch, fake)
        rows = run_matrix.run_live_cells(
            _cells(4),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=out,
        )
        # Cell k starts only after cells 1..k-1 are already on disk.
        assert seen_before == [0, 1, 2, 3]
        assert len(_read_csv_rows(out)) == 4 == len(rows)

    def test_kill_mid_batch_keeps_completed_cells(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"

        def fake(cid, model, **kw):
            if cid == "repo__task-4":
                raise KeyboardInterrupt  # a real kill: BaseException, not caught per-cell
            return _outcome()

        self._patch(monkeypatch, fake)
        with pytest.raises(KeyboardInterrupt):
            run_matrix.run_live_cells(
                _cells(8),
                {},
                _HASHES,
                _VERSIONS,
                timeout=10,
                verbose=False,
                workers=1,
                results_path=out,
            )
        # The 3 cells before the kill survived on disk; nothing in flight was lost wholesale.
        on_disk = _read_csv_rows(out)
        got = sorted(r["challenge_id"] for r in on_disk)
        assert got == ["repo__task-1", "repo__task-2", "repo__task-3"]
        # Atomicity: the interrupted checkpoint left a COMPLETE, re-loadable results.csv
        # — every RESULTS_FIELDS column present on every surviving row — not a half-written
        # file, and os.replace left no sibling .tmp turd behind.
        from benchmark.routing import integrity

        for r in on_disk:
            assert set(r.keys()) == set(integrity.RESULTS_FIELDS)
        assert not (tmp_path / "results.csv.tmp").exists()

    def test_ceiling_persists_completed_only(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"
        self._patch(monkeypatch, lambda cid, model, **kw: _outcome(cost=1.0))
        rows = run_matrix.run_live_cells(
            _cells(8),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            max_cost=3.0,
            results_path=out,
        )
        on_disk = _read_csv_rows(out)
        assert 0 < len(on_disk) == len(rows) < 8
        assert all(float(r["real_cost"]) == 1.0 for r in on_disk)

    def test_skipped_cell_writes_nothing(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"

        def fake(cid, model, **kw):
            if cid == "repo__task-3":
                raise infer.HarnessInfraError("docker down")
            return _outcome()

        self._patch(monkeypatch, fake)
        run_matrix.run_live_cells(
            _cells(6),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=out,
        )
        got = sorted(r["challenge_id"] for r in _read_csv_rows(out))
        assert "repo__task-3" not in got and len(got) == 5

    def test_empty_cells_no_write(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"
        self._patch(monkeypatch, lambda cid, model, **kw: _outcome())
        rows = run_matrix.run_live_cells(
            [], {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=1, results_path=out
        )
        assert rows == [] and not out.exists()

    def test_parallel_all_rows_land_no_dups(self, monkeypatch, tmp_path):
        serial_out = tmp_path / "serial.csv"
        parallel_out = tmp_path / "parallel.csv"
        self._patch(monkeypatch, lambda cid, model, **kw: _outcome())
        run_matrix.run_live_cells(
            _cells(8),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=serial_out,
        )
        run_matrix.run_live_cells(
            _cells(8),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=4,
            results_path=parallel_out,
        )

        def keys(path):
            rows = _read_csv_rows(path)
            return sorted((r["challenge_id"], r["model"], r["reasoning"]) for r in rows)

        pkeys = keys(parallel_out)
        assert len(pkeys) == 8 == len(set(pkeys))  # no lost/dup rows under concurrency
        assert pkeys == keys(serial_out)  # parallel CSV identical to serial

    def test_only_main_thread_checkpoints(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"
        real_merge = run_matrix.merge_rows
        checkpoint_threads: list[str] = []
        worker_threads: list[str] = []
        # A short sleep forces the pool to genuinely overlap cells, so the
        # main-thread-only assertion below is meaningful and not vacuously true
        # because every cell happened to run on the main thread anyway.
        lock = threading.Lock()

        def spy(rows, path, *a, **k):
            checkpoint_threads.append(threading.current_thread().name)
            return real_merge(rows, path, *a, **k)

        def fake(cid, model, **kw):
            time.sleep(0.02)
            with lock:
                worker_threads.append(threading.current_thread().name)
            return _outcome()

        monkeypatch.setattr(run_matrix, "merge_rows", spy)
        self._patch(monkeypatch, fake)
        # 2 challenges × 4 models: intra-challenge fan-out puts multiple workers to work
        # within a batch, so the multiple-worker assertion below is meaningful.
        run_matrix.run_live_cells(
            _grid_cells(2, ("m0", "m1", "m2", "m3")),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=4,
            results_path=out,
        )
        # Concurrency actually happened: cells ran off the main thread. Without this
        # the main-only checkpoint assertion could pass simply because nothing ran
        # in parallel — moving the checkpoint into _run_one_cell would then not be
        # caught. This guards the guard.
        assert threading.main_thread().name not in set(worker_threads)
        assert len(set(worker_threads)) > 1  # multiple pool workers did the work
        # Worker threads must never touch the CSV — only the as_completed main loop.
        assert len(checkpoint_threads) == 8
        assert set(checkpoint_threads) == {threading.main_thread().name}

    def test_final_remerge_is_idempotent(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"
        self._patch(monkeypatch, lambda cid, model, **kw: _outcome())
        rows = run_matrix.run_live_cells(
            _cells(4),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=out,
        )
        history = out.parent / "artifacts" / "results_history.csv"
        assert not history.exists()  # checkpointing alone supersedes nothing
        # main()'s final merge re-merges already-checkpointed rows: a pure no-op.
        run_matrix.merge_rows(rows, out)
        assert len(_read_csv_rows(out)) == 4
        assert not history.exists()  # no spurious history growth

    def test_supersession_archives_changed_row_once(self, tmp_path):
        out = tmp_path / "results.csv"
        base = run_matrix._build_row(
            "repo__task-1", "m", _outcome(cost=0.01), _HASHES, _VERSIONS, {}
        )
        run_matrix.merge_rows([base], out)
        changed = run_matrix._build_row(
            "repo__task-1", "m", _outcome(cost=0.99), _HASHES, _VERSIONS, {}
        )
        run_matrix.merge_rows([changed], out)
        history = out.parent / "artifacts" / "results_history.csv"
        assert len(_read_csv_rows(history)) == 1  # old row archived exactly once
        assert len(_read_csv_rows(out)) == 1  # results.csv keeps only current


class TestCheckpointDeterminism:
    """The load-bearing guarantee: checkpointing never changes the bytes written.

    serial(checkpoint) == parallel(N)(checkpoint) == in-memory(no-checkpoint) merge.
    """

    def _patch(self, monkeypatch) -> None:
        monkeypatch.setattr(infer, "run_live_cell", lambda cid, model, **kw: _fixed_outcome())
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])

    def test_serial_parallel_inmemory_byte_identical(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cells = _cells(8)
        serial_out = tmp_path / "serial.csv"
        parallel_out = tmp_path / "parallel.csv"
        inmem_out = tmp_path / "inmem.csv"

        run_matrix.run_live_cells(
            cells,
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=serial_out,
        )
        run_matrix.run_live_cells(
            cells,
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=4,
            results_path=parallel_out,
        )
        # The no-checkpoint path: rows collected in memory, then merged once (what
        # main() does when results_path is None). Must yield the same file.
        rows = run_matrix.run_live_cells(
            cells, {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=4
        )
        run_matrix.merge_rows(rows, inmem_out)

        serial_bytes = serial_out.read_bytes()
        assert serial_bytes == parallel_out.read_bytes()  # checkpoint order-independent
        assert serial_bytes == inmem_out.read_bytes()  # checkpoint == in-memory merge
        # Non-trivial content (not two empty files comparing equal).
        assert len(_read_csv_rows(serial_out)) == 8


class TestCostCeilingBoundaries:
    """Off-by-one and sign edge cases on --max-cost, plus the always-keep invariant."""

    def _patch(self, monkeypatch, cost: float) -> None:
        monkeypatch.setattr(infer, "run_live_cell", lambda cid, model, **kw: _outcome(cost=cost))
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])

    def _run(self, cells, tmp_path, max_cost):
        out = tmp_path / "results.csv"
        rows = run_matrix.run_live_cells(
            cells,
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            max_cost=max_cost,
            results_path=out,
        )
        return rows, out

    def test_zero_ceiling_runs_nothing(self, monkeypatch, tmp_path):
        # --max-cost 0 must spend nothing: the serial guard trips before cell 1, so
        # no row is produced and (crucially) no empty file is created.
        self._patch(monkeypatch, cost=1.0)
        rows, out = self._run(_cells(8), tmp_path, max_cost=0.0)
        assert rows == []
        assert not out.exists()

    def test_negative_ceiling_runs_nothing(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, cost=1.0)
        rows, out = self._run(_cells(8), tmp_path, max_cost=-5.0)
        assert rows == []
        assert not out.exists()

    def test_ceiling_is_inclusive_at_exact_boundary(self, monkeypatch, tmp_path):
        # >= boundary: 3 cells at $1 hit spent==$3==ceiling, which STOPS the loop,
        # so exactly the 2 cells that ran while spent<3 are kept (cell 3 is when the
        # guard sees spent==3). Pins the inclusive semantics of _over_budget.
        self._patch(monkeypatch, cost=1.0)
        rows, out = self._run(_cells(8), tmp_path, max_cost=3.0)
        assert len(rows) == 3
        assert len(_read_csv_rows(out)) == 3

    def test_single_cell_over_ceiling_is_still_kept(self, monkeypatch, tmp_path):
        # One expensive cell blows past the ceiling on its own — it is already PAID,
        # so it must be persisted, never discarded for exceeding the cap.
        self._patch(monkeypatch, cost=5.0)
        rows, out = self._run(_cells(8), tmp_path, max_cost=3.0)
        assert len(rows) == 1
        assert float(_read_csv_rows(out)[0]["real_cost"]) == 5.0

    def test_high_ceiling_never_reached_runs_all(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, cost=1.0)
        rows, out = self._run(_cells(8), tmp_path, max_cost=1000.0)
        assert len(rows) == 8
        assert len(_read_csv_rows(out)) == 8

    def test_parallel_ceiling_keeps_all_completed_no_loss(self, monkeypatch, tmp_path):
        # Ceiling crossed mid parallel batch: in-flight/completed cells are all paid,
        # so every collected row must be on disk (returned == persisted) and none
        # fabricated. At least the cells needed to cross the ceiling completed.
        out = tmp_path / "results.csv"
        self._patch(monkeypatch, cost=1.0)
        rows = run_matrix.run_live_cells(
            _cells(8),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=4,
            max_cost=3.0,
            results_path=out,
        )
        on_disk = _read_csv_rows(out)
        keys_rows = sorted((r["challenge_id"], r["reasoning"]) for r in rows)
        keys_disk = sorted((r["challenge_id"], r["reasoning"]) for r in on_disk)
        assert keys_rows == keys_disk  # nothing paid-for lost between return and disk
        assert 3 <= len(on_disk) <= 8  # crossed the ceiling, cancelled some pending
        assert all(float(r["real_cost"]) == 1.0 for r in on_disk)  # nothing fabricated


class TestChallengeMajorOrdering:
    """Cells complete challenge-at-a-time, so a ceiling leaves a prefix of fully-covered
    challenges — fixing the old skew where fast models raced ahead across many challenges
    while slow ones lagged, leaving a --max-cost cut with many partially-covered challenges.
    """

    _MODELS: Final[tuple[str, str, str, str]] = ("m0", "m1", "m2", "m3")

    def _patch(self, monkeypatch, cost: float = 1.0) -> None:
        monkeypatch.setattr(infer, "run_live_cell", lambda cid, model, **kw: _outcome(cost=cost))
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])

    def _coverage(self, rows) -> dict[str, set[str]]:
        cov: dict[str, set[str]] = {}
        for r in rows:
            cov.setdefault(r["challenge_id"], set()).add(r["model"])
        return cov

    def test_group_by_challenge_preserves_order_and_reunites_splits(self):
        # A challenge split across the (missing + stale) concatenation is reunited at its
        # first appearance; challenge order = first-appearance order, unchanged.
        cells = [
            ("repo__task-1", "m0", "default"),
            ("repo__task-2", "m0", "default"),
            ("repo__task-1", "m1", "high"),  # task-1 reappears (e.g. a stale arm)
        ]
        grouped = run_matrix._group_by_challenge(cells)
        assert [cid for cid, _ in grouped] == ["repo__task-1", "repo__task-2"]
        assert grouped[0][1] == [
            ("repo__task-1", "m0", "default"),
            ("repo__task-1", "m1", "high"),
        ]

    def test_ceiling_leaves_prefix_of_fully_covered_challenges(self, monkeypatch, tmp_path):
        # 6 challenges × 4 models @ $1, ceiling $10. Challenge-major completion → the kept
        # challenges are a prefix, each FULLY covered (all 4 models); none left partial.
        self._patch(monkeypatch, cost=1.0)
        out = tmp_path / "results.csv"
        rows = run_matrix.run_live_cells(
            _grid_cells(6, self._MODELS),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=4,
            max_cost=10.0,
            results_path=out,
        )
        cov = self._coverage(rows)
        covered = list(cov.keys())
        full = set(self._MODELS)
        # Every emitted challenge is fully covered (all models) — no partial challenge.
        assert all(models == full for models in cov.values()), cov
        # The covered set is a prefix of the challenge order.
        expected_prefix = [f"repo__task-{c}" for c in range(1, len(covered) + 1)]
        assert covered == expected_prefix
        assert 0 < len(covered) < 6  # ceiling actually cut
        # On-disk matches returned (nothing paid-for lost).
        assert len(_read_csv_rows(out)) == len(rows)

    def test_ceiling_prefix_holds_serial_too(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, cost=1.0)
        out = tmp_path / "results.csv"
        rows = run_matrix.run_live_cells(
            _grid_cells(6, self._MODELS),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            max_cost=10.0,
            results_path=out,
        )
        cov = self._coverage(rows)
        covered = list(cov.keys())
        assert all(models == set(self._MODELS) for models in cov.values()), cov
        assert covered == [f"repo__task-{c}" for c in range(1, len(covered) + 1)]
        assert 0 < len(covered) < 6

    def test_grid_serial_parallel_byte_identical(self, monkeypatch, tmp_path):
        # Byte-identity under intra-challenge parallelism: a full grid run must write the
        # same CSV serial vs parallel (the write sorts by key, but we pin it anyway).
        monkeypatch.setattr(
            infer, "run_live_cell", lambda cid, model, **kw: _fixed_outcome(cost=0.01)
        )
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])
        cells = _grid_cells(6, self._MODELS)
        serial_out = tmp_path / "serial.csv"
        parallel_out = tmp_path / "parallel.csv"
        run_matrix.run_live_cells(
            cells,
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=serial_out,
        )
        run_matrix.run_live_cells(
            cells,
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=4,
            results_path=parallel_out,
        )
        assert serial_out.read_bytes() == parallel_out.read_bytes()
        assert len(_read_csv_rows(serial_out)) == 24


class TestCheckpointResumability:
    """A second run over the same cells upserts; supersession archives exactly once."""

    def _patch(self, monkeypatch, outcome_fn) -> None:
        monkeypatch.setattr(infer, "run_live_cell", outcome_fn)
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])

    def test_rerun_identical_cells_upserts_no_dups_no_history(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"
        history = out.parent / "artifacts" / "results_history.csv"
        cells = _cells(4)
        self._patch(monkeypatch, lambda cid, model, **kw: _fixed_outcome(cost=0.01))
        for _ in range(2):  # resume: run the same cells twice through the checkpoint path
            run_matrix.run_live_cells(
                cells,
                {},
                _HASHES,
                _VERSIONS,
                timeout=10,
                verbose=False,
                workers=1,
                results_path=out,
            )
        # Upsert, not append: still 4 unique rows, and an identical re-run supersedes
        # nothing (no history file at all).
        disk = _read_csv_rows(out)
        keys = {(r["challenge_id"], r["model"], r["reasoning"]) for r in disk}
        assert len(disk) == 4 == len(keys)
        assert not history.exists()

    def test_rerun_changed_cells_archives_each_once(self, monkeypatch, tmp_path):
        out = tmp_path / "results.csv"
        history = out.parent / "artifacts" / "results_history.csv"
        cells = _cells(4)
        self._patch(monkeypatch, lambda cid, model, **kw: _fixed_outcome(cost=0.01))
        run_matrix.run_live_cells(
            cells,
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=out,
        )
        # Re-run with a CHANGED outcome (different cost) → each of the 4 rows is
        # superseded exactly once; results.csv still holds only the 4 current rows.
        self._patch(monkeypatch, lambda cid, model, **kw: _fixed_outcome(cost=0.99))
        run_matrix.run_live_cells(
            cells,
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=out,
        )
        assert len(_read_csv_rows(history)) == 4  # one archived row per changed cell
        assert len(_read_csv_rows(out)) == 4
        assert all(float(r["real_cost"]) == 0.99 for r in _read_csv_rows(out))
        # A third, idempotent re-run of the changed values grows history by nothing.
        run_matrix.run_live_cells(
            cells,
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            results_path=out,
        )
        assert len(_read_csv_rows(history)) == 4  # no double-archiving on idempotent merge

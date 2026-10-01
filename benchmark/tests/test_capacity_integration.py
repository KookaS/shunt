"""Capacity manager + RAM guard wired into ``run_matrix.run_live_cells``.

Fully offline: the container executor is monkeypatched, the capacity manager's preflight
seam is monkeypatched, and the fake handle yields a REAL ``ImageRetention`` so the
per-challenge ref-counting under test is the production code, not a stand-in. No docker,
no network, no /proc dependency leaks (``read_mem_available_bytes`` is injected).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

import pytest

from benchmark import config
from benchmark.runner import infer, run_matrix
from benchmark.runner.capacity import guard as capacity_guard
from benchmark.runner.capacity.guard import CapacityHandle, InsufficientDiskError
from benchmark.runner.capacity.retention import ImageRetention
from benchmark.runner.capacity.sizes import image_refs_for_ids
from benchmark.runner.capacity.types import CapacityVerdict, ResourceConfig, RetentionPolicy
from benchmark.runner.memory_guard import InsufficientMemoryError, MemoryConfig

GIB: Final[int] = 1024**3
_HASHES: Final[dict[str, str]] = {f"repo__task-{i}": "h" for i in range(1, 9)}
_VERSIONS: Final[dict[str, str]] = {"m": "v"}


@pytest.fixture(autouse=True)
def _no_cache_wall(monkeypatch: pytest.MonkeyPatch) -> None:
    """The uncached-enabled-model wall reads the real registry; neutralise it for every test."""
    monkeypatch.setattr(config, "models_missing_cache", lambda *a, **k: [])


def _cells(n: int) -> list[tuple[str, str, str]]:
    return [(f"repo__task-{i}", "m", "default") for i in range(1, n + 1)]


def _grid_cells(challenges: int, models: tuple[str, ...]) -> list[tuple[str, str, str]]:
    return [(f"repo__task-{c}", m, "default") for c in range(1, challenges + 1) for m in models]


def _outcome(cost: float = 0.01) -> dict[str, Any]:
    return {"pass": True, "in_tok": 10, "out_tok": 5, "calls": 1, "real_cost": cost}


def _recording_cell(ran: list[str], cost: float = 0.01) -> Any:
    """An ``infer.run_live_cell`` fake that records each challenge it was asked to run."""

    def _fake(cid: str, model: str, **kwargs: Any) -> dict[str, Any]:
        ran.append(cid)
        return _outcome(cost)

    return _fake


def _verdict(*, ok: bool = True) -> CapacityVerdict:
    return CapacityVerdict(
        ok=ok,
        needed_bytes=0,
        free_bytes=0,
        shortfall_bytes=0,
        per_image=(),
        docker_root="/",
        remedies=(),
    )


def _handle(retention: ImageRetention | None = None) -> CapacityHandle:
    return CapacityHandle(
        retention=retention or ImageRetention(ResourceConfig(), remove_fn=lambda _ref: 0),
        prefetch=None,
        verdict=_verdict(),
    )


def _fake_manager(handle: CapacityHandle, captured: dict[str, Any] | None = None) -> Any:
    """A contextmanager standing in for ``capacity_manager``, recording the refs it saw."""

    @contextmanager
    def _cm(refs: Any, *, config: Any, **kwargs: Any) -> Iterator[CapacityHandle]:
        if captured is not None:
            captured["refs"] = list(refs)
        yield handle

    return _cm


class TestDiskPreflight:
    def test_insufficient_disk_aborts_before_any_cell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[str] = []
        monkeypatch.setattr(infer, "run_live_cell", _recording_cell(ran))
        # The real capacity_manager calls its guard-module `preflight`; return a failing verdict.
        monkeypatch.setattr(capacity_guard, "preflight", lambda refs, **k: _verdict(ok=False))

        with pytest.raises(InsufficientDiskError):
            run_matrix.run_live_cells(
                _cells(3),
                {},
                _HASHES,
                _VERSIONS,
                timeout=10,
                verbose=False,
                workers=1,
                resources=ResourceConfig(),
            )
        assert ran == []  # refused before the first container

    def test_allow_insufficient_disk_proceeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_preflight(refs: Any, *, config: ResourceConfig, **k: Any) -> CapacityVerdict:
            seen["allow"] = config.allow_insufficient_disk
            # The real preflight sets ok=True itself when the config allows; mirror that.
            return _verdict(ok=config.allow_insufficient_disk)

        monkeypatch.setattr(capacity_guard, "preflight", fake_preflight)
        monkeypatch.setattr(infer, "run_live_cell", lambda cid, model, **k: _outcome())

        rows = run_matrix.run_live_cells(
            _cells(2),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            resources=ResourceConfig(allow_insufficient_disk=True),
        )
        assert seen["allow"] is True
        assert len(rows) == 2


class TestPerChallengeRetention:
    def test_releases_a_challenge_only_after_all_its_cells(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[tuple[str, str]] = []

        def remove(ref: str) -> int:
            events.append(("rm", ref))
            return 0

        resources = ResourceConfig(retention=RetentionPolicy.PER_CHALLENGE)
        retention = ImageRetention(resources, remove_fn=remove)
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            run_matrix, "capacity_manager", _fake_manager(_handle(retention), captured)
        )

        def fake(cid: str, model: str, **kwargs: Any) -> dict[str, Any]:
            events.append(("run", cid))
            return _outcome()

        monkeypatch.setattr(infer, "run_live_cell", fake)
        cells = _grid_cells(2, ("m0", "m1", "m2"))
        run_matrix.run_live_cells(
            cells, {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=1, resources=resources
        )

        ref_c1 = image_refs_for_ids(["repo__task-1"])[0]
        ref_c2 = image_refs_for_ids(["repo__task-2"])[0]
        # Ordered, de-duplicated refs for the run's challenges (one ref per challenge image).
        assert captured["refs"] == [ref_c1, ref_c2]
        assert events == [
            ("run", "repo__task-1"),
            ("run", "repo__task-1"),
            ("run", "repo__task-1"),
            ("rm", ref_c1),
            ("run", "repo__task-2"),
            ("run", "repo__task-2"),
            ("run", "repo__task-2"),
            ("rm", ref_c2),
        ]

    def test_keep_policy_removes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        removed: list[str] = []

        def remove(ref: str) -> int:
            removed.append(ref)
            return 0

        retention = ImageRetention(ResourceConfig(retention=RetentionPolicy.KEEP), remove_fn=remove)
        monkeypatch.setattr(run_matrix, "capacity_manager", _fake_manager(_handle(retention)))
        monkeypatch.setattr(infer, "run_live_cell", lambda cid, model, **k: _outcome())

        run_matrix.run_live_cells(
            _cells(2),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=1,
            resources=ResourceConfig(retention=RetentionPolicy.KEEP),
        )
        assert removed == []


class TestMemoryGuard:
    def test_cap_reduces_the_pool_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(run_matrix, "capacity_manager", _fake_manager(_handle()))
        # 6.5 GiB available - 2 GiB reserve = 4.5 GiB headroom; / 1.5 GiB bound = 3 workers.
        monkeypatch.setattr(run_matrix, "read_mem_available_bytes", lambda: 6 * GIB + GIB // 2)
        captured: dict[str, Any] = {}

        def fake_parallel(cells: Any, ctx: Any, workers: int, *a: Any, **k: Any) -> list[dict]:
            captured["workers"] = workers
            return []

        monkeypatch.setattr(run_matrix, "_run_cells_parallel", fake_parallel)
        run_matrix.run_live_cells(
            _grid_cells(1, ("m0", "m1", "m2", "m3")),
            {},
            _HASHES,
            _VERSIONS,
            timeout=10,
            verbose=False,
            workers=8,
            resources=ResourceConfig(),
            memory=MemoryConfig(),
        )
        assert captured["workers"] == 3  # requested 8, capped to what memory holds

    def test_refusal_happens_before_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(run_matrix, "capacity_manager", _fake_manager(_handle()))
        monkeypatch.setattr(run_matrix, "read_mem_available_bytes", lambda: 1 * GIB)
        launched: list[Any] = []

        def fake_pool(*a: Any, **k: Any) -> list[dict]:
            launched.append("pool")
            return []

        def fake_cell(cid: str, model: str, **k: Any) -> dict[str, Any]:
            launched.append(("cell", cid))
            return _outcome()

        monkeypatch.setattr(run_matrix, "_run_cells_parallel", fake_pool)
        monkeypatch.setattr(run_matrix, "_run_cells_serial", fake_pool)
        monkeypatch.setattr(infer, "run_live_cell", fake_cell)

        with pytest.raises(InsufficientMemoryError):
            run_matrix.run_live_cells(
                _cells(2),
                {},
                _HASHES,
                _VERSIONS,
                timeout=10,
                verbose=False,
                workers=4,
                resources=ResourceConfig(),
                memory=MemoryConfig(),
            )
        assert launched == []  # no pool built, no container started


class TestBackwardCompatibility:
    def test_no_resources_never_touches_capacity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("capacity manager must not run for a direct/library caller")

        monkeypatch.setattr(run_matrix, "capacity_manager", boom)
        monkeypatch.setattr(infer, "run_live_cell", lambda cid, model, **k: _outcome())
        rows = run_matrix.run_live_cells(
            _cells(2), {}, _HASHES, _VERSIONS, timeout=10, verbose=False, workers=1
        )
        assert len(rows) == 2

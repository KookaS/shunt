"""Strict `resources:` config handling and the `--config`-before-capacity ordering.

Offline: the collectors under test are monkeypatched, so no run, no docker, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from benchmark import config
from benchmark.runner import collect, ladder_collect, run_matrix
from benchmark.runner.capacity.guard import InsufficientDiskError
from benchmark.runner.capacity.types import CapacityVerdict, ResourceConfig, RetentionPolicy
from benchmark.runner.memory_guard import InsufficientMemoryError


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "custom.yaml"
    path.write_text(body)
    return path


# --- ResourceConfig.from_mapping is STRICT ---------------------------------------------


def test_from_mapping_reads_prefetch_alias_and_timeout() -> None:
    cfg = ResourceConfig.from_mapping({"prefetch": True, "prefetch_wait_timeout_s": 30})
    assert cfg.prefetch_enabled is True
    assert cfg.prefetch_wait_timeout_s == 30.0


def test_from_mapping_rejects_an_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown resources key"):
        ResourceConfig.from_mapping({"prefetch_windw": 5})


def test_from_mapping_rejects_a_typo_d_retention() -> None:
    with pytest.raises(ValueError, match="unknown retention"):
        ResourceConfig.from_mapping({"retention": "keepme"})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prefetch_workers", "two"),
        ("prefetch_workers", 2.5),
        ("disk_safety_factor", "lots"),
        ("disk_safety_factor", -1),
        ("prefetch_wait_timeout_s", -5),
        ("allow_insufficient_disk", "false"),
    ],
)
def test_from_mapping_rejects_bad_values(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        ResourceConfig.from_mapping({key: value})


# --- validate_config reports it (non-zero) ---------------------------------------------


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("resources:\n  retention_x: keep\n", "unknown resources key"),
        ("resources:\n  retention: keepme\n", "unknown retention"),
        ("resources:\n  disk_reserve_gib: many\n", "disk_reserve_gib"),
        ("resources:\n  container_memory_gib: -1\n", "container_memory_gib"),
    ],
)
def test_validate_flags_a_bad_resources_block(tmp_path: Path, body: str, needle: str) -> None:
    path = _write_config(tmp_path, "models: []\n" + body)
    errors = config.validate(str(path))
    assert any(needle in error for error in errors), errors


def test_validate_accepts_the_shipped_resources_block() -> None:
    errors = config.validate("benchmark/benchmark.yaml")
    assert errors == []


# --- --config resources beat the defaults (ordering fix) --------------------------------


def _capture_run(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake(config_path: str, **kwargs: Any) -> int:
        captured["resources"] = kwargs["resources"]
        captured["memory"] = kwargs["memory"]
        return 0

    monkeypatch.setattr(module, name, fake)
    return captured


def test_collect_main_builds_resources_after_loading_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(
        tmp_path,
        "resources:\n  retention: per-challenge\n  prefetch_wait_timeout_s: 42\n"
        "  container_memory_gib: 4\n",
    )
    captured = _capture_run(monkeypatch, collect, "run_collect")
    monkeypatch.setattr(sys, "argv", ["collect", "--config", str(path)])
    assert collect.main(str(path)) == 0
    assert captured["resources"].retention is RetentionPolicy.PER_CHALLENGE
    assert captured["resources"].prefetch_wait_timeout_s == 42.0
    assert captured["memory"].container_gib == 4.0


def test_ladder_main_builds_resources_after_loading_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(
        tmp_path, "resources:\n  retention: per-challenge\n  memory_reserve_gib: 3\n"
    )
    captured = _capture_run(monkeypatch, ladder_collect, "run_ladder")
    monkeypatch.setattr(sys, "argv", ["ladder", "--config", str(path)])
    assert ladder_collect.main(str(path)) == 0
    assert captured["resources"].retention is RetentionPolicy.PER_CHALLENGE
    assert captured["memory"].reserve_gib == 3.0


# --- capacity refusals exit non-zero with a verdict, no traceback -----------------------


class _ExplodingDispatch:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __call__(self, args: object) -> int:
        raise self._exc


def test_run_matrix_main_reports_a_disk_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    verdict = CapacityVerdict(
        ok=False,
        needed_bytes=1,
        free_bytes=0,
        shortfall_bytes=1,
        per_image=(),
        docker_root="/",
        remedies=("free space",),
    )
    monkeypatch.setattr(run_matrix, "_dispatch", _ExplodingDispatch(InsufficientDiskError(verdict)))
    monkeypatch.setattr(sys, "argv", ["run_matrix", "--config", "benchmark/benchmark.yaml"])
    assert run_matrix.main() == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err and "free space" in err and "Traceback" not in err


def test_run_matrix_main_reports_a_memory_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exc = InsufficientMemoryError(requested=4, effective=0, available_bytes=1, bound_gib=1.5)
    monkeypatch.setattr(run_matrix, "_dispatch", _ExplodingDispatch(exc))
    monkeypatch.setattr(sys, "argv", ["run_matrix", "--config", "benchmark/benchmark.yaml"])
    assert run_matrix.main() == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err and "insufficient memory" in err and "Traceback" not in err

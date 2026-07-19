"""Tests for routing metrics — the discriminating-sample (routing-signal) breakdown."""

from __future__ import annotations

from typing import Final

from benchmark.routing.metrics import (
    discriminating_set,
    discriminating_stats,
    task_signal,
)

MODELS: Final = ("cheap", "frontier")
ARMS: Final[dict[str, str]] = {"cheap": "off", "frontier": "high"}


def _cell(passed: bool) -> dict[str, dict]:
    return {"pass": passed, "cost": 0.01}


def _task(cheap_pass: bool, frontier_pass: bool) -> dict[str, dict[str, dict]]:
    return {
        "cheap": {"off": _cell(cheap_pass)},
        "frontier": {"high": _cell(frontier_pass)},
    }


class TestDiscriminatingStats:
    def test_empty_input(self):
        assert discriminating_stats({}, [], MODELS, ARMS) == {
            "n_tasks": 0,
            "n_fully_covered": 0,
            "n_all_pass": 0,
            "n_all_fail": 0,
            "n_discriminating": 0,
        }

    def test_all_pass_task_is_not_discriminating(self):
        results = {"t1": _task(True, True)}
        stats = discriminating_stats(results, ["t1"], MODELS, ARMS)
        assert stats["n_fully_covered"] == 1
        assert stats["n_all_pass"] == 1
        assert stats["n_discriminating"] == 0

    def test_all_fail_task_is_not_discriminating(self):
        results = {"t1": _task(False, False)}
        stats = discriminating_stats(results, ["t1"], MODELS, ARMS)
        assert stats["n_fully_covered"] == 1
        assert stats["n_all_fail"] == 1
        assert stats["n_discriminating"] == 0

    def test_mixed_task_is_discriminating(self):
        results = {"t1": _task(True, False)}
        stats = discriminating_stats(results, ["t1"], MODELS, ARMS)
        assert stats["n_fully_covered"] == 1
        assert stats["n_discriminating"] == 1
        assert stats["n_all_pass"] == 0
        assert stats["n_all_fail"] == 0

    def test_partially_covered_task_excluded_from_fully_covered(self):
        # frontier has no default-arm ("high") outcome — only a non-default arm.
        results = {"t1": {"cheap": {"off": _cell(True)}, "frontier": {"low": _cell(True)}}}
        stats = discriminating_stats(results, ["t1"], MODELS, ARMS)
        assert stats["n_tasks"] == 1  # has some coverage
        assert stats["n_fully_covered"] == 0
        assert stats["n_discriminating"] == 0

    def test_missing_model_excluded_from_fully_covered(self):
        results = {"t1": {"cheap": {"off": _cell(True)}}}
        stats = discriminating_stats(results, ["t1"], MODELS, ARMS)
        assert stats["n_tasks"] == 1
        assert stats["n_fully_covered"] == 0

    def test_task_with_no_coverage_not_counted(self):
        stats = discriminating_stats({}, ["t1"], MODELS, ARMS)
        assert stats["n_tasks"] == 0

    def test_realistic_small_matrix(self):
        results = {
            "all_pass": _task(True, True),
            "all_fail": _task(False, False),
            "mixed_a": _task(True, False),
            "mixed_b": _task(False, True),
            "partial": {"cheap": {"off": _cell(True)}},  # frontier missing
        }
        tasks = ["all_pass", "all_fail", "mixed_a", "mixed_b", "partial"]
        stats = discriminating_stats(results, tasks, MODELS, ARMS)
        assert stats == {
            "n_tasks": 5,
            "n_fully_covered": 4,
            "n_all_pass": 1,
            "n_all_fail": 1,
            "n_discriminating": 2,
        }

    def test_no_models_yields_zero_fully_covered(self):
        results = {"t1": _task(True, True)}
        stats = discriminating_stats(results, ["t1"], [], ARMS)
        assert stats["n_fully_covered"] == 0


class TestTaskSignal:
    def test_signals_match_the_four_classes(self):
        assert task_signal(_task(True, True), MODELS, ARMS) == "all_pass"
        assert task_signal(_task(False, False), MODELS, ARMS) == "all_fail"
        assert task_signal(_task(True, False), MODELS, ARMS) == "discriminating"

    def test_missing_cell_is_uncovered(self):
        assert task_signal({"cheap": {"off": _cell(True)}}, MODELS, ARMS) == "uncovered"

    def test_no_models_is_uncovered(self):
        assert task_signal(_task(True, True), [], ARMS) == "uncovered"


class TestDiscriminatingSet:
    def test_membership_and_counts_never_diverge(self):
        results = {
            "all_pass": _task(True, True),
            "all_fail": _task(False, False),
            "mixed_a": _task(True, False),
            "mixed_b": _task(False, True),
            "partial": {"cheap": {"off": _cell(True)}},
        }
        tasks = ["all_pass", "all_fail", "mixed_a", "mixed_b", "partial"]
        d_set, u_set = discriminating_set(results, tasks, MODELS, ARMS)
        stats = discriminating_stats(results, tasks, MODELS, ARMS)
        # The S5 regression guard: the set sizes equal the counter's numbers exactly.
        assert d_set == {"mixed_a", "mixed_b"}
        assert u_set == {"all_pass", "all_fail"}
        assert len(d_set) == stats["n_discriminating"]
        assert len(u_set) == stats["n_all_pass"] + stats["n_all_fail"]

    def test_uncovered_tasks_are_in_neither_set(self):
        results = {"partial": {"cheap": {"off": _cell(True)}}}
        d_set, u_set = discriminating_set(results, ["partial"], MODELS, ARMS)
        assert d_set == set()
        assert u_set == set()

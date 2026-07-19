"""CLI-surface tests for the unified benchmark runner."""

# Covers strategy dispatch (default cost_optimal, opt-in full), the full --live
# uncapped-spend confirmation, and the deprecated collect alias. No live/paid path
# is exercised — dispatch targets are stubbed, so nothing touches results.csv.

from __future__ import annotations

import sys
from argparse import Namespace
from typing import Final

from benchmark.runner import collect, run_matrix

_CONFIG: Final = "benchmark/config.yaml"


def _args(**over: object) -> Namespace:
    """A parsed-args stand-in with the runner's defaults, overridable per test."""
    base = dict(
        strategy="cost_optimal", config=_CONFIG, live=False, timeout=600, workers=1, max_cost=None
    )
    base.update(over)
    return Namespace(**base)


class TestStrategyDefault:
    def test_parser_defaults_to_cost_optimal(self):
        import argparse

        ap = argparse.ArgumentParser()
        run_matrix._add_args(ap, _CONFIG)
        assert ap.parse_args([]).strategy == "cost_optimal"

    def test_default_dispatch_calls_run_collect_not_full(self, monkeypatch):
        seen: list[str] = []

        def stub_collect(cfg: str, **kw: object) -> int:
            seen.append("collect")
            return 0

        monkeypatch.setattr(collect, "run_collect", stub_collect)
        monkeypatch.setattr(run_matrix, "_run_full", lambda a: seen.append("full") or 0)
        assert run_matrix._dispatch(_args()) == 0
        assert seen == ["collect"]


class TestStrategyFull:
    def test_full_dispatch_calls_run_full(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(run_matrix, "_run_full", lambda a: sentinel)
        assert run_matrix._dispatch(_args(strategy="full")) is sentinel

    def test_full_with_max_cost_skips_confirm(self, monkeypatch):
        # A capped live full run must not prompt — the confirm is only for uncapped spend.
        def boom() -> bool:
            raise AssertionError("confirm must not be called when --max-cost is set")

        monkeypatch.setattr(run_matrix, "_confirm_uncapped_live", boom)
        monkeypatch.setattr(run_matrix, "_run_full", lambda a: 0)
        assert run_matrix._dispatch(_args(strategy="full", live=True, max_cost=5.0)) == 0


class TestUncappedLiveConfirm:
    def _dispatch_full_live(self, monkeypatch, answer: str | None) -> int:
        monkeypatch.setattr(run_matrix, "_prompt_confirm", lambda _p: answer)
        monkeypatch.setattr(run_matrix, "_run_full", lambda a: 0)
        return run_matrix._dispatch(_args(strategy="full", live=True, max_cost=None))

    def test_yes_reaches_run_full(self, monkeypatch):
        assert self._dispatch_full_live(monkeypatch, "y") == 0

    def test_yes_uppercase_and_whitespace(self, monkeypatch):
        assert self._dispatch_full_live(monkeypatch, "  Y  ") == 0

    def test_no_aborts_nonzero(self, monkeypatch):
        assert self._dispatch_full_live(monkeypatch, "n") == 3

    def test_invalid_aborts_nonzero(self, monkeypatch):
        assert self._dispatch_full_live(monkeypatch, "x") == 3

    def test_non_interactive_aborts_nonzero(self, monkeypatch):
        # _prompt_confirm returns None for a non-TTY / EOF — a safe abort (never spends in CI).
        assert self._dispatch_full_live(monkeypatch, None) == 3

    def test_prompt_confirm_returns_none_when_not_a_tty(self, monkeypatch):
        monkeypatch.setattr(run_matrix.sys.stdin, "isatty", lambda: False)
        assert run_matrix._prompt_confirm("? ") is None


class TestCollectAlias:
    def test_alias_delegates_and_warns(self, monkeypatch, capsys):
        calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(sys, "argv", ["collect"])
        monkeypatch.setattr(collect, "run_collect", lambda cfg, **kw: calls.append((cfg, kw)) or 0)
        assert collect.main() == 0
        assert calls and calls[0][0] == _CONFIG
        assert "DEPRECATED" in capsys.readouterr().err

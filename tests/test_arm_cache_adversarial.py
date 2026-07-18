"""Adversarial stress-tests for the (challenge, model, reasoning-arm) outcome cache."""

# Ruthlessly probes the (challenge_id, model, reasoning) 3-tuple key, the
# arm_hash staleness anchor, legacy "default" aliasing, and select_arms
# determinism — the benchmark's "never recompute a paid cell / never
# double-spend" guarantee. Offline, deterministic.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

import pytest

from benchmark import config
from benchmark.routing import integrity
from benchmark.runner import run_matrix, sampling
from shunt.models.config import ReasoningArm, ReasoningConfig

_PRICING: Final[dict[str, dict[str, float]]] = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28}
}
_HASHES: Final[dict[str, str]] = {"c1": "a" * 64}
_VERSIONS: Final[dict[str, str]] = {"deepseek-v4-flash": "deepseek-v4-flash"}


def _outcome(**over: object) -> dict:
    base: dict[str, object] = {
        "pass": True,
        "in_tok": 100,
        "out_tok": 50,
        "calls": 3,
        "real_cost": 0.003,
        "timeout_flag": False,
        "image_digest": "sha256:abc",
        "computed_at": "2026-07-15T18:50:14+00:00",
    }
    base.update(over)
    return base


def _arm_map() -> dict[str, dict[str, str]]:
    """The real deepseek arm-hash anchors from the shipped registry."""
    config.load("benchmark/config.yaml")
    return integrity.arm_hashes(config.resolved_models())


def _row(arm: str, arm_map: dict[str, dict[str, str]], **over: object) -> dict:
    """A results row for deepseek at ``arm``, stamped with the real arm_hash anchor."""
    return run_matrix._build_row(
        "c1",
        "deepseek-v4-flash",
        _outcome(**over),
        _HASHES,
        _VERSIONS,
        _PRICING,
        None,
        arm,
        arm_map,
    )


def _bracket(default_arm: str, *ranked_ids: str) -> ReasoningConfig:
    """Synthetic bracket: arms ranked 0..n in id order, empty api params."""
    arms = [ReasoningArm(id=i, rank=r, api={}) for r, i in enumerate(ranked_ids)]
    return ReasoningConfig(default_arm=default_arm, arms=arms)


# ---------------------------------------------------------------------------
# Case 1 — run-twice-zero-recompute on the 3-tuple (proper arm_hash stamped).
# ---------------------------------------------------------------------------


class TestRunTwiceZeroRecompute:
    def test_single_arm_round_trip_is_present(self, tmp_path):
        arm_map = _arm_map()
        res = tmp_path / "results.csv"
        run_matrix.merge_rows([_row("high", arm_map)], res, tmp_path / "h.csv")
        cache = config.load_results(res)
        sel = {("c1", "deepseek-v4-flash"): ["high"]}
        st = run_matrix.classify_cells(
            ["c1"], ["deepseek-v4-flash"], cache, _HASHES, _VERSIONS, None, sel, arm_map
        )
        assert st.present == 1
        assert st.to_run == []

    def test_multiple_arms_per_cell_recompute_none(self, tmp_path):
        arm_map = _arm_map()
        res = tmp_path / "results.csv"
        hist = tmp_path / "h.csv"
        run_matrix.merge_rows([_row("high", arm_map), _row("none", arm_map)], res, hist)
        cache = config.load_results(res)
        assert sorted(cache["c1"]["deepseek-v4-flash"]) == ["high", "none"]
        sel = {("c1", "deepseek-v4-flash"): ["high", "none"]}
        st = run_matrix.classify_cells(
            ["c1"], ["deepseek-v4-flash"], cache, _HASHES, _VERSIONS, None, sel, arm_map
        )
        assert st.present == 2
        assert st.to_run == []
        assert not hist.exists(), "a clean multi-arm re-run must archive nothing"


# ---------------------------------------------------------------------------
# Case 2 — arm collision: two arms of one (challenge, model) are DISTINCT cells.
# This was the original 2-tuple bug (overwrite + spurious history). Prove fixed.
# ---------------------------------------------------------------------------


class TestArmCollisionNoOverwrite:
    def test_second_arm_does_not_overwrite_or_archive_sibling(self, tmp_path):
        arm_map = _arm_map()
        res = tmp_path / "results.csv"
        hist = tmp_path / "h.csv"
        run_matrix.merge_rows([_row("high", arm_map, real_cost=0.11)], res, hist)
        run_matrix.merge_rows([_row("none", arm_map, real_cost=0.22)], res, hist)
        assert not hist.exists(), "a NEW arm must never archive a sibling arm as supersession"
        cache = config.load_results(res)
        arms = cache["c1"]["deepseek-v4-flash"]
        assert set(arms) == {"high", "none"}
        assert arms["high"]["real_cost"] == pytest.approx(0.11)
        assert arms["none"]["real_cost"] == pytest.approx(0.22)

    def test_distinct_arm_keys_in_row_key(self):
        k_high = run_matrix._row_key({"challenge_id": "c1", "model": "m", "reasoning": "high"})
        k_none = run_matrix._row_key({"challenge_id": "c1", "model": "m", "reasoning": "none"})
        assert k_high != k_none
        assert k_high == ("c1", "m", "high")


# ---------------------------------------------------------------------------
# Case 3 — arm_hash staleness: changed api params recompute; unchanged do not.
# ---------------------------------------------------------------------------


class TestArmHashStaleness:
    def _cell(self, arm_hash: str) -> dict:
        return {
            "version_hash": "h1",
            "model_version": "deepseek-v4-flash",
            "image_digest": "",
            "arm_hash": arm_hash,
        }

    def _classify(self, cell: dict, arm_map: dict[str, dict[str, str]]):
        cache = {"c1": {"deepseek-v4-flash": {"high": cell}}}
        sel = {("c1", "deepseek-v4-flash"): ["high"]}
        return run_matrix.classify_cells(
            ["c1"], ["deepseek-v4-flash"], cache, {"c1": "h1"}, _VERSIONS, None, sel, arm_map
        )

    def test_unchanged_arm_params_is_present(self):
        arm_map = _arm_map()
        st = self._classify(self._cell(arm_map["deepseek-v4-flash"]["high"]), arm_map)
        assert st.present == 1
        assert st.to_run == []

    def test_changed_arm_params_is_stale(self):
        # Stored the anchor for a DIFFERENT arm ("max") under the "high" cell: the
        # api params moved, so the high cell must recompute rather than serve stale.
        arm_map = _arm_map()
        st = self._classify(self._cell(arm_map["deepseek-v4-flash"]["max"]), arm_map)
        assert st.stale == [("c1", "deepseek-v4-flash", "high")]

    def test_no_anchor_map_never_stales_on_arm_axis(self):
        # arm_hash_map absent (offline classify) ⇒ arm axis skipped, never stale.
        st = self._classify(self._cell("whatever"), None)
        assert st.present == 1


# ---------------------------------------------------------------------------
# Case 4 — aliasing safety: no two DISTINCT arm ids share identical api params
# (would double-spend on duplicate bandit arms scoring identically). Real registry.
# ---------------------------------------------------------------------------


class TestAliasingSafety:
    def test_no_api_duplicate_arms_within_any_model(self):
        config.load("benchmark/config.yaml")
        for name, model in config.resolved_models().items():
            if model.reasoning is None:
                continue
            hashes = [integrity.arm_hash_value(model, a.id) for a in model.reasoning.arms]
            assert len(hashes) == len(set(hashes)), (
                f"{name}: two arms resolve to identical api params (silent duplicate arm)"
            )

    def test_deepseek_low_med_excluded_from_registry(self):
        # DeepSeek low/med alias to high and must NOT be distinct arms.
        config.load("benchmark/config.yaml")
        ds = config.resolved_models()["deepseek-v4-flash"]
        assert ds.reasoning is not None
        ids = {a.id for a in ds.reasoning.arms}
        assert "low" not in ids and "med" not in ids
        assert ids == {"nothink", "high", "max"}

    def test_every_default_arm_matches_a_declared_arm(self):
        config.load("benchmark/config.yaml")
        for name, model in config.resolved_models().items():
            if model.reasoning is None:
                continue
            ids = {a.id for a in model.reasoning.arms}
            assert model.reasoning.default_arm in ids, f"{name}: default_arm not a declared arm"


# ---------------------------------------------------------------------------
# Case 5 — legacy migration: reasoning="default" alias-resolves at READ time to the
# model's real default_arm, no results.csv rewrite; collision behaviour documented.
# ---------------------------------------------------------------------------


class TestLegacyMigration:
    def _legacy_csv(self, tmp_path, rows: list[dict]) -> Path:
        p = tmp_path / "results.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(integrity.RESULTS_FIELDS))
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in integrity.RESULTS_FIELDS})
        return p

    def _row(self, reasoning: str, **over: object) -> dict:
        base: dict[str, object] = {
            "challenge_id": "c1",
            "model": "deepseek-v4-flash",
            "reasoning": reasoning,
            "pass": "True",
            "cost": "0.5",
            "version_hash": "a" * 64,
            "model_version": "deepseek-v4-flash",
        }
        base.update(over)
        return base

    def test_legacy_default_aliases_to_model_default_arm(self, tmp_path):
        config.load("benchmark/config.yaml")
        p = self._legacy_csv(tmp_path, [self._row("default")])
        cache = config.load_results(p)
        # deepseek's declared default_arm is "high": the legacy row resolves there.
        assert list(cache["c1"]["deepseek-v4-flash"]) == ["high"]

    def test_legacy_row_unrewritten_on_disk(self, tmp_path):
        # Aliasing is read-time only: the CSV on disk still literally says "default".
        config.load("benchmark/config.yaml")
        p = self._legacy_csv(tmp_path, [self._row("default")])
        config.load_results(p)
        assert "deepseek-v4-flash,default," in p.read_text()

    def test_explicit_arm_row_not_aliased(self, tmp_path):
        config.load("benchmark/config.yaml")
        p = self._legacy_csv(tmp_path, [self._row("none")])
        cache = config.load_results(p)
        assert list(cache["c1"]["deepseek-v4-flash"]) == ["none"]

    def test_legacy_and_explicit_default_arm_collide_at_read_deterministically(self, tmp_path):
        # A legacy "default" row AND an explicit "high" row for the same cell both map
        # to arm key "high" at read; results.csv keeps both (distinct write keys), but
        # load_results collapses them. Documented invariant: the resolution is
        # deterministic and prefers the explicit arm's data (sorted-write: default<high,
        # explicit read last-wins). No paid cell is LOST on disk (both rows persist).
        config.load("benchmark/config.yaml")
        arm_map = _arm_map()
        rows = [
            self._row("default", cost="0.99"),
            self._row("high", cost="0.11", arm_hash=arm_map["deepseek-v4-flash"]["high"]),
        ]
        p = self._legacy_csv(tmp_path, sorted(rows, key=lambda r: r["reasoning"]))
        cache = config.load_results(p)
        assert list(cache["c1"]["deepseek-v4-flash"]) == ["high"]
        assert cache["c1"]["deepseek-v4-flash"]["high"]["cost"] == pytest.approx(0.11)


# ---------------------------------------------------------------------------
# Case 6 — select_arms determinism / cache-stability (no RNG, default always in).
# ---------------------------------------------------------------------------


class TestSelectArmsDeterminism:
    def test_rerun_over_registry_selects_identical_sets(self):
        config.load("benchmark/config.yaml")
        cfgs = config.reasoning_configs()
        weights = config.arm_sampling_weights()
        ids = [f"repo__task-{i}" for i in range(200)]
        models = config.enabled_models()

        def sweep() -> dict[tuple[str, str], list[str]]:
            out: dict[tuple[str, str], list[str]] = {}
            for m in models:
                for cid in ids:
                    out[(cid, m)] = sampling.select_arms(cid, m, cfgs[m], weights)
            return out

        assert sweep() == sweep()

    def test_default_arm_always_selected_every_model(self):
        config.load("benchmark/config.yaml")
        cfgs = config.reasoning_configs()
        weights = config.arm_sampling_weights()
        for m in config.enabled_models():
            for cid in ("astropy__x-1", "django__y-2", "flask__z-3"):
                assert cfgs[m].default_arm in sampling.select_arms(cid, m, cfgs[m], weights)

    def test_selection_is_a_subset_of_the_bracket(self):
        config.load("benchmark/config.yaml")
        cfgs = config.reasoning_configs()
        weights = config.arm_sampling_weights()
        for m in config.enabled_models():
            bracket_ids = {a.id for a in cfgs[m].arms}
            arms = sampling.select_arms("some__challenge-9", m, cfgs[m], weights)
            assert set(arms) <= bracket_ids
            assert len(arms) == len(set(arms)), "no arm may be selected twice"


# ---------------------------------------------------------------------------
# Case 7 — edge cases: empty cache, single-arm bracket, unicode ids, round-trip.
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_results_csv_is_empty_cache(self, tmp_path):
        assert config.load_results(tmp_path / "absent.csv") == {}

    def test_single_arm_bracket_selects_only_that_arm(self):
        # kimi-k3 shape: one always-on arm which is also the default.
        bracket = _bracket("max", "max")
        for cid in ("a", "b", "c"):
            assert sampling.select_arms(cid, "kimi-k3", bracket, [0.5]) == ["max"]

    def test_unicode_challenge_id_round_trips_and_selects(self, tmp_path):
        arm_map = _arm_map()
        cid = "проект__tëst-\U0001f600-42"
        row = run_matrix._build_row(
            cid,
            "deepseek-v4-flash",
            _outcome(),
            {cid: "b" * 64},
            _VERSIONS,
            _PRICING,
            None,
            "high",
            arm_map,
        )
        res = tmp_path / "results.csv"
        run_matrix.merge_rows([row], res, tmp_path / "h.csv")
        cache = config.load_results(res)
        assert (
            cache[cid]["deepseek-v4-flash"]["high"]["arm_hash"]
            == arm_map["deepseek-v4-flash"]["high"]
        )
        bracket = _bracket("high", "none", "high")
        arms = sampling.select_arms(cid, "deepseek-v4-flash", bracket, [0.5])
        assert "high" in arms

    def test_csv_round_trip_preserves_three_tuple_and_arm_hash(self, tmp_path):
        arm_map = _arm_map()
        res = tmp_path / "results.csv"
        rows = [_row("high", arm_map), _row("none", arm_map)]
        run_matrix.merge_rows(rows, res, tmp_path / "h.csv")
        raw = run_matrix._read_raw_rows(res)
        assert set(raw) == {
            ("c1", "deepseek-v4-flash", "high"),
            ("c1", "deepseek-v4-flash", "none"),
        }
        assert (
            raw[("c1", "deepseek-v4-flash", "high")]["arm_hash"]
            == arm_map["deepseek-v4-flash"]["high"]
        )

    def test_challenge_present_for_one_arm_missing_another(self):
        arm_map = _arm_map()
        cell = {
            "version_hash": "h1",
            "model_version": "deepseek-v4-flash",
            "image_digest": "",
            "arm_hash": arm_map["deepseek-v4-flash"]["high"],
        }
        cache = {"c1": {"deepseek-v4-flash": {"high": cell}}}
        sel = {("c1", "deepseek-v4-flash"): ["high", "none"]}
        st = run_matrix.classify_cells(
            ["c1"], ["deepseek-v4-flash"], cache, {"c1": "h1"}, _VERSIONS, None, sel, arm_map
        )
        assert st.present == 1
        assert st.missing == [("c1", "deepseek-v4-flash", "none")]

    def test_merge_row_order_independent(self, tmp_path):
        arm_map = _arm_map()
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        rows_ab = [_row("high", arm_map), _row("none", arm_map)]
        rows_ba = [_row("none", arm_map), _row("high", arm_map)]
        run_matrix.merge_rows(rows_ab, a, tmp_path / "ha.csv")
        run_matrix.merge_rows(rows_ba, b, tmp_path / "hb.csv")
        msg = "results.csv must not depend on merge input order"
        assert a.read_text() == b.read_text(), msg


# ---------------------------------------------------------------------------
# Case 8 — schema: RESULTS_FIELDS carries reasoning + arm_hash; an old file with no
# arm_hash column degrades gracefully (no crash) rather than raising.
# ---------------------------------------------------------------------------


class TestSchema:
    def test_results_fields_carry_reasoning_and_arm_hash(self):
        assert "reasoning" in integrity.RESULTS_FIELDS
        assert "arm_hash" in integrity.RESULTS_FIELDS
        assert integrity.RESULTS_FIELDS[:3] == ("challenge_id", "model", "reasoning")

    def test_old_file_without_arm_hash_column_loads_without_crashing(self, tmp_path):
        # The committed legacy header has no arm_hash column. load_results must not
        # crash and must default the missing anchor to "".
        old_fields = tuple(c for c in integrity.RESULTS_FIELDS if c != "arm_hash")
        p = tmp_path / "results.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(old_fields))
            w.writeheader()
            w.writerow(
                {
                    "challenge_id": "c1",
                    "model": "deepseek-v4-flash",
                    "reasoning": "default",
                    "pass": "True",
                    "cost": "0.5",
                    "version_hash": "a" * 64,
                    "model_version": "deepseek-v4-flash",
                }
            )
        config.load("benchmark/config.yaml")
        cache = config.load_results(p)
        assert cache["c1"]["deepseek-v4-flash"]["high"]["arm_hash"] == ""


# ---------------------------------------------------------------------------
# REGRESSION GUARD (fixed 2026-07-17) — the arm_hash anchor must mirror the
# empty-stored guard in `_image_stale` (run_matrix.py:148-149). Before the fix,
# `_arm_stale` returned `str(cell.get("arm_hash","")) != expected`, so a legacy
# cell (predating the arm-hash column) with an EMPTY stored arm_hash always
# mismatched and was marked STALE. On the real main() path (which passes
# arm_hash_map) that recomputed all 69 committed legacy cells, re-spending the
# whole paid cache on byte-identical requests (one fixed request per
# (instance,model) regardless of arm). Fix mirrors
# `_image_stale`: `stored = str(cell.get("arm_hash","")); return bool(stored) and ...`.
# ---------------------------------------------------------------------------


class TestArmHashEmptyGuardRegression:
    def test_empty_stored_arm_hash_must_not_stale_paid_cell(self):
        # Mirrors test_integrity.py::test_empty_stored_digest_is_not_stale for the arm axis.
        arm_map = _arm_map()
        cell = {
            "version_hash": "h1",
            "model_version": "deepseek-v4-flash",
            "image_digest": "",
            "arm_hash": "",  # legacy row (pre-arm-hash): anchor was never recorded
        }
        cache = {"c1": {"deepseek-v4-flash": {"high": cell}}}
        sel = {("c1", "deepseek-v4-flash"): ["high"]}
        st = run_matrix.classify_cells(
            ["c1"], ["deepseek-v4-flash"], cache, {"c1": "h1"}, _VERSIONS, None, sel, arm_map
        )
        assert st.present == 1, "an empty stored arm_hash must degrade to no-op, not recompute"
        assert st.to_run == []

    def test_populated_arm_hash_still_stales_on_drift(self):
        # The guard must NOT mask real drift: a NON-EMPTY stored arm_hash that differs
        # from the resolved anchor is still stale (the arm's api params changed).
        arm_map = _arm_map()
        cell = {
            "version_hash": "h1",
            "model_version": "deepseek-v4-flash",
            "image_digest": "",
            "arm_hash": "stale-old-hash",
        }
        cache = {"c1": {"deepseek-v4-flash": {"high": cell}}}
        sel = {("c1", "deepseek-v4-flash"): ["high"]}
        st = run_matrix.classify_cells(
            ["c1"], ["deepseek-v4-flash"], cache, {"c1": "h1"}, _VERSIONS, None, sel, arm_map
        )
        assert st.stale == [("c1", "deepseek-v4-flash", "high")]

"""Invariance contract for the live-evaluation held-out task split.

The split must be a pure function of (task_id, salt, fraction) over the challenges.json
universe — immune to sample_size, seeds, arm weights, run order and collection coverage.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from benchmark import config
from benchmark.routing import live_split
from benchmark.runner import calibration, sampling


class TestNesting:
    def test_raising_the_fraction_only_adds_members(self):
        # I3: 0.2 must be a STRICT subset of 0.3 on the committed universe.
        small = set(live_split.holdout_tasks(fraction=0.2))
        large = set(live_split.holdout_tasks(fraction=0.3))
        assert small < large
        assert (len(small), len(large)) == (97, 152)


class TestSetIndependence:
    def test_truncated_universe_yields_the_intersection(self):
        # I1/I2: membership never depends on WHICH other tasks were considered.
        full = live_split.universe()
        truncated = full[:250]
        expected = set(live_split.holdout_tasks()) & set(truncated)
        got = {t for t in truncated if live_split.is_holdout(t)}
        assert got == expected
        assert 0 < len(got) < len(live_split.holdout_tasks())


class TestSaltNonAliasing:
    """I4: the fifth salt must not alias any of the four already in use."""

    SALTS = (
        live_split.LIVE_SPLIT_SALT,
        calibration.DEFAULT_SALT,
        sampling.ORDER_SALT,
        sampling.ARM_SALT_PREFIX,
        sampling.AUDIT_SALT,
    )

    def test_the_five_salt_literals_are_pairwise_distinct(self):
        assert len(set(self.SALTS)) == 5
        assert live_split.LIVE_SPLIT_SALT == "live-split-v1"

    @pytest.mark.parametrize("i", range(5))
    @pytest.mark.parametrize("j", range(5))
    def test_pairwise_overlap_matches_independence_not_identity(self, i: int, j: int):
        if i >= j:
            pytest.skip("unordered pairs only")
        ids = live_split.universe()
        f = live_split.DEFAULT_FRACTION
        a = {t for t in ids if calibration.in_calibration_holdout(t, f, self.SALTS[i])}
        b = {t for t in ids if calibration.in_calibration_holdout(t, f, self.SALTS[j])}
        assert a != b, "two salts produced an IDENTICAL draw — they alias"
        # Independent draws overlap at ~n*f^2 = 500*0.04 = 20; identity would give ~97.
        # Band is a generous binomial envelope, wide enough never to flake.
        assert 5 <= len(a & b) <= 45


class TestCommittedManifest:
    def test_digest_and_revision_match_a_freshly_computed_split(self):
        # I5: the regression test that fires when an upstream id or revision changes.
        committed = json.loads(live_split.MANIFEST_PATH.read_text())
        fresh = live_split.split_manifest()
        assert committed["tasks_digest"] == fresh["tasks_digest"]
        assert committed["dataset_revision"] == fresh["dataset_revision"]
        challenges = json.loads(live_split.CHALLENGES_PATH.read_text())
        assert committed["dataset_revision"] == challenges["dataset_revision"]
        assert committed["universe_size"] == len(challenges["challenges"]) == 500
        assert committed["salt"] == live_split.LIVE_SPLIT_SALT
        assert committed["fraction"] == live_split.DEFAULT_FRACTION
        assert sum(committed["per_repo_counts"].values()) == committed["holdout_count"]

    def test_digest_is_sha256_over_the_sorted_ids(self):
        committed = json.loads(live_split.MANIFEST_PATH.read_text())
        ids = committed["tasks"]
        assert ids == sorted(ids)
        expected = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        assert committed["tasks_digest"] == expected


class TestHyperparameterInvariance:
    def test_sample_size_and_seed_do_not_move_the_split(self, monkeypatch):
        # I1: the split is decided before any benchmark hyperparameter is read.
        before = live_split.holdout_tasks()
        mutated = copy.deepcopy(config.get())
        mutated.setdefault("benchmark", {})["sample_size"] = 25
        mutated["benchmark"]["seed"] = 1337
        mutated["models"] = ["deepseek-v4-flash"]
        monkeypatch.setattr(config, "_config", mutated)
        assert config.sample_size() == 25
        assert live_split.holdout_tasks() == before
        assert live_split.is_holdout("astropy__astropy-12907") == (
            "astropy__astropy-12907" in set(before)
        )

    def test_collection_coverage_is_irrelevant(self):
        # I2: the split is over the UNIVERSE, never the 200 collected tasks.
        holdout = set(live_split.holdout_tasks())
        assert len(live_split.universe()) == 500
        assert len(holdout) == 97


class TestHoldoutRatchet:
    def test_no_committed_task_may_leave_the_holdout(self):
        # The holdout may be GROWN but never SHRUNK: live collection was already performed
        # against the ids the committed manifest declares, so a task departing the holdout
        # retroactively invalidates every live cell measured on it. Lowering
        # DEFAULT_FRACTION, renaming an id, or bumping dataset_revision in a way that drops
        # a member must fail here — raising the fraction keeps it passing (superset).
        committed = set(json.loads(live_split.MANIFEST_PATH.read_text())["tasks"])
        today = set(live_split.holdout_tasks(fraction=live_split.DEFAULT_FRACTION))
        departed = sorted(committed - today)
        assert not departed, (
            f"{len(departed)} task(s) left the live holdout, invalidating live results "
            f"already collected against them: {departed}"
        )

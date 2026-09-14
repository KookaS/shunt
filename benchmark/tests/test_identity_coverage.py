"""Owner coverage policy: a collection-only extra skips a challenge its model IDENTITY
already covers under another registry id/channel.

Identity = the registry entry's ``version`` slug (``kimi-k3``). Direct requesty
``kimi-k3`` and free-promo ``kimi-k3-explabs`` are the SAME model on two channels, so
planning ``kimi-k3-explabs`` must not re-run the ~110 challenges direct ``kimi-k3``
already measured — MORE COVERAGE wins over re-running a covered challenge. Models
without a twin (``qwen3.8-27b-explabs``) and the direct/enabled baseline planning are
unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark import config
from benchmark.runner import run_matrix

OVERLAY: Path = Path(__file__).resolve().parents[2] / "configs" / "free-tier" / "overlay.yaml"

KIMI_EXTRA = "kimi-k3-explabs"
KIMI_DIRECT = "kimi-k3"
DS_EXTRA = "deepseek-v4-flash-explabs"
DS_DIRECT = "deepseek-v4-flash"
QWEN_EXTRA = "qwen3.8-27b-explabs"
QWEN_VERSION = "qwen3.8-27b"

C1, C2, C3 = "c1", "c2", "c3"


@pytest.fixture(scope="module")
def _cfg():
    config.load("benchmark/benchmark.yaml")


@pytest.fixture(autouse=True)
def _overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the non-shipped free overlay for every test (collection-only policy)."""
    monkeypatch.setattr(config, "_pricing", None)
    monkeypatch.setattr(config, "_free_registry", None)
    monkeypatch.setattr(config, "_free_registry_path_override", str(OVERLAY))


def _real_row(cid: str, version: str, calls: int = 5) -> dict:
    """A cached row that classifies PRESENT: matches its identity anchors and ran work.

    ``calls`` > 0 keeps it out of ``impute.is_zero_work``, so it is a REAL observation
    that can cover a challenge for its model identity.
    """
    return {
        "reasoning": "default",
        "model_version": version,
        "version_hash": f"h-{cid}",
        "arm_hash": "",
        "image_digest": "",
        "step_limit": "",
        "cost_limit": "",
        "scaffold_version": "",
        "sampling_hash": "",
        "prompt_hash": "",
        "stop_reason": "",
        "calls": calls,
        "real_cost": 0.0,
        "cost": 0.0,
    }


def _cache_with(twin: str, version: str, covered: list[str]) -> dict:
    """A results-cache where ``twin`` (the OTHER channel) holds real rows on ``covered``."""
    return {cid: {twin: {"default": _real_row(cid, version)}} for cid in covered}


def _classify_extra(
    cache: dict, extra: str, twin_version: str, tasks: list[str]
) -> run_matrix.CellStatus:
    """Classify ``extra``'s plan under the identity-coverage skip, all-default-arms."""
    twin = extra.removesuffix("-explabs")
    versions = {extra: twin_version, twin: twin_version}
    selected = {(cid, extra): ["default"] for cid in tasks}
    hashes = {cid: f"h-{cid}" for cid in tasks}
    return run_matrix.classify_cells(
        tasks,
        [extra],
        cache,
        hashes,
        versions,
        None,
        selected,
        None,
        identity_skip_models={extra},
    )


def test_extra_skips_challenge_covered_by_its_direct_twin(_cfg) -> None:
    # c1 is covered by direct requesty kimi-k3; c2 is not covered anywhere.
    cache = _cache_with(KIMI_DIRECT, "kimi-k3", [C1])
    status = _classify_extra(cache, KIMI_EXTRA, "kimi-k3", [C1, C2])
    assert (C1, KIMI_EXTRA, "default") not in status.missing
    assert (C1, KIMI_EXTRA, "default") not in status.stale
    assert (C1, KIMI_EXTRA, "default") not in status.to_run


def test_extra_keeps_challenge_not_covered_anywhere(_cfg) -> None:
    cache = _cache_with(KIMI_DIRECT, "kimi-k3", [C1])
    status = _classify_extra(cache, KIMI_EXTRA, "kimi-k3", [C1, C2])
    assert status.missing == [(C2, KIMI_EXTRA, "default")]


def test_deepseek_extra_skips_challenges_covered_by_direct_deepseek(_cfg) -> None:
    # Only c2 is covered by direct deepseek-v4-flash; c1 and c3 stay missing.
    cache = _cache_with(DS_DIRECT, "deepseek-v4-flash", [C2])
    status = _classify_extra(cache, DS_EXTRA, "deepseek-v4-flash", [C1, C2, C3])
    assert (C2, DS_EXTRA, "default") not in status.missing
    assert sorted(status.missing) == [
        (C1, DS_EXTRA, "default"),
        (C3, DS_EXTRA, "default"),
    ]


def test_twinless_extra_is_unaffected_by_other_identities(_cfg) -> None:
    # Rows under a DIFFERENT identity (claude-fable-5.1-explabs) and under no identity at all
    # must not suppress cells. (Under the bare-slug convention the `qwen3.8-27b-explabs` extra
    # now shares its identity with the paid `qwen3.8-27b` row, so the last assertion pins that
    # alignment rather than twinlessness.)
    from benchmark.routing import integrity

    cache = {
        C1: {"claude-fable-5.1-explabs": {"default": _real_row(C1, "claude-fable-5.1")}},
        C2: {QWEN_EXTRA: {"default": _real_row(C2, QWEN_VERSION)}},
    }
    status = _classify_extra(cache, QWEN_EXTRA, QWEN_VERSION, [C1, C2, C3])
    # c2 is the extra's OWN present row; c1 (foreign identity) and c3 are missing.
    assert status.present == 1
    assert sorted(status.missing) == [
        (C1, QWEN_EXTRA, "default"),
        (C3, QWEN_EXTRA, "default"),
    ]
    # Registry fact backing the test: the bare slug unifies the extra with its paid twin.
    config.register_collection_models([QWEN_EXTRA])
    versions = integrity.model_versions()
    assert {m for m, v in versions.items() if v == QWEN_VERSION} == {QWEN_VERSION, QWEN_EXTRA}


def test_no_other_registry_id_shares_twin_versions(_cfg) -> None:
    # kimi-k3 and deepseek-v4-flash each have EXACTLY one channel-namespaced twin —
    # the explabs id — so the identity rule maps one direct id to one extra.
    from benchmark.routing import integrity

    config.register_collection_models([KIMI_EXTRA, DS_EXTRA])
    versions = integrity.model_versions()
    assert versions[KIMI_DIRECT] == "kimi-k3"
    assert versions[KIMI_EXTRA] == "kimi-k3"
    assert versions[DS_DIRECT] == "deepseek-v4-flash"
    assert versions[DS_EXTRA] == "deepseek-v4-flash"
    assert {m for m, v in versions.items() if v == "kimi-k3"} == {KIMI_DIRECT, KIMI_EXTRA}
    assert {m for m, v in versions.items() if v == "deepseek-v4-flash"} == {
        DS_DIRECT,
        DS_EXTRA,
    }


def test_direct_baseline_planning_is_unchanged(_cfg) -> None:
    # Direct kimi-k3 is never identity-skipped: extra-explabs rows for a challenge do not
    # suppress the direct id's own cell — its presence is still judged per-id, as before.
    cache = {C1: {KIMI_EXTRA: {"default": _real_row(C1, "kimi-k3")}}}
    selected = {("c1", KIMI_DIRECT): ["default"], ("c2", KIMI_DIRECT): ["default"]}
    hashes = {"c1": "h-c1", "c2": "h-c2"}
    versions = {KIMI_DIRECT: "kimi-k3", KIMI_EXTRA: "kimi-k3"}
    status = run_matrix.classify_cells(
        [C1, C2],
        [KIMI_DIRECT],
        cache,
        hashes,
        versions,
        None,
        selected,
        None,
        identity_skip_models={KIMI_EXTRA},
    )
    # Both cells are missing for the direct id: nothing about the extra suppressed them.
    assert sorted(status.missing) == [
        (C1, KIMI_DIRECT, "default"),
        (C2, KIMI_DIRECT, "default"),
    ]


def test_extra_own_zero_work_row_does_not_suppress_recollect(_cfg) -> None:
    # A zero-work row never executed; without a twin's REAL row it must NOT count as
    # coverage, or the residue would strand the challenge permanently uncollected.
    cache = {C1: {KIMI_EXTRA: {"default": _real_row(C1, "kimi-k3", calls=0)}}}
    status = _classify_extra(cache, KIMI_EXTRA, "kimi-k3", [C1])
    # The extra's own residue is stale (zero-work), and no twin covers c1 -> collectable.
    assert (C1, KIMI_EXTRA, "default") in status.stale


def test_extra_stale_cell_is_dropped_when_twin_covers(_cfg) -> None:
    # The extra's own cell went stale (drifted identity hash) but direct kimi-k3 holds a
    # real row for the challenge -> the refresh is a re-run of a covered challenge: skip.
    stale = _real_row(C1, "kimi-k3")
    stale["version_hash"] = "stale-hash"
    cache = {
        C1: {KIMI_EXTRA: {"default": stale}, KIMI_DIRECT: {"default": _real_row(C1, "kimi-k3")}}
    }
    status = _classify_extra(cache, KIMI_EXTRA, "kimi-k3", [C1])
    assert (C1, KIMI_EXTRA, "default") not in status.stale
    assert (C1, KIMI_EXTRA, "default") not in status.to_run


def test_synthesized_extra_skips_challenge_covered_by_direct_twin(_cfg) -> None:
    # A runtime-synthesized `gpt-5-mini-explabs` (no registry row) takes identity
    # gpt-5-mini, so the identity-skip dedupes it against the direct twin exactly like a
    # hand-registered extra. `_classify_extra` derives the twin as the `-explabs` prefix.
    cache = _cache_with("gpt-5-mini", "gpt-5-mini", [C1])
    status = _classify_extra(cache, "gpt-5-mini-explabs", "gpt-5-mini", [C1, C2])
    assert (C1, "gpt-5-mini-explabs", "default") not in status.missing
    assert status.missing == [(C2, "gpt-5-mini-explabs", "default")]


# --- the fan-out cap: cap=1 is the oracle, higher caps admit a concordance subset -------


def _coverage(identity: str, others: list[str]) -> dict[tuple[str, str], set[str]]:
    return {(C1, identity): set(others)}


@pytest.mark.parametrize("n_others", [0, 1, 2, 3, 5])
def test_fanout_cap_one_is_byte_identical_to_covered_elsewhere(n_others: int) -> None:
    version = "kimi-k3"
    others = [f"channel-{i}" for i in range(n_others)]
    coverage = _coverage(version, others)
    versions = {"extra-explabs": version}
    expected = any(other != "extra-explabs" for other in others)
    assert (
        run_matrix._identity_fanout_reached(C1, "extra-explabs", versions, coverage, 1) is expected
    )
    assert run_matrix._covered_elsewhere(C1, "extra-explabs", versions, coverage) is expected


def test_higher_cap_admits_a_concordance_subset_until_the_ceiling() -> None:
    version = "kimi-k3"
    versions = {"extra-explabs": version}
    # One other channel already covers c1: cap=1 skips, cap=2 admits the second channel.
    one_other = _coverage(version, ["direct"])
    assert run_matrix._identity_fanout_reached(C1, "extra-explabs", versions, one_other, 1)
    assert not run_matrix._identity_fanout_reached(C1, "extra-explabs", versions, one_other, 2)
    # Two others: cap=2 now skips, cap=3 admits the third channel.
    two_others = _coverage(version, ["direct", "other"])
    assert run_matrix._identity_fanout_reached(C1, "extra-explabs", versions, two_others, 2)
    assert not run_matrix._identity_fanout_reached(C1, "extra-explabs", versions, two_others, 3)


def test_fanout_cap_leaves_a_twinless_model_alone() -> None:
    coverage = _coverage("kimi-k3", ["direct"])
    versions = {"qwen3.8-27b-explabs": QWEN_VERSION}
    assert not run_matrix._identity_fanout_reached(C1, "qwen3.8-27b-explabs", versions, coverage, 1)


def test_classify_cells_honours_a_raised_fanout_cap(_cfg) -> None:
    # The direct twin covers c1; with the default cap the extra's cell is skipped, and with
    # cap=2 it is kept — the mechanism the concordance subset rides.
    cache = _cache_with(KIMI_DIRECT, "kimi-k3", [C1])
    twin = KIMI_DIRECT
    versions = {KIMI_EXTRA: "kimi-k3", twin: "kimi-k3"}
    selected = {(C1, KIMI_EXTRA): ["default"]}
    hashes = {C1: f"h-{C1}"}
    for cap, skip in ((1, True), (2, False)):
        status = run_matrix.classify_cells(
            [C1],
            [KIMI_EXTRA],
            cache,
            hashes,
            versions,
            None,
            selected,
            None,
            identity_skip_models={KIMI_EXTRA},
            identity_fanout_cap=cap,
        )
        assert ((C1, KIMI_EXTRA, "default") not in status.missing) is skip

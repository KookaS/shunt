"""Cross-provider concordance: per-pair agreement, divergence warnings, and the subset wiring.

The concordance verdict is an instrument claim ("providers agree"), so the tests carry the
positive-control / shuffled-label-null pair as well as the identical-channel and planted-delta
cases. Every fixture is synthetic and lives here; the shipped module reads real corpus rows.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmark import config
from benchmark.routing import concordance, integrity, validate
from benchmark.routing.run_eval import _merge_free_corpus
from benchmark.runner import run_matrix

OVERLAY: Path = Path(__file__).resolve().parents[2] / "configs" / "free-tier" / "overlay.yaml"
FREE_CONFIG: Path = Path(__file__).resolve().parents[2] / "configs" / "free-tier" / "benchmark.yaml"

C1 = "c1"


def _row(
    cid: str,
    model: str,
    identity: str,
    provider: str,
    passed: bool,
    *,
    arm: str = "default",
    calls: str = "5",
    real_cost: str = "0.0",
) -> dict[str, str]:
    return {
        "challenge_id": cid,
        "model": model,
        "model_version": identity,
        "provider": provider,
        "reasoning": arm,
        "pass": str(passed),
        "calls": calls,
        "real_cost": real_cost,
    }


def _identical_corpus(n: int = 20) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(n):
        passed = index % 3 != 0
        rows.append(_row(f"c{index}", "model-a", "identity/x", "provider-a", passed))
        rows.append(_row(f"c{index}", "model-b", "identity/x", "provider-b", passed))
    return rows


def _planted_corpus(n: int = 20, b_passes: int = 14) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(n):
        rows.append(_row(f"c{index}", "model-a", "identity/x", "provider-a", True))
        rows.append(_row(f"c{index}", "model-b", "identity/x", "provider-b", index < b_passes))
    return rows


# ── observations and pairing ──────────────────────────────────────────────────


def test_zero_work_and_unlabelled_rows_are_skipped() -> None:
    rows = [
        _row("c0", "m", "id", "p", True),
        _row("c1", "m", "id", "p", True, calls="0"),
        {**_row("c2", "m", "id", "p", True), "provider": ""},
    ]
    obs = concordance.observations(rows)
    assert [item.challenge_id for item in obs] == ["c0"]


def test_pairing_is_deterministic_and_uses_every_provider_combination() -> None:
    obs = concordance.observations(
        [
            _row("c0", "a", "id", "p-a", True),
            _row("c0", "b", "id", "p-b", False),
            _row("c0", "c", "id", "p-c", True),
        ]
    )
    pairs = concordance.pair_observations(obs)
    assert [(p.provider_a, p.provider_b) for p in pairs] == [
        ("p-a", "p-b"),
        ("p-a", "p-c"),
        ("p-b", "p-c"),
    ]


# ── the measured verdict ──────────────────────────────────────────────────────


def test_identical_channels_are_not_flagged() -> None:
    report = concordance.concordance_report(_identical_corpus())
    assert report.n_cells == 20
    assert report.flagged == ()
    assert report.warnings == ()
    pair = report.pairs[0]
    assert pair.delta == 0.0
    assert pair.agreement == 1.0
    assert (pair.ci_lo, pair.ci_hi) == (0.0, 0.0)
    assert not pair.diverges


def test_planted_delta_is_flagged_with_a_ci_covering_it() -> None:
    planted_delta = 0.3
    report = concordance.concordance_report(_planted_corpus(20, b_passes=14))
    assert len(report.flagged) == 1
    pair = report.flagged[0]
    assert pair.delta == pytest.approx(planted_delta)
    assert pair.ci_lo <= planted_delta <= pair.ci_hi
    assert pair.agreement == pytest.approx(14 / 20)
    assert "DIVERGENCE" in report.warnings[0]


def test_a_significant_pair_is_reported_even_below_a_perfect_delta() -> None:
    report = concordance.concordance_report(_planted_corpus(40, b_passes=34))
    pair = report.pairs[0]
    assert pair.n_paired == 40
    assert pair.diverges
    assert pair.ci_lo > 0.0


# ── instrument validity: positive control + shuffled-label null ────────────────


def test_shuffled_labels_collapse_the_concordance_signal() -> None:
    planted = concordance.planted_control_corpus()
    positive = concordance.concordance_statistic(planted)
    shuffled = statistics.median(
        [
            concordance.concordance_statistic(concordance._shuffle_pass(planted, seed=draw))
            for draw in range(50)
        ]
    )
    assert positive > 5 * shuffled


def test_instrument_control_clears_both_legs() -> None:
    verdict = concordance.instrument_control(
        concordance.planted_control_corpus(), n_draws=400, seed=7
    )
    assert verdict.positive_passed
    assert verdict.null_at_chance
    assert verdict.admissible


# ── the named subset wiring ───────────────────────────────────────────────────


@pytest.fixture
def _free_cfg(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Capture the live config and restore it manually: `monkeypatch.setattr` records the value
    # at CALL time, so wrapping the `config.load(FREE_CONFIG)` body in it would record the free
    # config as the value to restore and leak it into later modules.
    previous = config._config
    monkeypatch.setattr(config, "_pricing", None)
    monkeypatch.setattr(config, "_free_registry", None)
    monkeypatch.setattr(config, "_free_registry_path_override", str(OVERLAY))
    yield
    config._config = previous


def _real_row(cid: str, identity: str) -> dict:
    return {
        "reasoning": "default",
        "model_version": identity,
        "version_hash": f"h-{cid}",
        "arm_hash": "",
        "image_digest": "",
        "step_limit": "",
        "cost_limit": "",
        "scaffold_version": "",
        "sampling_hash": "",
        "prompt_hash": "",
        "stop_reason": "",
        "calls": 5,
        "real_cost": 0.0,
        "cost": 0.0,
    }


def test_subset_names_three_identities_x_three_channels_x_twenty_challenges(_free_cfg) -> None:
    config.load(str(FREE_CONFIG))
    assert config.concordance_fanout_cap() == 3
    assert len(config.concordance_subset_models()) == 9
    assert len(config.concordance_subset_challenges()) == 20
    subset = config.concordance_config().get("subset", [])
    assert [entry["identity"] for entry in subset] == [
        "gemma-4-31b-it",
        "gpt-oss-120b",
        "laguna-s-2.1",
    ]
    assert all(len(entry["channels"]) == 3 for entry in subset)


def test_concordance_pairs_nine_on_the_committed_channel_set(_free_cfg) -> None:
    """The promised 3-way laguna comparison needs ONE identity per channel.

    Regression: vercel's version slug diverged from openrouter/kilo's, so the "3 providers"
    claim yielded 7 provider pairs (2+2 channels on one identity, 1 orphan) instead of 9.
    """
    config.load(str(FREE_CONFIG))
    config.set_free_registry(str(OVERLAY))
    subset = config.concordance_config().get("subset", [])
    channels = [c for entry in subset for c in entry["channels"]]
    config.register_collection_models(channels)
    versions = integrity.model_versions()
    overlay = config.free_registry()
    rows: list[dict[str, str]] = []
    for entry in subset:
        for channel in entry["channels"]:
            rows.append(_row(C1, channel, versions[channel], overlay[channel]["provider"], True))
    report = concordance.concordance_report(rows)
    assert len(report.pairs) == 9  # 3 identities x C(3 providers, 2)
    assert report.n_cells == 9
    laguna = next(entry for entry in subset if entry["identity"] == "laguna-s-2.1")
    assert {versions[channel] for channel in laguna["channels"]} == {"laguna-s-2.1"}


def test_subset_channels_run_beside_a_twin_where_dedupe_skips(_free_cfg) -> None:
    config.load(str(FREE_CONFIG))
    subset = config.concordance_subset_models()
    cap = config.concordance_fanout_cap()
    a, b, c = "groq-gpt-oss-120b", "sambanova-gpt-oss-120b", "cf-gpt-oss-120b"
    identity = "gpt-oss-120b"
    cache = {C1: {a: {"default": _real_row(C1, identity)}}}
    versions = {a: identity, b: identity, c: identity}
    selected = {(C1, model): ["default"] for model in (a, b, c)}
    status = run_matrix.classify_cells(
        [C1],
        [a, b, c],
        cache,
        {C1: f"h-{C1}"},
        versions,
        None,
        selected,
        None,
        identity_skip_models=set(),  # the subset is excluded from the cap-1 dedupe set
        identity_fanout_cap=cap,
        identity_fanout_models=subset,
    )
    assert status.present == 1  # channel a's own row
    assert sorted(status.missing) == [
        (C1, c, "default"),
        (C1, b, "default"),
    ]


def test_non_subset_extra_still_dedupes_at_cap_one(_free_cfg) -> None:
    config.load(str(FREE_CONFIG))
    a, b = "groq-gpt-oss-120b", "sambanova-gpt-oss-120b"
    identity = "gpt-oss-120b"
    cache = {C1: {a: {"default": _real_row(C1, identity)}}}
    versions = {a: identity, b: identity}
    selected = {(C1, b): ["default"]}
    status = run_matrix.classify_cells(
        [C1],
        [b],
        cache,
        {C1: f"h-{C1}"},
        versions,
        None,
        selected,
        None,
        identity_skip_models={b},
        identity_fanout_cap=3,
        identity_fanout_models=set(),  # b is outside the named subset
    )
    assert status.missing == []


def test_restrict_concordance_tasks_blanks_only_out_of_subset_challenges() -> None:
    selected = {("c1", "chan"): ["default"], ("c2", "chan"): ["default"]}
    restricted = run_matrix._restrict_concordance_tasks(selected, {"chan"}, {"c1"})
    assert restricted[("c1", "chan")] == ["default"]
    assert restricted[("c2", "chan")] == []


# ── multi-lane landing (merge_rows) ───────────────────────────────────────────


def test_two_lanes_of_one_identity_land_as_two_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    a = _row(C1, "groq-gpt-oss-120b", "gpt-oss-120b", "groq", True)
    b = _row(C1, "sambanova-gpt-oss-120b", "gpt-oss-120b", "sambanova", False)
    run_matrix.merge_rows([a], path, mode="replicate")
    run_matrix.merge_rows([b], path, mode="replicate")
    assert len(integrity.all_rows(path)) == 2


def test_reobserving_a_channel_with_unchanged_anchors_is_refused_under_supersede(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.csv"
    row = _row(C1, "groq-gpt-oss-120b", "gpt-oss-120b", "groq", True)
    run_matrix.merge_rows([row], path)
    changed = {**row, "computed_at": "2026-09-10T00:00:00+00:00", "cost": "0.001"}
    with pytest.raises(validate.DataIntegrityError):
        run_matrix.merge_rows([changed], path)


# ── run_eval --include-free-corpus ────────────────────────────────────────────


def test_merge_free_corpus_absent_is_a_noop(tmp_path: Path) -> None:
    assert _merge_free_corpus({"results": {}}, tmp_path / "missing.csv") == (0, 0)


def test_merge_free_corpus_adds_cells_and_challenges(tmp_path: Path) -> None:
    path = tmp_path / "results_free.csv"
    path.write_text(
        "challenge_id,model,reasoning,pass,cost,calls\nfree-c1,free-lane,default,True,0.0,5\n"
    )
    matrix = {"results": {"paid-c1": {"paid": {"pass": True}}}}
    assert _merge_free_corpus(matrix, path) == (1, 1)
    assert matrix["results"]["free-c1"]["free-lane"]["pass"] is True

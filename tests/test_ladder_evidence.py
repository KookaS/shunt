"""Pins the per-rung escalation evidence the committed results.csv currently yields."""

# The numbers below are NOT a target the analysis is tuned to hit — they are a snapshot of what
# the committed corpus says today, recorded so a silent change in results.csv (or in the pairing
# rule) fails a test instead of quietly rewriting a published claim. When the corpus legitimately
# grows, re-run `python -m benchmark.routing.scripts.ladder_evidence` and update PINNED_ROWS in
# the same change that grows it.
#
# A one-off probe recorded earlier, un-regenerable numbers for two of these rungs. They are kept
# in RECORDED_PROBE and reconciled deliberately: the test asserts the DIRECTION survived, not the
# counts, because the counts came from a smaller corpus and forcing agreement with them would be
# fitting the analysis to folklore.

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest

from benchmark import config
from benchmark.routing.scripts import ladder_evidence

CONFIG_PATH: Final = str(Path(config.__file__).resolve().parent / "benchmark.yaml")

# target -> (n, helps, hurts, delta, verdict) from the committed results.csv.
PINNED_ROWS: Final[dict[str, tuple[int, int, int, float, str]]] = {
    "qwen3.7-plus": (87, 6, 3, 0.0345, "INDISTINGUISHABLE"),
    "gpt-5-mini": (190, 4, 36, -0.1684, "NET-HARMFUL"),
    "kimi-k2.5": (121, 8, 10, -0.0165, "INDISTINGUISHABLE"),
    "zai-glm-5.2": (84, 14, 1, 0.1548, "NET-HELPFUL"),
    "kimi-k3": (110, 29, 3, 0.2364, "NET-HELPFUL"),
}

# The earlier un-regenerable probe: target -> (n, helps, hurts). Direction only is asserted.
RECORDED_PROBE: Final[dict[str, tuple[int, int, int]]] = {
    "gpt-5-mini": (140, 2, 18),
    "kimi-k3": (56, 13, 2),
}

_PRICE_ORDER: Final = ["qwen3.7-plus", "gpt-5-mini", "kimi-k2.5", "zai-glm-5.2", "kimi-k3"]


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    """The evidence payload built from the committed corpus at the script's defaults."""
    config.load(CONFIG_PATH)
    payload = ladder_evidence.build_evidence()
    assert payload is not None, "committed results.csv yielded no comparable rungs"
    return payload


@pytest.fixture(scope="module")
def rows(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["target"]: row for row in evidence["targets"]}


def test_base_is_the_cheap_tier(evidence: dict[str, Any]) -> None:
    """The default base is the cheapest enabled model — the tier a ladder escalates FROM."""
    assert evidence["base_model"] == "deepseek-v4-flash"
    assert [row["target"] for row in evidence["targets"]] == _PRICE_ORDER


@pytest.mark.parametrize("target", sorted(PINNED_ROWS))
def test_pinned_rung(rows: dict[str, dict[str, Any]], target: str) -> None:
    """Each rung reproduces its pinned n / helps / hurts / delta / verdict, or says what moved."""
    assert target in rows, f"{target} vanished from the corpus; rungs present: {sorted(rows)}"
    row = rows[target]
    actual = (row["n"], row["helps"], row["hurts"], row["delta"], row["verdict"])
    assert actual == PINNED_ROWS[target], (
        f"{target} moved: pinned={PINNED_ROWS[target]} current={actual}. "
        "If results.csv legitimately grew, update PINNED_ROWS in this file."
    )


def test_delta_is_determined_by_the_pair_counts(rows: dict[str, dict[str, Any]]) -> None:
    """delta == (helps - hurts) / n — ties cancel, so the counts fully determine the effect."""
    for row in rows.values():
        assert row["delta"] == pytest.approx((row["helps"] - row["hurts"]) / row["n"], abs=1e-4)


def test_price_multiple_comes_from_the_registry(rows: dict[str, dict[str, Any]]) -> None:
    """Price multiples are derived from the shipped registry, never restated in the script."""
    pricing = config.enabled_pricing()
    base = config.cost_per_1m("deepseek-v4-flash", pricing)
    for target, row in rows.items():
        assert row["price_multiple"] == pytest.approx(
            config.cost_per_1m(target, pricing) / base, abs=0.01
        )


def test_verdicts_are_adjudicated_against_the_null(rows: dict[str, dict[str, Any]]) -> None:
    """A non-neutral verdict requires the exact paired test to clear alpha AND leave the band."""
    for row in rows.values():
        low, high = row["null_ci95"]
        outside = row["delta"] < low or row["delta"] > high
        assert (row["p_value"] < 0.05) == (row["verdict"] != "INDISTINGUISHABLE")
        assert outside == (row["verdict"] != "INDISTINGUISHABLE")


def test_confidence_intervals_agree_with_the_verdicts(rows: dict[str, dict[str, Any]]) -> None:
    """The bootstrap interval excludes zero exactly where the null adjudication is non-neutral."""
    for row in rows.values():
        low, high = row["ci95"]
        excludes_zero = low > 0 or high < 0
        assert excludes_zero == (row["verdict"] != "INDISTINGUISHABLE")


@pytest.mark.parametrize("target", sorted(RECORDED_PROBE))
def test_recorded_probe_direction_survives(rows: dict[str, dict[str, Any]], target: str) -> None:
    """The earlier probe's DIRECTION reproduces; its counts came from a smaller corpus."""
    probe_n, probe_helps, probe_hurts = RECORDED_PROBE[target]
    row = rows[target]
    probe_sign = (probe_helps > probe_hurts) - (probe_helps < probe_hurts)
    current_sign = (row["helps"] > row["hurts"]) - (row["helps"] < row["hurts"])
    assert current_sign == probe_sign, (
        f"{target} REVERSED against the recorded probe: "
        f"probe n={probe_n} helps={probe_helps} hurts={probe_hurts}; "
        f"current n={row['n']} helps={row['helps']} hurts={row['hurts']}. "
        "A reversal is a finding, not a bug — re-read the claim that cites it before editing."
    )


def test_first_price_rung_is_measured_harmful(rows: dict[str, dict[str, Any]]) -> None:
    """The cheapest rung a price-ordered ladder reaches is net-harmful — the scoping falsifier."""
    # A price-ordered walk from the base reaches qwen3.7-plus (indistinguishable) and then
    # gpt-5-mini. If this ever flips, the "escalate to frontier only" scoping can be revisited.
    assert rows["gpt-5-mini"]["verdict"] == "NET-HARMFUL"
    assert rows["gpt-5-mini"]["ci95"][1] < 0
    assert rows["kimi-k3"]["verdict"] == "NET-HELPFUL"

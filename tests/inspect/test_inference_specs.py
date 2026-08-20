"""The inference family's text, asserted without importing a plotting backend."""

# The point of `specs.py` being matplotlib-free is that THIS file is: SH009 holds these strings
# byte-identical against `docs/inference.md`, and a text gate nobody can run without matplotlib
# is a text gate that rots.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from shunt.inspect.inference import specs

MAX_TITLE_CHARS = 90
MAX_CAVEAT_CHARS = 120


def test_specs_module_does_not_import_matplotlib() -> None:
    # A fresh interpreter, because this session has already imported matplotlib via other tests.
    # Asserting on the source text instead would pass on a comment that merely says the word.
    probe = (
        "import sys; import shunt.inspect.inference.specs; "
        "raise SystemExit(1 if 'matplotlib' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_seven_figures_with_unique_names_and_slugs() -> None:
    assert len(specs.FIGURES) == 7
    assert len({text.name for text in specs.FIGURES}) == 7
    assert [text.slug for text in specs.FIGURES] == [
        "inference-strata",
        "inference-cost",
        "inference-unit-economics",
        "inference-neighbourhood",
        "inference-policy",
        "inference-escalation",
        "inference-ope",
    ]
    assert all(text.filename == f"{text.name}.png" for text in specs.FIGURES)


def test_every_figure_carries_the_blocks_the_manifest_requires() -> None:
    for text in specs.FIGURES:
        assert text.title.strip(), text.name
        assert len(text.title) <= MAX_TITLE_CHARS, text.name
        assert text.subtitle.strip(), text.name
        assert text.reading.strip(), text.name
        assert text.goal.strip(), text.name
        assert text.definitions, text.name
        assert text.notes, text.name
        assert text.limitations, text.name


def test_caveats_are_single_line_and_within_the_frame_limit() -> None:
    for text in specs.FIGURES:
        if text.caveat is None:
            continue
        assert "\n" not in text.caveat, text.name
        assert text.caveat.strip() == text.caveat, text.name
        assert len(text.caveat) <= MAX_CAVEAT_CHARS, text.name


def test_cost_figure_never_calls_a_mixed_number_inference_cost() -> None:
    # The defect this family exists to correct: `total cost` over a seeded store presented as
    # inference cost. The axis label and the exclusion both have to be in the text, not the code.
    assert "seeded rows excluded" in specs.COST.title or "seeded" in specs.COST.subtitle
    assert "live inference cost (USD)" in specs.COST.subtitle
    assert "excluded by construction" in specs.COST.subtitle


def test_policy_figure_captions_the_seed_band_as_composition() -> None:
    assert specs.POLICY.caveat == "the seed band is corpus composition, not a routing decision"
    assert "not a choice the router made" in specs.POLICY.reading
    assert any("corpus composition" in note for note in specs.POLICY.notes)


def test_unit_economics_marks_the_reference_band_as_replayed() -> None:
    assert specs.UNIT_ECONOMICS.caveat is not None
    assert "REPLAYED BENCHMARK" in specs.UNIT_ECONOMICS.caveat


def test_hatching_means_replayed_everywhere_and_never_provisional() -> None:
    # One channel, one meaning. Hatching marks the seeded stratum on F1, F3 and F5, so the
    # provisional-n signal has to live somewhere else — the text has to agree with the canvas.
    assert "hatching marks replayed seeded rows" in specs.UNIT_ECONOMICS.subtitle
    assert "* marks n<10" in specs.UNIT_ECONOMICS.subtitle
    assert "hatched" not in dict(specs.UNIT_ECONOMICS.definitions)["provisional"]


def test_escalation_declares_panel_c_incomplete_and_names_the_derived_case() -> None:
    # N4: the engine returns early on an undeliverable rung with no hold token, so panel C is a
    # LOWER BOUND. The figure must say so on the canvas and in its limitations, not imply
    # completeness — that is the whole design decision, and it lives in these two strings.
    assert specs.ESCALATION.caveat is not None
    assert "ladder-evaluated holds only" in specs.ESCALATION.caveat
    assert any("lower bound" in limit for limit in specs.ESCALATION.limitations)
    assert any("counted nowhere on this figure" in limit for limit in specs.ESCALATION.limitations)
    assert specs.UNDELIVERABLE_LABEL in dict(specs.ESCALATION.definitions)
    assert "derived" in specs.UNDELIVERABLE_LABEL


def test_hold_and_rung_vocabularies_are_the_published_ones() -> None:
    assert specs.HOLD_TOKENS == (
        "disabled",
        "collapse_suppressed",
        "no_recurring_failure",
        "escalation_ceiling",
        "exploration_hold",
    )
    assert specs.RUNG_TOKENS == ("raise_effort", "raise_rank", "escalation_floor")
    # The derived case is deliberately NOT a sixth token: adding one changes the published
    # vocabulary and the engine, neither of which this figure owns.
    assert specs.UNDELIVERABLE_LABEL not in specs.HOLD_TOKENS


def test_policy_names_the_shipped_alarm_definition_not_a_local_proxy() -> None:
    # F5 previously divided entropy by the arms it OBSERVED and derived "frontier" from mean
    # cost. Both read all-clear on a collapsed router. The text has to name the real source.
    assert "top_capability_cluster" in " ".join(specs.POLICY.notes)
    assert "the registry's count" in dict(specs.POLICY.definitions)["normalized entropy"]
    assert "trailing window" in specs.POLICY.reading
    assert "not cumulative" in specs.POLICY.reading


def test_no_spec_string_names_a_function_that_does_not_exist() -> None:
    from shunt.inspect.inference import data

    names = ("escalation", "strata", "cost", "unit_economics", "neighbourhood", "policy", "ope")
    for name in names:
        assert hasattr(data, name)
    assert not hasattr(data, "escalation_breakdown")
    assert "escalation_breakdown" not in Path(specs.__file__ or "").read_text()


def test_escalation_limitation_is_a_grammatical_sentence() -> None:
    # It ships to the canvas and, via SH009, byte-identically into docs/inference.md.
    first = specs.ESCALATION.limitations[0]
    assert "Where escalation was not exploring" in first
    assert "on a session escalation was not exploring" not in first


def test_every_reading_block_explains_what_an_empty_panel_means() -> None:
    # A reader who sees an empty panel and no explanation concludes the render broke.
    for text in (specs.COST, specs.NEIGHBOURHOOD, specs.POLICY, specs.ESCALATION, specs.OPE):
        assert "empty" in text.reading.lower(), text.name


def test_ope_states_the_routing_contrast_asymmetry_wherever_a_reader_meets_it() -> None:
    # W7 measured it: routing's contrast compares against "took some other arm", an arbitrary
    # mixture of the remaining candidates, while escalation's compares two real actions. The
    # figure omits routing's contrast; these strings are why, on the canvas and in the docs.
    assert len(specs.ROUTING_CONTRAST_NOTE) <= MAX_CAVEAT_CHARS  # it ships AS the caveat
    assert "not a deployable policy" in specs.ROUTING_CONTRAST_NOTE
    assert any("not a policy anyone could deploy" in note for note in specs.OPE.notes)
    assert "Only escalation’s is drawn" in dict(specs.OPE.definitions)["contrast"]


def test_ope_declares_what_the_instrument_verdict_does_not_cover() -> None:
    # An ADMISSIBLE verdict is a gate against breakage, not a warrant of accuracy, and the
    # routing leg sees policy turns only. Both belong on the published page, not in a comment.
    assert any("gate against breakage" in limit for limit in specs.OPE.limitations)
    assert any("policy turns only" in limit for limit in specs.OPE.limitations)
    assert specs.OPE.caveat is None  # the caveat is computed: the refusal, or the asymmetry

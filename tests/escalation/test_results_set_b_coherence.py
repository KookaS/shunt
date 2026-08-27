"""results.md's Set B scorable count must equal the count derived from results.csv.

The doc once published "61 scorable of 69"; the true all-six-model scorable count is 74. The
derivation is reproduced here and the published count must equal it.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_CSV = _ROOT / "benchmark" / "routing" / "results.csv"
_DOC = _ROOT / "docs" / "results.md"


def _derived_set_b_scorable_count() -> int:
    """Tasks in results.csv where all six enabled models were measured."""
    # SUPERSET, not exact equality: the question is whether every enabled model has a cell,
    # not whether the row holds nothing else. results.csv legitimately carries cells for
    # models outside `enabled` — a probe-only collection such as zai-glm-5.3-flash's free
    # window — and under `==` each such cell silently DROPPED its task from the fully-measured
    # set (74 -> 33 when 41 probe cells landed), retracting a correct published number.
    import benchmark.config as config

    matrix = config.load_matrix()
    enabled = set(config.enabled_models())
    return sum(1 for cells in matrix["results"].values() if enabled <= set(cells))


def test_derived_set_b_count_is_74() -> None:
    assert _derived_set_b_scorable_count() == 74


def test_results_md_publishes_the_derived_count() -> None:
    text = _DOC.read_text(encoding="utf-8")
    # The Set B line must state 74 as the scorable count, derived from results.csv.
    m = re.search(r"Set B — .*?: (\d+) scorable", text)
    assert m, "Set B scorable count not found in results.md"
    assert int(m.group(1)) == _derived_set_b_scorable_count()
    assert "benchmark/routing/results.csv" in text, "doc must state the derivation source"


def test_no_site_in_results_md_states_the_stale_69_61_count() -> None:
    text = _DOC.read_text(encoding="utf-8")
    for stale in ("61 scorable of 69", "69/61", "61/69 tasks", "Set B (61 tasks)"):
        assert stale not in text, f"stale Set B count still published in results.md: {stale!r}"

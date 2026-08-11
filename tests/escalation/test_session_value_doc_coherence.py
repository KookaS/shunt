"""docs/escalation.md and the committed session-value figure must agree.

The doc once said 2.51× while the committed figure caption said lift 2.70x; the doc must defer
to the figure and its producer. This test greps both surfaces and asserts equality.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOC = _ROOT / "docs" / "escalation.md"
_CAPTION_ANCHOR = "#fig-session-value"


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


def _committed_caption_value() -> str:
    """The session-value lift from the committed figure caption in the same doc."""
    m = re.search(r"\*48 overlap tasks · .*?· lift ([\d.]+)x ·", _doc_text())
    assert m, "committed session-value caption not found in docs/escalation.md"
    return m.group(1)


def _prose_session_cadence_values() -> list[str]:
    """Every '2.xx× at session cadence'-style claim in the doc prose."""
    return re.findall(r"session cadence \(([\d.]+)", _doc_text())


def test_doc_prose_matches_committed_figure_caption() -> None:
    caption = _committed_caption_value()
    prose = _prose_session_cadence_values()
    assert prose, "no session-cadence value stated in the doc prose"
    # The figure caption is authoritative; every prose claim must quote the same value.
    for value in prose:
        assert value == caption, (
            f"doc prose says {value}× but the committed figure caption says {caption}× — "
            f"the doc must defer to the figure"
        )


def test_doc_cites_the_figure_and_its_producer() -> None:
    # The value is re-derivable from a committed producer, and the doc must cite it.
    text = _doc_text()
    assert _CAPTION_ANCHOR in text, "doc prose must cite the session-value figure"
    assert "session_eval.py" in text, "doc must cite the producer (session_eval.py)"


def test_the_committed_caption_value_is_reproducible() -> None:
    # The committed caption counts escalate 25/45 vs retry 7/34; the lift is the ratio.
    m = re.search(r"escalate ([\d]+)/([\d]+) vs retry ([\d]+)/([\d]+)", _doc_text())
    assert m, "committed caption counts not found"
    esc_n, esc_d, ret_n, ret_d = (int(g) for g in m.groups())
    lift = (esc_n / esc_d) / (ret_n / ret_d)
    assert lift == pytest.approx(float(_committed_caption_value()), abs=0.01)

"""docs/escalation.md, the shipped src/ comments and the committed metrics must agree."""

# The doc once said 2.51x while the committed figure caption said lift 2.70x; the doc must defer
# to the figure and its producer. The same stale literal then survived in two src/ comments while
# the doc was corrected, because this test only ever greped the doc. It now greps both surfaces.

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOC = _ROOT / "docs" / "escalation.md"
_METRICS = _ROOT / "benchmark" / "escalation" / "reports" / "metrics.json"
_CAPTION_ANCHOR = "#fig-session-value"

# Shipped files whose comments quote the session-cadence escalation value.
_SRC_FILES = (
    _ROOT / "src" / "shunt" / "config" / "router.yaml",
    _ROOT / "src" / "shunt" / "router" / "escalation.py",
    _ROOT / "configs" / "free-tier" / "router.yaml",
)

# A lift literal ("3.02x", "3.02×") anywhere in a paragraph that mentions session cadence.
_LIFT_RE = re.compile(r"(\d+\.\d+)\s*[x×]")
_SESSION_RE = re.compile(r"session[- ]cadence|session cadence", re.IGNORECASE)


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


def _committed_metrics_lift() -> float:
    """The session-cadence lift as committed by the eval producer."""
    payload = json.loads(_METRICS.read_text(encoding="utf-8"))
    return float(payload["session_value.png"]["lift"])


def _committed_caption_value() -> str:
    """The session-value lift from the committed figure caption in the same doc."""
    m = re.search(r"48 overlap tasks · .*?· lift ([\d.]+)x ·", _doc_text())
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


def _joined_comment_text(path: Path) -> str:
    """File text with comment-continuation line breaks folded away, so a claim reads as one span."""
    return re.sub(r"\n[ \t]*#[ \t]?", " ", path.read_text(encoding="utf-8"))


def _session_cadence_literals(text: str) -> list[str]:
    """Lift literals that are stated AS the session-cadence claim, not incidental ratios."""
    found: list[str] = []
    for match in _LIFT_RE.finditer(text):
        tail = text[match.end() : match.end() + 8].lstrip()
        is_labelled = tail.startswith(("lift", "ratio"))
        head = text[max(0, match.start() - 120) : match.start()]
        near_session = bool(_SESSION_RE.search(head))
        if is_labelled or near_session:
            found.append(match.group(1))
    return found


def test_shipped_src_comments_match_the_committed_metrics() -> None:
    """A stale lift literal in a shipped comment is the drift this test exists to catch."""
    expected = f"{_committed_metrics_lift():.2f}"
    seen = 0
    for path in _SRC_FILES:
        if not path.exists():
            continue
        for value in _session_cadence_literals(_joined_comment_text(path)):
            seen += 1
            assert value == expected, (
                f"{path.relative_to(_ROOT)} quotes a session-cadence lift of {value}x but "
                f"benchmark/escalation/reports/metrics.json says {expected}x"
            )
    assert seen, "no session-cadence lift literal found in any shipped src/ file"


# The retry contrast is the WEAKEST of the four baselines, and a shipped comment that quotes it
# alone contradicts docs/escalation-claim.md. These are the two arms that qualify it: the escalate
# arm loses to always-frontier and is indistinguishable from firing at random.
_QUALIFYING_ARMS = ("always_frontier", "random_escalate")


def _paired_differences() -> dict[str, dict[str, object]]:
    payload = json.loads(_METRICS.read_text(encoding="utf-8"))["session_value.png"]
    out = {"cheap_retry": payload["paired_difference"]}
    for arm, row in payload["comparisons"].items():
        out[arm] = row["paired_difference_vs_escalate"]
    return out


def _signed_forms(value: float) -> tuple[str, ...]:
    """A 3-decimal signed literal under both tie-breaking rules — a comment may use either."""
    # `-0.0385` is a real tie at 3dp: Python's round-half-even writes -0.038, a human writes
    # -0.039. Accepting both keeps the gate about the qualification being PRESENT and CURRENT
    # rather than about which rounding rule the author used.
    half_up = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return (f"{value:+.3f}", f"{half_up:+.3f}")


def _signed_literals(difference: dict[str, object]) -> list[tuple[str, ...]]:
    """The estimate and both interval bounds, as a shipped comment writes them."""
    low, high = difference["ci95"]  # type: ignore[misc]
    return [_signed_forms(float(v)) for v in (difference["estimate"], low, high)]


def test_a_shipped_comment_quoting_the_retry_contrast_also_carries_its_qualification() -> None:
    """The +0.416 retry contrast may never ship unqualified — that is the drift this catches."""
    differences = _paired_differences()
    retry = _signed_literals(differences["cheap_retry"])[0]
    checked = 0
    for path in _SRC_FILES:
        text = _joined_comment_text(path) if path.exists() else ""
        if not any(form in text for form in retry):
            continue
        checked += 1
        for arm in _QUALIFYING_ARMS:
            for forms in _signed_literals(differences[arm]):
                assert any(form in text for form in forms), (
                    f"{path.relative_to(_ROOT)} quotes the retry contrast {retry[-1]} but not "
                    f"the {arm} qualification {forms[-1]} — a shipped comment must not "
                    f"contradict the scoped claim it cites"
                )
    assert checked, f"no shipped src/ file quotes the retry contrast {retry[0]}"


def test_committed_caption_matches_the_committed_metrics() -> None:
    assert float(_committed_caption_value()) == pytest.approx(
        _committed_metrics_lift(), abs=0.005
    ), "docs/escalation.md caption disagrees with metrics.json"


def test_the_committed_caption_value_is_reproducible() -> None:
    # The committed caption counts escalate 28/45 vs retry 7/34; the lift is the ratio.
    m = re.search(r"escalate ([\d]+)/([\d]+) vs retry ([\d]+)/([\d]+)", _doc_text())
    assert m, "committed caption counts not found"
    esc_n, esc_d, ret_n, ret_d = (int(g) for g in m.groups())
    lift = (esc_n / esc_d) / (ret_n / ret_d)
    assert lift == pytest.approx(float(_committed_caption_value()), abs=0.01)

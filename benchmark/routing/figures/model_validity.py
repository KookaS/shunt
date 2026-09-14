"""model_validity.png — every evidenced model, why it is (or is not) inference-valid.

This is the routing half's ONE model census. It draws the full roster the committed corpus
names — the paid benchmark and live-pool models AND the separately-collected free channel —
against the five criteria `benchmark.routing.model_validity` defines: selected for inference
(in the live pool), triage pass (KEEP or EXCEPTION), capability rank measured not a price
prior, coverage at or above the `capability_rank.K` cell floor, and a paid channel. A model
failing any criterion is shown with the first failing reason, so an excluded model is NAMED
and EXPLAINED rather than dropped.

"EVIDENCED" has ONE definition (>=1 measured cell in either channel,
`model_validity.is_evidenced`), which is narrower than this roster: the roster also names
enabled/live slots and collection-only listings that carry no committed measurement. The
subtitle states both counts so a reader can tell them apart.

The other routing figures that draw a model at all read the same predicate's filtered set
(`model_validity.filter_valid`), so a dominated or never-measured model cannot stand beside a
served one as though they were the same kind of evidence. This canvas is where the excluded
ones live.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.colors import to_rgba

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import model_validity
from benchmark.routing.model_universe import display_name
from benchmark.routing.model_validity import ModelValidity

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from benchmark.routing.figures.context import RoutingContext

_PASS_BG: Final[str] = "#C8E6C9"
_FAIL_BG: Final[str] = "#FFCDD2"
_COVER_BG: Final[str] = "#BBDEFB"
_PAID_BG: Final[str] = "#ECEFF1"
_FREE_BG: Final[str] = "#FFE0B2"

# The neutral fill for a criterion that does not APPLY to a free-only row. A free-only identity
# is collection-only: it is never enabled, triaged or capability-ranked, so its triage and
# capability cells read "n/a" rather than the red "no" a paid row that actually failed carries.
# The paid-channel cell stays "no": the free channel is exactly the reason the row is out.
_NA_BG: Final[str] = "#F5F5F5"
_NA_CRITERIA: Final[frozenset[int]] = frozenset({3, 4})

# Panel B's two bars, moved off the red-green adjacency a CVD reader collapses: blue for the
# models that clear the bar, amber for those excluded, each bar carrying its printed count.
_VALID_BAR: Final[str] = "#0072B2"
_EXCLUDED_BAR: Final[str] = "#E69F00"

_GRID: Final[str] = "#FFFFFF"
_INK: Final[str] = "#1a1a1a"

# Column order for the matrix: channel and provider are documentary, the five CRITERIA are the
# verdict. Provider is a LABEL on the canonical identity, never part of its name.
_COLUMNS: Final[tuple[str, ...]] = (
    "channel",
    "provider(s)",
    *model_validity.CRITERIA,
    "verified\nchallenges",
)

# Reason classes for the census panel, in draw order (most severe first).
_REASON_LABELS: Final[tuple[str, ...]] = (
    "free channel (collection-only)",
    "collection-only promo probe",
    "live slot unmeasured",
    "probe-only (not enabled, not live)",
    "triage DROP",
    "inference-valid",
)


@dataclass(frozen=True)
class _Census:
    """The rows and the derived counts the two panels share."""

    rows: tuple[ModelValidity, ...]
    floor: int
    corpus: int

    @property
    def valid(self) -> int:
        return sum(1 for r in self.rows if r.valid)

    @property
    def invalid(self) -> int:
        return len(self.rows) - self.valid

    @property
    def paid(self) -> int:
        return sum(1 for r in self.rows if r.channel == model_validity.PAID)

    @property
    def free(self) -> int:
        return sum(1 for r in self.rows if r.channel == model_validity.FREE)

    @property
    def evidenced(self) -> int:
        """Identities with >=1 measured cell in either channel — the ONE evidenced count."""
        return sum(1 for r in self.rows if model_validity.is_evidenced(r))


def _reason_class(row: ModelValidity) -> str:
    """Bucket a row by the kind of failure, from the fields (never by parsing prose)."""
    if row.valid:
        return "inference-valid"
    if row.channel == model_validity.FREE:
        return "free channel (collection-only)"
    if row.collection_only:
        return "collection-only promo probe"
    if row.live:
        return "live slot unmeasured"
    if row.enabled:
        return "triage DROP"
    return "probe-only (not enabled, not live)"


def _cell_style(row: ModelValidity, col: int) -> tuple[tuple[float, float, float, float], str]:
    """(RGBA background, text) for one matrix cell."""
    channel_bg = _FREE_BG if row.channel == model_validity.FREE else _PAID_BG
    if col == 0:
        return to_rgba(channel_bg), row.channel
    if col == 1:
        # Provider is an axis, not an identity: every channel serving these weights is listed.
        return to_rgba(channel_bg), ", ".join(row.providers) or "—"
    # A free-only row is never submitted to triage or the capability rank, so those cells are
    # explicitly "n/a" on a neutral fill — not blank, and not a red "no" the row never earned.
    if row.channel == model_validity.FREE and col in _NA_CRITERIA:
        return to_rgba(_NA_BG), "n/a"
    if col == 7:
        frac = min(1.0, row.coverage_frac)
        return to_rgba(_COVER_BG, 0.35 + 0.65 * frac), f"{row.covered}"
    passed = row.criteria[col - 2]
    if col == 5:
        return to_rgba(_PASS_BG if passed else _FAIL_BG), str(row.cells)
    return to_rgba(_PASS_BG if passed else _FAIL_BG), "yes" if passed else "no"


def _draw_matrix(ax: Axes, census: _Census) -> None:
    rows = census.rows
    n_rows = len(rows)
    n_cols = len(_COLUMNS)
    rgba = np.ones((n_rows, n_cols, 4), dtype=float)
    for i, row in enumerate(rows):
        for j in range(n_cols):
            bg, text = _cell_style(row, j)
            rgba[i, j] = bg
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=5.6,
                color=_INK,
                family="monospace",
            )
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    # One white rule between every pair of cells, so a long column reads as cells not rows.
    ax.set_xticks(np.arange(-0.5, n_cols, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1.0), minor=True)
    ax.grid(which="minor", color=_GRID, linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(n_cols))
    # Disambiguate the two count columns ON THE CANVAS: "coverage ≥K cells" counts measured
    # DEFAULT-ARM CELLS (a challenge can contribute more than one), while the last column
    # counts DISTINCT challenges covered — different quantities that can read as duplicates.
    column_labels = list(_COLUMNS)
    column_labels[5] = "coverage\n≥K default-\narm cells"
    column_labels[7] = f"distinct\nchallenges\n(of {census.corpus})"
    ax.set_xticklabels(column_labels, fontsize=6.2, linespacing=1.1)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([display_name(r.model) for r in rows], fontsize=5.6, family="monospace")
    ax.tick_params(axis="y", length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plot_frame.panel_label(ax, "A · criteria matrix — every canonical weights identity")


def _draw_reasons(ax: Axes, census: _Census) -> None:
    counts = Counter(_reason_class(r) for r in census.rows)
    labels = [label for label in _REASON_LABELS if counts.get(label)]
    ys = list(range(len(labels)))[::-1]
    values = [counts[label] for label in labels]
    colours = [_VALID_BAR if label == "inference-valid" else _EXCLUDED_BAR for label in labels]
    ax.barh(ys, values, height=0.6, color=colours, alpha=0.9, zorder=2)
    for y, value in zip(ys, values, strict=True):
        ax.text(value + max(values) * 0.02, y, str(value), va="center", fontsize=7.5, color=_INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xlim(0, max(values) * 1.18 if values else 1.0)
    ax.set_xlabel("models", fontsize=8.5)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.0)
    plot_frame.panel_label(ax, "B · exclusion kind — why each named identity is out")


def _annotations(census: _Census) -> Annotations:
    return Annotations(
        subtitle_facts=(
            f"{census.valid} inference-valid of {len(census.rows)} canonical identities named "
            f"by committed evidence ({census.paid} paid, {census.free} free); "
            f"{census.evidenced} evidenced (≥1 measured cell in either channel)",
            f"coverage floor K={census.floor} measured default-arm cells "
            "(declared default arm, sole-arm fallback)",
        ),
        caveat=(
            "inference-invalid means not selected on committed evidence, not that the model is weak"
        ),
        notes=tuple(
            f"{display_name(r.model)}: {'VALID' if r.valid else 'invalid'} — {r.reason}"
            f" (channel {r.channel}, providers {', '.join(r.providers) or '—'}, "
            f"triage {r.triage}, capability {r.capability}, "
            f"{r.cells} cells, {r.covered}/{census.corpus} verified challenges)"
            for r in census.rows
        ),
        limitations=(
            "The criteria are order-independent only in the panel; the written reason reports "
            "the FIRST one a row fails, so a model that fails several shows only the first.",
            "'Evidenced' has ONE definition here: at least one measured cell in either channel. "
            f"So the {len(census.rows)}-identity roster holds {census.evidenced} evidenced "
            "identities, and the two counts differ on purpose — the roster also names live "
            "slots and collection-only listings with no committed measurement.",
            "The coverage cell count is the measured cells for the DECLARED default arm, "
            "falling back to a sole cached arm (the definition the universe's evidenced count "
            "reads). model_grid.png counts ONLY the strict declared arm, so its cell count can "
            "be smaller — this is why glm-5.2 shows 84 here and 21 there, and both labels say "
            "which definition they use.",
            "Free-channel coverage counts a handful of challenges and is not comparable with "
            "the paid corpus's; the two channels were collected under different campaigns.",
            "A free-only (collection-only) identity is never triaged or capability-ranked, so "
            "its triage and capability cells read 'n/a' rather than 'no'; the free channel — not "
            "a failed criterion — is what puts it outside the inference pool.",
            "A live slot with no committed measurement is flagged inference-invalid here, but "
            "it is a COVERAGE GAP, not evidence that the model is dominated — collect it and "
            "the verdict can change.",
            "This panel is the per-kind exclusion breakdown, not a second cumulative funnel. "
            "model_relevance.png carries the funnel (named to evidenced to paid to valid) and "
            "the pass-rate-vs-coverage plane; here panel B keeps each exclusion KIND distinct.",
        ),
        counts=(
            ("valid", census.valid),
            ("invalid", census.invalid),
            ("paid", census.paid),
            ("free", census.free),
            ("evidenced", census.evidenced),
            ("models", len(census.rows)),
        ),
    )


SPEC = FigureSpec(
    title="Only the measured live pool is inference-valid; the rest are named, not dropped",
    subtitle=(
        "canonical weights identity · provider is a label, never part of the name · criteria: "
        "live, triage pass (KEEP/EXCEPTION), capability measured, coverage, paid channel"
    ),
    caveat="invalid = not selected on committed evidence, not that the model is weak",
    reading=(
        "Panel A: one row per CANONICAL weights identity the committed evidence names (paid and "
        "free channels merged on `model_version`, never on a provider-prefixed listing id), one "
        "column per criterion — channel (paid or free), the serving provider(s), selected for "
        "inference (in the live pool), triage pass (KEEP or EXCEPTION), capability rank measured "
        "rather than a price prior, coverage at or above the K default-arm cell floor (the cell "
        "count is printed), and a paid channel. The last column is DISTINCT verified challenges "
        "covered (not the same quantity as the default-arm cell count in the coverage column: a "
        "challenge can contribute several cells), shaded by how much of the corpus the model "
        "holds. A green cell passes, a red one fails; a free-only row's triage and capability "
        "cells are NOT APPLICABLE and read 'n/a' on a neutral cell, because a collection-only "
        "identity is never triaged or capability-ranked, while its paid-channel cell stays 'no' "
        "— the fact that puts it outside the pool. "
        "The subtitle separates the roster ('canonical identities named') from 'evidenced' — at "
        "least one measured cell in either channel — because the roster also names slots no "
        "committed measurement covers. Panel B counts the identities by why they are out, keeping "
        "the free channel, the collection-only promo probes, the unmeasured live slots and the "
        "triage DROPs distinct."
    ),
    goal=(
        "Read the valid rows first — they are the only models the other routing figures show. "
        "Then read panel B to see that the excluded models are excluded for DIFFERENT reasons: "
        "a dominated benchmark model, an unmeasured frontier slot, and a collection-only probe "
        "are not the same kind of absence and must not be collapsed into one."
    ),
    definitions=(
        (
            "canonical identity",
            "the weights slug (`model_version`), so the same weights served by several "
            "providers are ONE row and a provider prefix or `-free` marker is never a name",
        ),
        (
            "inference-valid",
            "in the packaged live pool AND triage pass (KEEP or EXCEPTION) AND capability rank "
            "measured AND coverage >= K measured default-arm cells (declared default arm, "
            "sole-arm fallback) AND a paid channel",
        ),
        (
            "evidenced",
            "at least one measured cell in either channel (paid or free). NARROWER than the "
            "roster: the roster also names live slots and collection-only listings with no "
            "committed measurement, so 34 identities named is not 27 evidenced",
        ),
        (
            "capability measured",
            "the derived rank clears the confidence gate (K cells, CI width W, two qualifying "
            "peers); otherwise the model sits at its price-implied slot as a price prior",
        ),
        (
            "collection-only",
            "a row the benchmark may collect but never enable or route: the free overlay "
            "channel or a priced `-explabs` promo probe",
        ),
    ),
)


def build(ctx: RoutingContext) -> _Census | None:
    """The census the report already computed once, or a fresh read for a standalone use."""
    rows = ctx.validity
    if rows is None:
        rows = model_validity.validity_census()
    if not rows:
        return None
    corpus = rows[0].corpus
    return _Census(rows=tuple(rows), floor=model_validity.cell_floor(), corpus=corpus)


def render(ctx: RoutingContext) -> Path | None:
    """Draw model_validity.png; None when nothing has been evidenced."""
    census = build(ctx)
    if census is None:
        return None
    size = plot_frame.FigureSize("model_validity", 14.0, 12.5)
    fig = plot_frame.new_figure(size)
    axd = fig.subplot_mosaic(
        [["matrix", "matrix"], ["reasons", "reasons"]],
        height_ratios=(5.4, 1.0),
    )
    _draw_matrix(axd["matrix"], census)
    _draw_reasons(axd["reasons"], census)
    return plot_frame.save(
        fig,
        ctx.out_dir / "model_validity.png",
        SPEC,
        extra=_annotations(census),
        provenance=ctx.provenance(__name__),
        size=size,
    )

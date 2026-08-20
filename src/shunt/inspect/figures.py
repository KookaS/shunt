"""Diagnostic figures over the LIVE outcome store — the corpus the router queries.

Reads only public OutcomeStore methods and never prints: everything the CLI shows
travels back in `InspectResult` so this module stays lint-clean product code.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from shunt.inspect import plot_frame
from shunt.inspect.plot_style import model_color_map

if TYPE_CHECKING:
    from shunt.db.store import OutcomeStore

logger = logging.getLogger(__name__)

# Benchmark-seeded sessions are prefixed `bench:` by the eval harness; they draw as
# squares so the "did the benchmark data load into the live store?" check is visual.
SEED_PREFIX: Final[str] = "bench:"
_UNKNOWN_MODEL: Final[str] = "(unknown)"

_BAR_COLOR: Final[str] = "#1f77b4"
_EDGE_UNLABELED: Final[str] = "#9e9e9e"


@dataclass(frozen=True)
class InspectResult:
    """Everything the CLI prints after a render; ``figure_path`` is None on an empty corpus."""

    figure_path: Path | None
    n_embedded: int = 0
    n_labeled: int = 0
    n_tier2: int = 0
    seeded: int = 0
    live: int = 0
    pc1_share: float | None = None
    pc2_share: float | None = None
    pin_warning: str | None = None


@dataclass(frozen=True)
class _Census:
    """Counts the census panel renders — all read-only store aggregates."""

    total_sessions: int
    n_embedded: int
    n_labeled: int
    n_tier2: int
    seeded: int
    live: int
    index_size: int
    # Cost is split by stratum and never presented as one number: on a seeded rig the
    # whole-store total is overwhelmingly replayed benchmark spend, and this panel is titled
    # "Shunt inference corpus". `cost_unknown` is a COUNT of sessions the provider reported no
    # cost for — unknown, not a real 0.0, so it is stated rather than folded into either sum.
    live_cost: float
    seeded_cost: float
    cost_unknown: int
    per_model: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _PCABasis:
    """Centring mean + top right-singular vectors, so out-of-sample points project too."""

    mean: npt.NDArray[np.float64]
    components: npt.NDArray[np.float64]
    explained: npt.NDArray[np.float64]

    def transform(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project *vector* into the same subspace as the fitted corpus."""
        return (vector - self.mean) @ self.components


@dataclass(frozen=True)
class _ProjectionData:
    """Everything the projection panel needs, precomputed once in ``render``."""

    ids: list[str]
    projected: npt.NDArray[np.float64]
    explained: npt.NDArray[np.float64]
    model_colors: list[str]
    labels: dict[str, dict[str, Any]]
    latest_idx: int | None
    query_idx: int | None
    pin_point: tuple[float, float] | None
    label_lines: list[tuple[float, float, float, float]]
    pin_lines: list[tuple[float, float, float, float]]


def _fit_pca(vectors: npt.NDArray[np.float64], n_components: int) -> _PCABasis:
    """SVD PCA of *vectors* (rows = sessions): centred, top-*n_components* components."""
    n = min(n_components, vectors.shape[0], vectors.shape[1])
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular**2))
    explained = np.zeros(n) if total <= 0 else (singular[:n] ** 2) / total
    return _PCABasis(mean=mean, components=vt[:n].T, explained=explained)


def project(
    vectors: npt.NDArray[np.float64], n_components: int = 2
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Center *vectors* and project onto the top PCs; returns (projected, explained-ratios).

    The pure, unit-tested positive control the figure's axis variance annotations come
    from — numpy-only PCA, no sklearn, no hash/RNG stand-ins (SH008-safe).
    """
    fit = _fit_pca(vectors, n_components)
    return fit.transform(vectors), fit.explained


def _decode(rows: list[tuple[str, bytes]]) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Split (session_id, blob) rows into ids and a stacked float64 matrix."""
    ids = [sid for sid, _blob in rows]
    vectors = np.vstack(
        [np.frombuffer(blob, dtype=np.float32).astype(np.float64) for _sid, blob in rows]
    )
    return ids, vectors


def _palette(models: Sequence[str]) -> dict[str, str]:
    """One colour per distinct model, cycling Okabe-Ito in first-seen order."""
    # The same palette every benchmark figure uses: two figure families disagreeing on
    # what colour a model is makes them unreadable side by side, and tab10 is not
    # colourblind-safe.
    return model_color_map(list(dict.fromkeys(models)))


def _labels_by_id(labeled: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Outcome rows keyed by session_id, for O(1) lookups while drawing."""
    return {row["session_id"]: row for row in labeled}


def _outcome_class(row: dict[str, Any] | None) -> str:
    """Outcome bucket for one session: 'success', 'failure', or 'unlabeled'."""
    if row is None or row.get("tier2_outcome") is None:
        return "unlabeled"
    return "success" if row["tier2_outcome"] == "success" else "failure"


def _latest_labeled(labeled: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most recently-outcomed labeled session (by the outcome's created_at), or None."""
    dated = [r for r in labeled if r.get("created_at")]
    if not dated:
        return None
    return max(dated, key=lambda r: r["created_at"])


def _build_census(
    embedded_ids: list[str],
    labeled: list[dict[str, Any]],
    stats: dict[str, Any],
) -> _Census:
    """Assemble the census from the public store reads; labeled rows own model counts."""
    model_counts: dict[str, int] = {}
    for row in labeled:
        model = row.get("model_chosen") or _UNKNOWN_MODEL
        model_counts[model] = model_counts.get(model, 0) + 1
    per_model = tuple(sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return _Census(
        total_sessions=int(stats["session_count"]),
        n_embedded=len(embedded_ids),
        n_labeled=len(labeled),
        n_tier2=sum(1 for r in labeled if r.get("tier2_outcome") is not None),
        seeded=sum(1 for sid in embedded_ids if sid.startswith(SEED_PREFIX)),
        live=sum(1 for sid in embedded_ids if not sid.startswith(SEED_PREFIX)),
        index_size=int(stats["index_size"]),
        live_cost=float(stats["live_total_cost"]),
        seeded_cost=float(stats["seeded_total_cost"]),
        cost_unknown=int(stats["cost_unknown_count"]),
        per_model=per_model,
    )


def _session_meta(
    store: OutcomeStore, embedded_ids: set[str], limit: int = 500
) -> tuple[dict[str, str], str | None]:
    """(session_id -> model_chosen, most recent embedded session id) via get_sessions pages."""
    models: dict[str, str] = {}
    latest: tuple[str, str] | None = None
    offset = 0
    while True:
        rows = store.get_sessions(limit=limit, offset=offset)
        if not rows:
            break
        for row in rows:
            sid = row["session_id"]
            if sid in embedded_ids and row.get("model_chosen"):
                models[sid] = row["model_chosen"]
            timestamp = row.get("timestamp")
            if sid in embedded_ids and timestamp and (latest is None or timestamp > latest[0]):
                latest = (timestamp, sid)
        offset += limit
        if len(rows) < limit:
            break
    return models, latest[1] if latest else None


def _embed_prompt(text: str) -> tuple[npt.NDArray[np.float32] | None, str | None]:
    """Embed *text* through the shipped Embedder; (None, warning) when it cannot load."""
    from shunt.router.embedder import Embedder

    try:
        vector = Embedder().embed(text)
    except Exception as exc:
        logger.warning("prompt pin unavailable: %s", exc)
        return None, f"prompt pin unavailable ({type(exc).__name__}: {exc})"
    return vector, None


def _lines_for_query(
    store: OutcomeStore,
    qpoint: tuple[float, float],
    qvec: npt.NDArray[np.float64],
    sid_to_point: dict[str, tuple[float, float]],
    k: int,
    *,
    query_sid: str | None = None,
) -> list[tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) segments from *qpoint* to its k nearest neighbours' projections."""
    segments: list[tuple[float, float, float, float]] = []
    qx, qy = qpoint
    for sid, _distance in store.query_index(qvec, k):
        if sid == query_sid:
            continue
        point = sid_to_point.get(sid)
        if point is None:
            continue
        segments.append((qx, qy, point[0], point[1]))
    return segments


def _short_label(sid: str, max_chars: int = 24) -> str:
    """A session id shortened for on-canvas annotation."""
    return sid if len(sid) <= max_chars else sid[: max_chars - 1] + "…"


def _plot_segment(
    ax: Axes, seg: tuple[float, float, float, float], color: str, alpha: float
) -> None:
    """One neighbour-overlay line from query point to neighbour projection."""
    ax.plot([seg[0], seg[2]], [seg[1], seg[3]], color=color, alpha=alpha, lw=0.8, zorder=1)


def _scatter_by_outcome(
    ax: Axes,
    idx: list[int],
    data: _ProjectionData,
    marker: str,
    size: float,
) -> None:
    """One scatter per outcome class so face/edge encoding stays readable in the legend."""
    for cls in ("success", "failure", "unlabeled"):
        sub = [i for i in idx if _outcome_class(data.labels.get(data.ids[i])) == cls]
        if not sub:
            continue
        xs = data.projected[sub, 0]
        ys = data.projected[sub, 1]
        colors = [data.model_colors[i] for i in sub]
        if cls == "success":
            ax.scatter(xs, ys, marker=marker, s=size, c=colors, edgecolors="k", linewidths=0.4)
        elif cls == "failure":
            ax.scatter(
                xs,
                ys,
                marker=marker,
                s=size,
                facecolors="none",
                edgecolors=colors,
                linewidths=1.2,
            )
        else:
            ax.scatter(
                xs,
                ys,
                marker=marker,
                s=size,
                facecolors="none",
                edgecolors=_EDGE_UNLABELED,
                linewidths=0.7,
                alpha=0.55,
            )


def _projection_legend(ax: Axes) -> None:
    """Hand-built legend: fill = outcome, shape = origin, stars for pin/latest."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="#666666",
            label="live · verified pass",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#666666",
            label="live · verified fail",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=_EDGE_UNLABELED,
            label="live · unlabeled",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="none",
            markerfacecolor="#666666",
            label="seeded (bench:) · verified pass",
        ),
        Line2D(
            [],
            [],
            marker="*",
            color="none",
            markerfacecolor=plot_frame.INK,
            label="most recent session",
        ),
        Line2D(
            [],
            [],
            marker="X",
            color="none",
            markerfacecolor=plot_frame.INK,
            label="kNN query point (latest labeled)",
        ),
        Line2D(
            [],
            [],
            marker="P",
            color="none",
            markerfacecolor=plot_frame.CAVEAT_RED,
            label="prompt pin",
        ),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9, labelspacing=0.4)


def _draw_projection(ax: Axes, data: _ProjectionData) -> None:
    """Scatter the projected corpus, overlay neighbourhoods, and caption the axes."""
    for seeded, marker, size in ((False, "o", 24), (True, "s", 30)):
        idx = [i for i, sid in enumerate(data.ids) if sid.startswith(SEED_PREFIX) == seeded]
        if idx:
            _scatter_by_outcome(ax, idx, data, marker, size)
    for seg in data.label_lines:
        _plot_segment(ax, seg, plot_frame.INK, 0.45)
    for seg in data.pin_lines:
        _plot_segment(ax, seg, plot_frame.CAVEAT_RED, 0.55)
    if data.latest_idx is not None:
        i = data.latest_idx
        px, py = float(data.projected[i, 0]), float(data.projected[i, 1])
        ax.scatter([px], [py], marker="*", s=150, c=plot_frame.INK, zorder=5)
        ax.annotate(
            _short_label(data.ids[data.latest_idx]),
            (px, py),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=plot_frame.INK,
        )
    if data.query_idx is not None:
        i = data.query_idx
        qx, qy = float(data.projected[i, 0]), float(data.projected[i, 1])
        ax.scatter([qx], [qy], marker="X", s=130, c=plot_frame.INK, zorder=5)
        ax.annotate(
            _short_label(data.ids[i]),
            (qx, qy),
            textcoords="offset points",
            xytext=(-6, -6),
            fontsize=8,
            color=plot_frame.INK,
        )
    if data.pin_point is not None:
        ax.scatter(
            [data.pin_point[0]],
            [data.pin_point[1]],
            marker="P",
            s=110,
            c=plot_frame.CAVEAT_RED,
            zorder=5,
        )
        ax.annotate(
            "prompt pin",
            data.pin_point,
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=plot_frame.CAVEAT_RED,
        )
    ax.set_xlabel(f"PC1  {data.explained[0] * 100:.1f}% of variance")
    if data.explained.size > 1:
        ax.set_ylabel(f"PC2  {data.explained[1] * 100:.1f}% of variance")
    else:
        ax.set_ylabel("PC2  n/a")
    _projection_legend(ax)


def _draw_model_bars(ax: Axes, per_model: tuple[tuple[str, int], ...], *, ceiling: float) -> None:
    """Per-model labeled counts as axes-fraction bars below the stats block."""
    rows = per_model[:12]
    max_count = max(count for _m, count in rows)
    floor = 0.04
    row_h = min(0.07, (ceiling - floor) / len(rows))
    for i, (model, count) in enumerate(rows):
        y = ceiling - (i + 1) * row_h
        frac = count / max_count
        ax.add_patch(
            Rectangle(
                (0.05, y),
                0.9 * frac,
                row_h * 0.65,
                transform=ax.transAxes,
                facecolor=_BAR_COLOR,
                alpha=0.85,
            )
        )
        ax.text(
            0.05 + 0.9 * frac + 0.01,
            y + row_h * 0.3,
            f"{model}  {count}",
            transform=ax.transAxes,
            fontsize=7.5,
            family="monospace",
            va="center",
        )


def _draw_census(ax: Axes, census: _Census, *, top: float = 0.97) -> None:
    """Monospace text block (top) plus a per-model labeled-count bar list (below)."""
    ax.axis("off")
    lines = (
        f"total sessions: {census.total_sessions}",
        f"embedded:       {census.n_embedded}",
        f"labeled:        {census.n_labeled}   (tier-2: {census.n_tier2})",
        f"seeded (bench:):{census.seeded}    live: {census.live}",
        f"kNN index:      {census.index_size}",
        f"live cost:      ${census.live_cost:.4f}",
        f"seeded cost:    ${census.seeded_cost:.4f}",
        f"cost unknown:   {census.cost_unknown} sessions",
    )
    # Step shrinks to fit: the cold panel starts at top=0.55, where a fixed 0.095 would push
    # the last of these lines off the axes.
    step = min(0.095, (top - 0.02) / len(lines))
    y = top
    for line in lines:
        ax.text(0.03, y, line, transform=ax.transAxes, fontsize=9, family="monospace")
        y -= step
    if not census.per_model:
        ax.text(
            0.03,
            y,
            "no labeled sessions yet",
            transform=ax.transAxes,
            fontsize=9,
            color=plot_frame.MUTED,
        )
        return
    if y - 0.04 < 0.10:
        return  # no room below the stats block (cold panel) — the counts line carries it
    _draw_model_bars(ax, census.per_model, ceiling=y - 0.04)


def _render_cold(output_dir: Path, census: _Census) -> Path:
    """A single message + census panel for a corpus too small to project."""
    fig, axes = plot_frame.subplots(plot_frame.SINGLE, 1, 1)
    ax = cast(Axes, axes)
    spec = plot_frame.FigureSpec(
        title="Shunt inference corpus — too small to project",
        subtitle="PCA needs at least 2 embedded sessions to separate anything",
        caveat="The kNN index stays empty until verified Tier-2 outcomes exist.",
        reading=(
            "The census block is the whole figure: how many sessions the store holds, how many "
            "carry an embedding, and how many carry a verified outcome. There is no projection "
            "because fewer than two sessions are embedded."
        ),
        goal="Show why the projection is absent rather than writing an empty panel.",
    )
    ax.axis("off")
    ax.text(
        0.05,
        0.90,
        f"corpus too small — n={census.n_embedded} (still cold-start?)",
        transform=ax.transAxes,
        fontsize=12,
        color=plot_frame.INK,
    )
    _draw_census(ax, census, top=0.55)
    return plot_frame.save(fig, output_dir / "inspect_corpus.png", spec)


def _result(
    path: Path | None,
    census: _Census | None,
    *,
    pc1: float | None = None,
    pc2: float | None = None,
    warning: str | None = None,
) -> InspectResult:
    """Pack a census into the CLI-facing result (empty corpus passes a None census)."""
    if census is None:
        return InspectResult(figure_path=path)
    return InspectResult(
        figure_path=path,
        n_embedded=census.n_embedded,
        n_labeled=census.n_labeled,
        n_tier2=census.n_tier2,
        seeded=census.seeded,
        live=census.live,
        pc1_share=pc1,
        pc2_share=pc2,
        pin_warning=warning,
    )


def _draw_and_save(output_dir: Path, data: _ProjectionData, census: _Census) -> Path:
    """Assemble the two-panel WIDE figure and write the PNG through the frame."""
    fig, axes = plot_frame.subplots(plot_frame.WIDE, 1, 2)
    panels = cast(npt.NDArray[Any], axes)
    plot_frame.panel_label(panels[0], "embedding projection")
    plot_frame.panel_label(panels[1], "corpus census")
    _draw_projection(panels[0], data)
    _draw_census(panels[1], census)
    spec = plot_frame.FigureSpec(
        title="Shunt inference corpus — live outcome store",
        subtitle="PCA of the stored embeddings, colored by model",
        caveat="2D projection approximates cosine distance",
        reading=(
            "Left: every embedded session projected to its first two principal components and "
            "coloured by the model that served it; squares are benchmark-seeded rows, circles "
            "live ones. Right: the same corpus counted — embedded, labeled, seeded vs live."
        ),
        goal="Show what the router's kNN actually queries, and which stratum it is made of.",
    )
    path = output_dir / "inspect_corpus.png"
    return plot_frame.save(fig, path, spec)


def render(
    store: OutcomeStore,
    output_dir: Path,
    *,
    prompt: str | None = None,
    k: int = 5,
) -> InspectResult:
    """Render the diagnostic figure for the store's live corpus; returns the CLI summary."""
    emb_rows = store.get_all_embeddings()
    if not emb_rows:
        return _result(None, None)

    ids, vectors = _decode(emb_rows)
    labeled = store.labeled_outcome_rows()
    census = _build_census(ids, labeled, store.get_stats())
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(ids) < 2:
        return _result(_render_cold(output_dir, census), census)

    fit = _fit_pca(vectors, 2)
    projected = fit.transform(vectors)
    explained = fit.explained
    id_to_index = {sid: i for i, sid in enumerate(ids)}
    sid_to_point = {
        sid: (float(projected[i, 0]), float(projected[i, 1])) for i, sid in enumerate(ids)
    }
    models, latest_session_id = _session_meta(store, set(ids))
    palette = _palette([*models.values(), _UNKNOWN_MODEL])
    model_colors = [palette[models.get(sid, _UNKNOWN_MODEL)] for sid in ids]
    latest_idx = id_to_index.get(latest_session_id) if latest_session_id is not None else None

    pin: npt.NDArray[np.float32] | None = None
    pin_warning: str | None = None
    if prompt is not None:
        pin, pin_warning = _embed_prompt(prompt)

    label_lines, pin_lines, pin_point = [], [], None
    query_idx: int | None = None
    latest_labeled = _latest_labeled(labeled)
    if latest_labeled is not None and len(labeled) >= 2:
        query_sid = latest_labeled["session_id"]
        query_idx = id_to_index[query_sid]
        qpoint = (float(projected[query_idx, 0]), float(projected[query_idx, 1]))
        label_lines = _lines_for_query(
            store, qpoint, vectors[query_idx], sid_to_point, k, query_sid=query_sid
        )
    if pin is not None:
        pin64 = pin.astype(np.float64)
        point = fit.transform(pin64)
        pin_point = (float(point[0]), float(point[1]))
        pin_lines = _lines_for_query(store, pin_point, pin64, sid_to_point, k)

    data = _ProjectionData(
        ids=ids,
        projected=projected,
        explained=explained,
        model_colors=model_colors,
        labels=_labels_by_id(labeled),
        latest_idx=latest_idx,
        query_idx=query_idx,
        pin_point=pin_point,
        label_lines=label_lines,
        pin_lines=pin_lines,
    )
    path = _draw_and_save(output_dir, data, census)
    return _result(
        path,
        census,
        pc1=float(explained[0]) if explained.size else None,
        pc2=float(explained[1]) if explained.size > 1 else None,
        warning=pin_warning,
    )

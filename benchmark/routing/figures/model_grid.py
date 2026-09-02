"""model_grid.png — what each rung on the ladder costs, weighs and is measured to deliver."""

# THE BENCHMARK HALF of a drawer that ships in the wheel (`shunt.inspect.model_grid`). This
# module is only the ADAPTER: it turns the committed `results.csv` cache and the model
# registry into `GridRow`s and hands them over. The inference half does the same job against
# the live outcome store, so the two families draw one figure from two corpora and cannot
# disagree about what it means — the pattern `render_inference_figures.py` already proves.
#
# MEASURED CELLS ONLY. Every quality number here reads the raw challenge x model x arm cache,
# which holds observations. It never touches the completed/imputed matrix: 406 of that
# matrix's 1104 cells are monotone-imputed and near-exclusively pass-FILLED, so a rate drawn
# from it is biased upward by construction, and this figure's whole claim is a per-model rate.
#
# ONE ARM PER MODEL. The default arm, resolved from the registry — the arm the router would
# actually open on. Pooling arms would put a model's reasoning sweep into its headline rate
# and make the y axis depend on how much of the sweep happened to be sampled.
#
# THE BLEND IS THE CORPUS'S OWN MIX, applied identically to every model. A per-model mix would
# make the x axis measure "this model's verbosity" as much as its price, and the two are not
# separable on one axis. One mix, stated on the canvas, is a checkable number.
#
# A RUNG CAN BE MEASURED BEFORE IT IS IN THE CORPUS, and `data/external_rungs.yaml` is where
# that measurement lands. It is NOT a registry row on purpose: `benchmark.config.cost_per_1m`
# orders every cascade by list price and `shunt.models.config` ranks the shipped ladder by the
# same sum, so a registry addition re-orders the rank basis underneath already-published
# measurement claims — and a rung with zero cells in `results.csv` would become routable and
# rankable on evidence this corpus does not hold. The file is read HERE and nowhere else; no
# router, ranker, strategy or kill gate can see it. Its rows draw with a dagger, because they
# come from a different harness and their heights are not comparable cell-for-cell with the
# corpus rows beside them.

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

from benchmark import config
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from shunt.inspect import model_grid
from shunt.inspect.model_grid import GridData, GridRow
from shunt.inspect.plot_frame import FigureSpec
from shunt.models.config import load_registry, resolve_models

_UNDISCLOSED: Final[str] = "UNDISCLOSED"

#: Rungs measured OUTSIDE this corpus. See the file's own header for why they are not registry
#: rows: `benchmark.config.cost_per_1m` orders every cascade by list price, so a registry
#: addition moves the rank basis under a published claim. This adapter is the file's only
#: reader, and nothing it produces reaches a router, a ranker or a gate.
EXTERNAL_RUNGS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "data" / "external_rungs.yaml"
)

SPEC = FigureSpec(
    title="Every rung: what it costs, what it weighs, what it delivers",
    subtitle="one measured arm per model, nothing imputed · Wilson 95% intervals",
    caveat=("Panel A's x axis is a list price per TOKEN, never a bill per solved task."),
    reading=(
        "Panel A: each model at its blended token price (x, log — with a separate column at "
        "the left for locally-served rungs, which have no per-token list price at all and so "
        "cannot sit on a log axis; that column says nothing about what such a rung costs to "
        "run, which is UNDEFINED here and stated in the row's note) against its measured pass "
        "rate (y), with Wilson 95% whiskers. "
        "Marker area follows the square root of the active parameter count, the marker edge "
        "says whether the weights are hosted or local, and hue is the coarse total-size band. "
        "Panel B: one row per model, a hollow mark at total parameters and a filled mark at "
        "active parameters — the rule joining them IS the mixture-of-experts sparsity gap. "
        "A row whose name carries a dagger was measured outside this corpus under a different "
        "harness; its note below states which, and its height is not comparable cell-for-cell "
        "with the rows beside it. "
        "Panels C and D: per-call latency, hosted and local in separate axes because the "
        "two "
        "populations are not comparable."
    ),
    goal=(
        "Look for a rung that sits high and left in panel A — cheap per token and measured to "
        "resolve tasks. Then check panel B: if the high-and-left rungs are all long rules "
        "(very sparse), the ladder's cheap end is buying compute efficiency rather than size, "
        "and a small DENSE rung is not a substitute for one."
    ),
    definitions=(
        (
            "blended $/Mtok",
            "list input and output prices mixed at the corpus's own measured "
            "input:output token ratio, the same ratio for every model",
        ),
        (
            "active parameters",
            "what one token decodes through — a COMPUTE claim. All of a "
            "mixture's total parameters must still be resident to serve it",
        ),
        ("UNDISCLOSED", "the vendor publishes no parameter count. No estimate is substituted"),
        (
            "† (dagger)",
            "measured outside this corpus, on a different harness and a task draw that is "
            "neither paired with this corpus nor independent of it — the row's own note "
            "states the overlap. Plotted on the same axes, never pooled with the corpus rows",
        ),
    ),
    limitations=(
        "Pass rates come from an uneven sample: models were not run on identical task sets, "
        "so a rate difference across two rungs is not a paired comparison.",
        "Every row counts a censored cell in its denominator and not in its passes, so a row "
        "whose run censored heavily is a LOWER BOUND on that rung's rate, not an estimate of "
        "it. Where that matters the row's own note gives the censored count and the "
        "assumption-free bounds; read the marker as the floor it is.",
    ),
)


def _blend_ratio(raw: plot_style.RawResults, arms: dict[str, str]) -> tuple[float, int, int]:
    """The corpus's own input:output token split, over the same default-arm cells drawn."""
    in_tok = 0
    out_tok = 0
    for per_model in raw.values():
        for model, per_arm in per_model.items():
            row = per_arm.get(arms.get(model, ""))
            if row is None:
                continue
            in_tok += int(row.get("in_tok") or 0)
            out_tok += int(row.get("out_tok") or 0)
    total = in_tok + out_tok
    return (in_tok / total if total else 0.0), in_tok, out_tok


def _blended_price(pricing: dict, model: str, in_share: float) -> float | None:
    """One model's list price per 1M tokens at the corpus mix, or None when unpriced."""
    entry = pricing.get(model)
    if not isinstance(entry, dict):
        return None
    inp = entry.get("input_cost_per_1m", entry.get("input"))
    out = entry.get("output_cost_per_1m", entry.get("output"))
    if inp is None or out is None:
        return None
    return float(inp) * in_share + float(out) * (1.0 - in_share)


def _param(value: object) -> int | None:
    """A published count, or None for the literal UNDISCLOSED — never a guess."""
    return None if value == _UNDISCLOSED or not isinstance(value, int) else value


def _seed_clause(measured: dict[str, Any]) -> str:
    """The pooled-seeds phrase, silent for a rung measured on a single configuration."""
    seeds = measured.get("seeds")
    return f", seeds {seeds} pooled" if seeds else ""


def _cost_clause(measured: dict[str, Any]) -> str:
    """The billed cost per instance, or the row's own reason there is no dollar figure."""
    # A local rung's cost is UNDEFINED, not zero: $0 of hosted spend is not a price, and
    # amortising hardware over a workload would be an invented number. Such a row states
    # `cost_note` instead, and no dollar figure is emitted for it anywhere.
    billed = measured.get("measured_cost_per_instance_usd")
    if billed is None:
        return str(measured["cost_note"])
    return (
        f"${float(billed):.4f} per instance as billed, averaged over all "
        f"{measured['cells_run']} cells run (the n here is the model-attributable subset "
        "of those)"
    )


def _censoring_clause(measured: dict[str, Any]) -> str:
    """The count of censored cells, or the row's own longer statement of what they do to it."""
    return str(measured.get("censoring_summary") or f"{measured['censored']} cell(s) censored")


def _external_rows(in_share: float, *, path: Path = EXTERNAL_RUNGS_PATH) -> list[GridRow]:
    """The committed out-of-corpus measurements, priced on the SAME blend as every other row."""
    # The blend is the corpus's own mix applied to this rung's list price, exactly as it is
    # applied to a corpus row. Its own measured $/instance is a DIFFERENT quantity in a
    # different unit and cannot share the axis; it is printed in the row's note instead.
    if not path.is_file():
        return []
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows: list[GridRow] = []
    for name, entry in sorted((doc.get("rungs") or {}).items()):
        measured = entry["measurement"]
        pricing = entry["pricing"]
        size = entry.get("size") or {}
        local = entry["serving_mode"] == "local"
        rows.append(
            GridRow(
                name=name,
                # A LOCALLY SERVED RUNG HAS NO LIST PRICE, exactly as a local registry row has
                # none in `build()` below: nothing is billed per token, so there is no price to
                # put on the log axis and the row goes to panel A's category column. That is a
                # statement about the AXIS QUANTITY, not a cost claim — the row's own note
                # states that its dollar cost per solved task is UNDEFINED, never zero.
                x=None
                if local
                else float(pricing["input_cost_per_1m"]) * in_share
                + float(pricing["output_cost_per_1m"]) * (1.0 - in_share),
                serving_mode=entry["serving_mode"],
                n=int(measured["n"]),
                passes=int(measured["passes"]),
                total_params=_param(size.get("total_params")),
                active_params=_param(size.get("active_params")),
                latency_s=(),
                provenance_note=(
                    f"measured on {measured['scaffold']} over {measured['task_population']}"
                    f"{_seed_clause(measured)}; "
                    f"{measured['corpus_overlap']}; "
                    f"{_cost_clause(measured)}; "
                    f"{_censoring_clause(measured)}; "
                    f"verdict ceiling {measured['verdict_ceiling']} — NOT a corpus row, and "
                    f"not comparable cell-for-cell with the rows beside it"
                ),
            )
        )
    return rows


def _undrawn_rungs(drawn: set[str]) -> list[str]:
    """Models with cells in `results.csv` that this canvas does not draw, and why."""
    # THE CANVAS IS NOT THE CACHE. `ctx.raw` reaches this adapter already scoped to
    # `benchmark.yaml`'s enabled set (`report._only_enabled_models`), which legitimately
    # drops a probe-only collection — a model measured on a free window and never enabled
    # for the benchmark. That drop is deliberate; publishing a source line that says
    # "the measured default-arm cells of results.csv" while a filter silently narrows it
    # is not. This re-reads the cache — a csv parse, not a recomputation — purely so the
    # subtitle can NAME what it left out instead of implying a sweep it never ran.
    try:
        cached = config.load_results()
        enabled = set(config.enabled_models())
    except Exception:  # noqa: BLE001 (the caption degrades; the figure does not)
        return []
    counts: dict[str, int] = {}
    for per_model in cached.values():
        for model, per_arm in per_model.items():
            if model not in drawn:
                counts[model] = counts.get(model, 0) + len(per_arm)
    notes = []
    for name, cells in sorted(counts.items()):
        why = (
            "not enabled in benchmark.yaml"
            if enabled and name not in enabled
            else "no measured default-arm cell"
        )
        notes.append(f"{name} ({cells} cells, {why})")
    return notes


def _source_line(drawn: set[str]) -> str:
    """What the canvas is drawn FROM — including what the enabled-set filter removed."""
    base = "the measured default-arm cells of results.csv, plus any † out-of-corpus rung"
    undrawn = _undrawn_rungs(drawn)
    if not undrawn:
        return base
    return f"{base} · results.csv also holds " + "; ".join(undrawn) + " — not drawn"


def build(ctx: ctxmod.RoutingContext) -> GridData | None:
    """Turn the measured cache + the registry into the grid's rows."""
    raw = ctx.raw
    if not raw:
        return None
    models = sorted({m for per_model in raw.values() for m in per_model})
    arms = config.default_arm_ids(models)
    pricing = config.load_pricing()
    registry = resolve_models(load_registry())
    in_share, in_tok, out_tok = _blend_ratio(raw, arms)

    rows: list[GridRow] = []
    for name in models:
        stats = plot_style.arm_stats(raw, name, arms[name])
        if stats.n == 0:
            continue
        entry = registry.get(name)
        size = entry.size if entry is not None else None
        local = entry is not None and entry.serving_mode == "local"
        rows.append(
            GridRow(
                name=name,
                # A local rung's marginal token price is exactly zero, which is a different
                # kind of number from a list price and gets its own axis region.
                x=None if local else _blended_price(pricing, name, in_share),
                serving_mode=entry.serving_mode if entry is not None else "hosted",
                n=stats.n,
                passes=stats.passes,
                total_params=_param(size.total_params) if size else None,
                active_params=_param(size.active_params) if size else None,
                # Latency is declared MISSING on this corpus: `wall_clock_s` and
                # `latency_per_call_s` are blank on every committed row, and a blank
                # measurement column is never read as a value.
                latency_s=(),
            )
        )
    # THE CORPUS IS THE PRECONDITION, not the row count. An out-of-corpus row can never be the
    # only thing on this canvas: `source` and the blend both describe `results.csv`, so a
    # figure drawn from the external file alone would assert a provenance it does not have.
    if not rows:
        return None
    rows.extend(_external_rows(in_share))
    return GridData(
        rows=tuple(rows),
        x_label="blended $ per 1M tokens (log) — mix stated above",
        price_basis=(
            f"blend = {in_share * 100:.0f}% input / {(1 - in_share) * 100:.0f}% output, "
            f"the corpus's own {in_tok:,}:{out_tok:,} token split"
        ),
        source=_source_line({row.name for row in rows}),
        # THIS HALF PLOTS A LIST PRICE. The live half plots a measured bill and states the
        # opposite sentence; neither may inherit the other's.
        x_limitation=(
            "Panel A's x axis is a LIST PRICE at one token mix, not a measured bill: it "
            "answers what a token costs, never what a solved task costs. A locally served "
            "rung has no such price at all and sits in the category column, which is not a "
            "claim that running it is free — its cost is UNDEFINED and its note says so."
        ),
    )


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw model_grid.png — the three-panel ladder. None when no model was measured."""
    data = build(ctx)
    if data is None:
        return None
    return model_grid.render(ctx.out_dir / "model_grid.png", data, SPEC, ctx.provenance(__name__))

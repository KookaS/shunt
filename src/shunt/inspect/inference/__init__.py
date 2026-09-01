"""Inference-side figure family: the live router measured on its own outcome store."""

# The figure modules are imported INSIDE `render`, not at module scope. `estimators.py` is
# stdlib-only so it can run without a draw stack, and `python -c "import shunt"` must not pull
# matplotlib in; a top-level `from . import figures` here would undo both.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shunt.db.store import OutcomeStore
    from shunt.inspect.inference.data import SessionRow
    from shunt.models.config import ModelPool

GENERATOR: Final[str] = "shunt.inspect.inference:render"

# The manifest stays beside the code that writes it — it is source, not a published asset —
# while the PNGs live inside the docs tree so the page can link them relatively.
_PKG_DIR: Final[Path] = Path(__file__).resolve().parent
MANIFEST: Final[Path] = _PKG_DIR / "figures.json"
CANONICAL_PLOTS_DIR: Final[Path] = _PKG_DIR.parents[3] / "docs/assets/figures/inference"

_HALF: Final[str] = "inference"


@dataclass(frozen=True)
class Family:
    """One figure family these eight drawings serve: its half, its home, and its stamp."""

    # A family is the ONLY thing that varies between the measured `inference` set and an
    # illustrative one — same store shape, same eight drawings, same specs. Carrying the
    # watermark here rather than on a per-figure flag is what makes it unforgettable: the
    # renderer opens `plot_frame.watermarked(family.watermark)` around the whole draw, and
    # `plot_frame.save` is the only door a figure can leave by (SH007 denies `savefig`
    # elsewhere), so a figure added later inherits the mark without anyone remembering it.

    half: str
    # The committed figure directory. Rendering anywhere else is a scratch copy and must not
    # touch the committed manifest, which is what `manifest_for` enforces.
    plots_dir: Path
    manifest: Path
    watermark: str | None = None

    def manifest_for(self, plots_dir: Path) -> Path:
        """The committed manifest only when *plots_dir* IS this family's committed home."""
        # A directory NAME proves nothing — a tmp dir can be called `inference` too — so the
        # test is "is this THE committed directory", not "is it named like one".
        if plots_dir.resolve() == self.plots_dir.resolve():
            return self.manifest
        return plots_dir.parent / "figures.json"


INFERENCE: Final[Family] = Family(_HALF, CANONICAL_PLOTS_DIR, MANIFEST)


def _manifest_for(plots_dir: Path) -> Path:
    return INFERENCE.manifest_for(plots_dir)


@dataclass(frozen=True)
class InferenceReport:
    """What one render produced: the PNGs written, the manifest, and any figure that refused."""

    out_dir: Path
    manifest: Path
    data_digest: str
    figures: tuple[Path, ...]
    # A refusal is a result, not an error to swallow silently: F7's instrument gate fires before
    # a canvas exists, and the other six must still render. The reason travels with the report.
    inadmissible: str | None = None


def data_digest(rows: list[SessionRow]) -> str:
    """A content fingerprint of the corpus a figure was drawn from."""
    # Keyed on the adjudicated session rows rather than on the `.db` file: the store's
    # lifecycle columns (`created_at`, `updated_at`) are wall-clock by design, so a file hash
    # would differ on every rebuild of an identical corpus.
    digest = hashlib.blake2b(digest_size=8)
    for row in rows:
        digest.update(
            json.dumps(
                [
                    row.session_id,
                    row.timestamp.isoformat() if row.timestamp is not None else None,
                    row.model_chosen,
                    row.cost,
                    row.cost_known,
                    row.stratum,
                    row.selection_rule_used,
                    row.selection_propensity,
                    row.hold_reason,
                    row.rung,
                    row.undeliverable,
                    row.tier2_success,
                ],
                sort_keys=True,
            ).encode()
        )
    return digest.hexdigest()


def render(
    store: OutcomeStore,
    out_dir: Path,
    *,
    windows: tuple[int | None, ...] = (7, 30, None),
    family: Family = INFERENCE,
    now: datetime | None = None,
) -> InferenceReport:
    """Render the eight inference figures into `out_dir` and record their manifest rows."""
    # `now` reaches exactly one place — `data._in_window`, the family's single windowed
    # predicate — and defaults to the wall clock, so a measured render is unchanged. It is here
    # for a corpus with FROZEN timestamps, whose `7d`/`30d` panels would otherwise decay with
    # the calendar and take the committed PNGs stale with them.
    from shunt.inspect import plot_frame
    from shunt.inspect.inference import data as idata
    from shunt.inspect.inference import estimators, figures

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = idata.read_sessions(store)
    provenance = plot_frame.Provenance(
        generator=GENERATOR,
        data_digest=data_digest(rows),
        manifest=family.manifest_for(out_dir),
    )
    pool = _model_pool()
    inadmissible: str | None = None
    # The whole draw sits inside the family's stamp, so every canvas below — and every canvas
    # a later edit adds — is marked without the draw functions knowing the mark exists. On the
    # default family the watermark is None and this block is a no-op, byte for byte.
    with plot_frame.watermarked(family.watermark):
        written = [
            figures.draw_strata(out_dir, idata.strata(store, rows), provenance),
            figures.draw_cost(out_dir, idata.cost(rows, windows, now=now), provenance),
            figures.draw_unit_economics(out_dir, idata.unit_economics(rows), provenance),
            figures.draw_neighbourhood(out_dir, idata.neighbourhood(store, rows), provenance),
            figures.draw_policy(out_dir, idata.policy(rows, pool), provenance),
            figures.draw_escalation(out_dir, idata.escalation(rows, windows, now=now), provenance),
        ]
        written.append(figures.draw_model_grid(out_dir, idata.model_grid(rows, pool), provenance))
        try:
            estimates = estimators.certify(store)
            written.append(figures.draw_ope(out_dir, idata.ope(store, estimates), provenance))
        except estimators.InstrumentInadmissibleError as exc:
            # One estimator that failed its control must not take the family down: the seven
            # figures before it read the store, not the instrument. F7 leaves no PNG, so SH009 sees
            # a section with no file and the commit stops — the intended coupling, not a skip.
            inadmissible = str(exc)
    return InferenceReport(
        out_dir=out_dir,
        manifest=provenance.manifest,
        data_digest=provenance.data_digest,
        figures=tuple(written),
        inadmissible=inadmissible,
    )


def _model_pool() -> ModelPool:
    """The shipped registry F5 needs to know how many arms the router could have picked."""
    from shunt.models.config import ModelPool

    return ModelPool.load()


# ------------------------------------------------------------------ docs sections


def docs_section(png: str, row: dict[str, Any], *, half: str = _HALF) -> str:
    """One figure's SH009 markdown block, rendered from its manifest row."""
    # Generated rather than hand-written on purpose: SH009 holds title, subtitle, caveat and
    # notes byte-identical between `figures.json` and `docs/inference.md`, and half of those
    # strings are merged at render time from runtime counts. Copying them by hand is the drift
    # this function exists to make impossible.
    # THE MARKER NAMES THE ROW'S OWN GENERATOR, not this module. `figures.json` records the
    # entrypoint that drew each canvas, and this emitter serves more than one manifest: the
    # routing family's `model_grid.png` is drawn by `benchmark.routing.figures.model_grid`
    # and only its markdown block is rendered here. Hardcoding `GENERATOR` published a
    # provenance marker pointing at an entrypoint that never produces that figure — SH012
    # checks a marker EXISTS, never that it RESOLVES TO THE PRODUCER, so it stayed green.
    # The fallback keeps the inference and demo manifests (whose rows already record this
    # module) and a manifest-less row emitting exactly what they emitted before.
    slug = png.removesuffix(".png").replace("_", "-")
    title = str(row["title"])
    lines = [
        f"### {title} {{#fig-{slug}}}",
        "",
        f"![{title}](assets/figures/{half}/{png})",
        "",
        f"*{row['subtitle']}*",
    ]
    if row.get("caveat"):
        lines.append(f"> **Caveat.** {row['caveat']}")
    lines.append(f"**Reading.** {row['reading']}")
    lines.append("")
    lines.append(f"**What to look for.** {row['goal']}")
    terms = row.get("terms") or []
    if terms:
        lines.append("")
        lines.append("**Terms.** " + " ".join(f"*{term}* — {meaning}" for term, meaning in terms))
    notes = row.get("notes") or []
    if notes:
        lines.append("")
        lines.append("**Notes.** " + "\n".join(str(note) for note in notes))
    limitations = row.get("limitations") or []
    if limitations:
        lines.append("")
        lines.append("**Limits.** " + " ".join(str(item) for item in limitations))
    counts = row.get("n") or {}
    if counts:
        lines.append("")
        lines.append(
            "<!-- n: "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            + f" --><!-- generated-by: {row.get('generator') or GENERATOR} -->"
        )
    return "\n".join(lines) + "\n"


def docs_sections(manifest: Path) -> str:
    """Every figure's markdown block, in the family's declared order."""
    from shunt.inspect.inference import specs

    payload = json.loads(manifest.read_text())
    # The half is read from the manifest rather than assumed: `plot_frame.record` writes it
    # from the manifest's own directory, and it is what decides the `assets/figures/<half>/`
    # image url. Hardcoding `inference` here is how a second family's page would ship links
    # into the first family's directory.
    half = str(payload.get("half") or _HALF)
    rows: dict[str, Any] = payload.get("figures", {})
    blocks = [
        docs_section(text.filename, rows[text.filename], half=half)
        for text in specs.FIGURES
        if text.filename in rows
    ]
    return "\n".join(blocks)


__all__ = [
    "CANONICAL_PLOTS_DIR",
    "GENERATOR",
    "INFERENCE",
    "MANIFEST",
    "Family",
    "InferenceReport",
    "data_digest",
    "docs_section",
    "docs_sections",
    "render",
]

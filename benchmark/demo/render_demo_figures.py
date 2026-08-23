"""Render the demo figure family: the seven inference drawings over an invented corpus."""

# WHY THIS EXISTS. `docs/inference.md` is drawn from a seed-only store with no live rows, so
# five of its seven panels are empty and a reader who has never seen a populated one cannot
# tell what the page is for. This module draws the same seven over `demo_corpus`, so the shapes
# are legible. It answers "what does F3 look like when it has data", and nothing else.
#
# WHAT KEEPS IT HONEST. Four fences, none of them a caption:
#   * `Family(watermark=...)` stamps every canvas through `plot_frame.save` — the one door a
#     figure can leave by — so no drawing can omit the mark and none has to remember it;
#   * the half is its own: its own PNG directory, its own manifest, its own docs page, held
#     apart from the measurement halves by SH009 itself;
#   * `benchmark/pipeline.py` names `demo_corpus.py` (which carries `DEMO_SEED`) among this
#     job's digest inputs, so editing the generator or its seed marks the figures stale;
#   * no report, gate or analysis reads this half.
#
# The DRAWING is shipped code (`shunt.inspect.inference`) — the identical functions
# `render_inference_figures.py` calls. Only the store differs, which is the point.

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Final

from benchmark.routing import demo_corpus
from shunt.db.store import OutcomeStore
from shunt.inspect.inference import Family, render
from shunt.inspect.inference.estimators import InstrumentInadmissibleError

_PKG_DIR: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _PKG_DIR.parents[1]

WATERMARK: Final[str] = "SYNTHETIC — NOT MEASURED"
CANONICAL_PLOTS_DIR: Final[Path] = _REPO_ROOT / "docs/assets/figures/demo"
MANIFEST: Final[Path] = _PKG_DIR / "figures.json"

DEMO: Final[Family] = Family(
    half="demo",
    plots_dir=CANONICAL_PLOTS_DIR,
    manifest=MANIFEST,
    watermark=WATERMARK,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="render the illustrative demo figure family")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Draw a scratch copy here, manifest diverted beside it. Omit to rebuild the "
            f"COMMITTED demo figures into {CANONICAL_PLOTS_DIR}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: build the demo corpus, draw the seven figures, return the exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else CANONICAL_PLOTS_DIR

    # A throwaway store every time: the corpus is a pure function of DEMO_SEED, so persisting
    # it would only create a second copy of the same bytes for someone to mistake for data.
    with tempfile.TemporaryDirectory(prefix="shunt-demo-") as scratch:
        store = OutcomeStore(db_path=str(demo_corpus.build_demo_store(Path(scratch) / "demo.db")))
        try:
            # The demo's own clock. `demo_corpus` anchors every timestamp to a fixed date, so
            # without this the 7d/30d panels would be a function of the wall clock: they would
            # thin toward empty and the committed PNGs would go DRIFTED on the calendar alone.
            render(store, out_dir, family=DEMO, now=demo_corpus.DEMO_NOW)
        except InstrumentInadmissibleError as exc:
            print(f"INADMISSIBLE: {exc}", file=sys.stderr)
            return 1
        finally:
            store.close()
    print(f"figures: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

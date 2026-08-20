"""Render the inference figure family — the committed docs set, or a scratch copy."""

# Benchmark-side because only the committed mode's inputs live here: the LFS seed bundle and
# `docs_corpus`, which seeds a store from committed data alone. The DRAWING is all shipped code
# (`shunt.inspect.inference`), which is what lets the same seven figures render inside the rig
# container, where no `benchmark/` exists — there the entry point is
# `python -m shunt.inspect.inference` instead.
#
# Two modes, and the difference is the DATABASE, not the code:
#
#   no --out-dir   the COMMITTED docs figures. Seed-only, deterministic, no network and no live
#                  rig; the only mode that may touch the committed manifest.
#   --out-dir X    the same seven, drawn from whatever `SHUNT_DATA_DIR` points at (seed rows and
#                  live rows alike). `_committed_home()` diverts the manifest to
#                  `X/../figures.json` so a scratch render cannot dirty the committed one.

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmark.routing import docs_corpus
from shunt.db.store import OutcomeStore
from shunt.inspect.inference import CANONICAL_PLOTS_DIR, render
from shunt.inspect.inference.estimators import InstrumentInadmissibleError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="render the inference figure family")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Draw a scratch copy from the ambient SHUNT_DATA_DIR store into this directory, "
            "manifest diverted beside it. Omit to rebuild the COMMITTED docs figures from the "
            f"seed-only corpus into {CANONICAL_PLOTS_DIR}."
        ),
    )
    return parser


def _store_for(out_dir: Path | None) -> OutcomeStore:
    """The seed-only docs corpus for the committed set; the ambient live store otherwise."""
    if out_dir is None:
        return OutcomeStore(db_path=str(docs_corpus.build()))
    return OutcomeStore()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: render the seven figures; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else CANONICAL_PLOTS_DIR

    store = _store_for(args.out_dir)
    try:
        render(store, out_dir)
    except InstrumentInadmissibleError as exc:
        # F7 refuses BEFORE a canvas exists, so the family is incomplete and the exit must say
        # so: a zero here would let a half-drawn set reach the SH009 manifest gate as "rendered".
        print(f"INADMISSIBLE: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    print(f"figures: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

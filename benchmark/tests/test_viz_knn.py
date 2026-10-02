"""The kNN calibration bins must count every scored cell, float edges included.

`reliability` is the census `docs/routing.md` reports (n=3/139/465/647). Thirteen
live cells carry a weighted rate that rounds to 1.0000000000000002, and the old
strict `< hi` dropped them from every bin, so the figure summed to 1254 of 1267.
"""

from __future__ import annotations

import numpy as np

from benchmark.routing.scripts import viz_knn


def _bins(rates: np.ndarray) -> list[dict]:
    """Reliability bins for a one-model corpus whose cells are the given rates."""
    pass_mat = np.ones_like(rates).reshape(-1, 1)
    return viz_knn.reliability(rates.reshape(-1, 1), pass_mat)


def test_reliability_top_bin_counts_every_cell_including_the_float_edge() -> None:
    rates = np.concatenate(
        [
            np.full(3, 0.35),
            np.full(139, 0.55),
            np.full(465, 0.75),
            np.full(647, 0.95),
            np.full(13, 1.0000000000000002),
        ]
    )
    bins = _bins(rates)
    assert rates.size == 1267
    assert sum(b["n"] for b in bins) == 1267
    # The live census after the fix: the 13 float-edge cells join the top bin.
    assert [b["n"] for b in bins] == [3, 139, 465, 660]


def test_reliability_counts_a_lone_float_edge_cell() -> None:
    bins = _bins(np.array([1.0000000000000002]))
    assert sum(b["n"] for b in bins) == 1

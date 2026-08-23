"""Re-export shim: the layout audit now ships in `shunt.inspect.plot_contract`."""

# The audit moved into the wheel so the shipped figure code can run it without
# importing `benchmark` (SH006 forbids that direction). Benchmark call sites and
# `tests/test_plot_contract.py` keep importing `benchmark.plot_contract`; this file
# is the compatibility surface, and there is exactly one implementation.

from __future__ import annotations

from shunt.inspect.plot_contract import (
    LayoutError,
    Violation,
    assert_clean,
    audit,
)

__all__ = ["LayoutError", "Violation", "assert_clean", "audit"]

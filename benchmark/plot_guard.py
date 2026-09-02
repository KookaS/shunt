#!/usr/bin/env python3
"""Runtime wall: at build and test time, only the plot frame may write a figure to disk."""

from __future__ import annotations

import inspect
import runpy
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Final

from shunt.inspect import plot_frame

# SH007 is an AST DENYLIST: it can only refuse the spellings its author enumerated. A file
# holding `ax.set(title=...)`, an aliased `savefig` and a `getattr(fig, "savefig")` passed it
# at exit 0 until those three were added, and the next spelling (an attribute name computed at
# runtime, a save inside a third-party helper) is invisible to it by construction. This guard
# watches the BEHAVIOUR instead: whatever the call site looks like, the write itself has to
# come from inside the frame.
#
# WHERE IT IS INSTALLED, and why not everywhere. Two sites: `tests/conftest.py` (the whole
# suite, including every figure test) and the Makefile's `$(DRAW)` launcher, which runs a
# producer as `python -m benchmark.plot_guard <module>`. It is deliberately NOT installed from
# `benchmark/__init__.py`, `benchmark/pipeline.py` or the frame itself: each of those files is
# inside the figure jobs' import closure, so editing one marks committed PNGs STALE for
# `pipeline --check-figures` and demands a redraw that the guard cannot change the bytes of.
# The residual, stated plainly: producers spawned by `benchmark.pipeline` (`make
# benchmark-figures`) run unguarded, because the only hook that would reach them is such a
# file. SH007 is still the wall there.
#
# It is build-time and test-time only — the shipped router never imports it, and `shunt
# inspect` in a user's process is unaffected.

_FRAME_FILE: Final[str] = str(Path(plot_frame.__file__).resolve())

# `save()` legitimately reaches `print_figure` THROUGH matplotlib's own `Figure.savefig`, so
# the writer guard opens this token for the duration of an approved write rather than trying
# to allow-list matplotlib's internals. A ContextVar (not a module global: SH001) so a render
# in another task cannot inherit the permission.
_APPROVED: Final[ContextVar[bool]] = ContextVar("shunt_plot_guard_approved", default=False)
_INSTALLED: Final[dict[str, Any]] = {}


class FigureEscapedFrameError(RuntimeError):
    """A figure was written to disk from outside the plot frame."""


def _offender() -> Any:
    """The frame that called the guarded method: 0 = here, 1 = the wrapper, 2 = the caller."""
    frame = inspect.currentframe()
    for _ in range(2):
        if frame is None:
            return None
        frame = frame.f_back
    return frame


def _is_frame_module(frame: Any) -> bool:
    if frame is None:
        return False
    return str(Path(frame.f_code.co_filename).resolve()) == _FRAME_FILE


def _describe(frame: Any) -> str:
    if frame is None:
        return "<unknown caller>"
    return f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}()"


def _refuse(method: str, frame: Any) -> FigureEscapedFrameError:
    return FigureEscapedFrameError(
        f"{method} called from {_describe(frame)}, outside the plot frame. Write the figure "
        "with shunt.inspect.plot_frame.save(fig, path, FigureSpec(...)) (or "
        "benchmark.plot_frame, which re-exports it) so it carries its claim title band, passes "
        "the layout audit and records its manifest row. SH007 refuses the spellings it can "
        "see; this refuses the rest."
    )


def install() -> None:
    """Patch the matplotlib writers so a save outside the frame raises. Idempotent."""
    if _INSTALLED:
        return
    from matplotlib.backend_bases import FigureCanvasBase as Canvas
    from matplotlib.figure import Figure

    # This module is the runtime wall for the writers named below, so it is the one file that
    # has to reference them by name. The suppressions are per-line and deliberately narrow: a
    # bare save added anywhere else in this file is still a SH007 violation.
    original_savefig = Figure.savefig  # noqa: SH007 (the guard patches this writer)
    original_print = Canvas.print_figure  # noqa: SH007 (the guard patches this writer)

    def guarded_savefig(self: Figure, *args: Any, **kwargs: Any) -> Any:
        caller = _offender()
        if not _is_frame_module(caller):
            raise _refuse("Figure.savefig", caller)
        token = _APPROVED.set(True)
        try:
            return original_savefig(self, *args, **kwargs)
        finally:
            _APPROVED.reset(token)

    def guarded_print(self: Canvas, *args: Any, **kwargs: Any) -> Any:
        caller = _offender()
        if not _APPROVED.get() and not _is_frame_module(caller):
            raise _refuse("FigureCanvasBase.print_figure", caller)
        return original_print(self, *args, **kwargs)

    Figure.savefig = guarded_savefig  # type: ignore[method-assign]  # noqa: SH007 (patch site)
    Canvas.print_figure = guarded_print  # type: ignore[method-assign]  # noqa: SH007 (patch site)
    _INSTALLED["savefig"] = original_savefig
    _INSTALLED["print_figure"] = original_print


def uninstall() -> None:
    """Restore the unpatched writers — for the test that proves the guard is real."""
    if not _INSTALLED:
        return
    from matplotlib.backend_bases import FigureCanvasBase as Canvas
    from matplotlib.figure import Figure

    orig_savefig = _INSTALLED.pop("savefig")
    orig_print = _INSTALLED.pop("print_figure")
    Figure.savefig = orig_savefig  # type: ignore[method-assign]  # noqa: SH007 (restore site)
    Canvas.print_figure = orig_print  # type: ignore[method-assign]  # noqa: SH007 (restore site)


def _main(argv: list[str]) -> int:
    """`python -m benchmark.plot_guard <module> [args]` — run a producer behind the guard."""
    # The install has to happen INSIDE the producer's own process, and it cannot happen in
    # `benchmark/__init__.py`: that file is in every producer's import closure, so editing it
    # marks all 41 committed figures stale. This launcher sits outside every closure instead.
    if not argv:
        print("usage: python -m benchmark.plot_guard <module> [args...]", file=sys.stderr)
        return 2
    install()
    sys.argv = [argv[0], *argv[1:]]
    runpy.run_module(argv[0], run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))

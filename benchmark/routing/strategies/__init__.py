from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


def routing_text(task_id: str, task_meta: dict) -> str:
    """The text a routing strategy embeds: the task's problem statement, else its
    short description, else the id."""
    # The `description` (`<repo>@<commit12> - resolve <test-node-id>`) is a LABEL: measured
    # over the 500 committed tasks it is 62% test node-id, 14% repo name and 12% a random
    # commit prefix unique per task. Embedding it put the kNN neighbourhood over filenames
    # and repo names while the agent was handed `problem_statement` — router and routee never
    # saw the same text. Manifests built before problem_statement existed fall back.
    statement = str(task_meta.get("problem_statement") or "").strip()
    if statement:
        return statement
    return str(task_meta.get("description") or task_id)


@dataclass(frozen=True)
class BilledAttempt:
    """One billed attempt: which model served it, what it cost, and the tokens it moved."""

    # WHY A RECORD AND NOT A TUPLE. This was `(model, cost)`, and every consumer unpacked it
    # positionally. The context-cost model needs the tokens too, and widening a tuple silently
    # re-binds every `for model, cost in attempts` in the tree to two of five fields. The record
    # is deliberately NOT iterable so that re-binding is an AttributeError at the call site
    # rather than a wrong number in a published table.
    #
    # `in_tok` / `out_tok` / `calls` are the MEASURED columns of the cell that was billed. An
    # imputed cell carries none of them (see impute.ImputedCell), so they are 0 there — which is
    # exactly the condition `context_cost.token_complete_tasks` filters on rather than smoothing.

    model: str
    cost: float
    in_tok: int = 0
    out_tok: int = 0
    calls: int = 0


class Strategy(ABC):
    #: Per-attempt billing published by the most recent :meth:`select`, in billing order.
    #: ``None`` on a single-shot strategy, which never bills anything but the one cell it
    #: returns; a cascade REPLACES the list on every ``select``. Declared here rather than
    #: discovered by the caller with ``getattr``, so :attr:`sessions_burned` has one
    #: definition for every strategy instead of a default that hides a missing attribute.
    cascade_attempts: list[BilledAttempt] | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str: ...

    @property
    def sessions_burned(self) -> int:
        """How many sessions the most recent :meth:`select` made the user sit through."""
        # One billed attempt is one session waited through, so the default IS
        # `len(cascade_attempts)`; a single-shot strategy publishes none and reports the
        # single decision it made. A cascade therefore reports its real depth by
        # construction — the same list the cost model already requires is the one counted
        # here, so a new cascade cannot bill several attempts and still report 1.
        #
        # Override only where the session count is genuinely NOT the attempt count:
        # `SessionCascadeStrategy` bills unmeasured rungs onto its path without appending
        # them to `cascade_attempts`, and does override.
        if self.cascade_attempts is None:
            return 1
        return len(self.cascade_attempts)

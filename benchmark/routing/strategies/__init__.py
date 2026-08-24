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


class Strategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str: ...


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

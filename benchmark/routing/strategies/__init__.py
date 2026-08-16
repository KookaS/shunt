from __future__ import annotations

from abc import ABC, abstractmethod


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

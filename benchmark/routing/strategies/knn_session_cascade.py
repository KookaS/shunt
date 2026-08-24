"""kNN-seeded session cascade: the kNN pick opens the ladder, session cadence climbs it.

The OPT-IN `router.strategy: knn_cascade` (the shipped default, `session_cascade`, opens at the
cheapest model) — the first benchmark row that starts where the kNN pick lands.
"""

# WHY A SUBCLASS AND NOT A SECOND REPLAY. `SessionCascadeStrategy` already replays the shipped
# default's ladder through the product's own `EscalationRunner`; the ONLY thing this opt-in
# strategy does differently is where the first session opens. So everything downstream of the
# opening rung —
# `ArmLadder`, `_Trace`, `_bill`, `_apply`, `_cell`, `_mark_terminal_arm`, the runner seeding and
# `next_rung_rank` — is inherited verbatim, and the two rows cannot drift into measuring two
# different ladders. The one override is `_initial_rank_floor`.
#
# THE PICK IS THE kNN ROW'S PICK, not a re-implementation. `kNNStrategy` is instantiated and
# asked, so the index build, the neighbour query, the shipped `Embedder` (SH008 embedder
# isolation, `SHUNT_DISALLOW_REAL_EMBEDDER`, the 4000-char clip, the durable cache) and the live
# `RouterEngine.decide` selection are all the same code the `kNN` row runs. A local
# `TextEmbedding(...)` here would score the two rows in different embedding spaces.
#
# THE DISCLOSURE CARRIES OVER UNCHANGED. results.csv records a per-cell pass/fail and no
# failing-check identity, so the replay treats every failure of a task as recurring — the
# assumption most favourable to escalation, making the ladder climb as fast as the policy ever
# could. A cost reported here is therefore a LOWER bound on what the live ladder spends.
#
# A FLOOR IS A FLOOR. A task the base pick would have solved one rung DOWN is still billed at the
# kNN-selected rung, because the router never routes below its own pick. That is the failure mode
# unique to this row, and `session_cascade_control.py`'s second leg is what proves it is modelled
# rather than smoothed away.

from __future__ import annotations

from typing import TYPE_CHECKING

from . import Strategy
from .knn import kNNStrategy
from .session_cascade import DEFAULT_LADDER, SessionCascadeStrategy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .session_cascade import ArmLadder


class kNNSessionCascadeStrategy(SessionCascadeStrategy):  # noqa: N801 (kNN is the algorithm name)
    """The opt-in routing strategy: the kNN pick chooses the opening rung, then the session
    ladder climbs from there on a recurring verified failure. The shipped default,
    `session_cascade`, opens at the cheapest model and consults no neighbours.
    """

    def __init__(  # noqa: PLR0913 (one arg per knob of the two composed policies)
        self,
        k: int = 20,
        success_rate_threshold: float = 0.7,
        min_samples: int = 3,
        escalate_after_n: int = 2,
        stale_window: int = 10,
        ladder: str = DEFAULT_LADDER,
        arm_results: Mapping[str, Mapping[str, Mapping[str, dict]]] | None = None,
        arm_ladders: Mapping[str, ArmLadder] | None = None,
        rank_shortlist: int | None = None,
        picker: Strategy | None = None,
    ) -> None:
        super().__init__(
            escalate_after_n=escalate_after_n,
            stale_window=stale_window,
            ladder=ladder,
            arm_results=arm_results,
            arm_ladders=arm_ladders,
            rank_shortlist=rank_shortlist,
        )
        # Injectable so the positive control can plant a known floor without an ONNX load; None
        # means the real kNN row's own strategy object — the SAME class the `kNN` row runs.
        self._picker: Strategy = picker or kNNStrategy(
            k=k,
            success_rate_threshold=success_rate_threshold,
            min_samples=min_samples,
        )

    @property
    def name(self) -> str:
        """Display name shown in eval output and plots."""
        return "kNN-cascade"

    def _initial_rank_floor(self, task_id: str, matrix: dict, rungs: Sequence[str]) -> int:
        """The rung the kNN pick lands on, as an index into the price-ordered ladder."""
        # A pick outside the measured ladder has no rung to open on. That is a coverage gap, not
        # a cheap route, so it opens at the bottom and the ladder climbs from there — the same
        # degradation the base row already has when the pool is empty.
        chosen = self._picker.select(task_id, matrix.get("tasks", {}).get(task_id, {}), matrix)
        return rungs.index(chosen) if chosen in rungs else 0

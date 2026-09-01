"""The inference family's canvas text and manifest record, held free of matplotlib."""

# Split out from `figures.py` so the strings can be asserted byte-for-byte without importing
# matplotlib: SH009 holds these in bijection with `docs/inference.md`, and a text test that has
# to import a plotting backend is a text test nobody runs in the lint job. `figures.py` converts
# each entry to `plot_frame.FigureSpec`, which is where the length limits are enforced.
#
# Every string here is written for a corpus that may be entirely seeded. Emptiness is a finding
# in this family, not a defect, so the reading and limitation blocks state what an empty panel
# means rather than leaving the reader to guess it was a bug.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SEED_PREFIX: Final[str] = "bench:"

# The five HOLD tokens `decide_escalation` can emit. Any hold outside this set is a vocabulary
# drift and the figure surfaces it rather than dropping it into an "other" bucket.
#
# `disabled` is defined but UNREACHABLE from the serving path: `engine._task_key` returns None
# when escalation is off, and `_finalize_decision` returns on that None before `_maybe_escalate`
# ever builds a directive, so the live router cannot emit it and its bar is structurally zero.
# It stays in the vocabulary so a token that ever does appear is named rather than bucketed.
HOLD_TOKENS: Final[tuple[str, ...]] = (
    "disabled",
    "collapse_suppressed",
    "no_recurring_failure",
    "escalation_ceiling",
    "exploration_hold",
)

# Not a sixth token: a DERIVED category. `engine.py` returns early when a RAISE_* directive
# cannot be delivered, so that hold carries no `escalation_hold_reason` at all. It is recovered
# from the voided exploration record instead — see `data.escalation`.
UNDELIVERABLE_LABEL: Final[str] = "rung undeliverable (derived)"

RUNG_TOKENS: Final[tuple[str, ...]] = ("raise_effort", "raise_rank", "escalation_floor")


@dataclass(frozen=True)
class FigureText:
    """One figure's canvas text plus the blocks its docs section is written from."""

    name: str
    title: str
    subtitle: str
    caveat: str | None
    reading: str
    goal: str
    definitions: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def slug(self) -> str:
        return self.name.replace("_", "-")

    @property
    def filename(self) -> str:
        return f"{self.name}.png"


_STRATUM_TERM: Final[tuple[str, str]] = (
    "stratum",
    f"A session's origin. `seeded` rows were replayed into the store by the benchmark seeder "
    f"and carry the `{SEED_PREFIX}` session-id prefix; `live` rows were served by the router to "
    f"real traffic. `ambiguous` rows are those whose prefix and decision rule disagree.",
)
_VERIFIED_TERM: Final[tuple[str, str]] = (
    "verified success",
    "A session whose Tier-2 (test/typecheck) outcome is a pass. Tier-1 rows are excluded: they "
    "are never materialized and never become routing neighbours.",
)

STRATA: Final[FigureText] = FigureText(
    name="inference_strata",
    title="Two strata share one corpus, and the router cannot tell them apart",
    subtitle="lifecycle stage counts, arrival times and per-model labels, split seeded vs live",
    caveat=None,
    reading=(
        "Panel A counts sessions at five lifecycle stages per stratum: stored, embedded, "
        "labeled, Tier-2 and indexed. They are not a funnel and do not nest: `embedded` and "
        "`indexed` are the only two contained in `stored`, while `labeled` is counted off the "
        "append-only `outcome_events` log and `tier2` off the materialized `outcomes` view, so "
        "a later stage can exceed an earlier one. Read each bar as its own count, and read the "
        "red line for any adjacent pair that actually inverts. Panel B places every session on "
        "its `timestamp`; the seeded stratum is imported in one burst and so collapses to a "
        "single column, which is exactly why any recency-window read over the whole store "
        "reports the benchmark matrix rather than router behaviour. Panel C counts labeled "
        "sessions per model in each stratum."
    ),
    goal=(
        "Make the two populations sharing one outcome store visible before any figure quotes a "
        "number over them, so that a mixed aggregate is recognisable as mixed."
    ),
    definitions=(
        _STRATUM_TERM,
        (
            "stored",
            "A row in `sessions`: the router served the request and wrote the decision. Every "
            "other stage is a subset of the same population, counted a different way.",
        ),
        (
            "embedded",
            "A stored session that also carries an embedding vector "
            "(`sessions.embedding_blob`). Without one it can never be retrieved as a "
            "neighbour, however well it was labelled.",
        ),
        (
            "labeled (any tier)",
            "A session with at least one non-tombstoned event in the append-only "
            "`outcome_events` log, of EITHER tier. Tier-1 is the weak prior read off the wire; "
            "Tier-2 is a verified test or typecheck result.",
        ),
        (
            "verified (tier-2)",
            "A session whose Tier-2 verdict reached the materialized `outcomes` view. A "
            "Tier-1-only session is deliberately held out of that view until a Tier-2 "
            "corroborates it, so it never becomes a routing neighbour.",
        ),
        (
            "indexed",
            "An actual member of the kNN index: embedded, with a materialized outcome and no "
            "tombstone. This is the population a routing decision can draw a neighbour from.",
        ),
    ),
    notes=(
        "Stratum is decided by three signals: the session-id prefix, "
        '`decision_provenance.selection_rule_used == "benchmark_seed"`, and the winning '
        "outcome event's source. Rows where they disagree are counted as `ambiguous` and "
        "surfaced on the canvas rather than assigned to either stratum.",
        "The seeder writes one deterministic timestamp for the whole corpus, so panel B's seeded "
        "column has no width by construction.",
        "A stratum with no sessions draws NO bars in panels A and C, and its legend key reads "
        "`(none in this corpus)` at reduced strength. A zero-height bar and an absent population "
        "must not render the same, and the red line names the empty stratum outright.",
        "There is no Tier-1 bar, and its absence is the point: a Tier-1-only session is kept out "
        "of the materialized view and out of the trusted kNN index until a Tier-2 corroborates "
        "it, so it cannot influence a routing decision. It is not hidden either — `labeled (any "
        "tier)` counts both tiers and `verified (tier-2)` counts only the verified one, so the "
        "GAP between those two bars is exactly the Tier-1-only population.",
    ),
    limitations=(
        "Panel A counts sessions, not requests: a session serving many turns appears once.",
        "A session with no outcome event at all is stored and possibly embedded but never "
        "labeled. That gap is the store's, not this figure's.",
    ),
)

COST: Final[FigureText] = FigureText(
    name="inference_cost",
    title="Live inference cost by model and window, seeded rows excluded",
    subtitle="live inference cost (USD); replayed benchmark spend is excluded by construction",
    caveat=None,
    reading=(
        "Panel A is live spend per model over 7 days, 30 days and the whole store; a model "
        "absent from a window served nothing in it. Panel B is cost coverage: how many live "
        "sessions the provider actually reported a cost for, against how many it did not. "
        "Panel C accumulates live spend over time. An entirely empty figure means the corpus "
        "holds no live sessions, which is the honest answer for a seed-only render."
    ),
    goal=(
        "Report what live inference actually cost, and never let replayed benchmark spend be "
        "read as inference cost — the mislabel this family exists to correct."
    ),
    definitions=(
        _STRATUM_TERM,
        (
            "cost unknown",
            "A session the provider returned no `usage.cost` for. Counted, never summed and "
            "never zero-filled: an unreported cost is unknown, and a real 0.0 is a measurement.",
        ),
    ),
    notes=(
        "Every panel filters to the live stratum. The seeded row count excluded from the sum is "
        "printed in the subtitle, so the exclusion is stated rather than assumed.",
        "Cost is summed over `cost_known = 1` alone; the unknown count is reported beside it as "
        "coverage rather than folded into the total.",
    ),
    limitations=(
        "A window with no live sessions is empty, not zero-cost. The figure states which.",
        "Cost is the provider's reported figure, so a provider that under-reports cache reads "
        "under-reports here too.",
    ),
)

UNIT_ECONOMICS: Final[FigureText] = FigureText(
    name="inference_unit_economics",
    title="Cost per verified success, live traffic against the seeded reference band",
    subtitle="Wilson 95% intervals; hatching marks replayed seeded rows; * marks n<10",
    caveat="the grey band is REPLAYED BENCHMARK outcomes, not live inference",
    reading=(
        "Panel A is the verified-success rate per model with a Wilson 95% interval; a bar marked "
        "* is provisional (fewer than 10 labeled sessions) and its point estimate should not "
        "be ranked against another. Panel B divides spend by verified successes. Grey bars are "
        "the seeded reference band — replayed benchmark outcomes, a reference point and not a "
        "measurement of live routing. Coloured bars are live traffic; where there are none, the "
        "live claim is empty and says so."
    ),
    goal=(
        "Give the cost-per-success question a live answer where live data exists, and a visibly "
        "absent answer where it does not, with the seeded reference never standing in for it."
    ),
    definitions=(
        _VERIFIED_TERM,
        (
            "provisional",
            "Fewer than 10 labeled sessions in the cell. The interval is drawn but the point "
            "estimate is not comparable; the bar is marked * to say so. Hatching is a "
            "different signal entirely — it marks the replayed seeded stratum.",
        ),
    ),
    notes=(
        "The seeded band is drawn from replayed benchmark outcomes whose cost came from the "
        "benchmark run, not from live inference. It is a reference for shape, never a baseline "
        "for live spend.",
        "Cost per verified success is undefined where a model has zero verified successes; that "
        "cell is left empty rather than drawn as an infinite or zero bar.",
    ),
    limitations=(
        "The seeded band inherits the benchmark matrix's model mix, so its per-model n is a "
        "property of the sweep design and not of demand.",
        "Success is Tier-2 only. A model whose work is never verified contributes no successes "
        "however well it performed.",
    ),
)

NEIGHBOURHOOD: Final[FigureText] = FigureText(
    name="inference_neighbourhood",
    title="Do near neighbours agree? Reliability, distance and neighbour origin",
    subtitle="leave-one-out over every indexed session; k nearest, self excluded",
    caveat=None,
    reading=(
        "Panel A bins each session by the success rate of its k nearest neighbours and plots the "
        "realised success rate of the sessions in that bin against the diagonal; points on the "
        "diagonal mean the neighbourhood is calibrated, points below mean it is optimistic. "
        "Panel B is the distribution of neighbour distances — a corpus whose neighbours are all "
        "far away has no neighbourhood to speak of. Panel C asks, for each live decision, what "
        "fraction of its top-k neighbours were seeded rows; on a seed-only corpus there are no "
        "live decisions and the panel is empty."
    ),
    goal=(
        "Test the assumption the whole router rests on — that a near neighbour predicts an "
        "outcome — and show how much of a live decision's evidence is borrowed from the "
        "benchmark corpus."
    ),
    definitions=(
        (
            "leave-one-out",
            "Each indexed session is queried against the index and its own row dropped from the "
            "result, so a session never predicts itself.",
        ),
        (
            "neighbour origin mix",
            "The share of a live decision's k nearest neighbours that are seeded rows. A high "
            "share means the decision was made on replayed benchmark evidence.",
        ),
    ),
    notes=(
        "Panels A and B are computed over the indexed population, which is both strata; the "
        "reliability question is about the embedding space, not about origin.",
        "Distance is the index's own metric, reported unchanged.",
    ),
    limitations=(
        "Leave-one-out over a corpus imported in one burst measures the corpus, not the router's "
        "behaviour over time.",
        "A bin holding few sessions has a noisy realised rate; bin counts are printed so a bin "
        "resting on a handful of sessions is not read as a trend.",
    ),
)

POLICY: Final[FigureText] = FigureText(
    name="inference_policy",
    title="Model share over time, and whether the choice distribution has collapsed",
    subtitle="live share, rolling entropy and frontier share against the loop-health alarms",
    caveat="the seed band is corpus composition, not a routing decision",
    reading=(
        "Panel A is live model share within a trailing window — not cumulative share, which would "
        "dilute a recent collapse with history that has stopped being true. Where the corpus "
        "is seed-only it instead shows "
        "one hatched band: the seeded model distribution, which is the benchmark matrix's sweep "
        "design and not a choice the router made. Panel B tracks rolling choice entropy and "
        "frontier share against the loop-health alarm lines; entropy at or below the alarm means "
        "the distribution has concentrated onto a few arms. Panel C is each model's mean "
        "selection propensity against the exploration floor: a model below the floor has "
        "effectively stopped being tried. Panels B and C are empty where there is no live "
        "traffic to read, and say so on the canvas. Panel B is also empty when no model "
        "registry was supplied, because normalized entropy is undefined without the number "
        "of arms the router could have picked."
    ),
    goal=(
        "Detect a router that has collapsed onto one arm, without ever reading the benchmark "
        "corpus's own model mix as evidence about routing."
    ),
    definitions=(
        (
            "selection propensity",
            "The probability the routing policy assigned to the model it served. Written by the "
            "live router only; a replayed seed row never carries one.",
        ),
        (
            "normalized entropy",
            "Shannon entropy of the model-choice distribution in bits, divided by log2 of the "
            "number of models the router could have picked — the registry's count, not the "
            "count that happen to appear in the window. Same definition the shipped "
            "loop-health alarm uses, so the line drawn here is the line that fires.",
        ),
    ),
    notes=(
        "The seeded band is captioned as corpus composition on the canvas. Presenting the "
        "benchmark matrix's distribution as router behaviour is the exact misread this family "
        "exists to prevent.",
        "Alarm lines come from the shipped `LoopHealthThresholds` defaults, and both the "
        "frontier set and the candidate-arm count come from the shipped model registry via "
        "`top_capability_cluster`, not from a proxy derived here. A figure that re-derived "
        "either would drift from the alarm the router actually raises.",
    ),
    limitations=(
        "Entropy over a window holding fewer sessions than there are arms cannot reach 1.0 and "
        "so reads as collapse; the window size and the arm count are both printed.",
        "Propensity is missing for every non-policy decision — an escalated turn is imposed, not "
        "sampled — so panel C covers policy decisions only.",
    ),
)

ESCALATION: Final[FigureText] = FigureText(
    name="inference_escalation",
    title="Escalation: how often it fires, which rung, why it held, what followed",
    subtitle="live sessions only; hold panel is ladder-evaluated holds plus a derived bar",
    caveat=(
        "panel C covers ladder-evaluated holds only \u2014 a lower bound; "
        "`disabled` is unreachable live"
    ),
    reading=(
        "Panel A is the escalation rate per window. Panel B splits fired escalations by rung: "
        "`raise_effort` keeps the model and steps its reasoning arm, `raise_rank` moves to a "
        "higher-capability model, `escalation_floor` re-serves a rung this task already earned. "
        "Panel C breaks holds down by reason token, plus one derived bar for the holds the "
        "engine never tokenised. Panel D compares verified outcomes before and after an "
        "escalation fired. Empty panels mean the corpus holds no live escalations."
    ),
    goal=(
        "Show whether escalation fires when it should and whether it helps, with the holds "
        "accounted for honestly rather than counted only where the engine happened to name them."
    ),
    definitions=(
        (
            "hold",
            "Escalation ran and did not change what was served. A hold is not the same as "
            "escalation never running: the second leaves no record at all.",
        ),
        (
            UNDELIVERABLE_LABEL,
            "A directive that said raise, on a boundary where no rung could be delivered — no "
            "arm above, or every higher-rank model unhealthy. The engine returns early with the "
            "served model unchanged, so no hold-reason token is written; the case is recovered "
            "from the voided exploration record instead.",
        ),
    ),
    notes=(
        "The hold vocabulary is five tokens: `collapse_suppressed`, `no_recurring_failure`, "
        "`escalation_ceiling`, `exploration_hold`, and `disabled` \u2014 which a live router "
        "cannot emit, because the engine returns before the ladder runs when escalation is off, "
        "so that bar is structurally zero and is drawn only to say the vocabulary is complete. "
        "The derived bar is not a sixth token and is not written by the engine; it is inferred, "
        "and is drawn hatched to say so.",
        "Rung is read from `selection_rule_used` plus the presence of `escalated_reasoning_arm`, "
        "which is what distinguishes an effort step from a rank step.",
    ),
    limitations=(
        "The derived bar recovers only the undeliverable holds that were also being explored. "
        "Where escalation was not exploring, an undeliverable hold leaves no record at all and "
        "is counted nowhere on this figure — panel C is therefore a lower bound on holds, and "
        "is captioned as one.",
        "Panel D compares populations, not the same session under both arms; it is descriptive "
        "and carries no causal claim.",
    ),
)

# The one sentence a routing contrast may never be read without. The reduction that scores the
# multi-arm routing leg through the binary estimator makes the LEVEL exact and the CONTRAST a
# comparison against "took some other arm" — an arbitrary mixture of the remaining candidates,
# not a policy anyone could deploy. Escalation's contrast is a real two-arm decision. The figure
# therefore DRAWS escalation's contrast and omits routing's, and says so on the canvas.
ROUTING_CONTRAST_NOTE: Final[str] = (
    "routing contrast omitted: its complement is “some other arm”, not a deployable "
    "policy — unlike escalation’s"
)

OPE: Final[FigureText] = FigureText(
    name="inference_ope",
    title="Off-policy value of routing and escalation, and whether it is identified at all",
    subtitle=(
        "IPS, SNIPS and doubly-robust values with cluster-bootstrap intervals, beside the "
        "overlap diagnostics that decide whether any of them means anything"
    ),
    caveat=None,
    reading=(
        "Panel A is the value of the target routing policy (serve the top-scored candidate) "
        "against the dashed line the logged policy actually paid. Panel B is the same three "
        "estimators for `always_escalate` and `never_escalate`, plus the contrast "
        "V(escalate) - V(hold), which is the decision question a level cannot answer; the "
        "contrast is read against zero, not against the bars. Panel C is the empirical "
        "distribution of the importance weights, whose right tail is where an off-policy "
        "estimate goes wrong quietly. Panel D divides each identification floor into what the "
        "logs measured, so 1.0 is the floor and a short bar is the reason a panel refused. "
        "A panel is empty only when a leg has no logged decision at all; where a leg has "
        "decisions the estimator cannot use, the panel prints the refusal verbatim instead of "
        "drawing a bar, and that refusal is the figure's result."
    ),
    goal=(
        "Answer whether routing and escalation are worth what they cost, and refuse visibly "
        "when the logs cannot support an answer rather than drawing a plausible bar."
    ),
    definitions=(
        (
            "identified",
            "Both arms were realised under a logging policy that randomized, on enough "
            "independent sessions and at propensities far enough from zero for an inverse to "
            "mean something. Anything less is NOT_IDENTIFIED: undefined, not merely noisy.",
        ),
        (
            "importance weight",
            "The ratio of the target policy's probability of the logged action to the "
            "propensity the logging policy assigned it. Weights are clipped, and the count the "
            "clip actually bound is printed — a large maximum weight alone cannot say it.",
        ),
        (
            "effective sample size",
            "Kish ESS of those weights as a fraction of n: how many observations the estimate "
            "really rests on. A deterministic target gives every un-taken arm weight zero, so "
            "this fraction is capped by the share of rows that took the target's action even on "
            "logs with perfect overlap.",
        ),
        (
            "contrast",
            "V(target) - V(its complement), paired per decision and bootstrapped over the same "
            "session clusters. Only escalation’s is drawn; see the note below.",
        ),
        (
            "a value above 1",
            "Not an error, and not a success rate above 100%. IPS divides the weighted rewards "
            "by n rather than by the sum of the weights, so it is unnormalised and unbounded "
            "above; on a log with small propensities it exceeds 1 routinely. SNIPS divides by "
            "the weight sum and is bounded by the observed rewards; DR is bounded by its "
            "outcome model. A large gap between IPS and SNIPS is therefore a reading of the "
            "weight tail in panel C, not a disagreement about the policy's value.",
        ),
    ),
    notes=(
        "Routing’s contrast is omitted from panel A. The reduction that scores the "
        "multi-arm routing leg through the binary estimator makes the LEVEL exact and the "
        "contrast a comparison against “took some other arm” — an arbitrary mixture of "
        "the remaining candidates, not a policy anyone could deploy. Escalation’s contrast "
        "is a real two-arm decision and means what it says.",
        "Only estimators whose instrument certificate cleared both controls are drawn; an "
        "estimator that failed its control is named on the canvas and left undrawn rather than "
        "quietly averaged into the others.",
    ),
    limitations=(
        "The routing leg covers policy turns only. An escalated turn carries no candidate "
        "scores and a cold-start turn carries none either, so both are excluded before the "
        "estimator sees them; the excluded count is printed on panel D.",
        "An ADMISSIBLE instrument verdict is a gate against breakage — a filter that stopped "
        "filtering, an estimator that stopped weighting — not a warrant that these numbers "
        "are accurate to within a few points.",
    ),
)


MODEL_GRID: Final[FigureText] = FigureText(
    name="inference_model_grid",
    # NOT "the live router used". The store's Tier-2-labeled sessions are BOTH strata, and in
    # a seed-only store every row here is a replayed benchmark session. The old title asserted
    # a live measurement over replay; the subject set and its stratum split are now computed
    # into the subtitle by `data._grid_source`, so an empty live stratum is visible on the
    # canvas rather than only in a limitations paragraph below it.
    title="Every model in the outcome store: price, size, measured outcome",
    subtitle="the benchmark half's grid, redrawn over the outcome store — both strata, split "
    "stated below",
    caveat="Panel A's x axis is what the store was BILLED, so it is not the benchmark half's.",
    reading=(
        "Panel A: each model the store holds a Tier-2 label for, at its mean billed dollars "
        "per labeled session (x, log — with a separate column at the left for locally-served "
        "models, whose marginal price is exactly zero and cannot sit on a log axis) against its "
        "verified-success rate (y), with Wilson 95% whiskers. Marker area follows the square "
        "root of the active parameter count, the marker edge says hosted or local, and hue is "
        "the coarse total-size band. Panel B: a hollow mark at total parameters and a filled "
        "mark at active parameters, the rule between them being the mixture-of-experts "
        "sparsity gap. Panels C and D: per-call latency, hosted and local on separate axes."
    ),
    goal=(
        "Compare this canvas with the benchmark half's. The two draw the same three panels "
        "from different corpora, so a model that sits in one place here and elsewhere there "
        "is telling you the benchmark corpus and the served traffic are not the same "
        "workload — which is the gap the whole live-versus-benchmark question turns on."
    ),
    definitions=(
        _VERIFIED_TERM,
        (
            "$ per labeled session",
            "Billed spend divided by labeled sessions, read off the store's own cost rows. A "
            "model whose sessions are not all priced is left off the axis rather than plotted "
            "at a partial total. This is a MEASURED bill, not a list price, so it is not the "
            "quantity the benchmark half's panel A plots and the two must not be compared.",
        ),
        (
            "active parameters",
            "What one token decodes through — a COMPUTE claim. All of a mixture's total "
            "parameters must still be resident to serve it.",
        ),
        (
            "UNDISCLOSED",
            "The vendor publishes no parameter count. No estimate is substituted, and the row "
            "draws at a fixed reference marker that is deliberately off the size ramp.",
        ),
    ),
    notes=(
        "Panel A's x and the benchmark half's x are different quantities — billed dollars per "
        "session here, a blended list price per million tokens there. The two canvases share a "
        "shape, not an axis, and a model's horizontal position must never be carried between "
        "them.",
    ),
    limitations=(
        "STRATA ARE POOLED ON THIS PANEL, and the subtitle states the split. A row drawn "
        "entirely from seeded sessions was REPLAYED from the benchmark corpus: it repeats the "
        "benchmark half's finding rather than corroborating it, and nothing on this canvas "
        "may be read as live measurement until the subtitle reports live sessions.",
        "The x axis is a mean over sessions of very different sizes, so a model that served "
        "the longer sessions looks dearer per session without being dearer per token.",
        "Rates are not paired: models were not served the same sessions, so a height "
        "difference between two rows is not a controlled comparison.",
    ),
)

FIGURES: Final[tuple[FigureText, ...]] = (
    STRATA,
    COST,
    UNIT_ECONOMICS,
    NEIGHBOURHOOD,
    POLICY,
    ESCALATION,
    MODEL_GRID,
    OPE,
)

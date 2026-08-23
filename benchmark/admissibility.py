"""Re-export shim: the adjudicator now ships in `shunt.analysis.admissibility`.

It moved because SH006 forbids `src/shunt/` importing `benchmark/`, and the shipped inference
figures must state the instrument verdict beside every off-policy number they draw.
"""

from shunt.analysis.admissibility import AdmissibilityResult, admissibility_verdict, run_gate

__all__ = ["AdmissibilityResult", "admissibility_verdict", "run_gate"]

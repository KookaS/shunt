"""Instance-level instrument-validity gate: prove the replay can measure THIS instance at all."""

# WHY THIS EXISTS. The offline replay emits a verdict ("this step passed / failed"), so before any
# of its verdicts may be quoted it has to clear a two-sided control — the positive-control-plus-null
# discipline. Fixing the SELECTOR is not enough: some instance images cannot import the patched test
# module under ANY selector, and for those the replay is not measuring the agent, it is measuring a
# broken container. A negative result from an instrument never shown to produce a positive is a
# coverage-gap, not a finding.
#
# THE TWO CONTROLS, run in the same container, through the SAME assembled pipeline the steps use:
#
#   GOLD  (positive control)      base + gold test_patch + gold patch  MUST classify SUCCESS.
#                                 A known-correct fix is a planted signal: if the pipeline cannot
#                                 recover a pass here, it can never report one, and every red it
#                                 emits for this instance is uninformative (the django polarity).
#   BASE  (destroyed-signal ctrl) base + gold test_patch, fix REMOVED  MUST classify FAILURE.
#                                 With the signal destroyed the pipeline must NOT report success;
#                                 one that still does is stamping greens nothing earned (sympy).
#
# BOTH LEGS ARE ADJUDICATED THE WAY THE GRADER IS, AND BY THE SAME ADJUDICATOR AS THE STEPS.
# `check_instance` takes the `classify` callable the step replay uses — in the real pipeline
# `swebench_grading.GraderParity`, which reads per-test statuses out of the log through SWE-bench's
# own parser and counts only FAIL_TO_PASS ∪ PASS_TO_PASS. It used to be the whole-file exit code,
# which over-rejected: a test the gold patch does not fix, excluded by SWE-bench from BOTH lists,
# turned the positive control red on instances the grader resolves (measured with real containers:
# matplotlib-20676 `1 failed, 34 passed`, the failure `test_widgets.py::test_rectangle_selector`,
# in neither list; also astropy-13033, astropy-13236, astropy-13398, astropy-14096).
#
# THE BASE LEG MOVED WITH IT, AND THAT IS THE LOAD-BEARING HALF. Two-sidedness is only a
# DISCRIMINATION test if both legs answer the same question. Under the exit code, BASE reads
# FAILURE whenever anything in the file fails — including a permanently-broken test that has
# nothing to do with the fix — so an instance whose F2P tests already PASS without the fix could
# still satisfy the destroyed-signal leg, and its replay would then be stamping outcomes that do
# not depend on the agent at all. Per-test, that instance's BASE leg is RESOLVED, so it is
# rejected. Under one shared adjudicator the pair "GOLD resolved ∧ BASE not resolved" says exactly
# "some test in F2P ∪ P2P changes status between fix-present and fix-absent" — which is the
# discriminative power the steps rely on. Mixing the two adjudicators would instead satisfy the
# gate with zero discriminative power. "Not resolved at base" is NOT sufficient on its own,
# though: any red in F2P ∪ P2P satisfies it, and a P2P test passes at base by SWE-bench's own
# construction — so one flaky P2P can supply the red while every F2P test already passes without
# the fix. `adjudicate` therefore also requires the base leg's failure to be a FAIL_TO_PASS test.
#
# The negative control that must keep failing: psf/requests, whose gold-leg failure
# `test_BASICAUTH_TUPLE_HTTP_200_OK_GET` IS in PASS_TO_PASS and fails for want of network. Its
# rejection is genuine — the instance is unmeasurable offline — and per-test adjudication keeps it
# rejected (measured on requests-1724: at gold, 0 of its 6 F2P pass and 24 P2P still fail).
#
# RELATION TO THE SHARED RESEARCH GATE. This mirrors the contract of a shared adjudicator module
# that is not shipped in this repository — controls supplied by the caller, adjudicator separate
# from oracle, verdict about the INSTRUMENT and never about the hypothesis — but it cannot import
# it: a clean clone would not have it. The numeric adjudicator is also the wrong shape here. Its
# null rung asks a score to COLLAPSE TO CHANCE, which suits a continuous metric under shuffled
# labels; this instrument is discrete and its destroyed-signal leg must land on the DEFINITE
# OPPOSITE verdict (FAILURE), not at chance. Same discipline, correct encoding for the modality —
# the documented per-modality destroyed-signal control the shared gate explicitly allows for.

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Final

from benchmark import corpus_lock

if TYPE_CHECKING:
    from benchmark.runner.offline_replay import Classifier, ContainerExec
    from shunt.verifiers.base import VerifierResult

_LOG = logging.getLogger(__name__)

VERDICT_FILENAME = "admissibility.json"

# Every module whose source decides what a verdict IS: the classifier, the assembled replay the
# legs run through, this adjudicator, the normalizer that STAMPS per-step outcomes onto the
# trajectory, the verifier modules that rank/parse outcomes, and the record/schema modules a
# verdict round-trips through. Their combined digest is the instrument fingerprint — edit any of
# them and every cached verdict is invalidated automatically. A hand-bumped version constant
# would be the thing that goes stale silently, which is the failure this key exists to make
# impossible. Closure coverage is pinned by benchmark/tests/test_instrument_digest_closure.py.
_INSTRUMENT_MODULES: Final[tuple[str, ...]] = (
    "shunt.verifiers.parse",
    "shunt.verifiers.tier2",
    "shunt.verifiers.aggregator",
    "benchmark.runner.swebench_grading",
    "benchmark.runner.offline_replay",
    "benchmark.runner.replay_admissibility",
    "benchmark.runner.state_capture_audit",
    "benchmark.runner.step_snapshots",
    "benchmark.escalation.schema",
    "benchmark.escalation.authenticity",
    "benchmark.escalation.normalize.mini_swe_agent",
)


@lru_cache(maxsize=1)
def instrument_digest() -> str:
    """SHA-256 over the source of every module that determines a replay verdict."""
    sha = hashlib.sha256()
    for name in _INSTRUMENT_MODULES:
        spec = find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot fingerprint the replay instrument: {name} has no source")
        sha.update(name.encode())
        sha.update(Path(spec.origin).read_bytes())
    return sha.hexdigest()


def gate_key(
    *,
    dataset_revision: str,
    image_ref: str,
    test_cmd: str,
    test_selectors: list[str],
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
) -> str:
    """Cache key for one instance's verdict: everything a re-run could legitimately change."""
    # The gate costs two full container test runs, and it was being recomputed per TRAJECTORY —
    # 792 pairs for 166 instances, 626 of them redundant (1–31 h). Caching is only safe if a
    # stale entry cannot serve a wrong verdict, so the key covers both halves of "what was
    # measured": the SUBJECT (dataset revision, image, test command, selectors, and the F2P/P2P
    # lists the verdict is now adjudicated against) and the INSTRUMENT (the source digest above).
    payload = json.dumps(
        {
            "dataset_revision": dataset_revision,
            "image_ref": image_ref,
            "test_cmd": test_cmd,
            "test_selectors": test_selectors,
            "fail_to_pass": fail_to_pass or [],
            "pass_to_pass": pass_to_pass or [],
            "instrument": instrument_digest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# INSTANCES WHOSE REJECTION IS A KNOWN ARTIFACT OF THIS INSTRUMENT, NOT A PROPERTY OF THE INSTANCE.
#
# A rejection CLEARS every per-step stamp the instance owns, so an exclusion is a data loss and
# has to be auditable rather than silent. The verdict record already says WHY the legs classified
# as they did; what it cannot say is whether that classification is a real finding ("this instance
# is unmeasurable offline") or a diagnosed defect in the measuring apparatus that was deliberately
# left unfixed. This registry supplies the second reading, and it travels INTO the committed
# `admissibility.json` so a reader of the data — not just a reader of this file — sees it.
#
# IT IS DESCRIPTIVE AND ONLY DESCRIPTIVE. It never appears in the `admissible` decision, and
# `adjudicate` attaches it only to a verdict that already came out INADMISSIBLE. Anything else
# would turn documentation into a suppression list, which is precisely the failure mode a
# clearing gate must not have. An entry whose instance turns admissible is a STALE entry, and it
# is logged as one rather than silently ignored.
#
# Adding an entry edits this module and therefore moves `instrument_digest`, invalidating every
# cached verdict. That is the existing fingerprint contract (any edit here re-measures), not a
# new cost this registry introduces — and the alternative, a registry outside the fingerprint,
# would let recorded output change with nothing to notice.
KNOWN_ARTIFACTS: Final[dict[str, str]] = {
    "django__django-15098": (
        "KNOWN INSTRUMENT ARTIFACT, deliberately NOT fixed. Cost: 251 steps across 5 "
        "trajectories. The base leg demotes to `unknown` because NEITHER FAIL_TO_PASS id "
        "appears in the parsed status map, for two INDEPENDENT reasons that only together "
        "produce zero decided F2P. (1) `test_get_language_from_path_real "
        "(i18n.tests.MiscTests)` fails only through `subTest`, so no terminating `... FAIL` is "
        "written for the parent test and its sole record is the summary line `FAIL: "
        "test_get_language_from_path_real (i18n.tests.MiscTests) (path='/en-latn-us/')`, which "
        "`parse_log_django` keys as `line.split()[1]` — the BARE name, dropping both the "
        "`(module.Class)` qualifier and the subTest parameter. (2) Because that parent's status "
        "line was left unterminated, the NEXT test's header was appended to it, and the merged "
        "line `test_get_language_from_path_real (i18n.tests.MiscTests) ... "
        "test_get_supported_language_variant_null (i18n.tests.MiscTests) ... ok` was keyed WHOLE "
        "by the `... ok` branch — so the second F2P id is absent for an entirely different "
        "reason. A fix that only strips the subTest suffix recovers (1) and does nothing about "
        "(2); a genuine fix must handle BOTH, and must not turn either into a prefix or "
        "substring match that could credit one django test with another's status. SWE-bench's "
        "own grader reaches the CORRECT base verdict on this same map (`grade()` returns "
        "resolved=False, F2P 0/2, P2P 88/88) via its missing-implies-failed rule; the demotion "
        "is OUR `unobserved_signal` guard being deliberately stricter, because trusting an "
        "absent test is what fabricated ~30% of this corpus. The gold leg is RESOLVED (F2P 2/2, "
        "P2P 88/88) and the real grading harness resolved 2/5 of this instance's cells, so the "
        "251 cleared steps are the honest price of that conservatism, not a bug."
    ),
    "psf__requests-5414": (
        "KNOWN INSTRUMENT ARTIFACT, deliberately NOT fixed. Cost: 23 steps across 1 trajectory. "
        "The gold leg's log is `131 passed, 1 xfailed, 158 errors` at rc=1: 158 unrelated tests "
        "error on the `tests/conftest.py:42` fixture that imports `trustme` (absent from the "
        "image) and NOTHING failed — so the log carries no `FAILED <nodeid>` line, "
        "`_ran_and_failed` is False, and the `conftest.py.*ModuleNotFoundError` collection "
        "marker short-circuits the run to infra BEFORE any per-test grading. Per-test the same "
        "log is RESOLVED (F2P 1/1, P2P 130/130) and the real grading harness resolved 7/7 of "
        "this instance's cells. The blind spot is specifically the would-be-RESOLVED run: the "
        "base leg of this same instance does carry a `FAILED` line, so it grades normally "
        "(failure, F2P 0/1, P2P 130/130). This is NOT the network-blocked psf pattern — its P2P "
        "set is fully green offline, unlike requests-1724/1766/2317 (BASICAUTH) and 1921 "
        "(DIGESTAUTH), whose rejections are genuine unmeasurability. It stays rejected anyway: "
        "the only fix is to let a per-test grade override the collection-marker short-circuit, "
        "and that guard is currently the sole thing standing between a broken container and a "
        "fabricated red — trading a bounded 23-step loss for an unbounded fabrication risk is a "
        "bad trade. A safe fix must FIRST establish a positive criterion that the run actually "
        "executed the graded set (e.g. every F2P and P2P id decisively decided in the parsed "
        "map, not merely absent) and prove over the corpus that it re-admits no broken-container "
        "run. Fixing it would move the psf family from 5/5 rejected to 4/5, so it is a "
        "deliberate decision about that family's verdict, not a bug fix."
    ),
}


@dataclass(frozen=True)
class LegOutcome:
    """What one control leg's real container run classified as."""

    leg: str
    outcome: str
    exit_code: int | None
    is_infra_failure: bool
    detail: str

    @classmethod
    def of(cls, leg: str, result: VerifierResult) -> LegOutcome:
        return cls(
            leg=leg,
            outcome=result.outcome,
            exit_code=result.exit_code,
            is_infra_failure=result.is_infra_failure,
            detail=result.detail,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "leg": self.leg,
            "outcome": self.outcome,
            "exit_code": self.exit_code,
            "is_infra_failure": self.is_infra_failure,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LegOutcome:
        return cls(
            leg=str(payload["leg"]),
            outcome=str(payload["outcome"]),
            exit_code=None if payload["exit_code"] is None else int(str(payload["exit_code"])),
            is_infra_failure=bool(payload["is_infra_failure"]),
            detail=str(payload["detail"]),
        )


@dataclass(frozen=True)
class AdmissibilityVerdict:
    """Whether this instance's replay is a valid instrument. NOT a verdict on any agent."""

    instance_id: str
    admissible: bool
    reason: str
    base: LegOutcome
    gold: LegOutcome
    # The `gate_key` this verdict was measured under; "" for a pre-cache record, which never
    # matches a computed key and is therefore always re-measured rather than trusted.
    gate_key: str = field(default="")
    # The diagnosed defect behind this rejection when it is a KNOWN_ARTIFACTS entry, else "".
    # Never read by any decision — it exists so the committed record says "excluded on a known
    # instrument defect" where it would otherwise read as a measurement about the instance.
    known_artifact: str = field(default="")

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "admissible": self.admissible,
            "reason": self.reason,
            "base": self.base.to_dict(),
            "gold": self.gold.to_dict(),
            "gate_key": self.gate_key,
            "known_artifact": self.known_artifact,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AdmissibilityVerdict:
        base, gold = payload["base"], payload["gold"]
        if not isinstance(base, dict) or not isinstance(gold, dict):
            raise TypeError("admissibility record is missing its control legs")
        return cls(
            instance_id=str(payload["instance_id"]),
            admissible=bool(payload["admissible"]),
            reason=str(payload["reason"]),
            base=LegOutcome.from_dict(base),
            gold=LegOutcome.from_dict(gold),
            gate_key=str(payload.get("gate_key", "")),
            known_artifact=str(payload.get("known_artifact", "")),
        )


def _build(
    *,
    instance_id: str,
    admissible: bool,
    reason: str,
    base: LegOutcome,
    gold: LegOutcome,
    gate_key_value: str,
) -> AdmissibilityVerdict:
    """Attach the KNOWN_ARTIFACTS note to a rejection — the ONE place a verdict is constructed."""
    note = KNOWN_ARTIFACTS.get(instance_id, "")
    if note and admissible:
        # The registry claims this instance is rejected on a diagnosed instrument defect. It is
        # not rejected, so the entry now describes a reality that changed — say so loudly rather
        # than shipping a record whose prose contradicts its own verdict.
        _LOG.warning(
            "admissibility %s: STALE KNOWN_ARTIFACTS entry — the instance is now ADMISSIBLE; "
            "remove or rewrite it",
            instance_id,
        )
        note = ""
    return AdmissibilityVerdict(
        instance_id=instance_id,
        admissible=admissible,
        reason=reason,
        base=base,
        gold=gold,
        gate_key=gate_key_value,
        known_artifact=note,
    )


def adjudicate(
    instance_id: str,
    base: VerifierResult,
    gold: VerifierResult,
    fail_to_pass: tuple[str, ...],
    gate_key_value: str = "",
) -> AdmissibilityVerdict:
    """Decide admissibility from the two control legs' classifications (pure — no container)."""
    base_leg, gold_leg = LegOutcome.of("base", base), LegOutcome.of("gold", gold)
    gold_recovers = gold.outcome == "success"
    # THE DESTROYED-SIGNAL LEG MUST FAIL ON THE BUG, NOT ON SOMETHING ELSE. "base is not resolved"
    # is satisfied by ANY red in F2P ∪ P2P, and a P2P test passes at base by SWE-bench's own
    # construction — so a P2P test that is merely flaky supplies the red while every F2P test
    # already passes without the fix. That instance clears a two-sided gate with ZERO power on the
    # axis that encodes the bug, and its steps would then track the flaky test. The grader-parity
    # classifier orders F2P failures first, so `failing_check_id ∈ FAIL_TO_PASS` is exactly "at
    # least one fail-to-pass test is red at base". `fail_to_pass` is REQUIRED, and an empty tuple
    # means "this spec declares no signal", never "skip the check" — a default that quietly
    # disabled the strongest half of the gate is the same trap `replay_step.classify` avoids by
    # having no default at all.
    base_destroys = base.outcome == "failure" and base.failing_check_id in fail_to_pass
    if not base_destroys and base.outcome == "failure":
        reason = (
            f"INADMISSIBLE: the base leg fails, but on {base.failing_check_id!r} — not on any "
            "FAIL_TO_PASS test. Every test that encodes the bug already passes WITHOUT the fix, "
            "so the replay cannot tell a fixed state from an unfixed one for this instance."
        )
        return _build(
            instance_id=instance_id,
            admissible=False,
            reason=reason,
            base=base_leg,
            gold=gold_leg,
            gate_key_value=gate_key_value,
        )
    if gold_recovers and base_destroys:
        reason = (
            "ADMISSIBLE: gold patch classifies SUCCESS (the pipeline can recover a planted fix) "
            "and the base state classifies FAILURE (it does not report success without one)."
        )
    elif not gold_recovers and not base_destroys:
        reason = (
            f"INADMISSIBLE: gold leg is {gold.outcome!r} (not success) AND base leg is "
            f"{base.outcome!r} (not failure) — the replay cannot resolve this instance in either "
            "direction; its per-step outcomes carry no information about the agent."
        )
    elif not gold_recovers:
        reason = (
            f"INADMISSIBLE: gold leg is {gold.outcome!r}, not success — a known-correct fix does "
            f"not register as a pass ({gold.detail}), so every red this instance emits is "
            "uninformative. A negative here is a coverage-gap, not a falsification."
        )
    else:
        reason = (
            f"INADMISSIBLE: base leg is {base.outcome!r}, not failure — the pipeline reports a "
            f"non-failure with the fix REMOVED ({base.detail}), so its passes are unearned."
        )
    return _build(
        instance_id=instance_id,
        admissible=gold_recovers and base_destroys,
        reason=reason,
        base=base_leg,
        gold=gold_leg,
        gate_key_value=gate_key_value,
    )


def check_instance(
    *,
    instance_id: str,
    test_patch: str,
    gold_patch: str,
    test_cmd: str,
    test_selectors: list[str],
    exec_fn: ContainerExec,
    classify: Classifier,
    fail_to_pass: tuple[str, ...],
    gate_key_value: str = "",
) -> AdmissibilityVerdict:
    """Run both control legs in the container through the real step-replay path, then adjudicate."""
    # Imported here, not at module scope, to break the import cycle: `offline_replay` calls this
    # gate, so this module cannot import it eagerly. Using `replay_step` itself is deliberate —
    # a positive control has to exercise the ASSEMBLED instrument (reset → apply → run → classify),
    # not a parallel re-implementation that could pass while the real path is broken. `classify` is
    # threaded through for the same reason: the controls must be adjudicated by the SAME callable
    # the steps are, or the gate certifies an instrument nothing uses.
    from benchmark.runner.offline_replay import replay_step  # noqa: PLC0415

    base = replay_step("", test_patch, test_cmd, test_selectors, exec_fn, classify)
    gold = replay_step(gold_patch, test_patch, test_cmd, test_selectors, exec_fn, classify)
    verdict = adjudicate(instance_id, base, gold, fail_to_pass, gate_key_value)
    _LOG.info("admissibility %s: %s", instance_id, verdict.reason)
    if verdict.known_artifact:
        # Its own line, at WARNING: a supervising monitor reading the rebuild log must be able to
        # tell a diagnosed-and-accepted exclusion from an unexplained one without opening the JSON.
        _LOG.warning("admissibility %s: %s", instance_id, verdict.known_artifact)
    return verdict


def load_verdicts(out_dir: Path) -> dict[str, AdmissibilityVerdict]:
    """Every recorded per-instance verdict in *out_dir* (empty when the file is absent/bad)."""
    path = out_dir / VERDICT_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        _LOG.warning("unreadable %s — re-running every gate", path)
        return {}
    out: dict[str, AdmissibilityVerdict] = {}
    for instance_id, record in payload.items():
        try:
            out[instance_id] = AdmissibilityVerdict.from_dict(record)
        except (KeyError, TypeError, ValueError):
            _LOG.warning("malformed admissibility record for %s — will re-measure", instance_id)
    return out


def cached_verdict(instance_id: str, key: str, out_dir: Path) -> AdmissibilityVerdict | None:
    """A recorded verdict measured under exactly *key*, else None (re-measure)."""
    verdict = load_verdicts(out_dir).get(instance_id)
    if verdict is None or verdict.gate_key != key:
        return None
    _LOG.info("admissibility %s: cache hit (%s)", instance_id, verdict.reason)
    return verdict


def _raw_records(path: Path) -> dict[str, object]:
    """Every record in the verdict file as written, keeping ones this module cannot parse."""
    # Distinct from `load_verdicts`, which drops unparseable entries: a read-modify-write must
    # carry OTHER instances' records through byte-for-byte, or recording one verdict silently
    # deletes the audit trail of the rest.
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Unreachable once every writer goes through the lock below, but a file corrupted by an
        # older unsynchronised run must not crash the child (it used to: an uncaught
        # JSONDecodeError on 282 of 480 concurrent calls) — it must say so and rebuild.
        _LOG.warning("unreadable %s — rebuilding it from this verdict alone", path)
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def record_verdict(verdict: AdmissibilityVerdict, out_dir: Path) -> Path:
    """Persist the per-instance verdict so an exclusion is auditable, not silent."""
    path = out_dir / VERDICT_FILENAME
    # Read-modify-write over a file every parallel worker shares: unsynchronised, 413 of 480
    # verdicts were lost and the file was left unparseable (measured). The lock makes the
    # read+write one transaction; the atomic write makes the file whole at every instant.
    with corpus_lock.corpus_lock(out_dir):
        existing = _raw_records(path)
        existing[verdict.instance_id] = verdict.to_dict()
        corpus_lock.atomic_write_text(path, json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "KNOWN_ARTIFACTS",
    "VERDICT_FILENAME",
    "AdmissibilityVerdict",
    "LegOutcome",
    "adjudicate",
    "cached_verdict",
    "check_instance",
    "gate_key",
    "instrument_digest",
    "load_verdicts",
    "record_verdict",
]

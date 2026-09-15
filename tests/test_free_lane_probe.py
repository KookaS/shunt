"""Hermetic tests for the free-lane admission probe: static gates, live probe, controls.

Fakes are allowed here (tests only): the transports replay a fixed tool-call reply so the
live half is exercised without a network call or a provider key. The live entrypoint itself
is never run; what is tested is the decision machinery it wires together.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmark.runner import free_lane_probe as flp
from shunt.models.config import ModelConfig


def _tool_call(name: str = "bash", arguments: str = '{"command": "pwd"}') -> SimpleNamespace:
    return SimpleNamespace(id="call-1", function=SimpleNamespace(name=name, arguments=arguments))


def _reply(tool_calls: list[Any] | None = None, content: str = "") -> SimpleNamespace:
    message = SimpleNamespace(tool_calls=tool_calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _known_tool_transport(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
    # The scaffold's own schema is what the probe must send; assert it here so a
    # reimplementation with a hand-rolled tool would fail this test.
    assert tools == [flp.BASH_TOOL]
    return _reply([_tool_call()])


def _no_tool_transport(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
    return _reply(None, "I cannot run shell commands.")


def _app_gated_transport(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
    return _reply(None, "This model is only available on agentic harnesses.")


def _facts(**overrides: Any) -> flp.ListingFacts:
    base: dict[str, Any] = {
        "listing_id": "moonshotai/kimi-k3",
        "provider": "nvidia_nim",
        "declares_tools": True,
        "release_date": "2026-06-01",
        "first_seen": "2026-09-01",
        "expiration_date": None,
        "identity": "kimi-k3",
        "published_limits": {"rpm": 20, "rpd": 1000},
    }
    base.update(overrides)
    return flp.ListingFacts(**base)


def _context(**overrides: Any) -> flp.AdmissionContext:
    base: dict[str, Any] = {
        "scan_as_of": date(2026, 9, 10),
        "run_window_end": date(2026, 10, 10),
    }
    base.update(overrides)
    return flp.AdmissionContext(**base)


# --- the live tool-call probe -------------------------------------------------


class TestToolCallProbe:
    def test_uses_five_cases_and_the_scaffold_bash_tool(self) -> None:
        assert len(flp.TOOLCALL_CASES) == 5
        assert flp.BASH_TOOL["function"]["name"] == "bash"

    def test_known_tool_calling_transport_passes_every_case(self) -> None:
        result = flp.run_toolcall_probe(_known_tool_transport)
        assert result.passes == 5
        assert result.attempts == 5
        assert result.availability == 1.0
        assert not result.app_gated

    def test_no_tool_transport_collapses_to_zero(self) -> None:
        result = flp.run_toolcall_probe(_no_tool_transport)
        assert result.passes == 0
        assert result.availability == 0.0

    def test_non_bash_tool_is_not_a_pass(self) -> None:
        def wrong_tool(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
            return _reply([_tool_call(name="python", arguments='{"code": "1"}')])

        assert flp.run_toolcall_probe(wrong_tool).passes == 0

    def test_malformed_arguments_are_not_a_pass(self) -> None:
        def bad_json(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
            return _reply([_tool_call(arguments="not json")])

        assert flp.run_toolcall_probe(bad_json).passes == 0

    def test_app_gate_language_is_detected(self) -> None:
        result = flp.run_toolcall_probe(_app_gated_transport)
        assert result.app_gated

    def test_one_transport_error_does_not_abort_the_probe(self) -> None:
        calls = {"n": 0}

        def flaky(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("provider blip")
            return _reply([_tool_call()])

        result = flp.run_toolcall_probe(flaky)
        assert result.attempts == 5
        assert result.passes == 4
        assert result.errors


# --- the static gates ---------------------------------------------------------


class TestStaticGates:
    def test_a_clean_listing_passes_every_static_gate(self) -> None:
        checks = flp.static_gate_checks(_facts(), _context())
        assert all(c.passed for c in checks)
        assert {c.half for c in checks} == {flp.GateHalf.STATIC}

    def test_undeclared_tools_refuses(self) -> None:
        facts = _facts(declares_tools=False)
        checks = {c.name: c for c in flp.static_gate_checks(facts, _context())}
        assert not checks["declares_tool_calling"].passed

    def test_unknown_tool_surface_defers_to_the_live_probe(self) -> None:
        # `None` is not `False`: a catalogue silent on tools passes the static half and the
        # live tool-call probe decides, so the lane is never silently refused or admitted.
        facts = _facts(declares_tools=None)
        checks = {c.name: c for c in flp.static_gate_checks(facts, _context())}
        assert checks["declares_tool_calling"].passed
        assert "probe" in checks["declares_tool_calling"].reason
        # The assembled verdict still requires the live probe to pass.
        verdict = flp.evaluate_admission(_no_tool_transport, facts, _context())
        assert not verdict.admitted

    def test_stale_release_date_refuses(self) -> None:
        checks = {
            c.name: c for c in flp.static_gate_checks(_facts(release_date="2020-01-01"), _context())
        }
        assert not checks["release_date_within_18_months"].passed

    def test_unknown_release_date_refuses(self) -> None:
        checks = {c.name: c for c in flp.static_gate_checks(_facts(release_date=None), _context())}
        assert not checks["release_date_within_18_months"].passed

    def test_release_age_boundary_is_calendar_months(self) -> None:
        # 18 months before 2026-09-10 is 2025-03-10; one day older is the first refusal.
        edge = flp.static_gate_checks(_facts(release_date="2025-03-10"), _context())
        older = flp.static_gate_checks(_facts(release_date="2025-03-09"), _context())
        assert {c.name: c for c in edge}["release_date_within_18_months"].passed
        assert not {c.name: c for c in older}["release_date_within_18_months"].passed

    def test_first_sighting_refuses_the_consecutive_scan_gate(self) -> None:
        checks = {
            c.name: c for c in flp.static_gate_checks(_facts(first_seen="2026-09-10"), _context())
        }
        assert not checks["seen_in_2_consecutive_scans"].passed

    def test_expiration_inside_the_window_refuses(self) -> None:
        checks = {
            c.name: c
            for c in flp.static_gate_checks(_facts(expiration_date="2026-09-30"), _context())
        }
        assert not checks["no_expiration_in_window"].passed

    def test_unresolved_identity_refuses(self) -> None:
        checks = {c.name: c for c in flp.static_gate_checks(_facts(identity=None), _context())}
        assert not checks["identity_resolved"].passed

    def test_budget_below_one_cell_refuses(self) -> None:
        checks = {
            c.name: c
            for c in flp.static_gate_checks(
                _facts(published_limits={"rpm": 20, "rpd": 20}), _context()
            )
        }
        assert not checks["daily_budget_ge_one_cell"].passed


class TestResolveLimits:
    def test_published_limits_are_used_verbatim(self) -> None:
        limits = flp.resolve_limits({"rpm": 30, "rpd": 1000})
        assert (limits.rpm, limits.rpd, limits.known) == (30, 1000, True)

    def test_unknown_limits_get_the_declared_defaults_never_an_invented_number(self) -> None:
        limits = flp.resolve_limits({"rpm": None, "rpd": None})
        assert (limits.rpm, limits.rpd, limits.known) == (
            flp.UNKNOWN_LIMITS_RPM,
            flp.UNKNOWN_LIMITS_RPD,
            False,
        )

    def test_a_partially_published_row_is_not_marked_known(self) -> None:
        limits = flp.resolve_limits({"rpm": 30, "rpd": None})
        assert not limits.known
        assert limits.rpd == flp.UNKNOWN_LIMITS_RPD


# --- the assembled verdict ----------------------------------------------------


class TestEvaluateAdmission:
    def test_a_good_listing_on_a_tool_calling_model_is_admitted(self) -> None:
        verdict = flp.evaluate_admission(_known_tool_transport, _facts(), _context())
        assert verdict.admitted
        assert not verdict.refusals
        assert verdict.identity == "kimi-k3"

    def test_a_good_listing_on_a_no_tool_model_is_refused(self) -> None:
        verdict = flp.evaluate_admission(_no_tool_transport, _facts(), _context())
        assert not verdict.admitted
        assert {c.name for c in verdict.refusals} >= {
            "tool_call_probe",
            "availability_at_threshold",
        }

    def test_app_gating_refuses_even_when_a_tool_call_parses(self) -> None:
        def gated_with_tool(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
            return _reply([_tool_call()], "This model is only available on agentic harnesses.")

        verdict = flp.evaluate_admission(gated_with_tool, _facts(), _context())
        assert not verdict.admitted
        assert "not_app_gated" in {c.name for c in verdict.refusals}


# --- instrument validity: positive + destroyed-signal controls ----------------


class TestControls:
    def test_controls_admit_a_known_tool_caller_and_refuse_a_no_tool_model(self) -> None:
        outcome = flp.evaluate_controls(
            flp.AdmissionControls(_known_tool_transport, _no_tool_transport),
            positive_facts=_facts(),
            null_facts=_facts(),
            context=_context(),
        )
        assert outcome.positive.admitted
        assert not outcome.null.admitted
        assert outcome.adjudication.admissible
        assert outcome.adjudication.positive_passed
        assert outcome.adjudication.null_at_chance

    def test_each_control_transport_runs_exactly_once(self) -> None:
        # A live probe costs free quota; the control's verdict and its score must share one
        # probe run, so each transport is called once per case (5), never doubled.
        calls = {"positive": 0, "null": 0}

        def positive(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
            calls["positive"] += 1
            return _reply([_tool_call()])

        def null(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
            calls["null"] += 1
            return _reply(None, "no commands")

        flp.evaluate_controls(
            flp.AdmissionControls(positive, null),
            positive_facts=_facts(),
            null_facts=_facts(),
            context=_context(),
        )
        assert calls == {"positive": 5, "null": 5}

    def test_an_instrument_that_scores_high_on_the_null_is_inadmissible(self) -> None:
        # The killer shape: the probe reports tool calls even for the destroyed-signal
        # model, so its "pass" carries no information. The adjudicator must reject it.
        outcome = flp.evaluate_controls(
            flp.AdmissionControls(_known_tool_transport, _known_tool_transport),
            positive_facts=_facts(),
            null_facts=_facts(),
            context=_context(),
        )
        assert outcome.adjudication.positive_passed
        assert not outcome.adjudication.null_at_chance
        assert not outcome.adjudication.admissible


# --- wiring helpers -----------------------------------------------------------


class TestAnonymousFreeLaneAdmission:
    """A key-optional provider is admitted without a key; a key-required one still refuses."""

    @staticmethod
    def _model(provider: str, *, key_optional: bool) -> ModelConfig:
        return ModelConfig(
            name=f"{provider}-free",
            model_id="poolside/laguna-xs-2.1:free",
            provider=provider,
            base_url="https://api.kilo.ai/api/gateway",
            api_key_env_var="KILO_API_KEY",
            key_optional=key_optional,
        )

    def test_kilo_lane_with_no_key_is_admitted_as_anonymous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KILO_API_KEY", raising=False)
        captured: dict[str, Any] = {}

        def fake_transport(
            *, route: str, api_base: str, api_key: str, max_tokens: int = 512
        ) -> Any:
            captured.update(route=route, api_base=api_base, api_key=api_key)
            return _known_tool_transport

        monkeypatch.setattr(flp, "litellm_transport", fake_transport)
        transport = flp._transport_for(self._model("kilo_gateway", key_optional=True))
        assert callable(transport)
        # A harmless placeholder, never a billed path: the real key is absent.
        assert captured["api_key"] == flp.ANONYMOUS_API_KEY
        assert captured["route"] == "openai/poolside/laguna-xs-2.1:free"
        assert captured["api_base"] == "https://api.kilo.ai/api/gateway"

    def test_a_provider_that_requires_a_key_still_refuses_without_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KILO_API_KEY", raising=False)
        with pytest.raises(SystemExit, match=r"\$KILO_API_KEY is not set"):
            flp._transport_for(self._model("groq", key_optional=False))

    def test_an_anonymous_lane_still_prefers_a_real_key_when_one_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KILO_API_KEY", "real-key")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            flp,
            "litellm_transport",
            lambda **kw: captured.update(kw) or _known_tool_transport,
        )
        flp._transport_for(self._model("kilo_gateway", key_optional=True))
        assert captured["api_key"] == "real-key"


class TestWiringHelpers:
    def test_identity_lookup_reads_confirmed_and_proposed(self) -> None:
        proposal = {
            "confirmed": {"kimi-k3": {"listings": ["openrouter:moonshotai/kimi-k3:free"]}},
            "proposed": {"glm-5.3": {"listings": ["groq:glm-5.3"]}},
        }
        lookup = flp.identity_lookup(proposal)
        assert lookup[("openrouter", "moonshotai/kimi-k3:free")] == "kimi-k3"
        assert lookup[("groq", "glm-5.3")] == "glm-5.3"

    def test_snapshot_row_adapts_to_listing_facts(self) -> None:
        # This test previously omitted `release_date` from the row, so it passed while the
        # scanner wrote no such field and every real listing failed the age gate. It now
        # carries the metadata field the adapter must forward.
        row = {
            "listing_id": "moonshotai/kimi-k3",
            "provider": "nvidia_nim",
            "supports_tools": True,
            "release_date": "2026-06-01",
            "first_seen": "2026-09-01",
            "expiration_date": None,
            "published_limits": {"rpm": None, "rpd": None},
        }
        facts = flp.facts_from_snapshot_row(row, identity="kimi-k3")
        assert facts.identity == "kimi-k3"
        assert facts.declares_tools
        assert facts.release_date == "2026-06-01"
        assert facts.published_limits == {"rpm": None, "rpd": None}

    def test_a_missing_supports_tools_field_is_unknown_not_false(self) -> None:
        row = {
            "listing_id": "x",
            "provider": "p",
            "release_date": "2026-06-01",
            "first_seen": "2026-09-01",
            "expiration_date": None,
            "published_limits": {},
        }
        facts = flp.facts_from_snapshot_row(row, identity=None)
        assert facts.declares_tools is None


def test_entrypoint_requires_live_opt_in(capsys: pytest.CaptureFixture[str]) -> None:
    assert flp.main(["--registry", "does-not-matter.yaml"]) == 2
    assert "pass --live" in capsys.readouterr().err


# --- the blocker: release_date must really arrive from models.dev -------------


class TestCommittedSnapshotReleaseDates:
    """The regression that made 140/140 listings fail the age gate: nothing wrote release_date.

    The committed snapshot is real fetched data, so this asserts the field is POPULATED and
    that the real models.dev matches pass the gate — never that a fake date was injected.
    """

    def test_real_modelsdev_matches_carry_a_release_date(self) -> None:
        snapshot = flp.load_snapshot()
        active = [r for r in snapshot["listings"] if not r.get("withdrawn_at")]
        matched = [r for r in active if r.get("release_date")]
        assert matched, "no active listing carries release_date — the metadata join regressed"
        assert snapshot["metadata_sources"]["modelsdev"]["status"].startswith("ok")

    def test_at_least_one_real_match_passes_release_date_within_18_months(self) -> None:
        snapshot = flp.load_snapshot()
        identities = flp.identity_lookup(flp.load_proposal())
        context = _context()
        checks = [
            flp._gate_release_age(
                flp.facts_from_snapshot_row(
                    row, identity=identities.get((row["provider"], row["listing_id"]))
                ),
                context,
            )
            for row in snapshot["listings"]
            if not row.get("withdrawn_at") and row.get("release_date")
        ]
        assert checks, "no real models.dev match to verify"
        assert any(c.passed for c in checks)
        assert all("unknown" not in c.reason for c in checks)


# --- the pause document: batch report + wall clock ---------------------------


def _report_snapshot() -> dict[str, Any]:
    return {
        "scan_as_of": "2026-09-10",
        "listings": [
            {
                "listing_id": "good",
                "provider": "openrouter",
                "supports_tools": True,
                "release_date": "2026-06-01",
                "first_seen": "2026-08-01",
                "expiration_date": None,
                "published_limits": {"rpm": 20, "rpd": 1000},
                "withdrawn_at": None,
            },
            {
                "listing_id": "bad",
                "provider": "groq",
                "supports_tools": False,
                "release_date": None,
                "first_seen": "2026-09-10",
                "expiration_date": None,
                "published_limits": {"rpm": None, "rpd": None},
                "withdrawn_at": None,
            },
        ],
    }


def test_every_listing_gets_a_verdict_with_a_reason() -> None:
    proposal = {"confirmed": {"m": {"listings": ["openrouter:good"]}}, "proposed": {}}
    reports = flp.build_listing_reports(_report_snapshot(), proposal, _context())
    good = next(r for r in reports if r.facts.listing_id == "good")
    bad = next(r for r in reports if r.facts.listing_id == "bad")
    assert good.statically_admitted and not good.refusals
    assert not bad.statically_admitted
    assert bad.refusals and all(r.reason for r in bad.refusals)


def test_wall_clock_is_the_max_of_the_host_and_per_model_bounds() -> None:
    proposal = {"confirmed": {"m": {"listings": ["openrouter:good"]}}, "proposed": {}}
    reports = flp.build_listing_reports(_report_snapshot(), proposal, _context())
    wall = flp.estimate_wall_clock(reports)
    assert wall.models == 1
    assert wall.slowest_model_days == pytest.approx(20 * 85 / 1000)
    assert wall.host_bound_days == pytest.approx(20 / 8 / 24)
    assert wall.estimate_days == pytest.approx(wall.slowest_model_days)


def test_unknown_limits_lanes_are_reserved_at_the_declared_default() -> None:
    facts = _facts(identity="m", published_limits={"rpm": None, "rpd": None})
    report = flp.ListingReport(
        facts=facts, limits=flp.resolve_limits(facts.published_limits), checks=()
    )
    wall = flp.estimate_wall_clock([report])
    assert wall.per_model[0].capacity_rpd == flp.UNKNOWN_LIMITS_RPD
    assert wall.per_model[0].limits_known is False


def test_batch_report_writes_json_and_markdown_with_no_keys(tmp_path: Path) -> None:
    assert flp.main(["--all", "--out-dir", str(tmp_path)]) == 0
    stamp_dirs = list(tmp_path.iterdir())
    assert len(stamp_dirs) == 1
    payload = json.loads((stamp_dirs[0] / "report.json").read_text())
    assert "wall_clock" in payload
    assert payload["summary"]["listings"] == len(payload["listings"])
    for item in payload["listings"]:
        assert item["gates"]
        assert item["refusal_reasons"] is not None
    assert (stamp_dirs[0] / "report.md").read_text().startswith("# Free-lane admission scan")

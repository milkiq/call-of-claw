import re

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from trpg_agent.langchain.advisors import ADVISOR_SPECS, invoke_advisor
from trpg_agent.langchain.prompts import validate_runtime_prompts_are_generic
from trpg_agent.langchain.structured import (
    AuthorityGateResult,
    AuthorityMicroGateDecision,
    CriticReport,
    IntentMicroGateDecision,
    IntentRoutingDecision,
    MemoryCurationDecision,
    MemoryRecallMicroGateDecision,
    RiskMicroGateDecision,
    RulesAdjudicationAdvice,
    ScenarioDirectorDecision,
    SingleTurnAdvisorDecision,
    TargetMicroGateDecision,
)


def test_runtime_advisor_prompts_are_generic() -> None:
    forbidden_terms = [
        "Lasers",
        "Feelings",
        "lasers",
        "feelings",
        "激光",
        "感情",
        "姆姆",
        "水晶",
        "调频器",
        "维加",
        "浅蓝港",
        "铃兰",
        "达西",
        "海盗",
    ]

    assert validate_runtime_prompts_are_generic(forbidden_terms) == {}


def test_advisor_specs_cover_expected_roles() -> None:
    assert set(ADVISOR_SPECS) == {
        "intent_arbiter",
        "authority_gate",
        "authority_micro_gate",
        "intent_micro_gate",
        "risk_micro_gate",
        "target_micro_gate",
        "memory_recall_micro_gate",
        "rules_adjudicator",
        "scenario_director",
        "single_turn_advisor",
        "memory_curator",
        "critic_guardrail",
    }
    assert all(
        re.fullmatch(r"[a-z-]+-v[1-9][0-9]*", spec.prompt_version)
        for spec in ADVISOR_SPECS.values()
    )


def test_advisor_structured_contracts_validate() -> None:
    IntentRoutingDecision.model_validate(
        {
            "intent": {"kind": "action", "confidence": 0.8, "reason": "attempt"},
            "route": "risky_action",
            "needs_rules_resolution": True,
            "needs_scenario_director": True,
            "needs_memory_recall": False,
            "allow_direct_answer": False,
            "reasoning_summary": "The action has uncertain consequences.",
            "citations": [],
        }
    )
    AuthorityGateResult.model_validate(
        {
            "authority": {"ok": True, "reason": "grounded"},
            "allowed_next_step": "continue",
        }
    )
    AuthorityMicroGateDecision.model_validate(
        {
            "allowed": True,
            "boundary": False,
            "needs_clarification": False,
            "reason": "The player proposes an attempt rather than declaring an outcome.",
            "player_facing_boundary": None,
        }
    )
    IntentMicroGateDecision.model_validate(
        {
            "intent": {"kind": "action", "confidence": 0.8, "reason": "attempt"},
            "route": "free_action",
            "allow_direct_answer": False,
            "needs_scenario_director": True,
            "reason": "The input proposes an ordinary action.",
        }
    )
    RiskMicroGateDecision.model_validate(
        {
            "risky": True,
            "risk": "risky_uncertain",
            "needs_rules_resolution": True,
            "reason": "The result is uncertain and consequential.",
        }
    )
    TargetMicroGateDecision.model_validate(
        {
            "ambiguous": True,
            "needs_clarification": True,
            "clarification_question": "Which visible target do you mean?",
            "reason": "The visible scene has multiple plausible targets.",
        }
    )
    MemoryRecallMicroGateDecision.model_validate(
        {
            "needs_memory_recall": True,
            "reason": "The player asks about prior events.",
        }
    )
    RulesAdjudicationAdvice.model_validate(
        {
            "requires_resolution": True,
            "procedure_id": "loaded_procedure",
            "approach_id": "loaded_approach",
            "risk": "risky_uncertain",
            "stakes": "The result changes the immediate situation.",
            "citations": ["rules:procedure"],
        }
    )
    ScenarioDirectorDecision.model_validate(
        {
            "decision": "advance_pressure",
            "proposed_patches": [{"op": "increment", "path": ["clock", "value"], "value": 1}],
            "player_visible_context": "Pressure increases in a visible way.",
            "gm_only_reason": "The scenario clock advances after delay.",
            "citations": ["scenario:clock"],
        }
    )
    SingleTurnAdvisorDecision.model_validate(
        {
            "routing_decision": {
                "intent": {"kind": "action", "confidence": 0.8, "reason": "attempt"},
                "route": "free_action",
                "needs_rules_resolution": False,
                "needs_scenario_director": True,
                "needs_memory_recall": False,
                "allow_direct_answer": False,
                "reasoning_summary": "Proceed without rules resolution.",
                "uncertainty": None,
                "citations": [],
            },
            "rules_advice": {
                "requires_resolution": False,
                "procedure_id": None,
                "approach_id": None,
                "requested_roll": None,
                "risk": "none",
                "stakes": "No risky uncertainty.",
                "clarification_question": None,
                "citations": [],
            },
            "turn_plan": {
                "intent": {"kind": "action", "confidence": 0.8, "reason": "attempt"},
                "authority": {"ok": True, "reason": "grounded"},
                "decision": "free_action",
                "tool_requests": [],
                "narration_brief": "Proceed with a free action.",
                "citations": [],
            },
            "scenario_advice": {
                "decision": "no_change",
                "proposed_patches": [],
                "player_visible_context": "No scene change.",
                "gm_only_reason": "No patch needed.",
                "citations": [],
            },
            "reasoning_summary": "Combined contract validates.",
        }
    )
    MemoryCurationDecision.model_validate(
        {
            "canon_event_draft": {"event": "The player waited."},
            "memory_candidates": [
                {
                    "kind": "unresolved_thread",
                    "text": "A visible pressure remains unresolved.",
                    "scope": "session",
                    "confidence": 0.9,
                }
            ],
            "contradictions": [],
            "should_write": True,
        }
    )
    CriticReport.model_validate(
        {
            "ok": True,
            "blocks_output": False,
            "findings": [],
            "revised_final_text": None,
            "reasoning_summary": "The output follows the tool results.",
        }
    )


def test_invoke_advisor_parses_structured_output() -> None:
    model = FakeListChatModel(
        responses=[
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "player attempts"},
              "route": "risky_action",
              "needs_rules_resolution": true,
              "needs_scenario_director": true,
              "needs_memory_recall": false,
              "allow_direct_answer": false,
              "reasoning_summary": "The action is risky and uncertain.",
              "uncertainty": null,
              "citations": []
            }
            """
        ]
    )

    result = invoke_advisor(
        model=model,
        role="intent_arbiter",
        player_input="I try something risky.",
        context={"world_projection": {}},
    )

    assert isinstance(result.output, IntentRoutingDecision)
    assert result.output.needs_rules_resolution is True
    assert result.trace_metadata["advisor_role"] == "intent_arbiter"
    assert result.trace_metadata["prompt_version"] == "intent-arbiter-v4"
    assert result.trace_metadata["schema"] == "IntentRoutingDecision"
    assert result.trace_metadata["cached"] == "false"
    assert int(result.trace_metadata["elapsed_ms"]) >= 0
    assert int(result.trace_metadata["estimated_prompt_chars"]) > 0
    assert int(result.trace_metadata["estimated_response_chars"]) > 0


def test_invoke_advisor_reuses_cached_output(tmp_path) -> None:
    model = FakeListChatModel(
        responses=[
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "player attempts"},
              "route": "free_action",
              "needs_rules_resolution": false,
              "needs_scenario_director": true,
              "needs_memory_recall": false,
              "allow_direct_answer": false,
              "reasoning_summary": "The action is free.",
              "uncertainty": null,
              "citations": []
            }
            """,
            "this would fail if cache is not used"
        ]
    )
    kwargs = {
        "model": model,
        "role": "intent_arbiter",
        "player_input": "I inspect the area.",
        "context": {"world_projection": {"active_scene": "x"}},
        "sqlite_path": str(tmp_path / "advisor.sqlite"),
        "turn_id": "t1",
    }

    first = invoke_advisor(**kwargs)
    second = invoke_advisor(**kwargs)

    assert first.cached is False
    assert second.cached is True
    assert second.output.model_dump() == first.output.model_dump()
    assert second.trace_metadata["cached"] == "true"
    assert second.trace_metadata["attempt_count"] == "1"

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from coc.langchain.prompts import JSON_REPAIR_PROMPT


class IntentClassification(BaseModel):
    kind: Literal[
        "action",
        "info_query",
        "rules_query",
        "memory_recall",
        "character_setup",
        "passive",
        "boundary_claim",
        "clarify_needed",
    ]
    confidence: float = Field(ge=0, le=1)
    reason: str


class AuthorityResult(BaseModel):
    ok: bool
    reason: str
    unsupported_claim: str | None = None
    grounded_alternatives: list[str] = Field(default_factory=list)


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict
    reason: str


class ToolResult(BaseModel):
    tool_name: str
    ok: bool
    result: dict | list | str | int | float | bool | None = None
    error: str | None = None
    request: ToolRequest | None = None


class TurnPlan(BaseModel):
    intent: IntentClassification
    authority: AuthorityResult
    decision: Literal["answer", "free_action", "risky_action", "gm_move", "boundary", "clarify"]
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    narration_brief: str
    citations: list[str] = Field(default_factory=list)


class NarrationPlan(BaseModel):
    final_text: str
    canon_event_draft: dict | None = None
    memory_candidates: list[str] = Field(default_factory=list)


class IntentRoutingDecision(BaseModel):
    intent: IntentClassification
    route: Literal[
        "answer",
        "clarify",
        "free_action",
        "risky_action",
        "rules_query",
        "memory_recall",
        "boundary",
        "gm_move",
    ]
    needs_rules_resolution: bool = False
    needs_scenario_director: bool = True
    needs_memory_recall: bool = False
    allow_direct_answer: bool = False
    reasoning_summary: str
    uncertainty: str | None = None
    citations: list[str] = Field(default_factory=list)


class AuthorityGateResult(BaseModel):
    authority: AuthorityResult
    allowed_next_step: Literal["continue", "clarify", "boundary"]
    player_facing_boundary: str | None = None


class AuthorityMicroGateDecision(BaseModel):
    allowed: bool
    boundary: bool = False
    needs_clarification: bool = False
    reason: str
    player_facing_boundary: str | None = None


class IntentMicroGateDecision(BaseModel):
    intent: IntentClassification
    route: Literal[
        "answer",
        "clarify",
        "free_action",
        "rules_query",
        "memory_recall",
        "boundary",
        "gm_move",
    ]
    allow_direct_answer: bool = False
    needs_scenario_director: bool = True
    reason: str


class RiskMicroGateDecision(BaseModel):
    risky: bool
    risk: Literal["none", "low", "risky_uncertain", "high"] = "none"
    needs_rules_resolution: bool = False
    reason: str


class TargetMicroGateDecision(BaseModel):
    ambiguous: bool
    needs_clarification: bool = False
    clarification_question: str | None = None
    reason: str


class MemoryRecallMicroGateDecision(BaseModel):
    needs_memory_recall: bool
    reason: str


class RulesAdjudicationAdvice(BaseModel):
    requires_resolution: bool
    procedure_id: str | None = None
    approach_id: str | None = None
    check_id: str | None = None
    difficulty: str | None = None
    modifier: str | None = None
    pushed: bool = False
    risk: Literal["none", "low", "risky_uncertain", "high"] = "risky_uncertain"
    stakes: str
    clarification_question: str | None = None
    citations: list[str] = Field(default_factory=list)


class ScenarioDirectorDecision(BaseModel):
    decision: Literal[
        "no_change",
        "reveal",
        "transition",
        "advance_pressure",
        "consequence",
        "clarify",
        "ending",
    ]
    proposed_patches: list[dict] = Field(default_factory=list)
    transition_id: str | None = None
    trigger_evidence: list[str] = Field(default_factory=list)
    player_visible_context: str
    gm_only_reason: str
    citations: list[str] = Field(default_factory=list)


class ScenarioSurfaceSelectorDecision(BaseModel):
    decision: Literal["select", "fallback"]
    surface_id: str | None = None
    fallback_to_full_director: bool = False
    reason: str
    citations: list[str] = Field(default_factory=list)


class SingleTurnAdvisorDecision(BaseModel):
    routing_decision: IntentRoutingDecision
    rules_advice: RulesAdjudicationAdvice
    turn_plan: TurnPlan
    scenario_advice: ScenarioDirectorDecision
    reasoning_summary: str


class MemoryCandidate(BaseModel):
    kind: Literal[
        "canon",
        "character_state",
        "player_preference",
        "unresolved_thread",
        "episodic_summary",
        "procedural_note",
    ]
    text: str
    scope: Literal["session", "player", "campaign", "system"] = "session"
    confidence: float = Field(ge=0, le=1)
    metadata: dict = Field(default_factory=dict)


class MemoryCurationDecision(BaseModel):
    canon_event_draft: dict | None = None
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    should_write: bool = True


class CriticFinding(BaseModel):
    dimension: Literal[
        "hidden_leak",
        "unsupported_fact",
        "resolver_bypass",
        "canon_contradiction",
        "player_agency",
        "pacing",
        "clarification",
        "narration_quality",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    evidence: str | None = None


class CriticReport(BaseModel):
    ok: bool
    blocks_output: bool = False
    findings: list[CriticFinding] = Field(default_factory=list)
    revised_final_text: str | None = None
    reasoning_summary: str


class CompactToolRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(alias="tool", max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict, alias="args")
    reason: str = Field(default="", alias="why", max_length=160)


class IntentRoutingWire(BaseModel):
    route: Literal[
        "answer",
        "clarify",
        "free_action",
        "risky_action",
        "rules_query",
        "memory_recall",
        "boundary",
        "gm_move",
    ]
    flags: list[Literal["rules", "scenario", "memory", "direct"]] = Field(
        default_factory=list,
        max_length=4,
    )
    confidence: float = Field(default=0.75, ge=0, le=1)
    message: str = Field(default="", max_length=220)
    code: str | None = Field(default=None, max_length=80)
    refs: list[str] = Field(default_factory=list, max_length=5)


class RulesAdviceWire(BaseModel):
    resolve: bool = False
    proc: str | None = Field(default=None, max_length=80)
    approach: str | None = Field(default=None, max_length=80)
    check: str | None = Field(default=None, max_length=80)
    diff: str | None = Field(default=None, max_length=40)
    mod: str | None = Field(default=None, max_length=40)
    pushed: bool = False
    risk: Literal["none", "low", "risky_uncertain", "high"] = "none"
    stakes: str = Field(default="", max_length=220)
    clarify: str | None = Field(default=None, max_length=220)
    refs: list[str] = Field(default_factory=list, max_length=5)


class TurnPlanWire(BaseModel):
    decision: Literal["answer", "free_action", "risky_action", "gm_move", "boundary", "clarify"]
    brief: str = Field(default="", max_length=300)
    tools: list[CompactToolRequest] = Field(default_factory=list, max_length=4)
    message: str = Field(default="", max_length=220)
    code: str | None = Field(default=None, max_length=80)
    refs: list[str] = Field(default_factory=list, max_length=5)


class ScenarioDirectorWire(BaseModel):
    decision: Literal[
        "no_change",
        "reveal",
        "transition",
        "advance_pressure",
        "consequence",
        "clarify",
        "ending",
    ]
    patches: list[dict[str, Any]] = Field(default_factory=list, max_length=4)
    trans: str | None = Field(default=None, max_length=80)
    evidence: list[str] = Field(default_factory=list, max_length=6)
    visible: str = Field(default="", max_length=500)
    code: str | None = Field(default=None, max_length=160)
    refs: list[str] = Field(default_factory=list, max_length=5)


class SingleTurnAdvisorWire(BaseModel):
    routing: IntentRoutingWire
    rules: RulesAdviceWire
    plan: TurnPlanWire
    scenario: ScenarioDirectorWire
    summary: str = Field(default="", max_length=240)


class NarrationWire(BaseModel):
    text: str = Field(max_length=1800)


class CriticFindingWire(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dimension: Literal[
        "hidden_leak",
        "unsupported_fact",
        "resolver_bypass",
        "canon_contradiction",
        "player_agency",
        "pacing",
        "clarification",
        "narration_quality",
    ] = Field(alias="dim")
    severity: Literal["low", "medium", "high", "critical"] = Field(alias="sev")
    message: str = Field(alias="msg", max_length=220)
    evidence: str | None = Field(default=None, alias="ev", max_length=220)


class CriticWire(BaseModel):
    ok: bool
    block: bool = False
    findings: list[CriticFindingWire] = Field(default_factory=list, max_length=4)
    revision: str | None = Field(default=None, max_length=1800)
    summary: str = Field(default="", max_length=220)


class MemoryCandidateWire(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: str = Field(max_length=80)
    text: str = Field(max_length=500)
    scope: Literal["session", "player", "campaign", "system"] = "session"
    confidence: float = Field(default=0.7, ge=0, le=1, alias="conf")
    metadata: dict[str, Any] = Field(default_factory=dict, alias="meta")


class MemoryCurationWire(BaseModel):
    write: bool = True
    canon: dict[str, Any] | None = None
    mem: list[MemoryCandidateWire] = Field(default_factory=list, max_length=4)
    contradictions: list[str] = Field(default_factory=list, max_length=4)


class AuthorityMicroGateWire(BaseModel):
    ok: bool
    boundary: bool = False
    clarify: bool = False
    msg: str = Field(default="", max_length=180)
    text: str | None = Field(default=None, max_length=260)


class IntentMicroGateWire(BaseModel):
    route: Literal[
        "answer",
        "clarify",
        "free_action",
        "rules_query",
        "memory_recall",
        "boundary",
        "gm_move",
    ]
    intent: str | None = Field(default=None, max_length=80)
    confidence: float = Field(default=0.75, ge=0, le=1)
    direct: bool = False
    scenario: bool = True
    msg: str = Field(default="", max_length=180)


class RiskMicroGateWire(BaseModel):
    risky: bool
    risk: Literal["none", "low", "risky_uncertain", "high"] = "none"
    resolve: bool = False
    msg: str = Field(default="", max_length=180)


class TargetMicroGateWire(BaseModel):
    ambiguous: bool
    clarify: bool = False
    question: str | None = Field(default=None, max_length=240)
    msg: str = Field(default="", max_length=180)


class MemoryRecallMicroGateWire(BaseModel):
    recall: bool
    msg: str = Field(default="", max_length=180)


COMPACT_SCHEMA_BY_INTERNAL_SCHEMA: dict[str, type[BaseModel]] = {
    "IntentRoutingDecision": IntentRoutingWire,
    "RulesAdjudicationAdvice": RulesAdviceWire,
    "TurnPlan": TurnPlanWire,
    "ScenarioDirectorDecision": ScenarioDirectorWire,
    "SingleTurnAdvisorDecision": SingleTurnAdvisorWire,
    "NarrationPlan": NarrationWire,
    "CriticReport": CriticWire,
    "MemoryCurationDecision": MemoryCurationWire,
    "AuthorityMicroGateDecision": AuthorityMicroGateWire,
    "IntentMicroGateDecision": IntentMicroGateWire,
    "RiskMicroGateDecision": RiskMicroGateWire,
    "TargetMicroGateDecision": TargetMicroGateWire,
    "MemoryRecallMicroGateDecision": MemoryRecallMicroGateWire,
}


COMPACT_RESPONSE_CONTRACTS: dict[str, str] = {
    "IntentRoutingWire": (
        '{"route": answer|clarify|free_action|risky_action|rules_query|memory_recall|'
        'boundary|gm_move, "flags": [rules|scenario|memory|direct], '
        '"confidence": 0..1, "message": "<=220 chars", "code": null|string, '
        '"refs": []}'
    ),
    "RulesAdviceWire": (
        '{"resolve": bool, "proc": null|string, "approach": null|string, '
        '"check": null|string, "diff": null|string, "mod": null|string, "pushed": bool, '
        '"risk": none|low|risky_uncertain|high, "stakes": "<=220 chars", '
        '"clarify": null|string, "refs": []}'
    ),
    "TurnPlanWire": (
        '{"decision": answer|free_action|risky_action|gm_move|boundary|clarify, '
        '"brief": "<=300 chars", "tools": [{"tool": string, "args": {}, "why": string}], '
        '"message": "<=220 chars", "code": null|string, "refs": []}'
    ),
    "ScenarioDirectorWire": (
        '{"decision": no_change|reveal|transition|advance_pressure|consequence|clarify|ending, '
        '"patches": [object], "trans": null|string, "evidence": [string], '
        '"visible": "<=500 chars", "code": null|string, "refs": []}'
    ),
    "SingleTurnAdvisorWire": (
        '{"routing": IntentRoutingWire, "rules": RulesAdviceWire, "plan": TurnPlanWire, '
        '"scenario": ScenarioDirectorWire, "summary": "<=240 chars"}'
    ),
    "NarrationWire": '{"text": "player-facing GM reply, <=1800 chars"}',
    "CriticWire": (
        '{"ok": bool, "block": bool, '
        '"findings": [{"dim": dimension, "sev": severity, "msg": "<=220 chars", "ev": null}], '
        '"revision": null|string, "summary": "<=220 chars"}'
    ),
    "MemoryCurationWire": (
        '{"write": bool, "canon": null|object, '
        '"mem": [{"kind": kind, "text": "<=500 chars", "scope": session|player|campaign|system, '
        '"conf": 0..1, "meta": {}}], "contradictions": []}'
    ),
    "AuthorityMicroGateWire": (
        '{"ok": bool, "boundary": bool, "clarify": bool, "msg": "<=180 chars", '
        '"text": null|string}'
    ),
    "IntentMicroGateWire": (
        '{"route": answer|clarify|free_action|rules_query|memory_recall|boundary|gm_move, '
        '"intent": null|string, "confidence": 0..1, "direct": bool, '
        '"scenario": bool, "msg": "<=180 chars"}'
    ),
    "RiskMicroGateWire": (
        '{"risky": bool, "risk": none|low|risky_uncertain|high, '
        '"resolve": bool, "msg": "<=180 chars"}'
    ),
    "TargetMicroGateWire": (
        '{"ambiguous": bool, "clarify": bool only when ambiguity blocks safe advancement, '
        '"question": null|string, "msg": "<=180 chars"}'
    ),
    "MemoryRecallMicroGateWire": '{"recall": bool, "msg": "<=180 chars"}',
}


def compact_schema_for(schema: type[BaseModel]) -> type[BaseModel]:
    return COMPACT_SCHEMA_BY_INTERNAL_SCHEMA.get(schema.__name__, schema)


def compact_response_contract(schema: type[BaseModel]) -> str:
    return COMPACT_RESPONSE_CONTRACTS.get(schema.__name__, schema.__name__)


def adapt_compact_output(
    *,
    role: str,
    output: BaseModel,
    player_input: str = "",
    context: Mapping[str, object] | None = None,
) -> BaseModel:
    context = context or {}
    if isinstance(output, IntentRoutingWire):
        return _adapt_intent_routing(output)
    if isinstance(output, RulesAdviceWire):
        return _adapt_rules_advice(output)
    if isinstance(output, TurnPlanWire):
        return _adapt_turn_plan(output, context=context)
    if isinstance(output, ScenarioDirectorWire):
        return _adapt_scenario_director(output)
    if isinstance(output, SingleTurnAdvisorWire):
        routing = _adapt_intent_routing(output.routing)
        rules = _adapt_rules_advice(output.rules)
        turn_context = {**context, "routing_decision": routing.model_dump()}
        return SingleTurnAdvisorDecision(
            routing_decision=routing,
            rules_advice=rules,
            turn_plan=_adapt_turn_plan(output.plan, context=turn_context),
            scenario_advice=_adapt_scenario_director(output.scenario),
            reasoning_summary=output.summary or _reason("compact single-turn advisor", role),
        )
    if isinstance(output, NarrationWire):
        return NarrationPlan(final_text=output.text, canon_event_draft=None, memory_candidates=[])
    if isinstance(output, CriticWire):
        return _adapt_critic(output)
    if isinstance(output, MemoryCurationWire):
        return _adapt_memory_curation(output)
    if isinstance(output, AuthorityMicroGateWire):
        return AuthorityMicroGateDecision(
            allowed=output.ok,
            boundary=output.boundary,
            needs_clarification=output.clarify,
            reason=output.msg or _reason("compact authority micro-gate", role),
            player_facing_boundary=output.text,
        )
    if isinstance(output, IntentMicroGateWire):
        return IntentMicroGateDecision(
            intent=IntentClassification(
                kind=_normalize_intent_kind(output.intent, output.route),
                confidence=output.confidence,
                reason=output.msg or _reason("compact intent micro-gate", role),
            ),
            route=output.route,
            allow_direct_answer=output.direct,
            needs_scenario_director=output.scenario,
            reason=output.msg or _reason("compact intent micro-gate", role),
        )
    if isinstance(output, RiskMicroGateWire):
        return RiskMicroGateDecision(
            risky=output.risky,
            risk=output.risk,
            needs_rules_resolution=output.resolve,
            reason=output.msg or _reason("compact risk micro-gate", role),
        )
    if isinstance(output, TargetMicroGateWire):
        return TargetMicroGateDecision(
            ambiguous=output.ambiguous,
            needs_clarification=output.clarify,
            clarification_question=output.question,
            reason=output.msg or _reason("compact target micro-gate", role),
        )
    if isinstance(output, MemoryRecallMicroGateWire):
        return MemoryRecallMicroGateDecision(
            needs_memory_recall=output.recall,
            reason=output.msg or _reason("compact memory recall micro-gate", role),
        )
    return output


def _adapt_intent_routing(wire: IntentRoutingWire) -> IntentRoutingDecision:
    flags = set(wire.flags)
    route = wire.route
    return IntentRoutingDecision(
        intent=IntentClassification(
            kind=_intent_kind_for_route(route),
            confidence=wire.confidence,
            reason=wire.message or _reason("compact routing", route),
        ),
        route=route,
        needs_rules_resolution=("rules" in flags or route == "risky_action"),
        needs_scenario_director=(
            "scenario" in flags or route in {"free_action", "risky_action", "gm_move"}
        ),
        needs_memory_recall=("memory" in flags or route == "memory_recall"),
        allow_direct_answer=(
            "direct" in flags or route in {"answer", "rules_query", "memory_recall"}
        ),
        reasoning_summary=wire.message or _reason("compact routing", route),
        uncertainty=wire.code,
        citations=list(wire.refs),
    )


def _adapt_rules_advice(wire: RulesAdviceWire) -> RulesAdjudicationAdvice:
    return RulesAdjudicationAdvice(
        requires_resolution=wire.resolve,
        procedure_id=wire.proc,
        approach_id=wire.approach,
        check_id=wire.check,
        difficulty=wire.diff,
        modifier=wire.mod,
        pushed=wire.pushed,
        risk=wire.risk if wire.resolve else wire.risk,
        stakes=wire.stakes or ("Resolution needed." if wire.resolve else "No risky uncertainty."),
        clarification_question=wire.clarify,
        citations=list(wire.refs),
    )


def _adapt_turn_plan(
    wire: TurnPlanWire,
    *,
    context: Mapping[str, object],
) -> TurnPlan:
    routing = context.get("routing_decision")
    intent = _intent_from_context(routing, wire)
    return TurnPlan(
        intent=intent,
        authority=AuthorityResult(
            ok=wire.decision != "boundary",
            reason=wire.message or _reason("compact turn plan", wire.decision),
            unsupported_claim=wire.code if wire.decision == "boundary" else None,
            grounded_alternatives=[],
        ),
        decision=wire.decision,
        tool_requests=[
            ToolRequest(
                tool_name=request.tool_name,
                arguments=request.arguments,
                reason=request.reason or _reason("compact tool request", request.tool_name),
            )
            for request in wire.tools
        ],
        narration_brief=wire.brief or wire.message or _reason("compact turn plan", wire.decision),
        citations=list(wire.refs),
    )


def _adapt_scenario_director(wire: ScenarioDirectorWire) -> ScenarioDirectorDecision:
    return ScenarioDirectorDecision(
        decision=wire.decision,
        proposed_patches=list(wire.patches),
        transition_id=wire.trans,
        trigger_evidence=list(wire.evidence),
        player_visible_context=wire.visible,
        gm_only_reason=wire.code or _reason("compact scenario director", wire.decision),
        citations=list(wire.refs),
    )


def _adapt_critic(wire: CriticWire) -> CriticReport:
    return CriticReport(
        ok=wire.ok,
        blocks_output=wire.block,
        findings=[
            CriticFinding(
                dimension=finding.dimension,
                severity=finding.severity,
                message=finding.message,
                evidence=finding.evidence,
            )
            for finding in wire.findings
        ],
        revised_final_text=wire.revision,
        reasoning_summary=wire.summary or _reason("compact critic", "ok" if wire.ok else "issue"),
    )


def _adapt_memory_curation(wire: MemoryCurationWire) -> MemoryCurationDecision:
    return MemoryCurationDecision(
        canon_event_draft=wire.canon,
        memory_candidates=[
            MemoryCandidate(
                kind=_normalize_memory_kind(candidate.kind),
                text=candidate.text,
                scope=candidate.scope,
                confidence=candidate.confidence,
                metadata=candidate.metadata,
            )
            for candidate in wire.mem
        ],
        contradictions=list(wire.contradictions),
        should_write=wire.write,
    )


def _intent_from_context(routing: object, wire: TurnPlanWire) -> IntentClassification:
    if isinstance(routing, dict) and isinstance(routing.get("intent"), dict):
        try:
            return IntentClassification.model_validate(routing["intent"])
        except Exception:
            pass
    return IntentClassification(
        kind=_intent_kind_for_decision(wire.decision),
        confidence=0.7,
        reason=wire.message or _reason("compact turn plan", wire.decision),
    )


def _normalize_intent_kind(
    value: str | None,
    route: str,
) -> Literal[
    "action",
    "info_query",
    "rules_query",
    "memory_recall",
    "character_setup",
    "passive",
    "boundary_claim",
    "clarify_needed",
]:
    allowed = {
        "action",
        "info_query",
        "rules_query",
        "memory_recall",
        "character_setup",
        "passive",
        "boundary_claim",
        "clarify_needed",
    }
    normalized = (value or "").strip().lower()
    if normalized in allowed:
        return normalized  # type: ignore[return-value]
    return _intent_kind_for_route(route)


def _normalize_memory_kind(
    value: str,
) -> Literal[
    "canon",
    "character_state",
    "player_preference",
    "unresolved_thread",
    "episodic_summary",
    "procedural_note",
]:
    allowed = {
        "canon",
        "character_state",
        "player_preference",
        "unresolved_thread",
        "episodic_summary",
        "procedural_note",
    }
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized  # type: ignore[return-value]
    if "preference" in normalized:
        return "player_preference"
    if "character" in normalized or "npc" in normalized:
        return "character_state"
    if "thread" in normalized or "hook" in normalized or "opportunity" in normalized:
        return "unresolved_thread"
    if "summary" in normalized:
        return "episodic_summary"
    if "procedure" in normalized or "note" in normalized:
        return "procedural_note"
    return "canon"


def _intent_kind_for_route(route: str) -> Literal[
    "action",
    "info_query",
    "rules_query",
    "memory_recall",
    "character_setup",
    "passive",
    "boundary_claim",
    "clarify_needed",
]:
    if route == "rules_query":
        return "rules_query"
    if route == "memory_recall":
        return "memory_recall"
    if route == "answer":
        return "info_query"
    if route == "boundary":
        return "boundary_claim"
    if route == "clarify":
        return "clarify_needed"
    return "action"


def _intent_kind_for_decision(decision: str) -> Literal[
    "action",
    "info_query",
    "rules_query",
    "memory_recall",
    "character_setup",
    "passive",
    "boundary_claim",
    "clarify_needed",
]:
    if decision == "answer":
        return "info_query"
    if decision == "boundary":
        return "boundary_claim"
    if decision == "clarify":
        return "clarify_needed"
    return "action"


def _reason(prefix: str, value: str) -> str:
    return f"{prefix}: {value}"


class EvalScorecard(BaseModel):
    rules_correctness: int = Field(ge=1, le=5)
    fictional_authority: int = Field(ge=1, le=5)
    continuity: int = Field(ge=1, le=5)
    player_agency: int = Field(ge=1, le=5)
    pacing: int = Field(ge=1, le=5)
    progressive_disclosure: int = Field(ge=1, le=5)
    memory_behavior: int = Field(ge=1, le=5)
    narration_quality: int = Field(ge=1, le=5)
    trace_explainability: int = Field(default=5, ge=1, le=5)
    generic_architecture_compliance: int = Field(default=5, ge=1, le=5)


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def response_text(response: object) -> str:
    if isinstance(response, AIMessage):
        return str(response.content)
    return str(response)


def _render_prompt_payload(payload: Mapping[str, object]) -> dict[str, object]:
    rendered: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, dict | list):
            rendered[key] = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        else:
            rendered[key] = value
    return rendered


def parse_structured_model[StructuredModel: BaseModel](
    schema: type[StructuredModel],
    text: str,
) -> StructuredModel:
    return schema.model_validate(parse_json_object(text))


def invoke_structured_with_repair[StructuredModel: BaseModel](
    *,
    model: BaseChatModel,
    prompt: ChatPromptTemplate,
    payload: Mapping[str, object],
    schema: type[StructuredModel],
    model_kwargs: Mapping[str, object] | None = None,
) -> tuple[StructuredModel, list[dict[str, str]]]:
    """Invoke a prompt and repair malformed JSON once with the same model."""

    runnable_model = model.bind(**dict(model_kwargs or {})) if model_kwargs else model
    chain = prompt | runnable_model
    response = chain.invoke(_render_prompt_payload(payload))
    raw = response_text(response)
    attempts = [{"phase": "initial", "raw_output": raw}]
    try:
        return parse_structured_model(schema, raw), attempts
    except Exception as first_error:
        attempts[0]["error"] = str(first_error)

    repair_chain = JSON_REPAIR_PROMPT | runnable_model
    repair_response = repair_chain.invoke(
        {
            "schema": json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2),
            "raw_output": raw,
            "error": attempts[0]["error"],
        }
    )
    repaired = response_text(repair_response)
    attempts.append({"phase": "repair", "raw_output": repaired})
    try:
        return parse_structured_model(schema, repaired), attempts
    except Exception as second_error:
        attempts[1]["error"] = str(second_error)
        raise ValueError(
            f"Model output did not match {schema.__name__} after repair: {second_error}"
        ) from second_error

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from trpg_agent.langchain.prompts import JSON_REPAIR_PROMPT


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
    requested_roll: str | None = None
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
    player_visible_context: str
    gm_only_reason: str
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

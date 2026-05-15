from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from trpg_agent.langchain.prompts import (
    AUTHORITY_GATE_PROMPT,
    AUTHORITY_GATE_PROMPT_VERSION,
    AUTHORITY_MICRO_GATE_PROMPT,
    AUTHORITY_MICRO_GATE_PROMPT_VERSION,
    CRITIC_GUARDRAIL_PROMPT,
    CRITIC_GUARDRAIL_PROMPT_VERSION,
    INTENT_ARBITER_PROMPT,
    INTENT_ARBITER_PROMPT_VERSION,
    INTENT_MICRO_GATE_PROMPT,
    INTENT_MICRO_GATE_PROMPT_VERSION,
    MEMORY_CURATOR_PROMPT,
    MEMORY_CURATOR_PROMPT_VERSION,
    MEMORY_RECALL_MICRO_GATE_PROMPT,
    MEMORY_RECALL_MICRO_GATE_PROMPT_VERSION,
    RISK_MICRO_GATE_PROMPT,
    RISK_MICRO_GATE_PROMPT_VERSION,
    RULES_ADJUDICATOR_PROMPT,
    RULES_ADJUDICATOR_PROMPT_VERSION,
    SCENARIO_DIRECTOR_PROMPT,
    SCENARIO_DIRECTOR_PROMPT_VERSION,
    SCENARIO_SURFACE_SELECTOR_PROMPT,
    SCENARIO_SURFACE_SELECTOR_PROMPT_VERSION,
    SINGLE_TURN_ADVISOR_PROMPT,
    SINGLE_TURN_ADVISOR_PROMPT_VERSION,
    TARGET_MICRO_GATE_PROMPT,
    TARGET_MICRO_GATE_PROMPT_VERSION,
)
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
    ScenarioSurfaceSelectorDecision,
    SingleTurnAdvisorDecision,
    TargetMicroGateDecision,
    adapt_compact_output,
    compact_response_contract,
    compact_schema_for,
    invoke_structured_with_repair,
)
from trpg_agent.memory.store import SqliteStore
from trpg_agent.security.redaction import redact_secrets

AdvisorRole = Literal[
    "intent_arbiter",
    "authority_gate",
    "authority_micro_gate",
    "intent_micro_gate",
    "risk_micro_gate",
    "target_micro_gate",
    "memory_recall_micro_gate",
    "rules_adjudicator",
    "scenario_director",
    "scenario_surface_selector",
    "single_turn_advisor",
    "memory_curator",
    "critic_guardrail",
]
AdvisorContractMode = Literal["legacy", "compact"]


@dataclass(frozen=True)
class AdvisorSpec:
    role: AdvisorRole
    prompt: ChatPromptTemplate
    schema: type[BaseModel]
    prompt_version: str
    max_tokens: int | None = None
    compact_max_tokens: int | None = None


@dataclass(frozen=True)
class AdvisorResult:
    role: AdvisorRole
    prompt_version: str
    run_id: str
    output: BaseModel
    attempts: list[dict[str, str]]
    contract_mode: AdvisorContractMode = "legacy"
    cached: bool = False
    metrics: dict[str, str] = field(default_factory=dict)

    @property
    def trace_metadata(self) -> dict[str, str]:
        return {
            "advisor_role": self.role,
            "prompt_version": self.prompt_version,
            "advisor_run_id": self.run_id,
            "schema": self.output.__class__.__name__,
            "contract_mode": self.contract_mode,
            "cached": str(self.cached).lower(),
        } | self.metrics


ADVISOR_SPECS: dict[AdvisorRole, AdvisorSpec] = {
    "intent_arbiter": AdvisorSpec(
        role="intent_arbiter",
        prompt=INTENT_ARBITER_PROMPT,
        schema=IntentRoutingDecision,
        prompt_version=INTENT_ARBITER_PROMPT_VERSION,
        compact_max_tokens=500,
    ),
    "authority_gate": AdvisorSpec(
        role="authority_gate",
        prompt=AUTHORITY_GATE_PROMPT,
        schema=AuthorityGateResult,
        prompt_version=AUTHORITY_GATE_PROMPT_VERSION,
    ),
    "authority_micro_gate": AdvisorSpec(
        role="authority_micro_gate",
        prompt=AUTHORITY_MICRO_GATE_PROMPT,
        schema=AuthorityMicroGateDecision,
        prompt_version=AUTHORITY_MICRO_GATE_PROMPT_VERSION,
        max_tokens=350,
        compact_max_tokens=160,
    ),
    "intent_micro_gate": AdvisorSpec(
        role="intent_micro_gate",
        prompt=INTENT_MICRO_GATE_PROMPT,
        schema=IntentMicroGateDecision,
        prompt_version=INTENT_MICRO_GATE_PROMPT_VERSION,
        max_tokens=300,
        compact_max_tokens=180,
    ),
    "risk_micro_gate": AdvisorSpec(
        role="risk_micro_gate",
        prompt=RISK_MICRO_GATE_PROMPT,
        schema=RiskMicroGateDecision,
        prompt_version=RISK_MICRO_GATE_PROMPT_VERSION,
        max_tokens=350,
        compact_max_tokens=160,
    ),
    "target_micro_gate": AdvisorSpec(
        role="target_micro_gate",
        prompt=TARGET_MICRO_GATE_PROMPT,
        schema=TargetMicroGateDecision,
        prompt_version=TARGET_MICRO_GATE_PROMPT_VERSION,
        max_tokens=350,
        compact_max_tokens=180,
    ),
    "memory_recall_micro_gate": AdvisorSpec(
        role="memory_recall_micro_gate",
        prompt=MEMORY_RECALL_MICRO_GATE_PROMPT,
        schema=MemoryRecallMicroGateDecision,
        prompt_version=MEMORY_RECALL_MICRO_GATE_PROMPT_VERSION,
        max_tokens=250,
        compact_max_tokens=120,
    ),
    "rules_adjudicator": AdvisorSpec(
        role="rules_adjudicator",
        prompt=RULES_ADJUDICATOR_PROMPT,
        schema=RulesAdjudicationAdvice,
        prompt_version=RULES_ADJUDICATOR_PROMPT_VERSION,
        compact_max_tokens=450,
    ),
    "scenario_director": AdvisorSpec(
        role="scenario_director",
        prompt=SCENARIO_DIRECTOR_PROMPT,
        schema=ScenarioDirectorDecision,
        prompt_version=SCENARIO_DIRECTOR_PROMPT_VERSION,
        compact_max_tokens=650,
    ),
    "scenario_surface_selector": AdvisorSpec(
        role="scenario_surface_selector",
        prompt=SCENARIO_SURFACE_SELECTOR_PROMPT,
        schema=ScenarioSurfaceSelectorDecision,
        prompt_version=SCENARIO_SURFACE_SELECTOR_PROMPT_VERSION,
        max_tokens=260,
        compact_max_tokens=180,
    ),
    "single_turn_advisor": AdvisorSpec(
        role="single_turn_advisor",
        prompt=SINGLE_TURN_ADVISOR_PROMPT,
        schema=SingleTurnAdvisorDecision,
        prompt_version=SINGLE_TURN_ADVISOR_PROMPT_VERSION,
        compact_max_tokens=900,
    ),
    "memory_curator": AdvisorSpec(
        role="memory_curator",
        prompt=MEMORY_CURATOR_PROMPT,
        schema=MemoryCurationDecision,
        prompt_version=MEMORY_CURATOR_PROMPT_VERSION,
        compact_max_tokens=550,
    ),
    "critic_guardrail": AdvisorSpec(
        role="critic_guardrail",
        prompt=CRITIC_GUARDRAIL_PROMPT,
        schema=CriticReport,
        prompt_version=CRITIC_GUARDRAIL_PROMPT_VERSION,
        compact_max_tokens=650,
    ),
}


def advisor_spec(role: AdvisorRole) -> AdvisorSpec:
    return ADVISOR_SPECS[role]


def advisor_input_hash(
    *,
    role: AdvisorRole,
    prompt_version: str,
    contract_mode: AdvisorContractMode = "legacy",
    player_input: str,
    context: Mapping[str, object],
) -> str:
    payload = {
        "role": role,
        "prompt_version": prompt_version,
        "contract_mode": contract_mode,
        "player_input": player_input,
        "context": context,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def advisor_run_id(
    *,
    turn_id: str,
    role: AdvisorRole,
    prompt_version: str,
    input_hash: str,
) -> str:
    return f"{turn_id}:advisor:{role}:{prompt_version}:{input_hash[:16]}"


def invoke_advisor(
    *,
    model: BaseChatModel,
    role: AdvisorRole,
    player_input: str = "",
    context: Mapping[str, object],
    sqlite_path: str | None = None,
    turn_id: str | None = None,
    contract_mode: AdvisorContractMode = "legacy",
) -> AdvisorResult:
    spec = advisor_spec(role)
    parse_schema = spec.schema
    schema_prompt_payload: object = spec.schema.model_json_schema()
    model_kwargs = {"max_tokens": spec.max_tokens} if spec.max_tokens else None
    if contract_mode == "compact":
        parse_schema = compact_schema_for(spec.schema)
        schema_prompt_payload = compact_response_contract(parse_schema)
        if spec.compact_max_tokens:
            model_kwargs = {"max_tokens": spec.compact_max_tokens}
    input_hash = advisor_input_hash(
        role=role,
        prompt_version=spec.prompt_version,
        contract_mode=contract_mode,
        player_input=player_input,
        context=context,
    )
    run_id = advisor_run_id(
        turn_id=turn_id or "unpersisted",
        role=role,
        prompt_version=spec.prompt_version,
        input_hash=input_hash,
    )
    if sqlite_path and turn_id:
        store = SqliteStore(Path(sqlite_path))
        store.migrate()
        cached = store.get_advisor_run(run_id)
        if cached:
            cached_metrics = _metrics_from_attempts(cached.get("attempts", []))
            return AdvisorResult(
                role=role,
                prompt_version=spec.prompt_version,
                run_id=run_id,
                output=spec.schema.model_validate(cached["output"]),
                attempts=[{"phase": "cache"}],
                contract_mode=contract_mode,
                cached=True,
                metrics=cached_metrics,
            )

    started = time.perf_counter()
    output, attempts = invoke_structured_with_repair(
        model=model,
        prompt=spec.prompt,
        schema=parse_schema,
        payload={
            "player_input": player_input,
            "context": dict(context),
            "schema": schema_prompt_payload,
        },
        model_kwargs=model_kwargs,
    )
    if contract_mode == "compact":
        output = adapt_compact_output(
            role=role,
            output=output,
            player_input=player_input,
            context=context,
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    metrics = _advisor_metrics(
        elapsed_ms=elapsed_ms,
        player_input=player_input,
        context=context,
        attempts=attempts,
        schema_prompt=schema_prompt_payload,
    )
    attempts_for_store = [
        *attempts,
        {"phase": "metrics", **metrics},
    ]
    if sqlite_path and turn_id:
        store = SqliteStore(Path(sqlite_path))
        store.migrate()
        store.insert_advisor_run_once(
            run_id=run_id,
            turn_id=turn_id,
            role=role,
            prompt_version=spec.prompt_version,
            input_hash=input_hash,
            output=redact_secrets(output.model_dump()),
            attempts=redact_secrets(attempts_for_store),
        )
    return AdvisorResult(
        role=role,
        prompt_version=spec.prompt_version,
        run_id=run_id,
        output=output,
        attempts=attempts_for_store,
        contract_mode=contract_mode,
        metrics=metrics,
    )


def _advisor_metrics(
    *,
    elapsed_ms: int,
    player_input: str,
    context: Mapping[str, object],
    attempts: list[dict[str, str]],
    schema_prompt: object,
) -> dict[str, str]:
    context_key_chars = {
        str(key): len(json.dumps(value, ensure_ascii=False, default=str))
        for key, value in sorted(context.items(), key=lambda item: str(item[0]))
    }
    player_input_chars = len(player_input)
    context_chars = len(json.dumps(context, ensure_ascii=False, default=str))
    schema_chars = len(json.dumps(schema_prompt, ensure_ascii=False, default=str))
    prompt_chars = player_input_chars + context_chars + schema_chars
    response_chars = sum(len(str(attempt.get("raw_output", ""))) for attempt in attempts)
    return {
        "elapsed_ms": str(elapsed_ms),
        "estimated_prompt_chars": str(prompt_chars),
        "player_input_chars": str(player_input_chars),
        "context_chars": str(context_chars),
        "schema_chars": str(schema_chars),
        "context_key_chars_json": json.dumps(context_key_chars, ensure_ascii=False),
        "estimated_response_chars": str(response_chars),
        "attempt_count": str(len(attempts)),
    }


def _metrics_from_attempts(attempts: list[dict[str, str]]) -> dict[str, str]:
    for attempt in attempts:
        if attempt.get("phase") == "metrics":
            return {
                key: str(value)
                for key, value in attempt.items()
                if key != "phase"
            }
    return {}

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from trpg_agent.content.compiled import CompiledRuleset, load_compiled_ruleset
from trpg_agent.content.packages import PackageKind
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.content.retrieval import search_registry_text_indexed
from trpg_agent.content.visibility import AccessMode
from trpg_agent.context_budget import (
    build_advisor_context,
    build_context_budget_snapshot,
    compact_context_budget_trace,
)
from trpg_agent.graph.state import GraphState
from trpg_agent.langchain.advisors import AdvisorContractMode, AdvisorRole, invoke_advisor
from trpg_agent.langchain.prompts import (
    AUTHORITY_MICRO_GATE_PROMPT_VERSION,
    CORE_GM_PROMPT,
    CORE_GM_PROMPT_VERSION,
    CRITIC_GUARDRAIL_PROMPT_VERSION,
    INTENT_ARBITER_PROMPT_VERSION,
    INTENT_MICRO_GATE_PROMPT_VERSION,
    MEMORY_CURATOR_PROMPT_VERSION,
    MEMORY_RECALL_MICRO_GATE_PROMPT_VERSION,
    NARRATION_PROMPT,
    NARRATION_PROMPT_VERSION,
    RISK_MICRO_GATE_PROMPT_VERSION,
    RULES_ADJUDICATOR_PROMPT_VERSION,
    SCENARIO_DIRECTOR_PROMPT_VERSION,
    SCENARIO_SURFACE_SELECTOR_PROMPT_VERSION,
    SINGLE_TURN_ADVISOR_PROMPT_VERSION,
    TARGET_MICRO_GATE_PROMPT_VERSION,
)
from trpg_agent.langchain.structured import (
    AuthorityMicroGateDecision,
    AuthorityResult,
    CriticFinding,
    CriticReport,
    IntentClassification,
    IntentMicroGateDecision,
    IntentRoutingDecision,
    MemoryCandidate,
    MemoryCurationDecision,
    MemoryRecallMicroGateDecision,
    NarrationPlan,
    RiskMicroGateDecision,
    RulesAdjudicationAdvice,
    ScenarioDirectorDecision,
    ScenarioSurfaceSelectorDecision,
    SingleTurnAdvisorDecision,
    TargetMicroGateDecision,
    ToolRequest,
    ToolResult,
    TurnPlan,
    adapt_compact_output,
    compact_response_contract,
    compact_schema_for,
    invoke_structured_with_repair,
)
from trpg_agent.langchain.tools import build_langchain_tools
from trpg_agent.memory.projection import project_recent_summary
from trpg_agent.memory.store import SqliteStore
from trpg_agent.rules.compiled_resolver import RULES_PLUGIN_RUNTIME_VERSION, run_ruleset_resolver
from trpg_agent.rules.plugin_runtime import RulesDslPlugin, load_rules_plugin
from trpg_agent.scenario.director import (
    ScenarioPatchValidation,
    validate_scenario_director_decision,
)
from trpg_agent.scenario.runtime import load_or_initialize_world_state, sync_scene_details
from trpg_agent.security.redaction import redact_secrets
from trpg_agent.tools.content import load_content_span, search_content
from trpg_agent.tools.patches import WorldPatch, apply_world_patches

TURN_GRAPH_VERSION = "turn-graph-v2"
LOCAL_ADJUDICATION_VERSION = "local-adjudication-v1"
DETERMINISTIC_TOOL_VERSION = "deterministic-tools-v1"
AdvisorModelMap = Mapping[AdvisorRole, BaseChatModel]


def receive_input(state: GraphState) -> GraphState:
    next_state = {
        **state,
        "trace_events": [
            *state.get("trace_events", []),
            {"node": "receive_input", "player_input": state.get("player_input", "")},
        ],
    }
    return next_state


def _append_trace(state: GraphState, node: str, payload: dict[str, Any]) -> GraphState:
    return {
        **state,
        "trace_events": [
            *state.get("trace_events", []),
            {"node": node, **payload},
        ],
    }


def _registry_from_state(state: GraphState) -> ContentRegistry:
    content_dir = Path(state.get("content_dir") or Path.cwd() / "content")
    return ContentRegistry.load(content_dir, content_dir.parent)


def _model_for_role(
    role: AdvisorRole,
    *,
    default_model: BaseChatModel,
    advisor_models: AdvisorModelMap | None,
) -> BaseChatModel:
    if advisor_models and role in advisor_models:
        return advisor_models[role]
    return default_model


def _runtime_metadata(state: GraphState) -> dict[str, Any]:
    return {
        "graph_version": TURN_GRAPH_VERSION,
        "prompt_versions": {
            "core_gm": CORE_GM_PROMPT_VERSION,
            "intent_arbiter": INTENT_ARBITER_PROMPT_VERSION,
            "authority_micro_gate": AUTHORITY_MICRO_GATE_PROMPT_VERSION,
            "intent_micro_gate": INTENT_MICRO_GATE_PROMPT_VERSION,
            "risk_micro_gate": RISK_MICRO_GATE_PROMPT_VERSION,
            "target_micro_gate": TARGET_MICRO_GATE_PROMPT_VERSION,
            "memory_recall_micro_gate": MEMORY_RECALL_MICRO_GATE_PROMPT_VERSION,
            "rules_adjudicator": RULES_ADJUDICATOR_PROMPT_VERSION,
            "scenario_director": SCENARIO_DIRECTOR_PROMPT_VERSION,
            "scenario_surface_selector": SCENARIO_SURFACE_SELECTOR_PROMPT_VERSION,
            "single_turn_advisor": SINGLE_TURN_ADVISOR_PROMPT_VERSION,
            "memory_curator": MEMORY_CURATOR_PROMPT_VERSION,
            "narration": NARRATION_PROMPT_VERSION,
            "critic_guardrail": CRITIC_GUARDRAIL_PROMPT_VERSION,
            "local_adjudication": LOCAL_ADJUDICATION_VERSION,
            "deterministic_tools": DETERMINISTIC_TOOL_VERSION,
        },
        "rules_plugin_runtime_version": RULES_PLUGIN_RUNTIME_VERSION,
        "checkpoint_mode": state.get("checkpoint_mode", "none"),
        "advisor_contract_mode": _advisor_contract_mode(state),
        "context_budget_mode": _context_budget_mode(state),
        "conditional_advisors_mode": bool(state.get("conditional_advisors_mode")),
        "model": state.get("model_metadata", {}),
    }


def _advisor_contract_mode(state: Mapping[str, Any]) -> AdvisorContractMode:
    return "compact" if state.get("advisor_contract_mode") == "compact" else "legacy"


def _context_budget_mode(state: Mapping[str, Any]) -> str:
    mode = str(state.get("context_budget_mode") or "").strip().lower()
    if mode in {"enforced", "shadow"}:
        return mode
    if state.get("play_profile") == "fast":
        return "enforced"
    return "shadow"


def _context_packet(state: GraphState, role: str, **extra_context: Any) -> dict[str, Any]:
    return build_advisor_context(
        state,
        role,
        mode=_context_budget_mode(state),
        extra_context=extra_context,
    )


def _conditional_advisors_enabled(state: Mapping[str, Any]) -> bool:
    return bool(state.get("conditional_advisors_mode")) and not bool(
        state.get("single_turn_advisor_mode") or state.get("micro_gates_mode")
    )


def _with_advisor_skip_reason(
    state: GraphState,
    advisor: str,
    reason: str,
) -> GraphState:
    return {
        **state,
        "advisor_skip_reasons": {
            **dict(state.get("advisor_skip_reasons", {})),
            advisor: reason,
        },
    }


def _turn_complexity(state: Mapping[str, Any]) -> dict[str, Any]:
    routing = (
        state.get("routing_decision")
        if isinstance(state.get("routing_decision"), dict)
        else {}
    )
    plan = state.get("turn_plan") if isinstance(state.get("turn_plan"), dict) else {}
    return {
        "route": routing.get("route"),
        "decision": plan.get("decision"),
        "needs_rules_resolution": bool(routing.get("needs_rules_resolution")),
        "needs_scenario_director": bool(routing.get("needs_scenario_director")),
        "has_successful_resolver_result": _has_successful_resolver_result(state),  # type: ignore[arg-type]
        "has_validated_scenario_progress": _has_validated_scenario_progress(state),  # type: ignore[arg-type]
        "pending_rule_opportunities": len(_pending_rule_opportunities(state)),  # type: ignore[arg-type]
    }


def _should_direct_plan_from_routing(state: GraphState) -> bool:
    if not _conditional_advisors_enabled(state):
        return False
    if _pending_rule_opportunities(state):
        return False
    routing = state.get("routing_decision") or {}
    route = str(routing.get("route") or "")
    if route not in {"answer", "free_action", "rules_query", "memory_recall"}:
        return False
    if routing.get("needs_rules_resolution") or routing.get("needs_rules_review"):
        return False
    if route in {"answer", "rules_query", "memory_recall"} and not bool(
        routing.get("allow_direct_answer")
    ):
        return False
    return True


def _low_risk_local_review_reason(state: GraphState) -> str | None:
    if not _conditional_advisors_enabled(state):
        return None
    decision = str(state.get("turn_plan", {}).get("decision") or "")
    if decision not in {"answer", "free_action"}:
        return None
    if _has_successful_resolver_result(state) or _has_validated_scenario_progress(state):
        return None
    if state.get("tool_results"):
        return None
    if _pending_rule_opportunities(state):
        return None
    routing = state.get("routing_decision") or {}
    if routing.get("needs_rules_resolution") or routing.get("route") in {
        "risky_action",
        "boundary",
        "clarify",
        "gm_move",
    }:
        return None
    return "low_risk_no_tools_no_validated_patches"


def _structured_call_trace(
    *,
    role: str,
    prompt_version: str,
    schema_name: str,
    elapsed_ms: int,
    player_input: str,
    context: Mapping[str, Any],
    schema_prompt: object,
    attempts: list[dict[str, str]],
) -> dict[str, str]:
    context_key_chars = {
        str(key): len(json.dumps(value, ensure_ascii=False, default=str))
        for key, value in sorted(context.items(), key=lambda item: str(item[0]))
    }
    response_chars = sum(len(str(attempt.get("raw_output", ""))) for attempt in attempts)
    context_chars = len(json.dumps(context, ensure_ascii=False, default=str))
    schema_chars = len(json.dumps(schema_prompt, ensure_ascii=False, default=str))
    player_input_chars = len(player_input)
    return {
        "advisor_role": role,
        "prompt_version": prompt_version,
        "schema": schema_name,
        "contract_mode": _schema_contract_mode(schema_prompt),
        "cached": "false",
        "elapsed_ms": str(elapsed_ms),
        "estimated_prompt_chars": str(player_input_chars + context_chars + schema_chars),
        "player_input_chars": str(player_input_chars),
        "context_chars": str(context_chars),
        "schema_chars": str(schema_chars),
        "context_key_chars_json": json.dumps(context_key_chars, ensure_ascii=False),
        "estimated_response_chars": str(response_chars),
        "attempt_count": str(len(attempts)),
    }


def _schema_contract_mode(schema_prompt: object) -> str:
    return "compact" if isinstance(schema_prompt, str) else "legacy"


def _restore_existing_turn(
    *,
    state: GraphState,
    store: SqliteStore,
    turn_id: str,
) -> GraphState | None:
    existing_turn = store.get_turn(turn_id)
    if not existing_turn:
        return None

    trace = existing_turn.get("trace", {})
    next_state: GraphState = {
        **state,
        "session_id": existing_turn.get("session_id") or state.get("session_id", "default"),
        "replayed_turn": True,
        "final_output": existing_turn.get("output", ""),
        "trace_events": [
            *trace.get("trace_events", []),
            {"node": "replay_persisted_turn", "turn_id": turn_id},
        ],
        "retrieved_spans": trace.get("retrieved_spans", []),
        "context_budget": trace.get("context_budget", {}),
        "tool_results": trace.get("tool_results", []),
        "player_memory_hits": trace.get("player_memory_hits", []),
        "package_profiles": trace.get("package_profiles", state.get("package_profiles", [])),
        "routing_decision": trace.get("routing_decision", {}),
        "rules_advice": trace.get("rules_advice", {}),
        "scenario_director": trace.get("scenario_director", {}),
        "turn_plan": trace.get("turn_plan", {}),
        "narration_plan": trace.get("narration_plan", {}),
        "critic_report": trace.get("critic_report", {}),
        "memory_curation": trace.get("memory_curation", {}),
        "micro_gate_results": trace.get("micro_gate_results", {}),
        "world_projection": trace.get("world_projection", {}),
        "character_context": trace.get("character_context", state.get("character_context", {})),
        "runtime_metadata": trace.get("runtime_metadata", _runtime_metadata(state)),
    }
    return next_state


def load_runtime_context(state: GraphState) -> GraphState:
    content_dir = Path(state.get("content_dir") or Path.cwd() / "content")
    registry = ContentRegistry.load(content_dir, content_dir.parent)
    preload = state.get("runtime_preload") or {}
    next_state: GraphState = {
        **state,
        "session_id": state.get("session_id") or "default",
        "thread_id": state.get("thread_id") or state.get("session_id") or "default",
        "turn_id": state.get("turn_id") or f"turn-{uuid.uuid4().hex[:12]}",
        "content_dir": str(content_dir),
        "runtime_metadata": _runtime_metadata(state),
    }

    if not next_state.get("ruleset_id"):
        rulesets = registry.by_kind(PackageKind.RULESET)
        if rulesets:
            next_state["ruleset_id"] = rulesets[0].id
    if not next_state.get("scenario_id"):
        scenarios = registry.by_kind(PackageKind.SCENARIO)
        if scenarios:
            next_state["scenario_id"] = scenarios[0].id

    active_ids = [
        package_id
        for package_id in [
            next_state.get("ruleset_id"),
            next_state.get("scenario_id"),
            *next_state.get("active_extension_ids", []),
        ]
        if package_id
    ]
    if (
        preload.get("ruleset_id") == next_state.get("ruleset_id")
        and preload.get("scenario_id") == next_state.get("scenario_id")
    ):
        next_state["active_package_ids"] = list(preload.get("active_package_ids") or [])
        next_state["package_profiles"] = list(preload.get("package_profiles") or [])
    else:
        next_state["active_package_ids"] = registry.resolve_active_package_ids(
            list(dict.fromkeys(active_ids))
        )
        next_state["package_profiles"] = registry.package_profiles(next_state["active_package_ids"])
    runtime_metadata = dict(next_state.get("runtime_metadata", {}))
    runtime_metadata["content_packages"] = {
        package_id: registry.by_id[package_id].manifest.version
        for package_id in next_state["active_package_ids"]
        if package_id in registry.by_id
    }
    if next_state.get("ruleset_id"):
        try:
            if preload.get("ruleset_id") == next_state.get("ruleset_id") and isinstance(
                preload.get("compiled_ruleset"),
                dict,
            ):
                ruleset = CompiledRuleset.model_validate(preload["compiled_ruleset"])
            else:
                ruleset = load_compiled_ruleset(registry, next_state["ruleset_id"])
            runtime_metadata["ruleset_resolver_id"] = ruleset.resolver_id
            if preload.get("ruleset_id") == next_state.get("ruleset_id") and isinstance(
                preload.get("rules_plugin"),
                dict,
            ):
                plugin = RulesDslPlugin.model_validate(preload["rules_plugin"])
            else:
                plugin = load_rules_plugin(registry, next_state["ruleset_id"])
            if plugin is not None:
                runtime_metadata["rules_plugin_driver"] = plugin.driver
                runtime_metadata["rules_plugin_id"] = plugin.id
            if not next_state.get("character_context"):
                next_state["character_context"] = dict(ruleset.default_character_context)
        except Exception as error:  # pragma: no cover - content validation should catch this.
            runtime_metadata["ruleset_resolver_error"] = str(error)
    next_state["runtime_metadata"] = runtime_metadata

    sqlite_path = next_state.get("sqlite_path")
    store: SqliteStore | None = None
    if sqlite_path:
        store = SqliteStore(Path(sqlite_path))
        store.migrate()
        existing_turn = _restore_existing_turn(
            state=next_state,
            store=store,
            turn_id=next_state["turn_id"],
        )
        if existing_turn:
            return existing_turn

        store.upsert_session(
            session_id=next_state["session_id"],
            ruleset_id=next_state.get("ruleset_id"),
            scenario_id=next_state.get("scenario_id"),
        )
        canon_events = store.list_canon_events(next_state["session_id"])
        next_state["recent_canon"] = project_recent_summary(canon_events)
        world_state = load_or_initialize_world_state(
            store=store,
            session_id=next_state["session_id"],
            content_dir=content_dir,
            scenario_id=next_state.get("scenario_id"),
        )
        world_state["canon_event_count"] = len(canon_events)
        next_state["world_projection"] = world_state
        world_character_context = world_state.get("character_context")
        if isinstance(world_character_context, dict):
            merged_character_context = dict(next_state.get("character_context") or {})
            merged_character_context.update(world_character_context)
            next_state["character_context"] = merged_character_context
    elif next_state.get("scenario_id"):
        next_state["world_projection"] = load_or_initialize_world_state(
            store=None,
            session_id=next_state["session_id"],
            content_dir=content_dir,
            scenario_id=next_state.get("scenario_id"),
        )

    return _append_trace(
        next_state,
        "load_runtime_context",
        {
            "ruleset_id": next_state.get("ruleset_id"),
            "scenario_id": next_state.get("scenario_id"),
            "active_package_ids": next_state.get("active_package_ids", []),
            "package_profiles": [
                {
                    "id": profile.get("id"),
                    "kind": profile.get("kind"),
                    "references": len(profile.get("references", [])),
                }
                for profile in next_state.get("package_profiles", [])
            ],
            "runtime_metadata": next_state.get("runtime_metadata", {}),
            "character_context_keys": sorted(next_state.get("character_context", {})),
        },
    )


def route_after_runtime_context(state: GraphState) -> str:
    if state.get("replayed_turn"):
        return "replayed"
    return "new_turn"


def retrieve_memory(state: GraphState) -> GraphState:
    sqlite_path = state.get("sqlite_path")
    if not sqlite_path:
        return _append_trace(
            {**state, "memory_hits": [], "player_memory_hits": []},
            "retrieve_memory",
            {"memory_hits": 0, "player_memory_hits": 0},
        )

    store = SqliteStore(Path(sqlite_path))
    store.migrate()
    try:
        hits = store.recall_memories(
            query=state.get("player_input", ""),
            scope=state.get("session_id"),
            limit=5,
        )
        player_hits = store.recall_memories(
            query=state.get("player_input", ""),
            scope=state.get("session_id"),
            limit=5,
            include_gm_only=False,
        )
    except Exception as error:  # pragma: no cover - defensive against SQLite tokenizer quirks.
        return _append_trace(
            {**state, "memory_hits": [], "player_memory_hits": []},
            "retrieve_memory",
            {"memory_hits": 0, "error": str(error)},
        )
    return _append_trace(
        {**state, "memory_hits": hits, "player_memory_hits": player_hits},
        "retrieve_memory",
        {"memory_hits": len(hits), "player_memory_hits": len(player_hits)},
    )


def retrieve_content_spans(state: GraphState) -> GraphState:
    registry = _registry_from_state(state)
    sqlite_path = state.get("sqlite_path")
    if sqlite_path:
        retrieval = search_registry_text_indexed(
            registry,
            state.get("player_input", ""),
            sqlite_path=Path(sqlite_path),
            package_ids=state.get("active_package_ids", []),
            mode=AccessMode.GM,
            limit=6,
        )
    else:
        from trpg_agent.content.retrieval import search_registry_text_with_diagnostics

        retrieval = search_registry_text_with_diagnostics(
            registry,
            state.get("player_input", ""),
            package_ids=state.get("active_package_ids", []),
            mode=AccessMode.GM,
            limit=6,
        )
    spans = [_annotate_retrieved_span(span.to_dict(), state) for span in retrieval.spans]
    diagnostics = retrieval.diagnostics
    return _append_trace(
        {**state, "retrieved_spans": spans},
        "retrieve_content_spans",
        {
            "retrieved": [
                {
                    "package_id": span["package_id"],
                    "reference_id": span["reference_id"],
                    "visibility": span["visibility"],
                    "score": span["score"],
                }
                for span in spans
            ],
            "diagnostics": diagnostics,
        },
    )


def _annotate_retrieved_span(span: dict[str, Any], state: GraphState) -> dict[str, Any]:
    package_id = str(span.get("package_id") or "")
    reference_id = str(span.get("reference_id") or "")
    visibility = str(span.get("visibility") or "public")
    if package_id == state.get("ruleset_id"):
        bucket = "rules"
        purpose = "rules_adjudication"
    elif visibility == "gm_only":
        bucket = "scenario_gm"
        purpose = "gm_scenario_guidance"
    else:
        bucket = "scenario_public"
        purpose = "player_visible_context"
    return {
        **span,
        "citation_id": f"{package_id}:{reference_id}",
        "bucket": bucket,
        "mandatory": bucket == "rules",
        "purpose": purpose,
    }


def retrieve_context_parallel(state: GraphState) -> GraphState:
    """Read memory and content retrieval in parallel.

    Both branches are read-only with respect to durable state and depend only on runtime context.
    The trace keeps the original branch node names so existing eval coverage remains meaningful.
    """

    branch_state: GraphState = {**state, "trace_events": []}
    def timed_branch(fn: Any) -> tuple[GraphState, int]:
        started = time.perf_counter()
        return fn(branch_state), int((time.perf_counter() - started) * 1000)

    branch_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="trpg-context") as executor:
        memory_future = executor.submit(timed_branch, retrieve_memory)
        content_future = executor.submit(timed_branch, retrieve_content_spans)
        memory_state, memory_elapsed_ms = memory_future.result()
        content_state, content_elapsed_ms = content_future.result()
    parallel_elapsed_ms = int((time.perf_counter() - branch_started) * 1000)

    trace_events = [
        *state.get("trace_events", []),
        *memory_state.get("trace_events", []),
        *content_state.get("trace_events", []),
    ]
    next_state = {
        **state,
        "memory_hits": memory_state.get("memory_hits", []),
        "player_memory_hits": memory_state.get("player_memory_hits", []),
        "retrieved_spans": content_state.get("retrieved_spans", []),
        "trace_events": trace_events,
    }
    context_budget = build_context_budget_snapshot(next_state)
    next_state["context_budget"] = context_budget
    return _append_trace(
        next_state,
        "retrieve_context_parallel",
        {
            "branches": ["retrieve_memory", "retrieve_content_spans"],
            "branch_elapsed_ms": {
                "retrieve_memory": memory_elapsed_ms,
                "retrieve_content_spans": content_elapsed_ms,
                "parallel_total": parallel_elapsed_ms,
            },
            "memory_hits": len(next_state.get("memory_hits", [])),
            "player_memory_hits": len(next_state.get("player_memory_hits", [])),
            "retrieved_spans": len(next_state.get("retrieved_spans", [])),
            "context_budget": compact_context_budget_trace(context_budget),
        },
    )


def classify_player_intent(state: GraphState) -> GraphState:
    next_state = {
        **state,
        "intent": {
            "kind": "action",
            "confidence": 0.25,
            "reason": "local structural fallback does not infer natural-language intent",
        },
    }
    return _append_trace(next_state, "classify_player_intent", {"intent": "action"})


def _resolver_tool_request(
    state: GraphState,
    approach: str | None = None,
    procedure_id: str | None = None,
    check_id: str | None = None,
    difficulty: str | None = None,
    modifier: str | None = None,
    pushed: bool = False,
    risk: str = "risky_uncertain",
) -> ToolRequest:
    return ToolRequest(
        tool_name="run_ruleset_resolver",
        arguments={
            "content_dir": state.get("content_dir") or str(Path.cwd() / "content"),
            "ruleset_id": state.get("ruleset_id"),
            "action": state.get("player_input", ""),
            "approach": approach,
            "procedure_id": procedure_id,
            "check_id": check_id,
            "difficulty": difficulty,
            "modifier": modifier,
            "pushed": pushed,
            "risk": risk,
            "character_context": state.get("character_context", {}),
            "scene_context": state.get("world_projection", {}),
            "session_id": state.get("session_id", "default"),
            "turn_id": state.get("turn_id", "turn"),
            "sqlite_path": state.get("sqlite_path"),
        },
        reason="Risky and uncertain action must be resolved by the loaded ruleset resolver.",
    )


def _protected_resolver_arguments(state: GraphState) -> dict[str, Any]:
    return {
        "content_dir": state.get("content_dir") or str(Path.cwd() / "content"),
        "ruleset_id": state.get("ruleset_id"),
        "action": state.get("player_input", ""),
        "character_context": state.get("character_context", {}),
        "scene_context": state.get("world_projection", {}),
        "session_id": state.get("session_id", "default"),
        "turn_id": state.get("turn_id", "turn"),
        "sqlite_path": state.get("sqlite_path"),
    }


def _rules_plugin_arguments(rules_advice: dict[str, Any]) -> dict[str, Any]:
    allowed = {}
    for key in ["procedure_id", "check_id", "difficulty", "modifier", "pushed"]:
        value = rules_advice.get(key)
        if value not in (None, ""):
            allowed[key] = value
    return allowed


def _compiled_ruleset_from_state(state: GraphState):
    ruleset_id = state.get("ruleset_id")
    if not ruleset_id:
        return None
    try:
        return load_compiled_ruleset(_registry_from_state(state), ruleset_id)
    except Exception:
        return None


def _canonical_approach(state: GraphState, value: object) -> str | None:
    if not value:
        return None
    ruleset = _compiled_ruleset_from_state(state)
    if not ruleset:
        return str(value)
    raw = str(value).strip()
    if raw in ruleset.approaches:
        return raw
    lowered = raw.lower()
    for approach_id, spec in ruleset.approaches.items():
        if spec.label.lower() == lowered:
            return approach_id
    return None


def _approach_for_resolver_request(state: GraphState, value: object) -> str | None:
    canonical = _canonical_approach(state, value)
    if canonical:
        return canonical
    ruleset = _compiled_ruleset_from_state(state)
    if not ruleset:
        return None
    lowered_action = state.get("player_input", "").lower()
    for approach_id, spec in ruleset.approaches.items():
        if any(keyword.lower() in lowered_action for keyword in spec.keywords):
            return approach_id
    if ruleset.default_approach and ruleset.default_approach in ruleset.approaches:
        return ruleset.default_approach
    return next(iter(ruleset.approaches), None)


def _clock_tick_request(state: GraphState, reason: str) -> ToolRequest | None:
    clock = state.get("world_projection", {}).get("clock")
    if not isinstance(clock, dict):
        return None
    if int(clock.get("value", 0)) >= int(clock.get("max", 3)):
        return None
    return ToolRequest(
        tool_name="apply_world_patch",
        arguments={
            "patches": [{"op": "increment", "path": ["clock", "value"], "value": 1}],
            "reason": reason,
        },
        reason=reason,
    )


def plan_turn_locally(state: GraphState) -> GraphState:
    intent = IntentClassification.model_validate(state.get("intent", {}))
    authority = AuthorityResult(ok=True, reason="No unsupported authority claim detected.")
    decision = "free_action"
    narration_brief = (
        "Advance the proposed character intent as a free action; show only visible, grounded "
        "results and request a resolver if risk appears."
    )
    tool_requests: list[ToolRequest] = []

    if decision == "gm_move":
        clock_request = _clock_tick_request(
            state,
            "The player waits or delays, so scene pressure advances.",
        )
        if clock_request:
            tool_requests.append(clock_request)

    plan = TurnPlan(
        intent=intent,
        authority=authority,
        decision=decision,
        tool_requests=tool_requests,
        narration_brief=narration_brief,
        citations=[
            f"{span['package_id']}:{span['reference_id']}"
            for span in state.get("retrieved_spans", [])
        ],
    )
    return _append_trace(
        {
            **state,
            "authority_result": authority.model_dump(),
            "turn_plan": plan.model_dump(),
            "tool_requests": [request.model_dump() for request in tool_requests],
        },
        "plan_turn_locally",
        {"decision": decision, "tool_requests": [request.tool_name for request in tool_requests]},
    )


def _tool_context() -> list[dict[str, Any]]:
    tools = []
    for tool in build_langchain_tools():
        if tool.name == "roll_dice":
            continue
        schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
        tools.append({"name": tool.name, "description": tool.description, "schema": schema})
    return tools


def build_llm_adjudication_node(model: BaseChatModel):
    def adjudicate_with_llm(state: GraphState) -> GraphState:
        if _player_is_using_pending_rule_opportunity(state):
            plan = _pending_rule_opportunity_answer_turn_plan(state)
            next_state = {
                **state,
                "intent": plan.intent.model_dump(),
                "authority_result": plan.authority.model_dump(),
                "turn_plan": plan.model_dump(),
                "tool_requests": [request.model_dump() for request in plan.tool_requests],
                "routing_decision": {
                    **state.get("routing_decision", {}),
                    "needs_rules_resolution": False,
                    "needs_scenario_director": False,
                },
            }
            return _append_trace(
                next_state,
                "adjudicate_with_llm",
                {
                    "decision": plan.decision,
                    "tool_requests": [request.tool_name for request in plan.tool_requests],
                    "short_circuit": "pending_rule_opportunity_answer",
                },
            )
        if _pending_rule_opportunity_blocks_resolution(state):
            plan = _pending_rule_opportunity_clarification_turn_plan(state)
            next_state = {
                **state,
                "intent": plan.intent.model_dump(),
                "authority_result": plan.authority.model_dump(),
                "turn_plan": plan.model_dump(),
                "tool_requests": [],
                "routing_decision": {
                    **state.get("routing_decision", {}),
                    "needs_rules_resolution": False,
                    "needs_scenario_director": False,
                },
            }
            return _append_trace(
                next_state,
                "adjudicate_with_llm",
                {
                    "decision": plan.decision,
                    "tool_requests": [],
                    "short_circuit": "pending_rule_opportunity_blocks_resolution",
                },
            )
        if _rules_advice_requires_player_clarification(state):
            plan = _rules_clarification_turn_plan(state)
            next_state = {
                **state,
                "intent": plan.intent.model_dump(),
                "authority_result": plan.authority.model_dump(),
                "turn_plan": plan.model_dump(),
                "tool_requests": [],
            }
            return _append_trace(
                next_state,
                "adjudicate_with_llm",
                {
                    "decision": plan.decision,
                    "tool_requests": [],
                    "short_circuit": "rules_advice_requires_clarification",
                },
            )
        if state.get("routing_decision", {}).get("route") == "clarify":
            plan = _clarification_turn_plan(state)
            next_state = {
                **state,
                "intent": plan.intent.model_dump(),
                "authority_result": plan.authority.model_dump(),
                "turn_plan": plan.model_dump(),
                "tool_requests": [],
            }
            return _append_trace(
                next_state,
                "adjudicate_with_llm",
                {
                    "decision": plan.decision,
                    "tool_requests": [],
                    "short_circuit": "routing_requires_clarification",
                },
            )
        if state.get("micro_gates_mode"):
            plan = _micro_gate_turn_plan(state)
            next_state = {
                **state,
                "intent": plan.intent.model_dump(),
                "authority_result": plan.authority.model_dump(),
                "turn_plan": plan.model_dump(),
                "tool_requests": [request.model_dump() for request in plan.tool_requests],
            }
            return _append_trace(
                next_state,
                "adjudicate_with_llm",
                {
                    "decision": plan.decision,
                    "tool_requests": [request.tool_name for request in plan.tool_requests],
                    "short_circuit": "micro_gates_local_turn_plan",
                },
            )
        contract_mode = _advisor_contract_mode(state)
        plan_schema = TurnPlan
        plan_schema_prompt: object = TurnPlan.model_json_schema()
        plan_model_kwargs = None
        if contract_mode == "compact":
            plan_schema = compact_schema_for(TurnPlan)
            plan_schema_prompt = compact_response_contract(plan_schema)
            plan_model_kwargs = {"max_tokens": 650}
        packet = _context_packet(
            state,
            "core_gm",
            mode="turn_adjudication",
            tool_catalog=_tool_context(),
            required_schema=plan_schema_prompt,
        )
        adjudication_payload = {
            "player_input": state.get("player_input", ""),
            "context": packet["context"],
        }
        started = time.perf_counter()
        raw_plan, attempts = invoke_structured_with_repair(
            model=model,
            prompt=CORE_GM_PROMPT,
            schema=plan_schema,
            payload=adjudication_payload,
            model_kwargs=plan_model_kwargs,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        plan = TurnPlan.model_validate(
            (
                adapt_compact_output(
                    role="core_gm",
                    output=raw_plan,
                    player_input=state.get("player_input", ""),
                    context={"routing_decision": state.get("routing_decision", {})},
                )
                if contract_mode == "compact"
                else raw_plan
            ).model_dump()
        )
        plan = _sanitize_turn_plan_internal_text(state, plan)
        next_state = {
            **state,
            "intent": plan.intent.model_dump(),
            "authority_result": plan.authority.model_dump(),
            "turn_plan": plan.model_dump(),
            "tool_requests": [request.model_dump() for request in plan.tool_requests],
        }
        return _append_trace(
            next_state,
            "adjudicate_with_llm",
            {
                "decision": plan.decision,
                "tool_requests": [r.tool_name for r in plan.tool_requests],
                "advisor": _structured_call_trace(
                    role="core_gm",
                    prompt_version=CORE_GM_PROMPT_VERSION,
                    schema_name=plan_schema.__name__,
                    elapsed_ms=elapsed_ms,
                    player_input=state.get("player_input", ""),
                    context=packet["context"],
                    schema_prompt=plan_schema_prompt,
                    attempts=attempts,
                ),
                "context_packet": packet["trace"],
                "structured_attempts": [
                    {key: value for key, value in attempt.items() if key != "raw_output"}
                    for attempt in attempts
                ],
            },
        )

    return adjudicate_with_llm


def plan_turn_from_routing(state: GraphState) -> GraphState:
    routing = IntentRoutingDecision.model_validate(state.get("routing_decision", {}))
    decision = routing.route
    if decision in {"rules_query", "memory_recall"}:
        decision = "answer"
    if decision not in {"answer", "free_action"}:
        decision = "free_action"
    plan = TurnPlan(
        intent=routing.intent,
        authority=AuthorityResult(
            ok=True,
            reason="Conditional advisor runtime accepted the structured low-risk routing.",
        ),
        decision=decision,  # type: ignore[arg-type]
        tool_requests=[],
        narration_brief=_direct_plan_narration_brief(routing),
        citations=list(routing.citations or []),
    )
    next_state = _with_advisor_skip_reason(
        {
            **state,
            "intent": plan.intent.model_dump(),
            "authority_result": plan.authority.model_dump(),
            "turn_plan": plan.model_dump(),
            "tool_requests": [],
        },
        "core_gm",
        "conditional_direct_plan",
    )
    next_state = {**next_state, "turn_complexity": _turn_complexity(next_state)}
    return _append_trace(
        next_state,
        "plan_turn_from_routing",
        {
            "decision": plan.decision,
            "tool_requests": [],
            "turn_complexity": next_state["turn_complexity"],
            "advisor_skip_reasons": next_state.get("advisor_skip_reasons", {}),
        },
    )


def _direct_plan_narration_brief(routing: IntentRoutingDecision) -> str:
    if routing.route == "memory_recall":
        return (
            "Answer only from established recent canon and player-visible memory. Do not create "
            "new facts or scene changes."
        )
    if routing.route == "rules_query":
        return (
            "Answer the rules question using loaded public rules context. Do not resolve an "
            "action or roll dice."
        )
    if routing.route == "answer":
        return (
            "Answer using established visible facts, public rules context, and player-visible "
            "memory only."
        )
    return (
        "Advance only a low-risk visible free action. Do not create durable facts, rewards, "
        "hidden reveals, costs, or scene transitions unless validated by later tools."
    )


def _micro_gate_turn_plan(state: GraphState) -> TurnPlan:
    routing = IntentRoutingDecision.model_validate(state.get("routing_decision", {}))
    route = routing.route
    decision = route
    if decision in {"rules_query", "memory_recall"}:
        decision = "answer"
    if decision not in {"answer", "free_action", "risky_action", "gm_move", "boundary", "clarify"}:
        decision = "free_action"
    authority_gate = state.get("micro_gate_results", {}).get("authority_micro_gate") or {}
    authority = AuthorityResult(
        ok=decision != "boundary",
        reason=str(authority_gate.get("reason") or "Micro-gates preserved fictional authority."),
        unsupported_claim=(
            state.get("player_input", "") if decision == "boundary" else None
        ),
        grounded_alternatives=[
            "State it as an attempt and let the loaded rules decide the outcome.",
            "Ask what is visibly established before committing to a claim.",
        ]
        if decision == "boundary"
        else [],
    )
    rules_advice = state.get("rules_advice") or {}
    tool_requests: list[ToolRequest] = []
    if decision == "risky_action" and bool(rules_advice.get("requires_resolution")):
        tool_requests.append(
            _resolver_tool_request(
                state,
                approach=_approach_for_resolver_request(state, rules_advice.get("approach_id")),
                **_rules_plugin_arguments(rules_advice),
                risk=str(rules_advice.get("risk") or "risky_uncertain"),
            )
        )
    return TurnPlan(
        intent=routing.intent,
        authority=authority,
        decision=decision,  # type: ignore[arg-type]
        tool_requests=tool_requests,
        narration_brief=_micro_gate_narration_brief(state, decision),
        citations=list(routing.citations or []) + list(rules_advice.get("citations") or []),
    )


def _micro_gate_narration_brief(state: GraphState, decision: str) -> str:
    briefs = {
        "answer": (
            "Answer using only established facts, player-visible memory, public rules context, "
            "and any requested player-visible scenario context."
        ),
        "free_action": (
            "Advance the proposed intent as a free action; show visible results without creating "
            "unauthorized success, rewards, or durable facts."
        ),
        "risky_action": (
            "The action is risky and unresolved; narrate only according to rules resolution and "
            "tool results, without declaring success or failure independently."
        ),
        "gm_move": (
            "The player waits or watches; advance a light visible pressure grounded in established "
            "fiction without choosing for the player."
        ),
        "boundary": (
            "The declaration cannot directly become established fact; convert it into an attempt, "
            "plan, or question."
        ),
        "clarify": "Ask the player for the missing target, intent, or priority before advancing.",
    }
    return briefs.get(decision, briefs["free_action"])


def _sanitize_turn_plan_internal_text(state: GraphState, plan: TurnPlan) -> TurnPlan:
    if not _prefers_chinese(plan.narration_brief):
        return plan
    return plan.model_copy(
        update={"narration_brief": _micro_gate_narration_brief(state, plan.decision)}
    )


def _rules_advice_requires_player_clarification(state: GraphState) -> bool:
    advice = state.get("rules_advice") or {}
    if not advice.get("requires_resolution"):
        return False
    if str(advice.get("risk") or "").lower() == "low":
        return False
    question = str(advice.get("clarification_question") or "").strip()
    if not question:
        return False
    return not _looks_like_mechanical_clarification(question)


def _looks_like_mechanical_clarification(question: str) -> bool:
    lowered = question.lower()
    markers = [
        "approach",
        "roll",
        "mechanic",
        "procedure",
        "stat",
        "判定",
        "掷骰",
        "骰",
        "规则",
        "合并",
        "分别判定",
        "gm需裁定",
    ]
    return any(marker in lowered or marker in question for marker in markers)


def _pending_rule_opportunities(state: GraphState) -> list[dict[str, Any]]:
    world = state.get("world_projection", {})
    raw_items = world.get("pending_rule_opportunities") if isinstance(world, dict) else None
    if not isinstance(raw_items, list):
        return []
    opportunities: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "pending").lower() != "pending":
            continue
        opportunities.append(item)
    return opportunities


def _player_is_using_pending_rule_opportunity(state: GraphState) -> bool:
    if not _pending_rule_opportunities(state):
        return False
    routing = state.get("routing_decision") or {}
    route = str(routing.get("route") or "")
    intent = routing.get("intent") if isinstance(routing.get("intent"), dict) else {}
    intent_kind = str(intent.get("kind") or state.get("intent", {}).get("kind") or "")
    return route in {"answer", "rules_query", "memory_recall"} or intent_kind in {
        "info_query",
        "rules_query",
        "memory_recall",
    }


def _pending_rule_opportunity_blocks_resolution(state: GraphState) -> bool:
    if not _pending_rule_opportunities(state):
        return False
    if _player_is_using_pending_rule_opportunity(state):
        return False
    advice = state.get("rules_advice") or {}
    risk = str(advice.get("risk") or "").lower()
    if advice.get("requires_resolution") and risk != "low":
        return True
    routing = state.get("routing_decision") or {}
    if routing.get("needs_rules_resolution") or routing.get("route") == "risky_action":
        return True
    plan = state.get("turn_plan") or {}
    return plan.get("decision") == "risky_action"


def build_llm_intent_arbiter_node(model: BaseChatModel):
    def route_with_intent_arbiter(state: GraphState) -> GraphState:
        try:
            packet = _context_packet(
                state,
                "intent_arbiter",
                advisor_contract="IntentRoutingDecision",
            )
            result = invoke_advisor(
                model=model,
                role="intent_arbiter",
                player_input=state.get("player_input", ""),
                context=packet["context"],
                sqlite_path=state.get("sqlite_path"),
                turn_id=state.get("turn_id"),
                contract_mode=_advisor_contract_mode(state),
            )
            routing = result.output.model_dump()
            trace_payload = {
                "route": routing.get("route"),
                "needs_rules_resolution": routing.get("needs_rules_resolution"),
                "needs_scenario_director": routing.get("needs_scenario_director"),
                "advisor": result.trace_metadata,
                "context_packet": packet["trace"],
                "structured_attempts": [
                    {key: value for key, value in attempt.items() if key != "raw_output"}
                    for attempt in result.attempts
                ],
            }
        except Exception as error:
            fallback = _fallback_routing_decision(state, error)
            routing = fallback.model_dump()
            trace_payload = {
                "route": routing.get("route"),
                "needs_rules_resolution": routing.get("needs_rules_resolution"),
                "needs_scenario_director": routing.get("needs_scenario_director"),
                "advisor_error": str(error),
                "fallback": True,
            }

        if routing.get("route") == "clarify":
            routing = {
                **routing,
                "needs_rules_resolution": False,
                "needs_scenario_director": False,
                "needs_memory_recall": False,
                "allow_direct_answer": True,
            }
            trace_payload["route"] = routing.get("route")
            trace_payload["needs_rules_resolution"] = routing.get("needs_rules_resolution")
            trace_payload["needs_scenario_director"] = routing.get("needs_scenario_director")
            trace_payload["normalized_clarification"] = True

        return _append_trace(
            {
                **state,
                "routing_decision": routing,
                "intent": routing["intent"],
            },
            "route_with_intent_arbiter",
            trace_payload,
        )

    return route_with_intent_arbiter


MICRO_GATE_ROLES: tuple[AdvisorRole, ...] = (
    "authority_micro_gate",
    "intent_micro_gate",
    "risk_micro_gate",
    "target_micro_gate",
    "memory_recall_micro_gate",
)


def build_llm_micro_gates_node(
    model: BaseChatModel,
    *,
    advisor_models: AdvisorModelMap | None = None,
):
    def run_micro_gates(state: GraphState) -> GraphState:
        results: dict[str, Any] = {}
        traces: dict[str, Any] = {}
        with ThreadPoolExecutor(
            max_workers=len(MICRO_GATE_ROLES),
            thread_name_prefix="trpg-micro-gate",
        ) as executor:
            futures = {
                executor.submit(
                    _run_micro_gate,
                    role=role,
                    model=_model_for_role(
                        role,
                        default_model=model,
                        advisor_models=advisor_models,
                    ),
                    state=state,
                ): role
                for role in MICRO_GATE_ROLES
            }
            for future, role in futures.items():
                gate_result, gate_trace = future.result()
                results[role] = gate_result
                traces[role] = gate_trace

        routing = _routing_from_micro_gates(state, results)
        trace_payload = {
            "route": routing.get("route"),
            "needs_rules_resolution": routing.get("needs_rules_resolution"),
            "needs_scenario_director": routing.get("needs_scenario_director"),
            "gate_traces": traces,
        }
        if routing.get("route") == "clarify":
            routing = {
                **routing,
                "needs_rules_resolution": False,
                "needs_scenario_director": False,
                "needs_memory_recall": False,
                "allow_direct_answer": True,
            }
            trace_payload["normalized_clarification"] = True
        return _append_trace(
            {
                **state,
                "micro_gate_results": results,
                "routing_decision": routing,
                "intent": routing["intent"],
            },
            "run_micro_gates",
            trace_payload,
        )

    return run_micro_gates


def _run_micro_gate(
    *,
    role: AdvisorRole,
    model: BaseChatModel,
    state: GraphState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        result = invoke_advisor(
            model=model,
            role=role,
            player_input=state.get("player_input", ""),
            context=_micro_gate_context(state, role),
            sqlite_path=state.get("sqlite_path"),
            turn_id=state.get("turn_id"),
            contract_mode=_advisor_contract_mode(state),
        )
        return result.output.model_dump(), {
            "advisor": result.trace_metadata,
            "structured_attempts": [
                {key: value for key, value in attempt.items() if key != "raw_output"}
                for attempt in result.attempts
            ],
        }
    except Exception as error:
        fallback = _fallback_micro_gate_decision(role, state, error)
        return fallback, {"fallback": True, "advisor_error": str(error)}


def _micro_gate_context(state: GraphState, role: AdvisorRole) -> dict[str, Any]:
    common = {
        "ruleset_id": state.get("ruleset_id"),
        "scenario_id": state.get("scenario_id"),
        "turn_id": state.get("turn_id"),
        "context_clip_policy": role,
    }
    if role == "authority_micro_gate":
        return common | {
            "visible_world": _player_visible_world_context(state),
            "recent_canon": _recent_canon_clip(state),
            "player_visible_memory_hits": _player_visible_memory_clip(state),
            "public_retrieved_spans": _public_retrieved_span_clip(state, limit=3),
        }
    if role == "intent_micro_gate":
        return common | {
            "visible_world": _player_visible_world_context(state),
            "recent_canon": _recent_canon_clip(state),
            "player_visible_memory_hits": _player_visible_memory_clip(state, limit=3),
            "public_retrieved_spans": _public_retrieved_span_clip(state, limit=3),
        }
    if role == "risk_micro_gate":
        return common | {
            "ruleset_profile": _ruleset_profile_clip(state),
            "rules_retrieved_spans": _rules_retrieved_span_clip(state, limit=3),
            "visible_world": _player_visible_world_context(state),
            "character_context": state.get("character_context", {}),
        }
    if role == "target_micro_gate":
        return common | {
            "visible_world": _player_visible_world_context(state),
            "public_package_profiles": _single_turn_visible_package_profiles(state),
            "public_retrieved_spans": _public_retrieved_span_clip(state, limit=3),
        }
    if role == "memory_recall_micro_gate":
        return common | {
            "recent_canon": _recent_canon_clip(state, limit=5),
            "player_visible_memory_hits": _player_visible_memory_clip(state),
            "memory_hit_count": len(state.get("memory_hits", [])),
            "player_visible_memory_hit_count": len(state.get("player_memory_hits", [])),
        }
    return common


def _routing_from_micro_gates(
    state: GraphState,
    gate_results: dict[str, Any],
) -> dict[str, Any]:
    authority = AuthorityMicroGateDecision.model_validate(
        gate_results.get("authority_micro_gate")
        or _fallback_micro_gate_decision(
            "authority_micro_gate",
            state,
            RuntimeError("missing authority micro-gate"),
        )
    )
    intent = IntentMicroGateDecision.model_validate(
        gate_results.get("intent_micro_gate")
        or _fallback_micro_gate_decision(
            "intent_micro_gate",
            state,
            RuntimeError("missing intent micro-gate"),
        )
    )
    risk = RiskMicroGateDecision.model_validate(
        gate_results.get("risk_micro_gate")
        or _fallback_micro_gate_decision(
            "risk_micro_gate",
            state,
            RuntimeError("missing risk micro-gate"),
        )
    )
    target = TargetMicroGateDecision.model_validate(
        gate_results.get("target_micro_gate")
        or _fallback_micro_gate_decision(
            "target_micro_gate",
            state,
            RuntimeError("missing target micro-gate"),
        )
    )
    memory = MemoryRecallMicroGateDecision.model_validate(
        gate_results.get("memory_recall_micro_gate")
        or _fallback_micro_gate_decision(
            "memory_recall_micro_gate",
            state,
            RuntimeError("missing memory micro-gate"),
        )
    )

    route = intent.route
    intent_kind = intent.intent.kind
    uncertainty: str | None = None
    allow_direct_answer = intent.allow_direct_answer
    needs_memory_recall = route == "memory_recall"
    needs_scenario_director = intent.needs_scenario_director

    if authority.boundary or not authority.allowed:
        route = "boundary"
        intent_kind = "boundary_claim"
        allow_direct_answer = True
        needs_scenario_director = False
        uncertainty = authority.reason
    elif authority.needs_clarification or target.needs_clarification:
        route = "clarify"
        intent_kind = "clarify_needed"
        allow_direct_answer = True
        needs_scenario_director = False
        uncertainty = target.clarification_question or target.reason or authority.reason
    elif memory.needs_memory_recall:
        route = "memory_recall"
        intent_kind = "memory_recall"
        needs_memory_recall = True
        allow_direct_answer = True
    elif risk.needs_rules_resolution or risk.risky:
        route = "risky_action"
        intent_kind = "action"
        needs_scenario_director = True
    elif route == "answer":
        allow_direct_answer = True
    elif route in {"rules_query", "memory_recall"}:
        allow_direct_answer = True
        needs_scenario_director = False
    elif route in {"boundary", "clarify"}:
        allow_direct_answer = True
        needs_scenario_director = False

    needs_rules_resolution = route == "risky_action" and (
        risk.needs_rules_resolution or risk.risky
    )
    routing = IntentRoutingDecision(
        intent=IntentClassification(
            kind=intent_kind,  # type: ignore[arg-type]
            confidence=_micro_gate_confidence(route=route, gate_results=gate_results),
            reason=_micro_gate_reason_summary(
                authority=authority,
                intent=intent,
                risk=risk,
                target=target,
                memory=memory,
            ),
        ),
        route=route,  # type: ignore[arg-type]
        needs_rules_resolution=needs_rules_resolution,
        needs_scenario_director=needs_scenario_director
        and route in {"answer", "free_action", "risky_action", "gm_move"},
        needs_memory_recall=needs_memory_recall,
        allow_direct_answer=allow_direct_answer,
        reasoning_summary="Parallel micro-gates produced a bounded routing decision.",
        uncertainty=uncertainty,
        citations=[],
    )
    return routing.model_dump()


def _micro_gate_confidence(*, route: str, gate_results: dict[str, Any]) -> float:
    if any("fallback" in str(result).lower() for result in gate_results.values()):
        return 0.45
    if route in {"clarify", "boundary", "risky_action"}:
        return 0.75
    return 0.65


def _micro_gate_reason_summary(
    *,
    authority: AuthorityMicroGateDecision,
    intent: IntentMicroGateDecision,
    risk: RiskMicroGateDecision,
    target: TargetMicroGateDecision,
    memory: MemoryRecallMicroGateDecision,
) -> str:
    pieces = [
        f"authority: {authority.reason}",
        f"intent: {intent.reason}",
        f"risk: {risk.reason}",
        f"target: {target.reason}",
        f"memory: {memory.reason}",
    ]
    return " | ".join(piece[:220] for piece in pieces)


def _fallback_micro_gate_decision(
    role: AdvisorRole,
    state: GraphState,
    error: Exception,
) -> dict[str, Any]:
    reason = f"Micro-gate failed; structural fallback used without keyword routing: {error}"
    if role == "authority_micro_gate":
        return AuthorityMicroGateDecision(
            allowed=True,
            boundary=False,
            needs_clarification=False,
            reason=reason,
            player_facing_boundary=None,
        ).model_dump()
    if role == "intent_micro_gate":
        return IntentMicroGateDecision(
            intent=IntentClassification(
                kind="action",
                confidence=0.2,
                reason=reason,
            ),
            route="free_action",
            allow_direct_answer=False,
            needs_scenario_director=True,
            reason=reason,
        ).model_dump()
    if role == "risk_micro_gate":
        return RiskMicroGateDecision(
            risky=False,
            risk="none",
            needs_rules_resolution=False,
            reason=reason,
        ).model_dump()
    if role == "target_micro_gate":
        return TargetMicroGateDecision(
            ambiguous=False,
            needs_clarification=False,
            clarification_question=None,
            reason=reason,
        ).model_dump()
    if role == "memory_recall_micro_gate":
        return MemoryRecallMicroGateDecision(
            needs_memory_recall=False,
            reason=reason,
        ).model_dump()
    raise ValueError(f"Unsupported micro-gate role: {role}")


def _player_visible_world_context(state: GraphState) -> dict[str, Any]:
    world = state.get("world_projection", {})
    if not isinstance(world, dict):
        return {}
    scene = world.get("scene")
    visible_scene: dict[str, Any] = {}
    if isinstance(scene, dict):
        for key in ("id", "title", "public_summary"):
            if key in scene:
                visible_scene[key] = scene[key]
    visible_world = {
        "active_scene": world.get("active_scene"),
        "scene": visible_scene,
        "clock": _player_visible_value(world.get("clock")),
        "revealed_facts": _player_visible_value(world.get("revealed_facts", [])),
        "known_clues": _player_visible_value(world.get("known_clues", [])),
        "pending_rule_opportunities": _player_visible_value(
            world.get("pending_rule_opportunities", [])
        ),
    }
    return {key: value for key, value in visible_world.items() if value not in (None, [], {})}


def _player_visible_value(value: Any) -> Any:
    if isinstance(value, dict):
        visibility = str(value.get("visibility") or value.get("access") or "").lower()
        if visibility in {"gm_only", "secret", "hidden"}:
            return None
        visible: dict[str, Any] = {}
        for key, item in value.items():
            if key in {
                "gm_only",
                "gm_only_reason",
                "secret",
                "secrets",
                "hidden",
                "private",
            }:
                continue
            clipped = _player_visible_value(item)
            if clipped is not None:
                visible[key] = clipped
        return visible
    if isinstance(value, list):
        return [
            item
            for item in (_player_visible_value(item) for item in value)
            if item is not None
        ]
    return value


def _recent_canon_clip(state: GraphState, *, limit: int = 3) -> list[str]:
    return [str(item)[:600] for item in state.get("recent_canon", [])[-limit:]]


def _player_visible_memory_clip(state: GraphState, *, limit: int = 5) -> list[dict[str, Any]]:
    return _memory_hit_clip(state.get("player_memory_hits", []), limit=limit)


def _memory_hit_clip(hits: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for hit in hits[:limit]:
        if not isinstance(hit, dict):
            continue
        metadata = hit.get("metadata")
        clipped.append(
            {
                "kind": hit.get("kind"),
                "scope": hit.get("scope"),
                "text": str(hit.get("text", ""))[:600],
                "metadata": _player_visible_value(metadata) if isinstance(metadata, dict) else {},
            }
        )
    return clipped


def _public_retrieved_span_clip(state: GraphState, *, limit: int) -> list[dict[str, Any]]:
    return [
        _span_clip(span)
        for span in state.get("retrieved_spans", [])
        if str(span.get("visibility") or "public") == "public"
    ][:limit]


def _rules_retrieved_span_clip(state: GraphState, *, limit: int) -> list[dict[str, Any]]:
    ruleset_id = state.get("ruleset_id")
    return [
        _span_clip(span)
        for span in state.get("retrieved_spans", [])
        if not ruleset_id or span.get("package_id") == ruleset_id
    ][:limit]


def _span_clip(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "package_id": span.get("package_id"),
        "reference_id": span.get("reference_id"),
        "visibility": span.get("visibility"),
        "score": span.get("score"),
        "text": str(span.get("text", ""))[:1200],
    }


def _ruleset_profile_clip(state: GraphState) -> dict[str, Any]:
    ruleset_id = state.get("ruleset_id")
    for profile in state.get("package_profiles", []):
        if not isinstance(profile, dict):
            continue
        if profile.get("id") == ruleset_id:
            return {
                "id": profile.get("id"),
                "kind": profile.get("kind"),
                "name": profile.get("name"),
                "description": profile.get("description"),
                "capabilities": profile.get("capabilities", []),
                "references": profile.get("references", []),
            }
    return {}


def _clarification_turn_plan(state: GraphState) -> TurnPlan:
    return TurnPlan(
        intent=IntentClassification(
            kind="clarify_needed",
            confidence=0.8,
            reason="The player action needs a more specific target before the GM can advance.",
        ),
        authority=AuthorityResult(
            ok=True,
            reason="Clarification preserves player agency and avoids choosing a target for them.",
        ),
        decision="clarify",
        tool_requests=[],
        narration_brief=_target_clarification_brief(state),
        citations=[],
    )


def _rules_clarification_turn_plan(state: GraphState) -> TurnPlan:
    return TurnPlan(
        intent=IntentClassification(
            kind="clarify_needed",
            confidence=0.85,
            reason="The rules advisor identified a consequential choice that needs player input.",
        ),
        authority=AuthorityResult(
            ok=True,
            reason="Clarifying before resolution preserves player agency and avoids choosing a "
            "mechanical approach for the player.",
        ),
        decision="clarify",
        tool_requests=[],
        narration_brief=_rules_clarification_brief(state),
        citations=[],
    )


def _pending_rule_opportunity_answer_turn_plan(state: GraphState) -> TurnPlan:
    opportunity = _pending_rule_opportunities(state)[0]
    patches = [
        WorldPatch(op="set", path=["pending_rule_opportunities"], value=[]),
    ]
    if bool(opportunity.get("grants_prepared")):
        patches.append(WorldPatch(op="set", path=["character_context", "prepared"], value=True))
    prompt = str(opportunity.get("prompt") or "").strip()
    effect = str(opportunity.get("effect") or "").strip()
    brief = _pending_rule_opportunity_answer_text(
        state=state,
        prompt=prompt,
        effect=effect,
        grants_prepared=bool(opportunity.get("grants_prepared")),
    )
    return TurnPlan(
        intent=IntentClassification(
            kind="info_query",
            confidence=0.9,
            reason="The player is using a pending rules-granted opportunity to ask a question.",
        ),
        authority=AuthorityResult(
            ok=True,
            reason="The question is supported by a pending rules-granted opportunity.",
        ),
        decision="answer",
        tool_requests=[
            ToolRequest(
                tool_name="apply_world_patch",
                arguments={
                    "patches": [patch.model_dump() for patch in patches],
                    "reason": "Consume pending rules-granted opportunity.",
                },
                reason="Pending opportunity is consumed before the next risky resolution.",
            )
        ],
        narration_brief=brief,
        citations=[],
    )


def _pending_rule_opportunity_clarification_turn_plan(state: GraphState) -> TurnPlan:
    opportunity = _pending_rule_opportunities(state)[0]
    prompt = str(opportunity.get("prompt") or "").strip()
    effect = str(opportunity.get("effect") or "").strip()
    return TurnPlan(
        intent=IntentClassification(
            kind="clarify_needed",
            confidence=0.9,
            reason=(
                "A pending rules-granted opportunity must be used or waived before another "
                "risky resolution."
            ),
        ),
        authority=AuthorityResult(
            ok=True,
            reason="Clarifying before resolution preserves the player's earned option.",
        ),
        decision="clarify",
        tool_requests=[],
        narration_brief=_pending_rule_opportunity_clarification_text(
            state=state,
            prompt=prompt,
            effect=effect,
        ),
        citations=[],
    )


def _pending_rule_opportunity_answer_text(
    *,
    state: GraphState,
    prompt: str,
    effect: str,
    grants_prepared: bool,
) -> str:
    pieces = [
        "Answer the pending rules-granted question using only visible scene context, "
        "recent action, and established facts.",
        "Do not introduce new success, failure, damage, structural changes, or unauthorized facts.",
    ]
    if grants_prepared:
        pieces.append(
            "After answering, state that the next relevant action is treated as prepared "
            "if it uses this answer."
        )
    return " ".join(pieces)


def _pending_rule_opportunity_clarification_text(
    *,
    state: GraphState,
    prompt: str,
    effect: str,
) -> str:
    base = "Before another risky resolution, you still have a pending rules-granted opportunity."
    return (
        f"{base} Ask the player whether they use it now or explicitly waive it and continue."
    )


def _target_clarification_brief(state: GraphState) -> str:
    return (
        "The player's input needs a clearer priority, first target, or first step. Ask the "
        "player to name the specific person, system, object, location, or first step before "
        "advancing play."
    )


def _target_clarification_text(state: GraphState) -> str:
    player_input = state.get("player_input", "")
    texture = _clarification_scene_texture(state)
    if _prefers_chinese(player_input):
        texture_sentence = f"当前能确认的现场信息：{texture} " if texture else ""
        return (
            f"你说的“{player_input}”还需要明确优先处理的目标或第一步。"
            f"{texture_sentence}"
            "请说明你现在最先要询问、联系、检查或操作的具体对象，我再继续主持。"
        )
    texture_sentence = f"Visible context: {texture} " if texture else ""
    return (
        f"'{player_input}' needs a clearer priority or first target. "
        f"{texture_sentence}"
        "Name the specific person, system, object, or first step you want to handle, "
        "and I will continue from there."
    )


def _rules_clarification_brief(state: GraphState) -> str:
    question = str(state.get("rules_advice", {}).get("clarification_question") or "").strip()
    if not question or re.search(r"[\u4e00-\u9fff]", question):
        return _target_clarification_brief(state)
    return (
        "A consequential fictional choice changes how the action should be resolved. "
        f"Clarification needed: {question} Ask for the main fictional approach before "
        "continuing."
    )


def _rules_clarification_text(state: GraphState) -> str:
    player_input = state.get("player_input", "")
    question = str(state.get("rules_advice", {}).get("clarification_question") or "").strip()
    if not question:
        return _target_clarification_text(state)
    if _prefers_chinese(player_input):
        if not _prefers_chinese(question):
            return (
                "这个动作的重点会影响接下来怎么裁定。"
                "请先说明你最主要的做法、目标或优先顺序，我再沿这个做法继续。"
            )
        return (
            "这个动作的重点会影响接下来怎么裁定。"
            f"{question}"
            "请先选定主要做法，我再沿这个做法继续。"
        )
    if _prefers_chinese(question):
        return (
            "The focus of this action changes how it should be resolved. "
            "Choose the main fictional approach, target, or priority first and I will continue "
            "from there."
        )
    return (
        "The focus of this action changes how it should be resolved. "
        f"{question} "
        "Choose the main approach first and I will continue from there."
    )


def _context_text(value: object) -> str:
    if isinstance(value, dict):
        for key in ("content", "text", "fact", "summary"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""
    return str(value).strip()


def _ensure_terminal_punctuation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if re.search(r"[。！？.!?]$", stripped):
        return stripped
    if re.search(r"[\u4e00-\u9fff]", stripped):
        return f"{stripped}。"
    return f"{stripped}."


def _join_context_fragments(pieces: list[str]) -> str:
    cleaned = [piece.strip() for piece in pieces if piece.strip()]
    if not cleaned:
        return ""
    if any(re.search(r"[\u4e00-\u9fff]", piece) for piece in cleaned):
        fragments = [re.sub(r"[。！？.!?]+$", "", piece).strip() for piece in cleaned]
        return f"{'；'.join(fragment for fragment in fragments if fragment)}。"
    return " ".join(_ensure_terminal_punctuation(piece) for piece in cleaned)


def _clarification_scene_texture(state: GraphState) -> str:
    world = state.get("world_projection", {})
    scene = world.get("scene")
    if not isinstance(scene, dict) or not scene.get("public_summary"):
        return ""
    clauses = _split_context_sentences(str(scene["public_summary"]))
    if not clauses:
        return ""
    return " ".join(clauses[:2])[:120]


def route_after_intent_arbiter(state: GraphState) -> str:
    routing = state.get("routing_decision", {})
    if (
        routing.get("needs_rules_resolution")
        or routing.get("needs_rules_review")
        or routing.get("route") == "risky_action"
    ):
        return "rules_advice"
    if _should_direct_plan_from_routing(state):
        return "direct_plan"
    return "adjudicate"


def _fallback_routing_decision(state: GraphState, error: Exception) -> IntentRoutingDecision:
    return IntentRoutingDecision(
        intent=IntentClassification(
            kind="clarify_needed",
            confidence=0.35,
            reason=(
                "Intent advisor failed; structural fallback used without keyword routing: "
                f"{error}"
            ),
        ),
        route="clarify",
        needs_rules_resolution=False,
        needs_scenario_director=False,
        needs_memory_recall=False,
        allow_direct_answer=True,
        reasoning_summary="Fallback routing used because the intent advisor failed.",
        uncertainty=str(error),
        citations=[],
    )


def build_llm_rules_adjudicator_node(model: BaseChatModel):
    def advise_rules_with_llm(state: GraphState) -> GraphState:
        try:
            packet = _context_packet(
                state,
                "rules_adjudicator",
                advisor_contract="RulesAdjudicationAdvice",
            )
            result = invoke_advisor(
                model=model,
                role="rules_adjudicator",
                player_input=state.get("player_input", ""),
                context=packet["context"],
                sqlite_path=state.get("sqlite_path"),
                turn_id=state.get("turn_id"),
                contract_mode=_advisor_contract_mode(state),
            )
            advice = result.output.model_dump()
            trace_payload = {
                "requires_resolution": advice.get("requires_resolution"),
                "procedure_id": advice.get("procedure_id"),
                "approach_id": advice.get("approach_id"),
                "risk": advice.get("risk"),
                "advisor": result.trace_metadata,
                "context_packet": packet["trace"],
                "structured_attempts": [
                    {key: value for key, value in attempt.items() if key != "raw_output"}
                    for attempt in result.attempts
                ],
            }
        except Exception as error:
            fallback = RulesAdjudicationAdvice(
                requires_resolution=True,
                procedure_id=None,
                approach_id=None,
                risk="risky_uncertain",
                stakes="Rules advisor failed; deterministic resolver fallback will decide.",
                clarification_question=None,
                citations=[],
            )
            advice = fallback.model_dump()
            trace_payload = {
                "requires_resolution": True,
                "procedure_id": None,
                "approach_id": None,
                "risk": "risky_uncertain",
                "advisor_error": str(error),
                "fallback": True,
            }
        return _append_trace(
            {**state, "rules_advice": advice},
            "advise_rules_with_llm",
            trace_payload,
        )

    return advise_rules_with_llm


def build_llm_single_turn_advisor_node(model: BaseChatModel):
    def advise_turn_with_single_llm(state: GraphState) -> GraphState:
        if _player_is_using_pending_rule_opportunity(state):
            plan = _pending_rule_opportunity_answer_turn_plan(state)
            next_state = {
                **state,
                "intent": plan.intent.model_dump(),
                "authority_result": plan.authority.model_dump(),
                "routing_decision": {
                    "intent": plan.intent.model_dump(),
                    "route": "answer",
                    "needs_rules_resolution": False,
                    "needs_scenario_director": False,
                    "needs_memory_recall": False,
                    "allow_direct_answer": True,
                    "reasoning_summary": "Player is using a pending rules-granted opportunity.",
                    "uncertainty": None,
                    "citations": [],
                },
                "rules_advice": RulesAdjudicationAdvice(
                    requires_resolution=False,
                    procedure_id=None,
                    approach_id=None,
                    risk="none",
                    stakes="Pending question opportunity is answered without a new resolution.",
                    clarification_question=None,
                    citations=[],
                ).model_dump(),
                "turn_plan": plan.model_dump(),
                "tool_requests": [request.model_dump() for request in plan.tool_requests],
            }
            return _append_trace(
                next_state,
                "advise_turn_with_single_llm",
                {"short_circuit": "pending_rule_opportunity_answer"},
            )

        try:
            packet = _context_packet(
                state,
                "single_turn_advisor",
                mode="single_turn_adjudication",
                tool_catalog=_tool_context(),
                advisor_contract="SingleTurnAdvisorDecision",
            )
            result = invoke_advisor(
                model=model,
                role="single_turn_advisor",
                player_input=state.get("player_input", ""),
                context=packet["context"],
                sqlite_path=state.get("sqlite_path"),
                turn_id=state.get("turn_id"),
                contract_mode=_advisor_contract_mode(state),
            )
            advice = SingleTurnAdvisorDecision.model_validate(result.output.model_dump())
            advisor_trace = result.trace_metadata
            context_packet_trace = packet["trace"]
            structured_attempts = [
                {key: value for key, value in attempt.items() if key != "raw_output"}
                for attempt in result.attempts
            ]
        except Exception as error:
            routing = _fallback_routing_decision(state, error)
            rules_advice = RulesAdjudicationAdvice(
                requires_resolution=bool(routing.needs_rules_resolution),
                procedure_id=None,
                approach_id=None,
                risk="risky_uncertain" if routing.needs_rules_resolution else "none",
                stakes="Single-turn advisor failed; deterministic fallback used.",
                clarification_question=None,
                citations=[],
            )
            scenario_advice = ScenarioDirectorDecision(
                decision="no_change",
                proposed_patches=[],
                player_visible_context="Single-turn advisor fallback proposed no scene change.",
                gm_only_reason=f"Single-turn advisor failed safely: {error}",
                citations=[],
            )
            plan = _single_turn_fallback_plan(state, routing, rules_advice)
            advice = SingleTurnAdvisorDecision(
                routing_decision=routing,
                rules_advice=rules_advice,
                turn_plan=plan,
                scenario_advice=scenario_advice,
                reasoning_summary="Fallback used because single-turn advisor failed.",
            )
            advisor_trace = {"fallback": "true", "error": str(error)}
            context_packet_trace = {}
            structured_attempts = []

        routing = advice.routing_decision.model_dump()
        rules_advice = advice.rules_advice.model_dump()
        plan = _harden_single_turn_plan(
            state=state,
            routing=advice.routing_decision,
            rules_advice=advice.rules_advice,
            plan=advice.turn_plan,
        )
        next_state = {
            **state,
            "single_turn_advice": advice.model_dump(),
            "single_turn_scenario_advice": advice.scenario_advice.model_dump(),
            "intent": plan.intent.model_dump(),
            "authority_result": plan.authority.model_dump(),
            "routing_decision": routing,
            "rules_advice": rules_advice,
            "turn_plan": plan.model_dump(),
            "tool_requests": [request.model_dump() for request in plan.tool_requests],
        }
        return _append_trace(
            next_state,
            "advise_turn_with_single_llm",
            {
                "decision": plan.decision,
                "route": routing.get("route"),
                "requires_resolution": rules_advice.get("requires_resolution"),
                "scenario_decision": advice.scenario_advice.decision,
                "advisor": advisor_trace,
                "context_packet": context_packet_trace,
                "structured_attempts": structured_attempts,
            },
        )

    return advise_turn_with_single_llm


def _single_turn_visible_spans(state: GraphState) -> list[dict[str, Any]]:
    return [
        span
        for span in state.get("retrieved_spans", [])
        if str(span.get("visibility") or "public") == "public"
    ]


def _single_turn_visible_package_profiles(state: GraphState) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for profile in state.get("package_profiles", []):
        if not isinstance(profile, dict):
            continue
        references = [
            reference
            for reference in profile.get("references", [])
            if isinstance(reference, dict)
            and str(reference.get("visibility") or "public") == "public"
        ]
        profiles.append({**profile, "references": references})
    return profiles


def _single_turn_fallback_plan(
    state: GraphState,
    routing: IntentRoutingDecision,
    rules_advice: RulesAdjudicationAdvice,
) -> TurnPlan:
    decision = "risky_action" if routing.needs_rules_resolution else routing.route
    if decision in {"rules_query", "memory_recall"}:
        decision = "answer"
    if decision not in {"answer", "free_action", "risky_action", "gm_move", "boundary", "clarify"}:
        decision = "free_action"
    plan = TurnPlan(
        intent=routing.intent,
        authority=AuthorityResult(ok=True, reason="Fallback preserved the player proposal."),
        decision=decision,  # type: ignore[arg-type]
        tool_requests=[],
        narration_brief=rules_advice.stakes,
        citations=[*routing.citations, *rules_advice.citations],
    )
    return _harden_single_turn_plan(
        state=state,
        routing=routing,
        rules_advice=rules_advice,
        plan=plan,
    )


def _harden_single_turn_plan(
    *,
    state: GraphState,
    routing: IntentRoutingDecision,
    rules_advice: RulesAdjudicationAdvice,
    plan: TurnPlan,
) -> TurnPlan:
    if _pending_rule_opportunity_blocks_resolution(
        {
            **state,
            "routing_decision": routing.model_dump(),
            "rules_advice": rules_advice.model_dump(),
            "turn_plan": plan.model_dump(),
        }
    ):
        return _pending_rule_opportunity_clarification_turn_plan(state)
    if plan.decision == "clarify":
        return _single_turn_clarification_turn_plan(state, rules_advice)
    if routing.needs_rules_resolution or rules_advice.requires_resolution:
        plan = plan.model_copy(update={"decision": "risky_action"})
    return _sanitize_turn_plan_internal_text(state, plan)


def _single_turn_clarification_turn_plan(
    state: GraphState,
    rules_advice: RulesAdjudicationAdvice,
) -> TurnPlan:
    question = str(rules_advice.clarification_question or "").strip()
    if not question or re.search(r"[\u4e00-\u9fff]", question):
        question = (
            "The target or fictional approach is not clear yet. Ask the player to name the "
            "specific place, position, object, or method before advancing play."
        )
    return TurnPlan(
        intent=IntentClassification(
            kind="clarify_needed",
            confidence=0.9,
            reason="Single-turn advisor requested clarification; local text hardening applied.",
        ),
        authority=AuthorityResult(
            ok=True,
            reason="Clarification preserves player agency and prevents choosing a target.",
        ),
        decision="clarify",
        tool_requests=[],
        narration_brief=question,
        citations=list(rules_advice.citations),
    )


def build_llm_narration_node(model: BaseChatModel):
    def narrate_with_llm(state: GraphState) -> GraphState:
        if state.get("turn_plan", {}).get("decision") == "clarify":
            final_text = _clarification_player_text(state)
            next_state = {
                **state,
                "narration_plan": {
                    "final_text": final_text,
                    "canon_event_draft": None,
                    "memory_candidates": [],
                },
                "final_output": final_text,
            }
            return _append_trace(
                next_state,
                "narrate_with_llm",
                {"final_output": final_text, "short_circuit": "clarification"},
            )
        if _has_failed_required_resolution(state):
            final_text = _failed_required_resolution_text(state)
            return _append_trace(
                {
                    **state,
                    "narration_plan": {
                        "final_text": final_text,
                        "canon_event_draft": None,
                        "memory_candidates": [],
                    },
                    "final_output": final_text,
                },
                "narrate_with_llm",
                {
                    "final_output": final_text,
                    "blocked_reason": "required_rules_resolution_failed",
                },
            )

        try:
            contract_mode = _advisor_contract_mode(state)
            narration_schema = NarrationPlan
            narration_schema_prompt: object = NarrationPlan.model_json_schema()
            narration_model_kwargs = None
            if contract_mode == "compact":
                narration_schema = compact_schema_for(NarrationPlan)
                narration_schema_prompt = compact_response_contract(narration_schema)
                narration_model_kwargs = {"max_tokens": 1100}
            packet = _context_packet(state, "narrator")
            narration_payload = {
                "player_input": state.get("player_input", ""),
                "schema": narration_schema_prompt,
                "context": packet["context"],
            }
            started = time.perf_counter()
            raw_narration, attempts = invoke_structured_with_repair(
                model=model,
                prompt=NARRATION_PROMPT,
                schema=narration_schema,
                payload=narration_payload,
                model_kwargs=narration_model_kwargs,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            narration = NarrationPlan.model_validate(
                (
                    adapt_compact_output(
                        role="narration",
                        output=raw_narration,
                        player_input=state.get("player_input", ""),
                        context={},
                    )
                    if contract_mode == "compact"
                    else raw_narration
                ).model_dump()
            )
        except Exception as error:
            final_text = _fallback_narration_text(state)
            next_state = {
                **state,
                "final_output": final_text,
                "narration_plan": {
                    "final_text": final_text,
                    "canon_event_draft": None,
                    "memory_candidates": [],
                },
            }
            return _append_trace(
                next_state,
                "narrate_with_llm",
                {
                    "final_output": final_text,
                    "fallback": True,
                    "error": str(error),
                },
            )
        final_text = _ensure_required_dice_results(
            narration.final_text.strip(),
            state,
        )
        next_state = {
            **state,
            "narration_plan": {**narration.model_dump(), "final_text": final_text},
            "final_output": final_text,
        }
        return _append_trace(
            next_state,
            "narrate_with_llm",
            {
                "final_output": final_text,
                "advisor": _structured_call_trace(
                    role="narration",
                    prompt_version=NARRATION_PROMPT_VERSION,
                    schema_name=narration_schema.__name__,
                    elapsed_ms=elapsed_ms,
                    player_input=state.get("player_input", ""),
                    context=packet["context"],
                    schema_prompt=narration_schema_prompt,
                    attempts=attempts,
                ),
                "context_packet": packet["trace"],
                "structured_attempts": [
                    {key: value for key, value in attempt.items() if key != "raw_output"}
                    for attempt in attempts
                ],
            },
        )

    return narrate_with_llm


def _fallback_narration_text(state: GraphState) -> str:
    decision = str(state.get("turn_plan", {}).get("decision") or "free_action")
    brief = str(state.get("turn_plan", {}).get("narration_brief") or "").strip()
    player_input = state.get("player_input", "")
    prefers_chinese = _prefers_chinese(player_input)
    resolver_payload = _successful_resolver_payload(state)
    scenario_context = _fallback_visible_scenario_context(state)

    if decision == "risky_action" and resolver_payload:
        dice = resolver_payload.get("dice_result") or {}
        dice_text = (
            f"{dice.get('expression')} -> {dice.get('rolls')}"
            if dice.get("expression") and dice.get("rolls") is not None
            else str(resolver_payload.get("dice_expression") or "")
        )
        band = str(resolver_payload.get("band_label") or resolver_payload.get("band") or "")
        consequence = str(resolver_payload.get("consequence") or "").strip()
        if prefers_chinese:
            pieces = [
                f"你执行这个行动：{player_input[:100]}。",
                f"这个行动的判定已经落定：{dice_text}。",
                f"成功数 {resolver_payload.get('successes')}，结果档位：{band}。",
            ]
            if consequence:
                pieces.append(consequence)
            patch_summary = _resolver_patch_fallback_summary(
                resolver_payload,
                prefers_chinese=True,
            )
            if patch_summary:
                pieces.append(patch_summary)
            if scenario_context:
                pieces.append(scenario_context)
            pieces.append("你接下来怎么做？")
            return " ".join(piece for piece in pieces if piece).strip()
        pieces = [
            f"You carry out the action: {player_input[:100]}.",
            f"The action is resolved: {dice_text}.",
            f"Successes: {resolver_payload.get('successes')}; result band: {band}.",
        ]
        if consequence:
            pieces.append(consequence)
        patch_summary = _resolver_patch_fallback_summary(
            resolver_payload,
            prefers_chinese=False,
        )
        if patch_summary:
            pieces.append(patch_summary)
        if scenario_context:
            pieces.append(scenario_context)
        pieces.append("What do you do next?")
        return " ".join(piece for piece in pieces if piece).strip()

    if decision == "clarify":
        return _clarification_player_text(state)
    if decision == "boundary":
        return _boundary_fallback_text(state, brief)
    if decision == "gm_move" and scenario_context:
        return scenario_context
    context = _local_output_context(state, decision).strip()
    player_brief = _player_facing_brief_or_default(state, decision, brief)
    if player_brief and context:
        return f"{player_brief} {context}".strip()
    return player_brief or context or (
        "你先确认当前可见局势，再决定下一步怎么做。"
        if prefers_chinese
        else "You take in the visible situation before choosing your next move."
    )


def _failed_required_resolution_text(state: GraphState) -> str:
    if _prefers_chinese(state.get("player_input", "")):
        return (
            "你把行动推进到关键一刻，局势悬在那里，结果还没落定。"
            "我不会替这个有风险的行动宣布成功或失败；请补充角色资料、"
            "明确做法，或重新发起这个行动，我会从这一刻继续裁定。"
        )
    return (
        "Your action reaches the decisive moment, but the outcome is not resolved yet. "
        "I will not declare success or failure for a risky action without the rules result; "
        "add the missing character details, clarify the approach, or restate the action and I "
        "will adjudicate from that moment."
    )


def _player_facing_brief_or_default(
    state: GraphState,
    decision: str,
    brief: str,
) -> str:
    prefers_chinese = _prefers_chinese(state.get("player_input", ""))
    if not prefers_chinese:
        return brief
    if brief and _prefers_chinese(brief):
        return brief
    defaults = {
        "answer": "我只依据已经建立且对玩家可见的信息回答。",
        "free_action": "这个行动可以作为当前角色意图推进；我只呈现可见结果。",
        "risky_action": "这个行动有风险，结果会依据规则工具和已授权结果呈现。",
        "gm_move": "你暂缓行动观察局势，场景压力继续推进。",
    }
    return defaults.get(decision, "")


def _boundary_fallback_text(state: GraphState, brief: str) -> str:
    prefers_chinese = _prefers_chinese(state.get("player_input", ""))
    if brief and _prefers_chinese(brief) == prefers_chinese:
        return brief
    if prefers_chinese:
        return (
            "这个声明还不能直接成为既定事实。请把它描述成尝试、计划或问题，"
            "或先调查已经可见的信息，我会继续主持。"
        )
    return (
        "That declaration cannot become established fact directly. Frame it as an attempt, "
        "plan, or question, or investigate what is visibly established, and I will continue."
    )


def _clarification_player_text(state: GraphState) -> str:
    if _pending_rule_opportunities(state):
        return _pending_rule_opportunity_player_text(state)
    if _rules_advice_requires_player_clarification(state):
        return _rules_clarification_text(state)
    return _target_clarification_text(state)


def _pending_rule_opportunity_player_text(state: GraphState) -> str:
    opportunity = _pending_rule_opportunities(state)[0]
    prompt = _same_language_content(
        state,
        str(opportunity.get("prompt") or "").strip(),
    )
    effect = _same_language_content(
        state,
        str(opportunity.get("effect") or "").strip(),
    )
    if _prefers_chinese(state.get("player_input", "")):
        base = "在继续进行新的风险判定前，你还有一个待使用的规则机会。"
        prompt_text = f" {prompt}" if prompt else " 你可以向 GM 问一个关于当前局势的问题。"
        effect_text = f" {effect}" if effect else ""
        return f"{base}{prompt_text}{effect_text} 你要现在使用它，还是明确放弃后继续？"
    base = "Before another risky resolution, you still have a pending rules-granted opportunity."
    prompt_text = (
        f" {prompt}"
        if prompt
        else " You may ask the GM one question about the current situation."
    )
    effect_text = f" {effect}" if effect else ""
    return (
        f"{base}{prompt_text}{effect_text} Do you use it now, "
        "or explicitly waive it and continue?"
    )


def _same_language_content(state: GraphState, text: str) -> str:
    if not text:
        return ""
    if _prefers_chinese(text) == _prefers_chinese(state.get("player_input", "")):
        return text
    return ""


def _fallback_visible_scenario_context(state: GraphState) -> str:
    context = str(state.get("scenario_director", {}).get("player_visible_context") or "").strip()
    lowered = context.lower()
    if not context:
        return ""
    if "unavailable" in lowered or "no scenario patch proposed" in lowered:
        return ""
    return context


def _resolver_patch_fallback_summary(
    resolver_payload: dict[str, Any],
    *,
    prefers_chinese: bool,
) -> str:
    patches = resolver_payload.get("world_patches")
    if not isinstance(patches, list) or not patches:
        return ""
    clock_advanced = any(
        isinstance(patch, dict)
        and patch.get("op") in {"increment", "add"}
        and "clock" in [str(part) for part in patch.get("path", [])]
        for patch in patches
    )
    if clock_advanced:
        return "场景压力随之推进。" if prefers_chinese else "The scene pressure advances."
    return "结果带来了一个已授权的场景变化。" if prefers_chinese else (
        "The result authorizes a scene change."
    )


def _successful_resolver_payload(state: GraphState) -> dict[str, Any] | None:
    for result in state.get("tool_results", []):
        if result.get("tool_name") != "run_ruleset_resolver" or not result.get("ok"):
            continue
        payload = result.get("result")
        if isinstance(payload, dict):
            return payload
    return None


def _ensure_required_dice_results(final_text: str, state: GraphState) -> str:
    missing_lines: list[str] = []
    for result in state.get("tool_results", []):
        if result.get("tool_name") != "run_ruleset_resolver" or not result.get("ok"):
            continue
        payload = result.get("result") or {}
        if not isinstance(payload, dict):
            continue
        dice = payload.get("dice_result") or {}
        if not isinstance(dice, dict):
            continue
        expression = str(dice.get("expression") or "")
        rolls = dice.get("rolls")
        total = dice.get("total")
        if not expression or rolls is None or total is None:
            continue
        exact = f"{expression} -> {rolls}"
        if expression in final_text and str(total) in final_text:
            continue
        if _prefers_chinese(state.get("player_input", "")):
            missing_lines.append(f"骰点：{exact}，总计 {total}。")
        else:
            missing_lines.append(f"Roll: {exact}, total {total}.")
    if not missing_lines:
        return final_text
    separator = "\n\n" if final_text else ""
    return f"{final_text}{separator}{' '.join(missing_lines)}"


def _detect_player_roll_request(final_output: str) -> CriticFinding | None:
    patterns = [
        r"请.{0,12}(掷骰|投骰|扔骰|掷\s*\d*d\d+|投\s*\d*d\d+)",
        r"(你|玩家).{0,10}(需要|必须|要).{0,10}(掷骰|投骰|扔骰)",
        r"\bplease\s+roll\b",
        r"\broll\s+(?:the\s+)?dice\b",
        r"\broll\s+\d*d\d+\b",
    ]
    if not any(re.search(pattern, final_output, flags=re.IGNORECASE) for pattern in patterns):
        return None
    return CriticFinding(
        dimension="resolver_bypass",
        severity="high",
        message="Narration asks the player to roll dice manually; the runtime must roll via tools.",
        evidence=final_output,
    )


def _prefers_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _has_failed_required_resolution(state: GraphState) -> bool:
    if state.get("turn_plan", {}).get("decision") != "risky_action":
        return False
    return any(
        result.get("tool_name") == "run_ruleset_resolver" and not result.get("ok")
        for result in state.get("tool_results", [])
        if isinstance(result, dict)
    )


def critic_guardrail_locally(state: GraphState) -> GraphState:
    local_review_reason = _low_risk_local_review_reason(state)
    if local_review_reason:
        state = _with_advisor_skip_reason(state, "critic_guardrail", local_review_reason)
    findings: list[CriticFinding] = []
    final_output = state.get("final_output", "")
    if _has_failed_required_resolution(state):
        findings.append(
            CriticFinding(
                dimension="resolver_bypass",
                severity="high",
                message="Risky action reached output without successful resolver result.",
                evidence=final_output,
            )
        )
    roll_request_issue = _detect_player_roll_request(final_output)
    if roll_request_issue:
        findings.append(roll_request_issue)
    agency_issue = _detect_player_agency_violation(final_output)
    if agency_issue:
        findings.append(agency_issue)

    revised_text = None
    if roll_request_issue:
        revised_text = _roll_request_fallback_text(state)
    elif agency_issue:
        revised_text = _critic_fallback_text(state)
    report = CriticReport(
        ok=not findings,
        blocks_output=bool(agency_issue or roll_request_issue),
        findings=findings,
        revised_final_text=revised_text,
        reasoning_summary="Local critic fallback checked resolver bypass and player-roll requests.",
    )
    next_output = report.revised_final_text or final_output
    return _append_trace(
        {**state, "critic_report": report.model_dump(), "final_output": next_output},
        "critic_guardrail_locally",
        {
            "ok": report.ok,
            "blocks_output": report.blocks_output,
            "findings": [
                finding.dimension
                for finding in findings
            ],
            "advisor_skip_reasons": state.get("advisor_skip_reasons", {}),
        },
    )


def build_llm_critic_guardrail_node(model: BaseChatModel):
    def critic_guardrail_with_llm(state: GraphState) -> GraphState:
        if state.get("eval_smoke_mode"):
            local_state = critic_guardrail_locally(state)
            return _append_trace(
                local_state,
                "critic_guardrail_with_llm",
                {
                    "ok": local_state.get("critic_report", {}).get("ok"),
                    "blocks_output": local_state.get("critic_report", {}).get("blocks_output"),
                    "short_circuit": "eval_smoke_mode",
                },
            )
        if not state.get("sqlite_path"):
            return critic_guardrail_locally(state)
        if state.get("turn_plan", {}).get("decision") == "clarify":
            report = CriticReport(
                ok=True,
                blocks_output=False,
                findings=[],
                revised_final_text=None,
                reasoning_summary="Low-impact target clarification bypassed critic repair.",
            )
            return _append_trace(
                {**state, "critic_report": {**report.model_dump(), "repaired": False}},
                "critic_guardrail_with_llm",
                {
                    "ok": True,
                    "blocks_output": False,
                    "repaired": False,
                    "short_circuit": "clarification",
                    "findings": [],
                },
            )
        try:
            packet = _context_packet(
                state,
                "critic_guardrail",
                advisor_contract="CriticReport",
            )
            result = invoke_advisor(
                model=model,
                role="critic_guardrail",
                player_input=state.get("player_input", "") if packet["mode"] == "enforced" else "",
                context=packet["context"],
                sqlite_path=state.get("sqlite_path"),
                turn_id=state.get("turn_id"),
                contract_mode=_advisor_contract_mode(state),
            )
            report = CriticReport.model_validate(result.output.model_dump())
            advisor_trace = result.trace_metadata
            context_packet_trace = packet["trace"]
            structured_attempts = [
                {key: value for key, value in attempt.items() if key != "raw_output"}
                for attempt in result.attempts
            ]
        except Exception as error:
            report = CriticReport(
                ok=True,
                blocks_output=False,
                findings=[],
                revised_final_text=None,
                reasoning_summary=f"Critic advisor failed safely: {error}",
            )
            advisor_trace = {"fallback": "true", "error": str(error)}
            context_packet_trace = {}
            structured_attempts = []

        final_output_before_critic = state.get("final_output", "")
        roll_request_issue = _detect_player_roll_request(final_output_before_critic)
        agency_issue = _detect_player_agency_violation(final_output_before_critic)
        extra_findings = [
            finding for finding in [roll_request_issue, agency_issue] if finding is not None
        ]
        if extra_findings:
            findings = [*report.findings, *extra_findings]
            report = report.model_copy(update={"ok": False, "findings": findings})
        if _has_validated_scenario_progress(state):
            filtered_findings = [
                finding for finding in report.findings if finding.dimension != "clarification"
            ]
            if len(filtered_findings) != len(report.findings):
                report = report.model_copy(
                    update={
                        "findings": filtered_findings,
                        "ok": not filtered_findings,
                    }
                )
        forced_block = _critic_requires_repair(report, state)
        if forced_block and not report.blocks_output:
            report = report.model_copy(update={"ok": False, "blocks_output": True})
        elif (
            not forced_block
            and report.blocks_output
            and _has_validated_scenario_progress(state)
        ):
            report = report.model_copy(update={"blocks_output": False, "revised_final_text": None})
        elif not forced_block and not report.blocks_output:
            report = report.model_copy(update={"ok": True, "revised_final_text": None})

        final_output = state.get("final_output", "")
        repaired = False
        force_deterministic_fallback = any(
            finding.dimension in {"player_agency", "clarification"} for finding in report.findings
        )
        force_roll_fallback = roll_request_issue is not None
        if (
            report.blocks_output
            and report.revised_final_text
            and not force_deterministic_fallback
            and not force_roll_fallback
        ):
            final_output = _ensure_required_dice_results(report.revised_final_text.strip(), state)
            repaired = final_output != state.get("final_output", "")
            report = report.model_copy(update={"revised_final_text": final_output})
        elif report.blocks_output:
            if force_roll_fallback:
                final_output = _roll_request_fallback_text(state)
            elif any(finding.dimension == "clarification" for finding in report.findings):
                final_output = _clarification_fallback_text(state)
            else:
                final_output = _critic_fallback_text(state)
            repaired = True

        next_state = {
            **state,
            "critic_report": {
                **report.model_dump(),
                "repaired": repaired,
                "forced_block": forced_block,
                "deterministic_fallback": force_deterministic_fallback and report.blocks_output,
            },
            "final_output": final_output,
        }
        if repaired and "narration_plan" in next_state:
            next_state["narration_plan"] = {
                **dict(next_state.get("narration_plan", {})),
                "final_text": final_output,
            }
        return _append_trace(
            next_state,
            "critic_guardrail_with_llm",
            {
                "ok": report.ok,
                "blocks_output": report.blocks_output,
                "forced_block": forced_block,
                "repaired": repaired,
                "findings": [finding.dimension for finding in report.findings],
                "advisor": advisor_trace,
                "context_packet": context_packet_trace,
                "structured_attempts": structured_attempts,
            },
        )

    return critic_guardrail_with_llm


def _critic_requires_repair(report: CriticReport, state: GraphState | None = None) -> bool:
    always_block = {
        "hidden_leak",
        "resolver_bypass",
        "canon_contradiction",
        "player_agency",
        "clarification",
    }
    severity_block = {"narration_quality", "pacing"}
    medium_severity_block = {"unsupported_fact"}
    return any(
        finding.dimension in always_block
        or (
            finding.dimension in severity_block
            and finding.severity in {"high", "critical"}
        )
        or (
            finding.dimension in medium_severity_block
            and finding.severity in {"medium", "high", "critical"}
        )
        for finding in report.findings
    )


def _has_validated_scenario_progress(state: GraphState) -> bool:
    scenario_director = state.get("scenario_director", {})
    if scenario_director.get("validated_patches"):
        return True
    for result in state.get("tool_results", []):
        if result.get("tool_name") != "scenario_director" or not result.get("ok"):
            continue
        payload = result.get("result")
        if isinstance(payload, dict) and payload.get("world_patches"):
            return True
    return False


def _detect_player_agency_violation(final_output: str) -> CriticFinding | None:
    lowered = final_output.lower()
    markers = [
        "不由自主",
        "忍不住",
        "你想要",
        "你想跟",
        "你决定",
        "要么",
        "you want",
        "you decide",
        "you feel compelled",
        "you cannot help",
        "either ",
    ]
    if not any(marker in lowered or marker in final_output for marker in markers):
        return None
    return CriticFinding(
        dimension="player_agency",
        severity="medium",
        message="Narration appears to dictate the player character's internal state or action.",
        evidence=final_output,
    )


def _critic_fallback_text(state: GraphState) -> str:
    visible_context = _visible_scene_context_for_fallback(state)
    if _prefers_chinese(state.get("player_input", "")):
        if visible_context:
            return f"你把注意力放在能确认的现场迹象上：{visible_context} 你接下来怎么做？"
        return (
            "你把注意力放在能确认的现场迹象上。新的发现、代价或场景变化会先通过"
            "规则、场景补丁或明确揭示来建立。你接下来怎么做？"
        )
    if visible_context:
        return (
            "You focus on the visible, established details: "
            f"{visible_context} What do you do next?"
        )
    return (
        "You focus on the visible, established details. Any new discovery, cost, or scene change "
        "will be established through rules, scenario patches, or an explicit reveal. What do you "
        "do next?"
    )


def _roll_request_fallback_text(state: GraphState) -> str:
    resolver_payload = _successful_resolver_payload(state)
    if resolver_payload:
        return _fallback_narration_text(state)
    if _prefers_chinese(state.get("player_input", "")):
        return (
            "这个行动不需要你手动掷骰；需要判定时我会调用规则工具完成。"
            "请确认你要执行的具体动作、目标和准备，我会继续裁定。"
        )
    return (
        "You do not need to roll manually; when a check is needed, I will use the rules tool. "
        "Confirm the concrete action, target, and preparation, and I will continue adjudicating."
    )


def _clarification_fallback_text(state: GraphState) -> str:
    player_input = state.get("player_input", "")
    if _prefers_chinese(player_input):
        return (
            f"你说的“{player_input}”还需要明确优先处理的目标或第一步。"
            "请说明你现在最先要询问、联系、检查或操作的具体对象，我再继续主持。"
        )
    return (
        f"The priority or first target in '{player_input}' is not clear yet. "
        "Name the specific person, system, object, or first step you want to handle, "
        "and I will continue from there."
    )


def _visible_scene_context_for_fallback(state: GraphState) -> str:
    pieces: list[tuple[str, int]] = []
    scenario_context = state.get("scenario_director", {}).get("player_visible_context")
    if scenario_context:
        pieces.append((str(scenario_context).strip(), 1))
    world = state.get("world_projection", {})
    revealed = world.get("revealed_facts")
    if isinstance(revealed, list):
        pieces.extend((str(item).strip(), 2) for item in revealed[-1:] if str(item).strip())
    scene = world.get("scene")
    if isinstance(scene, dict) and scene.get("public_summary"):
        pieces.append((str(scene["public_summary"]).strip(), 2))
    deduped_sentences: list[str] = []
    seen: set[str] = set()
    signatures: list[set[str]] = []
    for piece, budget in pieces:
        accepted_from_piece = 0
        for sentence in _split_context_sentences(piece):
            if accepted_from_piece >= budget:
                break
            if not sentence or sentence in seen:
                continue
            signature = _context_signature(sentence)
            if _is_redundant_context_sentence(signature, signatures):
                continue
            deduped_sentences.append(sentence)
            accepted_from_piece += 1
            seen.add(sentence)
            if signature:
                signatures.append(signature)
            if len(deduped_sentences) >= 4:
                return _join_context_fragments(deduped_sentences)[:900]
    return _join_context_fragments(deduped_sentences)[:900]


def _split_context_sentences(piece: str) -> list[str]:
    text = " ".join(piece.strip().split())
    if not text:
        return []
    sentences = _raw_context_sentences(text)
    fragments: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) >= 16 and re.search(r"[，,；;]", sentence):
            fragments.extend(_split_context_clauses(sentence))
        else:
            fragments.append(sentence)
    return fragments


def _raw_context_sentences(piece: str) -> list[str]:
    text = " ".join(piece.strip().split())
    if not text:
        return []
    return [
        sentence.strip()
        for sentence in re.findall(r"[^。！？!?.]+[。！？!?.]?", text)
        if sentence.strip()
    ]


def _split_context_clauses(sentence: str) -> list[str]:
    clauses = re.findall(r"[^，,；;]+[，,；;]?", sentence)
    cleaned = [clause.strip().rstrip("，,；;") for clause in clauses if clause.strip()]
    return cleaned or [sentence]


def _context_signature(sentence: str) -> set[str]:
    normalized = re.sub(r"\s+", "", sentence.lower())
    cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    cjk_bigrams = {
        cjk_text[index : index + 2]
        for index in range(max(0, len(cjk_text) - 1))
    }
    latin_tokens = set(re.findall(r"[a-z0-9]{3,}", normalized))
    return cjk_bigrams | latin_tokens


def _is_redundant_context_sentence(signature: set[str], seen: list[set[str]]) -> bool:
    if not signature:
        return False
    for existing in seen:
        overlap = len(signature & existing)
        smaller = max(1, min(len(signature), len(existing)))
        if overlap >= 5 and overlap / smaller >= 0.32:
            return True
        if overlap >= 3 and smaller <= 8 and overlap / smaller >= 0.45:
            return True
    return False


def curate_memory_locally(state: GraphState) -> GraphState:
    candidates: list[MemoryCandidate] = []
    lower = state.get("player_input", "").lower()
    if "以后" in state.get("player_input", "") or "prefer" in lower:
        candidates.append(
            MemoryCandidate(
                kind="player_preference",
                text=state.get("player_input", ""),
                scope="session",
                confidence=0.72,
                metadata={"visibility": "public", "source": "local_preference_detector"},
            )
        )
    curation = MemoryCurationDecision(
        canon_event_draft=None,
        memory_candidates=candidates,
        contradictions=[],
        should_write=bool(candidates),
    )
    local_review_reason = _low_risk_local_review_reason(state)
    if local_review_reason:
        state = _with_advisor_skip_reason(
            state,
            "memory_curator",
            "low_risk_local_curation" if candidates else "no_durable_event",
        )
    return _append_trace(
        {**state, "memory_curation": curation.model_dump()},
        "curate_memory_locally",
        {
            "memory_candidates": [candidate.kind for candidate in candidates],
            "advisor_skip_reasons": state.get("advisor_skip_reasons", {}),
        },
    )


def build_llm_memory_curator_node(model: BaseChatModel):
    def curate_memory_with_llm(state: GraphState) -> GraphState:
        if state.get("eval_smoke_mode"):
            local_state = curate_memory_locally(state)
            return _append_trace(
                local_state,
                "curate_memory_with_llm",
                {
                    "should_write": local_state.get("memory_curation", {}).get("should_write"),
                    "short_circuit": "eval_smoke_mode",
                },
            )
        if state.get("turn_plan", {}).get("decision") == "clarify":
            curation = MemoryCurationDecision(
                canon_event_draft=None,
                memory_candidates=[],
                contradictions=[],
                should_write=False,
            )
            return _append_trace(
                {**state, "memory_curation": curation.model_dump()},
                "curate_memory_with_llm",
                {
                    "should_write": False,
                    "memory_candidates": [],
                    "contradictions": [],
                    "short_circuit": "clarification",
                },
            )
        if not state.get("sqlite_path"):
            return curate_memory_locally(state)
        try:
            packet = _context_packet(
                state,
                "memory_curator",
                advisor_contract="MemoryCurationDecision",
            )
            result = invoke_advisor(
                model=model,
                role="memory_curator",
                player_input=state.get("player_input", "") if packet["mode"] == "enforced" else "",
                context=packet["context"],
                sqlite_path=state.get("sqlite_path"),
                turn_id=state.get("turn_id"),
                contract_mode=_advisor_contract_mode(state),
            )
            curation = MemoryCurationDecision.model_validate(result.output.model_dump())
            advisor_trace = result.trace_metadata
            context_packet_trace = packet["trace"]
            structured_attempts = [
                {key: value for key, value in attempt.items() if key != "raw_output"}
                for attempt in result.attempts
            ]
        except Exception as error:
            curation = MemoryCurationDecision(
                canon_event_draft=None,
                memory_candidates=[],
                contradictions=[f"Memory curator failed safely: {error}"],
                should_write=False,
            )
            advisor_trace = {"fallback": "true", "error": str(error)}
            context_packet_trace = {}
            structured_attempts = []
        return _append_trace(
            {**state, "memory_curation": curation.model_dump()},
            "curate_memory_with_llm",
            {
                "should_write": curation.should_write,
                "memory_candidates": [candidate.kind for candidate in curation.memory_candidates],
                "contradictions": curation.contradictions,
                "advisor": advisor_trace,
                "context_packet": context_packet_trace,
                "structured_attempts": structured_attempts,
            },
        )

    return curate_memory_with_llm


def build_parallel_review_and_memory_node(
    critic_node,
    memory_node,
):
    def review_and_curate_parallel(state: GraphState) -> GraphState:
        branch_state: GraphState = {**state, "trace_events": []}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="trpg-review") as executor:
            critic_future = executor.submit(critic_node, branch_state)
            memory_future = executor.submit(memory_node, branch_state)
            critic_state = critic_future.result()
            memory_state = memory_future.result()

        memory_curation = memory_state.get("memory_curation", {})
        discarded_memory = False
        if critic_state.get("final_output") != state.get("final_output"):
            memory_curation = MemoryCurationDecision(
                canon_event_draft=None,
                memory_candidates=[],
                contradictions=["Discarded because critic repaired the player-facing output."],
                should_write=False,
            ).model_dump()
            discarded_memory = True

        next_state = {
            **critic_state,
            "memory_curation": memory_curation,
            "trace_events": [
                *state.get("trace_events", []),
                *critic_state.get("trace_events", []),
                *memory_state.get("trace_events", []),
            ],
        }
        return _append_trace(
            next_state,
            "review_and_curate_parallel",
            {
                "branches": ["critic_guardrail", "memory_curation"],
                "discarded_memory": discarded_memory,
            },
        )

    return review_and_curate_parallel


def persist_memory_curation(state: GraphState) -> GraphState:
    sqlite_path = state.get("sqlite_path")
    curation = MemoryCurationDecision.model_validate(
        state.get("memory_curation") or MemoryCurationDecision().model_dump()
    )
    if state.get("turn_plan", {}).get("decision") == "clarify":
        return _append_trace(
            state,
            "persist_memory_curation",
            {"persisted": 0, "skipped": "low_impact_target_disambiguation"},
        )
    if not sqlite_path:
        return _append_trace(state, "persist_memory_curation", {"persisted": 0})
    critic_report = state.get("critic_report")
    if (
        critic_report
        and _critic_requires_repair(CriticReport.model_validate(critic_report), state)
        and not critic_report.get("repaired")
    ):
        return _append_trace(
            state,
            "persist_memory_curation",
            {"persisted": 0, "skipped": "critic_blocking_findings"},
        )

    store = SqliteStore(Path(sqlite_path))
    store.migrate()
    session_id = state.get("session_id", "default")
    turn_id = state.get("turn_id", "turn")
    persisted = 0
    skip_curated_writes = False
    filtered_conflicts = 0
    if curation.should_write and not curation.contradictions and curation.canon_event_draft:
        inserted = store.insert_canon_event(
            event_id=f"{turn_id}:curated-canon",
            session_id=session_id,
            turn_id=turn_id,
            event_type="curated_canon",
            payload=curation.canon_event_draft,
        )
        persisted += int(inserted)

    if curation.should_write and not skip_curated_writes:
        for index, candidate in enumerate(curation.memory_candidates):
            if _memory_candidate_conflicts(candidate, curation.contradictions):
                filtered_conflicts += 1
                continue
            if candidate.confidence < 0.5 or not candidate.text.strip():
                continue
            metadata = _memory_metadata(candidate, state)
            store.upsert_memory(
                memory_id=f"{turn_id}:memory:{index}",
                scope=_memory_scope(candidate.scope, session_id),
                kind=candidate.kind,
                text=candidate.text.strip(),
                metadata=metadata,
            )
            persisted += 1

    persisted += _persist_episodic_summary_if_needed(store, state)
    return _append_trace(
        state,
        "persist_memory_curation",
        {
            "persisted": persisted,
            "candidate_count": len(curation.memory_candidates),
            **(
                {"filtered_conflicting_candidates": filtered_conflicts}
                if filtered_conflicts
                else {}
            ),
            **(
                {"skipped": "memory_curator_contradictions"}
                if curation.contradictions and persisted == 0
                else {}
            ),
        },
    )


def _memory_candidate_conflicts(candidate: MemoryCandidate, contradictions: list[str]) -> bool:
    if not contradictions:
        return False
    candidate_norm = re.sub(r"\s+", " ", candidate.text.lower()).strip()
    candidate_tokens = set(re.findall(r"[a-z0-9]{4,}", candidate_norm))
    candidate_signature = _context_signature(candidate.text)
    if not candidate_signature:
        return True
    for contradiction in contradictions:
        contradiction_norm = re.sub(r"\s+", " ", contradiction.lower()).strip()
        if candidate_norm and (
            candidate_norm in contradiction_norm or contradiction_norm in candidate_norm
        ):
            return True
        contradiction_tokens = set(re.findall(r"[a-z0-9]{4,}", contradiction_norm))
        token_overlap = candidate_tokens & contradiction_tokens
        if token_overlap and (
            "unsupported" in candidate_norm
            or "unsupported" in contradiction_norm
            or "not supported" in contradiction_norm
        ):
            return True
        if len(token_overlap) >= 2:
            return True
        contradiction_signature = _context_signature(contradiction)
        if _is_redundant_context_sentence(candidate_signature, [contradiction_signature]):
            return True
    return False


def _memory_metadata(candidate: MemoryCandidate, state: GraphState) -> dict[str, Any]:
    visibility = candidate.metadata.get("visibility")
    if not visibility:
        visibility = "gm_only" if candidate.kind == "procedural_note" else "public"
    return {
        **candidate.metadata,
        "visibility": visibility,
        "turn_id": state.get("turn_id"),
        "ruleset_id": state.get("ruleset_id"),
        "scenario_id": state.get("scenario_id"),
    }


def _memory_scope(scope: str, session_id: str) -> str:
    if scope == "session":
        return session_id
    return f"{scope}:default"


def _persist_episodic_summary_if_needed(store: SqliteStore, state: GraphState) -> int:
    session_id = state.get("session_id", "default")
    turn_count = len(store.list_turns(session_id)) + 1
    if turn_count < 10 or turn_count % 10 != 0:
        return 0
    recent = store.list_canon_events(session_id)[-5:]
    if not recent:
        return 0
    summary = " / ".join(str(event.get("payload", {}))[:160] for event in recent)
    store.upsert_memory(
        memory_id=f"{session_id}:episodic-summary:{turn_count // 10}",
        scope=session_id,
        kind="episodic_summary",
        text=summary,
        metadata={
            "visibility": "public",
            "turn_count": turn_count,
            "source": "episodic_summary_policy",
        },
    )
    return 1


def ensure_resolution_tools(state: GraphState) -> GraphState:
    plan = state.get("turn_plan", {})
    if plan.get("decision") in {"clarify", "boundary"}:
        return _append_trace(state, "ensure_resolution_tools", {"changed": False})
    routing = state.get("routing_decision", {})
    rules_advice = state.get("rules_advice", {})
    rules_requires_resolution = rules_advice.get("requires_resolution")
    rules_risk = str(rules_advice.get("risk") or "").lower()
    advisor_requires_risky_resolution = bool(
        rules_requires_resolution is True and rules_risk in {"risky_uncertain", "high"}
    )
    risky_by_route = bool(
        (
            routing.get("needs_rules_resolution")
            or routing.get("route") == "risky_action"
            or advisor_requires_risky_resolution
        )
        and rules_requires_resolution is not False
    )
    risky_by_plan = plan.get("decision") == "risky_action"
    if (
        not (risky_by_route or risky_by_plan)
        or not state.get("ruleset_id")
    ):
        return _append_trace(state, "ensure_resolution_tools", {"changed": False})
    if _pending_rule_opportunity_blocks_resolution(state):
        plan = _pending_rule_opportunity_clarification_turn_plan(state)
        next_state = {
            **state,
            "intent": plan.intent.model_dump(),
            "authority_result": plan.authority.model_dump(),
            "turn_plan": plan.model_dump(),
            "tool_requests": [],
        }
        return _append_trace(
            next_state,
            "ensure_resolution_tools",
            {
                "changed": True,
                "blocked_by": "pending_rule_opportunity",
                "tool_requests": [],
            },
        )

    next_requests: list[dict[str, Any]] = []
    has_resolver = False
    rejected_tool_requests: list[dict[str, Any]] = []
    advised_approach = _approach_for_resolver_request(state, rules_advice.get("approach_id"))
    advised_risk = str(rules_advice.get("risk") or "risky_uncertain")
    advised_plugin_args = _rules_plugin_arguments(rules_advice)
    for raw_request in state.get("tool_requests", []):
        request = ToolRequest.model_validate(raw_request)
        if request.tool_name == "run_ruleset_resolver":
            has_resolver = True
            arguments = dict(request.arguments)
            arguments["approach"] = _approach_for_resolver_request(
                state,
                arguments.get("approach") or advised_approach,
            )
            if not arguments.get("risk") and advised_risk:
                arguments["risk"] = advised_risk
            for key, value in advised_plugin_args.items():
                arguments.setdefault(key, value)
            arguments.pop("requested_roll", None)
            arguments.update(_protected_resolver_arguments(state))
            request = request.model_copy(update={"arguments": arguments})
            next_requests.append(request.model_dump())
        elif request.tool_name == "roll_dice":
            rejected_tool_requests.append(
                {
                    "tool_name": "roll_dice",
                    "reason": "manual_roll_command_only",
                }
            )
        elif request.tool_name == "apply_world_patch":
            continue
        else:
            next_requests.append(request.model_dump())

    if not has_resolver:
        resolver_request = _resolver_tool_request(
            state,
            approach=advised_approach,
            **advised_plugin_args,
            risk=advised_risk,
        )
        next_requests.append(resolver_request.model_dump())

    next_plan = dict(plan)
    next_plan["tool_requests"] = next_requests
    if risky_by_route and next_plan.get("decision") != "risky_action":
        next_plan["decision"] = "risky_action"
        next_plan["narration_brief"] = (
            "This action is risky and uncertain; the loaded rules extension must adjudicate it."
        )

    return _append_trace(
        {**state, "turn_plan": next_plan, "tool_requests": next_requests},
        "ensure_resolution_tools",
        {
            "changed": next_requests != state.get("tool_requests", []),
            "tool_requests": [
                request.get("tool_name")
                for request in next_requests
                if isinstance(request, dict)
            ],
            "rejected_tool_requests": rejected_tool_requests,
            "risky_by_route": risky_by_route,
            "used_rules_advice": bool(rules_advice),
        },
    )


def execute_deterministic_tools(state: GraphState) -> GraphState:
    results: list[dict[str, Any]] = []
    for raw_request in state.get("tool_requests", []):
        request = ToolRequest.model_validate(raw_request)
        try:
            result = _execute_tool_request(request, state)
            tool_result = ToolResult(
                tool_name=request.tool_name,
                ok=True,
                result=result,
                request=request,
            )
        except Exception as error:
            tool_result = ToolResult(
                tool_name=request.tool_name,
                ok=False,
                error=str(error),
                request=request,
            )
        results.append(tool_result.model_dump())

    return _append_trace(
        {**state, "tool_results": results},
        "execute_deterministic_tools",
        {"tool_results": results},
    )


def _scenario_director_needed(state: GraphState) -> bool:
    return _scenario_director_skip_reason(state) is None


def _scenario_director_skip_reason(state: GraphState) -> str | None:
    if not state.get("sqlite_path"):
        return "no_durable_runtime"
    if not state.get("scenario_id"):
        return "no_loaded_scenario"
    if _has_failed_required_resolution(state):
        return "failed_required_resolution"
    decision = state.get("turn_plan", {}).get("decision")
    if decision in {"clarify", "boundary"}:
        return f"{decision}_turn"
    routing = state.get("routing_decision", {})
    if routing:
        if routing.get("route") == "clarify":
            return "clarification_route"
        if decision == "answer":
            if not bool(routing.get("needs_scenario_director", False)):
                return "routing_not_needed_for_answer"
            return None
        if not bool(routing.get("needs_scenario_director", True)):
            return "routing_not_needed"
        return None
    if decision not in {
        "free_action",
        "risky_action",
        "gm_move",
    }:
        return "decision_not_scene_affecting"
    return None


def direct_scenario_locally(state: GraphState) -> GraphState:
    decision = ScenarioDirectorDecision(
        decision="no_change",
        proposed_patches=[],
        player_visible_context="No local scenario change proposed.",
        gm_only_reason="Local fallback leaves scenario changes to deterministic tool results.",
        citations=[],
    )
    return _append_trace(
        {**state, "scenario_director": decision.model_dump()},
        "direct_scenario_locally",
        {"decision": decision.decision, "validated_patches": [], "rejected_patches": []},
    )


def _scenario_surface_selector_eligible(state: GraphState) -> bool:
    if not _conditional_advisors_enabled(state):
        return False
    if _scenario_director_skip_reason(state):
        return False
    if state.get("tool_results"):
        return False
    if _has_successful_resolver_result(state) or _has_failed_required_resolution(state):
        return False
    if _pending_rule_opportunities(state):
        return False
    routing = state.get("routing_decision", {})
    route = str(routing.get("route") or "")
    if route not in {"answer", "free_action"}:
        return False
    if routing.get("needs_rules_resolution") or routing.get("needs_rules_review"):
        return False
    decision = str(state.get("turn_plan", {}).get("decision") or "")
    if decision not in {"answer", "free_action"}:
        return False
    return bool(_available_visible_surfaces(state))


def _available_visible_surfaces(state: GraphState) -> list[dict[str, Any]]:
    scene = state.get("world_projection", {}).get("scene", {})
    if not isinstance(scene, dict):
        return []
    used_texts = {
        str(item).strip()
        for item in [
            *state.get("world_projection", {}).get("revealed_facts", []),
            *state.get("world_projection", {}).get("known_clues", []),
        ]
        if str(item).strip()
    }
    surfaces: list[dict[str, Any]] = []
    for raw_surface in scene.get("visible_surfaces", []):
        if not isinstance(raw_surface, dict):
            continue
        surface_id = str(raw_surface.get("id") or "").strip()
        text = str(raw_surface.get("text") or "").strip()
        if not surface_id or not text:
            continue
        one_shot = bool(raw_surface.get("one_shot", True))
        if one_shot and text in used_texts:
            continue
        surfaces.append(
            {
                "id": surface_id,
                "text": text,
                "tags": [str(tag) for tag in raw_surface.get("tags", [])[:8]],
                "one_shot": one_shot,
            }
        )
    return surfaces


def build_llm_scenario_surface_selector_node(model: BaseChatModel):
    def select_scenario_surface_with_llm(state: GraphState) -> GraphState:
        candidates = _available_visible_surfaces(state)
        if not candidates:
            return _append_trace(
                {
                    **state,
                    "scenario_surface_selector": {
                        "fallback_to_full_director": True,
                        "fallback_reason": "no_visible_surface_candidates",
                    },
                },
                "select_scenario_surface_with_llm",
                {
                    "fallback_to_full_director": True,
                    "fallback_reason": "no_visible_surface_candidates",
                },
            )
        try:
            packet = _context_packet(
                state,
                "scenario_surface_selector",
                surface_candidates=candidates,
                advisor_contract="ScenarioSurfaceSelectorDecision",
            )
            result = invoke_advisor(
                model=model,
                role="scenario_surface_selector",
                player_input=state.get("player_input", ""),
                context=packet["context"],
                sqlite_path=state.get("sqlite_path"),
                turn_id=state.get("turn_id"),
                contract_mode=_advisor_contract_mode(state),
            )
            selector = ScenarioSurfaceSelectorDecision.model_validate(
                result.output.model_dump()
            )
            advisor_trace = result.trace_metadata
            context_packet_trace = packet["trace"]
            structured_attempts = [
                {key: value for key, value in attempt.items() if key != "raw_output"}
                for attempt in result.attempts
            ]
        except Exception as error:
            selector = ScenarioSurfaceSelectorDecision(
                decision="select",
                surface_id=str(candidates[0]["id"]),
                fallback_to_full_director=False,
                reason=(
                    "Surface selector failed; deterministic recovery selected the first "
                    "package-authorized visible surface."
                ),
                citations=[],
            )
            advisor_trace = {"selector_error": str(error), "deterministic_recovery": "true"}
            context_packet_trace = {}
            structured_attempts = []

        if _selector_empty_output_fallback(selector):
            selector = ScenarioSurfaceSelectorDecision(
                decision="select",
                surface_id=str(candidates[0]["id"]),
                fallback_to_full_director=False,
                reason=(
                    "Surface selector returned an empty fallback; deterministic recovery "
                    "selected the first package-authorized visible surface."
                ),
                citations=[],
            )
            advisor_trace = {
                **advisor_trace,
                "deterministic_recovery": "true",
                "selector_recovery_reason": "empty_selector_output",
            }

        selected = _visible_surface_by_id(candidates, selector.surface_id)
        fallback_reason = _surface_selector_fallback_reason(selector, selected)
        selector_state = selector.model_dump() | {
            "fallback_to_full_director": bool(fallback_reason),
            "fallback_reason": fallback_reason,
        }
        trace_payload = {
            "decision": selector.decision,
            "surface_id": selector.surface_id,
            "fallback_to_full_director": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "advisor": advisor_trace,
            "context_packet": context_packet_trace,
            "structured_attempts": structured_attempts,
        }
        if fallback_reason or not selected:
            return _append_trace(
                {**state, "scenario_surface_selector": selector_state},
                "select_scenario_surface_with_llm",
                trace_payload,
            )

        next_state = _with_advisor_skip_reason(
            {**state, "scenario_surface_selector": selector_state},
            "scenario_director",
            "conditional_surface_selector",
        )
        decision = _scenario_decision_from_visible_surface(
            surface=selected,
            selector=selector,
            state=next_state,
        )
        return _apply_scenario_director_decision(
            state=next_state,
            decision=decision,
            node="select_scenario_surface_with_llm",
            trace_payload=trace_payload
            | {
                "advisor_skip_reasons": next_state.get("advisor_skip_reasons", {}),
                "selected_surface": {
                    "id": selected["id"],
                    "one_shot": selected["one_shot"],
                    "tags": selected["tags"],
                },
            },
        )

    return select_scenario_surface_with_llm


def _visible_surface_by_id(
    candidates: list[dict[str, Any]],
    surface_id: str | None,
) -> dict[str, Any] | None:
    if not surface_id:
        return None
    for surface in candidates:
        if surface.get("id") == surface_id:
            return surface
    return None


def _surface_selector_fallback_reason(
    selector: ScenarioSurfaceSelectorDecision,
    selected: dict[str, Any] | None,
) -> str | None:
    if selector.fallback_to_full_director or selector.decision == "fallback":
        return "selector_requested_full_director"
    if not selected:
        return "invalid_or_missing_surface_id"
    return None


def _selector_empty_output_fallback(selector: ScenarioSurfaceSelectorDecision) -> bool:
    if selector.decision != "fallback" or selector.surface_id:
        return False
    reason = selector.reason.strip().lower()
    return reason in {
        "",
        "no output provided",
        "no valid selection",
        "no valid selection possible",
        "no valid surface selection possible",
        "no_output_provided",
    }


def _scenario_decision_from_visible_surface(
    *,
    surface: dict[str, Any],
    selector: ScenarioSurfaceSelectorDecision,
    state: GraphState,
) -> ScenarioDirectorDecision:
    text = str(surface.get("text") or "").strip()
    one_shot = bool(surface.get("one_shot", True))
    patches = (
        [{"op": "append", "path": ["revealed_facts"], "value": text}]
        if one_shot
        else []
    )
    citation = f"{state.get('scenario_id')}:visible_surface:{surface.get('id')}"
    return ScenarioDirectorDecision(
        decision="reveal" if patches else "no_change",
        proposed_patches=patches,
        player_visible_context=text,
        gm_only_reason=(
            "A low-risk conditional selector chose an authorized player-visible "
            f"surface. Selector reason: {selector.reason}"
        ),
        citations=[citation, *selector.citations],
    )


def _apply_scenario_director_decision(
    *,
    state: GraphState,
    decision: ScenarioDirectorDecision,
    node: str,
    trace_payload: dict[str, Any],
) -> GraphState:
    validation = validate_scenario_director_decision(
        decision=decision,
        state=state,
        content_dir=Path(state.get("content_dir") or Path.cwd() / "content"),
        scenario_id=state.get("scenario_id"),
    )
    validation = _limit_progressive_scenario_reveals(validation, state)
    player_visible_context = _scoped_scenario_visible_context(decision, validation, state)
    next_tool_results = list(state.get("tool_results", []))
    if validation.patches:
        next_tool_results.append(
            ToolResult(
                tool_name="scenario_director",
                ok=True,
                result={
                    "decision": decision.decision,
                    "player_visible_context": player_visible_context,
                    "world_patches": [patch.model_dump() for patch in validation.patches],
                },
            ).model_dump()
        )

    next_state = {
        **state,
        "tool_results": next_tool_results,
        "scenario_director": {
            **decision.model_dump(),
            "player_visible_context": player_visible_context,
            "validated_patches": [patch.model_dump() for patch in validation.patches],
            "rejected_patches": validation.rejected,
        },
    }
    return _append_trace(
        next_state,
        node,
        {
            **trace_payload,
            "decision": decision.decision,
            "validated_patches": [patch.model_dump() for patch in validation.patches],
            "rejected_patches": validation.rejected,
        },
    )


def build_llm_scenario_director_node(model: BaseChatModel):
    def direct_scenario_with_llm(state: GraphState) -> GraphState:
        skip_reason = _scenario_director_skip_reason(state)
        if skip_reason:
            decision = ScenarioDirectorDecision(
                decision="no_change",
                proposed_patches=[],
                player_visible_context="Scenario director not needed for this route.",
                gm_only_reason="Routing did not request scenario intelligence.",
                citations=[],
            )
            next_state = _with_advisor_skip_reason(
                {**state, "scenario_director": decision.model_dump()},
                "scenario_director",
                skip_reason,
            )
            return _append_trace(
                next_state,
                "direct_scenario_with_llm",
                {
                    "decision": "no_change",
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "advisor_skip_reasons": next_state.get("advisor_skip_reasons", {}),
                },
            )

        pre_advice = _single_turn_scenario_advice_for_director(state)
        if pre_advice:
            decision = pre_advice
            advisor_trace = {"source": "single_turn_advisor", "cached": "true"}
            context_packet_trace = {"source": "single_turn_advisor", "cached": "true"}
            structured_attempts = [{"phase": "single_turn_scenario_advice"}]
        else:
            try:
                packet = _context_packet(
                    state,
                    "scenario_director",
                    advisor_contract="ScenarioDirectorDecision",
                )
                result = invoke_advisor(
                    model=model,
                    role="scenario_director",
                    player_input=state.get("player_input", ""),
                    context=packet["context"],
                    sqlite_path=state.get("sqlite_path"),
                    turn_id=state.get("turn_id"),
                    contract_mode=_advisor_contract_mode(state),
                )
                decision = ScenarioDirectorDecision.model_validate(result.output.model_dump())
                advisor_trace = result.trace_metadata
                context_packet_trace = packet["trace"]
                structured_attempts = [
                    {key: value for key, value in attempt.items() if key != "raw_output"}
                    for attempt in result.attempts
                ]
            except Exception as error:
                decision = ScenarioDirectorDecision(
                    decision="no_change",
                    proposed_patches=[],
                    player_visible_context=(
                        "Scenario director unavailable; no scenario patch proposed."
                    ),
                    gm_only_reason=f"Scenario advisor failed safely: {error}",
                    citations=[],
                )
                advisor_trace = {"fallback": "true", "error": str(error)}
                context_packet_trace = {}
                structured_attempts = []

        return _apply_scenario_director_decision(
            state=state,
            decision=decision,
            node="direct_scenario_with_llm",
            trace_payload={
                "advisor": advisor_trace,
                "context_packet": context_packet_trace,
                "structured_attempts": structured_attempts,
            },
        )

    return direct_scenario_with_llm


def _single_turn_scenario_advice_for_director(
    state: GraphState,
) -> ScenarioDirectorDecision | None:
    if not state.get("single_turn_advisor_mode"):
        return None
    if _has_successful_resolver_result(state):
        return None
    raw_advice = state.get("single_turn_scenario_advice")
    if not isinstance(raw_advice, dict):
        return None
    try:
        return ScenarioDirectorDecision.model_validate(raw_advice)
    except Exception:
        return None


def _has_successful_resolver_result(state: GraphState) -> bool:
    return any(
        result.get("tool_name") == "run_ruleset_resolver" and result.get("ok")
        for result in state.get("tool_results", [])
        if isinstance(result, dict)
    )


def _scoped_scenario_visible_context(
    decision: ScenarioDirectorDecision,
    validation: ScenarioPatchValidation,
    state: GraphState,
) -> str:
    return decision.player_visible_context


def _limit_progressive_scenario_reveals(
    validation: ScenarioPatchValidation,
    state: GraphState,
) -> ScenarioPatchValidation:
    """Keep one ordinary observation from dumping a whole scene's clue stack."""

    decision = state.get("turn_plan", {}).get("decision")
    if decision not in {"free_action", "gm_move", "answer"}:
        return validation

    accepted: list[WorldPatch] = []
    rejected = list(validation.rejected)
    reveal_count = 0
    for patch in validation.patches:
        if patch.op == "append" and patch.path in (["revealed_facts"], ["known_clues"]):
            if reveal_count >= 1:
                rejected.append(
                    {
                        "patch": patch.model_dump(),
                        "reason": "progressive_disclosure_limit_one_reveal_per_turn",
                    }
                )
                continue
            trimmed = _trim_atomic_reveal_text(patch.value)
            if trimmed != patch.value:
                rejected.append(
                    {
                        "patch": patch.model_dump(),
                        "reason": "progressive_disclosure_trimmed_dense_reveal",
                    }
                )
                patch = patch.model_copy(update={"value": trimmed})
            reveal_count += 1
        accepted.append(patch)
    return ScenarioPatchValidation(patches=accepted, rejected=rejected)


def _trim_atomic_reveal_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) <= 90 and _sentence_count(text) <= 1:
        return text
    first_end = _first_sentence_end(text)
    if first_end is not None:
        first = text[:first_end].strip()
        if first:
            return first
    if len(text) <= 120:
        return text
    clauses = re.split(r"([,，;；])", text)
    if not clauses:
        return text[:120].rstrip() + "..."
    result = ""
    for index in range(0, len(clauses), 2):
        candidate = result + "".join(clauses[index : index + 2])
        if len(candidate) > 120:
            break
        result = candidate
    return result.strip() or text[:120].rstrip() + "..."


def _first_sentence_end(text: str) -> int | None:
    match = re.search(r"[。！？.!?]", text)
    if not match:
        return None
    return match.end()


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[。！？.!?]", text))


def _execute_tool_request(request: ToolRequest, state: GraphState) -> dict | list:
    arguments = dict(request.arguments)
    if request.tool_name == "roll_dice":
        raise ValueError("roll_dice is only available through the /roll command.")

    if request.tool_name == "search_content":
        if not arguments.get("content_dir"):
            arguments["content_dir"] = state.get("content_dir") or str(Path.cwd() / "content")
        return search_content(**arguments)

    if request.tool_name == "load_content_span":
        if not arguments.get("content_dir"):
            arguments["content_dir"] = state.get("content_dir") or str(Path.cwd() / "content")
        return load_content_span(**arguments)

    if request.tool_name == "run_ruleset_resolver":
        arguments.update(_protected_resolver_arguments(state))
        arguments["approach"] = _approach_for_resolver_request(state, arguments.get("approach"))
        arguments.pop("requested_roll", None)
        return run_ruleset_resolver(**arguments)

    if request.tool_name == "apply_world_patch":
        patches = arguments.get("patches") or []
        return {"world_patches": patches, "reason": arguments.get("reason", "")}

    if request.tool_name == "write_canon_event":
        sqlite_path = state.get("sqlite_path")
        if not sqlite_path:
            raise ValueError("write_canon_event requires sqlite_path in graph state")
        store = SqliteStore(Path(sqlite_path))
        store.migrate()
        payload = dict(arguments.get("payload") or {})
        event_id = str(arguments.get("event_id") or f"{state.get('turn_id')}:canon")
        inserted = store.insert_canon_event(
            event_id=event_id,
            session_id=state.get("session_id", "default"),
            turn_id=state.get("turn_id"),
            event_type=str(arguments.get("event_type") or "tool_event"),
            payload=payload,
        )
        return {"event_id": event_id, "inserted": inserted}

    raise ValueError(f"Unknown deterministic tool: {request.tool_name}")


def apply_world_patch_results(state: GraphState) -> GraphState:
    raw_patches: list[dict[str, Any]] = []
    for tool_result in state.get("tool_results", []):
        if not tool_result.get("ok"):
            continue
        result = tool_result.get("result")
        if isinstance(result, dict):
            patches = result.get("world_patches") or []
            raw_patches.extend(patch for patch in patches if isinstance(patch, dict))

    if not raw_patches:
        return _append_trace(state, "apply_world_patch_results", {"applied": []})

    patches = [WorldPatch.model_validate(patch) for patch in raw_patches]
    applied = apply_world_patches(state.get("world_projection", {}), patches)
    next_world = applied.state
    clock = next_world.get("clock")
    if isinstance(clock, dict) and "max" in clock and "value" in clock:
        clock["value"] = min(int(clock["value"]), int(clock["max"]))
    next_world = sync_scene_details(
        state=next_world,
        content_dir=Path(state.get("content_dir") or Path.cwd() / "content"),
        scenario_id=state.get("scenario_id"),
    )
    next_character_context = state.get("character_context", {})
    world_character_context = next_world.get("character_context")
    if isinstance(world_character_context, dict):
        next_character_context = dict(next_character_context)
        next_character_context.update(world_character_context)

    sqlite_path = state.get("sqlite_path")
    if sqlite_path:
        store = SqliteStore(Path(sqlite_path))
        store.migrate()
        next_world, inserted = store.commit_session_state_once(
            application_id=f"{state.get('turn_id', 'turn')}:world-patches",
            session_id=state.get("session_id", "default"),
            turn_id=state.get("turn_id", "turn"),
            patches=[patch.model_dump() for patch in patches],
            resulting_state=next_world,
        )
    else:
        inserted = True

    return _append_trace(
        {**state, "world_projection": next_world, "character_context": next_character_context},
        "apply_world_patch_results",
        {
            "applied": [patch.model_dump() for patch in applied.applied],
            "persisted_effect": inserted,
        },
    )


def emit_turn_output(state: GraphState) -> GraphState:
    plan = state.get("turn_plan", {})
    decision = plan.get("decision", "clarify")
    brief = plan.get("narration_brief", "")
    tool_lines: list[str] = []
    prefers_chinese = _prefers_chinese(state.get("player_input", ""))
    for result in state.get("tool_results", []):
        if result.get("tool_name") == "roll_dice" and result.get("ok"):
            payload = result.get("result") or {}
            if prefers_chinese:
                tool_lines.append(
                    "骰点结果："
                    f"{payload.get('expression')} -> {payload.get('rolls')}，"
                    f"总计 {payload.get('total')}。"
                )
            else:
                tool_lines.append(
                    f"Roll: {payload.get('expression')} -> {payload.get('rolls')}, "
                    f"total {payload.get('total')}."
                )
        if result.get("tool_name") == "run_ruleset_resolver" and result.get("ok"):
            payload = result.get("result") or {}
            dice = payload.get("dice_result") or {}
            if prefers_chinese:
                tool_lines.append(
                    "判定结果："
                    f"{dice.get('expression')} -> {dice.get('rolls')}，"
                    f"成功数 {payload.get('successes')}，档位 {payload.get('band_label')}。"
                )
            else:
                tool_lines.append(
                    f"Resolution: {dice.get('expression')} -> {dice.get('rolls')}; "
                    f"successes {payload.get('successes')}, band {payload.get('band_label')}."
                )
    suffix = (" " + " ".join(tool_lines)) if tool_lines else ""
    context = _local_output_context(state, decision)
    player_brief = _player_facing_brief_or_default(state, str(decision), str(brief))
    final_output = f"[turn_plan:{decision}] {player_brief}{context}{suffix}".strip()
    return _append_trace(
        {**state, "final_output": final_output},
        "emit_turn_output",
        {"final_output": final_output},
    )


def _local_output_context(state: GraphState, decision: str) -> str:
    if decision not in {"free_action", "gm_move"}:
        return ""
    pieces: list[str] = []
    player_input = state.get("player_input", "").strip()
    prefers_chinese = _prefers_chinese(player_input)
    if decision == "free_action" and player_input:
        if prefers_chinese:
            pieces.append(f"行动焦点：{player_input[:80]}。")
        else:
            pieces.append(f"Action focus: {player_input[:80]}.")
    world = state.get("world_projection", {})
    scene = world.get("scene")
    if isinstance(scene, dict) and scene.get("public_summary"):
        summary = str(scene["public_summary"]).strip()
        if prefers_chinese:
            pieces.append(f"可见局势：{summary[:140]}。")
        else:
            pieces.append(f"Visible situation: {summary[:140]}.")
    clock = world.get("clock")
    if isinstance(clock, dict) and "value" in clock and "max" in clock:
        if prefers_chinese:
            pieces.append(f"当前压力：{clock.get('value')}/{clock.get('max')}。")
        else:
            pieces.append(f"Current pressure: {clock.get('value')}/{clock.get('max')}.")
    return (" " + " ".join(pieces)) if pieces else ""


def persist_turn(state: GraphState) -> GraphState:
    sqlite_path = state.get("sqlite_path")
    if not sqlite_path:
        return _append_trace(state, "persist_turn", {"persisted": False})

    store = SqliteStore(Path(sqlite_path))
    store.migrate()
    store.upsert_session(
        session_id=state.get("session_id", "default"),
        ruleset_id=state.get("ruleset_id"),
        scenario_id=state.get("scenario_id"),
    )
    trace = {
        "trace_events": state.get("trace_events", []),
        "retrieved_spans": [
            {
                "package_id": span.get("package_id"),
                "reference_id": span.get("reference_id"),
                "visibility": span.get("visibility"),
                "score": span.get("score"),
            }
            for span in state.get("retrieved_spans", [])
        ],
        "recent_canon": state.get("recent_canon", []),
        "memory_hits": state.get("memory_hits", []),
        "player_memory_hits": state.get("player_memory_hits", []),
        "context_budget": state.get("context_budget", {}),
        "package_profiles": state.get("package_profiles", []),
        "tool_results": state.get("tool_results", []),
        "routing_decision": state.get("routing_decision", {}),
        "rules_advice": state.get("rules_advice", {}),
        "scenario_surface_selector": state.get("scenario_surface_selector", {}),
        "scenario_director": state.get("scenario_director", {}),
        "turn_plan": state.get("turn_plan", {}),
        "narration_plan": state.get("narration_plan", {}),
        "critic_report": state.get("critic_report", {}),
        "memory_curation": state.get("memory_curation", {}),
        "micro_gate_results": state.get("micro_gate_results", {}),
        "world_projection": state.get("world_projection", {}),
        "character_context": state.get("character_context", {}),
        "runtime_metadata": state.get("runtime_metadata", {}),
    }
    trace = redact_secrets(trace)
    inserted_turn = store.insert_turn(
        turn_id=state.get("turn_id", f"turn-{uuid.uuid4().hex[:12]}"),
        session_id=state.get("session_id", "default"),
        player_input=state.get("player_input", ""),
        output=state.get("final_output", ""),
        trace=trace,
    )
    inserted_canon = store.insert_canon_event(
        event_id=f"{state.get('turn_id')}:turn-summary",
        session_id=state.get("session_id", "default"),
        turn_id=state.get("turn_id"),
        event_type="turn_summary",
        payload={
            "player_action": state.get("player_input", ""),
            "narration": state.get("final_output", ""),
            "decision": state.get("turn_plan", {}).get("decision"),
            "draft": state.get("narration_plan", {}).get("canon_event_draft"),
            "world_projection": state.get("world_projection", {}),
            "character_context": state.get("character_context", {}),
            "tool_results": state.get("tool_results", []),
        },
    )
    inserted_critic = False
    if state.get("critic_report"):
        inserted_critic = store.insert_critic_report_once(
            report_id=f"{state.get('turn_id')}:critic",
            session_id=state.get("session_id", "default"),
            turn_id=state.get("turn_id", "turn"),
            report=state.get("critic_report", {}),
        )
    return _append_trace(
        state,
        "persist_turn",
        {
            "persisted": inserted_turn,
            "canon_inserted": inserted_canon,
            "critic_inserted": inserted_critic,
        },
    )


def route_after_context_retrieval(state: GraphState) -> str:
    if state.get("single_turn_advisor_mode"):
        return "single_turn_advisor"
    if state.get("micro_gates_mode"):
        return "micro_gates"
    return "intent_arbiter"


def route_after_tools_for_scenario(state: GraphState) -> str:
    if _scenario_surface_selector_eligible(state):
        return "surface_selector"
    return "scenario_director"


def route_after_surface_selector(state: GraphState) -> str:
    selector = state.get("scenario_surface_selector", {})
    if selector.get("fallback_to_full_director"):
        return "scenario_director"
    return "apply_patches"


def route_after_narration(state: GraphState) -> str:
    if _low_risk_local_review_reason(state):
        return "local_review"
    if state.get("parallel_review_mode") and not state.get("eval_smoke_mode"):
        return "parallel_review"
    return "critic_guardrail"


def build_turn_graph(checkpointer: Any | None = None):
    graph = StateGraph(GraphState)
    graph.add_node("receive_input", receive_input)
    graph.add_node("load_runtime_context", load_runtime_context)
    graph.add_node("retrieve_context_parallel", retrieve_context_parallel)
    graph.add_node("classify_player_intent", classify_player_intent)
    graph.add_node("plan_turn_locally", plan_turn_locally)
    graph.add_node("ensure_resolution_tools", ensure_resolution_tools)
    graph.add_node("execute_deterministic_tools", execute_deterministic_tools)
    graph.add_node("direct_scenario_locally", direct_scenario_locally)
    graph.add_node("apply_world_patch_results", apply_world_patch_results)
    graph.add_node("emit_turn_output", emit_turn_output)
    graph.add_node("critic_guardrail_locally", critic_guardrail_locally)
    graph.add_node("curate_memory_locally", curate_memory_locally)
    graph.add_node("persist_memory_curation", persist_memory_curation)
    graph.add_node("persist_turn", persist_turn)
    graph.add_edge(START, "receive_input")
    graph.add_edge("receive_input", "load_runtime_context")
    graph.add_conditional_edges(
        "load_runtime_context",
        route_after_runtime_context,
        {"replayed": END, "new_turn": "retrieve_context_parallel"},
    )
    graph.add_edge("retrieve_context_parallel", "classify_player_intent")
    graph.add_edge("classify_player_intent", "plan_turn_locally")
    graph.add_edge("plan_turn_locally", "ensure_resolution_tools")
    graph.add_edge("ensure_resolution_tools", "execute_deterministic_tools")
    graph.add_edge("execute_deterministic_tools", "direct_scenario_locally")
    graph.add_edge("direct_scenario_locally", "apply_world_patch_results")
    graph.add_edge("apply_world_patch_results", "emit_turn_output")
    graph.add_edge("emit_turn_output", "critic_guardrail_locally")
    graph.add_edge("critic_guardrail_locally", "curate_memory_locally")
    graph.add_edge("curate_memory_locally", "persist_memory_curation")
    graph.add_edge("persist_memory_curation", "persist_turn")
    graph.add_edge("persist_turn", END)
    return graph.compile(checkpointer=checkpointer)


def build_turn_graph_with_model(
    model: BaseChatModel,
    checkpointer: Any | None = None,
    advisor_models: AdvisorModelMap | None = None,
):
    graph = StateGraph(GraphState)
    graph.add_node("receive_input", receive_input)
    graph.add_node("load_runtime_context", load_runtime_context)
    graph.add_node("retrieve_context_parallel", retrieve_context_parallel)
    graph.add_node(
        "advise_turn_with_single_llm",
        build_llm_single_turn_advisor_node(model),
    )
    graph.add_node(
        "run_micro_gates",
        build_llm_micro_gates_node(model, advisor_models=advisor_models),
    )
    graph.add_node(
        "route_with_intent_arbiter",
        build_llm_intent_arbiter_node(
            _model_for_role("intent_arbiter", default_model=model, advisor_models=advisor_models)
        ),
    )
    graph.add_node(
        "advise_rules_with_llm",
        build_llm_rules_adjudicator_node(
            _model_for_role(
                "rules_adjudicator",
                default_model=model,
                advisor_models=advisor_models,
            )
        ),
    )
    graph.add_node("adjudicate_with_llm", build_llm_adjudication_node(model))
    graph.add_node("plan_turn_from_routing", plan_turn_from_routing)
    graph.add_node("ensure_resolution_tools", ensure_resolution_tools)
    graph.add_node("execute_deterministic_tools", execute_deterministic_tools)
    graph.add_node(
        "select_scenario_surface_with_llm",
        build_llm_scenario_surface_selector_node(
            _model_for_role(
                "scenario_surface_selector",
                default_model=model,
                advisor_models=advisor_models,
            )
        ),
    )
    graph.add_node(
        "direct_scenario_with_llm",
        build_llm_scenario_director_node(
            _model_for_role(
                "scenario_director",
                default_model=model,
                advisor_models=advisor_models,
            )
        ),
    )
    graph.add_node("apply_world_patch_results", apply_world_patch_results)
    graph.add_node("narrate_with_llm", build_llm_narration_node(model))
    critic_node = build_llm_critic_guardrail_node(
        _model_for_role(
            "critic_guardrail",
            default_model=model,
            advisor_models=advisor_models,
        )
    )
    memory_node = build_llm_memory_curator_node(
        _model_for_role(
            "memory_curator",
            default_model=model,
            advisor_models=advisor_models,
        )
    )
    graph.add_node("critic_guardrail_with_llm", critic_node)
    graph.add_node("curate_memory_with_llm", memory_node)
    graph.add_node("critic_guardrail_locally", critic_guardrail_locally)
    graph.add_node("curate_memory_locally", curate_memory_locally)
    graph.add_node(
        "review_and_curate_parallel",
        build_parallel_review_and_memory_node(critic_node, memory_node),
    )
    graph.add_node("persist_memory_curation", persist_memory_curation)
    graph.add_node("persist_turn", persist_turn)
    graph.add_edge(START, "receive_input")
    graph.add_edge("receive_input", "load_runtime_context")
    graph.add_conditional_edges(
        "load_runtime_context",
        route_after_runtime_context,
        {"replayed": END, "new_turn": "retrieve_context_parallel"},
    )
    graph.add_conditional_edges(
        "retrieve_context_parallel",
        route_after_context_retrieval,
        {
            "single_turn_advisor": "advise_turn_with_single_llm",
            "micro_gates": "run_micro_gates",
            "intent_arbiter": "route_with_intent_arbiter",
        },
    )
    graph.add_edge("advise_turn_with_single_llm", "ensure_resolution_tools")
    graph.add_conditional_edges(
        "run_micro_gates",
        route_after_intent_arbiter,
        {
            "rules_advice": "advise_rules_with_llm",
            "direct_plan": "plan_turn_from_routing",
            "adjudicate": "adjudicate_with_llm",
        },
    )
    graph.add_conditional_edges(
        "route_with_intent_arbiter",
        route_after_intent_arbiter,
        {
            "rules_advice": "advise_rules_with_llm",
            "direct_plan": "plan_turn_from_routing",
            "adjudicate": "adjudicate_with_llm",
        },
    )
    graph.add_edge("advise_rules_with_llm", "adjudicate_with_llm")
    graph.add_edge("plan_turn_from_routing", "ensure_resolution_tools")
    graph.add_edge("adjudicate_with_llm", "ensure_resolution_tools")
    graph.add_edge("ensure_resolution_tools", "execute_deterministic_tools")
    graph.add_conditional_edges(
        "execute_deterministic_tools",
        route_after_tools_for_scenario,
        {
            "surface_selector": "select_scenario_surface_with_llm",
            "scenario_director": "direct_scenario_with_llm",
        },
    )
    graph.add_conditional_edges(
        "select_scenario_surface_with_llm",
        route_after_surface_selector,
        {
            "scenario_director": "direct_scenario_with_llm",
            "apply_patches": "apply_world_patch_results",
        },
    )
    graph.add_edge("direct_scenario_with_llm", "apply_world_patch_results")
    graph.add_edge("apply_world_patch_results", "narrate_with_llm")
    graph.add_conditional_edges(
        "narrate_with_llm",
        route_after_narration,
        {
            "parallel_review": "review_and_curate_parallel",
            "critic_guardrail": "critic_guardrail_with_llm",
            "local_review": "critic_guardrail_locally",
        },
    )
    graph.add_edge("critic_guardrail_with_llm", "curate_memory_with_llm")
    graph.add_edge("curate_memory_with_llm", "persist_memory_curation")
    graph.add_edge("critic_guardrail_locally", "curate_memory_locally")
    graph.add_edge("curate_memory_locally", "persist_memory_curation")
    graph.add_edge("review_and_curate_parallel", "persist_memory_curation")
    graph.add_edge("persist_memory_curation", "persist_turn")
    graph.add_edge("persist_turn", END)
    return graph.compile(checkpointer=checkpointer)

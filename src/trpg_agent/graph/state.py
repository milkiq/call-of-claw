from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    session_id: str
    thread_id: str
    turn_id: str
    player_input: str
    content_dir: str
    sqlite_path: str
    ruleset_id: str
    scenario_id: str
    active_package_ids: list[str]
    active_extension_ids: list[str]
    model_metadata: dict[str, Any]
    eval_smoke_mode: bool
    parallel_review_mode: bool
    single_turn_advisor_mode: bool
    micro_gates_mode: bool
    runtime_metadata: dict[str, Any]
    checkpoint_mode: str
    replayed_turn: bool
    world_projection: dict[str, Any]
    character_context: dict[str, Any]
    recent_canon: list[str]
    retrieved_spans: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    player_memory_hits: list[dict[str, Any]]
    package_profiles: list[dict[str, Any]]
    routing_decision: dict[str, Any]
    rules_advice: dict[str, Any]
    scenario_director: dict[str, Any]
    intent: dict[str, Any]
    authority_result: dict[str, Any]
    turn_plan: dict[str, Any]
    tool_requests: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    narration_plan: dict[str, Any]
    critic_report: dict[str, Any]
    memory_curation: dict[str, Any]
    single_turn_advice: dict[str, Any]
    single_turn_scenario_advice: dict[str, Any]
    micro_gate_results: dict[str, Any]
    final_output: str
    trace_refs: list[str]
    trace_events: list[dict[str, Any]]

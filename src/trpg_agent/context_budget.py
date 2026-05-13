from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_BUCKET_BUDGETS: dict[str, int] = {
    "stable_prefix": 4_000,
    "local_scene": 3_000,
    "local_rules": 4_000,
    "player_visible_memory": 2_500,
    "recent_canon": 2_500,
    "retrieved_public": 4_000,
    "retrieved_gm": 4_000,
    "tool_results": 2_000,
    "style_state": 1_500,
}

ADVISOR_CONTEXT_TARGETS: dict[str, int] = {
    "intent_arbiter": 6_000,
    "rules_adjudicator": 8_000,
    "core_gm": 10_000,
    "scenario_director": 10_000,
    "critic_guardrail": 12_000,
    "memory_curator": 8_000,
    "narrator": 9_000,
    "single_turn_advisor": 10_000,
}

CONTEXT_PACKET_VERSION = "context-packet-v1"
ContextBudgetMode = str


@dataclass(frozen=True)
class ContextPolicy:
    role: str
    max_chars: int
    allowed_buckets: tuple[str, ...]
    allowed_visibility: tuple[str, ...]
    mandatory_buckets: tuple[str, ...] = ()


ROLE_CONTEXT_POLICIES: dict[str, ContextPolicy] = {
    "intent_arbiter": ContextPolicy(
        role="intent_arbiter",
        max_chars=6_000,
        allowed_buckets=(
            "ids",
            "package_index",
            "world_summary",
            "recent_canon",
            "player_memory",
            "public_spans",
        ),
        allowed_visibility=("public", "revealed"),
    ),
    "rules_adjudicator": ContextPolicy(
        role="rules_adjudicator",
        max_chars=8_000,
        allowed_buckets=(
            "ids",
            "package_index",
            "rules_spans",
            "character_context",
            "routing_decision",
            "world_summary",
            "recent_canon",
        ),
        allowed_visibility=("public", "revealed", "gm_only"),
        mandatory_buckets=("rules_spans", "routing_decision"),
    ),
    "core_gm": ContextPolicy(
        role="core_gm",
        max_chars=10_000,
        allowed_buckets=(
            "ids",
            "mode",
            "package_index",
            "public_spans",
            "rules_spans",
            "world_summary",
            "character_context",
            "recent_canon",
            "player_memory",
            "routing_decision",
            "rules_advice",
            "tool_catalog",
            "required_schema",
        ),
        allowed_visibility=("public", "revealed", "gm_only"),
        mandatory_buckets=("routing_decision", "rules_advice"),
    ),
    "scenario_director": ContextPolicy(
        role="scenario_director",
        max_chars=10_000,
        allowed_buckets=(
            "ids",
            "package_index",
            "scenario_spans",
            "public_spans",
            "world_summary",
            "recent_canon",
            "memory",
            "turn_plan_summary",
            "tool_result_summaries",
        ),
        allowed_visibility=("public", "revealed", "gm_only"),
        mandatory_buckets=("scenario_spans", "turn_plan_summary", "tool_result_summaries"),
    ),
    "narrator": ContextPolicy(
        role="narrator",
        max_chars=9_000,
        allowed_buckets=(
            "archivist_packet",
            "player_visible_memory",
            "recent_canon",
            "tool_result_summaries",
        ),
        allowed_visibility=("public", "revealed", "player_visible", "validated_player_visible"),
        mandatory_buckets=("archivist_packet", "tool_result_summaries"),
    ),
    "critic_guardrail": ContextPolicy(
        role="critic_guardrail",
        max_chars=12_000,
        allowed_buckets=(
            "archivist_packet",
            "final_text",
            "turn_plan_summary",
            "tool_result_summaries",
            "world_summary",
            "recent_canon",
            "player_memory",
            "forbidden_boundaries",
        ),
        allowed_visibility=("public", "revealed", "player_visible", "validated_player_visible"),
        mandatory_buckets=("archivist_packet", "final_text", "tool_result_summaries"),
    ),
    "memory_curator": ContextPolicy(
        role="memory_curator",
        max_chars=8_000,
        allowed_buckets=(
            "ids",
            "final_text",
            "turn_plan_summary",
            "narration_summary",
            "critic_summary",
            "tool_result_summaries",
            "world_summary",
            "recent_canon",
            "memory",
        ),
        allowed_visibility=("public", "revealed", "gm_only", "player_visible"),
    ),
    "single_turn_advisor": ContextPolicy(
        role="single_turn_advisor",
        max_chars=10_000,
        allowed_buckets=(
            "ids",
            "mode",
            "package_index",
            "public_spans",
            "rules_spans",
            "world_summary",
            "character_context",
            "recent_canon",
            "player_memory",
            "tool_catalog",
            "required_schema",
        ),
        allowed_visibility=("public", "revealed"),
        mandatory_buckets=("rules_spans",),
    ),
}


def build_context_budget_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    buckets = _bucket_values(state)
    summaries = []
    raw_total = 0
    kept_total = 0
    for name, value in buckets.items():
        raw_chars = _json_chars(value)
        budget_chars = DEFAULT_BUCKET_BUDGETS[name]
        kept_chars = min(raw_chars, budget_chars)
        raw_total += raw_chars
        kept_total += kept_chars
        summaries.append(
            {
                "name": name,
                "raw_chars": raw_chars,
                "budget_chars": budget_chars,
                "would_keep_chars": kept_chars,
                "would_clip_chars": max(0, raw_chars - kept_chars),
            }
        )
    return {
        "mode": "shadow",
        "raw_total_chars": raw_total,
        "budgeted_total_chars": kept_total,
        "would_clip_chars": max(0, raw_total - kept_total),
        "buckets": summaries,
        "advisor_targets": ADVISOR_CONTEXT_TARGETS,
    }


def compact_context_budget_trace(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": snapshot.get("mode", "shadow"),
        "raw_total_chars": snapshot.get("raw_total_chars", 0),
        "budgeted_total_chars": snapshot.get("budgeted_total_chars", 0),
        "would_clip_chars": snapshot.get("would_clip_chars", 0),
        "buckets": [
            {
                "name": bucket.get("name"),
                "raw_chars": bucket.get("raw_chars", 0),
                "budget_chars": bucket.get("budget_chars", 0),
                "would_clip_chars": bucket.get("would_clip_chars", 0),
            }
            for bucket in snapshot.get("buckets", [])
            if isinstance(bucket, Mapping)
        ],
    }


def build_advisor_context(
    state: Mapping[str, Any],
    role: str,
    *,
    mode: ContextBudgetMode = "enforced",
    extra_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the context passed to an LLM role through the context firewall."""

    policy = ROLE_CONTEXT_POLICIES.get(role, ROLE_CONTEXT_POLICIES["core_gm"])
    raw_context = _raw_role_context(state, role, extra_context=extra_context or {})
    raw_chars = _json_chars(raw_context)
    decisions: list[dict[str, Any]] = []
    if mode != "enforced":
        legacy_context = _legacy_role_context(state, role, extra_context=extra_context or {})
        legacy_chars = _json_chars(legacy_context)
        return {
            "role": role,
            "mode": "shadow",
            "version": CONTEXT_PACKET_VERSION,
            "context": legacy_context,
            "trace": {
                "role": role,
                "mode": "shadow",
                "version": CONTEXT_PACKET_VERSION,
                "budget_target_chars": policy.max_chars,
                "raw_chars": raw_chars,
                "sent_chars": legacy_chars,
                "clipped_chars": 0,
                "decisions": [],
                "citations": _collect_citations(legacy_context),
            },
        }

    filtered = {
        key: value
        for key, value in raw_context.items()
        if key in policy.allowed_buckets or key == "advisor_contract"
    }
    dropped_bucket_keys = [
        key for key in raw_context if key not in filtered and key != "advisor_contract"
    ]
    decisions.extend(
        {
            "item_id": key,
            "action": "dropped",
            "reason_code": "bucket_not_allowed_for_role",
        }
        for key in dropped_bucket_keys
    )
    decisions.extend(_hidden_span_denials(state, role))
    filtered, visibility_decisions = _filter_context_visibility(filtered, policy)
    decisions.extend(visibility_decisions)
    budgeted, budget_decisions = _enforce_context_budget(
        filtered,
        policy=policy,
    )
    decisions.extend(budget_decisions)
    sent_chars = _json_chars(budgeted)
    return {
        "role": role,
        "mode": "enforced",
        "version": CONTEXT_PACKET_VERSION,
        "context": budgeted,
        "trace": {
            "role": role,
            "mode": "enforced",
            "version": CONTEXT_PACKET_VERSION,
            "budget_target_chars": policy.max_chars,
            "raw_chars": raw_chars,
            "sent_chars": sent_chars,
            "clipped_chars": max(0, raw_chars - sent_chars),
            "decisions": decisions,
            "citations": _collect_citations(budgeted),
        },
    }


def _bucket_values(state: Mapping[str, Any]) -> dict[str, Any]:
    retrieved_public = []
    retrieved_gm = []
    for span in state.get("retrieved_spans", []):
        if not isinstance(span, Mapping):
            continue
        if span.get("visibility") == "gm_only":
            retrieved_gm.append(span)
        else:
            retrieved_public.append(span)
    return {
        "stable_prefix": {
            "runtime_metadata": state.get("runtime_metadata", {}),
            "package_profiles": state.get("package_profiles", []),
        },
        "local_scene": {
            "world_projection": state.get("world_projection", {}),
            "character_context": state.get("character_context", {}),
        },
        "local_rules": {
            "ruleset_id": state.get("ruleset_id"),
            "active_package_ids": state.get("active_package_ids", []),
        },
        "player_visible_memory": state.get("player_memory_hits", []),
        "recent_canon": state.get("recent_canon", []),
        "retrieved_public": retrieved_public,
        "retrieved_gm": retrieved_gm,
        "tool_results": state.get("tool_results", []),
        "style_state": state.get("style_state", {}),
    }


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _raw_role_context(
    state: Mapping[str, Any],
    role: str,
    *,
    extra_context: Mapping[str, Any],
) -> dict[str, Any]:
    ids = {
        "ruleset_id": state.get("ruleset_id"),
        "scenario_id": state.get("scenario_id"),
        "active_package_ids": state.get("active_package_ids", []),
    }
    context: dict[str, Any] = {
        "ids": ids,
        "package_index": _package_index(state, role),
        "public_spans": _spans_for_bucket(state, role, bucket="public_spans"),
        "rules_spans": _spans_for_bucket(state, role, bucket="rules_spans"),
        "scenario_spans": _spans_for_bucket(state, role, bucket="scenario_spans"),
        "world_summary": _world_summary(state.get("world_projection", {})),
        "character_context": _limited_mapping(state.get("character_context", {}), text_limit=500),
        "recent_canon": _recent_canon(state),
        "memory": _memory_hits(state.get("memory_hits", []), limit=4),
        "player_memory": _memory_hits(state.get("player_memory_hits", []), limit=4),
        "player_visible_memory": _memory_hits(state.get("player_memory_hits", []), limit=4),
        "routing_decision": _routing_summary(state.get("routing_decision", {})),
        "rules_advice": _rules_advice_summary(state.get("rules_advice", {})),
        "turn_plan_summary": _turn_plan_summary(state.get("turn_plan", {})),
        "tool_result_summaries": _tool_result_summaries(state.get("tool_results", [])),
        "archivist_packet": _archivist_packet(state),
        "final_text": state.get("final_output", ""),
        "narration_summary": _narration_summary(state.get("narration_plan", {})),
        "critic_summary": _critic_summary(state.get("critic_report", {})),
        "forbidden_boundaries": _forbidden_boundaries(state),
    }
    context.update(extra_context)
    return context


def _legacy_role_context(
    state: Mapping[str, Any],
    role: str,
    *,
    extra_context: Mapping[str, Any],
) -> dict[str, Any]:
    advisor_contract = extra_context.get("advisor_contract")
    base_ids = {
        "ruleset_id": state.get("ruleset_id"),
        "scenario_id": state.get("scenario_id"),
    }
    if role == "intent_arbiter":
        return {
            **base_ids,
            "recent_canon": state.get("recent_canon", []),
            "memory_hits": state.get("memory_hits", []),
            "player_memory_hits": state.get("player_memory_hits", []),
            "retrieved_spans": state.get("retrieved_spans", []),
            "world_projection": state.get("world_projection", {}),
            "character_context": state.get("character_context", {}),
            "package_profiles": state.get("package_profiles", []),
            "advisor_contract": advisor_contract,
        }
    if role == "rules_adjudicator":
        return {
            **base_ids,
            "routing_decision": state.get("routing_decision", {}),
            "recent_canon": state.get("recent_canon", []),
            "memory_hits": state.get("memory_hits", []),
            "player_memory_hits": state.get("player_memory_hits", []),
            "retrieved_spans": state.get("retrieved_spans", []),
            "world_projection": state.get("world_projection", {}),
            "character_context": state.get("character_context", {}),
            "package_profiles": state.get("package_profiles", []),
            "advisor_contract": advisor_contract,
        }
    if role == "core_gm":
        return {
            "mode": extra_context.get("mode", "turn_adjudication"),
            **base_ids,
            "recent_canon": state.get("recent_canon", []),
            "memory_hits": state.get("memory_hits", []),
            "player_memory_hits": state.get("player_memory_hits", []),
            "retrieved_spans": state.get("retrieved_spans", []),
            "world_projection": state.get("world_projection", {}),
            "character_context": state.get("character_context", {}),
            "package_profiles": state.get("package_profiles", []),
            "routing_decision": state.get("routing_decision", {}),
            "rules_advice": state.get("rules_advice", {}),
            "available_tools": extra_context.get("tool_catalog", []),
            "required_schema": extra_context.get("required_schema"),
        }
    if role == "scenario_director":
        return {
            **base_ids,
            "world_projection": state.get("world_projection", {}),
            "recent_canon": state.get("recent_canon", []),
            "memory_hits": state.get("memory_hits", []),
            "player_memory_hits": state.get("player_memory_hits", []),
            "retrieved_spans": state.get("retrieved_spans", []),
            "turn_plan": state.get("turn_plan", {}),
            "tool_results": state.get("tool_results", []),
            "package_profiles": state.get("package_profiles", []),
            "advisor_contract": advisor_contract,
        }
    if role == "narrator":
        return {
            **base_ids,
            "recent_canon": state.get("recent_canon", []),
            "memory_hits": state.get("player_memory_hits", []),
            "retrieved_spans": state.get("retrieved_spans", []),
            "world_projection": state.get("world_projection", {}),
            "character_context": state.get("character_context", {}),
            "scenario_director": state.get("scenario_director", {}),
            "turn_plan": state.get("turn_plan", {}),
            "tool_results": state.get("tool_results", []),
        }
    if role == "critic_guardrail":
        return {
            "final_text": state.get("final_output", ""),
            "player_input": state.get("player_input", ""),
            "turn_plan": state.get("turn_plan", {}),
            "tool_results": state.get("tool_results", []),
            "applied_world_projection": state.get("world_projection", {}),
            "scenario_director": state.get("scenario_director", {}),
            "recent_canon": state.get("recent_canon", []),
            "player_memory_hits": state.get("player_memory_hits", []),
            "gm_memory_hits": state.get("memory_hits", []),
            "retrieved_spans": state.get("retrieved_spans", []),
            "advisor_contract": advisor_contract,
        }
    if role == "memory_curator":
        return {
            "player_input": state.get("player_input", ""),
            "final_output": state.get("final_output", ""),
            "turn_plan": state.get("turn_plan", {}),
            "narration_plan": state.get("narration_plan", {}),
            "critic_report": state.get("critic_report", {}),
            "tool_results": state.get("tool_results", []),
            "world_projection": state.get("world_projection", {}),
            "recent_canon": state.get("recent_canon", []),
            "memory_hits": state.get("memory_hits", []),
            "advisor_contract": advisor_contract,
        }
    if role == "single_turn_advisor":
        return {
            **base_ids,
            "recent_canon": state.get("recent_canon", []),
            "memory_hits": state.get("memory_hits", []),
            "player_memory_hits": state.get("player_memory_hits", []),
            "retrieved_spans": _public_retrieved_spans(state),
            "world_projection": state.get("world_projection", {}),
            "character_context": state.get("character_context", {}),
            "package_profiles": _public_package_profiles(state),
            "available_tools": extra_context.get("tool_catalog", []),
            "advisor_contract": advisor_contract,
        }
    return dict(extra_context)


def _package_index(state: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    allow_hidden_refs = role in {
        "rules_adjudicator",
        "core_gm",
        "scenario_director",
        "memory_curator",
    }
    for profile in state.get("package_profiles", []):
        if not isinstance(profile, Mapping):
            continue
        references = []
        for reference in profile.get("references", []):
            if not isinstance(reference, Mapping):
                continue
            visibility = _visibility(reference)
            if not allow_hidden_refs and visibility == "gm_only":
                continue
            references.append(
                {
                    "id": reference.get("id"),
                    "title": _clip_text(reference.get("title"), 120),
                    "visibility": visibility,
                    "tags": list(reference.get("tags", []))[:6],
                }
            )
        result.append(
            {
                "id": profile.get("id"),
                "kind": profile.get("kind"),
                "name": _clip_text(profile.get("name"), 120),
                "description": _clip_text(profile.get("description"), 240),
                "capabilities": list(profile.get("capabilities", []))[:8],
                "references": references[:12],
            }
        )
    return result


def _public_retrieved_spans(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(span)
        for span in state.get("retrieved_spans", [])
        if isinstance(span, Mapping) and _visibility(span) == "public"
    ]


def _public_package_profiles(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for profile in state.get("package_profiles", []):
        if not isinstance(profile, Mapping):
            continue
        references = [
            dict(reference)
            for reference in profile.get("references", [])
            if isinstance(reference, Mapping) and _visibility(reference) == "public"
        ]
        profiles.append({**dict(profile), "references": references})
    return profiles


def _spans_for_bucket(
    state: Mapping[str, Any],
    role: str,
    *,
    bucket: str,
) -> list[dict[str, Any]]:
    ruleset_id = state.get("ruleset_id")
    scenario_id = state.get("scenario_id")
    spans: list[dict[str, Any]] = []
    for raw_span in state.get("retrieved_spans", []):
        if not isinstance(raw_span, Mapping):
            continue
        package_id = raw_span.get("package_id")
        visibility = _visibility(raw_span)
        if bucket == "rules_spans" and package_id != ruleset_id:
            continue
        if bucket == "scenario_spans" and package_id not in {scenario_id, None}:
            continue
        if bucket == "public_spans" and visibility == "gm_only":
            continue
        if (
            role in {"intent_arbiter", "single_turn_advisor", "narrator"}
            and visibility == "gm_only"
        ):
            continue
        spans.append(_span_context(raw_span, bucket=bucket))
    limits = {
        "rules_spans": 4,
        "scenario_spans": 4,
        "public_spans": 3,
    }
    return spans[: limits.get(bucket, 4)]


def _span_context(span: Mapping[str, Any], *, bucket: str) -> dict[str, Any]:
    citation_id = _citation_id(span)
    visibility = _visibility(span)
    return {
        "package_id": span.get("package_id"),
        "reference_id": span.get("reference_id"),
        "citation_id": citation_id,
        "bucket": bucket,
        "visibility": visibility,
        "mandatory": bucket == "rules_spans",
        "purpose": _span_purpose(bucket, visibility),
        "title": _clip_text(span.get("title"), 160),
        "score": span.get("score"),
        "text": _clip_text(span.get("text"), 900),
    }


def _span_purpose(bucket: str, visibility: str) -> str:
    if bucket == "rules_spans":
        return "rules_adjudication"
    if visibility == "gm_only":
        return "gm_scenario_guidance"
    return "player_visible_context"


def _world_summary(world: Any) -> dict[str, Any]:
    if not isinstance(world, Mapping):
        return {}
    scene = world.get("scene")
    summary = {
        "active_scene": world.get("active_scene"),
        "clock": world.get("clock"),
        "revealed_facts": list(world.get("revealed_facts", []))[-8:],
        "known_clues": list(world.get("known_clues", []))[-8:],
        "canon_event_count": world.get("canon_event_count"),
    }
    if isinstance(scene, Mapping):
        summary["scene"] = {
            "id": scene.get("id"),
            "title": scene.get("title"),
            "public_summary": _clip_text(scene.get("public_summary"), 700),
            "transitions": _clip_json(scene.get("transitions", []), 900),
        }
    return summary


def _recent_canon(state: Mapping[str, Any], *, limit: int = 4) -> list[str]:
    return [_clip_text(item, 500) for item in state.get("recent_canon", [])[-limit:]]


def _memory_hits(hits: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(hits, list):
        return []
    result: list[dict[str, Any]] = []
    for hit in hits[:limit]:
        if not isinstance(hit, Mapping):
            continue
        result.append(
            {
                "kind": hit.get("kind"),
                "scope": hit.get("scope"),
                "text": _clip_text(hit.get("text"), 450),
                "metadata": _clip_json(hit.get("metadata", {}), 500),
            }
        )
    return result


def _routing_summary(routing: Any) -> dict[str, Any]:
    if not isinstance(routing, Mapping):
        return {}
    intent = routing.get("intent") if isinstance(routing.get("intent"), Mapping) else {}
    return {
        "intent": {
            "kind": intent.get("kind") if isinstance(intent, Mapping) else None,
            "confidence": intent.get("confidence") if isinstance(intent, Mapping) else None,
        },
        "route": routing.get("route"),
        "needs_rules_resolution": routing.get("needs_rules_resolution"),
        "needs_scenario_director": routing.get("needs_scenario_director"),
        "needs_memory_recall": routing.get("needs_memory_recall"),
        "allow_direct_answer": routing.get("allow_direct_answer"),
        "uncertainty": _clip_text(routing.get("uncertainty"), 180),
        "citations": list(routing.get("citations", []))[:6],
    }


def _rules_advice_summary(advice: Any) -> dict[str, Any]:
    if not isinstance(advice, Mapping):
        return {}
    return {
        "requires_resolution": advice.get("requires_resolution"),
        "procedure_id": advice.get("procedure_id"),
        "approach_id": advice.get("approach_id"),
        "risk": advice.get("risk"),
        "stakes": _clip_text(advice.get("stakes"), 220),
        "clarification_question": _clip_text(advice.get("clarification_question"), 220),
        "citations": list(advice.get("citations", []))[:6],
    }


def _turn_plan_summary(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        return {}
    intent = plan.get("intent") if isinstance(plan.get("intent"), Mapping) else {}
    return {
        "intent": {
            "kind": intent.get("kind") if isinstance(intent, Mapping) else None,
            "confidence": intent.get("confidence") if isinstance(intent, Mapping) else None,
        },
        "decision": plan.get("decision"),
        "tool_requests": [
            {
                "tool_name": request.get("tool_name"),
                "reason": _clip_text(request.get("reason"), 180),
            }
            for request in plan.get("tool_requests", [])
            if isinstance(request, Mapping)
        ],
        "narration_brief": _clip_text(plan.get("narration_brief"), 360),
        "citations": list(plan.get("citations", []))[:8],
    }


def _tool_result_summaries(results: Any) -> list[dict[str, Any]]:
    if not isinstance(results, list):
        return []
    summaries: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        summary: dict[str, Any] = {
            "tool_name": item.get("tool_name"),
            "ok": item.get("ok"),
            "error": _clip_text(item.get("error"), 220),
        }
        result = item.get("result")
        if isinstance(result, Mapping):
            summary["result"] = _summarize_tool_payload(result)
        else:
            summary["result"] = _clip_json(result, 500)
        request = item.get("request")
        if isinstance(request, Mapping):
            summary["request"] = {
                "tool_name": request.get("tool_name"),
                "reason": _clip_text(request.get("reason"), 180),
            }
        summaries.append(summary)
    return summaries[:6]


def _summarize_tool_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "dice_expression",
        "band",
        "success_count",
        "approach",
        "risk",
        "consequence",
        "player_visible_result",
        "player_visible_context",
        "decision",
        "error",
    ]
    summary = {
        key: _clip_json(payload.get(key), 600)
        for key in keys
        if key in payload and payload.get(key) is not None
    }
    dice = payload.get("dice_result")
    if isinstance(dice, Mapping):
        summary["dice_result"] = {
            "expression": dice.get("expression"),
            "rolls": dice.get("rolls"),
            "total": dice.get("total"),
        }
    patches = payload.get("world_patches")
    if isinstance(patches, list):
        summary["world_patch_count"] = len(patches)
    if not summary:
        return _clip_json(payload, 900)
    return summary


def _archivist_packet(state: Mapping[str, Any]) -> dict[str, Any]:
    scenario_director = state.get("scenario_director")
    visible_context = ""
    validated_patches: list[Any] = []
    if isinstance(scenario_director, Mapping):
        visible_context = _clip_text(scenario_director.get("player_visible_context"), 900)
        validated_patches = list(scenario_director.get("validated_patches", []))[:6]
    packet = {
        "ruleset_id": state.get("ruleset_id"),
        "scenario_id": state.get("scenario_id"),
        "player_visible_scene": _world_summary(state.get("world_projection", {})),
        "recent_canon": _recent_canon(state, limit=3),
        "player_visible_memory": _memory_hits(state.get("player_memory_hits", []), limit=3),
        "public_spans": _spans_for_bucket(state, "narrator", bucket="public_spans")[:3],
        "turn_plan": _turn_plan_summary(state.get("turn_plan", {})),
        "tool_results": _tool_result_summaries(state.get("tool_results", [])),
        "validated_scenario_context": visible_context,
        "validated_world_patches": _clip_json(validated_patches, 900),
        "allowed_citations": [],
        "no_go_boundaries": [
            "Do not reveal gm_only spans unless they are included as validated "
            "player-visible context.",
            "Do not invent tool results, dice results, durable facts, or scene transitions.",
        ],
    }
    packet["allowed_citations"] = _collect_citations(packet)
    return packet


def _narration_summary(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        return {}
    return {
        "final_text": _clip_text(plan.get("final_text"), 900),
        "canon_event_draft": _clip_json(plan.get("canon_event_draft"), 600),
        "memory_candidates": [
            _clip_text(item, 300) for item in plan.get("memory_candidates", [])[:4]
        ],
    }


def _critic_summary(report: Any) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        return {}
    return {
        "ok": report.get("ok"),
        "blocks_output": report.get("blocks_output"),
        "findings": _clip_json(report.get("findings", []), 900),
        "revised_final_text": _clip_text(report.get("revised_final_text"), 900),
    }


def _forbidden_boundaries(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "narrator_hidden_visibility_denied": True,
        "gm_only_citations": [
            _citation_id(span)
            for span in state.get("retrieved_spans", [])
            if isinstance(span, Mapping) and _visibility(span) == "gm_only"
        ],
    }


def _hidden_span_denials(state: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    if role not in {"intent_arbiter", "single_turn_advisor", "narrator", "critic_guardrail"}:
        return []
    decisions: list[dict[str, Any]] = []
    for span in state.get("retrieved_spans", []):
        if not isinstance(span, Mapping) or _visibility(span) != "gm_only":
            continue
        decisions.append(
            {
                "item_id": _citation_id(span),
                "action": "dropped",
                "reason_code": "visibility_denied_for_role",
            }
        )
    return decisions


def _filter_context_visibility(
    context: dict[str, Any],
    policy: ContextPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    filtered: dict[str, Any] = {}
    for key, value in context.items():
        filtered_value, dropped = _filter_value_visibility(
            value,
            allowed_visibility=set(policy.allowed_visibility),
            item_id=key,
        )
        filtered[key] = filtered_value
        decisions.extend(dropped)
    return filtered, decisions


def _filter_value_visibility(
    value: Any,
    *,
    allowed_visibility: set[str],
    item_id: str,
) -> tuple[Any, list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    if isinstance(value, list):
        kept = []
        for index, item in enumerate(value):
            if isinstance(item, Mapping) and _visibility(item) not in allowed_visibility:
                decisions.append(
                    {
                        "item_id": item.get("citation_id") or f"{item_id}[{index}]",
                        "action": "dropped",
                        "reason_code": "visibility_denied_for_role",
                    }
                )
                continue
            kept.append(item)
        return kept, decisions
    if isinstance(value, Mapping) and _visibility(value) not in allowed_visibility:
        return {}, [
            {
                "item_id": value.get("citation_id") or item_id,
                "action": "dropped",
                "reason_code": "visibility_denied_for_role",
            }
        ]
    return value, decisions


def _enforce_context_budget(
    context: dict[str, Any],
    *,
    policy: ContextPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    if _json_chars(context) <= policy.max_chars:
        return context, decisions

    ordered_keys = list(context)
    mandatory = set(policy.mandatory_buckets) | {"advisor_contract"}
    budgeted = dict(context)
    for key in reversed(ordered_keys):
        if key in mandatory:
            continue
        value = budgeted.get(key)
        if isinstance(value, list) and value:
            while len(value) > 1 and _json_chars(budgeted) > policy.max_chars:
                dropped = value.pop()
                decisions.append(
                    {
                        "item_id": _decision_item_id(dropped, key),
                        "action": "dropped",
                        "reason_code": "budget_exceeded",
                    }
                )
        if _json_chars(budgeted) <= policy.max_chars:
            return budgeted, decisions

    for key, value in list(budgeted.items()):
        if key in mandatory:
            continue
        clipped = _clip_to_chars(value, max_chars=1_000)
        if _json_chars(clipped) < _json_chars(value):
            budgeted[key] = clipped
            decisions.append(
                {
                    "item_id": key,
                    "action": "summarized",
                    "reason_code": "budget_exceeded",
                }
            )
        if _json_chars(budgeted) <= policy.max_chars:
            return budgeted, decisions

    return budgeted, decisions


def _decision_item_id(value: Any, key: str) -> str:
    if isinstance(value, Mapping):
        return str(value.get("citation_id") or value.get("id") or key)
    return key


def _collect_citations(value: Any) -> list[str]:
    citations: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            citation = item.get("citation_id")
            if citation:
                citations.append(str(citation))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return list(dict.fromkeys(citations))


def _visibility(value: Mapping[str, Any]) -> str:
    return str(value.get("visibility") or "public")


def _citation_id(value: Mapping[str, Any]) -> str:
    package_id = str(value.get("package_id") or "unknown_package")
    reference_id = str(value.get("reference_id") or value.get("id") or "unknown_reference")
    return f"{package_id}:{reference_id}"


def _limited_mapping(value: Any, *, text_limit: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _clip_json(item, text_limit) for key, item in value.items()}


def _clip_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def _clip_json(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return _clip_text(value, limit)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result[str(key)] = _clip_json(item, max(80, limit // max(1, len(value))))
        return result
    if isinstance(value, list):
        return [_clip_json(item, max(80, limit // max(1, len(value)))) for item in value[:8]]
    return value


def _clip_to_chars(value: Any, *, max_chars: int) -> Any:
    if _json_chars(value) <= max_chars:
        return value
    if isinstance(value, list):
        result = list(value)
        while result and _json_chars(result) > max_chars:
            result.pop()
        return result
    if isinstance(value, Mapping):
        return _clip_json(value, max_chars)
    return _clip_text(value, max_chars)

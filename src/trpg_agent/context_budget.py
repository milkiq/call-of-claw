from __future__ import annotations

import json
from collections.abc import Mapping
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
    "scenario_director": 10_000,
    "critic_guardrail": 12_000,
    "memory_curator": 8_000,
    "narrator": 9_000,
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

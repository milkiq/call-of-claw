from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from coc.graph.build_turn_graph import (
    AdvisorModelMap,
    build_turn_graph,
    build_turn_graph_with_model,
)
from coc.graph.state import GraphState

RUNTIME_LATENCY_BUDGETS: dict[str, dict[str, int]] = {
    "fast": {"turn_ms": 15_000, "node_ms": 5_000},
    "balanced": {"turn_ms": 45_000, "node_ms": 15_000},
    "theatrical": {"turn_ms": 90_000, "node_ms": 30_000},
}
DEFAULT_RUNTIME_BUDGET_PROFILE = "balanced"


def checkpoint_path_for(sqlite_path: Path) -> Path:
    return sqlite_path.with_name(f"{sqlite_path.stem}.checkpoints.sqlite")


def delete_turn_graph_checkpoints(
    sqlite_path: Path,
    *,
    session_ids: list[str] | None = None,
    all_sessions: bool = False,
) -> dict[str, int]:
    checkpoint_path = checkpoint_path_for(sqlite_path)
    if not checkpoint_path.exists():
        return {"checkpoints": 0, "writes": 0}
    if not all_sessions and not session_ids:
        return {"checkpoints": 0, "writes": 0}
    counts = {"checkpoints": 0, "writes": 0}
    with sqlite3.connect(checkpoint_path) as conn:
        if all_sessions:
            for table in ("writes", "checkpoints"):
                counts[table] = _delete_checkpoint_rows(conn, table)
            return counts
        for session_id in list(dict.fromkeys(session_ids or [])):
            for table in ("writes", "checkpoints"):
                counts[table] += _delete_checkpoint_rows(conn, table, session_id=session_id)
    return counts


def _delete_checkpoint_rows(
    conn: sqlite3.Connection,
    table: str,
    *,
    session_id: str | None = None,
) -> int:
    if not _checkpoint_table_exists(conn, table):
        return 0
    if session_id is None:
        return _deleted_count(conn.execute(f"DELETE FROM {table}"))
    return _deleted_count(
        conn.execute(
            f"""
            DELETE FROM {table}
            WHERE thread_id = ?
               OR thread_id LIKE ?
            """,
            (session_id, f"{session_id}:%"),
        )
    )


def _checkpoint_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table,),
    ).fetchone()
    return row is not None


def _deleted_count(cursor: sqlite3.Cursor) -> int:
    return max(0, int(cursor.rowcount or 0))


def turn_graph_invoke_config(state: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    base_thread_id = str(state.get("thread_id") or state.get("session_id") or "default")
    turn_id = state.get("turn_id")
    thread_id = f"{base_thread_id}:{turn_id}" if turn_id else base_thread_id
    return {"configurable": {"thread_id": thread_id}}


@contextmanager
def durable_turn_graph(
    *,
    sqlite_path: Path,
    model: BaseChatModel | None = None,
    advisor_models: AdvisorModelMap | None = None,
) -> Iterator[Any]:
    checkpoint_path = checkpoint_path_for(sqlite_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        if model is None:
            yield build_turn_graph(checkpointer=checkpointer)
        else:
            yield build_turn_graph_with_model(
                model,
                checkpointer=checkpointer,
                advisor_models=advisor_models,
            )


def invoke_turn_graph(graph: Any, state: GraphState) -> GraphState:
    return _run_turn_graph_with_profile(graph, state)


def stream_turn_graph(
    graph: Any,
    state: GraphState,
    *,
    on_node: Any | None = None,
) -> GraphState:
    return _run_turn_graph_with_profile(graph, state, on_node=on_node)


def _run_turn_graph_with_profile(
    graph: Any,
    state: GraphState,
    *,
    on_node: Any | None = None,
) -> GraphState:
    final_state: GraphState = {}
    node_profiles: list[dict[str, Any]] = []
    start = time.monotonic()
    previous = start
    for chunk in graph.stream(
        state,
        turn_graph_invoke_config(state),
        stream_mode="updates",
    ):
        if not isinstance(chunk, dict):
            continue
        for node, update in chunk.items():
            now = time.monotonic()
            node_profiles.append(
                _node_profile(
                    node=str(node),
                    update=update,
                    sequence=len(node_profiles) + 1,
                    elapsed_ms=_elapsed_ms(previous, now),
                    completed_at_ms=_elapsed_ms(start, now),
                )
            )
            previous = now
            if on_node is not None:
                on_node(str(node), update if isinstance(update, dict) else {})
            if isinstance(update, dict):
                final_state.update(update)
    completed = time.monotonic()
    final_state["runtime_profile"] = _runtime_profile(
        state=final_state or state,
        nodes=node_profiles,
        total_elapsed_ms=_elapsed_ms(start, completed),
    )
    return final_state


def _node_profile(
    *,
    node: str,
    update: Any,
    sequence: int,
    elapsed_ms: int,
    completed_at_ms: int,
) -> dict[str, Any]:
    update_keys = sorted(str(key) for key in update.keys()) if isinstance(update, dict) else []
    return {
        "node": node,
        "category": _node_category(node),
        "sequence": sequence,
        "elapsed_ms": elapsed_ms,
        "completed_at_ms": completed_at_ms,
        "update_keys": update_keys,
    }


def _runtime_profile(
    *,
    state: Mapping[str, Any],
    nodes: list[dict[str, Any]],
    total_elapsed_ms: int,
) -> dict[str, Any]:
    budget_profile = _runtime_budget_profile(state)
    budget = RUNTIME_LATENCY_BUDGETS[budget_profile]
    trace_events = state.get("trace_events", [])
    fallback_count = _count_trace_markers(trace_events, _is_fallback_marker)
    timeout_count = _count_trace_markers(trace_events, _is_timeout_marker)
    advisor_timeout_count = _count_trace_markers(trace_events, _is_advisor_timeout_marker)
    slowest_nodes = sorted(
        (
            {
                "node": str(node.get("node", "")),
                "category": str(node.get("category", "graph_orchestration")),
                "elapsed_ms": int(node.get("elapsed_ms", 0)),
                "sequence": int(node.get("sequence", 0)),
            }
            for node in nodes
        ),
        key=lambda item: item["elapsed_ms"],
        reverse=True,
    )[:5]
    category_elapsed_ms: dict[str, int] = {}
    category_node_count: dict[str, int] = {}
    for node in nodes:
        category = str(node.get("category", "graph_orchestration"))
        category_elapsed_ms[category] = category_elapsed_ms.get(category, 0) + int(
            node.get("elapsed_ms", 0)
        )
        category_node_count[category] = category_node_count.get(category, 0) + 1
    return {
        "budget_profile": budget_profile,
        "budget": budget,
        "total_elapsed_ms": total_elapsed_ms,
        "node_count": len(nodes),
        "nodes": nodes,
        "slowest_nodes": slowest_nodes,
        "category_elapsed_ms": category_elapsed_ms,
        "category_node_count": category_node_count,
        "fallback_count": fallback_count,
        "timeout_count": timeout_count,
        "advisor_timeout_count": advisor_timeout_count,
        "within_turn_budget": total_elapsed_ms <= budget["turn_ms"],
        "within_node_budget": all(
            int(node.get("elapsed_ms", 0)) <= budget["node_ms"] for node in nodes
        ),
    }


def _runtime_budget_profile(state: Mapping[str, Any]) -> str:
    requested = str(
        state.get("runtime_budget_profile") or state.get("play_profile") or ""
    ).strip()
    if requested in RUNTIME_LATENCY_BUDGETS:
        return requested
    return DEFAULT_RUNTIME_BUDGET_PROFILE


def _node_category(node: str) -> str:
    if node in {
        "retrieve_context_parallel",
        "retrieve_memory",
        "retrieve_content_spans",
        "load_runtime_context",
    }:
        return "retrieval_io"
    if node in {"execute_deterministic_tools", "ensure_resolution_tools"}:
        return "deterministic_tool"
    if node in {"critic_guardrail_with_llm", "critic_guardrail_locally"}:
        return "critic"
    if node in {
        "curate_memory_with_llm",
        "curate_memory_locally",
        "persist_memory_curation",
    }:
        return "memory_curation"
    if node in {
        "route_with_intent_arbiter",
        "run_micro_gates",
        "advise_turn_with_single_llm",
        "advise_rules_with_llm",
        "adjudicate_with_llm",
        "select_scenario_surface_with_llm",
        "direct_scenario_with_llm",
        "narrate_with_llm",
    }:
        return "provider_wait"
    return "graph_orchestration"


def _elapsed_ms(start: float, end: float) -> int:
    return max(0, int(round((end - start) * 1000)))


def _count_trace_markers(value: Any, predicate: Any, *, parent_key: str = "") -> int:
    if isinstance(value, Mapping):
        return sum(
            _count_trace_markers(
                item,
                predicate,
                parent_key=f"{parent_key}.{key}" if parent_key else str(key),
            )
            for key, item in value.items()
        ) + (1 if predicate(parent_key, value) else 0)
    if isinstance(value, list | tuple):
        return sum(_count_trace_markers(item, predicate, parent_key=parent_key) for item in value)
    return 1 if predicate(parent_key, value) else 0


def _is_fallback_marker(parent_key: str, value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"fallback", "deterministic_fallback"} and bool(item):
                return True
    return False


def _is_timeout_marker(parent_key: str, value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return "timed out" in lowered or "timeout" in lowered
    key = parent_key.lower().rsplit(".", maxsplit=1)[-1]
    return key in {"timeout", "timed_out", "timedout", "did_timeout"} and bool(value)


def _is_advisor_timeout_marker(parent_key: str, value: Any) -> bool:
    parent = parent_key.lower()
    if "advisor" not in parent and "gate" not in parent:
        return False
    return _is_timeout_marker(parent_key, value)

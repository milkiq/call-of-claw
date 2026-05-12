from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from trpg_agent.graph.build_turn_graph import (
    AdvisorModelMap,
    build_turn_graph,
    build_turn_graph_with_model,
)
from trpg_agent.graph.state import GraphState


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
    return graph.invoke(state, turn_graph_invoke_config(state))


def stream_turn_graph(
    graph: Any,
    state: GraphState,
    *,
    on_node: Any | None = None,
) -> GraphState:
    final_state: GraphState = {}
    for chunk in graph.stream(
        state,
        turn_graph_invoke_config(state),
        stream_mode="updates",
    ):
        if not isinstance(chunk, dict):
            continue
        for node, update in chunk.items():
            if on_node is not None:
                on_node(str(node), update if isinstance(update, dict) else {})
            if isinstance(update, dict):
                final_state.update(update)
    return final_state

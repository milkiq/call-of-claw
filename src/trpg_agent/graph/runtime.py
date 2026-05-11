from __future__ import annotations

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

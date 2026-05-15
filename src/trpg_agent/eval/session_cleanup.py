from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trpg_agent.graph.runtime import delete_turn_graph_checkpoints
from trpg_agent.memory.store import SqliteStore

TEST_SESSION_ID_PREFIXES = (
    "online-playtest-",
    "eval-durable-turn-replay-",
    "live-eval-",
    "long-play-",
)
TEST_SESSION_EXACT_IDS = frozenset(
    {
        "roll-boundary-smoke",
    }
)


def is_test_session_id(session_id: str) -> bool:
    return session_id in TEST_SESSION_EXACT_IDS or session_id.startswith(
        TEST_SESSION_ID_PREFIXES
    )


def cleanup_sessions(
    *,
    store: SqliteStore,
    sqlite_path: Path,
    session_ids: list[str],
) -> dict[str, Any]:
    target_ids = list(dict.fromkeys(session_id for session_id in session_ids if session_id))
    database_counts = store.delete_sessions(target_ids)
    checkpoint_counts = delete_turn_graph_checkpoints(
        sqlite_path,
        session_ids=target_ids,
    )
    return {
        "session_ids": target_ids,
        "database": database_counts,
        "checkpoints": checkpoint_counts,
    }


def cleanup_known_test_sessions(
    *,
    store: SqliteStore,
    sqlite_path: Path,
) -> dict[str, Any]:
    target_ids = [
        str(session["id"])
        for session in store.list_session_summaries()
        if is_test_session_id(str(session["id"]))
    ]
    return cleanup_sessions(store=store, sqlite_path=sqlite_path, session_ids=target_ids)


def cleanup_metadata(cleanup: dict[str, Any]) -> dict[str, str]:
    return {
        "session_cleanup": "true",
        "session_cleanup_ids": ",".join(cleanup.get("session_ids", [])),
        "session_cleanup_database": json.dumps(
            cleanup.get("database", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "session_cleanup_checkpoints": json.dumps(
            cleanup.get("checkpoints", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }

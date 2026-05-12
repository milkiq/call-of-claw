from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


class SqliteStore:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def migrate(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  id TEXT PRIMARY KEY,
                  ruleset_id TEXT,
                  scenario_id TEXT,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS turns (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  input TEXT NOT NULL,
                  output TEXT NOT NULL,
                  trace_json TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS canon_events (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  turn_id TEXT,
                  type TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS memories (
                  id TEXT PRIMARY KEY,
                  scope TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  text TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS dice_rolls (
                  id TEXT PRIMARY KEY,
                  turn_id TEXT,
                  expression TEXT NOT NULL,
                  result_json TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS eval_runs (
                  id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS session_state (
                  session_id TEXT PRIMARY KEY,
                  state_json TEXT NOT NULL,
                  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS world_patch_applications (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  turn_id TEXT NOT NULL,
                  patches_json TEXT NOT NULL,
                  resulting_state_json TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS advisor_runs (
                  id TEXT PRIMARY KEY,
                  turn_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  prompt_version TEXT NOT NULL,
                  input_hash TEXT NOT NULL,
                  output_json TEXT NOT NULL,
                  attempts_json TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS critic_reports (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  turn_id TEXT NOT NULL,
                  report_json TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS schema_migrations (
                  version INTEGER PRIMARY KEY,
                  description TEXT NOT NULL,
                  applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                  id UNINDEXED,
                  scope UNINDEXED,
                  kind UNINDEXED,
                  text,
                  metadata_json UNINDEXED
                );
                """
            )
            migrations = [
                (1, "initial durable runtime schema"),
                (2, "advisor metrics and production hardening metadata"),
            ]
            for version, description in migrations:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO schema_migrations (version, description)
                    VALUES (?, ?)
                    """,
                    (version, description),
                )

    def schema_version(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(version) AS version
                FROM schema_migrations
                """
            ).fetchone()
        return int(row["version"] or 0) if row else 0

    def list_schema_migrations(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT version, description, applied_at
                FROM schema_migrations
                ORDER BY version
                """
            ).fetchall()
        return [
            {
                "version": row["version"],
                "description": row["description"],
                "applied_at": row["applied_at"],
            }
            for row in rows
        ]

    def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT state_json
                FROM session_state
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["state_json"])

    def set_session_state(self, *, session_id: str, state: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO session_state (session_id, state_json)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  state_json = excluded.state_json,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, json.dumps(state, ensure_ascii=False)),
            )

    def upsert_session(
        self,
        *,
        session_id: str,
        ruleset_id: str | None = None,
        scenario_id: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, ruleset_id, scenario_id)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  ruleset_id = COALESCE(excluded.ruleset_id, sessions.ruleset_id),
                  scenario_id = COALESCE(excluded.scenario_id, sessions.scenario_id),
                  updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, ruleset_id, scenario_id),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, ruleset_id, scenario_id, created_at, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "ruleset_id": row["ruleset_id"],
            "scenario_id": row["scenario_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ruleset_id, scenario_id, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC, id
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "ruleset_id": row["ruleset_id"],
                "scenario_id": row["scenario_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def list_session_summaries(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  sessions.id,
                  sessions.ruleset_id,
                  sessions.scenario_id,
                  sessions.created_at,
                  sessions.updated_at,
                  COUNT(DISTINCT turns.id) AS turn_count,
                  COUNT(DISTINCT memories.id) AS memory_count,
                  CASE WHEN session_state.session_id IS NULL THEN 0 ELSE 1 END AS has_state
                FROM sessions
                LEFT JOIN turns ON turns.session_id = sessions.id
                LEFT JOIN memories ON memories.scope = sessions.id
                LEFT JOIN session_state ON session_state.session_id = sessions.id
                GROUP BY sessions.id
                ORDER BY sessions.updated_at DESC, sessions.id
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "ruleset_id": row["ruleset_id"],
                "scenario_id": row["scenario_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "turn_count": int(row["turn_count"] or 0),
                "memory_count": int(row["memory_count"] or 0),
                "has_state": bool(row["has_state"]),
            }
            for row in rows
        ]

    def delete_sessions(self, session_ids: list[str]) -> dict[str, int]:
        unique_ids = list(dict.fromkeys(session_id for session_id in session_ids if session_id))
        if not unique_ids:
            return self._empty_delete_counts()
        with self.connect() as conn:
            return self._delete_sessions_in_conn(conn, unique_ids)

    def delete_all_sessions(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()
            session_ids = [str(row["id"]) for row in rows]
            counts = (
                self._delete_sessions_in_conn(conn, session_ids)
                if session_ids
                else self._empty_delete_counts()
            )
            counts["memories_fts"] += self._deleted_count(
                conn.execute("DELETE FROM memories_fts")
            )
            counts["memories"] += self._deleted_count(conn.execute("DELETE FROM memories"))
            return counts

    @staticmethod
    def _empty_delete_counts() -> dict[str, int]:
        return {
            "sessions": 0,
            "turns": 0,
            "canon_events": 0,
            "session_state": 0,
            "world_patch_applications": 0,
            "critic_reports": 0,
            "advisor_runs": 0,
            "dice_rolls": 0,
            "memories": 0,
            "memories_fts": 0,
        }

    @staticmethod
    def _deleted_count(cursor: sqlite3.Cursor) -> int:
        return max(0, int(cursor.rowcount or 0))

    def _delete_sessions_in_conn(
        self,
        conn: sqlite3.Connection,
        session_ids: list[str],
    ) -> dict[str, int]:
        counts = self._empty_delete_counts()
        for session_id in session_ids:
            turn_rows = conn.execute(
                """
                SELECT id
                FROM turns
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            turn_ids = [str(row["id"]) for row in turn_rows]
            for turn_id in turn_ids:
                counts["advisor_runs"] += self._deleted_count(
                    conn.execute(
                        "DELETE FROM advisor_runs WHERE turn_id = ?",
                        (turn_id,),
                    )
                )
                counts["dice_rolls"] += self._deleted_count(
                    conn.execute(
                        "DELETE FROM dice_rolls WHERE turn_id = ?",
                        (turn_id,),
                    )
                )

            memory_rows = conn.execute(
                """
                SELECT id
                FROM memories
                WHERE scope = ?
                """,
                (session_id,),
            ).fetchall()
            memory_ids = [str(row["id"]) for row in memory_rows]
            for memory_id in memory_ids:
                counts["memories_fts"] += self._deleted_count(
                    conn.execute(
                        "DELETE FROM memories_fts WHERE id = ?",
                        (memory_id,),
                    )
                )
            counts["memories"] += self._deleted_count(
                conn.execute(
                    "DELETE FROM memories WHERE scope = ?",
                    (session_id,),
                )
            )
            counts["critic_reports"] += self._deleted_count(
                conn.execute(
                    "DELETE FROM critic_reports WHERE session_id = ?",
                    (session_id,),
                )
            )
            counts["world_patch_applications"] += self._deleted_count(
                conn.execute(
                    "DELETE FROM world_patch_applications WHERE session_id = ?",
                    (session_id,),
                )
            )
            counts["session_state"] += self._deleted_count(
                conn.execute(
                    "DELETE FROM session_state WHERE session_id = ?",
                    (session_id,),
                )
            )
            counts["canon_events"] += self._deleted_count(
                conn.execute(
                    "DELETE FROM canon_events WHERE session_id = ?",
                    (session_id,),
                )
            )
            counts["turns"] += self._deleted_count(
                conn.execute(
                    "DELETE FROM turns WHERE session_id = ?",
                    (session_id,),
                )
            )
            counts["sessions"] += self._deleted_count(
                conn.execute(
                    "DELETE FROM sessions WHERE id = ?",
                    (session_id,),
                )
            )
        return counts

    def insert_turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        player_input: str,
        output: str,
        trace: dict[str, Any],
    ) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO turns
                  (id, session_id, input, output, trace_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    player_input,
                    output,
                    json.dumps(trace, ensure_ascii=False),
                ),
            )
            return cursor.rowcount > 0

    def get_turn(self, turn_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, input, output, trace_json, created_at
                FROM turns
                WHERE id = ?
                """,
                (turn_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "input": row["input"],
            "output": row["output"],
            "trace": json.loads(row["trace_json"]),
            "created_at": row["created_at"],
        }

    def list_turns(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, input, output, trace_json, created_at
                FROM turns
                WHERE session_id = ?
                ORDER BY created_at, id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "input": row["input"],
                "output": row["output"],
                "trace": json.loads(row["trace_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_advisor_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, turn_id, role, prompt_version, input_hash, output_json,
                       attempts_json, created_at
                FROM advisor_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "turn_id": row["turn_id"],
            "role": row["role"],
            "prompt_version": row["prompt_version"],
            "input_hash": row["input_hash"],
            "output": json.loads(row["output_json"]),
            "attempts": json.loads(row["attempts_json"]),
            "created_at": row["created_at"],
        }

    def insert_advisor_run_once(
        self,
        *,
        run_id: str,
        turn_id: str,
        role: str,
        prompt_version: str,
        input_hash: str,
        output: dict[str, Any],
        attempts: list[dict[str, Any]],
    ) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO advisor_runs
                  (id, turn_id, role, prompt_version, input_hash, output_json, attempts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    turn_id,
                    role,
                    prompt_version,
                    input_hash,
                    json.dumps(output, ensure_ascii=False),
                    json.dumps(attempts, ensure_ascii=False),
                ),
            )
            return cursor.rowcount > 0

    def list_advisor_runs(self, turn_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, prompt_version, input_hash, output_json, attempts_json,
                       created_at
                FROM advisor_runs
                WHERE turn_id = ?
                ORDER BY created_at, id
                """,
                (turn_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "prompt_version": row["prompt_version"],
                "input_hash": row["input_hash"],
                "output": json.loads(row["output_json"]),
                "attempts": json.loads(row["attempts_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_advisor_runs_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT advisor_runs.id, advisor_runs.turn_id, advisor_runs.role,
                       advisor_runs.prompt_version, advisor_runs.input_hash,
                       advisor_runs.output_json, advisor_runs.attempts_json,
                       advisor_runs.created_at
                FROM advisor_runs
                JOIN turns ON turns.id = advisor_runs.turn_id
                WHERE turns.session_id = ?
                ORDER BY advisor_runs.created_at, advisor_runs.id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "turn_id": row["turn_id"],
                "role": row["role"],
                "prompt_version": row["prompt_version"],
                "input_hash": row["input_hash"],
                "output": json.loads(row["output_json"]),
                "attempts": json.loads(row["attempts_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def commit_session_state_once(
        self,
        *,
        application_id: str,
        session_id: str,
        turn_id: str,
        patches: list[dict[str, Any]],
        resulting_state: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Persist a world patch effect once and return the committed state."""

        patches_json = json.dumps(patches, ensure_ascii=False)
        state_json = json.dumps(resulting_state, ensure_ascii=False)
        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT resulting_state_json
                FROM world_patch_applications
                WHERE id = ?
                """,
                (application_id,),
            ).fetchone()
            if existing:
                return json.loads(existing["resulting_state_json"]), False

            conn.execute(
                """
                INSERT INTO world_patch_applications
                  (id, session_id, turn_id, patches_json, resulting_state_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (application_id, session_id, turn_id, patches_json, state_json),
            )
            conn.execute(
                """
                INSERT INTO session_state (session_id, state_json)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  state_json = excluded.state_json,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, state_json),
            )
        return resulting_state, True

    def list_world_patch_applications(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, turn_id, patches_json, resulting_state_json, created_at
                FROM world_patch_applications
                WHERE session_id = ?
                ORDER BY created_at, id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "turn_id": row["turn_id"],
                "patches": json.loads(row["patches_json"]),
                "resulting_state": json.loads(row["resulting_state_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def insert_canon_event(
        self,
        *,
        event_id: str,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
        turn_id: str | None = None,
    ) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO canon_events
                  (id, session_id, turn_id, type, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    turn_id,
                    event_type,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return cursor.rowcount > 0

    def list_canon_events(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, type, payload_json, created_at
                FROM canon_events
                WHERE session_id = ?
                ORDER BY created_at, id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "type": row["type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def insert_dice_roll(
        self,
        *,
        roll_id: str,
        turn_id: str | None,
        expression: str,
        result: dict[str, Any],
    ) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO dice_rolls
                  (id, turn_id, expression, result_json)
                VALUES (?, ?, ?, ?)
                """,
                (roll_id, turn_id, expression, json.dumps(result, ensure_ascii=False)),
            )
            return cursor.rowcount > 0

    def get_dice_roll(self, roll_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, turn_id, expression, result_json, created_at
                FROM dice_rolls
                WHERE id = ?
                """,
                (roll_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "turn_id": row["turn_id"],
            "expression": row["expression"],
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
        }

    def upsert_memory(
        self,
        *,
        memory_id: str,
        scope: str,
        kind: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, scope, kind, text, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  scope = excluded.scope,
                  kind = excluded.kind,
                  text = excluded.text,
                  metadata_json = excluded.metadata_json,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (memory_id, scope, kind, text, payload),
            )
            conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            conn.execute(
                """
                INSERT INTO memories_fts (id, scope, kind, text, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, scope, kind, text, payload),
            )

    def recall_memories(
        self,
        *,
        query: str,
        scope: str | None = None,
        limit: int = 5,
        include_gm_only: bool = True,
    ) -> list[dict[str, Any]]:
        search = " ".join(token for token in query.replace('"', " ").split() if token)
        row_limit = limit if include_gm_only else max(limit * 4, limit)
        with self.connect() as conn:
            if search:
                rows = conn.execute(
                    """
                    SELECT id, scope, kind, text, metadata_json
                    FROM memories_fts
                    WHERE memories_fts MATCH ?
                      AND (? IS NULL OR scope = ?)
                    LIMIT ?
                    """,
                    (search, scope, scope, row_limit),
                ).fetchall()
                if not rows:
                    rows = conn.execute(
                        """
                        SELECT id, scope, kind, text, metadata_json
                        FROM memories
                        WHERE text LIKE ?
                          AND (? IS NULL OR scope = ?)
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (f"%{search}%", scope, scope, row_limit),
                    ).fetchall()
                if not rows:
                    rows = conn.execute(
                        """
                        SELECT id, scope, kind, text, metadata_json
                        FROM memories
                        WHERE (? IS NULL OR scope = ?)
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (scope, scope, row_limit),
                    ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, scope, kind, text, metadata_json
                    FROM memories
                    WHERE (? IS NULL OR scope = ?)
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (scope, scope, row_limit),
                ).fetchall()
        results = [
            {
                "id": row["id"],
                "scope": row["scope"],
                "kind": row["kind"],
                "text": row["text"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]
        if not include_gm_only:
            results = [
                memory
                for memory in results
                if memory.get("metadata", {}).get("visibility") != "gm_only"
            ]
        return results[:limit]

    def list_memories(self, scope: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, scope, kind, text, metadata_json, created_at, updated_at
                FROM memories
                WHERE (? IS NULL OR scope = ?)
                ORDER BY updated_at, id
                """,
                (scope, scope),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "scope": row["scope"],
                "kind": row["kind"],
                "text": row["text"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def insert_critic_report_once(
        self,
        *,
        report_id: str,
        session_id: str,
        turn_id: str,
        report: dict[str, Any],
    ) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO critic_reports
                  (id, session_id, turn_id, report_json)
                VALUES (?, ?, ?, ?)
                """,
                (report_id, session_id, turn_id, json.dumps(report, ensure_ascii=False)),
            )
            return cursor.rowcount > 0

    def list_critic_reports(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, turn_id, report_json, created_at
                FROM critic_reports
                WHERE session_id = ?
                ORDER BY created_at, id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "turn_id": row["turn_id"],
                "report": json.loads(row["report_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def insert_eval_run(self, *, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs (id, kind, payload_json)
                VALUES (?, ?, ?)
                """,
                (run_id, kind, json.dumps(payload, ensure_ascii=False)),
            )

    def list_eval_runs(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, payload_json, created_at
                FROM eval_runs
                ORDER BY created_at, id
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

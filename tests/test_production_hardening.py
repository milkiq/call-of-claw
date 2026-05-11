from pathlib import Path

from trpg_agent.content.registry import ContentRegistry
from trpg_agent.graph.build_turn_graph import build_turn_graph
from trpg_agent.memory.store import SCHEMA_VERSION, SqliteStore
from trpg_agent.security.redaction import redact_secrets


def test_store_records_schema_migrations(tmp_path) -> None:
    store = SqliteStore(tmp_path / "migration.sqlite")
    store.migrate()

    assert store.schema_version() == SCHEMA_VERSION
    versions = [row["version"] for row in store.list_schema_migrations()]
    assert versions == sorted(versions)
    assert SCHEMA_VERSION in versions


def test_redacts_secret_shapes_in_nested_trace() -> None:
    payload = {
        "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
        "nested": ["Bearer very-secret-token-value"],
    }

    assert redact_secrets(payload) == {
        "api_key": "[REDACTED]",
        "nested": ["[REDACTED]"],
    }


def test_persisted_turn_trace_redacts_model_metadata_secrets(tmp_path) -> None:
    sqlite_path = tmp_path / "trace.sqlite"
    result = build_turn_graph().invoke(
        {
            "player_input": "我检查周围",
            "session_id": "redaction-session",
            "turn_id": "redaction-turn",
            "content_dir": str(Path.cwd() / "content"),
            "sqlite_path": str(sqlite_path),
            "model_metadata": {"api_key": "sk-abcdefghijklmnopqrstuvwxyz"},
        }
    )

    store = SqliteStore(sqlite_path)
    turn = store.get_turn("redaction-turn")
    assert result["final_output"]
    assert turn is not None
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in str(turn["trace"])
    assert "[REDACTED]" in str(turn["trace"])


def test_content_validation_checks_compiled_package_compatibility() -> None:
    registry = ContentRegistry.load(Path.cwd() / "content", Path.cwd())

    assert registry.validate() == []

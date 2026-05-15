from pathlib import Path

from coc.memory.canon import import_canon_jsonl
from coc.memory.store import SqliteStore


def test_canon_import_is_idempotent(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "test.sqlite")
    store.migrate()
    canon_path = tmp_path / "canon.jsonl"
    canon_path.write_text(
        '{"type":"turn","playerAction":"look","narration":"A door is visible."}\n'
        '{"type":"turn","playerAction":"open","narration":"The door opens."}\n',
        encoding="utf-8",
    )

    first = import_canon_jsonl(store, canon_path)
    second = import_canon_jsonl(store, canon_path)

    assert first > 0
    assert second == 0
    assert len(store.list_canon_events("imported-canon")) == first


def test_turn_and_dice_persistence_are_idempotent(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "test.sqlite")
    store.migrate()
    store.upsert_session(session_id="s1", ruleset_id="r1", scenario_id="sc1")

    inserted = store.insert_turn(
        turn_id="t1",
        session_id="s1",
        player_input="look",
        output="door",
        trace={"nodes": []},
    )
    duplicate = store.insert_turn(
        turn_id="t1",
        session_id="s1",
        player_input="look again",
        output="changed",
        trace={"nodes": ["changed"]},
    )
    roll_inserted = store.insert_dice_roll(
        roll_id="r1",
        turn_id="t1",
        expression="1d6",
        result={"expression": "1d6", "rolls": [4], "total": 4, "roll_id": "r1"},
    )
    roll_duplicate = store.insert_dice_roll(
        roll_id="r1",
        turn_id="t1",
        expression="1d6",
        result={"expression": "1d6", "rolls": [1], "total": 1, "roll_id": "r1"},
    )

    assert inserted is True
    assert duplicate is False
    assert len(store.list_turns("s1")) == 1
    assert roll_inserted is True
    assert roll_duplicate is False
    assert store.get_dice_roll("r1")["result"]["total"] == 4


def test_memory_recall_uses_sqlite_store(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "test.sqlite")
    store.migrate()
    store.upsert_memory(
        memory_id="m1",
        scope="s1",
        kind="semantic",
        text="The old door is locked.",
        metadata={"source": "test"},
    )

    hits = store.recall_memories(query="door", scope="s1")

    assert len(hits) == 1
    assert hits[0]["metadata"]["source"] == "test"


def test_memory_recall_falls_back_to_substring_for_cjk_text(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "test.sqlite")
    store.migrate()
    store.upsert_memory(
        memory_id="m1",
        scope="s1",
        kind="player_preference",
        text="以后请简短回顾",
        metadata={"visibility": "public"},
    )

    hits = store.recall_memories(
        query="简短回顾",
        scope="s1",
        include_gm_only=False,
    )

    assert [hit["text"] for hit in hits] == ["以后请简短回顾"]


def test_memory_recall_falls_back_to_recent_session_memories(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "test.sqlite")
    store.migrate()
    store.upsert_memory(
        memory_id="m1",
        scope="s1",
        kind="unresolved_thread",
        text="The station clock is advancing.",
        metadata={"visibility": "public"},
    )

    hits = store.recall_memories(query="unrelated safety channel", scope="s1")

    assert [hit["text"] for hit in hits] == ["The station clock is advancing."]

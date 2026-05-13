import json
import sqlite3

from typer.testing import CliRunner

from trpg_agent.app.cli import app
from trpg_agent.graph.runtime import checkpoint_path_for
from trpg_agent.memory.store import SqliteStore


def _insert_checkpoint_rows(sqlite_path, thread_ids: list[str]) -> None:
    checkpoint_path = checkpoint_path_for(sqlite_path)
    with sqlite3.connect(checkpoint_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
              thread_id TEXT NOT NULL,
              checkpoint_ns TEXT NOT NULL DEFAULT '',
              checkpoint_id TEXT NOT NULL,
              parent_checkpoint_id TEXT,
              type TEXT,
              checkpoint BLOB,
              metadata BLOB,
              PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );

            CREATE TABLE IF NOT EXISTS writes (
              thread_id TEXT NOT NULL,
              checkpoint_ns TEXT NOT NULL DEFAULT '',
              checkpoint_id TEXT NOT NULL,
              task_id TEXT NOT NULL,
              idx INTEGER NOT NULL,
              channel TEXT NOT NULL,
              type TEXT,
              blob BLOB,
              PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
            """
        )
        for thread_id in thread_ids:
            checkpoint_id = f"{thread_id}:checkpoint"
            conn.execute(
                """
                INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id)
                VALUES (?, '', ?)
                """,
                (thread_id, checkpoint_id),
            )
            conn.execute(
                """
                INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel)
                VALUES (?, '', ?, 'task', 0, 'state')
                """,
                (thread_id, checkpoint_id),
            )


def _checkpoint_thread_ids(sqlite_path) -> list[str]:
    checkpoint_path = checkpoint_path_for(sqlite_path)
    with sqlite3.connect(checkpoint_path) as conn:
        return [
            str(row[0])
            for row in conn.execute("SELECT thread_id FROM checkpoints ORDER BY thread_id")
        ]


def test_session_cli_start_play_recap_inspect_and_export(tmp_path) -> None:
    runner = CliRunner()
    env = {"TRPG_AGENT_SQLITE": str(tmp_path / "cli.sqlite")}

    start = runner.invoke(
        app,
        [
            "session",
            "start",
            "--session-id",
            "cli-session",
            "--ruleset-id",
            "sum_target_smoke",
            "--scenario-id",
            "storm_watch_survival",
            "--reset",
        ],
        env=env,
    )
    assert start.exit_code == 0
    assert "session: cli-session" in start.output

    play = runner.invoke(
        app,
        [
            "play",
            "--local",
            "--session-id",
            "cli-session",
            "--input",
            "风险行动 2d6",
            "--ruleset-id",
            "sum_target_smoke",
            "--scenario-id",
            "storm_watch_survival",
            "--json",
        ],
        env=env,
    )
    assert play.exit_code == 0
    play_payload = json.loads(play.output)
    assert play_payload["turn_plan"]["decision"] == "risky_action"
    assert play_payload["tool_results"]

    recap = runner.invoke(
        app,
        ["session", "recap", "--session-id", "cli-session", "--limit", "1"],
        env=env,
    )
    assert recap.exit_code == 0
    assert "recent turns:" in recap.output

    inspect_public = runner.invoke(
        app,
        ["session", "inspect", "--session-id", "cli-session"],
        env=env,
    )
    assert inspect_public.exit_code == 0
    public_payload = json.loads(inspect_public.output)
    assert "gm_traces" not in public_payload
    assert public_payload["world_projection"]["scene"]["public_summary"]

    export_path = tmp_path / "session-export.json"
    export = runner.invoke(
        app,
        [
            "session",
            "export",
            "--session-id",
            "cli-session",
            "--output",
            str(export_path),
            "--include-gm",
        ],
        env=env,
    )
    assert export.exit_code == 0
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["gm_traces"]

    quality = runner.invoke(
        app,
        ["session", "quality-report", "--session-id", "cli-session"],
        env=env,
    )
    assert quality.exit_code == 0
    assert "resolver_bypass_count: 0" in quality.output


def test_eval_observation_report_cli_outputs_json(tmp_path) -> None:
    runner = CliRunner()
    env = {"TRPG_AGENT_SQLITE": str(tmp_path / "observation.sqlite")}

    result = runner.invoke(
        app,
        ["eval", "observation-report", "--source", "store", "--json"],
        env=env,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["source"] == "store"
    assert "advisor_event_count" in payload


def test_session_cli_list_includes_counts_and_json(tmp_path) -> None:
    runner = CliRunner()
    env = {"TRPG_AGENT_SQLITE": str(tmp_path / "session-list.sqlite")}
    start = runner.invoke(
        app,
        [
            "session",
            "start",
            "--session-id",
            "list-session",
            "--ruleset-id",
            "sum_target_smoke",
            "--scenario-id",
            "storm_watch_survival",
        ],
        env=env,
    )
    assert start.exit_code == 0

    listed = runner.invoke(app, ["session", "list"], env=env)
    assert listed.exit_code == 0
    assert "list-session" in listed.output
    assert "turns=0" in listed.output
    assert "memories=0" in listed.output
    assert "state=yes" in listed.output

    listed_json = runner.invoke(app, ["session", "list", "--json"], env=env)
    assert listed_json.exit_code == 0
    payload = json.loads(listed_json.output)
    assert payload["sessions"][0]["id"] == "list-session"
    assert payload["sessions"][0]["turn_count"] == 0
    assert payload["sessions"][0]["has_state"] is True


def test_session_cli_delete_single_session_preserves_eval_runs(tmp_path) -> None:
    runner = CliRunner()
    sqlite_path = tmp_path / "session-delete.sqlite"
    env = {"TRPG_AGENT_SQLITE": str(sqlite_path)}
    for session_id in ["delete-me", "keep-me"]:
        result = runner.invoke(
            app,
            [
                "session",
                "start",
                "--session-id",
                session_id,
                "--ruleset-id",
                "sum_target_smoke",
                "--scenario-id",
                "storm_watch_survival",
            ],
            env=env,
        )
        assert result.exit_code == 0

    store = SqliteStore(sqlite_path)
    store.insert_eval_run(run_id="eval-1", kind="test", payload={"ok": True})
    store.upsert_memory(
        memory_id="delete-me:memory:1",
        scope="delete-me",
        kind="episodic_summary",
        text="delete me",
    )
    store.upsert_memory(
        memory_id="keep-me:memory:1",
        scope="keep-me",
        kind="episodic_summary",
        text="keep me",
    )
    store.insert_turn(
        turn_id="delete-turn",
        session_id="delete-me",
        player_input="look",
        output="narration",
        trace={"nodes": ["test"]},
    )
    store.insert_canon_event(
        event_id="delete-canon",
        session_id="delete-me",
        turn_id="delete-turn",
        event_type="turn",
        payload={"ok": True},
    )
    store.insert_dice_roll(
        roll_id="delete-roll",
        turn_id="delete-turn",
        expression="1d6",
        result={"total": 4},
    )
    store.insert_advisor_run_once(
        run_id="delete-advisor",
        turn_id="delete-turn",
        role="intent_arbiter",
        prompt_version="test",
        input_hash="hash",
        output={"ok": True},
        attempts=[],
    )
    store.insert_critic_report_once(
        report_id="delete-critic",
        session_id="delete-me",
        turn_id="delete-turn",
        report={"approved": True},
    )
    store.commit_session_state_once(
        application_id="delete-world-patch",
        session_id="delete-me",
        turn_id="delete-turn",
        patches=[],
        resulting_state={"test": True},
    )
    _insert_checkpoint_rows(sqlite_path, ["delete-me:delete-turn", "keep-me:turn"])

    deleted = runner.invoke(
        app,
        ["session", "delete", "--session-id", "delete-me", "--yes", "--json"],
        env=env,
    )
    assert deleted.exit_code == 0
    payload = json.loads(deleted.output)
    assert payload["deleted"] is True
    assert payload["session_ids"] == ["delete-me"]
    assert payload["database"]["sessions"] == 1
    assert store.get_session("delete-me") is None
    assert store.get_session("keep-me") is not None
    assert store.get_session_state("delete-me") is None
    assert store.list_turns("delete-me") == []
    assert store.list_canon_events("delete-me") == []
    assert store.get_dice_roll("delete-roll") is None
    assert store.get_advisor_run("delete-advisor") is None
    assert store.list_critic_reports("delete-me") == []
    assert store.list_world_patch_applications("delete-me") == []
    assert store.list_memories(scope="delete-me") == []
    assert len(store.list_memories(scope="keep-me")) == 1
    assert store.list_eval_runs()[0]["id"] == "eval-1"
    assert _checkpoint_thread_ids(sqlite_path) == ["keep-me:turn"]


def test_session_cli_delete_all_sessions_preserves_eval_runs(tmp_path) -> None:
    runner = CliRunner()
    sqlite_path = tmp_path / "session-delete-all.sqlite"
    env = {"TRPG_AGENT_SQLITE": str(sqlite_path)}
    for session_id in ["one", "two"]:
        result = runner.invoke(
            app,
            [
                "session",
                "start",
                "--session-id",
                session_id,
                "--ruleset-id",
                "sum_target_smoke",
                "--scenario-id",
                "storm_watch_survival",
            ],
            env=env,
        )
        assert result.exit_code == 0
    store = SqliteStore(sqlite_path)
    store.insert_eval_run(run_id="eval-1", kind="test", payload={"ok": True})
    store.upsert_memory(
        memory_id="orphan-memory",
        scope="orphan",
        kind="episodic_summary",
        text="stale memory",
    )
    _insert_checkpoint_rows(sqlite_path, ["one:turn", "two:turn"])

    deleted = runner.invoke(
        app,
        ["session", "delete", "--all", "--yes", "--json"],
        env=env,
    )
    assert deleted.exit_code == 0
    payload = json.loads(deleted.output)
    assert payload["deleted"] is True
    assert payload["all"] is True
    assert sorted(payload["session_ids"]) == ["one", "two"]
    assert store.list_sessions() == []
    assert store.list_memories() == []
    assert store.list_eval_runs()[0]["id"] == "eval-1"
    assert _checkpoint_thread_ids(sqlite_path) == []


def test_eval_long_play_cli(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["eval", "long-play", "--turns", "6", "--session-id", "cli-long-play"],
        env={"TRPG_AGENT_SQLITE": str(tmp_path / "long-play.sqlite")},
    )

    assert result.exit_code == 0
    assert "long_play: 1/1 passed" in result.output


def test_eval_release_gates_cli(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["eval", "release-gates", "--long-play-turns", "6"],
        env={"TRPG_AGENT_SQLITE": str(tmp_path / "release.sqlite")},
    )

    assert result.exit_code == 0
    assert "release_gates:" in result.output


def test_eval_advisor_metrics_cli_summarizes_and_compares(tmp_path) -> None:
    runner = CliRunner()
    sqlite_path = tmp_path / "metrics.sqlite"
    env = {"TRPG_AGENT_SQLITE": str(sqlite_path)}
    store = SqliteStore(sqlite_path)
    store.migrate()
    for session_id, response_chars in [("legacy", 1000), ("compact", 500)]:
        store.upsert_session(session_id=session_id, ruleset_id="r", scenario_id="s")
        turn_id = f"{session_id}-turn"
        store.insert_turn(
            turn_id=turn_id,
            session_id=session_id,
            player_input="act",
            output="out",
            trace={},
        )
        store.insert_advisor_run_once(
            run_id=f"{session_id}-advisor",
            turn_id=turn_id,
            role="intent_arbiter",
            prompt_version="intent-arbiter-v5",
            input_hash="hash",
            output={"ok": True},
            attempts=[
                {"phase": "initial", "raw_output": "x" * response_chars},
                {
                    "phase": "metrics",
                    "elapsed_ms": "1000",
                    "estimated_prompt_chars": "2000",
                    "estimated_response_chars": str(response_chars),
                    "attempt_count": "1",
                },
            ],
        )

    listed = runner.invoke(
        app,
        ["eval", "advisor-metrics", "--session-id", "legacy", "--json"],
        env=env,
    )
    assert listed.exit_code == 0
    payload = json.loads(listed.output)
    assert payload["advisor_run_count"] == 1
    assert payload["totals"]["avg_response_chars"] == 1000

    compared = runner.invoke(
        app,
        [
            "eval",
            "advisor-metrics",
            "--compare",
            "legacy",
            "--compare",
            "compact",
            "--json",
        ],
        env=env,
    )
    assert compared.exit_code == 0
    comparison = json.loads(compared.output)
    assert comparison["delta"]["avg_response_chars_reduction_pct"] == 50.0


def test_play_help_lists_profile_and_hides_experiment_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["play", "--help"])

    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "--single-turn-advisor" not in result.output
    assert "--parallel-review" not in result.output
    assert "--advisor-contracts" not in result.output
    assert "--micro-gates" not in result.output
    assert "--use-llm" not in result.output
    assert "--progress" in result.output
    assert "--no-progress" in result.output


def test_play_profile_resolver_sets_expected_flags() -> None:
    from trpg_agent.app.cli import _resolve_play_profile

    fast = _resolve_play_profile("fast")
    assert fast.use_llm is True
    assert fast.micro_gates is True
    assert fast.parallel_review is True
    assert fast.advisor_contracts == "compact"
    assert fast.runtime_budget_profile == "fast"

    local = _resolve_play_profile("fast", local=True)
    assert local.use_llm is False
    assert local.micro_gates is False
    assert local.parallel_review is False
    assert local.advisor_contracts == "legacy"

    overridden = _resolve_play_profile("balanced", micro_gates=True, advisor_contracts="compact")
    assert overridden.micro_gates is True
    assert overridden.advisor_contracts == "compact"


def test_online_playtest_help_keeps_eval_experiment_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "online-playtest", "--help"])

    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "--single-turn-advis" in result.output
    assert "--micro-gates" in result.output
    assert "--parallel-review" in result.output
    assert "--advisor-contracts" in result.output


def test_interactive_play_creates_character_and_prints_resume(tmp_path) -> None:
    runner = CliRunner()
    sqlite_path = tmp_path / "interactive.sqlite"
    result = runner.invoke(
        app,
        [
            "play",
            "--local",
            "--session-id",
            "interactive-cli",
            "--ruleset-id",
            "lasers_feelings_smoke",
            "--scenario-id",
            "crystal_stop_singing_smoke",
        ],
        input="阿岚\n冷静的导航员\n5\n救回船长\n她\n/quit\n",
        env={"TRPG_AGENT_SQLITE": str(sqlite_path)},
    )

    assert result.exit_code == 0
    assert "session-id: interactive-cli" in result.output
    assert "resume: trpg play --session-id interactive-cli" in result.output
    store = SqliteStore(sqlite_path)
    state = store.get_session_state("interactive-cli")
    character_context = state["character_context"]
    assert character_context["number"] == 5
    assert character_context["player_character"]["name"] == "阿岚"
    assert character_context["player_character"]["concept"] == "冷静的导航员"


def test_interactive_play_resumes_existing_character(tmp_path) -> None:
    runner = CliRunner()
    sqlite_path = tmp_path / "interactive-resume.sqlite"
    env = {"TRPG_AGENT_SQLITE": str(sqlite_path)}
    first = runner.invoke(
        app,
        [
            "play",
            "--local",
            "--session-id",
            "interactive-resume",
            "--ruleset-id",
            "lasers_feelings_smoke",
            "--scenario-id",
            "crystal_stop_singing_smoke",
        ],
        input="阿岚\n冷静的导航员\n4\n救回船长\n她\n我检查广播塔\n/quit\n",
        env=env,
    )
    assert first.exit_code == 0

    second = runner.invoke(
        app,
        ["play", "--local", "--session-id", "interactive-resume"],
        input="/quit\n",
        env=env,
    )

    assert second.exit_code == 0
    assert "resuming character: 阿岚" in second.output
    assert "你的角色叫什么" not in second.output


def test_interactive_play_adds_character_to_existing_turn_session(tmp_path) -> None:
    runner = CliRunner()
    sqlite_path = tmp_path / "existing-turn.sqlite"
    env = {"TRPG_AGENT_SQLITE": str(sqlite_path)}
    one_turn = runner.invoke(
        app,
        [
            "play",
            "--local",
            "--session-id",
            "existing-turn",
            "--ruleset-id",
            "lasers_feelings_smoke",
            "--scenario-id",
            "crystal_stop_singing_smoke",
            "--input",
            "我观察广播塔",
        ],
        env=env,
    )
    assert one_turn.exit_code == 0
    store = SqliteStore(sqlite_path)
    assert "player_character" not in store.get_session_state("existing-turn").get(
        "character_context",
        {},
    )

    interactive = runner.invoke(
        app,
        ["play", "--local", "--session-id", "existing-turn"],
        input="阿岚\n冷静的导航员\n4\n救回船长\n她\n/quit\n",
        env=env,
    )

    assert interactive.exit_code == 0
    assert "你的角色叫什么" in interactive.output
    state = store.get_session_state("existing-turn")
    assert state["character_context"]["player_character"]["name"] == "阿岚"

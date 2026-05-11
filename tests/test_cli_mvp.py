import json

from typer.testing import CliRunner

from trpg_agent.app.cli import app
from trpg_agent.memory.store import SqliteStore


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


def test_play_help_lists_experiment_and_progress_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["play", "--help"])

    assert result.exit_code == 0
    assert "--single-turn-advisor" in result.output
    assert "--parallel-review" in result.output
    assert "--progress" in result.output
    assert "--no-progress" in result.output


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

from pathlib import Path

from trpg_agent.app.config import AppConfig
from trpg_agent.eval.playtest import (
    PlaytestMetrics,
    _findings_from_metrics,
    build_session_quality_summary,
    run_scripted_long_play,
)
from trpg_agent.memory.store import SqliteStore


def _config(tmp_path):
    root = Path.cwd()
    return AppConfig(
        root_dir=root,
        content_dir=root / "content",
        seeds_dir=root / "seeds",
        data_dir=tmp_path,
        sqlite_path=tmp_path / "playtest.sqlite",
        langsmith_tracing=False,
        langsmith_project="test",
    )


def test_scripted_long_play_replays_without_duplicates(tmp_path) -> None:
    config = _config(tmp_path)

    result = run_scripted_long_play(
        config,
        turns=12,
        session_id="test-long-play",
        persist=False,
    )

    assert result.failed == 0
    assert result.metadata["persisted_turns"] == "12"
    assert result.metadata["replay_restored"] == "True"
    assert result.metadata["resolver_bypass_count"] == "0"


def test_session_quality_summary_uses_persisted_metrics(tmp_path) -> None:
    config = _config(tmp_path)
    run_scripted_long_play(config, turns=5, session_id="summary-session", persist=False)

    summary = build_session_quality_summary(
        SqliteStore(config.sqlite_path),
        "summary-session",
    )

    assert summary["turns"] == 5
    assert summary["resolver_bypass_count"] == 0
    assert "latest_output" in summary


def test_50_turn_long_play_scores_repetition_hooks_and_memory_qa(tmp_path) -> None:
    config = _config(tmp_path)

    result = run_scripted_long_play(
        config,
        turns=50,
        session_id="quality-long-play",
        persist=False,
    )

    assert result.failed == 0
    assert result.metadata["consecutive_repeated_outputs"] == "0"
    assert float(result.metadata["max_repeated_output_ratio"]) <= 0.25
    assert int(result.metadata["unresolved_hook_count"]) >= 1
    assert float(result.metadata["unresolved_hook_quality"]) >= 0.8
    assert int(result.metadata["memory_qa_checks"]) >= 3
    assert float(result.metadata["memory_qa_accuracy"]) >= 0.8


def test_long_play_quality_findings_cover_repetition_hooks_and_memory_qa() -> None:
    metrics = PlaytestMetrics(
        session_id="bad-session",
        requested_turns=50,
        persisted_turns=50,
        canon_events=50,
        memories=0,
        world_patch_applications=0,
        critic_reports=50,
        replay_restored=True,
        consecutive_repeated_outputs=3,
        max_repeated_output_ratio=0.5,
        unresolved_hook_count=0,
        unresolved_hook_quality=0.0,
        memory_qa_checks=2,
        memory_qa_passed=1,
        memory_qa_accuracy=0.5,
        trace_node_coverage={
            "load_runtime_context": 50,
            "retrieve_memory": 50,
            "retrieve_content_spans": 50,
            "apply_world_patch_results": 50,
            "critic_guardrail_locally": 50,
        },
    )

    case_ids = {finding.case_id for finding in _findings_from_metrics(metrics)}

    assert "long-play-consecutive-repetition" in case_ids
    assert "long-play-repeated-content" in case_ids
    assert "long-play-missing-hooks" in case_ids
    assert "long-play-memory-qa-coverage" in case_ids
    assert "long-play-memory-qa-accuracy" in case_ids

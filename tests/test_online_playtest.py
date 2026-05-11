from __future__ import annotations

from types import SimpleNamespace

from trpg_agent.eval.online_playtest import (
    _offers_pending_question_opportunity,
    _online_findings_from_metrics,
    _online_should_run_llm_judge,
    _online_smoke_fast_path,
    _policy_player_action,
    _public_state_from_trace,
    _sample_transcript,
    _transcript_markdown,
)
from trpg_agent.eval.scorecard import EvalResult, EvalScorecard


def test_sample_transcript_keeps_opening_periodic_and_ending_turns() -> None:
    transcript = [
        {"turn": index, "player": f"p{index}", "gm": f"g{index}"}
        for index in range(1, 101)
    ]

    sampled = _sample_transcript(transcript)
    sampled_turns = [turn["turn"] for turn in sampled]

    assert sampled_turns[:8] == list(range(1, 9))
    assert 10 in sampled_turns
    assert 50 in sampled_turns
    assert sampled_turns[-12:] == list(range(89, 101))
    assert len(sampled_turns) == len(set(sampled_turns))


def test_public_state_from_trace_strips_private_runtime_fields() -> None:
    trace = [
        {
            "world_projection": {
                "active_scene": "scene_1",
                "clock": {"value": 1, "max": 3},
                "revealed_facts": ["visible fact"],
                "known_clues": ["public clue"],
                "scene": {"public_summary": "Visible scene."},
                "npc_stance": {"hidden": "private"},
            }
        }
    ]

    public_state = _public_state_from_trace(trace)

    assert public_state == {
        "active_scene": "scene_1",
        "clock": {"value": 1, "max": 3},
        "revealed_facts": ["visible fact"],
        "known_clues": ["public clue"],
        "scene": {"public_summary": "Visible scene."},
    }


def test_policy_player_action_uses_visible_scene_pressure() -> None:
    action = _policy_player_action(
        turn_number=2,
        current_gm_output="停泊入口就在前方，异常声音正在干扰自动流程。",
        transcript=[],
        public_state={"clock": {"value": 2, "max": 3}},
    )

    assert "异常信号" in action or "靠港" in action or "阻止" in action


def test_policy_player_uses_pending_question_opportunity() -> None:
    action = _policy_player_action(
        turn_number=3,
        current_gm_output="你还有一个待使用的规则机会。你可以向GM问一个关于当前局势的问题。",
        transcript=[],
        public_state={},
    )

    assert _offers_pending_question_opportunity("You may ask the GM one question.")
    assert action.startswith("我问")
    assert "准备充分" in action


def test_online_repetition_gate_ignores_tiny_smoke_runs() -> None:
    metrics = SimpleNamespace(
        requested_turns=2,
        persisted_turns=2,
        replay_restored=True,
        resolver_bypass_count=0,
        consecutive_repeated_outputs=0,
        max_repeated_output_ratio=0.5,
        memory_qa_accuracy=1.0,
        memory_qa_passed=1,
        memory_qa_checks=1,
    )

    assert _online_findings_from_metrics(metrics) == []


def test_online_auto_judge_skips_short_policy_smoke() -> None:
    assert _online_should_run_llm_judge(
        judge_mode="auto",
        player_mode="policy",
        turns=5,
    ) is False
    assert _online_should_run_llm_judge(
        judge_mode="auto",
        player_mode="llm",
        turns=5,
    ) is True
    assert _online_should_run_llm_judge(
        judge_mode="llm",
        player_mode="policy",
        turns=2,
    ) is True
    assert _online_smoke_fast_path(player_mode="policy", turns=5) is True
    assert _online_smoke_fast_path(player_mode="policy", turns=6) is False


def test_transcript_markdown_contains_dialogue_and_scorecard() -> None:
    result = EvalResult(
        run_id="run",
        kind="online_playtest",
        total=2,
        passed=2,
        findings=[],
        scorecard=EvalScorecard(
            rules_correctness=5,
            fictional_authority=5,
            continuity=5,
            player_agency=5,
            pacing=5,
            progressive_disclosure=5,
            memory_behavior=5,
            narration_quality=5,
        ),
        metadata={},
    )

    markdown = _transcript_markdown(
        run_id="run",
        session_id="session",
        transcript=[{"turn": 1, "player": "I inspect.", "gm": "You see a door."}],
        result=result,
    )

    assert "# Online Playtest Transcript: run" in markdown
    assert "**Player:** I inspect." in markdown
    assert "**GM:** You see a door." in markdown
    assert "No blocking findings" in markdown

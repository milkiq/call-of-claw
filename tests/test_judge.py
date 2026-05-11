from langchain_core.language_models.fake_chat_models import FakeListChatModel

from trpg_agent.eval.judge import JudgeReport, run_llm_judge, run_static_quality_gate
from trpg_agent.eval.scorecard import EvalFinding, EvalScorecard


def test_static_quality_gate_finds_empty_output() -> None:
    result = run_static_quality_gate(output="")

    assert result.failed == 1
    assert result.findings[0].dimension == "narration_quality"


def test_judge_report_schema() -> None:
    report = JudgeReport(
        summary="ok",
        scorecard=EvalScorecard(),
        findings=[
            EvalFinding(
                case_id="case",
                dimension="player_agency",
                severity="medium",
                message="issue",
            )
        ],
    )

    assert report.findings[0].suggested_area == "eval"


def test_llm_judge_repairs_malformed_json() -> None:
    model = FakeListChatModel(
        responses=[
            "not json",
            """
            {
              "summary": "ok",
              "scorecard": {
                "rules_correctness": 5,
                "fictional_authority": 5,
                "continuity": 5,
                "player_agency": 5,
                "pacing": 5,
                "progressive_disclosure": 5,
                "memory_behavior": 5,
                "narration_quality": 5,
                "trace_explainability": 5
              },
              "findings": []
            }
            """,
        ]
    )

    result = run_llm_judge(
        model,
        transcript=[{"turn_id": "t1", "player": "look", "gm": "A door."}],
        trace=[],
        evidence=[],
    )

    assert result.failed == 0
    assert result.metadata["structured_attempts"] == "2"


def test_llm_judge_filters_positive_observations_from_findings() -> None:
    model = FakeListChatModel(
        responses=[
            """
            {
              "summary": "mostly ok",
              "scorecard": {
                "rules_correctness": 4,
                "fictional_authority": 5,
                "continuity": 5,
                "player_agency": 5,
                "pacing": 5,
                "progressive_disclosure": 5,
                "memory_behavior": 5,
                "narration_quality": 4,
                "trace_explainability": 4
              },
              "findings": [
                {
                  "case_id": "positive",
                  "dimension": "rules_correctness",
                  "severity": "low",
                  "message": "Rules were correctly applied in this turn.",
                  "evidence": "",
                  "suggested_area": "eval"
                },
                {
                  "case_id": "issue",
                  "dimension": "narration_quality",
                  "severity": "medium",
                  "message": "Dice result is missing from narration.",
                  "evidence": "",
                  "suggested_area": "graph.narration"
                }
              ]
            }
            """
        ]
    )

    result = run_llm_judge(
        model,
        transcript=[{"turn_id": "t1", "player": "force", "gm": "It opens."}],
        trace=[],
        evidence=[],
    )

    assert [finding.case_id for finding in result.findings] == ["issue"]
    assert result.metadata["filtered_non_actionable_findings"] == "1"

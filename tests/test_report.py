from trpg_agent.eval.report import build_quality_report
from trpg_agent.eval.scorecard import EvalFinding, EvalResult, score_from_findings


def test_quality_report_aggregates_eval_results() -> None:
    finding = EvalFinding(
        case_id="case-1",
        dimension="narration_quality",
        severity="medium",
        message="Too terse.",
        suggested_area="graph.output",
    )
    result = EvalResult(
        run_id="run-1",
        kind="test",
        total=2,
        passed=1,
        findings=[finding],
        scorecard=score_from_findings([finding]),
    )

    report = build_quality_report([result])

    assert report.total_cases == 2
    assert report.failed_cases == 1
    assert report.findings_by_area == {"graph.output": 1}
    assert "Cases: 1/2 passed" in report.to_markdown()


def test_quality_report_includes_score_movement() -> None:
    first = EvalResult(
        run_id="run-1",
        kind="test",
        total=1,
        passed=0,
        scorecard=score_from_findings(
            [
                EvalFinding(
                    case_id="case-1",
                    dimension="generic_architecture_compliance",
                    severity="medium",
                    message="Architecture issue.",
                )
            ]
        ),
    )
    second = EvalResult(run_id="run-2", kind="test", total=1, passed=1)

    report = build_quality_report([first, second])

    assert report.score_movement["generic_architecture_compliance"] > 0
    assert "## Score Movement" in report.to_markdown()

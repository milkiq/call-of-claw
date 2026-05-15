from coc.eval.roadmap import derive_roadmap_from_results
from coc.eval.scorecard import EvalFinding, EvalResult


def test_roadmap_derives_items_from_findings() -> None:
    result = EvalResult(
        run_id="run",
        kind="test",
        total=1,
        passed=0,
        findings=[
            EvalFinding(
                case_id="case-1",
                dimension="progressive_disclosure",
                severity="critical",
                message="leak",
                suggested_area="retrieval.visibility",
            )
        ],
    )

    roadmap = derive_roadmap_from_results([result])

    assert roadmap.items[0].priority == "P0"
    assert roadmap.items[0].suggested_area == "retrieval.visibility"

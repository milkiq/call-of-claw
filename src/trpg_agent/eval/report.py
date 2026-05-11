from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from trpg_agent.eval.scorecard import EvalResult, EvalScorecard
from trpg_agent.memory.store import SqliteStore


class QualityReport(BaseModel):
    source_run_ids: list[str] = Field(default_factory=list)
    total_cases: int = 0
    passed_cases: int = 0
    findings_by_area: dict[str, int] = Field(default_factory=dict)
    findings_by_severity: dict[str, int] = Field(default_factory=dict)
    average_scorecard: EvalScorecard = Field(default_factory=EvalScorecard)
    score_movement: dict[str, int] = Field(default_factory=dict)

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    def to_markdown(self) -> str:
        lines = [
            "# TRPG Agent Quality Report",
            "",
            f"Runs: {len(self.source_run_ids)}",
            f"Cases: {self.passed_cases}/{self.total_cases} passed",
            "",
            "## Average Scorecard",
        ]
        for key, value in self.average_scorecard.model_dump().items():
            lines.append(f"- {key}: {value}/5")
        if self.score_movement:
            lines.extend(["", "## Score Movement"])
            for key, delta in sorted(self.score_movement.items()):
                sign = "+" if delta > 0 else ""
                lines.append(f"- {key}: {sign}{delta}")
        lines.append("")
        lines.append("## Findings")
        if not self.findings_by_area:
            lines.append("No failing findings in selected runs.")
        else:
            lines.append("By severity:")
            for severity, count in sorted(self.findings_by_severity.items()):
                lines.append(f"- {severity}: {count}")
            lines.append("")
            lines.append("By area:")
            for area, count in sorted(self.findings_by_area.items()):
                lines.append(f"- {area}: {count}")
        return "\n".join(lines)


def build_quality_report(results: list[EvalResult]) -> QualityReport:
    if not results:
        return QualityReport()

    area_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    score_totals: defaultdict[str, int] = defaultdict(int)
    for result in results:
        for finding in result.findings:
            area_counts[finding.suggested_area] += 1
            severity_counts[finding.severity] += 1
        for key, value in result.scorecard.model_dump().items():
            score_totals[key] += int(value)

    divisor = len(results)
    averages = {
        key: max(1, min(5, round(total / divisor)))
        for key, total in score_totals.items()
    }
    first_scores = results[0].scorecard.model_dump()
    last_scores = results[-1].scorecard.model_dump()
    movement = {
        key: int(last_scores.get(key, 0)) - int(first_scores.get(key, 0))
        for key in sorted(set(first_scores) | set(last_scores))
        if int(last_scores.get(key, 0)) - int(first_scores.get(key, 0)) != 0
    }
    return QualityReport(
        source_run_ids=[result.run_id for result in results],
        total_cases=sum(result.total for result in results),
        passed_cases=sum(result.passed for result in results),
        findings_by_area=dict(area_counts),
        findings_by_severity=dict(severity_counts),
        average_scorecard=EvalScorecard.model_validate(averages),
        score_movement=movement,
    )


def build_quality_report_from_store(store: SqliteStore, *, limit: int = 20) -> QualityReport:
    rows = store.list_eval_runs()
    selected = rows[-limit:] if limit > 0 else rows
    results = [
        EvalResult.model_validate(row["payload"])
        for row in selected
        if isinstance(row.get("payload"), dict)
    ]
    return build_quality_report(results)

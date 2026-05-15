from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

EvalDimension = Literal[
    "rules_correctness",
    "fictional_authority",
    "continuity",
    "player_agency",
    "pacing",
    "progressive_disclosure",
    "memory_behavior",
    "narration_quality",
    "trace_explainability",
    "generic_architecture_compliance",
    "infrastructure",
]


class EvalFinding(BaseModel):
    case_id: str
    dimension: EvalDimension
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    evidence: str = ""
    suggested_area: str = "eval"


class EvalScorecard(BaseModel):
    rules_correctness: int = Field(default=5, ge=1, le=5)
    fictional_authority: int = Field(default=5, ge=1, le=5)
    continuity: int = Field(default=5, ge=1, le=5)
    player_agency: int = Field(default=5, ge=1, le=5)
    pacing: int = Field(default=5, ge=1, le=5)
    progressive_disclosure: int = Field(default=5, ge=1, le=5)
    memory_behavior: int = Field(default=5, ge=1, le=5)
    narration_quality: int = Field(default=5, ge=1, le=5)
    trace_explainability: int = Field(default=5, ge=1, le=5)
    generic_architecture_compliance: int = Field(default=5, ge=1, le=5)


class EvalResult(BaseModel):
    run_id: str
    kind: str
    total: int
    passed: int
    findings: list[EvalFinding] = Field(default_factory=list)
    scorecard: EvalScorecard = Field(default_factory=EvalScorecard)
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def to_console_text(self) -> str:
        lines = [f"{self.kind}: {self.passed}/{self.total} passed"]
        for finding in self.findings:
            lines.append(
                f"FAIL {finding.case_id} [{finding.severity}/{finding.dimension}]: "
                f"{finding.message}"
            )
        return "\n".join(lines)


def score_from_findings(findings: list[EvalFinding]) -> EvalScorecard:
    scores = EvalScorecard()
    penalties = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    values = scores.model_dump()
    for finding in findings:
        if finding.dimension == "infrastructure":
            continue
        values[finding.dimension] = max(
            1,
            int(values[finding.dimension]) - penalties[finding.severity],
        )
    return EvalScorecard.model_validate(values)


@dataclass(frozen=True)
class RegressionResult:
    total: int
    passed: int
    failures: list[str]

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def to_console_text(self) -> str:
        lines = [f"regression: {self.passed}/{self.total} passed"]
        lines.extend(f"FAIL {failure}" for failure in self.failures)
        return "\n".join(lines)

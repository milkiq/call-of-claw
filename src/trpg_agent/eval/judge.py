from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from trpg_agent.eval.scorecard import EvalFinding, EvalResult, EvalScorecard
from trpg_agent.langchain.structured import invoke_structured_with_repair

JUDGE_PROMPT_VERSION = "generic-trpg-judge-v2"

JUDGE_SYSTEM_PROMPT = """You are a generic TRPG playtest judge.

Evaluate only against generic GM quality standards and the provided evidence. Do not assume any
specific ruleset or scenario beyond the loaded evidence spans. Findings must identify concrete
turn-level or trace-level issues.

Use findings only for actionable defects. Do not put strengths, confirmations, correct behavior, or
positive observations in findings; mention those only in the summary.
Write summary and findings in English, even when the transcript or evidence is multilingual.
"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "Transcript:\n{transcript}\n\nTrace:\n{trace}\n\nEvidence spans:\n{evidence}\n\n"
            "Return only a strictly valid JSON object matching this schema. Findings must be an "
            "empty list when there are no actionable defects:\n"
            "{{\n"
            '  "summary": "string",\n'
            '  "scorecard": {{\n'
            '    "rules_correctness": 1-5,\n'
            '    "fictional_authority": 1-5,\n'
            '    "continuity": 1-5,\n'
            '    "player_agency": 1-5,\n'
            '    "pacing": 1-5,\n'
            '    "progressive_disclosure": 1-5,\n'
            '    "memory_behavior": 1-5,\n'
            '    "narration_quality": 1-5,\n'
            '    "trace_explainability": 1-5,\n'
            '    "generic_architecture_compliance": 1-5\n'
            "  }},\n"
            '  "findings": []\n'
            "}}",
        ),
    ]
)


class JudgeReport(BaseModel):
    summary: str
    scorecard: EvalScorecard = Field(default_factory=EvalScorecard)
    findings: list[EvalFinding] = Field(default_factory=list)


def run_llm_judge(
    model: BaseChatModel,
    *,
    transcript: list[dict[str, Any]],
    trace: list[dict[str, Any]] | None = None,
    evidence: list[str] | None = None,
    run_id: str | None = None,
) -> EvalResult:
    """Run a LangChain structured-output LLM-as-judge pass."""

    report, attempts = invoke_structured_with_repair(
        model=model,
        prompt=JUDGE_PROMPT,
        schema=JudgeReport,
        payload={
            "transcript": json.dumps(transcript, ensure_ascii=False, indent=2),
            "trace": json.dumps(trace or [], ensure_ascii=False, indent=2),
            "evidence": "\n\n---\n\n".join(evidence or []),
        },
    )
    findings = [finding for finding in report.findings if _is_actionable_finding(finding)]
    return EvalResult(
        run_id=run_id or f"judge-{uuid.uuid4().hex[:12]}",
        kind="llm_judge",
        total=1,
        passed=0 if findings else 1,
        findings=findings,
        scorecard=report.scorecard,
        metadata={
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "structured_attempts": str(len(attempts)),
            "filtered_non_actionable_findings": str(len(report.findings) - len(findings)),
        },
    )


def _is_actionable_finding(finding: EvalFinding) -> bool:
    message = finding.message.lower()
    positive_markers = [
        "correctly ",
        "excellent",
        "strong",
        "preserve player agency",
        "preserves player agency",
        "faithfully",
    ]
    defect_markers = [
        " but ",
        "however",
        "missing",
        "mismatch",
        "failed",
        "fails",
        "failing",
        "not ",
        " no ",
        "without",
        "issue",
        "error",
        "inconsisten",
        "below threshold",
        "leak",
        "unsupported",
        "bypass",
    ]
    if any(marker in message for marker in positive_markers) and not any(
        marker in message for marker in defect_markers
    ):
        return False
    return True


def run_static_quality_gate(
    *,
    output: str,
    case_id: str = "static-output-quality",
    forbidden_terms: list[str] | None = None,
) -> EvalResult:
    """Local quality gate used when no LLM judge model is configured."""

    findings: list[EvalFinding] = []
    if not output.strip():
        findings.append(
            EvalFinding(
                case_id=case_id,
                dimension="narration_quality",
                severity="high",
                message="Output is empty.",
                suggested_area="graph.output",
            )
        )
    for term in forbidden_terms or []:
        if term in output:
            findings.append(
                EvalFinding(
                    case_id=case_id,
                    dimension="progressive_disclosure",
                    severity="high",
                    message=f"Output included forbidden term: {term}",
                    evidence=output,
                    suggested_area="retrieval.visibility",
                )
            )
    return EvalResult(
        run_id=f"static-judge-{uuid.uuid4().hex[:12]}",
        kind="static_quality_gate",
        total=1,
        passed=0 if findings else 1,
        findings=findings,
        metadata={"judge_prompt_version": "static"},
    )

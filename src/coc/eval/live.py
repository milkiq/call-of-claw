from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models.chat_models import BaseChatModel

from coc.app.config import AppConfig
from coc.eval.judge import run_llm_judge
from coc.eval.scorecard import EvalFinding, EvalResult, EvalScorecard, score_from_findings
from coc.eval.session_cleanup import cleanup_metadata, cleanup_sessions
from coc.graph.runtime import durable_turn_graph, invoke_turn_graph
from coc.memory.store import SqliteStore


@dataclass(frozen=True)
class LiveEvalCase:
    id: str
    player_input: str
    forbidden_terms: list[str] = field(default_factory=list)
    ruleset_id: str | None = None
    scenario_id: str | None = None


def load_live_eval_cases(path: Path) -> list[LiveEvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    entries = raw if isinstance(raw, list) else [raw]
    return [LiveEvalCase(**entry) for entry in entries]


def run_live_eval(
    config: AppConfig,
    model: BaseChatModel,
    *,
    limit: int = 3,
    min_score: int = 3,
    session_prefix: str = "live-eval",
    persist: bool = True,
    model_metadata: dict[str, str] | None = None,
    case_path: Path | None = None,
    cleanup_session: bool = False,
) -> EvalResult:
    path = case_path or config.root_dir / "tests" / "live_eval_cases" / "smoke.yaml"
    cases = load_live_eval_cases(path)[: max(0, limit)]
    result_run_id = f"live-{uuid.uuid4().hex[:12]}"
    run_session_prefix = f"{session_prefix}-{result_run_id}"
    transcript: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    evidence: list[str] = []
    evidence_keys: set[tuple[str, str]] = set()
    findings: list[EvalFinding] = []

    with durable_turn_graph(sqlite_path=config.sqlite_path, model=model) as graph:
        for index, case in enumerate(cases, start=1):
            turn_id = f"{case.id}-{uuid.uuid4().hex[:8]}"
            try:
                result = invoke_turn_graph(
                    graph,
                    {
                        "player_input": case.player_input,
                        "session_id": f"{run_session_prefix}-{index}",
                        "thread_id": f"{run_session_prefix}-{index}",
                        "turn_id": turn_id,
                        "content_dir": str(config.content_dir),
                        "sqlite_path": str(config.sqlite_path),
                        "ruleset_id": case.ruleset_id,
                        "scenario_id": case.scenario_id,
                        "checkpoint_mode": "sqlite",
                        "model_metadata": model_metadata or {},
                    },
                )
            except Exception as error:
                findings.append(
                    EvalFinding(
                        case_id=case.id,
                        dimension="infrastructure",
                        severity="high",
                        message=f"Live play graph failed: {error}",
                        suggested_area="graph.llm",
                    )
                )
                continue

            final_output = str(result.get("final_output", ""))
            transcript.append(
                {
                    "case_id": case.id,
                    "turn_id": turn_id,
                    "player": case.player_input,
                    "gm": final_output,
                    "ruleset_id": case.ruleset_id,
                    "scenario_id": case.scenario_id,
                }
            )
            trace.append(
                {
                    "case_id": case.id,
                    "turn_id": turn_id,
                    "trace_events": result.get("trace_events", []),
                    "ruleset_id": result.get("ruleset_id"),
                    "scenario_id": result.get("scenario_id"),
                    "recent_canon": result.get("recent_canon", []),
                    "memory_hits": result.get("memory_hits", []),
                    "routing_decision": result.get("routing_decision", {}),
                    "turn_plan": result.get("turn_plan", {}),
                    "tool_results": result.get("tool_results", []),
                    "world_projection": result.get("world_projection", {}),
                }
            )
            for span in result.get("retrieved_spans", []):
                key = (str(span.get("package_id")), str(span.get("reference_id")))
                if key not in evidence_keys:
                    evidence.append(str(span.get("text", ""))[:4000])
                    evidence_keys.add(key)
            for term in case.forbidden_terms:
                if term and term in final_output:
                    findings.append(
                        EvalFinding(
                            case_id=case.id,
                            dimension="progressive_disclosure",
                            severity="high",
                            message=f"Live output included forbidden hidden term: {term}",
                            evidence=final_output,
                            suggested_area="graph.narration",
                        )
                    )

    judge_scorecard: EvalScorecard | None = None
    if transcript:
        try:
            judge_result = run_llm_judge(
                model,
                transcript=transcript,
                trace=trace,
                evidence=evidence,
                run_id=f"live-judge-{uuid.uuid4().hex[:12]}",
            )
            findings.extend(judge_result.findings)
            judge_scorecard = judge_result.scorecard
        except Exception as error:
            findings.append(
                EvalFinding(
                    case_id="live-judge",
                    dimension="infrastructure",
                    severity="high",
                    message=f"LLM judge failed: {error}",
                    suggested_area="eval.judge",
                )
            )

    if judge_scorecard:
        for dimension, score in judge_scorecard.model_dump().items():
            if int(score) < min_score:
                findings.append(
                    EvalFinding(
                        case_id="live-judge-scorecard",
                        dimension=dimension,
                        severity="high",
                        message=f"Judge score for {dimension} was below threshold.",
                        evidence=f"score={score}, threshold={min_score}",
                        suggested_area="eval.judge",
                    )
                )

    total = len(cases) + (1 if transcript else 0)
    failed_case_ids = {finding.case_id for finding in findings}
    result = EvalResult(
        run_id=result_run_id,
        kind="live_llm_eval",
        total=total,
        passed=max(0, total - len(failed_case_ids)),
        findings=findings,
        scorecard=judge_scorecard or score_from_findings(findings),
        metadata={
            "session_prefix": session_prefix,
            "run_session_prefix": run_session_prefix,
            "min_score": str(min_score),
            **(model_metadata or {}),
        },
    )
    if cleanup_session:
        store = SqliteStore(config.sqlite_path)
        store.migrate()
        result.metadata.update(
            cleanup_metadata(
                cleanup_sessions(
                    store=store,
                    sqlite_path=config.sqlite_path,
                    session_ids=[
                        f"{run_session_prefix}-{index}"
                        for index in range(1, len(cases) + 1)
                    ],
                )
            )
        )
    if persist:
        store = SqliteStore(config.sqlite_path)
        store.migrate()
        store.insert_eval_run(run_id=result.run_id, kind=result.kind, payload=result.model_dump())
    return result

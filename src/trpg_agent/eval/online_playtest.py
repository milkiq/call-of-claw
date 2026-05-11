from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from trpg_agent.app.config import AppConfig
from trpg_agent.eval.judge import run_llm_judge
from trpg_agent.eval.playtest import collect_playtest_metrics
from trpg_agent.eval.scorecard import EvalFinding, EvalResult, score_from_findings
from trpg_agent.graph.build_turn_graph import TURN_GRAPH_VERSION
from trpg_agent.graph.runtime import durable_turn_graph, invoke_turn_graph
from trpg_agent.langchain.structured import invoke_structured_with_repair
from trpg_agent.memory.store import SqliteStore

PLAYER_SIMULATOR_PROMPT_VERSION = "player-simulator-v1"


class PlayerTurnDecision(BaseModel):
    action: str
    intent_summary: str
    risk_tolerance: Literal["cautious", "balanced", "bold"] = "balanced"


PLAYER_SIMULATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an independent TRPG playtest player, not the GM.

Your job is to choose the next player action from the visible transcript only. Play like a real
single-player TRPG participant: understand the apparent objective, follow unresolved hooks, ask
specific questions when needed, take risks when the fiction calls for it, and push the story
forward.

Do not use hidden GM knowledge. Do not evaluate the system. Do not write GM narration. Return a
concrete first-person player action in the same language as the recent transcript.
""".strip(),
        ),
        (
            "human",
            "Turn number: {turn_number}\n\n"
            "Current GM output:\n{current_gm_output}\n\n"
            "Recent transcript:\n{recent_transcript}\n\n"
            "Public state summary:\n{public_state}\n\n"
            "Return only a strictly valid JSON object matching this schema:\n{schema}",
        ),
    ]
)


def run_online_playtest(
    config: AppConfig,
    model: BaseChatModel,
    *,
    turns: int = 100,
    min_score: int = 4,
    player_mode: Literal["policy", "llm"] = "policy",
    judge_mode: Literal["auto", "llm", "static"] = "auto",
    per_call_timeout_seconds: int | None = 90,
    single_turn_advisor: bool = False,
    micro_gates: bool = False,
    parallel_review: bool = False,
    session_id: str | None = None,
    ruleset_id: str | None = None,
    scenario_id: str | None = None,
    output_dir: Path | None = None,
    model_metadata: dict[str, str] | None = None,
) -> EvalResult:
    model = _model_with_timeout(model, per_call_timeout_seconds)
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    run_id = f"online-playtest-{uuid.uuid4().hex[:12]}"
    play_session_id = session_id or run_id
    out_dir = output_dir or config.data_dir / "online-playtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    evidence: list[str] = []
    evidence_keys: set[tuple[str, str]] = set()
    current_gm_output = ""
    smoke_fast_path = _online_smoke_fast_path(
        player_mode=player_mode,
        turns=turns,
    ) and not single_turn_advisor and not micro_gates

    with durable_turn_graph(sqlite_path=config.sqlite_path, model=model) as graph:
        for index in range(1, turns + 1):
            player_input = (
                _initial_player_action()
                if index == 1
                else _next_player_action(
                    model=model,
                    player_mode=player_mode,
                    turn_number=index,
                    current_gm_output=current_gm_output,
                    transcript=transcript,
                    public_state=_public_state_from_trace(trace),
                )
            )
            turn_id = f"{play_session_id}-turn-{index:03d}"
            result = invoke_turn_graph(
                graph,
                {
                    "player_input": player_input,
                    "session_id": play_session_id,
                    "thread_id": play_session_id,
                    "turn_id": turn_id,
                    "content_dir": str(config.content_dir),
                    "sqlite_path": str(config.sqlite_path),
                    "ruleset_id": ruleset_id,
                    "scenario_id": scenario_id,
                    "checkpoint_mode": "sqlite",
                    "eval_smoke_mode": smoke_fast_path,
                    "single_turn_advisor_mode": single_turn_advisor,
                    "micro_gates_mode": micro_gates,
                    "parallel_review_mode": parallel_review,
                    "model_metadata": model_metadata or {},
                },
            )
            current_gm_output = str(result.get("final_output", ""))
            transcript.append(
                {
                    "turn": index,
                    "turn_id": turn_id,
                    "player": player_input,
                    "gm": current_gm_output,
                }
            )
            trace.append(_trace_entry_from_result(index, result))
            for span in result.get("retrieved_spans", []):
                key = (str(span.get("package_id")), str(span.get("reference_id")))
                if key not in evidence_keys:
                    evidence.append(str(span.get("text", ""))[:4000])
                    evidence_keys.add(key)

        replay_index = max(1, min(turns, 3))
        replay_turn = transcript[replay_index - 1]
        replay = invoke_turn_graph(
            graph,
            {
                "player_input": replay_turn["player"],
                "session_id": play_session_id,
                "thread_id": play_session_id,
                "turn_id": replay_turn["turn_id"],
                "content_dir": str(config.content_dir),
                "sqlite_path": str(config.sqlite_path),
                "ruleset_id": ruleset_id,
                "scenario_id": scenario_id,
                "checkpoint_mode": "sqlite",
                "eval_smoke_mode": smoke_fast_path,
                "single_turn_advisor_mode": single_turn_advisor,
                "micro_gates_mode": micro_gates,
                "parallel_review_mode": parallel_review,
                "model_metadata": model_metadata or {},
            },
        )

    metrics = collect_playtest_metrics(
        store=store,
        session_id=play_session_id,
        requested_turns=turns,
        replay_restored=bool(replay.get("replayed_turn")),
    )
    findings = _online_findings_from_metrics(metrics)
    judge_scorecard = None
    llm_judge_used = _online_should_run_llm_judge(
        judge_mode=judge_mode,
        player_mode=player_mode,
        turns=turns,
    )
    if llm_judge_used:
        try:
            judge_result = run_llm_judge(
                model,
                transcript=_sample_transcript(transcript),
                trace=_sample_trace(trace),
                evidence=evidence,
                run_id=f"online-judge-{uuid.uuid4().hex[:12]}",
            )
            findings.extend(judge_result.findings)
            judge_scorecard = judge_result.scorecard
        except Exception as error:
            findings.append(
                EvalFinding(
                    case_id="online-playtest-judge",
                    dimension="infrastructure",
                    severity="high",
                    message=f"Online playtest LLM judge failed: {error}",
                    suggested_area="eval.online_playtest",
                )
            )

    if judge_scorecard:
        for dimension, score in judge_scorecard.model_dump().items():
            if int(score) < min_score:
                findings.append(
                    EvalFinding(
                        case_id="online-playtest-scorecard",
                        dimension=dimension,
                        severity="high",
                        message=f"Judge score for {dimension} was below threshold.",
                        evidence=f"score={score}, threshold={min_score}",
                        suggested_area="eval.online_playtest",
                    )
                )

    transcript_path = out_dir / f"{run_id}-transcript.md"
    report_path = out_dir / f"{run_id}-report.json"
    result = EvalResult(
        run_id=run_id,
        kind="online_playtest",
        total=2,
        passed=0 if findings else 2,
        findings=findings,
        scorecard=judge_scorecard or score_from_findings(findings),
        metadata={
            "graph_version": TURN_GRAPH_VERSION,
            "player_simulator_prompt_version": PLAYER_SIMULATOR_PROMPT_VERSION,
            "player_mode": player_mode,
            "judge_mode": judge_mode,
            "llm_judge_used": str(llm_judge_used),
            "smoke_fast_path": str(smoke_fast_path),
            "single_turn_advisor": str(single_turn_advisor),
            "micro_gates": str(micro_gates),
            "parallel_review": str(parallel_review),
            "per_call_timeout_seconds": str(per_call_timeout_seconds or ""),
            "session_id": play_session_id,
            "turns": str(turns),
            "ruleset_id": ruleset_id or "",
            "scenario_id": scenario_id or "",
            "transcript_path": str(transcript_path),
            "report_path": str(report_path),
            **(model_metadata or {}),
            **metrics.to_metadata(),
        },
    )
    transcript_path.write_text(
        _transcript_markdown(
            run_id=run_id,
            session_id=play_session_id,
            transcript=transcript,
            result=result,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "result": result.model_dump(),
                "metrics": metrics.model_dump(),
                "transcript": transcript,
                "trace_sample": _sample_trace(trace),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    store.insert_eval_run(run_id=result.run_id, kind=result.kind, payload=result.model_dump())
    return result


def _online_should_run_llm_judge(
    *,
    judge_mode: Literal["auto", "llm", "static"],
    player_mode: Literal["policy", "llm"],
    turns: int,
) -> bool:
    if judge_mode == "llm":
        return True
    if judge_mode == "static":
        return False
    return not (player_mode == "policy" and turns <= 5)


def _online_smoke_fast_path(
    *,
    player_mode: Literal["policy", "llm"],
    turns: int,
) -> bool:
    return player_mode == "policy" and turns <= 5


def _model_with_timeout(
    model: BaseChatModel,
    timeout_seconds: int | None,
) -> BaseChatModel:
    if not timeout_seconds or timeout_seconds <= 0:
        return model
    current = getattr(model, "timeout_seconds", None)
    if current is None:
        return model
    target = min(int(current), int(timeout_seconds)) if int(current) > 0 else int(timeout_seconds)
    if target == int(current):
        return model
    try:
        return model.model_copy(update={"timeout_seconds": target})
    except Exception:
        return model


def _initial_player_action() -> str:
    return "我先观察当前处境，确认最紧急的危险、目标位置，以及有什么可以立刻行动的入口。"


def _next_player_action(
    *,
    model: BaseChatModel,
    player_mode: Literal["policy", "llm"],
    turn_number: int,
    current_gm_output: str,
    transcript: list[dict[str, Any]],
    public_state: dict[str, Any],
) -> str:
    if player_mode == "policy":
        return _policy_player_action(
            turn_number=turn_number,
            current_gm_output=current_gm_output,
            transcript=transcript,
            public_state=public_state,
        )
    return _choose_llm_player_action(
        model=model,
        turn_number=turn_number,
        current_gm_output=current_gm_output,
        transcript=transcript,
        public_state=public_state,
    )


def _choose_llm_player_action(
    *,
    model: BaseChatModel,
    turn_number: int,
    current_gm_output: str,
    transcript: list[dict[str, Any]],
    public_state: dict[str, Any],
) -> str:
    try:
        decision, _ = invoke_structured_with_repair(
            model=model,
            prompt=PLAYER_SIMULATOR_PROMPT,
            schema=PlayerTurnDecision,
            payload={
                "turn_number": turn_number,
                "current_gm_output": current_gm_output,
                "recent_transcript": _recent_transcript_text(transcript),
                "public_state": public_state,
                "schema": PlayerTurnDecision.model_json_schema(),
            },
        )
        action = decision.action.strip()
    except Exception:
        action = ""
    if action:
        return action[:500]
    return _fallback_player_action(turn_number)


def _policy_player_action(
    *,
    turn_number: int,
    current_gm_output: str,
    transcript: list[dict[str, Any]],
    public_state: dict[str, Any],
) -> str:
    text = current_gm_output.lower()
    if _offers_pending_question_opportunity(current_gm_output):
        return "我问：当前局势里最能让我下一步行动准备充分的关键弱点、机会或安全切入点是什么？"
    clock = public_state.get("clock")
    clock_ratio = 0.0
    if isinstance(clock, dict) and int(clock.get("max", 0)):
        clock_ratio = int(clock.get("value", 0)) / int(clock.get("max", 1))
    recent_player_actions = " / ".join(str(turn.get("player", "")) for turn in transcript[-5:])

    candidates = [
        (
            any(term in text for term in ["入口", "舱门", "door", "hatch"]),
            "我仔细检查入口和舱门周围的可见痕迹，确认是否能安全通过，以及有没有需要先处理的危险。",
        ),
        (
            any(term in text for term in ["靠港", "停泊", "dock", "docking"]),
            "我手动接管靠港流程，优先让航线远离异常干扰，同时寻找安全的对接窗口。",
        ),
        (
            any(term in text for term in ["逃生艇", "沉睡", "微笑", "sleep", "pod"]),
            "我尝试用安全频道联系漂浮载具里的人，同时扫描他们的生命体征和周围是否有危险源。",
        ),
        (
            any(term in text for term in ["广播", "旋律", "声音", "signal", "sound"]),
            "我尝试定位异常信号的来源，并寻找能暂时屏蔽、削弱或避开它的方法。",
        ),
        (
            any(term in text for term in ["大厅", "走廊", "房间", "hall", "corridor", "room"]),
            "我谨慎进入下一个可达区域，先观察出口、遮蔽物、活动迹象和最明显的线索。",
        ),
        (
            clock_ratio >= 0.66,
            "时间压力已经很高，我选择一个最直接的阻止办法，愿意承担风险立刻行动。",
        ),
    ]
    offset = turn_number % len(candidates)
    for enabled, action in candidates[offset:] + candidates[:offset]:
        if enabled and action not in recent_player_actions:
            return action
    return _fallback_player_action(turn_number)


def _fallback_player_action(turn_number: int) -> str:
    actions = [
        "我检查刚才发现的线索，寻找能推进当前目标的具体入口。",
        "我选择一个最直接但谨慎的办法，继续向目标位置靠近。",
        "我询问现场是否有能帮助判断风险的明显标志。",
        "如果局势继续恶化，我准备承担风险采取行动来阻止它。",
    ]
    return actions[(turn_number - 1) % len(actions)]


def _offers_pending_question_opportunity(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "向gm问一个",
        "问一个关于当前局势的问题",
        "提一个关于当前局势的问题",
        "待使用的规则机会",
        "ask the gm one question",
        "ask one question about the current situation",
        "pending rules-granted opportunity",
    ]
    return any(marker in lowered for marker in markers)


def _recent_transcript_text(transcript: list[dict[str, Any]], *, limit: int = 8) -> str:
    recent = transcript[-limit:]
    lines: list[str] = []
    for turn in recent:
        lines.append(f"Turn {turn['turn']} Player: {turn['player']}")
        lines.append(f"Turn {turn['turn']} GM: {turn['gm']}")
    return "\n".join(lines)


def _public_state_from_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {}
    latest = trace[-1]
    world = latest.get("world_projection")
    if not isinstance(world, dict):
        return {}
    return {
        "active_scene": world.get("active_scene"),
        "clock": world.get("clock"),
        "revealed_facts": world.get("revealed_facts"),
        "known_clues": world.get("known_clues"),
        "scene": world.get("scene"),
    }


def _trace_entry_from_result(turn_number: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": turn_number,
        "turn_id": result.get("turn_id"),
        "trace_events": result.get("trace_events", []),
        "routing_decision": result.get("routing_decision", {}),
        "micro_gate_results": result.get("micro_gate_results", {}),
        "turn_plan": result.get("turn_plan", {}),
        "tool_results": result.get("tool_results", []),
        "world_projection": result.get("world_projection", {}),
        "critic_report": result.get("critic_report", {}),
    }


def _sample_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(transcript) <= 30:
        return transcript
    selected_indexes = set(range(0, min(8, len(transcript))))
    selected_indexes.update(range(9, len(transcript), 10))
    selected_indexes.update(range(max(0, len(transcript) - 12), len(transcript)))
    return [transcript[index] for index in sorted(selected_indexes)]


def _sample_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sampled_turns = {entry["turn"] for entry in _sample_transcript(trace)}
    return [entry for entry in trace if entry.get("turn") in sampled_turns]


def _online_findings_from_metrics(metrics: Any) -> list[EvalFinding]:
    findings: list[EvalFinding] = []
    if metrics.persisted_turns != metrics.requested_turns:
        findings.append(
            EvalFinding(
                case_id="online-playtest-turn-count",
                dimension="continuity",
                severity="high",
                message=(
                    f"Expected {metrics.requested_turns} persisted turns, got "
                    f"{metrics.persisted_turns}."
                ),
                suggested_area="eval.online_playtest",
            )
        )
    if not metrics.replay_restored:
        findings.append(
            EvalFinding(
                case_id="online-playtest-replay",
                dimension="infrastructure",
                severity="critical",
                message="Durable replay did not restore a previously persisted online turn.",
                suggested_area="graph.runtime",
            )
        )
    if metrics.resolver_bypass_count:
        findings.append(
            EvalFinding(
                case_id="online-playtest-resolver-bypass",
                dimension="rules_correctness",
                severity="critical",
                message="Risky online turns reached output without a successful resolver result.",
                evidence=str(metrics.resolver_bypass_count),
                suggested_area="graph.rules",
            )
        )
    if metrics.consecutive_repeated_outputs:
        findings.append(
            EvalFinding(
                case_id="online-playtest-consecutive-repetition",
                dimension="pacing",
                severity="medium",
                message="Online playtest produced consecutive repeated normalized outputs.",
                evidence=str(metrics.consecutive_repeated_outputs),
                suggested_area="graph.output",
            )
        )
    if metrics.requested_turns >= 10 and metrics.max_repeated_output_ratio > 0.2:
        findings.append(
            EvalFinding(
                case_id="online-playtest-repeated-content",
                dimension="pacing",
                severity="medium",
                message="A single normalized output pattern dominates the online transcript.",
                evidence=f"{metrics.max_repeated_output_ratio:.2f}",
                suggested_area="graph.output",
            )
        )
    if metrics.requested_turns >= 50 and metrics.memory_qa_accuracy < 0.8:
        findings.append(
            EvalFinding(
                case_id="online-playtest-memory-qa",
                dimension="memory_behavior",
                severity="high",
                message="Online long-play memory QA accuracy fell below threshold.",
                evidence=(
                    f"{metrics.memory_qa_passed}/{metrics.memory_qa_checks} "
                    f"({metrics.memory_qa_accuracy:.2f})"
                ),
                suggested_area="memory.eval",
            )
        )
    return findings


def _transcript_markdown(
    *,
    run_id: str,
    session_id: str,
    transcript: list[dict[str, Any]],
    result: EvalResult,
) -> str:
    lines = [
        f"# Online Playtest Transcript: {run_id}",
        "",
        f"- Session: `{session_id}`",
        f"- Turns: {len(transcript)}",
        f"- Result: {result.passed}/{result.total} passed",
        f"- Scorecard: `{result.scorecard.model_dump()}`",
        "",
        "## Findings",
        "",
    ]
    if result.findings:
        for finding in result.findings:
            lines.append(
                f"- `{finding.dimension}` {finding.severity}: {finding.message}"
            )
    else:
        lines.append("- No blocking findings.")
    lines.extend(["", "## Dialogue", ""])
    for turn in transcript:
        lines.extend(
            [
                f"### Turn {turn['turn']:03d}",
                "",
                f"**Player:** {turn['player']}",
                "",
                f"**GM:** {turn['gm']}",
                "",
            ]
        )
    return "\n".join(lines)

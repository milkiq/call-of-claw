from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import BaseModel, Field

from trpg_agent.app.config import AppConfig
from trpg_agent.eval.scorecard import EvalFinding, EvalResult, score_from_findings
from trpg_agent.graph.build_turn_graph import TURN_GRAPH_VERSION
from trpg_agent.graph.runtime import durable_turn_graph, invoke_turn_graph
from trpg_agent.memory.store import SqliteStore

SCRIPTED_LONG_PLAY_INPUTS = [
    "我检查周围有什么明显危险",
    "我等待一下观察变化",
    "风险行动 1d6",
    "刚刚发生了什么",
    "以后请简短回顾",
    "我尝试继续推进当前目标",
    "我查看公开状态里最紧急的威胁",
    "我向附近的人说明我的计划",
]


class PlaytestMetrics(BaseModel):
    session_id: str
    requested_turns: int
    persisted_turns: int
    canon_events: int
    memories: int
    world_patch_applications: int
    critic_reports: int
    replay_restored: bool
    duplicate_turn_ids: int = 0
    duplicate_world_patch_ids: int = 0
    duplicate_canon_ids: int = 0
    resolver_bypass_count: int = 0
    critical_critic_findings: int = 0
    player_agency_violations: int = 0
    clarification_turns: int = 0
    clarification_rate: float = 0.0
    first_turn_clarification: bool = False
    consecutive_repeated_outputs: int = 0
    max_repeated_output_ratio: float = 0.0
    unresolved_hook_count: int = 0
    unresolved_hook_quality: float = 1.0
    memory_qa_checks: int = 0
    memory_qa_passed: int = 0
    memory_qa_accuracy: float = 1.0
    trace_node_coverage: dict[str, int] = Field(default_factory=dict)

    def to_metadata(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in self.model_dump().items()
            if key != "trace_node_coverage"
        } | {
            f"trace_node.{key}": str(value)
            for key, value in self.trace_node_coverage.items()
        }


def run_scripted_long_play(
    config: AppConfig,
    *,
    turns: int = 50,
    session_id: str | None = None,
    ruleset_id: str | None = None,
    scenario_id: str | None = None,
    persist: bool = True,
) -> EvalResult:
    """Run a deterministic local long-play smoke test through the durable graph."""

    store = SqliteStore(config.sqlite_path)
    store.migrate()
    play_session_id = session_id or f"long-play-{uuid.uuid4().hex[:10]}"
    outputs: list[dict[str, Any]] = []

    scripted_inputs = [
        SCRIPTED_LONG_PLAY_INPUTS[index % len(SCRIPTED_LONG_PLAY_INPUTS)]
        for index in range(turns)
    ]
    model = FakeListChatModel(responses=_long_play_model_responses(scripted_inputs))
    with durable_turn_graph(sqlite_path=config.sqlite_path, model=model) as graph:
        for index in range(1, turns + 1):
            player_input = scripted_inputs[index - 1]
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
                    "eval_smoke_mode": True,
                },
            )
            outputs.append(result)

        replay_turn_id = f"{play_session_id}-turn-{max(1, min(turns, 3)):03d}"
        replay = invoke_turn_graph(
            graph,
            {
                "player_input": SCRIPTED_LONG_PLAY_INPUTS[2 % len(SCRIPTED_LONG_PLAY_INPUTS)],
                "session_id": play_session_id,
                "thread_id": play_session_id,
                "turn_id": replay_turn_id,
                "content_dir": str(config.content_dir),
                "sqlite_path": str(config.sqlite_path),
                "ruleset_id": ruleset_id,
                "scenario_id": scenario_id,
                "checkpoint_mode": "sqlite",
                "eval_smoke_mode": True,
            },
        )

    metrics = collect_playtest_metrics(
        store=store,
        session_id=play_session_id,
        requested_turns=turns,
        replay_restored=bool(replay.get("replayed_turn")),
    )
    findings = _findings_from_metrics(metrics)
    result = EvalResult(
        run_id=f"long-play-{uuid.uuid4().hex[:12]}",
        kind="long_play",
        total=1,
        passed=0 if findings else 1,
        findings=findings,
        scorecard=score_from_findings(findings),
        metadata={
            "graph_version": TURN_GRAPH_VERSION,
            "ruleset_id": ruleset_id or "",
            "scenario_id": scenario_id or "",
            **metrics.to_metadata(),
        },
    )
    if persist:
        store.insert_eval_run(run_id=result.run_id, kind=result.kind, payload=result.model_dump())
    return result


def _long_play_model_responses(scripted_inputs: list[str]) -> list[str]:
    responses: list[str] = []
    for player_input in scripted_inputs:
        route, intent_kind, decision, needs_resolution = _long_play_fixture_route(player_input)
        responses.append(
            _json_fixture(
                {
                    "intent": {
                        "kind": intent_kind,
                        "confidence": 0.9,
                        "reason": "long-play fixture",
                    },
                    "route": route,
                    "needs_rules_resolution": needs_resolution,
                    "needs_scenario_director": False,
                    "needs_memory_recall": route == "memory_recall",
                    "allow_direct_answer": route in {
                        "answer",
                        "rules_query",
                        "memory_recall",
                        "boundary",
                    },
                    "reasoning_summary": "Long-play fixture supplied routing.",
                    "uncertainty": None,
                    "citations": [],
                }
            )
        )
        if needs_resolution:
            responses.append(
                _json_fixture(
                    {
                        "requires_resolution": True,
                        "procedure_id": None,
                        "approach_id": None,
                        "risk": "risky_uncertain",
                        "stakes": "Long-play fixture requires deterministic resolver resolution.",
                        "clarification_question": None,
                        "citations": [],
                    }
                )
            )
        responses.append(
            _json_fixture(
                {
                    "intent": {
                        "kind": intent_kind,
                        "confidence": 0.9,
                        "reason": "long-play fixture",
                    },
                    "authority": {
                        "ok": decision != "boundary",
                        "reason": "long-play fixture",
                    },
                    "decision": decision,
                    "tool_requests": [],
                    "narration_brief": f"Long-play fixture turn plan for {decision}.",
                    "citations": [],
                }
            )
        )
        responses.append(
            _json_fixture(
                {
                    "final_text": _long_play_fixture_output(
                        player_input=player_input,
                        decision=decision,
                    ),
                    "canon_event_draft": None,
                    "memory_candidates": [],
                }
            )
        )
    return responses


def _long_play_fixture_route(player_input: str) -> tuple[str, str, str, bool]:
    if "刚刚" in player_input:
        return "memory_recall", "memory_recall", "answer", False
    if "以后" in player_input:
        return "answer", "info_query", "answer", False
    return "free_action", "action", "free_action", False


def _long_play_fixture_output(*, player_input: str, decision: str) -> str:
    if "刚刚" in player_input:
        return "最近已建立事实：你已经观察局势并推进过一次风险行动。"
    if decision == "risky_action":
        return "判定结果将依据规则工具叙事。"
    if "以后" in player_input:
        return "已记录你的偏好：之后用简短回顾。"
    return f"[turn_plan:{decision}] 你围绕“{player_input}”继续推进当前可见局势。"


def _json_fixture(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def collect_playtest_metrics(
    *,
    store: SqliteStore,
    session_id: str,
    requested_turns: int,
    replay_restored: bool,
) -> PlaytestMetrics:
    turns = store.list_turns(session_id)
    canon_events = store.list_canon_events(session_id)
    world_patches = store.list_world_patch_applications(session_id)
    critic_reports = store.list_critic_reports(session_id)
    memories = store.list_memories(scope=session_id)

    turn_ids = [turn["id"] for turn in turns]
    patch_ids = [patch["id"] for patch in world_patches]
    canon_ids = [event["id"] for event in canon_events]
    trace_node_coverage: dict[str, int] = {}
    resolver_bypass_count = 0
    player_agency_violations = 0
    clarification_turns = 0
    first_turn_clarification = False
    for index, turn in enumerate(turns):
        trace = turn.get("trace", {})
        for event in trace.get("trace_events", []):
            node = str(event.get("node", "unknown"))
            trace_node_coverage[node] = trace_node_coverage.get(node, 0) + 1
        if _turn_has_resolver_bypass(trace):
            resolver_bypass_count += 1
        if _looks_like_player_agency_violation(turn.get("output", "")):
            player_agency_violations += 1
        if _turn_asked_for_clarification(trace):
            clarification_turns += 1
            if index == 0:
                first_turn_clarification = True
    repetition = _repetition_metrics(turns)
    unresolved_hooks = _unresolved_hook_metrics(store, session_id)
    memory_qa = _memory_qa_metrics(
        store=store,
        session_id=session_id,
        requested_turns=requested_turns,
        turns=turns,
    )

    critical_findings = 0
    for report in critic_reports:
        for finding in report.get("report", {}).get("findings", []):
            if finding.get("severity") == "critical":
                critical_findings += 1

    return PlaytestMetrics(
        session_id=session_id,
        requested_turns=requested_turns,
        persisted_turns=len(turns),
        canon_events=len(canon_events),
        memories=len(memories),
        world_patch_applications=len(world_patches),
        critic_reports=len(critic_reports),
        replay_restored=replay_restored,
        duplicate_turn_ids=len(turn_ids) - len(set(turn_ids)),
        duplicate_world_patch_ids=len(patch_ids) - len(set(patch_ids)),
        duplicate_canon_ids=len(canon_ids) - len(set(canon_ids)),
        resolver_bypass_count=resolver_bypass_count,
        critical_critic_findings=critical_findings,
        player_agency_violations=player_agency_violations,
        clarification_turns=clarification_turns,
        clarification_rate=(clarification_turns / len(turns)) if turns else 0.0,
        first_turn_clarification=first_turn_clarification,
        consecutive_repeated_outputs=repetition["consecutive_repeated_outputs"],
        max_repeated_output_ratio=repetition["max_repeated_output_ratio"],
        unresolved_hook_count=unresolved_hooks["unresolved_hook_count"],
        unresolved_hook_quality=unresolved_hooks["unresolved_hook_quality"],
        memory_qa_checks=memory_qa["memory_qa_checks"],
        memory_qa_passed=memory_qa["memory_qa_passed"],
        memory_qa_accuracy=memory_qa["memory_qa_accuracy"],
        trace_node_coverage=trace_node_coverage,
    )


def build_session_quality_summary(store: SqliteStore, session_id: str) -> dict[str, Any]:
    turns = store.list_turns(session_id)
    metrics = collect_playtest_metrics(
        store=store,
        session_id=session_id,
        requested_turns=len(turns),
        replay_restored=True,
    )
    return {
        "session_id": session_id,
        "turns": metrics.persisted_turns,
        "canon_events": metrics.canon_events,
        "memories": metrics.memories,
        "critic_reports": metrics.critic_reports,
        "world_patch_applications": metrics.world_patch_applications,
        "resolver_bypass_count": metrics.resolver_bypass_count,
        "critical_critic_findings": metrics.critical_critic_findings,
        "player_agency_violations": metrics.player_agency_violations,
        "clarification_turns": metrics.clarification_turns,
        "clarification_rate": metrics.clarification_rate,
        "first_turn_clarification": metrics.first_turn_clarification,
        "consecutive_repeated_outputs": metrics.consecutive_repeated_outputs,
        "max_repeated_output_ratio": metrics.max_repeated_output_ratio,
        "unresolved_hook_count": metrics.unresolved_hook_count,
        "unresolved_hook_quality": metrics.unresolved_hook_quality,
        "memory_qa_accuracy": metrics.memory_qa_accuracy,
        "latest_output": turns[-1]["output"] if turns else "",
    }


def _findings_from_metrics(metrics: PlaytestMetrics) -> list[EvalFinding]:
    findings: list[EvalFinding] = []
    if metrics.persisted_turns != metrics.requested_turns:
        findings.append(
            EvalFinding(
                case_id="long-play-turn-count",
                dimension="continuity",
                severity="high",
                message=(
                    f"Expected {metrics.requested_turns} persisted turns, got "
                    f"{metrics.persisted_turns}."
                ),
                suggested_area="eval.playtest",
            )
        )
    if not metrics.replay_restored:
        findings.append(
            EvalFinding(
                case_id="long-play-replay",
                dimension="infrastructure",
                severity="critical",
                message="Durable replay did not restore a previously persisted turn.",
                suggested_area="graph.runtime",
            )
        )
    if (
        metrics.duplicate_turn_ids
        or metrics.duplicate_world_patch_ids
        or metrics.duplicate_canon_ids
    ):
        findings.append(
            EvalFinding(
                case_id="long-play-duplicates",
                dimension="infrastructure",
                severity="critical",
                message="Long-play replay produced duplicated durable ids.",
                evidence=(
                    f"turns={metrics.duplicate_turn_ids}, "
                    f"world_patches={metrics.duplicate_world_patch_ids}, "
                    f"canon={metrics.duplicate_canon_ids}"
                ),
                suggested_area="memory.store",
            )
        )
    if metrics.resolver_bypass_count:
        findings.append(
            EvalFinding(
                case_id="long-play-resolver-bypass",
                dimension="rules_correctness",
                severity="critical",
                message="Risky turns reached output without a successful resolver result.",
                evidence=str(metrics.resolver_bypass_count),
                suggested_area="graph.rules",
            )
        )
    if metrics.critical_critic_findings:
        findings.append(
            EvalFinding(
                case_id="long-play-critical-critic",
                dimension="narration_quality",
                severity="critical",
                message="Critical critic findings occurred during long play.",
                evidence=str(metrics.critical_critic_findings),
                suggested_area="graph.critic",
            )
        )
    if metrics.player_agency_violations:
        findings.append(
            EvalFinding(
                case_id="long-play-player-agency",
                dimension="player_agency",
                severity="high",
                message="Long-play output appeared to dictate player intent or action.",
                evidence=str(metrics.player_agency_violations),
                suggested_area="graph.narration",
            )
        )
    if metrics.consecutive_repeated_outputs:
        findings.append(
            EvalFinding(
                case_id="long-play-consecutive-repetition",
                dimension="pacing",
                severity="medium",
                message="Long-play produced consecutive repeated normalized outputs.",
                evidence=str(metrics.consecutive_repeated_outputs),
                suggested_area="graph.output",
            )
        )
    if metrics.max_repeated_output_ratio > 0.25:
        findings.append(
            EvalFinding(
                case_id="long-play-repeated-content",
                dimension="pacing",
                severity="medium",
                message="A single normalized output pattern dominates the long-play transcript.",
                evidence=f"{metrics.max_repeated_output_ratio:.2f}",
                suggested_area="graph.output",
            )
        )
    if metrics.unresolved_hook_count == 0:
        findings.append(
            EvalFinding(
                case_id="long-play-missing-hooks",
                dimension="continuity",
                severity="medium",
                message="Long-play ended with no trackable unresolved hook.",
                suggested_area="memory.hooks",
            )
        )
    elif metrics.unresolved_hook_quality < 0.8:
        findings.append(
            EvalFinding(
                case_id="long-play-hook-quality",
                dimension="continuity",
                severity="medium",
                message="Unresolved hooks are too vague, duplicated, or missing source metadata.",
                evidence=f"{metrics.unresolved_hook_quality:.2f}",
                suggested_area="memory.hooks",
            )
        )
    if metrics.requested_turns >= 50 and metrics.memory_qa_checks < 3:
        findings.append(
            EvalFinding(
                case_id="long-play-memory-qa-coverage",
                dimension="memory_behavior",
                severity="high",
                message="50-turn memory QA did not have enough independent checks.",
                evidence=str(metrics.memory_qa_checks),
                suggested_area="memory.eval",
            )
        )
    if metrics.memory_qa_accuracy < 0.8:
        findings.append(
            EvalFinding(
                case_id="long-play-memory-qa-accuracy",
                dimension="memory_behavior",
                severity="high",
                message="Long-play memory QA accuracy fell below threshold.",
                evidence=(
                    f"{metrics.memory_qa_passed}/{metrics.memory_qa_checks} "
                    f"({metrics.memory_qa_accuracy:.2f})"
                ),
                suggested_area="memory.eval",
            )
        )
    required_nodes = {
        "load_runtime_context",
        "retrieve_memory",
        "retrieve_content_spans",
        "apply_world_patch_results",
        "critic_guardrail_locally",
    }
    missing = sorted(required_nodes - set(metrics.trace_node_coverage))
    if missing:
        findings.append(
            EvalFinding(
                case_id="long-play-trace-coverage",
                dimension="trace_explainability",
                severity="medium",
                message="Long-play trace missed required runtime nodes.",
                evidence=", ".join(missing),
                suggested_area="graph.trace",
            )
        )
    return findings


def _turn_has_resolver_bypass(trace: dict[str, Any]) -> bool:
    plan = trace.get("turn_plan", {})
    if plan.get("decision") != "risky_action":
        return False
    return not any(
        result.get("tool_name") == "run_ruleset_resolver" and result.get("ok")
        for result in trace.get("tool_results", [])
        if isinstance(result, dict)
    )


def _turn_asked_for_clarification(trace: dict[str, Any]) -> bool:
    plan = trace.get("turn_plan", {})
    if plan.get("decision") == "clarify":
        return True
    routing = trace.get("routing_decision", {})
    return routing.get("route") == "clarify"


def _looks_like_player_agency_violation(output: str) -> bool:
    lowered = output.lower()
    markers = [
        "不由自主",
        "忍不住",
        "你想要",
        "你想跟",
        "你决定",
        "you want",
        "you decide",
        "you feel compelled",
    ]
    return any(marker in lowered or marker in output for marker in markers)


def _repetition_metrics(turns: list[dict[str, Any]]) -> dict[str, float | int]:
    outputs = [_normalize_output(turn.get("output", "")) for turn in turns]
    outputs = [output for output in outputs if output]
    if not outputs:
        return {"consecutive_repeated_outputs": 0, "max_repeated_output_ratio": 0.0}
    consecutive = sum(
        1 for left, right in zip(outputs, outputs[1:], strict=False) if left == right
    )
    counts = Counter(outputs)
    return {
        "consecutive_repeated_outputs": consecutive,
        "max_repeated_output_ratio": max(counts.values()) / len(outputs),
    }


def _normalize_output(output: str) -> str:
    normalized = re.sub(r"\s+", " ", output).strip().lower()
    normalized = re.sub(r"\[[0-9,\s]+\]", "[rolls]", normalized)
    normalized = re.sub(r"\b\d+d\d+\b", "XdY", normalized)
    normalized = re.sub(r"成功数\s*\d+", "成功数 N", normalized)
    normalized = re.sub(r"总计\s*\d+", "总计 N", normalized)
    normalized = re.sub(r"当前压力：\d+/\d+", "当前压力：N/N", normalized)
    return normalized


def _unresolved_hook_metrics(store: SqliteStore, session_id: str) -> dict[str, float | int]:
    hooks = _collect_unresolved_hooks(store, session_id)
    if not hooks:
        return {"unresolved_hook_count": 0, "unresolved_hook_quality": 0.0}
    seen: set[str] = set()
    valid = 0
    for hook in hooks:
        text = str(hook.get("text", "")).strip()
        key = _normalize_output(text)
        has_source = bool(hook.get("source"))
        has_status = hook.get("status") == "open"
        is_specific = len(text) >= 20
        is_duplicate = key in seen
        seen.add(key)
        if has_source and has_status and is_specific and not is_duplicate:
            valid += 1
    return {
        "unresolved_hook_count": len(hooks),
        "unresolved_hook_quality": valid / len(hooks),
    }


def _collect_unresolved_hooks(store: SqliteStore, session_id: str) -> list[dict[str, str]]:
    hooks: list[dict[str, str]] = []
    state = store.get_session_state(session_id) or {}
    scene = state.get("scene")
    if isinstance(scene, dict) and scene.get("public_summary"):
        hooks.append(
            {
                "kind": "active_scene",
                "text": str(scene["public_summary"]),
                "source": f"scene:{scene.get('id', 'unknown')}",
                "status": "open",
            }
        )
    clock = state.get("clock")
    if isinstance(clock, dict) and int(clock.get("value", 0)) < int(clock.get("max", 0)):
        hooks.append(
            {
                "kind": "clock",
                "text": (
                    f"{clock.get('id', 'clock')} pressure remains at "
                    f"{clock.get('value', 0)}/{clock.get('max', 0)}."
                ),
                "source": "world_state:clock",
                "status": "open",
            }
        )
    for memory in store.list_memories(scope=session_id):
        if memory.get("kind") != "unresolved_thread":
            continue
        hooks.append(
            {
                "kind": "memory",
                "text": str(memory.get("text", "")),
                "source": str(memory.get("metadata", {}).get("source") or memory.get("id")),
                "status": str(memory.get("metadata", {}).get("status") or "open"),
            }
        )
    return hooks


def _memory_qa_metrics(
    *,
    store: SqliteStore,
    session_id: str,
    requested_turns: int,
    turns: list[dict[str, Any]],
) -> dict[str, float | int]:
    checks: list[bool] = []
    canon_events = store.list_canon_events(session_id)
    if turns:
        first_input = str(turns[0].get("input", ""))
        checks.append(
            any(
                event.get("payload", {}).get("player_action") == first_input
                for event in canon_events
                if isinstance(event.get("payload"), dict)
            )
        )

    if any("简短回顾" in str(turn.get("input", "")) for turn in turns):
        recalled = store.recall_memories(
            query="简短回顾",
            scope=session_id,
            include_gm_only=False,
            limit=5,
        )
        checks.append(any("简短回顾" in memory.get("text", "") for memory in recalled))

    expected_summaries = requested_turns // 10
    if expected_summaries:
        summaries = [
            memory
            for memory in store.list_memories(scope=session_id)
            if memory.get("kind") == "episodic_summary"
        ]
        checks.append(len(summaries) >= expected_summaries)
        if summaries:
            latest_turn_count = max(
                int(summary.get("metadata", {}).get("turn_count", 0))
                for summary in summaries
            )
            checks.append(latest_turn_count >= expected_summaries * 10)

    recall_outputs = [
        turn.get("output", "")
        for turn in turns
        if any(term in str(turn.get("input", "")) for term in ["刚刚", "刚才", "之前"])
    ]
    if recall_outputs:
        checks.append(any("最近已建立事实" in output for output in recall_outputs))

    passed = sum(1 for check in checks if check)
    total = len(checks)
    return {
        "memory_qa_checks": total,
        "memory_qa_passed": passed,
        "memory_qa_accuracy": (passed / total) if total else 1.0,
    }

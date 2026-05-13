from __future__ import annotations

import json
import uuid
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from trpg_agent.app.config import AppConfig
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.eval.cases import EvalCase, load_eval_cases
from trpg_agent.eval.scorecard import EvalFinding, EvalResult, score_from_findings
from trpg_agent.graph.build_turn_graph import (
    TURN_GRAPH_VERSION,
    build_turn_graph_with_model,
)
from trpg_agent.graph.runtime import durable_turn_graph, invoke_turn_graph
from trpg_agent.langchain.prompts import validate_core_prompt_is_generic
from trpg_agent.memory.canon import import_canon_jsonl
from trpg_agent.memory.store import SqliteStore
from trpg_agent.tools.dice import roll_dice_once


def default_case_dir(config: AppConfig) -> Path:
    return config.root_dir / "tests" / "eval_cases"


def run_eval_cases(
    config: AppConfig,
    *,
    case_dir: Path | None = None,
    kind: str = "deterministic",
    case_kinds: set[str] | None = None,
    persist: bool = True,
) -> EvalResult:
    cases = load_eval_cases(case_dir or default_case_dir(config))
    registry = ContentRegistry.load(config.content_dir, config.root_dir)
    if case_kinds is not None:
        cases = [case for case in cases if case.kind in case_kinds]
    findings: list[EvalFinding] = []
    for case in cases:
        findings.extend(run_case(config, case))

    passed = len(cases) - len({finding.case_id for finding in findings})
    result = EvalResult(
        run_id=f"{kind}-{uuid.uuid4().hex[:12]}",
        kind=kind,
        total=len(cases),
        passed=passed,
        findings=findings,
        scorecard=score_from_findings(findings),
        metadata={
            "case_dir": str(case_dir or default_case_dir(config)),
            "graph_version": TURN_GRAPH_VERSION,
            "content_packages": str(
                {
                    package.id: package.manifest.version
                    for package in registry.packages
                }
            ),
        },
    )
    if persist:
        store = SqliteStore(config.sqlite_path)
        store.migrate()
        store.insert_eval_run(run_id=result.run_id, kind=result.kind, payload=result.model_dump())
    return result


def run_case(config: AppConfig, case: EvalCase) -> list[EvalFinding]:
    if case.kind == "deterministic":
        return run_deterministic_case(config, case)
    if case.kind == "turn":
        return run_turn_case(config, case)
    return [
        EvalFinding(
            case_id=case.id,
            dimension="infrastructure",
            severity="high",
            message=f"Unknown eval case kind: {case.kind}",
            suggested_area="eval.runner",
        )
    ]


def run_deterministic_case(config: AppConfig, case: EvalCase) -> list[EvalFinding]:
    findings: list[EvalFinding] = []
    expectation = case.expectation

    if expectation.no_core_prompt_terms:
        banned = validate_core_prompt_is_generic(expectation.core_prompt_forbidden_terms)
        if banned:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="progressive_disclosure",
                    severity="critical",
                    message="Core prompt contains concrete smoke-test terms.",
                    evidence=", ".join(banned),
                    suggested_area="langchain.prompts",
                )
            )

    if expectation.content_registry_valid:
        issues = ContentRegistry.load(config.content_dir, config.root_dir).validate()
        if issues:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="infrastructure",
                    severity="high",
                    message="Content registry validation failed.",
                    evidence="; ".join(issues),
                    suggested_area="content.registry",
                )
            )

    if expectation.deterministic_dice:
        first = roll_dice_once("2d6", f"{case.id}-roll", seed="eval")
        second = roll_dice_once("2d6", f"{case.id}-roll", seed="eval")
        if first != second:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="infrastructure",
                    severity="critical",
                    message="Dice tool did not replay deterministically.",
                    suggested_area="tools.dice",
                )
            )

    if expectation.canon_import_idempotent:
        store = SqliteStore(config.sqlite_path)
        store.migrate()
        import_canon_jsonl(store, config.seeds_dir / "canon-log.jsonl")
        imported_again = import_canon_jsonl(store, config.seeds_dir / "canon-log.jsonl")
        if imported_again != 0:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="memory_behavior",
                    severity="high",
                    message="Canon import duplicated already imported events.",
                    evidence=f"duplicates={imported_again}",
                    suggested_area="memory.canon",
                )
            )

    if expectation.durable_turn_replay:
        store = SqliteStore(config.sqlite_path)
        store.migrate()
        run_suffix = uuid.uuid4().hex[:8]
        session_id = f"eval-{case.id}-{run_suffix}"
        turn_id = f"{case.id}-{run_suffix}-turn"
        state = {
            "player_input": case.input or "我等待一下",
            "session_id": session_id,
            "thread_id": session_id,
            "turn_id": turn_id,
            "content_dir": str(config.content_dir),
            "sqlite_path": str(config.sqlite_path),
            "checkpoint_mode": "sqlite",
        }
        try:
            with durable_turn_graph(sqlite_path=config.sqlite_path) as graph:
                first = invoke_turn_graph(graph, state)
                second = invoke_turn_graph(graph, state)
        except Exception as error:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="infrastructure",
                    severity="critical",
                    message=f"Durable turn replay check failed: {error}",
                    suggested_area="graph.runtime",
                )
            )
        else:
            if first.get("final_output") != second.get("final_output"):
                findings.append(
                    EvalFinding(
                        case_id=case.id,
                        dimension="infrastructure",
                        severity="critical",
                        message="Durable replay returned a different final output.",
                        suggested_area="graph.runtime",
                    )
                )
            if not second.get("replayed_turn"):
                findings.append(
                    EvalFinding(
                        case_id=case.id,
                        dimension="infrastructure",
                        severity="high",
                        message="Second durable invocation did not restore the persisted turn.",
                        suggested_area="graph.runtime",
                    )
                )
            if len(store.list_turns(session_id)) != 1:
                findings.append(
                    EvalFinding(
                        case_id=case.id,
                        dimension="infrastructure",
                        severity="critical",
                        message="Durable replay duplicated persisted turns.",
                        suggested_area="memory.store",
                    )
                )

    return findings


def run_turn_case(config: AppConfig, case: EvalCase) -> list[EvalFinding]:
    if not case.input:
        return [
            EvalFinding(
                case_id=case.id,
                dimension="infrastructure",
                severity="high",
                message="Turn eval case has no input.",
                suggested_area="eval.cases",
            )
        ]

    model = FakeListChatModel(responses=_turn_case_llm_responses(case))
    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": case.input,
            "content_dir": str(config.content_dir),
            "ruleset_id": case.ruleset_id,
            "scenario_id": case.scenario_id,
            "eval_smoke_mode": True,
        }
    )
    findings: list[EvalFinding] = []
    expectation = case.expectation
    intent_kind = result.get("intent", {}).get("kind")
    if expectation.intent_kind and intent_kind != expectation.intent_kind:
        findings.append(
            EvalFinding(
                case_id=case.id,
                dimension="trace_explainability",
                severity="medium",
                message=f"Expected intent {expectation.intent_kind}, got {intent_kind}.",
                suggested_area="graph.intent",
            )
        )

    decision = result.get("turn_plan", {}).get("decision")
    if expectation.decision and decision != expectation.decision:
        findings.append(
            EvalFinding(
                case_id=case.id,
                dimension="fictional_authority",
                severity="medium",
                message=f"Expected decision {expectation.decision}, got {decision}.",
                suggested_area="graph.adjudication",
            )
        )

    if expectation.min_retrieved_spans is not None:
        retrieved = len(result.get("retrieved_spans", []))
        if retrieved < expectation.min_retrieved_spans:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="progressive_disclosure",
                    severity="high",
                    message=(
                        f"Expected at least {expectation.min_retrieved_spans} retrieved "
                        f"span(s), got {retrieved}."
                    ),
                    suggested_area="content.retrieval",
                )
            )

    requested_tools = [
        request.get("tool_name")
        for request in result.get("tool_requests", [])
        if isinstance(request, dict)
    ]
    for tool_name in expectation.required_tool_names:
        if tool_name not in requested_tools:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="rules_correctness",
                    severity="high",
                    message=f"Expected tool request was missing: {tool_name}.",
                    suggested_area="graph.tools",
                )
            )
    for tool_name in expectation.no_tool_names:
        if tool_name in requested_tools:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="rules_correctness",
                    severity="high",
                    message=f"Unexpected tool request: {tool_name}.",
                    suggested_area="graph.tools",
                )
            )

    trace_nodes = [
        event.get("node")
        for event in result.get("trace_events", [])
        if isinstance(event, dict)
    ]
    for node in expectation.required_trace_nodes:
        if node not in trace_nodes:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="trace_explainability",
                    severity="high",
                    message=f"Expected trace node was missing: {node}.",
                    suggested_area="graph.trace",
                )
            )

    for dotted_path, expected in expectation.world_state_equals.items():
        actual = _get_dotted(result.get("world_projection", {}), dotted_path)
        if actual != expected:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="continuity",
                    severity="high",
                    message=f"Expected world state {dotted_path}={expected!r}, got {actual!r}.",
                    suggested_area="graph.world_state",
                )
            )

    final_output = str(result.get("final_output", ""))
    if expectation.graph_output_prefix and not final_output.startswith(
        expectation.graph_output_prefix
    ):
        findings.append(
            EvalFinding(
                case_id=case.id,
                dimension="narration_quality",
                severity="medium",
                message="Graph output did not use the expected prefix.",
                evidence=final_output,
                suggested_area="graph.output",
            )
        )
    for text in expectation.output_contains:
        if text not in final_output:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="narration_quality",
                    severity="medium",
                    message=f"Output did not include expected text: {text}",
                    evidence=final_output,
                    suggested_area="graph.output",
                )
            )
    for text in expectation.output_not_contains:
        if text in final_output:
            findings.append(
                EvalFinding(
                    case_id=case.id,
                    dimension="progressive_disclosure",
                    severity="high",
                    message=f"Output included forbidden text: {text}",
                    evidence=final_output,
                    suggested_area="graph.output",
                )
            )
    return findings


def _turn_case_llm_responses(case: EvalCase) -> list[str]:
    if case.llm_responses:
        return [
            response
            if isinstance(response, str)
            else json.dumps(response, ensure_ascii=False)
            for response in case.llm_responses
        ]

    expectation = case.expectation
    intent_kind = expectation.intent_kind or "action"
    decision = expectation.decision or "free_action"
    needs_resolution = decision == "risky_action" or (
        "run_ruleset_resolver" in expectation.required_tool_names
    )
    route = _route_for_fixture(intent_kind=intent_kind, decision=decision)
    responses: list[dict[str, object]] = [
        _routing_fixture(
            intent_kind=intent_kind,
            route=route,
            needs_rules_resolution=needs_resolution,
        )
    ]
    if needs_resolution:
        responses.append(_rules_advice_fixture(case))
    if route != "clarify":
        responses.append(_turn_plan_fixture(case, intent_kind=intent_kind, decision=decision))
        responses.append(_narration_fixture(case, decision=decision))
    return [json.dumps(response, ensure_ascii=False) for response in responses]


def _route_for_fixture(*, intent_kind: str, decision: str) -> str:
    if decision == "answer":
        if intent_kind in {"rules_query", "memory_recall"}:
            return intent_kind
        return "answer"
    if decision in {"free_action", "risky_action", "gm_move", "boundary", "clarify"}:
        return decision
    return "free_action"


def _routing_fixture(
    *,
    intent_kind: str,
    route: str,
    needs_rules_resolution: bool,
) -> dict[str, object]:
    return {
        "intent": {
            "kind": intent_kind,
            "confidence": 0.9,
            "reason": "offline fixture",
        },
        "route": route,
        "needs_rules_resolution": needs_rules_resolution,
        "needs_scenario_director": route in {"free_action", "risky_action", "gm_move"},
        "needs_memory_recall": route == "memory_recall",
        "allow_direct_answer": route in {"answer", "rules_query", "memory_recall", "boundary"},
        "reasoning_summary": "Offline eval fixture supplied routing.",
        "uncertainty": None,
        "citations": [],
    }


def _rules_advice_fixture(case: EvalCase) -> dict[str, object]:
    return {
        "requires_resolution": True,
        "procedure_id": None,
        "approach_id": None,
        "risk": "risky_uncertain",
        "stakes": "Offline eval fixture requires deterministic resolver resolution.",
        "clarification_question": None,
        "citations": [],
    }


def _turn_plan_fixture(case: EvalCase, *, intent_kind: str, decision: str) -> dict[str, object]:
    return {
        "intent": {
            "kind": intent_kind,
            "confidence": 0.9,
            "reason": "offline fixture",
        },
        "authority": {
            "ok": decision != "boundary",
            "reason": "offline fixture",
            "unsupported_claim": case.input if decision == "boundary" else None,
            "grounded_alternatives": [],
        },
        "decision": decision,
        "tool_requests": [],
        "narration_brief": f"Offline fixture turn plan for {decision}.",
        "citations": [],
    }


def _narration_fixture(case: EvalCase, *, decision: str) -> dict[str, object]:
    prefix = case.expectation.graph_output_prefix or f"[turn_plan:{decision}]"
    contains = " ".join(case.expectation.output_contains)
    final_text = f"{prefix} offline fixture output"
    if contains:
        final_text = f"{final_text} {contains}"
    return {
        "final_text": final_text,
        "canon_event_draft": None,
        "memory_candidates": [],
    }


def _get_dotted(payload: dict, dotted_path: str) -> object:
    cursor: object = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor

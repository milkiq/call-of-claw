import re
from pathlib import Path

from trpg_agent.graph.build_turn_graph import (
    _micro_gate_context,
    _routing_from_micro_gates,
    apply_world_patch_results,
    build_llm_adjudication_node,
    build_llm_micro_gates_node,
    build_llm_narration_node,
    build_llm_scenario_director_node,
    build_parallel_review_and_memory_node,
    build_turn_graph,
    critic_guardrail_locally,
    curate_memory_locally,
    ensure_resolution_tools,
    execute_deterministic_tools,
    load_runtime_context,
    retrieve_context_parallel,
)
from trpg_agent.graph.runtime import (
    _runtime_profile,
    checkpoint_path_for,
    durable_turn_graph,
    invoke_turn_graph,
    stream_turn_graph,
)
from trpg_agent.memory.store import SqliteStore


def _routing_response(
    *,
    route: str = "free_action",
    intent_kind: str = "action",
    needs_rules_resolution: bool = False,
    allow_direct_answer: bool = False,
) -> str:
    needs_rules_json = "true" if needs_rules_resolution else "false"
    allow_direct_json = "true" if allow_direct_answer else "false"
    return f"""
    {{
      "intent": {{"kind": "{intent_kind}", "confidence": 0.9, "reason": "routed"}},
      "route": "{route}",
      "needs_rules_resolution": {needs_rules_json},
      "needs_scenario_director": true,
      "needs_memory_recall": false,
      "allow_direct_answer": {allow_direct_json},
      "reasoning_summary": "Generic routing decision.",
      "uncertainty": null,
      "citations": []
    }}
    """


def _rules_advice_response(
    *,
    approach_id: str | None = None,
) -> str:
    approach_json = f'"{approach_id}"' if approach_id else "null"
    return f"""
    {{
      "requires_resolution": true,
      "procedure_id": null,
      "approach_id": {approach_json},
      "risk": "risky_uncertain",
      "stakes": "The action has uncertain consequences.",
      "clarification_question": null,
      "citations": []
    }}
    """


def test_bootstrap_turn_graph_runs() -> None:
    graph = build_turn_graph()
    result = graph.invoke({"player_input": "我检查门口"})

    assert result["intent"]["kind"] == "action"
    assert result["turn_plan"]["decision"] == "free_action"
    assert result["final_output"].startswith("[turn_plan:free_action]")
    assert result["trace_events"]


def test_stream_turn_graph_reports_progress_nodes(tmp_path: Path) -> None:
    seen_nodes: list[str] = []
    state = {
        "player_input": "我检查门口",
        "session_id": "stream-progress",
        "thread_id": "stream-progress",
        "turn_id": "stream-progress-turn",
        "content_dir": str(Path.cwd() / "content"),
        "sqlite_path": str(tmp_path / "stream.sqlite"),
        "ruleset_id": "lasers_feelings_smoke",
        "scenario_id": "crystal_stop_singing_smoke",
        "checkpoint_mode": "sqlite",
    }

    with durable_turn_graph(sqlite_path=tmp_path / "stream.sqlite") as graph:
        result = stream_turn_graph(
            graph,
            state,
            on_node=lambda node, _update: seen_nodes.append(node),
        )

    assert result["final_output"]
    assert result["runtime_profile"]["budget_profile"] == "balanced"
    assert result["runtime_profile"]["node_count"] >= 1
    assert result["runtime_profile"]["total_elapsed_ms"] >= 0
    assert result["runtime_profile"]["slowest_nodes"]
    assert result["runtime_profile"]["slowest_nodes"][0]["category"]
    retrieval_trace = next(
        event for event in result["trace_events"] if event["node"] == "retrieve_content_spans"
    )
    parallel_trace = next(
        event for event in result["trace_events"] if event["node"] == "retrieve_context_parallel"
    )
    assert retrieval_trace["diagnostics"]["search_backend"] in {"sqlite_fts", "scan_fallback"}
    assert "branch_elapsed_ms" in parallel_trace
    assert parallel_trace["context_budget"]["mode"] == "shadow"
    assert "load_runtime_context" in seen_nodes
    assert "retrieve_context_parallel" in seen_nodes
    assert "persist_turn" in seen_nodes


def test_runtime_profile_counts_fallback_and_timeout_markers() -> None:
    profile = _runtime_profile(
        state={
            "trace_events": [
                {"node": "advisor", "fallback": True, "timeout_seconds": 90},
                {"node": "advisor", "advisor_error": "The read operation timed out"},
                {"node": "micro_gate", "gate_traces": [{"error": "timeout"}]},
            ]
        },
        nodes=[
            {"node": "advisor", "elapsed_ms": 42, "sequence": 1},
            {"node": "narrate", "elapsed_ms": 7, "sequence": 2},
        ],
        total_elapsed_ms=49,
    )

    assert profile["fallback_count"] == 1
    assert profile["timeout_count"] == 2
    assert profile["advisor_timeout_count"] == 2
    assert profile["slowest_nodes"][0]["node"] == "advisor"


def test_recorded_info_query_does_not_become_entry_clarification() -> None:
    graph = build_turn_graph()
    result = graph.invoke(
        {
            "player_input": "现在飞船里都有谁，尝试通讯其他人和飞船",
            "content_dir": str(Path.cwd() / "content"),
            "ruleset_id": "lasers_feelings_smoke",
            "scenario_id": "crystal_stop_singing_smoke",
        }
    )

    assert "不止一个入口" not in result["final_output"]
    assert "门/舱门" not in result["final_output"]


def test_recorded_high_risk_actions_use_resolver_when_advised() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    for player_input in [
        "我要关闭船中的广播",
        "我要用手枪贴着他开枪避免无法命中，我用布条死死塞住自己的耳朵",
    ]:
        model = FakeListChatModel(
            responses=[
                _routing_response(route="risky_action", needs_rules_resolution=True),
                _rules_advice_response(),
                """
                {
                  "intent": {"kind": "action", "confidence": 0.9, "reason": "advisor risk"},
                  "authority": {"ok": true, "reason": "grounded"},
                  "decision": "risky_action",
                  "tool_requests": [],
                  "narration_brief": "Resolve through the loaded rules.",
                  "citations": []
                }
                """,
                """
                {
                  "final_text": "判定结果已由规则工具产生。",
                  "canon_event_draft": null,
                  "memory_candidates": []
                }
                """,
            ]
        )
        result = build_turn_graph_with_model(model).invoke(
            {
                "player_input": player_input,
                "content_dir": str(Path.cwd() / "content"),
                "ruleset_id": "lasers_feelings_smoke",
                "scenario_id": "crystal_stop_singing_smoke",
            }
        )
        requested_tools = [request["tool_name"] for request in result.get("tool_requests", [])]

        assert result["turn_plan"]["decision"] == "risky_action"
        assert "run_ruleset_resolver" in requested_tools
        assert "请掷骰" not in result["final_output"]


def test_non_entry_clarification_uses_generic_target_text() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_adjudication_node(FakeListChatModel(responses=[]))
    result = node(
        {
            "player_input": "现在飞船里都有谁，尝试通讯其他人和飞船",
            "routing_decision": {
                "intent": {"kind": "clarify_needed", "confidence": 0.9, "reason": "ambiguous"},
                "route": "clarify",
                "needs_rules_resolution": False,
                "needs_scenario_director": False,
                "needs_memory_recall": False,
                "allow_direct_answer": True,
                "reasoning_summary": "Need a target.",
                "uncertainty": "unclear target",
                "citations": [],
            },
        }
    )

    assert "specific person" in result["turn_plan"]["narration_brief"]
    assert not re.search(r"[\u4e00-\u9fff]", result["turn_plan"]["narration_brief"])
    assert "不止一个入口" not in result["turn_plan"]["narration_brief"]


def test_local_critic_blocks_player_roll_request() -> None:
    result = critic_guardrail_locally(
        {
            "player_input": "我要用手枪贴着他开枪",
            "turn_plan": {"decision": "risky_action"},
            "tool_results": [],
            "final_output": "请掷骰（1d6），并说明你具体如何执行这一枪。",
            "trace_events": [],
        }
    )

    assert result["critic_report"]["blocks_output"] is True
    assert result["critic_report"]["findings"][0]["dimension"] == "resolver_bypass"
    assert "请掷骰" not in result["final_output"]


def test_context_retrieval_runs_parallel_branches() -> None:
    state = load_runtime_context(
        {
            "player_input": "I inspect the current scene.",
            "content_dir": str(Path.cwd() / "content"),
            "trace_events": [],
        }
    )

    result = retrieve_context_parallel(state)
    nodes = [event["node"] for event in result["trace_events"]]

    assert "retrieve_memory" in nodes
    assert "retrieve_content_spans" in nodes
    assert "retrieve_context_parallel" in nodes
    assert isinstance(result["retrieved_spans"], list)
    assert result["memory_hits"] == []


def test_parallel_review_and_memory_merges_safe_outputs() -> None:
    node = build_parallel_review_and_memory_node(critic_guardrail_locally, curate_memory_locally)

    result = node(
        {
            "player_input": "以后请用简短风格回复。",
            "final_output": "已记录。",
            "turn_plan": {"decision": "answer"},
            "tool_results": [],
            "trace_events": [],
        }
    )
    nodes = [event["node"] for event in result["trace_events"]]

    assert result["critic_report"]["ok"] is True
    assert result["memory_curation"]["should_write"] is True
    assert nodes[-1] == "review_and_curate_parallel"
    assert "critic_guardrail_locally" in nodes
    assert "curate_memory_locally" in nodes


def test_micro_gate_context_clips_hidden_story_content() -> None:
    state = {
        "ruleset_id": "generic_rules",
        "scenario_id": "hidden_scenario",
        "world_projection": {
            "active_scene": "scene_1",
            "scene": {
                "id": "scene_1",
                "title": "Visible Room",
                "public_summary": "A public doorway is visible.",
                "gm_only": "secret culprit waits behind the wall",
            },
            "revealed_facts": [
                {"visibility": "public", "content": "A public clue is visible."},
                {"visibility": "gm_only", "content": "secret motive"},
            ],
        },
        "retrieved_spans": [
            {
                "package_id": "hidden_scenario",
                "reference_id": "public",
                "visibility": "public",
                "score": 1.0,
                "text": "public scene text",
            },
            {
                "package_id": "hidden_scenario",
                "reference_id": "secret",
                "visibility": "gm_only",
                "score": 0.9,
                "text": "secret hidden text",
            },
        ],
        "player_memory_hits": [
            {
                "kind": "canon",
                "scope": "session",
                "text": "The player saw the public clue.",
                "metadata": {"visibility": "public"},
            }
        ],
    }

    context = _micro_gate_context(state, "authority_micro_gate")
    rendered = str(context)

    assert "public doorway" in rendered
    assert "public scene text" in rendered
    assert "public clue" in rendered
    assert "secret culprit" not in rendered
    assert "secret hidden text" not in rendered
    assert "secret motive" not in rendered


def test_micro_gates_fallback_does_not_treat_text_dice_as_resolution() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_micro_gates_node(FakeListChatModel(responses=[]))

    result = node(
        {
            "player_input": "我执行这个动作 1d6",
            "ruleset_id": "generic_rules",
            "trace_events": [],
        }
    )

    assert result["routing_decision"]["route"] == "free_action"
    assert result["routing_decision"]["needs_rules_resolution"] is False
    assert result["micro_gate_results"]["risk_micro_gate"]["risky"] is False
    assert result["trace_events"][-1]["node"] == "run_micro_gates"


def test_micro_gates_fallback_does_not_keyword_route_risk() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_micro_gates_node(FakeListChatModel(responses=[]))

    result = node(
        {
            "player_input": "我观察当前处境，确认最紧急的危险。",
            "ruleset_id": "generic_rules",
            "trace_events": [],
        }
    )

    assert result["routing_decision"]["route"] == "free_action"
    assert result["micro_gate_results"]["risk_micro_gate"]["risky"] is False


def test_micro_gate_target_ambiguity_without_clarify_does_not_override_route() -> None:
    result = _routing_from_micro_gates(
        {},
        {
            "authority_micro_gate": {
                "allowed": True,
                "boundary": False,
                "needs_clarification": False,
                "reason": "Authority is valid.",
            },
            "intent_micro_gate": {
                "intent": {
                    "kind": "action",
                    "confidence": 0.8,
                    "reason": "The player is asking for visible feedback.",
                },
                "route": "free_action",
                "allow_direct_answer": False,
                "needs_scenario_director": True,
                "reason": "Proceed with a bounded visible response.",
            },
            "risk_micro_gate": {
                "risky": False,
                "risk": "none",
                "needs_rules_resolution": False,
                "reason": "No uncertain risky outcome is being resolved.",
            },
            "target_micro_gate": {
                "ambiguous": True,
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "Several visible details exist, but safe visible feedback can proceed.",
            },
            "memory_recall_micro_gate": {
                "needs_memory_recall": False,
                "reason": "No memory recall requested.",
            },
        },
    )

    assert result["route"] == "free_action"
    assert result["needs_scenario_director"] is True
    assert result["intent"]["kind"] == "action"


def test_micro_gate_answer_route_can_request_scenario_director() -> None:
    result = _routing_from_micro_gates(
        {},
        {
            "authority_micro_gate": {
                "allowed": True,
                "boundary": False,
                "needs_clarification": False,
                "reason": "Authority is valid.",
            },
            "intent_micro_gate": {
                "intent": {
                    "kind": "info_query",
                    "confidence": 0.85,
                    "reason": "The player asks for current visible context.",
                },
                "route": "answer",
                "allow_direct_answer": True,
                "needs_scenario_director": True,
                "reason": "Answer should include visible scene context.",
            },
            "risk_micro_gate": {
                "risky": False,
                "risk": "none",
                "needs_rules_resolution": False,
                "reason": "No risky action.",
            },
            "target_micro_gate": {
                "ambiguous": False,
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "No blocking target ambiguity.",
            },
            "memory_recall_micro_gate": {
                "needs_memory_recall": False,
                "reason": "No memory recall requested.",
            },
        },
    )

    assert result["route"] == "answer"
    assert result["needs_scenario_director"] is True
    assert result["allow_direct_answer"] is True


def test_micro_gates_adjudication_skips_core_gm_plan_call() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_adjudication_node(FakeListChatModel(responses=[]))

    result = node(
        {
            "player_input": "我观察当前处境",
            "micro_gates_mode": True,
            "routing_decision": {
                "intent": {"kind": "action", "confidence": 0.8, "reason": "observation"},
                "route": "free_action",
                "needs_rules_resolution": False,
                "needs_scenario_director": True,
                "needs_memory_recall": False,
                "allow_direct_answer": False,
                "reasoning_summary": "Micro-gates routed a safe observation.",
                "uncertainty": None,
                "citations": [],
            },
            "micro_gate_results": {
                "authority_micro_gate": {
                    "allowed": True,
                    "boundary": False,
                    "needs_clarification": False,
                    "reason": "The player proposes an observation.",
                    "player_facing_boundary": None,
                }
            },
            "trace_events": [],
        }
    )

    assert result["turn_plan"]["decision"] == "free_action"
    assert result["tool_requests"] == []
    assert result["trace_events"][-1]["short_circuit"] == "micro_gates_local_turn_plan"


def test_pending_rule_opportunity_blocks_new_risky_resolution() -> None:
    state = {
        "player_input": "我立刻强行破解控制台",
        "ruleset_id": "lasers_feelings_smoke",
        "routing_decision": {"route": "risky_action", "needs_rules_resolution": True},
        "rules_advice": {
            "requires_resolution": True,
            "risk": "risky_uncertain",
            "approach_id": None,
        },
        "world_projection": {
            "pending_rule_opportunities": [
                {
                    "id": "previous:exact-target",
                    "prompt": "你可以向GM问一个关于当前局势的问题，GM必须诚实回答。",
                    "effect": "如果答案能帮助你的下一步行动，下一次相关判定可视为准备充分。",
                    "grants_prepared": True,
                    "status": "pending",
                }
            ]
        },
        "turn_plan": {"decision": "risky_action", "tool_requests": []},
        "tool_requests": [],
        "trace_events": [],
    }

    result = ensure_resolution_tools(state)

    assert result["turn_plan"]["decision"] == "clarify"
    assert result["tool_requests"] == []
    assert "pending rules-granted opportunity" in result["turn_plan"]["narration_brief"]
    assert not re.search(r"[\u4e00-\u9fff]", result["turn_plan"]["narration_brief"])
    assert not any(
        request.get("tool_name") == "run_ruleset_resolver"
        for request in result["turn_plan"]["tool_requests"]
    )


def test_pending_rule_opportunity_question_consumes_and_prepares() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_adjudication_node(FakeListChatModel(responses=[]))
    state = {
        "player_input": "我问：下一步最安全的切入点是什么？",
        "session_id": "pending-session",
        "turn_id": "pending-turn",
        "routing_decision": {"route": "answer", "needs_rules_resolution": False},
        "world_projection": {
            "pending_rule_opportunities": [
                {
                    "id": "previous:exact-target",
                    "prompt": "你可以向GM问一个关于当前局势的问题，GM必须诚实回答。",
                    "effect": "如果答案能帮助你的下一步行动，下一次相关判定可视为准备充分。",
                    "grants_prepared": True,
                    "status": "pending",
                }
            ]
        },
        "character_context": {"prepared": False},
        "trace_events": [],
    }

    planned = node(state)
    executed = execute_deterministic_tools(planned)
    applied = apply_world_patch_results(executed)

    assert planned["turn_plan"]["decision"] == "answer"
    assert planned["routing_decision"]["needs_scenario_director"] is False
    assert planned["tool_requests"][0]["tool_name"] == "apply_world_patch"
    assert applied["world_projection"]["pending_rule_opportunities"] == []
    assert applied["world_projection"]["character_context"]["prepared"] is True
    assert applied["character_context"]["prepared"] is True


def test_llm_narration_falls_back_when_model_fails() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_narration_node(FakeListChatModel(responses=[]))

    result = node(
        {
            "player_input": "I inspect the exit.",
            "turn_plan": {
                "decision": "free_action",
                "narration_brief": "You inspect the exit.",
            },
            "tool_results": [],
            "world_projection": {},
            "trace_events": [],
        }
    )

    assert result["final_output"].startswith("You inspect the exit.")
    assert result["narration_plan"]["final_text"] == result["final_output"]
    assert result["trace_events"][-1]["fallback"] is True


def test_llm_narration_fallback_uses_resolver_result_without_debug_marker() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_narration_node(FakeListChatModel(responses=[]))

    result = node(
        {
            "player_input": "我强行打开舱门",
            "turn_plan": {
                "decision": "risky_action",
                "narration_brief": "必须依据规则解析叙事。",
            },
            "tool_results": [
                {
                    "tool_name": "run_ruleset_resolver",
                    "ok": True,
                    "result": {
                        "dice_expression": "1d6",
                        "dice_result": {"expression": "1d6", "rolls": [4], "total": 4},
                        "successes": 1,
                        "band_label": "有代价的成功",
                        "consequence": "行动成功，但需要一个麻烦、代价或时钟推进。",
                        "world_patches": [
                            {"op": "increment", "path": ["clock", "value"], "value": 1}
                        ],
                    },
                }
            ],
            "scenario_director": {"player_visible_context": "警报声变得更急促。"},
            "trace_events": [],
        }
    )

    assert not result["final_output"].startswith("[turn_plan:")
    assert "1d6 -> [4]" in result["final_output"]
    assert "有代价的成功" in result["final_output"]
    assert "场景压力随之推进" in result["final_output"]
    assert "警报声变得更急促" in result["final_output"]


def test_llm_narration_fallback_hides_unavailable_scenario_marker() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_narration_node(FakeListChatModel(responses=[]))

    result = node(
        {
            "player_input": "我扫描危险源",
            "turn_plan": {"decision": "risky_action", "narration_brief": "依据规则结果。"},
            "tool_results": [
                {
                    "tool_name": "run_ruleset_resolver",
                    "ok": True,
                    "result": {
                        "dice_expression": "1d6",
                        "dice_result": {"expression": "1d6", "rolls": [3], "total": 3},
                        "successes": 1,
                        "band_label": "有代价的成功",
                        "consequence": "行动成功，但需要一个麻烦、代价或时钟推进。",
                        "world_patches": [
                            {"op": "increment", "path": ["clock", "value"], "value": 1}
                        ],
                    },
                }
            ],
            "scenario_director": {
                "player_visible_context": (
                    "Scenario director unavailable; no scenario patch proposed."
                )
            },
            "trace_events": [],
        }
    )

    assert "Scenario director unavailable" not in result["final_output"]
    assert "场景压力随之推进" in result["final_output"]


def test_llm_turn_graph_parses_turn_plan() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "player acts"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "free_action",
              "tool_requests": [],
              "narration_brief": "You inspect the doorway.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "You inspect the doorway.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """
        ]
    )

    result = build_turn_graph_with_model(model).invoke({"player_input": "I inspect the doorway"})

    assert result["intent"]["kind"] == "action"
    assert result["routing_decision"]["route"] == "free_action"
    assert result["turn_plan"]["decision"] == "free_action"
    assert result["final_output"] == "You inspect the doorway."
    assert result["narration_plan"]["final_text"] == "You inspect the doorway."


def test_single_turn_advisor_path_can_request_resolver() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            """
            {
              "routing_decision": {
                "intent": {"kind": "action", "confidence": 0.9, "reason": "risky action"},
                "route": "risky_action",
                "needs_rules_resolution": true,
                "needs_scenario_director": true,
                "needs_memory_recall": false,
                "allow_direct_answer": false,
                "reasoning_summary": "The action is risky and uncertain.",
                "uncertainty": null,
                "citations": []
              },
              "rules_advice": {
                "requires_resolution": true,
                "procedure_id": null,
                "approach_id": null,
                "risk": "risky_uncertain",
                "stakes": "The loaded resolver must decide the outcome.",
                "clarification_question": null,
                "citations": []
              },
              "turn_plan": {
                "intent": {"kind": "action", "confidence": 0.9, "reason": "risky action"},
                "authority": {"ok": true, "reason": "grounded"},
                "decision": "risky_action",
                "tool_requests": [],
                "narration_brief": "Use the loaded resolver before narrating the outcome.",
                "citations": []
              },
              "scenario_advice": {
                "decision": "no_change",
                "proposed_patches": [],
                "player_visible_context": "No pre-resolution scene change.",
                "gm_only_reason": "Wait for resolver output.",
                "citations": []
              },
              "reasoning_summary": "Combined advisor kept hard boundaries."
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "I force the jammed hatch open.",
            "turn_id": "single-turn-risky",
            "single_turn_advisor_mode": True,
            "eval_smoke_mode": True,
        }
    )
    nodes = [event["node"] for event in result["trace_events"]]

    assert "advise_turn_with_single_llm" in nodes
    assert "route_with_intent_arbiter" not in nodes
    assert result["turn_plan"]["decision"] == "risky_action"
    assert result["tool_results"][0]["tool_name"] == "run_ruleset_resolver"
    assert result["tool_results"][0]["ok"] is True
    assert result["final_output"].startswith("The resolver result is narrated.")
    assert "Roll: 1d6" in result["final_output"]


def test_single_turn_advisor_clarification_is_player_facing() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            """
            {
              "routing_decision": {
                "intent": {"kind": "clarify_needed", "confidence": 0.9, "reason": "target unclear"},
                "route": "clarify",
                "needs_rules_resolution": false,
                "needs_scenario_director": false,
                "needs_memory_recall": false,
                "allow_direct_answer": false,
                "reasoning_summary": "The target needs clarification.",
                "uncertainty": "which location",
                "citations": []
              },
              "rules_advice": {
                "requires_resolution": false,
                "procedure_id": null,
                "approach_id": null,
                "risk": "none",
                "stakes": "Clarify target.",
                "clarification_question": "Ask the player which specific location.",
                "citations": []
              },
              "turn_plan": {
                "intent": {"kind": "clarify_needed", "confidence": 0.9, "reason": "target unclear"},
                "authority": {"ok": true, "reason": "grounded"},
                "decision": "clarify",
                "tool_requests": [],
                "narration_brief": "Ask the player which specific location.",
                "citations": []
              },
              "scenario_advice": {
                "decision": "no_change",
                "proposed_patches": [],
                "player_visible_context": "No scene change.",
                "gm_only_reason": "Clarification first.",
                "citations": []
              },
              "reasoning_summary": "Clarification required."
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "我继续向目标位置靠近。",
            "turn_id": "single-turn-clarify",
            "single_turn_advisor_mode": True,
            "eval_smoke_mode": True,
        }
    )

    assert result["turn_plan"]["decision"] == "clarify"
    assert not re.search(r"[\u4e00-\u9fff]", result["turn_plan"]["narration_brief"])
    assert "具体对象" in result["final_output"]
    assert "Ask the player" not in result["final_output"]


def test_llm_turn_graph_does_not_override_advisor_route_with_target_keywords() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            """
            {
              "intent": {
                "kind": "action",
                "confidence": 0.75,
                "reason": "The target might be ambiguous, but the advisor chose to proceed."
              },
              "route": "free_action",
              "needs_rules_resolution": false,
              "needs_scenario_director": true,
              "needs_memory_recall": false,
              "allow_direct_answer": false,
              "reasoning_summary": "目标指代模糊，但模型原本想推进。",
              "uncertainty": "The target is mildly unclear.",
              "citations": []
            }
            """,
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "advisor route"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "free_action",
              "tool_requests": [],
              "narration_brief": "Proceed exactly as advised.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The advisor route was preserved.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke({"player_input": "我检查门口"})

    assert result["routing_decision"]["route"] == "free_action"
    assert "assumed_target_only" not in result["routing_decision"]
    assert "needs_rules_review" not in result["routing_decision"]
    assert result["turn_plan"]["decision"] == "free_action"
    assert result["tool_requests"] == []
    assert result["final_output"] == "The advisor route was preserved."


def test_rules_resolution_depends_on_advisor_not_target_keyword() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            _rules_advice_response(approach_id="lasers"),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "investigation"},
              "authority": {"ok": true, "reason": "rules advisor requires resolution"},
              "decision": "free_action",
              "tool_requests": [],
              "narration_brief": "Resolve the inspection.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The roll result shapes what the doorway reveals.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {"player_input": "I inspect the doorway", "turn_id": "advisor-rules-turn"}
    )

    assert "assumed_target_only" not in result["routing_decision"]
    assert result["rules_advice"]["requires_resolution"] is True
    assert result["turn_plan"]["decision"] == "risky_action"
    assert result["tool_results"][0]["tool_name"] == "run_ruleset_resolver"
    assert result["final_output"].startswith("The roll result shapes what the doorway reveals.")
    assert "Roll:" in result["final_output"]


def test_high_ambiguity_target_remains_clarification_without_local_default() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            """
            {
              "intent": {
                "kind": "clarify_needed",
                "confidence": 0.85,
                "reason": "Multiple plausible entrances exist."
              },
              "route": "clarify",
              "needs_rules_resolution": false,
              "needs_scenario_director": true,
              "needs_memory_recall": false,
              "allow_direct_answer": false,
              "reasoning_summary": "The target could be several doors.",
              "uncertainty": "Multiple plausible entrances could match the word door.",
              "citations": []
            }
            """,
            """
            {
              "requires_resolution": true,
              "procedure_id": null,
              "approach_id": null,
              "risk": "low",
              "stakes": "A quick look can proceed with the default target.",
              "clarification_question": "Which exact entrance do you mean?",
              "citations": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke({"player_input": "我检查门口"})

    assert result["routing_decision"]["route"] == "clarify"
    assert "assumed_target_only" not in result["routing_decision"]
    assert result["turn_plan"]["decision"] == "clarify"
    assert "具体对象" in result["final_output"]


def test_rules_advice_clarification_prevents_ambiguous_approach_roll() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            """
            {
              "requires_resolution": true,
              "procedure_id": "core_roll",
              "approach_id": "lasers",
              "risk": "risky_uncertain",
              "stakes": "The chosen focus changes the outcome.",
              "clarification_question": "你是主要用技术扫描，还是主要尝试唤醒对方？",
              "citations": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {"player_input": "我联系他们，同时扫描生命体征"}
    )

    assert result["turn_plan"]["decision"] == "clarify"
    assert not re.search(r"[\u4e00-\u9fff]", result["turn_plan"]["narration_brief"])
    assert result["tool_requests"] == []
    assert result["tool_results"] == []
    assert "技术扫描" in result["final_output"]
    assert "主要做法" in result["final_output"]


def test_mechanical_rules_clarification_is_not_asked_to_player() -> None:
    from trpg_agent.graph.build_turn_graph import _rules_advice_requires_player_clarification

    state = {
        "rules_advice": {
            "requires_resolution": True,
            "risk": "risky_uncertain",
            "clarification_question": "GM需裁定是否合并为一次Laser判定或分别判定。",
        }
    }

    assert _rules_advice_requires_player_clarification(state) is False


def test_llm_turn_graph_repairs_malformed_turn_plan() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(),
            "not json",
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "player acts"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "free_action",
              "tool_requests": [],
              "narration_brief": "You inspect the doorway.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "You inspect the doorway after repair.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke({"player_input": "I inspect the doorway"})

    assert result["turn_plan"]["decision"] == "free_action"
    assert result["final_output"] == "You inspect the doorway after repair."
    adjudication_event = next(
        event for event in result["trace_events"] if event["node"] == "adjudicate_with_llm"
    )
    assert len(adjudication_event["structured_attempts"]) == 2


def test_llm_risky_action_cannot_bypass_resolver() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            _rules_advice_response(),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "answer",
              "tool_requests": [
                {
                  "tool_name": "apply_world_patch",
                  "arguments": {
                    "patches": [{"op": "set", "path": ["active_scene"], "value": "scene_2"}],
                    "reason": "bad shortcut"
                  },
                  "reason": "bad shortcut"
                }
              ],
              "narration_brief": "You repair it.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "我尝试强行修理导航",
            "turn_id": "llm-risk-turn",
        }
    )

    assert result["turn_plan"]["decision"] == "risky_action"
    assert result["routing_decision"]["needs_rules_resolution"] is True
    assert result["rules_advice"]["requires_resolution"] is True
    assert [request["tool_name"] for request in result["tool_requests"]] == [
        "run_ruleset_resolver"
    ]
    assert [request["tool_name"] for request in result["turn_plan"]["tool_requests"]] == [
        "run_ruleset_resolver"
    ]
    assert result["tool_results"][0]["tool_name"] == "run_ruleset_resolver"


def test_rules_adjudicator_advice_feeds_resolver_request() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            _rules_advice_response(approach_id="feelings"),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "risky_action",
              "tool_requests": [],
              "narration_brief": "Resolve through the loaded rules.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "我尝试说服守卫",
            "turn_id": "llm-rules-advice-turn",
        }
    )

    request_args = result["tool_requests"][0]["arguments"]
    resolver_payload = result["tool_results"][0]["result"]
    assert request_args["approach"] == "feelings"
    assert resolver_payload["approach"] == "feelings"
    assert any(event["node"] == "advise_rules_with_llm" for event in result["trace_events"])


def test_turn_graph_can_use_independent_advisor_models() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    intent_model = FakeListChatModel(
        responses=[_routing_response(route="risky_action", needs_rules_resolution=True)]
    )
    rules_model = FakeListChatModel(responses=[_rules_advice_response(approach_id="feelings")])
    gm_model = FakeListChatModel(
        responses=[
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "risky_action",
              "tool_requests": [],
              "narration_brief": "Resolve through the loaded rules.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(
        gm_model,
        advisor_models={
            "intent_arbiter": intent_model,
            "rules_adjudicator": rules_model,
        },
    ).invoke(
        {
            "player_input": "我尝试说服守卫",
            "turn_id": "independent-advisor-turn",
        }
    )

    assert result["routing_decision"]["route"] == "risky_action"
    assert result["rules_advice"]["approach_id"] == "feelings"
    assert result["tool_results"][0]["result"]["approach"] == "feelings"


def test_intent_advisor_failure_falls_back_safely() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            "not json",
            "still not json",
            """
            {
              "requires_resolution": true,
              "procedure_id": null,
              "approach_id": null,
              "risk": "risky_uncertain",
              "stakes": "Fallback resolver path.",
              "clarification_question": null,
              "citations": []
            }
            """,
            """
            {
              "intent": {"kind": "action", "confidence": 0.5, "reason": "fallback"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "free_action",
              "tool_requests": [],
              "narration_brief": "Fallback plan.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {"player_input": "我尝试修理导航 1d6", "turn_id": "intent-fallback-turn"}
    )

    route_event = next(
        event for event in result["trace_events"] if event["node"] == "route_with_intent_arbiter"
    )
    assert route_event["fallback"] is True
    assert result["routing_decision"]["route"] == "clarify"
    assert result["routing_decision"]["needs_rules_resolution"] is False
    assert result["tool_results"] == []


def test_rules_advisor_failure_keeps_resolver_path() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            "not json",
            "still not json",
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "free_action",
              "tool_requests": [],
              "narration_brief": "Fallback rules advice still resolves.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {"player_input": "我尝试强行修理导航", "turn_id": "rules-fallback-turn"}
    )

    rules_event = next(
        event for event in result["trace_events"] if event["node"] == "advise_rules_with_llm"
    )
    assert rules_event["fallback"] is True
    assert result["rules_advice"]["requires_resolution"] is True
    assert result["tool_results"][0]["tool_name"] == "run_ruleset_resolver"


def test_resolver_request_protected_arguments_are_sanitized() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            _rules_advice_response(),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "risky_action",
              "tool_requests": [
                {
                  "tool_name": "run_ruleset_resolver",
                  "arguments": {
                    "content_dir": "/tmp/wrong",
                    "ruleset_id": "wrong",
                    "action": "wrong action",
                    "session_id": "wrong",
                    "turn_id": "wrong"
                  },
                  "reason": "malformed resolver request"
                }
              ],
              "narration_brief": "Resolve through the loaded rules.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "我尝试强行修理导航",
            "session_id": "protected-session",
            "turn_id": "protected-turn",
        }
    )

    arguments = result["tool_requests"][0]["arguments"]
    assert arguments["ruleset_id"] == result["ruleset_id"]
    assert arguments["action"] == "我尝试强行修理导航"
    assert arguments["session_id"] == "protected-session"
    assert arguments["turn_id"] == "protected-turn"
    assert arguments["content_dir"].endswith("/content")
    assert result["tool_results"][0]["ok"] is True


def test_resolver_request_drops_llm_roll_and_invalid_approach() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            _rules_advice_response(approach_id="激光"),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "risky_action",
              "tool_requests": [
                {
                  "tool_name": "run_ruleset_resolver",
                  "arguments": {
                    "approach": "激光",
                    "requested_roll": "1d6 base plus bonuses from the loaded rules"
                  },
                  "reason": "resolve with current rules"
                }
              ],
              "narration_brief": "Resolve through the loaded rules.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "I carefully force the hatch open",
            "ruleset_id": "sum_target_smoke",
            "turn_id": "sanitized-request-turn",
        }
    )

    arguments = result["tool_requests"][0]["arguments"]
    assert arguments["approach"] == "force"
    assert "requested_roll" not in arguments
    assert arguments["character_context"] == {"target_total": 7}
    assert result["tool_results"][0]["ok"] is True
    assert result["tool_results"][0]["result"]["dice_expression"] == "2d6"


def test_natural_language_dice_expression_does_not_override_resolver_dice() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            _rules_advice_response(approach_id="lasers"),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "risky_action",
              "tool_requests": [],
              "narration_brief": "Resolve through the loaded rules.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "I scan the pod and roll 2d6",
            "ruleset_id": "lasers_feelings_smoke",
            "turn_id": "natural-language-dice-turn",
        }
    )

    assert result["tool_results"][0]["ok"] is True
    assert result["tool_results"][0]["result"]["dice_expression"] == "1d6"


def test_llm_roll_dice_request_cannot_override_resolver_dice() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            _rules_advice_response(approach_id="lasers"),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "risky_action",
              "tool_requests": [
                {
                  "tool_name": "roll_dice",
                  "arguments": {"expression": "2d6", "roll_id": "llm-roll"},
                  "reason": "incorrect direct dice request"
                }
              ],
              "narration_brief": "Resolve through the loaded rules.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The resolver result is narrated.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "I scan the pod",
            "ruleset_id": "lasers_feelings_smoke",
            "turn_id": "llm-roll-dice-turn",
        }
    )

    assert [request["tool_name"] for request in result["tool_requests"]] == [
        "run_ruleset_resolver"
    ]
    ensure_trace = next(
        event for event in result["trace_events"] if event["node"] == "ensure_resolution_tools"
    )
    assert ensure_trace["rejected_tool_requests"] == [
        {"tool_name": "roll_dice", "reason": "manual_roll_command_only"}
    ]
    assert result["tool_results"][0]["ok"] is True
    assert result["tool_results"][0]["result"]["dice_expression"] == "1d6"


def test_failed_required_resolution_blocks_llm_narration() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_narration_node

    node = build_llm_narration_node(FakeListChatModel(responses=[]))
    result = node(
        {
            "player_input": "I try a risky action.",
            "turn_plan": {"decision": "risky_action"},
            "tool_results": [
                {
                    "tool_name": "run_ruleset_resolver",
                    "ok": False,
                    "error": "resolver failed",
                }
            ],
            "trace_events": [],
        }
    )

    assert "outcome is not resolved yet" in result["final_output"]
    assert "不建立行动结果" not in result["final_output"]
    event = result["trace_events"][-1]
    assert event["node"] == "narrate_with_llm"
    assert event["blocked_reason"] == "required_rules_resolution_failed"


def test_narration_appends_missing_resolver_dice_result() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_narration_node

    node = build_llm_narration_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "final_text": "The hatch buckles but the situation worsens.",
                  "canon_event_draft": null,
                  "memory_candidates": []
                }
                """
            ]
        )
    )
    result = node(
        {
            "player_input": "I force the hatch open.",
            "turn_plan": {"decision": "risky_action"},
            "tool_results": [
                {
                    "tool_name": "run_ruleset_resolver",
                    "ok": True,
                    "result": {
                        "dice_result": {
                            "expression": "2d6",
                            "rolls": [4, 2],
                            "total": 6,
                        }
                    },
                }
            ],
            "trace_events": [],
        }
    )

    assert "Roll: 2d6 -> [4, 2], total 6." in result["final_output"]
    assert result["narration_plan"]["final_text"] == result["final_output"]


def test_compact_narration_contract_accepts_text_only_output() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_narration_node(
        FakeListChatModel(responses=['{"text": "You keep pressure on the visible problem."}'])
    )

    result = node(
        {
            "player_input": "I keep working on it.",
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "trace_events": [],
            "advisor_contract_mode": "compact",
        }
    )

    assert result["final_output"] == "You keep pressure on the visible problem."
    assert result["narration_plan"]["memory_candidates"] == []


def test_intent_arbiter_allows_question_without_keyword_fallback() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(
                route="answer",
                intent_kind="info_query",
                needs_rules_resolution=False,
                allow_direct_answer=True,
            ),
            """
            {
              "intent": {"kind": "info_query", "confidence": 0.9, "reason": "question"},
              "authority": {"ok": true, "reason": "question only"},
              "decision": "answer",
              "tool_requests": [],
              "narration_brief": "This is a question, not an attempted action.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "You can ask how a risky attempt would be handled before doing it.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {"player_input": "Can I attack the locked door?"}
    )

    assert result["routing_decision"]["route"] == "answer"
    assert result["turn_plan"]["decision"] == "answer"
    assert result.get("tool_requests", []) == []


def test_turn_graph_executes_ruleset_resolver() -> None:
    planned = ensure_resolution_tools(
        {
            "player_input": "风险行动",
            "ruleset_id": "lasers_feelings_smoke",
            "turn_id": "turn-test",
            "routing_decision": {"route": "risky_action", "needs_rules_resolution": True},
            "rules_advice": {"requires_resolution": True, "risk": "risky_uncertain"},
            "turn_plan": {"decision": "risky_action", "tool_requests": []},
            "trace_events": [],
        }
    )
    result = execute_deterministic_tools(planned)

    assert result["turn_plan"]["decision"] == "risky_action"
    assert result["tool_results"][0]["tool_name"] == "run_ruleset_resolver"
    assert result["tool_results"][0]["ok"] is True
    assert result["tool_requests"][0]["tool_name"] == "run_ruleset_resolver"


def test_resolver_result_can_transition_scene_from_compiled_scenario() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    model = FakeListChatModel(
        responses=[
            _routing_response(route="risky_action", needs_rules_resolution=True),
            _rules_advice_response(approach_id="force"),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "risky"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "risky_action",
              "tool_requests": [],
              "narration_brief": "Resolve through the loaded rules.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "The hatch opens after the resolver result.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
        ]
    )
    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "I carefully force the hatch open and repair the docking clamp",
            "ruleset_id": "sum_target_smoke",
            "scenario_id": "crystal_stop_singing_smoke",
            "session_id": "s1",
            "turn_id": "x",
        }
    )

    assert result["tool_results"][0]["ok"] is True
    assert result["tool_results"][0]["result"]["band"] == "success"
    assert {"op": "set", "path": ["active_scene"], "value": "scene_2"} in result[
        "tool_results"
    ][0]["result"]["world_patches"]
    assert result["world_projection"]["active_scene"] == "scene_2"
    assert result["world_projection"]["scene"]["id"] == "scene_2"


def test_turn_graph_local_mode_does_not_keyword_route_passive(tmp_path) -> None:
    sqlite_path = tmp_path / "turn.sqlite"
    graph = build_turn_graph()
    result = graph.invoke(
        {
            "player_input": "我等待一下",
            "session_id": "clock-session",
            "thread_id": "clock-session",
            "turn_id": "clock-turn",
            "sqlite_path": str(sqlite_path),
        }
    )

    assert result["turn_plan"]["decision"] == "free_action"
    assert result["world_projection"]["clock"]["value"] == 0
    store = SqliteStore(sqlite_path)
    assert store.get_session_state("clock-session")["clock"]["value"] == 0


def test_turn_graph_persists_turn_once(tmp_path) -> None:
    sqlite_path = tmp_path / "turn.sqlite"
    graph = build_turn_graph()
    state = {
        "player_input": "我检查门口",
        "session_id": "s1",
        "thread_id": "s1",
        "turn_id": "t1",
        "sqlite_path": str(sqlite_path),
    }

    graph.invoke(state)
    graph.invoke(state)

    store = SqliteStore(sqlite_path)
    assert len(store.list_turns("s1")) == 1
    assert len(store.list_canon_events("s1")) == 1


def test_turn_graph_replays_existing_turn_without_reapplying_world_patches(tmp_path) -> None:
    sqlite_path = tmp_path / "turn.sqlite"
    graph = build_turn_graph()
    state = {
        "player_input": "我等待一下",
        "session_id": "s1",
        "thread_id": "s1",
        "turn_id": "t1",
        "sqlite_path": str(sqlite_path),
    }

    first = graph.invoke(state)
    second = graph.invoke(state)

    store = SqliteStore(sqlite_path)
    assert first["world_projection"]["clock"]["value"] == 0
    assert second["world_projection"]["clock"]["value"] == 0
    assert store.get_session_state("s1")["clock"]["value"] == 0
    assert len(store.list_turns("s1")) == 1
    assert len(store.list_canon_events("s1")) == 1
    assert len(store.list_world_patch_applications("s1")) == 0
    assert second["replayed_turn"] is True


def test_world_patch_application_is_idempotent_after_partial_replay(tmp_path) -> None:
    from trpg_agent.graph.build_turn_graph import apply_world_patch_results

    sqlite_path = tmp_path / "turn.sqlite"
    state = {
        "session_id": "s1",
        "thread_id": "s1",
        "turn_id": "t1",
        "sqlite_path": str(sqlite_path),
        "world_projection": {"clock": {"value": 0, "max": 3}},
        "tool_results": [
            {
                "tool_name": "apply_world_patch",
                "ok": True,
                "result": {
                    "world_patches": [
                        {"op": "increment", "path": ["clock", "value"], "value": 1}
                    ]
                },
            }
        ],
        "trace_events": [],
    }

    first = apply_world_patch_results(state)
    replay_state = {**state, "world_projection": first["world_projection"]}
    second = apply_world_patch_results(replay_state)

    store = SqliteStore(sqlite_path)
    assert first["world_projection"]["clock"]["value"] == 1
    assert second["world_projection"]["clock"]["value"] == 1
    assert store.get_session_state("s1")["clock"]["value"] == 1
    assert len(store.list_world_patch_applications("s1")) == 1


def test_durable_turn_graph_uses_sqlite_checkpointer_and_persists_metadata(tmp_path) -> None:
    sqlite_path = tmp_path / "turn.sqlite"
    state = {
        "player_input": "我检查门口",
        "session_id": "s1",
        "thread_id": "s1",
        "turn_id": "t1",
        "sqlite_path": str(sqlite_path),
        "checkpoint_mode": "sqlite",
        "model_metadata": {"provider": "test"},
    }

    with durable_turn_graph(sqlite_path=sqlite_path) as graph:
        result = invoke_turn_graph(graph, state)

    store = SqliteStore(sqlite_path)
    trace = store.get_turn("t1")["trace"]
    assert checkpoint_path_for(sqlite_path).exists()
    assert result["runtime_metadata"]["graph_version"] == "turn-graph-v2"
    assert trace["runtime_metadata"]["checkpoint_mode"] == "sqlite"
    assert trace["runtime_metadata"]["model"]["provider"] == "test"
    assert trace["runtime_metadata"]["resolver_registry_version"] == "resolver-registry-v1"
    assert trace["runtime_metadata"]["content_packages"]["lasers_feelings_smoke"] == "0.1.0"
    assert trace["runtime_metadata"]["ruleset_resolver_id"] == "threshold_d6"


def test_durable_turn_graph_resumes_after_interrupted_apply_step(tmp_path) -> None:
    sqlite_path = tmp_path / "turn.sqlite"
    state = {
        "player_input": "我等待一下",
        "session_id": "s1",
        "thread_id": "s1",
        "turn_id": "t1",
        "sqlite_path": str(sqlite_path),
        "checkpoint_mode": "sqlite",
    }

    with durable_turn_graph(sqlite_path=sqlite_path) as graph:
        config = {"configurable": {"thread_id": f"{state['thread_id']}:{state['turn_id']}"}}
        interrupted = graph.invoke(
            state,
            config,
            interrupt_after=["apply_world_patch_results"],
        )
        resumed = graph.invoke(None, config)

    store = SqliteStore(sqlite_path)
    assert "final_output" not in interrupted
    assert resumed["final_output"].startswith("[turn_plan:free_action]")
    assert resumed["world_projection"]["clock"]["value"] == 0
    assert store.get_session_state("s1")["clock"]["value"] == 0
    assert len(store.list_world_patch_applications("s1")) == 0
    assert len(store.list_turns("s1")) == 1
    assert len(store.list_canon_events("s1")) == 1


def test_durable_turn_graph_handles_30_turn_session_without_duplicate_state(tmp_path) -> None:
    sqlite_path = tmp_path / "turn.sqlite"

    with durable_turn_graph(sqlite_path=sqlite_path) as graph:
        for index in range(30):
            state = {
                "player_input": "我检查门口",
                "session_id": "long-session",
                "thread_id": "long-session",
                "turn_id": f"long-turn-{index}",
                "sqlite_path": str(sqlite_path),
                "checkpoint_mode": "sqlite",
            }
            first = invoke_turn_graph(graph, state)
            second = invoke_turn_graph(graph, state)
            assert second["final_output"] == first["final_output"]
            assert second["replayed_turn"] is True

    store = SqliteStore(sqlite_path)
    assert len(store.list_turns("long-session")) == 30
    assert len(store.list_canon_events("long-session")) == 30


def test_llm_scenario_director_validates_and_applies_package_patches(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_turn_graph_with_model

    sqlite_path = tmp_path / "scenario.sqlite"
    model = FakeListChatModel(
        responses=[
            _routing_response(route="free_action", needs_rules_resolution=False),
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "enters"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "free_action",
              "tool_requests": [],
              "narration_brief": "The scene changes if validated.",
              "citations": []
            }
            """,
            """
            {
              "decision": "transition",
              "proposed_patches": [
                {"operation": "set", "path": "active_scene", "value": "scene_2"},
                {
                  "operation": "add",
                  "path": "/world_projection/revealed_facts/-",
                  "value": {
                    "id": "dock_entrance",
                    "content": "Dock entrance observed.",
                    "source": "scene_1_observation",
                    "visibility": "player"
                  }
                },
                {"op": "set", "path": ["scene", "title"], "value": "invalid"}
              ],
              "player_visible_context": "The action reaches a new scene.",
              "gm_only_reason": "Transition allowed by package affordance.",
              "citations": ["crystal_stop_singing_smoke:compiled_scenario"]
            }
            """,
            """
            {
              "final_text": "You move into the next area.",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
            """
            {
              "ok": true,
              "blocks_output": false,
              "findings": [],
              "revised_final_text": null,
              "reasoning_summary": "ok"
            }
            """,
            """
            {
              "canon_event_draft": null,
              "memory_candidates": [],
              "contradictions": [],
              "should_write": false
            }
            """,
        ]
    )

    result = build_turn_graph_with_model(model).invoke(
        {
            "player_input": "I enter the station.",
            "session_id": "scenario-session",
            "thread_id": "scenario-session",
            "turn_id": "scenario-turn",
            "sqlite_path": str(sqlite_path),
        }
    )

    assert result["world_projection"]["active_scene"] == "scene_2"
    assert result["scenario_director"]["validated_patches"] == [
        {"op": "set", "path": ["active_scene"], "value": "scene_2"},
        {"op": "append", "path": ["revealed_facts"], "value": "Dock entrance observed."},
    ]
    assert result["world_projection"]["revealed_facts"] == ["Dock entrance observed."]
    assert result["scenario_director"]["rejected_patches"]
    assert any(event["node"] == "direct_scenario_with_llm" for event in result["trace_events"])


def test_scenario_director_limits_dense_observation_reveals(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_scenario_director_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "decision": "reveal",
                  "proposed_patches": [
                    {
                      "operation": "add",
                      "path": "/revealed_facts/-",
                      "value": "Green light. Cut marks. Escape pods."
                    },
                    {
                      "operation": "add",
                      "path": "/known_clues/-",
                      "value": "The intruders forced entry through the docking hatch."
                    }
                  ],
                  "player_visible_context": "The docking area demands attention.",
                  "gm_only_reason": "Observation at the hatch.",
                  "citations": ["crystal_stop_singing_smoke:compiled_scenario"]
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "I inspect the hatch.",
            "session_id": "limit-session",
            "turn_id": "limit-turn",
            "sqlite_path": str(tmp_path / "limit.sqlite"),
            "scenario_id": "crystal_stop_singing_smoke",
            "content_dir": str(Path.cwd() / "content"),
            "turn_plan": {"decision": "free_action"},
            "world_projection": {
                "active_scene": "scene_1",
                "clock": {"id": "lullaby_broadcast", "value": 0, "max": 3},
            },
            "tool_results": [],
            "trace_events": [],
        }
    )

    assert result["scenario_director"]["validated_patches"] == [
        {"op": "append", "path": ["revealed_facts"], "value": "Green light."}
    ]
    reasons = {item["reason"] for item in result["scenario_director"]["rejected_patches"]}
    assert "progressive_disclosure_trimmed_dense_reveal" in reasons
    assert "progressive_disclosure_limit_one_reveal_per_turn" in reasons


def test_scenario_director_runs_for_answer_when_route_requests_context(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_scenario_director_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "decision": "no_change",
                  "proposed_patches": [],
                  "player_visible_context": "A public warning light is active.",
                  "gm_only_reason": "Answer route requested visible scene context.",
                  "citations": []
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "What do I see right now?",
            "session_id": "answer-scenario-session",
            "turn_id": "answer-scenario-turn",
            "sqlite_path": str(tmp_path / "answer-scenario.sqlite"),
            "scenario_id": "crystal_stop_singing_smoke",
            "content_dir": str(Path.cwd() / "content"),
            "routing_decision": {
                "route": "answer",
                "needs_scenario_director": True,
            },
            "turn_plan": {"decision": "answer"},
            "world_projection": {"active_scene": "scene_1"},
            "tool_results": [],
            "trace_events": [],
        }
    )

    assert result["scenario_director"]["player_visible_context"] == (
        "A public warning light is active."
    )
    assert result["trace_events"][-1].get("skipped") is not True


def test_scenario_director_accepts_typed_clue_patch_alias(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    node = build_llm_scenario_director_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "decision": "reveal",
                  "proposed_patches": [
                    {
                      "type": "clue",
                      "value": {
                        "id": "hatch_marks",
                        "content": "The hatch frame has tool marks.",
                        "source": "scene",
                        "visibility": "player"
                      }
                    }
                  ],
                  "player_visible_context": "The hatch frame has tool marks.",
                  "gm_only_reason": "Observation at the hatch.",
                  "citations": ["crystal_stop_singing_smoke:compiled_scenario"]
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "I inspect the hatch.",
            "session_id": "typed-clue-session",
            "turn_id": "typed-clue-turn",
            "sqlite_path": str(tmp_path / "typed-clue.sqlite"),
            "scenario_id": "crystal_stop_singing_smoke",
            "content_dir": str(Path.cwd() / "content"),
            "turn_plan": {"decision": "free_action"},
            "world_projection": {
                "active_scene": "scene_1",
                "clock": {"id": "lullaby_broadcast", "value": 0, "max": 3},
            },
            "tool_results": [],
            "trace_events": [],
        }
    )

    assert result["scenario_director"]["validated_patches"] == [
        {"op": "append", "path": ["known_clues"], "value": "The hatch frame has tool marks."}
    ]


def test_critic_repair_cannot_change_tool_results_or_world_state(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": false,
                  "blocks_output": true,
                  "findings": [
                    {
                      "dimension": "narration_quality",
                      "severity": "high",
                      "message": "Output omitted the tool result.",
                      "evidence": "bad"
                    }
                  ],
                  "revised_final_text": "The corrected narration mentions the roll.",
                  "reasoning_summary": "repair narration only"
                }
                """
            ]
        )
    )
    state = {
            "player_input": "I force it.",
            "session_id": "critic-session",
            "turn_id": "critic-turn",
            "sqlite_path": str(tmp_path / "critic.sqlite"),
        "final_output": "bad",
        "turn_plan": {"decision": "risky_action"},
        "tool_results": [
            {
                "tool_name": "run_ruleset_resolver",
                "ok": True,
                "result": {"dice_result": {"expression": "1d6", "rolls": [3], "total": 3}},
            }
        ],
        "world_projection": {"clock": {"value": 0, "max": 3}},
        "trace_events": [],
    }

    result = node(state)

    assert result["final_output"].startswith("The corrected narration")
    assert result["tool_results"] == state["tool_results"]
    assert result["world_projection"] == state["world_projection"]


def test_critic_low_quality_findings_are_advisory_not_failed(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": false,
                  "blocks_output": false,
                  "findings": [
                    {
                      "dimension": "narration_quality",
                      "severity": "low",
                      "message": "Minor wording issue.",
                      "evidence": "minor"
                    },
                    {
                      "dimension": "pacing",
                      "severity": "low",
                      "message": "Minor pacing concern.",
                      "evidence": "minor"
                    }
                  ],
                  "revised_final_text": null,
                  "reasoning_summary": "advisory"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "I inspect.",
            "session_id": "advisory-session",
            "turn_id": "advisory-turn",
            "sqlite_path": str(tmp_path / "advisory.sqlite"),
            "final_output": "You inspect the area. What do you do?",
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "world_projection": {},
            "trace_events": [],
        }
    )

    assert result["critic_report"]["ok"] is True
    assert result["critic_report"]["blocks_output"] is False
    assert result["critic_report"]["repaired"] is False
    assert result["final_output"] == "You inspect the area. What do you do?"


def test_critic_medium_unsupported_fact_is_actionable(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": true,
                  "blocks_output": false,
                  "findings": [
                    {
                      "dimension": "unsupported_fact",
                          "severity": "medium",
                          "message": "Unsupported embellishment with durable implication.",
                      "evidence": "extra adjective"
                    }
                  ],
                  "revised_final_text": null,
                  "reasoning_summary": "defect detected"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "我观察外面。",
            "session_id": "low-unsupported-session",
            "turn_id": "low-unsupported-turn",
            "sqlite_path": str(tmp_path / "low-unsupported.sqlite"),
            "final_output": "你看到一个带有额外修饰的现场。",
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "world_projection": {
                "scene": {"public_summary": "现场有一个已经建立的可见事实。"}
            },
            "trace_events": [],
        }
    )

    assert result["critic_report"]["blocks_output"] is True
    assert result["critic_report"]["forced_block"] is True
    assert result["critic_report"]["repaired"] is True
    assert "额外修饰" not in result["final_output"]


def test_critic_unsupported_fact_forces_repair_before_memory_write(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import (
        build_llm_critic_guardrail_node,
        persist_memory_curation,
    )

    sqlite_path = tmp_path / "critic-block.sqlite"
    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": true,
                  "blocks_output": false,
                  "findings": [
                    {
                      "dimension": "unsupported_fact",
                      "severity": "high",
                      "message": "The narration names a tool not established by context.",
                      "evidence": "phase cutter"
                    }
                  ],
                  "revised_final_text": null,
                  "reasoning_summary": "defect detected"
                }
                """
            ]
        )
    )
    state = {
        "player_input": "我检查痕迹",
        "session_id": "critic-session",
        "turn_id": "critic-turn",
        "sqlite_path": str(sqlite_path),
        "final_output": "你看到某个未建立的专名工具。",
        "turn_plan": {"decision": "free_action"},
        "tool_results": [],
        "world_projection": {},
        "memory_curation": {
            "canon_event_draft": None,
            "memory_candidates": [
                {
                    "kind": "episodic_summary",
                    "text": "Should persist after repair.",
                    "scope": "session",
                    "confidence": 0.9,
                    "metadata": {"visibility": "public"},
                }
            ],
            "contradictions": [],
            "should_write": True,
        },
        "trace_events": [],
    }

    criticized = node(state)
    persisted = persist_memory_curation(criticized)

    assert criticized["critic_report"]["blocks_output"] is True
    assert criticized["critic_report"]["repaired"] is True
    assert "能确认的现场迹象" in criticized["final_output"]
    store = SqliteStore(sqlite_path)
    store.migrate()
    assert persisted["trace_events"][-1]["persisted"] == 1
    assert [memory["text"] for memory in store.list_memories("critic-session")] == [
        "Should persist after repair."
    ]


def test_critic_player_agency_finding_uses_deterministic_fallback(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": false,
                  "blocks_output": true,
                  "findings": [
                    {
                      "dimension": "player_agency",
                      "severity": "low",
                      "message": "Narration describes the PC's desire.",
                      "evidence": "you want"
                    }
                  ],
                  "revised_final_text": "You feel a comforting attraction.",
                  "reasoning_summary": "repair attempted"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "I listen.",
            "session_id": "agency-session",
            "turn_id": "agency-turn",
            "sqlite_path": str(tmp_path / "agency.sqlite"),
            "final_output": "You want to get closer.",
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "world_projection": {},
            "trace_events": [],
        }
    )

    assert "comforting attraction" not in result["final_output"]
    assert result["critic_report"]["deterministic_fallback"] is True


def test_critic_detects_player_agency_language_even_when_model_misses_it(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": true,
                  "blocks_output": false,
                  "findings": [],
                  "revised_final_text": null,
                  "reasoning_summary": "missed issue"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "我聆听旋律。",
            "session_id": "agency-session",
            "turn_id": "agency-missed-turn",
            "sqlite_path": str(tmp_path / "agency-missed.sqlite"),
            "final_output": "你的手指不由自主地想跟着打节拍。",
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "world_projection": {},
            "trace_events": [],
        }
    )

    assert result["critic_report"]["blocks_output"] is True
    assert "不由自主" not in result["final_output"]


def test_critic_fallback_preserves_visible_scene_context(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": true,
                  "blocks_output": false,
                  "findings": [
                    {
                      "dimension": "unsupported_fact",
                      "severity": "high",
                      "message": "Unsupported detail.",
                      "evidence": "extra"
                    }
                  ],
                  "revised_final_text": null,
                  "reasoning_summary": "unsupported detail"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "我检查入口。",
            "session_id": "scene-context-session",
            "turn_id": "scene-context-turn",
            "sqlite_path": str(tmp_path / "scene-context.sqlite"),
            "final_output": "入口有额外未建立细节。",
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "world_projection": {
                "scene": {"public_summary": "门环缓慢旋转，自动导航正偏向危险中心。"}
            },
            "trace_events": [],
        }
    )

    assert "门环缓慢旋转" in result["final_output"]
    assert "额外未建立细节" not in result["final_output"]


def test_critic_fallback_dedupes_semantically_repeated_visible_context(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": false,
                  "blocks_output": true,
                  "findings": [
                    {
                      "dimension": "player_agency",
                      "severity": "medium",
                      "message": "Forced binary choice.",
                      "evidence": "要么"
                    }
                  ],
                  "revised_final_text": null,
                  "reasoning_summary": "deterministic fallback required"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "我检查控制台。",
            "session_id": "scene-context-dedupe-session",
            "turn_id": "scene-context-dedupe-turn",
            "sqlite_path": str(tmp_path / "scene-context-dedupe.sqlite"),
            "final_output": "要么接管控制台，要么被牵着走。",
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "scenario_director": {
                "player_visible_context": (
                    "自动导航正在自行偏转，试图把船驶向危险中心。"
                    "方向舵在你眼前缓缓转动。"
                )
            },
            "world_projection": {
                "scene": {
                    "public_summary": (
                        "外壳有切割痕迹，停泊环缓慢旋转，逃生艇漂浮在外。"
                        "自动导航被干扰，正试图靠近危险中心。"
                    )
                },
                "revealed_facts": ["入口边缘有切割痕迹。"],
            },
            "trace_events": [],
        }
    )

    assert result["final_output"].count("自动导航") == 1
    assert "切割痕迹" in result["final_output"]
    assert "逃生艇" in result["final_output"]
    assert "要么" not in result["final_output"]


def test_critic_does_not_special_case_assumed_target_marker(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": false,
                  "blocks_output": true,
                  "findings": [
                    {
                      "dimension": "narration_quality",
                      "severity": "high",
                      "message": "Final text omitted a broader active threat.",
                      "evidence": "player_visible_context"
                    },
                    {
                      "dimension": "clarification",
                      "severity": "low",
                      "message": "A more specific target could still be requested.",
                      "evidence": "door"
                    }
                  ],
                  "revised_final_text": "你检查门口，同时看到整片战场和远处威胁全部展开。",
                  "reasoning_summary": "over-broad repair suggested"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "我检查门口。",
            "session_id": "critic-session",
            "turn_id": "critic-turn",
            "sqlite_path": str(tmp_path / "critic.sqlite"),
            "routing_decision": {},
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "scenario_director": {"player_visible_context": "远处威胁正在升级。"},
            "world_projection": {
                "scene": {"public_summary": "门口附近有切割痕迹。远处威胁正在升级。"}
            },
            "final_output": "你检查最近的门口。门口附近有切割痕迹。你接下来怎么做？",
            "trace_events": [],
        }
    )

    assert "整片战场" not in result["final_output"]
    assert result["critic_report"]["forced_block"] is True
    assert result["critic_report"]["repaired"] is True
    assert "请说明" in result["final_output"]
    assert [finding["dimension"] for finding in result["critic_report"]["findings"]] == [
        "narration_quality",
        "clarification",
    ]


def test_critic_clarification_finding_forces_question(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": true,
                  "blocks_output": false,
                  "findings": [
                    {
                      "dimension": "clarification",
                      "severity": "low",
                      "message": "Target is ambiguous.",
                      "evidence": "door"
                    }
                  ],
                  "revised_final_text": "You inspect the station.",
                  "reasoning_summary": "ambiguous target"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "我检查门口",
            "session_id": "clarify-session",
            "turn_id": "clarify-turn",
            "sqlite_path": str(tmp_path / "clarify.sqlite"),
            "final_output": "你检查中继站入口。",
            "turn_plan": {"decision": "free_action"},
            "tool_results": [],
            "world_projection": {},
            "trace_events": [],
        }
    )

    assert result["critic_report"]["blocks_output"] is True
    assert "请说明" in result["final_output"]
    assert "You inspect" not in result["final_output"]


def test_critic_clarification_does_not_hide_validated_scenario_progress(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import build_llm_critic_guardrail_node

    node = build_llm_critic_guardrail_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "ok": false,
                  "blocks_output": true,
                  "findings": [
                    {
                      "dimension": "narration_quality",
                      "severity": "medium",
                      "message": "Could be tighter.",
                      "evidence": "long"
                    },
                    {
                      "dimension": "clarification",
                      "severity": "low",
                      "message": "Target might be ambiguous.",
                      "evidence": "door"
                    }
                  ],
                  "revised_final_text": "Please clarify the target.",
                  "reasoning_summary": "overcautious"
                }
                """
            ]
        )
    )

    result = node(
        {
            "player_input": "我检查门口",
            "session_id": "scenario-progress-session",
            "turn_id": "scenario-progress-turn",
            "sqlite_path": str(tmp_path / "scenario-progress.sqlite"),
            "final_output": "你看到入口外有切割痕迹和漂浮的逃生艇。你接下来怎么做？",
            "turn_plan": {"decision": "free_action"},
            "scenario_director": {
                "validated_patches": [
                    {"op": "append", "path": ["revealed_facts"], "value": "入口有切割痕迹。"}
                ]
            },
            "tool_results": [
                {
                    "tool_name": "scenario_director",
                    "ok": True,
                    "result": {
                        "world_patches": [
                            {
                                "op": "append",
                                "path": ["revealed_facts"],
                                "value": "入口有切割痕迹。",
                            }
                        ]
                    },
                }
            ],
            "world_projection": {"revealed_facts": ["入口有切割痕迹。"]},
            "trace_events": [],
        }
    )

    assert result["critic_report"]["blocks_output"] is False
    assert result["critic_report"]["forced_block"] is False
    assert result["final_output"].startswith("你看到入口外")
    assert "clarification" not in [
        finding["dimension"] for finding in result["critic_report"]["findings"]
    ]


def test_memory_curator_persists_visible_and_gm_only_memory(tmp_path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from trpg_agent.graph.build_turn_graph import (
        build_llm_memory_curator_node,
        persist_memory_curation,
    )

    sqlite_path = tmp_path / "memory.sqlite"
    state = {
        "player_input": "Please remember I prefer concise recaps.",
        "session_id": "memory-session",
        "turn_id": "memory-turn",
        "sqlite_path": str(sqlite_path),
        "final_output": "Noted.",
        "turn_plan": {"decision": "answer"},
        "trace_events": [],
    }
    curator = build_llm_memory_curator_node(
        FakeListChatModel(
            responses=[
                """
                {
                  "canon_event_draft": null,
                  "memory_candidates": [
                    {
                      "kind": "player_preference",
                      "text": "The player prefers concise recaps.",
                      "scope": "session",
                      "confidence": 0.9,
                      "metadata": {"visibility": "public"}
                    },
                    {
                      "kind": "procedural_note",
                      "text": "Keep future recaps brief unless the player asks for detail.",
                      "scope": "session",
                      "confidence": 0.8,
                      "metadata": {"visibility": "gm_only"}
                    }
                  ],
                  "contradictions": [],
                  "should_write": true
                }
                """
            ]
        )
    )

    curated = curator(state)
    persisted = persist_memory_curation(curated)
    replayed = persist_memory_curation(curated)

    store = SqliteStore(sqlite_path)
    all_memories = store.recall_memories(query="recaps", scope="memory-session", limit=5)
    visible = store.recall_memories(
        query="recaps",
        scope="memory-session",
        limit=5,
        include_gm_only=False,
    )

    assert persisted["trace_events"][-1]["persisted"] == 2
    assert replayed["trace_events"][-1]["persisted"] == 2
    assert {memory["kind"] for memory in all_memories} == {
        "player_preference",
        "procedural_note",
    }
    assert {memory["kind"] for memory in visible} == {"player_preference"}


def test_memory_curation_with_contradictions_skips_curated_writes(tmp_path) -> None:
    from trpg_agent.graph.build_turn_graph import persist_memory_curation

    sqlite_path = tmp_path / "memory-contradiction.sqlite"
    state = {
        "player_input": "I inspect.",
        "session_id": "memory-contradiction-session",
        "turn_id": "memory-contradiction-turn",
        "sqlite_path": str(sqlite_path),
        "final_output": "You see an unsupported embellishment.",
        "turn_plan": {"decision": "free_action"},
        "memory_curation": {
            "canon_event_draft": {"summary": "Unsupported embellishment."},
            "memory_candidates": [
                {
                    "kind": "canon",
                    "text": "Unsupported embellishment should not persist.",
                    "scope": "session",
                    "confidence": 0.95,
                    "metadata": {"visibility": "public"},
                }
            ],
            "contradictions": ["The embellishment is not supported by source context."],
            "should_write": True,
        },
        "trace_events": [],
    }

    persisted = persist_memory_curation(state)

    store = SqliteStore(sqlite_path)
    store.migrate()
    assert persisted["trace_events"][-1]["persisted"] == 0
    assert persisted["trace_events"][-1]["skipped"] == "memory_curator_contradictions"
    assert store.list_memories("memory-contradiction-session") == []
    assert store.list_canon_events("memory-contradiction-session") == []


def test_memory_curation_filters_only_conflicting_candidates(tmp_path) -> None:
    from trpg_agent.graph.build_turn_graph import persist_memory_curation

    sqlite_path = tmp_path / "memory-partial-contradiction.sqlite"
    state = {
        "player_input": "I dock badly.",
        "session_id": "memory-partial-session",
        "turn_id": "memory-partial-turn",
        "sqlite_path": str(sqlite_path),
        "final_output": "The ship is stuck. Metal tearing suggests unsupported damage.",
        "turn_plan": {"decision": "risky_action"},
        "memory_curation": {
            "canon_event_draft": None,
            "memory_candidates": [
                {
                    "kind": "canon",
                    "text": "Metal tearing caused hull damage.",
                    "scope": "session",
                    "confidence": 0.9,
                    "metadata": {"visibility": "public"},
                },
                {
                    "kind": "unresolved_thread",
                    "text": "The ship is stuck in the docking ring and the clock advanced.",
                    "scope": "session",
                    "confidence": 0.9,
                    "metadata": {"visibility": "public"},
                },
            ],
            "contradictions": ["Metal tearing caused hull damage is unsupported."],
            "should_write": True,
        },
        "trace_events": [],
    }

    persisted = persist_memory_curation(state)

    store = SqliteStore(sqlite_path)
    memories = store.list_memories("memory-partial-session")
    assert persisted["trace_events"][-1]["persisted"] == 1
    assert persisted["trace_events"][-1]["filtered_conflicting_candidates"] == 1
    assert [memory["text"] for memory in memories] == [
        "The ship is stuck in the docking ring and the clock advanced."
    ]

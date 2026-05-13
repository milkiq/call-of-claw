from trpg_agent.context_budget import build_advisor_context


def _state() -> dict:
    return {
        "ruleset_id": "rules",
        "scenario_id": "scenario",
        "active_package_ids": ["rules", "scenario"],
        "package_profiles": [
            {
                "id": "rules",
                "kind": "ruleset",
                "name": "Rules",
                "description": "Rules package",
                "capabilities": ["resolution"],
                "references": [
                    {
                        "id": "core",
                        "title": "Core roll",
                        "visibility": "public",
                        "tags": ["rules"],
                    }
                ],
            },
            {
                "id": "scenario",
                "kind": "scenario",
                "name": "Scenario",
                "description": "Scenario package",
                "capabilities": ["scenes"],
                "references": [
                    {
                        "id": "secret",
                        "title": "Hidden truth",
                        "visibility": "gm_only",
                        "tags": ["secret"],
                    }
                ],
            },
        ],
        "retrieved_spans": [
            {
                "package_id": "rules",
                "reference_id": "core",
                "title": "Core roll",
                "visibility": "public",
                "score": 4,
                "text": "Roll the loaded resolver for risky uncertainty.",
            },
            {
                "package_id": "scenario",
                "reference_id": "secret",
                "title": "Hidden truth",
                "visibility": "gm_only",
                "score": 9,
                "text": "The hidden villain is behind the door.",
            },
        ],
        "world_projection": {
            "active_scene": "scene_1",
            "scene": {
                "id": "scene_1",
                "title": "Docking Bay",
                "public_summary": "The visible docking bay is unstable.",
            },
        },
        "turn_plan": {
            "decision": "risky_action",
            "tool_requests": [{"tool_name": "run_ruleset_resolver", "reason": "Risky."}],
            "narration_brief": "Use the resolver result.",
        },
        "tool_results": [
            {
                "tool_name": "run_ruleset_resolver",
                "ok": True,
                "result": {
                    "band": "success_with_cost",
                    "dice_result": {"expression": "1d6", "rolls": [2], "total": 2},
                    "world_patches": [{"op": "increment", "path": ["clock", "value"], "value": 1}],
                },
                "request": {
                    "tool_name": "run_ruleset_resolver",
                    "arguments": {"large": "x" * 5000},
                    "reason": "Risky action.",
                },
            }
        ],
        "scenario_director": {
            "player_visible_context": "The alarm grows louder.",
            "validated_patches": [],
        },
    }


def test_narrator_context_filters_gm_only_spans() -> None:
    packet = build_advisor_context(_state(), "narrator", mode="enforced")

    context_text = str(packet["context"])
    assert "hidden villain" not in context_text
    assert "Hidden truth" not in context_text
    assert packet["trace"]["sent_chars"] <= packet["trace"]["budget_target_chars"]
    assert any(
        decision["reason_code"] == "visibility_denied_for_role"
        for decision in packet["trace"]["decisions"]
    )


def test_rules_context_keeps_rule_span_and_uses_package_index() -> None:
    packet = build_advisor_context(
        _state(),
        "rules_adjudicator",
        mode="enforced",
        extra_context={"advisor_contract": "RulesAdjudicationAdvice"},
    )

    context = packet["context"]
    assert context["rules_spans"][0]["citation_id"] == "rules:core"
    assert "package_profiles" not in context
    assert context["advisor_contract"] == "RulesAdjudicationAdvice"


def test_tool_result_summary_preserves_dice_and_drops_large_arguments() -> None:
    packet = build_advisor_context(_state(), "narrator", mode="enforced")

    summary = packet["context"]["archivist_packet"]["tool_results"][0]
    assert summary["result"]["dice_result"] == {
        "expression": "1d6",
        "rolls": [2],
        "total": 2,
    }
    assert "large" not in str(summary)


def test_shadow_mode_preserves_legacy_narrator_context_shape() -> None:
    packet = build_advisor_context(_state(), "narrator", mode="shadow")

    context = packet["context"]
    assert "retrieved_spans" in context
    assert "archivist_packet" not in context
    assert any(span["visibility"] == "gm_only" for span in context["retrieved_spans"])
    assert packet["trace"]["mode"] == "shadow"

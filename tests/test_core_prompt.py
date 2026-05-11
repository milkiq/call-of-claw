from trpg_agent.langchain.prompts import RUNTIME_SYSTEM_PROMPTS, validate_core_prompt_is_generic


def test_core_prompt_has_no_smoke_content_terms() -> None:
    forbidden_terms = [
        "Lasers",
        "Feelings",
        "姆姆",
        "水晶",
        "调频器",
        "维加",
        "Sanity",
        "Cthulhu",
    ]
    assert validate_core_prompt_is_generic(forbidden_terms) == []


def test_runtime_prompts_forbid_player_manual_dice_rolls() -> None:
    for role in ["core_gm", "rules_adjudicator", "single_turn_advisor", "narration"]:
        prompt = RUNTIME_SYSTEM_PROMPTS[role].lower()
        assert "do not ask the player to roll" in prompt or "never ask the player to roll" in prompt
    assert "asks the player to roll dice manually" in RUNTIME_SYSTEM_PROMPTS[
        "critic_guardrail"
    ].lower()

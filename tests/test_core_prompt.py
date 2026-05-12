import re

from trpg_agent.app.cli import CHARACTER_CREATION_EXTRACTION_PROMPT
from trpg_agent.eval.judge import JUDGE_PROMPT
from trpg_agent.eval.online_playtest import PLAYER_SIMULATOR_PROMPT
from trpg_agent.langchain import prompts as runtime_prompts
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


def test_runtime_prompts_protect_broad_observation_from_forced_clarification() -> None:
    target_prompt = RUNTIME_SYSTEM_PROMPTS["target_micro_gate"].lower()
    intent_prompt = RUNTIME_SYSTEM_PROMPTS["intent_micro_gate"].lower()
    scenario_prompt = RUNTIME_SYSTEM_PROMPTS["scenario_director"].lower()

    assert "clarification is a last resort" in target_prompt
    assert "set clarify=false" in target_prompt
    assert "current visible situation" in intent_prompt
    assert "at least one grounded visible fact" in scenario_prompt


def test_llm_instruction_prompt_templates_are_english_only() -> None:
    prompts = {
        name: prompt
        for name, prompt in vars(runtime_prompts).items()
        if name.endswith("_PROMPT") and hasattr(prompt, "messages")
    }
    prompts.update(
        {
            "JUDGE_PROMPT": JUDGE_PROMPT,
            "PLAYER_SIMULATOR_PROMPT": PLAYER_SIMULATOR_PROMPT,
            "CHARACTER_CREATION_EXTRACTION_PROMPT": CHARACTER_CREATION_EXTRACTION_PROMPT,
        }
    )

    offenders: dict[str, str] = {}
    for name, prompt in prompts.items():
        for index, message in enumerate(prompt.messages):
            template = getattr(getattr(message, "prompt", None), "template", "")
            if re.search(r"[\u4e00-\u9fff]", template):
                offenders[f"{name}[{index}]"] = template

    assert offenders == {}

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from trpg_agent.content.registry import ContentRegistry
from trpg_agent.tools.dice import roll_dice_once
from trpg_agent.tools.patches import WorldPatch


class RuleValueSpec(BaseModel):
    label: str
    default: int = 50
    keywords: list[str] = Field(default_factory=list)


class DifficultySpec(BaseModel):
    label: str
    divisor: int = 1


class PatchEffectSpec(BaseModel):
    op: Literal["increment"] = "increment"
    path: list[str]
    value: int = 1
    pushed_value: int | None = None
    cap_path: list[str] | None = None
    apply_on: list[str] = Field(default_factory=lambda: ["failure"])


class CheckSpec(BaseModel):
    label: str
    kind: Literal["skill", "attribute", "state", "opposed"] = "skill"
    source: str
    default: int = 50
    procedure_id: str = "skill_check"
    keywords: list[str] = Field(default_factory=list)


class ProcedureSpec(BaseModel):
    label: str
    kind: Literal["skill", "attribute", "state", "opposed"] = "skill"
    default_check_id: str | None = None
    default_difficulty: str = "regular"
    failure_effect: str = "Failure adds pressure or a complication authorized by the scene."
    pushed_failure_effect: str | None = None
    success_effect: str = "The action succeeds."
    state_patches: list[PatchEffectSpec] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class RulesDslPlugin(BaseModel):
    schema_version: int
    id: str
    package_id: str
    driver: Literal["rules_dsl_v1"]
    dice_expression: str = "1d100"
    default_procedure_id: str = "skill_check"
    default_check_id: str | None = None
    difficulty_levels: dict[str, DifficultySpec] = Field(
        default_factory=lambda: {
            "regular": DifficultySpec(label="Regular", divisor=1),
            "hard": DifficultySpec(label="Hard", divisor=2),
            "extreme": DifficultySpec(label="Extreme", divisor=5),
        }
    )
    attributes: dict[str, RuleValueSpec] = Field(default_factory=dict)
    skills: dict[str, RuleValueSpec] = Field(default_factory=dict)
    checks: dict[str, CheckSpec] = Field(default_factory=dict)
    procedures: dict[str, ProcedureSpec] = Field(default_factory=dict)
    result_bands: dict[str, str] = Field(default_factory=dict)
    gm_constraints: list[str] = Field(default_factory=list)


def load_rules_plugin(registry: ContentRegistry, ruleset_id: str) -> RulesDslPlugin | None:
    package = registry.by_id.get(ruleset_id)
    if package is None:
        return None
    plugin_path: Path | None = None
    for reference in package.manifest.references:
        if "plugin" in reference.tags:
            plugin_path = package.reference_path(reference)
            break
    if plugin_path is None:
        candidate = package.root_dir / "plugin.yaml"
        plugin_path = candidate if candidate.exists() else None
    if plugin_path is None:
        return None
    raw = yaml.safe_load(plugin_path.read_text(encoding="utf-8")) or {}
    plugin = RulesDslPlugin.model_validate(raw)
    if plugin.package_id != ruleset_id:
        raise ValueError(
            f"rules plugin package_id mismatch for {ruleset_id}: {plugin.package_id}"
        )
    return plugin


def resolve_rules_dsl(
    *,
    plugin: RulesDslPlugin,
    action: str,
    character_context: dict[str, Any],
    scene_context: dict[str, Any],
    session_id: str,
    turn_id: str,
    sqlite_path: str | None,
    procedure_id: str | None = None,
    check_id: str | None = None,
    difficulty: str | None = None,
    modifier: str | None = None,
    pushed: bool = False,
    load_or_roll: Any,
) -> dict[str, Any]:
    selected_check_id = _select_check_id(plugin, action=action, check_id=check_id)
    check = plugin.checks[selected_check_id]
    selected_procedure_id = _select_procedure_id(
        plugin,
        check=check,
        procedure_id=procedure_id,
    )
    procedure = plugin.procedures[selected_procedure_id]
    selected_difficulty = _select_difficulty(plugin, difficulty or procedure.default_difficulty)
    target_base = _target_base(check, character_context)
    target_value = max(1, target_base // plugin.difficulty_levels[selected_difficulty].divisor)
    candidate_count = 2 if modifier in {"bonus", "penalty"} else 1
    candidates = [
        load_or_roll(
            expression=plugin.dice_expression,
            roll_id=f"{turn_id}:resolver:{plugin.package_id}:{index}",
            seed=f"{session_id}:{selected_check_id}:{selected_difficulty}",
            turn_id=turn_id,
            sqlite_path=sqlite_path,
        )
        for index in range(1, candidate_count + 1)
    ]
    selected_roll = _select_roll(candidates, modifier=modifier)
    total = int(selected_roll["total"])
    success_level = _success_level(total=total, base_target=target_base)
    if total > target_value:
        success_level = "failure"
    successes = 0 if success_level == "failure" else 1
    band = success_level
    consequence = _consequence(
        procedure=procedure,
        success_level=success_level,
        pushed=pushed,
        plugin=plugin,
    )
    patches = _authorized_patches(
        procedure=procedure,
        success_level=success_level,
        pushed=pushed,
        scene_context=scene_context,
    )
    constraints = [
        (
            "Rules plugin result is authoritative: "
            f"procedure={selected_procedure_id}, check={selected_check_id}, "
            f"difficulty={selected_difficulty}, target={target_value}, "
            f"selected_roll={total}, success_level={success_level}."
        ),
        (
            "Dice result must be narrated exactly: "
            f"{plugin.dice_expression} candidates "
            f"{[candidate['total'] for candidate in candidates]}, selected {total}."
        ),
        f"Narrate only this authorized consequence: {consequence}",
        *plugin.gm_constraints,
    ]
    if pushed:
        constraints.append(
            "This was a pushed roll. On failure, narrate the pushed failure effect clearly."
        )
    return {
        "resolver_id": plugin.driver,
        "ruleset_id": plugin.package_id,
        "action": action,
        "approach": selected_check_id,
        "procedure_id": selected_procedure_id,
        "check_id": selected_check_id,
        "target_number": target_value,
        "target_value": target_value,
        "base_target_value": target_base,
        "difficulty_level": selected_difficulty,
        "modifier": modifier or "none",
        "pushed": pushed,
        "dice_expression": plugin.dice_expression,
        "dice_result": selected_roll,
        "roll_candidates": candidates,
        "selected_roll": selected_roll,
        "successes": successes,
        "exact_target_hits": 0,
        "success_level": success_level,
        "band": band,
        "band_label": success_level.replace("_", " ").title(),
        "consequence": consequence,
        "authorized_effects": [consequence],
        "world_patches": [patch.model_dump() for patch in patches],
        "narration_constraints": constraints,
    }


def _select_check_id(
    plugin: RulesDslPlugin,
    *,
    action: str,
    check_id: str | None,
) -> str:
    if check_id and check_id in plugin.checks:
        return check_id
    lowered = action.lower()
    for candidate_id, check in plugin.checks.items():
        if any(keyword.lower() in lowered for keyword in check.keywords):
            return candidate_id
    if plugin.default_check_id and plugin.default_check_id in plugin.checks:
        return plugin.default_check_id
    if not plugin.checks:
        raise ValueError(f"rules plugin {plugin.id} has no checks")
    return next(iter(plugin.checks))


def _select_procedure_id(
    plugin: RulesDslPlugin,
    *,
    check: CheckSpec,
    procedure_id: str | None,
) -> str:
    if procedure_id and procedure_id in plugin.procedures:
        return procedure_id
    if check.procedure_id in plugin.procedures:
        return check.procedure_id
    if plugin.default_procedure_id in plugin.procedures:
        return plugin.default_procedure_id
    if not plugin.procedures:
        raise ValueError(f"rules plugin {plugin.id} has no procedures")
    return next(iter(plugin.procedures))


def _select_difficulty(plugin: RulesDslPlugin, difficulty: str | None) -> str:
    if difficulty and difficulty in plugin.difficulty_levels:
        return difficulty
    if "regular" in plugin.difficulty_levels:
        return "regular"
    return next(iter(plugin.difficulty_levels))


def _target_base(check: CheckSpec, character_context: dict[str, Any]) -> int:
    if check.kind == "skill":
        value = _nested_lookup(character_context, ["skills", check.source], check.default)
    elif check.kind == "attribute":
        value = _nested_lookup(character_context, ["attributes", check.source], check.default)
    else:
        value = _nested_lookup(character_context, [check.source], check.default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(check.default)


def _nested_lookup(root: dict[str, Any], path: list[str], default: Any) -> Any:
    cursor: Any = root
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _select_roll(candidates: list[dict[str, Any]], *, modifier: str | None) -> dict[str, Any]:
    if modifier == "bonus":
        return min(candidates, key=lambda item: int(item["total"]))
    if modifier == "penalty":
        return max(candidates, key=lambda item: int(item["total"]))
    return candidates[0]


def _success_level(*, total: int, base_target: int) -> str:
    if total == 1:
        return "critical_success"
    if total <= max(1, base_target // 5):
        return "extreme_success"
    if total <= max(1, base_target // 2):
        return "hard_success"
    if total <= base_target:
        return "success"
    return "failure"


def _consequence(
    *,
    procedure: ProcedureSpec,
    success_level: str,
    pushed: bool,
    plugin: RulesDslPlugin,
) -> str:
    if success_level != "failure":
        return plugin.result_bands.get(success_level) or procedure.success_effect
    if pushed and procedure.pushed_failure_effect:
        return procedure.pushed_failure_effect
    return plugin.result_bands.get("failure") or procedure.failure_effect


def _authorized_patches(
    *,
    procedure: ProcedureSpec,
    success_level: str,
    pushed: bool,
    scene_context: dict[str, Any],
) -> list[WorldPatch]:
    patches: list[WorldPatch] = []
    for effect in procedure.state_patches:
        if success_level not in effect.apply_on:
            continue
        amount = effect.pushed_value if pushed and effect.pushed_value is not None else effect.value
        if amount <= 0:
            continue
        if effect.cap_path:
            current = _nested_lookup(scene_context, effect.path, 0)
            cap = _nested_lookup(scene_context, effect.cap_path, None)
            try:
                current_value = int(current)
                cap_value = int(cap)
            except (TypeError, ValueError):
                continue
            if current_value >= cap_value:
                continue
            amount = min(amount, cap_value - current_value)
        patches.append(WorldPatch(op=effect.op, path=effect.path, value=amount))
    return patches


def roll_plugin_dice_once(
    *,
    expression: str,
    roll_id: str,
    seed: str,
) -> dict[str, Any]:
    return roll_dice_once(expression, roll_id, seed)

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


class DicePoolSpec(BaseModel):
    base_sides: int = 100
    base_dice: int = 1
    max_dice: int = 1
    bonus_flags: list[str] = Field(default_factory=list)


class DifficultySpec(BaseModel):
    label: str
    divisor: int = 1


class PatchEffectSpec(BaseModel):
    op: Literal["set", "append", "increment"] = "increment"
    path: list[str]
    value: Any = 1
    pushed_value: Any | None = None
    cap_path: list[str] | None = None
    apply_on: list[str] = Field(default_factory=lambda: ["failure"])


class ExactTargetSpec(BaseModel):
    counts_as_success: bool = True
    tag: str = "exact_target"
    prompt: str = "You may ask the GM one question about the current situation."
    effect: str | None = None
    grants_prepared: bool = False


class PluginBandSpec(BaseModel):
    id: str
    label: str
    summary: str
    world_patches: list[dict[str, Any]] = Field(default_factory=list)


class CheckSpec(BaseModel):
    label: str
    kind: Literal["skill", "attribute", "state", "opposed"] = "skill"
    source: str
    default: int = 50
    procedure_id: str = "skill_check"
    success_when: Literal["below", "above", "at_or_below", "at_or_above"] = "at_or_below"
    keywords: list[str] = Field(default_factory=list)


class ProcedureSpec(BaseModel):
    label: str
    kind: Literal["skill", "attribute", "state", "opposed"] = "skill"
    resolution: Literal["target_under", "count_successes", "sum_compare"] = "target_under"
    success_level_mode: Literal["granular", "binary"] = "granular"
    default_check_id: str | None = None
    default_difficulty: str = "regular"
    failure_effect: str = "Failure adds pressure or a complication authorized by the scene."
    pushed_failure_effect: str | None = None
    success_effect: str = "The action succeeds."
    state_patches: list[PatchEffectSpec] = Field(default_factory=list)
    consume_flags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class RulesDslPlugin(BaseModel):
    schema_version: int
    id: str
    package_id: str
    driver: Literal["rules_dsl_v1"]
    dice_expression: str = "1d100"
    dice: DicePoolSpec | None = None
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
    bands: dict[str, PluginBandSpec] = Field(default_factory=dict)
    exact_target: ExactTargetSpec | None = None
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
    expression = _dice_expression(plugin, character_context)
    candidate_count = 2 if modifier in {"bonus", "penalty"} else 1
    candidates = [
        load_or_roll(
            expression=expression,
            roll_id=f"{turn_id}:resolver:{plugin.package_id}:{index}",
            seed=session_id,
            turn_id=turn_id,
            sqlite_path=sqlite_path,
        )
        for index in range(1, candidate_count + 1)
    ]
    selected_roll = _select_roll(candidates, modifier=modifier)
    result = _resolve_by_procedure(
        plugin=plugin,
        procedure=procedure,
        check=check,
        roll=selected_roll,
        target_base=target_base,
        target_value=target_value,
    )
    exact_patches = _exact_target_patches(
        plugin=plugin,
        exact_hits=result["exact_target_hits"],
        turn_id=turn_id,
    )
    patches = [
        *_authorized_patches(
            procedure=procedure,
            success_level=result["success_level"],
            pushed=pushed,
            scene_context=scene_context,
        ),
        *_band_patches(result["band_spec"], scene_context),
        *exact_patches,
        *_consume_flag_patches(procedure, character_context),
    ]
    consequence = _consequence(
        procedure=procedure,
        success_level=result["success_level"],
        band_spec=result["band_spec"],
        pushed=pushed,
        plugin=plugin,
    )
    constraints = [
        (
            "Rules plugin result is authoritative: "
            f"procedure={selected_procedure_id}, check={selected_check_id}, "
            f"difficulty={selected_difficulty}, target={target_value}, "
            f"selected_roll={selected_roll['total']}, success_level={result['success_level']}."
        ),
        (
            "Dice result must be narrated exactly: "
            f"{expression} candidates {[candidate['total'] for candidate in candidates]}, "
            f"selected {selected_roll['total']}."
        ),
        _authorized_consequence_constraint(consequence, patches),
        *plugin.gm_constraints,
    ]
    if pushed:
        constraints.append(
            "This was a pushed roll. On failure, narrate the pushed failure effect clearly."
        )
    if plugin.exact_target and result["exact_target_hits"]:
        constraints.append(
            "The exact-target rule created a pending player opportunity. Before resolving "
            "another risky roll, give the player a clear chance to use or waive it."
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
        "dice_expression": expression,
        "dice_result": selected_roll,
        "roll_candidates": candidates,
        "selected_roll": selected_roll,
        "successes": result["successes"],
        "exact_target_hits": result["exact_target_hits"],
        "success_level": result["success_level"],
        "band": result["band"],
        "band_label": result["band_label"],
        "consequence": consequence,
        "authorized_effects": [consequence],
        "world_patches": [patch.model_dump() for patch in patches],
        "narration_constraints": constraints,
    }


def _resolve_by_procedure(
    *,
    plugin: RulesDslPlugin,
    procedure: ProcedureSpec,
    check: CheckSpec,
    roll: dict[str, Any],
    target_base: int,
    target_value: int,
) -> dict[str, Any]:
    if procedure.resolution == "count_successes":
        successes = 0
        exact_hits = 0
        for value in roll.get("rolls") or []:
            die = int(value)
            if die == target_base:
                exact_hits += 1
                if plugin.exact_target and plugin.exact_target.counts_as_success:
                    successes += 1
            elif _compare(die, target_base, check.success_when):
                successes += 1
        return _success_count_result(plugin, successes=successes, exact_hits=exact_hits)
    if procedure.resolution == "sum_compare":
        total = int(roll["total"])
        successes = 1 if _compare(total, target_value, check.success_when) else 0
        return _success_count_result(plugin, successes=successes, exact_hits=0)

    total = int(roll["total"])
    if procedure.success_level_mode == "binary":
        success_level = "success" if total <= target_value else "failure"
    else:
        success_level = _percentile_success_level(total=total, base_target=target_base)
    if procedure.success_level_mode != "binary" and total > target_value:
        success_level = "failure"
    successes = 0 if success_level == "failure" else 1
    return {
        "successes": successes,
        "exact_target_hits": 0,
        "success_level": success_level,
        "band": success_level,
        "band_label": success_level.replace("_", " ").title(),
        "band_spec": plugin.bands.get(success_level),
    }


def _success_count_result(
    plugin: RulesDslPlugin,
    *,
    successes: int,
    exact_hits: int,
) -> dict[str, Any]:
    band_spec = _band_for_successes(plugin, successes)
    band_id = band_spec.id if band_spec else ("success" if successes else "failure")
    band_label = band_spec.label if band_spec else band_id.replace("_", " ").title()
    return {
        "successes": successes,
        "exact_target_hits": exact_hits,
        "success_level": band_id,
        "band": band_id,
        "band_label": band_label,
        "band_spec": band_spec,
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


def _dice_expression(plugin: RulesDslPlugin, character_context: dict[str, Any]) -> str:
    if plugin.dice is None:
        return plugin.dice_expression
    count = plugin.dice.base_dice
    for flag in plugin.dice.bonus_flags:
        if bool(_nested_lookup(character_context, [flag], False)):
            count += 1
    count = max(1, min(count, plugin.dice.max_dice))
    return f"{count}d{plugin.dice.base_sides}"


def _select_roll(candidates: list[dict[str, Any]], *, modifier: str | None) -> dict[str, Any]:
    if modifier == "bonus":
        return min(candidates, key=lambda item: int(item["total"]))
    if modifier == "penalty":
        return max(candidates, key=lambda item: int(item["total"]))
    return candidates[0]


def _compare(value: int, target: int, mode: str) -> bool:
    if mode == "below":
        return value < target
    if mode == "above":
        return value > target
    if mode == "at_or_above":
        return value >= target
    return value <= target


def _percentile_success_level(*, total: int, base_target: int) -> str:
    if total == 1:
        return "critical_success"
    if total <= max(1, base_target // 5):
        return "extreme_success"
    if total <= max(1, base_target // 2):
        return "hard_success"
    if total <= base_target:
        return "success"
    return "failure"


def _band_for_successes(plugin: RulesDslPlugin, successes: int) -> PluginBandSpec | None:
    if not plugin.bands:
        return None
    numeric_keys = [int(key) for key in plugin.bands if str(key).lstrip("-").isdigit()]
    if not numeric_keys:
        return plugin.bands.get("success" if successes else "failure")
    key = str(min(max(successes, min(numeric_keys)), max(numeric_keys)))
    return plugin.bands[key]


def _consequence(
    *,
    procedure: ProcedureSpec,
    success_level: str,
    band_spec: PluginBandSpec | None,
    pushed: bool,
    plugin: RulesDslPlugin,
) -> str:
    if band_spec:
        return band_spec.summary
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
        if isinstance(amount, int) and amount <= 0:
            continue
        if effect.cap_path:
            current = _nested_lookup(scene_context, effect.path, 0)
            cap = _nested_lookup(scene_context, effect.cap_path, None)
            try:
                current_value = int(current)
                cap_value = int(cap)
                amount = int(amount)
            except (TypeError, ValueError):
                continue
            if current_value >= cap_value:
                continue
            amount = min(amount, cap_value - current_value)
        patches.append(WorldPatch(op=effect.op, path=effect.path, value=amount))
    return patches


def _band_patches(
    band_spec: PluginBandSpec | None,
    scene_context: dict[str, Any],
) -> list[WorldPatch]:
    if band_spec is None:
        return []
    patches: list[WorldPatch] = []
    for raw_patch in band_spec.world_patches:
        patch = WorldPatch.model_validate(raw_patch)
        if _patch_target_exists(scene_context, patch):
            patches.append(patch)
    return patches


def _patch_target_exists(scene_context: dict[str, Any], patch: WorldPatch) -> bool:
    cursor: Any = scene_context
    for key in patch.path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    return bool(patch.path) and isinstance(cursor, dict)


def _exact_target_patches(
    *,
    plugin: RulesDslPlugin,
    exact_hits: int,
    turn_id: str,
) -> list[WorldPatch]:
    if not plugin.exact_target or not exact_hits:
        return []
    return [
        WorldPatch(
            op="append",
            path=["pending_rule_opportunities"],
            value={
                "id": f"{turn_id}:exact-target",
                "ruleset_id": plugin.package_id,
                "tag": plugin.exact_target.tag,
                "source_turn_id": turn_id,
                "prompt": plugin.exact_target.prompt,
                "effect": plugin.exact_target.effect,
                "grants_prepared": plugin.exact_target.grants_prepared,
                "status": "pending",
            },
        )
    ]


def _consume_flag_patches(
    procedure: ProcedureSpec,
    character_context: dict[str, Any],
) -> list[WorldPatch]:
    patches: list[WorldPatch] = []
    for flag in procedure.consume_flags:
        if character_context.get(flag):
            patches.append(WorldPatch(op="set", path=["character_context", flag], value=False))
    return patches


def _authorized_consequence_constraint(
    consequence: str,
    patches: list[WorldPatch],
) -> str:
    if patches:
        return (
            f"Narrate only the resolver consequence '{consequence}' and these authorized "
            f"world patches: {[patch.model_dump() for patch in patches]}. Do not add extra "
            "costs, clock advances, NPC harm, conditions, offscreen events, or complications "
            "that are not listed here."
        )
    return (
        f"Narrate only the resolver consequence '{consequence}'. No world patches, costs, "
        "clock advances, NPC harm, conditions, offscreen events, or complications are authorized."
    )


def roll_plugin_dice_once(
    *,
    expression: str,
    roll_id: str,
    seed: str,
) -> dict[str, Any]:
    return roll_dice_once(expression, roll_id, seed)

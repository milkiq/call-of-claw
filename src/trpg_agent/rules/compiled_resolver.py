from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from trpg_agent.content.compiled import CompiledRuleset, ResolutionBand, load_compiled_ruleset
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.memory.store import SqliteStore
from trpg_agent.rules.resolver_runtime import ResolverRegistry
from trpg_agent.tools.dice import roll_dice_once
from trpg_agent.tools.patches import WorldPatch

RESOLVER_REGISTRY_VERSION = "resolver-registry-v1"


class RulesetResolverInput(BaseModel):
    content_dir: str
    ruleset_id: str
    action: str
    approach: str | None = None
    risk: str = "risky_uncertain"
    character_context: dict[str, Any] = Field(default_factory=dict)
    scene_context: dict[str, Any] = Field(default_factory=dict)
    session_id: str = "default"
    turn_id: str = "turn"
    sqlite_path: str | None = None


class RulesetResolverResult(BaseModel):
    resolver_id: str
    ruleset_id: str
    action: str
    approach: str
    target_number: int
    dice_expression: str
    dice_result: dict[str, Any]
    successes: int
    exact_target_hits: int = 0
    band: str
    band_label: str
    consequence: str
    world_patches: list[dict[str, Any]] = Field(default_factory=list)
    narration_constraints: list[str] = Field(default_factory=list)


def run_ruleset_resolver(
    content_dir: str,
    ruleset_id: str,
    action: str,
    approach: str | None = None,
    risk: str = "risky_uncertain",
    character_context: dict[str, Any] | None = None,
    scene_context: dict[str, Any] | None = None,
    session_id: str = "default",
    turn_id: str = "turn",
    sqlite_path: str | None = None,
) -> dict:
    request = RulesetResolverInput(
        content_dir=content_dir,
        ruleset_id=ruleset_id,
        action=action,
        approach=approach,
        risk=risk,
        character_context=character_context or {},
        scene_context=scene_context or {},
        session_id=session_id,
        turn_id=turn_id,
        sqlite_path=sqlite_path,
    )
    registry = ContentRegistry.load(Path(content_dir), Path(content_dir).parent)
    ruleset = load_compiled_ruleset(registry, ruleset_id)
    resolver = default_resolver_registry().get(ruleset.resolver_id)
    return resolver.resolve(ruleset, request).model_dump()


def resolve_compiled_ruleset(
    ruleset: CompiledRuleset,
    request: RulesetResolverInput,
) -> RulesetResolverResult:
    return ThresholdDiceResolver().resolve(ruleset, request)


class ThresholdDiceResolver:
    id = "threshold_d6"

    def resolve(
        self,
        ruleset: CompiledRuleset,
        request: RulesetResolverInput,
    ) -> RulesetResolverResult:
        selected_approach = _select_approach(ruleset, request.approach, request.action)
        target = int(request.character_context.get("number", ruleset.default_target))
        dice_count = _dice_count(ruleset, request.character_context)
        expression = f"{dice_count}d{ruleset.dice.base_sides}"
        roll_id = f"{request.turn_id}:resolver:{ruleset.package_id}:1"
        dice_result = _load_or_roll(
            expression=expression,
            roll_id=roll_id,
            seed=request.session_id,
            turn_id=request.turn_id,
            sqlite_path=request.sqlite_path,
        )
        approach_spec = ruleset.approaches[selected_approach]
        successes = 0
        exact_hits = 0
        for roll in dice_result["rolls"]:
            if roll == target:
                exact_hits += 1
                if ruleset.exact_target and ruleset.exact_target.counts_as_success:
                    successes += 1
            elif approach_spec.success_when == "below" and roll < target:
                successes += 1
            elif approach_spec.success_when == "above" and roll > target:
                successes += 1

        band = ruleset.band_for_successes(successes)
        patches = _world_patches_for_band(band, request.scene_context, request.action)
        if ruleset.exact_target and exact_hits:
            patches.append(
                WorldPatch(
                    op="append",
                    path=["pending_rule_opportunities"],
                    value={
                        "id": f"{request.turn_id}:exact-target",
                        "ruleset_id": ruleset.package_id,
                        "tag": ruleset.exact_target.tag,
                        "source_turn_id": request.turn_id,
                        "prompt": ruleset.exact_target.prompt,
                        "effect": ruleset.exact_target.effect,
                        "grants_prepared": ruleset.exact_target.grants_prepared,
                        "status": "pending",
                    },
                )
            )
        if request.character_context.get("prepared"):
            patches.append(
                WorldPatch(
                    op="set",
                    path=["character_context", "prepared"],
                    value=False,
                )
            )
        constraints = [
            f"Resolver result is authoritative: {successes} success(es), band={band.id}.",
            f"Dice result must be narrated exactly: {dice_result['expression']} -> "
            f"{dice_result['rolls']} total {dice_result['total']}.",
            _authorized_consequence_constraint(band, patches),
            *ruleset.gm_constraints,
        ]
        if ruleset.exact_target and exact_hits:
            constraints.append(
                "The exact-target rule created a pending player opportunity. Before resolving "
                "another risky roll, give the player a clear chance to use or waive it."
            )
        return RulesetResolverResult(
            resolver_id=ruleset.resolver_id,
            ruleset_id=ruleset.package_id,
            action=request.action,
            approach=selected_approach,
            target_number=target,
            dice_expression=expression,
            dice_result=dice_result,
            successes=successes,
            exact_target_hits=exact_hits,
            band=band.id,
            band_label=band.label,
            consequence=band.summary,
            world_patches=[patch.model_dump() for patch in patches],
            narration_constraints=constraints,
        )


class SumTargetResolver:
    id = "sum_target"

    def resolve(
        self,
        ruleset: CompiledRuleset,
        request: RulesetResolverInput,
    ) -> RulesetResolverResult:
        selected_approach = _select_approach(ruleset, request.approach, request.action)
        target = int(request.character_context.get("target_total", ruleset.default_target))
        dice_count = _dice_count(ruleset, request.character_context)
        expression = f"{dice_count}d{ruleset.dice.base_sides}"
        roll_id = f"{request.turn_id}:resolver:{ruleset.package_id}:1"
        dice_result = _load_or_roll(
            expression=expression,
            roll_id=roll_id,
            seed=request.session_id,
            turn_id=request.turn_id,
            sqlite_path=request.sqlite_path,
        )
        successes = 1 if int(dice_result["total"]) >= target else 0
        band = ruleset.band_for_successes(successes)
        patches = _world_patches_for_band(band, request.scene_context, request.action)
        constraints = [
            f"Resolver result is authoritative: total={dice_result['total']}, "
            f"target={target}, band={band.id}.",
            f"Dice result must be narrated exactly: {dice_result['expression']} -> "
            f"{dice_result['rolls']} total {dice_result['total']}.",
            _authorized_consequence_constraint(band, patches),
            *ruleset.gm_constraints,
        ]
        return RulesetResolverResult(
            resolver_id=ruleset.resolver_id,
            ruleset_id=ruleset.package_id,
            action=request.action,
            approach=selected_approach,
            target_number=target,
            dice_expression=expression,
            dice_result=dice_result,
            successes=successes,
            exact_target_hits=0,
            band=band.id,
            band_label=band.label,
            consequence=band.summary,
            world_patches=[patch.model_dump() for patch in patches],
            narration_constraints=constraints,
        )


class PercentileUnderResolver:
    id = "percentile_under"

    def resolve(
        self,
        ruleset: CompiledRuleset,
        request: RulesetResolverInput,
    ) -> RulesetResolverResult:
        selected_approach = _select_approach(ruleset, request.approach, request.action)
        target = int(request.character_context.get("percentile_target", ruleset.default_target))
        expression = f"{ruleset.dice.base_dice}d{ruleset.dice.base_sides}"
        roll_id = f"{request.turn_id}:resolver:{ruleset.package_id}:1"
        dice_result = _load_or_roll(
            expression=expression,
            roll_id=roll_id,
            seed=request.session_id,
            turn_id=request.turn_id,
            sqlite_path=request.sqlite_path,
        )
        total = int(dice_result["total"])
        successes = 1 if total <= target else 0
        band = ruleset.band_for_successes(successes)
        patches = _world_patches_for_band(band, request.scene_context, request.action)
        constraints = [
            f"Resolver result is authoritative: percentile={total}, "
            f"target={target}, band={band.id}.",
            f"Dice result must be narrated exactly: {dice_result['expression']} -> "
            f"{dice_result['rolls']} total {dice_result['total']}.",
            _authorized_consequence_constraint(band, patches),
            *ruleset.gm_constraints,
        ]
        return RulesetResolverResult(
            resolver_id=ruleset.resolver_id,
            ruleset_id=ruleset.package_id,
            action=request.action,
            approach=selected_approach,
            target_number=target,
            dice_expression=expression,
            dice_result=dice_result,
            successes=successes,
            exact_target_hits=0,
            band=band.id,
            band_label=band.label,
            consequence=band.summary,
            world_patches=[patch.model_dump() for patch in patches],
            narration_constraints=constraints,
        )


def default_resolver_registry() -> ResolverRegistry:
    registry = ResolverRegistry()
    registry.register(ThresholdDiceResolver())
    registry.register(SumTargetResolver())
    registry.register(PercentileUnderResolver())
    return registry


def registered_resolver_ids() -> list[str]:
    return default_resolver_registry().ids()


def _select_approach(ruleset: CompiledRuleset, approach: str | None, action: str) -> str:
    if approach and approach in ruleset.approaches:
        return approach
    if approach:
        lowered_approach = approach.lower()
        for approach_id, spec in ruleset.approaches.items():
            if spec.label.lower() == lowered_approach:
                return approach_id
    lowered = action.lower()
    for approach_id, spec in ruleset.approaches.items():
        if any(keyword.lower() in lowered for keyword in spec.keywords):
            return approach_id
    if ruleset.default_approach and ruleset.default_approach in ruleset.approaches:
        return ruleset.default_approach
    return next(iter(ruleset.approaches))


def _dice_count(
    ruleset: CompiledRuleset,
    character_context: dict[str, Any],
) -> int:
    count = ruleset.dice.base_dice
    for key in ["prepared", "expert", "helped"]:
        if character_context.get(key):
            count += 1
    return max(1, min(count, ruleset.dice.max_dice))


def _load_or_roll(
    *,
    expression: str,
    roll_id: str,
    seed: str,
    turn_id: str,
    sqlite_path: str | None,
) -> dict[str, Any]:
    if sqlite_path:
        store = SqliteStore(Path(sqlite_path))
        store.migrate()
        existing = store.get_dice_roll(roll_id)
        if existing:
            return existing["result"]
    result = roll_dice_once(expression, roll_id, seed)
    if sqlite_path:
        store.insert_dice_roll(
            roll_id=roll_id,
            turn_id=turn_id,
            expression=expression,
            result=result,
        )
    return result


def _world_patches_for_band(
    band: ResolutionBand,
    scene_context: dict[str, Any],
    action: str,
) -> list[WorldPatch]:
    patches: list[WorldPatch] = []
    for raw_patch in getattr(band, "world_patches", []):
        patch = WorldPatch.model_validate(raw_patch)
        if _patch_target_exists(scene_context, patch):
            patches.append(patch)
    patches.extend(_transition_patches_for_band(getattr(band, "id", ""), scene_context, action))
    return patches


def _patch_target_exists(scene_context: dict[str, Any], patch: WorldPatch) -> bool:
    cursor: Any = scene_context
    for key in patch.path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    return bool(patch.path) and isinstance(cursor, dict)


def _transition_patches_for_band(
    band_id: str,
    scene_context: dict[str, Any],
    action: str,
) -> list[WorldPatch]:
    scene = scene_context.get("scene")
    if not isinstance(scene, dict):
        return []
    transitions = scene.get("transitions")
    if not isinstance(transitions, list):
        return []

    lowered_action = action.lower()
    patches: list[WorldPatch] = []
    for transition in transitions:
        if not isinstance(transition, dict):
            continue
        result_bands = transition.get("result_bands") or []
        if result_bands and band_id not in result_bands:
            continue
        keywords = transition.get("action_keywords") or []
        if keywords and not any(str(keyword).lower() in lowered_action for keyword in keywords):
            continue
        target = transition.get("to")
        if not target or target == scene_context.get("active_scene"):
            continue
        patches.append(WorldPatch(op="set", path=["active_scene"], value=str(target)))
        break
    return patches


def _authorized_consequence_constraint(
    band: ResolutionBand,
    patches: list[WorldPatch],
) -> str:
    if patches:
        patch_summary = [patch.model_dump() for patch in patches]
        return (
            f"Narrate only the resolver consequence '{band.summary}' and these authorized "
            f"world patches: {patch_summary}. Do not add extra costs, clock advances, NPC harm, "
            "conditions, offscreen events, or complications that are not listed here."
        )
    return (
        f"Narrate only the resolver consequence '{band.summary}'. No world patches, costs, "
        "clock advances, NPC harm, conditions, offscreen events, or complications are authorized."
    )

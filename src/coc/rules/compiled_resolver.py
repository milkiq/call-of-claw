from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from coc.content.compiled import load_compiled_ruleset
from coc.content.registry import ContentRegistry
from coc.memory.store import SqliteStore
from coc.rules.plugin_runtime import load_rules_plugin, resolve_rules_dsl
from coc.tools.dice import roll_dice_once

RULES_PLUGIN_RUNTIME_VERSION = "rules-plugin-runtime-v1"


class RulesetResolverInput(BaseModel):
    content_dir: str
    ruleset_id: str
    action: str
    approach: str | None = None
    procedure_id: str | None = None
    check_id: str | None = None
    difficulty: str | None = None
    modifier: str | None = None
    pushed: bool = False
    opposed_by: str | None = None
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
    procedure_id: str | None = None
    check_id: str | None = None
    target_number: int
    target_value: int | None = None
    difficulty_level: str | None = None
    modifier: str | None = None
    pushed: bool = False
    dice_expression: str
    dice_result: dict[str, Any]
    roll_candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_roll: dict[str, Any] | None = None
    successes: int
    success_level: str | None = None
    exact_target_hits: int = 0
    band: str
    band_label: str
    consequence: str
    authorized_effects: list[str] = Field(default_factory=list)
    world_patches: list[dict[str, Any]] = Field(default_factory=list)
    narration_constraints: list[str] = Field(default_factory=list)


def run_ruleset_resolver(
    content_dir: str,
    ruleset_id: str,
    action: str,
    approach: str | None = None,
    procedure_id: str | None = None,
    check_id: str | None = None,
    difficulty: str | None = None,
    modifier: str | None = None,
    pushed: bool = False,
    opposed_by: str | None = None,
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
        procedure_id=procedure_id,
        check_id=check_id,
        difficulty=difficulty,
        modifier=modifier,
        pushed=pushed,
        opposed_by=opposed_by,
        risk=risk,
        character_context=character_context or {},
        scene_context=scene_context or {},
        session_id=session_id,
        turn_id=turn_id,
        sqlite_path=sqlite_path,
    )
    registry = ContentRegistry.load(Path(content_dir), Path(content_dir).parent)
    ruleset = load_compiled_ruleset(registry, ruleset_id)
    plugin = load_rules_plugin(registry, ruleset_id)
    if plugin is None:
        raise ValueError(f"ruleset {ruleset_id} has no package-local rules plugin")
    if ruleset.resolver_id != plugin.driver:
        raise ValueError(
            f"ruleset {ruleset_id} resolver_id {ruleset.resolver_id} does not match "
            f"plugin driver {plugin.driver}"
        )
    return resolve_rules_dsl(
        plugin=plugin,
        action=request.action,
        character_context=request.character_context,
        scene_context=request.scene_context,
        session_id=request.session_id,
        turn_id=request.turn_id,
        sqlite_path=request.sqlite_path,
        procedure_id=request.procedure_id,
        check_id=request.check_id or request.approach,
        difficulty=request.difficulty,
        modifier=request.modifier,
        pushed=request.pushed,
        load_or_roll=_load_or_roll,
    )


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

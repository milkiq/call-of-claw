from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from trpg_agent.content.registry import ContentRegistry


class DiceSpec(BaseModel):
    base_sides: int
    base_dice: int
    max_dice: int


class ApproachSpec(BaseModel):
    label: str
    success_when: str
    keywords: list[str] = Field(default_factory=list)


class ExactTargetSpec(BaseModel):
    counts_as_success: bool = True
    tag: str = "exact_target"
    prompt: str = "You may ask the GM one question about the current situation."
    effect: str | None = None
    grants_prepared: bool = False


class ResolutionBand(BaseModel):
    id: str
    label: str
    summary: str
    world_patches: list[dict] = Field(default_factory=list)


class CharacterCreationQuestion(BaseModel):
    id: str
    prompt: str
    field: str
    required: bool = True
    choices: list[str] = Field(default_factory=list)
    numeric_range: list[int] | None = None


class CharacterMechanicalAssignment(BaseModel):
    field: str
    source_question: str | None = None
    default: Any = None
    allowed_values: list[Any] = Field(default_factory=list)
    min: int | float | None = None
    max: int | float | None = None


class CharacterCreationSpec(BaseModel):
    enabled: bool = False
    intro: str = ""
    questions: list[CharacterCreationQuestion] = Field(default_factory=list)
    mechanical_assignments: list[CharacterMechanicalAssignment] = Field(default_factory=list)
    summary_template: str = "Character: {name} - {concept}"


class CompiledRuleset(BaseModel):
    schema_version: int
    id: str
    package_id: str
    resolver_id: str
    plugin_ref: str | None = None
    default_target: int
    default_approach: str | None = None
    dice: DiceSpec
    approaches: dict[str, ApproachSpec]
    exact_target: ExactTargetSpec | None = None
    bands: dict[str, ResolutionBand]
    gm_constraints: list[str] = Field(default_factory=list)
    default_character_context: dict = Field(default_factory=dict)
    rules_model: dict[str, Any] = Field(default_factory=dict)
    character_creation: CharacterCreationSpec = Field(default_factory=CharacterCreationSpec)

    def band_for_successes(self, successes: int) -> ResolutionBand:
        key = str(min(max(successes, 0), max(int(key) for key in self.bands)))
        return self.bands[key]


class ClockState(BaseModel):
    id: str
    value: int = 0
    max: int = 3


class ScenarioState(BaseModel):
    active_scene: str
    clock: ClockState
    revealed_facts: list[str] = Field(default_factory=list)
    known_clues: list[str] = Field(default_factory=list)
    npc_stance: dict[str, str] = Field(default_factory=dict)


class ScenarioMove(BaseModel):
    type: Literal["clock_tick", "reveal", "transition", "consequence"]
    amount: int = 1
    reason: str


class SceneTransition(BaseModel):
    to: str
    when: str
    result_bands: list[str] = Field(default_factory=list)
    action_keywords: list[str] = Field(default_factory=list)


class VisibleSurfaceSpec(BaseModel):
    id: str
    text: str
    tags: list[str] = Field(default_factory=list)
    one_shot: bool = True


class CompiledScene(BaseModel):
    title: str
    public_summary: str
    visible_surfaces: list[VisibleSurfaceSpec] = Field(default_factory=list)
    gm_only: list[str] = Field(default_factory=list)
    default_gm_move: ScenarioMove | None = None
    transitions: list[SceneTransition] = Field(default_factory=list)


class ScenarioEnding(BaseModel):
    when: str
    summary: str


class CompiledScenario(BaseModel):
    schema_version: int
    id: str
    package_id: str
    initial_scene: str
    opening: str
    initial_state: ScenarioState
    scenes: dict[str, CompiledScene]
    endings: dict[str, ScenarioEnding] = Field(default_factory=dict)
    patch_allowlist: list[list[str]] = Field(
        default_factory=lambda: [
            ["active_scene"],
            ["clock", "value"],
            ["revealed_facts"],
            ["known_clues"],
            ["npc_stance", "*"],
        ]
    )
    gm_constraints: list[str] = Field(default_factory=list)

    def scene(self, scene_id: str) -> CompiledScene:
        try:
            return self.scenes[scene_id]
        except KeyError as error:
            raise KeyError(f"Unknown compiled scene: {scene_id}") from error


def _compiled_path(registry: ContentRegistry, package_id: str) -> Path:
    package = registry.by_id[package_id]
    for reference in package.manifest.references:
        if "compiled" in reference.tags:
            return package.reference_path(reference)
    return package.root_dir / "compiled.yaml"


def load_compiled_ruleset(registry: ContentRegistry, package_id: str) -> CompiledRuleset:
    path = _compiled_path(registry, package_id)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CompiledRuleset.model_validate(raw)


def load_compiled_scenario(registry: ContentRegistry, package_id: str) -> CompiledScenario:
    path = _compiled_path(registry, package_id)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CompiledScenario.model_validate(raw)

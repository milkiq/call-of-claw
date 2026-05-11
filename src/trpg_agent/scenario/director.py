from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trpg_agent.content.compiled import CompiledScenario, load_compiled_scenario
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.langchain.structured import ScenarioDirectorDecision
from trpg_agent.tools.patches import WorldPatch


@dataclass(frozen=True)
class ScenarioPatchValidation:
    patches: list[WorldPatch] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_trace(self) -> dict[str, Any]:
        return {
            "accepted": [patch.model_dump() for patch in self.patches],
            "rejected": self.rejected,
        }


def load_scenario_from_context(
    *,
    content_dir: Path,
    scenario_id: str | None,
) -> CompiledScenario | None:
    if not scenario_id:
        return None
    registry = ContentRegistry.load(content_dir, content_dir.parent)
    return load_compiled_scenario(registry, scenario_id)


def validate_scenario_director_decision(
    *,
    decision: ScenarioDirectorDecision,
    state: dict[str, Any],
    content_dir: Path,
    scenario_id: str | None,
) -> ScenarioPatchValidation:
    scenario = load_scenario_from_context(content_dir=content_dir, scenario_id=scenario_id)
    if not scenario:
        return ScenarioPatchValidation(
            rejected=[
                {
                    "reason": "no_loaded_scenario",
                    "patch": patch,
                }
                for patch in decision.proposed_patches
            ]
        )

    accepted: list[WorldPatch] = []
    rejected: list[dict[str, Any]] = []
    existing = {
        tuple(patch)
        for patch in _existing_patch_fingerprints(state.get("tool_results", []))
    }
    for raw_patch in decision.proposed_patches:
        try:
            patch = WorldPatch.model_validate(_normalize_patch(raw_patch))
            _validate_patch_authority(patch, scenario, state.get("world_projection", {}))
        except Exception as error:
            rejected.append({"patch": raw_patch, "reason": str(error)})
            continue
        fingerprint = tuple(_patch_fingerprint(patch))
        if fingerprint in existing:
            rejected.append({"patch": raw_patch, "reason": "duplicate_existing_patch"})
            continue
        existing.add(fingerprint)
        accepted.append(patch)
    return ScenarioPatchValidation(patches=accepted, rejected=rejected)


def _normalize_patch(raw_patch: dict[str, Any]) -> dict[str, Any]:
    patch = dict(raw_patch)
    patch_type = str(patch.get("type") or "").strip().lower()
    if "op" not in patch and patch_type in {"clue", "known_clue", "reveal", "fact"}:
        patch["op"] = "append"
    if "path" not in patch:
        if patch_type in {"clue", "known_clue"}:
            patch["path"] = ["known_clues"]
        elif patch_type in {"reveal", "fact"}:
            patch["path"] = ["revealed_facts"]
    if "op" not in patch and "operation" in patch:
        patch["op"] = patch.pop("operation")
    if "value" not in patch:
        for alias in ("content", "text", "fact"):
            if alias in patch:
                patch["value"] = patch[alias]
                break
    op_aliases = {
        "add": "append",
        "push": "append",
        "replace": "set",
        "update": "set",
        "increase": "increment",
        "tick": "increment",
    }
    if isinstance(patch.get("op"), str):
        patch["op"] = op_aliases.get(str(patch["op"]).lower(), patch["op"])
    path = patch.get("path")
    if isinstance(path, str):
        if path.startswith("/"):
            patch["path"] = [part for part in path.split("/") if part]
        else:
            patch["path"] = [part for part in path.split(".") if part]
    if patch.get("path") and patch["path"][0] in {"world_projection", "world_state"}:
        patch["path"] = patch["path"][1:]
    if patch.get("path") in (["revealed_facts", "-"], ["known_clues", "-"]):
        patch["path"] = [patch["path"][0]]
        if patch.get("op") == "set":
            patch["op"] = "append"
    if patch.get("path") in (["revealed_facts"], ["known_clues"]) and isinstance(
        patch.get("value"),
        dict,
    ):
        for alias in ("content", "text", "fact", "summary"):
            value = patch["value"].get(alias)
            if isinstance(value, str) and value.strip():
                patch["value"] = value
                break
    return patch


def _existing_patch_fingerprints(tool_results: list[dict[str, Any]]) -> list[list[Any]]:
    fingerprints: list[list[Any]] = []
    for tool_result in tool_results:
        if not tool_result.get("ok"):
            continue
        result = tool_result.get("result")
        if not isinstance(result, dict):
            continue
        for raw_patch in result.get("world_patches") or []:
            try:
                fingerprints.append(_patch_fingerprint(WorldPatch.model_validate(raw_patch)))
            except Exception:
                continue
    return fingerprints


def _patch_fingerprint(patch: WorldPatch) -> list[Any]:
    return [patch.op, *patch.path, repr(patch.value)]


def _validate_patch_authority(
    patch: WorldPatch,
    scenario: CompiledScenario,
    world_projection: dict[str, Any],
) -> None:
    if not _path_allowed(patch.path, scenario.patch_allowlist):
        raise ValueError(f"patch path is not scenario-authorized: {'.'.join(patch.path)}")

    if patch.path == ["active_scene"]:
        if patch.op != "set":
            raise ValueError("active_scene patch must use set")
        if patch.value not in scenario.scenes:
            raise ValueError(f"unknown target scene: {patch.value}")
        return

    if patch.path == ["clock", "value"]:
        if patch.op not in {"set", "increment"}:
            raise ValueError("clock.value patch must use set or increment")
        clock = world_projection.get("clock")
        if not isinstance(clock, dict):
            raise ValueError("world state has no clock")
        current = int(clock.get("value", 0))
        maximum = int(clock.get("max", 0))
        next_value = int(patch.value) if patch.op == "set" else current + int(patch.value)
        if next_value < 0 or next_value > maximum:
            raise ValueError("clock.value patch exceeds clock bounds")
        return

    if patch.path in (["revealed_facts"], ["known_clues"]):
        if patch.op != "append":
            raise ValueError("reveal and clue patches must use append")
        if not isinstance(patch.value, str) or not patch.value.strip():
            raise ValueError("reveal and clue patches require non-empty text")
        return

    if len(patch.path) == 2 and patch.path[0] == "npc_stance":
        if patch.op != "set":
            raise ValueError("npc_stance patches must use set")
        if not isinstance(patch.value, str) or not patch.value.strip():
            raise ValueError("npc_stance patch requires non-empty text")
        return

    raise ValueError(f"unsupported scenario patch path: {'.'.join(patch.path)}")


def _path_allowed(path: list[str], allowlist: list[list[str]]) -> bool:
    for allowed in allowlist:
        if allowed == path:
            return True
        if allowed and allowed[-1] == "*" and path[: len(allowed) - 1] == allowed[:-1]:
            return True
    return False

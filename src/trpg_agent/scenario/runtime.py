from __future__ import annotations

from pathlib import Path

from trpg_agent.content.compiled import CompiledScenario, load_compiled_scenario
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.memory.store import SqliteStore


def initial_world_state(scenario: CompiledScenario) -> dict:
    state = scenario.initial_state.model_dump()
    return sync_scene_details_for_scenario(state, scenario)


def sync_scene_details_for_scenario(state: dict, scenario: CompiledScenario) -> dict:
    scene = scenario.scene(state["active_scene"])
    state["scene"] = {
        "id": state["active_scene"],
        "title": scene.title,
        "public_summary": scene.public_summary,
        "transitions": [transition.model_dump() for transition in scene.transitions],
    }
    state["endings"] = {
        ending_id: ending.model_dump()
        for ending_id, ending in scenario.endings.items()
    }
    return state


def sync_scene_details(
    *,
    state: dict,
    content_dir: Path,
    scenario_id: str | None,
) -> dict:
    if not scenario_id or "active_scene" not in state:
        return state
    registry = ContentRegistry.load(content_dir, content_dir.parent)
    scenario = load_compiled_scenario(registry, scenario_id)
    return sync_scene_details_for_scenario(state, scenario)


def load_or_initialize_world_state(
    *,
    store: SqliteStore | None,
    session_id: str,
    content_dir: Path,
    scenario_id: str | None,
) -> dict:
    if store:
        existing = store.get_session_state(session_id)
        if existing is not None:
            return sync_scene_details(
                state=existing,
                content_dir=content_dir,
                scenario_id=scenario_id,
            )
    if not scenario_id:
        return {}
    registry = ContentRegistry.load(content_dir, content_dir.parent)
    scenario = load_compiled_scenario(registry, scenario_id)
    state = initial_world_state(scenario)
    if store:
        store.set_session_state(session_id=session_id, state=state)
    return state


def start_session(
    *,
    store: SqliteStore,
    session_id: str,
    content_dir: Path,
    ruleset_id: str,
    scenario_id: str,
    reset: bool = False,
) -> tuple[dict, str]:
    registry = ContentRegistry.load(content_dir, content_dir.parent)
    scenario = load_compiled_scenario(registry, scenario_id)
    state = initial_world_state(scenario)
    store.migrate()
    store.upsert_session(
        session_id=session_id,
        ruleset_id=ruleset_id,
        scenario_id=scenario_id,
    )
    if reset or store.get_session_state(session_id) is None:
        store.set_session_state(session_id=session_id, state=state)
        store.insert_canon_event(
            event_id=f"{session_id}:session-start",
            session_id=session_id,
            event_type="session_start",
            payload={
                "ruleset_id": ruleset_id,
                "scenario_id": scenario_id,
                "opening": scenario.opening,
                "initial_state": state,
            },
        )
    else:
        state = store.get_session_state(session_id) or state
    return state, scenario.opening

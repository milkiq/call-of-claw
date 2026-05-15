from pathlib import Path

from trpg_agent.content.compiled import load_compiled_ruleset, load_compiled_scenario
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.memory.store import SqliteStore
from trpg_agent.rules.compiled_resolver import registered_resolver_ids, run_ruleset_resolver
from trpg_agent.rules.plugin_runtime import load_rules_plugin
from trpg_agent.scenario.runtime import start_session, sync_scene_details


def test_compiled_packages_load() -> None:
    root = Path.cwd()
    registry = ContentRegistry.load(root / "content", root)

    ruleset = load_compiled_ruleset(registry, "lasers_feelings_smoke")
    sum_ruleset = load_compiled_ruleset(registry, "sum_target_smoke")
    percentile_ruleset = load_compiled_ruleset(registry, "percentile_smoke")
    coc_ruleset = load_compiled_ruleset(registry, "coc7_light_investigation")
    scenario = load_compiled_scenario(registry, "crystal_stop_singing_smoke")
    mystery = load_compiled_scenario(registry, "old_manor_mystery")
    survival = load_compiled_scenario(registry, "storm_watch_survival")
    black_tide = load_compiled_scenario(registry, "black_tide_beacon")

    assert ruleset.resolver_id == "threshold_d6"
    assert sum_ruleset.resolver_id == "sum_target"
    assert percentile_ruleset.resolver_id == "percentile_under"
    assert coc_ruleset.resolver_id == "rules_dsl_v1"
    assert load_rules_plugin(registry, "coc7_light_investigation") is not None
    assert ruleset.default_character_context == {
        "number": 4,
        "expert": False,
        "prepared": False,
        "helped": False,
    }
    assert sum_ruleset.default_character_context == {"target_total": 7}
    assert percentile_ruleset.default_character_context == {"percentile_target": 55}
    assert ruleset.character_creation.enabled is True
    assert {question.id for question in ruleset.character_creation.questions} == {
        "name",
        "concept",
        "number",
        "drive",
        "style",
    }
    assert ruleset.character_creation.mechanical_assignments[0].field == "number"
    assert ruleset.character_creation.mechanical_assignments[0].source_question == "number"
    assert sum_ruleset.character_creation.enabled is True
    assert sum_ruleset.character_creation.mechanical_assignments[0].field == "target_total"
    assert percentile_ruleset.character_creation.enabled is True
    assert percentile_ruleset.character_creation.mechanical_assignments[0].field == (
        "percentile_target"
    )
    assert scenario.initial_scene == "scene_1"
    assert mystery.initial_scene == "foyer"
    assert survival.initial_scene == "station_roof"
    assert black_tide.initial_scene == "harbor_road"
    assert scenario.initial_state.clock.value == 0
    assert {"threshold_d6", "sum_target", "percentile_under"}.issubset(
        set(registered_resolver_ids())
    )


def test_compiled_ruleset_resolver_replays(tmp_path: Path) -> None:
    root = Path.cwd()
    sqlite_path = tmp_path / "resolver.sqlite"

    first = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="lasers_feelings_smoke",
        action="强行修理导航 1d6",
        session_id="s1",
        turn_id="t1",
        sqlite_path=str(sqlite_path),
    )
    second = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="lasers_feelings_smoke",
        action="强行修理导航 1d6",
        session_id="s1",
        turn_id="t1",
        sqlite_path=str(sqlite_path),
    )

    assert first == second
    assert first["approach"] == "lasers"
    assert first["dice_expression"] == "1d6"
    assert first["dice_result"]["roll_id"] == "t1:resolver:lasers_feelings_smoke:1"
    assert first["band"] in {"failure", "success_with_cost", "full_success", "critical_success"}


def test_exact_target_emits_pending_rule_opportunity(tmp_path: Path) -> None:
    root = Path.cwd()

    result = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="lasers_feelings_smoke",
        action="强行修理导航 1d6",
        session_id="exact-seed",
        turn_id="exact-3",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )

    opportunities = [
        patch["value"]
        for patch in result["world_patches"]
        if patch["path"] == ["pending_rule_opportunities"]
    ]

    assert result["dice_result"]["rolls"] == [4]
    assert result["exact_target_hits"] == 1
    assert opportunities
    assert opportunities[0]["status"] == "pending"
    assert opportunities[0]["grants_prepared"] is True
    assert "pending player opportunity" in " ".join(result["narration_constraints"])


def test_compiled_ruleset_owns_approach_keywords(tmp_path: Path) -> None:
    root = Path.cwd()

    result = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="lasers_feelings_smoke",
        action="我说服维加合作",
        session_id="s1",
        turn_id="t2",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )

    assert result["approach"] == "feelings"


def test_second_resolver_family_runs_without_core_graph_changes(tmp_path: Path) -> None:
    root = Path.cwd()

    first = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="sum_target_smoke",
        action="I carefully force the hatch open",
        session_id="s1",
        turn_id="sum-1",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )
    second = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="sum_target_smoke",
        action="I carefully force the hatch open",
        session_id="s1",
        turn_id="sum-1",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )

    assert first == second
    assert first["resolver_id"] == "sum_target"
    assert first["dice_expression"] == "2d6"
    assert first["target_number"] == 7
    assert first["successes"] in {0, 1}


def test_resolver_can_emit_content_owned_scene_transition_patch() -> None:
    root = Path.cwd()
    registry = ContentRegistry.load(root / "content", root)
    scenario = load_compiled_scenario(registry, "crystal_stop_singing_smoke")
    scene_context = scenario.initial_state.model_dump()
    scene_context = sync_scene_details(
        state=scene_context,
        content_dir=root / "content",
        scenario_id="crystal_stop_singing_smoke",
    )

    result = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="sum_target_smoke",
        action="I carefully force the hatch open",
        session_id="s1",
        turn_id="x",
        scene_context=scene_context,
    )

    assert result["band"] == "success"
    assert {"op": "set", "path": ["active_scene"], "value": "scene_2"} in result[
        "world_patches"
    ]
    assert any("Do not add extra costs" in text for text in result["narration_constraints"])


def test_same_action_can_use_different_resolver_families(tmp_path: Path) -> None:
    root = Path.cwd()
    action = "I carefully force the hatch open"

    threshold = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="lasers_feelings_smoke",
        action=action,
        session_id="s1",
        turn_id="cross-1",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )
    sum_target = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="sum_target_smoke",
        action=action,
        session_id="s1",
        turn_id="cross-2",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )

    assert threshold["resolver_id"] == "threshold_d6"
    assert sum_target["resolver_id"] == "sum_target"
    assert threshold["dice_expression"] != sum_target["dice_expression"]


def test_third_resolver_family_runs_without_core_graph_changes(tmp_path: Path) -> None:
    root = Path.cwd()

    result = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="percentile_smoke",
        action="I carefully inspect the failing antenna",
        session_id="s1",
        turn_id="percentile-1",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )

    assert result["resolver_id"] == "percentile_under"
    assert result["dice_expression"] == "1d100"
    assert result["target_number"] == 55
    assert result["successes"] in {0, 1}


def test_rules_dsl_plugin_supports_multi_check_difficulty_and_modifiers(tmp_path: Path) -> None:
    root = Path.cwd()
    character_context = {
        "attributes": {"int": 70, "dex": 50},
        "skills": {"spot_hidden": 60, "mechanical_repair": 40},
        "luck": 35,
        "sanity": 45,
    }

    regular = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="coc7_light_investigation",
        action="我检查灯塔镜片上的盐痕",
        check_id="spot_hidden",
        difficulty="regular",
        character_context=character_context,
        session_id="dsl",
        turn_id="dsl-regular",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )
    hard = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="coc7_light_investigation",
        action="我修理无线电并稳定灯塔机械",
        check_id="mechanical_repair",
        difficulty="hard",
        modifier="bonus",
        character_context=character_context,
        session_id="dsl",
        turn_id="dsl-hard",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )
    replay = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="coc7_light_investigation",
        action="我修理无线电并稳定灯塔机械",
        check_id="mechanical_repair",
        difficulty="hard",
        modifier="bonus",
        character_context=character_context,
        session_id="dsl",
        turn_id="dsl-hard",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )

    assert regular["resolver_id"] == "rules_dsl_v1"
    assert regular["check_id"] == "spot_hidden"
    assert regular["target_value"] == 60
    assert hard == replay
    assert hard["check_id"] == "mechanical_repair"
    assert hard["difficulty_level"] == "hard"
    assert hard["target_value"] == 20
    assert hard["modifier"] == "bonus"
    assert len(hard["roll_candidates"]) == 2
    assert hard["selected_roll"]["total"] == min(
        candidate["total"] for candidate in hard["roll_candidates"]
    )


def test_rules_dsl_plugin_supports_sanity_luck_and_pushed_pressure(tmp_path: Path) -> None:
    root = Path.cwd()
    scene_context = {
        "clock": {"id": "black_tide_pressure", "value": 1, "max": 5},
    }
    character_context = {"luck": 35, "sanity": 45}

    luck = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="coc7_light_investigation",
        action="我靠运气避开突然涌上的黑潮",
        procedure_id="luck_check",
        check_id="luck",
        character_context=character_context,
        scene_context=scene_context,
        session_id="dsl",
        turn_id="dsl-luck",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )
    sanity = run_ruleset_resolver(
        content_dir=str(root / "content"),
        ruleset_id="coc7_light_investigation",
        action="我直视那个会回应我想法的石头",
        procedure_id="sanity_check",
        check_id="sanity",
        difficulty="hard",
        pushed=True,
        character_context=character_context,
        scene_context=scene_context,
        session_id="dsl",
        turn_id="dsl-sanity",
        sqlite_path=str(tmp_path / "resolver.sqlite"),
    )

    assert luck["procedure_id"] == "luck_check"
    assert luck["check_id"] == "luck"
    assert sanity["procedure_id"] == "sanity_check"
    assert sanity["check_id"] == "sanity"
    assert sanity["pushed"] is True
    if sanity["success_level"] == "failure":
        assert {
            "op": "increment",
            "path": ["clock", "value"],
            "value": 2,
        } in sanity["world_patches"]


def test_session_start_initializes_scenario_state(tmp_path: Path) -> None:
    root = Path.cwd()
    store = SqliteStore(tmp_path / "session.sqlite")

    state, opening = start_session(
        store=store,
        session_id="s1",
        content_dir=root / "content",
        ruleset_id="lasers_feelings_smoke",
        scenario_id="crystal_stop_singing_smoke",
        reset=True,
    )

    assert "怎么做" in opening
    assert state["active_scene"] == "scene_1"
    assert store.get_session_state("s1")["clock"]["value"] == 0
    assert len(store.list_canon_events("s1")) == 1


def test_scene_details_sync_after_active_scene_patch() -> None:
    root = Path.cwd()
    state = {
        "active_scene": "scene_2",
        "clock": {"id": "lullaby_broadcast", "value": 1, "max": 3},
        "scene": {"id": "scene_1", "title": "stale"},
    }

    synced = sync_scene_details(
        state=state,
        content_dir=root / "content",
        scenario_id="crystal_stop_singing_smoke",
    )

    assert synced["scene"]["id"] == "scene_2"
    assert synced["scene"]["title"] == "海盗在唱反调"

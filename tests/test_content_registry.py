from pathlib import Path

from coc.content.compiled import load_compiled_scenario
from coc.content.packages import PackageKind
from coc.content.registry import ContentRegistry
from coc.content.retrieval import search_registry_text, search_registry_text_indexed
from coc.content.visibility import AccessMode


def test_content_registry_loads_smoke_packages() -> None:
    root = Path.cwd()
    registry = ContentRegistry.load(root / "content", root)

    assert registry.validate() == []
    assert len(registry.by_kind(PackageKind.AGENT_SKILL)) >= 1
    assert len(registry.by_kind(PackageKind.CAPABILITY_SKILL)) >= 1
    assert len(registry.by_kind(PackageKind.RULESET)) >= 3
    assert len(registry.by_kind(PackageKind.SCENARIO)) >= 3
    assert len(registry.by_kind(PackageKind.EVALUATOR)) >= 1


def test_content_retrieval_enforces_visibility() -> None:
    root = Path.cwd()
    registry = ContentRegistry.load(root / "content", root)

    gm_hits = search_registry_text(
        registry,
        "维加 调频器",
        package_ids=["crystal_stop_singing_smoke"],
        mode=AccessMode.GM,
    )
    player_hits = search_registry_text(
        registry,
        "维加 调频器",
        package_ids=["crystal_stop_singing_smoke"],
        mode=AccessMode.PLAYER,
    )

    assert gm_hits
    assert player_hits == []


def test_indexed_content_retrieval_enforces_visibility_and_reports_diagnostics(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    registry = ContentRegistry.load(root / "content", root)

    gm_result = search_registry_text_indexed(
        registry,
        "维加 调频器",
        sqlite_path=tmp_path / "content-index.sqlite",
        package_ids=["crystal_stop_singing_smoke"],
        mode=AccessMode.GM,
    )
    player_result = search_registry_text_indexed(
        registry,
        "维加 调频器",
        sqlite_path=tmp_path / "content-index.sqlite",
        package_ids=["crystal_stop_singing_smoke"],
        mode=AccessMode.PLAYER,
    )

    assert gm_result.spans
    assert player_result.spans == []
    assert gm_result.diagnostics["search_backend"] == "sqlite_fts"
    assert gm_result.diagnostics["files_scanned"] == 0
    assert gm_result.diagnostics["retrieved_chars"] > 0


def test_capability_skill_package_is_reusable_extension() -> None:
    root = Path.cwd()
    registry = ContentRegistry.load(root / "content", root)

    active_ids = registry.resolve_active_package_ids(["crystal_stop_singing_smoke"])
    profiles = registry.package_profiles(active_ids)

    assert "clue_hygiene_skill" in active_ids
    skill_profile = next(profile for profile in profiles if profile["id"] == "clue_hygiene_skill")
    assert skill_profile["kind"] == PackageKind.CAPABILITY_SKILL.value
    assert skill_profile["disclosure"]["advisor_manifest_first"] is True


def test_compiled_scenario_visible_surfaces_are_public_not_gm_only() -> None:
    root = Path.cwd()
    registry = ContentRegistry.load(root / "content", root)

    for package in registry.by_kind(PackageKind.SCENARIO):
        scenario = load_compiled_scenario(registry, package.id)
        surface_count = 0
        for scene in scenario.scenes.values():
            surface_count += len(scene.visible_surfaces)
            for surface in scene.visible_surfaces:
                assert surface.id
                assert surface.text.strip()
                for secret in scene.gm_only:
                    assert surface.text not in secret
                    assert secret not in surface.text
        assert surface_count > 0

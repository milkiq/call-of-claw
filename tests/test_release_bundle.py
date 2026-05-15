import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from coc.app.cli import app
from coc.app.config import load_config
from coc.content.registry import ContentRegistry
from coc.release.builder import ReleaseBuildSpec, build_release_bundle


def _copy_content(root: Path) -> None:
    source = Path.cwd() / "content"
    for relative in [
        Path("rulesets/coc7_light_investigation"),
        Path("scenarios/black_tide_beacon"),
        Path("agent_skills/generic_gm_adjudication"),
        Path("capability_skills/clue_hygiene"),
    ]:
        destination = root / "content" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / relative, destination)


def _write_release_json(root: Path) -> None:
    (root / "release.json").write_text(
        json.dumps(
            {
                "defaultProfile": "fast",
                "defaultRulesetId": "coc7_light_investigation",
                "defaultScenarioId": "black_tide_beacon",
            }
        ),
        encoding="utf-8",
    )


def test_load_config_uses_env_root_and_release_defaults(tmp_path, monkeypatch) -> None:
    _write_release_json(tmp_path)
    monkeypatch.setenv("COC_ROOT", str(tmp_path))

    config = load_config()

    assert config.root_dir == tmp_path.resolve()
    assert config.release_defaults is not None
    assert config.release_defaults.default_profile == "fast"
    assert config.release_defaults.default_ruleset_id == "coc7_light_investigation"
    assert config.release_defaults.default_scenario_id == "black_tide_beacon"


def test_play_uses_release_json_defaults(tmp_path) -> None:
    _copy_content(tmp_path)
    _write_release_json(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["play", "--local", "--session-id", "release-default", "--input", "我观察灯塔", "--json"],
        env={
            "COC_ROOT": str(tmp_path),
            "COC_SQLITE": str(tmp_path / "data" / "release.sqlite"),
        },
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["final_output"]
    assert "coc7_light_investigation" in payload["runtime_metadata"]["content_packages"]
    assert "black_tide_beacon" in payload["runtime_metadata"]["content_packages"]


def test_doctor_reports_missing_and_valid_llm_config(tmp_path) -> None:
    _copy_content(tmp_path)
    _write_release_json(tmp_path)
    runner = CliRunner()
    env = {"COC_ROOT": str(tmp_path)}

    missing = runner.invoke(app, ["doctor", "--json"], env=env)
    assert missing.exit_code == 1
    missing_payload = json.loads(missing.output)
    assert any(
        check["name"] == "llm config" and check["status"] == "error"
        for check in missing_payload["checks"]
    )

    (tmp_path / "llm.config.json").write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "apiKey": "test-key",
                "baseURL": "https://api.example.com/v1",
                "model": "test-model",
            }
        ),
        encoding="utf-8",
    )
    valid = runner.invoke(app, ["doctor", "--json"], env=env)
    assert valid.exit_code == 0
    valid_payload = json.loads(valid.output)
    assert any(
        check["name"] == "llm config" and check["status"] == "ok"
        for check in valid_payload["checks"]
    )
    assert "test-key" not in valid.output


def test_release_builder_creates_safe_bundle(tmp_path) -> None:
    result = build_release_bundle(
        project_root=Path.cwd(),
        spec=ReleaseBuildSpec(
            name="black-tide-test",
            ruleset_id="coc7_light_investigation",
            scenario_id="black_tide_beacon",
            default_profile="balanced",
            output_dir=tmp_path,
            build_executable=False,
        ),
    )

    assert result.bundle_dir.exists()
    assert result.archive_path.exists()
    assert result.executable_path is None
    assert "coc7_light_investigation" in result.included_package_ids
    assert "black_tide_beacon" in result.included_package_ids
    assert "generic_gm_adjudication" in result.included_package_ids
    assert "clue_hygiene_skill" in result.included_package_ids
    assert (result.bundle_dir / "README-PLAY.md").exists()
    assert (result.bundle_dir / "llm.config.example.json").exists()
    assert (result.bundle_dir / "release.json").exists()
    assert (result.bundle_dir / "data").is_dir()
    assert not (result.bundle_dir / "llm.config.json").exists()
    readme = (result.bundle_dir / "README-PLAY.md").read_text(encoding="utf-8")
    assert "./coc play" in readme
    assert ".\\coc.exe play" in readme
    assert ("./" + "tr" + "pg") not in readme
    assert ("tr" + "pg.exe") not in readme
    assert not any((result.bundle_dir / "data").iterdir())
    assert (
        result.bundle_dir / "content" / "rulesets" / "coc7_light_investigation" / "manifest.yaml"
    ).exists()
    assert (
        result.bundle_dir / "content" / "scenarios" / "black_tide_beacon" / "manifest.yaml"
    ).exists()
    registry = ContentRegistry.load(result.bundle_dir / "content", result.bundle_dir)
    assert registry.validate() == []

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from coc.content.registry import ContentRegistry


@dataclass(frozen=True)
class ReleaseBuildSpec:
    name: str
    ruleset_id: str
    scenario_id: str
    default_profile: str = "balanced"
    output_dir: Path = Path("dist/releases")
    force: bool = False
    build_executable: bool = True


@dataclass(frozen=True)
class ReleaseBuildResult:
    bundle_dir: Path
    archive_path: Path
    executable_path: Path | None
    included_package_ids: list[str]


def build_release_bundle(
    *,
    project_root: Path,
    spec: ReleaseBuildSpec,
) -> ReleaseBuildResult:
    project_root = project_root.resolve()
    output_dir = (
        (project_root / spec.output_dir).resolve()
        if not spec.output_dir.is_absolute()
        else spec.output_dir
    )
    registry = ContentRegistry.load(project_root / "content", project_root)
    _require_package(registry, spec.ruleset_id, "ruleset")
    _require_package(registry, spec.scenario_id, "scenario")
    included_package_ids = registry.resolve_active_package_ids([spec.ruleset_id, spec.scenario_id])

    bundle_dir = output_dir / spec.name
    archive_path = output_dir / f"{spec.name}.zip"
    if bundle_dir.exists() or archive_path.exists():
        if not spec.force:
            raise FileExistsError(f"release output already exists: {bundle_dir} or {archive_path}")
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
    if archive_path.exists() and spec.force:
        archive_path.unlink()
    bundle_dir.mkdir(parents=True, exist_ok=False)
    (bundle_dir / "data").mkdir()

    _copy_content_packages(
        registry=registry,
        content_dir=project_root / "content",
        bundle_content_dir=bundle_dir / "content",
        package_ids=included_package_ids,
    )
    _write_release_config(bundle_dir, spec)
    _write_llm_config_example(bundle_dir)
    _write_play_readme(bundle_dir, spec)

    executable_path = None
    if spec.build_executable:
        executable_path = _build_pyinstaller_executable(
            project_root=project_root,
            bundle_dir=bundle_dir,
            build_root=output_dir / f".{spec.name}-build",
        )

    archive_base = archive_path.with_suffix("")
    shutil.make_archive(str(archive_base), "zip", root_dir=output_dir, base_dir=spec.name)
    return ReleaseBuildResult(
        bundle_dir=bundle_dir,
        archive_path=archive_path,
        executable_path=executable_path,
        included_package_ids=included_package_ids,
    )


def _require_package(registry: ContentRegistry, package_id: str, expected_kind: str) -> None:
    package = registry.by_id.get(package_id)
    if package is None:
        raise ValueError(f"unknown {expected_kind} package: {package_id}")
    if package.kind.value != expected_kind:
        raise ValueError(f"{package_id} is {package.kind.value}, not {expected_kind}")


def _copy_content_packages(
    *,
    registry: ContentRegistry,
    content_dir: Path,
    bundle_content_dir: Path,
    package_ids: list[str],
) -> None:
    for package_id in package_ids:
        package = registry.by_id[package_id]
        relative_root = package.root_dir.relative_to(content_dir)
        destination = bundle_content_dir / relative_root
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package.root_dir, destination)


def _write_release_config(bundle_dir: Path, spec: ReleaseBuildSpec) -> None:
    payload = {
        "defaultProfile": spec.default_profile,
        "defaultRulesetId": spec.ruleset_id,
        "defaultScenarioId": spec.scenario_id,
    }
    (bundle_dir / "release.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_llm_config_example(bundle_dir: Path) -> None:
    payload = {
        "provider": "openai-compatible",
        "apiKey": "YOUR_API_KEY",
        "baseURL": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0,
        "timeoutSeconds": 90,
    }
    (bundle_dir / "llm.config.example.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_play_readme(bundle_dir: Path, spec: ReleaseBuildSpec) -> None:
    text = f"""# Call of Claw Play Bundle

This bundle includes one playable TRPG setup:

- Ruleset: `{spec.ruleset_id}`
- Scenario: `{spec.scenario_id}`
- Default profile: `{spec.default_profile}`

## Configure LLM

Copy `llm.config.example.json` to `llm.config.json`, then replace `apiKey`, `baseURL`, and
`model` with your own provider settings. Do not share your filled `llm.config.json`.

## macOS / Linux

```bash
cp llm.config.example.json llm.config.json
chmod +x ./coc
./coc doctor
./coc play
```

## Windows PowerShell

```powershell
copy llm.config.example.json llm.config.json
.\\coc.exe doctor
.\\coc.exe play
```

Use `/help` inside play for commands, and `/quit` to exit. The command printed on exit can resume
the same session later.
"""
    (bundle_dir / "README-PLAY.md").write_text(text, encoding="utf-8")


def _build_pyinstaller_executable(
    *,
    project_root: Path,
    bundle_dir: Path,
    build_root: Path,
) -> Path:
    try:
        import PyInstaller.__main__ as pyinstaller
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyInstaller is required for release builds. "
            "Install with `pip install -e '.[release]'`."
        ) from error

    build_root.mkdir(parents=True, exist_ok=True)
    entrypoint = project_root / "src" / "coc" / "app" / "cli.py"
    args = [
        str(entrypoint),
        "--name",
        "coc",
        "--onefile",
        "--console",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(bundle_dir),
        "--workpath",
        str(build_root / "work"),
        "--specpath",
        str(build_root / "spec"),
        "--paths",
        str(project_root / "src"),
        "--collect-submodules",
        "coc",
        "--collect-submodules",
        "langchain",
        "--collect-submodules",
        "langgraph",
        "--collect-submodules",
        "langgraph.checkpoint.sqlite",
        "--collect-submodules",
        "langsmith",
    ]
    pyinstaller.run(args)
    executable_name = "coc.exe" if os.name == "nt" else "coc"
    executable_path = bundle_dir / executable_name
    if not executable_path.exists():
        raise RuntimeError(f"PyInstaller did not create expected executable: {executable_path}")
    return executable_path

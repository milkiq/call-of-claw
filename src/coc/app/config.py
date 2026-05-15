from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseDefaults:
    path: Path
    default_profile: str | None = None
    default_ruleset_id: str | None = None
    default_scenario_id: str | None = None


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    content_dir: Path
    seeds_dir: Path
    data_dir: Path
    sqlite_path: Path
    langsmith_tracing: bool
    langsmith_project: str
    release_defaults: ReleaseDefaults | None = None


def discover_root_dir(root_dir: Path | None = None) -> Path:
    if root_dir is not None:
        return root_dir.resolve()
    env_root = os.getenv("COC_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def load_release_defaults(root_dir: Path) -> ReleaseDefaults | None:
    path = root_dir / "release.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid release config: {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid release config: {path}: expected JSON object")
    return ReleaseDefaults(
        path=path,
        default_profile=_optional_str(raw, "defaultProfile", "default_profile"),
        default_ruleset_id=_optional_str(raw, "defaultRulesetId", "default_ruleset_id"),
        default_scenario_id=_optional_str(raw, "defaultScenarioId", "default_scenario_id"),
    )


def _optional_str(raw: dict, camel_key: str, snake_key: str) -> str | None:
    value = raw.get(camel_key, raw.get(snake_key))
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"release config field {camel_key} must be a string")
    return value


def load_config(root_dir: Path | None = None) -> AppConfig:
    root = discover_root_dir(root_dir)
    data_dir = root / "data"
    return AppConfig(
        root_dir=root,
        content_dir=root / "content",
        seeds_dir=root / "seeds",
        data_dir=data_dir,
        sqlite_path=Path(os.getenv("COC_SQLITE", str(data_dir / "coc.sqlite"))),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"},
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "coc-dev"),
        release_defaults=load_release_defaults(root),
    )

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    content_dir: Path
    seeds_dir: Path
    data_dir: Path
    sqlite_path: Path
    langsmith_tracing: bool
    langsmith_project: str


def load_config(root_dir: Path | None = None) -> AppConfig:
    root = root_dir or Path.cwd()
    data_dir = root / "data"
    return AppConfig(
        root_dir=root,
        content_dir=root / "content",
        seeds_dir=root / "seeds",
        data_dir=data_dir,
        sqlite_path=Path(os.getenv("TRPG_AGENT_SQLITE", str(data_dir / "trpg-agent.sqlite"))),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"},
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "trpg-agent-dev"),
    )

from pathlib import Path

from trpg_agent.app.config import load_config
from trpg_agent.eval.regression import run_regression


def test_regression_smoke(tmp_path: Path) -> None:
    config = load_config(Path.cwd())
    config = config.__class__(
        root_dir=config.root_dir,
        content_dir=config.content_dir,
        seeds_dir=config.seeds_dir,
        data_dir=tmp_path,
        sqlite_path=tmp_path / "regression.sqlite",
        langsmith_tracing=False,
        langsmith_project=config.langsmith_project,
    )

    result = run_regression(config)

    assert result.failed == 0, result.findings

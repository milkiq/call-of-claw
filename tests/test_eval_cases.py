from pathlib import Path

from trpg_agent.app.config import load_config
from trpg_agent.eval.cases import load_eval_cases
from trpg_agent.eval.runner import run_eval_cases


def test_load_eval_cases() -> None:
    cases = load_eval_cases(Path("tests/eval_cases"))

    assert {case.id for case in cases} >= {"core-prompt-generic", "turn-action-bootstrap"}


def test_eval_case_runner(tmp_path: Path) -> None:
    config = load_config(Path.cwd())
    config = config.__class__(
        root_dir=config.root_dir,
        content_dir=config.content_dir,
        seeds_dir=config.seeds_dir,
        data_dir=tmp_path,
        sqlite_path=tmp_path / "eval.sqlite",
        langsmith_tracing=False,
        langsmith_project=config.langsmith_project,
    )

    result = run_eval_cases(config, persist=True)

    assert result.failed == 0, [finding.message for finding in result.findings]
    assert result.total >= 8

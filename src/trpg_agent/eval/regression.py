from __future__ import annotations

from trpg_agent.app.config import AppConfig
from trpg_agent.eval.runner import run_eval_cases
from trpg_agent.eval.scorecard import EvalResult


def run_regression(config: AppConfig) -> EvalResult:
    return run_eval_cases(config, kind="regression")

from __future__ import annotations

from coc.app.config import AppConfig
from coc.eval.runner import run_eval_cases
from coc.eval.scorecard import EvalResult


def run_regression(config: AppConfig) -> EvalResult:
    return run_eval_cases(config, kind="regression")

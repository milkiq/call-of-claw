from __future__ import annotations

import os
from collections.abc import Mapping

from coc.app.config import AppConfig


def configure_langsmith(config: AppConfig, metadata: Mapping[str, str] | None = None) -> None:
    """Configure LangSmith tracing through environment variables.

    LangChain and LangGraph pick these variables up automatically. Tests can leave tracing disabled.
    """

    if not config.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", config.langsmith_project)
    if metadata:
        os.environ["COC_TRACE_METADATA"] = ",".join(
            f"{key}={value}" for key, value in sorted(metadata.items())
        )

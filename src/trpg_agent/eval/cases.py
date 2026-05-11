from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class CaseExpectation(BaseModel):
    output_contains: list[str] = Field(default_factory=list)
    output_not_contains: list[str] = Field(default_factory=list)
    intent_kind: str | None = None
    decision: str | None = None
    graph_output_prefix: str | None = None
    min_retrieved_spans: int | None = None
    no_tool_names: list[str] = Field(default_factory=list)
    required_tool_names: list[str] = Field(default_factory=list)
    required_trace_nodes: list[str] = Field(default_factory=list)
    world_state_equals: dict[str, object] = Field(default_factory=dict)
    no_core_prompt_terms: bool = False
    core_prompt_forbidden_terms: list[str] = Field(default_factory=list)
    deterministic_dice: bool = False
    content_registry_valid: bool = False
    canon_import_idempotent: bool = False
    durable_turn_replay: bool = False


class EvalCase(BaseModel):
    id: str
    kind: Literal["deterministic", "turn"]
    description: str
    input: str | None = None
    ruleset_id: str | None = None
    scenario_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    llm_responses: list[dict[str, Any] | str] = Field(default_factory=list)
    expectation: CaseExpectation = Field(default_factory=CaseExpectation)


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for case_path in sorted(path.glob("*.yaml")):
        raw = yaml.safe_load(case_path.read_text(encoding="utf-8")) or []
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            cases.append(EvalCase.model_validate(entry))
    return cases

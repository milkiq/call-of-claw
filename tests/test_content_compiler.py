from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from trpg_agent.content.compiler import compile_content_draft, write_content_draft
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.rules.plugin_runtime import load_rules_plugin


def test_content_compiler_writes_valid_ruleset_package(tmp_path: Path) -> None:
    draft_payload = {
        "manifest": {
            "schema_version": 1,
            "id": "compiled_test_rules",
            "kind": "ruleset",
            "name": "Compiled Test Rules",
            "description": "A tiny compiled ruleset.",
            "version": "0.1.0",
            "entrypoint": "rules.md",
            "references": [
                {
                    "id": "source_rules",
                    "title": "Source rules",
                    "path": "rules.md",
                    "visibility": "public",
                    "tags": ["rules"],
                },
                {
                    "id": "compiled_rules",
                    "title": "Compiled rules",
                    "path": "compiled.yaml",
                    "visibility": "tool_only",
                    "tags": ["compiled", "rules"],
                },
                {
                    "id": "rules_plugin",
                    "title": "Rules plugin",
                    "path": "plugin.yaml",
                    "visibility": "tool_only",
                    "tags": ["plugin", "rules"],
                },
            ],
        },
        "compiled": {
            "schema_version": 1,
            "id": "compiled_test_rules_compiled",
            "package_id": "compiled_test_rules",
            "resolver_id": "rules_dsl_v1",
            "plugin_ref": "rules_plugin",
            "default_target": 50,
            "dice": {"base_sides": 100, "base_dice": 1, "max_dice": 1},
            "approaches": {
                "notice": {
                    "label": "Notice",
                    "success_when": "below",
                    "keywords": ["notice"],
                }
            },
            "bands": {
                "0": {"id": "failure", "label": "Failure", "summary": "It fails."},
                "1": {"id": "success", "label": "Success", "summary": "It succeeds."},
            },
        },
        "plugin": {
            "schema_version": 1,
            "id": "compiled_test_rules_plugin",
            "package_id": "compiled_test_rules",
            "driver": "rules_dsl_v1",
            "default_procedure_id": "skill_check",
            "default_check_id": "notice",
            "procedures": {
                "skill_check": {
                    "label": "Skill Check",
                    "kind": "skill",
                    "default_check_id": "notice",
                }
            },
            "checks": {
                "notice": {
                    "label": "Notice",
                    "kind": "skill",
                    "source": "notice",
                    "default": 50,
                    "procedure_id": "skill_check",
                }
            },
        },
        "source_filename": "rules.md",
        "source_text": "# Rules\nUse a notice check.",
        "notes": "",
    }
    model = FakeListChatModel(responses=[json.dumps(draft_payload)])

    draft = compile_content_draft(
        model=model,
        kind="ruleset",
        package_id="compiled_test_rules",
        source="# Rules\nUse a notice check.",
    )
    package_dir = write_content_draft(
        kind="ruleset",
        package_id="compiled_test_rules",
        draft=draft,
        output_dir=tmp_path,
    )
    registry = ContentRegistry.load(tmp_path, tmp_path)

    assert package_dir.exists()
    assert registry.validate() == []
    assert load_rules_plugin(registry, "compiled_test_rules").driver == "rules_dsl_v1"


def test_content_compiler_rejects_unknown_scenario_transition() -> None:
    draft_payload = {
        "manifest": {
            "schema_version": 1,
            "id": "bad_scenario",
            "kind": "scenario",
            "name": "Bad Scenario",
            "description": "Invalid transition.",
            "version": "0.1.0",
            "entrypoint": "scenario.md",
            "references": [
                {
                    "id": "compiled_scenario",
                    "title": "Compiled scenario",
                    "path": "compiled.yaml",
                    "visibility": "tool_only",
                    "tags": ["compiled", "scenario"],
                }
            ],
        },
        "compiled": {
            "schema_version": 1,
            "id": "bad_scenario_compiled",
            "package_id": "bad_scenario",
            "initial_scene": "start",
            "opening": "Start.",
            "initial_state": {
                "active_scene": "start",
                "clock": {"id": "clock", "value": 0, "max": 3},
            },
            "scenes": {
                "start": {
                    "title": "Start",
                    "public_summary": "A room.",
                    "transitions": [{"to": "missing", "when": "The player leaves."}],
                }
            },
        },
        "source_filename": "scenario.md",
        "source_text": "# Scenario",
    }
    model = FakeListChatModel(responses=[json.dumps(draft_payload)])

    with pytest.raises(ValueError, match="unknown scene"):
        compile_content_draft(
            model=model,
            kind="scenario",
            package_id="bad_scenario",
            source="# Scenario",
        )

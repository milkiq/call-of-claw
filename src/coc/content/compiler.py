from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from coc.content.compiled import CompiledRuleset, CompiledScenario
from coc.content.packages import ContentManifest
from coc.langchain.structured import invoke_structured_with_repair
from coc.rules.plugin_runtime import RulesDslPlugin

CONTENT_COMPILER_PROMPT_VERSION = "content-compiler-v1"


class ContentCompileDraft(BaseModel):
    manifest: dict[str, Any]
    compiled: dict[str, Any]
    plugin: dict[str, Any] | None = None
    source_filename: str
    source_text: str
    notes: str = ""


CONTENT_COMPILER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You compile tabletop roleplaying content into this project's package format.

Return only JSON matching the requested schema. Keep package-specific rules and scenario facts in
the package data, never in core prompts. For rulesets, emit a manifest, compiled ruleset, and
plugin.yaml data when a deterministic rules DSL can express the rules. For scenarios, emit a
manifest and compiled scenario. Use English for internal field names and ids. Preserve user-facing
content language from the source when appropriate.
""".strip(),
        ),
        (
            "human",
            "Kind: {kind}\nPackage id: {package_id}\n\nSource:\n{source}\n\n"
            "Return JSON matching this schema:\n{schema}",
        ),
    ]
)


def compile_content_draft(
    *,
    model: BaseChatModel,
    kind: Literal["ruleset", "scenario"],
    package_id: str,
    source: str,
) -> ContentCompileDraft:
    draft, _attempts = invoke_structured_with_repair(
        model=model,
        prompt=CONTENT_COMPILER_PROMPT,
        schema=ContentCompileDraft,
        payload={
            "kind": kind,
            "package_id": package_id,
            "source": source,
            "schema": ContentCompileDraft.model_json_schema(),
        },
        model_kwargs={"max_tokens": 4000},
    )
    validate_content_draft(kind=kind, package_id=package_id, draft=draft)
    return draft


def validate_content_draft(
    *,
    kind: Literal["ruleset", "scenario"],
    package_id: str,
    draft: ContentCompileDraft,
) -> None:
    manifest = ContentManifest.model_validate(draft.manifest)
    if manifest.id != package_id:
        raise ValueError(f"manifest id mismatch: {manifest.id} != {package_id}")
    if manifest.kind.value != kind:
        raise ValueError(f"manifest kind mismatch: {manifest.kind.value} != {kind}")
    if kind == "ruleset":
        compiled = CompiledRuleset.model_validate(draft.compiled)
        if compiled.package_id != package_id:
            raise ValueError(f"compiled ruleset package_id mismatch: {compiled.package_id}")
        if draft.plugin is None:
            raise ValueError("ruleset compile draft requires plugin data")
        plugin = RulesDslPlugin.model_validate(draft.plugin)
        if plugin.package_id != package_id:
            raise ValueError(f"rules plugin package_id mismatch: {plugin.package_id}")
        missing_checks = [
            check_id
            for check_id, check in plugin.checks.items()
            if check.procedure_id not in plugin.procedures
        ]
        if missing_checks:
            raise ValueError(f"checks reference missing procedures: {missing_checks}")
    else:
        compiled = CompiledScenario.model_validate(draft.compiled)
        if compiled.package_id != package_id:
            raise ValueError(f"compiled scenario package_id mismatch: {compiled.package_id}")
        for scene_id, scene in compiled.scenes.items():
            for transition in scene.transitions:
                if transition.to not in compiled.scenes:
                    raise ValueError(
                        f"scene {scene_id} transition references unknown scene {transition.to}"
                    )


def draft_to_json(draft: ContentCompileDraft) -> str:
    return json.dumps(draft.model_dump(), ensure_ascii=False, indent=2)


def write_content_draft(
    *,
    kind: Literal["ruleset", "scenario"],
    package_id: str,
    draft: ContentCompileDraft,
    output_dir: Path,
    force: bool = False,
) -> Path:
    validate_content_draft(kind=kind, package_id=package_id, draft=draft)
    package_dir = output_dir / package_id
    if package_dir.exists() and not force:
        raise FileExistsError(f"package directory already exists: {package_dir}")
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "manifest.yaml").write_text(
        yaml.safe_dump(draft.manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (package_dir / "compiled.yaml").write_text(
        yaml.safe_dump(draft.compiled, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (package_dir / draft.source_filename).write_text(draft.source_text, encoding="utf-8")
    if kind == "ruleset" and draft.plugin is not None:
        (package_dir / "plugin.yaml").write_text(
            yaml.safe_dump(draft.plugin, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return package_dir

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PackageKind(StrEnum):
    AGENT_SKILL = "agent_skill"
    CAPABILITY_SKILL = "capability_skill"
    RULESET = "ruleset"
    SCENARIO = "scenario"
    EXTENSION = "extension"
    EVALUATOR = "evaluator"


class Visibility(StrEnum):
    PUBLIC = "public"
    GM_ONLY = "gm_only"
    TOOL_ONLY = "tool_only"


class VisibilityDefaults(BaseModel):
    default: Visibility = Visibility.PUBLIC


class ContentReference(BaseModel):
    id: str
    title: str
    path: str
    visibility: Visibility = Visibility.PUBLIC
    tags: list[str] = Field(default_factory=list)


class ExtensionPromptSpec(BaseModel):
    id: str
    role: str
    reference_id: str


class ExtensionToolSpec(BaseModel):
    id: str
    description: str
    schema_ref: str | None = None


class ProgressiveDisclosurePolicy(BaseModel):
    advisor_manifest_first: bool = True
    default_access: Literal["manifest", "retrieved_spans", "full_package"] = "retrieved_spans"
    require_citations_for_package_specific_decisions: bool = True


class ContentManifest(BaseModel):
    schema_version: int = Field(alias="schema_version")
    id: str
    kind: PackageKind
    name: str
    description: str
    version: str = "0.1.0"
    entrypoint: str
    visibility: VisibilityDefaults = Field(default_factory=VisibilityDefaults)
    dependencies: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    references: list[ContentReference] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    extension_prompts: list[ExtensionPromptSpec] = Field(default_factory=list)
    extension_tools: list[ExtensionToolSpec] = Field(default_factory=list)
    disclosure: ProgressiveDisclosurePolicy = Field(default_factory=ProgressiveDisclosurePolicy)
    tests: list[str] = Field(default_factory=list)


class ContentPackage(BaseModel):
    manifest: ContentManifest
    root_dir: Path
    manifest_path: Path

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def kind(self) -> PackageKind:
        return self.manifest.kind

    @property
    def entrypoint_path(self) -> Path:
        return (self.root_dir / self.manifest.entrypoint).resolve()

    def reference_path(self, reference: ContentReference) -> Path:
        return (self.root_dir / reference.path).resolve()

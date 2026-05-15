from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import ValidationError

from trpg_agent.content.packages import ContentManifest, ContentPackage, PackageKind


class ContentRegistry:
    def __init__(self, packages: list[ContentPackage], project_root: Path):
        self.packages = packages
        self.project_root = project_root.resolve()
        self.by_id = {package.id: package for package in packages}

    @classmethod
    def load(cls, content_dir: Path, project_root: Path | None = None) -> ContentRegistry:
        project = (project_root or Path.cwd()).resolve()
        packages: list[ContentPackage] = []
        if not content_dir.exists():
            return cls([], project)
        for manifest_path in sorted(content_dir.rglob("manifest.yaml")):
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            try:
                manifest = ContentManifest.model_validate(raw)
            except ValidationError as error:
                raise ValueError(f"{manifest_path} failed manifest validation:\n{error}") from error
            packages.append(
                ContentPackage(
                    manifest=manifest,
                    root_dir=manifest_path.parent,
                    manifest_path=manifest_path,
                )
            )
        return cls(packages, project)

    def group_by_kind(self) -> dict[str, list[ContentPackage]]:
        grouped: dict[str, list[ContentPackage]] = defaultdict(list)
        for package in self.packages:
            grouped[package.kind.value].append(package)
        return dict(grouped)

    def by_kind(self, kind: PackageKind) -> list[ContentPackage]:
        return [package for package in self.packages if package.kind == kind]

    def package_profiles(self, package_ids: list[str] | None = None) -> list[dict]:
        selected_ids = set(package_ids or [])
        selected = [
            package
            for package in self.packages
            if not selected_ids or package.id in selected_ids
        ]
        return [
            {
                "id": package.id,
                "kind": package.kind.value,
                "name": package.manifest.name,
                "description": package.manifest.description,
                "version": package.manifest.version,
                "capabilities": package.manifest.capabilities,
                "dependencies": package.manifest.dependencies,
                "extensions": package.manifest.extensions,
                "references": [
                    {
                        "id": reference.id,
                        "title": reference.title,
                        "visibility": reference.visibility.value,
                        "tags": reference.tags,
                    }
                    for reference in package.manifest.references
                ],
                "disclosure": package.manifest.disclosure.model_dump(),
            }
            for package in selected
        ]

    def resolve_active_package_ids(self, package_ids: list[str]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()

        def visit(package_id: str) -> None:
            if package_id in seen or package_id not in self.by_id:
                return
            seen.add(package_id)
            package = self.by_id[package_id]
            resolved.append(package_id)
            for dependency_id in [*package.manifest.dependencies, *package.manifest.extensions]:
                visit(dependency_id)

        for package_id in package_ids:
            visit(package_id)
        return resolved

    def validate(self) -> list[str]:
        issues: list[str] = []
        seen: set[str] = set()
        for package in self.packages:
            if package.id in seen:
                issues.append(f"duplicate package id: {package.id}")
            seen.add(package.id)
            if package.manifest.schema_version != 1:
                issues.append(
                    f"{package.id}: unsupported manifest schema_version "
                    f"{package.manifest.schema_version}"
                )
            if not re.fullmatch(r"\d+\.\d+\.\d+", package.manifest.version):
                issues.append(f"{package.id}: version must use MAJOR.MINOR.PATCH")
            if not package.entrypoint_path.exists():
                issues.append(f"{package.id}: missing entrypoint {package.entrypoint_path}")
            for dependency in package.manifest.dependencies:
                if dependency == package.id:
                    issues.append(f"{package.id}: package cannot depend on itself")
                if dependency not in self.by_id:
                    issues.append(f"{package.id}: missing dependency {dependency}")
            for extension in package.manifest.extensions:
                if extension == package.id:
                    issues.append(f"{package.id}: package cannot extend itself")
                if extension not in self.by_id:
                    issues.append(f"{package.id}: missing extension {extension}")
            for reference in package.manifest.references:
                path = package.reference_path(reference)
                if not path.exists():
                    issues.append(f"{package.id}: missing reference {reference.id} at {path}")
                try:
                    path.relative_to(self.project_root)
                except ValueError:
                    issues.append(f"{package.id}: reference escapes project root: {reference.id}")
                if "compiled" in reference.tags and path.exists():
                    issues.extend(_validate_compiled_reference(package.id, reference.id, path))
                if "plugin" in reference.tags and path.exists():
                    issues.extend(_validate_plugin_reference(package.id, reference.id, path))
            reference_ids = {reference.id for reference in package.manifest.references}
            for prompt in package.manifest.extension_prompts:
                if prompt.reference_id not in reference_ids:
                    issues.append(
                        f"{package.id}: extension prompt {prompt.id} references "
                        f"missing reference {prompt.reference_id}"
                    )
        return issues


def _validate_compiled_reference(
    package_id: str,
    reference_id: str,
    path: Path,
) -> list[str]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        return [f"{package_id}: compiled reference {reference_id} cannot be read: {error}"]
    issues: list[str] = []
    if raw.get("schema_version") != 1:
        issues.append(f"{package_id}: compiled reference {reference_id} must use schema_version 1")
    if raw.get("package_id") != package_id:
        issues.append(
            f"{package_id}: compiled reference {reference_id} package_id mismatch "
            f"({raw.get('package_id')})"
        )
    return issues


def _validate_plugin_reference(
    package_id: str,
    reference_id: str,
    path: Path,
) -> list[str]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        return [f"{package_id}: plugin reference {reference_id} cannot be read: {error}"]
    issues: list[str] = []
    if raw.get("schema_version") != 1:
        issues.append(f"{package_id}: plugin reference {reference_id} must use schema_version 1")
    if raw.get("package_id") != package_id:
        issues.append(
            f"{package_id}: plugin reference {reference_id} package_id mismatch "
            f"({raw.get('package_id')})"
        )
    if raw.get("driver") not in {"rules_dsl_v1"}:
        issues.append(
            f"{package_id}: plugin reference {reference_id} unsupported driver "
            f"({raw.get('driver')})"
        )
    return issues

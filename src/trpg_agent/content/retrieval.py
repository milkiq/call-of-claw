from __future__ import annotations

import re
from dataclasses import dataclass

from trpg_agent.content.packages import ContentPackage
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.content.visibility import AccessMode, can_load_reference


@dataclass(frozen=True)
class RetrievedSpan:
    package_id: str
    reference_id: str
    title: str
    path: str
    text: str
    visibility: str
    score: int = 0

    def to_dict(self) -> dict[str, str | int]:
        return {
            "package_id": self.package_id,
            "reference_id": self.reference_id,
            "title": self.title,
            "path": self.path,
            "text": self.text,
            "visibility": self.visibility,
            "score": self.score,
        }


def query_terms(query: str) -> list[str]:
    """Build lightweight multilingual search terms without assuming a ruleset."""

    lower = query.lower()
    terms = re.findall(r"[a-z0-9_]+", lower)
    for block in re.findall(r"[\u4e00-\u9fff]+", lower):
        if len(block) <= 2:
            terms.append(block)
        else:
            terms.extend(block[index : index + 2] for index in range(len(block) - 1))
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def search_package_text(
    package: ContentPackage,
    query: str,
    *,
    mode: AccessMode = AccessMode.GM,
    limit: int = 5,
) -> list[RetrievedSpan]:
    terms = query_terms(query)
    hits: list[RetrievedSpan] = []
    for reference in package.manifest.references:
        if not can_load_reference(reference, mode):
            continue
        path = package.reference_path(reference)
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        haystack = f"{reference.title}\n{' '.join(reference.tags)}\n{text}".lower()
        score = sum(1 for term in terms if term in haystack) if terms else 1
        if score:
            hits.append(
                RetrievedSpan(
                    package_id=package.id,
                    reference_id=reference.id,
                    title=reference.title,
                    path=str(path),
                    text=text[:4000],
                    visibility=reference.visibility.value,
                    score=score,
                )
            )
    return sorted(hits, key=lambda span: (-span.score, span.package_id, span.reference_id))[:limit]


def search_registry_text(
    registry: ContentRegistry,
    query: str,
    *,
    package_ids: list[str] | None = None,
    mode: AccessMode = AccessMode.GM,
    limit: int = 8,
) -> list[RetrievedSpan]:
    selected_ids = set(package_ids or [])
    packages = [
        package
        for package in registry.packages
        if not selected_ids or package.id in selected_ids
    ]
    hits: list[RetrievedSpan] = []
    for package in packages:
        hits.extend(search_package_text(package, query, mode=mode, limit=limit))
    return sorted(hits, key=lambda span: (-span.score, span.package_id, span.reference_id))[:limit]


def load_reference_text(
    registry: ContentRegistry,
    *,
    package_id: str,
    reference_id: str,
    mode: AccessMode = AccessMode.GM,
) -> RetrievedSpan:
    package = registry.by_id[package_id]
    for reference in package.manifest.references:
        if reference.id != reference_id:
            continue
        if not can_load_reference(reference, mode):
            raise PermissionError(f"Reference {reference_id} is not visible in {mode} mode")
        path = package.reference_path(reference)
        text = path.read_text(encoding="utf-8")
        return RetrievedSpan(
            package_id=package.id,
            reference_id=reference.id,
            title=reference.title,
            path=str(path),
            text=text,
            visibility=reference.visibility.value,
            score=1,
        )
    raise KeyError(f"Unknown reference {package_id}:{reference_id}")

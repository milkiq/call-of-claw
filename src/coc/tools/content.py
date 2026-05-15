from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from coc.content.registry import ContentRegistry
from coc.content.retrieval import load_reference_text, search_registry_text
from coc.content.visibility import AccessMode


class SearchContentInput(BaseModel):
    content_dir: str = Field(description="Project content directory.")
    query: str = Field(description="Natural language query for relevant content.")
    package_ids: list[str] = Field(default_factory=list, description="Optional package filter.")
    mode: AccessMode = Field(default=AccessMode.GM, description="Visibility access mode.")
    limit: int = Field(default=5, ge=1, le=20)


class LoadContentSpanInput(BaseModel):
    content_dir: str = Field(description="Project content directory.")
    package_id: str
    reference_id: str
    mode: AccessMode = Field(default=AccessMode.GM, description="Visibility access mode.")


def search_content(
    content_dir: str,
    query: str,
    package_ids: list[str] | None = None,
    mode: AccessMode = AccessMode.GM,
    limit: int = 5,
) -> list[dict]:
    registry = ContentRegistry.load(Path(content_dir), Path(content_dir).parent)
    return [
        span.to_dict()
        for span in search_registry_text(
            registry,
            query,
            package_ids=package_ids or [],
            mode=mode,
            limit=limit,
        )
    ]


def load_content_span(
    content_dir: str,
    package_id: str,
    reference_id: str,
    mode: AccessMode = AccessMode.GM,
) -> dict:
    registry = ContentRegistry.load(Path(content_dir), Path(content_dir).parent)
    return load_reference_text(
        registry,
        package_id=package_id,
        reference_id=reference_id,
        mode=mode,
    ).to_dict()

from __future__ import annotations

from trpg_agent.content.packages import ContentReference
from trpg_agent.content.visibility import AccessMode, can_load_reference


def enforce_disclosure(reference: ContentReference, mode: AccessMode) -> None:
    if not can_load_reference(reference, mode):
        raise PermissionError(f"Reference {reference.id} is not visible in {mode} mode")

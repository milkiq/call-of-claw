from __future__ import annotations

from enum import StrEnum

from coc.content.packages import ContentReference, Visibility


class AccessMode(StrEnum):
    PLAYER = "player"
    GM = "gm"
    TOOL = "tool"


def can_load_reference(reference: ContentReference, mode: AccessMode) -> bool:
    if reference.visibility == Visibility.PUBLIC:
        return True
    if reference.visibility == Visibility.GM_ONLY:
        return mode in {AccessMode.GM, AccessMode.TOOL}
    if reference.visibility == Visibility.TOOL_ONLY:
        return mode == AccessMode.TOOL
    return False

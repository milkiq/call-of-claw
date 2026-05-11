from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field


class WorldPatch(BaseModel):
    op: Literal["set", "append", "increment"]
    path: list[str]
    value: Any = None


class PatchResult(BaseModel):
    state: dict[str, Any]
    applied: list[WorldPatch] = Field(default_factory=list)


class ApplyWorldPatchInput(BaseModel):
    patches: list[WorldPatch]
    reason: str = ""


def apply_world_patches(state: dict[str, Any], patches: list[WorldPatch]) -> PatchResult:
    next_state = deepcopy(state)
    applied: list[WorldPatch] = []
    for patch in patches:
        if not patch.path:
            raise ValueError("Patch path cannot be empty")
        cursor: Any = next_state
        for key in patch.path[:-1]:
            if not isinstance(cursor, dict):
                raise ValueError(f"Patch path crosses non-object at {key}")
            cursor = cursor.setdefault(key, {})
        final = patch.path[-1]
        if patch.op == "set":
            cursor[final] = patch.value
        elif patch.op == "append":
            cursor.setdefault(final, [])
            if not isinstance(cursor[final], list):
                raise ValueError(f"Patch target {'.'.join(patch.path)} is not a list")
            cursor[final].append(patch.value)
        elif patch.op == "increment":
            cursor[final] = int(cursor.get(final, 0)) + int(patch.value)
        applied.append(patch)
    return PatchResult(state=next_state, applied=applied)

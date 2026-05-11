from __future__ import annotations

import random
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field


class RollDiceInput(BaseModel):
    expression: str = Field(description="Dice expression in NdM form, e.g. 2d6.")
    roll_id: str = Field(description="Stable roll identifier used for replay.")
    seed: str | None = Field(default=None, description="Optional deterministic seed.")


@dataclass(frozen=True)
class RollDiceResult:
    expression: str
    rolls: list[int]
    total: int
    roll_id: str


def parse_expression(expression: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)d(\d+)\s*", expression.lower())
    if not match:
        raise ValueError(f"Unsupported dice expression: {expression}")
    count = int(match.group(1))
    sides = int(match.group(2))
    if count < 1 or count > 100:
        raise ValueError("Dice count must be between 1 and 100")
    if sides < 2 or sides > 1000:
        raise ValueError("Dice sides must be between 2 and 1000")
    return count, sides


def roll_dice_once(expression: str, roll_id: str, seed: str | None = None) -> dict:
    count, sides = parse_expression(expression)
    rng = random.Random(f"{seed or ''}:{roll_id}:{expression}")
    rolls = [rng.randint(1, sides) for _ in range(count)]
    result = RollDiceResult(expression=expression, rolls=rolls, total=sum(rolls), roll_id=roll_id)
    return {
        "expression": result.expression,
        "rolls": result.rolls,
        "total": result.total,
        "roll_id": result.roll_id,
    }

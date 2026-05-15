from coc.tools.dice import parse_expression, roll_dice_once


def test_dice_roll_is_replayable() -> None:
    first = roll_dice_once("3d6", "turn-1-roll-1", seed="test")
    second = roll_dice_once("3d6", "turn-1-roll-1", seed="test")

    assert first == second
    assert len(first["rolls"]) == 3


def test_parse_expression() -> None:
    assert parse_expression("2d10") == (2, 10)

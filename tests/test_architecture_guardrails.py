from __future__ import annotations

from pathlib import Path

SMOKE_CONTENT_TERMS = [
    "Lasers",
    "Feelings",
    "lasers",
    "feelings",
    "激光",
    "感情",
    "姆姆",
    "水晶",
    "调频器",
    "维加",
    "浅蓝港",
    "铃兰",
    "达西",
    "海盗",
]


def test_core_source_does_not_embed_smoke_rules_or_scenario_terms() -> None:
    src_root = Path.cwd() / "src" / "trpg_agent"
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for term in SMOKE_CONTENT_TERMS:
            if term in text:
                offenders.append(f"{path.relative_to(Path.cwd())}: {term}")

    assert offenders == []


def test_online_graph_has_no_natural_language_keyword_routing_helpers() -> None:
    source = (Path.cwd() / "src" / "trpg_agent" / "graph" / "build_turn_graph.py").read_text(
        encoding="utf-8"
    )
    banned_fragments = [
        "_local_intent_kind",
        "_looks_risky",
        "_input_mentions_entry_target",
        "_should_assume_nearest_target",
        "assumed_target_only",
        "nearest_current_door_or_hatch",
        "risky_by_text",
    ]

    assert [fragment for fragment in banned_fragments if fragment in source] == []

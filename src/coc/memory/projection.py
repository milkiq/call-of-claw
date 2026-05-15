from __future__ import annotations

from typing import Any


def project_recent_summary(events: list[dict[str, Any]], limit: int = 5) -> list[str]:
    summaries: list[str] = []
    for event in events[-limit:]:
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            action = payload.get("playerAction") or payload.get("player_action")
            narration = payload.get("narration")
            if action or narration:
                summary = " | ".join(str(value) for value in [action, narration] if value)
                summaries.append(summary[:500])
            else:
                summaries.append(str(payload)[:500])
        else:
            summaries.append(str(payload)[:500])
    return summaries

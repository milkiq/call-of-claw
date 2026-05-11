from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trpg_agent.memory.store import SqliteStore

DEFAULT_IMPORTED_SESSION_ID = "imported-canon"


def stable_event_id(payload: dict[str, Any], line_number: int) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"canon-{line_number}-{digest}"


def import_canon_jsonl(
    store: SqliteStore,
    path: Path,
    *,
    session_id: str = DEFAULT_IMPORTED_SESSION_ID,
) -> int:
    if not path.exists():
        return 0
    imported = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        event_id = stable_event_id(payload, line_number)
        if store.insert_canon_event(
            event_id=event_id,
            session_id=session_id,
            event_type=str(payload.get("type", "imported")),
            payload=payload,
        ):
            imported += 1
    return imported

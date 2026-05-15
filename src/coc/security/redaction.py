from __future__ import annotations

import re
from typing import Any

SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "apiKey",
    "authorization",
    "access_token",
    "refresh_token",
    "secret",
    "password",
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{12,}"),
]


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if str(key) in SECRET_KEY_NAMES or str(key).lower() in SECRET_KEY_NAMES:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, str):
        redacted = value
        for pattern in SECRET_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value

from __future__ import annotations

from typing import Any, Protocol


class RulesetResolver(Protocol):
    id: str

    def resolve(self, ruleset: Any, request: Any) -> Any:
        """Resolve a ruleset-specific deterministic request."""

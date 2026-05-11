from __future__ import annotations

from trpg_agent.rules.base import RulesetResolver


class ResolverRegistry:
    def __init__(self) -> None:
        self._resolvers: dict[str, RulesetResolver] = {}

    def register(self, resolver: RulesetResolver) -> None:
        self._resolvers[resolver.id] = resolver

    def get(self, resolver_id: str) -> RulesetResolver:
        try:
            return self._resolvers[resolver_id]
        except KeyError as error:
            raise KeyError(f"Unknown ruleset resolver: {resolver_id}") from error

    def ids(self) -> list[str]:
        return sorted(self._resolvers)

from __future__ import annotations

from langchain_core.tools import StructuredTool

from trpg_agent.rules.compiled_resolver import RulesetResolverInput, run_ruleset_resolver
from trpg_agent.tools.content import (
    LoadContentSpanInput,
    SearchContentInput,
    load_content_span,
    search_content,
)
from trpg_agent.tools.patches import ApplyWorldPatchInput


def build_langchain_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=search_content,
            name="search_content",
            description="Search loaded content packages under a visibility access mode.",
            args_schema=SearchContentInput,
        ),
        StructuredTool.from_function(
            func=load_content_span,
            name="load_content_span",
            description="Load one content reference after enforcing visibility policy.",
            args_schema=LoadContentSpanInput,
        ),
        StructuredTool.from_function(
            func=run_ruleset_resolver,
            name="run_ruleset_resolver",
            description=(
                "Resolve a risky and uncertain action using the currently loaded compiled "
                "ruleset extension."
            ),
            args_schema=RulesetResolverInput,
        ),
        StructuredTool.from_function(
            func=lambda patches, reason="": {
                "world_patches": [
                    patch.model_dump() if hasattr(patch, "model_dump") else patch
                    for patch in patches
                ],
                "reason": reason,
            },
            name="apply_world_patch",
            description="Request deterministic world-state patches to be applied by the graph.",
            args_schema=ApplyWorldPatchInput,
        ),
    ]

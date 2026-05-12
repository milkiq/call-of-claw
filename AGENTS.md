# Project Development Requirements

This project is a generic TRPG GM agent runtime. Future development must preserve the boundary
between the generic GM engine and concrete TRPG rules/scenarios.

Use `docs/gm-agent-architecture-milestones.md` as the staged roadmap for reliability, long-session
operation, specialist advisor agents, memory, evaluation, and generic play quality.

## Architecture Red Lines

- Core prompts, graph nodes, tools, memory, storage, and eval runners must not embed concrete
  ruleset terms, scenario entities, NPC names, clue names, or smoke-test content.
- Online graph must not classify player intent, risk, target ambiguity, memory recall, passive
  choice, or unsupported authority claims by substring or keyword checks over player input. Those
  decisions must come from structured advisor outputs.
- Core graph must not classify a ruleset-specific action mode.
- Local no-model mode is only a structural debug fallback. It may parse explicit `NdM` dice syntax
  and slash commands, but it must not guess natural-language intent, risk, or target categories.
- Ruleset-specific adjudication belongs in compiled ruleset packages and resolver extensions.
- Scenario-specific GM requirements, secrets, scenes, clocks, NPCs, clues, and endings belong in
  compiled scenario packages.
- LLM output is advisory until validated by structured schemas and deterministic tools. Durable
  facts must come from tool results, world patches, or canon events.
- Risky and uncertain actions must go through the loaded resolver. The graph must not allow LLMs to
  bypass resolver calls with direct world patches or prose success.
- GM-only content may be used for GM reasoning, but player-facing output must not reveal it unless
  play has established access.
- LLM-facing instruction prompts and framework-generated internal advisor fields must be English.
  Player input, retrieved content, and content-package data may be multilingual; final
  player-facing text must match the player's language.
- Tests and content packages may contain smoke-test terms. `src/trpg_agent` must not.
- Offline turn eval must use fake advisor/fixture responses rather than relying on core graph
  keyword guessing.

## Required Checks

Run these before handing off changes:

```bash
.venv/bin/ruff check .
.venv/bin/pytest
.venv/bin/trpg content check
.venv/bin/trpg eval all --offline
```

Use live LLM evaluation when changing prompts, resolver flow, disclosure behavior, or narration:

```bash
.venv/bin/trpg eval live --limit 3
```

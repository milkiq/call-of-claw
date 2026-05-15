# Project Development Requirements

This project is a generic TRPG GM agent runtime. Future development must preserve the boundary
between the generic GM engine and concrete TRPG rules/scenarios.

Use `docs/gm-agent-architecture-milestones.md` as the staged roadmap. Milestones 0-13 are the
foundation baseline for reliability, long-session operation, specialist advisor agents, memory,
evaluation, and generic play quality. New development should prioritize Milestones 14-23: runtime
profiling, context budgeting, indexed retrieval, conditional advisors, Archivist/Narrator
separation, style state, prefix cache, rules plugins, structured scenario transitions, semi-open
long-play evaluation, and user-facing play profiles.

## Architecture Red Lines

- Core prompts, graph nodes, tools, memory, storage, and eval runners must not embed concrete
  ruleset terms, scenario entities, NPC names, clue names, or smoke-test content.
- Online graph must not classify player intent, risk, target ambiguity, memory recall, passive
  choice, or unsupported authority claims by substring or keyword checks over player input. Those
  decisions must come from structured advisor outputs.
- Core graph must not classify a ruleset-specific action mode.
- Local no-model mode is only a structural debug fallback. It may parse slash commands, but it must
  not guess natural-language intent, risk, target categories, or mechanically interpret `NdM` dice
  expressions inside ordinary player prose.
- Ruleset-specific adjudication belongs in compiled ruleset packages and resolver extensions.
- Scenario-specific GM requirements, secrets, scenes, clocks, NPCs, clues, and endings belong in
  compiled scenario packages.
- Scenario fast paths may expose only package-owned public fields such as compiled
  `visible_surfaces`; they must not derive hidden reveals, transitions, consequences, clocks, or
  clues from player-input keywords.
- GM style, table tone, and NPC voice are style state or package data. They must not authorize
  durable facts, world changes, hidden reveals, rule effects, or player decisions.
- LLM output is advisory until validated by structured schemas and deterministic tools. Durable
  facts must come from tool results, world patches, or canon events.
- Risky and uncertain actions must go through the loaded resolver. The graph must not allow LLMs to
  bypass resolver calls with direct world patches or prose success.
- Rule dice are resolver-owned. LLMs may request rules resolution and suggest loaded rule ids, but
  they must not choose final dice expressions or dice-pool sizes. Free manual rolls must use the
  out-of-band `/roll <NdM>` command and are not authoritative rules results.
- GM-only content may be used for GM reasoning, but player-facing output must not reveal it unless
  play has established access.
- LLM context loading must go through the programmatic context firewall. Do not pass raw
  `retrieved_spans`, full `package_profiles`, full `tool_results`, or full `turn_plan` directly to
  player-facing LLM nodes; use role-specific context packets and keep reason codes in trace
  metadata rather than prompt text.
- LLM-facing instruction prompts and framework-generated internal advisor fields must be English.
  Player input, retrieved content, and content-package data may be multilingual; final
  player-facing text must match the player's language.
- Tests and content packages may contain smoke-test terms. `src/trpg_agent` must not.
- Offline turn eval must use fake advisor/fixture responses rather than relying on core graph
  keyword guessing.
- Do not solve latency by weakening resolver, visibility, critic, or durable-state boundaries.
  Optimize through profiling, context budgeting, indexed retrieval, conditional advisors, and
  provider-friendly stable prefixes first.
- Conditional advisor skips must be based on structured advisor outputs plus validated graph state,
  never raw player-input keyword checks. Every skipped advisor must leave a machine-readable skip
  reason in trace/state, and resolver results, scenario patches, hidden-content exposure risk, and
  pending rules opportunities must stay on the full safety path.
- Before changing context budgets, retrieval, advisor scheduling, or provider settings, inspect
  `trpg eval observation-report` so optimization work is tied to measured slow nodes, prompt size,
  fallback counts, and timeout counts.
- User-facing play configuration should go through `--profile fast|balanced|theatrical` and
  `--local`; do not re-expose low-level experiment flags on `trpg play` unless they are promoted to
  documented profile behavior.
- Eval and development smoke tests must not leave playable test sessions in the default local
  database unless a human explicitly needs to inspect that session. Use `--keep-session` only for
  deliberate retention, and otherwise rely on eval command cleanup or run
  `.venv/bin/trpg session cleanup-tests --yes` after tests.

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

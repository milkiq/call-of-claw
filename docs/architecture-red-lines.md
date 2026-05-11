# GM Agent Architecture Red Lines

## Summary

The runtime is a generic TRPG GM engine. Concrete rules and scenarios are data and extensions, not
core agent behavior. The core can decide when something needs rules, tools, memory, or clarification;
it cannot decide using a specific game's categories unless those categories came from a loaded
compiled package.

## Red Lines

- No concrete ruleset terms, scenario entities, NPCs, clues, locations, or smoke-test keywords in
  `src/trpg_agent`.
- No natural-language keyword routing in the online graph. Intent, risk, target ambiguity, memory
  recall, passive action, and boundary claims must come from structured advisor outputs, not
  substring checks over player input.
- No ruleset-specific action classifier in the graph. The resolver decides rule-specific approach,
  dice, target, success count, and band.
- No scenario-specific scene logic in the graph. Scene titles, secrets, clocks, transitions, and
  endings come from compiled scenario packages.
- No durable state changes from prose. Durable facts must be tool results, world patches, session
  state, or append-only canon events.
- No LLM bypass of deterministic adjudication. Risky actions must call the loaded resolver even if
  the LLM tries to answer directly or apply a patch.
- No hidden information leakage. GM-only spans can guide reasoning, but player-facing narration must
  expose only what play has established.
- No framework-level defaults that silently encode one ruleset. If a resolver needs target numbers,
  exact-target behavior, or approach keywords, those values must be in compiled ruleset content.
- No scenario director patch becomes durable until validated against the loaded scenario package's
  generic patch authority, scene set, and current state bounds.
- Critic repair may alter only player-facing narration text. It must not mutate tool results,
  resolver results, world patches, memory writes, or canon.
- Structural parsers are allowed only for machine-like syntax and commands, such as slash commands,
  explicit `NdM` dice expressions, and output-policy guardrails.
- Offline turn eval must be fake-advisor or fixture driven. It must not depend on the core graph
  guessing natural-language intent, risk, or target keywords.

## Current Self-Check Findings

- Fixed: a graph-level action approach helper had encoded the smoke ruleset's approaches. Approach
  selection is now owned by compiled ruleset data and resolver runtime.
- Fixed: smoke-test forbidden prompt terms and live eval inputs were embedded in `src`. They now
  live in test/eval data.
- Fixed: compiled ruleset schema no longer constrains target numbers to the smoke ruleset range or
  defaults to one dice shape.
- Added: `tests/test_architecture_guardrails.py` fails if smoke rules or scenario terms enter
  `src/trpg_agent`.
- Added: `tests/test_architecture_guardrails.py` fails if online graph keyword-routing helper names
  or soft target-assumption fields return to `build_turn_graph.py`.
- Fixed: online routing, micro-gate aggregation, and resolver enforcement no longer use local
  natural-language keyword lists for player intent, risk, target, or boundary decisions.
- Fixed: local turn eval uses fake LLM fixtures synthesized from eval expectations, so regression
  cases test advisor contracts rather than local keyword guesses.
- Added: scenario director, critic, and memory curator are separate advisor nodes with structured
  trace state and idempotent persistence boundaries.

## Current GM Loop

1. CLI receives a player input and session id.
2. Graph loads session metadata, resolved package ids, manifest-only package profiles, session
   state, recent canon, and player/GM memory views.
3. Graph retrieves relevant content spans under GM visibility.
4. Intent and rules advisors produce routing/rules advice. Local no-model mode is a structural
   debug fallback and does not infer natural-language intent, risk, or target ambiguity.
5. Local or LLM adjudication produces a generic `TurnPlan`.
6. Graph enforces red lines:
   - risky and uncertain action must request `run_ruleset_resolver`;
   - LLM-requested direct world patches are dropped for risky actions;
   - explicit `NdM` dice syntax can be passed to the resolver as a structural requested roll.
7. Deterministic tools run:
   - content search/load;
   - ruleset resolver;
   - world patch request;
   - canon writes.
8. Scenario director may propose scene, reveal, pressure, consequence, or ending patches; validation
   rejects patches outside package authority and normalizes common structured LLM patch variants
   before applying them.
9. World patches are applied to session state and persisted.
10. LLM narration receives the turn plan, retrieved spans, current world projection, validated
    scenario state, tool results, and player-visible memory context, then returns player-facing text.
11. Critic guardrail checks and may perform one narration-only repair.
12. Memory curator proposes canon/memory writes; graph validates and persists them with stable ids.
13. Turn output, trace, critic report, tool results, world projection, and canon summary are
    redacted for common secret shapes and persisted.
14. Evaluation can replay fixture-driven fake-advisor cases, run long-play durability checks, run
    release gates, or run live LLM judge against transcript, trace, and evidence.

## Known Remaining Risks

- The current resolver implementation covers two smoke resolver families. Additional rulesets should
  add resolver modules or resolver configs rather than widening core graph logic.
- Scenario transitions are package-validated and advisor-guided, but future milestones should add
  richer deterministic transition resolvers for complex scenarios.
- Critic repair depends on validated reveal/world patches being present before narration review.
  Scenario patch rejection therefore needs live-eval coverage, because an overly strict validator can
  make grounded narration look unsupported.
- The current long-play runner is deterministic and scripted. It now checks repeated output,
  unresolved hook quality, and 50-turn memory QA, but future work still needs semi-open
  player-simulator runs and observer/post-session judge agents.

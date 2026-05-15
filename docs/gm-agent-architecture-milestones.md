# GM Agent Architecture Milestones

This document turns the architecture direction into executable milestones. The target is a generic
single-player TRPG GM agent that is reliable, can run for long sessions, and can flexibly host
different rulesets and scenarios without coupling core runtime code to any concrete game.

## Operating Principles

- The core runtime is generic. Rulesets, scenarios, clocks, NPCs, endings, and special GM procedures
  are loaded content or extensions.
- LLMs may judge, suggest, summarize, and narrate. They do not directly create durable facts.
- Durable facts must be produced through deterministic tools, validated world patches, session
  state, or append-only canon events.
- The main graph is the authority. Specialist LLMs are advisors inside controlled graph nodes.
- Every milestone must improve both play quality and observability. A feature is not complete until
  it has trace evidence and regression coverage.
- Local deterministic tests are required for all changes. Live LLM evaluation is required for prompt,
  routing, memory, disclosure, and narration changes.

## Target Runtime Shape

```text
player input
-> load durable session context
-> retrieve short-term state, long-term memory, canon, and content spans
-> intent arbiter advisor
-> authority and boundary gate
-> rules adjudicator advisor, when needed
-> deterministic resolver and tools
-> scenario director advisor
-> validated world patch application
-> memory curator advisor
-> narrator advisor
-> critic and guardrail advisor
-> persist turn, trace, canon, memory, and eval signals
```

The orchestrator remains a deterministic LangGraph workflow. Each advisor receives a deliberately
scoped context package and returns structured output.

## Milestone 0: Architectural Baseline

Goal: Keep the current project from regressing while later milestones change the runtime.

Current status: mostly implemented.

Implementation work:

- Maintain architecture red lines in `AGENTS.md` and `docs/architecture-red-lines.md`.
- Keep concrete smoke-test rules and scenario terms out of `src/trpg_agent`.
- Keep current CLI play, deterministic eval, content check, and architecture guardrail tests working.
- Treat the current keyword-based intent and risk detection as bootstrap fallback only.

Verification:

```bash
.venv/bin/ruff check .
.venv/bin/pytest
.venv/bin/trpg content check
.venv/bin/trpg eval all --offline
```

Acceptance:

- Architecture guardrail tests fail if concrete smoke content enters core source.
- Risky actions still cannot bypass `run_ruleset_resolver`.
- Existing local play and eval flows remain usable.

## Milestone 1: Durable Graph Runtime

Goal: Make long-running sessions resumable and replay-safe.

Current status: implemented for the current runtime. The turn graph can be compiled with a SQLite
checkpointer, CLI/live eval use that durable runtime, checkpoint thread ids are scoped per turn,
persisted turns can be replayed by `turn_id`, dice/advisor/world patch effects are idempotent, trace
metadata records graph/prompt/checkpoint/content/resolver versions, and offline eval includes a
durable replay gate.

Implementation work:

- Add a LangGraph checkpointer for turn execution.
- Standardize `thread_id`, `session_id`, `turn_id`, and per-advisor run ids.
- Move non-deterministic and side-effecting operations behind idempotent tool/task boundaries:
  dice rolls, LLM calls that affect routing, DB writes, canon writes, memory writes, eval writes.
- Add idempotency keys for all persistent writes.
- Record graph version, prompt versions, model config, content package versions, and resolver version
  in every trace and persisted turn.
- Add resume tests for interrupted turn execution.

Verification:

- Unit test: repeated invocation with the same `turn_id` does not duplicate dice, canon, memory, or
  session patches.
- Unit test: replay from a checkpoint returns the same tool results.
- Integration test: simulate interruption after tool execution and resume.
- CLI eval includes deterministic replay checks.

Acceptance:

- A 30-turn scripted session can be stopped and resumed without duplicated state changes.
- Replaying a persisted turn preserves dice results and world patches.
- Failed LLM calls do not leave half-applied durable state.

## Milestone 2: Specialist Advisor Contracts

Goal: Replace the single overloaded GM decision step with explicit advisor schemas.

Current status: implemented for the current runtime. Advisor schemas, generic prompts, prompt
versions, a role-specific runner, prompt genericity tests, per-role model selection, advisor output
caching, and safe fallback behavior exist. Intent and rules advisors are wired into the LLM turn
graph; later milestones will wire scenario, memory, and critic advisors.

Implementation work:

- Add structured schemas:
  - `IntentRoutingDecision`
  - `AuthorityGateResult`
  - `RulesAdjudicationAdvice`
  - `ScenarioDirectorDecision`
  - `MemoryCurationDecision`
  - `NarrationPlan`
  - `CriticReport`
- Add generic prompts for each advisor. Prompts must not mention concrete rulesets or scenarios.
- Add an advisor runner abstraction that supports:
  - role-specific prompt version;
  - role-specific model selection;
  - structured output parsing and repair;
  - trace metadata;
  - fallback behavior.
- Keep local deterministic fallback nodes for offline tests.

Verification:

- Schema tests for every advisor.
- Prompt guardrail test: no smoke-test content appears in advisor prompts.
- Fake model graph tests proving each advisor output changes only its allowed part of state.

Acceptance:

- The graph can run with independent fake models per advisor.
- The graph can run with one shared real model for all advisors.
- Advisor failures degrade to clarification or safe local fallback rather than corrupting state.

## Milestone 3: Intent Arbiter and Routing

Goal: Use a dedicated LLM session to judge what the player is doing and how the graph should flow.

Current status: implemented for the current runtime. The LLM turn graph runs `intent_arbiter`
before GM adjudication, persists the routing decision into trace state, passes it into the turn
planner, and uses it as the primary signal for resolver enforcement. Keyword risk detection remains
only for the local/no-routing fallback path, and advisor failure degrades to safe generic fallback
routing.

Implementation work:

- Add `intent_arbiter` node before turn planning.
- Provide the arbiter with player input, current public state, relevant rules/scenario summaries,
  recent canon, and memory hits.
- Arbiter returns generic routing signals:
  - answer directly;
  - ask clarification;
  - free fictional action;
  - risky or uncertain action;
  - rules question;
  - memory recall;
  - boundary claim;
  - scenario director needed;
  - rules resolver needed.
- Downgrade keyword risk detection to safety fallback only.
- Add graph guards: if arbiter says rules resolver is needed, later narration cannot skip it.

Verification:

- Dataset covering ambiguous player inputs, questions, passive choices, risky actions, impossible
  claims, and memory recalls.
- LLM judge evaluates whether routing matches normal TRPG GM practice.
- Regression test proves no ruleset-specific action mode is hardcoded in the graph.

Acceptance:

- On live eval, routing score averages at least 4/5 across the routing dataset.
- No risky action reaches narration without resolver or explicit clarification.
- False positives from keyword-only routing are no longer the primary path.

## Milestone 4: Rules Adjudicator and Resolver Extension Boundary

Goal: Let rules intelligence come from loaded ruleset content and resolver extensions, not core GM
logic.

Current status: implemented for the current runtime. The LLM turn graph runs `rules_adjudicator`
only when the intent arbiter routes a turn to rules resolution. Its output is stored as advisory
trace state and can provide resolver parameters such as loaded approach ids, requested rolls, and
risk labels. Resolver protected arguments are sanitized by the graph. Dice, targets, success counts,
bands, and world patches remain owned by deterministic resolver extensions. The resolver registry
has two families, `threshold_d6` and `sum_target`, and cross-ruleset tests prove the core graph does
not change when a second ruleset is added.

Implementation work:

- Add `rules_adjudicator` node that runs only when routing requires rules.
- Provide the rules advisor with loaded ruleset content, relevant scenario context, character state,
  and player action.
- The advisor may suggest:
  - which loaded rule procedure applies;
  - which loaded approach/stat/skill id applies;
  - stakes and fictional risk;
  - whether clarification is needed.
- Deterministic resolver remains responsible for dice, targets, success counts, result bands, and
  resolver-owned world patches.
- Introduce resolver family registry so new rulesets add resolver modules/configs without widening
  core graph logic.
- Add tests with at least two different resolver families before declaring the boundary mature.

Verification:

- Unit tests for resolver registry and extension loading.
- Cross-ruleset tests proving core graph source contains no concrete rule labels.
- Live eval cases where the same player action is adjudicated differently under different loaded
  rulesets.

Acceptance:

- Adding a second non-threshold ruleset does not require changes to core graph routing.
- Rules advisor cannot directly write world state.
- Resolver output is deterministic, persisted, replayable, and cited in narration.

## Milestone 5: Scenario Director and Transition Intelligence

Goal: Move scene transitions, reveals, clocks, NPC reactions, and ending pressure into a dedicated
scenario intelligence loop.

Current status: implemented for the current runtime. The LLM graph now runs a `scenario_director`
advisor after deterministic tools and before world patch application. Its patch proposals are
validated against the loaded scenario package allowlist, current scene set, clock bounds, duplicate
patches, and generic visibility-safe patch shapes before becoming authoritative world patches. The
local graph includes a deterministic no-change scenario director step for trace parity. Scenario
packages now carry transition affordances, patch allowlists, and GM constraints in compiled content.
The validator accepts common structured LLM patch variants such as JSON Pointer paths,
`operation=add`, and reveal payloads that place player-facing fact text in `value.content`, then
normalizes them into durable world patches. Live eval found and verified this boundary after a false
reject on reveal facts.

Implementation work:

- Add `scenario_director` node after deterministic tools and before final patch application.
- Provide current scene, available scenario spans, GM-only context, public state, canon, memory hits,
  resolver results, and player input.
- Director returns structured decisions:
  - no change;
  - reveal public information;
  - propose scene transition;
  - advance clock or pressure;
  - introduce consequence;
  - request clarification;
  - propose ending condition.
- Compile scenarios into transition affordances and patch constraints.
- Validate director patches against scenario package ids, path allowlists, visibility policy, and
  current state.
- Reject or downgrade patches that leak hidden facts, skip required rules resolution, or exceed
  scenario authority.

Verification:

- Scenario transition dataset with success, failure, partial success, passive waiting, wrong
  direction, and investigation cases.
- Deterministic patch validation tests.
- LLM judge checks whether transitions feel like normal GM judgment rather than rigid branching.

Acceptance:

- A 20-turn smoke scenario can transition scenes without hardcoded scene logic.
- Hidden content is not exposed until scenario director proposes a validated reveal.
- Scene state remains consistent after repeated transitions and resumed turns.

## Milestone 6: Long-Term Memory Architecture

Goal: Support long sessions and future campaigns without overwhelming context or losing continuity.

Current status: implemented as a SQLite-backed baseline. The graph distinguishes canon, world state,
turn traces, player-visible memory, GM-only memory, character context, player preferences,
procedural notes, unresolved threads, and episodic summaries. The LLM graph runs `memory_curator`
after critic approval; the local graph runs a small deterministic fallback. Curated memory and canon
drafts are written through stable turn-scoped ids, and retrieval can filter GM-only memory out of
player-facing contexts. A simple episodic summary policy writes summary memories every ten durable
turns.

Implementation work:

- Separate memory types:
  - turn trace;
  - canon event;
  - world state projection;
  - character state;
  - player preference;
  - unresolved thread;
  - episodic summary;
  - procedural note for future GM behavior.
- Add `memory_curator` node after narration or critic approval.
- Curator proposes canon and memory writes; graph validates and persists them.
- Add summarization policy for long sessions:
  - recent turns stay detailed;
  - older turns become episodic summaries;
  - unresolved hooks stay separately retrievable;
  - durable facts remain structured canon.
- Add retrieval policy that distinguishes player-visible memory from GM-only memory.
- Prepare vector-store abstraction while keeping SQLite FTS as the local baseline.

Verification:

- 50-turn scripted replay with recall questions at different distances.
- Tests for no duplicate memory writes on replay.
- Tests for contradictions: new memory cannot silently overwrite established canon.
- LLM judge scores continuity and memory usefulness.

Acceptance:

- The agent can answer "what happened earlier" after 50 turns using canon/memory, not prompt luck.
- Old summaries do not overwrite concrete canon facts.
- GM-only memory is never narrated as public fact without reveal.

## Milestone 7: Critic, Guardrails, and Repair Loop

Goal: Add a separate quality and safety check before player-facing output is finalized.

Current status: implemented for the LLM graph with a deterministic local fallback. The
`critic_guardrail` advisor runs after narration and before memory persistence. It receives final
text, turn plan, tools, applied state, memory, retrieved visibility metadata, and scenario director
state. A bounded repair pass may replace only `final_output`/`narration_plan.final_text`; tool
results and world state remain unchanged. Critic reports are persisted idempotently for durable
sessions and included in turn traces. Blocking findings for resolver bypass, hidden leakage,
unsupported facts, canon contradiction, player agency, and missing clarification fail closed; player
agency and clarification repairs use deterministic in-fiction fallbacks. Live eval caught a critic
overcorrection case caused by rejected reveal patches; after patch normalization, revealed facts are
available to the critic and the corrected live eval passed.

Implementation work:

- Add `critic_guardrail` node after narration.
- Critic receives final text, player input, turn plan, tool results, visibility metadata, and applied
  patches.
- Critic checks:
  - hidden information leakage;
  - unsupported durable facts in prose;
  - resolver bypass;
  - contradiction with canon;
  - player agency violation;
  - bad pacing or unusable narration;
  - missing clarification.
- Add one bounded repair pass for narration only. Repair cannot alter tool results or applied state.
- Persist critic findings for eval.

Verification:

- Adversarial eval cases for hidden clue leakage, impossible claims, skipped dice, contradiction, and
  narration choosing the player's action.
- Unit test: critic repair cannot change tool results or world patches.
- Live eval: critic improves or blocks low-quality output.

Acceptance:

- Known leakage and resolver-bypass prompts fail closed.
- Repair pass improves final output without changing durable state.
- Critic reports are visible in traces and eval reports.

## Milestone 8: Skill and Content Package Model

Goal: Clarify what belongs in skills versus rulesets and scenarios.

Current status: implemented as a stricter content-package baseline. Manifests now support
`capability_skill` packages, capabilities, extension prompt/tool declarations, and progressive
disclosure policy metadata while preserving existing `agent_skill`, ruleset, scenario, extension,
and evaluator packages. The registry resolves dependencies/extensions into active package ids,
provides manifest-only package profiles to advisors, validates extension prompt references, and
keeps retrieved span text separate from manifest profiles. A reusable `clue_hygiene_skill`
capability package is attached to the smoke scenario without changing core prompts or graph logic.

Implementation work:

- Define package categories:
  - ruleset package: rules text, procedures, resolver config, rule-specific advisor instructions;
  - scenario package: scenes, secrets, clocks, NPCs, transitions, endings, scenario-specific GM
    requirements;
  - capability skill: reusable agent capability, such as mystery pacing, horror tone, tactical combat
    presentation, clue hygiene, or safety tools;
  - evaluator skill: quality rubrics and judge instructions.
- Do not bind generic GM core to any fixed ruleset or scenario.
- Allow ruleset/scenario packages to declare extension prompts and tools under strict schemas.
- Build package discovery and progressive disclosure:
  - advisor sees package manifest first;
  - loads relevant spans on demand;
  - cannot read everything by default unless policy permits.

Verification:

- Content package validation tests.
- Progressive disclosure tests: advisor must cite loaded spans for package-specific decisions.
- Cross-package tests where one capability skill works with multiple rulesets/scenarios.

Acceptance:

- A new scenario can be added without editing core prompts or graph logic.
- A capability skill can be reused across different scenarios.
- A ruleset-specific instruction is present only in that ruleset package or compiled extension.

## Milestone 9: Automated Quality System

Goal: Make quality improvement continuous instead of anecdotal.

Current status: implemented as the current automated quality baseline. Offline eval includes
Milestone 5-9 trace coverage, content validation, replay, and package checks. LLM judge scorecards
now include generic architecture compliance. Eval run metadata records graph version and content
package versions, and `trpg eval quality-report` summarizes pass/fail, average scorecards, findings,
and score movement across selected runs. Roadmap derivation continues to consume persisted eval
findings. The current required check baseline passes `ruff`, `pytest`, `trpg content check`, and
offline eval; the latest bounded live eval passed 2/2 after fixing the scenario patch normalization
boundary, and the latest quality report passed 30/30 selected cases.

Implementation work:

- Expand datasets:
  - deterministic regression;
  - routing cases;
  - rules adjudication cases;
  - scenario transition cases;
  - memory recall cases;
  - leakage and authority boundary cases;
  - long-play cases;
  - cross-ruleset genericity cases.
- Add LLM-as-judge scorecards for:
  - rules correctness;
  - fictional authority;
  - continuity;
  - player agency;
  - pacing;
  - progressive disclosure;
  - memory behavior;
  - narration quality;
  - generic architecture compliance.
- Add paired comparison against previous prompt/runtime versions.
- Store eval runs with graph version, prompt versions, model config, package ids, and trace ids.
- Add `trpg eval quality-report` that summarizes trends, failures, and next roadmap items.

Verification:

- Offline eval must run without network or live model where possible.
- Live eval can run with bounded cases and clear cost controls.
- Failing eval cases automatically produce actionable failure categories.

Acceptance:

- Every architecture milestone adds or updates eval cases.
- Quality reports show both pass/fail and score movement.
- Roadmap can be derived from failing eval clusters.

## Milestone 10: Long-Play Reliability

Goal: Prove that the agent can host a long single-player session without state drift.

Current status: implemented as a deterministic long-play reliability baseline. `trpg eval long-play`
drives the durable graph through a scripted multi-turn session, replays a persisted turn, and records
turn count, canon count, memory count, world patch applications, critic reports, resolver bypasses,
critical critic findings, player-agency markers, duplicate ids, and trace-node coverage. The
release-gate command includes this long-play check, with the default set to 50 turns. The long-play
gate now also scores repeated normalized output, unresolved hook presence/quality, and memory QA
accuracy after 50 turns. Memory QA checks whether early canon remains available, player preference
memory can be recalled through the public memory path, episodic summaries are written at 10-turn
intervals, and recall-style turns surface persisted facts.

Implementation work:

- Build automated agent playtest runner:
  - player simulator agent;
  - GM subject agent;
  - observer critic agent;
  - post-session judge.
- Add scripted and semi-open playtest modes.
- Add long-session health metrics:
  - turn count;
  - unresolved hooks;
  - repeated content;
  - contradiction count;
  - memory recall success;
  - hidden leak count;
  - resolver bypass count;
  - average judge score by dimension.
- Add context compaction and recap checkpoints.

Verification:

- 50-turn smoke long play with repeated-content, unresolved-hook, and memory-QA gates.
- 100-turn durability and memory stress test.
- Resume long play from midpoint checkpoint.
- Compare transcript quality against previous version.

Acceptance:

- 50-turn session average judge score is at least 4/5 on continuity, agency, memory, and pacing.
- Zero critical hidden leaks or resolver bypasses.
- 100-turn technical stress test completes without duplicated durable state.

## Milestone 11: Genericity Across Games

Goal: Demonstrate that the runtime is a generic TRPG GM agent, not a test-case host.

Current status: implemented as a package-level genericity baseline. The content registry now carries
three ruleset packages using three resolver families: threshold dice, sum-target dice, and
percentile-under d100. It also carries three scenario packages with different structures: a pressure
clock scenario, an investigation scenario, and a survival pressure scenario. Offline eval includes
cross-product turn cases that run non-default rulesets and scenarios through the same core graph.

Implementation work:

- Add at least three materially different ruleset packages:
  - one light threshold or target-number ruleset;
  - one percentile or d100-style ruleset;
  - one fiction-first or move-based ruleset.
- Add at least three scenario packages with different structures:
  - mystery/investigation;
  - survival or pressure-clock scenario;
  - social or faction scenario.
- Add cross-product eval cases using multiple rulesets and scenarios.
- Confirm each game-specific behavior lives in packages/extensions.

Verification:

- Architecture source scan for concrete package terms.
- Cross-ruleset live eval proving different rule procedures are chosen through package context.
- Cross-scenario eval proving scene transitions come from scenario context.

Acceptance:

- Core graph does not change when adding the third ruleset and third scenario.
- Generic GM prompt remains unchanged or changes only for game-agnostic GM obligations.
- Quality scores remain acceptable across all test packages.

## Milestone 12: Playable Single-Player MVP

Goal: Reach a practical solo play experience that is reliable enough for regular use.

Current status: implemented as a local CLI MVP. `trpg session start` initializes scenario state,
`trpg play --input` runs one deterministic or LLM-backed turn, and `trpg play` without `--input`
starts a reusable interactive loop with `/recap`, `/session`, `/quit`, generated session ids, resume
commands, and ruleset-provided character creation before the first turn. `trpg session recap`
summarizes recent turns and public state, `trpg session inspect` exposes public state by default and
GM trace data only with `--gm-trace`, `trpg session export` writes transcript/state JSON, and
`trpg session quality-report` summarizes persisted session metrics. Tests exercise both the
start-play-recap-inspect-export-quality path and the interactive create/resume character path.

Implementation work:

- Add a CLI or minimal local interface for starting, resuming, inspecting, and exporting sessions.
- Support session commands:
  - start;
  - play;
  - recap;
  - inspect public state;
  - inspect GM trace for debugging;
  - export transcript;
  - run session quality report.
- Add configurable model profiles for fast, balanced, and high-quality play.
- Add failure handling for model timeout, malformed JSON, missing package, invalid patch, and resolver
  errors.

Verification:

- End-to-end manual playtest checklist.
- Automated 30-turn and 50-turn playtests.
- Live quality report after playtest.
- Exported transcript review by LLM judge.

Acceptance:

- A user can start a fresh session, play at least 30 turns, pause, resume, ask recap questions, and
  export the transcript.
- The agent handles ambiguous inputs by asking useful questions rather than guessing destructively.
- The game does not depend on any concrete smoke-test content unless that content package is loaded.

## Milestone 13: Production Hardening

Goal: Make the system maintainable for ongoing development by Codex or other agents.

Current status: implemented as a maintainability baseline. SQLite migrations are recorded in a
`schema_migrations` table, persisted traces are redacted for common API-key and bearer-token shapes,
advisor traces include latency and estimated prompt/response character counts, content validation
checks semver package versions and compiled package compatibility, and `trpg eval release-gates`
runs content validation, offline eval, durable replay, and long-play reliability. Prompt and
developer workflow docs were added for future agents.

Implementation work:

- Add migration tests for persistent schema changes.
- Add prompt version changelog.
- Add package versioning and compatibility checks.
- Add trace redaction policy for secrets and API keys.
- Add cost and latency reporting per advisor.
- Add model fallback policy.
- Add developer documentation for adding a ruleset, scenario, resolver, advisor, evaluator, and
  capability skill.

Verification:

- Full test and eval suite passes from a clean checkout.
- Schema migration from previous DB version works.
- Cost-bounded live eval works.
- Developer docs are validated by adding a tiny new package in tests.

Acceptance:

- A new development agent can follow docs to add a small ruleset or scenario without editing core
  GM logic.
- Runtime failures are diagnosable from traces and persisted eval reports.
- Prompt, graph, package, and resolver changes are all attributable in quality reports.

## Post-Research Roadmap: Speed, Style, and Scalable Genericity

Deep research in `docs/deep-research-report.md` reframes Milestones 0-13 as the foundation
baseline. They establish replay safety, advisor boundaries, package-owned rules and scenarios,
memory, critic guardrails, genericity checks, and a playable CLI. The next development phase should
not add heavier always-on multi-agent coordination. It should tighten the runtime path around:

- progressive disclosure and explicit context budgets;
- an Archivist/Narrator split, where the Archivist prepares authorized facts and the Narrator only
  writes player-facing text;
- conditional lightweight advisors instead of every advisor running every turn;
- indexed retrieval and provider-friendly stable prompt prefixes;
- a style state that improves GM voice without becoming canon or world state;
- plugin-oriented rules and scenario transitions that keep game-specific behavior out of core code.

Recent test evidence should guide this sequence: required offline checks pass, bounded live eval can
pass, and a 2-turn online A/B showed `clarification_rate=0` for both legacy and compact contracts.
However, true online runs still showed high latency and provider read timeouts; compact contracts
reduced response size but did not reliably reduce wall-clock latency. Therefore the next milestones
prioritize profiling, context shape, retrieval, and conditional routing before making compact mode
or micro-gates the default.

## Milestone 14: Runtime Profiling and Latency Budget

Goal: Explain where online play spends time before changing the runtime path.

Current status: baseline implemented. Advisor diagnostics record elapsed time and estimated prompt
and response size. `invoke_turn_graph()` and `stream_turn_graph()` now attach a `runtime_profile`
with per-node wall-clock timings, the selected latency budget profile, slowest nodes, and
trace-derived fallback/timeout counts, and coarse latency categories. Online play reports include
runtime metadata and flag timeout or fallback markers as infrastructure findings. `trpg eval
observation-report` can summarize historical report files and stored advisor runs, including older
reports that predate `runtime_profile`. Remaining work is finer provider-level attribution where a
backend exposes cache/read/write timings.

Implementation work:

- Add wall-clock timing for every graph node and every parallel branch.
  - Done for graph nodes and context retrieval branches.
- Record LLM elapsed time, timeout/fallback status, repair attempts, cache hits, and prompt/response
  character estimates in one node-latency report.
  - Done through advisor diagnostics, prompt breakdown metadata, observation reports, and graph-level
    runtime reports.
- Split latency categories into retrieval I/O, graph orchestration, provider wait, schema repair,
  deterministic tools, critic, and memory curation.
  - Done as coarse runtime node categories; schema repair remains advisor-attempt metadata.
- Add explicit runtime budgets for `fast`, `balanced`, and `theatrical` profiles.
  - Done as reporting budgets; CLI play-profile switching is Milestone 23.
- Ensure online reports surface timeout/fallback counts even when the playtest technically passes.
  - Done through runtime metadata and infrastructure findings.

Verification:

- Unit tests cover node timing metadata, timeout/fallback marker counting, runtime summary
  serialization, and online runtime findings.
- 2-5 turn online smoke with node-latency table and timeout/fallback counts remains a live
  verification step when model access is available.
- Regression test proves fallback and timeout events do not disappear from online quality reports.

Acceptance:

- A developer can identify the three slowest nodes in every online playtest report.
- Online playtest reports distinguish provider latency from graph overhead.
- Passing online smoke cannot hide advisor timeout/fallback risk.

## Milestone 15: Context Budgeter and Retrieval Index

Goal: Stop sending large undifferentiated context blobs and stop scanning content files every turn.

Current status: implemented for the fast profile and still observable in shadow mode for the
balanced/theatrical profiles. Advisor input now passes through role-specific `ContextPacket`
construction, memory uses SQLite FTS, and content retrieval has an indexed SQLite backend with scan
fallback. The budgeter records stable prefix, scene, rules, memory, canon, retrieved public/GM,
tool-result, and style buckets; fast-profile LLM prompts receive the enforced packet instead of the
full graph state.

Implementation work:

- Add a `ContextBudgeter` that builds explicit buckets for stable prefix, local scene, local rules,
  player-visible memory, recent canon, retrieved spans, and style state.
  - Done in shadow mode.
- Give each bucket a default token/character budget and a clear clipping priority.
  - Done with enforced role-specific packets for the fast profile; balanced/theatrical retain shadow
    mode until quality gates promote enforcement.
- Add a local content index backed by SQLite FTS/BM25 while preserving the current
  `search_registry_text()` interface.
  - Done as an indexed runtime path with scan fallback.
- Store package id, reference id, title, tags, visibility, text, and package version in the index.
  - Done.
- Rebuild the index when package version, manifest, or reference mtime changes.
  - Done for manifest/reference mtime and package version.
- Mark retrieved spans with bucket, visibility, citation id, and whether they are mandatory or
  discardable.
  - Done for runtime retrieved spans and context-packet copies; role filtering records machine-readable
    reason codes in trace metadata.

Verification:

- Content retrieval tests cover visibility, GM-only exclusion for player contexts, CJK queries,
  package filtering, indexed retrieval diagnostics, and scan fallback surface.
- Prompt-size and context-firewall tests verify role packets, hidden-content filtering, and tool
  result summarization.
- Offline eval and content check pass with indexed retrieval enabled.

Acceptance:

- Adding content packages does not force each turn to read every reference file.
- Narrator prompts are smaller without losing required visible facts or rule citations.
- Hidden content remains unavailable to player-facing context unless revealed by validated play.

## Milestone 16: Conditional Advisor Runtime

Goal: Run only the advisors that a turn actually needs.

Current status: first safe slice implemented for the `fast` profile. Conditional advisor mode now
keeps the normal router, but can build a direct low-risk turn plan from structured routing output,
select package-authorized visible scene surfaces for low-risk observation turns, skip scenario
direction when runtime/routing says no scenario intelligence is needed, and use local critic/memory
review when no resolver result, scenario patch, tool result, hidden-content risk, or pending rules
opportunity is present. Micro-gates and compact contracts remain explicit A/B experiments because
true online tests showed provider timeouts and no reliable wall-clock win.

Implementation work:

- Classify each turn into a generic complexity profile:
  - safe observation;
  - direct information answer;
  - memory recall;
  - risky action;
  - scene transition;
  - boundary or authority risk;
  - high-risk output review.
  - Partially done through `turn_complexity` trace metadata built from structured routing, turn
    plan, resolver, scenario, and rule-opportunity state.
- Build a fast path for safe observation and direct answers: lightweight router, Archivist packet,
  Narrator, and deterministic lint.
  - Done for structured low-risk answer, rules-query, memory-recall, and free-action routes that do
    not require rules resolution or scenario direction.
- Add a package-owned visible-surface selector before full scenario direction.
  - Done for the `fast` profile. The selector receives only active-scene `visible_surfaces`,
    structured routing, turn-plan summary, and public/revealed scene state; invalid or uncertain
    selector output falls back to the full scenario director. Malformed optional selector output can
    recover with the first package-authorized visible surface and records `selector_error` without
    weakening visibility boundaries.
- Skip scenario direction only when programmatic runtime checks or structured routing show it is not
  needed.
  - Done with `scenario_director` skip reasons for no durable runtime, no loaded scenario,
    clarification/boundary turns, failed required resolution, conditional visible-surface
    selection, and routes that do not need scenario intelligence.
- Keep full critic mandatory for resolver results, hidden-context exposure, scenario patches,
  boundary repairs, and any high-risk output.
  - Done for the implemented fast slice: those states continue to the full LLM advisor path.
- Trigger memory curation on durable events, scene changes, explicit player preference updates, or
  fixed turn intervals, rather than every low-impact turn.
  - Partially done by using local no-write curation for low-risk turns without durable events.
- Persist the skip reason for every advisor that does not run.
  - Done for the current skips: `core_gm`, `scenario_director`, `critic_guardrail`, and
    `memory_curator`.

Verification:

- Fixture tests for each complexity profile and advisor skip reason.
  - Started with low-risk direct plan, visible-surface selector, selector fallback, and
    risky-action full-path regression tests.
- Online 2-5 turn A/B comparing advisor call counts, fallback count, clarification rate, and quality
  score.
  - Required before promoting conditional scheduling beyond the `fast` profile.
- Regression tests for risky action resolver enforcement under fast path.
  - Done for the current direct-plan bypass guard.

Acceptance:

- Safe observation turns use fewer LLM calls than the full path.
  - Done in graph tests; live wall-clock benefit remains provider-dependent and must be measured.
- Resolver bypass, hidden leaks, and unsupported facts remain at zero in regression and live smoke.
  - Regression guard exists for resolver bypass; live smoke remains required.
- Trace explains why each advisor ran or was skipped.
  - Done for implemented skip paths through `advisor_skip_reasons`.

## Milestone 17: Archivist/Narrator Split

Goal: Separate fact assembly and disclosure policy from player-facing prose.

Current status: first boundary implemented in the context budgeter. Narrator now consumes an
`ArchivistTurnPacket` in enforced mode instead of raw retrieved spans. Existing nodes still keep the
full graph state for replay/debug; the remaining work is critic enforcement against the packet and
promotion from fast-profile enforcement to the default profile.

Implementation work:

- Add an `ArchivistTurnPacket` containing only authorized facts for this turn:
  current visible scene, relevant rules envelope, tool/resolver results, validated scenario
  context, unresolved hooks, allowed citations, and explicit no-go boundaries.
  - Done for narrator context in enforced mode.
- Let the Archivist own progressive disclosure and context budgeting.
  - Done as deterministic role-specific packet construction; LLM context planning remains a future
    experiment and cannot bypass visibility/budget checks.
- Let Narrator consume only player input, the Archivist packet, and style state.
  - Done in enforced mode.
- Prevent Narrator from directly consuming GM-only retrieved spans or full package text.
  - Done in enforced mode with tests.
- Extend critic checks to verify final text stays inside the Archivist packet.
  - Follow-up.

Verification:

- Unit tests prove hidden retrieved spans cannot reach Narrator context.
- Fake-model graph tests show Narrator can produce final text from packet-only context.
- Critic tests catch final text that introduces facts outside the packet.

Acceptance:

- Narrator is a prose layer, not a fact-discovery layer.
- Unsupported durable facts do not increase after the split.
- Prompt size and final-text controllability improve against the current path.

## Milestone 18: Style State and Human-Likeness Evaluation

Goal: Improve GM voice without weakening rules, fact, or agency guardrails.

Current status: not implemented. Narration prompts enforce safe and concise output, but there is no
explicit durable style layer or style judge.

Implementation work:

- Add session-level `style_state` separate from canon, memory, and world state.
- Include GM tone, narrative distance, question style, dice reveal style, pressure curve, dialogue
  ratio, allowed sensory palette, and taboo patterns.
- Add NPC voice state with stance, voice, last emotion, and secrecy boundary, provided by scenario
  packages or validated state.
- Add a style judge with dimensions:
  - `human_likeness`;
  - `voice_consistency`;
  - `npc_distinctiveness`.
- Build a small style comparison dataset before considering SFT or LoRA.

Verification:

- Schema tests for style state and NPC voice state.
- Judge tests that distinguish mechanical restatement, voice drift, and NPC sameness.
- Live eval compares existing scorecard with style judge results.

Acceptance:

- Narration feels less templated while player agency, hidden-leak, and unsupported-fact scores do
  not regress.
- Style state cannot authorize durable facts.
- NPC voice differences are visible in transcripts and measurable by the style judge.

## Milestone 19: Prefix Cache and Cross-Turn Reuse

Goal: Make caching improve normal play latency, not only replay idempotency.

Current status: turn-scoped advisor caching exists. Because run ids include `turn_id`, this is
correct for replay but weak for cross-turn acceleration.

Implementation work:

- Keep turn-scoped advisor cache unchanged for deterministic replay.
- Add versioned stable prefix builders for system prompts, package manifests, resolver/tool schemas,
  rules summaries, and style profiles.
- Place stable prompt sections before volatile turn context for providers that support prompt
  caching.
- Record stable prefix size, cache eligibility, and provider cache metadata when available.
- Add metrics that separate turn-cache hits from prefix-cache opportunities.

Verification:

- Unit tests for stable prefix hashing and version invalidation.
- Advisor metrics tests for prefix-size and cache-eligibility fields.
- Online A/B on a provider with prompt caching when available.

Acceptance:

- Replay safety remains unchanged.
- Stable context is versioned and reused across turns.
- Repeated play in the same ruleset/scenario shows lower provider-side latency where caching is
  supported.

## Milestone 20: Rules Plugin API and Thin Rules DSL

Goal: Make new rulesets easier to add without widening core graph logic.

Current status: initial ruleset-owned plugin support is implemented through `rules_dsl_v1`.
Compiled rulesets may reference a package-local `plugin.yaml`, rules advisors may select loaded
procedure/check/difficulty ids, and the resolver dispatches through the package plugin before
falling back to legacy built-in resolver families. `coc7_light_investigation` and
`black_tide_beacon` exercise multiple attributes, skills, difficulty levels, pushed checks, sanity,
luck, and pressure clocks. Remaining work is to harden the lifecycle into explicit prepare /
resolve / explain-public phases and add a non-target-number ruleset.

Flow before this milestone:

1. The core graph loaded a compiled ruleset and selected a built-in resolver family such as a
   threshold or sum resolver.
2. Rules-specific behavior could only grow by adding resolver code or by stretching generic
   resolver fields.
3. Scenario/rules tests were easy to add, but a new game with a materially different procedure
   risked pushing game-specific interpretation into the graph.

Flow after the first plugin implementation:

1. Play startup resolves the active ruleset/scenario, preloads package profiles and compiled
   package data, and warms the indexed content store.
2. The rules advisor can only request procedure/check/difficulty ids available in the loaded
   ruleset package.
3. The resolver loads the ruleset-owned plugin, validates requested ids/modifiers, performs
   deterministic dice through the shared dice tool, and emits a consequence envelope.
4. Narration consumes validated tool results and package context; it does not invent rules
   consequences or inspect ruleset-specific keywords in core graph code.

Implementation work:

- Standardize a ruleset plugin lifecycle:
  - prepare;
  - resolve;
  - explain public result.
- Define a thin JSON rules DSL for procedures, approach/stat ids, allowed modifiers, result bands,
  opportunities, and authorized effects.
- Ensure rules advisors can select only loaded ids.
- Ensure resolvers emit an authoritative consequence envelope consumed by Narrator.
- Add at least one fiction-first or move-based smoke ruleset to validate non-target-number play.

Verification:

- Plugin contract tests for prepare, resolve, and explain-public.
- Cross-ruleset golden cases where one fictional action maps differently by loaded ruleset.
- Architecture guardrail scan proving new ruleset terms stay out of core source.

Acceptance:

- Adding a new ruleset requires content/plugin/tests, not core graph edits.
- Narrator never invents rule consequences outside the resolver envelope.
- The runtime supports more than numeric target-family resolvers.

## Milestone 21: Scenario Transition Predicates

Goal: Replace action keyword transition matching with package-owned structured triggers.

Current status: scenario transition behavior is package-validated and advisor-guided, but richer
deterministic predicates are still a known risk.

Implementation work:

- Add structured transition predicates or triggers to scenario packages.
- Let Scenario Director output trigger evidence and candidate transition ids.
- Validate transitions against package predicates, current state, scene set, and patch authority.
- Keep legacy `action_keywords` compatibility but mark it deprecated in docs and tests.

Verification:

- Transition tests cover success, failure, partial success, passive waiting, wrong direction, and
  investigation.
- A new scenario transitions without `action_keywords`.
- Source scan proves no scenario-specific transition logic enters core graph.

Acceptance:

- Scene transitions feel like GM judgment but remain package-owned and replayable.
- Hidden scene content is revealed only through validated transition/reveal patches.
- Complex scenarios can advance without brittle action keyword lists.

## Milestone 22: Semi-Open Long Play and Observer System

Goal: Move beyond deterministic scripted long-play into semi-open play quality evaluation.

Current status: deterministic 50-turn long-play checks repetition, unresolved hooks, memory QA,
replay, and trace coverage. Future work needs player-simulator and observer agents.

Implementation work:

- Add a semi-open player simulator that acts only from visible transcript and public state.
- Add an observer judge for pacing, hook lifecycle, memory use, style consistency, repetition, and
  player agency.
- Track unresolved hook creation, updates, resolution, and stale hooks over 50-100 turns.
- Ensure long-play failures produce actionable roadmap categories.
- Add cost and timeout controls so incomplete online long-play still yields a diagnostic report.

Verification:

- 50-turn semi-open playtest with observer judge.
- 100-turn technical stress test for replay and durable idempotency.
- Memory QA across early, middle, and late session facts.

Acceptance:

- Semi-open play remains coherent for at least 50 turns.
- 100-turn stress does not duplicate durable state.
- Observer findings can be grouped into roadmap items.

## Milestone 23: Play Profiles and Delivery Quality

Goal: Make the default CLI playable without requiring the user to understand experimental flags.

Current status: baseline implemented. `trpg play` now exposes `--profile
fast|balanced|theatrical` plus `--local`, progress, JSON, session, ruleset, and scenario options.
The old experiment switches are hidden from normal play help. `trpg eval online-playtest` also
accepts `--profile` while keeping explicit experiment flags for A/B reproduction.

Implementation work:

- Add `--profile fast|balanced|theatrical` to `trpg play` and online eval.
  - Done.
- Define profile defaults:
  - `fast`: stable multi-advisor path, conditional low-risk advisor skips, legacy contracts,
    parallel review, enforced context budgeting, fast runtime budget;
  - `balanced`: stable multi-advisor path, legacy contracts, shadow context budgeting, balanced
    runtime budget;
  - `theatrical`: legacy contracts and theatrical runtime budget, ready for future style work.
  - Done as profile config. Micro-gates and compact contracts remain explicit online-eval A/B
    overrides because the latest live smoke showed provider-side schema repair fallbacks and
    first-turn clarification regressions.
- Keep explicit expert flags as overrides.
  - Done for online eval; play keeps hidden compatibility flags.
- Improve progress reporting to show stages such as retrieval, rules, scenario, narration, critic,
  and memory.
  - Already present and now profile-independent.
- Document profile tradeoffs in user-facing docs.
  - Partially done in production notes; fuller user docs remain follow-up.

Verification:

- CLI tests cover profile defaults, hidden play experiment flags, and eval experiment flags.
- 2-5 turn online smoke for each profile.
- Interactive smoke verifies progress output and session resume still work.

Acceptance:

- A user can start play with one profile choice instead of several experimental flags.
- The default profile is stable enough for normal solo play.
- Progress output explains what the GM is waiting on during slow LLM calls.

## Release Gates

### Technical Reliability Gate

- Deterministic tests pass.
- Content check passes.
- Offline eval passes.
- Durable replay test passes.
- No duplicated dice, canon, memory, or world patches under replay.

### Architecture Gate

- No concrete ruleset or scenario content appears in core source.
- New rules/scenario behavior is package-owned.
- LLM advisors cannot directly persist durable state.
- Narration cannot override tools or resolver results.

### Play Quality Gate

- Live LLM judge average score is at least 4/5 on key dimensions.
- No critical hidden leaks.
- No resolver bypass for risky uncertain actions.
- Player agency violations are below the configured threshold.
- Long-play transcript remains coherent after at least 50 turns.
- New style metrics do not regress reliability metrics.
- Online smoke reports node latency, fallback counts, clarification rate, and profile settings.

## Immediate Next Development Sequence

1. Milestone 14: add runtime profiling and latency budgets so online slowness is diagnosable.
2. Milestone 15: implement context budgeting and indexed content retrieval.
3. Milestone 16: implement conditional advisor scheduling while preserving safety gates.
4. Milestone 17: split Archivist fact assembly from Narrator prose generation.
5. Milestone 18: add style state and human-likeness evaluation.
6. Milestone 19: add provider-friendly stable prefix caching and cross-turn reuse metrics.
7. Milestone 20: standardize rules plugin APIs and a thin rules DSL.
8. Milestone 21: replace scenario action keywords with structured transition predicates.
9. Milestone 22: add semi-open long-play and observer evaluation.
10. Milestone 23: package runtime choices into `fast`, `balanced`, and `theatrical` play profiles.

This order assumes Milestones 0-13 remain the reliability foundation. The next phase should optimize
the path through that foundation rather than adding heavier always-on agents.

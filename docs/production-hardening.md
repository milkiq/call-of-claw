# Production Hardening Notes

## Durable Schema

The SQLite store records applied migrations in `schema_migrations`. Tests assert the current schema
version so future migrations are explicit and replayable.

## Trace Redaction

Persisted turn traces pass through secret redaction before storage. The policy redacts common key
names such as `api_key`, `authorization`, `access_token`, and string shapes such as `sk-*`, bearer
tokens, and AWS-style access keys.

## Advisor Diagnostics

Advisor trace metadata includes:

- `elapsed_ms`;
- `estimated_prompt_chars`;
- `player_input_chars`;
- `context_chars`;
- `schema_chars`;
- `context_key_chars_json`;
- `estimated_response_chars`;
- `attempt_count`;
- prompt version;
- schema name;
- advisor contract mode;
- advisor skip reasons;
- cache status.

These metrics are coarse and provider-independent. They are intended for trend analysis and
debugging, not billing-grade accounting.

`trpg eval advisor-metrics` summarizes these diagnostics per session and can compare a legacy
session with a compact-contract session.

`trpg eval observation-report` summarizes report files and persisted advisor runs. It highlights
runtime-profile coverage, slowest runtime nodes, advisor fallback/timeout counts, prompt-size
breakdowns, and missing retrieval diagnostics. Use it before changing context budgets, retrieval,
advisor scheduling, or provider settings.

Online and long-play reports also include `clarification_turns`, `clarification_rate`, and
`first_turn_clarification`. These are trace-derived quality signals used to catch regressions where
safe current-situation observation gets converted into unnecessary clarification.

## Runtime Profiling

Turn graph execution attaches a `runtime_profile` to the returned graph state. The profile records
the selected latency budget, per-node completion timing, total elapsed time, slowest nodes, node
count, coarse latency category summaries, and trace-derived fallback/timeout counters. This is
reporting-only instrumentation; it must not influence routing, rules adjudication, scenario
movement, or narration.

Online playtest reports copy the runtime summary into metadata under keys such as
`runtime_total_elapsed_ms`, `runtime_slowest_nodes`, `runtime_fallback_count`,
`runtime_timeout_count`, `runtime_advisor_timeout_count`, and `runtime_node_count`. Timeout and
fallback counters produce infrastructure findings so a technically completed run cannot hide
provider or fallback-risk behavior.

## Context and Retrieval Diagnostics

Content retrieval records `search_backend`, `files_scanned`, `chars_scanned`, `retrieved_chars`,
index rebuild state, and fallback status. Runtime play uses a SQLite FTS-backed content index when a
SQLite path is available and falls back to the old scan backend if the index cannot be used.

Context budgeting runs in two modes. `shadow` records bucket sizes and would-clip counts for stable
prefix, local scene, local rules, player-visible memory, recent canon, retrieved public/GM spans,
tool results, and style state without changing advisor inputs. `enforced` builds role-specific
`ContextPacket` inputs for LLM calls, filters by visibility, records machine-readable drop reason
codes, and sends summaries such as `package_index`, `tool_result_summaries`,
`turn_plan_summary`, and `ArchivistTurnPacket` instead of the full graph state.

The fast play profile enables enforced context budgeting. Balanced and theatrical profiles continue
to use shadow mode until live A/B quality gates promote enforcement. Context approval and clipping
are programmatic; no additional LLM approval step is introduced. A future context-planner experiment
may request reference ids, but the system must still apply visibility, budget, and citation checks
before loading text.

Recent live A/B showed that enforced context with legacy contracts passed the 2-turn smoke, while
compact contracts and micro-gates caused schema repair fallbacks or first-turn clarification
regressions on the configured provider. Compact contracts and micro-gates remain explicit eval
overrides rather than normal `fast` defaults.

## Model Fallback Policy

Local deterministic graph execution remains available without an LLM. Live commands catch model
setup and invocation failures and point developers to offline checks. Any future model fallback must
preserve advisor contracts, prompt versions, trace metadata, and deterministic tool authority.

## Play Profiles

`trpg play` exposes `--profile fast|balanced|theatrical` as the normal user-facing runtime choice.
The default `balanced` profile uses the stable multi-advisor path and shadow context budgeting.
`fast` uses the stable multi-advisor path with conditional low-risk advisor skips, legacy contracts,
parallel review, enforced context budgeting, and the fast latency budget. The conditional fast path
may skip the core GM planner when structured routing already proves a direct low-risk answer or
free action, may use a narrow scenario surface selector for package-authorized visible observation
details, may skip scenario direction when programmatic runtime checks or structured routing say it
is unnecessary, and may use local critic/memory review when no tool result, resolver result,
scenario patch, hidden-content risk, or pending rules opportunity is present. The surface selector
can only choose ids from the active scene's compiled `visible_surfaces`; invalid, uncertain, or
consequential selections fall back to the full scenario director. If the optional selector call
itself returns malformed output, fast mode may recover with the first package-authorized visible
surface and records `selector_error` in trace without treating it as a safety fallback.
`theatrical` preserves the stable path with shadow context budgeting and the theatrical latency
budget for future style work. Use `--local` for the no-model structural debug fallback.

Online eval accepts the same `--profile` and still exposes explicit experiment flags for A/B
reproduction. Reports must record both the selected profile and resolved graph flags.

## Release Gates

`trpg eval release-gates` runs content validation, offline eval, durable replay coverage, and
long-play reliability. Live quality gates remain separate because they require a configured model and
network access.

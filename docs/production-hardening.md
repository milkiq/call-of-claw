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

Context budgeting currently runs in shadow mode. It records bucket sizes and would-clip counts for
stable prefix, local scene, local rules, player-visible memory, recent canon, retrieved public/GM
spans, tool results, and style state. Shadow data must not change advisor inputs until budget
enforcement has separate quality gates.

## Model Fallback Policy

Local deterministic graph execution remains available without an LLM. Live commands catch model
setup and invocation failures and point developers to offline checks. Any future model fallback must
preserve advisor contracts, prompt versions, trace metadata, and deterministic tool authority.

## Play Profiles

`trpg play` exposes `--profile fast|balanced|theatrical` as the normal user-facing runtime choice.
The default `balanced` profile uses the stable multi-advisor path. `fast` enables compact contracts,
micro-gates, parallel review, and the fast latency budget. `theatrical` preserves the stable path
with the theatrical latency budget for future style work. Use `--local` for the no-model structural
debug fallback.

Online eval accepts the same `--profile` and still exposes explicit experiment flags for A/B
reproduction. Reports must record both the selected profile and resolved graph flags.

## Release Gates

`trpg eval release-gates` runs content validation, offline eval, durable replay coverage, and
long-play reliability. Live quality gates remain separate because they require a configured model and
network access.

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
- `estimated_response_chars`;
- `attempt_count`;
- prompt version;
- schema name;
- cache status.

These metrics are coarse and provider-independent. They are intended for trend analysis and
debugging, not billing-grade accounting.

## Model Fallback Policy

Local deterministic graph execution remains available without an LLM. Live commands catch model
setup and invocation failures and point developers to offline checks. Any future model fallback must
preserve advisor contracts, prompt versions, trace metadata, and deterministic tool authority.

## Release Gates

`trpg eval release-gates` runs content validation, offline eval, durable replay coverage, and
long-play reliability. Live quality gates remain separate because they require a configured model and
network access.

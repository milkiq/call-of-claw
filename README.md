# TRPG Agent

Generic TRPG GM agent runtime built around LangChain, LangGraph, LangSmith, Pydantic, and SQLite.

The core GM agent is intentionally rules-agnostic and scenario-agnostic. Concrete TRPG rules and
scenario-specific GM requirements are loaded through content packages and compiled extensions.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
trpg content check
trpg eval regression
trpg eval all --offline
trpg eval report
pytest
```

## Current Status

This is the new Python foundation. The old demo architecture is not part of this runtime.
Architecture guardrails are documented in [docs/architecture-red-lines.md](docs/architecture-red-lines.md)
and enforced by tests. The staged development roadmap is documented in
[docs/gm-agent-architecture-milestones.md](docs/gm-agent-architecture-milestones.md).

Implemented runtime pieces:

- Generic core GM prompt policy with smoke-content leakage tests.
- Content package registry with visibility-aware retrieval.
- Handwritten compiled smoke ruleset/scenario packages used by the generic runtime.
- LangGraph turn graph with local adjudication, optional LLM adjudication, deterministic tool
  execution, ruleset resolver calls, world-state patching, LLM narration, trace events, and
  optional SQLite turn/canon persistence.
- SQLite LangGraph checkpointing for CLI/live play, persisted turn replay by `turn_id`, and
  idempotent world patch application records.
- Specialist advisor contracts for intent routing, authority gating, rules advice, scenario
  direction, memory curation, and critic guardrails.
- LLM turn graph now uses the intent arbiter advisor before GM adjudication and resolver
  enforcement.
- Rules adjudicator advisor can feed advisory resolver parameters while deterministic resolver
  remains authoritative for dice, success bands, and world patches.
- Advisor outputs are cached by turn/role/prompt/context, advisor failures fall back safely, and
  advisor models can be selected independently.
- Resolver family registry includes two smoke resolver families used to verify ruleset extension
  boundaries.
- Deterministic dice and ruleset resolver tools exposed through LangChain tool schemas.
- SQLite session state, canon, memory, dice, turn, and eval-run storage.
- Local regression/eval cases, live LLM judge evaluation, persisted quality report, and roadmap
  derivation.
- Interactive CLI play loop with session reuse, `/recap`, `/session`, `/quit`, and ruleset-provided
  character creation before the first turn.

Live LLM checks use `llm.config.json`:

```bash
trpg session start --session-id demo --reset
trpg play --use-llm --input '我检查门口'
trpg play --use-llm --session-id demo --input '我尝试强行修理导航'
trpg eval live --limit 3
trpg eval all
```

Interactive play:

```bash
trpg play --session-id demo
trpg play --local --session-id local-demo
trpg play --session-id demo --single-turn-advisor --parallel-review --no-progress
```

When `--input` is omitted, `trpg play` starts a reusable loop. A new session asks the loaded
ruleset's compiled character-creation questions, then accepts player actions until `/quit`. Exiting
prints the session id and resume command. In an interactive terminal the input line uses
`prompt_toolkit` editing and shows safe progress stages while the GM graph is working.

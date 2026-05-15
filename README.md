# Call of Claw

Generic TRPG GM agent runtime built around LangChain, LangGraph, LangSmith, Pydantic, and SQLite.

The core GM agent is intentionally rules-agnostic and scenario-agnostic. Concrete TRPG rules and
scenario-specific GM requirements are loaded through content packages and compiled extensions.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
coc content check
coc eval regression
coc eval all --offline
coc eval report
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
coc session start --session-id demo --reset
coc play --use-llm --input '我检查门口'
coc play --use-llm --session-id demo --input '我尝试强行修理导航'
coc eval live --limit 3
coc eval all
```

Interactive play:

```bash
coc play --session-id demo
coc play --local --session-id local-demo
coc play --session-id demo --single-turn-advisor --parallel-review --no-progress
```

When `--input` is omitted, `coc play` starts a reusable loop. A new session asks the loaded
ruleset's compiled character-creation questions, then accepts player actions until `/quit`. Exiting
prints the session id and resume command. In an interactive terminal the input line uses
`prompt_toolkit` editing and shows safe progress stages while the GM graph is working.

## Release Bundle

Build a shareable macOS or Windows folder bundle from the current platform:

```bash
pip install -e '.[release]'
coc release build \
  --name black-tide-beacon \
  --ruleset-id coc7_light_investigation \
  --scenario-id black_tide_beacon \
  --profile balanced
```

The bundle is written under `dist/releases/` and contains the `coc` executable, active content
packages, `release.json`, `README-PLAY.md`, `data/`, and `llm.config.example.json`. It does not
include local sessions, eval data, or real API configs. The recipient copies
`llm.config.example.json` to `llm.config.json`, fills their own provider settings, runs
`coc doctor`, then runs `coc play`.

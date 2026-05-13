# Context Clipping Strategy

The runtime should not send one large undifferentiated JSON blob to every LLM call. Each advisor
gets only the context category needed for its contract.

## Categories

- Player input: the current player utterance, always included.
- Visible scene context: active scene id/title/public summary, visible clock, revealed facts,
  known clues, and pending player-facing rule opportunities. GM-only fields are removed.
- Rules context: ruleset id, resolver/profile metadata, character context, and retrieved rules
  spans. This may include non-player-facing rules text, but not hidden scenario secrets. Natural
  language dice expressions in player input are treated as text, not as resolver parameters.
- Target context: visible scene context plus public package references and public retrieved spans.
  Used only to decide whether the player target or intent needs clarification.
- Authority context: visible scene context, recent canon, player-visible memory, and public spans.
  Used only to decide whether the player asserted unsupported authority.
- Memory recall context: recent canon, player-visible memory hits, and memory hit counts. Used only
  to decide whether the player is asking about established history.
- Scenario director context: full scenario-authorized context after rules/tools, including GM-only
  material if the package allows it, because this advisor proposes hidden-safe patch changes.
- Narration context: turn plan, tool results, visible world state, player-visible memory, and
  retrieved content needed to write the player-facing answer.
- Critic context: final text plus the supporting trace needed to catch hidden leaks, unsupported
  facts, resolver bypasses, and agency violations.
- Memory curation context: final turn transcript, canon draft, critic status, and visibility labels.

## Experimental Micro-Gate Path

`--micro-gates` enables four parallel, narrow advisors:

- `authority_micro_gate`
- `risk_micro_gate`
- `target_micro_gate`
- `memory_recall_micro_gate`

Their structured outputs are merged into `routing_decision`. They do not replace the rules
adjudicator, scenario director, narrator, critic, or durable memory writer. This keeps the
experiment bounded: micro-gates can route the graph faster, but the authoritative resolution and
quality checks remain in the existing runtime.

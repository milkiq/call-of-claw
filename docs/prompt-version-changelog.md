# Prompt Version Changelog

This file records runtime prompt contract changes that can affect advisor caching, eval
comparability, or live play quality.

## Current Versions

- `advisor-contracts=compact`: optional short wire JSON contracts for runtime advisors. Compact
  outputs are adapted back into the existing internal schemas, and advisor cache keys include the
  contract mode so legacy and compact runs remain comparable.
- `core-gm-v5`: generic GM obligations only; v5 keeps framework-generated internal structured
  fields in English and treats broad current-situation observation as playable visible feedback
  rather than automatic target clarification.
- `intent-arbiter-v6`: route selection and advisor needs; v6 requires generated routing fields to
  be English, keeps unestablished prior sources/reveals routed to boundary, and routes broad
  current-situation requests toward answer/free-action feedback.
- `authority-gate-v3`, `authority-micro-gate-v3`, `intent-micro-gate-v3`, `risk-micro-gate-v2`,
  `target-micro-gate-v3`, `memory-recall-micro-gate-v2`: narrow routing gates. Each prompt answers
  one clipped-context question, cannot narrate outcomes or write state, and returns generated
  schema fields in English. The target gate now distinguishes non-blocking ambiguity from
  clarification that truly blocks safe advancement.
- `rules-adjudicator-v5`: rules procedure advice without direct world writes; v5 keeps
  `stakes`/`clarification_question` in English, keeps mechanical mapping as a GM responsibility,
  and forbids asking the player to roll manually.
- `scenario-director-v7`: scenario intelligence with package-bounded patch proposals; v7 keeps
  advisory reasoning in English and requires minimal grounded visible feedback for safe observation,
  inspection, and answer turns when public scene context is available.
- `memory-curator-v3`: durable memory and canon curation; v3 keeps curator reasoning/procedural
  notes in English and preserves original wording only when the text itself is the durable fact.
- `single-turn-advisor-v5`: combined advisory contract; v5 keeps generated internal schema fields
  in English, forbids manual player dice requests, and prevents broad observation from collapsing
  into target clarification.
- `generic-narration-v9`: player-facing narration constrained by tools, patches, memory, and
  visibility; v9 makes `final_text` match player language while incorporating requested visible
  scenario context for answer/free-action/GM-move turns.
- `critic-guardrail-v8`: narration-only critic and repair contract; v8 keeps findings/reasoning in
  English and requires `revised_final_text` to match the final/player language.
- `generic-trpg-judge-v2`, `player-simulator-v2`: eval prompts are English-only instruction
  surfaces. Judge output stays English; simulated player `action` matches the transcript language.

## Change Rules

- Bump an advisor prompt version whenever schema instructions, authority boundaries, or output
  interpretation changes.
- Do not bump prompt versions for typo-only edits that cannot affect model behavior.
- Record the reason, expected eval impact, and required regression/live eval before merging.
- Never put concrete ruleset or scenario lore into generic runtime prompts; package-specific
  requirements belong in content packages or compiled extensions.

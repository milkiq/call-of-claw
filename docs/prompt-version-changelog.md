# Prompt Version Changelog

This file records runtime prompt contract changes that can affect advisor caching, eval
comparability, or live play quality.

## Current Versions

- `core-gm-v3`: generic GM obligations only; no concrete ruleset or scenario terms. v3 adds a
  generic unsupported-premise boundary for unestablished prior sources, reveals, and permissions.
- `intent-arbiter-v4`: route selection and advisor needs; v4 removes example target keywords and
  routes unestablished prior sources/reveals to boundary.
- `authority-gate-v2`, `authority-micro-gate-v2`, `intent-micro-gate-v1`, `risk-micro-gate-v1`,
  `target-micro-gate-v1`, `memory-recall-micro-gate-v1`: experimental narrow routing gates. Each
  prompt answers one clipped-context question and cannot narrate outcomes, write state, or reveal
  hidden content.
- `rules-adjudicator-v4`: rules procedure advice without direct world writes; v4 keeps mechanical
  mapping as a GM responsibility and uses `clarification_question` only for missing fictional
  target, priority, consent, or intent, and forbids asking the player to roll manually.
- `scenario-director-v5`: scenario intelligence with package-bounded patch proposals; v5 keeps
  visible context within structured routing scope instead of bundling a separate scene beat,
  whole-scene tactical summary, or unrelated active threat.
- `memory-curator-v2`: durable memory and canon curation; contradictions require `should_write=false`
  and must not be restated as memory candidates.
- `single-turn-advisor-v3`: combined advisory contract; v3 routes unestablished prior
  sources/reveals to boundary and forbids manual player dice requests.
- `generic-narration-v7`: player-facing narration constrained by tools, patches, memory, and
  visibility; v7 requires boundary narration to correct unsupported premises in player-visible
  terms, forbids sensory prose that implies extra damage, injury, conditions, breaches, equipment
  loss, or lasting constraints beyond resolver-authorized consequences, and forbids asking the
  player to roll manually.
- `critic-guardrail-v7`: narration-only critic and repair contract; hidden leaks, resolver
  bypasses, canon contradictions, and player-agency violations are blocking at any severity.
  Unsupported facts block at medium or higher severity; v7 treats vivid prose that implies
  unauthorized lasting consequences as medium-or-higher unsupported facts and blocks manual player
  dice requests as resolver bypasses.

## Change Rules

- Bump an advisor prompt version whenever schema instructions, authority boundaries, or output
  interpretation changes.
- Do not bump prompt versions for typo-only edits that cannot affect model behavior.
- Record the reason, expected eval impact, and required regression/live eval before merging.
- Never put concrete ruleset or scenario lore into generic runtime prompts; package-specific
  requirements belong in content packages or compiled extensions.

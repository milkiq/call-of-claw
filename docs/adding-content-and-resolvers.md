# Adding Content and Resolvers

Use this checklist when adding a new game, scenario, capability skill, evaluator, or deterministic
rules plugin.

## Ruleset Package

1. Create `content/rulesets/<id>/manifest.yaml`.
2. Add public source rules, a `compiled.yaml` reference tagged `[compiled, rules]`, and a
   package-local `plugin.yaml` reference tagged `[plugin, rules]`.
3. Set `compiled.yaml.package_id` to the manifest id and `resolver_id: rules_dsl_v1`.
4. Put procedures, checks, dice shape, target rules, bands, opportunities, and authorized effects in
   `plugin.yaml`.
5. Add `character_creation` questions and mechanical assignments if the ruleset needs a specific
   character creation flow. The generic GM agent must only read this compiled spec; it must not
   hard-code game-specific character setup in core prompts or graph routing.
6. Do not add a built-in resolver family for a new ruleset. Extend the rules DSL only when a
   capability is genuinely reusable across games.
7. Add plugin tests proving deterministic replay and package-owned check/procedure selection.

## Scenario Package

1. Create `content/scenarios/<id>/manifest.yaml`.
2. Keep source text `gm_only` unless it is safe for player-facing retrieval.
3. Add `compiled.yaml` with scenes, public summaries, GM-only facts, patch allowlist, structured
   transition ids/triggers, clocks, and endings.
4. Put scenario-specific GM requirements in `gm_constraints`.
5. Do not use keyword transition fields. Scene changes must go through Scenario Director proposals
   and package trigger validation.
6. Add eval cases that use the scenario through normal graph routing.

## Capability Skill

1. Create `content/capability_skills/<id>/manifest.yaml`.
2. Keep reusable agent behavior in `SKILL.md`.
3. Attach it through a scenario or ruleset `extensions` list.
4. Do not make the core GM prompt inherit the skill permanently.

## Evaluator Skill

1. Create `content/evaluators/<id>/manifest.yaml`.
2. Store judge rubrics as content, not source constants.
3. Add tests or eval cases that prove the evaluator can be discovered by manifest.

## Rules DSL Extension

1. Extend `rules_dsl_v1` only for reusable mechanics that cannot be represented by current
   procedures/checks/dice/bands/effects.
2. Keep all concrete rules labels in package content, not graph routing.
3. Persist dice with stable `turn_id:resolver:<package>:<index>` ids.
4. Return narration constraints that forbid unlisted costs or state changes.

## Required Checks

- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
- `.venv/bin/trpg content check`
- `.venv/bin/trpg eval all --offline`
- `.venv/bin/trpg eval release-gates`
- Bounded live eval when prompts, routing, memory, disclosure, narration, or critic behavior changes.

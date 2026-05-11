# Adding Content and Resolvers

Use this checklist when adding a new game, scenario, capability skill, evaluator, or deterministic
resolver.

## Ruleset Package

1. Create `content/rulesets/<id>/manifest.yaml`.
2. Add public source rules and a `compiled.yaml` reference tagged `[compiled, rules]`.
3. Set `compiled.yaml.package_id` to the manifest id.
4. Choose an existing `resolver_id` when possible.
5. Add `character_creation` questions and mechanical assignments if the ruleset needs a specific
   character creation flow. The generic GM agent must only read this compiled spec; it must not
   hard-code game-specific character setup in core prompts or graph routing.
6. Add a resolver implementation only if the resolution family is materially different.
7. Add a resolver test proving deterministic replay and package-owned approach selection.

## Scenario Package

1. Create `content/scenarios/<id>/manifest.yaml`.
2. Keep source text `gm_only` unless it is safe for player-facing retrieval.
3. Add `compiled.yaml` with scenes, public summaries, GM-only facts, patch allowlist, transitions,
   clocks, and endings.
4. Put scenario-specific GM requirements in `gm_constraints`.
5. Add eval cases that use the scenario through normal graph routing.

## Capability Skill

1. Create `content/capability_skills/<id>/manifest.yaml`.
2. Keep reusable agent behavior in `SKILL.md`.
3. Attach it through a scenario or ruleset `extensions` list.
4. Do not make the core GM prompt inherit the skill permanently.

## Evaluator Skill

1. Create `content/evaluators/<id>/manifest.yaml`.
2. Store judge rubrics as content, not source constants.
3. Add tests or eval cases that prove the evaluator can be discovered by manifest.

## Resolver Implementation

1. Implement a resolver class in `src/trpg_agent/rules/compiled_resolver.py`.
2. Register it in `default_resolver_registry`.
3. Keep all concrete rules labels in compiled content, not graph routing.
4. Persist dice with stable `turn_id:resolver:<package>:<index>` ids.
5. Return narration constraints that forbid unlisted costs or state changes.

## Required Checks

- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
- `.venv/bin/trpg content check`
- `.venv/bin/trpg eval all --offline`
- `.venv/bin/trpg eval release-gates`
- Bounded live eval when prompts, routing, memory, disclosure, narration, or critic behavior changes.

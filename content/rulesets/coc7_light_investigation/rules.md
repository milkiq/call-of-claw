# COC7 Light Investigation

This is an original, simplified percentile investigation-horror ruleset for testing a generic TRPG
GM agent. It is inspired by the broad shape of d100 investigative play, but it is not a reproduction
of any published rules text.

Characters have attributes, skills, Luck, and Sanity. Risky uncertain actions use one loaded check:
skill, attribute, luck, sanity, or opposed. The runtime rolls `1d100`; results at or below the target
count as success.

Difficulty changes the target:

- Regular: full target.
- Hard: half target.
- Extreme: one fifth target.

Bonus and penalty are simplified for this test package: roll two d100 candidates and take the lower
for bonus or the higher for penalty. A pushed roll is a second attempt after failure; if it fails,
the GM must apply a clearer, stronger consequence from the scene or pressure clock.

The GM must never ask the player to roll manually. The rules plugin performs all rule dice.

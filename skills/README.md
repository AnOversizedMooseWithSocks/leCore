# leCore skill files

Copies of the agent skills for the skill folders, kept in the repo so they travel with the code they
describe. Each got the sweep-176 pass: a section, in the skill's own voice, on the typed-decision doors
(`typed`, `systemone_lint`, `route` tiered, `plan_change`, `decision_outcome`, the reflex, `verify_decision`,
`swarm_step` / `swarm_evaluate`) with the numbers those doors measured in that skill's kind of session, and
the sandbox lessons that applied there. The syntax reference the sections point at is
`docs/TYPED_DECISIONS.md` (JSON generated live by `tools/gen_typed_examples.py`).

| skill | what it drives | sweep-176 section |
|---|---|---|
| `lecore-boot/` | boot, both ends, the swarm, land it | Step 0/7/7b/7c: the natural path, the typed doors over `/invoke`, the code workflow, re-tiling after boot, the sandbox lessons |
| `lecore-render/` | render, bake, relight, export | kill-safe strips, the `__main__` guard, `sdf_scene_shader`, materials as typed decisions |
| `lecore-simulate/` | bodies, fluids, heat, fields, growth | §14: solver / material / preset as typed decisions, `swarm_step` with the solver's own numbers as done_when, checkpoint-per-frame |
| `lestudio/` | the 2-D studio | the benchmark-poster swarm recipe; verify text by geometry |
| `lestudio3d/` | the 3-D studio | §9: materials 8/8 and tools 1–3/8 → 8/8 by geometry, the rasteriser loop, the bake trap, the PyPI shadow, agents on one scene |

Every verb a section names exists on `UnifiedMind` (checked against the live mind when the pass was made).

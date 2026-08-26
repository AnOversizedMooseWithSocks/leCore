"""CreatureMind -- the reference DEMO of building a specialized mind ON the one UnifiedMind.

KEPT NEGATIVE (no engine door on purpose): this is a WORKED EXAMPLE meant to be READ, not imported -- it shows how a
caller composes a domain-specific mind on top of UnifiedMind's faculties. Wiring UnifiedMind to instantiate a demo
of itself would be circular; the value is the pattern in the source, not a callable. The faculties it demonstrates
(memory, recall, procedural generation) are each already wired on UnifiedMind directly.

THE THREE MINDS -- one division of labour, so this never gets confusing again:
  - UnifiedMind     : THE ONE MIND (holographic_unified.py) -- every general faculty (the single encoder,
                      the memory, recall, planning, denoising, the decision machinery). Everything builds on
                      it. New capability that any mind could use belongs there, not here.
  - CreatureMind    : a SPECIALIZED LAYER on UnifiedMind  <<< THIS CLASS.  Subclasses UnifiedMind, inherits
                      every faculty, and adds only domain wiring (sense / act / learn). The template for any
                      new specialized mind (image generation, a market mind, ...).
  - HolographicMind : the RL ENGINE (holographic_creature.py) -- a per-action prototype value memory + greedy
                      policy that UnifiedMind uses internally; MEASURED to beat value-learning on the unified
                      memory (exp_value_memory.py). NOT an agent pattern: build agents from THIS class, not
                      from the engine.

THE PATTERN
-----------
There is ONE mind in holostuff -- UnifiedMind -- where all general, composable functionality lives: the
single encoder, the self-organising memory, recall / recognise, planning and re-anchoring, traversal,
denoising, the decision machinery. A *specialized* mind is a thin LAYER on top. It inherits every faculty
and adds only its domain wiring -- it never reimplements a primitive the one mind already has.

This is that layer for a reactive creature / agent, and it is meant to read as the template for any other
specialized mind (image generation, a market mind, whatever): **subclass UnifiedMind, name your domain's
moving parts, and drive the inherited faculties.** Each member of the project's advisory panel builds the
same way -- a general substrate, then a specialization on top -- which is the whole point of keeping the
substrate one mind.

WHAT IT DEMONSTRATES (and what it deliberately does NOT do)
  * It uses the mind's ONE encoder (`perceive`) for its egocentric senses -- there is no separate creature
    encoder. The mind's own docstring says it plainly: "the memory and the brain never encode anything
    themselves." A specialized layer honours that.
  * Its policy, learning, planning, and recall are the INHERITED faculties -- `decide` / `reinforce`,
    `plan` / `replan_needed`, `recall` / `recognize` -- reached on `self`, not rebuilt.
  * So a CreatureMind is, in one object, a full mind that can also act, learn, and navigate. That is the
    exemplar: the specialization is a handful of convenience methods over the one mind, nothing more.

This sits BESIDE the standalone `HolographicMind` (the lower-level RL engine UnifiedMind still wraps today)
during the migration; the migration plan's later phases retire that duplication so there is genuinely one
encoder and one memory. CreatureMind is the shape that migration is heading toward, written down now.
"""

from holographic.misc.holographic_unified import UnifiedMind


class CreatureMind(UnifiedMind):
    """A reactive creature/agent as a specialized layer on the one mind. Subclass-and-wire is the whole
    recipe: name the actions, then act / learn / plan through the inherited faculties."""

    def __init__(self, dim=1024, actions=("N", "S", "E", "W"), seed=0, **kw):
        super().__init__(dim=dim, seed=seed, **kw)     # the full mind: encoder, memory, faculties
        self.actions(list(actions))                    # wire the inherited decision machinery to these actions

    # ---- the creature loop, expressed entirely over inherited faculties --------------------------
    def sense(self, senses):
        """Egocentric senses (a role->value dict) -> a state vector, through the ONE encoder. No separate
        creature encoder: this is `perceive` in 'record' mode, the same encoder everything else uses."""
        return self.perceive(senses, "record")

    def act(self, senses, explore=True, epsilon=None, avoid=("danger", "wall")):
        """Choose an action from raw senses, using the inherited decision machinery (which perceives the
        senses through the one encoder and applies the brain's safety reflexes). The creature loop is just
        sense -> decide; nothing here is creature-specific except naming the call."""
        return self.decide(senses, modality="record", explore=explore, epsilon=epsilon,
                            senses=senses, avoid=avoid)

    def learn(self, senses, action, reward):
        """Fold one (senses, action, reward) experience into the inherited value memory via reinforce."""
        return self.reinforce(senses, action, reward, modality="record")


def _selftest():
    """CI-fast: prove a CreatureMind is, in ONE object, a full mind (it can recall, plan, denoise -- the
    inherited faculties) that ALSO acts and learns -- and that it uses the one encoder for its senses."""
    import numpy as np
    m = CreatureMind(dim=512, actions=("N", "S", "E", "W"), seed=0)

    # (1) it uses the ONE encoder for senses (no separate creature encoder)
    sv = m.sense({"food_x": "east", "danger_E": "yes"})
    assert sv.shape == (512,)
    assert np.allclose(sv, m.perceive({"food_x": "east", "danger_E": "yes"}, "record"))

    # (2) the creature loop runs on the inherited decision machinery: act -> learn -> act
    a = m.act({"food_x": "east"}, explore=False)
    assert a in ("N", "S", "E", "W")
    for _ in range(5):
        m.learn({"food_x": "east"}, "E", 1.0)
        m.learn({"food_x": "west"}, "W", 1.0)
    assert m.act({"food_x": "east"}, explore=False) == "E"   # it learned the rewarded action

    # (3) it ALSO has the full mind's faculties in the same object -- e.g. corridor planning (inherited)
    rng = np.random.default_rng(0)
    tiles = rng.standard_normal((9, 512)); tiles /= np.linalg.norm(tiles, axis=1, keepdims=True)
    def field_step(cur):
        i = int(np.argmax(tiles @ (cur / (np.linalg.norm(cur) + 1e-12))))
        return tiles[i + 1] if i + 1 < len(tiles) else None
    p = m.plan(tiles[0], field_step, max_steps=8, floor=0.12)
    assert p.route == list(range(1, 9))                      # the inherited planning faculty, on the creature


if __name__ == "__main__":
    _selftest()
    print("holographic_creature_mind (CreatureMind) selftest passed -- a specialized layer on the one mind")

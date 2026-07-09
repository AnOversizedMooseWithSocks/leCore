"""CI wrapper for CreatureMind -- the reference demo of a specialized mind built ON the one UnifiedMind.
The selftest proves it uses the one encoder for senses (no separate creature encoder), runs the
act/learn loop on the inherited decision machinery, and carries the full mind's faculties (planning) in
the same object -- i.e. the specialization is a thin layer, exactly the pattern."""
from holographic.agents_and_reasoning.holographic_creature_mind import _selftest


def test_creature_mind_is_a_layer_on_the_one_mind():
    _selftest()

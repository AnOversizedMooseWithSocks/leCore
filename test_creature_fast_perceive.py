"""Compiled, fully in-VSA perception (FastCreatureEncoder): the per-step role/filler binds are precomputed
once into a codebook, so perceiving recurring senses is a gather+sum -- no per-step FFT. Bit-identical to
the plain encoder; with the holographic value head the whole perceive->decide->learn loop is array ops."""

import numpy as np

from holographic_creature import (CreatureEncoder, FastCreatureEncoder, GridWorld,
                                  HolographicMind, run_episode)

DIM = 256


def test_compiled_perceive_is_bit_identical_to_the_base_encoder():
    base = CreatureEncoder(DIM, seed=1)
    fast = FastCreatureEncoder(DIM, seed=1)
    for s in ({"wall_N": "yes", "goal_E": "far"}, {"goal_E": "near", "wall_W": "yes"}, {}):
        assert np.array_equal(base.encode(s), fast.encode(s))


def test_compiled_perceive_caches_binds_to_zero_per_step_at_steady_state():
    fast = FastCreatureEncoder(DIM, seed=1)
    senses = {"wall_N": "yes", "goal_E": "far", "wall_S": "no"}
    for _ in range(50):
        fast.encode(senses)
    assert fast.binds_done == 3 and fast.binds_saved == 3 * 49      # 3 features bound once, reused thereafter


def test_perception_codebook_is_a_matrix():
    fast = FastCreatureEncoder(DIM, seed=1)
    fast.encode({"wall_N": "yes", "goal_E": "far"})
    mat, keys = fast.perception_codebook()
    assert mat.shape == (2, DIM) and len(keys) == 2


def test_full_in_vsa_loop_learns_a_maze():
    # compiled perceive (gather+sum) + routed hypervector brain (decide=dot, learn=bundle): all array ops.
    enc = FastCreatureEncoder(DIM, seed=1)
    mind = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=0.5, novelty_bonus=0.2,
                           memory_cap=12000, seed=1, value_backend="routed")
    world = GridWorld(7, 7, maze=True, fixed_seed=3)
    for ep in range(120):
        mind.epsilon = max(0.05, 0.5 * (1.0 - ep / 120))
        run_episode(world, enc, mind, learn=True, explore=True, mem=4, corridor_reflex=True, max_steps=90)
    got = 0
    for _ in range(20):
        run_episode(world, enc, mind, learn=False, explore=False, eval_epsilon=0.05, mem=4,
                    corridor_reflex=True, max_steps=90)
        got += world.escaped
    assert got / 20 >= 0.8
    assert enc.binds_saved > enc.binds_done * 10                    # steady state avoided the vast majority of FFTs

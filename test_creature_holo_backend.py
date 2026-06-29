"""The holographic value backend wired into the LIVE creature (value_backend='holo'): the whole brain runs
on a two-bundle hypervector policy. Same maze control as the tabular default, fixed-size savable policy.
Default ('table') stays bit-identical."""

import numpy as np

from holographic_creature import HolographicMind, CreatureEncoder, GridWorld, run_episode

DIM = 256


def test_default_backend_is_table_and_unchanged():
    m = HolographicMind(DIM, GridWorld.ACTIONS, seed=0)
    assert m.value_backend == "table" and m._holo is False and m._value_head is None
    # table path still works: absorb an experience, recall its value
    s = np.random.default_rng(0).normal(size=DIM); s /= np.linalg.norm(s)
    m._absorb(s, 1, 0.9)
    v, _ = m.value(s, 1)
    assert v > 0.5


def test_holo_backend_routes_value_and_learning_to_the_head():
    m = HolographicMind(DIM, GridWorld.ACTIONS, seed=0, value_backend="holo")
    assert m._holo and m._value_head is not None
    s = np.random.default_rng(1).normal(size=DIM); s /= np.linalg.norm(s)
    for _ in range(4):
        m._absorb(s, 2, 1.0); m._absorb(s, 0, 0.1)
    assert int(np.argmax([m.value(s, a)[0] for a in range(len(m.actions))])) == 2   # recalls the best action


def test_holo_policy_is_a_fixed_size_hypervector_program():
    m = HolographicMind(DIM, GridWorld.ACTIONS, seed=0, value_backend="holo")
    before = m._value_head.nbytes
    rng = np.random.default_rng(2)
    for _ in range(300):
        m._absorb(rng.normal(size=DIM), int(rng.integers(4)), float(rng.uniform()))
    Q, N = m._value_head.policy_vectors()
    assert m._value_head.nbytes == before                     # storage does not grow with experience
    assert Q.shape == (len(GridWorld.ACTIONS), DIM)


def test_holo_backend_learns_to_escape_a_maze():
    # end-to-end on the creature's REAL task: train with the holo backend, then it escapes reliably.
    enc = CreatureEncoder(DIM, seed=1)
    mind = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=0.50, novelty_bonus=0.2,
                           memory_cap=12000, seed=1, value_backend="holo")
    world = GridWorld(7, 7, maze=True, fixed_seed=3)
    for ep in range(120):
        mind.epsilon = max(0.05, 0.50 * (1.0 - ep / 120))
        run_episode(world, enc, mind, learn=True, explore=True, mem=4, corridor_reflex=True, max_steps=90)
    got = 0
    for _ in range(20):
        run_episode(world, enc, mind, learn=False, explore=False, eval_epsilon=0.05, mem=4,
                    corridor_reflex=True, max_steps=90)
        got += world.escaped
    assert got / 20 >= 0.8                                     # learns the maze on a hypervector policy


def test_routed_backend_and_brain_switch():
    from holographic_unified import UnifiedMind
    um = UnifiedMind(dim=256, seed=0)
    um.actions(["N", "S", "E", "W"], value_backend="routed")
    assert um._brain.value_backend == "routed" and um._brain._holo
    um.use_holographic_brain(routed=False)             # swap in place
    assert um._brain.value_backend == "holo"
    # routed backend learns via decide/reinforce path
    s = np.random.default_rng(0).normal(size=256); s /= np.linalg.norm(s)
    um.actions(["N", "S", "E", "W"], value_backend="routed")
    for _ in range(4):
        um._brain._absorb(s, 2, 1.0); um._brain._absorb(s, 0, 0.1)
    assert int(np.argmax([um._brain.value(s, a)[0] for a in range(4)])) == 2

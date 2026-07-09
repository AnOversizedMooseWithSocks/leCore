"""Sweep 3 item 4: online label-free concept growth -- distinct clusters form distinct concepts."""
import numpy as np
from holographic.simulation_and_physics.holographic_emergence import EmergentConcepts


def _unit(rng, d):
    v = rng.standard_normal(d); return v / np.linalg.norm(v)


def test_tight_cluster_forms_a_concept():
    rng = np.random.default_rng(0); D = 128
    c = _unit(rng, D)
    ec = EmergentConcepts(seed=0)
    for _ in range(8):
        ec.perceive(c + 0.03 * rng.standard_normal(D))
    assert len(ec.concepts) >= 1


def test_two_clusters_two_concepts():
    rng = np.random.default_rng(1); D = 128
    a = _unit(rng, D); b = _unit(rng, D)
    ec = EmergentConcepts(seed=0)
    for _ in range(10):
        ec.perceive(a + 0.02 * rng.standard_normal(D))
        ec.perceive(b + 0.02 * rng.standard_normal(D))
    assert len(ec.concepts) >= 2                        # label-free: it discovered both


def test_deterministic():
    rng = np.random.default_rng(2); D = 64
    stream = [_unit(rng, D) for _ in range(20)]
    e1 = EmergentConcepts(seed=0); e2 = EmergentConcepts(seed=0)
    for x in stream:
        e1.perceive(x); e2.perceive(x)
    assert len(e1.concepts) == len(e2.concepts)

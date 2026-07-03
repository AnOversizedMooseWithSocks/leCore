"""Sweep 3 item 3: reaction-diffusion HyperCA -- patterns emerge, stays finite, deterministic."""
import numpy as np
from holographic_automaton import HyperCA


def test_step_finite_and_normalized():
    ca = HyperCA(size=24, dim=32, seed=0)
    for _ in range(15):
        ca.step()
    assert np.isfinite(ca.grid).all()
    norms = np.linalg.norm(ca.grid, axis=2)
    assert np.allclose(norms, 1.0, atol=1e-6)          # cells stay unit vectors


def test_pattern_emerges():
    ca = HyperCA(size=32, dim=32, seed=0)
    start = ca.grid.copy()
    for _ in range(30):
        ca.step()
    # the field organized: it moved away from the initial random state
    assert np.abs(ca.grid - start).mean() > 1e-3


def test_deterministic():
    a = HyperCA(size=20, dim=24, seed=0); b = HyperCA(size=20, dim=24, seed=0)
    for _ in range(10):
        a.step(); b.step()
    assert np.array_equal(a.grid, b.grid)

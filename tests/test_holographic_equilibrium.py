"""CI wrapper for Equilibrium Propagation (Family 3 of the learning program -- the local-gradient corner).
The module ships its own asserts in _selftest: EP's symmetric contrastive update matches the true loss
gradient (cosine vs finite differences), and it learns a nonlinear task (two moons) past a linear model,
deterministically. This collects that check into the suite."""
from holographic.simulation_and_physics.holographic_equilibrium import _selftest


def test_holographic_equilibrium_selftest():
    _selftest()

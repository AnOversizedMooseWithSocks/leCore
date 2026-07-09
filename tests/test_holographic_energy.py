"""CI wrapper for the learned energy memory (EP trains the cleanup's attractors). The module ships its
own asserts in _selftest: on a continuous 2-D manifold the learned energy beats both the fixed soft
modern-Hopfield cleanup and a matched-memory codebook of random samples (the curse of dimensionality),
deterministically -- while on discrete atoms the hard 1-NN cleanup's exact recovery wins (the kept
negative). This collects that check into the suite."""
from holographic.simulation_and_physics.holographic_energy import _selftest


def test_holographic_energy_selftest():
    _selftest()

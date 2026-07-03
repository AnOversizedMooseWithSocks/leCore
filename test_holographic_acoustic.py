"""Acoustic (A2): impedance spread, big-mismatch reflection, energy conservation, absorption."""
import numpy as np
from holographic_acoustic import impedance, interface, wall_absorption, reflect_absorb


def test_impedance_spread():
    za, zw, zs = impedance("air"), impedance("water"), impedance("steel")
    assert za < zw < zs and 380 < za < 460 and 1.3e6 < zw < 1.6e6 and zs > 3e7


def test_big_mismatch_reflects_and_conserves_energy():
    R_aw, T_aw = interface("air", "water"); R_as, T_as = interface("air", "steel")
    assert R_aw > 0.99 and R_as > 0.999
    assert abs(R_aw + T_aw - 1.0) < 1e-12 and abs(R_as + T_as - 1.0) < 1e-12
    R_wi, T_wi = interface("water", "ice")
    assert T_wi > T_aw                                             # similar impedances transmit more


def test_absorption_split():
    hr, ha = reflect_absorb("concrete"); sr, sa = reflect_absorb("acoustic_foam")
    assert hr > 0.95 and sa > 0.8
    assert abs(hr + ha - 1.0) < 1e-12 and abs(sr + sa - 1.0) < 1e-12
    assert 0.0 <= wall_absorption("wood") <= 1.0

"""Tearable cloth (#2): overstretched links snap and the sheet separates; stronger materials tear less."""
import numpy as np
from holographic_tear import TearableCloth, tear_strength, TEAR_STRENGTH


def test_pulled_sheet_tears_and_separates():
    cloth = TearableCloth(rows=12, cols=12, material="paper", compliance=3e-3)
    n0 = cloth.n_constraints()
    assert cloth.connected_components() == 1
    for _ in range(120):
        cloth.step(pull=(0.0, -1200.0), gravity=(0.0, -9.8))
    assert cloth.torn > 0 and cloth.n_constraints() < n0
    assert cloth.connected_components() > 1


def test_stronger_material_tears_less():
    def torn(mat):
        c = TearableCloth(rows=12, cols=12, material=mat, compliance=3e-3)
        for _ in range(120):
            c.step(pull=(0.0, -1200.0), gravity=(0.0, -9.8))
        return c.torn
    assert torn("rubber") < torn("paper")
    assert tear_strength("rubber") > tear_strength("paper") > tear_strength("wet_paper")


def test_gentle_load_does_not_tear():
    calm = TearableCloth(rows=10, cols=10, material="cotton", compliance=1e-3)
    for _ in range(60):
        calm.step(gravity=(0.0, -0.5))
    assert calm.torn == 0 and calm.connected_components() == 1


def test_full_tear_gives_two_substantial_pieces():
    big = TearableCloth(rows=14, cols=14, material="wet_paper", compliance=4e-3)
    for _ in range(150):
        big.step(pull=(0.0, -1500.0), gravity=(0.0, -9.8))
    sizes = big.piece_sizes()
    assert len(sizes) >= 2 and sizes[1] >= 5


def test_deterministic():
    a = TearableCloth(rows=8, cols=8, material="paper", compliance=3e-3)
    b = TearableCloth(rows=8, cols=8, material="paper", compliance=3e-3)
    for _ in range(50):
        a.step(pull=(0.0, -1000.0), gravity=(0.0, -9.8)); b.step(pull=(0.0, -1000.0), gravity=(0.0, -9.8))
    assert a.torn == b.torn and np.array_equal(a.body.x, b.body.x)

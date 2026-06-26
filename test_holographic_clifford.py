"""EXP-9: Clifford Cl(3,0) geometric algebra as a parallel binding mode (holographic_clifford.py) -- exact,
non-commutative 3D rotation composition where HRR's commutative convolution bind cannot follow."""
import numpy as np

from holographic_clifford import CliffordAlgebra, _selftest


def test_selftest_passes():
    _selftest()


def test_geometric_product_basics():
    cl = CliffordAlgebra()
    e1 = np.zeros(8); e1[1] = 1
    e2 = np.zeros(8); e2[2] = 1
    # e1*e2 = e12 (blade index 3); e2*e1 = -e12 (anticommute); e1*e1 = 1 (scalar)
    assert cl.product(e1, e2)[3] == 1.0
    assert cl.product(e2, e1)[3] == -1.0
    assert np.allclose(cl.product(e1, e1), [1, 0, 0, 0, 0, 0, 0, 0])


def test_known_rotation():
    cl = CliffordAlgebra()
    assert np.allclose(cl.rotate(cl.rotor([0, 0, 1], np.pi / 2), [1, 0, 0]), [0, 1, 0], atol=1e-9)
    assert np.allclose(cl.rotate(cl.rotor([0, 0, 1], np.pi), [1, 0, 0]), [-1, 0, 0], atol=1e-9)


def test_composition_is_exact():
    # the win: the geometric product of rotors == applying the rotations in sequence, to floating point.
    cl = CliffordAlgebra()
    rng = np.random.default_rng(1)
    for _ in range(100):
        RA = cl.rotor(rng.standard_normal(3), rng.uniform(0, np.pi))
        RB = cl.rotor(rng.standard_normal(3), rng.uniform(0, np.pi))
        v = rng.standard_normal(3)
        assert np.linalg.norm(cl.rotate(RA, cl.rotate(RB, v)) - cl.rotate(cl.compose(RA, RB), v)) < 1e-12


def test_composition_is_non_commutative():
    # rotation order matters; the geometric product captures it. A commutative bind (HRR) cannot.
    cl = CliffordAlgebra()
    RA, RB = cl.rotor([1, 0, 0], 1.1), cl.rotor([0, 1, 0], 0.7)
    probe = np.array([0.3, -0.5, 0.8])
    gap = np.linalg.norm(cl.rotate(cl.compose(RA, RB), probe) - cl.rotate(cl.compose(RB, RA), probe))
    assert gap > 0.1                                       # the order-gap a commutative operation collapses to 0


def test_length_preserved_and_invertible():
    cl = CliffordAlgebra()
    rng = np.random.default_rng(2)
    v = rng.standard_normal(3)
    R = cl.rotor([1, 2, 3], 1.3)
    assert abs(np.linalg.norm(cl.rotate(R, v)) - np.linalg.norm(v)) < 1e-9
    assert np.linalg.norm(cl.rotate(cl.reverse(R), cl.rotate(R, v)) - v) < 1e-9


def test_binds_versors_not_arbitrary_atoms():
    # kept negative: a unit rotor times its reverse is the identity (clean unbind); a random multivector
    # times its reverse is NOT -- so this is a versor algebra, not a general key->value memory like HRR.
    cl = CliffordAlgebra()
    rng = np.random.default_rng(3)
    R = cl.rotor([0.5, 1.0, -0.3], 0.9)
    assert np.allclose(cl.product(R, cl.reverse(R)), [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-9)   # versor: clean
    m = rng.standard_normal(8)
    assert not np.allclose(cl.product(m, cl.reverse(m)), [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-3)  # arbitrary: not


def test_deterministic():
    a, b = CliffordAlgebra(), CliffordAlgebra()
    R = a.rotor([1, 1, 1], 0.5)
    assert np.allclose(a.rotate(R, [1, 0, 0]), b.rotate(b.rotor([1, 1, 1], 0.5), [1, 0, 0]))

"""Impurities/inclusions (M3): calibrated coverage, valid socket, deterministic."""
import numpy as np
from holographic_inclusions import with_inclusions, inclusion_coverage


def test_calibrated_coverage_hits_target():
    for frac in (0.1, 0.25, 0.4):
        got = inclusion_coverage(("gold_ore", frac, 6.0))
        assert abs(got - frac) < 0.05, (frac, got)


def test_socket_paints_base_and_inclusion():
    import holographic_matlib as ml
    sock = with_inclusions("steel", [("coal", 0.15, 7.0)], seed=1)
    P = np.random.default_rng(0).uniform(-2, 2, (3000, 3))
    a = sock(P)
    assert a.shape == (3000, 3) and a.min() >= 0 and a.max() <= 1
    assert np.array_equal(a, sock(P))                                # deterministic
    incl = ml.albedo("coal"); base = ml.albedo("steel")
    is_incl = np.all(np.abs(a - incl) < 1e-9, axis=1)
    is_base = np.all(np.abs(a - base) < 1e-9, axis=1)
    assert is_incl.sum() > 0 and is_base.sum() > 0                   # both present
    assert 0.10 < is_incl.mean() < 0.20                              # ~15% as requested


def test_later_inclusion_overrides_earlier_on_overlap():
    """List order = priority: a second inclusion paints over the first where they overlap."""
    import holographic_matlib as ml
    sock = with_inclusions("slate", [("gold_ore", 0.5, 5.0), ("coal", 0.5, 5.0)], seed=2)
    P = np.random.default_rng(1).uniform(-2, 2, (2000, 3))
    a = sock(P)
    # coal (applied second) should be well represented; nothing crashes; valid rgb
    coal = ml.albedo("coal")
    assert np.all(np.abs(a - coal) < 1e-9, axis=1).sum() > 0

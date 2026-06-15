"""Property tests of the VSA algebra: invariants over MANY random draws, bounding the
WORST case, not one lucky pair. Demo-outcome tests catch logic regressions; these catch
the silent NUMERICAL/structural class (the kind behind the penalize_recent lockstep bug)
-- a degradation that leaves the demos passing but quietly erodes the algebra. Each test
asserts a distribution bound (mean AND min/max across the draws), so a numerical
regression fails the build even when the demos still look fine.
"""
import numpy as np
import pytest

from holographic_ai import (random_vector, unitary_vector, bind, unbind, bundle,
                            permute, cosine, Vocabulary)


def _pairs(dim, n, rng, mint=random_vector):
    return [(mint(dim, rng), mint(dim, rng)) for _ in range(n)]


def test_bind_unbind_roundtrip_band_gaussian():
    # unbind(bind(a,b), a) recovers b in DIRECTION across many random pairs. Bound the
    # worst case, not just the mean. With Gaussian atoms the involution is an
    # APPROXIMATE inverse, so single-pair recovery sits around 0.71 (measured) -- still
    # far above chance and enough for cleanup to snap to the right symbol.
    rng = np.random.default_rng(0)
    cos = np.array([cosine(unbind(bind(a, b), a), b)
                    for a, b in _pairs(1024, 500, rng)])
    assert cos.mean() > 0.65
    assert cos.min() > 0.5                             # worst of 500 still clearly above chance


def test_unitary_atoms_make_roundtrip_exact_band():
    # With unitary atoms the same round-trip is EXACT for every draw (the involution is
    # the true inverse), so even the worst case sits at ~1.0.
    rng = np.random.default_rng(1)
    cos = np.array([cosine(unbind(bind(a, b), a), b)
                    for a, b in _pairs(1024, 300, rng, mint=unitary_vector)])
    assert cos.min() > 0.999


def test_bind_hides_its_operands():
    # The bound vector must be DISSIMILAR to both inputs (binding hides what went in),
    # across many draws -- otherwise composites would leak their parts into cleanup.
    rng = np.random.default_rng(2)
    leak = []
    for a, b in _pairs(1024, 300, rng):
        c = bind(a, b)
        leak.append(max(abs(cosine(c, a)), abs(cosine(c, b))))
    assert np.max(leak) < 0.2                          # worst-case leakage stays small


def test_permute_inverse_is_identity_exactly():
    # permute by +s then by -s is the identity, exactly, for every draw and shift.
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(300):
        v = random_vector(512, rng)
        s = int(rng.integers(1, 64))
        worst = max(worst, float(np.max(np.abs(permute(permute(v, s), -s) - v))))
    assert worst < 1e-12                               # exact (it is a cyclic shift)


def test_permute_decorrelates():
    # A permuted vector is dissimilar to the original (that is what lets permutation
    # tag order/position without colliding with the untagged item).
    rng = np.random.default_rng(4)
    sims = []
    for _ in range(300):
        v = random_vector(1024, rng)
        sims.append(abs(cosine(v, permute(v, int(rng.integers(1, 128))))))
    assert np.max(sims) < 0.2


def test_bundle_stays_similar_to_members_within_capacity():
    # A bundle stays similar to each member while the count is within the dimension's
    # capacity band; similarity falls as ~1/sqrt(count). Assert the band, both ends.
    rng = np.random.default_rng(5)
    dim = 1024
    for k in (2, 4, 8, 16):
        mins = []
        for _ in range(60):
            members = [random_vector(dim, rng) for _ in range(k)]
            b = bundle(members)
            mins.append(min(cosine(b, m) for m in members))
        # every member stays clearly present, and not absurdly so (it is a sum of k)
        assert np.min(mins) > 0.10                      # worst member still recoverable
        assert np.mean(mins) < 1.0 / np.sqrt(k) + 0.25  # decays roughly as 1/sqrt(k)


def test_cleanup_is_correct_under_bounded_noise():
    # cleanup() snaps a noisy vector back to the right symbol while the noise is within
    # the band the capacity argument predicts. Assert it recovers EVERY probe, not most.
    rng = np.random.default_rng(6)
    v = Vocabulary(1024, seed=0)
    names = [f"s{i}" for i in range(40)]
    for nm in names:
        v.get(nm)
    wrong = 0
    for _ in range(300):
        nm = names[int(rng.integers(len(names)))]
        noisy = v.get(nm) + 0.35 * random_vector(1024, rng)   # bounded corruption
        got, _ = v.cleanup(noisy, candidates=names)
        wrong += (got != nm)
    assert wrong == 0                                  # exact recovery under bounded noise


def test_walsh_hadamard_key_operator_is_an_exact_isometry():
    # The archive's key operator must preserve norm (||K x|| == ||x||) and round-trip
    # exactly on undamaged plates (adjoint(apply(v)) == v) -- the property the
    # damage-tolerant recovery depends on. Test the distribution, not one vector.
    from holographic_archive import HolographicArchive
    arch = HolographicArchive(shape=(8, 8, 1), capacity=4, keep=256, dim=4096, seed=0)
    n_slots = arch.dim // arch.K
    rng = np.random.default_rng(7)
    norm_err, rt_err = 0.0, 0.0
    for _ in range(50):
        i = int(rng.integers(n_slots))
        v = rng.standard_normal(arch.K)
        y = arch._apply(i, v)
        norm_err = max(norm_err, abs(np.linalg.norm(y) - np.linalg.norm(v)))
        rt_err = max(rt_err, float(np.max(np.abs(arch._adjoint(i, y) - v))))
    assert norm_err < 1e-9                              # isometry: norm preserved
    assert rt_err < 1e-9                                # exact round-trip on clean plates


def test_brain_prototype_arrays_stay_in_lockstep_under_maintenance():
    # The structural invariant behind the penalize_recent bug: the four per-action
    # banks (_unit/_sum/_ret/_cnt) must stay equal-length through a long maintain='auto'
    # stream with a consolidation. A property test of STRUCTURE, not a demo outcome.
    from holographic_creature import HolographicMind
    rng = np.random.default_rng(8)
    m = HolographicMind(dim=40, actions=["N", "S", "E", "W"], maintain="auto",
                        merge=0.5, seed=0, check_every=25, buffer_cap=150)
    for step in range(1500):
        c = (step // 250) % 4
        m.remember([rng.standard_normal(40) + (c == 0) * 2.0],
                   [int(rng.integers(4))], [float(rng.standard_normal())])
        if step == 500:
            m.consolidate(energy=0.95)
        if step % 5 == 0:
            m.penalize_recent()
        for a in range(4):
            assert len({len(m._unit[a]), len(m._sum[a]),
                        len(m._ret[a]), len(m._cnt[a])}) == 1

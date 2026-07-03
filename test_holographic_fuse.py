"""Fill 2: spectral fusion -- matches op-by-op to tolerance, fewer FFTs, tie-sensitive stays out."""
import numpy as np
from holographic_fuse import (fuse, fuse_record, leaf, fbind, funbind, fbundle, fpermute,
                              reset_fft_counts, fft_counts)
from holographic_ai import bind, unbind, bundle, permute, bundle_bind


def _atoms(rng, k, d):
    v = [rng.standard_normal(d) for _ in range(k)]
    return [x / np.linalg.norm(x) for x in v]


def test_each_op_matches_kernel():
    rng = np.random.default_rng(0); D = 1024
    a, b, c, d = _atoms(rng, 4, D)
    assert np.abs(fuse(fbind(a, b)) - bind(a, b)).max() < 1e-12
    assert np.abs(fuse(funbind(bind(a, b), a)) - unbind(bind(a, b), a)).max() < 1e-10
    assert np.abs(fuse(fpermute(a, 5)) - permute(a, 5)).max() < 1e-12
    assert np.abs(fuse(fbundle([a, b, c])) - bundle([a, b, c])).max() < 1e-12


def test_keystone_chain():
    rng = np.random.default_rng(1); D = 1024
    a, b, c, d = _atoms(rng, 4, D)
    ref = unbind(bundle([bind(a, b), c]), d)
    got = fuse(funbind(fbundle([fbind(a, b), leaf(c)]), d))
    assert np.abs(got - ref).max() < 1e-10


def test_fewer_ffts_than_opbyop():
    rng = np.random.default_rng(2); D = 1024
    for K in (4, 8, 16):
        atoms = _atoms(rng, K + 1, D)
        e = leaf(atoms[0])
        for x in atoms[1:]:
            e = fbind(e, x)
        reset_fft_counts(); fuse(e); c = fft_counts()
        assert c["rfft"] + c["irfft"] <= K + 2
        assert c["rfft"] + c["irfft"] < 3 * K


def test_record_matches_bundle_bind():
    rng = np.random.default_rng(3); D = 512
    keys = _atoms(rng, 6, D); vals = _atoms(rng, 6, D)
    assert np.abs(fuse_record(keys, vals) - bundle_bind(keys, vals)).max() < 1e-10


def test_deterministic():
    rng = np.random.default_rng(4); D = 512
    a, b, c, d = _atoms(rng, 4, D)
    e = funbind(fbundle([fbind(a, b), leaf(c)]), d)
    assert np.array_equal(fuse(e), fuse(e))


def test_sum_is_plain_superposition():
    """Fill 4 needs a plain (un-normalized) Sum for recipe's superpose op."""
    from holographic_fuse import fsum
    rng = np.random.default_rng(9); D = 512
    a, b, c = _atoms(rng, 3, D)
    got = fuse(fsum([a, b, c]))
    assert np.abs(got - (a + b + c)).max() < 1e-10          # plain sum, no renormalization

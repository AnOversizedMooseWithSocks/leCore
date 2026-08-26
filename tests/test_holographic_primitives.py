"""Vectorized VSA-program primitives (holographic_ai): the common operations programs run -- encoding a
record (role/filler bind+bundle) and decoding a trace against many keys -- as ONE batched FFT instead of a
Python loop. They match the scalar loops to FFT-batch epsilon; nearest() is an exact matmul cleanup."""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, bundle, unbind, cosine, involution, involution_batch, unbind_all, bundle_bind, nearest


def _rv(rng, D=512):
    v = rng.normal(size=D); return v / np.linalg.norm(v)


def test_bundle_bind_matches_the_encode_loop():
    rng = np.random.default_rng(0); K = 16
    roles = np.stack([_rv(rng) for _ in range(K)]); vals = np.stack([_rv(rng) for _ in range(K)])
    loop = bundle([bind(roles[i], vals[i]) for i in range(K)])
    assert np.max(np.abs(bundle_bind(roles, vals) - loop)) < 1e-10


def test_unbind_all_matches_the_decode_loop():
    rng = np.random.default_rng(1); K = 16
    trace = _rv(rng); keys = np.stack([_rv(rng) for _ in range(K)])
    loop = np.stack([unbind(trace, keys[i]) for i in range(K)])
    assert np.max(np.abs(unbind_all(trace, keys) - loop)) < 1e-10


def test_involution_batch_matches_scalar():
    rng = np.random.default_rng(2)
    K = np.stack([_rv(rng) for _ in range(8)])
    assert all(np.array_equal(involution_batch(K)[i], involution(K[i])) for i in range(8))


def test_nearest_is_exact_argmax():
    rng = np.random.default_rng(3)
    cb = np.stack([_rv(rng) for _ in range(200)])
    q = cb[42] + 0.3 * _rv(rng)
    j_loop = int(np.argmax([cosine(q, cb[i]) for i in range(len(cb))]))
    assert nearest(q, cb)[0] == j_loop


def test_classifier_encode_runs_through_bundle_bind():
    from holographic.agents_and_reasoning.holographic_ai import HolographicLearner
    enc = HolographicLearner(512, seed=0)
    ex = {"color": "red", "size": "big", "shape": "round"}
    v = enc.encode(ex)
    items = sorted(ex.items())                       # same get-call order the encoder uses
    roles = np.stack([enc.vocab.get(f) for f, _ in items])
    vals = np.stack([enc.vocab.get(val) for _, val in items])
    assert np.allclose(v, bundle_bind(roles, vals))


def test_faculties_delegate():
    from holographic.misc.holographic_unified import UnifiedMind
    um = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    roles = np.stack([_rv(rng, 256) for _ in range(4)]); vals = np.stack([_rv(rng, 256) for _ in range(4)])
    assert np.allclose(um.encode_pairs(roles, vals), bundle_bind(roles, vals))
    trace = _rv(rng, 256); keys = np.stack([_rv(rng, 256) for _ in range(4)])
    assert np.allclose(um.unbind_keys(trace, keys), unbind_all(trace, keys))


def test_partitioned_route_matches_cosine_loop():
    """The nearest pass: PartitionedMemory.route now uses nearest() (argmax of a matmul) instead of a Python
    cosine loop over the anchors -- exact, since the anchors are unit random_vectors."""
    from holographic.agents_and_reasoning.holographic_ai import PartitionedMemory, cosine
    pm = PartitionedMemory(256, num_partitions=8, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        k = rng.normal(size=256)
        loop = int(np.argmax([cosine(k, a) for a in pm.anchors]))
        assert pm.route(k) == loop

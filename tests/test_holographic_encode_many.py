"""encode_many: the batch the texture unit's sample path never had -- and the verdict it produces."""
import numpy as np
import pytest

from holographic.io_and_interop.holographic_encoders import ScalarEncoder
from holographic.rendering.holographic_shader import bake_nd, fetch_nd
from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder


def _enc(dim=512, n=3):
    return VectorFunctionEncoder(n, dim=dim, bounds=[(-1, 1)] * n, seed=0)


def test_encode_many_matches_stacking_encode_to_machine_epsilon():
    # NOT bit-identical, and it must not be asserted so: binding all axes' spectra at once reassociates the
    # products that pairwise `bind` performs in sequence. The same reassociation the emitted C twin shows.
    enc = _enc()
    P = np.random.default_rng(0).uniform(-0.9, 0.9, (64, 3))
    loop = np.stack([enc.encode(p) for p in P])
    batch = enc.encode_many(P)
    assert batch.shape == loop.shape
    assert np.abs(loop - batch).max() < 1e-14
    assert not np.array_equal(loop, batch)


def test_encode_many_is_faster_than_the_loop():
    import time

    enc = _enc(dim=1024)
    P = np.random.default_rng(0).uniform(-0.9, 0.9, (512, 3))

    def _t(fn, n=2):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    assert _t(lambda: enc.encode_many(P)) < _t(lambda: np.stack([enc.encode(p) for p in P]))


def test_the_scalar_encoder_batches_too_and_honours_a_fitted_warp():
    # `_warp_u`, not `_warp_y`. A first version reached for a name that does not exist -- harmless while no warp is
    # fitted, and a crash the moment one is. Fit one and check the batch agrees with the loop.
    e = ScalarEncoder(dim=256, lo=-1.0, hi=1.0, seed=0)
    xs = np.linspace(-0.9, 0.9, 32)
    assert np.abs(np.stack([e.encode(x) for x in xs]) - e.encode_many(xs)).max() < 1e-14

    if hasattr(e, "fit_resolution"):
        e.fit_resolution(np.random.default_rng(0).normal(0.0, 0.2, 500).clip(-1, 1))
        assert getattr(e, "_warp_x", None) is not None
        assert np.abs(np.stack([e.encode(x) for x in xs]) - e.encode_many(xs)).max() < 1e-14


def test_encode_many_refuses_the_wrong_arity():
    with pytest.raises(ValueError, match="coordinates"):
        _enc().encode_many(np.zeros((4, 2)))


def test_fetch_nd_uses_the_batch_path():
    g = [np.linspace(-1, 1, 8)] * 3
    G = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)
    vals = np.sin(G[:, 0]) * np.cos(G[:, 1]).reshape(-1)
    b = bake_nd(tuple(g), vals.reshape(8, 8, 8), dim=256, seed=0)
    P = np.random.default_rng(0).uniform(-0.8, 0.8, (16, 3))
    out = fetch_nd(b, P)
    assert np.shape(out) == (16,)
    assert np.isfinite(out).all()


def test_kept_negative_a_texture_bake_loses_to_direct_evaluation_on_a_cpu():
    # "Sample O(1)" is O(1) in the number of BAKED SAMPLES, not in `dim`: a fetch costs one O(dim log dim)
    # transform per point. Measured 166x SLOWER at 512 hits, and the bake does not even hold five octaves of fBm.
    import time

    from holographic.misc.holographic_pattern import fbm, field_lerp

    tex = field_lerp(fbm(octaves=5), 0.05, 0.95)
    g = [np.linspace(-1, 1, 16)] * 3
    G = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)
    b = bake_nd(tuple(g), np.asarray(tex(G), float).reshape(16, 16, 16), dim=512, seed=0)

    P = np.random.default_rng(0).uniform(-0.8, 0.8, (256, 3))

    def _t(fn, n=2):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    assert _t(lambda: fetch_nd(b, P)) > 5.0 * _t(lambda: tex(P))       # the bake is far slower on a CPU

    got = np.asarray(fetch_nd(b, P), float).ravel()
    want = np.asarray(tex(P), float)
    assert float(np.corrcoef(got, want)[0, 1]) < 0.9                   # ... and it does not hold the field

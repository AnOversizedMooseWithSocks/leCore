"""Tests for holographic_fieldhome -- the Field home (R2: one .sample interface over spatial field backends)."""
import numpy as np
from holographic_fieldhome import Field, field_backends


def _oracle(P):
    return 1.0 - np.linalg.norm(np.asarray(P, float), axis=1)


def _grid(lo, hi, N):
    axis = [lo[d] + np.arange(N) / N * (hi[d] - lo[d]) for d in range(3)]
    gx, gy, gz = np.meshgrid(axis[0], axis[1], axis[2], indexing="ij")
    return _oracle(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)).reshape(N, N, N), axis


def test_dense_and_callable_identical_at_nodes():
    lo = np.array([-1., -1., -1.]); hi = np.array([1., 1., 1.]); N = 16
    grid, axis = _grid(lo, hi, N)
    f_dense = Field.grid(grid, lo, hi)
    f_call = Field.callable(_oracle)
    probe = np.array([[axis[0][2], axis[1][5], axis[2][9]],
                      [axis[0][7], axis[1][1], axis[2][3]]])
    assert np.allclose(f_dense.sample(probe), f_call.sample(probe), atol=1e-9)   # two backends, identical values


def test_uniform_sample_interface():
    lo = np.array([-1., -1., -1.]); hi = np.array([1., 1., 1.])
    grid, _ = _grid(lo, hi, 12)
    for f in (Field.grid(grid, lo, hi), Field.callable(_oracle)):
        v = f.sample(np.zeros((5, 3)))
        assert v.shape == (5,) and np.isfinite(v).all()


def test_sparse_backend_routes():
    from holographic_sparsefield import SparseField
    def sdf(P):
        return np.linalg.norm(np.asarray(P, float), axis=1) - 0.5
    sp = SparseField.from_field(sdf, np.array([-1., -1., -1.]), np.array([1., 1., 1.]), voxel=0.1, band=0.3)
    f = Field.sparse(sp)
    assert f.kind == "sparse" and f.backend is sp
    # inside the band it agrees with the true SDF
    pts = np.array([[0.5, 0, 0], [0, 0.55, 0]])
    assert np.allclose(f.sample(pts), sdf(pts), atol=0.05)


def test_backends_listed():
    assert set(field_backends()) >= {"callable", "dense", "sparse"}


def test_repr_and_kind():
    f = Field.callable(_oracle)
    assert f.kind == "callable" and "callable" in repr(f)

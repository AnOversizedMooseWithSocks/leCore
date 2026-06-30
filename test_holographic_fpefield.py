"""Tests for FS-5: the surface carried as a single hypervector (holographic_fpefield.HolographicField)."""

import numpy as np

from holographic_fpe import VectorFunctionEncoder
from holographic_fpefield import HolographicField


def _sphere_field(R=0.6, B=1.3, dim=2048, bw=18.0):
    enc = VectorFunctionEncoder(3, dim=dim, bounds=[(-B, B)] * 3, bandwidth=bw, seed=0)
    g = np.linspace(-0.9, 0.9, 12)
    P = np.array([(x, y, z) for x in g for y in g for z in g])
    W = np.linalg.norm(P, axis=1) - R
    return enc, P, W, HolographicField(enc, P, W)


def test_selftest_runs():
    import holographic_fpefield
    holographic_fpefield._selftest()


def test_surface_is_a_single_vector_with_correct_sign():
    enc, P, W, field = _sphere_field()
    assert field.f.shape == (enc.dim,)                              # the whole surface is ONE hypervector
    assert float(field.value([[0.0, 0.0, 0.0]])[0]) < 0.0          # inside the sphere
    assert float(field.value([[0.7, 0.7, 0.7]])[0]) > 0.0          # outside (within the sampled cloud)


def test_field_value_uses_the_batched_read_semantics():
    enc, _, _, field = _sphere_field()
    pts = np.random.default_rng(0).uniform(-0.8, 0.8, (24, 3))
    loop = np.array([enc.query(field.f, p) for p in pts])
    assert np.allclose(field.value(pts), loop, atol=1e-12)


def test_edit_is_bind_exactly():
    """The headline: translating the whole field by a SINGLE binding makes value_shifted(x) == value_orig(x - delta)
    to machine precision, and the surface's zero-crossing moves by exactly the delta."""
    enc, P, W, field = _sphere_field()
    d = np.array([0.25, 0.0, 0.0])
    moved = field.translate(d)
    cg = np.linspace(-0.5, 0.5, 6)
    X = np.array([(a, b, c) for a in cg for b in cg for c in cg])
    assert np.max(np.abs(moved.value(X) - field.value(X - d))) < 1e-9    # exact translation of the field

    ray = np.linspace(0.0, 1.1, 120)
    def xcross(fld):
        v = fld.value(np.stack([ray, np.zeros_like(ray), np.zeros_like(ray)], axis=1))
        idx = np.where(np.diff(np.sign(v)) != 0)[0]
        return float(ray[idx[0]])
    assert abs((xcross(moved) - xcross(field)) - 0.25) < 0.02       # surface moved by exactly the delta

    # additive / non-destructive / deterministic
    assert np.array_equal(field.translate(d).f, moved.f)
    assert np.array_equal(field.f, HolographicField(enc, P, W).f)   # original untouched


def test_union_is_bundle():
    enc, P, W, field = _sphere_field()
    other = HolographicField(enc, P + np.array([0.7, 0.0, 0.0]), W)
    both = field.union(other)
    uv = both.value([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [1.6, 0.0, 0.0]])
    assert uv[0] < 0 and uv[1] < 0 and uv[2] > 0                    # inside at both centres, outside far away
    # union of fields from different encoders is rejected
    other_enc = VectorFunctionEncoder(3, dim=2048, bounds=[(-1.3, 1.3)] * 3, bandwidth=18.0, seed=1)
    bad = HolographicField(other_enc, P, W)
    try:
        field.union(bad)
        assert False, "should reject cross-encoder union"
    except ValueError:
        pass


def test_from_mesh_builds_and_reconstructs():
    """from_mesh samples a real mesh's signed distance and bundles it; the 0-level re-extracts a surface near the
    original scale (a smoothed estimate)."""
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
    R = 0.6
    m = marching_tetrahedra_vec(*sample_field(lambda P: np.linalg.norm(P, axis=1) - R, ((-1, -1, -1), (1, 1, 1)), 20), 0.0)
    field = HolographicField.from_mesh(m, ((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3)), dim=2048, bandwidth=18.0, grid=12)
    assert field.f.shape == (2048,)
    # +x zero-crossing near R (the smoothed-estimate band)
    ray = np.linspace(0.0, 1.1, 120)
    v = field.value(np.stack([ray, np.zeros_like(ray), np.zeros_like(ray)], axis=1))
    idx = np.where(np.diff(np.sign(v)) != 0)[0]
    assert len(idx) and 0.45 < float(ray[idx[0]]) < 0.85
    extract = field.surface(((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3)), res=20)
    assert extract.n_faces > 0


def test_delta_editing_is_additive_exact_undo_and_model_size_independent():
    """A local edit is a delta hypervector: apply_delta adds it, remove_delta subtracts it (exact undo by linearity),
    the edit is local (far side unchanged), and the cost is model-size-independent (the model is one fixed vector)."""
    import time
    enc, P, W, field = _sphere_field()
    q = np.array([0.6, 0.0, 0.0])
    bump = np.array([q + o for o in [(0, 0, 0), (0.05, 0, 0), (0, 0.05, 0), (0, -0.05, 0), (0, 0, 0.05), (0, 0, -0.05)]])
    delta = field.make_delta(bump, np.full(len(bump), -0.35))

    edited = field.apply_delta(delta)
    assert edited.value([q])[0] < field.value([q])[0]              # surface pushed out at the edit
    assert abs(float(edited.value([-q])[0]) - float(field.value([-q])[0])) < 0.02   # local: far side ~unchanged
    assert np.array_equal(field.f, _sphere_field()[3].f)           # apply did not mutate the original

    undone = edited.remove_delta(delta)
    assert np.max(np.abs(undone.f - field.f)) < 1e-9               # EXACT undo (linearity)

    # a redo/undo history is just adding/subtracting deltas; two edits compose and unwind exactly
    d2 = field.make_delta(np.array([[-0.6, 0, 0]]), np.array([-0.3]))
    two = field.apply_delta(delta).apply_delta(d2)
    back = two.remove_delta(d2).remove_delta(delta)
    assert np.max(np.abs(back.f - field.f)) < 1e-9

    # model-size independence: a 10x-bigger model is still one vector; the same delta applies at the same cost
    gL = np.linspace(-0.95, 0.95, 22)
    PL = np.array([(x, y, z) for x in gL for y in gL for z in gL])
    big = HolographicField(enc, PL, np.linalg.norm(PL, axis=1) - 0.6)
    assert big.f.shape == field.f.shape == (enc.dim,)
    assert big.apply_delta(delta).f.shape == (enc.dim,)

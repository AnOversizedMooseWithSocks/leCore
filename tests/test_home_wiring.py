"""The consolidation HOMES are "one door -- route, don't rewrite". A door that does not know about a capability is
a silo, and a door that offers two things side by side as if they were the same KIND of thing is worse.

This suite guards two joins found by an organisation audit:

  * `transformhome.Transform` offered `bind` and `scaling` side by side. `bind` is the abelian IDEAL of the affine
    group -- a convolution algebra can represent nothing else -- while `scaling` is the CENTRE of the linear part
    and is not a bind at all. The one door for transforms was silent about the structure that governs them.
  * `computehome.Compute` lists `fuse` -- "collapse a bind expression tree into ~2 FFTs" -- and did not know about
    `TransformBank`, which is **`fuse` with the leaves precomputed**. Wiring it exposed that the bank's published
    13.5x was measured against SEQUENTIAL BINDS, a strawman: `fuse` had already beaten those. The honest number is
    4.2x-5.3x over `fuse`.
"""

import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_ai import bind, unitary_vector
from holographic.misc.holographic_computehome import Compute
from holographic.misc.holographic_transformhome import Transform, transform_kinds


# ---------------------------------------------------------------------------------------------------------
# the TRANSFORM home now knows which floor each of its doors opens onto
# ---------------------------------------------------------------------------------------------------------

def test_the_transform_home_can_name_the_floor_of_its_own_doors():
    # `Transform.bind` is the abelian ideal; `Transform.scaling` is the centre, and is not a bind.
    assert Transform.hypervector_layer()["level"] == 1
    assert Transform.layer(lambda x: 1.7 * x)["name"] == "scale"
    assert Transform.layer(lambda x: x + np.array([0.1, 0.0, 0.0]))["name"] == "translation"
    assert Transform.layer(lambda x: 1.7 * x)["bankable"] is False
    assert Transform.layer(lambda x: x + np.array([0.1, 0.0, 0.0]))["bankable"] is True


def test_the_transform_home_reaches_the_projective_ceiling():
    assert Transform.is_affine(Transform.translation([1.0, 2.0, 3.0])) is True
    from holographic.mesh_and_geometry.holographic_projectivetower import projective
    assert Transform.is_affine(projective([0.1, 0.0, 0.0])) is False

    word = Transform.compose_word([Transform.translation([0.1, 0.2, 0.3]), projective([0.1, 0.0, 0.0])])
    assert Transform.is_affine(word) is False       # one projective letter makes the whole word projective


def test_the_transform_home_reaches_the_bank_and_the_adjoint():
    b = Transform.bank(128, seed=0)
    b.add_rotation("r3", 3)
    v = np.random.default_rng(0).normal(size=128)
    assert np.abs(b.apply("r3", v) - np.roll(v, 3)).max() < 1e-10
    assert b.layer_of("r3")["name"] == "translation"     # even the entry called "rotation"

    from holographic.mesh_and_geometry.holographic_equivariance import apply_affine, shade
    rng = np.random.default_rng(0)
    tri = rng.normal(size=(3, 3))
    L = np.array([0.0, 0.0, 1.0])
    A = 1.7 * np.eye(3)
    assert abs(Transform.adjoint(tri, L, A) - shade(apply_affine(A, 0, tri), L)) < 1e-12


def test_the_home_advertises_its_new_kinds():
    kinds = transform_kinds()
    assert "bind(vsa)" in kinds and "matrix(4x4)" in kinds        # the old doors
    for new in ("tower(which floor)", "bank(prebuilt spectra)", "projective(the ceiling)"):
        assert new in kinds, new


def test_the_home_routes_and_does_not_rewrite():
    # "route, don't rewrite" -- the home's answers must be the modules' answers, identically.
    from holographic.mesh_and_geometry.holographic_grouptower import classify_transform, hypervector_layer

    assert Transform.hypervector_layer() == hypervector_layer()
    assert Transform.layer(lambda x: 1.7 * x) == classify_transform(lambda x: 1.7 * x)


# ---------------------------------------------------------------------------------------------------------
# the COMPUTE home: the bank is `fuse` with the leaves precomputed
# ---------------------------------------------------------------------------------------------------------

def _chain(D=512, k=6, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=D)
    atoms = [unitary_vector(D, rng) for _ in range(k)]
    b = Compute.transform_bank(D, seed=0)
    for i, a in enumerate(atoms):
        b.add("t%d" % i, a)
    return v, atoms, b, ["t%d" % i for i in range(k)]


def test_fuse_and_the_bank_and_sequential_binds_all_agree():
    v, atoms, b, names = _chain()

    seq = v
    for a in atoms:
        seq = bind(seq, a)

    expr = Compute.leaf(v)
    for a in atoms:
        expr = Compute.bind(expr, Compute.leaf(a))
    fused = np.asarray(Compute.fuse(expr))

    chained = b.apply_chain(names, v)
    assert np.abs(fused - seq).max() < 1e-12
    assert np.abs(chained - seq).max() < 1e-12
    assert np.abs(fused - chained).max() < 1e-12


def test_kept_negative_the_banks_baseline_was_fuse_not_sequential_binds():
    # `fuse` already collapsed the tree to one forward transform per distinct LEAF plus one inverse. The bank wins
    # only because its leaves are ALREADY spectra: one forward transform of the INPUT. A win without the strongest
    # available baseline is not a win.
    import time

    v, atoms, b, names = _chain(D=1024, k=8)

    def _t(fn, n=6):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    def _fuse():
        e = Compute.leaf(v)
        for a in atoms:
            e = Compute.bind(e, Compute.leaf(a))
        return Compute.fuse(e)

    def _seq():
        x = v
        for a in atoms:
            x = bind(x, a)
        return x

    t_seq, t_fuse, t_bank = _t(_seq), _t(_fuse), _t(lambda: b.apply_chain(names, v))
    assert t_fuse < t_seq                       # fuse already beat sequential binds, before the bank existed
    assert t_bank < t_fuse                      # ... and the bank beats fuse, by precomputing the leaves


def test_the_compute_home_routes_the_batch_encoder():
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder

    enc = VectorFunctionEncoder(3, dim=256, bounds=[(-1, 1)] * 3, seed=0)
    P = np.random.default_rng(0).uniform(-0.5, 0.5, (8, 3))
    assert np.array_equal(Compute.encode_many(enc, P), enc.encode_many(P))


# ---------------------------------------------------------------------------------------------------------
# the audit itself: a facade is not a gap
# ---------------------------------------------------------------------------------------------------------

def test_the_reachability_audit_knows_a_facade_from_a_gap():
    # 13 consolidation homes sat in the IMPORT-ONLY "review" bucket forever, indistinguishable from real gaps.
    # A number that never moves is a blind spot, not a baseline.
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("_ra", os.path.join(root, "tools", "reachability_audit.py"))
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)

    home = open(os.path.join(root, "holographic", "misc", "holographic_transformhome.py"),
                encoding="utf-8", errors="replace").read()
    assert ra._is_facade("holographic_transformhome", home) is True
    assert ra._is_facade("holographic_grouptower", "some docstring") is False   # naming alone is not the declaration
    assert ra._is_facade("holographic_fakehome", "no marker here") is False     # ... it must SAY it is a facade

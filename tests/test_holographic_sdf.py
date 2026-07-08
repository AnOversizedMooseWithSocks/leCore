"""Tests for S1 the SDF/shader algebra (holographic_sdf): a Cartesian 3D signed-distance expression tree
that evaluates (-> mesh via the marching bridge), represents itself as a holographic recipe, round-trips a
compact DSL, and emits a complete Shadertoy GLSL shader. The Menger sponge is a first-class fractal primitive."""

import numpy as np

from holographic.mesh_and_geometry.holographic_sdf import sphere, box, torus, menger, SDF, parse_dsl, to_callable, node_kinds, _selftest


def test_primitive_distances():
    assert abs(sphere(1.0).eval([[2, 0, 0]])[0] - 1.0) < 1e-9
    assert abs(sphere(1.0).eval([[0, 0, 0]])[0] + 1.0) < 1e-9
    assert abs(box(1, 1, 1).eval([[2, 0, 0]])[0] - 1.0) < 1e-9
    assert abs(torus(1.0, 0.25).eval([[1.0, 0.0, 0.0]])[0] + 0.25) < 1e-9


def test_csg_union_is_min():
    a = sphere(1.0); c = sphere(1.0).translate([1.5, 0, 0])
    P = [[0.75, 0, 0]]
    assert abs(a.union(c).eval(P)[0] - min(a.eval(P)[0], c.eval(P)[0])) < 1e-12


def test_smooth_union_is_creaseless():
    a = sphere(1.0); c = sphere(1.0).translate([1.5, 0, 0])
    P = np.hstack([np.linspace(0, 1.5, 60)[:, None], np.zeros((60, 2))])
    hard = float(np.max(np.abs(np.diff(SDF("union", (), [a, c]).eval(P), 2))))
    soft = float(np.max(np.abs(np.diff(a.smooth_union(c, 0.4).eval(P), 2))))
    assert soft < hard


def test_domain_repetition_tiles():
    rep = sphere(0.3).repeat([2.0, 0, 0])
    assert abs(rep.eval([[0.4, 0, 0]])[0] - rep.eval([[2.4, 0, 0]])[0]) < 1e-9


def test_renders_to_watertight_mesh():
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    vals, axes = sample_field(to_callable(sphere(0.6)), ((-1, -1, -1), (1, 1, 1)), 24)
    mesh = marching_tetrahedra_vec(vals, axes, 0.0)
    assert mesh.n_faces > 0 and mesh.is_manifold()


def test_dsl_roundtrip():
    tree = sphere(1.0).smooth_union(box(0.5, 0.5, 0.5).translate([1, 0, 0]), 0.3).rounded(0.05)
    back = parse_dsl(tree.to_dsl())
    Q = np.random.default_rng(0).uniform(-2, 2, (50, 3))
    assert np.allclose(tree.eval(Q), back.eval(Q), atol=1e-9)


def test_holographic_recipe():
    from holographic.misc.holographic_typed import tree_to_recipe, op_kinds
    tree = sphere(1.0).union(torus(0.8, 0.2))
    rec = tree_to_recipe(512, 0, tree.to_tree())
    assert rec is not None and len(op_kinds(rec)) > 0


def test_glsl_emit_is_complete_and_roundtrips_dsl():
    tree = sphere(1.0).smooth_union(torus(0.8, 0.2), 0.3)
    glsl = tree.to_glsl()
    assert "float map(vec3 p)" in glsl and "mainImage" in glsl and "opSmin" in glsl
    assert tree.to_dsl() in glsl                       # the shader carries its own DSL


def test_menger_fractal():
    spng = menger(3, 1.0)
    assert spng.eval([[0.0, 0.0, 0.0]])[0] > 0          # central cross carved out
    assert "for(int m=0;m<3;m++)" in spng.to_glsl()
    assert "menger" in node_kinds(spng)


def test_selftest_runs():
    _selftest()

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


# ---------------------------------------------------------------------------
# make_shape / dsl_grammar -- the reach half (J-3D-13/14)
# ---------------------------------------------------------------------------

def test_make_shape_transform_order_spins_in_place():
    """THE assertion for this faculty. scale -> rotate -> translate. Rotating AFTER translating swings the
    object around the world ORIGIN instead of spinning it where it stands -- it reads as "my object jumped
    somewhere else", and a single still frame cannot tell you which of the two happened."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_sdf import make_sdf_shape
    bar = make_sdf_shape("box", bx=1.0, by=0.1, bz=0.1, position=(3.0, 0.0, 0.0), rotate=(0, 0, 1, np.pi / 2))
    assert float(bar.eval(np.array([[3.0, 0.0, 0.0]]))[0]) < 0.0, "must still be centred at (3,0,0)"
    assert float(bar.eval(np.array([[3.0, 0.9, 0.0]]))[0]) < 0.0, "after a 90deg z-turn it extends along y"
    assert float(bar.eval(np.array([[3.9, 0.0, 0.0]]))[0]) > 0.0, "...and no longer along x"


def test_every_shape_kind_builds_and_wrong_guesses_teach():
    from holographic.mesh_and_geometry.holographic_sdf import make_sdf_shape, SHAPE_KINDS, SDF
    for k in SHAPE_KINDS:
        assert isinstance(make_sdf_shape(k), SDF), "kind %r did not build" % k
    try:
        make_sdf_shape("blob")
        raise AssertionError("an unknown kind must raise, not silently pick a default")
    except KeyError as exc:
        assert "sphere" in str(exc), "the error must TEACH the vocabulary"
    try:
        make_sdf_shape("sphere", bx=1.0)                     # right kind, wrong parameter name
        raise AssertionError("a wrong parameter must raise rather than be silently dropped")
    except TypeError as exc:
        assert "'r'" in str(exc) or "['r']" in str(exc), "the error must name the parameters that DO apply"


def test_grammar_matches_the_parser_it_documents():
    """A grammar describing a node set the parser does not implement is worse than no grammar: it sends the
    reader confidently down a path that raises."""
    from holographic.mesh_and_geometry.holographic_sdf import dsl_grammar, parse_dsl, ARITY
    g = dsl_grammar()
    assert {r["kind"] for r in g["nodes"]} == set(ARITY), "grammar and parser disagree on the node set"
    assert all(r["does"] for r in g["nodes"]), "every node needs a plain-language line or the table is a cipher"
    assert parse_dsl(g["example"]) is not None, "the grammar's own example must parse"


def test_authoring_loop_closes_through_the_mind():
    """Cross-faculty, and the point of the whole arc: build geometry, put it in the document, read the
    document back, and light it -- with nothing imported past lecore."""
    import lecore
    import numpy as np
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    sc.add(name="floor", geometry=m.shape("floor", h=0.0), material="matte_gray", transform=np.eye(4))
    sc.add(name="ball", geometry=m.shape("ball", r=0.6, position=(-1.0, 0.6, 0.0)),
           material="copper", transform=np.eye(4))
    info = m.scene_info(sc)
    names = {o["name"] for o in info["objects"]}
    assert names == {"floor", "ball"}, "scene_info must report what was just added: %s" % names
    assert m.scene_light("dome") is not None
    assert "3-D primitive" in str(m.find_capability("make a sphere")[0])


def test_sdf_to_device_bridge_is_real():
    """W1, the merge-integration gap: sdf_dialect emitted WGSL that nothing dispatched while wgpurun
    dispatched WGSL that nothing emitted. Pins the bridge WITHOUT needing an adapter -- the shader is
    inspectable text, the CPU reference is analytically checkable, and the emitted map() is proven against
    Python through the C dialect (the same executable bar sdfemit uses, because WGSL cannot run here)."""
    import numpy as np
    import lecore
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    from holographic.mesh_and_geometry.holographic_sdfemit import validate_c
    m = lecore.UnifiedMind(dim=64, seed=0)
    tree = sphere(1.0)

    src = m.sdf_trace_shader(tree, 32, 24, steps=96)
    assert "fn map(" in src and "fn sdf_depth(" in src
    assert "for (var s: i32 = 0; s < 96;" in src, "the trace must be a BOUNDED loop -- a shader needs a static trip count"
    assert "%(" not in src and "%%" not in src, "unexpanded format tokens would ship a broken shader"

    # the CPU reference is checkable against analytic truth: a unit sphere at z=3 is 2.0 away on the axis
    d = m.sdf_depth_cpu(tree, 33, 25, eye=(0.0, 0.0, 3.0))
    assert d.shape == (25, 33)
    assert abs(float(d[12, 16]) - 2.0) < 5e-3, float(d[12, 16])
    assert float(d[0, 0]) == -1.0 and float(d[-1, -1]) == -1.0, "corner rays must MISS and say so"

    # the emitted map() itself is bit-identical to Python where it CAN be executed
    r = validate_c(tree, np.random.default_rng(0).uniform(-2, 2, (128, 3)), dialect="c_f64")
    assert r["bit_identical"], r

    # the device path REFUSES rather than pretending, when there is no adapter
    from holographic.io_and_interop.holographic_wgpurun import available
    if not available():
        try:
            m.sdf_depth_device(tree, 8, 8)
            assert False, "must raise without an adapter"
        except ImportError:
            pass
    else:
        assert m.sdf_depth_agrees(tree, 24, 18)["agrees"]


def test_sdf_trace_consults_the_placement_layer():
    """W2/W3: the render arc never asked the placement layer, so the one path that pays for a device could
    not ask. Pins the workload arithmetic (the part a caller gets wrong) and the measured asymmetry that
    makes this the ONLY render path worth offloading."""
    import lecore
    from holographic.io_and_interop.holographic_gpureport import (MIN_BYTES_PROVISIONAL,
                                                                  MIN_INTENSITY_PROVISIONAL)
    m = lecore.UnifiedMind(dim=64, seed=0)

    w = m.sdf_trace_workload(512, 384, steps=96)
    assert w["n_bytes"] == 512 * 384 * 4 * 2, "bytes MOVED (one f32 in, one f32 out), not bytes touched"
    # intensity is resolution-INDEPENDENT: both terms scale with the pixel count
    assert w["flops_per_byte"] == m.sdf_trace_workload(64, 64, steps=96)["flops_per_byte"]
    assert w["n_bytes"] >= MIN_BYTES_PROVISIONAL
    assert w["flops_per_byte"] > 30 * MIN_INTENSITY_PROVISIONAL, "the trace should clear the bar by a wide margin"

    # halving the steps halves the intensity -- the verdict turns on march depth, not on resolution
    assert abs(m.sdf_trace_workload(512, 384, steps=48)["flops_per_byte"]
               - w["flops_per_byte"] / 2.0) < 1e-9

    r = m.sdf_trace_placement(512, 384)
    assert r["placement"] in ("cpu", "device", "unit", "pool")
    assert r["workload"]["n_bytes"] == w["n_bytes"], "the verdict must carry the numbers that produced it"
    assert "device" in r["considered"]

    # THE KEPT NEGATIVE, pinned: an elementwise postfx pass is transfer-bound and must NOT clear the bar.
    # 1920x1080 RGB in+out, ~6 flops/pixel-channel -> 0.8 flops/byte against a 4.0 bar.
    nb = 1920 * 1080 * 3 * 4 * 2
    assert (1920 * 1080 * 3 * 6) / nb < MIN_INTENSITY_PROVISIONAL, \
        "an elementwise image pass is transfer-bound; wiring a backend= into postfx would not pay"

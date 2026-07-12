"""Cross-faculty integration test for the geometry-kernel arc (K1/K2/K3/K4/K7/K8 + grouping cycle guard).

The unit selftests prove each module in isolation; this proves the FACULTIES compose through a live UnifiedMind the
way a modeling-app backend would actually call them -- the "lands with a cross-faculty integration test" rule, whose
hard lesson on record is that a shared kernel is not a shared manifold.
"""
import numpy as np

import pytest

import lecore


def _mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_k2_ssi_traces_the_sphere_sphere_circle():
    m = _mind()
    def sA(P): P = np.asarray(P, float); return np.linalg.norm(P, axis=1) - 1.0
    def sB(P): P = np.asarray(P, float); return np.linalg.norm(P - np.array([1., 0, 0]), axis=1) - 1.0
    curves = m.surface_intersect(sA, sB, lo=(-1.5, -1.5, -1.5), hi=(2, 1.5, 1.5), res=20)
    assert curves, "sphere-sphere should intersect"
    circle = max(curves, key=len)
    assert len(circle) > 20
    assert abs(float(circle[:, 0].mean()) - 0.5) < 1e-3        # the circle sits in x=0.5
    assert np.max(np.abs(sA(circle))) < 1e-5 and np.max(np.abs(sB(circle))) < 1e-5


def test_k1_k4_offset_then_k7_dxf_export():
    m = _mind()
    sq = np.array([[0., 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    off = m.offset_curve(sq, 0.1, closed=True)                 # grows outward -> area (1.2)^2 = 1.44
    x = off[:, 0]; y = off[:, 1]
    area = abs(0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))
    assert abs(area - 1.44) < 0.03, area
    dxf = m.polylines_to_dxf([off])
    assert "POLYLINE" in dxf and dxf.strip().endswith("EOF")


def test_k8_sketch_solves_then_k7_stl_export():
    m = _mind()
    s = m.sketch2d()
    a = s.add_point(0.1, -0.2); b = s.add_point(3, 0.3); c = s.add_point(2.7, 2.1); d = s.add_point(-0.2, 1.8)
    s.fix(a); s.horizontal(a, b); s.horizontal(d, c); s.vertical(a, d); s.vertical(b, c)
    s.distance(a, b, 4.0); s.distance(a, d, 2.0)
    assert s.solve()["satisfied"]
    assert abs(np.linalg.norm(s.pts[b] - s.pts[a]) - 4.0) < 1e-6
    verts = np.array([[*s.pts[a], 0], [*s.pts[b], 0], [*s.pts[c], 0], [*s.pts[d], 0]])
    stl = m.mesh_to_stl(verts, [(0, 1, 2, 3)])
    assert stl.count("facet normal") == 2                      # a quad -> two facets


def test_grouping_cycle_guard_holds_through_the_scene():
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.misc import holographic_grouping as G
    sc = Scene()
    o1 = sc.add(name="o1"); o2 = sc.add(name="o2")
    g = G.group_objects(sc, [o1, o2], "grp")
    # o1 is a member of g, so parenting g under o1 would cycle -> must be refused
    try:
        sc.set_parent(g, o1)
        assert False, "cycle should have been refused"
    except ValueError:
        pass


def test_predicate_faculties_are_exact():
    m = _mind()
    assert m.orient2d((0, 0), (1, 0), (0, 1)) == 1
    assert m.orient2d((0, 0), (1, 1), (2, 2)) == 0             # exactly collinear
    assert m.orient2d((0, 0), (1, 1), (2, 2 + 1e-11)) == 1     # a hair off -> not swallowed


def test_k9_surface_curvature_matches_analytic():
    m = _mind()
    R = 2.0
    def sphere(u, v):
        return np.array([R * np.cos(u) * np.sin(v), R * np.sin(u) * np.sin(v), R * np.cos(v)])
    c = m.surface_curvature(sphere, 0.7, 1.0)
    assert abs(c["gaussian"] - 1.0 / R ** 2) < 1e-3           # sphere K = 1/R^2
    assert abs(abs(c["k1"]) - 1.0 / R) < 5e-3
    # a vertical cylinder wall drafts at ~0 deg for a +z pull
    def cyl(u, v):
        return np.array([R * np.cos(u), R * np.sin(u), v])
    assert abs(m.draft_angle(cyl, 1.0, 0.5, pull_dir=(0, 0, 1))) < 1.0


def test_k10_osnaps():
    m = _mind()
    V = np.array([[0., 0, 0], [5, 0, 0]])
    mp = m.snap_to_midpoints([2.4, 0.2, 0], V, [[0, 1]])
    assert abs(mp["position"][0] - 2.5) < 1e-6
    A = np.array([[-1., 0], [1, 0]]); B = np.array([[0, -1.], [0, 1]])
    xi = m.snap_to_intersections([0.1, 0.1], [A, B])
    assert abs(xi["position"][0]) < 1e-9 and abs(xi["position"][1]) < 1e-9


def test_k5_fillet_is_exact_radius_and_local():
    m = _mind()
    px = lambda P: np.asarray(P, float)[:, 0]
    py = lambda P: np.asarray(P, float)[:, 1]
    r = 0.3
    f = m.fillet_union(px, py, r)
    # a point on the expected arc (0, r) is on the zero-set
    assert abs(float(f(np.array([[0.0, r, 0.0]]))[0])) < 1e-9
    # local: away from the crease it equals the sharp union
    far = np.array([[5.0, 0.1, 0.0]])
    assert abs(float(f(far)[0]) - float(min(px(far)[0], py(far)[0]))) < 1e-9


def test_k6_brep_cube_is_valid_genus0_solid():
    m = _mind()
    rep = m.brep_validate(m.brep_box())
    assert rep["closed_manifold"] and rep["euler_ok"]
    assert (rep["V"], rep["E"], rep["F"]) == (8, 12, 6)
    assert rep["genus"] == 0


def test_brep_membership_and_boolean_classification():
    m = _mind()
    cube = m.brep_box(lo=(-1, -1, -1), hi=(1, 1, 1))
    inside = m.point_in_brep(cube, np.array([[0., 0, 0], [3., 0, 0]]))
    assert bool(inside[0]) and not bool(inside[1])
    A = m.brep_box(lo=(-1, -1, -1), hi=(1, 1, 1))
    B = m.brep_box(lo=(0.3, -2, -2), hi=(3, 2, 2))
    res = m.brep_boolean_faces(A, B, "difference")
    assert len(res["straddle"]) >= 1          # boundary-crossing faces flagged for an SSI split


def test_nodegraph_drives_real_sdf_compute_typed_and_serializable():
    m = _mind()
    g = m.node_graph()
    s = g.add("sdf_sphere", {"radius": 1.0})
    b = g.add("sdf_box", {"size": (0.8, 0.8, 0.8)})
    u = g.add("sdf_union")
    g.connect(s, "out", u, "a")
    g.connect(b, "out", u, "b")
    from holographic.mesh_and_geometry.holographic_sdf import as_eval
    ev = as_eval(g.evaluate(u)["out"])
    assert float(ev(np.array([[0.0, 0, 0]]))[0]) < 0        # inside the union
    assert float(ev(np.array([[5.0, 0, 0]]))[0]) > 0        # outside
    # type check: wiring a texture into an sdf input is refused
    tc = g.add("texture_const", {"color": (1, 0, 0)})
    try:
        g.connect(tc, "out", u, "a"); assert False
    except TypeError:
        pass
    # serialize round-trip evaluates the same
    import json
    from holographic.scene_and_pipeline.holographic_nodegraph import NodeGraph
    g2 = NodeGraph.from_dict(g.reg, json.loads(json.dumps(g.to_dict())))
    ev2 = as_eval(g2.evaluate(u)["out"])
    Q = np.array([[0.9, 0.0, 0.0]])
    assert abs(float(ev2(Q)[0]) - float(ev(Q)[0])) < 1e-9
    # expanded palette: a transform + smooth_union + bake node all evaluate through the shell
    tr = g.add("sdf_translate", {"t": (1.0, 0.0, 0.0)})
    g.connect(s, "out", tr, "a")
    assert float(as_eval(g.evaluate(tr)["out"])(np.array([[1.0, 0, 0]]))[0]) < 0   # sphere moved to x=1
    assert len(g.reg.types) >= 18
    # end-to-end: the graph terminates in renderable geometry (CSG -> mesh -> +material)
    mn = g.add("sdf_to_mesh", {"lo": (-2, -2, -2), "hi": (2, 2, 2), "res": 32})
    g.connect(u, "out", mn, "a")
    mesh = g.evaluate(mn)["out"]
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
    mat = g.add("material_lib", {"name": "matte_gray"})
    asg = g.add("assign_material")
    g.connect(mn, "out", asg, "mesh"); g.connect(mat, "out", asg, "material")
    r = g.evaluate(asg)["out"]
    assert "mesh" in r and "material" in r
    # geometry-modifier nodes: subdivide adds detail, smooth preserves topology, decimate removes faces (the
    # node uses the fast O(F log F) heap QEM, so this is quick now)
    sub = g.add("mesh_subdivide", {"levels": 1})
    g.connect(mn, "out", sub, "mesh")
    assert len(g.evaluate(sub)["out"].vertices) > len(mesh.vertices)
    sm = g.add("mesh_smooth", {"lam": 0.5, "iters": 2})
    g.connect(mn, "out", sm, "mesh")
    assert len(g.evaluate(sm)["out"].vertices) == len(mesh.vertices)
    dec = g.add("mesh_decimate", {"ratio": 0.5})   # fast=True in the node -> heap decimator
    g.connect(mn, "out", dec, "mesh")
    assert len(g.evaluate(dec)["out"].faces) < len(mesh.faces)
    # material SOCKET graph: color + scalar nodes drive a PBR material's channels (shader-editor pattern)
    col = g.add("color", {"rgba": (0.2, 0.5, 0.9, 1.0)})
    rough = g.add("scalar", {"value": 0.25})
    mpbr = g.add("material_pbr", {"name": "blue"})
    g.connect(col, "out", mpbr, "base_color")
    g.connect(rough, "out", mpbr, "roughness")
    pbr = g.evaluate(mpbr)["out"]
    assert abs(pbr.roughness - 0.25) < 1e-9 and tuple(pbr.base_color)[:3] == (0.2, 0.5, 0.9)
    try:
        g.connect(rough, "out", mpbr, "base_color"); assert False   # scalar cannot drive a color socket
    except TypeError:
        pass
    # texture-driven material: a texture drives the albedo -> a callable socket f(points)->rgb
    tr = g.add("texture_const", {"color": (1.0, 0.0, 0.0)})
    tb = g.add("texture_const", {"color": (0.0, 0.0, 1.0)})
    tm = g.add("texture_mix", {"t": 0.5})
    g.connect(tr, "out", tm, "a"); g.connect(tb, "out", tm, "b")
    mtex = g.add("material_textured", {"name": "purple"})
    g.connect(tm, "out", mtex, "tex")
    pm = g.evaluate(mtex)["out"]
    assert pm["textured"] and callable(pm["albedo"])
    c = pm["albedo"](np.array([[0.0, 0, 0]]))
    assert c.shape == (1, 3) and 0.4 < c[0][0] < 0.6 and c[0][1] < 0.1


def test_nodegraph_audio_and_shader_drive_generically():
    m = _mind()
    from holographic.mesh_and_geometry.holographic_sdf import as_eval
    g = m.node_graph()
    clip = g.add("audio_clip", {"freq": 220.0, "dur": 1.0, "tremolo": 4.0})
    band = g.add("audio_band", {"band": 0, "lo": 0.4, "hi": 1.6})
    g.connect(clip, "out", band, "signal")
    sph = g.add("sdf_sphere", {"radius": 1.0})
    g.connect(band, "out", sph, "radius")                     # audio scalar drives a drivable param
    v0 = float(as_eval(g.evaluate(sph, t=0.0)["out"])(np.array([[0.0, 0, 0]]))[0])
    v1 = float(as_eval(g.evaluate(sph, t=0.5)["out"])(np.array([[0.0, 0, 0]]))[0])
    assert abs(v0 - v1) > 1e-3                                 # time advances -> audio-reactive geometry
    # a shader field animates with time
    sf = g.add("shader_field", {"freq": (2.0, 0.0, 0.0), "speed": 3.0})
    P = np.array([[0.3, 0.0, 0.0]])
    assert abs(float(g.evaluate(sf, t=0.0)["out"](P)[0]) - float(g.evaluate(sf, t=0.5)["out"](P)[0])) > 1e-3
    # type safety: an sdf cannot drive a scalar param
    sp2 = g.add("sdf_sphere")
    try:
        g.connect(sph, "out", sp2, "radius"); assert False
    except TypeError:
        pass




@pytest.mark.slow  # the finished B-rep boolean routes two solids through SDF marching + analytic re-stitch (~30s),
                   # over the 15s per-test budget. Deselected by default; runs under --run-slow. (Verified: passes.)
def test_k6_brep_boolean_is_watertight_and_volume_correct():
    m = _mind()
    a = m.brep_box(lo=(-1, -1, -1), hi=(1, 1, 1))     # vol 8
    b = m.brep_box(lo=(0, 0, 0), hi=(2, 2, 2))         # vol 8, overlap [0,1]^3 vol 1
    bnds = ((-1.5, -1.5, -1.5), (2.5, 2.5, 2.5))
    for op, expected in (("union", 15.0), ("intersection", 1.0), ("difference", 7.0)):
        r = m.brep_boolean(a, b, op, res=56, bounds=bnds)
        assert r._boolean_report["closed_manifold"], (op, "not watertight")
        assert abs(abs(r._boolean_report["volume"]) - expected) < 0.6, (op, r._boolean_report["volume"])
        # analytic re-stitch: polygonal faces recovered, same volume, far fewer faces
        ra = m.brep_boolean(a, b, op, res=56, bounds=bnds, analytic=True)
        assert abs(abs(ra._boolean_report["volume"]) - expected) < 0.6
        assert ra._boolean_report["n_faces"] < r._boolean_report["n_faces"] // 10
        assert any(len(f.outer) > 3 for f in ra.faces)   # real polygons, not triangles


if __name__ == "__main__":
    test_k2_ssi_traces_the_sphere_sphere_circle()
    test_k1_k4_offset_then_k7_dxf_export()
    test_k8_sketch_solves_then_k7_stl_export()
    test_grouping_cycle_guard_holds_through_the_scene()
    test_predicate_faculties_are_exact()
    test_k9_surface_curvature_matches_analytic()
    test_k10_osnaps()
    test_k5_fillet_is_exact_radius_and_local()
    test_k6_brep_cube_is_valid_genus0_solid()
    test_k6_brep_boolean_is_watertight_and_volume_correct()
    test_nodegraph_drives_real_sdf_compute_typed_and_serializable()
    test_nodegraph_audio_and_shader_drive_generically()
    test_brep_membership_and_boolean_classification()
    print("geometry-kernel integration OK: SSI circle; offset->DXF; sketch->STL; grouping cycle guard; exact "
          "predicates; surface curvature+draft; osnaps; exact-radius fillet")

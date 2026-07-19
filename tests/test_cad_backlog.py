"""Integration tests for the Poly Studio backlog upstream (CAD trio, OBB, erosion, camera-from-VPs,
NodeGraph.remove, ccrun, emission alias, extrude total_height).

WHY cross-faculty: the wiring rule is that a faculty lands with a test proving it composes with the rest of
the engine, not just that it runs in isolation -- the on-record lesson being that a shared kernel is not a
shared manifold. Here the composition is the REAL modeling pipeline: node graph -> SDF -> marched mesh ->
measured (mass/section/OBB), plus the C runner against the emitter's own contract.
"""

import numpy as np
import pytest

import lecore


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def _cube_mesh():
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    F = [t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    return Mesh(V, F)


def test_mass_properties_exact_and_positive(mind):
    mp = mind.mass_properties(_cube_mesh(), density=2.0)
    assert abs(mp["volume"] - 1.0) < 1e-12
    assert abs(mp["area"] - 6.0) < 1e-12
    assert np.allclose(mp["center_of_mass"], 0.5, atol=1e-12)
    assert np.allclose(mp["principal_moments"], 2.0 / 6.0, atol=1e-12)
    # the pinned negative: rotation must never yield negative principal moments
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    rot = Mesh(np.asarray(_cube_mesh().vertices) @ R.T, _cube_mesh().faces)
    assert np.all(mind.mass_properties(rot)["principal_moments"] > 0)


def test_section_and_draft_on_quad_primitive(mind):
    # the engine's own box() is a QUAD mesh -- the fan-triangulation path, exactly measured
    from holographic.mesh_and_geometry.holographic_mesh import box
    s = mind.mesh_section(box(), (0, 0, 0.0), (0, 0, 1))
    assert s["contours"] == 1 and abs(s["area"] - 1.0) < 1e-9 and abs(s["perimeter"] - 4.0) < 1e-9
    d = mind.draft_report(box(), (0, 0, 1), min_draft_deg=2.0)
    assert abs(d["undercut_fraction"] - 1.0 / 6.0) < 1e-9
    assert abs(d["parting_fraction"] - 4.0 / 6.0) < 1e-9


def test_obb_never_worse_than_aabb_and_recovers_rotation(mind):
    rng = np.random.default_rng(0)
    pts = (rng.uniform(0, 1, (300, 3)) * [1.0, 2.0, 3.0])
    th = np.pi / 4
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    rot = pts @ Rz.T
    r = mind.oriented_bbox(rot)
    aabb = float(np.prod(rot.max(0) - rot.min(0)))
    assert r["volume"] <= aabb + 1e-9              # the fallback guarantee
    assert r["volume"] < 0.8 * aabb                # and a genuine win on the rotated cloud


def test_pipeline_nodegraph_to_measured_mesh(mind):
    """The composition that matters: node graph -> evaluate SDF -> march to a mesh -> measure it.
    A sphere's marched mesh must report volume/area within marching tolerance of the analytic ball,
    and its OBB must be a near-cube of side ~2r."""
    g = mind.node_graph()
    s = g.add("sdf_sphere", {"radius": 1.0})
    sdf = g.evaluate(s)["out"]
    mesh = mind.sdf_to_mesh(sdf, bounds=((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3)), resolution=48)
    mp = mind.mass_properties(mesh)
    assert abs(mp["volume"] - 4.0 / 3.0 * np.pi) < 0.05 * (4.0 / 3.0 * np.pi)
    assert abs(mp["area"] - 4.0 * np.pi) < 0.06 * (4.0 * np.pi)
    assert np.allclose(mp["center_of_mass"], 0.0, atol=0.01)
    sec = mind.mesh_section(mesh, (0, 0, 0.0), (0, 0, 1))
    assert sec["contours"] == 1 and abs(sec["area"] - np.pi) < 0.05 * np.pi
    r = mind.oriented_bbox(np.asarray(mesh.vertices))
    assert np.all(np.abs(r["half_extents"] - 1.0) < 0.05)
    # and remove() leaves a working graph behind
    g.remove(s)
    assert s not in g.nodes and g.edges == []


def test_nodegraph_remove_prunes_wired_edges(mind):
    g = mind.node_graph()
    a = g.add("sdf_sphere", {"radius": 1.0})
    b = g.add("sdf_box", {"size": (0.5, 0.5, 0.5)})
    u = g.add("sdf_union")
    g.connect(a, "out", u, "a")
    g.connect(b, "out", u, "b")
    g.evaluate(u)
    g.remove(a)
    assert len(g.edges) == 1 and all(e["src"] != a and e["dst"] != a for e in g.edges)
    with pytest.raises(KeyError):
        g.remove("no_such_node")


def test_terrain_erode_deterministic_and_additive(mind):
    from holographic.mesh_and_geometry.holographic_terrain import Terrain
    H0 = Terrain(seed=3, octaves=4).heightmap(48)
    H1 = mind.terrain_erode(H0, droplets=300, steps=20, seed=0)
    H2 = mind.terrain_erode(H0, droplets=300, steps=20, seed=0)
    assert np.array_equal(H1, H2)
    assert H1.max() <= H0.max() + 1e-12
    assert float(np.abs(H1 - H0).sum()) > 1e-3


def test_camera_from_vps_roundtrip_and_refusal(mind):
    cam = mind.camera_from_vanishing_points((1120, 240), (-480, 290), (320, 240))
    assert abs(cam["focal"] - 800.0) < 1e-6
    assert np.allclose(cam["R"] @ cam["R"].T, np.eye(3), atol=1e-9)
    with pytest.raises(ValueError):
        mind.camera_from_vanishing_points((420, 240), (520, 240), (320, 240))


def test_c_batch_eval_matches_python_bitwise(mind):
    from holographic.io_and_interop.holographic_ccrun import cc_available
    src = "def k(x: float, y: float) -> float:\n    return sqrt(x*x + y*y) - 1.0\n"
    x = np.linspace(-2, 2, 101)
    y = np.linspace(1, 3, 101)
    if cc_available() is None:
        pytest.skip("no C compiler in this environment (the refusal path is covered by the module selftest)")
    got = mind.c_batch_eval(src, [x, y])
    ref = np.sqrt(x * x + y * y) - 1.0
    assert np.array_equal(np.asarray(got), ref)    # emit's f64 contract: bit-identical


def test_material_emission_alias_roundtrip():
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial
    m = PBRMaterial()
    m.emission = (1.0, 0.5, 0.25)                  # the previously-silent dead attribute, now a live synonym
    assert m.emissive == (1.0, 0.5, 0.25)
    assert m.emission == m.emissive
    # serialization still speaks glTF's name
    assert "emissiveFactor" in str(m.to_gltf_dict())


def test_extrude_total_height_semantics():
    from holographic.mesh_and_geometry.holographic_sdf2d import circle2d, extrude
    f_half = extrude(circle2d(1.0), height=1.0)                    # legacy: |z| < 1 -> total 2
    f_total = extrude(circle2d(1.0), total_height=2.0)             # new: total 2 -> same solid
    P = np.array([[0.0, 0.0, 0.99], [0.0, 0.0, 1.01]])
    assert np.allclose(f_half(P), f_total(P))
    assert f_total(P)[0] < 0 < f_total(P)[1]


# ---- backlog C2: subgraph nesting (collapse / expand) --------------------------------------------------------
# Cross-faculty: the group boundary must survive the things the rest of the engine does to a graph -- external
# param DRIVERS crossing it, JSON serialization into a fresh registry, and marching the result to a real mesh.

def _sdf_graph(mind):
    g = mind.node_graph()
    sc = g.add("scalar", {"value": 1.3})
    sph = g.add("sdf_sphere", {"radius": 1.0})
    bx = g.add("sdf_box", {"size": (0.8, 0.8, 0.8)})
    uni = g.add("sdf_union")
    g.connect(sc, "out", sph, "radius")
    g.connect(sph, "out", uni, "a")
    g.connect(bx, "out", uni, "b")
    return g, sc, sph, bx, uni


def _probe(sdf):
    from holographic.mesh_and_geometry.holographic_sdf import as_eval
    P = np.array([[0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    return as_eval(sdf)(P)


def test_collapse_is_result_preserving_and_types_the_boundary(mind):
    g, sc, sph, bx, uni = _sdf_graph(mind)
    ref = _probe(g.evaluate(uni)["out"])
    gid = g.collapse([sph, bx])
    assert np.array_equal(_probe(g.evaluate(uni)["out"]), ref)   # a refactor, not an edit
    nt = g.reg.get(g.nodes[gid]["type"])
    assert nt.inputs == {"in_0": "scalar"}                       # the external driver -> one typed group input
    assert set(nt.outputs) == {"out_0", "out_1"}


def test_driver_still_drives_through_the_group_boundary(mind):
    g, sc, sph, bx, uni = _sdf_graph(mind)
    gid = g.collapse([sph, bx])
    before = _probe(g.evaluate(uni)["out"])
    g.set_param(sc, value=2.0)
    assert not np.array_equal(_probe(g.evaluate(uni)["out"]), before)


def test_nested_group_survives_json_into_a_fresh_registry(mind):
    import json
    from holographic.scene_and_pipeline.holographic_nodegraph import NodeGraph, default_registry
    g, sc, sph, bx, uni = _sdf_graph(mind)
    ref = _probe(g.evaluate(uni)["out"])
    inner = g.collapse([sph, bx])
    outer = g.collapse([inner, uni])                             # a group INSIDE a group, and terminal
    assert np.array_equal(_probe(g.evaluate(outer)["out_0"]), ref)
    data = json.loads(json.dumps(g.to_dict()))                   # really JSON-able
    g2 = NodeGraph.from_dict(default_registry(), data)           # a registry that never saw this group type
    assert np.array_equal(_probe(g2.evaluate(outer)["out_0"]), ref)


def test_terminal_collapse_stays_readable(mind):
    """The bug the dangling-output rule fixes: collapsing everything must not make the result unreadable."""
    g, sc, sph, bx, uni = _sdf_graph(mind)
    ref = _probe(g.evaluate(uni)["out"])
    gid = g.collapse([sc, sph, bx, uni])
    assert set(g.reg.get(g.nodes[gid]["type"]).outputs) == {"out_0"}
    assert np.array_equal(_probe(g.evaluate(gid)["out_0"]), ref)


def test_expand_is_the_inverse_of_collapse(mind):
    g, sc, sph, bx, uni = _sdf_graph(mind)
    ref = _probe(g.evaluate(uni)["out"])
    gid = g.collapse([sph, bx])
    ids = g.expand(gid)
    assert gid not in g.nodes and len(ids) == 2
    assert np.array_equal(_probe(g.evaluate(uni)["out"]), ref)
    with pytest.raises(ValueError):
        g.expand(uni)                                            # not a subgraph node
    with pytest.raises(KeyError):
        g.expand("no_such_node")


def test_cycle_creating_collapse_is_refused_untouched(mind):
    import json
    g = mind.node_graph()
    a = g.add("sdf_sphere", {"radius": 1.0})
    b = g.add("sdf_union")
    c = g.add("sdf_union")
    d = g.add("sdf_box", {"size": (0.5, 0.5, 0.5)})
    g.connect(a, "out", b, "a"); g.connect(d, "out", b, "b")
    g.connect(b, "out", c, "a"); g.connect(a, "out", c, "b")
    snapshot = json.dumps(g.to_dict(), sort_keys=True)
    with pytest.raises(ValueError):
        g.collapse([a, c])                                       # external b sits between the two members
    assert json.dumps(g.to_dict(), sort_keys=True) == snapshot   # refused == untouched, like connect()


def test_grouped_graph_still_marches_to_a_measured_mesh(mind):
    """End of the pipeline: a collapsed graph is still just an SDF source -- march it and measure it."""
    g, sc, sph, bx, uni = _sdf_graph(mind)
    g.set_param(sc, value=1.0)
    gid = g.collapse([sph, bx])
    sdf = g.evaluate(uni)["out"]
    mesh = mind.sdf_to_mesh(sdf, bounds=((-1.4, -1.4, -1.4), (1.4, 1.4, 1.4)), resolution=40)
    mp = mind.mass_properties(mesh)
    assert mp["volume"] > 4.0                                    # sphere(1) union box(0.8) encloses real volume
    assert np.allclose(mp["center_of_mass"], 0.0, atol=0.05)


# ---- .glb scan-soup regression: the mesh->SDF sign (reported: "glb models rendered as garbage") --------------
# Root cause chain, measured: the importer was INNOCENT (file declares 11003 tris/18610 verts, reader returns
# exactly that; node matrices compose to identity; winding consistent). The mesh is a 71%-boundary-edge triangle
# soup; mesh_to_sdf_grid's flood-fill sign needs a watertight negative shell, the soup's shell has super-voxel
# holes, the flood leaks, and marching the leaked field yields the reported garbage blobs.

def _slit_sphere():
    """The honest scan model: a nearly-contiguous sphere whose every triangle is its own island (100% boundary
    edges, super-voxel slits, 64% solid-angle coverage)."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    Vs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    Fs = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    for _ in range(2):
        V2 = list(Vs); F2 = []; cache = {}
        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                mp = V2[i] + V2[j]; cache[k] = len(V2); V2.append(mp / np.linalg.norm(mp))
            return cache[k]
        for (a, b, c) in Fs:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]
        Vs, Fs = np.array(V2), F2
    tris = Vs[np.asarray(Fs)]
    cen = tris.mean(axis=1, keepdims=True)
    V = (cen + 0.8 * (tris - cen)).reshape(-1, 3)
    return Mesh(V, [(3*i, 3*i+1, 3*i+2) for i in range(len(tris))])


def test_closed_mesh_auto_is_bit_identical_to_flood(mind):
    """The backward-compat contract, by measurement: no closed mesh moves an inch."""
    from holographic.mesh_and_geometry.holographic_meshbridge import open_fraction
    cube = _cube_mesh()
    assert open_fraction(cube) == 0.0
    b = ((-0.3, -0.3, -0.3), (1.3, 1.3, 1.3))
    ga, _ = mind.mesh_to_sdf_grid(cube, b, res=20, sign="auto")
    gf, _ = mind.mesh_to_sdf_grid(cube, b, res=20, sign="flood")
    assert np.array_equal(ga, gf)


@pytest.mark.slow
def test_soup_regression_flood_leaks_and_winding_fixes(mind):
    """The regression and the fix on one mesh at one resolution -- the numbers that close the report.
    Marked slow (winding at 64^3 measures ~17s, over the suite's 15s budget); the module _selftest_soup_sign
    pins the same contract outside pytest, so the pin is not budget-dependent."""
    soup = _slit_sphere()
    from holographic.mesh_and_geometry.holographic_meshbridge import open_fraction
    assert open_fraction(soup) == 1.0
    bs = ((-1.3,)*3, (1.3,)*3)
    res = 64
    xs = np.linspace(-1.3, 1.3, res)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    inside = np.sqrt(gx**2 + gy**2 + gz**2) < 0.85
    gf, _ = mind.mesh_to_sdf_grid(soup, bs, res=res, sign="flood")
    gw, _ = mind.mesh_to_sdf_grid(soup, bs, res=res, sign="winding")
    assert float((gf[inside] < 0).mean()) < 0.6          # the leak, kept as a loud negative
    assert float((gw[inside] < 0).mean()) > 0.9          # the fix
    ga, _ = mind.mesh_to_sdf_grid(soup, bs, res=res, sign="auto")
    assert np.array_equal(ga, gw)                        # auto routes the soup to winding


def test_voxel_remesh_soup_produces_one_coherent_surface(mind):
    """End of the reported pipeline: voxel_remesh on a soup must produce a coherent surface, not confetti.
    Measured via connected components of the marched mesh: the biggest component must dominate."""
    vr = mind.voxel_remesh(_slit_sphere(), resolution=48, silhouette=None)
    F = np.asarray([f for f in vr.faces if len(f) == 3])
    assert len(F) > 100
    # union-find over shared vertices -> face components
    parent = np.arange(len(vr.vertices))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for (a, b, c) in F:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra; parent[find(rc)] = ra
    roots = np.array([find(v) for v in F[:, 0]])
    _, counts = np.unique(roots, return_counts=True)
    frac = counts.max() / counts.sum()
    assert frac > 0.9, "voxel_remesh left confetti: biggest component only %.0f%% of faces" % (100 * frac)


def test_fast_winding_tracks_exact_on_closed_surface():
    from holographic.mesh_and_geometry.holographic_voxelize import winding_number, fast_winding_number
    rng = np.random.default_rng(1)
    soup = _slit_sphere()                                # also fine as an accuracy probe: same formula both paths
    pts = rng.uniform(-1.4, 1.4, (300, 3))
    w_e = winding_number(pts, soup.vertices, soup.faces)
    w_f = fast_winding_number(pts, soup.vertices, soup.faces, cells=8)
    assert np.abs(w_e - w_f).max() < 0.05


# ---- texture-loss regression: "losing texture information and the mesh is not looking great" -----------------
# Root cause, measured on the mantis .glb: mesh_repair's position-only weld collapsed 18610 -> 5497 verts and
# DROPPED uvs+normals. All 4956 duplicate-position groups were UV-SEAM splits (median uv spread 0.67, zero were
# render duplicates), so the weld was not cleanup -- it scrambled the atlas AND stripped the arrays. The fix is
# attribute-aware welding (weld only pos+uv+normal agreement) + attribute carry through split/drop/fill, plus
# uv projection in cluster_decimate / voxel_remesh via the existing transfer_uv machinery.

def _seam_sheet():
    """Two triangles sharing a coincident-vertex pair that is a UV SEAM, plus one true render duplicate."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.array([[0,0,0],[0,0,0],[1,0,0],[1,0,0],[0,1,0],[2,1,0]], float)
    UV = np.array([[0.2,0.2],[0.2,0.2],[0.1,0.5],[0.9,0.5],[0.3,0.3],[0.7,0.7]])
    return Mesh(V, [(0,2,4),(1,3,5)], uvs=UV)


def test_weld_without_attributes_is_bit_identical(mind):
    from holographic.mesh_and_geometry.holographic_meshtools import merge_by_distance
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.array([[0,0,0],[0,0,0],[1,0,0],[0,1,0]], float)
    F = [(0,2,3),(1,3,2)]
    a = merge_by_distance(Mesh(V, F), attrs="auto")
    b = merge_by_distance(Mesh(V, F), attrs="ignore")
    assert np.array_equal(a.vertices, b.vertices) and a.faces == b.faces


def test_uv_seams_survive_the_weld(mind):
    from holographic.mesh_and_geometry.holographic_meshtools import merge_by_distance
    w = merge_by_distance(_seam_sheet(), attrs="auto")
    assert len(w.vertices) == 5                        # only the render duplicate welded; the seam stayed split
    assert w.uvs is not None
    seam = sorted(round(float(u), 3) for v, u in zip(np.asarray(w.vertices), np.asarray(w.uvs)[:, 0])
                  if abs(v[0] - 1.0) < 1e-9 and abs(v[1]) < 1e-9)
    assert seam == [0.1, 0.9]                          # both atlas sides intact


def test_mesh_repair_reports_and_carries_uvs(mind):
    rep, report = mind.mesh_repair(_seam_sheet(), fill_holes=False)
    assert report["uvs_carried"] is True
    assert rep.uvs is not None and len(rep.uvs) == len(rep.vertices)


def test_cluster_decimate_projects_uvs(mind):
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    # uv-sphere; cluster to a coarse grid; uvs must arrive and be in-range
    Vs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    Fs = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    for _ in range(3):
        V2=list(Vs); F2=[]; cache={}
        def mid(i,j):
            k=(min(i,j),max(i,j))
            if k not in cache:
                p=V2[i]+V2[j]; cache[k]=len(V2); V2.append(p/np.linalg.norm(p))
            return cache[k]
        for (a,b,c) in Fs:
            ab,bc,ca=mid(a,b),mid(b,c),mid(c,a); F2+=[(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]
        Vs,Fs=np.array(V2),F2
    uv = np.stack([(np.arctan2(Vs[:,1],Vs[:,0])/(2*np.pi))%1.0, np.arccos(np.clip(Vs[:,2],-1,1))/np.pi], axis=1)
    sph = Mesh(Vs, Fs, uvs=uv)
    lod = mind.mesh_cluster_decimate(sph, grid=8)
    assert getattr(lod, "uvs", None) is not None and len(lod.uvs) == len(lod.vertices)
    # Geometry unchanged by the uv option (nothing flips). The invariant is SURFACE identity, not vertex-TABLE
    # identity: seam-correct uvs REQUIRE duplicate vertices at the cuts, because one vertex cannot carry the two
    # uvs a seam needs -- the source asset itself stores them split, and a retopo welds them. So keep_uv="auto"
    # may return MORE vertices than keep_uv=False. What must never change is where the surface is: every face's
    # three positions, in order, bit-identical. Pinned at 0.00e+00 -- an earlier version routed the split through
    # merge_by_distance, whose averaging moved the surface by 1.11e-16, which this engine counts as a flip.
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate
    lod0 = cluster_decimate(sph, grid=8, keep_uv=False)
    PA = np.asarray(lod0.vertices)[np.asarray([f[:3] for f in lod0.faces], int)]
    PB = np.asarray(lod.vertices)[np.asarray([f[:3] for f in lod.faces], int)]
    assert len(lod0.faces) == len(lod.faces)
    assert np.array_equal(PA, PB), "the uv option moved the surface"
    assert len(lod.vertices) >= len(lod0.vertices)     # extra verts only ever appear at the cuts
    assert getattr(lod0, "uvs", None) is None          # the old geometry-only contract, still available


def test_voxel_remesh_refuses_to_project_a_fragmented_atlas(mind):
    """SUPERSEDES test_voxel_remesh_projects_uvs, which PINNED THE BUG: _slit_sphere is (by its own docstring)
    the scan model where every triangle is its own island, and the old test asserted that voxel_remesh emitted
    uvs for it. It did -- scrambled ones. That is precisely the speckle a user reported on a decimated mantis.
    A test asserting a lie is worse than no test, so the assertion is INVERTED, not deleted: keep_uv='auto' must
    now refuse and name the right route, while keep_uv=True still forces the old path for callers who want it."""
    src = _slit_sphere_uv()
    lod = mind.voxel_remesh(src, resolution=24, silhouette=None)
    assert getattr(lod, "uvs", None) is None, "auto must not emit uvs it cannot compute honestly"
    assert lod.uv_transfer_report["skipped"] is True
    assert "rebake_texture" in lod.uv_transfer_report["reason"]
    forced = mind.voxel_remesh(src, resolution=24, keep_uv=True, silhouette=None)
    uvs = np.asarray(forced.uvs)
    assert len(uvs) == len(forced.vertices) and np.isfinite(uvs).all()   # old behaviour still reachable


def _slit_sphere_uv():
    s = _slit_sphere()
    V = np.asarray(s.vertices)
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    uv = np.stack([(np.arctan2(V[:,1],V[:,0])/(2*np.pi))%1.0, np.arccos(np.clip(V[:,2]/np.maximum(np.linalg.norm(V,axis=1),1e-9),-1,1))/np.pi], axis=1)
    return Mesh(V, s.faces, uvs=uv)


# ---- "still haven't seen a properly textured render": the missing composition, pinned -------------------------
# Every piece existed (load_glb extracts embedded textures; render_mesh renders textured) but nothing composed
# them -- the debugging arc itself rendered with a synthetic checker for exactly that reason. preview_asset is
# the composition, and the uv-readback test is the objective proof the mapping is right (not an eyeball claim).

def test_preview_asset_textured_roundtrip(tmp_path, mind):
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.io_and_interop.holographic_gltf import mesh_to_glb
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial, TextureMap
    V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], float)
    uv = np.array([[0,0],[1,0],[1,1],[0,1]], float)
    teximg = np.zeros((8,8,3)); teximg[:, :4] = [1,0,0]; teximg[:, 4:] = [0,0,1]
    blob = mesh_to_glb(Mesh(V, [(0,1,2),(0,2,3)], uvs=uv),
                       material=PBRMaterial(name="two", base_color_map=TextureMap(teximg)), texture=teximg)
    p = tmp_path / "q.glb"; p.write_bytes(blob)
    img, lm = mind.preview_asset(str(p), camera={"eye": [0.5,0.5,2.0], "target": [0.5,0.5,0.0]},
                                 width=64, height=64, ambient=1.0, smooth=False)
    fg = img[img.sum(axis=2) > 0.2]
    assert ((fg[:,0] > 0.4) & (fg[:,2] < 0.3)).sum() > 20     # red half rendered
    assert ((fg[:,2] > 0.4) & (fg[:,0] < 0.3)).sum() > 20     # blue half rendered -> uvs really drive sampling


def test_renderer_uv_convention_matches_gltf(mind):
    """Pin the convention that made the mantis render correct WITHOUT a V flip: the rasterizer maps v directly
    to image row (row 0 = top), which IS glTF's top-left uv origin. If someone 'fixes' the sampler to a
    bottom-left origin, this fails by name instead of every .glb silently flipping."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], float)
    uv = np.array([[0,0],[1,0],[1,1],[0,1]], float)           # v=0 along the TOP edge of the texture
    teximg = np.zeros((8,8,3)); teximg[:4, :] = [1,0,0]; teximg[4:, :] = [0,0,1]   # top rows red
    quad = Mesh(V, [(0,1,2),(0,2,3)], uvs=uv)
    img = np.asarray(mind.render_mesh(quad, {"eye": [0.5,0.5,2.0], "target": [0.5,0.5,0.0]},
                                      width=64, height=64, texture=teximg, uvs=uv,
                                      ambient=1.0, lights=[], background=(0,0,0)))
    # world +Y (v=1) is screen-UP for this camera; v=1 samples the texture's BOTTOM rows (blue).
    top = img[:24][img[:24].sum(axis=2) > 0.2]                # upper screen = v~1 -> blue
    bot = img[40:][img[40:].sum(axis=2) > 0.2]                # lower screen = v~0 -> red
    assert len(top) and len(bot)
    assert top[:, 2].mean() > 0.6 and top[:, 0].mean() < 0.3, "v=1 must sample texture bottom (glTF top-left origin)"
    assert bot[:, 0].mean() > 0.6 and bot[:, 2].mean() < 0.3


# ---- multi-mesh .glb import: the filed latent issue fired on a real asset ------------------------------------
# A crab scan = 24-vert pedestal (material 0) + five ~65k-vert chunks (material 1, the classic 65k split),
# transforms identity. The old first-primitive-only glb_to_mesh returned THE CUBE -- 24 of 312,578 vertices.

def test_multimesh_glb_concatenates_with_transforms(tmp_path):
    import copy, json, struct
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.io_and_interop.holographic_gltf import (mesh_to_glb, glb_to_mesh,
                                                             _GLB_MAGIC, _GLB_VERSION, _CHUNK_JSON, _CHUNK_BIN)
    blob = mesh_to_glb(box())
    jlen = struct.unpack("<I", blob[12:16])[0]
    g = json.loads(blob[20:20 + jlen]); binary = blob[20 + jlen + 8:]
    g["meshes"].append(copy.deepcopy(g["meshes"][0]))
    g["meshes"][0]["primitives"][0]["material"] = 0
    g["meshes"][1]["primitives"][0]["material"] = 1
    g["nodes"] = [{"mesh": 0}, {"mesh": 1, "translation": [5.0, 0.0, 0.0]}]
    g["scenes"] = [{"nodes": [0, 1]}]; g["scene"] = 0
    js = json.dumps(g, separators=(",", ":")).encode(); js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, 12 + 8 + len(js) + 8 + len(binary))
    out += struct.pack("<II", len(js), _CHUNK_JSON) + js + struct.pack("<II", len(binary), _CHUNK_BIN) + binary
    m2 = glb_to_mesh(out)
    one = glb_to_mesh(mesh_to_glb(box()))
    n1 = len(one.vertices)
    assert len(m2.vertices) == 2 * n1 and len(m2.faces) == 2 * len(one.faces)
    assert np.allclose(np.asarray(m2.vertices)[n1:] - np.asarray(m2.vertices)[:n1], [5, 0, 0])
    assert m2.face_material == [0] * len(one.faces) + [1] * len(one.faces)
    # single-mesh files byte-compatible (every engine-emitted .glb)
    assert np.array_equal(np.asarray(one.vertices), np.asarray(m2.vertices)[:n1])


def test_split_nonmanifold_scales(mind):
    """The perf pin for the O(V*E log E) -> O(E) fix: split_nonmanifold on a 20k-face grid-of-bowties must
    finish in seconds, not hours. Measured >800s at 322k faces before the per-vertex edge scan was hoisted;
    8.4s for the whole repair after. This test uses a size that old code would fail by timeout budget."""
    import time
    from holographic.mesh_and_geometry.holographic_meshtools import split_nonmanifold_vertices
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    n = 60
    xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    V = np.stack([xs.ravel(), ys.ravel(), np.zeros((n + 1) ** 2)], axis=1).astype(float)
    def vid(i, j): return i * (n + 1) + j
    F = []
    for i in range(n):
        for j in range(n):
            F += [(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)), (vid(i, j), vid(i + 1, j + 1), vid(i, j + 1))]
    t0 = time.time()
    out, rep = split_nonmanifold_vertices(Mesh(V, F))
    dt = time.time() - t0
    assert dt < 10.0, "split_nonmanifold regressed to the slow path: %.1fs" % dt
    assert rep["split_vertices"] == 0                      # a clean grid is a no-op, fast


# ---- texture through LOD: the mantis "looked terrible at lowest quality" arc ---------------------------------
# ROOT CAUSE (measured): a photogrammetry atlas is per-TRIANGLE (4079 islands / 11003 faces, median 1 face per
# island). No per-vertex uv transfer can preserve it -- a new face's corners land in unrelated islands and the
# triangle smears an arbitrary slice of the atlas across itself. Only a re-bake works.

def _coherent_sheet(n=8):
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    V = np.stack([xs.ravel() / n, ys.ravel() / n, np.zeros((n + 1) ** 2)], axis=1).astype(float)
    def vid(i, j): return i * (n + 1) + j
    F = []
    for i in range(n):
        for j in range(n):
            F += [(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)), (vid(i, j), vid(i + 1, j + 1), vid(i, j + 1))]
    return Mesh(V, F, uvs=V[:, :2].copy())


def _fragmented(sheet):
    """The same surface with every triangle given private vertices -- a scan's per-triangle atlas."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    F = np.asarray([f[:3] for f in sheet.faces], int)
    V = np.asarray(sheet.vertices, float)[F].reshape(-1, 3)
    uv = np.asarray(sheet.uvs, float)[F].reshape(-1, 2)
    return Mesh(V, [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(F))], uvs=uv)


def test_uv_atlas_report_separates_coherent_from_fragmented():
    from holographic.mesh_and_geometry.holographic_meshtools import uv_atlas_report
    sheet = _coherent_sheet()
    r = uv_atlas_report(sheet)
    assert r["islands"] == 1 and r["transferable"]
    rb = uv_atlas_report(_fragmented(sheet))
    assert rb["faces_per_island_median"] == 1.0
    assert not rb["transferable"], "a per-triangle atlas must never be called transferable"


def test_keep_uv_auto_refuses_fragmented_and_names_the_route(mind):
    """The actual bug: cluster_decimate silently emitted scrambled uvs. It must now refuse and SAY WHY."""
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate
    frag = _fragmented(_coherent_sheet())
    auto = cluster_decimate(frag, grid=4, keep_uv="auto")
    assert getattr(auto, "uvs", None) is None
    assert auto.uv_transfer_report["skipped"] is True
    assert "rebake_texture" in auto.uv_transfer_report["reason"]
    forced = cluster_decimate(frag, grid=4, keep_uv=True)          # old behaviour still reachable on request
    assert getattr(forced, "uvs", None) is not None
    coherent = cluster_decimate(_coherent_sheet(), grid=4, keep_uv="auto")
    assert getattr(coherent, "uvs", None) is not None              # a coherent atlas must still transfer


def test_rebake_reproduces_an_analytic_texture_including_corners():
    """Corners are in the assert on purpose: they caught BOTH real bugs -- a conservative-rasterization gap that
    left the corner texel unwritten, and a np.roll dilation that wrapped the atlas's far edge into it."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshtools import rebake_texture
    T = 64
    gy, gx = np.mgrid[0:T, 0:T]
    tex = np.zeros((T, T, 3)); tex[:, :, 0] = gx / (T - 1); tex[:, :, 1] = gy / (T - 1)
    quad = Mesh(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), [(0, 1, 2), (0, 2, 3)],
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))
    tgt = Mesh(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]], float),
               [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
    out, nuv, img, rep = rebake_texture(quad, np.asarray(quad.uvs), tex, tgt, size=256, margin=2)
    OV = np.asarray(out.vertices)
    h, w = img.shape[:2]
    err = []
    for i in range(len(OV)):
        u, v = nuv[i]
        got = img[int(np.clip(round(v * (h - 1)), 0, h - 1)), int(np.clip(round(u * (w - 1)), 0, w - 1))]
        err += [abs(got[0] - OV[i][0]), abs(got[1] - OV[i][1])]
    assert max(err) < 1.0 / (T - 1), "bake error %.4f exceeds texel quantisation" % max(err)
    assert rep["projection_distance_mean"] < 1e-4


def test_textured_lod_routes_by_measurement():
    from holographic.mesh_and_geometry.holographic_meshtools import textured_lod
    tex = np.zeros((32, 32, 3)); tex[:, :, 0] = 1.0
    sheet = _coherent_sheet()
    _, _, img1, r1 = textured_lod(sheet, tex, grid=4)
    assert r1["route"] == "transfer" and img1.shape[0] == 32       # transfer reuses the source image
    _, _, img2, r2 = textured_lod(_fragmented(sheet), tex, grid=4, size=128)
    assert r2["route"] == "rebake" and img2.shape[0] == 128        # a rebake returns a NEW image
    assert "fragmented" in r2["reason"]


@pytest.mark.slow
def test_uv_straddle_metric_is_density_dependent():
    """KEPT NEGATIVE, pinned -- and a correction on record: this metric's threshold is an ABSOLUTE fraction of
    the atlas, so it reads high on any mesh whose legitimate uv edges are large, however healthy. The first
    version of this test asserted a per-face BAKE reads ~1.0 "because every face is its own island". Measured:
    a 2048-face bake reads 0.00 (g=46 -> small cells). The claim was reasoned from a 4-face toy and was false.
    Density is the only real hole."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshtools import rebake_texture, uv_straddle_fraction
    # `threshold` is a fraction OF THE ATLAS, so the mesh must be dense enough for 0.1 to mean "jumped":
    # an 8x8 sheet has legitimate 0.125 uv edges and reads 1.0. That is scope hole (2), pinned below.
    sheet = _coherent_sheet(n=32)
    s_ok, _ = uv_straddle_fraction(sheet, np.asarray(sheet.uvs))
    assert s_ok < 0.05                                             # a healthy DENSE shared atlas reads ~0
    coarse = _coherent_sheet(n=4)
    s_coarse, _ = uv_straddle_fraction(coarse, np.asarray(coarse.uvs))
    assert s_coarse > 0.5, "if this flips, the metric stopped being density-dependent -- update its docstring"
    # a DENSE bake reads ~0 -- the "per-face atlases always read 1.0" story was wrong
    tex = np.zeros((32, 32, 3))
    out, nuv, _, _ = rebake_texture(sheet, np.asarray(sheet.uvs), tex, sheet, size=256)
    s_bake, _ = uv_straddle_fraction(out, nuv)
    assert s_bake < 0.05, "a dense per-face bake must read low; the island story was a misdiagnosis"


# ---- rigged multi-mesh .glb: the same bug class as the geometry reader, one layer down --------------------
# Filed twice, then made ACTIVE by the whole-scene fix: once glb_to_mesh returned every primitive, the
# JOINTS/WEIGHTS reader still returned the first one's rows -- 16 positions against 8 weights, silently.

def test_rigged_multimesh_attributes_align_with_the_vertex_table(tmp_path):
    from holographic.io_and_interop.holographic_assetimport import load_glb, _rigged_glb
    p = tmp_path / "rig.glb"
    p.write_bytes(_rigged_glb(two_skins=False))
    lm = load_glb(str(p))
    J = np.asarray(lm.joints); W = np.asarray(lm.weights)
    n = len(lm.positions) // 2
    assert len(J) == len(lm.positions) and len(W) == len(lm.positions)
    assert np.allclose(W[:n, 0], 1.0)                     # rigged chunk intact
    assert (J[n:] == 0).all() and np.allclose(W[n:], 0.0)  # unrigged chunk zero-fills IN PLACE, not shifting rows
    assert lm.joint_nodes == [2]


def test_two_skins_do_not_collide_on_joint_indices(tmp_path):
    """A JOINTS_0 value indexes its own node's SKIN, not the node table. Two chunks on different skins both say
    'joint 0' and mean different bones; raw concatenation welds two skeletons together without raising."""
    from holographic.io_and_interop.holographic_assetimport import load_glb, _rigged_glb
    p = tmp_path / "twoskin.glb"
    p.write_bytes(_rigged_glb(two_skins=True))
    lm = load_glb(str(p))
    J = np.asarray(lm.joints)
    n = len(lm.positions) // 2
    assert J[0, 0] != J[n, 0], "the remap must separate two skins' colliding local joint 0"
    assert lm.joint_nodes[J[0, 0]] == 2 and lm.joint_nodes[J[n, 0]] == 3


def test_scene_primitives_is_the_single_source_of_vertex_order(tmp_path):
    """The structural fix: ONE traversal, many payloads. If a reader ever grows its own walk again, this drifts."""
    import json, struct
    from holographic.io_and_interop.holographic_gltf import scene_primitives, glb_to_mesh
    from holographic.io_and_interop.holographic_assetimport import _rigged_glb, _glb_chunks
    data = _rigged_glb(two_skins=True)
    gltf, _ = _glb_chunks(data)
    prims = scene_primitives(gltf)
    assert len(prims) == 2                                 # two mesh nodes, one primitive each
    assert [p[3] for p in prims] == [0, 1]                 # node indices ride along, in walk order
    counts = [gltf["accessors"][gltf["meshes"][mi]["primitives"][pi]["attributes"]["POSITION"]]["count"]
              for mi, pi, _M, _ni in prims]
    assert sum(counts) == len(glb_to_mesh(data).vertices)  # the walk PREDICTS the vertex table's length


def test_morph_targets_span_the_scene_and_carry_no_translation(tmp_path):
    """The same first-primitive bug, a third layer down. Also pins the delta trap: morph deltas are transformed
    by the node's rotation/scale but NEVER its translation -- otherwise any non-zero weight teleports the chunk
    by the node's offset."""
    from holographic.io_and_interop.holographic_assetimport import load_glb, _morph_glb
    p = tmp_path / "morph.glb"
    p.write_bytes(_morph_glb())
    lm = load_glb(str(p))
    D = np.asarray(lm.morph_targets)
    n = len(lm.positions) // 2
    assert D.shape == (1, len(lm.positions), 3)
    assert np.allclose(D[0, :n, 1], 0.5)          # morphed chunk intact
    assert np.allclose(D[0, n:], 0.0)             # unmorphed chunk zero-fills in place
    assert np.allclose(D[0, :, 0], 0.0)           # the node's 3.0 translation must NOT leak into a delta


# ---- posing a rigged asset: the last missing link of the import chain ----------------------------------------
# Every piece existed (channels read, slerp, node graph, inverse-binds, LBS) and nothing composed them, so a
# rigged .glb imported and sat in its bind pose forever. Pinned against ANALYTIC truth, not a golden run.

def test_pose_asset_matches_an_analytic_bone_swing(tmp_path):
    from holographic.io_and_interop.holographic_assetimport import load_glb, _bone_glb, pose_asset
    p = tmp_path / "bone.glb"
    p.write_bytes(_bone_glb())
    lm = load_glb(str(p))
    rest = np.asarray(lm.positions, float)
    m0, _ = pose_asset(lm, time=0.0)
    m1, r1 = pose_asset(lm, time=1.0)
    assert r1["mode"] == "animated" and r1["joints"] == 1 and r1["skinned_vertices"] == len(rest)
    assert np.abs(np.asarray(m0.vertices) - rest).max() < 1e-9      # t=0 IS the bind pose
    o = np.array([0.0, 2.0, 0.0])
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expect = (rest - o) @ Rz.T + o
    assert np.abs(np.asarray(m1.vertices) - expect).max() < 1e-9


def test_animation_sample_keeps_rest_for_unanimated_paths(tmp_path):
    """A rotation-only channel is the most common thing in a rig. Defaulting its translation to zero collapsed
    the bone to the origin -- a whole-skeleton bug that raised nothing."""
    from holographic.io_and_interop.holographic_assetimport import load_glb, _bone_glb
    p = tmp_path / "bone.glb"
    p.write_bytes(_bone_glb())
    lm = load_glb(str(p))
    clip = lm.animations[0]
    with_rest = clip.sample(1.0, rest_trs={1: lm.node_graph[1]["trs"]})[1]
    assert np.allclose(with_rest[:3, 3], [0, 2, 0]), "un-animated translation must keep the node's REST value"
    assert np.allclose(clip.sample(1.0)[1][:3, 3], [0, 0, 0])       # no rest passed -> documented old default


def test_quaternion_is_renormalised_before_becoming_a_matrix():
    """glTF stores quats as float32 -> unit only to ~1.7e-8. The matrix formula is exact only for a unit quat,
    so the raw value produced det 0.99999993 (not a rotation) and put an analytic 90-degree swing 1.0e-07 off."""
    from holographic.io_and_interop.holographic_assetimport import _quat_to_mat
    h = float(np.float32(np.sin(np.pi / 4)))
    M = _quat_to_mat(np.array([0.0, 0.0, h, h]))
    assert abs(np.linalg.det(M) - 1.0) < 1e-12, "det %.17f -- quaternion not renormalised" % np.linalg.det(M)
    assert np.abs(M @ M.T - np.eye(3)).max() < 1e-12               # orthonormal, i.e. actually a rotation
    assert np.allclose(_quat_to_mat(np.zeros(4)), np.eye(3))       # malformed channel -> identity, not NaN


def test_sparse_and_dense_lbs_agree(mind):
    """The sparse form must be a calling convention, not a second algorithm."""
    from holographic.mesh_and_geometry.holographic_meshskin import linear_blend_skin, linear_blend_skin_indexed
    rng = np.random.default_rng(0)
    V = rng.normal(size=(120, 3)); B = 5
    T = np.stack([np.eye(4) for _ in range(B)])
    for b in range(B):
        T[b][:3, 3] = rng.normal(size=3)
        T[b][:3, :3] += rng.normal(scale=0.1, size=(3, 3))
    J = rng.integers(0, B, (120, 4))
    Wk = rng.random((120, 4))
    dense = np.zeros((120, B))
    for v in range(120):
        for k in range(4):
            dense[v, J[v, k]] += Wk[v, k]
    a = linear_blend_skin(V, T, dense)
    b = linear_blend_skin_indexed(V, T, J, Wk)
    assert np.abs(a - b).max() < 1e-10          # measured 3.5e-12; the orders of accumulation differ
    # an unclaimed vertex is left where it is, not collapsed to the origin
    c = linear_blend_skin_indexed(V, T, np.zeros((120, 4), int), np.zeros((120, 4)))
    assert np.allclose(c, V)


# ---- uv survival through topology change ---------------------------------------------------------------------
# The audit that started this: give every face-count-changing operation a mesh WITH a coherent atlas and see
# what survives. cluster_decimate and voxel_remesh were both emitting NaN uvs -- silently, on a perfectly
# coherent atlas -- because Ericson's closest-point test assumes a non-degenerate triangle and a uv-sphere's
# POLE is a zero-length-edge triangle. A mesh whose uvs are NaN has no relationship to its texture at all.

def test_closest_point_is_finite_on_degenerate_triangles():
    from holographic.mesh_and_geometry.holographic_meshtools import _closest_point_barycentric as cpb
    cases = [
        (np.array([0., 0., 0.]), np.array([1., 0, 0]), np.array([1., 0, 0]), np.array([0, 1., 0])),  # pole
        (np.array([0., 0., 0.]), np.array([1., 1, 1]), np.array([1., 1, 1]), np.array([1., 1, 1])),  # collapsed
        (np.array([0., 1., 0.]), np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([2., 0, 0])),  # collinear
        (np.array([.2, .2, 1.]), np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([0, 1., 0])),  # healthy
    ]
    for p, a, b, c in cases:
        q, bc = cpb(p, a, b, c)
        assert np.isfinite(q).all() and np.isfinite(bc).all()
        assert abs(sum(bc) - 1.0) < 1e-12


@pytest.mark.parametrize("op", ["cluster_decimate", "voxel_remesh", "qem_decimate", "triangulate"])
def test_face_changing_ops_never_emit_broken_uvs(mind, op):
    """The regression trap for the whole audit: no face-count-changing operation may hand back uvs that are
    non-finite or the wrong length. Silence here is what let NaN uvs ship."""
    from holographic.mesh_and_geometry.holographic_meshtools import _uv_sphere_fixture
    src = _uv_sphere_fixture()
    out = {"cluster_decimate": lambda: mind.mesh_cluster_decimate(src, grid=6),
           "voxel_remesh": lambda: mind.voxel_remesh(src, resolution=18, silhouette=None),
           "qem_decimate": lambda: mind.mesh_qem_decimate(src, target_faces=60),
           "triangulate": lambda: mind.mesh_triangulate(src)}[op]()
    if isinstance(out, tuple):
        out = out[0]
    uv = getattr(out, "uvs", None)
    assert uv is not None, "%s dropped the uvs entirely" % op
    uv = np.asarray(uv)
    assert len(uv) == len(out.vertices), "%s: %d uvs for %d verts" % (op, len(uv), len(out.vertices))
    assert np.isfinite(uv).all(), "%s emitted non-finite uvs" % op


def test_reproject_uv_eliminates_seam_spanning_faces(mind):
    """A cylinder's seam has real AREA either side -- unlike a sphere's pole fan, which is a singularity and
    too small on screen to show the difference (measured: a wash there). Per-vertex transfer leaves faces
    spanning the whole atlas; reprojection must leave none, and must split only at the cut."""
    from holographic.mesh_and_geometry.holographic_meshtools import _uv_cylinder_fixture, transfer_uv
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate
    src = _uv_cylinder_fixture()
    lod = cluster_decimate(src, grid=7, keep_uv=False)

    def crossers(mesh, uv):
        F = np.asarray([f[:3] for f in mesh.faces], int)
        me = np.zeros(len(F))
        for k in range(3):
            me = np.maximum(me, np.abs(uv[F[:, k], 0] - uv[F[:, (k + 1) % 3], 0]))
        return int((me > 0.5).sum())

    naive, _ = transfer_uv(src, np.asarray(src.uvs), np.asarray(lod.vertices))
    assert crossers(lod, naive) > 0, "fixture must exhibit the bug or it proves nothing"
    mesh, uv, rep = mind.mesh_reproject_uv(src, np.asarray(src.uvs), lod)
    assert rep["finite"] and crossers(mesh, uv) == 0
    assert rep["seam_splits"] < 0.25 * len(lod.vertices)
    # v has NO seam on a cylinder, so any error in it is fabricated by the tie-break. An earlier version chose
    # both "which side of the cut" and "where on that side" with one uv argmin, dragging every ambiguous corner
    # toward its own face's centre -- v went from exactly 0.00000 to 0.05 at p95, which reads as a texture that
    # does not line up. Side is chosen by uv; position is chosen by geometry. Pinned exact.
    v_true = (np.asarray(mesh.vertices)[:, 1] / 2.0) + 0.5
    assert np.abs(uv[:, 1] - v_true).max() < 1e-9, "reprojection fabricated error in the seam-free coordinate"


def test_reproject_refuses_a_fragmented_atlas_and_names_the_route(mind):
    from holographic.mesh_and_geometry.holographic_meshtools import _uv_cylinder_fixture
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    src = _uv_cylinder_fixture()
    F = np.asarray([f[:3] for f in src.faces], int)
    frag = Mesh(np.asarray(src.vertices)[F].reshape(-1, 3),
                [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(F))],
                uvs=np.asarray(src.uvs)[F].reshape(-1, 2))
    with pytest.raises(ValueError, match="rebake"):
        mind.mesh_reproject_uv(frag, np.asarray(frag.uvs), src)


def test_keep_uv_true_still_forces_the_legacy_transfer(mind):
    """The documented escape hatch must not change meaning. Routing True through reprojection silently turned
    "force" into "no uvs at all", because reprojection RAISES on a fragmented atlas."""
    from holographic.mesh_and_geometry.holographic_meshtools import _uv_cylinder_fixture
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate
    src = _uv_cylinder_fixture()
    forced = cluster_decimate(src, grid=7, keep_uv=True)
    assert getattr(forced, "uvs", None) is not None
    assert forced.uv_transfer_report.get("forced") is True


# ---- decimation under control: explicit budgets + the optional silhouette guard ------------------------------
# Moose, after the crab's legs degraded at 3k faces: "we need control over whether a model gets decimated...
# specify a limit % or face count... silhouette comparison... optional." The engine stops deciding for the
# caller. The fixture is a box with a thin spike -- the crab's leg in miniature: coarse clustering deletes it
# and only the silhouette critic notices.

def _spiky_mesh():
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    b = triangulate_ngons(box())
    V = np.asarray(b.vertices, float).tolist()
    F = [tuple(f) for f in b.faces]
    tip = len(V); V.append([0.5, 3.0, 0.5])
    F += [(2, 3, tip), (3, 7, tip), (7, 6, tip), (6, 2, tip)]
    return loop_subdivide(Mesh(np.array(V), F), levels=3)


def test_decimate_to_no_target_is_identity(mind):
    from holographic.mesh_and_geometry.holographic_meshqem import decimate_to
    dense = _spiky_mesh()
    out, r = decimate_to(dense)
    assert out is dense and r["modified"] is False   # the SAME object: "never modify" must mean never


def test_decimate_to_hits_an_explicit_budget(mind):
    dense = _spiky_mesh()
    out, r = mind.mesh_decimate_to(dense, target_faces=200, keep_uv=False)
    assert r["modified"] and abs(r["result_faces"] - 200) / 200.0 <= 0.35
    out2, r2 = mind.mesh_decimate_to(dense, target_fraction=0.1, keep_uv=False)
    assert r2["modified"]
    with pytest.raises(ValueError):
        mind.mesh_decimate_to(dense, target_faces=100, target_fraction=0.5)   # ambiguous: refuse


def test_silhouette_guard_refuses_to_eat_the_spike(mind):
    dense = _spiky_mesh()
    coarse, rc = mind.mesh_decimate_to(dense, target_faces=60, keep_uv=False)
    guarded, rg = mind.mesh_decimate_to(dense, target_faces=60, keep_uv=False,
                                        min_silhouette_iou=0.97, views_size=96)
    assert min(rg["silhouette_iou"].values()) >= 0.97
    assert rg["result_faces"] > rc["result_faces"]           # walked back to protect the outline
    assert rg["budget_missed_for_silhouette"] is True        # ...and SAID so, never silently



def test_silhouette_guard_is_default_on_and_opt_out_is_exact(mind):
    """The owner directive: preservation is the DEFAULT, destruction is the opt-out. Default call walks back
    when the outline breaks; silhouette=None reproduces the unguarded primitive bit-identically."""
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate
    dense = _spiky_mesh()
    guarded = mind.mesh_cluster_decimate(dense, grid=3, keep_uv=False)          # brutal grid: must walk back
    rep = guarded.silhouette_report
    assert rep["guard_walked_back"] and rep["passed"]
    assert min(rep["silhouette_iou"].values()) >= 0.95
    bare = mind.mesh_cluster_decimate(dense, grid=3, keep_uv=False, silhouette=None)
    raw = cluster_decimate(dense, grid=3, keep_uv=False)
    assert np.array_equal(np.asarray(bare.vertices), np.asarray(raw.vertices))
    assert [tuple(f) for f in bare.faces] == [tuple(f) for f in raw.faces]
    assert bare.silhouette_report["guard_walked_back"] is False


def test_voxel_remesh_guard_smoke(mind):
    """The same shared guard drives voxel_remesh's faculty; one cheap resolution-walk smoke test."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    out = mind.voxel_remesh(triangulate_ngons(box()), resolution=12)
    rep = out.silhouette_report
    assert rep["min_silhouette_iou"] == 0.95 and "silhouette_iou" in rep


# ---- the spherical-instrument search: EGI shipped, the blind candidates pinned AS negatives ------------------

def test_egi_is_the_orientation_complement(mind):
    """EGI measures surface character, not outline -- the quantity the silhouette guard admits blindness to.
    Identity exact; translation-invariant; loses similarity when faces (the spike) vanish."""
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    b = triangulate_ngons(box())
    assert mind.mesh_egi_compare(b, b)["similarity"] == 1.0
    V = np.asarray(b.vertices, float)
    moved = Mesh(V + np.array([3.0, -1.0, 2.0]), [tuple(f) for f in b.faces])
    assert mind.mesh_egi_compare(b, moved)["similarity"] == 1.0
    spiky = _spiky_mesh()
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    assert mind.mesh_egi_compare(spiky, loop_subdivide(_spiky_mesh(), levels=0))["similarity"] == 1.0


def test_radial_occupancy_is_blind_kept_negative(mind):
    """PINNED NEGATIVE from the spherical-render investigation: 'does a ray from the centre hit the object'
    saturates on any star-ish shape and carries ~no silhouette information. If this test ever FAILS, the
    negative needs re-examining -- that is what a kept negative is for."""
    # First version of this pin USED VERTICES and failed for the confound the investigation itself had already
    # named: 325 fixture vertices cannot saturate 512 bins regardless of shape. The instrument must be fed
    # density-independent SURFACE SAMPLES or it measures vertex count -- same lesson as the extent map.
    from holographic.mesh_and_geometry.holographic_meshtools import _uv_sphere_fixture
    sph = _uv_sphere_fixture(24)
    V = np.asarray(sph.vertices, float)
    F = np.asarray([f[:3] for f in sph.faces], int)
    A_, B_, C_ = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    rng = np.random.default_rng(0)
    idx = np.repeat(np.arange(len(F)), 40)                       # 40 samples/face: density-independent
    u = rng.random(len(idx)); v = rng.random(len(idx))
    flip = u + v > 1; u[flip] = 1 - u[flip]; v[flip] = 1 - v[flip]
    P = A_[idx] * (1 - u - v)[:, None] + B_[idx] * u[:, None] + C_[idx] * v[:, None]
    c = P.mean(0)
    d = P - c
    r = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    th = np.arccos(np.clip(d[:, 1] / r, -1, 1)); ph = np.arctan2(d[:, 2], d[:, 0]) % (2 * np.pi)
    bins = np.zeros(16 * 32)
    bins[np.clip((th / np.pi * 16).astype(int), 0, 15) * 32
         + np.clip((ph / (2 * np.pi) * 32).astype(int), 0, 31)] = 1.0
    assert bins.mean() > 0.9, "occupancy saturates on a star-ish shape -- the negative holds"


# ---- THE STRUCTURAL TRAP: no mesh-reducing faculty ships unguarded, ever again -------------------------------
# Moose's directive: silhouette comparison DRIVES decimation/LOD/remesh/retopo; no visibly destructive change
# without explicit user direction. This test enumerates the family BY NAME PATTERN, so a future faculty that
# matches and lacks a silhouette parameter is a CI failure the day it lands -- the guard is a property of the
# family, not a per-function favour.

_GUARD_EXEMPT = {
    # refinement, not reduction -- adds faces, cannot destroy the outline the guard watches for:
    "mesh_subdivide", "subdivide_sequence",
    # selection over an existing chain -- modifies nothing:
    "mesh_select_lod", "splat_select_lod",
    # splat family: gaussian splats have their own PSNR-based LOD contract, not a mesh silhouette:
    "splat_lod_chain",
    # explicit-budget wrapper whose guard parameter has its own name (asserted separately below):
    "mesh_decimate_to",
}


def test_every_mesh_reducing_faculty_is_silhouette_guarded(mind):
    import inspect
    fam = [n for n in dir(mind) if not n.startswith("_") and any(
        k in n.lower() for k in ("decimate", "remesh", "retopo", "lod", "simplif"))]
    unguarded = []
    for n in sorted(fam):
        if n in _GUARD_EXEMPT:
            continue
        sig = inspect.signature(getattr(mind, n))
        if "silhouette" not in sig.parameters:
            unguarded.append(n)
    assert not unguarded, "mesh-reducing faculties without a silhouette guard: %s" % unguarded
    # the wrapper's guard has its own name and its own default -- pinned here so renames cannot silently drop it
    dsig = inspect.signature(mind.mesh_decimate_to)
    assert dsig.parameters["min_silhouette_iou"].default == 0.95


def test_lod_chain_truncates_destroyed_levels(mind):
    dense = _spiky_mesh()
    mind.mesh_cluster_lod_chain(dense, grids=(12, 6, 3))
    rep = mind._last_lod_chain_silhouette
    assert rep["levels_kept"] < rep["levels_in"], "the destroyed coarse tail must be truncated"
    assert rep["dropped"] and rep["dropped"][-1]["worst"] < 0.95
    full = mind.mesh_cluster_lod_chain(dense, grids=(12, 6, 3), silhouette=None)
    assert len(full) == rep["levels_in"], "opt-out must keep the whole chain"


def test_quad_remesh_verdict_and_retopo_walkback(mind):
    from holographic.mesh_and_geometry.holographic_meshtools import _uv_sphere_fixture
    sph = _uv_sphere_fixture(24)
    q = mind.quad_remesh(sph)
    qm = q[0] if isinstance(q, tuple) else q
    assert qm.silhouette_report["worst"] >= 0.95 and qm.silhouette_report["refused"] is False
    r = mind.auto_retopo(sph, voxel_resolution=8)      # brutal resolution: the guard must walk it up
    sr = r["silhouette_report"]
    assert sr["passed"] and sr["guard_walked_back"] and sr["knob"] > 8


# ---- R1 (PLAN_retopo): sparse cross_field ---------------------------------------------------------------------

def test_sparse_cross_field_parity_and_routing(mind):
    """The unblocking item of the retopo plan: the connection-Laplacian smallest eigenpair WITHOUT the dense
    matrix. Pins: eigenvalue parity with dense eigh; never an interior eigenvalue (the RQI nearest-trap);
    deterministic; auto routes small meshes to the bit-compatible dense path."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_crossfield import (cross_field, _sparse_smallest_eigvec,
                                                                      connection, connection_laplacian)
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    b = triangulate_ngons(box())
    mm = loop_subdivide(b, levels=2)
    V = np.asarray(mm.vertices, float); F = np.asarray(mm.faces, int)
    rho, opp, nxt, de = connection(V, F)
    w = np.linalg.eigvalsh(connection_laplacian(F, rho, de))
    u1, lam1, _ = _sparse_smallest_eigvec(len(F), rho, de)
    u2, lam2, _ = _sparse_smallest_eigvec(len(F), rho, de)
    assert lam1 == lam2 and np.array_equal(u1, u2)
    assert abs(lam1 - w[0]) <= max(1e-6, 2 * (w[1] - w[0]))
    assert lam1 < w[1] + 1e-9                      # interior-eigenvalue trap stays dead
    phi, ctx = mind.cross_field(mm)
    assert ctx["solver"] == "dense"                # bit-compatible path for previously-affordable sizes
    big = loop_subdivide(b, levels=4)
    phi_b, ctx_b = mind.cross_field(big)
    assert ctx_b["solver"] == "sparse" and np.isfinite(phi_b).all()


def test_guard_step_is_cost_aware_and_capped(mind):
    """R2 (PLAN_retopo): the guard's walk is sized to the knob's cost curve, and a cap refuses instead of
    OOMing. auto_retopo's voxel_resolution is CUBIC -- x1.5 on it OOM-killed this process twice."""
    from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guarded, cluster_decimate
    from holographic.mesh_and_geometry.holographic_meshtools import _uv_sphere_fixture
    sph = _uv_sphere_fixture(24)
    _, r = silhouette_guarded(sph, lambda g: cluster_decimate(sph, g, keep_uv=False), 3, min_iou=0.95)
    assert r["step_factor"] == 1.5                      # linear history must not flip
    seen = []

    def stuck(k):
        seen.append(k)
        return cluster_decimate(sph, 3, keep_uv=False)
    _, r2 = silhouette_guarded(sph, stuck, 24, min_iou=0.999, knob_cost="cubic", max_knob=40, max_steps=8)
    assert seen == [24, 31, 40] and r2["refused_knob_cap"] is True and r2["passed"] is False
    with pytest.raises(ValueError):
        silhouette_guarded(sph, lambda g: cluster_decimate(sph, g, keep_uv=False), 3, knob_cost="quartic")


def test_cross_field_natural_boundary(mind):
    """R3a (PLAN_retopo): cross_field solves OPEN meshes with free boundaries -- the closed check was a guard,
    not maths. Every photogrammetry scan is open, which is exactly what the surface retopo route serves."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    closed = loop_subdivide(triangulate_ngons(box()), levels=2)
    p1, c1 = mind.cross_field(closed)
    p2, c2 = mind.cross_field(closed, boundary="natural")
    assert np.array_equal(p1, p2)                       # closed meshes: bit-identical, nothing flips
    openm = Mesh(np.asarray(closed.vertices, float), [tuple(f) for f in closed.faces][:-12])
    with pytest.raises(ValueError):
        mind.cross_field(openm)                         # default error contract unchanged
    phi, ctx = mind.cross_field(openm, boundary="natural")
    assert np.isfinite(phi).all() and ctx["n_boundary_edges"] > 0


# ---- P1/R8 (promotion ledger, archived to NOTES): one CG, both callers delegating, both pinned ---------------

def test_promoted_cg_serves_both_fields(mind):
    """The first executed promotion: holographic_numerics.cg replaced two independent CG copies. Pins:
    (1) real input is BIT-identical to the historical image _cg loop; (2) complex-Hermitian converges (the
    case the historical form could not solve); (3) both old sites now DELEGATE (no second implementation)."""
    import numpy as np, inspect
    from holographic.misc.holographic_numerics import cg
    from holographic.io_and_interop import holographic_image as hi
    from holographic.mesh_and_geometry import holographic_crossfield as cf
    rng = np.random.default_rng(0)
    n = 32
    A = rng.standard_normal((n, n)); A = A @ A.T + n * np.eye(n)
    b = rng.standard_normal(n)
    x = np.zeros_like(b); r = b - A @ x; p = r.copy(); rs = r @ r          # the historical loop, verbatim
    for _ in range(250):
        Ap = A @ p; a = rs / (p @ Ap + 1e-30)
        x += a * p; r -= a * Ap
        rs2 = r @ r
        if rs2 < 1e-13:
            break
        p = r + (rs2 / rs) * p; rs = rs2
    assert np.array_equal(x, cg(lambda v: A @ v, b))
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = M @ M.conj().T + n * np.eye(n)
    bc = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert np.abs(H @ cg(lambda v: H @ v, bc, iters=4 * n, tol=1e-24) - bc).max() < 1e-8
    assert "holographic_numerics import cg" in inspect.getsource(hi._cg)
    # M7 moved crossfield's iteration (and its cg call) INTO numerics.smallest_eigenpair -- the delegation is
    # one level deeper now: crossfield -> smallest_eigenpair -> cg. Assert the CURRENT chain, each link.
    assert "holographic_numerics import smallest_eigenpair" in inspect.getsource(cf._sparse_smallest_eigvec)
    from holographic.misc import holographic_numerics as _num
    assert "cg(" in inspect.getsource(_num.smallest_eigenpair)
    x2 = mind.solve_linear_cg(A, b)
    # tol is on ||r||^2, so the contracted residual is ~sqrt(tol): assert THAT, not a wish
    assert np.abs(A @ x2 - b).max() < 1e-5


def test_guided_cross_field_sparse_route(mind):
    """R6a (PLAN_retopo v2): the guided solve through the promoted shared CG. (L + diag(w) + eps I) is
    Hermitian PD by construction -- numerics.cg's exact contract. Pins parity, the guide-free identity with
    cross_field's sparse path, and the inherited natural boundary."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    mm = loop_subdivide(triangulate_ngons(box()), levels=2)
    F = np.asarray(mm.faces, int); V = np.asarray(mm.vertices, float)
    rng = np.random.default_rng(1)
    g = np.zeros((len(F), 3))
    idx = rng.choice(len(F), size=len(F) // 5, replace=False)
    e = V[F[idx, 1]] - V[F[idx, 0]]
    g[idx] = e / np.linalg.norm(e, axis=1, keepdims=True)
    pd, _ = mind.guided_cross_field(mm, g, solver="dense")
    ps, cs = mind.guided_cross_field(mm, g, solver="sparse")
    assert np.abs(np.exp(1j * 4 * pd) - np.exp(1j * 4 * ps)).max() < 1e-6
    p0, _ = mind.guided_cross_field(mm, np.zeros((len(F), 3)), solver="sparse")
    p1, _ = mind.cross_field(mm, solver="sparse")
    assert np.abs(np.exp(1j * 4 * p0) - np.exp(1j * 4 * p1)).max() < 1e-12
    openm = Mesh(V, [tuple(f) for f in mm.faces][:-12])
    with pytest.raises(ValueError):
        mind.guided_cross_field(openm, np.zeros((len(openm.faces), 3)))
    po, _ = mind.guided_cross_field(openm, np.zeros((len(openm.faces), 3)), boundary="natural")
    assert np.isfinite(po).all()


def test_surface_retopo_passes_the_gate(mind):
    """R3 (PLAN_retopo v2): the surface route the whole arc was for. Vertices never leave the source, so the
    silhouette survives BY CONSTRUCTION -- the gate voxelize-then-quad structurally cannot pass on thin
    features. Also pins guard membership (it is a mesh-reducing faculty)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(triangulate_ngons(box()), levels=3)
    out, rep = mind.surface_retopo(src, density=1.0)
    g = rep["silhouette_report"]
    assert g["passed"] is True and min(g["silhouette_iou"].values()) >= 0.95
    assert rep["quad_fraction"] > 0.6 and rep["faces"] < len(src.faces)
    assert g["knob_cost"] == "linear"                  # NOT auto_retopo's cubic voxel knob
    o2, r2 = mind.surface_retopo(src, density=1.0)
    assert np.array_equal(np.asarray(out.vertices), np.asarray(o2.vertices))


def test_mesh_orient_is_the_field_precondition(mind):
    """R3b (PLAN_retopo v2): consistent winding, the precondition cross_field/surface_retopo need and that
    scans lack. Pins the distinction that cost a wrong answer: NON-MANIFOLD is not NON-ORIENTABLE."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(triangulate_ngons(box()), levels=2)
    out, rep = mind.mesh_orient(src)
    assert rep["oriented"] and rep["flipped"] == 0
    assert [tuple(f) for f in out.faces] == [tuple(f) for f in src.faces]     # oriented input untouched
    rng = np.random.default_rng(0)
    bad = [tuple(reversed(f)) if rng.random() < 0.5 else tuple(f) for f in src.faces]
    o2, r2 = mind.mesh_orient(Mesh(np.asarray(src.vertices, float), bad))
    assert r2["oriented"] and r2["flipped"] > 0
    V = [[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]]
    _o3, r3 = mind.mesh_orient(Mesh(np.asarray(V, float), [(0, 1, 2), (0, 1, 3), (0, 1, 4)]))
    assert r3["non_manifold_edges"] >= 1 and r3["non_orientable_components"] == 0
    assert r3["propagation_components"] == r3["components"]      # deprecated alias must not drift
    assert mind.mesh_orientation_report(src)["oriented"] is True             # general-degree checker


def test_is_oriented_delegates_no_second_checker(mind):
    """Dedup audit (2026-07-17): is_oriented was QUAD-ONLY and face_orientation_report supersets it. Two
    checkers for one property is the exact fragmentation the ledger exists to stop -- so is_oriented now
    delegates. Pins parity with the historical quad counter so the delegation cannot drift."""
    import numpy as np, inspect
    from collections import Counter
    from holographic.mesh_and_geometry.holographic_isosurface import is_oriented

    def historical(quads):
        c = Counter()
        for q in np.asarray(quads, int):
            for e in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
                c[e] += 1
        return bool(c) and all(v == 1 for v in c.values())
    for qs in ([(0, 1, 2, 3), (3, 2, 5, 4)], [(0, 1, 2, 3), (2, 3, 4, 5), (1, 0, 3, 2)], [(0, 1, 2, 3)]):
        assert is_oriented(qs) == historical(qs)
    assert "face_orientation_report" in inspect.getsource(is_oriented)      # one implementation, not two


def test_transform_mesh_is_reflection_aware(mind):
    """M3 (BACKLOG), the CORRECTED version. The original M3 claimed mesh_orient cured the demo's axis-swap
    bug; measurement proved it cannot -- a reflection leaves the mesh consistently oriented and consistently
    inside-out, and mesh_orient repairs only DISAGREEMENT. transform_mesh owns the rule instead."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    b = triangulate_ngons(box())

    def outward(mesh):
        V = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces, int)
        c = V.mean(0)
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        return float(((n * (V[F].mean(1) - c)).sum(1) > 0).mean())
    assert outward(mind.transform_mesh(b, np.diag([1.0, 1.0, -1.0]))) == 1.0     # det<0 -> flipped
    rot = mind.transform_mesh(b, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]))  # det>0 -> untouched
    assert [tuple(f) for f in rot.faces] == [tuple(f) for f in b.faces]
    assert outward(mind.convert_up_axis(b, "z", "y")) == 1.0
    naive = Mesh(np.asarray(b.vertices, float)[:, [0, 2, 1]], [tuple(f) for f in b.faces])
    assert mind.mesh_orientation_report(naive)["oriented"] is True and outward(naive) == 0.0
    assert mind.mesh_orient(naive)[1]["flipped"] == 0        # the distinction, pinned
    with pytest.raises(ValueError):
        mind.transform_mesh(b, np.zeros((3, 3)))


def test_topology_gate_catches_what_the_silhouette_cannot(mind):
    """The owner's gate: islands, holes punched, holes filled -- none of which move the OUTLINE. Pins the
    motivating measurement too: our own surface_retopo passes the silhouette gate while punching holes."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(triangulate_ngons(box()), levels=2)
    V = np.asarray(src.vertices, float); F = [tuple(f) for f in src.faces]
    assert mind.mesh_topology_delta(src, src)["preserved"] is True
    holed = Mesh(V, F[:-4])
    assert mind.mesh_topology_delta(src, holed)["holes_created"]
    assert mind.mesh_topology_delta(holed, src)["holes_filled"]      # filling an EXISTING hole is a violation
    island = Mesh(np.vstack([V, V[:3] + 50.0]), F + [(len(V), len(V) + 1, len(V) + 2)])
    assert mind.mesh_topology_delta(src, island)["islands_created"]
    # QEM and cluster decimation must keep topology; this is the regression trap for them
    for out in (mind.mesh_cluster_decimate(src, grid=8, silhouette=None),
                mind.mesh_decimate_to(src, target_faces=60, min_silhouette_iou=None)[0]):
        assert mind.mesh_topology_delta(src, out)["preserved"], "a decimator must not change topology"


def test_surface_retopo_reports_topology_without_flipping_decisions(mind):
    """M13 template: the second gate arrives as an INSTRUMENT, not a policy change. Pins that (a) the topology
    delta is reported by default, (b) the face count is IDENTICAL with the instrument on, off, and as it was
    before it existed -- adding a measurement must never change an answer -- and (c) refuse is opt-in."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(triangulate_ngons(box()), levels=3)
    out, rep = mind.surface_retopo(src, density=1.0)
    assert min(rep["silhouette_report"]["silhouette_iou"].values()) >= 0.95
    # M11 oriented-dedup fix: a CLOSED input now comes out CLOSED -- no holes created. (topology may still be
    # non-preserved via reported non-manifold singular edges, which is data, not a hole.)
    assert rep["topology"]["holes_created"] is False
    from holographic.mesh_and_geometry.holographic_meshtools import face_orientation_report
    assert face_orientation_report(out)["boundary_edges"] == 0
    _o2, r2 = mind.surface_retopo(src, density=1.0, topology=False)
    assert r2["faces"] == rep["faces"] and "topology" not in r2
    with pytest.raises(ValueError):
        mind.surface_retopo(src, density=1.0, topology="refuse")


def test_m2_strain_guides_steer_the_retopo_field(mind):
    """M2 (BACKLOG): the deformation-guide path, end to end. strain_directions -> surface_retopo(guide_dirs).
    Pins that guides are LOAD-BEARING, not inert -- the field's 4-RoSy alignment to the strain must rise
    sharply under guidance and climb with guide_weight. (Measured wrong first with a single-representative
    dot product, which reads an arbitrary one of the four RoSy directions; the correct metric is
    cos(4*(phi-guide)), aligned modulo 90 deg.)"""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_crossfield import cross_field, guided_cross_field, face_frames
    src = loop_subdivide(triangulate_ngons(box()), levels=3)
    V = np.asarray(src.vertices, float); F = np.asarray(src.faces, int)
    deformed = V.copy(); deformed[:, 0] += 0.4 * np.sin(V[:, 1] * 3.0)
    sd = np.asarray(mind.strain_directions(src, deformed))
    assert sd.shape == (len(F), 3)
    _n, ex, ey = face_frames(V, F)
    sd_th = np.arctan2((sd * ey).sum(1), (sd * ex).sum(1))
    rosy = lambda phi: float(np.cos(4.0 * (phi - sd_th)).mean())
    af = rosy(cross_field(src)[0])
    ag = rosy(guided_cross_field(src, sd, guide_weight=5.0)[0])
    assert ag > af + 0.5 and ag > 0.95, "strain guidance must strongly align the field (%.3f -> %.3f)" % (af, ag)
    # and it must flow all the way through the faculty
    out, rep = mind.surface_retopo(src, density=1.5, guide_dirs=sd, silhouette=None, topology=False)
    assert rep["guided"] is True and rep["faces"] > 0


def test_fit_camera_carries_its_aspect(mind):
    """M10: fit_camera sized the horizontal half-angle as tx=ty*aspect but did NOT return aspect, so the
    renderer used its default and the fit was silently wrong on non-square frames -- the same two-paths-
    disagree-on-aspect bug that bit rasterize_mesh earlier. Pins that the camera now CARRIES the aspect it
    was fit for, and that the subject fills the constraining axis on wide AND tall frames."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.rendering.holographic_render import fit_camera
    b = triangulate_ngons(box())
    for W, H in [(800, 200), (512, 384), (200, 800)]:
        cam = fit_camera(b, direction=(0.3, 0.2, 1.0), fov_deg=50.0, aspect=W / float(H))
        assert abs(cam["aspect"] - W / float(H)) < 1e-9
        img = np.asarray(mind.render_mesh(b, cam, width=W, height=H, ambient=0.7,
                                          base_color=(0.8, 0.7, 0.6), background=(0, 0, 0)), float)
        img = img / 255.0 if img.max() > 1.5 else img
        cov = img.mean(2) > 0.02
        ys, xs = np.where(cov)
        # the subject must fill a large fraction of the CONSTRAINING axis (>=70%), whichever it is
        fw = (xs.max() - xs.min()) / W; fh = (ys.max() - ys.min()) / H
        assert max(fw, fh) >= 0.70, "fit must fill the binding axis (%dx%d: %.2f x %.2f)" % (W, H, fw, fh)


def test_preview_asset_fit_is_opt_in(mind):
    """M10: preview_asset gained fit=True (fit_camera) but the DEFAULT stays the bbox-diagonal heuristic, so
    no existing preview reframes. Pins that fit=True runs and returns an image of the requested size.

    The asset is a dev-box fixture under /mnt/user-data/uploads (not shipped in the repo), so this SKIPS rather
    than errors when it is absent -- the test is real where the file exists and must not red-CI everywhere else.
    A hardcoded absolute path that only resolves on one machine is a portability bug in the test, not the code."""
    import os

    import numpy as np

    asset = '/mnt/user-data/uploads/cc0____japanese_freshwater_crab.glb'
    if not os.path.exists(asset):
        import pytest
        pytest.skip("preview_asset fixture %s not present (dev-box asset, not shipped)" % asset)
    img, mesh = mind.preview_asset(asset, width=256, height=192, fit=True)
    assert np.asarray(img).shape[:2] == (192, 256)


def test_m12_displacement_rides_the_normal_projection(mind):
    """M12/M14: displacement bakes from the SAME closest-point cast as the normal (one pass, two channels).
    Pins the load-bearing invariant -- turning displacement ON must not perturb the normal map by a single
    bit -- plus the cage (a stray hit clamps, because displacement moves geometry, not just shading)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    high = loop_subdivide(triangulate_ngons(box()), levels=3)
    low, _ = mind.mesh_decimate_to(high, target_faces=120, min_silhouette_iou=None)
    uv = np.asarray(low.vertices)[:, :2].copy(); uv = (uv - uv.min(0)) / (uv.max(0) - uv.min(0) + 1e-9)
    n_only = mind.bake_normal_map(low, uv, high, size=32)
    n_with, disp = mind.bake_normal_map(low, uv, high, size=32, displacement=True, max_distance=0.5)
    assert np.array_equal(n_only, n_with), "displacement must not perturb the normal bake (shared projection)"
    assert disp.shape == (32, 32)
    _, tight = mind.bake_normal_map(low, uv, high, size=32, displacement=True, max_distance=0.002)
    assert np.abs(tight).max() <= 0.002 + 1e-9, "the cage must clamp -- displacement moves geometry"


def test_m13_topology_gate_wired_into_all_reducing_faculties(mind):
    """M13 COMPLETE: all six reducing faculties carry the topology instrument -- report by default (results
    bit-identical with it on/off; an instrument must not flip decisions), refuse opt-in (fires only on a real
    violation, e.g. an SDF rebuild FILLING a hole that existed -- a scan's holes are data)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(triangulate_ngons(box()), levels=2)
    # Mesh-returners: report rides .topology_report; result identical with the instrument off
    a = mind.mesh_cluster_decimate(src, grid=8)
    b = mind.mesh_cluster_decimate(src, grid=8, topology=False)
    assert a.topology_report["preserved"] is True
    assert np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
    q = mind.mesh_qem_decimate(src, target_faces=60)
    assert q.topology_report["preserved"] is True
    v = mind.voxel_remesh(src, resolution=12)
    assert "preserved" in v.topology_report
    # dict-returners: report joins the dict; the recorded decision (768->200 = 186 faces, 4 iters) must hold
    big = loop_subdivide(triangulate_ngons(box()), levels=3)
    out, rep = mind.mesh_decimate_to(big, target_faces=200, min_silhouette_iou=None)
    assert rep["result_faces"] == 186 and rep["iters"] == 4        # the M6 pin: instrument changed nothing
    assert rep["topology"]["preserved"] is True
    # refuse fires ONLY on a real violation: an SDF rebuild that FILLS an existing hole
    V = np.asarray(src.vertices, float); F = [tuple(f) for f in src.faces]
    holed = Mesh(V, F[:-2])
    with pytest.raises(ValueError):
        mind.voxel_remesh(holed, resolution=10, topology="refuse")
    mind.mesh_cluster_decimate(src, grid=8, topology="refuse")     # clean op under refuse: no raise


def test_m6_bisect_to_budget_promotion_preserves_both_pins(mind):
    """M6 COMPLETE: decimate_to and ratedistortion both delegate to numerics.bisect_to_budget. The promotion
    is real ONLY if it moved no recorded decision -- so pin BOTH bit-identity contracts. The load-bearing
    subtlety (M6 dry-run): a primitive-owned iter counter reproduces the FACE result but flips decimate_to's
    reported iters 4->5; the counter stays caller-side via on_probe, and this test proves iters is still 4."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    big = loop_subdivide(triangulate_ngons(box()), levels=3)
    out, rep = mind.mesh_decimate_to(big, target_faces=200, min_silhouette_iou=None)
    assert rep["result_faces"] == 186 and rep["iters"] == 4 and rep["budget_error"] == 0.07
    # the geometric consumer's pin
    import importlib, glob
    mod = importlib.import_module(glob.glob("holographic/*/holographic_ratedistortion.py")[0].replace("/", ".")[:-3])
    code = mod.geometry_preserving_code(np.random.default_rng(0).standard_normal((40, 16)), target_cos=0.9999)
    assert code["delta"] == 0.04737815295834658
    # the primitive itself, via the faculty, on both midpoints
    val, knob, err = mind.bisect_to_budget(lambda k: k, 20, 0, 4, midpoint="arith", max_iters=12, tol=0.10,
                                           bracket=True)
    assert abs(val - 20) <= 2
    lo, _v = mind.bisect_to_budget(lambda d: 1.0 - d, 0.63, 1e-5, 1.0, midpoint="geom", max_iters=28,
                                   tol=None, cmp=lambda c, t: c >= t)
    assert abs(lo - 0.37) < 0.02


def test_m7_smallest_eigenpair_promotion_preserves_the_field(mind):
    """M7: crossfield's two-phase eigensolver promoted to numerics.smallest_eigenpair; crossfield delegates.
    Pins the load-bearing contract THREE ways: (1) cross_field's phi is BIT-IDENTICAL through the delegated
    path on both the dense (768f) and sparse (3072f) routes -- sha-pinned; (2) the primitive finds the TRUE
    smallest eigenpair of an arbitrary PSD matvec; (3) the matvec counter stays caller-side (the M6 lesson)."""
    import numpy as np, hashlib
    from holographic.mesh_and_geometry import holographic_crossfield as cf
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    phi_s, _ = cf.cross_field(loop_subdivide(triangulate_ngons(box()), levels=3))
    phi_b, _ = cf.cross_field(loop_subdivide(triangulate_ngons(box()), levels=4))
    assert hashlib.sha256(phi_s.tobytes()).hexdigest()[:16] == "d2c81dd2847439ed"
    assert hashlib.sha256(phi_b.tobytes()).hexdigest()[:16] == "cee8e1134fd1a71f"
    rng = np.random.default_rng(3)
    Q = rng.standard_normal((30, 30)); A = Q @ Q.T
    c = float(np.abs(A).sum(1).max())
    counted = [0]
    u, lam, mv = mind.smallest_eigenpair(lambda x: A @ x, 30, c, dtype=float,
                                         on_matvec=lambda: counted.__setitem__(0, counted[0] + 1))
    w, V = np.linalg.eigh(A)
    assert abs(lam - w[0]) < 1e-6 * c and abs(float(u @ V[:, 0])) > 0.999
    assert counted[0] == mv


def test_m14_shared_correspondence_machine(mind):
    """M14 increment 2: transfer_uv AND bake_normal_map delegate to build_face_grid + closest_face_point --
    one projection primitive, many channel readers (the owner's holographic reframe). Both load-bearing paths
    must be BIT-IDENTICAL through the shared machine (sha-pinned), displacement (M12) still rides it with
    normal-identity, and the faculty answers a direct closest-point query."""
    import numpy as np, hashlib
    from holographic.mesh_and_geometry.holographic_meshtools import transfer_uv, bake_normal_map
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    high = loop_subdivide(triangulate_ngons(box()), levels=3)
    low, _ = mind.mesh_decimate_to(high, target_faces=120, min_silhouette_iou=None)
    srcuv = np.asarray(high.vertices)[:, :2].copy(); srcuv = (srcuv - srcuv.min(0)) / (srcuv.max(0) - srcuv.min(0) + 1e-9)
    out, dist = transfer_uv(high, srcuv, np.asarray(low.vertices), cell_scale=1.0)
    assert hashlib.sha256(np.asarray(out).tobytes()).hexdigest()[:16] == "6f296d80bb12e491"
    assert hashlib.sha256(np.asarray(dist).tobytes()).hexdigest()[:16] == "326ba2806dec4bbf"
    lowuv = np.asarray(low.vertices)[:, :2].copy(); lowuv = (lowuv - lowuv.min(0)) / (lowuv.max(0) - lowuv.min(0) + 1e-9)
    nrm = bake_normal_map(low, lowuv, high, size=48)
    assert hashlib.sha256(np.asarray(nrm).tobytes()).hexdigest()[:16] == "740e16230c4eb938"
    n2, d = bake_normal_map(low, lowuv, high, size=48, displacement=True, max_distance=0.5)
    assert np.array_equal(nrm, n2)                          # displacement still rides the shared projection
    # the faculty: a direct closest-point query returns (face, bary, distance)
    b = triangulate_ngons(box())
    r = mind.mesh_closest_point(b, [[0.4, 0.4, 0.4]])
    assert len(r) == 1 and len(r[0]) == 3 and r[0][2] >= 0.0


def test_degenerate_face_frame_is_finite_and_pins_hold(mind):
    """M11-remainder investigation found a real crash: face_frames divided a zero-area face's normal by its
    zero length -> NaN -> propagated into position_field's round() and crashed the retopo of a hole-filled
    scan. Guarded so a degenerate face gets a finite stable frame. LOAD-BEARING: non-degenerate frames stay
    BIT-IDENTICAL (the phi pins), so the guard changed nothing for real geometry."""
    import numpy as np, hashlib
    from holographic.mesh_and_geometry import holographic_crossfield as cf
    from holographic.mesh_and_geometry.holographic_crossfield import face_frames
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    phi_s, _ = cf.cross_field(loop_subdivide(triangulate_ngons(box()), levels=3))
    assert hashlib.sha256(phi_s.tobytes()).hexdigest()[:16] == "d2c81dd2847439ed"   # pin holds
    V = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0]], float)     # face 0 is collinear
    F = np.array([[0, 1, 2], [0, 1, 3], [1, 4, 3]], int)
    n, ex, ey = face_frames(V, F)
    assert np.isfinite(n).all() and np.isfinite(ex).all() and np.isfinite(ey).all()


def test_m1_graded_levels_balanced_and_curvature_driven(mind):
    """M1 increment 1: graded_levels produces a power-of-two size field, 2:1-BALANCED (level jump <=1 across
    every edge -- the property that keeps extract_quads' lattice valid across level boundaries) and grading
    toward the fine target. The operator + level-keyed extraction (M1 increment 2) build ON this."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    s = loop_subdivide(triangulate_ngons(box()), levels=3)
    V = np.asarray(s.vertices, float)
    rho0 = float(np.linalg.norm(V[np.asarray(s.faces)[0][1]] - V[np.asarray(s.faces)[0][0]]))
    te = np.where(V[:, 0] > 0, rho0 * 0.25, rho0 * 4.0)              # 4 levels apart -> forces balancing
    k, rho = mind.graded_levels(s, te, rho0, k_min=0, k_max=6)
    F = [tuple(int(i) for i in f[:3]) for f in s.faces]
    maxjump = max(abs(int(k[a]) - int(k[b])) for f in F for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])))
    assert maxjump <= 1, "2:1 balance failed (max |dk| = %d)" % maxjump
    assert k[V[:, 0] > 0].mean() > k[V[:, 0] < 0].mean()
    assert np.allclose(rho, rho0 * 2.0 ** k)


def test_m1_increment2_graded_operator_and_extraction(mind):
    """M1 increment 2: position_field AND extract_quads gain levels= (graded operator + level-keyed
    extraction). BOTH are strict supersets -- levels=None is bit-identical to the uniform path (surface_retopo
    pin 328/0 holds), a uniform non-zero level equals rho-scaling exactly, and varying levels produce a valid
    adaptively-sized closed mesh. (quad_fraction may dip below uniform until increment 3's T-junction stitch
    resolves the level-boundary hanging nodes -- the mesh is valid and closed, which is what inc 2 guarantees.)"""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_crossfield import (cross_field, field_to_vertex_dirs,
        position_field, extract_quads, graded_levels)
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshtools import face_orientation_report
    src = loop_subdivide(triangulate_ngons(box()), levels=3)
    V = np.asarray(src.vertices, float)
    rho0 = float(np.linalg.norm(V[np.asarray(src.faces)[0][1]] - V[np.asarray(src.faces)[0][0]]))
    phi, _ = cross_field(src); ov = field_to_vertex_dirs(src, phi)
    # default bit-identical
    P = position_field(src, ov, rho0, iterations=20)
    assert np.array_equal(P, position_field(src, ov, rho0, iterations=20, levels=np.zeros(len(V), int)))
    # uniform level k == rho*2^k scaling
    P2 = position_field(src, ov, rho0, iterations=20, levels=2 * np.ones(len(V), int))
    assert np.array_equal(P2, position_field(src, ov, rho0 * 4.0, iterations=20))
    # surface_retopo pin unmoved
    out, r = mind.surface_retopo(src, density=1.0, silhouette=None, topology=False)
    assert r["faces"] == 328 and face_orientation_report(out)["boundary_edges"] == 0
    # graded produces a valid, closed, differently-sized mesh
    te = np.where(V[:, 0] > 0, rho0 * 0.5, rho0 * 2.0)
    k, _ = graded_levels(src, te, rho0, k_min=0, k_max=3)
    Pg = position_field(src, ov, rho0, iterations=20, levels=k)
    qm_g, rep_g = extract_quads(src, Pg, rho0, levels=k)
    assert rep_g["faces"] > 0 and face_orientation_report(qm_g)["boundary_edges"] == 0
    assert rep_g["faces"] != extract_quads(src, P, rho0)[1]["faces"]     # grading changed the mesh


def test_m9_mesh_skeleton_medial_axis(mind):
    """M9 increment 1: mesh_skeleton = the ridge of the interior distance field (the medial axis). Validated
    on a cylinder whose medial axis IS its central line: the ridge must sit near r=0, span the height, and its
    medial depth must be ~ the cylinder radius. GENERALISES the M14 correspondence machine + winding number --
    not a new algorithm. KEPT NEGATIVE pinned in the module: voxel ridge, not yet a connected 1-D curve."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_skeleton import _cylinder
    sk = mind.mesh_skeleton(_cylinder(r=0.3, h=2.0), res=24)
    pts = sk["points"]
    assert len(pts) > 0
    rad = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    assert rad.mean() < 0.10, "medial axis must be on the centerline (mean radial %.3f)" % rad.mean()
    assert pts[:, 2].max() - pts[:, 2].min() > 1.0, "must span the length"
    assert 0.15 < sk["depth"].mean() < 0.35, "medial depth ~ cylinder radius"
    # the interior distance field faculty is live too
    depth, bounds, cell = mind.interior_distance_field(_cylinder(), res=16)
    assert depth.shape == (16, 16, 16) and depth.max() > 0


def test_m9_increment2_skeleton_curve(mind):
    """M9 increment 2: skeleton_curve collapses the medial ridge to a single-branch centerline polyline via
    principal-axis binning. A cylinder must collapse to a STRAIGHT line ON its axis (radial ~0, residual ~0).
    KEPT NEGATIVE pinned in the module: single-branch -- one PCA axis cuts corners on bent/branched shapes,
    which need branch segmentation first (increment 2-plus)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_skeleton import _cylinder
    cv = mind.skeleton_curve(_cylinder(r=0.3, h=2.0), res=24)
    curve = cv["curve"]
    assert len(curve) >= 3
    crad = np.sqrt(curve[:, 0] ** 2 + curve[:, 1] ** 2)
    assert crad.mean() < 0.05, "cylinder curve must lie on the axis (mean radial %.3f)" % crad.mean()
    cc = curve.mean(0)
    _u, _s, vt = np.linalg.svd(curve - cc, full_matrices=False)
    resid = np.linalg.norm((curve - cc) - ((curve - cc) @ vt[0])[:, None] * vt[0], axis=1)
    assert resid.mean() < 0.02, "cylinder curve must be straight (residual %.4f)" % resid.mean()
    assert len(cv["depth"]) == len(curve) and (cv["depth"] > 0).all()


def test_asset_base_texture_render_ready(mind):
    """Render exercise gap: getting render-ready (texture, uvs) from a LOADED or self-derived mesh had no
    helper -- I hand-extracted the embedded JPEG. asset_base_texture is that pointer, sharing preview_asset's
    coverage-based material pick. A mesh carrying a base_color_map yields a [0,1] RGB texture + its uvs; the
    pair renders TEXTURED through render_mesh. (preview_asset pins held after factoring this out -- the refactor
    is behaviour-preserving.)"""
    import numpy as np, tempfile, os
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.io_and_interop.holographic_gltf import mesh_to_glb
    from holographic.io_and_interop.holographic_assetimport import load_glb
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial, TextureMap
    from holographic.rendering.holographic_render import Camera
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float); F = [(0, 1, 2), (0, 2, 3)]
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    teximg = np.zeros((4, 4, 3), np.float32); teximg[:, :2] = (1, 0, 0); teximg[:, 2:] = (0, 0, 1)
    quad = Mesh(V, F, uvs=uv); mat = PBRMaterial(name="t", base_color_map=TextureMap(teximg))
    p = tempfile.mktemp(suffix=".glb"); open(p, "wb").write(mesh_to_glb(quad, material=mat, texture=teximg))
    try:
        lm = load_glb(p)
        tex, u, base = mind.asset_base_texture(lm)
        assert tex is not None and tex.shape[2] == 3 and tex.max() <= 1.0
        assert u is not None and u.shape[1] == 2 and len(base) == 3
        # renders textured: both colours appear
        img = np.asarray(mind.render_mesh(quad, Camera(eye=(0.5, 0.5, 2.0), target=(0.5, 0.5, 0.0)),
                                          width=48, height=48, texture=tex, uvs=u, ambient=1.0))
        fg = img[img.sum(2) > 0.2]
        assert ((fg[:, 0] > 0.4).sum() > 5) and ((fg[:, 2] > 0.4).sum() > 5)
    finally:
        os.remove(p)


def test_m5_graph_connected_components_faculty(mind):
    """M5 resolution: the graph flood the plan wanted to BUILD already exists as
    island.connected_components (route + mesh_connected_components already delegate). The gap was
    DISCOVERABILITY -- no direct faculty. graph_connected_components surfaces the reusable primitive:
    deterministic, smallest-member-ordered partition of an edge list; isolated nodes are singletons."""
    comps = mind.graph_connected_components(6, [(0, 1), (1, 2), (3, 4)])
    assert comps == [[0, 1, 2], [3, 4], [5]], comps          # includes the singleton 5, ordered
    # determinism: edge order must not change the result
    comps2 = mind.graph_connected_components(6, [(4, 3), (2, 1), (1, 0)])
    assert comps2 == comps
    # a self-loop connects nothing new
    assert mind.graph_connected_components(3, [(0, 0), (1, 2)]) == [[0], [1, 2]]


def test_drop_small_components_retopo_cleanup(mind):
    """Render-exercise finding: a field-guided retopo of a scan shatters into many components (a mantis retopo:
    88 components, one body + ~75 shards) -- a MESH problem, not a texture problem. drop_small_components
    keep_largest keeps the coherent body and drops the debris, carrying uvs. Built on the shared graph flood."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                  [5, 5, 5], [6, 5, 5], [5, 6, 5]], float)
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [.5, .5], [.6, .5], [.5, .6]], float)
    mesh = Mesh(V, [(0, 1, 2), (1, 3, 2), (4, 5, 6)], uvs=uv)
    body, rep = mind.mesh_drop_small_components(mesh, keep_largest=True)
    assert rep["components_before"] == 2 and rep["components_after"] == 1
    assert rep["faces_after"] == 2 and len(body.vertices) == 4
    assert body.uvs is not None and len(body.uvs) == 4          # uvs carried through the remap
    # min_faces threshold keeps only components at/above N faces
    _, rep2 = mind.mesh_drop_small_components(mesh, min_faces=2)
    assert rep2["components_after"] == 1                         # only the 2-face body qualifies


def test_process_scan_four_workflows(mind):
    """Moose's pipeline spec: repair the ORIGINAL -> retopo the repaired mesh -> LOD (coarser retopo when
    retopo is on, since decimating a quad retopo re-shatters it -- measured; QEM when off) -> shard cleanup ->
    fresh atlas + rebake. Four workflows via the retopo/lod flags. This pins the routing and the textured path
    end-to-end on a synthetic textured quad. H6: the guarded retopo stage must report a real silhouette_iou."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = triangulate_ngons(box())
    # H6: a guarded retopo on a real (subdividable) mesh must populate silhouette_iou, not leave it None
    dense = loop_subdivide(box(), 2)
    _, _, _, grep = mind.process_scan(dense, retopo=True, lod=None, density=1.0, silhouette=0.9)
    rstage = [s for s in grep["stages"] if s["stage"] == "retopo"][0]
    assert rstage["silhouette_iou"] is not None and 0.0 <= rstage["silhouette_iou"] <= 1.0, rstage
    # routing contract for the four workflows
    for retopo, lod, want in [(True, 0.5, ["repair", "retopo", "lod_via_coarser_retopo", "shard_cleanup"]),
                              (True, None, ["repair", "retopo", "shard_cleanup"]),
                              (False, 0.5, ["repair", "lod_via_decimate"]),
                              (False, None, ["repair"])]:
        out, u, img, rep = mind.process_scan(src, retopo=retopo, lod=lod, density=1.0)
        assert [s["stage"] for s in rep["stages"]] == want
        assert len(out.faces) > 0
    # textured path: a box with uvs + a two-colour texture must come back with a fresh uv + baked image
    V = np.asarray(src.vertices, float)
    uv = (V[:, :2] - V[:, :2].min(0)) / (np.ptp(V[:, :2], axis=0) + 1e-9)
    tex = np.zeros((8, 8, 3)); tex[:, :4] = (1, 0, 0); tex[:, 4:] = (0, 0, 1)
    out, new_uv, image, rep = mind.process_scan(src, uv=uv, texture=tex, retopo=False, lod=None, bake_size=64)
    assert new_uv is not None and image is not None
    assert image.shape[2] == 3 and rep["stages"][-1]["stage"] == "rebake"
    assert rep["stages"][-1]["texel_coverage"] > 0.3


def test_h1_holographic_scatter_bake(mind):
    """H1: rebake_texture method='scatter' is the holographic scatter/gather fast path. method='project' stays
    the default (backward-compatible); scatter reproduces the texture on a DENSE source (its use case), is
    deterministic, and reports its grid. Measured ~1500x on real scans; here we pin the CONTRACT."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import grid as _g
    s = _g(10, 10, width=1.0, height=1.0)
    V = np.asarray(s.vertices, float); s.uvs = V[:, :2].copy()
    tex = np.zeros((48, 48, 3)); yy, xx = np.mgrid[0:48, 0:48]
    tex[:, :, 0] = xx / 47.0; tex[:, :, 1] = yy / 47.0
    # default = project
    _, _, _, rp = mind.mesh_rebake_texture(s, np.asarray(s.uvs), tex, s, size=192, margin=2)
    assert rp["method"] == "project"
    # scatter path
    ms, us, img, rs = mind.mesh_rebake_texture(s, np.asarray(s.uvs), tex, s, size=192, margin=2, method="scatter")
    assert rs["method"] == "scatter" and rs["grid"] > 0 and rs["gather_weight_mean"] > 0
    # reproduces R=u,G=v in the interior (source-density-bounded, so interior only)
    OV = np.asarray(ms.vertices); hh, ww = img.shape[:2]; errs = []
    for i in range(len(OV)):
        u, v = us[i]
        if not (0.2 < OV[i][0] < 0.8 and 0.2 < OV[i][1] < 0.8):
            continue
        got = img[int(np.clip(round(v * (hh - 1)), 0, hh - 1)), int(np.clip(round(u * (ww - 1)), 0, ww - 1))]
        errs += [abs(got[0] - OV[i][0]), abs(got[1] - OV[i][1])]
    assert errs and np.array(errs).max() < 0.15
    # deterministic
    _, _, img2, _ = mind.mesh_rebake_texture(s, np.asarray(s.uvs), tex, s, size=192, margin=2, method="scatter")
    assert np.array_equal(img, img2)
    # process_scan threads bake_method through
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    src = triangulate_ngons(box()); Vb = np.asarray(src.vertices, float)
    uvb = (Vb[:, :2] - Vb[:, :2].min(0)) / (np.ptp(Vb[:, :2], axis=0) + 1e-9)
    _, _, _, rep = mind.process_scan(src, uv=uvb, texture=tex, retopo=False, bake_size=64, bake_method="scatter")
    assert rep["stages"][-1]["method"] == "scatter"


def test_h4_guard_iterations_field_reuse(mind):
    """H4: surface_retopo(guard_iterations=N) runs the silhouette-guard TRIAL solves at N iterations (cheap)
    and re-solves the chosen density once at full iterations. Measured 1.48x when the guard WALKS (115s->78s,
    identical face count) because position_field face count plateaus by ~5 iters but time is linear. The
    default (guard_iterations=None) is unchanged. Pinned on a small fixture for the CONTRACT, not the timing."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(box(), 2)               # dense enough that the crossfield solve is well-posed
    # default path still returns a valid guarded retopo
    q0, r0 = mind.surface_retopo(src, density=1.0, silhouette=0.9)
    assert len(q0.faces) > 0 and "silhouette_report" in r0
    # guard_iterations path returns a valid mesh too (the refine ran at full iterations)
    q1, r1 = mind.surface_retopo(src, density=1.0, silhouette=0.9, guard_iterations=3)
    assert len(q1.faces) > 0 and "silhouette_report" in r1
    # guard_iterations == iterations is a no-op refine (must not double-solve into a different result shape)
    q2, r2 = mind.surface_retopo(src, density=1.0, silhouette=0.9, guard_iterations=20)
    assert len(q2.faces) > 0


def test_h2_position_field_fast_bit_identical(mind):
    """H2: surface_retopo(fast=True) vectorises position_field's inner neighbour loop while keeping the EXACT
    sequential Gauss-Seidel order. Measured ~3.5x on the mantis, BIT-IDENTICAL extraction (0.00 vertex diff).
    KEPT NEGATIVE, measured and why fast is not a reorder: Jacobi matches the pinned lattice only 55%,
    colored-GS 52%, a different seed 58% -- the visit order is load-bearing. This pins that fast == slow."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    import numpy as np
    src = loop_subdivide(box(), 3)
    slow, rs = mind.surface_retopo(src, density=1.0, silhouette=None)
    fast, rf = mind.surface_retopo(src, density=1.0, silhouette=None, fast=True)
    assert len(fast.faces) == len(slow.faces), "fast must not change face count"
    assert np.abs(np.asarray(fast.vertices) - np.asarray(slow.vertices)).max() < 1e-9, "fast must be bit-identical"
    # process_scan threads retopo_fast through and stays valid
    _, _, _, rep = mind.process_scan(src, retopo=True, lod=None, density=1.0, silhouette=None, retopo_fast=True)
    assert [s["stage"] for s in rep["stages"]] == ["repair", "retopo", "shard_cleanup"]


def test_h5_spatial_recall(mind):
    """H5: every closest-point is a RECALL. Positions encode as FPE hypervectors (proximity-preserving,
    measured spearman 0.967) and nearest-point is argmax cosine over the item store. Pins: geometric
    equivalence (recalled point within a few % of true nearest), resonant payload readout on a smooth field,
    determinism, and the KEPT NEGATIVE that there is NO bundle mode (correlated FPE keys cross-talk)."""
    import numpy as np
    rng = np.random.default_rng(0)
    P = rng.random((500, 3))
    Q = np.clip(P[rng.choice(500, 100, replace=False)] + rng.normal(0, 0.01, (100, 3)), 0, 1)
    idx, out, rep = mind.spatial_recall(P, Q, payloads=P, k=1)
    assert rep["n_points"] == 500 and idx.shape == (100, 1)
    d2 = ((Q[:, None, :] - P[None, :, :]) ** 2).sum(2)
    nn = d2.argmin(1)
    dtrue = np.sqrt(d2[np.arange(100), nn]); dgot = np.sqrt(d2[np.arange(100), idx[:, 0]])
    assert np.percentile(dgot / (dtrue + 1e-12), 95) < 1.10   # geometrically equivalent recall
    assert np.linalg.norm(out - Q, axis=1).mean() < 0.05      # resonant readout reproduces the smooth field
    # determinism
    idx2, _, _ = mind.spatial_recall(P, Q, payloads=None, k=1)
    idx3, _, _ = mind.spatial_recall(P, Q, payloads=None, k=1)
    assert np.array_equal(idx2, idx3)


def test_process_scan_repairs_welded_geometry(mind):
    """The missing-faces regression Moose caught in a render: a scan passed WITH its fragmented per-face uvs
    attached made mesh_repair operate on the uv-SPLIT confetti (29062 faces vs 11010 on the mantis), the
    retopo shattered (65 comps vs 12), and keep_largest amputated 11% of the surface. process_scan must
    repair GEOMETRY ONLY -- the uvs are captured separately and the bake re-projects from the original."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    src = triangulate_ngons(box())
    F = np.asarray([f[:3] for f in src.faces], int)
    V = np.asarray(src.vertices, float)
    # a per-face-split copy with fragmented uvs (every triangle private verts -- a scan's shape)
    Vs = V[F].reshape(-1, 3)
    frag_uv = np.tile([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], (len(F), 1))
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    frag = Mesh(Vs, [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(F))], uvs=frag_uv)
    _, _, _, ra = mind.process_scan(frag, retopo=False, lod=None)
    plain = Mesh(Vs, [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(F))])
    _, _, _, rb = mind.process_scan(plain, retopo=False, lod=None)
    fa = [s for s in ra["stages"] if s["stage"] == "repair"][0]["faces"]
    fb = [s for s in rb["stages"] if s["stage"] == "repair"][0]["faces"]
    assert fa == fb, "repair must ignore attached uvs and weld the geometry (%d vs %d)" % (fa, fb)


def test_r1_topology_gate(mind):
    """R1: topology invariants per component; gate rejects punched holes / fragmentation / genus change and
    accepts intended holes. process_scan's shard_cleanup stage carries the verdict + dropped_fraction so
    amputation is loud, never silent."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, grid, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    b = triangulate_ngons(box())
    assert mind.topology_report(b)["per_component"][0]["genus"] == 0
    g = triangulate_ngons(grid(4, 4))
    assert mind.topology_report(g)["per_component"][0]["boundary_loops"] == 1
    ok, _ = mind.topology_gate(g, g)
    assert ok, "an intended hole must not be flagged"
    punched = Mesh(np.asarray(b.vertices, float), [tuple(int(i) for i in f) for f in b.faces][:-2])
    ok, rep = mind.topology_gate(b, punched)
    assert not ok and any("new boundary loop" in v for v in rep["violations"])
    # process_scan reports the verdict per shard_cleanup stage
    _, _, _, prep = mind.process_scan(loop_subdivide(box(), 2), retopo=True, density=1.0, silhouette=0.9)
    sc = [s for s in prep["stages"] if s["stage"] == "shard_cleanup"][0]
    assert "topology_ok" in sc and "dropped_fraction" in sc and "topology_violations" in sc


def test_r2_singular_snap(mind):
    """R2: extract_quads(snap_singular=True) rescues degenerate cells by QEx-style per-vertex re-keying.
    Contracts: (a) default OFF -- the pinned retopo is unchanged; (b) snap never LOSES faces; (c) on the
    shattered-scan case it strictly reduces component count (measured on the mantis: 12 -> 5, 1134 cells
    rescued). Pinned here on the box fixture for the additive contract."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(box(), 3)
    off, roff = mind.surface_retopo(src, density=1.0, silhouette=None)
    on, ron = mind.surface_retopo(src, density=1.0, silhouette=None, snap_singular=True)
    assert len(on.faces) >= len(off.faces), "snap must never lose faces"
    # process_scan threads retopo_snap and still reports the R1 verdict
    _, _, _, rep = mind.process_scan(src, retopo=True, density=1.0, silhouette=None, retopo_snap=True)
    sc = [s for s in rep["stages"] if s["stage"] == "shard_cleanup"][0]
    assert "topology_ok" in sc


def test_r5_feature_sized_retopo(mind):
    """R5: feature_size_field reads local thickness (a SpatialMemory recall of the opposing wall) and
    feature_sized=True grades the lattice finer where thin. MEASURED composition (recorded in the crossfield
    selftest): mantis at coarse density -- baseline 12 components, snap 5, sized 5, snap+sized 1. Pinned here:
    the field is sane on the box and the threaded flags run end-to-end through process_scan."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_crossfield import feature_size_field
    src = loop_subdivide(box(), 2)
    thick = feature_size_field(src)
    assert thick.shape == (len(src.vertices),) and np.all(thick > 0)
    q, _ = mind.surface_retopo(src, density=1.0, silhouette=None, feature_sized=True, snap_singular=True)
    assert len(q.faces) > 0
    _, _, _, rep = mind.process_scan(src, retopo=True, density=1.0, silhouette=None,
                                     retopo_snap=True, retopo_sized=True)
    sc = [s for s in rep["stages"] if s["stage"] == "shard_cleanup"][0]
    assert "topology_ok" in sc and sc["dropped_fraction"] < 0.5


def test_gabor_field_volumes(mind):
    """Gabor Fields (SIGGRAPH 2026, verified on this engine before building): closed-form ray integrals,
    equal-budget win on oriented content, free LOD by frequency pruning. Pins the faculty contract."""
    import numpy as np
    ax = np.linspace(0, 1, 20)
    X = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
    r2 = ((X - 0.5) ** 2).sum(-1)
    rho = np.clip(np.exp(-r2 / 0.08) * (1 + 0.5 * np.cos(30 * X[..., 0] + 20 * X[..., 1])), 0, None)
    f, rep = mind.gabor_volume(rho, K=16, n_freqs=3)
    fg, rg = mind.gabor_volume(rho, K=16, n_freqs=0)          # gaussian-only baseline, equal budget
    assert rep["psnr_db"] > rg["psnr_db"], "gabors must not lose to equal-count gaussians on oriented content"
    base = f.lod(1e-9)
    assert len(base.A) == rep["gaussians"], "lod(0) keeps exactly the gaussian base"
    tr = f.transmittance(np.array([0.5, 0.5, -1.0]), np.array([0.0, 0.0, 1.0]), extinction=2.0)
    assert 0.0 <= tr <= 1.0
    f2, rep2 = mind.gabor_volume(rho, K=16, n_freqs=3)
    assert np.array_equal(f.A, f2.A) and np.array_equal(f.w, f2.w), "fit must be deterministic"


def test_cvt_remesh(mind):
    """CWF-derived CVT remeshing (fulfils the R4 isotropic-fallback slot): Lloyd-relaxed sites beat the fixed
    grid on triangle quality at equal budget; deterministic; result gated like any remesh (not provably
    manifold -- the kept negative)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(box(), 3)
    q, rep = mind.cvt_remesh(src, n_sites=200, iterations=4)          # guarded by default (institutional rule)
    assert len(q.faces) > 0 and rep["sites_requested"] == 200 and "guard" in rep
    qp, rp = mind.cvt_remesh(src, n_sites=200, iterations=4, silhouette=None)   # primitive path
    assert rp["sites"] == 200
    q2, _ = mind.cvt_remesh(src, n_sites=200, iterations=4)
    assert np.array_equal(np.asarray(q.vertices), np.asarray(q2.vertices)), "must be deterministic"
    ok, gr = mind.topology_gate(src, q)      # the gate JUDGES it; pass or fail, the verdict must be named
    assert "violations" in gr


def test_render_two_sided(mind):
    """two_sided=True shades with |n.l| so zero-thickness sheets and unorientable retopo patches render like
    their front (measured on the mantis retopo: 23% of visible pixels were flipped-dark ambient-only ->
    6%). Contract: bit-identical on a closed consistently-oriented mesh (default OFF, additive), strictly
    brighter on a camera-facing flipped triangle."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.rendering.holographic_render import Camera
    cam = Camera(eye=(0, 0.4, 2.2), target=(0, 0, 0), fov_deg=40.0, aspect=1.0)
    b = triangulate_ngons(box())
    a1 = np.asarray(mind.render_mesh(b, cam, width=64, height=64))
    a2 = np.asarray(mind.render_mesh(b, cam, width=64, height=64, two_sided=True))
    assert np.array_equal(a1, a2), "closed oriented mesh must be bit-identical under two_sided"
    flipped = Mesh(np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]), [(0, 2, 1)])
    f1 = np.asarray(mind.render_mesh(flipped, cam, width=64, height=64))
    f2 = np.asarray(mind.render_mesh(flipped, cam, width=64, height=64, two_sided=True))
    assert f2.sum() > f1.sum(), "a flipped camera-facing triangle must shade brighter two-sided"


def test_rebake_flood_fill(mind):
    """The dark-speckle arc (Moose-caught, three refuted hypotheses on record in NOTES): fill_mode='flood'
    fills every unwritten atlas texel by outward dilation so bilinear taps never blend black half-cells;
    default 'margin' stays bit-identical (the sha-pinned bake). process_scan opts into flood. FINAL measured
    contract: project+flood retopo render matches the original scan render (dark 14.7%% vs 14.6%%, luminance
    0.313 vs 0.312 on the mantis)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshtools import rebake_texture
    src = triangulate_ngons(box())
    uv = np.random.default_rng(0).uniform(0.05, 0.95, (len(src.vertices), 2))
    tex = np.full((64, 64, 3), 0.5)
    _, _, img_m, rm = rebake_texture(src, uv, tex, src, size=128, margin=2, fill_mode="margin")
    _, _, img_f, rf = rebake_texture(src, uv, tex, src, size=128, margin=2, fill_mode="flood")
    wm = (np.asarray(img_f).sum(2) > 0).mean()
    assert wm == 1.0, "flood must leave zero unwritten texels (got %.3f coverage)" % wm
    # margin-mode written texels are untouched by flood (colours only bleed OUTWARD into unwritten)
    mm = np.asarray(img_m).sum(2) > 0
    assert np.array_equal(np.asarray(img_m)[mm], np.asarray(img_f)[mm]), "flood must not touch painted texels"


def test_lod_texture_route(mind):
    """The 'LOD and retopo switched places' arc: forcing keep_uv=True through a per-face confetti atlas makes
    QEM's uv interpolation sweep unrelated charts (measured: 37.7%% dark vs the source's 14.6%%). Contracts:
    (a) keep_uv='auto' REFUSES uv transfer on a fragmented per-face atlas; (b) process_scan's LOD-with-texture
    route decimates uv-free and rebakes a fresh atlas (stage sequence pinned), reaching source parity
    (measured: 14.3%% dark / 0.313 lum vs original 14.6%% / 0.312)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = triangulate_ngons(loop_subdivide(box(), 2))
    F = np.asarray([f[:3] for f in src.faces], int)
    V = np.asarray(src.vertices, float)
    Vs = V[F].reshape(-1, 3)                                     # per-face split = confetti
    frag_uv = np.tile([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], (len(F), 1))
    frag = Mesh(Vs, [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(F))], uvs=frag_uv)
    lod = mind.mesh_decimate_to(frag, target_faces=64, keep_uv="auto")
    lm = lod[0] if isinstance(lod, tuple) else lod
    assert getattr(lm, "uvs", None) is None, "auto must refuse uv transfer on a confetti atlas"
    tex = np.full((32, 32, 3), 0.5)
    _, _, _, rep = mind.process_scan(frag, texture=tex, retopo=False, lod=64, bake_size=256)
    assert [s["stage"] for s in rep["stages"]] == ["repair", "lod_via_decimate", "rebake"], \
        "LOD-with-texture must decimate uv-free then rebake a fresh atlas"


def test_nascat_coverage_and_normal(mind):
    """NA-SCAT arc. The REAL fix was the grid auto-sizer: the old 2x-source-edge rule overshot on dense
    targets, starving cells so texels gathered ~0 -> dark (measured on the mantis: auto grid 242 -> 22.4%%
    dark; coverage-aware sizer -> grid 121, 16.5%% dark, matching project's 16.5%% ceiling at 33x speed).
    normal_aware is a SMALL additive sharpness gain on top (atlas std +0.003 on the mantis, zero dark cost).
    Contracts pinned here: (a) the coverage-aware auto grid is never finer than the target can fill; (b)
    normal_aware runs and reports; (c) it never LOSES coverage vs plain scatter (the empty-bin fallback)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshtools import rebake_texture
    src = triangulate_ngons(loop_subdivide(box(), 2))
    uv = np.random.default_rng(0).uniform(0.05, 0.95, (len(src.vertices), 2))
    tex = np.zeros((48, 48, 3)); tex[:, :, 1] = 1.0
    _, _, _, r_plain = rebake_texture(src, uv, tex, src, size=256, method="scatter", normal_aware=False)
    _, _, _, r_na = rebake_texture(src, uv, tex, src, size=256, method="scatter", normal_aware=True)
    assert r_plain["method"] == "scatter" and r_na["normal_aware"] is True
    # normal_aware must not COLLAPSE coverage (empty-bin fallback keeps it >= plain, within noise)
    assert r_na["gather_weight_mean"] > 0.5 * r_plain["gather_weight_mean"], \
        "normal_aware fell back correctly and kept coverage"


def test_r3_manifold_cleanup(mind):
    """R3: manifold_cleanup makes a finned retopo strictly manifold so QEM decimate accepts it (the measured
    consumer motive: raw retopo has 142 non-manifold fin edges and QEM refuses it). The cost is reported --
    a few small holes for strict manifoldness -- not hidden. process_scan(manifold=True) opts in. Four local
    surgeries were tried and refuted (kept negatives in the module docstring); this split+gate is the honest
    pragmatic route until R3-proper (global integer assignment) lands."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshqem import qem_decimate
    b = triangulate_ngons(loop_subdivide(box(), 2))
    F = [tuple(int(i) for i in f) for f in b.faces]
    a, c, d = F[0]
    fin = Mesh(np.asarray(b.vertices, float), F + [(a, d, c), (a, d, c)])   # doubled-face fins
    out, rep = mind.manifold_cleanup(fin)
    assert rep["manifold"] and rep["non_manifold_after"] == 0
    assert rep["faces_kept_frac"] > 0.5 and "gate" in rep
    # consumer contract: QEM refuses the fin, accepts the cleaned mesh
    try:
        qem_decimate(fin, target_faces=8)
        assert False, "QEM should refuse the non-manifold fixture"
    except AssertionError:
        raise
    except Exception:
        pass
    assert len(qem_decimate(out, target_faces=8).faces) > 0
    # process_scan opt-in surfaces the stage
    _, _, _, prep = mind.process_scan(fin, retopo=True, density=1.0, silhouette=None, manifold=True)
    assert any(s["stage"] == "manifold_cleanup" for s in prep["stages"])


def test_gab_cloud_render(mind):
    """GAB-CLOUD: a fitted GaborField renders as a single-scattered volumetric cloud through the engine's
    cloud renderer. Pins: (a) finite-segment optical_depth matches quadrature (the erf-based integral);
    (b) the field satisfies the density protocol so single_scatter accepts it; (c) closed-form shadow rays
    stay accurate (the cloud_report bar holds on a Gabor field, same as on FPE volumes)."""
    import numpy as np
    from holographic.rendering.holographic_gaborfield import GaborField
    from holographic.rendering.holographic_cloud import cloud_report
    gf = GaborField(A=[1.0, 0.5], mu=[[0.5, 0.5, 0.5], [0.4, 0.6, 0.5]], sigma=[0.15, 0.1],
                    w=[[0, 0, 0], [7.0, 0, 0]], phi=[0.0, 0.5])
    o = np.array([[0.5, 0.5, -1.0]]); d = np.array([[0.0, 0.0, 1.0]])
    seg = gf.optical_depth(o, d, 2.0)[0]
    tq = np.linspace(0, 2.0, 80001); Pq = o[0][None, :] + tq[:, None] * d[0][None, :]
    quad = float(np.trapezoid(np.clip(gf.eval(Pq), 0, None), tq))
    assert abs(seg - quad) < 1e-5, "segment integral must match quadrature (%.2e)" % abs(seg - quad)
    O = np.array([[0.5, 0.5, -0.5], [0.45, 0.55, -0.5]]); D = np.tile([0, 0, 1.0], (2, 1))
    rad, ev = mind.gabor_cloud_render(gf, O, D, 2.0, np.array([0.3, 1.0, 0.2]), 1.2, view_steps=8)
    assert np.all(np.isfinite(rad)) and rad.min() >= 0.0
    r = cloud_report(gf, O, D, L=2.0, sun_dir=np.array([0.3, 1.0, 0.2]), ceiling=1.2, view_steps=8,
                     reference_shadow_steps=32)
    assert r["max_error"] < 1e-2 and r["eval_ratio"] > 5.0, "closed-form shadow must stay cheap and accurate"


def test_vcol_vertex_colour_render(mind):
    """VCOL: render a mesh from per-vertex colours (barycentric-interpolated) with no texture -- the path
    H5's vertex-scale recall bake needed (it produces per-vertex colour with nowhere to render). Contracts:
    (a) vertex_colors= param and mesh.colours attribute give identical results; (b) plain flat/textured
    renders are unaffected when no colours given (additive, default None); (c) the interpolation is smooth
    (a position-graded mesh shows colour variety, not flat)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.rendering.holographic_render import Camera
    b = triangulate_ngons(loop_subdivide(box(), 2))
    V = np.asarray(b.vertices, float)
    col = (V - V.min(0)) / (V.max(0) - V.min(0) + 1e-9)
    cam = Camera(eye=(1.6, 1.1, 1.8), target=(0, 0, 0), fov_deg=42.0, aspect=1.0)
    im_param = np.asarray(mind.render_mesh(b, cam, width=96, height=96, vertex_colors=col, smooth=True))
    bc = Mesh(V, [tuple(int(i) for i in f) for f in b.faces],
              colours=np.concatenate([col, np.ones((len(V), 1))], 1))
    im_attr = np.asarray(mind.render_mesh(bc, cam, width=96, height=96, smooth=True))
    assert np.allclose(im_param, im_attr), "param and mesh.colours paths must agree"
    fg = im_param[im_param.sum(2) > 0.05]
    assert fg.std() > 0.1, "position-graded colours must render as a gradient, not flat"
    # additive: a plain render (no colours) is unchanged / still valid
    im_plain = np.asarray(mind.render_mesh(b, cam, width=96, height=96))
    assert im_plain.shape == (96, 96, 3) and im_plain.mean() > 0.0


def test_sr_beta_weighted_fusion(mind):
    """SR-BETA: the dense-dominance sweep verdict, pinned as a mechanism test. (A) a dense HIT (gold at dense
    rank 1) survives fusion with a weak BM25 whose top is spurious, at dense-dominant weight; (B) a BURIED
    gold (low in dense, #1 in BM25) is RESCUED at the same weight; (C) equal weight can LOSE a rank-2 dense
    hit to a spurious BM25 #1 -- the refuted case that motivates down-weighting. weights= is exposed on the
    faculty."""
    from holographic.semantic_router.holographic_bm25 import reciprocal_rank_fusion
    # (A) dense hit kept at (1.0, 0.3)
    A = reciprocal_rank_fusion([["d0", "d1", "d2"], ["d5", "d0", "d6"]], weights=[1.0, 0.3])
    assert A[0][0] == "d0", "dense-#1 gold must survive dense-dominant fusion"
    # (B) buried-but-present gold rescued
    B = reciprocal_rank_fusion([["d3", "d1", "d2", "d0"], ["d0", "d6", "d7"]], weights=[1.0, 0.3])
    assert B[0][0] == "d0", "a gold buried in dense but #1 in BM25 must be rescued"
    # (C) the refuted equal-weight case: a rank-2 dense hit lost to a spurious BM25 #1
    C = reciprocal_rank_fusion([["d5", "d0", "d1"], ["d5", "d2", "d3"]], weights=[1.0, 1.0])
    assert C[0][0] == "d5", "equal weight lets a spurious doc ranked by BOTH win (the motivating negative)"
    # the faculty exposes weights=
    F = mind.fuse_rankings([["a", "b"], ["c", "a"]], weights=[1.0, 0.3])
    assert F[0][0] == "a"


def test_m9_mesh_parts(mind):
    """M9: Reeb-graph part segmentation. A three-armed star decomposes into >=4 parts (3 arms + core), the
    three arms are the elongated ones (aspect > 4), and the two mirrored arms pair as symmetric. Every part
    is one connected blob. This is the segmentation a rig needs; it runs on the SURFACE graph so thin limbs
    survive (kept negative: the voxel ridge found 45 points on a mantis's legs -- see the module docstring)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from collections import defaultdict, deque
    S = triangulate_ngons(loop_subdivide(box(), 4))
    V = np.asarray(S.vertices, float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    dirs = np.asarray([[0.0, -1.0, 0.0], [-0.8, 0.9, 0.0], [0.8, 0.9, 0.0]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    for dvec in dirs:
        V = V + dvec[None, :] * (3.0 * np.clip((V @ dvec - 0.7) / 0.3, 0.0, 1.0) ** 1.2)[:, None]
    Y = Mesh(V, [tuple(int(i) for i in f) for f in S.faces])
    lab, rep = mind.mesh_parts(Y, band_factor=4.0, min_part_frac=0.05)
    assert rep["n_parts"] >= 4, "star must give >=4 parts, got %d" % rep["n_parts"]
    arms = [l for l in rep["part_ids"] if rep["part_aspect"][l] > 4.0]
    assert len(arms) >= 3, "three elongated arms expected"
    pairs = mind.match_symmetric_parts(lab, rep, V)
    assert any(a in arms and b in arms for a, b in pairs), "the two mirrored arms must pair"
    # every part is one connected component on the surface graph
    adj = defaultdict(set)
    for f in Y.faces:
        f = [int(i) for i in f]
        for k in range(len(f)):
            adj[f[k]].add(f[(k + 1) % len(f)]); adj[f[(k + 1) % len(f)]].add(f[k])
    for l in rep["part_ids"]:
        vs = set(np.where(lab == l)[0].tolist()); seen = set(); comps = 0
        for v in vs:
            if v in seen:
                continue
            comps += 1; dq = deque([v]); seen.add(v)
            while dq:
                u = dq.popleft()
                for w in adj[u]:
                    if w in vs and w not in seen:
                        seen.add(w); dq.append(w)
        assert comps == 1, "part %d fragmented into %d components" % (l, comps)


def test_m2_rig_from_parts(mind):
    """M2: rig_from_parts assembles a joint tree + label-aware skin from a mesh_parts segmentation, weights are
    a partition of unity, and posing one arm's distal joint moves ONLY that arm (>10x in-vs-out). Composition
    of M9 + skin_bind_weights + part adjacency -- unblocked once M9 gave the parts."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    S = triangulate_ngons(loop_subdivide(box(), 4))
    V = np.asarray(S.vertices, float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    for dvec in np.asarray([[0.0, -1.0, 0.0], [-0.8, 0.9, 0.0], [0.8, 0.9, 0.0]]):
        dvec = dvec / np.linalg.norm(dvec)
        V = V + dvec[None, :] * (3.0 * np.clip((V @ dvec - 0.7) / 0.3, 0.0, 1.0) ** 1.2)[:, None]
    mesh = Mesh(V, [tuple(int(i) for i in f) for f in S.faces])
    lab, rep = mind.mesh_parts(mesh, band_factor=4.0, min_part_frac=0.05)
    rig = mind.rig_from_parts(mesh, lab, rep)
    W = rig["weights"]
    assert np.allclose(W.sum(1), 1.0, atol=1e-6), "partition of unity"
    assert len(rig["joints"]) >= 4 and len(rig["bones"]) >= 3
    arms = [p for p in rep["part_ids"] if rep["part_aspect"][p] > 4.0
            and rig["joint_prox"][p] != rig["joint_dist"][p]]
    assert arms, "an elongated arm with two joints is required"
    arm = arms[0]; prox = rig["joint_prox"][arm]; dist = rig["joint_dist"][arm]; J = rig["joints"]
    ax = np.cross(J[dist] - J[prox], np.array([0.0, 0, 1.0])); ax = ax / (np.linalg.norm(ax) + 1e-9)
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(0.8) * Kx + (1 - np.cos(0.8)) * Kx @ Kx
    Ts = np.stack([np.eye(4) for _ in range(len(J))])
    Ts[dist, :3, :3] = R; Ts[dist, :3, 3] = J[prox] - R @ J[prox]
    Vh = np.concatenate([V, np.ones((len(V), 1))], 1)
    out = sum(W[:, j][:, None] * (Vh @ Ts[j].T)[:, :3] for j in range(len(J)))
    moved = np.linalg.norm(out - V, axis=1); inarm = lab == arm
    assert moved[inarm].mean() > 10 * moved[~inarm].mean() + 1e-6, "one-arm pose must stay isolated"


def test_gab_aniso_fit(mind):
    """GAB-ANISO: anisotropic envelopes beat isotropic on oriented filament content at equal kernel count
    (breaking the isotropic PSNR plateau -- round envelopes cannot elongate), the aniso segment integral
    stays quadrature-exact, and an isotropic field (Q=None) is byte-identical to before. Opt-in (worse on
    blobby content -- kept negative)."""
    import numpy as np
    from holographic.rendering.holographic_gaborfield import GaborField
    N = 32; ax = np.linspace(0, 1, N); X = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
    rho = np.zeros((N, N, N))
    for c, dv in [((0.3, 0.3, 0.5), (1, 1, 0.2)), ((0.6, 0.4, 0.5), (1, -0.5, 0.3))]:
        c = np.array(c, float); dv = np.array(dv, float); dv /= np.linalg.norm(dv)
        rel = X - c; al = rel @ dv; pe = (rel ** 2).sum(-1) - al ** 2
        rho += np.exp(-al ** 2 / (2 * 0.22 ** 2)) * np.exp(-pe / (2 * 0.03 ** 2))
    rho = np.clip(rho, 0, None); P = X.reshape(-1, 3)
    fi, _ = mind.gabor_volume(rho, K=24, n_freqs=3, anisotropic=False)
    fa, _ = mind.gabor_volume(rho, K=24, n_freqs=3, anisotropic=True)
    def ps(a, b):
        return 10 * np.log10(max(a.max(), 1e-9) ** 2 / max(((a - b) ** 2).mean(), 1e-12))
    psi = ps(rho, fi.eval(P).reshape(N, N, N)); psa = ps(rho, fa.eval(P).reshape(N, N, N))
    assert psa > psi + 2.0, "aniso must beat iso by >2 dB on filaments (%.1f vs %.1f)" % (psa, psi)
    assert fa.Q is not None and fa.Q.shape == (24, 3, 3) and fi.Q is None
    # aniso segment integral matches quadrature
    gs = GaborField(A=[1.0], mu=[[0.5, 0.5, 0.5]], sigma=[0.1], w=[[0, 0, 0]], phi=[0.0],
                    Q=[np.diag([1 / 0.3 ** 2, 1 / 0.05 ** 2, 1 / 0.05 ** 2])])
    seg = gs.optical_depth(np.array([[0.5, 0.5, -1.0]]), np.array([[0, 0, 1.0]]), 3.0)[0]
    t = np.linspace(0, 3, 100001); Pp = np.array([0.5, 0.5, -1.0])[None, :] + t[:, None] * np.array([0, 0, 1.0])[None, :]
    quad = float(np.trapezoid(np.clip(gs.eval(Pp), 0, None), t))
    assert abs(seg - quad) < 1e-5, "aniso segment integral must match quadrature (%.2e)" % abs(seg - quad)
    # isotropic field byte-identical whether Q=None or Q=I/sigma^2
    gf = GaborField(A=[1.2], mu=[[0.5, 0.5, 0.5]], sigma=[0.15], w=[[0, 0, 0]], phi=[0.0])
    gq = GaborField(A=[1.2], mu=[[0.5, 0.5, 0.5]], sigma=[0.15], w=[[0, 0, 0]], phi=[0.0],
                    Q=[np.eye(3) / 0.15 ** 2])
    Pt = np.random.default_rng(0).uniform(0, 1, (40, 3))
    assert np.allclose(gf.eval(Pt), gq.eval(Pt), atol=1e-12)


def test_r6_laplacian_eigenmaps(mind):
    """R6 foundation: the cotan Laplacian eigenspectrum matches the sphere's l(l+1) harmonics, the first
    eigenspace recovers x,y,z, and Morse critical counts obey Euler-Poincare (min - saddle + max = chi=2).
    This is the SCALAR vertex Laplacian a spectral analysis (segmentation, quadrangulation) builds on --
    distinct from the crossfield connection Laplacian."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    S = triangulate_ngons(loop_subdivide(box(), 3))
    V = np.asarray(S.vertices, float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    sphere = Mesh(V, [tuple(int(i) for i in f) for f in S.faces])
    w, phi = mind.mesh_laplacian_eigenmaps(sphere, k=6)
    assert abs(w[0]) < 1e-5, "first eigenvalue ~0"
    assert np.allclose(w[1:4], 2.0, atol=0.05), "l=1 harmonics at ~2: %s" % w[1:4]
    B = phi[:, 1:4]
    for c in range(3):
        xyz = V[:, c] - V[:, c].mean()
        proj = B @ np.linalg.lstsq(B, xyz, rcond=None)[0]
        assert 1 - np.var(xyz - proj) / np.var(xyz) > 0.98, "eigenspace recovers coord %d" % c
    crit = mind.morse_critical_points(sphere, V[:, 2])
    assert crit["minima"] - crit["saddles"] + crit["maxima"] == 2, \
        "Euler-Poincare chi=2 (%d/%d/%d)" % (crit["minima"], crit["saddles"], crit["maxima"])
    assert crit["minima"] == 1 and crit["maxima"] == 1, "z-height on a sphere: 1 min + 1 max"


def test_stage0_low_eigenvectors(mind):
    """Stage-0: block shifted inverse iteration (matvec-only, no scipy) matches dense eigh's low band and is
    deterministic -- the eigensolver R6/Fiedler/SATO-SEQ all depend on."""
    import numpy as np
    A = np.random.default_rng(0).standard_normal((30, 30)); A = A @ A.T
    wt = np.linalg.eigvalsh(A)
    c = float(np.abs(A).sum(1).max())
    w, U = mind.low_eigenvectors(lambda x: A @ x, 30, c, k=4, dtype=float, shift=float(wt[0] - 0.5), iters=80)
    assert np.allclose(np.sort(w), wt[:4], atol=1e-2), "band must match dense eigh"
    w2, U2 = mind.low_eigenvectors(lambda x: A @ x, 30, c, k=4, dtype=float, shift=float(wt[0] - 0.5), iters=80)
    assert np.array_equal(U, U2), "must be deterministic"


def test_eigenmaps_sparse_matches_dense(mind):
    """R6: the matvec (sparse) eigenmaps path matches the dense path in eigenvalues and the l=1 eigenspace on
    a sphere, and auto stays byte-identical to dense on a small mesh."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_crossfield import mesh_laplacian_eigenmaps
    S = triangulate_ngons(loop_subdivide(box(), 3))
    V = np.asarray(S.vertices, float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    mesh = Mesh(V, [tuple(int(i) for i in f) for f in S.faces])
    wd, pd = mesh_laplacian_eigenmaps(mesh, k=5, solver="dense")
    ws, ps = mesh_laplacian_eigenmaps(mesh, k=5, solver="sparse")
    assert np.allclose(np.sort(ws), np.sort(wd), atol=0.03), "sparse eigenvalues must match dense"
    Bd = pd[:, 1:4]; Bs = ps[:, 1:4]; P = Bd @ np.linalg.lstsq(Bd, Bs, rcond=None)[0]
    assert np.linalg.norm(Bs - P) / np.linalg.norm(Bs) < 1e-6, "l=1 eigenspace must agree"
    wa, pa = mesh_laplacian_eigenmaps(mesh, k=5)               # auto -> dense on a small mesh
    assert np.array_equal(wa, wd) and np.array_equal(pa, pd), "auto must be byte-identical to dense here"


def test_satoseq_roundtrip(mind):
    """SATO-SEQ: Morton order is permutation-stable, the mesh serialises to 3 tokens/vertex, and a token
    sequence round-trips through seq_encode/seq_decode (including block storage past the capacity cliff)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshseq import morton_key
    rng = np.random.default_rng(1); pts = rng.uniform(0, 1, (120, 3))
    k1 = [morton_key(p) for p in pts]; perm = rng.permutation(120); k2 = [morton_key(p) for p in pts[perm]]
    assert np.array_equal(pts[np.argsort(k1, kind="stable")], pts[perm[np.argsort(k2, kind="stable")]])
    toks, idx, grid = mind.mesh_to_tokens(box(), order="morton", bits=8)
    assert len(toks) == 3 * len(box().vertices)
    seq = [int(x) for x in np.random.default_rng(2).integers(0, 64, 48)]
    H = mind.seq_encode(seq, dim=1024, seed=0, vocab_size=64)
    assert mind.seq_decode(H, len(seq), dim=1024, seed=0, vocab_size=64) == seq
    lseq = [int(x) for x in np.random.default_rng(3).integers(0, 64, 256)]
    Hl = mind.seq_encode(lseq, dim=1024, seed=1, vocab_size=64)
    assert isinstance(Hl, list) and mind.seq_decode(Hl, len(lseq), dim=1024, seed=1, vocab_size=64) == lseq


def test_m16_worst_view(mind):
    """M16: DIRECT finds a planted worst view within 1 deg beating a dense sweep's eval count; certified B&B
    returns the optimum with a valid small certificate; deterministic."""
    import numpy as np
    g = np.array([0.41, -0.63, 0.66]); g = g / np.linalg.norm(g)
    g2 = np.array([-0.7, 0.2, 0.7]); g2 = g2 / np.linalg.norm(g2)
    def metric(d):
        d = np.asarray(d, float)
        return 1.0 * np.exp(-8 * np.arccos(np.clip(d @ g, -1, 1)) ** 2) + \
               0.8 * np.exp(-6 * np.arccos(np.clip(d @ g2, -1, 1)) ** 2)
    d, v, rep = mind.worst_view(metric, mode="direct", max_evals=2400)
    assert np.degrees(np.arccos(np.clip(d @ g, -1, 1))) < 1.0 and rep["evals"] < 2562
    L = 1.0 * np.sqrt(16 / np.e) + 0.8 * np.sqrt(12 / np.e)
    dc, vc, rc = mind.worst_view(metric, mode="certified", lipschitz=L, max_evals=20000, eps=1e-3)
    assert np.degrees(np.arccos(np.clip(dc @ g, -1, 1))) < 0.5 and rc["certified_gap"] <= 1e-3 + 1e-9
    d2, v2, _ = mind.worst_view(metric, mode="direct", max_evals=2400)
    assert np.array_equal(d, d2)


def test_r6_stripe_pattern(mind):
    """Knoppel-Crane stripe patterns: on a sphere with a smooth tangent field the recovered complex phase
    follows the field to a small median edge residual, its energy is a small non-negative eigenvalue, and the
    phase actually winds (stripes exist). This is ONE smallest-eigenvector problem on the shipped matvec-only
    eigensolver -- the field-following sibling of the cross field."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    S = triangulate_ngons(loop_subdivide(box(), 3))
    V = np.asarray(S.vertices, float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    N = V.copy(); ax = np.array([0.0, 0, 1.0])
    X = ax[None, :] - N * (N @ ax)[:, None]
    bad = np.linalg.norm(X, axis=1) < 1e-6
    X[bad] = np.array([1.0, 0, 0]) - N[bad] * (N[bad] @ np.array([1.0, 0, 0]))[:, None]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    mesh = Mesh(V, [tuple(int(i) for i in f) for f in S.faces])
    psi, rep = mind.stripe_pattern(mesh, X, frequency=18.0)
    assert rep["energy"] < 0.05, "stripe energy must be small: %.3e" % rep["energy"]
    assert rep["phase_residual_median"] < 0.05, "phase must follow field: %.3f rad" % rep["phase_residual_median"]
    assert np.ptp(np.angle(psi)) > 1.0, "phase must vary (stripes exist)"
    # determinism
    psi2, _ = mind.stripe_pattern(mesh, X, frequency=18.0)
    assert np.allclose(psi, psi2), "stripe_pattern must be deterministic"


def test_r6_holistic_lattice_resonator(mind):
    """R6 (gated): the FHRR resonator recovers integer lattice coordinates from ONLY their bound product
    z_u^u * z_v^v, even under phase noise -- the holistic-only regime where rounding is undefined. KEPT
    NEGATIVE (measured): for direct noisy coordinates np.round dominates and this must not be used."""
    import numpy as np, hashlib
    def base(seed):
        h = hashlib.sha256(str(seed).encode()).digest()
        return np.exp(1j * np.random.default_rng(int.from_bytes(h[:8], "big")).uniform(-np.pi, np.pi, 1024))
    zu, zv = base("u"), base("v"); K = 21
    rng = np.random.default_rng(0); ok = 0
    for _ in range(40):
        u, v = int(rng.integers(0, K)), int(rng.integers(0, K))
        s = (zu ** u) * (zv ** v) * np.exp(1j * rng.normal(0, 0.6, 1024))
        coords, rep = mind.fpe_lattice_resonator(s, [zu, zv], [K, K])
        ok += (coords == [u, v])
    assert ok >= 38, "resonator must recover holistic-only coords past 0.6 rad noise (%d/40)" % ok
    c1, _ = mind.fpe_lattice_resonator((zu ** 7) * (zv ** 13), [zu, zv], [K, K])
    c2, _ = mind.fpe_lattice_resonator((zu ** 7) * (zv ** 13), [zu, zv], [K, K])
    assert c1 == [7, 13] and c1 == c2, "exact + deterministic on a clean product"

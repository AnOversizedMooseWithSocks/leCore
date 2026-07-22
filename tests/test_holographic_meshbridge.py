"""Tests for the mesh<->SDF<->splat bridge (FWD-11): marching-tetrahedra isosurface extraction (SDF->mesh),
mesh->SDF sampling, and the splat->mesh path. Measured against analytic references: a closed-manifold sphere with
vertices on the sphere, outward orientation, signed-distance correctness, the splat blob, resolution scaling,
determinism."""

import numpy as np

from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra, mesh_to_sdf, sphere_sdf, metaball_field


def _extract_sphere(radius=1.0, res=24, half=1.5):
    vals, axes = sample_field(sphere_sdf(radius=radius), ((-half,) * 3, (half,) * 3), res=res)
    return marching_tetrahedra(vals, axes, level=0.0)


# ---- SDF -> mesh ----------------------------------------------------------------------------------
def test_sdf_to_mesh_is_a_closed_manifold_sphere():
    m = _extract_sphere()
    assert m.n_faces > 0 and m.is_manifold()
    assert m.is_closed() and m.euler_characteristic() == 2      # genus-0 closed surface (watertight)


def test_extracted_vertices_lie_on_the_sphere():
    radii = np.linalg.norm(_extract_sphere(radius=1.0).vertices, axis=1)
    assert abs(float(radii.mean()) - 1.0) < 0.02 and float(radii.std()) < 0.03


def test_extracted_sphere_radius_scales():
    radii = np.linalg.norm(_extract_sphere(radius=0.7, half=1.2).vertices, axis=1)
    assert abs(float(radii.mean()) - 0.7) < 0.02               # a different radius extracts correctly


def test_marching_tets_orientation_is_outward():
    m = _extract_sphere(res=20)
    V = m.vertices
    outward = sum(1 for (a, b, c) in m.faces
                  if np.dot(np.cross(V[b] - V[a], V[c] - V[a]), (V[a] + V[b] + V[c]) / 3.0) > 0)
    assert outward == m.n_faces                                # every face normal points outward


def test_resolution_scaling_adds_faces():
    f12 = _extract_sphere(res=12).n_faces
    f20 = _extract_sphere(res=20).n_faces
    f28 = _extract_sphere(res=28).n_faces
    assert f12 < f20 < f28                                     # finer grid -> more, finer triangles


# ---- mesh -> SDF ----------------------------------------------------------------------------------
def test_mesh_to_sdf_matches_analytic():
    m = _extract_sphere()
    probes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -1.5]])
    got = mesh_to_sdf(m, probes)
    analytic = np.linalg.norm(probes, axis=1) - 1.0
    assert np.allclose(got, analytic, atol=0.05)


def test_mesh_to_sdf_sign_is_inside_negative_outside_positive():
    m = _extract_sphere()
    got = mesh_to_sdf(m, np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert got[0] < 0 and got[1] > 0


# ---- splat -> mesh --------------------------------------------------------------------------------
def test_splat_field_meshes_to_a_closed_blob():
    vals, axes = sample_field(metaball_field(np.array([[-0.4, 0, 0], [0.4, 0, 0]]), radius=0.4),
                              ((-1.5,) * 3, (1.5,) * 3), res=24)
    blob = marching_tetrahedra(vals, axes, level=0.5)
    assert blob.n_faces > 0 and blob.is_manifold() and blob.is_closed()


# ---- determinism ----------------------------------------------------------------------------------
def test_marching_tets_is_deterministic():
    vals, axes = sample_field(sphere_sdf(), ((-1.5,) * 3, (1.5,) * 3), res=20)
    assert np.array_equal(marching_tetrahedra(vals, axes, 0.0).vertices,
                          marching_tetrahedra(vals, axes, 0.0).vertices)


def test_marching_tetrahedra_vec_matches_per_cell():
    """The vectorized marcher (marching_tetrahedra_vec, the case-table-RAM parallel path) is geometrically identical
    to the per-cell marching_tetrahedra: same vertex/face counts, same faces by position, same orientation, same
    manifoldness -- on a sphere, two spheres, a torus, and a non-manifold grid. A faithful parallelization."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra, marching_tetrahedra_vec

    bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    def two_spheres(P):
        a = np.linalg.norm(P - np.array([0.3, 0, 0]), axis=1) - 0.35
        b = np.linalg.norm(P - np.array([-0.3, 0, 0]), axis=1) - 0.35
        return np.minimum(a, b)

    def torus(P):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        q = np.sqrt(x * x + y * y) - 0.5
        return np.sqrt(q * q + z * z) - 0.22

    def face_pos_set(m):
        return {tuple(sorted(tuple(np.round(m.vertices[vi], 9)) for vi in f)) for f in m.faces}

    for fn, res in [(sphere, 24), (sphere, 31), (two_spheres, 28), (torus, 30)]:
        vals, axes = sample_field(fn, bounds, res)
        ref = marching_tetrahedra(vals, axes, 0.0)
        vec = marching_tetrahedra_vec(vals, axes, 0.0)
        assert vec.n_faces == ref.n_faces
        assert len(vec.vertices) == len(ref.vertices)
        assert face_pos_set(vec) == face_pos_set(ref)               # same faces, same positions
        assert vec.is_manifold() == ref.is_manifold()               # reproduces grid-dependent manifoldness too
        # orientation is well-defined only where the mesh is manifold (degenerate triangles at a grid vertex landing
        # on the surface have ambiguous winding); there, the net oriented area must agree.
        if ref.is_manifold():
            nref = sum(np.cross(ref.vertices[b] - ref.vertices[a], ref.vertices[c] - ref.vertices[a])
                       for a, b, c in ref.faces)
            nvec = sum(np.cross(vec.vertices[b] - vec.vertices[a], vec.vertices[c] - vec.vertices[a])
                       for a, b, c in vec.faces)
            assert np.allclose(nref, nvec, atol=1e-9)


def test_mesh_to_field_is_signed_subvoxel_accurate_and_deterministic():
    """mesh_distance_grid is the mesh -> FIELD direction by TILING: each triangle scatter-mins into its local voxel
    block. SIGNED so |sample| near the surface resolves distance to well under a voxel (an unsigned field's kink
    cannot). Verify against the analytic sphere distance, that the sign is right just inside/outside, and determinism."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import sphere_sdf, sample_field, marching_tetrahedra_vec, mesh_distance_grid, sample_distance_grid

    vals, axes = sample_field(sphere_sdf(radius=1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=24)
    m = marching_tetrahedra_vec(vals, axes, 0.0)
    grid, gax = mesh_distance_grid(m, ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=48)

    # sub-voxel accuracy: points just off the unit sphere, |sample| vs analytic |p|-1
    pts = m.vertices[:80] * 1.04
    samp = sample_distance_grid(grid, gax, pts)
    truth = np.linalg.norm(pts, axis=1) - 1.0
    voxel = 3.0 / 47
    assert np.abs(samp - truth).max() < 0.5 * voxel               # well under a voxel -- the signed-field win

    # sign is right just inside (negative) and just outside (positive), within the band
    s_in = sample_distance_grid(grid, gax, m.vertices[:20] * 0.97)
    s_out = sample_distance_grid(grid, gax, m.vertices[:20] * 1.03)
    assert np.all(s_in < 0) and np.all(s_out > 0)

    # deterministic
    g2, _ = mesh_distance_grid(m, ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=48)
    assert np.array_equal(grid, g2)


def test_mesh_to_field_matches_brute_surface_distance():
    """The point of the field: build once, then query many points cheaply -- and the answer should track the exact
    brute point-to-mesh distance near the surface (an LOD/decimation deviation). Compare |field sample| to
    surface_deviation on a decimated mesh's vertices; they should agree to about the grid resolution."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import sphere_sdf, sample_field, marching_tetrahedra_vec, mesh_distance_grid, sample_distance_grid
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate, surface_deviation

    vals, axes = sample_field(sphere_sdf(radius=1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=28)
    m = marching_tetrahedra_vec(vals, axes, 0.0)
    coarse = cluster_decimate(m, 14)

    bmean, bmax = surface_deviation(coarse, m)                     # exact brute reference
    lo = m.vertices.min(0) - 0.05; hi = m.vertices.max(0) + 0.05
    grid, gax = mesh_distance_grid(m, (lo, hi), res=64)
    d = np.abs(sample_distance_grid(grid, gax, coarse.vertices))   # field-read deviation, O(V)
    voxel = float((hi - lo).max()) / 63
    assert abs(float(d.mean()) - bmean) < voxel                   # agrees to ~grid resolution
    assert abs(float(d.max()) - bmax) < 2.0 * voxel


def test_batched_closest_point_kernel_matches_brute_paired_and_allpairs():
    """The batched closest-point kernel (_closest_points_on_triangles) is the generalization under the whole
    point-to-mesh family: it must match the single-triangle kernel to machine epsilon in BOTH the all-pairs shape
    (N points x F triangles) and the paired shape (F triangles x their own points), and point_set_to_mesh (its
    all-pairs wrapper) must match the brute surface distance, signed and unsigned. NOTE: this kernel is correct
    infrastructure, NOT a speedup for large meshes -- the all-pairs reduction is memory-bandwidth-bound and the
    per-triangle brute loop stays the cache-appropriate path (measured, see the module docstring)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import _closest_point_on_triangle, _closest_points_on_triangles, point_set_to_mesh, mesh_to_sdf, sphere_sdf, sample_field, marching_tetrahedra_vec

    rng = np.random.default_rng(7)

    # all-pairs shape: N points x 1 triangle broadcast == the single-triangle kernel
    a, b, c = rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(3)
    P = rng.standard_normal((150, 3))
    o_single = _closest_point_on_triangle(P, a, b, c)
    o_batch = _closest_points_on_triangles(P[:, None, :], a, b, c)[:, 0, :]
    assert np.abs(o_single - o_batch).max() < 1e-12

    # paired shape: F triangles, each with its own block of points
    A = rng.standard_normal((5, 3)); B = rng.standard_normal((5, 3)); C = rng.standard_normal((5, 3))
    Pp = rng.standard_normal((5, 20, 3))
    cp = _closest_points_on_triangles(Pp, A[:, None, :], B[:, None, :], C[:, None, :])   # (5,20,3)
    for ti in range(5):                                            # each row must match the single-triangle kernel
        ref = _closest_point_on_triangle(Pp[ti], A[ti], B[ti], C[ti])
        assert np.abs(cp[ti] - ref).max() < 1e-12

    # point_set_to_mesh exactly matches the brute loop, unsigned and signed
    v, ax = sample_field(sphere_sdf(radius=1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=16)
    m = marching_tetrahedra_vec(v, ax, 0.0)
    Q = m.vertices[:100] * 1.06
    best = np.full(len(Q), np.inf)
    for f in m.faces:
        d = np.linalg.norm(Q - _closest_point_on_triangle(Q, m.vertices[f[0]], m.vertices[f[1]], m.vertices[f[2]]), axis=1)
        best = np.minimum(best, d)
    assert np.abs(point_set_to_mesh(Q, m.vertices, m.faces) - best).max() < 1e-9

    probes = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.6, 0.0]])
    s_old = mesh_to_sdf(m, probes)
    s_new = point_set_to_mesh(probes, m.vertices, m.faces, signed=True)
    assert np.abs(s_old - s_new).max() < 1e-9 and np.all(np.sign(s_old) == np.sign(s_new))


def test_flood_fill_makes_a_full_remarchable_sdf():
    """flood_fill_sign turns the banded SDF (interior defaults to wrong +band) into a FULL signed SDF by flooding the
    outside from the boundary; the negative band shell blocks it, so the enclosed interior is filled negative. The
    result must be RE-MARCHABLE -- marching it at level 0 reconstructs the surface -- which is what lets an imported
    mesh inherit field-native LOD. Verify interior sign, exterior sign, reconstruction, and determinism."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import sphere_sdf, sample_field, marching_tetrahedra_vec, mesh_distance_grid, mesh_to_sdf_grid, flood_fill_sign

    v, ax = sample_field(sphere_sdf(radius=1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=20)
    m = marching_tetrahedra_vec(v, ax, 0.0)
    bnds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))

    banded, _ = mesh_distance_grid(m, bnds, res=56)
    full, faxes = mesh_to_sdf_grid(m, bnds, res=56)
    mid = full.shape[0] // 2
    assert banded[mid, mid, mid] > 0 and full[mid, mid, mid] < 0      # banded wrong (+), full filled (-)
    assert full[0, 0, 0] > 0                                          # far corner stays outside

    remarch = marching_tetrahedra_vec(full, faxes, 0.0)
    assert remarch.is_closed() and remarch.n_faces > 0
    rr = np.linalg.norm(remarch.vertices, axis=1)
    assert abs(float(rr.mean()) - 1.0) < 0.03                         # reconstructs the unit sphere

    # the band distances are untouched by the fill; only far-interior voxels flip
    assert np.array_equal(full[np.abs(banded) < banded.max()], banded[np.abs(banded) < banded.max()])
    # deterministic
    assert np.array_equal(flood_fill_sign(banded, banded.max()), flood_fill_sign(banded, banded.max()))


def test_grid_accelerated_point_to_mesh_is_exact_near_surface_and_culls_work():
    """point_set_to_mesh_grid culls the O(N*F) scan with a vectorized uniform-grid index (sort-based binning + the
    ranges trick, no Python dicts). For near-surface queries on a roughly uniform mesh it must match the brute
    distance EXACTLY (to machine epsilon) with no misses; a far query (beyond the grid radius) returns +inf -- the
    documented approximate-by-construction caveat. Signed matches mesh_to_sdf near the surface. Deterministic."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import point_set_to_mesh_grid, _closest_point_on_triangle, mesh_to_sdf, sphere_sdf, sample_field, marching_tetrahedra_vec

    v, ax = sample_field(sphere_sdf(radius=1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=24)
    m = marching_tetrahedra_vec(v, ax, 0.0)
    Q = m.vertices * 1.03                                          # near-surface queries

    # brute reference
    best = np.full(len(Q), np.inf)
    for f in m.faces:
        best = np.minimum(best, np.linalg.norm(Q - _closest_point_on_triangle(Q, m.vertices[f[0]], m.vertices[f[1]], m.vertices[f[2]]), axis=1))

    d = point_set_to_mesh_grid(Q, m.vertices, m.faces, radius=2)
    assert not np.any(np.isinf(d))                                # near-surface: nothing missed
    assert np.abs(d - best).max() < 1e-9                          # and EXACT for what it finds

    # far query beyond the grid reach -> +inf (the honest caveat, not a wrong number)
    far = np.array([[5.0, 5.0, 5.0]])
    assert np.isinf(point_set_to_mesh_grid(far, m.vertices, m.faces, radius=1)[0])

    # signed near-surface matches mesh_to_sdf
    probes = m.vertices[:30] * np.array([0.97])[:, None]          # just inside
    s_grid = point_set_to_mesh_grid(probes, m.vertices, m.faces, radius=2, signed=True)
    s_ref = mesh_to_sdf(m, probes)
    ok = ~np.isinf(s_grid)
    assert np.all(s_grid[ok] < 0) and np.abs(s_grid[ok] - s_ref[ok]).max() < 1e-9

    # deterministic
    assert np.array_equal(point_set_to_mesh_grid(Q, m.vertices, m.faces, radius=2),
                          point_set_to_mesh_grid(Q, m.vertices, m.faces, radius=2))


def test_shell_build_matches_scatter_and_remarches():
    """mesh_distance_grid's default 'shell' build (O surface area) gives the same near-surface field as the exact
    'scatter' build (machine epsilon), and its flood-filled SDF re-marches to the same closed surface -- the work-
    culling lesson applied to the build, verified equivalent for the uses that matter."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import mesh_distance_grid, sample_distance_grid, flood_fill_sign, sphere_sdf, sample_field, marching_tetrahedra_vec
    m = marching_tetrahedra_vec(*sample_field(sphere_sdf(radius=0.6), ((-1, -1, -1), (1, 1, 1)), 40), 0.0)
    bnds = (np.array([-1., -1, -1]), np.array([1., 1, 1]))
    gs, ax = mesh_distance_grid(m, bnds, res=56, method="scatter")
    gh, _ = mesh_distance_grid(m, bnds, res=56, method="shell")
    near = m.vertices * 1.02
    assert np.abs(sample_distance_grid(gs, ax, near) - sample_distance_grid(gh, ax, near)).max() < 1e-9  # same near surface
    band = 4.0 * (2.0 / 55)
    full = flood_fill_sign(gh, band)
    mid = full.shape[0] // 2
    assert full[mid, mid, mid] < 0 < full[0, 0, 0]                # interior negative, corner positive
    re = marching_tetrahedra_vec(full, ax, 0.0)
    assert re.n_faces > 0 and re.is_closed()                      # re-marches to a closed surface


def test_sculpt_prepare_guarded_conversion():
    """The sculpt-mode bug, closed: the guard pulls the winding lever on touching shells (where flood fill
    leaks and escalating resolution makes it WORSE), refuses unrecoverable slivers with the ladder report,
    and returns a grid/mesh pair that are the same level of the same field."""
    import numpy as np
    import lecore
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshbridge import marching_tetrahedra_vec
    m = lecore.UnifiedMind(dim=64, seed=0)

    def box_with_fin(fin):
        def bx(cx, cy, cz, hx, hy, hz):
            s = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
            return [(cx+hx*a, cy+hy*b, cz+hz*c) for a, b, c in s]
        V = bx(0,0,0,0.7,0.35,0.5) + bx(0,0.55,0,fin,0.2,0.4)
        Fq = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,4,7,3),(1,2,6,5)]
        F = []
        for base in (0, 8):
            for q in Fq:
                a,b,c,d = [x+base for x in q]; F += [(a,b,c),(a,c,d)]
        return Mesh(np.array(V, float), F)

    # the reproduced bug: auto fails on touching shells; the guard recovers via winding at EQUAL resolution
    r = m.sculpt_prepare(box_with_fin(0.02), resolution=48)
    assert r["report"]["sign"] == "winding" and r["report"]["iou"] >= 0.95
    assert r["report"]["ladder"][0][1] == "auto" and r["report"]["ladder"][0][2] < 0.8

    # the cache and the visible mesh are the SAME field: re-marching the grid reproduces the mesh
    back = marching_tetrahedra_vec(r["grid"], r["axes"], level=0.0)
    assert len(back.vertices) == len(r["mesh"].vertices)

    # unrecoverable sliver -> loud refusal carrying the ladder
    try:
        m.sculpt_prepare(box_with_fin(0.002), resolution=32, max_resolution=96)
        assert False, "must refuse"
    except ValueError as exc:
        assert "REFUSES" in str(exc)

    # explicit opt-out is single-pass and says so
    u = m.sculpt_prepare(box_with_fin(0.002), resolution=24, silhouette=None)
    assert u["report"]["iou"] is None and u["report"]["ladder"] == []

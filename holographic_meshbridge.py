"""The mesh <-> SDF <-> splat bridge (FWD-11): three views of one surface, made convertible.

WHY THIS MODULE EXISTS
----------------------
Tier 3, the highest-value remaining item, and the one the native-first direction elevates: it connects the
explicit-mesh work (FWD-1..10) back to the engine's NATIVE implicit and splat representations. A surface can be
carried three ways -- as an explicit MESH (vertices + faces), as an implicit SDF (a scalar field, negative inside,
whose zero level-set IS the surface), or as a SPLAT field (a superposition of Gaussian primitives, the engine's
`holographic_splat`). The same geometry, three costumes -- exactly the project's recurring thesis. This module is
the bridge that turns one into another and measures that the round-trip holds.

THE GENUINELY NEW PIECE: ISOSURFACE EXTRACTION (SDF -> mesh)
  The mesh kernel says so itself in its header -- "no marching cubes, nothing that a Blender-class modeler edits."
  So extracting a mesh from an implicit field was the one missing direction, and it is what unlocks the bridge.
  This module supplies it via MARCHING TETRAHEDRA rather than marching cubes, on purpose:
    * Marching tetrahedra has a tiny, unambiguous case set (per tet: 0, 1, or 2 triangles by how many of its 4
      corners are inside), where marching cubes has 256 cases AND the notorious ambiguous-face problem.
    * It is MANIFOLD BY CONSTRUCTION: a crossing point lives on a grid edge and is shared by every tet touching
      that edge (welded by edge identity here), and a tet's quad split is interior, so adjacent patches always
      agree on their shared boundary -- no cracks, no ambiguous faces.
  The cube is split into 6 tetrahedra sharing a main diagonal (the standard Kuhn decomposition).

WHAT IT PROVIDES
  * marching_tetrahedra(values, axes, level=0.0) -- extract the level-set isosurface of a sampled 3-D scalar
    field as a watertight, outward-oriented triangle Mesh. The bridge's core (SDF/field -> mesh).
  * sample_field(func, bounds, res) -- evaluate a scalar field func(points)->values on a res^3 grid.
  * mesh_to_sdf(mesh, points) -- the reverse direction: signed distance from a mesh at query points
    (closest-point-on-triangle, sign from the nearest face normal). Mesh -> implicit.
  * sphere_sdf / metaball_field -- analytic fields for testing and the splat bridge.

THE BRIDGE, MEASURED (the bar, against analytic references)
  * SDF -> mesh: extracting the analytic sphere SDF |p|-r gives a CLOSED MANIFOLD (chi = 2) whose vertices all
    lie on the sphere of radius r (to grid resolution).
  * mesh -> SDF: sampling a sphere mesh's signed distance matches the analytic |p|-r at interior/exterior probes.
  * SPLAT -> mesh: a superposition of Gaussian "splats" (a metaball field) iso-extracts to a smooth blob mesh --
    the splat representation entering the mesh world through the SAME extractor (holostuff's `bundle` is a sum of
    Gaussians; thresholding that sum is an isosurface).

DETERMINISM (per ISA.md)
  Grid sampling and the marching pass are pure; crossings are welded by a fixed edge key and triangles emitted in
  a fixed cell/tet order. Same field in -> byte-identical mesh out (asserted).

KEPT NEGATIVES (loud)
  * mesh_to_sdf signs by the NEAREST FACE NORMAL -- exact for convex-ish closed meshes, but it can mis-sign deep
    concavities or thin sheets, where a generalized winding number is needed (not shipped). The magnitude
    (unsigned distance) is always right; only the inside/outside sign has this caveat.
  * Marching tetrahedra resolution is the grid's: sharp features below the cell size are rounded, and the
    extracted surface is the field's smoothed level set, not the original mesh's exact triangles (the round-trip
    recovers the SHAPE to grid resolution, not the connectivity).
  * It extracts triangle soup welded by edge -- it does not guarantee well-shaped triangles; a downstream
    remesh/Taubin smooth (FWD-4) is the cleanup, which is exactly why those faculties exist.
"""

import numpy as np

from holographic_mesh import Mesh

# The 8 cube corners as (i,j,k) bit offsets, and the 6-tetrahedra (Kuhn) decomposition sharing the 0-7 diagonal.
_CUBE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
_TETS = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]


def sample_field(func, bounds, res):
    """Evaluate a scalar field `func` (points (N,3) -> values (N,)) on a res^3 grid spanning
    bounds = ((x0,y0,z0),(x1,y1,z1)). Returns (values (res,res,res), axes=(xs,ys,zs))."""
    (x0, y0, z0), (x1, y1, z1) = bounds
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    zs = np.linspace(z0, z1, res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    values = np.asarray(func(pts), float).reshape(res, res, res)
    return values, (xs, ys, zs)


def _orient(v0, v1, v2, idx, toward):
    """Return the index triple (i0,i1,i2) wound so the triangle normal points TOWARD the point `toward` (the
    outside / higher-field side). v0,v1,v2 are the positions of idx=(i0,i1,i2)."""
    normal = np.cross(v1 - v0, v2 - v0)
    centroid = (v0 + v1 + v2) / 3.0
    if np.dot(normal, toward - centroid) < 0.0:
        return (idx[0], idx[2], idx[1])     # flip winding
    return idx


def marching_tetrahedra(values, axes, level=0.0):
    """Extract the `level` isosurface of a sampled scalar field via marching tetrahedra. `values` is (nx,ny,nz),
    `axes`=(xs,ys,zs) the per-axis coordinates. Returns a watertight triangle Mesh oriented so normals point toward
    INCREASING field (outward, for an SDF where inside is negative)."""
    values = np.asarray(values, float)
    xs, ys, zs = axes
    nx, ny, nz = values.shape

    def pos(coord):
        i, j, k = coord
        return np.array([xs[i], ys[j], zs[k]])

    weld = {}                                # edge (sorted coord pair) -> vertex index
    verts = []

    def crossing(ca, cb):
        key = (ca, cb) if ca <= cb else (cb, ca)
        if key in weld:
            return weld[key]
        fa = values[ca]
        fb = values[cb]
        t = 0.5 if fa == fb else (level - fa) / (fb - fa)
        p = pos(ca) + t * (pos(cb) - pos(ca))
        idx = len(verts)
        verts.append(p)
        weld[key] = idx
        return idx

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corners = [(i + dx, j + dy, k + dz) for (dx, dy, dz) in _CUBE]
                for tet in _TETS:
                    cc = [corners[t] for t in tet]                 # the 4 grid coords of this tet
                    fv = [values[c] for c in cc]
                    inside = [m for m in range(4) if fv[m] < level]
                    outside = [m for m in range(4) if fv[m] >= level]
                    if not inside or not outside:
                        continue
                    P = [pos(c) for c in cc]
                    if len(inside) == 1:                            # one triangle, normal away from the inside corner
                        A = inside[0]
                        tri = [crossing(cc[A], cc[o]) for o in outside]
                        idx = _orient(verts[tri[0]], verts[tri[1]], verts[tri[2]], tuple(tri), P[outside[0]])
                        faces.append(idx)
                    elif len(inside) == 3:                          # one triangle, normal toward the lone outside corner
                        D = outside[0]
                        tri = [crossing(cc[D], cc[m]) for m in inside]
                        idx = _orient(verts[tri[0]], verts[tri[1]], verts[tri[2]], tuple(tri), P[D])
                        faces.append(idx)
                    else:                                          # two inside, two outside -> a quad (two triangles)
                        a, b = inside
                        c, d = outside
                        q = [crossing(cc[a], cc[c]), crossing(cc[b], cc[c]),
                             crossing(cc[b], cc[d]), crossing(cc[a], cc[d])]     # perimeter order
                        toward = 0.5 * (P[c] + P[d])
                        for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                            idx = _orient(verts[tri[0]], verts[tri[1]], verts[tri[2]], tri, toward)
                            faces.append(idx)
    return Mesh(np.array(verts) if verts else np.zeros((0, 3)), faces)


def _closest_point_on_triangle(P, a, b, c):
    """Closest point on triangle (a,b,c) to each row of P (N,3), vectorised (Ericson's region test). Returns the
    closest points (N,3)."""
    ab = b - a
    ac = c - a
    ap = P - a
    d1 = ap @ ab
    d2 = ap @ ac
    bp = P - b
    d3 = bp @ ab
    d4 = bp @ ac
    cp = P - c
    d5 = cp @ ab
    d6 = cp @ ac
    out = np.zeros_like(P)
    done = np.zeros(len(P), dtype=bool)

    def fill(mask, pts):
        m = mask & ~done
        out[m] = pts[m]
        done[m] = True

    fill((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, P.shape))                    # vertex a region
    fill((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, P.shape))                   # vertex b region
    fill((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, P.shape))                   # vertex c region
    vc = d1 * d4 - d3 * d2                                                      # edge ab
    v_ab = np.divide(d1, d1 - d3, out=np.zeros(len(P)), where=(d1 - d3) != 0)
    fill((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v_ab[:, None] * ab)
    vb = d5 * d2 - d1 * d6                                                      # edge ac
    w_ac = np.divide(d2, d2 - d6, out=np.zeros(len(P)), where=(d2 - d6) != 0)
    fill((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w_ac[:, None] * ac)
    va = d3 * d6 - d5 * d4                                                      # edge bc
    denom_bc = (d4 - d3) + (d5 - d6)
    w_bc = np.divide(d4 - d3, denom_bc, out=np.zeros(len(P)), where=denom_bc != 0)
    fill((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w_bc[:, None] * (c - b))
    denom = va + vb + vc                                                       # interior (barycentric)
    denom = np.where(denom != 0, denom, 1.0)
    vv = (vb / denom)[:, None]
    ww = (vc / denom)[:, None]
    fill(np.ones(len(P), dtype=bool), a + ab * vv + ac * ww)
    return out


def _face_normal(V, f):
    a, b, c = V[f[0]], V[f[1]], V[f[2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else n


def mesh_to_sdf(mesh, points):
    """Signed distance from `mesh` at each query point (N,3): the unsigned distance to the nearest triangle, signed
    by the nearest face normal (negative inside). The mesh -> implicit direction. Returns (N,). Kept negative: the
    nearest-normal sign is exact for convex-ish closed meshes but can mis-sign deep concavities (the magnitude is
    always right)."""
    P = np.asarray(points, float)
    tris = [f for f in mesh.faces if len(f) == 3] or [tuple(t) for t in mesh.triangulate()]
    V = mesh.vertices
    best = np.full(len(P), np.inf)
    sign = np.ones(len(P))
    for f in tris:
        cp = _closest_point_on_triangle(P, V[f[0]], V[f[1]], V[f[2]])
        dist = np.linalg.norm(P - cp, axis=1)
        closer = dist < best
        if np.any(closer):
            best = np.where(closer, dist, best)
            nrm = _face_normal(V, f)
            s = np.sign(np.einsum("ij,j->i", P - cp, nrm))         # + outside, - inside (by the face normal)
            s = np.where(s == 0, 1.0, s)
            sign = np.where(closer, s, sign)
    return sign * best


def sphere_sdf(center=(0.0, 0.0, 0.0), radius=1.0):
    """The analytic signed distance to a sphere: f(p) = |p - center| - radius (the exact reference)."""
    center = np.asarray(center, float)

    def f(P):
        return np.linalg.norm(np.asarray(P, float) - center, axis=1) - radius
    return f


def metaball_field(centers, radius=0.5):
    """A sum-of-Gaussians 'metaball' field -- the engine's SPLAT representation as an implicit field (a `bundle`
    of Gaussian primitives is a superposition; thresholding it is an isosurface). Returns a callable whose level
    set wraps the splats. Use a level like 0.5 with marching_tetrahedra to mesh it."""
    centers = np.asarray(centers, float)

    def f(P):
        P = np.asarray(P, float)
        acc = np.zeros(len(P))
        for ctr in centers:
            d2 = np.sum((P - ctr) ** 2, axis=1)
            acc += np.exp(-d2 / (2.0 * radius ** 2))               # a Gaussian splat
        return acc
    return f


# =====================================================================================================
# Self-test -- SDF -> mesh (analytic sphere), mesh -> SDF (probes), splat -> mesh (metaball blob).
# =====================================================================================================
def _selftest():
    # --- SDF -> mesh: extract the analytic unit sphere; closed manifold, chi=2, vertices on the sphere ---
    vals, axes = sample_field(sphere_sdf(radius=1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=24)
    sphere = marching_tetrahedra(vals, axes, level=0.0)
    assert sphere.n_faces > 0 and sphere.is_manifold(), "the extracted surface must be a manifold"
    assert sphere.is_closed() and sphere.euler_characteristic() == 2, \
        f"a sphere isosurface is a closed genus-0 manifold: chi={sphere.euler_characteristic()}"
    radii = np.linalg.norm(sphere.vertices, axis=1)
    assert abs(float(radii.mean()) - 1.0) < 0.02 and float(radii.std()) < 0.03, \
        f"extracted vertices should lie on the unit sphere: mean r={radii.mean():.3f}, std={radii.std():.3f}"

    # --- mesh -> SDF: a sphere mesh's signed distance matches the analytic |p|-1 at probe points ---
    probes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -1.5]])
    sdf_vals = mesh_to_sdf(sphere, probes)
    analytic = np.linalg.norm(probes, axis=1) - 1.0
    assert np.allclose(sdf_vals, analytic, atol=0.05), f"mesh SDF vs analytic: {sdf_vals} vs {analytic}"
    assert sdf_vals[0] < 0 and sdf_vals[1] > 0, "origin inside (negative), far point outside (positive)"

    # --- SPLAT -> mesh: a sum of Gaussian splats iso-extracts to a watertight blob ---
    blob_vals, blob_axes = sample_field(metaball_field(np.array([[-0.4, 0, 0], [0.4, 0, 0]]), radius=0.4),
                                        ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=24)
    blob = marching_tetrahedra(blob_vals, blob_axes, level=0.5)
    assert blob.n_faces > 0 and blob.is_manifold() and blob.is_closed(), "the splat blob meshes to a closed manifold"

    # --- determinism ---
    assert np.array_equal(marching_tetrahedra(vals, axes, 0.0).vertices, marching_tetrahedra(vals, axes, 0.0).vertices)

    print(f"holographic_meshbridge selftest: ok (SDF->mesh: unit sphere extracted, {sphere.n_faces} faces, closed "
          f"manifold chi={sphere.euler_characteristic()}, vertices on sphere mean r={radii.mean():.3f}+/-{radii.std():.3f}; "
          f"mesh->SDF matches analytic |p|-1 at probes (<0.05); splat->mesh: {blob.n_faces}-face blob, closed "
          f"manifold; deterministic)")


if __name__ == "__main__":
    _selftest()

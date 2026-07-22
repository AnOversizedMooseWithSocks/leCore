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

from holographic.mesh_and_geometry.holographic_mesh import Mesh

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


# A 16-entry CASE TABLE: corner m is INSIDE when f < level, and bit m of the case index is that flag. Each case maps
# to its triangle topology (each triangle = 3 edges, an edge = a sorted pair of the 4 tet-corner indices) and the
# point the surface orients TOWARD. This is the per-cell branch of marching_tetrahedra turned into a content-
# addressable lookup -- a sign-pattern -> triangles RAM -- so the whole grid can be marched with vectorized array ops
# instead of a Python per-cell loop. Built once at import.
def _build_tet_case_table():
    table = {}
    for mask in range(16):
        inside = [m for m in range(4) if (mask >> m) & 1]
        outside = [m for m in range(4) if not ((mask >> m) & 1)]
        if not inside or not outside:
            table[mask] = ([], None)
        elif len(inside) == 1:                                  # one triangle, normal toward the lone outside corner
            A = inside[0]
            table[mask] = ([[tuple(sorted((A, o))) for o in outside]], ("corner", outside[0]))
        elif len(inside) == 3:                                  # one triangle, toward the lone outside corner
            D = outside[0]
            table[mask] = ([[tuple(sorted((D, m))) for m in inside]], ("corner", D))
        else:                                                  # two in / two out -> a quad (split q0-q2), toward c+d
            a, b = inside; c, d = outside
            q = [tuple(sorted((a, c))), tuple(sorted((b, c))), tuple(sorted((b, d))), tuple(sorted((a, d)))]
            table[mask] = ([[q[0], q[1], q[2]], [q[0], q[2], q[3]]], ("mid", (c, d)))
    return table


_TET_CASE_TABLE = _build_tet_case_table()


def marching_tetrahedra_vec(values, axes, level=0.0, return_keys=False):
    """VECTORIZED marching tetrahedra -- identical result to marching_tetrahedra (same vertices, same faces, same
    orientation, to machine epsilon) but the whole grid is processed as parallel NumPy ARRAY operations instead of a
    Python per-cell loop. Each tetrahedron's triangle topology is read from _TET_CASE_TABLE (the sign-pattern RAM),
    crossings are deduplicated by a packed edge key (np.unique), and orientation is one batched normal test. ~6-14x
    faster than the per-cell version at working grid sizes (the gap is the Python loop, not the arithmetic).

    `values` is (nx,ny,nz), `axes`=(xs,ys,zs). Returns a watertight triangle Mesh oriented toward INCREASING field
    (outward for an SDF where inside is negative). KEPT HONEST: like the per-cell version it is non-manifold at grid
    sizes where a vertex lands exactly on the isosurface (it reproduces that case identically -- it is a faithful
    parallelization, not a different algorithm).

    return_keys=True ALSO returns a STABLE per-vertex identity array: each marched vertex sits on one grid edge, and
    its key = pack(lo_corner, hi_corner) is canonical and deterministic -- the SAME physical edge gets the SAME key
    in any extraction at this (resolution, bounds), regardless of which other crossings exist. So after a LOCAL field
    edit + re-march, a frontend can track vertices by KEY (persistent identity) instead of by array index (which
    renumbers whenever a crossing is added/removed). Returns (mesh, keys) with keys[v] the int identity of vertex v.
    (Keys are tied to the grid; they are stable across EDITS at a fixed resolution, not across resolution changes.)"""
    values = np.asarray(values, float)
    xs, ys, zs = axes
    nx, ny, nz = values.shape
    cnx, cny, cnz = nx - 1, ny - 1, nz - 1
    if cnx <= 0 or cny <= 0 or cnz <= 0:
        return (Mesh(np.zeros((0, 3)), []), np.zeros(0, np.int64)) if return_keys else Mesh(np.zeros((0, 3)), [])
    NYZ = ny * nz
    BIG = nx * ny * nz                                          # packs an unordered edge (lo,hi corner codes) into 1 int

    ci, cj, ck = np.meshgrid(np.arange(cnx), np.arange(cny), np.arange(cnz), indexing="ij")
    ci = ci.ravel(); cj = cj.ravel(); ck = ck.ravel()

    tri_keys = []
    tri_toward = []
    for tet in _TETS:
        offs = [_CUBE[t] for t in tet]
        fv = []; gco = []; gw = []
        for (dx, dy, dz) in offs:
            fv.append(values[dx:dx + cnx, dy:dy + cny, dz:dz + cnz].ravel())
            I = ci + dx; J = cj + dy; K = ck + dz
            gco.append(I * NYZ + J * nz + K)                   # a unique code per grid corner
            gw.append(np.stack([xs[I], ys[J], zs[K]], axis=1))
        fv = np.stack(fv, axis=1)
        inside = fv < level
        case = (inside[:, 0].astype(np.int64) | (inside[:, 1].astype(np.int64) << 1)
                | (inside[:, 2].astype(np.int64) << 2) | (inside[:, 3].astype(np.int64) << 3))
        for mask, (tris, toward) in _TET_CASE_TABLE.items():
            if not tris:
                continue
            idx = np.where(case == mask)[0]
            if idx.size == 0:
                continue
            if toward[0] == "corner":
                tw = gw[toward[1]][idx]
            else:
                c, d = toward[1]; tw = 0.5 * (gw[c][idx] + gw[d][idx])
            for tri in tris:
                cols = []
                for (ca, cb) in tri:
                    code_a = gco[ca][idx]; code_b = gco[cb][idx]
                    cols.append(np.minimum(code_a, code_b) * BIG + np.maximum(code_a, code_b))
                tri_keys.append(np.stack(cols, axis=1))
                tri_toward.append(tw)

    if not tri_keys:
        return Mesh(np.zeros((0, 3)), [])
    all_keys = np.concatenate(tri_keys, axis=0)
    all_toward = np.concatenate(tri_toward, axis=0)
    uniq, inv = np.unique(all_keys.ravel(), return_inverse=True)
    faces_idx = inv.reshape(-1, 3)

    lo = uniq // BIG; hi = uniq % BIG                          # decode each unique edge's two corners
    li, lr = lo // NYZ, lo % NYZ; lj, lk = lr // nz, lr % nz
    hii, hr = hi // NYZ, hi % NYZ; hj, hk = hr // nz, hr % nz
    fa = values[li, lj, lk]; fb = values[hii, hj, hk]
    wa = np.stack([xs[li], ys[lj], zs[lk]], axis=1)
    wb = np.stack([xs[hii], ys[hj], zs[hk]], axis=1)
    denom = fb - fa
    t = np.where(denom == 0, 0.5, (level - fa) / np.where(denom == 0, 1.0, denom))
    P = wa + t[:, None] * (wb - wa)

    v0 = P[faces_idx[:, 0]]; v1 = P[faces_idx[:, 1]]; v2 = P[faces_idx[:, 2]]
    normal = np.cross(v1 - v0, v2 - v0)
    centroid = (v0 + v1 + v2) / 3.0
    flip = np.sum(normal * (all_toward - centroid), axis=1) < 0.0
    faces_idx[flip] = faces_idx[flip][:, [0, 2, 1]]
    mesh = Mesh(P, [(int(a), int(b), int(c)) for a, b, c in faces_idx])
    if return_keys:
        return mesh, uniq.astype(np.int64)                     # uniq[v] = vertex v's canonical edge identity
    return mesh


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


def _closest_points_on_triangles(P, A, B, C):
    """Closest point on each triangle (A,B,C) to each point P, fully BROADCAST (Ericson's region test). P, A, B, C
    broadcast to a common (..., 3); returns the closest points (..., 3). This is the single batched kernel under the
    whole point-to-mesh family: pass PAIRED inputs (F triangles each with its own block of points, shapes (F,B,3) and
    (F,1,3)) or ALL-PAIRS (N points x F triangles, shapes (N,1,3) and (1,F,3)) and the region logic runs over the
    leading axes in ONE vectorized pass -- replacing the per-triangle Python loop that gated mesh_distance_grid,
    mesh_to_sdf, and surface_deviation. The original _closest_point_on_triangle is the (N,3)-vs-one-triangle case."""
    P, A, B, C = np.broadcast_arrays(np.asarray(P, float), np.asarray(A, float),
                                     np.asarray(B, float), np.asarray(C, float))
    ab = B - A
    ac = C - A
    ap = P - A
    d1 = np.sum(ap * ab, axis=-1)
    d2 = np.sum(ap * ac, axis=-1)
    bp = P - B
    d3 = np.sum(bp * ab, axis=-1)
    d4 = np.sum(bp * ac, axis=-1)
    cq = P - C
    d5 = np.sum(cq * ab, axis=-1)
    d6 = np.sum(cq * ac, axis=-1)
    out = np.zeros(P.shape)
    done = np.zeros(P.shape[:-1], dtype=bool)

    def fill(mask, pts):                                            # first matching region wins (Ericson's ordering)
        m = mask & ~done
        out[m] = np.broadcast_to(pts, out.shape)[m]
        done[m] = True

    def sdiv(num, den):
        return np.divide(num, den, out=np.zeros_like(num), where=den != 0)

    fill((d1 <= 0) & (d2 <= 0), A)                                  # vertex a region
    fill((d3 >= 0) & (d4 <= d3), B)                                 # vertex b region
    fill((d6 >= 0) & (d5 <= d6), C)                                 # vertex c region
    vc = d1 * d4 - d3 * d2                                          # edge ab
    fill((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + sdiv(d1, d1 - d3)[..., None] * ab)
    vb = d5 * d2 - d1 * d6                                          # edge ac
    fill((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + sdiv(d2, d2 - d6)[..., None] * ac)
    va = d3 * d6 - d5 * d4                                          # edge bc
    denom_bc = (d4 - d3) + (d5 - d6)
    fill((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), B + sdiv(d4 - d3, denom_bc)[..., None] * (C - B))
    denom = va + vb + vc                                            # interior (barycentric)
    denom = np.where(denom != 0, denom, 1.0)
    fill(np.ones(P.shape[:-1], dtype=bool), A + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None])
    return out


def _cache_chunk(n_tri, bytes_per_level=6 * 1024 * 1024):
    """Pick an ALL-PAIRS point chunk so one chunk's hot working set -- the (chunk, n_tri) distance/closest-point
    temporaries the kernel streams over -- fits in a fast CPU cache level (default ~6MB, an L2/L3 working budget),
    while the reused triangle arrays (A,B,C, ~n_tri*72 bytes) stay resident across chunks. Blocking to cache is what
    keeps the batched kernel from thrashing memory bandwidth on a big triangle set. Floored at 8, capped at 4096."""
    per_point = max(n_tri, 1) * 8 * 8                              # ~8 (chunk,n_tri) float64 temporaries per point row
    return int(min(4096, max(8, bytes_per_level // max(per_point, 1))))


def point_set_to_mesh_grid(P, V, faces, radius=2, cells_per_axis=None, signed=False):
    """Min distance from each query point P (N,3) to a triangle set, ACCELERATED by a vectorized uniform-grid index
    that CULLS the work -- the genuine speedup the batched all-pairs kernel could not give (that one is memory-bound;
    this one does far less arithmetic). The whole thing is array ops, NO Python per-cell dicts (those were measured
    slower):

      BUILD: bin each triangle into its centroid's grid cell, sort triangles by cell id (argsort), and build a CSR
      [start,count] per cell by bincount+cumsum.
      QUERY: each query looks only at the (2*radius+1)^3 cells around its own cell; the ragged 'gather the triangles
      in those cells' is done with the vectorized RANGES trick (cumsum of an increment array), giving a flat
      (query, candidate-triangle) edge list; the exact closest-point distance is computed for those FEW edges
      (_closest_points_on_triangles, paired) and reduced per query with np.minimum.at.

    This turns an O(N*F) scan into O(N * candidates), where candidates ~ (2r+1)^3 * (triangles per cell) -- a large
    reduction for a surface mesh (most cells are empty). Returns (N,) unsigned distance, or signed-by-nearest-face-
    normal if `signed`.

    KEPT HONEST (APPROXIMATE by construction): a triangle sits in its CENTROID cell only, and a query only sees a
    finite radius, so the TRUE nearest triangle is guaranteed found only when it lies within `radius` cells of the
    query's cell -- correct for near-surface queries on a roughly uniform mesh (the LOD/deviation use), but a large
    triangle whose centroid is far, or a query far from the surface, can be missed. A query whose neighbourhood holds
    no triangle returns +inf (caller can widen `radius` or fall back to the exact point_set_to_mesh). Raise `radius`
    or `cells_per_axis` to trade speed for guaranteed coverage."""
    V = np.asarray(V, float)
    F = np.asarray([f[:3] for f in faces], dtype=int)
    A = V[F[:, 0]]; B = V[F[:, 1]]; C = V[F[:, 2]]
    P = np.asarray(P, float)
    cent = (A + B + C) / 3.0
    lo = np.minimum(V.min(axis=0), P.min(axis=0))
    hi = np.maximum(V.max(axis=0), P.max(axis=0))
    span = np.maximum(hi - lo, 1e-9)
    if cells_per_axis is None:
        cells_per_axis = int(np.clip(round(2.0 * len(F) ** (1.0 / 3.0)), 4, 128))   # ~triangle-sized cells
    dims = np.array([cells_per_axis] * 3, dtype=int)
    cell = (hi - lo) / dims
    cell = np.where(cell > 0, cell, 1.0)

    def cell_of(X):                                                  # point(s) -> integer cell coords, clamped
        return np.clip(((X - lo) / cell).astype(int), 0, dims - 1)

    ny, nz = int(dims[1]), int(dims[2])
    tri_cell = cell_of(cent)
    tri_lin = tri_cell[:, 0] * ny * nz + tri_cell[:, 1] * nz + tri_cell[:, 2]        # (F,) linear cell id
    ncells = int(dims[0] * ny * nz)
    order = np.argsort(tri_lin, kind="stable")                      # triangles grouped by cell in this order
    counts = np.bincount(tri_lin, minlength=ncells)                 # triangles per cell
    starts = np.zeros(ncells, dtype=int)
    starts[1:] = np.cumsum(counts)[:-1]                             # CSR offsets into `order`

    off = np.stack(np.meshgrid(*([np.arange(-radius, radius + 1)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    qc = cell_of(P)                                                 # (N,3)
    ncoord = qc[:, None, :] + off[None, :, :]                       # (N,B,3) neighbour cells
    inb = np.all((ncoord >= 0) & (ncoord < dims), axis=2)           # (N,B) in-bounds
    ncl = np.clip(ncoord, 0, dims - 1)
    ncl_lin = ncl[..., 0] * ny * nz + ncl[..., 1] * nz + ncl[..., 2]                 # (N,B)
    pair_start = starts[ncl_lin].ravel()                           # (N*B,)
    pair_count = np.where(inb, counts[ncl_lin], 0).ravel()         # (N*B,) 0 for out-of-bounds
    pair_query = np.repeat(np.arange(len(P)), off.shape[0])        # (N*B,) which query each pair belongs to

    keep = pair_count > 0
    s = pair_start[keep]; c = pair_count[keep]; q = pair_query[keep]
    best = np.full(len(P), np.inf)
    sgn = np.ones(len(P))
    if c.size:
        total = int(c.sum())
        incr = np.ones(total, dtype=int)                          # vectorized concatenated ranges [s_i, s_i+c_i)
        seg = np.cumsum(c) - c                                    # where each range begins in the flat output
        incr[seg[1:]] = s[1:] - (s[:-1] + c[:-1] - 1)            # jump from one range's end to the next's start
        incr[0] = s[0]
        pos = np.cumsum(incr)                                     # positions into `order`
        tri = order[pos]                                          # candidate triangle indices (flat)
        eq = np.repeat(q, c)                                      # the query for each candidate edge
        Pe = P[eq]
        cp = _closest_points_on_triangles(Pe, A[tri], B[tri], C[tri])               # paired: one query vs one triangle
        de = np.linalg.norm(Pe - cp, axis=1)
        np.minimum.at(best, eq, de)                              # nearest distance per query
        if signed:
            fn = np.cross(B - A, C - A)
            ln = np.linalg.norm(fn, axis=1)
            fn = fn / np.where(ln[:, None] > 1e-12, ln[:, None], 1.0)
            srt = np.lexsort((de, eq))                            # sort edges by query then distance
            qs = eq[srt]
            first = np.ones(len(qs), dtype=bool)
            first[1:] = qs[1:] != qs[:-1]                         # first per query = the minimum-distance edge
            win = srt[first]
            wq = qs[first]
            sv = np.sign(np.sum((P[wq] - cp[win]) * fn[tri[win]], axis=1))
            sgn[wq] = np.where(sv == 0, 1.0, sv)
    return sgn * best if signed else best


def point_set_to_mesh(P, V, faces, chunk=None, signed=False):
    """Min distance from each query point P (N,3) to a triangle set, via the batched kernel
    (_closest_points_on_triangles): per cache-sized chunk it forms the all-pairs closest points and reduces over the
    triangle axis. Returns (N,) unsigned distance, or signed-by-nearest-face-normal if `signed`. A clean, EXACT,
    unified point-to-mesh API (matches the brute loop to machine epsilon).

    *** KEPT NEGATIVE, MEASURED -- this is NOT faster than the per-triangle brute loop for a large mesh, and the
    cache-blocking sweep shows why. The all-pairs (chunk, F, 3) intermediates make this MEMORY-BANDWIDTH-BOUND:
    smaller chunks are faster (less working set) but no chunk beats the brute F-loop, which never materializes the
    (N,F) array and keeps a tiny per-triangle working set that stays cache-resident. For an ACTUAL speedup use
    point_set_to_mesh_grid, which culls the O(N*F) work with a vectorized spatial index instead of vectorizing the
    dense reduction. This batched form is kept as the exact reference kernel. ***"""
    F = np.asarray([f[:3] for f in faces], dtype=int)
    A = V[F[:, 0]]
    B = V[F[:, 1]]
    C = V[F[:, 2]]
    if signed:
        fn = np.cross(B - A, C - A)
        ln = np.linalg.norm(fn, axis=1)
        fn = np.where(ln[:, None] > 1e-12, fn / np.where(ln[:, None] > 1e-12, ln[:, None], 1.0), fn)
    P = np.asarray(P, float)
    if chunk is None:
        chunk = _cache_chunk(len(F))
    best = np.empty(len(P))
    sgn = np.ones(len(P))
    rng = np.arange(min(chunk, len(P)))
    for i in range(0, len(P), chunk):
        Pc = P[i:i + chunk]
        r = rng[:len(Pc)]
        cp = _closest_points_on_triangles(Pc[:, None, :], A[None, :, :], B[None, :, :], C[None, :, :])  # (c,F,3)
        d = np.linalg.norm(Pc[:, None, :] - cp, axis=2)            # (c,F)
        j = np.argmin(d, axis=1)                                   # nearest triangle per point
        best[i:i + chunk] = d[r, j]
        if signed:
            cpn = cp[r, j]                                         # the nearest closest-point
            s = np.sign(np.sum((Pc - cpn) * fn[j], axis=1))
            sgn[i:i + chunk] = np.where(s == 0, 1.0, s)
    return sgn * best if signed else best


def _face_normal(V, f):
    """Unit normal from the FIRST THREE vertices of `f` -- the triangle normal, valid for triangles and for planar
    faces.

    **NOT Newell's method, and not a duplicate of `meshverbs.newell_normal`**, though a structural AST scan reports
    the same shape. On a bent quad the two disagree by 0.47: Newell integrates every edge, this reads three
    vertices. *Structurally identical, semantically different.* Named here so nobody merges them."""
    a, b, c = V[f[0]], V[f[1]], V[f[2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else n


def mesh_distance_grid(mesh, bounds, res=48, band=None, method="shell"):
    """Build a SIGNED banded distance grid (a banded SDF) from a mesh -- the mesh -> field direction.

    method="shell" (default): the FAST build, O(SURFACE AREA) not O(triangle count). It applies the work-culling
    lesson to the build itself -- instead of SCATTERING a distance from every triangle (a Python loop), it marks the
    near-surface SHELL voxels (the voxels holding a vertex / centroid / edge-midpoint, dilated by the band width) and
    answers them with ONE vectorized grid-culled point-to-mesh query (point_set_to_mesh_grid). Since only the shell is
    touched, the cost no longer grows with the triangle count -- a big win on large meshes. Verified equivalent for
    every downstream use: near-surface samples match the scatter build to machine epsilon, and the flood-filled SDF
    re-marches identically (band-EDGE voxels, which are clamped to +-band and never touch the zero level, may differ
    in which get clamped -- benign).

    method="scatter": the original per-triangle scatter -- each triangle scatter-mins the signed distance into the
    local block of voxels within `band` of its bbox. Slower (O(F*block)) but the exact reference, and robust to
    pathologically large triangles (the shell build can under-cover a triangle wider than ~2*band, a documented edge
    case; marched meshes are finely tessellated and never hit it).

    SIGNED on purpose: an unsigned field has a kink at the surface so trilinear sampling there overestimates by ~half
    a voxel; a signed field crosses zero LINEARLY, so |sample| near it is sub-voxel accurate. Returns (grid res^3,
    (xs,ys,zs)). KEPT HONEST: nearest-face-normal sign (can mis-sign deep concavities / non-watertight); far interior
    defaults to +band (use mesh_to_sdf_grid for a flood-filled, re-marchable SDF)."""
    lo = np.asarray(bounds[0], float)
    hi = np.asarray(bounds[1], float)
    res = int(res)
    xs = np.linspace(lo[0], hi[0], res)
    ys = np.linspace(lo[1], hi[1], res)
    zs = np.linspace(lo[2], hi[2], res)
    h = (hi - lo) / max(res - 1, 1)
    if band is None:
        band = 4.0 * float(h.max())
    grid = np.full((res, res, res), band, float)                     # +band = outside, the default
    V = mesh.vertices

    if method == "shell":
        F = np.asarray([f[:3] for f in mesh.faces], dtype=int)
        A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        seeds = np.vstack([V, (A + B + C) / 3.0,                      # vertices + centroids + the 3 edge-midpoints
                           (A + B) / 2.0, (B + C) / 2.0, (A + C) / 2.0])
        sidx = np.clip(((seeds - lo) / h).astype(int), 0, res - 1)
        occ = np.zeros((res, res, res), dtype=bool)
        occ[sidx[:, 0], sidx[:, 1], sidx[:, 2]] = True
        bw = max(1, int(np.ceil(band / float(h.min()))))             # dilate the seeds out to the band width
        for _ in range(bw):
            nb = occ.copy()
            nb[1:, :, :] |= occ[:-1, :, :]; nb[:-1, :, :] |= occ[1:, :, :]
            nb[:, 1:, :] |= occ[:, :-1, :]; nb[:, :-1, :] |= occ[:, 1:, :]
            nb[:, :, 1:] |= occ[:, :, :-1]; nb[:, :, :-1] |= occ[:, :, 1:]
            occ = nb
        vox = np.argwhere(occ)                                        # the shell voxels -- the only ones to solve
        if len(vox):
            centers = lo + vox * h
            d = point_set_to_mesh_grid(centers, V, mesh.faces, radius=2, signed=True)
            d = np.where(np.isinf(d), band, np.clip(d, -band, band))  # out-of-reach band-edge -> +band (benign)
            grid[vox[:, 0], vox[:, 1], vox[:, 2]] = d
        return grid, (xs, ys, zs)

    for f in mesh.faces:                                             # method == "scatter": the exact reference
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm = nrm / ln
        tlo = np.minimum(np.minimum(a, b), c) - band                 # the triangle's bbox, grown by the band
        thi = np.maximum(np.maximum(a, b), c) + band
        ilo = np.clip(np.floor((tlo - lo) / h).astype(int), 0, res - 1)
        ihi = np.clip(np.ceil((thi - lo) / h).astype(int), 0, res - 1)
        bx, by, bz = xs[ilo[0]:ihi[0] + 1], ys[ilo[1]:ihi[1] + 1], zs[ilo[2]:ihi[2] + 1]
        if bx.size == 0 or by.size == 0 or bz.size == 0:
            continue
        gx, gy, gz = np.meshgrid(bx, by, bz, indexing="ij")          # this triangle's local block of voxel centres
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        cp = _closest_point_on_triangle(pts, a, b, c)
        diff = pts - cp
        d = np.linalg.norm(diff, axis=1)
        s = np.sign(np.einsum("ij,j->i", diff, nrm))                 # + outside, - inside (nearest-normal sign)
        s = np.where(s == 0, 1.0, s)
        signed = (s * d).reshape(gx.shape)
        sub = grid[ilo[0]:ihi[0] + 1, ilo[1]:ihi[1] + 1, ilo[2]:ihi[2] + 1]
        closer = np.abs(signed) < np.abs(sub)                        # nearest triangle wins, keeping its sign
        sub[closer] = signed[closer]                                 # local scatter-min by magnitude, in place
    return grid, (xs, ys, zs)


def sample_distance_grid(grid, axes, points):
    """Trilinearly sample a banded SDF grid (from mesh_distance_grid) at query points (N,3) -> (N,) SIGNED distances
    (take np.abs for unsigned surface distance). The O(V) read that turns the once-built grid into a point-to-surface
    distance for any points -- e.g. an LOD level's vertices against the original it was decimated from. Because the
    field is signed (zero-crossing is linear), this resolves distances well below the grid spacing near the surface."""
    xs, ys, zs = axes
    P = np.asarray(points, float)
    lo = np.array([xs[0], ys[0], zs[0]])
    h = np.array([xs[1] - xs[0] if len(xs) > 1 else 1.0,
                  ys[1] - ys[0] if len(ys) > 1 else 1.0,
                  zs[1] - zs[0] if len(zs) > 1 else 1.0])
    n = np.array(grid.shape)
    g = (P - lo) / h
    i0 = np.clip(np.floor(g).astype(int), 0, n - 2)
    frac = np.clip(g - i0, 0.0, 1.0)
    out = np.zeros(len(P))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, frac[:, 0], 1 - frac[:, 0]) *
                     np.where(dy, frac[:, 1], 1 - frac[:, 1]) *
                     np.where(dz, frac[:, 2], 1 - frac[:, 2]))
                out += w * grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


def flood_fill_sign(grid, band):
    """Turn a BANDED signed SDF (from mesh_distance_grid -- band voxels carry exact signed distance, far voxels
    default to +band) into a FULL signed SDF by flood-filling the OUTSIDE inward from the grid boundary. The negative
    band shell around the surface blocks the flood, so any far voxel the boundary cannot reach is ENCLOSED (interior)
    and is set to -band. The result has the surface as a true zero level set and is RE-MARCHABLE -- so an imported
    mesh, once it is a field, inherits the field-native LOD (re-march coarser) instead of needing mesh decimation.
    Returns a NEW grid (input untouched).

    The flood is a vectorized iterative 6-neighbour dilation through {value >= 0}, converging in O(grid diameter)
    cheap array-shift passes -- it touches no triangles, so it sidesteps the memory-bandwidth wall that the
    point-to-mesh batching hit. KEPT HONEST: it relies on the negative band shell being watertight on the grid, so a
    too-thin band (keep >= ~2 voxels) or a non-watertight / nearest-normal-mis-signed mesh lets the flood leak
    (interior wrongly stays positive). The band distances themselves are never changed and are always correct."""
    g = np.array(grid, dtype=float)                                # copy; never mutate the caller's grid
    outside = g >= 0.0                                             # candidate exterior (incl. far default +band)
    reach = np.zeros(g.shape, dtype=bool)
    reach[0, :, :] = reach[-1, :, :] = True                        # seed from all six boundary faces
    reach[:, 0, :] = reach[:, -1, :] = True
    reach[:, :, 0] = reach[:, :, -1] = True
    reach &= outside
    while True:                                                    # dilate, blocked by the negative shell (~outside)
        nxt = reach.copy()
        nxt[1:, :, :] |= reach[:-1, :, :]
        nxt[:-1, :, :] |= reach[1:, :, :]
        nxt[:, 1:, :] |= reach[:, :-1, :]
        nxt[:, :-1, :] |= reach[:, 1:, :]
        nxt[:, :, 1:] |= reach[:, :, :-1]
        nxt[:, :, :-1] |= reach[:, :, 1:]
        nxt &= outside
        if np.array_equal(nxt, reach):
            break
        reach = nxt
    g[outside & ~reach] = -band                                    # exterior-valued but unreachable = interior
    return g


def open_fraction(mesh):
    """Fraction of edges that are BOUNDARY (used by exactly one face). 0.0 = edge-closed; a scan-style triangle
    soup measures near 1 (the mantis regression case measured 0.71). The cheap probe that routes the sign method."""
    from collections import Counter
    cnt = Counter()
    for f in mesh.faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            cnt[(a, b) if a < b else (b, a)] += 1
    if not cnt:
        return 1.0
    return sum(1 for v in cnt.values() if v == 1) / float(len(cnt))


def mesh_to_sdf_grid(mesh, bounds, res=48, band=None, sign="auto", open_threshold=0.05):
    """Build a FULL, re-marchable signed distance grid from a mesh -- the complete mesh -> field conversion.
    Returns (grid res^3, (xs,ys,zs)).

    `sign` chooses where the INSIDE comes from, because the two failure modes are disjoint:
      * "flood"   -- the original path: banded nearest-normal sign + flood_fill_sign. Exact and cheap for
        edge-closed meshes; on an OPEN mesh the negative shell has holes, the flood LEAKS, and marching the
        result yields scattered garbage blobs (the mantis .glb regression: a Sketchfab scan soup with 71%
        boundary edges shredded into disconnected chunks -- reproduced, pinned).
      * "winding" -- magnitude from the same distance build, sign from the GENERALISED WINDING NUMBER
        (Jacobson et al. 2013) via the fast cluster-dipole sum (Barill et al. 2018; measured 113x over the
        exact sum on the regression scan). No closure or orientation assumption -- the published robust answer
        for soups. Costs O(voxels x cells) + exact near-cells.
      * "auto" (default) -- probe open_fraction(mesh): > `open_threshold` routes to "winding", else "flood".
        BACKWARD COMPATIBLE BY MEASUREMENT, not by promise: an edge-closed mesh takes the flood path
        BIT-IDENTICALLY (pinned by selftest); only the class whose flood output was documented garbage moves.
    """
    grid, axes = mesh_distance_grid(mesh, bounds, res=res, band=band)
    lo = np.asarray(bounds[0], float)
    hi = np.asarray(bounds[1], float)
    band_val = 4.0 * float(((hi - lo) / max(int(res) - 1, 1)).max()) if band is None else band
    if sign == "auto":
        sign = "winding" if open_fraction(mesh) > open_threshold else "flood"
    if sign == "flood":
        return flood_fill_sign(grid, band_val), axes
    if sign != "winding":
        raise ValueError("sign must be 'auto', 'flood', or 'winding', got %r" % (sign,))
    from holographic.mesh_and_geometry.holographic_voxelize import fast_winding_number
    xs, ys, zs = axes
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    tris = [f[:3] for f in mesh.faces]
    w = fast_winding_number(pts, mesh.vertices, tris).reshape(grid.shape)
    inside = np.abs(w) > 0.5                                     # |w|: robust to either global orientation
    # magnitude from the band build (clamped to band far away), sign purely from the winding number -- the
    # nearest-normal sign is DISCARDED because on a soup it is exactly the broken part
    return np.where(inside, -np.abs(grid), np.abs(grid)), axes


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
def voxel_remesh(mesh, resolution=64, pad=0.2, sign="auto", keep_uv="auto"):
    """VOXEL REMESH (Blender's Voxel Remesh): rebuild a mesh as a UNIFORM, watertight surface by sampling it into a
    signed-distance grid and re-marching -- the standard cleanup for messy, self-intersecting, non-manifold, or
    multi-shell input before retopo. Any tangle in becomes one clean closed surface out, at a density set by
    `resolution` (cells per axis). `pad` (RELATIVE to the mesh size) grows the bounds so the SDF band clears the grid
    edge and the surface always closes -- the default 0.2 is watertight; drop it only if you know the mesh is small
    in its box.

    A pure COMPOSE: mesh_to_sdf_grid (the winding/flood-filled SDF) -> marching_tetrahedra_vec (the mesher), both
    already here. Returns a watertight triangle Mesh. This is what turns the skin_skeleton block-out, a boolean mess,
    or an imported scan into something with consistent topology. KEPT NEGATIVE: uniform density loses sharp features
    below the cell size (a thin blade or a crisp corner rounds off -- raise resolution or crease after); the sign is
    nearest-normal, so a deeply non-watertight input can leak (voxel remesh wants a roughly-closed shell)."""
    V = np.asarray(mesh.vertices, float)
    span = float((V.max(0) - V.min(0)).max()) or 1.0
    p = float(pad) * span                                # pad relative to the mesh size, so the SDF band clears the
    lo = V.min(0) - p; hi = V.max(0) + p                 # grid boundary and the marched surface always closes
    grid, axes = mesh_to_sdf_grid(mesh, (lo.tolist(), hi.tolist()), res=int(resolution), sign=sign)
    out = marching_tetrahedra_vec(grid, axes, level=0.0)
    # keep_uv="auto": a remesh REPLACES the topology, so uvs can only be PROJECTED from the source surface
    # (transfer_uv -- exact carry is meaningless across a re-march). Same honest caveat as everywhere: at a UV
    # seam the nearest source triangle may sit on either atlas island; the projection residual is the error
    # signal. keep_uv=False = geometry only (the old output, byte-identical).
    src_uv = getattr(mesh, "uvs", None) if keep_uv in ("auto", True) else None
    if src_uv is not None and len(out.vertices):
        try:
            from holographic.mesh_and_geometry.holographic_meshtools import transfer_uv, uv_atlas_report
            atlas = uv_atlas_report(mesh)
            # same honesty rule as cluster_decimate: a fragmented (per-triangle) atlas cannot survive ANY
            # per-vertex transfer -- refuse and point at rebake_texture instead of emitting confetti uvs.
            if keep_uv is True:
                # FORCED legacy path: plain per-vertex transfer, exactly as before reprojection existed. This is
                # the documented escape hatch and it must not quietly change meaning -- a caller who asked to
                # force it gets the old behaviour (uvs present, seams smeared on a fragmented atlas), not a
                # refusal. Routing True through reproject_uv broke that: reprojection RAISES on a fragmented
                # atlas, so "force" silently became "no uvs at all". The selftest caught it.
                new_uv, _ = transfer_uv(mesh, np.asarray(src_uv, float), np.asarray(out.vertices, float))
                out.uvs = new_uv
                out.uv_transfer_report = dict(atlas, skipped=False, reprojected=False, forced=True)
            elif not atlas["transferable"]:
                out.uv_transfer_report = dict(atlas, skipped=True,
                                              reason="fragmented atlas (median %.1f faces/island): use "
                                                     "meshtools.rebake_texture, or keep_uv=True to force"
                                                     % atlas["faces_per_island_median"])
            else:
                # same reasoning as cluster_decimate: remeshing welds the seams, so REPROJECT (per-corner,
                # re-splitting the cuts) rather than transfer per vertex.
                from holographic.mesh_and_geometry.holographic_meshtools import reproject_uv
                try:
                    rmesh, new_uv, rrep = reproject_uv(mesh, np.asarray(src_uv, float), out)
                    rmesh.uv_transfer_report = dict(atlas, skipped=False, reprojected=True, **{
                        k: rrep[k] for k in ("seam_splits", "projection_distance_mean", "finite")})
                    return rmesh
                except ValueError:
                    out.uv_transfer_report = dict(atlas, skipped=True,
                                                  reason="reprojection refused: fragmented atlas -- use "
                                                         "meshtools.rebake_texture")
                    return out
        except Exception:
            pass
    return out


def metaball_mesh(centers, radius=0.5, level=0.5, resolution=48, pad=0.6):
    """METABALL MESH (Blender metaballs / the classic soft-blob base mesh): sum-of-Gaussians field -> marching-cubes
    surface. `centers` (n,3) are the blob positions, `radius` their spread, `level` the isovalue (higher = tighter
    around each center, lower = blobs merge more). Overlapping blobs FUSE smoothly -- the organic-blob route that
    complements skin_skeleton (blobs where skeleton branch-stitching gets ugly: a torso lump, a knuckle cluster).
    Returns a watertight triangle Mesh. `pad` (relative to the blob spread) frames the grid so the surface closes.

    Reuses metaball_field (the engine's splat-as-implicit-field) -- a `bundle` of Gaussians IS a superposition, and
    thresholding it is an isosurface, the 'as above so below' identity in mesh form. KEPT NEGATIVE: isotropic-triangle
    blob topology (a base mesh to retopo onto, like skin_skeleton); `level` vs `radius` interact -- too high a level on
    far-apart centers yields separate shells (they only merge where the summed field clears `level`)."""
    C = np.asarray(centers, float)
    field = metaball_field(C, radius=radius)
    ext = radius * (1.0 + float(pad))
    lo = C.min(0) - ext; hi = C.max(0) + ext
    n = int(resolution)
    xs = np.linspace(lo[0], hi[0], n); ys = np.linspace(lo[1], hi[1], n); zs = np.linspace(lo[2], hi[2], n)
    GX, GY, GZ = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
    vals = field(P).reshape(n, n, n)
    # metaball_field is a SUM (higher inside), but marching_tetrahedra expects inside-negative; negate around `level`
    return marching_tetrahedra_vec(level - vals, (xs, ys, zs), level=0.0)


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

    # --- mesh -> FIELD (banded SDF by tiling): a coarse vertex's |sample| matches the analytic surface distance,
    #     sub-voxel, where an unsigned field's kink could not ---
    dgrid, daxes = mesh_distance_grid(sphere, ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=48)
    near = sphere.vertices[:50] * 1.03                      # points just OFF the unit sphere (within the band)
    samp = sample_distance_grid(dgrid, daxes, near)
    truth = np.linalg.norm(near, axis=1) - 1.0             # analytic signed distance to the unit sphere
    voxel = 3.0 / 47
    assert np.abs(samp - truth).max() < 0.5 * voxel, "signed banded SDF must resolve distance to well under a voxel"
    assert np.array_equal(mesh_distance_grid(sphere, ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=48)[0], dgrid)  # deterministic

    # --- the BATCHED closest-point kernel matches the single-triangle one to machine epsilon, both shapes ---
    _rng = np.random.default_rng(1)
    _ta, _tb, _tc = _rng.standard_normal(3), _rng.standard_normal(3), _rng.standard_normal(3)
    _qp = _rng.standard_normal((60, 3))
    _o1 = _closest_point_on_triangle(_qp, _ta, _tb, _tc)
    _o2 = _closest_points_on_triangles(_qp[:, None, :], _ta, _tb, _tc)[:, 0, :]   # all-pairs broadcast shape
    assert np.abs(_o1 - _o2).max() < 1e-12, "batched kernel must match the single-triangle kernel"
    _pm = point_set_to_mesh(near, sphere.vertices, sphere.faces)                  # the unified API, vs the selftest's analytic check
    assert np.abs(_pm - np.abs(np.linalg.norm(near, axis=1) - 1.0)).max() < 0.02, "point_set_to_mesh tracks the analytic distance"
    _pg = point_set_to_mesh_grid(near, sphere.vertices, sphere.faces, radius=2)   # the grid-accelerated kernel, near-surface
    assert not np.any(np.isinf(_pg)) and np.abs(_pg - _pm).max() < 1e-9, "grid point-to-mesh is exact near the surface"

    # --- FULL SDF via flood-fill: interior filled negative, re-marchable back to the surface ---
    _full, _faxes = mesh_to_sdf_grid(sphere, ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=48)
    _mid = _full.shape[0] // 2
    assert _full[_mid, _mid, _mid] < 0.0, "the interior must be filled negative by the flood fill"
    assert _full[0, 0, 0] > 0.0, "a far corner stays outside (positive)"
    _remarch = marching_tetrahedra(_full, _faxes, 0.0)
    assert _remarch.is_closed() and _remarch.n_faces > 0, "the full SDF must re-march to a closed surface"
    _rr = np.linalg.norm(_remarch.vertices, axis=1)
    assert abs(float(_rr.mean()) - 1.0) < 0.03, "re-marched surface should sit back on the unit sphere"

    # --- determinism ---
    assert np.array_equal(marching_tetrahedra(vals, axes, 0.0).vertices, marching_tetrahedra(vals, axes, 0.0).vertices)

    # voxel_remesh: a box rebuilds to a uniform closed surface (the standard messy-input cleanup). Marching
    # tetrahedra can be non-manifold at grid-coincident points (its declared negative), so check watertightness via
    # boundary edges (a robust closed-surface test) rather than the strict half-edge manifold assertion.
    from holographic.mesh_and_geometry.holographic_mesh import box as _vbox
    from collections import Counter as _VC
    _vr = voxel_remesh(_vbox(), resolution=36)
    _vec = _VC()
    for _vf in _vr.faces:
        for _a in range(len(_vf)):
            _vec[tuple(sorted((_vf[_a], _vf[(_a + 1) % len(_vf)])))] += 1
    assert _vr.n_faces > 0 and sum(1 for _cc in _vec.values() if _cc == 1) == 0   # no boundary edges = closed

    # metaball_mesh: two overlapping Gaussian blobs FUSE into one watertight shell (the soft-blob base-mesh route).
    _mb = metaball_mesh(np.array([[0.0, 0, 0], [0.4, 0, 0]]), radius=0.4, level=0.5, resolution=32)
    _mbe = _VC()
    for _mf in _mb.faces:
        for _a in range(len(_mf)):
            _mbe[tuple(sorted((_mf[_a], _mf[(_a + 1) % len(_mf)])))] += 1
    assert _mb.n_faces > 0 and sum(1 for _cc in _mbe.values() if _cc == 1) == 0   # fused, closed

    print(f"holographic_meshbridge selftest: ok (SDF->mesh: unit sphere extracted, {sphere.n_faces} faces, closed "
          f"manifold chi={sphere.euler_characteristic()}, vertices on sphere mean r={radii.mean():.3f}+/-{radii.std():.3f}; "
          f"mesh->SDF matches analytic |p|-1 at probes (<0.05); mesh->FIELD signed banded SDF resolves surface distance "
          f"to <half a voxel (the kink fix); splat->mesh: {blob.n_faces}-face blob, closed manifold; deterministic)")


if __name__ == "__main__":
    _selftest()


def _selftest_soup_sign():
    """Pin the .glb-scan regression and its fix. Three contracts:
    (1) BIT-IDENTICAL backward compat: an edge-closed mesh under sign='auto' equals the old flood path exactly.
    (2) THE REGRESSION, kept loud: on an open triangle soup the flood path mis-signs the interior badly.
    (3) THE FIX: winding sign recovers a sane interior on the same soup at the same resolution."""
    rng = np.random.default_rng(0)
    from holographic.mesh_and_geometry.holographic_mesh import Mesh

    # (1) closed cube: auto must route to flood and match it to the bit
    Vc = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    Fc = [t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    cube = Mesh(Vc, Fc)
    assert open_fraction(cube) == 0.0
    b = ((-0.3, -0.3, -0.3), (1.3, 1.3, 1.3))
    g_auto, _ = mesh_to_sdf_grid(cube, b, res=24, sign="auto")
    g_old, _ = mesh_to_sdf_grid(cube, b, res=24, sign="flood")
    assert np.array_equal(g_auto, g_old), "closed mesh must take the flood path BIT-IDENTICALLY"

    # a SLIT SOUP of the unit sphere -- the honest model of a scan like the mantis: a NEARLY-CONTIGUOUS surface
    # whose every triangle is disconnected. Built by subdividing an octahedron onto the sphere (128 tris) and
    # SHRINKING each triangle 20% about its centroid with its own private vertices: 100% boundary edges,
    # slit gaps ~0.064 (super-voxel at res 64, voxel 0.041), solid-angle coverage 0.8^2 = 64%.
    # KEPT NEGATIVE FROM WRITING THIS TEST: the first soup (isolated random patches) covered only ~8% of the
    # sphere's solid angle, so the winding number CORRECTLY said "not inside" (w ~ coverage). Winding sign fixes
    # surfaces with holes; it cannot invent a surface that mostly is not there. That is a property, not a bug.
    # 2 subdivisions (128 tris), not 3: the slit width scales with EDGE length (slit ~ shrink * edge / sqrt(3)),
    # so fewer/bigger triangles buy wider slits at the SAME coverage. Measured while writing this: at 512 tris
    # the slit (0.032) is SMALLER than the res-64 voxel (0.041), the shell closes on the grid, and flood signs
    # 100% -- the test silently tests nothing. At 128 tris the slit is 0.064 > voxel and the leak fires.
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
    tris = Vs[np.asarray(Fs)]                                    # (512, 3, 3)
    centroids = tris.mean(axis=1, keepdims=True)
    shrunk = centroids + 0.8 * (tris - centroids)                # 20% slit, every triangle its own island
    V = shrunk.reshape(-1, 3)
    F = [(3*i, 3*i+1, 3*i+2) for i in range(len(tris))]
    soup = Mesh(V, F)
    assert open_fraction(soup) == 1.0
    bs = ((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3))
    # res 64 pins the BROKEN side of the flood's sharp resolution threshold (sweep on record: a dense-patch
    # draft of this soup flood-signed 100% at res 40 -- the dilated shell closed ON THE GRID and blocked the
    # flood -- and 3% at res 64 once the holes exceeded the voxel. The failure is a THRESHOLD, not a decay,
    # which is why the mantis .glb "suddenly" regressed when render fidelity rose.)
    res = 64
    xs = np.linspace(-1.3, 1.3, res)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    truly_inside = np.sqrt(gx**2 + gy**2 + gz**2) < 0.85         # safely interior, away from the slitted shell
    g_flood, _ = mesh_to_sdf_grid(soup, bs, res=res, sign="flood")
    g_wind, _ = mesh_to_sdf_grid(soup, bs, res=res, sign="winding")
    flood_ok = float((g_flood[truly_inside] < 0).mean())
    wind_ok = float((g_wind[truly_inside] < 0).mean())
    # (2) the regression: the leaked flood misses most of the interior on a soup. Pinned as a NEGATIVE so a
    # future "improvement" to flood_fill_sign that silently fixes soups gets noticed and promoted deliberately.
    assert flood_ok < 0.6, "KEPT NEGATIVE moved: flood now signs soup interiors (%.2f) -- re-evaluate the auto route" % flood_ok
    # (3) the fix: winding signs the same interior, at the same res. Bound is 0.9 not 0.99 because deep-interior
    # w equals the solid-angle coverage (~0.64 here) -- above threshold everywhere but not by miles; the mantis
    # itself (coverage near 1 with slits) measured w up to 1.01.
    assert wind_ok > 0.9, wind_ok
    # and auto routes the soup to winding
    g_auto2, _ = mesh_to_sdf_grid(soup, bs, res=res, sign="auto")
    assert np.array_equal(g_auto2, g_wind)
    print("soup-sign selftest OK (closed: auto==flood bit-identical; soup interior signed: flood %.0f%% vs "
          "winding %.0f%%)" % (100 * flood_ok, 100 * wind_ok))


if __name__ == "__main__":
    _selftest_soup_sign()


# ======================================================================================================
# SCULPT-MODE PREPARATION -- the GUARDED mesh -> sdf-cache conversion
# ======================================================================================================

def sculpt_prepare(mesh, resolution=48, silhouette=0.95, max_resolution=160, pad_frac=0.08,
                   n_azimuth=6, band=None):
    """Prepare a mesh for SCULPT MODE: build the SDF cache and the sculptable remesh, GUARDED so the
    conversion cannot silently change the shape.

    WHY THIS EXISTS (the measured bug it closes): the raw pipeline -- mesh_to_sdf_grid -> march -- has two
    independent shape-drift modes, both reproduced and quantified before this was written:
      * THIN FEATURES die at coarse resolution (a 0.04-thick fin on a test box: worst-view silhouette IoU
        0.621 at res 32, the fin's top simply gone).
      * TOUCHING/COINCIDENT SHELLS break the flood-fill sign, and escalating resolution makes it WORSE, not
        better (the same box: IoU 0.734 @ 48 -> 0.460 @ 64 -> 0.250 @ 96 under sign='auto' -- higher res
        resolves the zero-thickness contact and the fill floods through). The generalised winding number is
        robust on the same mesh (0.954 / 0.967 / 0.967, monotone).
    A resolution-only guard therefore CANNOT save the second mode; the guard must also switch the SIGN method.

    THE LADDER, in cost order: (1) convert at `resolution` with sign='auto'; (2) if the worst-view IoU is
    below `silhouette`, retry the SAME resolution with sign='winding' (the coincident-shell fix); (3) keep the
    better sign and escalate resolution x1.5 (re-testing that sign) until the floor passes, `max_resolution`
    is hit, or improvement stalls; (4) if the floor is still unmet, raise ValueError CARRYING THE FULL REPORT
    -- the destructive conversion is REFUSED, not shipped, which is the point. `silhouette=None` runs exactly
    one unguarded conversion (preservation is the default; destruction is the explicit opt-out, matching
    silhouette_guarded's convention).

    Returns {'mesh': the sculptable remarch, 'grid': (res,res,res) signed distances, 'axes': (xs,ys,zs),
    'bounds': the padded box, 'report': {'iou': worst-view IoU, 'view': worst view, 'sign': sign used,
    'resolution': final res, 'ladder': [(res, sign, iou), ...]}} -- everything sculpt mode needs in one call,
    with the grid and mesh guaranteed to be the SAME level of the SAME field.
    Deterministic; cost is the ladder's converts + one silhouette sweep per rung (~1 s per rung at res 48).

    KEPT HONEST: SHARP low-poly shapes (a raw tetrahedron, hard-edged primitives) can plateau BELOW a 0.95
    floor at any resolution -- corner rounding is intrinsic to a trilinear grid SDF, not a resolution bug
    (measured: a tetrahedron stalls at IoU ~0.78 even at res 122). The refusal is then correct behaviour;
    the caller chooses a lower floor or silhouette=None knowingly, which is the entire point of the guard."""
    from holographic.rendering.holographic_render import silhouette_sweep
    V = np.asarray(mesh.vertices, float)
    lo, hi = V.min(0), V.max(0)
    pad = float(pad_frac) * float((hi - lo).max())
    bounds = (tuple(lo - pad), tuple(hi + pad))

    def rung(res, sign):
        grid, axes = mesh_to_sdf_grid(mesh, bounds, res=int(res), band=band, sign=sign)
        back = marching_tetrahedra_vec(grid, axes, level=0.0)
        if silhouette is None:
            return grid, axes, back, None
        sw = silhouette_sweep(mesh, back, n_azimuth=n_azimuth)
        return grid, axes, back, sw

    ladder = []
    res = int(resolution)
    grid, axes, back, sw = rung(res, "auto")
    if silhouette is None:
        return {"mesh": back, "grid": grid, "axes": axes, "bounds": bounds,
                "report": {"iou": None, "view": None, "sign": "auto", "resolution": res, "ladder": []}}
    floor = float(silhouette)
    ladder.append((res, "auto", float(sw["worst"])))
    best = ("auto", grid, axes, back, sw)
    if sw["worst"] < floor:
        g2, a2, b2, s2 = rung(res, "winding")                  # the coincident-shell lever, same cost rung
        ladder.append((res, "winding", float(s2["worst"])))
        if s2["worst"] > sw["worst"]:
            best = ("winding", g2, a2, b2, s2)
    while best[4]["worst"] < floor and res * 1.5 <= float(max_resolution):
        prev = float(best[4]["worst"])
        res = int(round(res * 1.5))
        g2, a2, b2, s2 = rung(res, best[0])
        ladder.append((res, best[0], float(s2["worst"])))
        if s2["worst"] > best[4]["worst"]:
            best = (best[0], g2, a2, b2, s2)
        if float(s2["worst"]) <= prev + 1e-4:
            break                                              # stalled: more cells are not the missing lever
    sign_used, grid, axes, back, sw = best
    report = {"iou": float(sw["worst"]), "view": sw["worst_view"], "sign": sign_used,
              "resolution": res, "ladder": ladder}
    if sw["worst"] < floor:
        raise ValueError("sculpt_prepare REFUSES a destructive conversion: worst-view silhouette IoU %.3f "
                         "< floor %.2f after the ladder %s. The mesh's shape would not survive sculpt mode; "
                         "inspect the report, fix the mesh (coincident shells? sub-cell features?), or "
                         "lower the floor / pass silhouette=None to accept the loss explicitly."
                         % (sw["worst"], floor, ladder))
    out = {"mesh": back, "grid": grid, "axes": axes, "bounds": bounds, "report": report}
    return out


def _selftest_sculpt_prepare():
    """The guarded sculpt-cache conversion: the two measured drift modes and their levers, pinned."""
    def _box_with_fin(fin):
        def bx(cx, cy, cz, hx, hy, hz):
            s = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
            return [(cx+hx*a, cy+hy*b, cz+hz*c) for a, b, c in s]
        V = bx(0,0,0,0.7,0.35,0.5) + bx(0,0.55,0,fin,0.2,0.4)
        Fq = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,4,7,3),(1,2,6,5)]
        F = []
        for base in (0, 8):
            for q in Fq:
                a,b,c,d = [x+base for x in q]; F += [(a,b,c),(a,c,d)]
        from holographic.mesh_and_geometry.holographic_mesh import Mesh
        return Mesh(np.array(V, float), F)

    # touching shells: sign='auto' flood-leaks (measured 0.734@48, WORSENING with res: 0.460@64, 0.250@96);
    # the guard must recover by pulling the WINDING lever at the same resolution, not by escalating cells
    r = sculpt_prepare(_box_with_fin(0.02), resolution=48)
    assert r["report"]["sign"] == "winding", r["report"]
    assert r["report"]["iou"] >= 0.95, r["report"]
    assert r["report"]["ladder"][0][1] == "auto" and r["report"]["ladder"][0][2] < 0.8, \
        "the auto rung must have failed first (that IS the reproduced bug)"

    # an unrecoverable sub-cell sliver REFUSES rather than shipping a decapitated mesh
    try:
        sculpt_prepare(_box_with_fin(0.002), resolution=32, max_resolution=96)
        raise AssertionError("a sub-cell sliver must refuse")
    except ValueError as exc:
        assert "REFUSES" in str(exc) and "IoU" in str(exc)

    # the cache IS the mesh's field: marching the returned grid reproduces the returned mesh
    back = marching_tetrahedra_vec(r["grid"], r["axes"], level=0.0)
    assert len(back.vertices) == len(r["mesh"].vertices)

    # explicit opt-out: silhouette=None converts exactly once, unguarded, and says so in the report
    u = sculpt_prepare(_box_with_fin(0.002), resolution=24, silhouette=None)
    assert u["report"]["iou"] is None and u["report"]["ladder"] == []
    print("OK: sculpt_prepare selftest passed (winding lever recovers the touching-shell leak at equal res; "
          "sub-cell sliver refused with the report; grid==mesh field; opt-out is single-pass)")


if __name__ == "__main__":
    _selftest_sculpt_prepare()

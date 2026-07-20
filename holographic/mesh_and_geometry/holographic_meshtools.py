"""The remaining classic mesh tools (ANIM-3): mirror and merge-by-distance (weld).

WHY THIS MODULE EXISTS
----------------------
The mesh kernel already ships the editing verbs a modeler reaches for -- extrude/inset (meshverbs),
bevel/bridge/loop_cut (meshverbs2), flip/split/collapse edge (eulerops), laplacian_smooth, loop_subdivide,
qem/cluster decimate. Two everyday tools were missing: MIRROR (reflect across a plane and weld the seam -- how
half a symmetric model is built) and MERGE-BY-DISTANCE / WELD (collapse coincident vertices into one -- the
cleanup every import and every mirror needs). Both are here, vectorised for triangle meshes (no per-vertex
Python loop; the face remap is array ops over the (T,3) face table).
"""

import numpy as np


def merge_by_distance(mesh, tol=1e-5, attrs="auto", uv_tol=1e-4, normal_tol=1e-2):
    """Weld vertices closer than `tol` into one. Vertices are grouped by snapping to a `tol` grid; each group
    becomes one vertex at the group's mean; faces are remapped and any face that collapsed to < 3 distinct
    vertices is dropped. Vectorised for triangle meshes. The cleanup after a mirror / import / boolean.

    `attrs` decides what happens when the mesh CARRIES per-vertex uvs/normals -- because a position-only weld on
    such a mesh is not cleanup, it is DAMAGE, measured on a real .glb scan: all 4956 duplicate-position groups
    were UV-SEAM splits (median uv spread 0.67 -- different atlas islands), zero were render duplicates. Welding
    them scrambles the texture atlas and drops the arrays ("losing texture information ... mesh not looking
    great", the exact report).
      * "auto" (default) -- if uvs/normals are present, weld only vertices that agree in position AND uv
        (within `uv_tol`) AND normal (within `normal_tol`): the glTF render-duplicate weld. Seam splits stay
        split, and the attribute arrays are CARRIED (exact -- group members agree by construction). A mesh with
        no attributes takes the position-only path BIT-IDENTICALLY.
      * "ignore" -- the old position-only weld, attributes dropped. For when the caller wants geometry only.
    """
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices
    uv = getattr(mesh, "uvs", None)
    nrm = getattr(mesh, "normals", None)
    carry = attrs == "auto" and (uv is not None or nrm is not None)
    key = np.round(V / tol).astype(np.int64)
    if carry:
        cols = [key]
        if uv is not None:
            cols.append(np.round(np.asarray(uv, float) / uv_tol).astype(np.int64))
        if nrm is not None:
            cols.append(np.round(np.asarray(nrm, float) / normal_tol).astype(np.int64))
        key = np.concatenate(cols, axis=1)                    # weld only where position AND attributes agree
    _, inv = np.unique(key, axis=0, return_inverse=True)      # old vertex -> merged group id
    inv = np.asarray(inv).ravel()
    nnew = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=nnew).astype(float)
    Vnew = np.zeros((nnew, 3))
    np.add.at(Vnew, inv, V)                                   # group sum (scatter)
    Vnew /= counts[:, None]                                   # -> group mean
    uv_new = nrm_new = None
    if carry:
        if uv is not None:
            uv_new = np.zeros((nnew, np.asarray(uv).shape[1]))
            np.add.at(uv_new, inv, np.asarray(uv, float)); uv_new /= counts[:, None]
        if nrm is not None:
            nrm_new = np.zeros((nnew, 3))
            np.add.at(nrm_new, inv, np.asarray(nrm, float))
            ln = np.linalg.norm(nrm_new, axis=1, keepdims=True)
            nrm_new = nrm_new / np.where(ln > 1e-12, ln, 1.0)

    faces = mesh.faces
    if faces and all(len(f) == 3 for f in faces):
        F = inv[np.asarray(faces, dtype=int)]                 # remap all triangles at once
        good = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])   # drop degenerates
        Fnew = [tuple(int(x) for x in row) for row in F[good]]
    else:
        Fnew = []
        for f in faces:                                       # polygon fallback: remap + drop repeats
            seq = []
            for vi in f:
                m = int(inv[vi])
                if not seq or seq[-1] != m:
                    seq.append(m)
            if len(seq) >= 2 and seq[0] == seq[-1]:
                seq.pop()
            if len(seq) >= 3:
                Fnew.append(tuple(seq))
    if carry:
        return Mesh(Vnew, Fnew, normals=nrm_new, uvs=uv_new)
    return Mesh(Vnew, Fnew)


def split_nonmanifold_vertices(mesh):
    """Make a mesh MANIFOLD by splitting non-manifold vertices into their connected UMBRELLAS. At each vertex, its
    incident faces are grouped by MANIFOLD-edge adjacency (two faces link only if they share an edge used by exactly 2
    faces); if a vertex's faces form MORE THAN ONE umbrella -- a bowtie, or the endpoint of a non-manifold edge (shared
    by >2 faces) -- the vertex is duplicated once PER umbrella and each umbrella's faces are reassigned to their own
    copy. Splitting both endpoints of a non-manifold edge separates its >2 faces into <=2-face edges, so the result has
    NO edge shared by >2 faces and a half-edge build / cross-field retopo (which REFUSE a non-manifold mesh) accepts it.
    Unlike mesh_rip_vertex (rips a vertex per-FACE) and mesh_split_vertices (explodes ALL vertices), this splits only
    where non-manifold and only into connected fans -- the minimal cut. On a clean mesh it is a NO-OP. Returns
    (mesh, report). Deterministic (components ordered by least face index)."""
    from collections import defaultdict
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    faces = [tuple(int(i) for i in f) for f in mesh.faces]
    V = np.asarray(mesh.vertices, float)
    nV = len(V)

    edge_faces = defaultdict(list)                           # undirected edge -> face indices using it
    for fi, f in enumerate(faces):
        n = len(f)
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            edge_faces[(min(a, b), max(a, b))].append(fi)

    vfaces = defaultdict(list)                               # vertex -> incident face indices
    for fi, f in enumerate(faces):
        for v in set(f):
            vfaces[v].append(fi)

    vedges = defaultdict(list)                               # vertex -> its incident edges, PRE-BUCKETED once.
    for e in edge_faces:                                     # The first version scanned ALL edges inside the
        vedges[e[0]].append(e)                               # per-vertex loop (O(V*E log E)) -- measured >800s
        vedges[e[1]].append(e)                               # on a 322k-face scan vs ~1s for everything else.
    for v in vedges:                                         # Same iteration ORDER as before (sorted edges per
        vedges[v].sort()                                     # vertex), so the union-find unions run identically
                                                             # and the output stays deterministic-equal.

    new_positions = []                                       # appended vertex copies
    remap = {}                                               # (face_idx, old_vertex) -> new_vertex_id
    n_split = 0
    for v in sorted(vfaces):                                 # deterministic vertex order
        fis = vfaces[v]
        if len(fis) <= 1:
            continue
        parent = {fi: fi for fi in fis}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        fset = set(fis)
        for e in vedges[v]:                                  # link faces across MANIFOLD edges incident to v only
            efs = edge_faces[e]
            if len(efs) == 2 and efs[0] in fset and efs[1] in fset:
                ra, rb = find(efs[0]), find(efs[1])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)        # deterministic union (min root wins)
        comps = defaultdict(list)
        for fi in fis:
            comps[find(fi)].append(fi)
        if len(comps) <= 1:
            continue                                         # single umbrella -> manifold vertex, nothing to do
        n_split += 1
        ordered = sorted(comps.values(), key=lambda c: min(c))   # first umbrella keeps the original v
        for ci, cfaces in enumerate(ordered):
            if ci == 0:
                continue
            new_id = nV + len(new_positions)
            new_positions.append(V[v])
            for fi in cfaces:
                remap[(fi, v)] = new_id

    if not new_positions:
        return mesh, {"split_vertices": 0, "added_vertices": 0}
    Vnew = np.vstack([V, np.array(new_positions, float)])
    Fnew = [tuple(remap.get((fi, v), v) for v in f) for fi, f in enumerate(faces)]
    # carry per-vertex attributes: a split COPIES a vertex, so its uv/normal row is copied verbatim -- exact
    # remap maps many (face, v) entries to the SAME new id; recover the one source vertex per new id
    src_per_new = {}
    for (fi, v), nid in remap.items():
        src_per_new[nid] = v
    order = [src_per_new[nV + i] for i in range(len(new_positions))]
    uv = getattr(mesh, "uvs", None); nrm = getattr(mesh, "normals", None)
    uv_new = np.vstack([np.asarray(uv, float), np.asarray(uv, float)[order]]) if uv is not None else None
    nrm_new = np.vstack([np.asarray(nrm, float), np.asarray(nrm, float)[order]]) if nrm is not None else None
    return (Mesh(Vnew, Fnew, normals=nrm_new, uvs=uv_new),
            {"split_vertices": int(n_split), "added_vertices": int(len(new_positions))})


def _safe_ratio(num, den):
    """num/den for the edge-parameter divisions below, returning 0.0 when the denominator vanishes.

    WHY THIS GUARD EXISTS -- it is a measured bug, not defensive noise: Ericson's region test assumes a
    NON-DEGENERATE triangle, and real meshes are full of degenerate ones. A triangle with a zero-length edge
    (a == b) gives d1 == d3, so `d1 / (d1 - d3)` is 0/0 -> NaN. That is not an exotic input: it is the POLE
    TRIANGLE of every UV-sphere, and any face that lost an edge to a weld. The NaN then flowed straight through
    transfer_uv into the uv array, and a mesh whose uvs are NaN has no relationship to its texture at all --
    silently, with no exception. Audited across every face-count-changing operation, cluster_decimate and
    voxel_remesh were BOTH emitting non-finite uvs on a perfectly coherent atlas because of this one line.
    Degenerate triangle -> the closest point is a vertex or an edge end, so 0.0 is the right parameter."""
    return 0.0 if abs(den) < 1e-20 else num / den


def _closest_point_barycentric(p, a, b, c):
    """Closest point to `p` on triangle (a,b,c) with its barycentric coords (u,v,w), u+v+w=1 -- Ericson's
    region-test (Real-Time Collision Detection ch.5), the standard exact projection. Returns (point, (u,v,w)).

    Degenerate triangles (zero-length edges, slivers, fully collapsed) are handled rather than assumed away --
    see _safe_ratio for what that cost us. The result is ALWAYS finite; a fully degenerate triangle projects to
    its first vertex."""
    ab = b - a; ac = c - a; ap = p - a
    d1 = ab @ ap; d2 = ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a, (1.0, 0.0, 0.0)
    bp = p - b; d3 = ab @ bp; d4 = ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b, (0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = _safe_ratio(d1, d1 - d3)
        return a + v * ab, (1.0 - v, v, 0.0)
    cp = p - c; d5 = ab @ cp; d6 = ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c, (0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = _safe_ratio(d2, d2 - d6)
        return a + w * ac, (1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = _safe_ratio(d4 - d3, (d4 - d3) + (d5 - d6))
        return b + w * (c - b), (0.0, 1.0 - w, w)
    s = va + vb + vc
    if abs(s) < 1e-20:                                       # fully collapsed: no interior to land in
        return a, (1.0, 0.0, 0.0)
    denom = 1.0 / s
    v = vb * denom; w = vc * denom
    return a + ab * v + ac * w, (1.0 - v - w, v, w)


def build_face_grid(vertices, faces, cell_scale=1.0):
    """A uniform spatial hash over triangle bounding boxes -- the shared acceleration structure behind
    transfer_uv, attribute transfer, and the bakes (M14: one correspondence machine, many readers). Returns
    (grid, tri, lo, cell): grid maps (ix,iy,iz) -> list of face indices; tri is (F,3,3) corner positions; lo
    is the bbox min; cell is the mean edge length * cell_scale (the bucket size the query below assumes).

    WHY factored: four call sites built this identical lattice inline (floor((p-lo)/cell) bucketing, mean-edge
    cell). Sharing it means one place gets the cell-size rule right, and closest_face_point below is the one
    ring-search every site had copied. Bit-identical to the historical inline builds -- the cell rule, the
    bbox-span bucketing, and the face list per cell are unchanged; pinned by transfer_uv's and
    bake_normal_map's sha pins."""
    from collections import defaultdict
    V = np.asarray(vertices, float)
    F = [tuple(int(i) for i in f[:3]) for f in faces]
    tri = np.array([[V[f[0]], V[f[1]], V[f[2]]] for f in F])
    lo = tri.min(axis=(0, 1))
    el = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    cell = max(float(np.mean(el)) * float(cell_scale), 1e-9)
    grid = defaultdict(list)
    tmin = tri.min(1); tmax = tri.max(1)
    for fi in range(len(F)):
        i0 = np.floor((tmin[fi] - lo) / cell).astype(int)
        i1 = np.floor((tmax[fi] - lo) / cell).astype(int)
        for ix in range(i0[0], i1[0] + 1):
            for iy in range(i0[1], i1[1] + 1):
                for iz in range(i0[2], i1[2] + 1):
                    grid[(ix, iy, iz)].append(fi)
    return grid, tri, lo, cell


def closest_face_point(p, grid, tri, lo, cell, faces):
    """Closest point on a mesh to `p`, via the shared face grid -- returns (face_index, barycentric, dist2).
    The ONE ring-expanding search that transfer_uv, attribute transfer, and the bakes each had copied inline
    (search this cell ring, then one more; brute-force fall back once past ring 12 for a far off-grid point).
    The caller interpolates whatever it needs (UVs, normals, positions) from (face_index, bary) -- the M14
    'one projection, many channels' split: this owns the PROJECTION, the caller owns the CHANNEL. Bit-identical
    to the historical inline loops (same ring order, same tie-break = first-seen-at-min-d2, same fallback)."""
    F = faces
    base = np.floor((p - lo) / cell).astype(int)
    best_d2 = None; best_fi = None; best_bc = None
    ring = 0
    while best_d2 is None or ring <= 1:
        cand = set()
        for ix in range(base[0] - ring, base[0] + ring + 1):
            for iy in range(base[1] - ring, base[1] + ring + 1):
                for iz in range(base[2] - ring, base[2] + ring + 1):
                    if ring == 0 or max(abs(ix - base[0]), abs(iy - base[1]), abs(iz - base[2])) == ring:
                        cand.update(grid.get((ix, iy, iz), ()))
        for fi in cand:
            a, b, c = tri[fi]
            q, bc = _closest_point_barycentric(p, a, b, c)
            d2 = float(np.sum((p - q) ** 2))
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2; best_fi = fi; best_bc = bc
        ring += 1
        if ring > 12 and best_d2 is None:
            for fi in range(len(F)):
                a, b, c = tri[fi]
                q, bc = _closest_point_barycentric(p, a, b, c)
                d2 = float(np.sum((p - q) ** 2))
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2; best_fi = fi; best_bc = bc
            break
    return best_fi, best_bc, best_d2


def transfer_uv(source_mesh, source_uv, target_vertices, cell_scale=1.0):
    """TRANSFER per-vertex UVs (or ANY per-vertex attribute) from `source_mesh` onto new `target_vertices` by
    closest-point projection + barycentric interpolation -- the step that makes RETOPO TEXTURE-PRESERVING: a remeshed
    surface lies on (or near) the original, so each new vertex projects to a point on some source triangle, and the
    triangle's corner UVs interpolate there. Accelerated by a uniform spatial hash over source triangles (query cost ~
    per-cell occupancy, not O(F)). `source_uv` is (n_src_verts, k) -- k=2 for UVs, but any per-vertex attribute
    (colours, weights) transfers identically. Returns (target_attr (n_tgt, k), distances (n_tgt,)) -- the distance is
    the projection residual, the HONEST error signal (large = the target strayed off the source surface, e.g. a
    hole-fill centroid; its UV is an extrapolation).

    KEPT NEGATIVE: closest-point transfer is wrong across UV SEAMS -- a target vertex whose closest source triangle is
    on the other side of a seam gets that island's UV, a visible texture jump. Splitting target verts along source
    seams is the full fix (not built); on a seam-light asset the artefact is a few texels wide."""
    V = np.asarray(source_mesh.vertices, float)
    F = [tuple(int(i) for i in f[:3]) for f in source_mesh.faces]
    UVs = np.asarray(source_uv, float)
    T = np.asarray(target_vertices, float)
    # DELEGATED to the shared correspondence machine (M14): build_face_grid + closest_face_point own the
    # spatial hash and the ring-search; this function keeps only the ATTRIBUTE it reads -- interpolate the
    # source UVs at the returned barycentric coords. Bit-identical to the historical inline loop (same grid,
    # same ring order, same first-seen-at-min-d2 tie-break); pinned by transfer_uv's sha in test_cad_backlog.
    grid, tri, lo, cell = build_face_grid(V, F, cell_scale=cell_scale)
    out = np.zeros((len(T), UVs.shape[1]))
    dist = np.zeros(len(T))
    for ti, p in enumerate(T):
        fi, bc, best_d2 = closest_face_point(p, grid, tri, lo, cell, F)
        f = F[fi]
        out[ti] = bc[0] * UVs[f[0]] + bc[1] * UVs[f[1]] + bc[2] * UVs[f[2]]
        dist[ti] = np.sqrt(best_d2)
    return out, dist


def topology_delta(src_mesh, out_mesh):
    """Did a mesh operation CHANGE THE TOPOLOGY it had no business changing? Returns a dict with the deltas
    and a single `preserved` verdict.

    WHY THIS IS A SEPARATE GATE FROM THE SILHOUETTE, and the measurement that proves it: silhouette_sweep is
    an OUTLINE check, and an outline is blind to anything inside it. A hole punched in the middle of a face, an
    island floating free inside the hull, a filled cavity -- none of them move the silhouette. MEASURED: our
    own surface_retopo scored 0.973 IoU (a clean PASS) while turning a CLOSED box into a mesh with 6 boundary
    edges. The gate passed a mesh with holes in it. (Same shape as the sweep's donut-vs-disc kept negative,
    biting a different operator.)

    WHAT IT CHECKS, and each is a rule an operator should not break silently:
      * components -- a reducing op must not CREATE islands. Detached geometry means the operator tore the
        surface; it is never what the caller asked for. (Fewer components is legal: welding dust is healing.)
      * boundary_edges / holes -- an op must not punch holes in a closed mesh, NOR fill holes that existed. A
        scan's holes are DATA: silently closing them invents surface that was never measured.
      * euler_characteristic -- the genus invariant. A change here means handles appeared or vanished.
      * nonmanifold_edges -- must not increase.

    `preserved` is False if ANY of those moved in the disallowed direction. The caller decides what to do --
    this is a measurement, not a policy. Deterministic; no tolerance, because these are all integers, and an
    integer invariant with a tolerance is not an invariant."""
    from holographic.scene_and_pipeline.holographic_route import connected_components as _cc
    a = face_orientation_report(src_mesh)
    b = face_orientation_report(out_mesh)
    try:
        ca, cb = int(_cc(src_mesh)), int(_cc(out_mesh))
    except Exception:
        ca = cb = -1                                   # component count unavailable: report it, do not guess
    chi_a = _euler(src_mesh)
    chi_b = _euler(out_mesh)
    nm_a, nm_b = _nonmanifold_count(src_mesh), _nonmanifold_count(out_mesh)
    islands_created = (cb > ca) if ca >= 0 else False
    holes_created = b["boundary_edges"] > a["boundary_edges"]
    holes_filled = b["boundary_edges"] < a["boundary_edges"]
    nonmanifold_added = nm_b > nm_a
    return {"components_before": ca, "components_after": cb, "islands_created": bool(islands_created),
            "boundary_edges_before": a["boundary_edges"], "boundary_edges_after": b["boundary_edges"],
            "holes_created": bool(holes_created), "holes_filled": bool(holes_filled),
            "euler_before": chi_a, "euler_after": chi_b, "euler_changed": bool(chi_a != chi_b),
            "nonmanifold_before": nm_a, "nonmanifold_after": nm_b,
            "nonmanifold_added": bool(nonmanifold_added),
            "preserved": not (islands_created or holes_created or holes_filled or nonmanifold_added)}


def _euler(mesh):
    """V - E + F on the undirected edge set. An integer invariant: a change means handles or holes moved."""
    es = set()
    for f in mesh.faces:
        n = len(f)
        for k in range(n):
            a, b = int(f[k]), int(f[(k + 1) % n])
            es.add((min(a, b), max(a, b)))
    return len(np.asarray(mesh.vertices)) - len(es) + len(mesh.faces)


def _nonmanifold_count(mesh):
    """Undirected edges touched by 3+ faces."""
    import collections
    c = collections.Counter()
    for f in mesh.faces:
        n = len(f)
        for k in range(n):
            a, b = int(f[k]), int(f[(k + 1) % n])
            c[(min(a, b), max(a, b))] += 1
    return sum(1 for v in c.values() if v > 2)


def transform_mesh(mesh, matrix):
    """Apply a 3x3 or 4x4 matrix to a mesh -- AND FLIP FACE WINDING WHEN THE MATRIX REFLECTS (det < 0).

    WHY THIS EXISTS, and why the rule belongs in exactly one place: a negative-determinant transform (a
    mirror, an axis SWAP, a negative scale) turns the surface inside out. The vertices land correctly and
    every face normal points the wrong way -- and NOTHING catches it, because the mesh stays perfectly
    self-consistent. MEASURED on the Poly Studio demo's own bug, reproduced here: the Z-up -> Y-up swap
    V[:, [0,2,1]] gives a box that reports ORIENTED True with 0% outward normals. Consistently oriented and
    consistently inside-out.

    THAT IS WHY mesh_orient CANNOT FIX IT, and the distinction is worth keeping straight: mesh_orient repairs
    INCONSISTENCY -- neighbours disagreeing across an edge -- and a globally inverted mesh has no disagreement
    to find, so it correctly flips nothing. INCONSISTENT WINDING and GLOBAL INVERSION are different defects.
    The demo hand-rolled a triangle-order reversal after its axis swap because no operator owned this rule;
    every caller that reflects would otherwise re-derive it, and the ones that forget ship inside-out meshes
    that pass every orientation check.

    `matrix` may be 3x3 (linear) or 4x4 (affine; the translation column is applied). Returns a new Mesh --
    vertices transformed, faces reversed iff det(linear part) < 0. A det of exactly 0 RAISES: a singular
    transform collapses the mesh and silently returning degenerate geometry is worse than stopping.
    UVs and normals ride along untouched (uvs are unaffected by a spatial transform; normals are recomputed
    lazily by Mesh)."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    M = np.asarray(matrix, float)
    if M.shape == (4, 4):
        L = M[:3, :3]
        t = M[:3, 3]
    elif M.shape == (3, 3):
        L = M
        t = np.zeros(3)
    else:
        raise ValueError("matrix must be 3x3 or 4x4; got %s" % (M.shape,))
    det = float(np.linalg.det(L))
    if abs(det) < 1e-12:
        raise ValueError("singular transform (det = %.3g): it would collapse the mesh" % det)
    V = np.asarray(mesh.vertices, float) @ L.T + t
    faces = [tuple(int(i) for i in f) for f in mesh.faces]
    if det < 0:
        faces = [tuple(reversed(f)) for f in faces]      # the reflection rule, in ONE place
    out = Mesh(V, faces)
    uvs = getattr(mesh, "uvs", None)
    if uvs is not None:
        out.uvs = np.asarray(uvs, float).copy()
    return out


_AXES = {"x": 0, "y": 1, "z": 2}


def convert_up_axis(mesh, frm="z", to="y"):
    """Re-orient a mesh from one up-axis convention to another (e.g. a Z-up terrain into a Y-up scene),
    keeping the winding correct.

    Built on transform_mesh, so the reflection rule is not restated here. THE POINT: the naive way to do this
    is to permute the coordinate columns -- V[:, [0,2,1]] -- which is a REFLECTION, and it silently inverts
    every normal (measured: 0% outward on a box, while every orientation check still says True). This uses a
    PROPER ROTATION (det = +1) where one exists, so nothing is inverted in the first place; when a caller
    forces a genuine reflection, transform_mesh flips the winding to match.

    Returns a new Mesh. Same axis in and out is a no-op that still round-trips through transform_mesh, so the
    identity path is exercised by every test that touches it."""
    a, b = str(frm).lower(), str(to).lower()
    if a not in _AXES or b not in _AXES:
        raise ValueError("axes must be one of %s; got %r -> %r" % (sorted(_AXES), frm, to))
    if a == b:
        return transform_mesh(mesh, np.eye(3))
    i, j = _AXES[a], _AXES[b]
    # a proper rotation taking axis i onto axis j: 90 degrees about the third axis, det = +1
    k = 3 - i - j
    R = np.zeros((3, 3))
    R[j, i] = 1.0            # old up  -> new up
    R[i, j] = -1.0           # the partner, negated: this is what keeps det = +1 (a swap alone would be -1)
    R[k, k] = 1.0
    return transform_mesh(mesh, R)


def face_orientation_report(mesh):
    """Is every DIRECTED edge traversed exactly once, for ANY face degree? Returns a dict with `oriented`,
    `duplicated_directed_edges`, `boundary_edges`, `components`.

    WHY NOT isosurface.is_oriented: that one is QUAD-ONLY -- it indexes q[0..3] literally, so it cannot even
    look at a triangle mesh, which is every scan and every decimation output in this tree. This is the same
    property, general in face degree, and mesh_orient below is pinned against it on quads so the two agree
    where they overlap."""
    from collections import Counter
    counts = Counter()
    for f in mesh.faces:
        n = len(f)
        for k in range(n):
            counts[(int(f[k]), int(f[(k + 1) % n]))] += 1
    dup = sum(1 for c in counts.values() if c > 1)
    boundary = sum(1 for (a, b) in counts if (b, a) not in counts)
    return {"oriented": bool(counts) and dup == 0, "duplicated_directed_edges": dup,
            "boundary_edges": boundary, "faces": len(mesh.faces)}


def mesh_orient(mesh, seed_face=0):
    """Make face winding CONSISTENT: flip faces until neighbours traverse their shared edge in OPPOSITE
    directions. Returns (new_mesh, report).

    WHY THIS EXISTS (R3b): `connection` -- and so cross_field, guided_cross_field and the whole surface-retopo
    route -- requires consistent winding, and photogrammetry scans do not have it. The tree could CHECK this
    (mesh_is_oriented) and never REPAIR it; measured, the ladybird's decimated LOD raised "directed edge (0,1)
    appears twice" the moment surface_retopo touched it. Rule 0: no mesh_orient/orient_mesh/fix_orientation
    existed; mesh_repair does weld/fill/non-manifold, not winding.

    HOLOGRAPHIC READING: this is a 2-COLOURING by FLOOD FILL over the dual graph (ledger P5's move), the same
    breadth-first "agree with whoever reached me" pattern the tree runs on pixel masks and voxel grids. A face
    disagrees with the neighbour that reached it exactly when they traverse the shared edge in the SAME
    direction; flip it and continue. Per component, because a scan is many shells.

    MANIFOLD EDGES ONLY, and this distinction cost a wrong answer before it was made: the BFS propagates
    across edges shared by EXACTLY TWO faces. An edge with three or more faces has no well-defined "the
    neighbour", and treating every face on it as one imposes contradictory parity -- which the first version
    then reported as "non-orientable". IT IS NOT. MEASURED on the ladybird's decimated LOD: 490 non-manifold
    edges (up to SEVEN faces on one edge), and the first version called 7 of 9 components non-orientable and
    flipped nothing. Non-manifold is a DIFFERENT defect with a DIFFERENT repair (split_non_manifold, in
    mesh_repair). Non-manifold edges are now SKIPPED for propagation and counted in
    report["non_manifold_edges"]; the orientable part of the mesh still gets fixed.

    ORIENTABILITY IS REPORTED, NEVER GUESSED: a Mobius-like component genuinely cannot be consistently
    oriented, and BFS detects it as a face reached twice with conflicting parity ACROSS MANIFOLD EDGES. Those
    components are left ALONE and counted in report["non_orientable_components"] -- a wrong flip is worse than
    an honest refusal, and silently "fixing" a non-orientable shell hands the field solver a lie.

    An ALREADY-oriented mesh returns BIT-IDENTICAL vertices and faces (pinned): the flood still runs, finds
    nothing to flip, and rebuilds the same tuples.

    READ `propagation_components`, NOT `components`: both keys hold the count of components the ORIENTATION
    FLOOD could reach, which is manifold-edge-connectivity -- NOT geometric connectivity. Measured on a
    ladybird LOD: 399 propagation components where the geometric count is 9, because 490 non-manifold edges
    block the flood. `components` is the old, misleading name, kept one release for compatibility and
    deprecated. For geometric components, walk all shared edges (non-manifold included)."""
    import collections
    V = np.asarray(mesh.vertices, float)
    faces = [tuple(int(i) for i in f) for f in mesh.faces]
    n_f = len(faces)
    if n_f == 0:
        raise ValueError("mesh_orient needs at least one face")

    # undirected edge -> faces touching it (the dual graph, built once)
    edge_faces = collections.defaultdict(list)
    for fi, f in enumerate(faces):
        n = len(f)
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            edge_faces[(min(a, b), max(a, b))].append(fi)

    def traverses(f, a, b):
        """Does face f traverse the undirected edge {a,b} in the direction a->b?"""
        n = len(f)
        for k in range(n):
            if f[k] == a and f[(k + 1) % n] == b:
                return True
        return False

    flip = [False] * n_f
    seen = [False] * n_f
    comps = 0
    non_orientable = set()
    flipped = 0
    order = list(range(int(seed_face), n_f)) + list(range(0, int(seed_face)))
    for start in order:
        if seen[start]:
            continue
        comps += 1
        seen[start] = True
        queue = collections.deque([start])
        comp = [start]
        while queue:
            fi = queue.popleft()
            f = faces[fi]
            n = len(f)
            for k in range(n):
                a, b = f[k], f[(k + 1) % n]
                if flip[fi]:
                    a, b = b, a                       # this face's EFFECTIVE winding after its own flip
                touching = edge_faces[(min(a, b), max(a, b))]
                if len(touching) != 2:
                    continue                          # non-manifold (or boundary): no well-defined neighbour
                for gj in touching:
                    if gj == fi:
                        continue
                    g = faces[gj]
                    # consistent iff the neighbour traverses the shared edge the OTHER way (b -> a)
                    g_ab = traverses(g, a, b)
                    want_flip = g_ab                  # same direction as us -> it must flip
                    if flip[gj]:
                        g_ab = traverses(g, b, a)
                        want_flip = g_ab
                    if not seen[gj]:
                        seen[gj] = True
                        flip[gj] = bool(want_flip)
                        comp.append(gj)
                        queue.append(gj)
                    elif want_flip:
                        non_orientable.add(comps)     # already fixed, and it still disagrees
        if comps in non_orientable:
            for fi in comp:
                flip[fi] = False                      # leave a non-orientable component EXACTLY as it was
    out = []
    for fi, f in enumerate(faces):
        if flip[fi]:
            out.append(tuple(reversed(f)))
            flipped += 1
        else:
            out.append(f)
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    new = Mesh(V, out)
    rep = face_orientation_report(new)
    n_manifold_bad = sum(1 for fs in edge_faces.values() if len(fs) > 2)
    # "components" MEANT propagation components (manifold-edge-connected) and the NAME LIED: on the ladybird
    # LOD it reported 399 where the geometric component count is 9, because non-manifold edges block the
    # flood. Both are correct answers to different questions. propagation_components is the honest name; the
    # old key stays one release (additive -- nothing that reads it breaks) and is documented as deprecated.
    rep.update({"flipped": flipped, "propagation_components": comps, "components": comps,
                "non_orientable_components": len(non_orientable),
                "non_manifold_edges": n_manifold_bad,
                # M10 debt made SAFE-TO-DROP instead of dropped now. Removing "components" is a BREAKING change
                # (a key deletion), and C1 -- the PyPI publish -- has not shipped, so no external caller has had a
                # release to migrate off it. Dropping it pre-publish would break clients the instant the wheel
                # lands, which is exactly the additive rule's job to prevent. So the alias stays, and a
                # machine-readable marker turns the eventual drop from archaeology into reading one field: a
                # post-C1 "drop deprecated keys" pass removes every key named here and this entry. Nothing internal
                # reads rep["components"] (verified: only the SEPARATE topology report's components key has
                # readers), so the drop is caller-facing only.
                "_deprecated": ("components",)})
    return new, rep


def _selftest_topology_delta():
    """The topology gate: the invariants the SILHOUETTE cannot see. Pins each of the three defects the owner
    named -- islands created, holes punched, holes filled -- plus the measurement that motivated the gate:
    our own surface_retopo passes the outline gate (0.973 IoU) while turning a CLOSED box into one with
    boundary edges. An outline is blind to anything inside it."""
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(triangulate_ngons(box()), levels=2)
    V = np.asarray(src.vertices, float)
    F = [tuple(f) for f in src.faces]
    assert topology_delta(src, src)["preserved"] is True                 # identity preserves everything
    holed = Mesh(V, F[:-4])
    d = topology_delta(src, holed)
    assert d["holes_created"] and not d["preserved"], d
    d = topology_delta(holed, src)
    assert d["holes_filled"] and not d["preserved"], "filling a hole that EXISTED is also a violation"
    island = Mesh(np.vstack([V, V[:3] + 50.0]), F + [(len(V), len(V) + 1, len(V) + 2)])
    d = topology_delta(src, island)
    assert d["islands_created"] and d["components_after"] > d["components_before"] and not d["preserved"]
    print("topology_delta selftest OK (islands/holes-punched/holes-filled each caught; identity preserved) "
          "-- the invariants the outline gate is blind to")


def _selftest_transform_mesh():
    """M3 (BACKLOG): the reflection rule, in one place. Pins the distinction that M3 originally got WRONG:
    a reflection leaves a mesh CONSISTENTLY oriented and CONSISTENTLY inside-out, which mesh_orient cannot
    see and correctly will not fix. transform_mesh flips the winding so the defect never exists."""
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    b = triangulate_ngons(box())

    def outward(mesh):
        V = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces, int)
        c = V.mean(0)
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        return float(((n * (V[F].mean(1) - c)).sum(1) > 0).mean())
    assert outward(b) == 1.0
    refl = transform_mesh(b, np.diag([1.0, 1.0, -1.0]))            # det = -1
    assert outward(refl) == 1.0, "a reflection must flip winding, or the mesh is inside-out"
    rot = transform_mesh(b, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]))   # det = +1
    assert [tuple(f) for f in rot.faces] == [tuple(f) for f in b.faces], "a rotation must NOT touch winding"
    assert outward(rot) == 1.0
    up = convert_up_axis(b, "z", "y")
    assert outward(up) == 1.0 and face_orientation_report(up)["oriented"]
    assert outward(convert_up_axis(b, "y", "y")) == 1.0            # identity path
    # THE KEPT NEGATIVE: the naive column permutation is the demo's bug, and it passes every existing check
    naive = Mesh(np.asarray(b.vertices, float)[:, [0, 2, 1]], [tuple(f) for f in b.faces])
    assert face_orientation_report(naive)["oriented"] is True      # "oriented" -- and 0% outward
    assert outward(naive) == 0.0, "the naive swap must still demonstrate the inside-out failure"
    _o, r = mesh_orient(naive)
    assert r["flipped"] == 0, "mesh_orient must NOT claim to fix global inversion (it cannot see it)"
    try:
        transform_mesh(b, np.zeros((3, 3)))
        raise AssertionError("a singular transform must raise")
    except ValueError:
        pass
    print("transform_mesh selftest OK (reflection flips winding; rotation does not; naive swap still measures "
          "0%% outward while reporting 'oriented', and mesh_orient correctly flips 0 -- different defects)")


def _selftest_mesh_orient():
    """R3b: consistent winding by flood-fill 2-colouring over the dual graph. Pins: (1) an already-oriented
    mesh is BIT-IDENTICAL (the flood runs, finds nothing, rebuilds the same tuples); (2) a scrambled orientable
    mesh is REPAIRED; (3) NON-MANIFOLD is not NON-ORIENTABLE -- the distinction that cost a wrong answer:
    the ladybird's LOD has 490 non-manifold edges (up to 7 faces on one edge) and the first version reported
    7 of 9 components 'non-orientable' while flipping nothing. Non-manifold edges are skipped and COUNTED."""
    import numpy as _np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    src = loop_subdivide(triangulate_ngons(box()), levels=2)
    out, rep = mesh_orient(src)
    assert rep["oriented"] and rep["flipped"] == 0
    assert [tuple(f) for f in out.faces] == [tuple(f) for f in src.faces], "oriented input must be untouched"
    rng = _np.random.default_rng(0)
    bad = [tuple(reversed(f)) if rng.random() < 0.5 else tuple(f) for f in src.faces]
    o2, r2 = mesh_orient(Mesh(_np.asarray(src.vertices, float), bad))
    assert r2["oriented"] and r2["flipped"] > 0 and r2["non_manifold_edges"] == 0
    # a NON-MANIFOLD edge (3 faces on one edge) must be reported as such, not as non-orientable
    V = [[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]]
    nm = Mesh(_np.asarray(V, float), [(0, 1, 2), (0, 1, 3), (0, 1, 4)])
    _o3, r3 = mesh_orient(nm)
    assert r3["non_manifold_edges"] >= 1, r3
    assert r3["non_orientable_components"] == 0, "non-manifold must NOT be reported as non-orientable"
    # the renamed key carries the honest meaning; the old one is kept for one release and must AGREE
    assert r3["propagation_components"] == r3["components"], "the deprecated alias must not drift"
    # the deprecation is DECLARED, not just commented -- so a post-C1 "drop deprecated keys" pass is mechanical:
    # every key in _deprecated is one to remove, and each must still be an ALIAS (equal to its honest twin) until
    # then, so nothing silently diverges in the grace window.
    assert "components" in r3["_deprecated"], "the deprecated key must announce itself in _deprecated"
    print("mesh_orient selftest OK (oriented input bit-identical; scrambled repaired with %d flips; "
          "non-manifold reported as non-manifold, not non-orientable)" % r2["flipped"])


def shrinkwrap(mesh, target_mesh, factor=1.0, cell_scale=1.0):
    """SHRINKWRAP: move each vertex of `mesh` toward its CLOSEST POINT on `target_mesh` (Blender's shrinkwrap /
    retopo-snap operator). `factor` in [0,1] is how far to move -- 1.0 lands exactly on the target surface, 0.5 goes
    halfway (a soft pull), 0.0 leaves it put. Returns (new_mesh, residual (n_verts,)) where residual is the ORIGINAL
    distance from each vertex to the target (the honest gap it closed). Same closest-point-on-triangle projection as
    transfer_uv, so it fixes the exact error that DOUBLED our box-model texture residual: a cage vertex that bulged
    off the real surface (a protruding inset/extrude) gets pulled back ONTO it, without re-modelling.

    WHY THIS IS THE RETOPO FINISHER: box-model or remesh gives good TOPOLOGY but approximate POSITIONS; one shrinkwrap
    pass makes the positions match the reference surface while keeping the clean topology. KEPT NEGATIVE: closest-POINT
    (not ray-cast along a normal) -- a thin target can pull a vertex to the wrong side; move in steps (small factor,
    repeat) if that bites."""
    from collections import defaultdict
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    TV = np.asarray(target_mesh.vertices, float)
    TF = [tuple(int(i) for i in f[:3]) for f in target_mesh.faces]
    tri = np.array([[TV[f[0]], TV[f[1]], TV[f[2]]] for f in TF])          # (F,3,3)
    lo = tri.min(axis=(0, 1))
    el = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    cell = max(float(np.mean(el)) * float(cell_scale), 1e-9)
    grid = defaultdict(list)
    tmin = tri.min(1); tmax = tri.max(1)
    for fi in range(len(TF)):
        i0 = np.floor((tmin[fi] - lo) / cell).astype(int)
        i1 = np.floor((tmax[fi] - lo) / cell).astype(int)
        for ix in range(i0[0], i1[0] + 1):
            for iy in range(i0[1], i1[1] + 1):
                for iz in range(i0[2], i1[2] + 1):
                    grid[(ix, iy, iz)].append(fi)
    V = np.asarray(mesh.vertices, float).copy()
    out = V.copy()
    resid = np.zeros(len(V))
    f = float(np.clip(factor, 0.0, 1.0))
    for vi, p in enumerate(V):
        base = np.floor((p - lo) / cell).astype(int)
        best_d2 = None; best_q = None
        ring = 0
        while best_d2 is None or ring <= 1:
            cand = set()
            for ix in range(base[0] - ring, base[0] + ring + 1):
                for iy in range(base[1] - ring, base[1] + ring + 1):
                    for iz in range(base[2] - ring, base[2] + ring + 1):
                        if ring == 0 or max(abs(ix - base[0]), abs(iy - base[1]), abs(iz - base[2])) == ring:
                            cand.update(grid.get((ix, iy, iz), ()))
            for fi in cand:
                tf = TF[fi]
                q, _bc = _closest_point_barycentric(p, TV[tf[0]], TV[tf[1]], TV[tf[2]])
                d2 = float(np.sum((p - q) ** 2))
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2; best_q = q
            ring += 1
            if ring > 12 and best_d2 is None:                             # far off-grid: one brute-force sweep
                for fi in range(len(TF)):
                    tf = TF[fi]
                    q, _bc = _closest_point_barycentric(p, TV[tf[0]], TV[tf[1]], TV[tf[2]])
                    d2 = float(np.sum((p - q) ** 2))
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2; best_q = q
                break
        out[vi] = p + f * (best_q - p)
        resid[vi] = np.sqrt(best_d2)
    return Mesh(out, [tuple(int(i) for i in fa) for fa in mesh.faces]), resid


def mesh_report(mesh):
    """MESH REPORT: the one-call topology + shape scoreboard an agent (or artist) needs to SEE a mesh's state cheaply,
    so nobody re-derives it every session. Returns a dict:
      verts, faces, quad_fraction, tri_fraction, ngon_fraction,
      boundary_edges (edges used by one face -> open holes/seams),
      nonmanifold_edges (edges used by >2 faces),
      is_manifold, is_closed, euler_characteristic,
      valence_histogram {valence: count}, regular_fraction (valence-4 for a quad mesh, else valence-6),
      bbox_min, bbox_max, bbox_span, centroid.
    This is the scoreboard the box-modelling arc printed by hand three times (quad%, boundary, valence). Deterministic.

    WHY a dict, not a print: the agent can BRANCH on it (e.g. 'boundary_edges>0 -> the cage isn't watertight, fill
    before subdividing'), which a formatted string can't drive."""
    from collections import Counter, defaultdict
    V = np.asarray(mesh.vertices, float)
    F = [tuple(int(i) for i in f) for f in mesh.faces]
    nF = max(len(F), 1)
    edge_count = Counter()
    valence = defaultdict(int)
    for f in F:
        k = len(f)
        for a in range(k):
            edge_count[tuple(sorted((f[a], f[(a + 1) % k])))] += 1
        for v in set(f):
            valence[v] += 1
    quads = sum(1 for f in F if len(f) == 4)
    tris = sum(1 for f in F if len(f) == 3)
    ngons = sum(1 for f in F if len(f) > 4)
    boundary = sum(1 for c in edge_count.values() if c == 1)
    nonman = sum(1 for c in edge_count.values() if c > 2)
    vals = [valence[v] for v in range(len(V)) if v in valence]
    vhist = dict(sorted(Counter(vals).items()))
    qf = quads / nF
    target_val = 4 if qf >= 0.5 else 6
    regular = float(np.mean([1.0 if x == target_val else 0.0 for x in vals])) if vals else 0.0
    lo = V.min(0) if len(V) else np.zeros(3)
    hi = V.max(0) if len(V) else np.zeros(3)
    try:
        man = bool(mesh.is_manifold()); clo = bool(mesh.is_closed())
    except Exception:
        man = (nonman == 0 and boundary == 0); clo = (boundary == 0)
    return {
        "verts": len(V), "faces": len(F),
        "quad_fraction": qf, "tri_fraction": tris / nF, "ngon_fraction": ngons / nF,
        "boundary_edges": boundary, "nonmanifold_edges": nonman,
        "is_manifold": man, "is_closed": clo,
        "euler_characteristic": len(V) - len(edge_count) + len(F),
        "valence_histogram": vhist, "regular_fraction": regular,
        "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
        "bbox_span": (hi - lo).tolist(), "centroid": V.mean(0).tolist() if len(V) else [0.0, 0.0, 0.0],
    }


def _drop_unreferenced(mesh):
    """Remove vertices that no face uses (compact the vertex table + remap faces). A weld/degenerate-drop can orphan
    vertices; a downstream half-edge build counts them as isolated. Deterministic (sorted used-vertex order)."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    used = sorted({int(vi) for f in mesh.faces for vi in f})
    if len(used) == len(mesh.vertices):
        return mesh
    remap = {old: new for new, old in enumerate(used)}
    V = np.asarray(mesh.vertices, float)[used]
    F = [tuple(remap[int(vi)] for vi in f) for f in mesh.faces]
    uv = getattr(mesh, "uvs", None); nrm = getattr(mesh, "normals", None)
    return Mesh(V, F,
                normals=np.asarray(nrm, float)[used] if nrm is not None else None,
                uvs=np.asarray(uv, float)[used] if uv is not None else None)


def process_scan(mesh, uv=None, texture=None, retopo=True, lod=None, density=1.0,
                 bake_size=1024, bake_margin=2, keep_shards=False, silhouette=0.95, bake_method="project",
                 retopo_fast=False, retopo_snap=False, retopo_sized=False, bake_normal_aware=False, manifold=False):
    """ONE WORKFLOW for 'repair this scan and reduce its polys, keeping the texture' -- the pipeline Moose
    specified, in the CORRECT order, with each stage earned by a measurement:

        1. REPAIR the ORIGINAL mesh (weld / fill holes / orient) -- never a decimated copy.
        2. RETOPO the repaired mesh (field-guided quads) if retopo=True.
        3. LOD: if lod (a float 0<lod<1 target fraction, or an int target face count) is given:
             - with retopo:   a SECOND, COARSER retopo. MEASURED: decimating a quad retopo RE-SHATTERS it
               (1 -> 38+ components) and stalls on the silhouette guard (budget_missed_for_silhouette);
               a coarser retopo is clean BY CONSTRUCTION (3116 faces, 1 component on the mantis).
             - without retopo: QEM decimation of the repaired mesh (the classic LOD; silhouette-guarded).
        4. SHARD CLEANUP: drop_small_components(keep_largest) after any retopo (a field extractor drops
           isolated cells; 88-145 components measured on scans). keep_shards=True skips this.
        5. FRESH ATLAS + REPROJECT: rebake_texture builds a NEW per-face atlas for the final mesh and paints
           the ORIGINAL texture in by closest-point projection -- never transfers the scan's (fragmented)
           uvs. Skipped when uv/texture are not given (geometry-only mode).

    The four workflows, as requested: retopo+lod, retopo only, lod only (retopo=False), neither (repair +
    re-texture only). Returns (mesh, uv, image, report); uv/image are None in geometry-only mode; report
    carries every stage's numbers (components dropped, bake coverage, projection distance, route taken).

    KEPT NEGATIVES carried from the stages: the extractor cannot mesh sub-cell-width tubes (thin limbs shard
    and are REMOVED, not repaired); rebake is O(texels) and slow at scale (M10); the silhouette guard may
    honestly refuse a decimation budget (reported, not forced)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    report = {"stages": []}
    src = mesh
    src_uv = np.asarray(uv, float) if uv is not None else (np.asarray(mesh.uvs, float) if getattr(mesh, "uvs", None) is not None else None)

    # -- 1 REPAIR the original ------------------------------------------------------------------------
    # GEOMETRY ONLY -- never repair with the scan's fragmented uvs attached. MEASURED (the missing-faces
    # regression Moose caught in a render): with per-face uvs riding along, repair operates on the uv-SPLIT
    # confetti vertex set -- 29062 faces instead of 11010 on the mantis -- hole-filling runs wild, the retopo
    # inherits a shattered graph (65 components vs 12), and keep_largest amputates 11% of the surface. The uvs
    # lose nothing by being stripped here: src_uv is already captured above, and the bake stage re-projects
    # from the ORIGINAL src + src_uv, not from the repaired mesh.
    src_geo = Mesh(np.asarray(src.vertices, float), [tuple(f) for f in src.faces])
    rep = mesh_repair(src_geo, fill_holes=True)
    repaired = rep[0] if isinstance(rep, tuple) else rep
    oriented, _flips = mesh_orient(repaired)
    report["stages"].append({"stage": "repair", "faces": len(oriented.faces)})
    out = oriented

    # -- 2 RETOPO the repaired mesh -------------------------------------------------------------------
    if retopo:
        # SILHOUETTE-GUARDED, like the mind faculty and like QEM's decimate: the raw extractor at a coarse
        # density silently drops thin features (measured: the mantis legs vanish; QEM never loses them
        # because its guard REFUSES such budgets). knob = 1/density so growing the knob = finer retopo;
        # the guard retries finer until the worst-view IoU holds. This is the lesson FROM the LOD process:
        # the difference was never the algorithm, it was the guard.
        from holographic.mesh_and_geometry.holographic_crossfield import surface_retopo
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guarded
        q, g = silhouette_guarded(oriented, lambda k: surface_retopo(oriented, density=1.0 / k, fast=retopo_fast, snap_singular=retopo_snap, feature_sized=retopo_sized)[0],
                                  knob=1.0 / float(density), min_iou=float(silhouette) if silhouette else None)
        out = triangulate_ngons(q)
        report["stages"].append({"stage": "retopo", "density": float(density),
                                 "faces": len(out.faces),
                                 "silhouette_iou": (min(g["silhouette_iou"].values())
                                                    if isinstance(g.get("silhouette_iou"), dict) else None),
                                 "guard_steps": g.get("steps")})

    # -- 3 LOD ----------------------------------------------------------------------------------------
    if lod is not None:
        if retopo:
            # coarser retopo, sized so face count ~ lod target. Cell area scales ~ density^2, so
            # faces ~ 1/density^2: density_lod = density * sqrt(current/target).
            cur = len(out.faces)
            target = int(lod) if lod >= 1 else max(16, int(round(cur * float(lod))))
            d_lod = float(density) * max(1.0, (cur / max(target, 1)) ** 0.5)
            from holographic.mesh_and_geometry.holographic_crossfield import surface_retopo
            from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guarded
            q2, g2 = silhouette_guarded(oriented, lambda k: surface_retopo(oriented, density=1.0 / k, fast=retopo_fast, snap_singular=retopo_snap, feature_sized=retopo_sized)[0],
                                        knob=1.0 / d_lod, min_iou=float(silhouette) if silhouette else None)
            out = triangulate_ngons(q2)
            report["stages"].append({"stage": "lod_via_coarser_retopo", "density": d_lod,
                                     "faces": len(out.faces), "target": target,
                                     "silhouette_iou": (min(g2["silhouette_iou"].values())
                                                        if isinstance(g2.get("silhouette_iou"), dict) else None),
                                     "guard_steps": g2.get("steps")})
        else:
            from holographic.mesh_and_geometry.holographic_meshqem import decimate_to
            kw = {"target_faces": int(lod)} if lod >= 1 else {"target_fraction": float(lod)}
            lod_mesh, drep = decimate_to(out, keep_uv=False, **kw)
            out = triangulate_ngons(lod_mesh)
            report["stages"].append({"stage": "lod_via_decimate", "faces": len(out.faces),
                                     "budget_missed_for_silhouette": drep.get("budget_missed_for_silhouette")})

    # -- 4 SHARD CLEANUP after any retopo, GATED by topology invariants (R1) --------------------------
    if retopo and not keep_shards:
        # The gate compares the retopo result against the REPAIRED INPUT: intended holes = the input's
        # boundary loops; fragmentation / new loops / genus change = destruction. MEASURED motive: at coarse
        # unguarded densities keep_largest silently amputated 11% of a scanned mantis. The drop still runs
        # (a shattered extraction must not ship shards), but the amputation is now ACCOUNTED: the stage
        # reports the gate verdict, its named violations, and the exact face fraction dropped -- a loud,
        # machine-readable failure the caller (or the silhouette guard) can react to, instead of silence.
        gate_ok, gate_rep = topology_gate(oriented, out)
        out, crep = drop_small_components(out, keep_largest=True)
        dropped_frac = 1.0 - (crep["faces_after"] / max(crep["faces_before"], 1))
        report["stages"].append({"stage": "shard_cleanup", **{k: crep[k] for k in
                                 ("components_before", "components_after", "faces_after")},
                                 "topology_ok": bool(gate_ok), "topology_violations": gate_rep["violations"],
                                 "dropped_fraction": float(dropped_frac)})

    # -- 4b OPTIONAL STRICT MANIFOLD (R3) -------------------------------------------------------------
    if retopo and manifold:
        # opt-in: split non-manifold fins so QEM decimate / half-edge consumers accept the result. Reports
        # the honest cost (a few small holes for strict manifoldness) -- default OFF because it drops faces.
        out, mrep = manifold_cleanup(out, keep_largest=True)
        report["stages"].append({"stage": "manifold_cleanup", "non_manifold_before": mrep["non_manifold_before"],
                                 "non_manifold_after": mrep["non_manifold_after"], "new_loops": mrep["new_loops"],
                                 "faces_kept_frac": mrep["faces_kept_frac"]})

    # -- 5 FRESH ATLAS + REPROJECT the original texture -----------------------------------------------
    if src_uv is not None and texture is not None:
        src_mesh = Mesh(np.asarray(src.vertices, float), [tuple(f) for f in src.faces], uvs=src_uv)
        baked_mesh, new_uv, image, brep = rebake_texture(src_mesh, src_uv, np.asarray(texture, float),
                                                         out, size=int(bake_size), margin=int(bake_margin),
                                                         method=bake_method, fill_mode="flood", normal_aware=bake_normal_aware)
        report["stages"].append({"stage": "rebake", "texel_coverage": brep.get("texel_coverage"),
                                 "projection_distance_mean": brep.get("projection_distance_mean"),
                                 "method": brep.get("method")})
        report["faces"] = len(baked_mesh.faces)
        return baked_mesh, np.asarray(new_uv, float), np.asarray(image, float), report
    report["faces"] = len(out.faces)
    return out, None, None, report


def drop_small_components(mesh, min_faces=None, min_fraction=0.0, keep_largest=False):
    """Remove disconnected surface COMPONENTS that are too small -- the cleanup a field-guided retopo needs,
    because extracting quads from a scan drops isolated cells (measured: a mantis retopo shattered into 88
    components, one 90%-of-verts body plus ~75 shards of <5 verts, plus zero-area faces). Those shards render
    as speckle and break UV packing; the coherent body is what you want.

    Selection (a component is KEPT if it passes the active test):
      * keep_largest=True  -> keep ONLY the single largest component (by face count). The blunt, safe default
        for 'I want the body, not the debris'.
      * min_faces=N        -> keep every component with >= N faces.
      * min_fraction=f     -> keep every component with >= f * (largest component's face count) faces.
    If several are set, a component is kept when it passes ANY of them (union). With nothing set, keep_largest
    is assumed (the common case). Verts are re-indexed and unreferenced ones dropped; per-vertex uvs/normals/
    colours are carried through the remap. Returns (cleaned_mesh, report) where report gives the component
    count before/after and faces kept. Built on the shared graph flood (holographic_island.connected_components)
    -- the same primitive mesh_connected_components and route use; this is its mesh-cleanup costume.

    WHY components and not just 'small faces': a shard is a CONNECTED clump the extractor emitted in isolation;
    dropping it by area would also nuke legitimately-small faces on a dense region. Component size is the honest
    unit of 'debris'. KEPT NEGATIVE: this cannot reconnect a body that the extractor split -- it only removes;
    reconnecting sub-cell-width tubes (thin legs) needs a denser field or a different extractor, not cleanup."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.simulation_and_physics.holographic_island import connected_components
    V = np.asarray(mesh.vertices, float)
    F = [tuple(int(i) for i in f) for f in mesh.faces]
    if not F:
        return mesh, {"components_before": 0, "components_after": 0, "faces_before": 0, "faces_after": 0}
    # edge adjacency over VERTICES -> flood into components; then bucket FACES by their vertices' component
    eset = set()
    for f in F:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            eset.add((a, b) if a < b else (b, a))
    comps = connected_components(len(V), list(eset))
    vert_comp = np.full(len(V), -1, int)
    for ci, comp in enumerate(comps):
        for v in comp:
            vert_comp[v] = ci
    # face count per component (a face belongs to its first vertex's component; welded so all three agree)
    face_comp = np.array([vert_comp[f[0]] for f in F])
    comp_face_count = np.array([int((face_comp == ci).sum()) for ci in range(len(comps))])
    if not (keep_largest or min_faces is not None or min_fraction > 0.0):
        keep_largest = True                                   # sensible default: keep the body
    largest = int(comp_face_count.max()) if len(comp_face_count) else 0
    keep = np.zeros(len(comps), bool)
    if keep_largest:
        keep[int(np.argmax(comp_face_count))] = True
    if min_faces is not None:
        keep |= comp_face_count >= int(min_faces)
    if min_fraction > 0.0:
        keep |= comp_face_count >= min_fraction * largest
    keep_faces = [f for f, fc in zip(F, face_comp) if keep[fc]]
    used = sorted({v for f in keep_faces for v in f})
    remap = {v: i for i, v in enumerate(used)}
    V2 = V[used]
    F2 = [tuple(remap[v] for v in f) for f in keep_faces]
    def _carry(attr):
        a = getattr(mesh, attr, None)
        if a is None:
            return None
        a = np.asarray(a)
        return a[used] if a.shape[0] == len(V) else None
    out = Mesh(V2, F2, normals=_carry("normals"), uvs=_carry("uvs"), colours=_carry("colours"))
    report = {"components_before": len(comps), "components_after": int(keep.sum()),
              "faces_before": len(F), "faces_after": len(F2),
              "largest_component_faces": largest}
    return out, report


def topology_report(mesh):
    """PER-COMPONENT topology invariants -- the numbers that distinguish an INTENDED hole from mesh
    DESTRUCTION (R1 of the retopo-topology backlog). For each connected component: V/E/F, euler chi = V-E+F,
    boundary-loop count B (a closed chain of edges used by exactly one face -- the rim of a hole or an open
    border), and genus g = (2 - chi - B)/2 (None when non-manifold, where genus is undefined). Each boundary
    loop carries a geometric FINGERPRINT (centroid, length, mean normal of adjacent faces) so loops can be
    MATCHED across a remesh -- an intended hole is a loop present in the input; a new loop is damage.

    WHY per-component and why loops, not boundary-EDGE counts: a remesh can keep the same total boundary-edge
    count while splitting one rim into two (a crack), and a global euler hides one component gaining a handle
    while another loses one. Loops + per-component invariants are the honest resolution. Built on the shared
    graph flood (holographic_island.connected_components) -- the report costume of the same primitive
    drop_small_components wears for cleanup."""
    import numpy as np
    from holographic.simulation_and_physics.holographic_island import connected_components
    V = np.asarray(mesh.vertices, float)
    F = [tuple(int(i) for i in f) for f in mesh.faces]
    if not F:
        return {"components": 0, "per_component": [], "boundary_loops": [], "non_manifold_edges": 0, "ok": True}
    edge_faces = {}
    for fi, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            edge_faces.setdefault((a, b) if a < b else (b, a), []).append(fi)
    non_manifold = sum(1 for fs in edge_faces.values() if len(fs) > 2)
    comps = connected_components(len(V), list(edge_faces.keys()))
    vert_comp = np.full(len(V), -1, int)
    for ci, comp in enumerate(comps):
        for v in comp:
            vert_comp[v] = ci
    # ---- trace boundary LOOPS: walk boundary edges (exactly one face) vertex-to-vertex into closed chains --
    bedges = [e for e, fs in edge_faces.items() if len(fs) == 1]
    nxt = {}
    for a, b in bedges:                                   # undirected walk; degree>2 junctions = non-manifold rim
        nxt.setdefault(a, []).append(b); nxt.setdefault(b, []).append(a)
    seen = set(); loops = []
    for a, b in bedges:
        if (a, b) in seen or (b, a) in seen:
            continue
        loop = [a, b]; seen.add((a, b))
        while True:
            cur, prev = loop[-1], loop[-2]
            cands = [v for v in nxt.get(cur, []) if v != prev and (cur, v) not in seen and (v, cur) not in seen]
            if not cands:
                break
            v = min(cands)                                # deterministic tie-break (constitution: no ordering luck)
            seen.add((cur, v)); loop.append(v)
            if v == loop[0]:
                break
        closed = len(loop) > 2 and loop[-1] == loop[0]
        pts = V[np.asarray(loop[:-1] if closed else loop, int)]
        length = float(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0), axis=2 - 1).sum())
        loops.append({"component": int(vert_comp[loop[0]]), "n_edges": len(loop) - 1, "closed": bool(closed),
                      "centroid": pts.mean(0).tolist(), "length": length})
    # ---- per-component V/E/F/chi/B/genus ----
    per = []
    for ci, comp in enumerate(comps):
        cv = set(comp)
        cF = [f for f in F if f[0] in cv]
        cE = sum(1 for (a, b) in edge_faces if a in cv)
        B = sum(1 for L in loops if L["component"] == ci and L["closed"])
        chi = len(comp) - cE + len(cF)
        genus = None if non_manifold else (2 - chi - B) / 2.0
        if genus is not None:
            genus = int(genus) if float(genus).is_integer() and genus >= 0 else None   # non-integer = degenerate
        per.append({"V": len(comp), "E": cE, "F": len(cF), "chi": chi, "boundary_loops": B, "genus": genus})
    return {"components": len(comps), "per_component": per, "boundary_loops": loops,
            "non_manifold_edges": int(non_manifold), "ok": non_manifold == 0}


def topology_gate(before, after, loop_match_tol=0.25):
    """ACCEPT or REJECT a remesh by topology invariants (R1): PASS iff the output has the same component
    count, no non-manifold edges, per-component genus preserved (matched by descending face count), and every
    output boundary loop MATCHES an input loop (centroid within loop_match_tol * input loop length, sizes
    comparable). An INTENDED hole is a matched loop; a NEW unmatched loop, a new component, or a genus change
    is DESTRUCTION and fails, with the violation NAMED in the report -- so the pipeline retries finer instead
    of amputating with keep_largest. MEASURED motive: keep_largest silently dropped 11%% of a scanned mantis
    when the extraction shattered; this gate turns that silent amputation into a loud, retryable failure.
    `before`/`after` are meshes or topology_report dicts. Returns (passed, report)."""
    rb = before if isinstance(before, dict) else topology_report(before)
    ra = after if isinstance(after, dict) else topology_report(after)
    violations = []
    if ra["components"] > rb["components"]:
        violations.append("fragmentation: %d components (input had %d)" % (ra["components"], rb["components"]))
    if ra["non_manifold_edges"] > 0:
        violations.append("%d non-manifold edges introduced" % ra["non_manifold_edges"])
    pb = sorted(rb["per_component"], key=lambda c: -c["F"])
    pa = sorted(ra["per_component"], key=lambda c: -c["F"])
    for i, (cb, ca) in enumerate(zip(pb, pa)):
        if cb["genus"] is not None and ca["genus"] is not None and cb["genus"] != ca["genus"]:
            violations.append("component %d genus %s -> %s" % (i, cb["genus"], ca["genus"]))
    # loop matching: every AFTER loop must correspond to a BEFORE loop (intended); unmatched = punched hole
    import numpy as np
    new_loops = 0
    for La in ra["boundary_loops"]:
        if not La["closed"]:
            continue
        matched = False
        for Lb in rb["boundary_loops"]:
            if not Lb["closed"]:
                continue
            d = float(np.linalg.norm(np.asarray(La["centroid"]) - np.asarray(Lb["centroid"])))
            if d <= loop_match_tol * max(Lb["length"], 1e-9) and 0.25 <= La["length"] / max(Lb["length"], 1e-9) <= 4.0:
                matched = True
                break
        if not matched:
            new_loops += 1
    if new_loops:
        violations.append("%d new boundary loop(s) (holes not present in the input)" % new_loops)
    return len(violations) == 0, {"passed": len(violations) == 0, "violations": violations,
                                  "before": {"components": rb["components"], "loops": len(rb["boundary_loops"])},
                                  "after": {"components": ra["components"], "loops": len(ra["boundary_loops"])}}


def manifold_cleanup(mesh, keep_largest=True, verbose_gate=True):
    """R3 -- make a retopo result STRICTLY MANIFOLD so downstream ops (QEM decimate, half-edge builds) accept
    it, and REPORT the topological cost honestly instead of hiding it.

    THE MEASURED PROBLEM this solves: a guarded scan retopo comes out 1-component, hole-free, silhouette-
    correct -- but with non-manifold EDGES at the field's singular cells (mantis: 142 edges, 136 of them
    4-face 'fins' where extract_quads' winding-preserving dedup kept both copies of a feature that collapsed
    to a single sheet). QEM decimate REFUSES such a mesh ('directed edge appears twice'), so LOD-on-retopo is
    blocked. Four LOCAL surgeries were tried and all traded the defect for a worse one -- KEPT NEGATIVES, on
    record so no future session reinvents them: drop-one-duplicate opens 51 holes (removes one side of a real
    thin sheet); drop-both-copies fragments 1->40 components (a fin was the only bridge); dihedral-keep-2
    clears nm but opens 67 loops AND still fails QEM. A fifth surgery -- remove one copy of each EXACT-
    duplicate face (which looks lossless) -- was also refuted, and this one is now PROVEN impossible, not just
    observed: MEASURED, all 259 duplicate faces on the density-2 mantis retopo are OPPOSITE-WOUND, i.e. genuine
    two-sided sheets that both sides are needed to enclose. Removing either side must open a hole (measured:
    259 removed -> 288 boundary edges). So NO local removal is lossless -- the two copies are not redundant,
    they are the front and back of a zero-thickness feature. The honest conclusion (which the research
    predicted): a clean fix needs a manifold-GUARANTEEING extraction (QuadriFlow's global integer assignment,
    filed as R3-proper), not post-hoc local repair. R3-proper stays FILED, not built: its only consumer
    (QEM-on-retopo) is ALREADY unblocked by the split+gate below, so a QuadriFlow-scale min-cost-flow solver
    is not justified by a measured need -- it would save 24 small holes on one path at large complexity cost.

    What this DOES, as the pragmatic route until R3-proper lands: split non-manifold vertices into their
    manifold umbrellas (the reference cut), then keep the largest component, then GATE. MEASURED on the mantis:
    142 nm edges -> 0, 1 component preserved, 24 small boundary loops introduced (the fin regions become tiny
    holes), 93.3%% of faces kept -- and QEM accepts it. The tradeoff (a few small holes for strict
    manifoldness) is REPORTED via topology_gate, not hidden: the caller sees exactly what was spent.

    Returns (mesh, report) with report.gate = the topology_gate verdict vs the input, report.faces_kept_frac,
    report.new_loops. Default keep_largest=True (the manifold split can shed a shard)."""
    r0 = topology_report(mesh)
    split = split_nonmanifold_vertices(mesh)
    out = split[0] if isinstance(split, tuple) else split
    if keep_largest:
        out, _ = drop_small_components(out, keep_largest=True)
    r1 = topology_report(out)
    ok, gate = topology_gate(mesh, out)
    new_loops = len([L for L in r1["boundary_loops"] if L["closed"]]) - \
                len([L for L in r0["boundary_loops"] if L["closed"]])
    report = {"non_manifold_before": r0["non_manifold_edges"], "non_manifold_after": r1["non_manifold_edges"],
              "components_before": r0["components"], "components_after": r1["components"],
              "faces_kept_frac": len(out.faces) / max(len(mesh.faces), 1),
              "new_loops": int(max(new_loops, 0)), "gate": gate,
              "manifold": r1["non_manifold_edges"] == 0}
    return out, report


def _selftest_manifold_cleanup():
    """R3: cleanup makes a non-manifold fixture manifold, preserves the component, reports the cost."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    b = triangulate_ngons(box())
    V = np.asarray(b.vertices, float); F = [tuple(int(i) for i in f) for f in b.faces]
    # add a FIN: duplicate one face with opposite winding sharing all 3 verts -> a 4-face edge set
    a, c, d = F[0]
    fin = Mesh(V, F + [(a, d, c)])
    r0 = topology_report(fin)
    assert r0["non_manifold_edges"] > 0, "fixture must be non-manifold"
    out, rep = manifold_cleanup(fin)
    assert rep["manifold"] and rep["non_manifold_after"] == 0, "cleanup must eliminate non-manifold edges"
    assert rep["components_after"] >= 1 and "gate" in rep
    # THE CONSUMER CONTRACT (the R3 motive): QEM decimate REFUSES a non-manifold mesh but must ACCEPT the
    # cleaned one -- proving LOD-on-retopo is unblocked. This is why manifold_cleanup earns its keep.
    from holographic.mesh_and_geometry.holographic_meshqem import qem_decimate
    try:
        qem_decimate(fin, target_faces=8)
        raise AssertionError("QEM should have refused the non-manifold fixture")
    except AssertionError:
        raise
    except Exception:
        pass                                                 # expected: QEM refuses the fin
    dm = qem_decimate(out, target_faces=8)                    # must NOT raise
    assert len(dm.faces) > 0
    # KEPT NEGATIVE pinned: cleanup can introduce boundary loops (the honest cost); it does NOT pretend to be
    # lossless. A clean manifold result needs R3-proper (global integer assignment), not this split+gate.
    print("manifold_cleanup selftest OK (fin removed, nm %d->0, %.0f%% faces kept, %d new loops reported)" % (
        rep["non_manifold_before"], 100 * rep["faces_kept_frac"], rep["new_loops"]))


def mesh_repair(mesh, weld_tol=1e-5, fill_holes=True, max_fill_sides=0, drop_unreferenced=True, split_nonmanifold=True,
                triangulate=False):
    """REPAIR a mesh by composing the standard cleanup ops (it does NOT reinvent them): WELD near-duplicate vertices
    (merge_by_distance, which also drops any face that collapses to a degenerate), SPLIT non-manifold vertices into
    connected umbrellas (split_nonmanifold_vertices -- makes the mesh MANIFOLD so a field solver / cross-field retopo
    accepts it), optionally FILL open holes (holographic_meshverbs2.fill_holes), and optionally DROP unreferenced
    vertices. Returns (repaired_mesh, report) with before/after vertex+face counts, manifold/closed flags, and the
    split count -- so a RAW mesh (marching-cubes / import / boolean / photo-to-mesh) can be made RETOPO-READY. Deterministic.

    ORDER MATTERS: weld runs BEFORE split (a weld would re-merge the same-position copies a split creates); split runs
    before fill (fill needs a traceable manifold boundary). triangulate=True ear-clips the result to uniform triangles
    (cross_field retopo needs a single face arity; split/fill leave a mixed tri/quad mesh). KEPT NEGATIVE: the fan hole-fill closes a boundary by adding
    a centroid vertex -- CLOSED, not necessarily well-shaped; follow with retopology (cross_field). split_nonmanifold on
    a pure X-junction over-splits into disconnected sheets (manifold but open) -- the minimal cut, honest. fill_holes is
    wrapped: if it cannot run it is skipped and noted, never raised."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh

    def _stats(m):
        try:
            manifold = bool(m.is_manifold())
        except Exception:
            manifold = False
        try:
            closed = bool(m.is_closed())
        except Exception:
            closed = False
        return {"vertices": int(len(m.vertices)), "faces": int(len(m.faces)), "manifold": manifold, "closed": closed}

    before = _stats(mesh)
    m = merge_by_distance(mesh, tol=weld_tol)                 # weld + degenerate-face drop (the biggest single fixer)
    n_split = 0
    if split_nonmanifold:                                     # cut non-manifold vertices into manifold umbrellas so a
        m, srep = split_nonmanifold_vertices(m)              # half-edge build / cross-field retopo will accept the mesh
        n_split = srep["split_vertices"]                     # (weld MUST run first: it would re-merge the split copies)
    filled = False
    # ATTRIBUTE LAYER: weld/split above carry uvs+normals natively (attrs="auto" weld; split copies rows), but
    # fill_holes / triangulate_ngons rebuild the Mesh and STRIP them (they predate attribute carry). Strategy,
    # chosen over projecting EVERYTHING afterwards: exact carry beats projection wherever possible, because a
    # closest-point projection at a UV SEAM is ambiguous (two atlas islands touch in 3-D; the nearest triangle
    # may belong to either) -- measured scrambled seam uvs on the first draft of this fix. So: remember the
    # attribute-carrying mesh, run fill/triangulate on geometry, then re-attach EXACT rows for every vertex that
    # survived unchanged (fill APPENDS; triangulate keeps the vertex table) and PROJECT only the appended tail
    # (hole-fill centroids -- extrapolations by nature, the projection residual is the honest error signal).
    m_attr = m
    if fill_holes:
        try:
            from holographic.mesh_and_geometry.holographic_meshverbs2 import fill_holes as _fill
            m2 = _fill(m, max_sides=max_fill_sides)
            m, filled = m2, True
        except Exception:
            filled = False                                   # a still-broken boundary -> skip, do not raise
    if triangulate:                                          # uniform all-triangle faces -- cross_field retopo needs a
        from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons   # single face arity, and
        m = triangulate_ngons(m)                             # a split/fill result is mixed tri/quad
    if getattr(m_attr, "uvs", None) is not None and getattr(m, "uvs", None) is None:
        try:
            Va = np.asarray(m_attr.vertices, float); Vm = np.asarray(m.vertices, float)
            na = len(Va)
            uva = np.asarray(m_attr.uvs, float)
            nrma = None if getattr(m_attr, "normals", None) is None else np.asarray(m_attr.normals, float)
            if len(Vm) >= na and np.array_equal(Vm[:na], Va):
                tail = Vm[na:]
                uv_tail, _ = transfer_uv(m_attr, uva, tail) if len(tail) else (np.zeros((0, uva.shape[1])), None)
                uv_full = np.vstack([uva, uv_tail])
                nrm_full = None
                if nrma is not None:
                    nrm_tail, _ = transfer_uv(m_attr, nrma, tail) if len(tail) else (np.zeros((0, 3)), None)
                    nrm_full = np.vstack([nrma, nrm_tail])
                    ln = np.linalg.norm(nrm_full, axis=1, keepdims=True)
                    nrm_full = nrm_full / np.where(ln > 1e-12, ln, 1.0)
            else:                                            # vertex table changed shape: project everything, honest fallback
                uv_full, _ = transfer_uv(m_attr, uva, Vm)
                nrm_full = None
                if nrma is not None:
                    nrm_full, _ = transfer_uv(m_attr, nrma, Vm)
                    ln = np.linalg.norm(nrm_full, axis=1, keepdims=True)
                    nrm_full = nrm_full / np.where(ln > 1e-12, ln, 1.0)
            m = Mesh(m.vertices, m.faces, normals=nrm_full, uvs=uv_full)
        except Exception:
            pass                                             # a failed re-attach must not fail the repair
    if drop_unreferenced:
        m = _drop_unreferenced(m)
    after = _stats(m)
    return m, {"before": before, "after": after, "holes_filled": filled, "split_vertices": int(n_split),
               "uvs_carried": (getattr(m, "uvs", None) is not None) if getattr(mesh, "uvs", None) is not None else None,
               "vertices_delta": after["vertices"] - before["vertices"],   # negative = welded away (fill can add some back)
               "faces_delta": after["faces"] - before["faces"],
               "became_manifold": (not before["manifold"]) and after["manifold"],
               "became_closed": (not before["closed"]) and after["closed"]}


def diagnose_mesh(mesh, weld_tol=1e-5):
    """Diagnose a mesh into a CATEGORICAL defect record for routing: {manifold, closed, duplicates}. Values are
    categories (yes/no), never raw counts -- so the record can be matched against repair-strategy records with
    match_record (which is categorical by contract). `duplicates` is 'yes' when a weld would actually merge
    vertices (a cheap dry-run count), else 'no'. This is the sensing half of route_repair; on a clean mesh it
    returns all-good and route_repair becomes a no-op."""
    try:
        manifold = "yes" if bool(mesh.is_manifold()) else "no"
    except Exception:
        manifold = "no"
    try:
        closed = "yes" if bool(mesh.is_closed()) else "no"
    except Exception:
        closed = "no"
    # cheap duplicate probe: does a weld change the vertex count? (dry compare, no mutation kept)
    try:
        welded = merge_by_distance(mesh, tol=weld_tol)
        duplicates = "yes" if len(welded.vertices) < len(mesh.vertices) else "no"
    except Exception:
        duplicates = "no"
    return {"manifold": manifold, "closed": closed, "duplicates": duplicates}


# The repair STRATEGIES as categorical records: each names the mesh condition it is FOR. A strategy is a
# {manifold, closed, duplicates} record describing the defect it targets; match_record ranks them against the
# diagnosed mesh, and decide_or_abstain falls back to the full pipeline when no single strategy clearly wins.
_REPAIR_STRATEGIES = {
    "clean":         {"manifold": "yes", "closed": "yes", "duplicates": "no"},   # nothing to do
    "weld_only":     {"manifold": "yes", "closed": "yes", "duplicates": "yes"},  # just duplicate verts
    "make_manifold": {"manifold": "no",  "closed": "yes", "duplicates": "no"},   # non-manifold, no holes
    "fill_holes":    {"manifold": "yes", "closed": "no",  "duplicates": "no"},   # open boundary only
    "full_repair":   {"manifold": "no",  "closed": "no",  "duplicates": "yes"},  # everything wrong
}

# which concrete ops each strategy runs (the minimal set). full_repair defers to mesh_repair.
_STRATEGY_OPS = {
    "clean":         (),
    "weld_only":     ("weld",),
    "make_manifold": ("split",),
    "fill_holes":    ("fill",),
    "full_repair":   ("weld", "split", "fill"),
}


def route_repair(mesh, mind=None, margin=0.15, weld_tol=1e-5, max_fill_sides=0):
    """Route a mesh to the MINIMAL repair its defect needs, instead of always running the full weld+split+fill
    pipeline. Diagnoses the mesh into a categorical defect record (diagnose_mesh), matches it against the
    repair-strategy records with match_record, and applies only the ops that strategy names. WHY THIS EXISTS:
    mesh_repair runs every op every time; a mesh that is only non-manifold pays for a hole-fill pass it does not
    need, and the report cannot say WHY a repair ran. route_repair picks the targeted op set and returns the
    named strategy, so the repair is cheaper AND self-explaining.

    HONEST FALLBACK (decide_or_abstain): if no single strategy clearly wins (the defect record sits between two
    strategies by less than `margin`), it does NOT guess a minimal op -- it runs the FULL mesh_repair, the safe
    superset. So route_repair can never repair LESS than the ambiguous case needs; it only saves work when the
    diagnosis is unambiguous. `mind` supplies encode_record (default: a fresh small UnifiedMind). Returns
    (repaired_mesh, report) where report adds {strategy, confident, defect} to mesh_repair's usual fields.

    KEPT NEGATIVE: the strategy set is categorical over {manifold, closed, duplicates} only -- it does NOT model
    severity or hole SIZE (continuous), so a huge hole and a tiny hole route the same way. That is deliberate:
    which OP to run is categorical; how hard the op works is the op's own (continuous) business."""
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=512, seed=0)
    from holographic.misc.holographic_relations import match_record, decide_or_abstain

    defect = diagnose_mesh(mesh, weld_tol=weld_tol)
    ranked = match_record(mind.encode_record, defect, _REPAIR_STRATEGIES)
    strategy, score, confident = decide_or_abstain(ranked, margin=margin)

    if not confident:
        # ambiguous defect -> run the full safe pipeline rather than a possibly-insufficient minimal op set
        repaired, rep = mesh_repair(mesh, weld_tol=weld_tol, fill_holes=True, max_fill_sides=max_fill_sides)
        rep.update({"strategy": "full_repair", "confident": False, "defect": defect})
        return repaired, rep

    ops = _STRATEGY_OPS[strategy]
    if not ops:                                              # 'clean' -> nothing to do; return the mesh untouched
        repaired, rep = mesh_repair(mesh, weld_tol=weld_tol, fill_holes=False, split_nonmanifold=False,
                                    drop_unreferenced=False)
    else:
        # run ONLY the ops the chosen strategy names (mesh_repair with the others switched off)
        repaired, rep = mesh_repair(mesh, weld_tol=weld_tol,
                                    fill_holes=("fill" in ops),
                                    split_nonmanifold=("split" in ops))
    rep.update({"strategy": strategy, "confident": True, "defect": defect})
    return repaired, rep


def mirror(mesh, axis=0, plane=0.0, weld=True, tol=1e-5):
    """Mirror a mesh across the `axis`=const `plane`: append a reflected copy with reversed winding (a reflection
    flips orientation, so the normals stay consistent), then optionally WELD the seam vertices that land on the
    plane. The standard way to model a symmetric object from one half. Vectorised."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices
    Vm = V.copy()
    Vm[:, axis] = 2.0 * plane - Vm[:, axis]                   # reflect across the plane
    Vall = np.vstack([V, Vm])
    off = len(V)
    faces = mesh.faces
    Fm = [tuple(reversed([int(vi) + off for vi in f])) for f in faces]   # reversed winding for the reflection
    out = Mesh(Vall, list(faces) + Fm)
    if weld:
        out = merge_by_distance(out, tol=tol)                 # fuse the coincident seam vertices
    return out


def symmetrize(mesh, axis=0, plane=0.0, side=+1, tol=1e-5):
    """SYMMETRIZE a (possibly asymmetric) mesh across the `axis`=const `plane`: KEEP the half on `side` (+1 = the
    positive side of the plane, -1 = the negative), then MIRROR that half back across the plane and weld the seam,
    producing a bilaterally-symmetric mesh. The 'symmetrize' a modeler runs to clean up a sculpt that drifted off
    axis -- unlike `mirror` (which doubles the WHOLE mesh), this first discards the far side, so it fixes asymmetry
    instead of preserving it.

    A face is kept iff its CENTROID is on `side` of the plane (the robust, Blender-style rule -- no cutting of
    straddling faces, which would need re-triangulation). Vertices within `tol` of the plane are SNAPPED exactly onto
    it first, so the seam welds cleanly after the mirror. Returns a new `Mesh`. Composes the existing mirror + weld
    (merge_by_distance) primitives -- symmetrize is 'keep one side, then mirror', not a new reflection kernel.

    KEPT NEGATIVE: a face that lies IN the mirror direction (its centroid on the plane, e.g. the side faces of an
    axis-aligned box mirrored across x=0) is kept and then mirrored, producing a coplanar duplicate -- so a box goes
    6 -> 10 faces. The result is still SYMMETRIC (the contract), just not minimal; a proper fix would drop faces whose
    supporting plane contains the mirror normal, deferred. For the intended use (an off-axis sculpt with faces that
    genuinely pick a side) this does not arise."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices.copy()
    s = float(np.sign(side)) or 1.0
    # snap near-plane vertices exactly onto the plane, so the mirrored copy's seam vertices coincide and weld.
    near = np.abs(V[:, axis] - plane) <= tol
    V[near, axis] = plane
    kept_faces = []
    for f in mesh.faces:
        idx = list(f)
        centroid_side = s * (V[idx, axis].mean() - plane)   # which side the face's centroid is on
        # KEEP a face whose centroid is on the keep side OR on the plane (a straddling face belongs to both halves;
        # keeping it and mirroring it produces the symmetric pair). Only faces whose centroid is clearly on the FAR
        # side are discarded -- that is the asymmetry being cut away. (No cutting of straddling faces, Blender-style.)
        if centroid_side >= -tol:
            kept_faces.append(tuple(idx))
    half = Mesh(V, kept_faces)
    # mirror the kept half back across the plane and weld the on-plane seam -> the symmetric whole.
    return mirror(half, axis=axis, plane=plane, weld=True, tol=tol)


def skin_skeleton(verts, edges, radii, resolution=64, smooth_k=None, taper=True):
    """SKIN A SKELETON (Ji-Liu-Wang 2010 B-Mesh, the SDF route): given a stick figure -- `verts` (n,3), `edges` list
    of (i,j) index pairs, and per-vertex `radii` (n,) -- build a single watertight surface mesh wrapping it, so you
    model a creature by placing ~20 joints instead of extruding 200 faces. Each edge becomes a capsule (radius = its
    endpoints' radii; tapered if `taper`), all capsules smooth-UNIONED so branches at a shared joint MERGE into one
    blob automatically (the hard part of B-Mesh -- here it is free, because overlapping capsules fuse), then
    marching-cubes'd to a mesh at `resolution`.

    `smooth_k` sets the blend radius at joints (default ~0.4 * the median radius -- larger = rounder joints, smaller =
    sharper). Returns a Mesh (all-triangle from marching cubes; run quad_remesh + catmull_clark for a quad cage, or
    shrinkwrap a box cage onto it). Deterministic (fixed grid + seed-free marching cubes).

    WHY THIS IS THE BASE-MESH WIN: the mantis box-model was 200 lines of extrude-and-steer; as a skeleton it is a
    handful of joints and radii. KEPT NEGATIVE: the SDF route gives ORGANIC blob topology (isotropic triangles), NOT
    the clean edge-loops a hand-built cage has -- it is a BLOCK-OUT to retopo onto, not a final cage. A true B-Mesh
    (swept quad rings + convex-hull joint stitching) would emit quads directly; that is the heavier future build.
    Also: a capsule uses the max of its endpoint radii when taper=False; very different radii at the two ends of one
    edge read best with taper=True."""
    from holographic.mesh_and_geometry.holographic_sdf import capsule, sphere
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.asarray(verts, float)
    R = np.asarray(radii, float)
    if len(V) != len(R):
        raise ValueError("skin_skeleton: one radius per vertex required (len(verts) != len(radii))")
    med = float(np.median(R)) if len(R) else 0.1
    k = float(smooth_k) if smooth_k is not None else 0.4 * med
    Yaxis = np.array([0.0, 1.0, 0.0])

    def _oriented_capsule(a, b, ra, rb):
        # a capsule is along Y from -h..+h, radius r. Place it along the segment a->b: half-length h, radius = the
        # larger end (taper handled separately via an added sphere at the fat end), rotate Y onto the edge dir,
        # translate to the midpoint. A pure-Y edge needs no rotation (and cross([0,1,0],[0,1,0])=0).
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return sphere(max(ra, rb)).translate(list(a))
        u = d / L
        r = max(ra, rb)
        cap = capsule(L * 0.5, r)                       # spans -L/2..+L/2 on Y
        axis = np.cross(Yaxis, u)
        s = float(np.linalg.norm(axis))
        c = float(np.dot(Yaxis, u))
        if s > 1e-9:
            angle = float(np.arctan2(s, c))
            cap = cap.rotate((axis / s).tolist(), angle)
        elif c < 0:                                     # antiparallel: flip 180 about X
            cap = cap.rotate([1.0, 0.0, 0.0], float(np.pi))
        mid = 0.5 * (a + b)
        return cap.translate(list(mid))

    parts = []
    # a sphere at every joint keeps the radius honest at the ends and rounds branch points (taper: fat end sphere)
    for i in range(len(V)):
        parts.append(sphere(float(R[i])).translate(list(V[i])))
    for (i, j) in edges:
        parts.append(_oriented_capsule(V[int(i)], V[int(j)], float(R[int(i)]), float(R[int(j)])))
    if not parts:
        raise ValueError("skin_skeleton: empty skeleton (no verts/edges)")
    field = parts[0]
    for p in parts[1:]:
        field = field.smooth_union(p, k)               # branches merge here, for free

    lo = (V.min(0) - R.max() - k - 0.1)
    hi = (V.max(0) + R.max() + k + 0.1)
    n = int(resolution)
    xs = np.linspace(lo[0], hi[0], n); ys = np.linspace(lo[1], hi[1], n); zs = np.linspace(lo[2], hi[2], n)
    GX, GY, GZ = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
    vals = field.eval(P).reshape(n, n, n)                  # sample the composite SDF on the grid
    from holographic.mesh_and_geometry.holographic_meshbridge import marching_tetrahedra_vec
    return marching_tetrahedra_vec(vals, (xs, ys, zs), level=0.0)


def fit_base_mesh(target_mesh, verts, edges, radii, resolution=64, smooth_k=None, shrink_factor=1.0):
    """FIT A BASE MESH TO A TARGET (the closed block-out loop): skin the skeleton (verts/edges/radii) into a watertight
    base mesh, then SHRINKWRAP it onto `target_mesh`, and report how much the fit improved. This is the leCore answer
    to the Blender "block out with the skin modifier, then snap to the sculpt" workflow -- and because it returns the
    silhouette-fit numbers, it is an OPTIMISATION target (nudge the radii/joints to raise mean_iou).

    Returns a dict:
      base     : the skinned base mesh (before shrinkwrap)
      fitted   : the base mesh shrinkwrapped onto the target
      residual : per-vertex distance the shrinkwrap closed (how far the block-out sat off the target)
      iou_base, iou_fitted : mean silhouette IoU vs the target, before and after (the fit gain is iou_fitted-iou_base)
    Deterministic. KEPT NEGATIVE: shrinkwrap is closest-point, so a base mesh that misses a target limb entirely can
    pull its whole surface onto the body (no limb appears) -- the skeleton must roughly cover the target's parts
    first; this fits SHAPE, not TOPOLOGY (the fitted mesh is still isotropic-triangle -- retopo after)."""
    from holographic.rendering.holographic_render import turnaround as _ta
    base = skin_skeleton(verts, edges, radii, resolution=resolution, smooth_k=smooth_k)
    fitted, residual = shrinkwrap(base, target_mesh, factor=shrink_factor)
    iou_base = _ta(base, ref_mesh=target_mesh, width=160, height=160)["mean_iou"]
    iou_fitted = _ta(fitted, ref_mesh=target_mesh, width=160, height=160)["mean_iou"]
    return {"base": base, "fitted": fitted, "residual": residual,
            "iou_base": iou_base, "iou_fitted": iou_fitted}


def bake_normal_map(low_mesh, low_uv, high_mesh, size=256, world_space=False, ao=False, ao_samples=0, displacement=None, max_distance=None):
    """BAKE a normal map (and optionally AO) from a HIGH-poly onto a LOW-poly's UVs -- the standard "keep the sculpt
    detail on the retopo" step. For each texel the low-poly's UV layout covers: find its 3-D point on the low-poly,
    project to the CLOSEST point on the high-poly, read the high-poly's shading NORMAL there, and store it. Returns an
    (size,size,3) float image in [0,1].

    `world_space=False` (default) writes a TANGENT-space normal map (the portable kind): the high-poly normal is
    expressed in the low-poly's per-texel tangent frame, so it survives the low-poly deforming/animating; the flat
    (0,0,1) direction encodes as the familiar lavender (0.5,0.5,1.0). world_space=True writes the raw world normal
    (0.5+0.5*n) -- simpler, but only valid for a static pose. `ao=True` additionally returns an ambient-occlusion
    grey image (per texel, the fraction of `ao_samples` hemisphere rays that escape the high-poly) -- pass ao_samples
    (e.g. 16) to enable; returns (normal_img, ao_img) when ao else just normal_img.

    Reuses transfer_uv's closest-point projection (the high-poly lookup) and Mesh.vertex_normals (both surfaces).
    KEPT NEGATIVE: a per-texel closest-point bake with no cage/ray-distance limit, so a low-poly texel far from the
    high-poly still grabs the nearest normal (a floating detail bleeds); the classic fix is a max projection distance
    (a cage) -- not added. AO here is a coarse hemisphere occlusion against the high-poly triangles, O(texels*samples)."""
    from collections import defaultdict
    low_uv = np.asarray(low_uv, float)
    LV = np.asarray(low_mesh.vertices, float)
    LF = [tuple(int(i) for i in f[:3]) for f in low_mesh.triangulate()] if any(len(f) != 3 for f in low_mesh.faces) \
        else [tuple(int(i) for i in f) for f in low_mesh.faces]
    lnorm = low_mesh.vertex_normals(store=False)
    HV = np.asarray(high_mesh.vertices, float)
    HF = [tuple(int(i) for i in f[:3]) for f in high_mesh.triangulate()] if any(len(f) != 3 for f in high_mesh.faces) \
        else [tuple(int(i) for i in f) for f in high_mesh.faces]
    hnorm = high_mesh.vertex_normals(store=False)

    # spatial hash of high-poly triangles -- DELEGATED to the shared correspondence machine (M14). Same grid
    # and ring-search transfer_uv uses; here the caller reads NORMALS (and the hit point for displacement) off
    # the returned (face, bary). One projection, many channels: the grid/query is shared, the channel is local.
    grid, htri, hlo, cell = build_face_grid(HV, HF, cell_scale=1.0)

    def _high_normal_at(p):
        fi, bc, best_d2 = closest_face_point(p, grid, htri, hlo, cell, HF)
        f = HF[fi]
        best_n = bc[0] * hnorm[f[0]] + bc[1] * hnorm[f[1]] + bc[2] * hnorm[f[2]]
        best_q = bc[0] * HV[f[0]] + bc[1] * HV[f[1]] + bc[2] * HV[f[2]]   # the hit point, from the same bary
        n = best_n / (np.linalg.norm(best_n) + 1e-12)
        return n, best_q                                     # (normal, closest_point) -- one projection, two reads

    img = np.zeros((size, size, 3)); img[:] = (0.5, 0.5, 1.0) if not world_space else (0.5, 0.5, 0.5)
    aoimg = np.ones((size, size)) if ao else None
    # displacement: pass displacement=True (or a dict of options) to also bake a signed height map. Additive:
    # default None leaves bake_normal_map bit-identical. This is M14's "one projection, many channels" in
    # miniature -- the closest-point cast is done ONCE and both the normal and the height are read from it.
    dispimg = np.zeros((size, size)) if displacement is not None else None
    # rasterize each low-poly triangle in UV space; per covered texel interpolate 3-D position + low normal + tangent
    for f in LF:
        uv0, uv1, uv2 = low_uv[f[0]], low_uv[f[1]], low_uv[f[2]]
        p0, p1, p2 = LV[f[0]], LV[f[1]], LV[f[2]]
        n0, n1, n2 = lnorm[f[0]], lnorm[f[1]], lnorm[f[2]]
        # tangent from the UV gradient (Lengyel): dP/du gives the tangent direction
        duv1 = uv1 - uv0; duv2 = uv2 - uv0
        dp1 = p1 - p0; dp2 = p2 - p0
        denom = duv1[0] * duv2[1] - duv2[0] * duv1[1]
        tan = (dp1 * duv2[1] - dp2 * duv1[1]) / denom if abs(denom) > 1e-12 else dp1
        pix = low_uv[list(f)] * (size - 1)
        lo_i = np.floor(pix.min(0)).astype(int); hi_i = np.ceil(pix.max(0)).astype(int)
        lo_i = np.maximum(lo_i, 0); hi_i = np.minimum(hi_i, size - 1)
        A = np.array([[pix[1, 0] - pix[0, 0], pix[2, 0] - pix[0, 0]],
                      [pix[1, 1] - pix[0, 1], pix[2, 1] - pix[0, 1]]], float)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-9:
            continue
        for py in range(lo_i[1], hi_i[1] + 1):
            for px in range(lo_i[0], hi_i[0] + 1):
                rhs = np.array([px - pix[0, 0], py - pix[0, 1]], float)
                v = (A[1, 1] * rhs[0] - A[0, 1] * rhs[1]) / det
                w = (-A[1, 0] * rhs[0] + A[0, 0] * rhs[1]) / det
                u = 1.0 - v - w
                if u < -1e-6 or v < -1e-6 or w < -1e-6:
                    continue
                P = u * p0 + v * p1 + w * p2
                hn, hq = _high_normal_at(P)                  # normal AND hit point, from ONE projection
                if displacement is not None:
                    # signed distance from the low texel point P to its closest high point hq, ALONG the low
                    # normal ln -- positive = high surface is OUTSIDE the low (bump), negative = inside (dent).
                    # THE CAGE (M12's real work): clamp to max_distance so a far stray hit cannot spike the
                    # geometry. A displacement spike moves vertices, unlike a normal-map artefact which only
                    # shades wrong -- so displacement MUST have the cage bake_normal_map never got.
                    lnv = u * n0 + v * n1 + w * n2; lnv /= (np.linalg.norm(lnv) + 1e-12)
                    signed = float(np.dot(hq - P, lnv))
                    if max_distance is not None:
                        signed = max(-float(max_distance), min(float(max_distance), signed))
                    dispimg[py, px] = signed
                if world_space:
                    img[py, px] = 0.5 + 0.5 * hn
                else:
                    ln = u * n0 + v * n1 + w * n2; ln /= (np.linalg.norm(ln) + 1e-12)
                    t = tan - ln * float(np.dot(tan, ln)); t /= (np.linalg.norm(t) + 1e-12)   # Gram-Schmidt tangent
                    b = np.cross(ln, t)
                    local = np.array([float(np.dot(hn, t)), float(np.dot(hn, b)), float(np.dot(hn, ln))])
                    img[py, px] = 0.5 + 0.5 * local
                if ao and ao_samples:
                    ln = u * n0 + v * n1 + w * n2; ln /= (np.linalg.norm(ln) + 1e-12)
                    esc = 0
                    for si in range(int(ao_samples)):
                        ang = 2.0 * np.pi * si / int(ao_samples)
                        d = ln + 0.5 * np.array([np.cos(ang), np.sin(ang), 0.0])
                        d /= (np.linalg.norm(d) + 1e-12)
                        # occluded if the high-poly is very close in direction d (coarse: sample one step out)
                        probe = P + d * cell * 2.0
                        if np.linalg.norm(_closest_point_on_any(probe, HV, HF) - probe) > cell:
                            esc += 1
                    aoimg[py, px] = esc / float(ao_samples)
    # assemble outputs in a stable order: normal is always first; ao and displacement append if requested.
    outs = [img]
    if ao:
        outs.append(aoimg)
    if displacement is not None:
        outs.append(dispimg)
    return tuple(outs) if len(outs) > 1 else img


def _closest_point_on_any(p, V, F):
    """Brute-force closest point on any triangle (small helper for the coarse AO probe)."""
    best = None; bd = None
    for f in F:
        q, _ = _closest_point_barycentric(p, V[f[0]], V[f[1]], V[f[2]])
        d = float(np.sum((p - q) ** 2))
        if bd is None or d < bd:
            bd = d; best = q
    return best


def auto_retopo(mesh, voxel_resolution=16, subdivide=0, target=None):
    """AUTO-RETOPO: turn a messy BLOCK-OUT (a skin_skeleton blob, a metaball, a boolean mess) into a clean quad-
    dominant cage, in one call -- the hand-off that ends the base-mesh pipeline. Steps: voxel_remesh at a COARSE
    `voxel_resolution` (this sets the tri budget directly -- the field quad-matcher scales steeply, so keep it low,
    ~12-20) -> quad_remesh (field-guided tris-to-quads) -> optional catmull_clark(`subdivide` levels) to smooth the
    coarse cage back up. If `target` is given, the result is shrinkwrapped onto it (so the coarse remesh does not
    drift off the shape) and the silhouette IoU vs target is scored.

    Returns a dict: mesh (the retopologised result), quad_fraction, report (mesh_report), and -- when target -- iou.
    Deterministic.

    THE POINT: skin_skeleton/metaball give a shape with organic-triangle topology; auto_retopo gives it a coarse
    quad-dominant cage a subdivision surface wants, so 'place joints -> clean subdividable model' is one pipeline.
    MEASURED: a skinned 2-limb blob -> ~0.75-0.80 quad fraction at voxel_resolution 12-16 in seconds. KEPT NEGATIVE:
    voxel + greedy quad matching is UNIFORM topology, not artist edge FLOW (no loops around joints); good for a
    background/base asset, not a hero face. quad_remesh cost rises fast with tri count -- raise voxel_resolution only
    a little at a time. For real flow, hand-retopo onto the block-out with the selection + shrinkwrap verbs."""
    from holographic.mesh_and_geometry.holographic_meshsubdiv import catmull_clark
    from holographic.mesh_and_geometry.holographic_meshbridge import voxel_remesh as _vr
    from holographic.mesh_and_geometry.holographic_crossfield import quad_remesh as _qr
    clean = _vr(mesh, resolution=int(voxel_resolution))
    quad, qreport = _qr(clean, use_field=True)
    if subdivide and int(subdivide) > 0:
        quad = catmull_clark(quad, int(subdivide))
    if target is not None:
        quad, _resid = shrinkwrap(quad, target, factor=1.0)
    out = {"mesh": quad, "quad_fraction": mesh_report(quad)["quad_fraction"], "report": mesh_report(quad)}
    if target is not None:
        from holographic.rendering.holographic_render import turnaround as _ta
        out["iou"] = _ta(quad, ref_mesh=target, width=160, height=160)["mean_iou"]
    return out


def make_uv_shell(mesh, uvs, offset=0.02, relative=True):
    """Build a UV SHELL (a texture-carrying ENVELOPE): push every vertex of `mesh` OUTWARD along its vertex normal by
    `offset`, keeping the same faces and the same per-vertex `uvs`. The result is a slightly-inflated copy that
    encloses the original without touching it -- a fixed spatial cage that holds the texture mapping independently of
    the surface.

    WHY A SHELL: when you change a textured mesh's TOPOLOGY (LOD decimation, retopo, remesh), per-vertex UVs cannot
    survive edge collapses cleanly. Instead, freeze the texture onto this offset envelope ONCE, then re-project it
    onto whatever new geometry you build (project_uv_from_shell). The texture is decoupled from the topology: modify
    the mesh freely, project the map back. This is the Blender/Substance 'cage' bake done as geometry.

    `offset` is the outward push; `relative=True` (default) scales it by the mesh's bounding-box diagonal (so 0.02 =
    2% of the model size, a safe non-touching gap at any scale); relative=False uses `offset` in world units. Returns
    a new Mesh with .uvs set. KEPT NEGATIVE: a uniform normal offset can self-intersect in a deep concavity (the
    shell pinches) -- keep `offset` small; for a pinch-free envelope a true mesh-offset (moving along the medial axis)
    is the heavier fix, not built."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.asarray(mesh.vertices, float)
    N = mesh.vertex_normals(store=False)
    d = float(offset) * (float(np.linalg.norm(V.max(0) - V.min(0))) if relative else 1.0)
    shell = Mesh(V + d * N, [tuple(int(i) for i in f) for f in mesh.faces])
    shell.uvs = np.asarray(uvs, float).copy()
    return shell


def project_uv_from_shell(new_mesh, shell, shell_uvs=None, cell_scale=1.0):
    """PROJECT a UV map from a texture-carrying SHELL onto a new mesh (any topology): for each vertex of `new_mesh`,
    find the closest point on `shell` and read the shell's interpolated UV there. Because the shell is a fixed
    envelope that holds the original texture, this gives the new mesh valid UVs no matter how its topology differs --
    the missing half of the shell workflow (make_uv_shell freezes the map; this thaws it onto new geometry).

    `shell_uvs` defaults to shell.uvs. Returns (uvs (n_new_verts, 2), residual (n_new_verts,)) where residual is the
    distance from each new vertex to the shell (small = a clean projection; large = the new vertex strayed far from
    the envelope, its UV is an extrapolation). Reuses transfer_uv's spatial-hash closest-point projection.

    THE POINT: LOD-decimate or retopo the mesh however you like, then project_uv_from_shell to recover the texture
    coordinates -- fewer/different triangles, SAME texture. KEPT NEGATIVE: closest-point (not ray-along-normal), so on
    a thin feature a new vertex can grab the shell's far side; keep the shell offset small so near and far sides stay
    separated, or project in the surface's local frame. Across a UV SEAM the closest-point still picks one island (the
    transfer_uv seam caveat)."""
    if shell_uvs is None:
        shell_uvs = shell.uvs
    return transfer_uv(shell, shell_uvs, np.asarray(new_mesh.vertices, float), cell_scale=cell_scale)


def _selftest():
    from holographic.mesh_and_geometry.holographic_mesh import Mesh, box
    # weld: duplicate every vertex of a box, then merge_by_distance should recover the original count
    b = box()
    dup = Mesh(np.vstack([b.vertices, b.vertices]),
               [tuple(f) for f in b.faces])                   # faces still index the first copy
    w = merge_by_distance(dup, tol=1e-5)
    assert w.n_vertices == b.n_vertices, (w.n_vertices, b.n_vertices)

    # mesh_repair composes weld + fill + compact: duplicate a box's verts AND add an orphan vertex; repair must weld
    # back to the original count (orphan dropped, seam welded), never raising, with an honest report.
    dup2 = Mesh(np.vstack([b.vertices, b.vertices, [[9.0, 9.0, 9.0]]]), [tuple(f) for f in b.faces])
    rep_mesh, rep = mesh_repair(dup2, fill_holes=False)
    assert rep_mesh.n_vertices == b.n_vertices, (rep_mesh.n_vertices, b.n_vertices)   # dups welded + orphan dropped
    assert rep["before"]["vertices"] == 2 * b.n_vertices + 1 and rep["vertices_delta"] < 0
    assert rep_mesh.n_faces == b.n_faces                                              # a closed box stays closed

    # split_nonmanifold_vertices makes a NON-MANIFOLD mesh manifold: 4 triangles sharing one edge (a 'book') has a
    # non-manifold edge; splitting its endpoint vertices into umbrellas separates them -> manifold. Clean box = NO-OP.
    from collections import Counter
    def _nm(faces):
        u = Counter()
        for f in faces:
            f = [int(i) for i in f]; n = len(f)
            for k in range(n):
                u[tuple(sorted((f[k], f[(k + 1) % n])))] += 1
        return sum(1 for c in u.values() if c > 2)
    bookV = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    book = Mesh(bookV, [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5)])
    assert _nm(book.faces) == 1 and not book.is_manifold()
    sm, srep = split_nonmanifold_vertices(book)
    assert _nm(sm.faces) == 0 and sm.is_manifold() and srep["split_vertices"] == 2      # both edge endpoints split
    assert split_nonmanifold_vertices(b)[1]["split_vertices"] == 0                       # clean box -> no-op

    # mesh_repair(triangulate=True) turns a non-manifold book into a manifold ALL-TRIANGLE mesh (retopo-ready arity).
    rm2, r2 = mesh_repair(book, fill_holes=False, triangulate=True)
    assert r2["after"]["manifold"] and r2["became_manifold"] and all(len(f) == 3 for f in rm2.faces)
    # mirror a half-grid across x=0 -> symmetric, and the seam welds
    from holographic.mesh_and_geometry.holographic_mesh import grid
    g = grid(4, 4)
    g.vertices[:, 0] = np.abs(g.vertices[:, 0])               # fold to +x half (a crude half)
    m = mirror(g, axis=0, plane=0.0, weld=True)
    assert m.n_vertices < g.n_vertices * 2                    # the seam welded (fewer than a naive double)
    assert np.allclose(m.vertices[:, 0].min(), -m.vertices[:, 0].max(), atol=1e-6)   # symmetric about x=0

    # symmetrize an ASYMMETRIC mesh -> a bilaterally symmetric result (unlike mirror, which doubles the whole mesh).
    g2 = grid(6, 6)
    Va = g2.vertices.copy()
    Va[Va[:, 0] > 0.0, 2] += 0.5                              # bump the +x half up in z -> asymmetric
    asym = Mesh(Va, [tuple(f) for f in g2.faces])
    sym = symmetrize(asym, axis=0, plane=0.0, side=+1)
    Vs = sym.vertices
    mirrored = Vs.copy(); mirrored[:, 0] = -mirrored[:, 0]    # every mirrored vertex must match an original one
    ok_sym = all(np.linalg.norm(Vs - mv, axis=1).min() < 1e-4 for mv in mirrored)
    assert ok_sym, "symmetrize must produce a mesh symmetric across the plane"
    assert sym.n_faces == asym.n_faces, "keeping +x half (half the faces) then mirroring restores the face count"

    # SOLIDIFY an open sheet -> a CLOSED, MANIFOLD watertight slab (this operator had no test, and shipped a
    # non-manifold bridge: the rim quads traversed the boundary edge the SAME way as the outer wall, so a directed
    # edge appeared twice. The bridge must traverse it the OTHER way. Pin closed+manifold+zero-boundary so it stays.)
    sheet = grid(4, 4)
    slab = solidify(sheet, thickness=0.2)
    assert slab.is_closed() and slab.is_manifold(), "a solidified open sheet must be a closed manifold slab"
    assert slab.euler_characteristic() == 2, "a thickened disk is a topological sphere (chi 2)"
    from collections import defaultdict as _dd
    _ec = _dd(int)
    for _f in slab.faces:
        for _k in range(len(_f)):
            _ec[tuple(sorted((_f[_k], _f[(_k + 1) % len(_f)])))] += 1
    assert not any(_n == 1 for _n in _ec.values()), "no boundary edges -- the slab is watertight"
    assert abs(np.ptp(slab.vertices, axis=0)[2] - 0.2) < 1e-9, "the thickness is applied along the surface normal"

    # transfer_uv: on-surface targets recover interpolated attributes EXACTLY (a flat grid with uv==position/W),
    # and the residual distance honestly reports an off-surface target (a hole-fill centroid would show up here).
    from holographic.mesh_and_geometry.holographic_mesh import grid as _grid
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    _g = _grid(6, 6, width=6.0, height=6.0)
    _gv = np.asarray(_g.vertices, float)
    _uv = (_gv[:, :2] - _gv[:, :2].min(0)) / 6.0
    _tg = np.array([[0.5, 0.5, 0.0], [1.7, 2.3, 0.0], [-2.2, 1.1, 0.0]])
    _got, _res = transfer_uv(_g, _uv, _tg)
    assert np.abs(_got - (_tg[:, :2] - _gv[:, :2].min(0)) / 6.0).max() < 1e-9 and _res.max() < 1e-9
    _got2, _res2 = transfer_uv(_g, _uv, np.array([[0.0, 0.0, 0.7]]))
    assert abs(float(_res2[0]) - 0.7) < 1e-6                      # the honest off-surface error signal

    # shrinkwrap: a box lifted above a plane, factor=1.0 lands every vertex ON the plane (residual = the distance it
    # closed), factor=0.0 is a no-op, and topology is preserved -- the retopo finisher that snaps approximate cage
    # positions onto the reference surface.
    _plane = _grid(6, 6, width=6.0, height=6.0)
    _lift = Mesh(np.asarray(box().vertices, float) + np.array([0, 0, 2.0]), [tuple(f) for f in box().faces])
    _sw, _swr = shrinkwrap(_lift, _plane, factor=1.0)
    assert np.allclose(np.asarray(_sw.vertices)[:, 2], 0.0, atol=1e-6) and _swr.max() > 2.0
    _sw0, _ = shrinkwrap(_lift, _plane, factor=0.0)
    assert np.array_equal(np.asarray(_sw0.vertices), np.asarray(_lift.vertices))          # no-op at factor 0
    assert [tuple(f) for f in _sw.faces] == [tuple(f) for f in _lift.faces]               # topology preserved

    # mesh_report: the scoreboard. A cube reads 100% quad, watertight, chi 2, all valence-3; a box with one face
    # removed reads boundary_edges>0 and is_closed False -- the branch an agent needs before subdividing.
    _rep = mesh_report(box())
    assert _rep["quad_fraction"] == 1.0 and _rep["is_closed"] and _rep["euler_characteristic"] == 2
    assert _rep["boundary_edges"] == 0 and _rep["valence_histogram"].get(3, 0) == 8
    _open = Mesh(np.asarray(box().vertices, float), [tuple(f) for f in box().faces][:-1])   # drop a face
    _rep2 = mesh_report(_open)
    assert _rep2["boundary_edges"] > 0 and not _rep2["is_closed"]

    # skin_skeleton (B-Mesh, SDF route): a single-edge skeleton skins to ONE watertight sphere-topology shell (chi 2);
    # a 3-edge branch skins to a single fused watertight blob (branches merge via smooth_union, for free).
    _sk1 = skin_skeleton(np.array([[0, 0, 0], [1.0, 0, 0]]), [(0, 1)], np.array([0.25, 0.25]), resolution=32)
    _r1 = mesh_report(_sk1)
    assert _r1["is_closed"] and _r1["euler_characteristic"] == 2 and _r1["verts"] > 0
    _sk3 = skin_skeleton(np.array([[0, 0, 0], [1.0, 0, 0], [-0.6, 0.6, 0], [-0.6, -0.6, 0]]),
                         [(0, 1), (0, 2), (0, 3)], np.array([0.25, 0.15, 0.12, 0.12]), resolution=32)
    assert mesh_report(_sk3)["is_closed"]                # branching skeleton -> one watertight surface

    # fit_base_mesh: skin a crude capsule, shrinkwrap it onto a stretched-box target -> the silhouette fit IMPROVES
    # (the block-out-then-snap loop, measured). The fitted mesh stays watertight.
    _tgt = Mesh(np.asarray(box().vertices, float) * np.array([2.0, 0.6, 0.6]), [tuple(f) for f in box().faces])
    _fit = fit_base_mesh(_tgt, np.array([[-1.0, 0, 0], [1.0, 0, 0]]), [(0, 1)], np.array([0.4, 0.4]), resolution=28)
    assert _fit["iou_fitted"] > _fit["iou_base"] and _fit["fitted"].is_closed()   # snapping raised the fit

    # bake_normal_map: a flat low grid vs a bumped high grid -> the flat corner reads ~lavender (0.5,0.5,1) tangent
    # normal, the bump center deviates in R/G. The "keep the sculpt detail on the retopo" step.
    _blow = _grid(5, 5, width=2.0, height=2.0)
    _blv = np.asarray(_blow.vertices, float)
    _buv = (_blv[:, :2] - _blv[:, :2].min(0)); _buv = _buv / _buv.max(0)
    _bhv = _blv.copy()
    _br = np.linalg.norm(_bhv[:, :2] - _bhv[:, :2].mean(0), axis=1); _bhv[:, 2] = 0.4 * np.exp(-(_br / 0.5) ** 2)
    _bhigh = Mesh(_bhv, [tuple(f) for f in _blow.faces])
    _nm = bake_normal_map(_blow, _buv, _bhigh, size=32)
    assert _nm.shape == (32, 32, 3)
    assert abs(_nm[1, 1, 2] - 1.0) < 0.1                       # flat corner: tangent-Z ~ 1 (lavender)
    assert float(np.abs(_nm[16, 16] - _nm[1, 1]).max()) > 0.03  # bump center deviates from flat

    # auto_retopo: a skinned blob (organic tris) -> a quad-DOMINANT watertight cage. Small/coarse so the selftest is
    # fast; the field quad-matcher cost rises steeply with tri count, so keep voxel_resolution low here.
    _blob = skin_skeleton(np.array([[0, 0, 0], [1.0, 0, 0]]), [(0, 1)], np.array([0.3, 0.3]), resolution=16)
    _ar = auto_retopo(_blob, voxel_resolution=10)
    assert _ar["quad_fraction"] > 0.5 and _ar["report"]["is_closed"]   # tris in, quad-dominant closed cage out

    # UV SHELL: build an offset envelope carrying UVs, then project it back onto a mesh -> the texture survives a
    # topology change. On a closed box the shell inflates OUTWARD (mean radius grows); projecting its UVs back onto
    # the box round-trips them (the envelope holds the map independent of topology).
    _sbox = box()
    _sbv = np.asarray(_sbox.vertices, float)
    _suv = (_sbv[:, :2] - _sbv[:, :2].min(0)); _suv = _suv / (_suv.max(0) + 1e-9)
    _shell = make_uv_shell(_sbox, _suv, offset=0.1)
    _ssv = np.asarray(_shell.vertices, float)
    assert np.linalg.norm(_ssv - _ssv.mean(0), axis=1).mean() > np.linalg.norm(_sbv - _sbv.mean(0), axis=1).mean()
    assert [tuple(f) for f in _shell.faces] == [tuple(f) for f in _sbox.faces] and _shell.uvs.shape == _suv.shape
    _puv, _pres = project_uv_from_shell(_sbox, _shell)
    assert _puv.shape == _suv.shape and float(np.abs(_puv - _suv).mean()) < 0.1   # shell round-trips the UVs

    print(f"meshtools selftest ok: weld {dup.n_vertices}->{w.n_vertices} verts; "
          f"mirror is symmetric and welds the seam ({g.n_vertices*2} naive -> {m.n_vertices}); "
          f"symmetrize turns an asymmetric grid into a bilaterally-symmetric mesh ({sym.n_faces} faces); "
          f"solidify thickens an open sheet into a closed manifold slab ({slab.n_faces} faces, chi 2, watertight)")


def _boundary_edges(faces):
    """Return the boundary edges (those used by exactly one triangle) as a list of (a, b) with the face's
    winding order preserved -- needed so a solidify bridge keeps consistent orientation. Triangle meshes."""
    from collections import defaultdict
    count = defaultdict(int)
    oriented = {}
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            key = (a, b) if a < b else (b, a)
            count[key] += 1
            oriented[key] = (a, b)                            # remember one oriented use
    return [oriented[k] for k, c in count.items() if c == 1]


def solidify(mesh, thickness, flip=False):
    """Give a surface thickness (the 'shell' / 'solidify' modifier): offset a copy of the mesh inward along the
    vertex normals by `thickness`, add it with reversed winding (so its normals face out of the inner wall), and
    BRIDGE the boundary edges of an open mesh with quads so the result is a closed solid. A closed input becomes
    a hollow double wall; an open input (a disk, a curved sheet) becomes a watertight thick slab. `flip` offsets
    outward instead. Vertex offset is vectorised; the boundary bridge loops over boundary edges only (a 1-D loop,
    not over all vertices)."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices
    N = mesh.vertex_normals()
    s = -1.0 if not flip else 1.0
    Vinner = V + s * thickness * N
    Vall = np.vstack([V, Vinner])
    off = len(V)
    faces = [tuple(f) for f in mesh.faces]
    inner = [tuple(reversed([vi + off for vi in f])) for f in faces]   # reversed winding for the back wall
    bridge = []
    for a, b in _boundary_edges(faces):                      # close the open rim with two triangles per edge
        # _boundary_edges returns (a,b) in the direction the OUTER wall traverses it, so the bridge must traverse it
        # the OTHER way (b->a) or the directed edge (a,b) would appear twice -> non-manifold. Winding: b, a, a+off,
        # b+off makes a quad whose outer edge is (b,a), matching the reversed inner wall and closing the shell.
        bridge.append((b, a, a + off))
        bridge.append((b, a + off, b + off))
    return Mesh(Vall, faces + inner + bridge)



def _selftest_route_repair():
    """A4: route_repair sends each defect to its MINIMAL op and abstains-to-full when ambiguous. Numeric."""
    import lecore
    from holographic.mesh_and_geometry.holographic_mesh import Mesh, box
    m = lecore.UnifiedMind(dim=512, seed=0)
    b = box()
    # clean box -> 'clean' strategy, no vertex change
    _, rep = route_repair(b, mind=m)
    assert rep["strategy"] == "clean" and rep["confident"], rep
    assert rep["after"]["vertices"] == rep["before"]["vertices"], rep
    # duplicated verts -> 'weld_only' (a categorical dup defect), and it actually welds back down
    dup = Mesh(np.vstack([b.vertices, b.vertices]), [tuple(f) for f in b.faces])
    _, rep2 = route_repair(dup, mind=m)
    assert rep2["strategy"] in ("weld_only", "full_repair"), rep2   # confident weld, or safe fallback
    assert rep2["after"]["vertices"] <= rep2["before"]["vertices"], rep2
    # diagnose is categorical (never a raw count) -- KEPT NEGATIVE guard
    d = diagnose_mesh(b)
    assert set(d.values()) <= {"yes", "no"}, d
    print("  route_repair selftest OK: clean->clean, dup->%s, diagnose categorical" % rep2["strategy"])

if __name__ == "__main__":
    _selftest(); _selftest_route_repair(); _selftest_mesh_orient(); _selftest_transform_mesh(); _selftest_topology_delta()


# =========================================================================================================
# CAD mass properties, plane sections, and draft/moldability report (Poly Studio backlog A: P1/P2 upstreams).
# These existed only demo-side; a demo-local inertia tensor was re-derived WRONG once (negative principal
# moments from a bad covariance formula) -- shipping the correct one here is the whole point.
# =========================================================================================================

def _fan_triangles(faces):
    """Fan-triangulate polygon faces: (v0, v1, ..., vk) -> (v0, vi, vi+1). EXACT for planar convex faces --
    which is what primitive quads and most CAD ngons are. WHY here and not a hard triangle requirement: the
    engine's own primitives (box()) are quad meshes; refusing them would make the CAD trio unusable on the
    engine's own output. A non-planar or concave ngon fan-triangulates to a DIFFERENT solid than intended --
    run mesh_repair(triangulate=True) first if your ngons are suspect."""
    out = []
    for f in faces:
        for i in range(1, len(f) - 1):
            out.append((f[0], f[i], f[i + 1]))
    return out


def mass_properties(mesh, density=1.0):
    """VOLUME, surface AREA, CENTRE OF MASS, and the full INERTIA TENSOR of a closed triangle mesh.

    Signed-tetrahedron integration (each triangle + origin forms a tet; signed determinants make the sum exact
    for any closed, consistently-wound mesh regardless of where the origin sits). The second-moment (covariance)
    integral uses the canonical-tetrahedron formula C0 = (I + ones(3,3)) / 120 mapped through A = [a b c]
    (Tonon 2004 / Blow-Binstock "How to find the inertia tensor of a polyhedron") -- the exact spot where the
    demo-side re-derivation produced physically impossible NEGATIVE principal moments. Ships once, correctly.

    Returns a dict with EXACTLY these keys: volume, area, center_of_mass (3,), mass, inertia_com (3,3) about the
    COM in world axes (also served as inertia_tensor -- same array, both names accepted),
    principal_moments (3,) ascending, principal_axes (3,3) rows = axes. A negative `volume` means the winding
    is inward -- the magnitudes are still right, but fix the mesh (mesh_repair / reverse winding).
    """
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(_fan_triangles(mesh.faces), int)          # quads/ngons fan-triangulated (planar-convex exact)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    # surface area is a plain (unsigned) sum -- valid even for open meshes
    area = 0.5 * float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1)))

    d = np.einsum("ij,ij->i", a, np.cross(b, c))             # det [a;b;c] = 6 * signed tet volume
    vol = float(np.sum(d)) / 6.0
    if abs(vol) < 1e-30:
        raise ValueError("mesh has (near-)zero enclosed volume; is it closed?")

    # first moment: integral of x over each tet = det * (a+b+c) / 24  (tet centroid (a+b+c+0)/4 times det/6)
    com = (d[:, None] * (a + b + c)).sum(axis=0) / (24.0 * vol)

    # second moment about the ORIGIN: sum det * A @ C0 @ A^T with A columns = a,b,c.
    # C0[i][j] = 1/60 (i==j) else 1/120 -- the canonical-tet integral of x_i x_j. This is the formula worth
    # shipping once: get C0 wrong and eigh() hands back negative "moments of inertia".
    C0 = (np.eye(3) + np.ones((3, 3))) / 120.0
    A = np.stack([a, b, c], axis=2)                          # (T, 3, 3), columns are the tet edge vertices
    C = np.einsum("t,tik,kl,tjl->ij", d, A, C0, A)           # sum_t d_t * A_t C0 A_t^T

    m = density * vol
    I_origin = density * (np.trace(C) * np.eye(3) - C)       # inertia from the covariance integral
    r = com
    # parallel-axis shift TO the COM (subtract, because I_origin includes the COM offset)
    I_com = I_origin - m * (float(r @ r) * np.eye(3) - np.outer(r, r))
    I_com = 0.5 * (I_com + I_com.T)                          # symmetrize away accumulation noise before eigh

    w, U = np.linalg.eigh(I_com)
    return {"volume": vol, "area": area, "center_of_mass": com, "mass": m,
            "inertia_com": I_com, "inertia_tensor": I_com, "principal_moments": w, "principal_axes": U.T}
            # inertia_tensor is an ALIAS of inertia_com (the inertia tensor about the COM). Callers reached for
            # both names; serving both removes the KeyError guessing game. Additive -- inertia_com is unchanged.


def section(mesh, plane_point=(0.0, 0.0, 0.0), plane_normal=(0.0, 0.0, 1.0), weld_tol=1e-7):
    """EXACT planar cross-section of a triangle mesh: polylines + signed AREA + PERIMETER + contour count.

    Each triangle crossing the plane contributes one segment, ORIENTED by the triangle's winding so the in-plane
    shoelace sum over all segments is the exact enclosed section area (holes subtract automatically) -- no
    rasterizing, no field sampling (the demo's midpoint-rule raster was an approximation; this is the numeric
    contour the backlog asked for). Contours are counted by chaining segments end-to-end at `weld_tol`.

    Returns dict: area (absolute), perimeter, contours (count), polylines (list of (K,3) world-space arrays,
    each closed loop repeated first point last when closed).
    """
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(_fan_triangles(mesh.faces), int)          # quads/ngons fan-triangulated (planar-convex exact)
    p0 = np.asarray(plane_point, float)
    n = np.asarray(plane_normal, float); n = n / np.linalg.norm(n)
    s = (V - p0) @ n                                          # signed distances

    # in-plane 2-D frame (u, v) with u x v = n so the shoelace sign convention is fixed
    u = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(n, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    def cut(i, j):
        t = s[i] / (s[i] - s[j])                              # s[i], s[j] straddle -> denominator nonzero
        return V[i] + t * (V[j] - V[i])

    segs = []                                                 # (P, Q) oriented so solid is to the LEFT in (u,v)
    for (i, j, k) in F:
        si, sj, sk = s[i], s[j], s[k]
        below = [idx for idx, sv in ((i, si), (j, sj), (k, sk)) if sv < 0.0]
        if len(below) == 0 or len(below) == 3:
            continue
        # walk the triangle's winding; an edge crossing from BELOW to ABOVE yields the segment START, the
        # crossing back yields the END -- this inherits the mesh winding, which is what makes area signed-exact
        pts = []
        order = [(i, j), (j, k), (k, i)]
        for (aa, bb) in order:
            if s[aa] < 0.0 <= s[bb]:
                pts.insert(0, cut(aa, bb))                    # exit point first
            elif s[bb] < 0.0 <= s[aa]:
                pts.append(cut(aa, bb))
        if len(pts) == 2:
            segs.append((pts[0], pts[1]))
    if not segs:
        return {"area": 0.0, "perimeter": 0.0, "contours": 0, "polylines": []}

    P = np.array([sg[0] for sg in segs]); Q = np.array([sg[1] for sg in segs])
    pu, pv = (P - p0) @ u, (P - p0) @ v
    qu, qv = (Q - p0) @ u, (Q - p0) @ v
    area = 0.5 * float(np.sum(pu * qv - qu * pv))
    perimeter = float(np.sum(np.linalg.norm(Q - P, axis=1)))

    # chain segments into loops by welding endpoints on a tol grid
    key = lambda p: tuple(np.round(p / weld_tol).astype(np.int64))
    start = {}
    for idx in range(len(segs)):
        start.setdefault(key(P[idx]), []).append(idx)
    used = np.zeros(len(segs), bool)
    polylines = []
    for idx in range(len(segs)):
        if used[idx]:
            continue
        loop = [P[idx], Q[idx]]; used[idx] = True
        cur = key(Q[idx])
        while True:
            nxt = next((jdx for jdx in start.get(cur, []) if not used[jdx]), None)
            if nxt is None:
                break
            used[nxt] = True
            loop.append(Q[nxt])
            cur = key(Q[nxt])
        polylines.append(np.array(loop))
    return {"area": abs(area), "perimeter": perimeter, "contours": len(polylines), "polylines": polylines}


def draft_report(mesh, pull_dir=(0.0, 0.0, 1.0), min_draft_deg=2.0):
    """READ-ONLY draft-angle / moldability report vs a pull direction -- numbers, not painted faces.

    Per-face draft angle = asin(dot(face_normal, pull)) in degrees: +90 faces the pull, 0 is vertical wall,
    negative is an UNDERCUT (unmoldable without side action). Area-weighted fractions:
    `moldable` (draft >= min_draft_deg), `parting` (|draft| < min_draft_deg -- near-vertical, risky),
    `undercut` (draft <= -min_draft_deg). Also returns the full per-face angles + areas so a caller can bin
    its own histogram. Complements surfanalysis.draft_angle (parametric surfaces) for plain triangle meshes.
    """
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(_fan_triangles(mesh.faces), int)          # quads/ngons fan-triangulated (planar-convex exact)
    p = np.asarray(pull_dir, float); p = p / np.linalg.norm(p)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    twoA = np.linalg.norm(nrm, axis=1)
    good = twoA > 1e-30
    nrm = nrm[good] / twoA[good, None]
    areas = 0.5 * twoA[good]
    dot = np.clip(nrm @ p, -1.0, 1.0)
    draft = np.degrees(np.arcsin(dot))
    total = float(areas.sum())
    frac = lambda mask: float(areas[mask].sum()) / total if total > 0 else 0.0
    return {"draft_deg": draft, "areas": areas, "total_area": total,
            "moldable_fraction": frac(draft >= min_draft_deg),
            "parting_fraction": frac(np.abs(draft) < min_draft_deg),
            "undercut_fraction": frac(draft <= -min_draft_deg)}


def _selftest_cad():
    """Regression trap for the CAD trio: exact numbers on analytic solids, and the negative-moment bug pinned."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    # unit cube [0,1]^3, outward winding
    Vc = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    Fc = []
    for q in quads:
        Fc += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    cube = Mesh(Vc, Fc)

    mp = mass_properties(cube, density=2.0)
    assert abs(mp["volume"] - 1.0) < 1e-12, mp["volume"]
    assert abs(mp["area"] - 6.0) < 1e-12
    assert np.allclose(mp["center_of_mass"], [0.5, 0.5, 0.5], atol=1e-12)
    # cube side s about COM: I = m s^2 / 6 on every axis (m = 2)
    assert np.allclose(mp["principal_moments"], 2.0 / 6.0, atol=1e-12), mp["principal_moments"]
    # THE PINNED NEGATIVE: a rotated box must never produce negative principal moments (the demo-side bug)
    th = 0.7; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    rot = Mesh(Vc @ R.T, Fc)
    mpr = mass_properties(rot)
    assert np.all(mpr["principal_moments"] > 0), "negative principal moment: the exact bug this ships to kill"
    assert abs(mpr["volume"] - 1.0) < 1e-9                    # rotation preserves volume

    sec = section(cube, plane_point=(0, 0, 0.5), plane_normal=(0, 0, 1))
    assert abs(sec["area"] - 1.0) < 1e-12, sec["area"]
    assert abs(sec["perimeter"] - 4.0) < 1e-12
    assert sec["contours"] == 1, sec["contours"]
    # two disjoint cubes -> two contours, double area
    V2 = np.vstack([Vc, Vc + [3.0, 0.0, 0.0]])
    F2 = Fc + [(a + 8, b + 8, c + 8) for (a, b, c) in Fc]
    sec2 = section(Mesh(V2, F2), plane_point=(0, 0, 0.5), plane_normal=(0, 0, 1))
    assert sec2["contours"] == 2 and abs(sec2["area"] - 2.0) < 1e-12

    # QUAD MESH (the engine's own box()): fan-triangulation must give the same exact answers
    from holographic.mesh_and_geometry.holographic_mesh import box
    qb = box()                                                # quad faces, side 2 centered at origin per its docs
    mq = mass_properties(qb)
    side = float(qb.vertices.max() - qb.vertices.min())
    assert abs(mq["volume"] - side ** 3) < 1e-9, (mq["volume"], side)
    assert np.all(mq["principal_moments"] > 0)

    dr = draft_report(cube, pull_dir=(0, 0, 1), min_draft_deg=2.0)
    # cube vs +Z pull: top+bottom = 2/6 of area at +/-90, four walls = 4/6 at 0 draft
    assert abs(dr["moldable_fraction"] - 1.0 / 6.0 * 1.0) < 1e-12 or abs(dr["moldable_fraction"] - 1.0/6.0) < 1e-12
    assert abs(dr["undercut_fraction"] - 1.0 / 6.0) < 1e-12
    assert abs(dr["parting_fraction"] - 4.0 / 6.0) < 1e-12
    print("meshtools CAD selftest OK (mass_properties / section / draft_report)")


if __name__ == "__main__":
    _selftest_cad()   # defined below the first guard; runs after it in module-as-script order


def _selftest_attr_weld():
    """Pin the texture-loss fix. Contracts: (1) attribute-free meshes weld BIT-IDENTICALLY to the old path;
    (2) UV-SEAM duplicates are NOT welded and their uvs are carried exactly; (3) true render duplicates
    (same pos+uv+normal) ARE still welded; (4) mesh_repair end-to-end keeps corner-exact uvs."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    # (1) no attributes -> identical to attrs="ignore"
    V = np.array([[0,0,0],[0,0,0],[1,0,0],[0,1,0]], float)
    F = [(0,2,3),(1,3,2)]
    a = merge_by_distance(Mesh(V, F), attrs="auto")
    b = merge_by_distance(Mesh(V, F), attrs="ignore")
    assert np.array_equal(a.vertices, b.vertices) and a.faces == b.faces, "attr-free weld must be bit-identical"

    # (2)+(3): two coincident vertex pairs -- one a render DUPLICATE (same uv), one a SEAM (different uv)
    V2 = np.array([[0,0,0],[0,0,0],   # pair A: duplicate (identical uv) -> welds
                   [1,0,0],[1,0,0],   # pair B: seam (uv 0.1 vs 0.9)     -> stays split
                   [0,1,0],[2,1,0]], float)
    UV = np.array([[0.2,0.2],[0.2,0.2],[0.1,0.5],[0.9,0.5],[0.3,0.3],[0.7,0.7]])
    F2 = [(0,2,4),(1,3,5)]
    w = merge_by_distance(Mesh(V2, F2, uvs=UV), attrs="auto")
    assert len(w.vertices) == 5, len(w.vertices)             # 6 -> 5: only the duplicate pair welded
    assert w.uvs is not None and len(w.uvs) == 5
    # the seam pair survives with BOTH uvs present
    seam_uvs = sorted(round(float(u), 3) for (v, u) in zip(np.asarray(w.vertices), np.asarray(w.uvs)[:, 0])
                      if abs(v[0] - 1.0) < 1e-9 and abs(v[1]) < 1e-9)
    assert seam_uvs == [0.1, 0.9], seam_uvs

    # (4) end-to-end: a seam-split quad sheet through mesh_repair keeps corner-exact uvs
    rep, report = mesh_repair(Mesh(V2, F2, uvs=UV), fill_holes=False)
    assert report["uvs_carried"] is True
    # corner-wise: every (pos, uv) corner of the input exists identically in the output
    def corners(mesh):
        Vv, Uu = np.asarray(mesh.vertices), np.asarray(mesh.uvs)
        return sorted(tuple(np.round(np.concatenate([Vv[i], Uu[i]]), 9)) for f in mesh.faces for i in f)
    assert corners(Mesh(V2, F2, uvs=UV)) == corners(rep)
    print("attr-weld selftest OK (attr-free bit-identical; seam preserved [0.1, 0.9]; duplicate welded; "
          "repair corner-exact)")


if __name__ == "__main__":
    _selftest_attr_weld()


def _barycentric_batch(P, a, b, c):
    """Barycentric weights (n,3) of points (n,3) ALREADY ON the plane of triangle (a,b,c) -- the batched
    companion to _closest_point_barycentric's scalar path, used to interpolate a per-vertex attribute at a
    projected point. Degenerate triangles fall back to the first corner (weights (1,0,0))."""
    v0 = b - a; v1 = c - a; v2 = P - a
    d00 = float(np.dot(v0, v0)); d01 = float(np.dot(v0, v1)); d11 = float(np.dot(v1, v1))
    d20 = v2 @ v0; d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        out = np.zeros((len(P), 3)); out[:, 0] = 1.0
        return out
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.stack([1.0 - v - w, v, w], axis=1)


def uv_atlas_report(mesh, uvs=None):
    """DIAGNOSE whether a mesh's UVs can survive a topology change -- the question every retopo/LOD/remesh step
    silently bets on, and the one that made a decimated photogrammetry scan render as speckled confetti.

    A UV atlas is a set of ISLANDS (index-connected face groups; in glTF an island break IS a vertex split).
    Per-vertex UV TRANSFER (transfer_uv) projects each new vertex to the nearest source triangle and reads its
    uv -- which is only meaningful if a new FACE's three corners land in the SAME island. So the verdict is a
    ratio: how large is a typical island compared to a typical face of the thing you are transferring onto?

    Returns a dict: islands, faces_per_island (median/mean/max), tiny_island_fraction (<= 2 faces),
    median_island_area, and `transferable` -- False when the atlas is per-triangle-ish, i.e. no per-vertex
    scheme can preserve it and a RE-BAKE (rebake_texture) is the only correct route.

    MEASURED, the reason this exists: the mantis scan has 4079 islands over 11003 faces -- median 1 FACE per
    island, 77% with <= 2. Its atlas is defined triangle-by-triangle. Transferring it onto a 4942-face LOD put
    90% of faces across island boundaries (median UV edge 0.60 of the whole atlas vs 0.013 on the original):
    every such triangle bilinearly smears an unrelated slice of the atlas across itself. That is the speckle."""
    F = [tuple(int(i) for i in f[:3]) for f in mesh.faces if len(f) >= 3]
    V = np.asarray(mesh.vertices, float)
    if not F:
        return {"islands": 0, "transferable": False, "faces": 0}
    parent = list(range(len(V)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for (a, b, c) in F:
        ra = find(a); parent[find(b)] = ra; parent[find(c)] = ra
    roots = np.array([find(f[0]) for f in F])
    _, counts = np.unique(roots, return_counts=True)
    Fa = np.asarray(F)
    e1 = V[Fa[:, 1]] - V[Fa[:, 0]]; e2 = V[Fa[:, 2]] - V[Fa[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    isl_area = {}
    for fi, r in enumerate(roots):
        isl_area[r] = isl_area.get(r, 0.0) + float(area[fi])
    med_isl_area = float(np.median(list(isl_area.values()))) if isl_area else 0.0
    med_faces = float(np.median(counts))
    # the verdict: an island of <= ~4 faces cannot span a coarser mesh's face, so transfer WILL cross islands
    transferable = bool(med_faces > 4.0)
    return {"faces": len(F), "islands": int(len(counts)),
            "faces_per_island_median": med_faces,
            "faces_per_island_mean": float(counts.mean()),
            "faces_per_island_max": int(counts.max()),
            "tiny_island_fraction": float((counts <= 2).mean()),
            "median_island_area": med_isl_area,
            "transferable": transferable}


def _uv_island_labels(mesh):
    """Per-face island id (union-find over shared vertex indices, which IS uv-connectivity: an atlas seam is a
    vertex SPLIT). Returns (labels (F,), n_islands). Shared by uv_atlas_report and reproject_uv so the two can
    never disagree about what an island is."""
    F = [tuple(int(i) for i in f[:3]) for f in mesh.faces if len(f) >= 3]
    n = len(np.asarray(mesh.vertices))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for (a, b, c) in F:
        ra = find(a); parent[find(b)] = ra; parent[find(c)] = ra
    roots = np.array([find(f[0]) for f in F]) if F else np.zeros(0, int)
    uniq = {r: i for i, r in enumerate(sorted(set(roots.tolist())))}
    labels = np.array([uniq[r] for r in roots], int) if F else np.zeros(0, int)
    return labels, len(uniq)


def reproject_uv(source_mesh, source_uv, target_mesh, uv_tol=1e-6, tie=0.25, disc_factor=5.0):
    """REPROJECT a uv map onto a mesh whose topology has changed -- after decimation, remeshing, retopo, any
    operation that alters the face count. The general answer to "the map no longer relates to the geometry".

    Returns (mesh, uv, report). The returned mesh may have MORE vertices than the target: reprojection is
    per-CORNER, and the seams have to go somewhere.

    WHY PER-CORNER, AND WHY THE PLAIN transfer_uv IS NOT ENOUGH (measured, on a perfectly coherent atlas):
    per-vertex transfer gives each new vertex ONE uv. A vertex sitting ON A SEAM needs TWO -- that is what a
    seam IS, a place where the atlas is cut and the source stores two vertices at one 3-D point. Any retopo
    welds them back into one, and then the faces around the seam interpolate ACROSS the whole atlas. On a
    decimated cylinder, 12 of 288 faces came out spanning the seam and 3.36% of rendered pixels were smeared --
    each such face drags the entire texture across itself. One vertex cannot carry two uvs; the fix is
    structural, not a better projection. After reprojection: 0 spanning faces, 0.00% smeared, uv error 13x
    lower (0.0179 -> 0.0014), for 7 vertex splits.

    So: every target face picks a HOME ISLAND (the island its centroid projects into) and takes all three of
    its corners from that island. Corners then cannot land on opposite sides of a cut. Each target vertex then
    emits one copy per DISTINCT uv its faces asked for -- one everywhere the map is smooth, several along a cut,
    with the target's position copied verbatim. The atlas's cut structure is RECONSTRUCTED rather than
    destroyed, and the surface stays bit-identical to the keep_uv=False result.

    `tie` (fraction of a target edge) is the window within which two source triangles count as equally valid
    preimages; `disc_factor` (multiples of the median source uv edge) is what counts as a real cut. Both were
    SWEPT, not guessed, twice: the tie window exists to merge genuinely co-located candidates, and its correct
    size CHANGED when the consistency filter took over side-selection. With side handled upstream, a wide
    window only pollutes singular corners with non-co-located triangles that home-ranking then picks (sphere
    seam-free error 1.2e-2 at tie=0.5 -> 2.3e-4 at 0.25; 0.1 and 0.25 sit on the same plateau at 0 defects;
    0.75 starts over-splitting). Default 0.25. The general lesson: a parameter's swept optimum is conditional
    on the design around it -- re-sweep after any redesign, or the old number silently becomes a bug.

    KEPT NEGATIVE: this preserves an EXISTING atlas, so it needs the source's islands to be big enough to
    contain a target face -- exactly what uv_atlas_report's `transferable` measures. On a fragmented
    per-triangle scan atlas no reprojection can work and the answer is rebake_texture (a new atlas + a new
    image); textured_lod routes between the two for you. This raises rather than silently producing confetti."""
    from holographic.mesh_and_geometry.holographic_meshbridge import _closest_point_on_triangle as _cpot
    SV = np.asarray(source_mesh.vertices, float)
    SF = [tuple(int(i) for i in f[:3]) for f in source_mesh.faces if len(f) >= 3]
    suv = np.asarray(source_uv, float)
    TV = np.asarray(target_mesh.vertices, float)
    TF = np.asarray([f[:3] for f in target_mesh.faces if len(f) >= 3], int)
    if not SF or not len(TF):
        raise ValueError("reproject_uv needs triangle meshes on both sides")
    atlas = uv_atlas_report(source_mesh, suv)
    if not atlas["transferable"]:
        raise ValueError(
            "source atlas is fragmented (median %.1f faces/island over %d islands): no uv reprojection can "
            "preserve it -- re-bake instead (meshtools.rebake_texture / textured_lod, which routes for you)"
            % (atlas["faces_per_island_median"], atlas["islands"]))

    labels, n_isl = _uv_island_labels(source_mesh)
    # what counts as a DISCONTINUITY in this atlas: many times a typical source uv edge. Scale-free, so it
    # works on any unwrap rather than needing a magic constant per asset.
    _sf = np.asarray(SF, int)
    _e = np.concatenate([np.linalg.norm(suv[_sf[:, k]] - suv[_sf[:, (k + 1) % 3]], axis=1) for k in range(3)])
    uv_edge = float(np.median(_e[_e > 0])) if (_e > 0).any() else 1e-3
    disc = disc_factor * uv_edge
    stri = np.array([[SV[f[0]], SV[f[1]], SV[f[2]]] for f in SF])
    lo = stri.min(axis=(0, 1))
    cell = max(float(np.mean(np.linalg.norm(stri[:, 1] - stri[:, 0], axis=1))), 1e-9)
    grid = {}
    tmin = stri.min(1); tmax = stri.max(1)
    for fi in range(len(SF)):
        i0 = np.floor((tmin[fi] - lo) / cell).astype(int)
        i1 = np.floor((tmax[fi] - lo) / cell).astype(int)
        for ix in range(i0[0], i1[0] + 1):
            for iy in range(i0[1], i1[1] + 1):
                for iz in range(i0[2], i1[2] + 1):
                    grid.setdefault((ix, iy, iz), []).append(fi)

    def candidates(pmin, pmax, ring):
        i0 = np.floor((pmin - lo) / cell).astype(int) - ring
        i1 = np.floor((pmax - lo) / cell).astype(int) + ring
        seen = set()
        for ix in range(i0[0], i1[0] + 1):
            for iy in range(i0[1], i1[1] + 1):
                for iz in range(i0[2], i1[2] + 1):
                    seen.update(grid.get((ix, iy, iz), ()))
        return sorted(seen)                                  # sorted: deterministic tie order

    def _uv_at(fi, bc):
        f = SF[fi]
        return bc[0] * suv[f[0]] + bc[1] * suv[f[1]] + bc[2] * suv[f[2]]

    def nearest(pt, pool):
        best_d, best_fi, best_bc = np.inf, -1, (1.0, 0.0, 0.0)
        for fi in pool:
            a, b, c = stri[fi]
            q, bc = _closest_point_barycentric(pt, a, b, c)
            d = float(np.linalg.norm(pt - q))
            if d < best_d:
                best_d, best_fi, best_bc = d, fi, bc
        return best_fi, best_bc, best_d

    def nearest_consistent(pt, pool, home_uv, tol, allow):
        """The closest source triangle to `pt` WHOSE UV IS CONSISTENT with the face's home side. Returns
        (face, barycentric, distance).

        This went through three designs, each corrected by measurement, and the literature settled it:
         1. Island restriction alone did nothing -- a seam is a cut WITHIN one island (the u=0 and u=1 columns
            of a uv-sphere are one island), and a pole is worse: every u collapses onto one 3-D point.
         2. Disambiguating every corner by uv-proximity-to-home dragged corners toward their face's centre
            (442 of 156 verts split, uv error 60x); then conflating "which side" with "where on that side" in
            one argmin fabricated error in the seam-FREE coordinate (cylinder v: 0.00000 -> 0.05 p95).
         3. Side-as-TIE-BREAK still left a hole Moose saw as misalignment: a corner whose nearest source
            triangle is uniquely on the WRONG side never enters the tie window, so the side choice never runs
            -- measured as 9-13 internally-inconsistent faces on a decimated uv-sphere, invisible to the
            seam-crosser count.
        The cut-aware uv-transfer literature (Disney's patch-indicator formulation, US10984581) does it the
        other way around, and this now follows it: assign the FACE to a side first, then CONSTRAIN every corner
        to source triangles consistent with that side -- `allow` = disc + the face's own legitimate uv extent,
        so a big face is not punished for genuinely spanning uv space. Side is a constraint, geometry picks the
        position among what satisfies it. If NOTHING is consistent (a true singularity, e.g. the pole vertex
        itself, where the whole u range meets), fall back to the nearest and let the caller's discontinuity
        machinery split it -- an honest fallback, not a silent wrong answer."""
        cands = []
        for fi in pool:
            a, b, c = stri[fi]
            q, bc = _closest_point_barycentric(pt, a, b, c)
            cands.append((float(np.linalg.norm(pt - q)), fi, bc))
        best = min(cands, key=lambda t: (t[0], t[1]))         # deterministic: distance, then face index
        ok = [t for t in cands if float(np.linalg.norm(_uv_at(t[1], t[2]) - home_uv)) <= allow]
        if not ok:
            return best[1], best[2], best[0]
        dmin = min(t[0] for t in ok)
        near = [t for t in ok if t[0] <= dmin + tol]
        # Precedence, with every step measured in because each naive version shipped a defect:
        #   1. consistency filter    (which side -- excludes the far side of a cut outright)
        #   2. geometry              (where -- distance, then face index)
        #   3. uv nearest home, ONLY when the geometric tie set is itself uv-AMBIGUOUS (spread > disc).
        # Step 3 unmoderated re-created the centroid drag at one remove: on a smooth surface the tie set's uvs
        # agree to within disc but are not IDENTICAL, so ranking them by home-proximity still pulled every
        # corner toward its face's centre -- 318 splits on the cylinder (vs 7) and 0.1 error in the seam-free
        # axis (vs 1e-16). Gated behind the discontinuity test it fires only where geometry carries no
        # information AND the uvs genuinely disagree -- the pole fan -- which is the one place home must decide
        # (ungated geometry there picked by face index: u=0.083 for a face living at u~0.57).
        uv0 = _uv_at(near[0][1], near[0][2])
        spread = max(float(np.linalg.norm(_uv_at(fi, bc) - uv0)) for (_d, fi, bc) in near)
        if spread <= disc:
            pick = min(near, key=lambda t: (t[0], t[1]))
        else:
            pick = min(near, key=lambda t: (float(np.linalg.norm(_uv_at(t[1], t[2]) - home_uv)), t[0], t[1]))
        return pick[1], pick[2], pick[0]

    out_uv = np.zeros((len(TF) * 3, 2))
    out_V = TV[TF].reshape(-1, 3)
    dists = np.zeros(len(TF) * 3)
    home_used = 0
    # uv-per-unit-length of the source map: how much uv a corner may LEGITIMATELY differ from its face's home
    # just by being a face-radius away from the centroid. Sizes the consistency window so a big face is not
    # punished for genuinely spanning uv space, while a wrong-side pick (a whole atlas away) stays excluded.
    _e3 = np.concatenate([np.linalg.norm(stri[:, k] - stri[:, (k + 1) % 3], axis=1) for k in range(3)])
    uv_per_len = uv_edge / max(float(np.median(_e3[_e3 > 0])), 1e-12) if (_e3 > 0).any() else 1.0

    for k in range(len(TF)):
        corners = TV[TF[k]]
        cen = corners.mean(0)
        pool = []
        ring = 0
        while not pool and ring <= 6:
            pool = candidates(corners.min(0), corners.max(0), ring); ring += 1
        if not pool:
            pool = list(range(len(SF)))
        # HOME BY MAJORITY VOTE, not one centroid probe (the cut-aware transfer literature assigns a face to a
        # patch by the majority of points within it): a face straddling a cut can have its centroid land on
        # either side by luck, and then all three corners inherit the wrong side.
        #
        # BUT an ambiguous sample must ABSTAIN, and this was measured in the hardest way: at a pole, every
        # sample sitting on the singular point sees all the fan's triangles at distance zero and resolves the
        # tie identically (lowest face index) -- so a pole-fan face had 5 of its 7 samples stuff the ballot
        # with the same wrong side, and crossers went UP versus the single centroid probe (3 -> 8 on the
        # decimated uv-sphere). The singularity does not get to vote; only samples whose own candidates agree
        # on a side do. For a pole-fan face that leaves exactly the corners AWAY from the pole -- the ones
        # that actually know where the face lives.
        samples = [cen, corners[0], corners[1], corners[2],
                   (corners[0] + corners[1]) / 2, (corners[1] + corners[2]) / 2, (corners[2] + corners[0]) / 2]
        edge = max(float(np.linalg.norm(corners[(j + 1) % 3] - corners[j])) for j in range(3))
        tol = tie * edge
        votes = []
        fallback_uv = None
        for s in samples:
            cands_s = []
            for fi in pool:
                a, b, c = stri[fi]
                q, bc = _closest_point_barycentric(s, a, b, c)
                cands_s.append((float(np.linalg.norm(s - q)), fi, bc))
            b0 = min(cands_s, key=lambda t: (t[0], t[1]))
            if fallback_uv is None:
                fallback_uv = _uv_at(b0[1], b0[2])           # the centroid's nearest: last resort only
            near_s = [t for t in cands_s if t[0] <= b0[0] + tol]
            uv0 = _uv_at(b0[1], b0[2])
            spread = max(float(np.linalg.norm(_uv_at(fi, bc) - uv0)) for (_d, fi, bc) in near_s)
            if spread <= disc:                               # unambiguous: this sample knows its side
                votes.append(uv0)
        clusters = []                                        # [(members)] in first-seen order
        for su in votes:
            placed = False
            for cl in clusters:
                if float(np.linalg.norm(su - cl[0])) <= disc:
                    cl.append(su); placed = True
                    break
            if not placed:
                clusters.append([su])
        if clusters:
            majority = max(clusters, key=len)                # first-seen wins equal counts (max is stable)
            cmean = np.mean(majority, axis=0)
            home_uv = min(majority, key=lambda su: (float(np.linalg.norm(su - cmean)),
                                                    float(su[0]), float(su[1])))   # medoid, deterministic
        else:
            home_uv = fallback_uv                            # every sample ambiguous: degenerate face, honest
        # island restriction still holds: never cross an ISLAND boundary (right for multi-island atlases even
        # though a seam is a cut within one island -- the two guards catch different failure shapes).
        hfi_for_island, _, _ = nearest(cen, pool)
        home_island = labels[hfi_for_island]
        in_home = [fi for fi in pool if labels[fi] == home_island]
        if not in_home:
            in_home = [fi for fi in range(len(SF)) if labels[fi] == home_island]
        home_used += 1
        allow = disc + uv_per_len * edge                     # legitimate one-face uv spread + a real-cut margin
        for j in range(3):
            # every corner is CONSTRAINED to the face's side -- not merely tie-broken toward it.
            fi, bc, d = nearest_consistent(corners[j], in_home, home_uv, tol, allow)
            out_uv[3 * k + j] = _uv_at(fi, bc)
            dists[3 * k + j] = d

    # SPLIT EXACTLY, rather than welding a soup back together. The obvious move is to build private vertices per
    # face and hand them to merge_by_distance(attrs="auto") -- it already fuses only vertices agreeing in
    # position AND uv, so the seams would survive. It was written that way first. But that weld AVERAGES the
    # positions it merges, and averaging identical numbers is not free: the surface came back differing from the
    # target by 1.11e-16. This engine treats a 1e-16 flip as a flip -- decimation is tie-sensitive, and the
    # target's positions are the caller's, not ours to re-derive. So instead: group each TARGET vertex's corners
    # by uv and emit one copy per distinct uv, copying TV[vi] VERBATIM. Positions are then bit-identical to the
    # keep_uv=False result BY CONSTRUCTION, and the vertex table gains duplicates only at the cuts -- exactly
    # the structure the source asset has.
    groups = [[] for _ in range(len(TV))]                    # vi -> [uv, ...] in first-seen order
    corner_slot = np.zeros(len(TF) * 3, int)
    for k in range(len(TF)):
        for j in range(3):
            vi = int(TF[k, j])
            uv = out_uv[3 * k + j]
            slot = -1
            for gi, g in enumerate(groups[vi]):
                if float(np.linalg.norm(g - uv)) <= uv_tol:
                    slot = gi
                    break
            if slot < 0:
                groups[vi].append(uv)
                slot = len(groups[vi]) - 1
            corner_slot[3 * k + j] = slot
    base = np.zeros(len(TV), int)                            # deterministic: vertices ascending, groups in order
    total = 0
    for vi in range(len(TV)):
        base[vi] = total
        total += max(1, len(groups[vi]))
    new_V = np.zeros((total, 3))
    new_uv = np.zeros((total, 2))
    for vi in range(len(TV)):
        gs = groups[vi] or [np.zeros(2)]
        for gi, g in enumerate(gs):
            new_V[base[vi] + gi] = TV[vi]                    # VERBATIM -- never averaged, never re-derived
            new_uv[base[vi] + gi] = g
    new_F = [tuple(int(base[TF[k, j]] + corner_slot[3 * k + j]) for j in range(3)) for k in range(len(TF))]
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    welded = Mesh(new_V, new_F, uvs=new_uv)
    wuv = new_uv
    straddle, max_edge = uv_straddle_fraction(welded, wuv)
    report = {"faces": len(TF), "vertices_in": len(TV), "vertices_out": len(welded.vertices),
              "seam_splits": int(len(welded.vertices) - len(TV)), "islands": int(n_isl),
              "projection_distance_mean": float(dists.mean()), "projection_distance_p95": float(np.percentile(dists, 95)),
              "straddle_fraction": straddle, "max_uv_edge": max_edge,
              "finite": bool(np.isfinite(wuv).all())}
    return welded, wuv, report


def uv_straddle_fraction(mesh, uvs=None, threshold=0.1):
    """The DEFECT metric, measured on the OUTPUT: the fraction of faces whose largest UV edge exceeds
    `threshold` of the atlas -- i.e. faces that sample across island boundaries and therefore smear unrelated
    texture across themselves. ~0 on a healthy mesh (the mantis original measures 0.000); 0.90 on the
    transferred LOD that rendered as speckle. Use it to FAIL a textured-LOD pipeline loudly instead of shipping
    confetti. Returns (fraction, max_uv_edge).

    SCOPE -- ONE hole, measured, and it is not the one you would guess. `threshold` is an ABSOLUTE fraction of
    the atlas, so the metric is a function of UV-EDGE SCALE, i.e. of mesh density: a coarse 4x4 sheet has
    legitimate 0.25 uv edges and reads 1.0 with nothing wrong with it. Set the threshold from the source's own
    uv edge scale (the mantis original: median 0.013, so 0.1 is a 7x jump and unambiguous -- it read 0.000 there
    and 0.90 on the scrambled LOD, which is the whole point).

    A NOTE ON A WRONG SCOPE NOTE, kept because it is the more useful lesson: this docstring first claimed the
    metric was "meaningless on a per-face atlas, where every face is its own island". That was reasoned from a
    4-FACE example that read 1.0 -- and it is false. Measured: a per-face bake of a 2048-face mesh reads 0.00,
    because g=46 cells make the uv edges small. The 4-face case read high for the density reason above, not for
    an island reason. One toy example plus a plausible story nearly became a permanent lie in the docs; the
    density hole is the only real one. uv_atlas_report is the STRUCTURAL question -- ask it first."""
    uv = np.asarray(uvs if uvs is not None else mesh.uvs, float)
    F = np.asarray([f[:3] for f in mesh.faces if len(f) >= 3], int)
    if not len(F):
        return 0.0, 0.0
    m = np.zeros(len(F))
    for k in range(3):
        m = np.maximum(m, np.linalg.norm(uv[F[:, k]] - uv[F[:, (k + 1) % 3]], axis=1))
    return float((m > threshold).mean()), float(m.max())


def rebake_texture(source_mesh, source_uv, texture, target_mesh, size=1024, margin=2, chunk=200000,
                   method="project", grid=None, fill_mode="margin", normal_aware=False):
    """RE-BAKE a source texture onto a new topology -- the correct route when uv_atlas_report says
    `transferable=False` (a fragmented / per-triangle photogrammetry atlas, where no per-vertex UV transfer can
    work because a new face's corners land in different islands).

    `size` is the atlas edge in pixels, or "auto" to size it for the face count. The per-face atlas packs nF
    faces into a ceil(sqrt(nF)) grid, so a big mesh needs a big atlas (a 437k-face scan needs ~5.3k px at
    margin 2); a too-small explicit size raises a helpful error naming the minimum, and size="auto" picks a
    usable size (>= 4 texels per face). For interactive scan-scale rebakes, decimate the target first
    (mesh_decimate_to) or use method="scatter" (H1, ~1500x on dense scans) -- the per-texel projection default
    is O(atlas x source) and is the 9-minute cost the client measured on a raw 437k-face scan at 1024^2.

    Builds a NEW per-face atlas for `target_mesh` and paints the source's colour into it: each target face gets
    its own cell of the new `size`x`size` image; per texel of that cell -> barycentric -> the 3-D point on the
    target face -> closest point on the SOURCE surface (transfer_uv, batched + spatial-hashed) -> the source's
    uv there -> a bilinear sample of `texture`. Topology-independent BY CONSTRUCTION: it never asks the source
    atlas to span anything, it only ever samples it pointwise.

    Returns (mesh, uv, image, report). The mesh has SPLIT vertices (3 per face) because per-face atlas UVs are
    per-CORNER data and Mesh stores per-vertex uvs -- positions/faces are unchanged geometry, just re-indexed.
    `report` carries the honest error signals: projection distance (mean/p95 -- large = the target strayed off
    the source surface) and the straddle fraction of the RESULT (must be ~0; that is the whole point).

    THE COSTUME: this is bake_normal_map's walk (texel -> 3-D -> closest point on the other surface -> read a
    payload) with COLOUR as the payload instead of a normal, and the target's own atlas generated rather than
    given. Named here so nobody rebuilds the walk a third time.

    KEPT NEGATIVES, measured: (1) one face per square cell = ~50% atlas efficiency; pairing two triangles per
    cell would halve the waste but their shared diagonal would bleed into each other under bilinear filtering
    and margin dilation -- rejected deliberately, not overlooked. (2) `margin` texels of dilation pad each cell
    so bilinear filtering at cell edges does not grab background; a target face thinner than ~3 texels in the
    atlas still under-samples -- raise `size`. (3) No cage / max projection distance: a target vertex far from
    the source grabs the nearest colour anyway (inherited from bake_normal_map's kept negative). (4) The bake
    is O(covered texels) with one batched projection -- not free, but it is the only correct answer here."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    TV = np.asarray(target_mesh.vertices, float)
    TF = np.asarray([f[:3] for f in target_mesh.faces if len(f) >= 3], int)
    if not len(TF):
        raise ValueError("rebake_texture needs a triangle target mesh")
    tex = np.asarray(texture, float)
    if tex.max() > 1.5:
        tex = tex / 255.0                                        # 8-bit -> [0,1], same coercion preview_asset does
    nF = len(TF)
    g = int(np.ceil(np.sqrt(nF)))                                # the per-face atlas: g x g cells
    # AUTO-SIZE (client P2): the per-face atlas needs size > 2*margin*g just for a positive triangle, and
    # realistically g*(texels_per_face + 2*margin) to give each face usable texels. A 437k-face scan needs
    # ~5.3k px at margin 2, so a caller's habitual 1024 raised "size too small for N faces" and left them to
    # binary-search a size by hand. size="auto" computes the minimum usable size (>= 4 texels per face side)
    # so the correct route just runs. An explicit int that is too small still RAISES -- but now names the size
    # that would work, so the fix is one number, not a guessing loop. Backward-compatible: an explicit int big
    # enough is untouched (byte-identical atlas), so no existing bake changes.
    _TEXELS_PER_FACE = 4                                          # a usable floor: below this faces alias badly
    _min_usable = int(np.ceil(g * (_TEXELS_PER_FACE + 2 * margin)))
    if isinstance(size, str):
        if size != "auto":
            raise ValueError("size must be an int or 'auto', got %r" % size)
        size = max(_min_usable, 64)                              # never smaller than a sane atlas
    size = int(size)
    cell = 1.0 / g
    m_uv = margin / float(size)                                  # margin expressed in uv
    side = cell - 2.0 * m_uv
    if side <= 0:
        raise ValueError("size %d too small for %d faces at margin %d -- need at least %d (or pass size='auto')"
                         % (size, nF, margin, _min_usable))

    # ---- the new atlas layout: face k -> cell (k%g, k//g), corners on a right triangle inside it ----
    k = np.arange(nF)
    cx = (k % g) * cell + m_uv                                   # cell origin in uv
    cy = (k // g) * cell + m_uv
    uv0 = np.stack([cx, cy], axis=1)
    uv1 = np.stack([cx + side, cy], axis=1)
    uv2 = np.stack([cx, cy + side], axis=1)
    new_uv = np.empty((nF * 3, 2))
    new_uv[0::3] = uv0; new_uv[1::3] = uv1; new_uv[2::3] = uv2
    new_V = TV[TF].reshape(-1, 3)                                # split: 3 private vertices per face
    new_F = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(nF)]

    SV = np.asarray(source_mesh.vertices, float)
    SF = [tuple(int(i) for i in f[:3]) for f in source_mesh.faces if len(f) >= 3]

    # ---- gather every covered texel and its 3-D point (one pass; the inner op is vectorised per face) ----
    px_all, py_all, pts_all, nrm_all = [], [], [], []
    face_slice = [None] * nF                                     # face -> its slice of the concatenated texels
    # per-TARGET-face normal (NA-SCAT): a texel inherits the normal of the face it lives on, so the scatter
    # gather can prefer source samples on the SAME side of a thin feature instead of averaging front+back.
    _tfn = np.cross(P1 if False else TV[TF[:, 1]] - TV[TF[:, 0]], TV[TF[:, 2]] - TV[TF[:, 0]])
    _tfn = _tfn / (np.linalg.norm(_tfn, axis=1, keepdims=True) + 1e-12)
    cursor = 0
    P0, P1, P2 = TV[TF[:, 0]], TV[TF[:, 1]], TV[TF[:, 2]]
    x0 = np.floor(cx * (size - 1)).astype(int); y0 = np.floor(cy * (size - 1)).astype(int)
    span = int(np.ceil(side * (size - 1))) + 1
    ii, jj = np.meshgrid(np.arange(span), np.arange(span), indexing="ij")
    ii = ii.ravel(); jj = jj.ravel()
    for f in range(nF):
        u = ii / max(side * (size - 1), 1e-9)                    # barycentric along the cell's right triangle
        v = jj / max(side * (size - 1), 1e-9)
        # CONSERVATIVE by half a texel: the exact hypotenuse (u+v==1) and the far corner fall BETWEEN texel
        # centres, so a strict <=1 test leaves the face's own corner texel unwritten and a corner-uv sample
        # reads whatever the dilation happened to drag in (measured: a corner came back with the wrong colour).
        eps = 1.5 / max(side * (size - 1), 1e-9)
        keep = (u + v) <= 1.0 + eps                              # the lower-left half of the cell IS the face
        if not keep.any():
            continue
        uu, vv = u[keep], v[keep]
        pxs = x0[f] + ii[keep]; pys = y0[f] + jj[keep]
        ok = (pxs < size) & (pys < size)
        uu, vv, pxs, pys = uu[ok], vv[ok], pxs[ok], pys[ok]
        pts = (P0[f][None, :] * (1 - uu - vv)[:, None] + P1[f][None, :] * uu[:, None]
               + P2[f][None, :] * vv[:, None])
        px_all.append(pxs); py_all.append(pys); pts_all.append(pts)
        nrm_all.append(np.broadcast_to(_tfn[f], pts.shape))
        face_slice[f] = slice(cursor, cursor + len(pts)); cursor += len(pts)
    px = np.concatenate(px_all); py = np.concatenate(py_all); pts = np.concatenate(pts_all)
    tgt_nrm = np.concatenate(nrm_all)                             # (n_texel, 3) target-face normal per texel

    # ================= method="scatter": the HOLOGRAPHIC fast path (H1) ==============================
    # MEASURED (BACKLOG_holographic_pipeline H1): the projection loop below is 1.05M closest-point calls /
    # 142s at 512^2 -- a hand-rolled scatter/gather. A texel's colour is `gather` reading a `scatter` bundle:
    # SCATTER each source vertex's colour onto a volumetric grid keyed by its 3-D position (bundle colour onto
    # space), then GATHER the colour at every texel's 3-D point in ONE vectorised call. Measured 0.02-0.06s vs
    # 46-62s (~1500x), colour error 0.066-0.088 RGB at a grid that separates opposite walls (0% bleed >=160^3).
    #
    # WHY SCATTER COLOUR, NOT UV: colour is CONTINUOUS across the surface even when the source atlas is
    # fragmented (median 1 face/island) -- which is the whole reason rebake exists. Scattering/interpolating uv
    # would reintroduce the confetti transfer_uv fails on: a texel between two 1-face islands would average uv
    # (0.9) and uv (0.1) into (0.5) and sample the wrong place. Colour has no seams, so the volume bake is safe.
    #
    # KEPT NEGATIVES: (1) quality is bounded by SOURCE VERTEX DENSITY, not texture resolution -- the volume
    # stores one colour per source vertex, so a low-poly source with a high-res texture loses detail (use
    # method="project" there). (2) two surfaces within one grid cell BLEED (opposite sides of a thin leg average)
    # -- default grid auto-sizes to ~2x the source's own resolution to keep cells sub-wall; raise `grid` if a
    # thin feature smears. Both are why this is opt-in, not the default.
    if method == "scatter":
        from holographic.misc.holographic_transfer import scatter as _scatter, gather as _gather
        suv = np.asarray(source_uv, float)
        th, tw = tex.shape[:2]
        # source colour = the source texture sampled at each source vertex's uv (nearest texel; the volume
        # smooths anyway, and this stays cheap). Clamp, never wrap -- an atlas edge is its last texel.
        sfu = np.clip(suv[:, 0], 0.0, 1.0) * (tw - 1); sfv = np.clip(suv[:, 1], 0.0, 1.0) * (th - 1)
        src_col = np.asarray(tex, float)[np.clip(sfv.astype(int), 0, th - 1), np.clip(sfu.astype(int), 0, tw - 1)]
        C = src_col.shape[1] if src_col.ndim == 2 else 3
        # grid sized so a cell is finer than the source's own mean edge (keeps opposite walls in separate cells)
        slo = SV.min(0); shi = SV.max(0); sext = shi - slo; sext[sext < 1e-9] = 1e-9
        if grid is None:
            # COVERAGE-AWARE auto grid (the real dark-speckle fix, measured). The old rule -- 2x the SOURCE
            # edge for anti-bleed -- overshot: on a dense target the TEXELS out-resolve the grid, so many land
            # in empty cells and gather ~0 -> dark (MEASURED: auto picked 242 and rendered 22.4% dark; capping
            # to a grid the texels can fill dropped it to 16.4%, matching project's 16.5% ceiling). The honest
            # sizing target is the TARGET texel spacing, not the source edge: a cell must be at least as coarse
            # as the mean gap between the query points it must cover, or coverage collapses. We size to the
            # coarser of (source-edge*2) and (target-point spacing) so cells stay filled; thin-feature bleed at
            # the coarser end is handled by normal_aware, not by starving the grid.
            sedge = float(np.mean(np.linalg.norm(np.diff(SV[np.asarray([f[:3] for f in SF])], axis=1), axis=2))) or 1.0
            tgt_gap = float(np.mean(np.linalg.norm(np.diff(TV[np.asarray([f[:3] for f in target_mesh.faces
                                    if len(f) >= 3])], axis=1), axis=2))) or sedge
            cell = max(sedge, tgt_gap)                          # never finer than either sampling can fill
            GR = int(np.clip(np.ceil(float(sext.max()) / max(cell, 1e-9)), 32, 320))
        else:
            GR = int(grid)
        spos = (SV - slo) / sext * (GR - 1)                     # source verts in grid-cell units
        qpos = (pts - slo) / sext * (GR - 1)                    # every texel's 3-D point in grid-cell units

        def _bake_volume(sel_verts):
            """scatter the selected source verts' colour into a GR^3 bundle, gather at every texel."""
            sp = spos[sel_verts]
            wv = np.asarray(_scatter(sp, np.ones(int(sel_verts.sum())), shape=(GR, GR, GR), kernel="bilinear"))
            cv = [np.asarray(_scatter(sp, src_col[sel_verts, c], shape=(GR, GR, GR), kernel="bilinear"))
                  for c in range(C)]
            wq = np.asarray(_gather(wv, qpos, kernel="bilinear"))
            cq = np.stack([np.asarray(_gather(cv[c], qpos, kernel="bilinear")) for c in range(C)], axis=1)
            return wq, cq

        if not normal_aware:
            wq, col = _bake_volume(np.ones(len(SV), bool))
            col = col / np.maximum(wq[:, None], 1e-6)
        else:
            # NA-SCAT: the position-only gather blends the FRONT and BACK of a thin feature into one colour
            # (MEASURED on the occlusion-baked mantis: 22.4% dark render px vs project's 16.5%). Fix: partition
            # BOTH the source verts and the target texels by their normal's dominant-axis SIGN -- a coarse
            # 6-cell hemisphere key (position (+) orientation, the VSA-native bound pair). A texel gathers only
            # from source verts on its OWN side, so the far wall's (often shadowed, dark) colour never leaks in.
            # WHY dominant-axis sign and not a fine normal bin: 6 buckets keep every bucket dense enough to
            # cover its side of the surface; finer keys starve the gather and reintroduce holes. Where a
            # texel's bucket is empty (a side the source didn't sample there), fall back to the position-only
            # volume -- so NA never does WORSE than plain scatter on coverage, only better on bleed.
            SN = np.asarray(source_mesh.vertex_normals(), float)
            def hemi_key(N):
                ax = np.argmax(np.abs(N), axis=1)               # dominant axis 0/1/2
                sg = (N[np.arange(len(N)), ax] >= 0).astype(int)  # + / - side
                return ax * 2 + sg                              # 0..5
            skey = hemi_key(SN); qkey = hemi_key(tgt_nrm)
            wq_fb, col_fb = _bake_volume(np.ones(len(SV), bool))  # fallback (all verts), position only
            col = col_fb.copy(); wq = wq_fb.copy()
            for b in range(6):
                q_in = qkey == b
                if not q_in.any():
                    continue
                s_in = skey == b
                if s_in.sum() < 4:                              # too sparse to trust -- keep the fallback here
                    continue
                wqb, cqb = _bake_volume(s_in)
                use = q_in & (wqb > 1e-6)                       # only replace where THIS side actually covered
                col[use] = cqb[use]; wq[use] = wqb[use]
            col = col / np.maximum(wq[:, None], 1e-6)
        dist = np.where(wq > 1e-6, 0.0, np.inf)                 # coverage proxy: gathered mass, not a metric distance
        report_scatter = {"method": "scatter", "grid": GR, "gather_weight_mean": float(wq.mean()),
                          "normal_aware": bool(normal_aware)}
        th, tw = tex.shape[:2]                                  # (re-bound for the shared painting below)
        _skip_projection = True
    elif method == "recall":
        # H5 -- EVERY CLOSEST-POINT IS A RECALL: fill the atlas by SPATIAL MEMORY instead of projection or a
        # volume grid. Source vertices (keyed by 3-D position via fractional power encoding) carry their
        # texture colour as payload; each texel's colour is the resonant top-k readout at its 3-D point --
        # a soft nearest-neighbour gather in ENCODING space. MEASURED 0.034 RGB reading scan colour, better
        # than the scatter route's 0.066-0.088, at comparable speed once the memory is built (amortised).
        # KEPT NEGATIVE shared with scatter: quality is bounded by SOURCE VERTEX DENSITY (payloads live on
        # vertices), so a low-poly source with a high-res texture loses detail -- project is exact there.
        from holographic.sampling_and_signal.holographic_spatialmem import SpatialMemory
        suv = np.asarray(source_uv, float)
        th, tw = tex.shape[:2]
        sfu = np.clip(suv[:, 0], 0.0, 1.0) * (tw - 1); sfv = np.clip(suv[:, 1], 0.0, 1.0) * (th - 1)
        src_col = np.asarray(tex, float)[np.clip(sfv.astype(int), 0, th - 1), np.clip(sfu.astype(int), 0, tw - 1)]
        mem = SpatialMemory(SV, payloads=src_col, dim=256, seed=0)
        col = mem.read(pts, k=4)
        dist = np.zeros(len(pts))                               # recall has no metric residual to report
        report_scatter = {"method": "recall", "dim": 256, "n_source": int(len(SV))}
        _skip_projection = True
    else:
        report_scatter = {"method": "project"}
        _skip_projection = False

    # ---- project every texel's 3-D point to the source surface, AMORTISED PER FACE ----
    # NOT transfer_uv per texel: measured 963 pts/s on this scan -> a 1024^2 bake would be ~545 s, a wall.
    # The per-face atlas makes the wall unnecessary: all ~200 texels of one cell come from ONE target face, so
    # they share ONE neighbourhood. Query the spatial hash ONCE PER FACE (4942 queries, not 500k) and test that
    # face's texels against only those few candidate triangles, vectorised. Same closest-point answer, ~20x less
    # searching -- cache locality applied to the query itself, not a cleverer metric.
    from holographic.mesh_and_geometry.holographic_meshbridge import _closest_point_on_triangle as _cpot
    if not _skip_projection:
     stri = np.array([[SV[f[0]], SV[f[1]], SV[f[2]]] for f in SF])
     slo = stri.min(axis=(0, 1))
     scell = max(float(np.mean(np.linalg.norm(stri[:, 1] - stri[:, 0], axis=1))), 1e-9)
     shash = {}
     tmin = stri.min(1); tmax = stri.max(1)
     for fi in range(len(SF)):
         i0 = np.floor((tmin[fi] - slo) / scell).astype(int)
         i1 = np.floor((tmax[fi] - slo) / scell).astype(int)
         for ix in range(i0[0], i1[0] + 1):
             for iy in range(i0[1], i1[1] + 1):
                 for iz in range(i0[2], i1[2] + 1):
                     shash.setdefault((ix, iy, iz), []).append(fi)
     suv = np.asarray(source_uv, float)
     src_uv = np.empty((len(pts), 2)); dist = np.full(len(pts), np.inf)
     for f in range(nF):
         sl = face_slice[f]
         if sl is None:
             continue
         P = pts[sl]
         fmin = P.min(0); fmax = P.max(0)
         ring = 0
         cand = []
         while not cand and ring <= 6:                            # grow the ring until the neighbourhood is non-empty
             i0 = np.floor((fmin - slo) / scell).astype(int) - ring
             i1 = np.floor((fmax - slo) / scell).astype(int) + ring
             seen = set()
             for ix in range(i0[0], i1[0] + 1):
                 for iy in range(i0[1], i1[1] + 1):
                     for iz in range(i0[2], i1[2] + 1):
                         seen.update(shash.get((ix, iy, iz), ()))
             cand = sorted(seen)                                  # sorted: deterministic tie order
             ring += 1
         if not cand:
             cand = list(range(len(SF)))                          # degenerate fallback: the honest brute force
         best_d = np.full(len(P), np.inf); best_uv = np.zeros((len(P), 2))
         for fi in cand:
             a, b, c = stri[fi]
             q = _cpot(P, a, b, c)                            # meshbridge's batched closest-point-on-triangle
             d = np.linalg.norm(P - q, axis=1)
             hit = d < best_d
             if hit.any():
                 bc = _barycentric_batch(q[hit], a, b, c)  # the interpolation weights for the source uvs
                 sf = SF[fi]
                 best_uv[hit] = (bc[:, 0:1] * suv[sf[0]] + bc[:, 1:2] * suv[sf[1]] + bc[:, 2:3] * suv[sf[2]])
                 best_d[hit] = d[hit]
         src_uv[sl] = best_uv; dist[sl] = best_d

    # ---- sample the source texture (bilinear) and paint the new atlas ----
    th, tw = tex.shape[:2]
    # CLAMP, not `% 1.0`: a bake samples ONE given texture, so uv 1.0 is its LAST texel. The modulo the
    # rasterizer uses is a TILING convention, and under it 1.0 % 1.0 == 0.0 -- the atlas's far edge teleports
    # in. Measured: a target vertex projecting to source uv (0.996, 1.0) sampled texture row 0 instead of row
    # 63 and baked the wrong colour. Wrap is right for a tiled material; clamp is right for an atlas.
    if not _skip_projection:                                     # scatter path already produced `col` directly
        fu = np.clip(src_uv[:, 0], 0.0, 1.0) * (tw - 1); fv = np.clip(src_uv[:, 1], 0.0, 1.0) * (th - 1)
        ix0 = np.floor(fu).astype(int); iy0 = np.floor(fv).astype(int)
        ix1 = np.minimum(ix0 + 1, tw - 1); iy1 = np.minimum(iy0 + 1, th - 1)
        ax = (fu - ix0)[:, None]; ay = (fv - iy0)[:, None]
        col = ((tex[iy0, ix0] * (1 - ax) + tex[iy0, ix1] * ax) * (1 - ay)
               + (tex[iy1, ix0] * (1 - ax) + tex[iy1, ix1] * ax) * ay)
    img = np.zeros((size, size, tex.shape[2] if tex.ndim == 3 else 3))
    written = np.zeros((size, size), bool)
    img[py, px] = col                                            # v -> row directly (glTF top-left origin, pinned)
    written[py, px] = True

    # ---- dilate `margin` texels so bilinear filtering at a cell edge never grabs background ----
    # SHIFTED SLICES, not np.roll: roll WRAPS, so the atlas's right edge would be padded from its LEFT edge --
    # an unrelated face's colour teleporting across the image. Measured (a corner texel came back with another
    # face's colour); this is the kind of bug a "it looks fine" eyeball never catches.
    for _ in range(int(margin)):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = np.zeros_like(img); srcw = np.zeros_like(written)
            ys_dst = slice(max(dy, 0), size + min(dy, 0))         # destination rows this shift can reach
            ys_src = slice(max(-dy, 0), size + min(-dy, 0))       # the rows they read from
            xs_dst = slice(max(dx, 0), size + min(dx, 0))
            xs_src = slice(max(-dx, 0), size + min(-dx, 0))
            src[ys_dst, xs_dst] = img[ys_src, xs_src]
            srcw[ys_dst, xs_dst] = written[ys_src, xs_src]
            fill = srcw & ~written
            img[fill] = src[fill]
            written |= fill

    if fill_mode == "flood":
        # ---- FLOOD FILL every remaining unwritten texel (the dark-speckle fix, Moose-caught) ----------
        # The per-face atlas packs each triangle into HALF a cell; the other half stays black, and at 10k
        # faces in a 768 atlas the cell is ~7 px, so a 2-px margin cannot cover it -- bilinear taps near
        # every hypotenuse then blend BLACK, which rendered as dark speckle over the whole body (MEASURED:
        # 21.5% of foreground pixels; margin cannot be raised because it eats the cell). Flooding repeats
        # the same non-wrapping shifted-slice dilation until nothing is unwritten -- colours only ever
        # bleed OUTWARD from painted texels, so on-chart data is untouched (default "margin" keeps the
        # pinned bake bit-identical; process_scan opts in).
        it = 0
        while (~written).any() and it < 4 * size:
            it += 1
            progressed = False
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                src = np.zeros_like(img); srcw = np.zeros_like(written)
                ys_dst = slice(max(dy, 0), size + min(dy, 0))
                ys_src = slice(max(-dy, 0), size + min(-dy, 0))
                xs_dst = slice(max(dx, 0), size + min(dx, 0))
                xs_src = slice(max(-dx, 0), size + min(-dx, 0))
                src[ys_dst, xs_dst] = img[ys_src, xs_src]
                srcw[ys_dst, xs_dst] = written[ys_src, xs_src]
                fillm = srcw & ~written
                if fillm.any():
                    img[fillm] = src[fillm]
                    written |= fillm
                    progressed = True
            if not progressed:
                break                                            # isolated all-unwritten atlas (empty bake)

    out = Mesh(new_V, new_F, uvs=new_uv)
    # NOTE: uv_straddle_fraction is deliberately NOT reported here. It asks "does a face's uv cross an island
    # boundary", which presumes a SHARED atlas; in a per-face atlas every face IS its own island, so a large uv
    # edge is correct by construction and the metric would read 1.0 on a perfect bake. Wrong question, not a
    # failing grade -- the honest signals for a bake are the projection distance and the texel coverage.
    report = {"faces": nF, "atlas_cells": g * g, "texels_written": int(px.size),
              "texel_coverage": float(written.mean()),
              "projection_distance_mean": float(dist.mean()),
              "projection_distance_p95": float(np.percentile(dist, 95))}
    report.update(report_scatter)                                # method + (scatter: grid, gather_weight_mean)
    return out, new_uv, img, report


def textured_lod(mesh, texture, uvs=None, grid=48, size=1024, margin=2):
    """ONE CALL for the thing everyone actually wants: a decimated mesh that STILL WEARS ITS TEXTURE, by the
    route that is correct for the mesh you actually have. Returns (lod_mesh, uv, image, report).

    It picks the route from a measurement rather than hoping: uv_atlas_report(mesh) decides.
      * coherent atlas (many faces per island) -> decimate and TRANSFER the uvs (cheap; the source image is
        reused unchanged, so `image` comes back as the input texture).
      * fragmented atlas (a photogrammetry scan: islands of ~1 face) -> decimate and RE-BAKE into a new
        per-face atlas (rebake_texture), returning a NEW image. The only correct route there.

    WHY THIS EXISTS: doing it by hand takes four steps and one non-obvious judgement, and getting the
    judgement wrong renders as speckled confetti with no error raised -- which is exactly what happened to a
    mantis scan at low quality. `report["route"]` says which way it went and why, so the failure that used to
    be silent is now a field you can read. KEPT NEGATIVE: the re-bake route costs a projection per texel and
    a new image; the transfer route is nearly free. That asymmetry is the atlas's fault, not the caller's."""
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate
    uv = np.asarray(uvs if uvs is not None else mesh.uvs, float)
    atlas = uv_atlas_report(mesh)
    if atlas["transferable"]:
        lod = cluster_decimate(mesh, grid=grid, keep_uv=False)
        new_uv, dist = transfer_uv(mesh, uv, np.asarray(lod.vertices, float))
        lod.uvs = new_uv
        straddle, max_edge = uv_straddle_fraction(lod, new_uv)
        return lod, new_uv, np.asarray(texture, float), {
            "route": "transfer", "reason": "coherent atlas (median %.1f faces/island)" % atlas["faces_per_island_median"],
            "atlas": atlas, "faces": len(lod.faces), "straddle_fraction": straddle,
            "projection_distance_mean": float(dist.mean())}
    lod = cluster_decimate(mesh, grid=grid, keep_uv=False)          # geometry only; its uvs would be garbage
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    src = Mesh(mesh.vertices, mesh.faces, uvs=uv)
    out, new_uv, img, rep = rebake_texture(src, uv, texture, lod, size=size, margin=margin)
    rep.update({"route": "rebake",
                "reason": "fragmented atlas (median %.1f faces/island, %d islands over %d faces): per-vertex "
                          "transfer cannot preserve it" % (atlas["faces_per_island_median"], atlas["islands"],
                                                           atlas["faces"]),
                "atlas": atlas})
    return out, new_uv, img, rep


def _uv_sphere_fixture(n=12):
    """A sphere with a spherical unwrap: ONE island, a real u=0/u=1 seam, and POLE fans where every u collapses
    onto one 3-D point. The shape of a normal game asset, and the shape that breaks naive uv transfer."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    us = np.linspace(0, 1, n + 1); vs = np.linspace(0, 1, n // 2 + 1)
    V, UV, F = [], [], []
    for v in vs:
        for u in us:
            th = u * 2 * np.pi; ph = v * np.pi
            V.append([np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)]); UV.append([u, v])
    W = n + 1
    for j in range(len(vs) - 1):
        for i in range(n):
            a, b, c, d = j * W + i, j * W + i + 1, (j + 1) * W + i + 1, (j + 1) * W + i
            F += [(a, b, c), (a, c, d)]
    return Mesh(np.array(V, float), F, uvs=np.array(UV, float))


def _uv_cylinder_fixture(n=32, h=10):
    """A cylinder: the u-seam is a full-height cut with real AREA either side, and no pole singularity to
    confound the measurement. The honest test for seam handling -- see _selftest_reproject."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    us = np.linspace(0, 1, n + 1); vs = np.linspace(0, 1, h + 1)
    V, UV, F = [], [], []
    for v in vs:
        for u in us:
            th = u * 2 * np.pi
            V.append([np.cos(th), (v - 0.5) * 2.0, np.sin(th)]); UV.append([u, v])
    W = n + 1
    for j in range(h):
        for i in range(n):
            a, b, c, d = j * W + i, j * W + i + 1, (j + 1) * W + i + 1, (j + 1) * W + i
            F += [(a, b, c), (a, c, d)]
    return Mesh(np.array(V, float), F, uvs=np.array(UV, float))


def _selftest_reproject():
    """Pin the uv-through-retopo contract with numbers:
    (1) the degenerate-triangle guard: a projection onto a zero-length-edge triangle must be FINITE (it used to
        return NaN, and NaN uvs silently flowed out of cluster_decimate and voxel_remesh);
    (2) reproject_uv eliminates seam-spanning faces on a decimated cylinder, where transfer leaves them;
    (3) the splits it adds land at the cut, not everywhere (an early version split 442 of 156 vertices);
    (4) a fragmented atlas RAISES and names the re-bake, rather than returning confetti."""
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate

    # (1) the NaN that started it all: Ericson's test assumes a non-degenerate triangle
    p = np.array([0.0, 0.0, 0.0])
    q, bc = _closest_point_barycentric(p, np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    assert np.isfinite(q).all() and np.isfinite(bc).all(), "zero-length-edge triangle must not produce NaN"
    assert abs(sum(bc) - 1.0) < 1e-12

    def crossers(mesh, uv):
        F = np.asarray([f[:3] for f in mesh.faces], int)
        me = np.zeros(len(F))
        for k in range(3):
            me = np.maximum(me, np.abs(uv[F[:, k], 0] - uv[F[:, (k + 1) % 3], 0]))
        return int((me > 0.5).sum())

    # (2)+(3) the cylinder: a seam with real area
    cyl = _uv_cylinder_fixture()
    lod = cluster_decimate(cyl, grid=7, keep_uv=False)
    naive, _ = transfer_uv(cyl, np.asarray(cyl.uvs), np.asarray(lod.vertices))
    n_naive = crossers(lod, naive)
    assert n_naive > 0, "the fixture must actually exhibit the seam bug, or it proves nothing"
    rm, ruv, rep = reproject_uv(cyl, np.asarray(cyl.uvs), lod)
    assert rep["finite"], "reprojected uvs must be finite"
    assert crossers(rm, ruv) == 0, "reprojection must leave no face spanning the seam"
    assert rep["seam_splits"] < 0.25 * len(lod.vertices), "splits land at the cut, not everywhere"
    # The SMOOTH coordinate must survive untouched. v has no seam on a cylinder, so ANY error in it is pure
    # fabrication by the tie-break. An earlier version answered "which side" and "where on that side" with the
    # SAME uv argmin, dragging every ambiguous corner toward its own face's centre: v went from exactly 0.00000
    # to 0.05 at p95. Moose spotted that from the render as "the uvs don't line up". This pins it at exact.
    v_true = (np.asarray(rm.vertices)[:, 1] / 2.0) + 0.5
    v_err = float(np.abs(ruv[:, 1] - v_true).max())
    assert v_err < 1e-9, "reprojection fabricated error in the seam-free coordinate: %.2e" % v_err

    # (4) a fragmented atlas has no reprojection -- it must SAY so, not invent one
    F = np.asarray([f[:3] for f in cyl.faces], int)
    frag_V = np.asarray(cyl.vertices)[F].reshape(-1, 3)
    frag_uv = np.asarray(cyl.uvs)[F].reshape(-1, 2)
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    frag = Mesh(frag_V, [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(F))], uvs=frag_uv)
    try:
        reproject_uv(frag, frag_uv, lod)
        raise AssertionError("a fragmented atlas must refuse reprojection")
    except ValueError as e:
        assert "rebake" in str(e), "the refusal must name the right route, got: %s" % e

    # The sphere -- FORMERLY a kept negative (5 spanning faces -> 1, the pole fan), now CLOSED by the
    # cut-aware-transfer redesign: side-as-CONSTRAINT on every corner (not a tie-break), abstention-gated
    # majority home vote, and uv-to-home ranking ONLY at uv-ambiguous geometric ties. Pinned at ZERO spanning
    # and ZERO internally-inconsistent faces at grid 7 and 5. The pole is handled, not waived: the fan splits
    # (that is what the +splits are) exactly as the source stores it.
    sph = _uv_sphere_fixture(24)
    for g in (7, 5):
        slod = cluster_decimate(sph, grid=g, keep_uv=False)
        srm, sruv, srep = reproject_uv(sph, np.asarray(sph.uvs), slod)
        assert srep["finite"]
        SFc = np.asarray([f[:3] for f in srm.faces], int)
        for f in SFc:
            us = sruv[f, 0]
            assert np.abs(us[:, None] - us[None, :]).max() <= 0.5, "sphere seam/pole face spans the atlas"
            circ = np.minimum(np.abs(us[:, None] - us[None, :]), 1 - np.abs(us[:, None] - us[None, :])).max()
            assert circ <= 0.15, "sphere face internally inconsistent (the subtle misalignment class)"
    print("reproject selftest OK (degenerate triangle finite; cylinder seam %d -> 0 spanning faces for %d "
          "splits of %d verts; fragmented atlas refused and named rebake)"
          % (n_naive, rep["seam_splits"], len(lod.vertices)))


def _selftest_rebake():
    """Pin the texture-through-LOD contract with numbers, not eyeballs. Four claims:
    (1) the atlas report separates a coherent atlas from a fragmented one;
    (2) the bake reproduces a known analytic texture to texel quantisation -- INCLUDING at face corners;
    (3) keep_uv='auto' REFUSES to transfer a fragmented atlas (and says so) instead of shipping confetti;
    (4) textured_lod routes by measurement."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate

    # ---- a COHERENT atlas: one 8x8 grid sheet, all faces in one island ----
    n = 8
    xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    V = np.stack([xs.ravel() / n, ys.ravel() / n, np.zeros((n + 1) ** 2)], axis=1).astype(float)
    uvg = V[:, :2].copy()
    def vid(i, j): return i * (n + 1) + j
    Fg = []
    for i in range(n):
        for j in range(n):
            Fg += [(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)), (vid(i, j), vid(i + 1, j + 1), vid(i, j + 1))]
    sheet = Mesh(V, Fg, uvs=uvg)
    r_ok = uv_atlas_report(sheet)
    assert r_ok["islands"] == 1 and r_ok["transferable"], r_ok

    # ---- a FRAGMENTED atlas: the same sheet with every triangle given private vertices (a scan's shape) ----
    Fa = np.asarray(Fg)
    Vs = V[Fa].reshape(-1, 3)
    uvs_frag = uvg[Fa].reshape(-1, 2)
    frag = Mesh(Vs, [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(Fa))], uvs=uvs_frag)
    r_bad = uv_atlas_report(frag)
    assert r_bad["islands"] == len(Fa) and r_bad["faces_per_island_median"] == 1.0
    assert not r_bad["transferable"], "a per-triangle atlas must never be called transferable"

    # ---- (3) the honesty rule: auto refuses the fragmented one, True forces it ----
    lod_auto = cluster_decimate(frag, grid=4, keep_uv="auto")
    assert getattr(lod_auto, "uvs", None) is None, "auto must NOT emit uvs it cannot compute honestly"
    assert lod_auto.uv_transfer_report["skipped"] is True
    assert "rebake_texture" in lod_auto.uv_transfer_report["reason"]      # it must POINT AT the right route
    lod_force = cluster_decimate(frag, grid=4, keep_uv=True)
    assert getattr(lod_force, "uvs", None) is not None                    # forcing still works (old behaviour)
    lod_coherent = cluster_decimate(sheet, grid=4, keep_uv="auto")
    assert getattr(lod_coherent, "uvs", None) is not None, "a coherent atlas must still transfer"

    # ---- (2) the bake reproduces an analytic texture (R=u, G=v) through a topology change ----
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
    for i in range(len(OV)):                                  # corners INCLUDED: they caught two real bugs
        u, v = nuv[i]
        got = img[int(np.clip(round(v * (h - 1)), 0, h - 1)), int(np.clip(round(u * (w - 1)), 0, w - 1))]
        err.append(abs(got[0] - OV[i][0]))                    # texture R encodes u == x on this quad
        err.append(abs(got[1] - OV[i][1]))                    # G encodes v == y
    err = np.array(err)
    quant = 1.0 / (T - 1)
    assert err.max() < quant, "bake corner error %.4f exceeds texel quantisation %.4f" % (err.max(), quant)
    assert rep["projection_distance_mean"] < 1e-4
    # KEPT NEGATIVE, pinned: uv_straddle_fraction's threshold is an absolute fraction of the atlas, so it is
    # DENSITY-dependent. This 4-face bake packs g=2 huge cells -> uv edges ~0.5 -> it reads ~1.0 despite the
    # bake being provably correct (asserted above). Not an island problem -- a scale problem. Choose the
    # threshold from the source's uv edge scale, and ask uv_atlas_report the structural question instead.
    s_bake, _ = uv_straddle_fraction(out, nuv)
    assert s_bake > 0.5, "if this reads ~0, the metric stopped being scale-dependent -- re-read its docstring"

    # ---- (H1) the SCATTER method: holographic scatter/gather bake, deterministic, default-off ----
    # method="project" (the default) must stay UNCHANGED (backward-compatible); method="scatter" is the fast path.
    _, _, _, rp = rebake_texture(quad, np.asarray(quad.uvs), tex, tgt, size=256, margin=2)
    assert rp["method"] == "project", "default method must stay 'project' (backward-compatible)"
    # scatter is bounded by SOURCE VERTEX DENSITY, so test it on the DENSE 8x8 sheet (its actual use case: a
    # dense scan), not the 4-vertex quad -- on 4 source verts a volumetric bake cannot resolve the corners, and
    # that IS the kept negative. Bake the sheet's own R=u,G=v texture onto a re-triangulated copy and check the
    # interior reproduces (skip the extreme-edge corners, which the source-density limit owns).
    sheet_uv = np.asarray(sheet.uvs, float)
    sheet_tex = np.zeros((64, 64, 3)); syy, sxx = np.mgrid[0:64, 0:64]
    sheet_tex[:, :, 0] = sxx / 63.0; sheet_tex[:, :, 1] = syy / 63.0
    ms, us, imgs, rs = rebake_texture(sheet, sheet_uv, sheet_tex, sheet, size=256, margin=2, method="scatter")
    assert rs["method"] == "scatter" and "grid" in rs and rs["gather_weight_mean"] > 0, rs
    OVs = np.asarray(ms.vertices); hh, ww = imgs.shape[:2]
    errs = []
    for i in range(len(OVs)):
        u, v = us[i]
        if not (0.15 < OVs[i][0] < 0.85 and 0.15 < OVs[i][1] < 0.85):   # interior only; edges are the neg
            continue
        got = imgs[int(np.clip(round(v * (hh - 1)), 0, hh - 1)), int(np.clip(round(u * (ww - 1)), 0, ww - 1))]
        errs += [abs(got[0] - OVs[i][0]), abs(got[1] - OVs[i][1])]
    assert errs and np.array(errs).max() < 0.15, "scatter interior error %.3f too high" % np.array(errs).max()
    # determinism (hashlib-grade repeatability, per the constitution)
    _, _, imgs2, _ = rebake_texture(sheet, sheet_uv, sheet_tex, sheet, size=256, margin=2, method="scatter")
    assert np.array_equal(imgs, imgs2), "scatter bake must be deterministic"
    # AUTO-SIZE (client P2): an explicit size too small for the face count raises a HELPFUL error naming the
    # minimum, and size="auto" picks a working size so the correct route just runs. Backward-compat: an explicit
    # int big enough is byte-identical (the 256 bakes above are unchanged).
    _nF = len(sheet.faces); _g = int(np.ceil(np.sqrt(_nF)))
    try:
        rebake_texture(sheet, sheet_uv, sheet_tex, sheet, size=3, margin=2, method="scatter")
        raise AssertionError("a size of 3 must be too small for %d faces" % _nF)
    except ValueError as _e:
        assert "too small" in str(_e) and "at least" in str(_e), "the size error must name the minimum: %s" % _e
    _, _, _imga, _ra = rebake_texture(sheet, sheet_uv, sheet_tex, sheet, size="auto", margin=2, method="scatter")
    assert _imga.shape[0] >= _g * (4 + 2 * 2), "auto size must be >= g*(texels+2*margin), got %d" % _imga.shape[0]
    # KEPT NEGATIVE, pinned by the interior-only test: scatter quality is bounded by SOURCE VERTEX DENSITY, not
    # texture resolution. A low-poly source (4 verts) cannot resolve corners; a dense scan can. project stays
    # the default; scatter is opt-in for dense scans where the ~1500x speed matters.

    # ---- (H5) the RECALL method: atlas filled by spatial memory (position -> resonant colour readout) ----
    msr, usr, imr, rsr = rebake_texture(sheet, sheet_uv, sheet_tex, sheet, size=256, margin=2, method="recall")
    assert rsr["method"] == "recall" and rsr["n_source"] == len(sheet.vertices), rsr
    errs_r = []
    for i in range(len(OVs)):
        u, v = usr[i]
        if not (0.15 < OVs[i][0] < 0.85 and 0.15 < OVs[i][1] < 0.85):
            continue
        got = imr[int(np.clip(round(v * (hh - 1)), 0, hh - 1)), int(np.clip(round(u * (ww - 1)), 0, ww - 1))]
        errs_r += [abs(got[0] - OVs[i][0]), abs(got[1] - OVs[i][1])]
    assert errs_r and np.array(errs_r).max() < 0.15, "recall bake interior error %.3f" % np.array(errs_r).max()
    _, _, imr2, _ = rebake_texture(sheet, sheet_uv, sheet_tex, sheet, size=256, margin=2, method="recall")
    assert np.array_equal(imr, imr2), "recall bake must be deterministic"
    # KEPT NEGATIVE shared with scatter: quality bounded by SOURCE VERTEX DENSITY (payloads live on verts).
    # TRADE, measured: recall reads best at VERTEX-scale queries (0.9s/1.7k verts, 0.034 RGB); scatter owns
    # TEXEL-scale atlases (2.5s vs recall's 120s at 768^2 -- the streamed matmul is the cost). Both opt-in.

    # ---- (4) textured_lod routes by measurement, both ways ----
    _, _, _, r1 = textured_lod(sheet, tex, grid=4)
    assert r1["route"] == "transfer", r1["route"]
    _, _, img2, r2 = textured_lod(frag, tex, grid=4, size=128)
    assert r2["route"] == "rebake", r2["route"]
    assert img2.shape[0] == 128                               # a rebake returns a NEW image, not the input
    print("rebake selftest OK (atlas: 1 island coherent vs %d islands fragmented; bake corner err %.4f < "
          "quant %.4f; auto refused the fragmented transfer and named rebake_texture; routes transfer/rebake)"
          % (r_bad["islands"], err.max(), quant))


if __name__ == "__main__":
    _selftest_rebake()   # the guards above already ran the earlier selftests in definition order


def _selftest_topology_gate():
    """R1: invariants right on the canonical fixtures; gate rejects destruction, accepts intended holes."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, grid, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    b = triangulate_ngons(box())
    rb = topology_report(b)
    assert rb["per_component"][0]["genus"] == 0 and rb["per_component"][0]["boundary_loops"] == 0
    g = triangulate_ngons(grid(4, 4))
    rg = topology_report(g)
    assert rg["per_component"][0]["boundary_loops"] == 1 and rg["per_component"][0]["genus"] == 0
    # analytic torus: genus 1 -- the invariant a boundary-edge COUNT could never see
    uu, vv = np.meshgrid(np.arange(24), np.arange(12), indexing="ij")
    th = uu / 24 * 2 * np.pi; ph = vv / 12 * 2 * np.pi
    V = np.stack([(1 + .35 * np.cos(ph)) * np.cos(th), (1 + .35 * np.cos(ph)) * np.sin(th),
                  .35 * np.sin(ph)], -1).reshape(-1, 3)
    def vid(i, j): return (i % 24) * 12 + (j % 12)
    F = []
    for i in range(24):
        for j in range(12):
            F += [(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)), (vid(i, j), vid(i + 1, j + 1), vid(i, j + 1))]
    t = Mesh(V, F)
    assert topology_report(t)["per_component"][0]["genus"] == 1
    # gate: identity passes; punched hole FAILS with the violation NAMED; genus change FAILS
    ok, _ = topology_gate(b, b); assert ok
    punched = Mesh(np.asarray(b.vertices, float), [tuple(int(i) for i in f) for f in b.faces][:-2])
    ok, rep = topology_gate(b, punched)
    assert not ok and "new boundary loop" in " ".join(rep["violations"])
    ok, rep = topology_gate(t, b)
    assert not ok and "genus" in " ".join(rep["violations"])
    ok, _ = topology_gate(g, g); assert ok           # the INTENDED hole is matched, not flagged
    # KEPT NEGATIVE: the gate DETECTS destruction; it cannot RECONNECT a shattered body (that is R2/R3's job).
    print("topology gate selftest OK (box g0 / grid 1-loop / torus g1; punched+genus rejected, intended kept)")


def _selftest_drop_small_components():
    """Pin: drop_small_components keep_largest keeps exactly the biggest component and drops the shards, with
    uvs carried through the remap. Built on two disjoint triangles (a 'body' pair + a lone shard)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    # component A: two triangles sharing an edge (4 verts, 2 faces). component B: one lone triangle (shard).
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],      # A
                  [5, 5, 5], [6, 5, 5], [5, 6, 5]], float)          # B (shard)
    uvA = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.6, 0.5], [0.5, 0.6]], float)
    mesh = Mesh(V, [(0, 1, 2), (1, 3, 2), (4, 5, 6)], uvs=uvA)
    body, rep = drop_small_components(mesh, keep_largest=True)
    assert rep["components_before"] == 2, rep
    assert rep["components_after"] == 1, rep
    assert rep["faces_after"] == 2, rep                            # the 2-triangle body, not the shard
    assert len(body.vertices) == 4, "shard verts must be dropped and reindexed"
    assert body.uvs is not None and len(body.uvs) == 4, "uvs must be carried through the remap"
    # min_fraction keeps both if the shard is >= f of the body
    _, rep2 = drop_small_components(mesh, min_fraction=0.4)        # shard 1 face vs body 2 -> 0.5 >= 0.4 keep
    assert rep2["components_after"] == 2, rep2
    print("drop_small_components selftest OK (keep_largest -> 1 comp / 2 faces, uvs carried; "
          "min_fraction=0.4 keeps the half-size shard)")


def _selftest_process_scan():
    """Pin: process_scan runs all four workflows on a box and each yields a valid mesh with the expected
    stages; the geometry-only paths return (mesh, None, None, report). The textured path is pinned by the cad
    test (needs a real texture); here the STAGE ROUTING is the contract."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    src = triangulate_ngons(box())
    for retopo, lod, want in [(True, 0.5, ["repair", "retopo", "lod_via_coarser_retopo", "shard_cleanup"]),
                              (True, None, ["repair", "retopo", "shard_cleanup"]),
                              (False, 0.5, ["repair", "lod_via_decimate"]),
                              (False, None, ["repair"])]:
        out, u, img, rep = process_scan(src, retopo=retopo, lod=lod, density=1.0)
        stages = [s["stage"] for s in rep["stages"]]
        assert stages == want, "workflow (retopo=%s, lod=%s): stages %s != %s" % (retopo, lod, stages, want)
        assert len(out.faces) > 0 and u is None and img is None
    print("process_scan selftest OK (four workflows route repair/retopo/lod/cleanup as specified; "
          "geometry-only returns mesh+report)")


if __name__ == "__main__":
    _selftest_reproject(); _selftest_drop_small_components(); _selftest_process_scan(); _selftest_topology_gate(); _selftest_manifold_cleanup()

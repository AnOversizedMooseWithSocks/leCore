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


def merge_by_distance(mesh, tol=1e-5):
    """Weld vertices closer than `tol` into one. Vertices are grouped by snapping to a `tol` grid; each group
    becomes one vertex at the group's mean; faces are remapped and any face that collapsed to < 3 distinct
    vertices is dropped. Vectorised for triangle meshes (the (T,3) face table is remapped and degenerate-filtered
    as array ops). The cleanup after a mirror / import / boolean."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices
    key = np.round(V / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)      # old vertex -> merged group id
    inv = np.asarray(inv).ravel()
    nnew = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=nnew).astype(float)
    Vnew = np.zeros((nnew, 3))
    np.add.at(Vnew, inv, V)                                   # group sum (scatter)
    Vnew /= counts[:, None]                                   # -> group mean

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
        for e in sorted(edge_faces):                         # link faces across MANIFOLD edges incident to v only
            if v in e:
                efs = edge_faces[e]
                if len(efs) == 2 and efs[0] in fset and efs[1] in fset:
                    ra, rb = find(efs[0]), find(efs[1])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)    # deterministic union (min root wins)
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
    return Mesh(Vnew, Fnew), {"split_vertices": int(n_split), "added_vertices": int(len(new_positions))}


def _closest_point_barycentric(p, a, b, c):
    """Closest point to `p` on triangle (a,b,c) with its barycentric coords (u,v,w), u+v+w=1 -- Ericson's
    region-test (Real-Time Collision Detection ch.5), the standard exact projection. Returns (point, (u,v,w))."""
    ab = b - a; ac = c - a; ap = p - a
    d1 = ab @ ap; d2 = ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a, (1.0, 0.0, 0.0)
    bp = p - b; d3 = ab @ bp; d4 = ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b, (0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab, (1.0 - v, v, 0.0)
    cp = p - c; d5 = ab @ cp; d6 = ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c, (0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac, (1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), (0.0, 1.0 - w, w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom; w = vc * denom
    return a + ab * v + ac * w, (1.0 - v - w, v, w)


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
    from collections import defaultdict
    V = np.asarray(source_mesh.vertices, float)
    F = [tuple(int(i) for i in f[:3]) for f in source_mesh.faces]
    UVs = np.asarray(source_uv, float)
    T = np.asarray(target_vertices, float)
    # uniform grid over triangle bounding boxes
    tri = np.array([[V[f[0]], V[f[1]], V[f[2]]] for f in F])          # (F,3,3)
    lo = tri.min(axis=(0, 1)); hi = tri.max(axis=(0, 1))
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
    out = np.zeros((len(T), UVs.shape[1]))
    dist = np.zeros(len(T))
    for ti, p in enumerate(T):
        base = np.floor((p - lo) / cell).astype(int)
        best_d2 = None; best = None
        ring = 0
        while best_d2 is None or ring <= 1:                    # search this cell ring, then one more to be safe
            cand = set()
            for ix in range(base[0] - ring, base[0] + ring + 1):
                for iy in range(base[1] - ring, base[1] + ring + 1):
                    for iz in range(base[2] - ring, base[2] + ring + 1):
                        if ring == 0 or max(abs(ix - base[0]), abs(iy - base[1]), abs(iz - base[2])) == ring:
                            cand.update(grid.get((ix, iy, iz), ()))
            for fi in cand:
                f = F[fi]
                q, bc = _closest_point_barycentric(p, V[f[0]], V[f[1]], V[f[2]])
                d2 = float(np.sum((p - q) ** 2))
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = bc[0] * UVs[f[0]] + bc[1] * UVs[f[1]] + bc[2] * UVs[f[2]]
            ring += 1
            if ring > 12 and best_d2 is None:                  # far off-grid target: brute-force fall back once
                for fi in range(len(F)):
                    f = F[fi]
                    q, bc = _closest_point_barycentric(p, V[f[0]], V[f[1]], V[f[2]])
                    d2 = float(np.sum((p - q) ** 2))
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = bc[0] * UVs[f[0]] + bc[1] * UVs[f[1]] + bc[2] * UVs[f[2]]
                break
        out[ti] = best
        dist[ti] = np.sqrt(best_d2)
    return out, dist


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
    return Mesh(V, F)


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
    if drop_unreferenced:
        m = _drop_unreferenced(m)
    after = _stats(m)
    return m, {"before": before, "after": after, "holes_filled": filled, "split_vertices": int(n_split),
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


def bake_normal_map(low_mesh, low_uv, high_mesh, size=256, world_space=False, ao=False, ao_samples=0):
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

    # spatial hash of high-poly triangles for the closest-point lookup (same structure as transfer_uv)
    htri = np.array([[HV[f[0]], HV[f[1]], HV[f[2]]] for f in HF])
    hlo = htri.min(axis=(0, 1))
    el = np.linalg.norm(htri[:, 1] - htri[:, 0], axis=1)
    cell = max(float(np.mean(el)), 1e-9)
    grid = defaultdict(list)
    tmin = htri.min(1); tmax = htri.max(1)
    for fi in range(len(HF)):
        i0 = np.floor((tmin[fi] - hlo) / cell).astype(int); i1 = np.floor((tmax[fi] - hlo) / cell).astype(int)
        for ix in range(i0[0], i1[0] + 1):
            for iy in range(i0[1], i1[1] + 1):
                for iz in range(i0[2], i1[2] + 1):
                    grid[(ix, iy, iz)].append(fi)

    def _high_normal_at(p):
        base = np.floor((p - hlo) / cell).astype(int)
        best_d2 = None; best_n = None; ring = 0
        while best_d2 is None or ring <= 1:
            cand = set()
            for ix in range(base[0] - ring, base[0] + ring + 1):
                for iy in range(base[1] - ring, base[1] + ring + 1):
                    for iz in range(base[2] - ring, base[2] + ring + 1):
                        if ring == 0 or max(abs(ix - base[0]), abs(iy - base[1]), abs(iz - base[2])) == ring:
                            cand.update(grid.get((ix, iy, iz), ()))
            for fi in cand:
                f = HF[fi]
                q, bc = _closest_point_barycentric(p, HV[f[0]], HV[f[1]], HV[f[2]])
                d2 = float(np.sum((p - q) ** 2))
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_n = bc[0] * hnorm[f[0]] + bc[1] * hnorm[f[1]] + bc[2] * hnorm[f[2]]
            ring += 1
            if ring > 12 and best_d2 is None:
                for fi in range(len(HF)):
                    f = HF[fi]
                    q, bc = _closest_point_barycentric(p, HV[f[0]], HV[f[1]], HV[f[2]])
                    d2 = float(np.sum((p - q) ** 2))
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_n = bc[0] * hnorm[f[0]] + bc[1] * hnorm[f[1]] + bc[2] * hnorm[f[2]]
                break
        n = best_n / (np.linalg.norm(best_n) + 1e-12)
        return n

    img = np.zeros((size, size, 3)); img[:] = (0.5, 0.5, 1.0) if not world_space else (0.5, 0.5, 0.5)
    aoimg = np.ones((size, size)) if ao else None
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
                hn = _high_normal_at(P)
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
    return (img, aoimg) if ao else img


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
    _selftest(); _selftest_route_repair()

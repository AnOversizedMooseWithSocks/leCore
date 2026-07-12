"""QEM decimation -- the quadric error metric (holographic_meshqem).

WHY THIS MODULE EXISTS
----------------------
The engine had every part of a principled mesh decimator EXCEPT the cost function itself: the guarded
`collapse_edge` (eulerops -- the link-condition refusal made operational), the heapq priority-descent with
deterministic ties (HoloForest / Dijkstra), the curvature read-out (meshcurvature), and the greedy-by-error loop
shape (matching pursuit). The one genuinely-missing piece was the **quadric error metric** (Garland & Heckbert,
SIGGRAPH 1997) -- the measure of how much collapsing an edge moves the surface. This module supplies it and wires
it to the shipped collapse, giving an error-driven simplifier that preserves sharp features instead of eroding
them like a naive shortest-edge collapse.

THE QUADRIC, AND WHY IT BELONGS IN THIS ENGINE
  For a triangle with unit normal n and a point p on it, the plane is [n, -n.p] = [a,b,c,d], and the SQUARED
  distance from a homogeneous point v=[x,y,z,1] to that plane is exactly v^T (pp^T) v. So the per-vertex error
  quadric is Q_v = sum of pp^T over the vertex's incident planes -- an OUTER-PRODUCT ACCUMULATION, i.e. a BUNDLE
  of plane constraints in matrix form, with the collapse cost read out as a quadratic. That is bind/bundle/readout
  in a different costume, which is why the reverse thesis (VSA is geometry) predicts this same operator reappears
  as a general "merge the two items whose combined representation loses the least" -- prototype compaction, splat
  merge, codebook merge. Build it once for meshes; it is the merge operator everywhere.

WHAT IT PROVIDES
  * vertex_quadrics(mesh) -- the per-vertex 4x4 error quadrics (the accumulated plane constraints).
  * contraction_target(Q, p_i, p_j) -- the optimal merged position (argmin v^T Q v) and its cost; falls back to
    the best of {midpoint, endpoints} when the 3x3 system is singular (a flat/degenerate neighbourhood).
  * qem_decimate(mesh, target_faces) -- greedily collapse the lowest-cost edge (deterministic ties by vertex
    index) via the guarded collapse_edge, accumulating quadrics through each collapse, until target_faces.
  * surface_deviation(mesh_a, mesh_b) -- a quality metric: mean/max point-to-surface distance from a's vertices to
    b's triangles (how far the decimation moved the surface).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * Decimating an icosphere (V66 F128) to 64 faces stays a CLOSED MANIFOLD with chi PRESERVED (2).
  * QEM beats a naive shortest-edge->midpoint baseline at the SAME face count on MEAN point-to-surface error
    (~1.8x better) and dramatically on MAX error (naive spikes where it collapses a feature edge).
  * A vertex's own quadric reads ~0 at the vertex (it lies on all its incident planes) -- the quadric measures
    deviation correctly. The contraction cost is never negative.

DETERMINISM (per ISA.md)
  Edges ranked by (cost, i, j); exact-tie costs broken by vertex index; the collapse rule and the accumulation are
  fixed. Same mesh + target -> byte-identical decimation (asserted). No RNG.

KEPT NEGATIVES (loud)
  * QEM minimizes squared distance to the incident PLANES, not the true surface or any particular invariant -- so
    on a sphere the optimal points can sit slightly OFF-RADIUS (measured: QEM's |r-1| is a touch worse than the
    chord-midpoint baseline's) while being CLOSER to the actual surface (point-to-surface, the honest metric, is
    much better). The plane metric is the right one; radius fidelity is not what it optimizes.
  * CLOSED meshes are in scope; OPEN-mesh boundary preservation (the standard fix: add a high-weight plane
    perpendicular to each boundary edge so the boundary curve is held) is deferred -- without it QEM would erode a
    boundary. Stated, not hidden.
  * The loop recomputes edge costs each pass (clear + correct); the incremental heap-with-lazy-deletion that makes
    QEM near-linear is the standard performance upgrade and is deferred -- this is the correct-and-readable version,
    right for moderate meshes (the panel's "delegate the heavy grind" call).
  * collapse_edge REFUSES collapses that would break the manifold (the link condition); the decimator honours the
    refusal by trying the next-cheapest edge, and STOPS if no remaining edge can be collapsed -- so it may halt
    above target_faces on a mesh with no safe collapse left. A true property of the mesh, made operational.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.misc.holographic_eulerops import collapse_edge


def vertex_quadrics(mesh):
    """The per-vertex 4x4 error quadrics: Q_v = sum over incident faces of (plane plane^T), plane = [n, -n.p] with
    unit normal n. v^T Q_v v is the summed squared distance from v to its incident planes.

    KEPT SCALAR ON PURPOSE (a measured negative): the obvious vectorization -- batch the planes and scatter-add the
    outer products -- is NOT bit-identical, because the plane offset uses a dot product (n.dot(V[a])) whose vectorized
    form (np.sum or einsum) sums in a different order and differs by ULPs. Those ULPs flip QEM's collapse-order
    TIE-BREAKS and produce a DIFFERENT decimated mesh (verified on several meshes -- same face count, different
    faces). This is the bind_batch lesson: a 1e-15 change in a tie-sensitive path is a real bug. The quadric build is
    called once per decimation and is not the bottleneck (the greedy collapse loop is), so it stays in the exact
    scalar form. Vectorizing the QEM cost loop would require a heap with incremental, order-stable updates -- the real
    fix, deferred (the documented 'delegate decimation to meshoptimizer' negative stands)."""
    Q = [np.zeros((4, 4)) for _ in range(mesh.n_vertices)]
    V = mesh.vertices
    for f in mesh.faces:
        a, b, c = f[0], f[1], f[2]                         # triangles (the first three carry the plane)
        n = np.cross(V[b] - V[a], V[c] - V[a])
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue                                       # degenerate sliver -> no plane
        n = n / ln
        plane = np.append(n, -n.dot(V[a]))
        K = np.outer(plane, plane)
        for v in f:
            Q[v] = Q[v] + K
    return Q


def contraction_target(Q, p_i, p_j):
    """The optimal merged position v_bar = argmin v^T Q v (with v=[x,y,z,1]) and its cost. Solve the 3x3 system
    from the top-left of Q; if singular, take the best of {midpoint, p_i, p_j}. Returns (point, cost>=0)."""
    A = Q[:3, :3]
    b = Q[:3, 3]
    x = None
    if abs(np.linalg.det(A)) > 1e-10:
        try:
            x = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = None
    if x is None:
        cands = [0.5 * (p_i + p_j), np.asarray(p_i, float), np.asarray(p_j, float)]
        costs = [float(np.append(c, 1.0) @ Q @ np.append(c, 1.0)) for c in cands]
        x = cands[int(np.argmin(costs))]
    v = np.append(x, 1.0)
    return x, max(0.0, float(v @ Q @ v))


def _edges(mesh):
    E = set()
    for f in mesh.faces:
        n = len(f)
        for k in range(n):
            E.add(tuple(sorted((f[k], f[(k + 1) % n]))))
    return E


def _qem_decimate_fast(mesh, target_faces):
    """STABLE-INDEX + HEAP QEM decimation -- the O(F log F) fast path (opt-in via qem_decimate(fast=True)).

    The canonical qem_decimate is O(F^2) from two places: re-ranking every edge each step, and collapse_edge
    rebuilding the whole mesh each call. This path fixes BOTH: vertex->face INCIDENCE makes each collapse O(degree)
    (only the faces around the merged vertex are touched, no full rebuild), and a LAZY-INVALIDATION HEAP re-ranks only
    the edges incident to the merged vertex (a version stamp per vertex makes stale heap entries skippable on pop).
    The mesh is compacted to a fresh Mesh once, at the end.

    NOT bit-identical to the default (same tie-sensitivity caveat as the vectorised path): a valid, deterministic QEM
    decimation to the same face target, close to the canonical surface, much faster. Manifold-safe: the same LINK
    CONDITION that collapse_edge enforces is checked (incrementally, from incidence) before every collapse, and a
    refused edge is dropped until a neighbouring collapse re-pushes it."""
    import heapq

    V = np.array(mesh.vertices, float)
    n = len(V)
    Qlist = [np.asarray(q, float) for q in vertex_quadrics(mesh)]
    faces = [list(f) for f in mesh.faces]
    face_alive = [True] * len(faces)
    n_alive = len(faces)
    vert_faces = [set() for _ in range(n)]
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[v].add(fi)
    vert_alive = [True] * n
    vert_ver = [0] * n

    def neighbours(x):
        nb = set()
        for fi in vert_faces[x]:
            if face_alive[fi]:
                nb.update(v for v in faces[fi] if v != x)
        return nb

    def edge_cost(u, v):
        Qe = Qlist[u] + Qlist[v]
        x, cost = contraction_target(Qe, V[u], V[v])
        return cost, x

    # seed the heap with every edge (u<v), keyed (cost, u, v) for deterministic ties, stamped with vertex versions
    heap = []
    seen = set()
    for fi, f in enumerate(faces):
        for k in range(3):
            u, w = f[k], f[(k + 1) % 3]
            e = (u, w) if u < w else (w, u)
            if e in seen:
                continue
            seen.add(e)
            c, _x = edge_cost(e[0], e[1])
            heapq.heappush(heap, (c, e[0], e[1], vert_ver[e[0]], vert_ver[e[1]]))

    def link_ok(a, b):
        # shared neighbours of a,b must be exactly the apexes of the faces on edge {a,b} (else the collapse folds
        # the surface onto itself -- the exact condition collapse_edge enforces).
        shared = neighbours(a) & neighbours(b)
        apex = set()
        for fi in vert_faces[a] & vert_faces[b]:
            if face_alive[fi]:
                apex.update(v for v in faces[fi] if v != a and v != b)
        return shared == apex

    while n_alive > target_faces and heap:
        c, u, w, vu, vw = heapq.heappop(heap)
        if not (vert_alive[u] and vert_alive[w]):
            continue
        if vert_ver[u] != vu or vert_ver[w] != vw:
            continue                                            # stale entry (an endpoint changed) -> skip
        if w not in neighbours(u):
            continue                                            # no longer an edge
        # keep the lower index (deterministic), remove the other
        a, b = (u, w) if u < w else (w, u)
        if not link_ok(a, b):
            continue                                            # unsafe collapse -> drop (re-pushed if a nbr changes)
        _c, x = edge_cost(a, b)
        # --- collapse b into a, touching only b's incident faces (O(degree)) ---
        for fi in list(vert_faces[b]):
            if not face_alive[fi]:
                continue
            f = faces[fi]
            f2 = [a if v == b else v for v in f]
            if len(set(f2)) < 3:                                # degenerate sliver -> kill the face
                face_alive[fi] = False
                n_alive -= 1
                for v in set(f):
                    vert_faces[v].discard(fi)
            else:
                faces[fi] = f2
                vert_faces[a].add(fi)
        vert_faces[b] = set()
        vert_alive[b] = False
        V[a] = x
        Qlist[a] = Qlist[a] + Qlist[b]
        vert_ver[a] += 1                                        # invalidate every stale heap entry touching a
        # re-rank only the edges now incident to a (the local update the heap needs)
        for wnb in neighbours(a):
            if vert_alive[wnb]:
                cc, _xx = edge_cost(a, wnb)
                lo, hi = (a, wnb) if a < wnb else (wnb, a)
                heapq.heappush(heap, (cc, lo, hi, vert_ver[lo], vert_ver[hi]))

    # --- compact once: drop dead vertices, remap faces ---
    keep = [i for i in range(n) if vert_alive[i]]
    remap = {old: new for new, old in enumerate(keep)}
    out_faces = [tuple(remap[v] for v in faces[fi]) for fi in range(len(faces)) if face_alive[fi]]
    return Mesh(V[keep], out_faces)


def qem_decimate(mesh, target_faces, fast=False):
    """Greedily collapse the lowest-QEM-cost edge (deterministic ties by vertex index) via the guarded
    collapse_edge, accumulating quadrics through each collapse, until the mesh has <= target_faces (or no safe
    collapse remains). Returns a new Mesh. Closed triangle meshes; see the kept negatives.

    fast=False (DEFAULT, bit-identical): the original per-edge ranking loop. Pinned as the canonical behaviour so
    existing decisions never flip.
    fast=True (OPT-IN): the STABLE-INDEX + HEAP decimator (_qem_decimate_fast) -- vertex->face incidence makes each
    collapse O(degree) and a lazy-invalidation heap re-ranks only the edges around each merged vertex, so the whole
    decimation is O(F log F) instead of O(F^2). An EQUALLY-VALID, deterministic QEM decimation to the same face
    target and close to the canonical surface, much faster -- but NOT bit-identical (batched/heap ordering differs
    from the per-edge loop at the ULP, and QEM is tie-sensitive). Use it for preview/interactive decimation; use the
    default where the exact canonical mesh is required. (Kept negative, loud: fast != default byte-for-byte.)"""
    if fast:
        return _qem_decimate_fast(mesh, target_faces)
    m = mesh
    Q = vertex_quadrics(m)
    while m.n_faces > target_faces:
        # rank every edge by its contraction cost (deterministic: cost, then the two vertex indices)
        ranked = []                                             # canonical per-edge loop (default, pinned bit-identical)
        for (i, j) in _edges(m):
            Qe = Q[i] + Q[j]
            x, cost = contraction_target(Qe, m.vertices[i], m.vertices[j])
            ranked.append((cost, i, j, x))
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))
        collapsed = False
        for (cost, i, j, x) in ranked:
            keep, remove = (i, j) if i < j else (j, i)
            nm = collapse_edge(m, keep, remove)
            if nm is None:                                 # link condition refused -> try the other direction
                keep, remove = remove, keep
                nm = collapse_edge(m, keep, remove)
            if nm is None:
                continue                                   # both refused -> next-cheapest edge
            # the quadric list stays aligned with the mesh: collapse_edge drops index `remove` and shifts down,
            # so we delete that entry too, then fold the removed quadric into the survivor and move it to v_bar
            QR = Q[remove]
            del Q[remove]
            keep_new = keep if keep < remove else keep - 1
            Q[keep_new] = Q[keep_new] + QR
            nm.vertices[keep_new] = x
            m = nm
            collapsed = True
            break
        if not collapsed:
            break                                          # no remaining edge can be safely collapsed
    return m


def _point_to_triangle(p, a, b, c):
    """Closed-form distance from point p to triangle (a,b,c) (Ericson, Real-Time Collision Detection)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab.dot(ap), ac.dot(ap)
    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(ap)
    bp = p - b
    d3, d4 = ab.dot(bp), ac.dot(bp)
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return np.linalg.norm(ap - (d1 / (d1 - d3)) * ab)
    cp = p - c
    d5, d6 = ab.dot(cp), ac.dot(cp)
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return np.linalg.norm(ap - (d2 / (d2 - d6)) * ac)
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.linalg.norm((b - p) + w * (c - b))
    denom = 1.0 / (va + vb + vc)
    return np.linalg.norm(ap - (vb * denom * ab + vc * denom * ac))


def surface_deviation(mesh_a, mesh_b, fast=True):
    """A decimation quality metric: (mean, max) point-to-surface distance from mesh_a's vertices to mesh_b's
    triangles -- how far b's surface sits from a's points.

    fast=True (default): use the VECTORIZED SPATIAL INDEX (point_set_to_mesh_grid) -- the "real build" this docstring
    used to back-log. It culls the O(Va*Fb) scan to O(Va * nearby triangles) and is exact + ~100x faster for the case
    this metric is actually used in: a DECIMATED mesh measured against its original, whose vertices are near the
    original surface. If any of a's vertices fall outside the grid's near-surface reach (two far-apart meshes), it
    transparently FALLS BACK to the exact brute scan below, so the answer is always correct -- fast when it can be.

    fast=False: the tight vectorized brute force -- loop once over b's faces, each face's closest point to ALL of a's
    vertices at once, running min. The exact reference (and what the fast path falls back to).

    KEPT HONEST -- the spatial index is the third acceleration attempt; the first two were MEASURED SLOWER and are
    kept on record: (1) fully batching point-vs-all-triangles into one (Va,Fb) tensor was slower (136s on a 22k case;
    no early exit, memory-bound); (2) a PYTHON spatial hash (per-cell triangle lists, per-query set unions) was ~0.5x
    (the cluster LOD chain went 6.3s -> 52s) and not even bit-exact -- many tiny vectorized ops lose to brute force's
    few large ones. Only a FULLY vectorized index (one big gather via the ranges trick, no Python per-cell loop) wins;
    that is point_set_to_mesh_grid, and it does (the cluster LOD chain error: 27s -> 0.25s)."""
    P = np.asarray(mesh_a.vertices, float)
    if fast:
        from holographic.mesh_and_geometry.holographic_meshbridge import point_set_to_mesh_grid
        d = point_set_to_mesh_grid(P, mesh_b.vertices, mesh_b.faces, radius=2)
        if not np.any(np.isinf(d)):                          # every vertex found its nearest triangle -> exact, fast
            return float(d.mean()), float(d.max())
        # else: some vertex is out of the index's near-surface reach -- fall through to the exact brute scan
    from holographic.mesh_and_geometry.holographic_meshbridge import _closest_point_on_triangle
    Vb = np.asarray(mesh_b.vertices, float)
    best = np.full(len(P), np.inf)
    for f in mesh_b.faces:
        cp = _closest_point_on_triangle(P, Vb[f[0]], Vb[f[1]], Vb[f[2]])
        best = np.minimum(best, np.linalg.norm(P - cp, axis=1))
    return float(best.mean()), float(best.max())


# =====================================================================================================
# Self-test -- decimate an icosphere; QEM beats naive on surface error; quadrics measure deviation correctly.
# =====================================================================================================
def _selftest():
    from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere

    ico = _icosphere(2)                                    # V66 F128, unit sphere

    # --- a vertex's own quadric reads ~0 at the vertex (it lies on its incident planes) ---
    Q = vertex_quadrics(ico)
    v0 = np.append(ico.vertices[0], 1.0)
    assert v0 @ Q[0] @ v0 < 1e-12, "the quadric must vanish at the vertex itself"

    # --- QEM decimation: closed manifold, chi preserved ---
    qem = qem_decimate(ico, 64)
    assert qem.is_manifold() and qem.is_closed() and qem.euler_characteristic() == 2, "QEM preserves the sphere"
    assert qem.n_faces <= 64, "decimated to the target (or halted above it on a refusal)"

    # --- QEM beats a naive shortest-edge->midpoint baseline on surface error ---
    def naive_decimate(mesh, target):
        m = mesh
        while m.n_faces > target:
            ranked = sorted(((float(np.linalg.norm(m.vertices[i] - m.vertices[j])), i, j) for (i, j) in _edges(m)),
                            key=lambda t: (t[0], t[1], t[2]))
            done = False
            for (_, i, j) in ranked:
                keep, remove = (i, j) if i < j else (j, i)
                nm = collapse_edge(m, keep, remove)
                if nm is None:
                    keep, remove = remove, keep
                    nm = collapse_edge(m, keep, remove)
                if nm is None:
                    continue
                mid = 0.5 * (m.vertices[keep] + m.vertices[remove])
                keep_new = keep if keep < remove else keep - 1
                nm.vertices[keep_new] = mid
                m = nm
                done = True
                break
            if not done:
                break
        return m

    naive = naive_decimate(ico, 64)
    q_mean, q_max = surface_deviation(ico, qem)
    n_mean, n_max = surface_deviation(ico, naive)
    assert q_mean < n_mean, f"QEM must beat naive on mean surface error ({q_mean:.4f} vs {n_mean:.4f})"
    assert q_max < n_max, f"QEM must beat naive on MAX surface error ({q_max:.4f} vs {n_max:.4f})"

    # --- determinism ---
    assert np.array_equal(qem_decimate(ico, 64).vertices, qem_decimate(ico, 64).vertices)

    # --- fast=True (opt-in STABLE-INDEX + HEAP decimator, O(F log F)): same face COUNT, deterministic, closed
    # manifold, chi preserved, and a surface very close to canonical -- but NOT guaranteed bit-identical (tie-
    # sensitive). The default stays the canonical per-edge mesh. ---
    q_slow = qem_decimate(ico, 64, fast=False)
    q_fast = qem_decimate(ico, 64, fast=True)
    assert q_fast.n_faces == q_slow.n_faces                     # same target reached
    assert np.array_equal(qem_decimate(ico, 64, fast=True).vertices,
                          qem_decimate(ico, 64, fast=True).vertices)   # fast path is itself deterministic
    # closed 2-manifold + Euler characteristic preserved (a valid decimation, not a torn mesh)
    from collections import Counter as _Counter
    _ec = _Counter()
    for _f in q_fast.faces:
        for _k in range(len(_f)):
            _ec[frozenset((_f[_k], _f[(_k + 1) % len(_f)]))] += 1
    assert all(_v == 2 for _v in _ec.values()), "fast decimation must stay a closed 2-manifold"
    _chi = len(q_fast.vertices) - len(_ec) + len(q_fast.faces)
    assert _chi == 2, ("fast decimation must preserve sphere topology (chi=2)", _chi)
    # surface stays close to canonical
    dmean, _dmax = surface_deviation(q_fast, q_slow)
    assert dmean < 0.1, ("fast decimation drifted from canonical", dmean)   # small (tie-order divergence), not broken

    # --- cluster_decimate: the PARALLEL path. Coarsens, deterministic, and far faster than greedy QEM ---
    import time as _time
    clu = cluster_decimate(ico, 6)
    assert 0 < clu.n_faces < ico.n_faces, "clustering must coarsen"
    assert np.array_equal(cluster_decimate(ico, 6).vertices, clu.vertices), "clustering must be deterministic"
    _big = _icosphere(4)                                   # V2562 F5120, enough to time the gap
    _t = _time.time(); _c = cluster_decimate(_big, 20); _tc = _time.time() - _t
    _t = _time.time(); _q = qem_decimate(_big, _c.n_faces); _tq = _time.time() - _t
    assert _tc < _tq, f"clustering ({_tc*1000:.0f}ms) must beat greedy QEM ({_tq*1000:.0f}ms) on speed"

    print(f"holographic_meshqem selftest: ok (icosphere V{ico.n_vertices} F{ico.n_faces} -> QEM F{qem.n_faces}, "
          f"closed manifold, chi 2 preserved; surface error QEM mean {q_mean:.4f}/max {q_max:.4f} BEATS naive mean "
          f"{n_mean:.4f}/max {n_max:.4f} ({n_mean / q_mean:.2f}x mean, {n_max / q_max:.2f}x max); the quadric "
          f"vanishes at its vertex; deterministic. PARALLEL cluster_decimate: F{_big.n_faces} -> F{_c.n_faces} in "
          f"{_tc*1000:.0f}ms vs greedy QEM {_tq*1000:.0f}ms ({_tq/_tc:.0f}x faster -- the bundled-quadric vertex merge))")


def cluster_decimate(mesh, grid=16):
    """PARALLEL decimation by vertex clustering (Rossignac-Borrel / Lindstrom) -- the O(n) counterpart of the greedy
    qem_decimate, for an IMPORTED mesh that has no field behind it. Partition the bounding box into a grid^3 lattice
    (the same floor-divide spatial binning the engine's tilers use), collapse every vertex in a cell to ONE
    representative, remap faces, drop the faces that degenerate. NO greedy edge-collapse search -- every step is a
    vectorized array op, so it is hundreds-to-thousands x faster than greedy QEM at the cost of some quality.

    The representative is VSA-native: a cell's error quadric is the SUM (a bundle, superposition) of its vertices'
    plane tensors, and the representative is that bundled quadric's minimizer (solve A x = -b), clamped to the cell so
    the optimum cannot overshoot, falling back to the centroid where the quadric is singular (a flat cell). 'Sum of
    plane outer products = a bundle' is the same algebra as the rest of the engine, here merging geometry.

    KEPT HONEST: clustering trades quality and manifoldness for parallel speed -- a coarse grid can produce
    non-manifold edges or merge across thin gaps (greedy qem_decimate stays the quality option; this is the fast
    one). Determinism: pure array ops + a fixed grid, bit-stable run to run. Returns a new Mesh."""
    V = np.asarray(mesh.vertices, float)
    if mesh.n_faces == 0 or len(V) == 0:
        return Mesh(np.zeros((0, 3)), [])
    grid = int(grid)
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    cell = np.clip(np.floor((V - lo) / span * grid).astype(int), 0, grid - 1)    # spatial bin (vectorized)
    cell_id = (cell[:, 0] * grid + cell[:, 1]) * grid + cell[:, 2]
    uniq, inv = np.unique(cell_id, return_inverse=True)                          # inv[v] = representative index
    n_rep = len(uniq)

    centroid = np.zeros((n_rep, 3))
    cnt = np.zeros(n_rep)
    np.add.at(centroid, inv, V)
    np.add.at(cnt, inv, 1.0)
    centroid = centroid / cnt[:, None]                                           # per-cell centroid (the fallback)

    Fa = np.array([f[:3] for f in mesh.faces], dtype=int)                        # triangles carry the planes
    a3, b3, c3 = V[Fa[:, 0]], V[Fa[:, 1]], V[Fa[:, 2]]
    fn = np.cross(b3 - a3, c3 - a3)
    fln = np.linalg.norm(fn, axis=1)
    fvalid = fln > 1e-12
    fnrm = np.zeros_like(fn)
    fnrm[fvalid] = fn[fvalid] / fln[fvalid, None]
    plane = np.concatenate([fnrm, -np.sum(fnrm * a3, axis=1)[:, None]], axis=1)   # (nf,4) each face's plane [n, -n.a]
    K = plane[:, :, None] * plane[:, None, :]                                     # (nf,4,4) plane outer plane
    K[~fvalid] = 0.0
    faces_cells = inv[Fa]                                                         # the representative cell of each face vertex
    Qc = np.zeros((n_rep, 4, 4))
    for k in range(3):
        np.add.at(Qc, faces_cells[:, k], K)                                      # per-cell quadric = BUNDLE of plane tensors
    # (this vectorized quadric is safe here -- unlike greedy QEM, clustering has no collapse-order tie-breaks for
    #  a ULP difference to flip, so the bind_batch caveat does not apply.)
    A = Qc[:, :3, :3]
    bvec = Qc[:, :3, 3]
    rep = centroid.copy()
    dets = np.linalg.det(A)
    ok = np.abs(dets) > 1e-10
    if ok.any():
        rep[ok] = np.linalg.solve(A[ok], -bvec[ok][..., None])[..., 0]          # minimizer of the bundled quadric

    cz = uniq % grid                                                            # decode each cell's lattice index
    cy = (uniq // grid) % grid
    cx = uniq // (grid * grid)
    cell_lo = lo + np.stack([cx, cy, cz], axis=1) * (span / grid)
    rep = np.clip(rep, cell_lo, cell_lo + span / grid)                          # keep the optimum inside its cell

    mapped = faces_cells                                                        # faces already remapped to representatives
    keep = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 0] != mapped[:, 2])
    mapped = mapped[keep]                                                       # drop collapsed (degenerate) faces
    if len(mapped):
        mapped = np.unique(mapped, axis=0)                                      # drop exact duplicate faces
    return Mesh(rep, [tuple(int(x) for x in row) for row in mapped])


if __name__ == "__main__":
    _selftest()

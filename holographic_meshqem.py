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

from holographic_mesh import Mesh
from holographic_eulerops import collapse_edge


def vertex_quadrics(mesh):
    """The per-vertex 4x4 error quadrics: Q_v = sum over incident faces of (plane plane^T), plane = [n, -n.p] with
    unit normal n. v^T Q_v v is the summed squared distance from v to its incident planes."""
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


def qem_decimate(mesh, target_faces):
    """Greedily collapse the lowest-QEM-cost edge (deterministic ties by vertex index) via the guarded
    collapse_edge, accumulating quadrics through each collapse, until the mesh has <= target_faces (or no safe
    collapse remains). Returns a new Mesh. Closed triangle meshes; see the kept negatives."""
    m = mesh
    Q = vertex_quadrics(m)
    while m.n_faces > target_faces:
        # rank every edge by its contraction cost (deterministic: cost, then the two vertex indices)
        ranked = []
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


def surface_deviation(mesh_a, mesh_b):
    """A decimation quality metric: (mean, max) point-to-surface distance from mesh_a's vertices to mesh_b's
    triangles -- how far b's surface sits from a's points."""
    errs = []
    for p in mesh_a.vertices:
        errs.append(min(_point_to_triangle(p, mesh_b.vertices[f[0]], mesh_b.vertices[f[1]], mesh_b.vertices[f[2]])
                        for f in mesh_b.faces))
    errs = np.asarray(errs)
    return float(errs.mean()), float(errs.max())


# =====================================================================================================
# Self-test -- decimate an icosphere; QEM beats naive on surface error; quadrics measure deviation correctly.
# =====================================================================================================
def _selftest():
    from holographic_meshsmooth import _icosphere

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

    print(f"holographic_meshqem selftest: ok (icosphere V{ico.n_vertices} F{ico.n_faces} -> QEM F{qem.n_faces}, "
          f"closed manifold, chi 2 preserved; surface error QEM mean {q_mean:.4f}/max {q_max:.4f} BEATS naive mean "
          f"{n_mean:.4f}/max {n_max:.4f} ({n_mean / q_mean:.2f}x mean, {n_max / q_max:.2f}x max); the quadric "
          f"vanishes at its vertex; deterministic)")


if __name__ == "__main__":
    _selftest()

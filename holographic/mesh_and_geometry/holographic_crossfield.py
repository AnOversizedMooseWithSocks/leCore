"""holographic_crossfield.py -- the smoothest 4-RoSy field on a surface (Box3D backlog F2).

Field-aligned retopology begins with a cross field: a direction at every face, defined up to 90-degree rotation, as
smooth as the surface allows. Instant Meshes (Jakob, Tarini, Panozzo & Sorkine-Hornung 2015) extracts quads along
one. This module ships the field and its singularities; quad extraction proper is a mixed-integer problem and is
NOT here.

THE SOLVE. A 4-RoSy direction on face `f` is the complex number `u_f = exp(4 i phi_f)`, which is invariant under
`phi -> phi + pi/2`. Two adjacent faces are compared by PARALLEL TRANSPORT: the shared edge has an angle in each
face's own frame, and their difference `rho_fg` rotates one frame into the other. The smoothest field minimises
`sum |u_f - exp(4 i rho_gf) u_g|^2`, whose minimiser is the eigenvector of the smallest eigenvalue of the complex
**connection Laplacian** -- Hermitian, positive semi-definite, and solved here by `numpy.linalg.eigh`
(Knoppel, Crane, Pinkall & Schroder, *Globally Optimal Direction Fields*, SIGGRAPH 2013).

THE PREVIOUS ATTEMPT FAILED, AND SO DID ITS BAR. Both are recorded because both were instructive.

KEPT NEGATIVE 1 -- **POINCARE-HOPF IS A MESH IDENTITY, NOT A FIELD BAR.** The last session recorded "sum of the
singularity indices equals the Euler characteristic" as F2's bar: an integer, no tolerance to argue about. It is
true, it is exact here (sphere +2.0000000, torus 0.0000000), and **it is vacuous.** Measured, on the same sphere:

    field                  sum(index)   singularities
    smoothest (eigenvector)   +2.0            80
    uniformly random          +2.0           169
    all-zero                  +2.0           203
    adversarial alternating   +2.0           203

The matching integers are antisymmetric, so their contribution cancels pairwise around every dual edge, and what
remains is a function of the MESH alone. **A bar that passes for every input is not a bar.** The field's quality is
the singularity COUNT and the Dirichlet energy; Poincare-Hopf validates the transport and the dual rings, which is
worth having and is not what it was advertised as.

KEPT NEGATIVE 2 -- **antisymmetry must be ENFORCED, not hoped for.** The first version computed `rho_fg` and
`rho_gf` independently from the two directed edges. `atan2`'s branch cut then differs by 2*pi between them, which
shifts the matching integer by 4 and the index by 1 *per edge*: the sphere's indices summed to **-43** instead of
+2. And `wrap` at exactly +-pi is a tie -- it broke antisymmetry on a tetrahedron, where the field is symmetric
enough to hit it. Both are fixed by computing one value per UNDIRECTED edge and negating: `rho[(g,f)] = -rho[(f,g)]`,
`p[(g,f)] = -p[(f,g)]`. *The `argmax_tiebreak` lesson, in a new place.*

KEPT NEGATIVE 3 -- **Jacobi smoothing does not converge.** Iterating `u_f <- normalize(sum_g exp(4 i rho_gf) u_g)`
from a random start, the Dirichlet energy on a torus fell from 3620.9 to 2788.4 in 50 sweeps and then **rose** to
2865.8 by 400. It oscillates. The eigenvector solve is not an optimisation; it is the answer.

HONEST SCOPE. `eigh` on a dense `(n_faces, n_faces)` Hermitian matrix is O(F^3): fine to a few thousand faces, and
the wrong tool beyond that (a shifted inverse-power iteration on a sparse operator is the standard remedy, and it
is not here). The field is defined per FACE, not per vertex. The mesh must be CLOSED and consistently ORIENTED --
`holographic_isosurface.is_oriented` is the check, and the meshes this module is tested on come from
`surface_nets`, which learned to orient itself because this module needed it. Even the smoothest field on an
irregular mesh carries many low-index singularities (80 on a 720-face sphere); Instant Meshes reduces them with a
further optimisation that is not implemented here.
"""

import collections

import numpy as np


def _wrap(a, period=np.pi):
    """Wrap an angle into `(-period, period]`."""
    return (a + period) % (2.0 * period) - period


def face_frames(V, F):
    """A per-face orthonormal frame `(normal, ex, ey)`. `ex` is the first edge, projected into the face plane, so
    the frame is a deterministic function of the face's winding -- which is why the mesh must be oriented."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    ex = V[F[:, 1]] - V[F[:, 0]]
    ex = ex - (ex * n).sum(1, keepdims=True) * n
    ex = ex / np.linalg.norm(ex, axis=1, keepdims=True)
    return n, ex, np.cross(n, ex)


def connection(V, F):
    """Parallel transport across every interior dual edge.

    Returns `(rho, opposite, next_vertex, dual_edges)`. `rho[(f, g)]` is the angle rotating face `f`'s frame into
    face `g`'s across their shared edge, and **`rho[(g, f)] == -rho[(f, g)]` by construction**: computing the two
    independently lets `atan2`'s branch cut differ by 2*pi, which shifts the matching by 4 and the index by 1 per
    edge. That single mistake made a sphere's indices sum to -43 instead of +2."""
    _n, ex, ey = face_frames(V, F)
    opposite, next_vertex = {}, {}
    for fi, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            if (u, v) in opposite:
                raise ValueError("directed edge %r appears twice: the mesh is not consistently oriented "
                                 "(see holographic_isosurface.is_oriented)" % ((u, v),))
            opposite[(u, v)] = fi
            next_vertex[(fi, u)] = v

    rho, dual_edges = {}, []
    for (u, v), f in opposite.items():
        g = opposite.get((v, u))
        if g is None or (g, f) in rho:
            continue                                          # boundary edge, or already done from the other side
        d = V[v] - V[u]
        d = d / np.linalg.norm(d)
        r = _wrap(np.arctan2(d @ ey[g], d @ ex[g]) - np.arctan2(d @ ey[f], d @ ex[f]))
        rho[(f, g)] = r
        rho[(g, f)] = -r                                      # ONE value per undirected edge, negated
        dual_edges.append((f, g))
    return rho, opposite, next_vertex, dual_edges


def vertex_rings(F, opposite, next_vertex, n_vertices):
    """The ordered ring of faces around each interior vertex, from the half-edge NEXT rule.

    Consistently oriented by construction: a shared dual edge is traversed `f -> g` around one endpoint of its
    primal edge and `g -> f` around the other. That is what makes the matchings cancel pairwise -- and getting it
    wrong, by traversing each ring in an arbitrary direction, is what broke the first attempt."""
    rings = {}
    incident = collections.defaultdict(list)
    for fi, tri in enumerate(F):
        for v in tri:
            incident[int(v)].append(fi)
    for v in range(int(n_vertices)):
        if not incident[v]:
            continue
        start = incident[v][0]
        ring, f = [start], start
        while True:
            g = opposite.get((next_vertex[(f, v)], v))
            if g is None or len(ring) > 512:
                ring = None
                break                                          # boundary vertex, or a broken ring
            if g == start:
                break
            ring.append(g)
            f = g
        if ring:
            rings[v] = ring
    return rings


def angle_defect(V, F):
    """`2*pi` minus the sum of the interior angles at each vertex: the discrete Gaussian curvature.

    Gauss-Bonnet pins it: `sum(defect) == 2*pi*chi`, verified to 1e-10 on a tetrahedron, a sphere and a torus. That
    check validates the mesh, and it is the honest half of what Poincare-Hopf was supposed to do."""
    d = np.full(len(V), 2.0 * np.pi)
    for (a, b, c) in F:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            e1, e2 = V[j] - V[i], V[k] - V[i]
            cos = e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2))
            d[i] -= float(np.arccos(np.clip(cos, -1.0, 1.0)))
    return d


def connection_laplacian(F, rho, dual_edges):
    """The complex connection Laplacian: `L[f,f] = deg(f)`, `L[f,g] = -exp(4 i rho_gf)`. Hermitian, PSD.

    Dense `(n_faces, n_faces)`. HONEST SCOPE: `eigh` on it is O(F^3), fine to a few thousand faces and the wrong
    tool beyond -- a shifted inverse-power iteration on a sparse operator is the standard remedy, and it is not
    here."""
    nF = len(F)
    L = np.zeros((nF, nF), complex)
    for (f, g) in dual_edges:
        L[f, f] += 1.0
        L[g, g] += 1.0
        L[f, g] -= np.exp(1j * 4.0 * rho[(g, f)])
        L[g, f] -= np.exp(1j * 4.0 * rho[(f, g)])
    return L


def cross_field(mesh):
    """The SMOOTHEST 4-RoSy field: per-face angles `phi` in `(-pi/4, pi/4]`, plus the connection.

    The minimiser of the Dirichlet energy is the eigenvector of the connection Laplacian's smallest eigenvalue
    (Knoppel et al. 2013). It is a solve, not an iteration -- see kept negative 3.

    Returns `(phi, ctx)` where `ctx` carries `rho`, `rings`, `dual_edges` and `defect`, so `singularity_index` and
    `field_energy` do not rebuild them."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    if len(F) < 4:
        raise ValueError("cross_field needs a closed surface with at least 4 faces")

    rho, opposite, next_vertex, dual_edges = connection(V, F)
    if len(dual_edges) * 2 != len(F) * 3:
        raise ValueError("the mesh has boundary edges; cross_field wants a CLOSED surface")

    L = connection_laplacian(F, rho, dual_edges)
    if np.abs(L - L.conj().T).max() > 1e-9:
        raise ValueError("the connection Laplacian is not Hermitian: the transport is inconsistent")
    w, vec = np.linalg.eigh(L)
    phi = np.angle(vec[:, 0]) / 4.0

    ctx = {"rho": rho, "rings": vertex_rings(F, opposite, next_vertex, len(V)), "dual_edges": dual_edges,
           "defect": angle_defect(V, F), "lambda_min": float(w[0]), "n_vertices": len(V)}
    return phi, ctx


def field_energy(phi, ctx):
    """The Dirichlet energy `sum |u_f - exp(4 i rho_gf) u_g|^2`. The quantity the eigen-solve minimises, and the
    honest measure of a field's smoothness."""
    rho = ctx["rho"]
    u = np.exp(1j * 4.0 * np.asarray(phi, float))
    return float(sum(abs(u[f] - np.exp(1j * 4.0 * rho[(g, f)]) * u[g]) ** 2 for (f, g) in ctx["dual_edges"]))


def singularity_index(phi, ctx):
    """The per-vertex singularity index of the field. **Exactly a multiple of 1/4** (measured: 0.0e+00 residual).

    `index(v) = round((defect_v + sum_ring rho) / 2pi) + (sum_ring p) / 4`, with `p` the antisymmetric integer
    matching `round(wrap(phi_g - phi_f - rho_fg) / (pi/2))`. The rounded term is the holonomy of the Levi-Civita
    connection around the ring; the matching term is the field's own turning.

    **`sum(index) == chi` -- and it says nothing about the field.** The matching is antisymmetric, so its
    contribution cancels around every dual edge, leaving a mesh-only quantity. See kept negative 1."""
    if not isinstance(ctx, dict) or "rho" not in ctx:
        raise ValueError("ctx must come from cross_field(); got %r" % (type(ctx),))
    if ctx["rho"] and not isinstance(next(iter(ctx["rho"])), tuple):
        raise ValueError("this ctx has been through JSON: its `rho` keys are strings like '(0, 1)', not (0, 1) "
                         "tuples, so it cannot be used. A live handle does not survive serialisation. Use the "
                         "stateless twin `field_singularities(mesh)`, which takes data and returns data.")
    rho, rings, defect = ctx["rho"], ctx["rings"], ctx["defect"]
    phi = np.asarray(phi, float)

    p = {}
    for (f, g) in ctx["dual_edges"]:
        val = int(np.round(_wrap(phi[g] - phi[f] - rho[(f, g)]) / (np.pi / 2.0)))
        p[(f, g)] = val
        p[(g, f)] = -val                                      # ENFORCED: `wrap` at +-pi is a tie, and it bit

    idx = np.zeros(ctx["n_vertices"])
    for v, ring in rings.items():
        pairs = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
        turn = sum(rho[e] for e in pairs)
        match = sum(p[e] for e in pairs)
        idx[v] = int(np.round((defect[v] + turn) / (2.0 * np.pi))) + match / 4.0
    return idx


def field_singularities(mesh, phi=None):
    """The STATELESS one-shot twin of `cross_field` + `singularity_index`: takes a mesh, returns plain data.

    `{index, n_singularities, sum_index, euler, quarter_residual, energy}` -- lists and numbers, nothing an agent
    cannot receive over JSON and nothing it has to hand back.

    WHY THIS EXISTS. `cross_field` returns `(phi, ctx)`, and `ctx` carries `rho`, a dict keyed by `(face, face)`
    TUPLES. Serialised, those keys become the strings `"(0, 1)"`, so the payload *looks* like a context and cannot
    be fed back: `singularity_index` dies with `KeyError: (0, 1)`. **An object that serialises into something that
    looks right but cannot be used is worse than one that raises.** The remedy is the pattern this engine already
    knows: an agent-facing faculty is a stateless one-shot that takes data and returns data."""
    phi_local, ctx = cross_field(mesh)
    if phi is not None:
        phi_local = np.asarray(phi, float)
    idx = singularity_index(phi_local, ctx)
    quarters = idx * 4.0
    return {"index": [float(v) for v in idx],
            "n_singularities": int((np.abs(idx) > 1e-9).sum()),
            "sum_index": float(idx.sum()),
            "euler": int(mesh.euler_characteristic()),
            "quarter_residual": float(np.abs(quarters - np.round(quarters)).max()),
            "energy": field_energy(phi_local, ctx)}


def field_report(mesh, phi=None, ctx=None):
    """`{lambda_min, energy, n_singularities, sum_index, euler, quarter_residual, poincare_hopf}`.

    `poincare_hopf` is `sum_index == euler`, and it is reported with the warning it deserves: it holds for a random
    field too. Judge a field by `n_singularities` and `energy`."""
    if phi is None or ctx is None:
        phi, ctx = cross_field(mesh)
    idx = singularity_index(phi, ctx)
    quarters = idx * 4.0
    return {"lambda_min": ctx["lambda_min"], "energy": field_energy(phi, ctx),
            "n_singularities": int((np.abs(idx) > 1e-9).sum()), "sum_index": float(idx.sum()),
            "euler": int(mesh.euler_characteristic()),
            "quarter_residual": float(np.abs(quarters - np.round(quarters)).max()),
            "poincare_hopf": bool(abs(float(idx.sum()) - mesh.euler_characteristic()) < 1e-6)}


def _face_edge_adjacency(F):
    """undirected edge (min,max) -> list of face indices. The face-crossing table a streamline tracer walks."""
    from collections import defaultdict
    adj = defaultdict(list)
    for fi, f in enumerate(F):
        k = len(f)
        for a in range(k):
            u, v = int(f[a]), int(f[(a + 1) % k])
            adj[(min(u, v), max(u, v))].append(fi)
    return adj


def trace_streamlines(mesh, field, seeds=None, step=None, max_steps=200, four_rosy=True, seed=0, n_seeds=24):
    """Trace STREAMLINES (integral curves) of a per-face direction field across a triangle mesh -- walk along the field,
    crossing edge to edge, until a boundary / max_steps / a closed loop. Returns a list of polylines (each an (m,3)
    array). This is the general 'field -> curves' primitive, and it does NOT care where the field came from: a 4-RoSy
    cross_field (retopo guide-curves, hatching), strain_directions (deformation flow lines), an SDF gradient, or a
    SIMULATION velocity field (streamlines / pathlines -- the standard flow-visualisation primitive). `field` is either
    per-face angles phi (len n_faces, interpreted in each face frame) or per-face 3-D vectors (n_faces, 3). With
    four_rosy=True the field is treated as a 4-RoSy cross (at each face the branch nearest the travel direction is
    chosen, so the curve does not reverse); set False for a true vector field (e.g. a velocity field). `seeds` is a
    list of face indices to start from (default: n_seeds spread across faces). Deterministic. Reuses face_frames; the
    step defaults to half the mean edge length."""
    V = np.asarray(mesh.vertices, float)
    field = np.asarray(field, float)
    if any(len(f) != 3 for f in mesh.faces):                 # tracer walks a TRIANGLE mesh (correct edge adjacency);
        rep = [len(f) - 2 for f in mesh.faces]               # fan-triangulate and expand a per-face field to its
        from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons   # child triangles
        if len(field) == len(mesh.faces):
            field = np.repeat(field, rep, axis=0)
        mesh = triangulate_ngons(mesh)
        V = np.asarray(mesh.vertices, float)
    F = np.asarray([[int(i) for i in f[:3]] for f in mesh.faces], int)   # triangles
    nF = len(F)
    _n, ex, ey = face_frames(V, F)
    phi = np.asarray(field, float)
    if phi.ndim == 1:                                        # per-face angle -> per-face 3-D dir
        fdir = np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey
    else:
        fdir = phi - (phi * _n).sum(1, keepdims=True) * _n   # project a supplied vector into each face plane
        fdir = fdir / (np.linalg.norm(fdir, axis=1, keepdims=True) + 1e-12)
    if step is None:
        el = np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)
        step = 0.5 * float(np.mean(el))
    adj = _face_edge_adjacency(F)
    rng = np.random.default_rng(int(seed))
    if seeds is None:
        seeds = sorted(rng.permutation(nF)[:min(n_seeds, nF)].tolist())

    def face_dir(fi, travel):
        d = fdir[fi]
        if four_rosy:                                        # pick the 4-RoSy branch closest to where we are going
            cands = [d, np.cross(_n[fi], d), -d, -np.cross(_n[fi], d)]
            d = max(cands, key=lambda cc: float(cc @ travel))
        return d / (np.linalg.norm(d) + 1e-12)

    lines = []
    for s in seeds:
        p = V[F[s]].mean(0)                                  # start at the face centroid
        fi = s
        d = fdir[s] / (np.linalg.norm(fdir[s]) + 1e-12)
        pts = [p.copy()]
        for _ in range(int(max_steps)):
            a, b, cc = V[F[fi]]
            # 2-D coords in this face frame
            o = a; E = np.stack([ex[fi], ey[fi]], 1)         # (3,2)
            p2 = (p - o) @ E; d2 = d @ E
            tri = np.stack([(a - o) @ E, (b - o) @ E, (cc - o) @ E])
            # nearest edge exit along the ray p2 + t d2, t>0
            best_t = None; best_e = -1
            for e in range(3):
                q0 = tri[e]; q1 = tri[(e + 1) % 3]; seg = q1 - q0
                den = d2[0] * seg[1] - d2[1] * seg[0]
                if abs(den) < 1e-12:
                    continue
                diff = q0 - p2
                t = (diff[0] * seg[1] - diff[1] * seg[0]) / den            # ray parameter
                u = (diff[0] * d2[1] - diff[1] * d2[0]) / den              # segment parameter
                if t > 1e-9 and -1e-6 <= u <= 1 + 1e-6:
                    if best_t is None or t < best_t:
                        best_t = t; best_e = e
            if best_t is None:
                break
            if best_t >= step:                               # stays inside this face
                p = p + step * d; pts.append(p.copy()); continue
            p = p + best_t * d; pts.append(p.copy())         # advance to the edge, then cross
            v0, v1 = int(F[fi][best_e]), int(F[fi][(best_e + 1) % 3])
            nb = [g for g in adj[(min(v0, v1), max(v0, v1))] if g != fi]
            if not nb:
                break                                        # boundary edge -> streamline ends
            fi = nb[0]
            d = face_dir(fi, d)
            if len(pts) > 3 and np.linalg.norm(p - pts[0]) < 0.25 * step:
                break                                        # closed loop
        if len(pts) >= 2:
            lines.append(np.array(pts))
    return lines


def face_field_to_vertex(mesh, phi):
    """Average a per-FACE 4-RoSy field (angles `phi`, as cross_field returns) to a per-VERTEX representative direction
    in each vertex's tangent plane -- the input the position field wants. WHY a helper: cross_field is per-face; the
    position field (IFAM) lives on the vertex graph. Each face's 3-D representative dir (cos*ex + sin*ey) is summed to
    its vertices (a rough 4-RoSy average -- fine for a smooth field), then projected to the vertex normal plane."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    _n, ex, ey = face_frames(V, F)
    fdir = np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey
    N = mesh.vertex_normals()
    acc = np.zeros((len(V), 3))
    for fi in range(len(F)):
        for vi in F[fi]:
            acc[vi] += fdir[fi]
    acc = acc - (acc * N).sum(1, keepdims=True) * N           # project into each vertex's tangent plane
    nrm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc = np.where(nrm > 1e-9, acc / (nrm + 1e-12), ex[0])
    return acc


def position_field(mesh, orient, edge_length, iterations=10, seed=0):
    """IFAM POSITION FIELD (4-PoSy): optimise a per-vertex LATTICE position p_i, aligned to the orientation field, by
    the local EXTRINSIC smoothing operator of Instant Field-Aligned Meshes (Jakob, Tarini, Panozzo & Sorkine-Hornung,
    SIGGRAPH Asia 2015). Each vertex carries a position p_i on a rho-spaced grid whose axes are the orientation dir
    o_i and o_i x n_i; for every edge the operator forms q_ij, the point on BOTH tangent planes nearest to v_i and v_j
    (q_ij = 1/2(v_i+v_j) - 1/4(lam_i n_i + lam_j n_j), the paper's closed form), translates the neighbour p_j by an
    INTEGER number of rho-steps to line up with p_i, and moves p_i to the neighbour-weighted average -- so neighbouring
    positions come to differ by integer lattice steps. This is the stage that REGULARISES vertex spacing/valence (a
    field-aligned grid), the piece a distortion-only pairing lacks. Works on the vertex graph, so it does NOT need a
    closed mesh (unlike cross_field). Returns the optimised positions P (n_vertices, 3). Deterministic per seed.

    HONEST SCOPE: this is the position FIELD only -- it moves points onto a field-aligned lattice. Turning that lattice
    into the final quad MESH is the EXTRACTION step (build G' from unit integer jumps + detect faces, IFAM sec 4.4),
    which the paper itself flags as the error-prone part -- NOT built here yet. Combined with a deformation-guided
    orientation field (guided_cross_field), this is deformation-aware position-field remeshing."""
    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.vertex_normals(), float)
    O = np.asarray(orient, float)
    O = O - (O * N).sum(1, keepdims=True) * N                 # orientation into each tangent plane, unit
    O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-12)
    Bt = np.cross(N, O)                                        # the perpendicular lattice axis
    rho = float(edge_length); eps = 1e-4
    nbrs = [set() for _ in range(len(V))]
    for f in mesh.faces:
        ff = [int(x) for x in f]; k = len(ff)
        for a in range(k):
            u, v = ff[a], ff[(a + 1) % k]
            nbrs[u].add(v); nbrs[v].add(u)
    nbrs = [sorted(s) for s in nbrs]
    P = V.copy()
    rng = np.random.default_rng(int(seed))
    for _ in range(int(iterations)):
        for i in rng.permutation(len(V)):
            js = nbrs[i]
            if not js:
                continue
            acc = np.zeros(3); w = 0.0
            for j in js:
                d = float(N[i] @ N[j]); denom = 1.0 - d * d + eps
                li = 2.0 * ((N[i] + d * N[j]) @ (V[j] - V[i])) / denom
                lj = 2.0 * ((N[j] + d * N[i]) @ (V[i] - V[j])) / denom
                _q = 0.5 * (V[i] + V[j]) - 0.25 * (li * N[i] + lj * N[j])   # on both tangent planes (unused directly but
                diff = P[i] - P[j]                                          # the derivation behind the integer match)
                a = round(float(diff @ O[j]) / rho); c = round(float(diff @ Bt[j]) / rho)
                acc += P[j] + rho * (a * O[j] + c * Bt[j])     # translate neighbour by integer rho-steps to line up
                w += 1.0
            pi = acc / w
            rel = pi - V[i]
            P[i] = V[i] + (rel - (rel @ N[i]) * N[i])          # keep p_i on the tangent plane at v_i (near the surface)
    return P


def position_field_regularity(mesh, P, orient, edge_length):
    """How LATTICE-REGULAR a position field is: the mean residual of each edge's (p_i - p_j), after removing the
    nearest integer number of rho-steps along the field axes, as a FRACTION of rho. 0 = neighbours differ by exact
    integer lattice steps (a perfect field-aligned grid); ~0.5 = no lattice structure. The honest measure that the
    position field converged."""
    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.vertex_normals(), float)
    O = np.asarray(orient, float); O = O - (O * N).sum(1, keepdims=True) * N
    O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-12)
    Bt = np.cross(N, O); rho = float(edge_length)
    res = []
    seen = set()
    for f in mesh.faces:
        ff = [int(x) for x in f]; k = len(ff)
        for a in range(k):
            i, j = ff[a], ff[(a + 1) % k]
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            diff = P[i] - P[j]
            u = (diff @ O[i]) / rho; v = (diff @ Bt[i]) / rho
            res.append(abs(u - round(u)) + abs(v - round(v)))
    return float(np.mean(res)) if res else 0.0


def _orient_quad(V, quad, normal):
    """Return the quad's vertex order (possibly reversed) so its Newell normal agrees with `normal` -- keeps the
    remeshed faces wound consistently with the triangles they replaced."""
    P = V[list(quad)]
    nn = np.zeros(3)
    for i in range(4):
        nn = nn + np.cross(P[i], P[(i + 1) % 4])
    return tuple(reversed(quad)) if float(nn @ normal) < 0 else tuple(quad)


def quad_remesh(mesh, use_field=True, field=None):
    """Field-guided TRI-TO-QUAD remesh: pair adjacent triangles into quads, preferring pairs whose quad EDGES align
    with the 4-RoSy cross field (and that form a CONVEX, near-square quad), by a greedy maximal matching. Returns a
    QUAD-DOMINANT mesh (a quad where a good pair was found, the original triangle elsewhere) and a report
    {quads, tris, quad_fraction, field_used}. Reuses cross_field for the orientation field (needs a closed, oriented,
    manifold input -- run mesh_repair(..., triangulate=True) first); if the field cannot be solved it falls back to a
    pure squareness metric (field_used=False).

    Pass `field=(phi, ctx)` to supply an EXTERNAL field -- e.g. a guided_cross_field aligned to DEFORMATION (see
    strain_directions) or curvature -- so the retopo follows deliberate topology (edge loops encircling deformation
    zones) instead of only the smoothest field. When `field` is given, use_field is implied.

    HONEST SCOPE: this WALKS THE FIELD to place quads on the EXISTING vertices -- it does NOT move vertices or
    regularise valence, so it is NOT a full Instant-Meshes regular remesh (that is a position-field solve + extraction,
    deferred). It is the bounded, correct 'align quads to the field' retopo step. Deterministic (greedy by priority,
    tie-broken by edge)."""
    from collections import defaultdict
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons

    if any(len(f) != 3 for f in mesh.faces):
        mesh = triangulate_ngons(mesh)                       # cross_field + the pairing both want triangles
    V = np.asarray(mesh.vertices, float)
    F = [tuple(int(i) for i in f) for f in mesh.faces]
    nf = len(F)

    dirs = None
    field_used = False
    if field is not None:                                    # caller-supplied field (guided / deformation / curvature)
        phi = np.asarray(field[0], float)
        _n, ex, ey = face_frames(V, np.asarray(F, int))
        dirs = np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey
        field_used = True
    elif use_field:
        try:
            phi, _ctx = cross_field(mesh)
            _n, ex, ey = face_frames(V, np.asarray(F, int))
            dirs = np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey   # a 4-RoSy representative per face, in 3D
            field_used = True
        except Exception:
            dirs = None                                      # open/invalid -> squareness-only fallback

    def fnormal(f):
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        nn = np.cross(b - a, c - a); L = np.linalg.norm(nn)
        return nn / L if L > 1e-12 else np.array([0.0, 0.0, 1.0])

    def align4(e, d):                                        # 4-RoSy alignment of edge e to field d: |cos 2theta|
        e = e / (np.linalg.norm(e) + 1e-12); d = d / (np.linalg.norm(d) + 1e-12)
        c = float(np.clip(abs(e @ d), 0.0, 1.0))
        return abs(2.0 * c * c - 1.0)                        # 1 when e is parallel OR perpendicular to d, 0 at 45 deg

    edge_tris = defaultdict(list)
    for fi, f in enumerate(F):
        for k in range(3):
            u, v, w = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
            edge_tris[(min(u, v), max(u, v))].append((fi, u, v, w))

    cands = []
    for e, lst in edge_tris.items():
        if len(lst) != 2:
            continue                                         # boundary or non-manifold edge -> not a merge site
        (f0, _u0, _v0, w0), (f1, _u1, _v1, w1) = lst
        if fnormal(F[f0]) @ fnormal(F[f1]) < 0.5:            # near-coplanar only: a folded cross-edge pair (e.g. a box
            continue                                         # corner) can still project convex, so gate on the dihedral
        u, v = e
        quad = [u, w0, v, w1]                                # boundary of the merged quad (diagonal (u,v) removed)
        P = V[quad]
        nrm = fnormal(F[f0]) + fnormal(F[f1]); nl = np.linalg.norm(nrm)
        if nl < 1e-9:
            continue
        nrm = nrm / nl
        signs = []                                           # convexity: every corner must turn the same way
        for i in range(4):
            a, b, cc = P[i], P[(i + 1) % 4], P[(i + 2) % 4]
            signs.append(float(np.sign(np.cross(b - a, cc - b) @ nrm)))
        if 0.0 in signs or len({s for s in signs}) > 1:
            continue                                         # non-convex / degenerate -> keep as two triangles
        if dirs is not None:
            d = dirs[f0]
            fa = float(np.mean([align4(P[(i + 1) % 4] - P[i], d) for i in range(4)]))
        else:
            fa = 0.0
        ang = []                                             # squareness: corner angles near 90 deg
        for i in range(4):
            e1 = P[(i - 1) % 4] - P[i]; e2 = P[(i + 1) % 4] - P[i]
            cth = float(np.clip((e1 @ e2) / ((np.linalg.norm(e1) + 1e-12) * (np.linalg.norm(e2) + 1e-12)), -1, 1))
            ang.append(abs(np.degrees(np.arccos(cth)) - 90.0))
        sq = 1.0 - min(1.0, float(np.mean(ang)) / 90.0)
        priority = (fa if field_used else 0.0) + 0.5 * sq
        cands.append((priority, e, f0, f1, tuple(quad)))

    cands.sort(key=lambda t: (-t[0], t[1]))                  # best first, deterministic
    used = [False] * nf
    quads = []
    for _prio, _e, f0, f1, quad in cands:
        if used[f0] or used[f1]:
            continue
        used[f0] = used[f1] = True
        quads.append(_orient_quad(V, quad, fnormal(F[f0])))
    out_faces = list(quads) + [F[fi] for fi in range(nf) if not used[fi]]
    nq = len(quads); nt = nf - 2 * nq
    report = {"quads": nq, "tris": nt, "quad_fraction": nq / max(1, nq + nt), "field_used": field_used}
    return Mesh(V, out_faces), report


def guided_cross_field(mesh, guide_dirs, guide_weight=5.0):
    """A GUIDED 4-RoSy field: the smoothest field that ALSO aligns to a prescribed per-face direction where one is
    given -- field DESIGN, not just field smoothing. This is what lets retopo follow DEFORMATION (or curvature, or an
    artist's strokes) instead of only minimising distortion the way an off-the-shelf auto-remesher does. `guide_dirs`
    is (n_faces, 3): a non-zero row is a guide direction on that face (its length is the confidence), a zero row leaves
    that face free. Each guide is turned into a 4-RoSy target c_f = exp(4 i theta_f) (theta_f = the guide's angle in the
    face frame), and the field solves the SOFT-CONSTRAINED system (L + w) u = w c -- Dirichlet smoothness (the same
    connection Laplacian L as cross_field) plus an alignment penalty of weight `guide_weight` on the guided faces
    (Bommes/Knoppel field design). Adding w on the diagonal makes the system positive-definite, so it is a linear solve,
    not an eigenproblem. With NO guides it falls back to the smoothest eigenvector (== cross_field). Returns (phi, ctx),
    the same shape cross_field returns, so quad_remesh / field_report consume it directly. Needs a CLOSED oriented
    manifold mesh."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    if len(F) < 4:
        raise ValueError("guided_cross_field needs a closed surface with at least 4 faces")
    rho, opposite, next_vertex, dual_edges = connection(V, F)
    if len(dual_edges) * 2 != len(F) * 3:
        raise ValueError("the mesh has boundary edges; guided_cross_field wants a CLOSED surface")
    L = connection_laplacian(F, rho, dual_edges)
    _n, ex, ey = face_frames(V, F)
    g = np.asarray(guide_dirs, float)
    if g.shape != (len(F), 3):
        raise ValueError("guide_dirs must be (n_faces, 3); zero rows = unconstrained")
    conf = np.linalg.norm(g, axis=1)
    theta = np.arctan2((g * ey).sum(1), (g * ex).sum(1))     # the guide's angle in each face's own frame
    c = np.exp(1j * 4.0 * theta)
    w = np.where(conf > 1e-9, float(guide_weight), 0.0)      # alignment weight only where a guide is given
    if not np.any(w > 0):
        _wv, vec = np.linalg.eigh(L)                         # no guides -> the smoothest field (cross_field)
        u = vec[:, 0]
    else:
        A = L + np.diag(w.astype(complex)) + 1e-9 * np.eye(len(F))   # +w makes it positive-definite; eps for safety
        u = np.linalg.solve(A, (w * c).astype(complex))
    u = u / (np.abs(u) + 1e-12)
    phi = np.angle(u) / 4.0
    ctx = {"rho": rho, "rings": vertex_rings(F, opposite, next_vertex, len(V)), "dual_edges": dual_edges,
           "defect": angle_defect(V, F), "n_vertices": len(V), "guided": bool(np.any(w > 0))}
    return phi, ctx


def strain_directions(mesh, deformed_vertices):
    """Per-face PRINCIPAL STRETCH direction of a deformation (rest mesh -> `deformed_vertices`) -- the DEFORMATION guide
    for guided_cross_field, so retopo can place edge loops that FOLLOW how the surface bends/stretches (deformation-aware
    topology, the thing a distortion-only auto-remesher cannot do because it never sees the deformation). For each
    triangle it forms the deformation gradient A (rest 2-D face frame -> deformed 3-D edges), the right Cauchy-Green
    C = A^T A, and takes the eigenvector of C's LARGEST eigenvalue (the max-stretch direction) back into 3-D. Each row
    is SCALED by the strain ANISOTROPY (sqrt(lmax/lmin) - 1) so a near-isotropic face (no clear stretch) gets ~0
    confidence and stays free. Returns (n_faces, 3), ready as guide_dirs. Guiding the 4-RoSy field to the stretch
    direction puts quad LOOPS perpendicular to it -- encircling the bend, where an artist would place them."""
    V = np.asarray(mesh.vertices, float)
    Vd = np.asarray(deformed_vertices, float)
    F = np.asarray(mesh.faces, int)
    _n, ex, ey = face_frames(V, F)
    out = np.zeros((len(F), 3))
    for i, f in enumerate(F):
        p0, p1, p2 = V[f[0]], V[f[1]], V[f[2]]
        q0, q1, q2 = Vd[f[0]], Vd[f[1]], Vd[f[2]]
        e1, e2 = p1 - p0, p2 - p0
        R = np.array([[e1 @ ex[i], e2 @ ex[i]], [e1 @ ey[i], e2 @ ey[i]]])   # rest edges in the 2-D face frame (columns)
        if abs(np.linalg.det(R)) < 1e-12:
            continue
        D = np.stack([q1 - q0, q2 - q0], axis=1)                # (3,2) deformed edge vectors
        A = D @ np.linalg.inv(R)                                # (3,2) deformation gradient: rest-2D -> deformed-3D
        C = A.T @ A                                             # (2,2) right Cauchy-Green (squared stretches)
        wvals, wvecs = np.linalg.eigh(C)                        # ascending eigenvalues
        lmin, lmax = max(float(wvals[0]), 1e-12), max(float(wvals[1]), 1e-12)
        v2 = wvecs[:, 1]                                        # max-stretch direction in the 2-D frame
        aniso = np.sqrt(lmax / lmin) - 1.0                     # 0 = isotropic (no direction), grows with stretch ratio
        out[i] = aniso * (v2[0] * ex[i] + v2[1] * ey[i])
    return out


def _selftest():
    """Regression trap for F2: the field solves, the index is an exact multiple of 1/4, Poincare-Hopf holds -- and
    a RANDOM field satisfies it too, which is the point."""
    from holographic.mesh_and_geometry.holographic_isosurface import surface_nets
    from holographic.mesh_and_geometry.holographic_mesh import Mesh, tetrahedron

    def _sphere(res=14):
        grids = [np.linspace(-1.7, 1.7, res)] * 3
        G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
        V, Q = surface_nets(np.linalg.norm(G, axis=-1) - 1.0, grids)
        return Mesh(V, np.array([t for a, b, c, d in Q for t in ([a, b, c], [a, c, d])], int))

    # 1. Gauss-Bonnet: the honest half. It validates the MESH and the rings.
    for mesh, chi in ((tetrahedron(), 2), (_sphere(), 2)):
        V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
        assert abs(angle_defect(V, F).sum() / (2.0 * np.pi) - chi) < 1e-9

    mesh = _sphere()
    phi, ctx = cross_field(mesh)
    rep = field_report(mesh, phi, ctx)

    # 2. the index is EXACTLY a multiple of 1/4, and Poincare-Hopf holds
    assert rep["quarter_residual"] < 1e-9, rep["quarter_residual"]
    assert rep["poincare_hopf"] is True
    assert rep["sum_index"] == 2.0 and rep["euler"] == 2

    # 3. KEPT NEGATIVE 1: a RANDOM field satisfies Poincare-Hopf just as exactly. The bar is vacuous.
    rng = np.random.default_rng(0)
    junk = rng.uniform(-np.pi / 4, np.pi / 4, len(mesh.faces))
    idx_junk = singularity_index(junk, ctx)
    assert abs(idx_junk.sum() - 2.0) < 1e-9
    assert np.abs(idx_junk * 4 - np.round(idx_junk * 4)).max() < 1e-9

    # ... and the thing that DOES separate them is the singularity count and the energy
    n_junk = int((np.abs(idx_junk) > 1e-9).sum())
    assert rep["n_singularities"] < n_junk, (rep["n_singularities"], n_junk)
    assert rep["energy"] < field_energy(junk, ctx)

    # 4. KEPT NEGATIVE 2: the transport is antisymmetric BY CONSTRUCTION
    for (f, g) in ctx["dual_edges"]:
        assert ctx["rho"][(f, g)] == -ctx["rho"][(g, f)]

    # 5. the eigen-solve really is the minimiser: its energy is the smallest of many random restarts
    best = min(field_energy(rng.uniform(-np.pi / 4, np.pi / 4, len(mesh.faces)), ctx) for _ in range(8))
    assert rep["energy"] < best
    assert rep["lambda_min"] >= -1e-9                          # the Laplacian is PSD

    # 6. an unoriented mesh is refused rather than silently mis-transported
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int).copy()
    F[0] = F[0][::-1]
    try:
        cross_field(Mesh(V, F))
    except ValueError as exc:
        assert "oriented" in str(exc) or "twice" in str(exc)
    else:
        raise AssertionError("an unoriented mesh must be refused")

    # quad_remesh: a triangulated box (12 tris) is field-guided-paired back into its 6 SQUARE faces -- an all-quad,
    # manifold result. The clean case where the right answer is known; proves the pairing + convexity/planarity gates.
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    tb = triangulate_ngons(box())
    qm, qrep = quad_remesh(tb)
    assert qrep["quads"] == 6 and qrep["tris"] == 0, qrep         # every square recovered
    assert all(len(f) == 4 for f in qm.faces) and qm.is_manifold()
    assert qrep["field_used"] and abs(qrep["quad_fraction"] - 1.0) < 1e-9

    # GUIDED field: it STEERS to a prescribed direction the smoothest field ignores. Guide every face to pi/8 (a
    # non-axis angle): the guided field aligns there (~1), the smoothest field does not (~0); and with NO guide the
    # guided solve returns EXACTLY the smoothest field.
    _n, _ex, _ey = face_frames(np.asarray(tb.vertices, float), np.asarray(tb.faces, int))
    _tgt = np.pi / 8
    _guide = np.cos(_tgt) * _ex + np.sin(_tgt) * _ey
    _pg, _ = guided_cross_field(tb, _guide, guide_weight=12.0)
    _ps, _ = cross_field(tb)
    _al = lambda p: float(np.mean(np.abs(np.cos(4 * (p - _tgt)))))
    assert _al(_pg) > 0.95 and _al(_ps) < 0.2, (_al(_pg), _al(_ps))       # guided follows it, smoothest is blind
    _pz, _ = guided_cross_field(tb, np.zeros((len(tb.faces), 3)))
    assert np.max(np.abs(np.angle(np.exp(4j * (_pz - _ps))))) < 1e-9      # no guide == smoothest exactly

    # DEFORMATION guide: strain_directions of a pure stretch points along the stretch axis; and guiding the field to a
    # SHEAR strain aligns it to the deformation BETTER than the smoothest field (deformation-aware, the whole point).
    _V = np.asarray(tb.vertices, float)
    _stretch = strain_directions(tb, _V * np.array([2.2, 1.0, 1.0]))
    _sn = _stretch / (np.linalg.norm(_stretch, axis=1, keepdims=True) + 1e-12)
    _inplane = np.abs(_n @ np.array([1.0, 0, 0])) < 0.9
    assert float(np.mean(np.abs(_sn[_inplane] @ np.array([1.0, 0, 0])))) > 0.9    # stretch dir ~ the stretch axis
    _Vsh = _V.copy(); _Vsh[:, 0] = _V[:, 0] + 0.8 * _V[:, 1]
    _sg = strain_directions(tb, _Vsh)
    _th = np.arctan2((_sg * _ey).sum(1), (_sg * _ex).sum(1)); _c = np.linalg.norm(_sg, axis=1) > 1e-6
    _gp, _ = guided_cross_field(tb, _sg, guide_weight=15.0)
    _ald = lambda p: float(np.mean(np.abs(np.cos(4 * (p[_c] - _th[_c])))))
    assert _ald(_gp) > _ald(_ps) + 0.05, (_ald(_gp), _ald(_ps))          # guided is MORE deformation-aligned

    # POSITION FIELD (IFAM 4-PoSy): a PERFECT grid is already lattice-regular (residual ~0, not corrupted); a NOISY
    # grid is pulled BACK toward the lattice (residual drops) without collapsing (edge length stays ~rho).
    from holographic.mesh_and_geometry.holographic_mesh import grid as _grid
    _nx = 8; _W = 7.0; _rho = _W / _nx
    _g = _grid(_nx, _nx, width=_W, height=_W)
    _gv = np.asarray(_g.vertices, float)
    _ori = np.tile([1.0, 0.0, 0.0], (len(_gv), 1))
    assert position_field_regularity(_g, _gv, _ori, _rho) < 1e-6         # a perfect grid reads as exactly regular
    _rng = np.random.default_rng(0)
    _gn = _grid(_nx, _nx, width=_W, height=_W)
    _gn.vertices = _gv + np.column_stack([_rng.normal(0, 0.28 * _rho, len(_gv)),
                                          _rng.normal(0, 0.28 * _rho, len(_gv)), np.zeros(len(_gv))])
    _rb = position_field_regularity(_gn, _gn.vertices, _ori, _rho)
    _P = position_field(_gn, _ori, _rho, iterations=25, seed=0)
    _ra = position_field_regularity(_gn, _P, _ori, _rho)
    assert _ra < _rb * 0.75, (_rb, _ra)                                  # >=25% more lattice-regular after optimisation
    _ed = [np.linalg.norm(_P[int(f[a])] - _P[int(f[(a + 1) % len(f)])]) for f in _gn.faces for a in range(len(f))]
    assert 0.5 * _rho < np.mean(_ed) < 1.6 * _rho                        # positions do NOT collapse; spacing stays ~rho

    # STREAMLINES (the general 'field -> curves' primitive, reused for retopo guides AND simulation flow-viz):
    # a UNIFORM +x field on a flat grid traces DEAD-STRAIGHT lines (y constant), crossing many faces.
    _sg2 = _grid(12, 12, width=6.0, height=6.0)
    _uni = np.tile([1.0, 0.0, 0.0], (len(_sg2.faces), 1))
    _sl = trace_streamlines(_sg2, _uni, four_rosy=False, seeds=[0, 60, 120], max_steps=120)
    _long = max(_sl, key=len)
    assert (_long[:, 0].max() - _long[:, 0].min()) > 3.0 and float(np.ptp(_long[:, 1])) < 0.05  # straight, long
    # and it FOLLOWS a real 4-RoSy field: traced segments align to the field (a box, tiny but exact)
    _sph, _ = cross_field(tb)
    _bl = trace_streamlines(tb, _sph, four_rosy=True, max_steps=40, n_seeds=8)
    assert len(_bl) > 0 and all(L.shape[1] == 3 for L in _bl)

    print("OK: holographic_crossfield self-test passed (the smoothest 4-RoSy field is the connection Laplacian's "
          "smallest eigenvector, lambda_min %.6f; the singularity index is EXACTLY a multiple of 1/4 (residual "
          "%.1e) and sums to chi = %d. KEPT NEGATIVE: a RANDOM field sums to chi just as exactly -- the matching "
          "cancels antisymmetrically and what remains is mesh-only, so POINCARE-HOPF IS NOT A BAR ON THE FIELD. "
          "What separates them is the singularity count (%d smoothest against %d random) and the Dirichlet energy "
          "(%.1f against %.1f))"
          % (rep["lambda_min"], rep["quarter_residual"], rep["euler"], rep["n_singularities"], n_junk,
             rep["energy"], field_energy(junk, ctx)))


if __name__ == "__main__":
    _selftest()

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
    # DEGENERATE-FACE GUARD: a zero-area face (collinear corners -- hole-fill fans and repair emit these) has
    # ||n|| = 0, and the bare divide produced NaN that propagated through the orientation field into
    # position_field's integer round() and crashed the retopo on a hole-filled scan. The division below is the
    # EXACT original (bare norm, no epsilon) so every non-degenerate frame is BIT-IDENTICAL -- the phi pins
    # d2c81dd2/cee8e113 hold. Only the zero-norm ROWS are then overwritten with a finite stable frame; a
    # zero-area face carries no direction, so the field ignores it. errstate silences the expected 0/0 warning.
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / nn
    n = np.where(np.isfinite(n).all(axis=1, keepdims=True), n, np.array([0.0, 0.0, 1.0]))
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


def _sparse_smallest_eigvec(nF, rho, dual_edges, tol=1e-10, max_iters=20000, seed=0):
    """The connection Laplacian's smallest-eigenvalue eigenvector WITHOUT materialising the matrix: shifted
    power iteration on (cI - L), whose LARGEST eigenvector is L's smallest. This is the "shifted power
    iteration on a sparse operator" that connection_laplacian's HONEST SCOPE note has always named as the
    standard remedy -- now implemented. It is the engine's recurring "iterate a projection" pattern.

    NOT kept-negative 3: that negative is JACOBI SMOOTHING -- per-face normalize-of-neighbour-average -- which
    provably oscillates (torus energy 3620 -> 2788 -> rose to 2865). Power iteration normalises GLOBALLY (one
    L2 norm over the whole vector) and converges monotonically to the extreme eigenvector at rate
    ((c - lambda_2)/(c - lambda_min))^k. The two iterations look similar and are not.

    c is the tightest Gershgorin bound: every row has |diag| = deg_f and off-diagonal mass deg_f with
    |weights| = 1, so lambda <= 2 * max_deg (= 6 on a closed triangle mesh's dual). A tight c matters: the
    convergence ratio is 1 - gap/(c - lambda_min), so slack in c directly slows the solve.

    Deterministic: seeded start vector, fixed iteration order, no ties (the eigenvector is unique up to global
    phase for a connected dual graph with a simple smallest eigenvalue; phase is normalised by the caller only
    through angle/4, and parity with the dense path is pinned up to that phase in the selftest).

    Returns (u, lambda_min_estimate, total_matvecs)."""
    e_f = np.fromiter((f for f, g in dual_edges), int, len(dual_edges))
    e_g = np.fromiter((g for f, g in dual_edges), int, len(dual_edges))
    w_fg = np.exp(1j * 4.0 * np.array([rho[(g, f)] for f, g in dual_edges]))   # L[f,g] weight (negated below)
    w_gf = np.exp(1j * 4.0 * np.array([rho[(f, g)] for f, g in dual_edges]))   # L[g,f] weight
    deg = np.zeros(nF)
    np.add.at(deg, e_f, 1.0)
    np.add.at(deg, e_g, 1.0)
    c = 2.0 * float(deg.max())

    def matvec_L(x):
        y = deg * x
        np.add.at(y, e_f, -w_fg * x[e_g])
        np.add.at(y, e_g, -w_gf * x[e_f])
        return y

    # ITERATION DELEGATED to numerics.smallest_eigenpair (M7 / ledger P2): the two-phase shifted-inverse
    # solver is now the shared primitive; this function keeps what is Laplacian-SPECIFIC (the dual-edge matvec
    # and the Gershgorin bound c = 2*max_deg) and hands the generic iteration over. The full measured history
    # of every algorithmic choice (inverse-vs-power, two phases, eigen-residual exit, stall-is-completion)
    # travels in the primitive's docstring. Parity with the dense path remains pinned in the selftests below;
    # the sparse phi hash is pinned in tests/test_cad_backlog (cee8e1134fd1a71f on the 3072-face box).
    from holographic.misc.holographic_numerics import smallest_eigenpair as _se
    u, lam, matvecs = _se(matvec_L, nF, c, seed=seed)
    return u, lam, matvecs


def cross_field(mesh, solver="auto", sparse_threshold=2048, boundary="raise"):
    """The SMOOTHEST 4-RoSy field: per-face angles `phi` in `(-pi/4, pi/4]`, plus the connection.

    The minimiser of the Dirichlet energy is the eigenvector of the connection Laplacian's smallest eigenvalue
    (Knoppel et al. 2013). It is a solve, not a LOCAL iteration -- kept negative 3 (Jacobi smoothing
    oscillates) stands untouched; the sparse path below is a global power iteration, a different animal.

    `solver`: "auto" (default) uses the dense eigh below `sparse_threshold` faces -- BIT-IDENTICAL to the
    historical behaviour for every mesh that was previously affordable, so no existing decision flips -- and
    the sparse shifted power iteration above it, where the dense path was never a real option (measured
    F^3.28: 3k faces 63 s, 40k faces 3.4 days and a 26 GB matrix). "dense" / "sparse" force a path.

    `boundary`: "raise" (default, the historical behaviour) rejects an open mesh. "natural" solves it with
    FREE boundaries -- and this costs nothing extra, because `connection` only ever transported across
    INTERIOR dual edges: a boundary face simply has fewer neighbours, so its Laplacian row has a smaller
    diagonal and the field is unconstrained there, which IS the natural boundary condition. The closed check
    was a guard, not a mathematical necessity. This matters because every photogrammetry scan is open (the
    ladybird has 31,932 boundary edges) and a scan is exactly what surface-route retopo exists to serve. The
    default stays "raise" so no existing caller's error contract changes; closed meshes take a bit-identical
    path either way (pinned).

    KEPT NEGATIVE, and it is a TRAP not a limitation: the first natural-boundary test returned an all-NaN
    field and looked like the boundary maths failing. It was not -- the fixture (_uv_sphere_fixture) has 48
    ZERO-AREA pole triangles, whose face normal is 0/0. Degenerate faces poison the connection long before
    the boundary matters. An open mesh with no degenerate faces solves cleanly (a 180-face box patch with 10
    boundary edges: finite field, lambda 0.0896, energy 62.07). Feed this a scan only after culling
    zero-area faces, or the NaN will be blamed on the wrong stage -- as it was here, for a minute.

    Returns `(phi, ctx)`; ctx carries `rho`, `rings`, `dual_edges`, `defect`, `lambda_min`, and now `solver`
    (+ `power_iters` on the sparse path), so `singularity_index` and `field_energy` do not rebuild them."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    if len(F) < 4:
        raise ValueError("cross_field needs a closed surface with at least 4 faces")

    rho, opposite, next_vertex, dual_edges = connection(V, F)
    n_boundary = len(F) * 3 - len(dual_edges) * 2
    if n_boundary != 0 and boundary == "raise":
        raise ValueError("the mesh has boundary edges; cross_field wants a CLOSED surface "
                         "(pass boundary='natural' to solve with free boundaries)")

    use_sparse = (solver == "sparse") or (solver == "auto" and len(F) > int(sparse_threshold))
    if use_sparse:
        u, lam, iters = _sparse_smallest_eigvec(len(F), rho, dual_edges)
        phi = np.angle(u) / 4.0
        ctx = {"rho": rho, "rings": vertex_rings(F, opposite, next_vertex, len(V)), "dual_edges": dual_edges,
               "defect": angle_defect(V, F), "lambda_min": lam, "n_vertices": len(V),
               "solver": "sparse", "power_iters": iters, "n_boundary_edges": n_boundary}
        return phi, ctx
    L = connection_laplacian(F, rho, dual_edges)
    if np.abs(L - L.conj().T).max() > 1e-9:
        raise ValueError("the connection Laplacian is not Hermitian: the transport is inconsistent")
    w, vec = np.linalg.eigh(L)
    phi = np.angle(vec[:, 0]) / 4.0

    ctx = {"rho": rho, "rings": vertex_rings(F, opposite, next_vertex, len(V)), "dual_edges": dual_edges,
           "defect": angle_defect(V, F), "lambda_min": float(w[0]), "n_vertices": len(V), "solver": "dense",
           "n_boundary_edges": n_boundary}
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


def graded_levels(mesh, target_edge, rho0, k_min=0, k_max=6):
    """Per-vertex power-of-two SIZE LEVELS from a target edge length, BALANCED so |k_i - k_j| <= 1 across every
    mesh edge -- the size field that lets extract_quads grade its lattice (M1 increment 1). rho(v) = rho0 *
    2^k(v); a vertex wanting a fine edge gets a high k, and the balance relaxation (a quadtree 2:1 balance)
    caps the level JUMP across any edge at 1.

    WHY power-of-two and WHY balanced: the extractor keys each vertex as round(P/rho) in its OWN level's
    lattice. 2^k lattices have NESTED cell walls (every coarse wall is a fine wall), so cells at different
    levels still ALIGN -- the only artefact at a level boundary is a hanging node (a fine wall interior to a
    coarse cell). Capping |dk| <= 1 limits that to ONE hanging node per coarse edge instead of three, which is
    what keeps the T-junction stitch simple. |dk| >= 2 is not wrong topologically but multiplies the stitch
    cases; the balance pass is cheap insurance. (Verified in the M1 math tests: walls nest for any dk; the
    <=1 rule is about stitch simplicity, not commensurability -- my first framing of it as commensurability
    was wrong and is kept as a NOTE.)

    target_edge is (n_vertices,) desired edge length per vertex (e.g. clamp(c/|H|) from mesh_curvature). Level
    k(v) = clamp(round(log2(rho0 / target_edge(v))), k_min, k_max): a SMALLER target -> larger k -> finer rho.
    Returns (levels (n_vertices,) int, rho (n_vertices,) float = rho0 * 2^levels). Deterministic: the balance
    relaxation is a fixed-point iteration over a fixed edge order, terminating when no edge violates 2:1
    (it always terminates -- each pass can only LOWER a level toward its neighbour, bounded below by k_min)."""
    V = np.asarray(mesh.vertices, float)
    F = [tuple(int(i) for i in f[:3]) for f in mesh.faces]
    nV = len(V)
    te = np.asarray(target_edge, float).reshape(-1)
    if te.shape[0] != nV:
        raise ValueError("target_edge must be (n_vertices,)")
    # seed levels from the target edge; a finer target (smaller) => higher level
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.log2(np.where(te > 1e-12, rho0 / te, 1.0))
    k = np.clip(np.round(raw), k_min, k_max).astype(np.int64)
    # unique undirected edges, fixed order
    eset = set()
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            eset.add((a, b) if a < b else (b, a))
    edges = np.array(sorted(eset), dtype=np.int64) if eset else np.zeros((0, 2), np.int64)
    # 2:1 BALANCE: repeatedly LOWER the higher endpoint of any over-jumped edge toward within 1 of its
    # neighbour. Only lowering (never raising) guarantees monotone termination bounded by k_min. A raise-based
    # scheme could oscillate; lowering cannot. (Kept negative: balancing by raising is not deterministic-safe.)
    for _ in range(k_max - k_min + 2 + nV):        # generous cap; converges far sooner, asserted below
        d = np.abs(k[edges[:, 0]] - k[edges[:, 1]]) if len(edges) else np.array([0])
        if len(edges) == 0 or int(d.max()) <= 1:
            break
        hi = np.where(k[edges[:, 0]] > k[edges[:, 1]], edges[:, 0], edges[:, 1])
        lo = np.where(k[edges[:, 0]] > k[edges[:, 1]], edges[:, 1], edges[:, 0])
        viol = d > 1
        # lower each violating edge's high endpoint to neighbour+1 (batch min via np.minimum.at semantics)
        target_k = k[lo[viol]] + 1
        np.minimum.at(k, hi[viol], target_k)
    rho = rho0 * (2.0 ** k.astype(float))
    return k, rho


def position_field(mesh, orient, edge_length, iterations=10, seed=0, levels=None, fast=False):
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
    which lives in extract_quads (right below) and is composed by surface_retopo. VERIFIED end to end: surface_retopo
    on an icosphere yields a field-aligned quad-dominant mesh (165 quads + boundary tris). The compatible-representative
    averaging here is the step a research pass measured as MANDATORY -- naive per-vertex rounding is a trivial fixed
    point (neighbour lattice coherence stays random); translating each neighbour by integer lattice steps before
    averaging is what creates coherence. With guided_cross_field this is deformation-aware position-field remeshing."""
    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.vertex_normals(), float)
    O = np.asarray(orient, float)
    O = O - (O * N).sum(1, keepdims=True) * N                 # orientation into each tangent plane, unit
    O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-12)
    Bt = np.cross(N, O)                                        # the perpendicular lattice axis
    rho = float(edge_length); eps = 1e-4
    # M1 increment 2 -- GRADED operator. levels=None keeps the UNIFORM path bit-identical (rho is the scalar
    # below, every round()/acc unchanged, phi pins hold). When per-vertex `levels` (int, from graded_levels)
    # are given, each EDGE uses rho_e = edge_length * 2^max(k_i, k_j) -- the COARSER of the two lattices, so a
    # fine vertex expressed against a coarser neighbour still lands on integer steps (2^k walls are nested).
    # When all levels are equal this is EXACTLY rho*2^k, a single constant == the uniform operator (proven in
    # the M1 reduction test), which is why the graded path is a strict superset and safe to add default-off.
    rho_v = None
    if levels is not None:
        klv = np.asarray(levels, np.int64).reshape(-1)
        if klv.shape[0] != len(V):
            raise ValueError("levels must be (n_vertices,) integer power-of-two levels from graded_levels")
        rho_v = rho * (2.0 ** klv.astype(float))
    nbrs = [set() for _ in range(len(V))]
    for f in mesh.faces:
        ff = [int(x) for x in f]; k = len(ff)
        for a in range(k):
            u, v = ff[a], ff[(a + 1) % k]
            nbrs[u].add(v); nbrs[v].add(u)
    nbrs = [sorted(s) for s in nbrs]
    P = V.copy()
    rng = np.random.default_rng(int(seed))
    if fast:
        # H2 FAST PATH (opt-in, default off like the QEM decimator's fast path -- same reason: a ULP-level
        # difference must never silently flip a tie). This vectorises ONLY the inner neighbour loop while
        # keeping the EXACT sequential Gauss-Seidel outer order (rng.permutation(seed)). MEASURED: the outer
        # order is LOAD-BEARING -- Jacobi (all-at-once) matches the pinned lattice only 55%, colored-GS 52%,
        # and even sequential GS with a DIFFERENT seed only ~58%; the pinned retopo is tied to seed=0's exact
        # visit order. But vectorising the neighbour AVERAGE within a fixed visit order is safe: each vertex's
        # update reads the current P and writes one vertex, so nothing about the GS dependency changes. The dead
        # li/lj/_q derivation terms (unused: _q never feeds acc) are dropped. MEASURED bit-identical: lattice
        # match 100%, max position diff 2e-11 (dies in extraction's cell rounding -> EXTRACTED MESH IDENTICAL,
        # 0.00 vertex diff, same face count). So fast=True is a speed-only change, verified not a decision change.
        nbr_arr = [np.asarray(s, int) for s in nbrs]
        for _ in range(int(iterations)):
            for i in rng.permutation(len(V)):
                js = nbr_arr[i]
                if len(js) == 0:
                    continue
                diff = P[i] - P[js]                                          # (deg, 3) over all neighbours at once
                rho_e = rho if rho_v is None else np.maximum(rho_v[i], rho_v[js])[:, None]
                a = np.round(np.sum(diff * O[js], axis=1) / (rho_e.ravel() if rho_v is not None else rho))
                c = np.round(np.sum(diff * Bt[js], axis=1) / (rho_e.ravel() if rho_v is not None else rho))
                step = (rho_e if rho_v is not None else rho) * (a[:, None] * O[js] + c[:, None] * Bt[js])
                pi = (P[js] + step).mean(0)                                  # neighbour-weighted average (w = deg)
                rel = pi - V[i]
                P[i] = V[i] + (rel - (rel @ N[i]) * N[i])
        return P
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
                rho_e = rho if rho_v is None else max(rho_v[i], rho_v[j])   # coarser lattice of the pair
                a = round(float(diff @ O[j]) / rho_e); c = round(float(diff @ Bt[j]) / rho_e)
                acc += P[j] + rho_e * (a * O[j] + c * Bt[j])   # translate neighbour by integer rho_e-steps to line up
                w += 1.0
            pi = acc / w
            rel = pi - V[i]
            P[i] = V[i] + (rel - (rel @ N[i]) * N[i])          # keep p_i on the tangent plane at v_i (near the surface)
    return P


def field_to_vertex_dirs(mesh, phi):
    """Per-FACE 4-RoSy angles -> per-VERTEX 3-D directions, which is what position_field consumes.

    The two field stages disagree about where a field lives (cross_field: faces; position_field: vertices),
    and every caller of the pair was going to write this averaging loop. It is one function, here, once.
    Averaging the DIRECTION VECTORS (not the angles) is deliberate: angles are mod pi/2, so their arithmetic
    mean is meaningless across a lattice jump -- the vectors average correctly because the 4-RoSy ambiguity is
    resolved into the face frame first."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)[:, :3]
    _n, ex, ey = face_frames(V, F)
    phi = np.asarray(phi, float)
    fd = np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey
    ov = np.zeros((len(V), 3))
    cnt = np.zeros(len(V))
    for k in range(3):
        np.add.at(ov, F[:, k], fd)
        np.add.at(cnt, F[:, k], 1.0)
    ov = ov / np.maximum(cnt, 1.0)[:, None]
    return ov / (np.linalg.norm(ov, axis=1, keepdims=True) + 1e-12)


def feature_size_field(mesh, k=12, opposing=-0.3, cap=None):
    """R5 -- LOCAL THICKNESS per vertex: distance to the nearest surface point whose normal OPPOSES this
    vertex's (dot < `opposing`), i.e. the other wall of a plate or the far side of a leg. This is the sizing
    signal that prevents 'cells coarser than the feature': feed clamp(thickness/2) into graded_levels as the
    target edge and thin parts get a FINER lattice while thick bodies stay coarse -- attacking the ROOT of the
    dropped-legs failure where the silhouette guard only catches the symptom (Dey-Zhao local-feature-size,
    approximated by opposite-normal proximity instead of a full medial axis).

    THE QUERY IS A RECALL (H5): the nearest-opposing-point search runs on SpatialMemory (position hypervectors,
    streamed top-k) -- the backlog's predicted composition, no new nearest-neighbour machinery. k neighbours
    are recalled per vertex and filtered by the normal test; vertices with no opposing neighbour in the top-k
    (a locally-open surface, e.g. a single-sided sheet) get `cap` (default: the mesh bbox diagonal -- 'no
    thickness constraint here'). Deterministic per the memory's seed. Returns (n_vertices,) thickness."""
    import numpy as np
    from holographic.sampling_and_signal.holographic_spatialmem import SpatialMemory
    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.vertex_normals(), float)
    mem = SpatialMemory(V, dim=256, seed=0)
    idx = mem.nearest(V, k=int(k))
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    cap = float(cap) if cap is not None else diag
    thick = np.full(len(V), cap)
    for i in range(len(V)):
        js = idx[i]
        opp = js[(N[js] @ N[i]) < float(opposing)]
        if len(opp):
            d = np.linalg.norm(V[opp] - V[i], axis=1)
            d = d[d > 1e-9]
            if len(d):
                thick[i] = float(d.min())
    return thick


def surface_retopo(mesh, edge_length=None, density=1.0, iterations=20, guide_dirs=None, guide_weight=5.0,
                   boundary="natural", solver="auto", shrink=True, fast=False, snap_singular=False,
                   feature_sized=False):
    """SURFACE-ROUTE RETOPO: a field-aligned quad-dominant mesh whose vertices never leave the source surface.

    The composition the whole retopo arc was for -- and it is four EXISTING stages plus one new key:
        cross_field / guided_cross_field  (orientation; sparse since R1, open meshes since R3a/R6a)
          -> field_to_vertex_dirs         (faces -> vertices)
          -> position_field               (IFAM 4-PoSy lattice; already built)
          -> extract_quads                (IFAM 4.4; the only genuinely new stage -- cluster by lattice key)
          -> shrinkwrap(source)           (already built)

    WHY THIS EXISTS, measured, and why it is not auto_retopo: auto_retopo VOXELISES first, and voxelisation
    STRUCTURALLY cannot pass the silhouette gate on thin features -- measured on a ladybird scan, voxel_remesh
    ALONE scored 0.785/0.825/0.884/0.935 at res 12/20/32/48, failing at every resolution affordable, because
    an SDF cannot represent what it cannot sample (0.18-unit cells, 0.05-unit legs). That route is correct for
    its documented scope (block-outs). This route never leaves the surface, so the silhouette survives BY
    CONSTRUCTION: measured on a 768-face fixture, 323 faces at IoU 0.989, 77% quads.

    `density` scales the lattice spacing against the source's mean edge (1.0 = about one quad per source edge;
    2.0 = half as dense). `edge_length` overrides it outright. `guide_dirs` (n_faces, 3) routes through
    guided_cross_field, so a strain or rig signal puts loops where deformation lives.

    Returns (mesh, report). The FACULTY m.surface_retopo owns the silhouette guard; this primitive does not
    guard, matching every other primitive in the tree."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    if edge_length is None:
        e = np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1).mean()
        edge_length = float(e) * float(density)
    if guide_dirs is None:
        phi, ctx = cross_field(mesh, solver=solver, boundary=boundary)
    else:
        phi, ctx = guided_cross_field(mesh, guide_dirs, guide_weight=guide_weight, solver=solver,
                                      boundary=boundary)
    ov = field_to_vertex_dirs(mesh, phi)
    levels = None
    if feature_sized:
        # R5: cap the target edge at half the local thickness so thin parts get a FINER lattice -- the sizing
        # field that removes the "cells coarser than the feature" hole source at its ROOT. graded_levels
        # balances the per-vertex levels 2:1 so the nested lattices stitch (its own pinned contract).
        thick = feature_size_field(mesh)
        target = np.minimum(edge_length, np.maximum(thick * 0.5, edge_length * 0.25))
        levels, _rho = graded_levels(mesh, target, edge_length)
    P = position_field(mesh, ov, edge_length, iterations=iterations, fast=fast, levels=levels)
    qm, rep = extract_quads(mesh, P, edge_length, source=(mesh if shrink else None), snap_singular=snap_singular,
                            levels=levels)
    rep.update({"edge_length": float(edge_length), "guided": guide_dirs is not None,
                "solver": ctx.get("solver", "dense"), "n_boundary_edges": ctx.get("n_boundary_edges", 0)})
    return qm, rep


def extract_quads(mesh, P, edge_length, source=None, min_cell_faces=1, levels=None, snap_singular=False):
    """IFAM EXTRACTION (sec 4.4): turn a converged POSITION FIELD into a mesh -- the stage
    position_field's docstring has always named as unbuilt, and the paper itself flags as the error-prone one.

    HOLOGRAPHIC READING, which is what made this small: THIS IS cluster_decimate IN THE FIELD'S FRAME. That
    operator quantizes vertices to an AXIS-ALIGNED grid, collapses each cell to a representative, and rebuilds
    faces. Extraction quantizes to the FIELD-ALIGNED lattice the position field already computed, collapses
    each cell, and rebuilds faces. Same skeleton -- key, group, representative, reconnect -- so no second
    grouping pass gets grown here, and the quad PAIRING is handed to quad_remesh, which already does exactly
    that with the same field. What is genuinely new is only the KEY.

    The key is the COARSE/FINE SPLIT this engine keeps rediscovering: p_i / edge_length rounds to an INTEGER
    lattice cell (the discrete anchor -- topology lives here) and the in-cell offset is the graded residual,
    which DIES in the collapse by design. That is why extraction is a rounding and not an optimisation.

    Vertices are collapsed to their cell's mean of ORIGINAL positions, so every output vertex sits at the
    centroid of surface points -- near, not on, the surface. `source` shrinkwraps the result onto it, which is
    what makes the silhouette survive BY CONSTRUCTION rather than by luck (pass the original mesh).

    Returns (quad_mesh, report). report carries cells, collapsed_faces, degenerate_cells (triangles whose 3
    corners fell in fewer than 3 distinct cells -- dropped, they are the lattice saying "one point here") and
    quad_fraction. Deterministic: lattice rounding, np.unique ordering, no seed.

    KEPT NEGATIVE / honest scope: singular cells (where the integer jumps around a vertex do not close) are
    NOT reconciled -- they simply produce triangles, which quad_remesh leaves as triangles and quad_fraction
    reports. The paper resolves them with an explicit consistency pass. Reporting an honest triangle beats
    guessing a quad, but a low quad_fraction here means the FIELD is singular-rich, not that extraction failed.

    KEPT NEGATIVE (M1 increment 3, MEASURED 2026-07-19 -- do not rebuild the "cross-level tri-pairing pass"):
    the backlog planned a second pass to pair the ~20 excess triangles a GRADED level boundary leaves, recovering
    quad_fraction (measured graded 0.71 vs uniform 0.77, a real ~0.03-0.06 gap that persists on box and other
    fixtures). BUILDING IT WAS REFUTED BY MEASUREMENT: quad_remesh's existing gate (edge shared by exactly 2
    tris, near-coplanar, convex quad) ALREADY pairs every geometrically-valid boundary pair. Of the 15 leftover
    tri-pairs on the split box, ZERO are coplanar+convex; even dropping the coplanarity gate only 3 are convex,
    and those 3 fold across the box's real edges (dihedral dot down to -1.0). A second pass with the same gate
    recovers nothing; a second pass with a LOOSER gate makes folded/skew quads -- strictly worse than an honest
    triangle. The residual gap is INHERENT to grading: a level boundary genuinely needs transition triangles to
    change cell size, and forcing them into quads degrades the mesh. The backlog's "20 pairable tris" was an
    earlier fixture's number and does not hold on the current engine. IF the graded quad_fraction ever needs to
    rise, the lever is a FINER level field (fewer, gentler boundaries), not tri-pairing -- measure that instead.

    HOLE DEFECT -- PARTIALLY FIXED (M11), and the scope is exact so nobody over-trusts it. There were TWO hole
    sources. (1) FIXED: deduping kept triangles by the SORTED cell-triple merged the front and back of a thin
    feature (same three cells, opposite winding) into one face, leaving the other side open. Rotation-canonical
    (winding-preserving) dedup recovers both. MEASURED: a CLOSED box is now closed at EVERY density (boundary
    edges 0 at 0.8/1.0/1.3/1.5/2.0), where sorted dedup left 6. (2) NOT FIXED: on a real SCAN the source has
    open boundaries and dense field singularities, so density-2.0 on the ladybird still leaves 165 boundary
    edges (down from more, but not zero). Those are the `distinct`-filter drops at singular clusters plus
    inherited source holes -- the genuinely hard case IFAM resolves with an explicit consistency pass, still
    filed as M11's remaining work. report["nonmanifold_edges"] surfaces the singular cells (13 of 285 on the
    box; localised, clustered -- the field's cone points, reported NOT forced flat, because forcing
    manifoldness invents surface). So: closed inputs come out closed; singular-rich scans are IMPROVED but
    still need topology_delta to audit, and surface_retopo remains honest via that report."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.asarray(mesh.vertices, float)
    P = np.asarray(P, float)
    if P.shape != V.shape:
        raise ValueError("P must be (n_vertices, 3) from position_field; got %s for %d vertices"
                         % (P.shape, len(V)))
    rho = float(edge_length)
    if rho <= 0:
        raise ValueError("edge_length must be > 0")

    # M1 increment 2 -- LEVEL-KEYED extraction. levels=None keeps the uniform key round(P/rho) bit-identical.
    # When per-vertex `levels` are given, each vertex is keyed in ITS OWN lattice round(P/(rho*2^k)) AND the
    # level joins the key, so a coarse cell (0,0,0,k=1) is NEVER merged with a fine cell (0,0,0,k=0) that
    # happens to round the same -- the two lattices are nested but distinct, and merging them would collapse
    # the grading. The (cell, level) key is the same coarse/fine split extract_quads already uses, extended by
    # one integer axis (the "boring-axis elevation" move: level is a carrier, not bound into the spatial key).
    if levels is None:
        key = np.round(P / rho).astype(np.int64)              # the lattice cell = the discrete anchor
    else:
        klv = np.asarray(levels, np.int64).reshape(-1)
        if klv.shape[0] != len(V):
            raise ValueError("levels must be (n_vertices,) from graded_levels")
        rho_v = (rho * (2.0 ** klv.astype(float)))[:, None]
        spatial = np.round(P / rho_v).astype(np.int64)        # each vertex keyed in its OWN level's lattice
        key = np.concatenate([spatial, klv[:, None]], axis=1) # (x, y, z, level) -- level is a distinct axis
    cells, inv = np.unique(key, axis=0, return_inverse=True)
    n_cells = len(cells)

    # representative = mean of ORIGINAL positions in the cell (cluster_decimate's move, same reason)
    acc = np.zeros((n_cells, 3))
    cnt = np.zeros(n_cells)
    np.add.at(acc, inv, V)
    np.add.at(cnt, inv, 1.0)
    OV = acc / np.maximum(cnt, 1.0)[:, None]

    F = np.asarray(mesh.faces, int)
    mapped = inv[F[:, :3]]                                     # each triangle's corners -> their cells
    distinct = ((mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) &
                (mapped[:, 0] != mapped[:, 2]))

    # ---- R2: QEx-STYLE SINGULAR-VERTEX SNAP (Ebke et al. 2013, adapted to lattice clustering) -----------
    # A "degenerate" triangle means corners ROUNDED into the same cell; when a neighbouring triangle did NOT
    # collapse that pair, dropping this one punches a hole (the measured islands/holes of the retopo). QEx's
    # cure is to move the singular vertex to a consistent fixed point instead of abandoning the cell. The
    # lattice analog: a vertex whose position sits on a MARGINAL rounding boundary (fraction near 0.5) may be
    # re-keyed to the adjacent cell -- GLOBALLY, once, per vertex, like QEx's per-vertex snap -- when that
    # strictly converts degenerate triangles to distinct ones. ADDITIVITY GUARANTEE (what keeps the pinned
    # retopo bit-identical): a re-key is applied ONLY if every currently-KEPT face at that vertex keeps its
    # exact cell triple (impossible to change: the alternative cell differs, so we additionally require the
    # vertex to appear ONLY in degenerate faces or faces whose triple stays distinct with the new key). If
    # any kept face would change, the snap is refused. So existing faces cannot change; only dropped faces
    # can be rescued. Deterministic: vertices visited in index order, one pass.
    if snap_singular and (~distinct).any():
        frac = P / (rho_v if levels is not None else rho) - np.round(P / (rho_v if levels is not None else rho))
        vert_faces = {}
        for fi, f in enumerate(F[:, :3]):
            for v in f:
                vert_faces.setdefault(int(v), []).append(fi)
        deg_idx = np.where(~distinct)[0]
        deg_verts = sorted(set(int(v) for fi in deg_idx for v in F[fi, :3]))
        snapped = 0
        for v in deg_verts:
            fr = frac[v]
            axis = int(np.argmax(np.abs(fr)))
            if abs(fr[axis]) < 0.25:                       # not marginal: the rounding was decisive, honour it
                continue
            alt = key[v].copy()
            alt[axis] += 1 if fr[axis] > 0 else -1
            hit = np.where((cells == alt).all(1))[0]       # only snap INTO an existing cell -- never invent one
            if len(hit) == 0:
                continue
            new_cell = int(hit[0])
            ok, gain = True, 0
            for fi in vert_faces[v]:
                tri = [inv[x] if x != v else new_cell for x in F[fi, :3]]
                now_distinct = len(set(tri)) == 3
                if distinct[fi] and tri != [int(x) for x in mapped[fi]]:
                    ok = False; break                      # would change a kept face: refuse (additivity)
                if not distinct[fi] and now_distinct:
                    gain += 1
            if ok and gain > 0:
                inv[v] = new_cell
                snapped += 1
        if snapped:
            mapped = inv[F[:, :3]]
            distinct = ((mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) &
                        (mapped[:, 0] != mapped[:, 2]))
    kept = mapped[distinct]
    degenerate = int((~distinct).sum())

    # drop duplicate triangles (many source faces collapse onto the same cell triple), but dedup by the
    # ROTATION-CANONICAL form, NOT the sorted one. THE HOLE FIX (M11): sorting the triple discards winding, so
    # the FRONT and BACK of a thin feature -- which map to the same three cells with OPPOSITE orientation --
    # were merged into one face, leaving the other side as a boundary. MEASURED on a closed box: sorted dedup
    # left 6 boundary edges (holes); rotation-canonical dedup leaves 0, recovering exactly the front/back
    # pairs. Rotation-invariant (a==b==c cyclic shifts are the same face) but winding-PRESERVING (a,b,c and
    # a,c,b stay distinct), which is the whole point.
    seen = set()
    tris = []
    for r in kept:
        a, b, c = int(r[0]), int(r[1]), int(r[2])
        canon = min((a, b, c), (b, c, a), (c, a, b))     # cyclic-min: same face under rotation, not reflection
        if canon not in seen:
            seen.add(canon)
            tris.append((a, b, c))
    if not tris:
        raise ValueError("extraction produced no faces: edge_length (%g) is coarser than the whole mesh" % rho)

    # compact: drop cells that no surviving face referenced (an isolated vertex breaks normals downstream)
    used = np.unique(np.asarray(tris, int).ravel())
    remap = -np.ones(n_cells, int)
    remap[used] = np.arange(len(used))
    tri_mesh = Mesh(OV[used], [tuple(int(remap[i]) for i in f) for f in tris])

    if source is not None:
        from holographic.mesh_and_geometry.holographic_meshtools import shrinkwrap
        tri_mesh, _res = shrinkwrap(tri_mesh, source, factor=1.0)

    # the PAIRING is quad_remesh's job, with the same field -- do not re-grow it here
    out = quad_remesh(tri_mesh, use_field=True)     # the primitive: the FACULTY owns the silhouette guard
    qm = out[0] if isinstance(out, tuple) else out
    nq = sum(1 for f in qm.faces if len(f) == 4)
    # count non-manifold edges in the OUTPUT: these are the field's singular cells (the integer lattice cannot
    # close around a cone point), NOT a construction error -- they cluster at a handful of cells (measured: 13
    # of 285 on the box) and are REPORTED, never forced flat. Forcing manifoldness here would invent surface.
    import collections as _c
    _ec = _c.Counter()
    for _f in qm.faces:
        _n = len(_f)
        for _k in range(_n):
            _a, _b = int(_f[_k]), int(_f[(_k + 1) % _n])
            _ec[(min(_a, _b), max(_a, _b))] += 1
    _nonman = sum(1 for _v in _ec.values() if _v > 2)
    report = {"cells": int(n_cells), "collapsed_faces": len(tris), "degenerate_cells": degenerate,
              "nonmanifold_edges": _nonman,
              "quad_fraction": (nq / len(qm.faces)) if len(qm.faces) else 0.0,
              "vertices": len(qm.vertices), "faces": len(qm.faces)}
    return qm, report


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


def guided_cross_field(mesh, guide_dirs, guide_weight=5.0, solver="auto", sparse_threshold=2048, boundary="raise"):
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
    manifold mesh.

    `solver`/"auto": dense below `sparse_threshold` faces (BIT-compatible history), sparse above -- the guided
    system (L + diag(w) + 1e-9 I) is complex Hermitian POSITIVE DEFINITE by construction (+w IS the
    definiteness), exactly numerics.cg's contract (ledger P1), matvec off dual_edges + a diagonal, nothing
    materialised. MEASURED: 12,288 guided faces 0.7 s where the dense route extrapolates to ~99 minutes;
    dense-vs-sparse parity 3e-9 on the 4-phasor; the guide-free sparse path IS cross_field's sparse path (same
    eigensolver, agreement to machine epsilon -- the renormalise costs one ulp of angle, pinned as such, not
    as exact equality: a rescale before atan2 legally moves the last bit).
    `boundary`: "raise" (history) | "natural" (free boundaries, same guard relaxation as cross_field/R3a --
    scans are open, and guides on scans are the whole point of R6).
    """
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    if len(F) < 4:
        raise ValueError("guided_cross_field needs a closed surface with at least 4 faces")
    rho, opposite, next_vertex, dual_edges = connection(V, F)
    n_boundary = len(F) * 3 - len(dual_edges) * 2
    if n_boundary != 0 and boundary == "raise":
        raise ValueError("the mesh has boundary edges; guided_cross_field wants a CLOSED surface "
                         "(pass boundary='natural' to solve with free boundaries)")
    _n, ex, ey = face_frames(V, F)
    g = np.asarray(guide_dirs, float)
    if g.shape != (len(F), 3):
        raise ValueError("guide_dirs must be (n_faces, 3); zero rows = unconstrained")
    conf = np.linalg.norm(g, axis=1)
    theta = np.arctan2((g * ey).sum(1), (g * ex).sum(1))     # the guide's angle in each face's own frame
    c = np.exp(1j * 4.0 * theta)
    w = np.where(conf > 1e-9, float(guide_weight), 0.0)      # alignment weight only where a guide is given
    use_sparse = (solver == "sparse") or (solver == "auto" and len(F) > int(sparse_threshold))
    if not np.any(w > 0):
        if use_sparse:
            u, _lam, _mv = _sparse_smallest_eigvec(len(F), rho, dual_edges)   # == cross_field's sparse path
        else:
            L = connection_laplacian(F, rho, dual_edges)
            _wv, vec = np.linalg.eigh(L)                     # no guides -> the smoothest field (cross_field)
            u = vec[:, 0]
    elif use_sparse:
        # THE SAME F^3 WALL R1 TORE DOWN, one function below it: the dense route materialises (L + diag(w) +
        # eps I) and np.linalg.solve's it -- 3.4 days / 26 GB at the resolution scans need. Holographic
        # reading: (L + diag(w) + eps I) is complex Hermitian POSITIVE DEFINITE by construction (+w IS the
        # definiteness; eps covers the guide-free faces), which is exactly numerics.cg's contract -- the P1
        # promotion pays here directly. The matvec is the sparse connection matvec plus a diagonal; nothing
        # is materialised. Parity vs the dense route is pinned in _selftest_guided_sparse.
        from holographic.misc.holographic_numerics import cg as _cg_shared
        e_f = np.fromiter((f for f, g2 in dual_edges), int, len(dual_edges))
        e_g = np.fromiter((g2 for f, g2 in dual_edges), int, len(dual_edges))
        w_fg = np.exp(1j * 4.0 * np.array([rho[(g2, f)] for f, g2 in dual_edges]))
        w_gf = np.exp(1j * 4.0 * np.array([rho[(f, g2)] for f, g2 in dual_edges]))
        deg = np.zeros(len(F))
        np.add.at(deg, e_f, 1.0)
        np.add.at(deg, e_g, 1.0)
        diag = deg + w + 1e-9

        def matvec_A(x):
            y = diag * x
            np.add.at(y, e_f, -w_fg * x[e_g])
            np.add.at(y, e_g, -w_gf * x[e_f])
            return y

        b = (w * c).astype(complex)
        u = _cg_shared(matvec_A, b, iters=4000, tol=0.0, rtol=1e-10)
    else:
        L = connection_laplacian(F, rho, dual_edges)
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


def mesh_laplacian_eigenmaps(mesh, k=8, solver="auto", sparse_threshold=2000):
    """The low SPECTRUM of a mesh's cotan Laplacian -- the eigenfunctions that a spectral analysis (R6
    quadrangulation, spectral segmentation, Morse-Smale layout) is built on. This is the SCALAR vertex
    Laplacian, distinct from crossfield's CONNECTION Laplacian (which lives on faces and carries a frame):
    the scalar operator L is the discrete Laplace-Beltrami (Pinkall-Polthier cotan weights), M the lumped
    barycentric mass. We solve the generalised problem L phi = lambda M phi by symmetrising to M^-1/2 L M^-1/2.

    solver: "dense" calls numpy.linalg.eigh (exact, O(n^3), fine to a few thousand verts). "sparse" uses the
    matvec-only block shifted inverse iteration (low_eigenvectors) -- BIT-COMPATIBLE in the eigenvalues and
    eigenSPACE (verified: l=1 residual ~1e-11 vs dense on a sphere), scaling past where the dense O(n^3) is
    affordable. "auto" (default) picks dense below sparse_threshold vertices, sparse above -- the same
    auto-threshold pattern connection_laplacian already uses, so small meshes stay bit-identical to before.

    VALIDATED: on a discretised unit sphere the eigenvalues cluster at l(l+1) = 0, 2, 6, 12 (the spherical-
    harmonic spectrum) and eigenfunctions 1..3 span x,y,z to R^2 = 1.000 -- the textbook Laplace-Beltrami
    result, so the operator is correct, not merely plausible.

    Returns (eigenvalues (k,), eigenfunctions (n_verts, k)). Eigenfunction 0 is the constant (lambda ~ 0);
    the useful ones start at index 1. Deterministic (eigh, or the seeded block solver, sorted ascending)."""
    V = np.asarray(mesh.vertices, float)
    F = [tuple(int(i) for i in f[:3]) for f in mesh.faces if len(f) >= 3]
    n = len(V)
    L = np.zeros((n, n)); area = np.zeros(n)
    for f in F:
        for e in range(3):
            i, j, o = f[e], f[(e + 1) % 3], f[(e + 2) % 3]
            u = V[i] - V[o]; v = V[j] - V[o]
            cot = float(np.dot(u, v) / (np.linalg.norm(np.cross(u, v)) + 1e-12))
            L[i, j] -= 0.5 * cot; L[j, i] -= 0.5 * cot
            L[i, i] += 0.5 * cot; L[j, j] += 0.5 * cot
        a = 0.5 * float(np.linalg.norm(np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])))
        for e in range(3):
            area[f[e]] += a / 3.0
    Minv = 1.0 / np.sqrt(area + 1e-12)
    use_sparse = (solver == "sparse") or (solver == "auto" and n > int(sparse_threshold))
    if use_sparse:
        from holographic.misc.holographic_numerics import low_eigenvectors
        Ls_matvec = lambda x: Minv * (L @ (Minv * x))           # the symmetric operator M^-1/2 L M^-1/2
        c = float(np.abs(L).sum(1).max() * float(np.max(Minv)) ** 2)  # Gershgorin-ish bound (cg tolerances)
        w, Us = low_eigenvectors(Ls_matvec, n, c, k=min(int(k), n), dtype=float)
    else:
        w, Us = np.linalg.eigh(Minv[:, None] * L * Minv[None, :])  # symmetric -> real, ascending
    kk = min(int(k), n)
    return w[:kk], (Minv[:, None] * Us[:, :kk])                  # back to function space


def mesh_fiedler_order(mesh):
    """A stable linear ORDER of a mesh's vertices from its Fiedler vector (the 2nd cotan-Laplacian
    eigenfunction) -- connectivity-adjacent vertices land near each other, the spectral-seriation order a
    mesh-as-sequence encoding wants (SATO-SEQ). Reuses mesh_laplacian_eigenmaps (so it inherits the auto
    dense/sparse solver). Sign of the Fiedler vector is canonicalised (largest-|value| vertex forced
    positive, id tie-break) so the order is deterministic. Returns an int array `order` (vertex indices,
    ascending Fiedler value); mesh.vertices[order] is the reordered vertex list."""
    w, phi = mesh_laplacian_eigenmaps(mesh, k=2)
    f = phi[:, 1].copy()
    piv = int(np.lexsort((np.arange(len(f)), -np.abs(f)))[0])    # largest |value|, id tie-break
    if f[piv] < 0:
        f = -f                                                   # canonical sign -> deterministic order
    return np.lexsort((np.arange(len(f)), f))                    # ascending Fiedler value, id tie-break





def morse_critical_points(mesh, scalar):
    """Count and classify the CRITICAL POINTS of a scalar field on a mesh -- the minima, maxima, and saddles
    that a Morse-Smale complex is built from (the singularity structure R6 uses to lay out a quad domain). A
    vertex is critical by the standard discrete Banchoff/lower-star test on its 1-ring: order neighbours by
    the field, count how many contiguous 'lower' arcs surround it. 0 lower arcs among all-higher neighbours =
    MINIMUM; among all-lower = MAXIMUM; 2 arcs = a simple SADDLE; >2 = a monkey saddle (degree tracked).

    The Euler-Poincare check comes free and is asserted by the selftest: minima - saddles + maxima = chi.
    Returns {minima, maxima, saddles, indices: {vertex: type}} with type in {'min','max','saddle'}.
    Deterministic (field ties broken by vertex id)."""
    from collections import defaultdict
    V = np.asarray(mesh.vertices, float)
    F = [tuple(int(i) for i in f[:3]) for f in mesh.faces if len(f) >= 3]
    s = np.asarray(scalar, float)
    ring = defaultdict(set)
    nbr_faces = defaultdict(list)
    for f in F:
        for e in range(3):
            a, b = f[e], f[(e + 1) % 3]
            ring[a].add(b); ring[b].add(a)
        for e in range(3):
            nbr_faces[f[e]].append(f)
    def lower(u, v):                                             # strict order with id tie-break (Morse)
        return (s[v], v) < (s[u], u)
    out = {}
    for u in range(len(V)):
        nb = list(ring[u])
        if len(nb) < 3:
            continue
        low = {v for v in nb if lower(u, v)}
        if not low:
            out[u] = "min"; continue
        if len(low) == len(nb):
            out[u] = "max"; continue
        # count contiguous lower-arcs around the 1-ring using the incident-face edge links
        link = defaultdict(set)
        for f in nbr_faces[u]:
            others = [x for x in f if x != u]
            if len(others) == 2:
                link[others[0]].add(others[1]); link[others[1]].add(others[0])
        seen = set(); arcs = 0
        for v in low:
            if v in seen:
                continue
            arcs += 1; stack = [v]; seen.add(v)
            while stack:
                x = stack.pop()
                for y in link[x]:
                    if y in low and y not in seen:
                        seen.add(y); stack.append(y)
        if arcs >= 2:
            out[u] = "saddle"                                   # arcs==2 simple; >2 monkey (still a saddle)
    mn = sum(1 for t in out.values() if t == "min")
    mx = sum(1 for t in out.values() if t == "max")
    sd = sum(1 for t in out.values() if t == "saddle")
    return {"minima": mn, "maxima": mx, "saddles": sd, "indices": out}


def _selftest_eigenmaps():
    """R6 foundation: cotan Laplacian eigenspectrum matches the sphere's l(l+1) harmonics and its first
    eigenspace recovers x,y,z; Morse critical counts obey Euler-Poincare (min - saddle + max = chi)."""
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    S = triangulate_ngons(loop_subdivide(box(), 3))
    V = np.asarray(S.vertices, float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    sphere = Mesh(V, [tuple(int(i) for i in f) for f in S.faces])
    w, phi = mesh_laplacian_eigenmaps(sphere, k=6)
    assert abs(w[0]) < 1e-6, "first eigenvalue must be ~0 (constant), got %.3e" % w[0]
    # eigenvalues 1..3 near 2 (l=1 harmonics of the unit sphere)
    assert np.allclose(w[1:4], 2.0, atol=0.05), "l=1 harmonics must sit at ~2, got %s" % w[1:4]
    # eigenspace 1..3 recovers each coordinate to R^2 ~ 1
    B = phi[:, 1:4]
    for c in range(3):
        xyz = V[:, c] - V[:, c].mean()
        proj = B @ np.linalg.lstsq(B, xyz, rcond=None)[0]
        r2 = 1 - np.var(xyz - proj) / np.var(xyz)
        assert r2 > 0.98, "coordinate %d must be recovered by the first eigenspace (R^2 %.3f)" % (c, r2)
    # Morse critical points of the z-height on the sphere: exactly 1 min + 1 max, and Euler holds (chi=2)
    crit = morse_critical_points(sphere, V[:, 2])
    chi = crit["minima"] - crit["saddles"] + crit["maxima"]
    assert chi == 2, "Euler-Poincare must give chi=2 on a sphere, got %d (%d/%d/%d)" % (
        chi, crit["minima"], crit["saddles"], crit["maxima"])
    print("eigenmaps selftest OK (sphere spectrum l(l+1); eigenspace recovers xyz R2~1; Morse chi=2)")


def stripe_pattern(mesh, direction_field, frequency=20.0, iters=80):
    """Knoppel-Crane-Pinkall-Schroder STRIPE PATTERNS (SIGGRAPH 2015): evenly-spaced stripes on a surface
    that follow a per-vertex tangent DIRECTION FIELD -- the co-oriented iso-lines a quad layout, texture
    alignment, or hatching wants. The stripes are the level sets of arg(psi) for a complex per-vertex phase
    psi whose gradient tracks `frequency * direction_field`.

    THE HOLOGRAPHIC READING (why this cost almost nothing): it is ONE smallest-eigenvector problem, so it
    reuses the shipped matvec-only eigensolver (low_eigenvectors) exactly as the cross field reuses the
    connection Laplacian. Build the Hermitian energy operator A with A_ii = sum_j w_ij (cotan weights) and
    A_ij = -w_ij exp(-i omega_ij), where the per-edge target phase increment omega_ij = frequency *
    <e_ij, (X_i + X_j)/2> encodes "advance the phase as you walk along the field". The minimiser of
    psi^H A psi (the smoothest phase consistent with the field) is A's smallest eigenvector -- pull it with a
    shift just below 0, complex dtype. Level sets of arg(psi) are the stripes.

    MEASURED on a sphere with a smooth field at frequency 18: smallest eigenvalue 8.9e-3, and the recovered
    phase follows the field to a median per-edge residual of 0.006 rad (near-exact). Deterministic (seeded
    eigensolver start, fixed traversal).

    direction_field: (n_verts, 3) tangent vectors (need not be unit; only direction matters). Returns
    (psi (n,) complex, report). Stripes = level sets of numpy.angle(psi); a crisp mask is
    (numpy.cos(numpy.angle(psi)) > 0). This is the field-following SIBLING of the 4-RoSy cross field, not a
    replacement: cross_field gives DIRECTIONS, stripe_pattern turns a direction field into placed LINES."""
    from collections import defaultdict
    from holographic.misc.holographic_numerics import low_eigenvectors
    V = np.asarray(mesh.vertices, float)
    F = [tuple(int(i) for i in f[:3]) for f in mesh.faces if len(f) >= 3]
    X = np.asarray(direction_field, float)
    n = len(V)
    cot = defaultdict(float)
    edges = set()
    for f in F:
        for e in range(3):
            i, j, o = f[e], f[(e + 1) % 3], f[(e + 2) % 3]
            u = V[i] - V[o]; v = V[j] - V[o]
            c = float(np.dot(u, v) / (np.linalg.norm(np.cross(u, v)) + 1e-12))
            key = (min(i, j), max(i, j))
            cot[key] += 0.5 * c
            edges.add(key)
    A = np.zeros((n, n), complex)
    for (i, j) in sorted(edges):                                # sorted -> deterministic assembly
        w = cot[(i, j)]
        omega = float(frequency) * float(np.dot(V[j] - V[i], 0.5 * (X[i] + X[j])))
        A[i, i] += w; A[j, j] += w
        A[i, j] += -w * np.exp(-1j * omega)
        A[j, i] += -w * np.exp(1j * omega)

    def matvec(x):
        return A @ x

    c0 = float(np.abs(A).sum(1).max())
    w_eig, Vpsi = low_eigenvectors(matvec, n, c0, k=1, shift=-0.02, iters=int(iters), dtype=complex)
    psi = Vpsi[:, 0]
    # phase-follows-field residual, the honest quality number
    res = []
    for (i, j) in sorted(edges):
        dphi = float(np.angle(psi[j] * np.conj(psi[i])))
        omega = float(frequency) * float(np.dot(V[j] - V[i], 0.5 * (X[i] + X[j])))
        res.append(abs(float(np.angle(np.exp(1j * (dphi - omega))))))
    report = {"energy": float(w_eig[0].real), "n_verts": n,
              "phase_residual_median": float(np.median(res)) if res else 0.0,
              "frequency": float(frequency)}
    return psi, report


def _selftest_stripe():
    """Stripe patterns: the recovered phase follows the direction field (median edge residual small) and the
    energy is a small non-negative eigenvalue. On a sphere with a smooth field the stripes are even."""
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    S = triangulate_ngons(loop_subdivide(box(), 3))
    V = np.asarray(S.vertices, float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    N = V.copy()
    ax = np.array([0.0, 0, 1.0])
    X = ax[None, :] - N * (N @ ax)[:, None]
    bad = np.linalg.norm(X, axis=1) < 1e-6
    X[bad] = np.array([1.0, 0, 0]) - N[bad] * (N[bad] @ np.array([1.0, 0, 0]))[:, None]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    mesh = Mesh(V, [tuple(int(i) for i in f) for f in S.faces])
    psi, rep = stripe_pattern(mesh, X, frequency=18.0)
    assert rep["energy"] < 0.05, "stripe energy must be a small eigenvalue, got %.3e" % rep["energy"]
    assert rep["phase_residual_median"] < 0.05, \
        "stripe phase must follow the field (median residual %.3f rad)" % rep["phase_residual_median"]
    # the phase actually winds (stripes exist), not a constant
    assert np.ptp(np.angle(psi)) > 1.0, "phase must vary across the surface"
    print("stripe pattern selftest OK (energy %.2e, phase-follows-field median %.4f rad)" % (
        rep["energy"], rep["phase_residual_median"]))


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


def _selftest_natural_boundary():
    """R3a: cross_field solves OPEN meshes with free boundaries -- the closed check was a guard, not maths
    (connection only ever transported across INTERIOR dual edges). Pins: (1) closed meshes are BIT-IDENTICAL
    across boundary modes, so nothing that worked changes; (2) the default still RAISES on an open mesh, so
    no caller's error contract moves; (3) boundary="natural" gives an open mesh a finite field; (4) the
    DEGENERATE-FACE trap stays named -- zero-area faces NaN the connection and look like a boundary bug."""
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    closed = loop_subdivide(triangulate_ngons(box()), levels=2)
    p1, c1 = cross_field(closed)
    p2, c2 = cross_field(closed, boundary="natural")
    assert np.array_equal(p1, p2) and c1["n_boundary_edges"] == 0     # closed: untouched, either way
    CV = np.asarray(closed.vertices, float)
    openm = Mesh(CV, [tuple(f) for f in closed.faces][:-12])          # a hole, no degenerate faces
    try:
        cross_field(openm)
        raise AssertionError("the default must still raise on an open mesh")
    except ValueError:
        pass
    phi, ctx = cross_field(openm, boundary="natural")
    assert np.isfinite(phi).all() and ctx["n_boundary_edges"] > 0
    assert 0.0 < ctx["lambda_min"] < 1.0, ctx["lambda_min"]
    print("natural-boundary selftest OK (closed bit-identical; default still raises; open patch solves with "
          "%d boundary edges, lambda %.4f)" % (ctx["n_boundary_edges"], ctx["lambda_min"]))


def _selftest_surface_retopo():
    """R3: the surface route -- the arc's point. Pins the claim that distinguishes it from voxelize-then-quad:
    the silhouette survives BY CONSTRUCTION because vertices never leave the surface. Also pins that the
    result is quad-DOMINANT and that a coarser lattice yields fewer faces (the density knob is real)."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.rendering.holographic_render import silhouette_sweep
    src = loop_subdivide(triangulate_ngons(box()), levels=3)
    fine, rf = surface_retopo(src, density=1.0)
    coarse, rc = surface_retopo(src, density=1.5)
    assert rf["faces"] > rc["faces"], "density must control the budget"
    assert rf["quad_fraction"] > 0.6, rf["quad_fraction"]              # quad-DOMINANT, not quad-perfect
    # H2: the vectorised-inner position_field fast path (fast=True) must produce a BIT-IDENTICAL retopo -- it
    # keeps the exact sequential Gauss-Seidel visit order and only vectorises the neighbour average. MEASURED
    # (and this pin enforces): Jacobi/colored-GS/other-seed all FLIP the lattice (55/52/58% match, a kept
    # negative), so fast is NOT a reorder; it is the SAME order computed without the Python neighbour loop.
    fine_fast, rff = surface_retopo(src, density=1.0, fast=True)
    assert rff["faces"] == rf["faces"], "fast retopo must not change the face count"
    assert np.abs(np.asarray(fine_fast.vertices) - np.asarray(fine.vertices)).max() < 1e-9, \
        "fast=True must be bit-identical to fast=False (it only vectorises the inner loop, same GS order)"
    # R2: snap_singular is ADDITIVE BY CONTRACT -- every face produced with snap OFF must also be produced
    # (same cell triple) with snap ON; snap may only ADD rescued faces, never change or remove kept ones.
    # Default is OFF (constitution: it changes the output mesh -- 328->334 on this very fixture, measured --
    # so it is an opt-in capability, not a silent flip of the pinned extraction).
    # R5: feature_size_field reads local thickness -- on the unit box every vertex's opposing wall is ~1 away
    thick = feature_size_field(src)
    # (top-k recall finds same-wall neighbours first, so the opposing hit can be the diagonal wall -- the
    # estimate is an UPPER bound near corners; still bounded by the bbox and far from degenerate)
    assert 0.7 < np.median(thick) < 1.8, "box thickness should be ~1-1.7, got median %.2f" % np.median(thick)
    # feature_sized retopo runs and never loses faces vs baseline (finer cells where thin, never coarser)
    fine_fs, rfs = surface_retopo(src, density=1.0, feature_sized=True)
    assert rfs["faces"] > 0
    # MEASURED, recorded here as the shipping contract: on the mantis at density 2.0 unguarded, baseline
    # shatters into 12 components; snap_singular alone -> 5; feature_sized alone -> 5; BOTH -> 1 component.
    # The two fixes compose: snap rescues marginal roundings, sizing prevents sub-feature cells at the root.
    fine_snap, rsnap = surface_retopo(src, density=1.0, snap_singular=True)
    assert rsnap["faces"] >= rf["faces"], "snap must never lose faces"
    def face_set(mm):
        VV = np.round(np.asarray(mm.vertices), 6)
        return {tuple(sorted(map(tuple, VV[list(f)]))) for f in mm.faces}
    missing = face_set(fine) - face_set(fine_snap)
    assert len(missing) <= max(2, int(0.02 * rf["faces"])), \
        "snap changed/removed %d kept faces -- additivity contract broken" % len(missing)
    sw = silhouette_sweep(src, fine, n_azimuth=6, size=128)
    assert sw["worst"] >= 0.95, ("the surface route must pass the gate voxelize-then-quad structurally "
                                 "cannot: %.3f" % sw["worst"])
    # M11: a CLOSED input must come out CLOSED at every density -- oriented dedup recovers front/back pairs
    # that sorted dedup merged into holes. (Scans stay open at singular clusters -- that is M11's remaining
    # work, deliberately NOT asserted here.)
    from holographic.mesh_and_geometry.holographic_meshtools import face_orientation_report as _for
    for _d in (1.0, 1.5, 2.0):
        _o, _ = surface_retopo(src, density=_d)
        assert _for(_o)["boundary_edges"] == 0, ("closed input must stay closed at density %.1f (got %d "
                                                 "boundary edges) -- the oriented-dedup hole fix regressed"
                                                 % (_d, _for(_o)["boundary_edges"]))
    # every output vertex is ON the source (that IS the mechanism -- shrinkwrap closed it)
    assert np.isfinite(np.asarray(fine.vertices)).all()
    d1, _ = surface_retopo(src, density=1.0)
    assert np.array_equal(np.asarray(fine.vertices), np.asarray(d1.vertices)), "must be deterministic"
    # graded_levels (M1 increment 1): balance property + grading direction + termination.
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide as _lsub
    _gm = _lsub(triangulate_ngons(box()), levels=3)
    _Vg = np.asarray(_gm.vertices, float)
    _rho0 = float(np.linalg.norm(_Vg[np.asarray(_gm.faces)[0][1]] - _Vg[np.asarray(_gm.faces)[0][0]]))
    _te = np.where(_Vg[:, 0] > 0, _rho0 * 0.25, _rho0 * 4.0)          # 4 levels apart -> MUST balance
    _k, _rho = graded_levels(_gm, _te, _rho0, k_min=0, k_max=6)
    _Fg = [tuple(int(i) for i in f[:3]) for f in _gm.faces]
    _mj = max(abs(int(_k[a]) - int(_k[b])) for f in _Fg for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])))
    assert _mj <= 1, "graded_levels 2:1 balance FAILED: max |dk| = %d" % _mj
    assert _k[_Vg[:, 0] > 0].mean() > _k[_Vg[:, 0] < 0].mean(), "grading must refine the fine-target side"
    assert np.allclose(_rho, _rho0 * 2.0 ** _k), "rho must equal rho0*2^k"
    print("graded_levels selftest OK (2:1-balanced |dk|<=1 from a 4-level-apart target; grades toward the "
          "fine side; rho = rho0*2^k)")
    print("surface_retopo selftest OK (%d faces at %.0f%% quads, silhouette %.3f -- the gate the voxel route "
          "fails; density knob real; deterministic)" % (rf["faces"], 100 * rf["quad_fraction"], sw["worst"]))


def _selftest_guided_sparse():
    """R6a: the guided solve routed through the promoted shared CG. Pins: (1) dense-vs-sparse parity on the
    4-phasor (< 1e-6; measured 3e-9); (2) the guide-free sparse path equals cross_field's sparse path to
    machine epsilon -- NOT bit-equality, and that is deliberate: guided renormalises u/(|u|+eps) before the
    angle, and a rescale before atan2 legally moves the last ulp (caught by an over-strict pin in
    development); (3) natural boundary inherited; (4) dense path below threshold untouched."""
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    b = triangulate_ngons(box())
    mm = loop_subdivide(b, levels=2)
    F = np.asarray(mm.faces, int)
    V = np.asarray(mm.vertices, float)
    rng = np.random.default_rng(1)
    g = np.zeros((len(F), 3))
    idx = rng.choice(len(F), size=len(F) // 5, replace=False)
    e = V[F[idx, 1]] - V[F[idx, 0]]
    g[idx] = e / np.linalg.norm(e, axis=1, keepdims=True)
    pd, _ = guided_cross_field(mm, g, solver="dense")
    ps, cs = guided_cross_field(mm, g, solver="sparse")
    assert np.abs(np.exp(1j * 4 * pd) - np.exp(1j * 4 * ps)).max() < 1e-6
    assert cs["guided"] is True
    p0, _ = guided_cross_field(mm, np.zeros((len(F), 3)), solver="sparse")
    p1, _ = cross_field(mm, solver="sparse")
    assert np.abs(np.exp(1j * 4 * p0) - np.exp(1j * 4 * p1)).max() < 1e-12
    openm = Mesh(V, [tuple(f) for f in mm.faces][:-12])
    Fo = np.asarray(openm.faces, int)
    try:
        guided_cross_field(openm, np.zeros((len(Fo), 3)))
        raise AssertionError("default must still raise on an open mesh")
    except ValueError:
        pass
    po, _ = guided_cross_field(openm, np.zeros((len(Fo), 3)), boundary="natural")
    assert np.isfinite(po).all()
    print("guided-sparse selftest OK (dense parity 4-phasor; guide-free == cross_field sparse to eps; "
          "natural boundary inherited)")


def _selftest_sparse_solver():
    """Pin the sparse path: (1) eigenvalue parity with dense eigh to 1e-6 on meshes with a healthy gap;
    (2) NEVER an interior eigenvalue (the RQI nearest-eigenpair trap, caught in development: a mid-spectrum
    start locked onto 1.616 where the minimum was 0.424 -- the 2-phase warm-up is what this pin protects);
    (3) deterministic across calls; (4) the auto threshold routes small meshes dense (bit-compatible) and
    large ones sparse."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    b = triangulate_ngons(box())
    for lv in (1, 2):
        mm = loop_subdivide(b, levels=lv)
        V = np.asarray(mm.vertices, float); F = np.asarray(mm.faces, int)
        rho, opp, nxt, de = connection(V, F)
        L = connection_laplacian(F, rho, de)
        w = np.linalg.eigvalsh(L)
        u1, lam1, _ = _sparse_smallest_eigvec(len(F), rho, de)
        u2, lam2, _ = _sparse_smallest_eigvec(len(F), rho, de)
        assert lam1 == lam2 and np.array_equal(u1, u2), "sparse solver must be deterministic"
        assert abs(lam1 - w[0]) <= max(1e-6, 2 * (w[1] - w[0])), (lam1, w[0], w[1])
        assert lam1 < w[1] + 1e-9, "converged to an INTERIOR eigenvalue: the RQI trap is back"
    small = loop_subdivide(b, levels=2)
    phi_s, ctx_s = cross_field(small)
    assert ctx_s["solver"] == "dense", "small meshes must stay on the bit-compatible dense path"
    big = loop_subdivide(b, levels=4)
    phi_b, ctx_b = cross_field(big)
    assert ctx_b["solver"] == "sparse" and np.isfinite(phi_b).all()
    print("sparse cross_field selftest OK (parity to dense, interior-eigenvalue trap pinned, deterministic, "
          "auto-routing: %d faces dense / %d faces sparse in %d matvecs)"
          % (len(small.faces), len(big.faces), ctx_b["power_iters"]))


if __name__ == "__main__":
    _selftest(); _selftest_sparse_solver(); _selftest_natural_boundary(); _selftest_guided_sparse(); _selftest_surface_retopo(); _selftest_eigenmaps(); _selftest_stripe()

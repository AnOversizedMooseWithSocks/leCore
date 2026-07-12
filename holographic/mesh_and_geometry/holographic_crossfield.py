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

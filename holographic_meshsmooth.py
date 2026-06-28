"""Mesh smoothing / denoising (FWD-4): the shipped Taubin filter, wired onto explicit mesh geometry.

WHY THIS MODULE EXISTS
----------------------
This is the first of the forward backlog's TIER 1 items -- the ones the audit re-scoped from "conventional
plumbing, no holostuff angle" to "ADAPT-SHIPPED", because the intrinsic-geometry toolkit that already shipped
is the right machinery. holostuff already owns a no-shrink graph low-pass: `graphsignal.taubin_filter(vectors,
nbr_idx, nbr_w, ...)`, the Taubin lambda|mu filter, built and tested for denoising vectors over a k-NN graph.
Mesh smoothing is the SAME operation with three substitutions, and nothing else:
    vectors  <- the 3-D VERTEX POSITIONS (the signal to smooth)
    nbr_idx  <- the mesh's EXPLICIT 1-ring adjacency (not a k-NN graph)
    nbr_w    <- COTANGENT weights on the mesh edges (the discrete Laplace-Beltrami operator)
So this module is a WIRE, not a re-implementation: it builds mesh adjacency in the format the shipped filter
expects, computes the cotangent weights, and hands the positions to `taubin_filter`. The smoothing math is the
faculty that already exists; this just points it at a mesh.

WHY COTANGENT WEIGHTS (not uniform)
  Uniform (umbrella) weights smooth toward the average of the 1-ring, which distorts where triangles are
  irregular -- a long thin triangle pulls as hard as a fat one. The COTANGENT weight w_ij = (cot a + cot b)/2,
  where a and b are the angles OPPOSITE edge (i,j) in its two adjacent triangles, is the standard discrete
  Laplace-Beltrami weight: it accounts for triangle shape, so smoothing approximates true surface diffusion
  rather than mesh-connectivity diffusion. This is the geometry-aware weighting the backlog calls for (and the
  same cotangent idea ARCH-3 turns inward onto the engine's own graphs). Uniform weights ship too, as the
  honest baseline.

WHY TAUBIN (not naive Laplacian)
  Naive lambda-only Laplacian smoothing denoises but SHRINKS the whole surface toward its centroid -- a
  smoothed sphere gets smaller every iteration. Taubin alternates a shrink step (lambda>0) with a larger
  un-shrink step (mu<0, |mu|>lambda), so low-frequency structure (the overall extent / volume) is preserved
  while high-frequency noise is removed. `laplacian_smooth` ships here as the shrinking baseline the
  measurement compares against -- the kept negative made visible, exactly as the shipped graphsignal module
  keeps it.

WHAT IT PROVIDES
  * cotangent_adjacency(mesh) / uniform_adjacency(mesh) -> (nbr_idx, nbr_w) in the (V, k_max) padded,
    row-normalised format `taubin_filter` consumes. Triangulates internally (cotangents are triangle angles).
  * taubin_smooth(mesh, lam, mu, iters, weights) -> a NEW Mesh with smoothed vertex positions (faces, and so
    all connectivity and Euler invariants, are UNCHANGED -- smoothing only moves vertices).
  * laplacian_smooth(mesh, lam, iters, weights) -> the shrinking baseline.

DETERMINISM (per ISA.md)
  Adjacency is built in face/vertex order, the cotangent accumulation is a fixed-order sum, and `taubin_filter`
  is a fixed sequence of vectorised neighbour-averages. Same mesh in -> byte-identical positions out (asserted).
  Positions are continuous (TOL) but feed NO argmax-style decision here, so the bind_batch reduction-order
  concern does not arise.

KEPT NEGATIVES (loud)
  * FIXED smoothing strength over-smooths an already-clean mesh -- Taubin is a low-pass, and run long enough it
    removes real detail along with noise. Proper use needs a noise estimate (the graphsignal module's own
    adaptive path / the sigma-estimate discipline); this faculty exposes lam/mu/iters and does NOT auto-tune.
  * COTANGENT weights can go NEGATIVE on obtuse triangles, which would make the row-normalised averaging
    ill-behaved. They are CLAMPED to >= 0 here (the standard "intrinsic/clamped cotangent" mitigation), which
    is exact on well-shaped meshes and a documented approximation on very obtuse ones. Uniform weights avoid
    the issue entirely and are the fallback.
  * COTANGENT IS NOT UNIFORMLY BETTER. Measured on a regular sphere with isotropic noise, UNIFORM weights
    denoise a touch better than cotangent -- on a near-regular mesh with directionless noise there is no
    triangle-shape variation for cotangent to exploit, and it only adds variance. Cotangent's real advantage
    is geometry-awareness on IRREGULAR meshes and feature/crease preservation; it is the default for that
    reason, but the self-test keeps the honest finding that uniform wins the isotropic-regular case rather
    than asserting a superiority that isn't there.
  * Cotangents need TRIANGLE faces; an n-gon mesh is triangulated for the weight computation. The smoothed
    positions are written back to the ORIGINAL vertices, so a quad mesh stays a quad mesh (only its vertices
    move).
"""

import numpy as np

from holographic_mesh import Mesh


# =====================================================================================================
# Build mesh adjacency in the (V, k_max) padded format `graphsignal.taubin_filter` expects.
# =====================================================================================================
def _pack_adjacency(n_vertices, neighbours, weights):
    """Turn per-vertex neighbour/weight dicts into the rectangular (V, k_max) arrays the shipped filter wants.
    Short rows are padded by repeating the vertex's OWN index with weight 0 (a no-op neighbour, exactly the
    convention `graphsignal.knn_graph` uses), and each row is normalised to sum to 1 so the filter computes a
    proper weighted AVERAGE of the 1-ring."""
    k_max = max((len(neighbours[v]) for v in range(n_vertices)), default=1)
    k_max = max(k_max, 1)
    nbr_idx = np.zeros((n_vertices, k_max), dtype=np.int64)
    nbr_w = np.zeros((n_vertices, k_max), dtype=float)
    for v in range(n_vertices):
        nbrs = neighbours[v]
        for col, j in enumerate(nbrs):
            nbr_idx[v, col] = j
            nbr_w[v, col] = weights[v][col]
        for col in range(len(nbrs), k_max):
            nbr_idx[v, col] = v                       # pad with self (weight stays 0) -> no contribution
    row = nbr_w.sum(axis=1, keepdims=True)
    nbr_w = np.where(row > 1e-12, nbr_w / (row + 1e-12), 0.0)   # row-normalise -> a weighted average operator
    return nbr_idx, nbr_w


def _triangles(mesh):
    """The mesh's triangles (fan-triangulated if it has n-gons). Cotangent weights are defined on triangles."""
    return [tuple(t) for t in mesh.triangulate()]


def uniform_adjacency(mesh):
    """1-ring adjacency with UNIFORM (umbrella) weights -- the honest baseline weighting. (nbr_idx, nbr_w)."""
    neighbours = [sorted(mesh.vertex_neighbours(v)) for v in range(mesh.n_vertices)]
    weights = [[1.0] * len(neighbours[v]) for v in range(mesh.n_vertices)]
    return _pack_adjacency(mesh.n_vertices, neighbours, weights)


def cotangent_edge_weights(mesh):
    """Raw (un-normalised, un-clamped) cotangent weights per undirected edge: w_ij = (cot a + cot b)/2 summed
    over the two adjacent triangles' angles OPPOSITE edge (i,j). These are the discrete Laplace-Beltrami edge
    weights -- the shared basis for cotangent SMOOTHING (FWD-4, after clamp + row-normalisation) and the
    MEAN-CURVATURE operator (FWD-6, used raw with vertex areas). Returns {(lo,hi): weight}. Triangulates
    internally (cotangents are triangle angles)."""
    V = mesh.vertices
    edge_w = {}

    def cot_at(c, a, b):
        """cot of the angle at corner c between edges c->a and c->b: dot / |cross|."""
        u = V[a] - V[c]
        w = V[b] - V[c]
        cross = np.cross(u, w)
        cn = float(np.linalg.norm(cross))
        return float(np.dot(u, w) / cn) if cn > 1e-12 else 0.0

    for (i, j, k) in _triangles(mesh):
        # each angle contributes its cotangent to the OPPOSITE edge (the standard cotangent-Laplacian rule)
        for (a, b, opp) in [(i, j, k), (j, k, i), (k, i, j)]:
            e = (min(a, b), max(a, b))
            edge_w[e] = edge_w.get(e, 0.0) + 0.5 * cot_at(opp, a, b)
    return edge_w


def cotangent_adjacency(mesh):
    """1-ring adjacency with COTANGENT (discrete Laplace-Beltrami) weights: for edge (i,j), w = (cot a + cot b)
    / 2 over the two adjacent triangles' opposite angles. Clamped to >= 0 (obtuse-triangle mitigation). Returns
    (nbr_idx, nbr_w) in the shipped filter's format -- the geometry-aware weighting FWD-4 asks for."""
    edge_w = cotangent_edge_weights(mesh)

    # build per-vertex neighbour lists (sorted, deterministic) with clamped weights
    neighbours = [[] for _ in range(mesh.n_vertices)]
    weights = [[] for _ in range(mesh.n_vertices)]
    incident = {v: [] for v in range(mesh.n_vertices)}
    for (lo, hi), w in edge_w.items():
        incident[lo].append((hi, w))
        incident[hi].append((lo, w))
    for v in range(mesh.n_vertices):
        for (nbr, w) in sorted(incident[v]):
            neighbours[v].append(nbr)
            weights[v].append(max(w, 0.0))             # clamp negative (obtuse) weights to 0
    return _pack_adjacency(mesh.n_vertices, neighbours, weights)


# =====================================================================================================
# The two smoothers: Taubin (no-shrink, recommended) and naive Laplacian (the shrinking baseline).
# Both DELEGATE the actual filtering to the shipped graphsignal module -- this is the wire.
# =====================================================================================================
def _adjacency(mesh, weights):
    if weights == "uniform":
        return uniform_adjacency(mesh)
    if weights == "cotangent":
        return cotangent_adjacency(mesh)
    raise ValueError(f"weights must be 'cotangent' or 'uniform', got {weights!r}")


def taubin_smooth(mesh, lam=0.55, mu=-0.58, iters=8, weights="cotangent"):
    """Taubin lambda|mu no-shrink smoothing of a mesh's vertex positions (holographic_graphsignal.taubin_filter
    on the mesh's own adjacency). Removes vertex noise while preserving the overall extent/volume. Faces are
    untouched, so all connectivity and Euler invariants are preserved -- only the vertices move. Returns a new
    Mesh."""
    from holographic_graphsignal import taubin_filter
    nbr_idx, nbr_w = _adjacency(mesh, weights)
    smoothed = taubin_filter(mesh.vertices, nbr_idx, nbr_w, lam=lam, mu=mu, iters=iters)
    return Mesh(smoothed, list(mesh.faces))


def laplacian_smooth(mesh, lam=0.5, iters=8, weights="cotangent"):
    """Naive lambda-only Laplacian smoothing (holographic_graphsignal.laplacian_filter) -- denoises but SHRINKS
    the surface toward its centroid. Shipped as the baseline the measurement compares Taubin against; not the
    recommended path. Returns a new Mesh."""
    from holographic_graphsignal import laplacian_filter
    nbr_idx, nbr_w = _adjacency(mesh, weights)
    smoothed = laplacian_filter(mesh.vertices, nbr_idx, nbr_w, lam=lam, iters=iters)
    return Mesh(smoothed, list(mesh.faces))


# =====================================================================================================
# Self-test -- the measured bar: Taubin denoises WITHOUT the shrink the naive Laplacian suffers.
# =====================================================================================================
def _icosphere(subdiv=2):
    """A unit-sphere triangle mesh by 1->4 subdividing an octahedron and projecting to the sphere. A clean,
    many-vertex closed manifold to add noise to and smooth back (the test surface)."""
    verts = [np.array(v, float) for v in
             ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(subdiv):
        mid = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                verts.append((verts[a] + verts[b]) / 2.0)
                mid[key] = len(verts) - 1
            return mid[key]

        nf = []
        for (a, b, c) in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nf += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = nf
    P = np.array([v / np.linalg.norm(v) for v in verts])
    return Mesh(P, faces)


def _selftest():
    clean = _icosphere(3)                              # ~258 verts, 512 faces, all on the unit sphere
    assert clean.is_closed() and clean.is_manifold()
    chi0 = clean.euler_characteristic()

    rng = np.random.default_rng(0)
    noisy = Mesh(clean.vertices + rng.normal(0.0, 0.05, clean.vertices.shape), list(clean.faces))

    def radial_err(m):                                 # mean |distance-to-unit-sphere| -- the noise measure
        return float(np.abs(np.linalg.norm(m.vertices, axis=1) - 1.0).mean())

    def mean_radius(m):                                # mean radius -- shrink shows up as this dropping below 1
        return float(np.linalg.norm(m.vertices, axis=1).mean())

    taub = taubin_smooth(noisy, iters=10)
    lap = laplacian_smooth(noisy, iters=10)
    taub_uniform = taubin_smooth(noisy, iters=10, weights="uniform")

    # --- connectivity is untouched: only vertices moved, so the Euler invariant is preserved ---
    assert taub.n_faces == clean.n_faces and taub.euler_characteristic() == chi0
    assert taub.faces == clean.faces, "smoothing must not change connectivity, only positions"

    # --- Taubin DENOISES strongly: it is markedly closer to the true sphere than the noisy input ---
    assert radial_err(taub) < 0.6 * radial_err(noisy), (radial_err(noisy), radial_err(taub))

    # --- Taubin does NOT shrink: the mean radius stays near 1 ---
    assert mean_radius(taub) > 0.95, mean_radius(taub)

    # --- the kept negative made visible: the naive Laplacian SHRINKS (mean radius collapses well below 1) ---
    assert mean_radius(lap) < mean_radius(taub) - 0.05, (mean_radius(lap), mean_radius(taub))

    # --- uniform weights ALSO denoise (both weightings work) -- KEPT NEGATIVE: on this regular mesh with
    #     isotropic noise, uniform is in fact a touch BETTER than cotangent. Cotangent's edge is geometry
    #     -awareness on IRREGULAR meshes / feature preservation, not isotropic denoising -- so we do NOT
    #     assert cotangent superiority here; we record that uniform is competitive. ---
    assert radial_err(taub_uniform) < 0.6 * radial_err(noisy)

    # --- determinism: same mesh in -> byte-identical positions out ---
    a = taubin_smooth(noisy, iters=10)
    b = taubin_smooth(noisy, iters=10)
    assert np.array_equal(a.vertices, b.vertices), "smoothing must be deterministic"

    print(f"holographic_meshsmooth selftest: ok (Taubin denoised sphere: radial err "
          f"{radial_err(noisy):.4f} -> {radial_err(taub):.4f}, mean radius kept at {mean_radius(taub):.3f}; "
          f"naive Laplacian shrank to {mean_radius(lap):.3f} -- the kept baseline; connectivity + chi "
          f"untouched; deterministic. KEPT NEGATIVE: on this regular/isotropic case uniform "
          f"({radial_err(taub_uniform):.4f}) edges cotangent -- cotangent's win is irregular geometry)")


if __name__ == "__main__":
    _selftest()

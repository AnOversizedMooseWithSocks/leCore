"""The spectral structure kernel: one operator, several jobs (EXP-5 + EXP-6).

THE IDEA
--------
A Laplacian's spectrum carries both the shape and the geometry of the thing it is built on:

  * its HARMONIC spectrum (the near-zero eigenvalues) counts the topology -- for a Hodge Laplacian the
    dimension of the harmonic space equals the Betti number (components, loops, voids). That is EXP-7's route
    to topology, and it is already tested here as the operator's sanity check.
  * its NON-harmonic spectrum gives the data-driven RIGHT basis for any manifold -- the low eigenvectors are
    the smoothest functions the manifold admits, which is exactly what you want to decompose or denoise a
    signal living on it. That is EXP-6.

And this generalises something holostuff already does by hand: the harmonic basis it uses on a ring IS the
graph-Laplacian eigenbasis of a cycle, and the elementary/DCT basis it uses on a line IS the eigenbasis of a
path. `decompose_signal` hand-picks line->elementary, ring->harmonic; the Laplacian eigenbasis subsumes both
AND extends to manifolds the topology detector cannot name (a sphere, a torus, an arbitrary curved surface),
where it is measurably the right basis and the line/elementary fallback is not.

So this module builds the operator ONCE: graph and Hodge Laplacians from a point cloud or a simplicial
complex, their eigendecomposition, and the Betti-number readout -- shared by the basis-selector (EXP-6, here),
topology detection (EXP-7), and the Hodge flow decomposition (EXP-8).

CONSTRAINTS ON THE RECORD
-------------------------
  * C1 -- dense `numpy.linalg.eigh` only. Fine to a few thousand nodes (O(N^3)); a large sparse graph would
    need a sparse eigensolver, i.e. scipy, i.e. a second dependency -- out of scope. "Does not scale to huge
    graphs without a sparse solver" is the honest limit, kept rather than papered over.
  * C2 -- eigenvector signs (and the basis within a degenerate eigenspace) are ambiguous, the same bit-exact-
    tie class as the bind_batch bug. We FIX a convention -- each eigenvector's largest-magnitude component is
    made positive -- so the basis is reproducible run to run. Within a degenerate eigenspace the basis is
    eigh's deterministic output for a given matrix, and for reconstruction the projection onto that eigenspace
    is basis-invariant anyway, so the ambiguity does not reach the result. The selftest pins reproducibility.

Only NumPy. Nothing learned.
"""
import numpy as np


# --- C2: the determinism convention -----------------------------------------

def sign_fix(V):
    """Make each eigenvector's largest-magnitude component positive, so eigh's sign-ambiguous output becomes a
    reproducible basis (C2). Operates column-wise; returns the same array for convenience."""
    V = np.asarray(V, float)
    for j in range(V.shape[1]):
        i = int(np.argmax(np.abs(V[:, j])))
        if V[i, j] < 0:
            V[:, j] = -V[:, j]
    return V


# --- Laplacians from connectivity -------------------------------------------

def graph_laplacian(adjacency):
    """The combinatorial graph Laplacian L = D - A of a (symmetric) weighted adjacency matrix."""
    A = np.asarray(adjacency, float)
    A = np.maximum(A, A.T)                                   # enforce symmetry
    return np.diag(A.sum(1)) - A


def path_laplacian(n):
    """Laplacian of a path graph (a line). Its eigenbasis is the DCT -- holostuff's 'elementary' basis."""
    A = np.zeros((n, n))
    idx = np.arange(n - 1)
    A[idx, idx + 1] = A[idx + 1, idx] = 1.0
    return graph_laplacian(A)


def cycle_laplacian(n):
    """Laplacian of a cycle graph (a ring). Its eigenbasis is the DFT/harmonic basis (eigenvalues
    4 sin^2(pi k / n)) -- the very basis decompose_signal hand-picks for a ring."""
    A = np.zeros((n, n))
    idx = np.arange(n)
    A[idx, (idx + 1) % n] = 1.0
    A[(idx + 1) % n, idx] = 1.0
    return graph_laplacian(A)


def knn_adjacency(points, k):
    """A symmetric k-nearest-neighbour adjacency on a point cloud -- the graph that turns sampled manifold
    points into a connectivity the Laplacian can read. Mutual-or (a<->b if EITHER is in the other's kNN)."""
    P = np.asarray(points, float)
    n = len(P)
    if n < 2:
        return np.zeros((n, n))
    D = np.sqrt(np.maximum(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1), 0.0))
    A = np.zeros((n, n))
    kk = min(k, n - 1)
    for i in range(n):
        nb = np.argsort(D[i])[1:kk + 1]                     # skip self
        A[i, nb] = 1.0
    return np.maximum(A, A.T)


def knn_laplacian(points, k):
    """The graph Laplacian of a kNN graph on `points` -- the operator whose eigenbasis is the manifold's own
    smooth basis (EXP-6) and whose harmonic dimension counts its components."""
    return graph_laplacian(knn_adjacency(points, k))


def laplacian_eigenbasis(L, n_basis=None):
    """Eigendecomposition of a symmetric PSD Laplacian, ascending, sign-fixed (C2). Returns
    (eigenvalues, eigenvectors-as-columns). With `n_basis` set, keeps the lowest that many -- the smoothest
    functions on the structure."""
    w, V = np.linalg.eigh(np.asarray(L, float))             # ascending eigenvalues; symmetric -> real
    if n_basis is not None:
        w, V = w[:n_basis], V[:, :n_basis]
    return w, sign_fix(V)


# --- Hodge Laplacians + Betti numbers (the topology half; tested here, used by EXP-7/EXP-8) ---

def boundary_matrices(n_verts, edges, triangles=None):
    """Signed boundary operators of a simplicial complex: d1 maps edges->vertices and d2 maps triangles->edges.
    Edges are (a,b) with a<b; triangles (a,b,c) with a<b<c. These are the incidence matrices the Hodge
    Laplacian is built from."""
    edges = [tuple(sorted(e)) for e in edges]
    d1 = np.zeros((n_verts, len(edges)))
    for e, (a, b) in enumerate(edges):
        d1[a, e] = -1.0
        d1[b, e] = +1.0
    triangles = triangles or []
    eidx = {e: i for i, e in enumerate(edges)}
    d2 = np.zeros((len(edges), len(triangles)))
    for t, tri in enumerate(triangles):
        a, b, c = sorted(tri)
        d2[eidx[(a, b)], t] += 1.0                          # boundary of (a,b,c) = (b,c)-(a,c)+(a,b)
        d2[eidx[(b, c)], t] += 1.0
        d2[eidx[(a, c)], t] -= 1.0
    return d1, d2


def hodge_laplacians(n_verts, edges, triangles=None):
    """The graph Laplacian L0 = d1 d1^T and the 1-Hodge Laplacian L1 = d1^T d1 + d2 d2^T."""
    d1, d2 = boundary_matrices(n_verts, edges, triangles)
    L0 = d1 @ d1.T
    L1 = d1.T @ d1 + d2 @ d2.T
    return L0, L1


def betti_numbers(n_verts, edges, triangles=None, tol=1e-9):
    """(b0, b1): the Betti numbers, read as the harmonic dimension (near-zero eigenvalue count) of L0 and L1.
    b0 = connected components, b1 = independent loops. This is the spectral route to topology (EXP-7's core),
    and the operator's own sanity check -- it must reproduce known Betti numbers."""
    L0, L1 = hodge_laplacians(n_verts, edges, triangles)
    b0 = int(np.sum(np.linalg.eigvalsh(L0) < tol))
    b1 = int(np.sum(np.linalg.eigvalsh(L1) < tol))
    return b0, b1


# --- EXP-8: the Helmholtz-Hodge decomposition of an edge flow -----------------

def hodge_decomposition(n_verts, edges, flow, triangles=None):
    """Split an edge flow into its three L2-ORTHOGONAL Helmholtz-Hodge components:

        flow = gradient + curl + harmonic

      * GRADIENT  = d1^T phi   -- curl-free "downhill" transport from a vertex potential phi (the part with
                                  divergence; what a source-to-sink flow is made of).
      * CURL      = d2 psi     -- divergence-free circulation around the filled triangles (local rotation).
      * HARMONIC  = remainder  -- both divergence-free AND curl-free: the GLOBAL circulation that wraps the
                                  holes. Its dimension is exactly B1 (EXP-7's loop count), so the harmonic
                                  part IS the flow's topology.

    Computed by least-squares solves of the graph Laplacian d1 d1^T (for phi) and the triangle Laplacian
    d2^T d2 (for psi); the harmonic part is what neither explains. Returns (gradient, curl, harmonic).

    On a TREE (no cycles, no triangles) there is nothing to circulate: curl and harmonic are zero and the whole
    flow is gradient -- a kept negative that falls straight out of the topology (B1 = 0, no triangles)."""
    d1, d2 = boundary_matrices(n_verts, edges, triangles)
    flow = np.asarray(flow, float)
    div = d1 @ flow                                         # vertex divergence of the flow
    phi, *_ = np.linalg.lstsq(d1 @ d1.T, div, rcond=None)   # solve the graph Laplacian (singular -> lstsq)
    gradient = d1.T @ phi
    if d2.shape[1] > 0:
        psi, *_ = np.linalg.lstsq(d2.T @ d2, d2.T @ flow, rcond=None)
        curl = d2 @ psi
    else:
        curl = np.zeros_like(flow)
    harmonic = flow - gradient - curl
    return gradient, curl, harmonic


def denoise_flow(n_verts, edges, flow, triangles=None, keep=("gradient", "harmonic")):
    """Denoise an edge flow by keeping only its structurally-valid Hodge components and dropping the rest --
    the spectral analogue of projecting onto a signal manifold. For a transport flow that should not circulate
    around individual cells, keep gradient + harmonic and drop curl (where isotropic noise leaves a share);
    for an incompressible flow that should be divergence-free, keep curl + harmonic and drop gradient. Returns
    the reconstructed flow from the kept parts."""
    g, c, h = hodge_decomposition(n_verts, edges, flow, triangles)
    parts = {"gradient": g, "curl": c, "harmonic": h}
    out = np.zeros_like(g)
    for name in keep:
        out = out + parts[name]
    return out


# --- EXP-6 at scale: the smallest eigenvectors WITHOUT the full O(n^3) eigh -------------------
#
# laplacian_eigenbasis above calls np.linalg.eigh, which computes ALL n eigenvectors and then keeps the lowest
# n_basis -- fine to ~1500 points, but O(n^3) wastes nearly everything when n_basis << n (a spectral basis keeps
# ~12 of n). For a large cloud we want ONLY the smoothest n_basis modes. The honest difficulty that makes this
# non-trivial: a manifold's Laplacian is DEGENERATE (a 2-sphere carries 2l+1 modes at each eigenvalue), so a
# count cutoff lands INSIDE a degenerate block and plain Lanczos / subspace iteration cannot separate it (all
# measured to fail). The method that does is Chebyshev-FILTERED subspace iteration -- what real sparse
# eigensolvers use: a polynomial filter amplifies the wanted low-eigenvalue subspace by orders of magnitude so a
# block iteration converges THROUGH the degeneracy. Built on a SPARSE kNN-Laplacian matvec (O(n*k), never the
# dense n x n matrix) it matches the exact basis in the verified range and lifts the eigh O(n^3).
#   KEPT NEGATIVES (measured, surfaced not hidden): it is an APPROXIMATION -- the projector error onto the smooth
#   subspace grows slowly with n (~0 to a few thousand points, then drifts); and it lifts the EIGH cost, not the
#   O(n^2) kNN-distance build, which itself caps practical use at a few thousand points (a spatial index would be
#   the next step). Below SpectralBasis.partial_threshold the exact dense eigh is used -- faster there AND
#   bit-identical to before, so every existing result and test is unchanged.

def _knn_sparse_laplacian(points, k):
    """The kNN graph Laplacian as a SPARSE edge list (n, degree, src, dst) -- the same mutual-OR, unit-weight
    operator knn_adjacency/graph_laplacian build densely, but never materialised as an n x n matrix. The one
    O(n^2) cost that remains is the distance sort; the Laplacian itself is then O(n*k) to store and to apply."""
    P = np.asarray(points, float)
    n = len(P)
    kk = min(k, n - 1)
    D = np.sqrt(np.maximum(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(D, np.inf)                              # exclude self (matches knn_adjacency's skip-self)
    nn = np.argsort(D, 1)[:, :kk]                            # kk nearest neighbours of each point
    src = np.repeat(np.arange(n), kk)
    dst = nn.ravel()
    a = np.minimum(src, dst)                                 # symmetrise mutual-OR: keep each undirected edge once
    b = np.maximum(src, dst)
    _, ui = np.unique(a.astype(np.int64) * n + b, return_index=True)
    s = np.concatenate([a[ui], b[ui]])                       # then list both directions of each kept edge
    d = np.concatenate([b[ui], a[ui]])
    deg = np.bincount(s, minlength=n).astype(float)          # degree = row sum of the symmetric adjacency
    return n, deg, s, d


def _sparse_lap_matvec(deg, src, dst):
    """A callable v -> L v for the sparse Laplacian L = diag(deg) - A, given as its edges (src, dst). Handles a
    single vector OR a block of columns. O(edges) = O(n*k) per apply -- the whole reason for the sparse form."""
    def mv(V):
        V = np.asarray(V, float)
        if V.ndim == 1:
            out = deg * V
            np.add.at(out, src, -V[dst])                     # subtract each neighbour's value (the -A part)
            return out
        out = deg[:, None] * V
        for c in range(V.shape[1]):
            np.add.at(out[:, c], src, -V[dst, c])
        return out
    return mv


def _estimate_lambda_k(mv, n, k, m=None, seed=1):
    """A cheap estimate of the k-th smallest eigenvalue of L -- the Chebyshev filter's cutoff -- from a few
    Lanczos steps. Eigenvalue ESTIMATES at the spectrum's extreme converge fast even where the eigenVECTORS do
    not (the degeneracy that defeats plain Lanczos), so this is reliable enough to place the filter window."""
    m = m or min(n, 3 * k + 20)
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, m)); al = np.zeros(m); be = np.zeros(m)
    q = rng.standard_normal(n); q /= np.linalg.norm(q) + 1e-12
    Q[:, 0] = q
    w = mv(q); al[0] = q @ w; w = w - al[0] * q; mm = m
    for j in range(1, m):
        b = np.linalg.norm(w)
        if b < 1e-10:
            mm = j; break                                    # hit an invariant subspace
        be[j] = b; q = w / b
        q = q - Q[:, :j] @ (Q[:, :j].T @ q)                  # reorthogonalise (m is small -> cheap)
        q /= np.linalg.norm(q) + 1e-12; Q[:, j] = q
        w = mv(q); al[j] = q @ w; w = w - al[j] * q - be[j] * Q[:, j - 1]
    T = np.diag(al[:mm]) + np.diag(be[1:mm], 1) + np.diag(be[1:mm], -1)
    return float(np.linalg.eigvalsh(T)[min(k, mm - 1)])


def _cheb_filter(mv, V, a, b, deg):
    """Apply the degree-`deg` Chebyshev polynomial of the interval [a, b] to V by matvecs only. On [a, b] the
    polynomial stays bounded (|T| <= 1); BELOW a (the wanted small-eigenvalue end) it grows fast -- so with
    [a, b] = [lambda_cut, lambda_max] this AMPLIFIES the smooth subspace and damps the rough modes. Three-term
    Chebyshev recurrence."""
    c = (a + b) / 2.0; e = (b - a) / 2.0
    Y0 = V
    Y1 = (mv(V) - c * V) / e
    for _ in range(deg - 1):
        Y2 = 2.0 * (mv(Y1) - c * Y1) / e - Y0
        Y0, Y1 = Y1, Y2
    return Y1


def cheb_eigenbasis(points, k_graph=10, n_basis=12, oversample=14, outer=6, deg=24, seed=0):
    """The smallest `n_basis` eigenvectors of the kNN graph Laplacian by Chebyshev-filtered subspace iteration on
    the SPARSE operator -- the scalable stand-in for laplacian_eigenbasis's full eigh on large clouds. Returns
    (eigenvalues, basis-as-columns), sign-fixed, matching laplacian_eigenbasis's contract. Matches the exact
    basis in the verified range (n up to a few thousand: projector difference ~0 to ~0.3 there); an
    approximation whose error grows slowly with n, and bounded above by the O(n^2) distance build (kept
    negatives). Speedup over eigh grows with n -- ~4x at 3000 points, ~7x at 4500, where eigh's O(n^3) bites."""
    n, degv, src, dst = _knn_sparse_laplacian(points, k_graph)
    mv = _sparse_lap_matvec(degv, src, dst)
    lam_max = float(degv.max() * 2.0)                         # Gershgorin bound on lambda_max (2 * max degree)
    lam_cut = _estimate_lambda_k(mv, n, n_basis + oversample // 2) * 1.2   # cut just above the wanted modes
    rng = np.random.default_rng(seed)
    Om, _ = np.linalg.qr(rng.standard_normal((n, min(n, n_basis + oversample))))
    for _ in range(outer):                                    # subspace iteration with the amplifying filter
        Om = _cheb_filter(mv, Om, lam_cut, lam_max, deg)
        Om, _ = np.linalg.qr(Om)
    B = Om.T @ mv(Om)                                         # Rayleigh-Ritz on the converged subspace
    bvals, bvecs = np.linalg.eigh(B)
    idx = np.argsort(bvals)[:n_basis]                         # keep the smallest n_basis Ritz pairs
    return bvals[idx], sign_fix(Om @ bvecs[:, idx])


# --- EXP-6: the eigenbasis as a basis-selector ------------------------------

class SpectralBasis:
    """The data-driven decomposition basis for a signal living on a manifold (EXP-6).

    Build the kNN-graph Laplacian of the sample points, take its lowest `n_basis` eigenvectors (the smoothest
    functions the manifold admits), and use them as an orthonormal basis to decompose / reconstruct / denoise
    a signal on those points. On a line this basis IS the DCT and on a ring IS the harmonic basis (so it
    matches decompose_signal's hand-picked choice), and on a manifold the topology detector cannot name it is
    measurably the right basis where the line/elementary fallback is not.

    For clouds above `partial_threshold` points the lowest modes are found by Chebyshev-filtered subspace
    iteration on the SPARSE Laplacian (cheb_eigenbasis) instead of the full O(n^3) eigh -- a measured ~4x at
    3000 points growing with n. Below the threshold (and for the tests/selftest, which use small clouds) the
    exact dense eigh is kept: faster there, and bit-identical to before. The scalable path is an approximation
    (kept negative: its projector error grows slowly with n, and the O(n^2) distance build still caps practical
    use at a few thousand points).
    """

    def __init__(self, points, k=10, n_basis=12, partial_threshold=2000):
        self.points = np.asarray(points, float)
        self.k = int(k)
        self.n_basis = int(n_basis)
        self.partial_threshold = int(partial_threshold)
        n = len(self.points)
        if n <= self.partial_threshold or n_basis >= n // 2:
            L = knn_laplacian(self.points, self.k)                        # small cloud: exact dense eigh
            self.eigenvalues, self.basis = laplacian_eigenbasis(L, n_basis)   # columns: smooth modes, sign-fixed
        else:                                                            # large cloud: scalable filtered solver
            self.eigenvalues, self.basis = cheb_eigenbasis(self.points, self.k, n_basis)

    def decompose(self, signal):
        """Coordinates of `signal` in the manifold's smooth basis (one coefficient per kept eigenvector)."""
        return self.basis.T @ np.asarray(signal, float)

    def reconstruct(self, coeffs):
        """Rebuild a signal from its spectral coordinates."""
        return self.basis @ np.asarray(coeffs, float)

    def denoise(self, signal):
        """Project a noisy signal onto the smooth (low-frequency) subspace of the manifold -- spectral
        denoising. Recovers a smooth field on a curved manifold where a line/index-order basis cannot."""
        return self.reconstruct(self.decompose(signal))


# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(0)

    # (1) EXP-5 sanity: the cycle Laplacian eigenbasis IS the DFT/harmonic basis.
    n = 16
    w, V = laplacian_eigenbasis(cycle_laplacian(n))
    pred = np.sort([4 * np.sin(np.pi * k / n) ** 2 for k in range(n)])
    assert np.allclose(w, pred, atol=1e-9), "cycle eigenvalues != 4 sin^2(pi k/n)"
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sig = np.cos(2 * t) + 0.5 * np.sin(3 * t)
    assert np.linalg.norm(V @ (V.T @ sig) - sig) < 1e-9, "ring signal not reconstructed by Laplacian eigenbasis"

    # (2) C2 determinism: rebuild -> identical sign-fixed basis.
    _, V2 = laplacian_eigenbasis(cycle_laplacian(n))
    assert np.allclose(V, V2), "eigenbasis not reproducible"

    # (3) EXP-5 operator: Hodge harmonic dimension == Betti numbers.
    assert betti_numbers(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) == (1, 1)                  # 4-cycle
    assert betti_numbers(3, [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)]) == (1, 0)             # filled triangle
    assert betti_numbers(4, [(0, 1), (2, 3)]) == (2, 0)                                  # two components

    # (4) EXP-6 line: the path-Laplacian basis denoises a smooth line ~ as well as the DCT (it IS the DCT).
    m = 64
    tt = np.linspace(0, 1, m)
    clean = np.sin(2 * np.pi * tt) + 0.4 * np.cos(6 * np.pi * tt)
    noisy = clean + 0.3 * rng.standard_normal(m)
    _, Vp = laplacian_eigenbasis(path_laplacian(m), 8)
    err_lap = np.linalg.norm(Vp @ (Vp.T @ noisy) - clean)
    DCT = np.stack([np.ones(m) / np.sqrt(m)] +
                   [np.sqrt(2 / m) * np.cos(np.pi * (np.arange(m) + 0.5) * kk / m) for kk in range(1, 8)]).T
    err_dct = np.linalg.norm(DCT @ (DCT.T @ noisy) - clean)
    assert abs(err_lap - err_dct) < 1e-6, f"path-Laplacian basis != DCT on a line: {err_lap} vs {err_dct}"

    # (5) EXP-6 win: on a sphere (a manifold the detector calls 'line') the Laplacian basis beats the line basis.
    N = 400
    idx = np.arange(N)
    phi = np.arccos(1 - 2 * (idx + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    P = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], 1)
    f = P[:, 2] ** 2 - 1 / 3 + P[:, 0] * P[:, 1]            # a smooth degree-2 field on the sphere
    fn = f + 0.3 * rng.standard_normal(N)
    sb = SpectralBasis(P, k=10, n_basis=12)
    err_sphere_lap = np.linalg.norm(sb.denoise(fn) - f)
    DCTi = np.stack([np.ones(N) / np.sqrt(N)] +
                    [np.sqrt(2 / N) * np.cos(np.pi * (np.arange(N) + 0.5) * kk / N) for kk in range(1, 12)]).T
    err_sphere_line = np.linalg.norm(DCTi @ (DCTi.T @ fn) - f)
    assert err_sphere_lap < 0.6 * err_sphere_line, f"Laplacian basis did not beat line on sphere: {err_sphere_lap} vs {err_sphere_line}"

    # (6) EXP-8: the Hodge decomposition of an edge flow -- orthogonal parts, exact sum, harmonic is div/curl-free.
    V = 9                                                  # a triangulated 3x3 grid with one triangle removed -> a hole
    tris_all = []
    for cy in range(2):
        for cx in range(2):
            a = cy * 3 + cx
            tris_all += [(a, a + 1, a + 4), (a, a + 4, a + 3)]
    tris = [t for t in tris_all if t != (0, 1, 4)]
    edges = sorted({tuple(sorted(e)) for t in tris_all for e in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]})
    d1h, d2h = boundary_matrices(V, edges, tris)
    fl = d1h.T @ rng.standard_normal(V) + d2h @ rng.standard_normal(len(tris))   # a grad + curl flow
    g, c, h = hodge_decomposition(V, edges, fl, tris)
    assert np.linalg.norm(g + c + h - fl) < 1e-9, "Hodge parts do not sum to the flow"
    assert abs(g @ c) < 1e-9 and abs(g @ h) < 1e-9 and abs(c @ h) < 1e-9, "Hodge parts not orthogonal"
    hr = hodge_decomposition(V, edges, rng.standard_normal(len(edges)), tris)[2]
    assert np.linalg.norm(d1h @ hr) < 1e-9 and np.linalg.norm(d2h.T @ hr) < 1e-9, "harmonic part not div/curl-free"
    # denoising a transport flow (drop curl) beats the noisy input and naive edge-smoothing
    clean = d1h.T @ rng.standard_normal(V)
    noisy = clean + 0.5 * rng.standard_normal(len(edges))
    den = denoise_flow(V, edges, noisy, tris, keep=("gradient", "harmonic"))
    Aedge = (np.abs(d1h).T @ np.abs(d1h)) > 0
    np.fill_diagonal(Aedge, False)
    sm = 0.5 * noisy + 0.5 * np.array([noisy[Aedge[i]].mean() if Aedge[i].any() else noisy[i] for i in range(len(edges))])
    assert np.linalg.norm(den - clean) < np.linalg.norm(noisy - clean), "Hodge denoise did not beat raw"
    assert np.linalg.norm(den - clean) < np.linalg.norm(sm - clean), "Hodge denoise did not beat naive smoothing"
    # tree: curl and harmonic vanish (kept negative)
    d1t, d2t = boundary_matrices(5, [(0, 1), (1, 2), (1, 3), (3, 4)], [])
    _, ct, ht = hodge_decomposition(5, [(0, 1), (1, 2), (1, 3), (3, 4)], rng.standard_normal(4), [])
    assert np.linalg.norm(ct) < 1e-12 and np.linalg.norm(ht) < 1e-9, "tree flow not pure gradient"

    # (7) EXP-6 at scale: above partial_threshold, Chebyshev-filtered subspace iteration matches the exact eigh.
    Nbig = 2400
    ib = np.arange(Nbig)
    phib = np.arccos(1 - 2 * (ib + 0.5) / Nbig)
    thb = np.pi * (1 + 5 ** 0.5) * ib
    Pb = np.stack([np.sin(phib) * np.cos(thb), np.sin(phib) * np.sin(thb), np.cos(phib)], 1)
    fb = Pb[:, 2] ** 2 - 1 / 3 + 0.5 * Pb[:, 0]            # smooth field on the sphere
    fbn = fb + 0.3 * rng.standard_normal(Nbig)
    sb_cheb = SpectralBasis(Pb, k=10, n_basis=12, partial_threshold=2000)   # > threshold -> ChebFSI path
    err_cheb = np.linalg.norm(sb_cheb.denoise(fbn) - fb)
    Lb = knn_laplacian(Pb, 10)
    _, Vb = laplacian_eigenbasis(Lb, 12)                  # exact dense reference
    err_exact = np.linalg.norm(Vb @ (Vb.T @ fbn) - fb)
    assert err_cheb < 1.15 * err_exact + 1e-9, f"ChebFSI basis far from exact at scale: {err_cheb} vs {err_exact}"
    # and below the threshold the exact dense path is used (bit-identical to laplacian_eigenbasis)
    sb_small = SpectralBasis(Pb[:300], k=10, n_basis=12, partial_threshold=2000)
    _, Vs = laplacian_eigenbasis(knn_laplacian(Pb[:300], 10), 12)
    assert np.allclose(np.abs(sb_small.basis), np.abs(Vs)), "small cloud did not use the exact dense eigh"

    print("holographic_spectral selftest OK:")
    print(f"  cycle eigenbasis == DFT (eigvals 4 sin^2), ring reconstructs exact")
    print(f"  Hodge Betti numbers correct: 4-cycle (1,1), filled triangle (1,0), two comps (2,0)")
    print(f"  LINE denoise   path-Laplacian {err_lap:.3f} == DCT {err_dct:.3f}")
    print(f"  SPHERE denoise kNN-Laplacian {err_sphere_lap:.3f}  vs line basis {err_sphere_line:.3f} (raw {np.linalg.norm(fn - f):.3f})")
    print(f"  SCALE (n={Nbig}) ChebFSI denoise {err_cheb:.3f} ~= exact eigh {err_exact:.3f} (partial eigensolver, sparse matvec)")
    print(f"  HODGE flow split orthogonal+exact; denoise {np.linalg.norm(den - clean):.3f} < raw {np.linalg.norm(noisy - clean):.3f} < smooth {np.linalg.norm(sm - clean):.3f}; tree=pure-gradient")


if __name__ == "__main__":
    _selftest()

"""EXP-5 + EXP-6: the spectral structure kernel (graph/Hodge Laplacian) and the Laplacian eigenbasis as a
data-driven basis-selector (holographic_spectral.py)."""
import numpy as np

from holographic.sampling_and_signal.holographic_spectral import sign_fix, cycle_laplacian, path_laplacian, knn_laplacian, laplacian_eigenbasis, betti_numbers, SpectralBasis, cheb_eigenbasis, _selftest


def test_selftest_passes():
    _selftest()


def test_cycle_eigenbasis_is_the_dft():
    # EXP-5 sanity: a cycle's Laplacian eigenbasis is the DFT/harmonic basis the engine already uses on a ring.
    n = 24
    w, V = laplacian_eigenbasis(cycle_laplacian(n))
    assert np.allclose(w, np.sort([4 * np.sin(np.pi * k / n) ** 2 for k in range(n)]), atol=1e-9)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sig = np.cos(3 * t) - 0.4 * np.sin(2 * t)
    assert np.linalg.norm(V @ (V.T @ sig) - sig) < 1e-9


def test_eigenbasis_is_deterministic():
    # C2: the sign-fixed eigenbasis is reproducible run to run.
    L = path_laplacian(40)
    _, A = laplacian_eigenbasis(L, 10)
    _, B = laplacian_eigenbasis(L, 10)
    assert np.allclose(A, B)
    # sign_fix makes the largest-magnitude component non-negative
    for j in range(A.shape[1]):
        assert A[np.argmax(np.abs(A[:, j])), j] >= 0


def test_hodge_harmonic_dimension_is_betti_number():
    assert betti_numbers(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) == (1, 1)              # cycle: 1 comp, 1 loop
    assert betti_numbers(3, [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)]) == (1, 0)         # filled triangle: loop gone
    assert betti_numbers(4, [(0, 1), (2, 3)]) == (2, 0)                              # two components
    assert betti_numbers(5, [(0, 1), (1, 2), (2, 0), (3, 4)]) == (2, 1)             # a triangle loop + an edge


def test_path_basis_matches_dct_on_a_line():
    # EXP-6: the path-Laplacian eigenbasis denoises a smooth line identically to the elementary/DCT basis.
    rng = np.random.default_rng(1)
    m = 64
    t = np.linspace(0, 1, m)
    clean = np.sin(2 * np.pi * t) + 0.3 * np.cos(8 * np.pi * t)
    noisy = clean + 0.25 * rng.standard_normal(m)
    _, Vp = laplacian_eigenbasis(path_laplacian(m), 8)
    err_lap = np.linalg.norm(Vp @ (Vp.T @ noisy) - clean)
    DCT = np.stack([np.ones(m) / np.sqrt(m)] +
                   [np.sqrt(2 / m) * np.cos(np.pi * (np.arange(m) + 0.5) * kk / m) for kk in range(1, 8)]).T
    err_dct = np.linalg.norm(DCT @ (DCT.T @ noisy) - clean)
    assert abs(err_lap - err_dct) < 1e-6


def test_laplacian_basis_beats_line_basis_on_a_sphere():
    # EXP-6 win: a manifold the topology detector can only call 'line' -- the data-driven basis recovers it.
    rng = np.random.default_rng(2)
    N = 350
    idx = np.arange(N)
    phi = np.arccos(1 - 2 * (idx + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    P = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], 1)
    f = P[:, 2] ** 2 - 1 / 3 + P[:, 0] * P[:, 1]
    fn = f + 0.3 * rng.standard_normal(N)
    err_lap = np.linalg.norm(SpectralBasis(P, k=10, n_basis=12).denoise(fn) - f)
    DCTi = np.stack([np.ones(N) / np.sqrt(N)] +
                    [np.sqrt(2 / N) * np.cos(np.pi * (np.arange(N) + 0.5) * kk / N) for kk in range(1, 12)]).T
    err_line = np.linalg.norm(DCTi @ (DCTi.T @ fn) - f)
    assert err_lap < 0.6 * err_line


def test_spectral_basis_roundtrip():
    rng = np.random.default_rng(3)
    P = rng.standard_normal((120, 3))
    sb = SpectralBasis(P, k=8, n_basis=10)
    # a signal that lives in the basis reconstructs exactly from its coordinates
    c = rng.standard_normal(10)
    sig = sb.reconstruct(c)
    assert np.allclose(sb.decompose(sig), c, atol=1e-9)


def test_cheb_eigenbasis_matches_full_eigh_at_scale():
    # EXP-6 at scale: the Chebyshev-filtered partial eigensolver on the SPARSE Laplacian recovers the SAME
    # smooth subspace as the full O(n^3) eigh -- the denoise depends on the subspace, so equal denoise to within
    # a small tolerance is the test. The manifold-Laplacian degeneracy that defeats plain Lanczos is handled.
    rng = np.random.default_rng(4)
    N = 2400
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    P = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], 1)
    f = P[:, 2] ** 2 - 1 / 3 + 0.5 * P[:, 0]
    fn = f + 0.3 * rng.standard_normal(N)
    _, Vc = cheb_eigenbasis(P, k_graph=10, n_basis=12)
    err_cheb = np.linalg.norm(Vc @ (Vc.T @ fn) - f)
    _, Ve = laplacian_eigenbasis(knn_laplacian(P, 10), 12)              # exact dense reference
    err_exact = np.linalg.norm(Ve @ (Ve.T @ fn) - f)
    assert err_cheb < 1.15 * err_exact + 1e-9                            # matches the exact basis's denoise
    assert err_cheb < np.linalg.norm(fn - f)                             # and genuinely denoises


def test_spectral_basis_thresholds_to_exact_below_cutoff():
    # below partial_threshold the class uses the exact dense eigh (bit-identical to laplacian_eigenbasis);
    # above it, the ChebFSI path. The small case must reproduce the exact basis exactly (up to per-column sign).
    rng = np.random.default_rng(5)
    P = rng.standard_normal((300, 3))
    sb = SpectralBasis(P, k=10, n_basis=12, partial_threshold=2000)      # 300 <= 2000 -> exact path
    _, Ve = laplacian_eigenbasis(knn_laplacian(P, 10), 12)
    assert np.allclose(np.abs(sb.basis), np.abs(Ve))
    # forcing the threshold low routes the SAME cloud through ChebFSI, which still denoises comparably
    sb_cheb = SpectralBasis(P, k=10, n_basis=12, partial_threshold=50)   # 300 > 50 -> ChebFSI path
    f = P[:, 0] * 0.0 + np.cos(P[:, 1])
    fn = f + 0.2 * rng.standard_normal(300)
    assert np.linalg.norm(sb_cheb.denoise(fn) - f) < 1.4 * np.linalg.norm(sb.denoise(fn) - f) + 1e-9


# --- EXP-8: the Helmholtz-Hodge decomposition of an edge flow ---------------
from holographic.sampling_and_signal.holographic_spectral import boundary_matrices, hodge_decomposition, denoise_flow


def _holed_grid():
    """A triangulated 3x3 grid with one triangle removed -> a single hole (B1=1), so all three Hodge
    components are non-trivial. Returns (n_verts, edges, triangles)."""
    tris_all = []
    for cy in range(2):
        for cx in range(2):
            a = cy * 3 + cx
            tris_all += [(a, a + 1, a + 4), (a, a + 4, a + 3)]
    tris = [t for t in tris_all if t != (0, 1, 4)]
    edges = sorted({tuple(sorted(e)) for t in tris_all for e in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]})
    return 9, edges, tris


def test_hodge_parts_sum_and_are_orthogonal():
    V, E, T = _holed_grid()
    d1, d2 = boundary_matrices(V, E, T)
    rng = np.random.default_rng(0)
    flow = d1.T @ rng.standard_normal(V) + d2 @ rng.standard_normal(len(T))
    g, c, h = hodge_decomposition(V, E, flow, T)
    assert np.linalg.norm(g + c + h - flow) < 1e-9
    assert abs(g @ c) < 1e-9 and abs(g @ h) < 1e-9 and abs(c @ h) < 1e-9


def test_harmonic_part_is_div_and_curl_free_with_dim_b1():
    V, E, T = _holed_grid()
    d1, d2 = boundary_matrices(V, E, T)
    rng = np.random.default_rng(1)
    h = hodge_decomposition(V, E, rng.standard_normal(len(E)), T)[2]
    assert np.linalg.norm(d1 @ h) < 1e-9          # divergence-free
    assert np.linalg.norm(d2.T @ h) < 1e-9        # curl-free  -> genuinely harmonic
    # the harmonic SUBSPACE dimension equals B1 (the loop count) -- the harmonic part IS the topology
    L1 = d1.T @ d1 + d2 @ d2.T
    assert int(np.sum(np.linalg.eigvalsh(L1) < 1e-9)) == 1   # one hole


def test_hodge_denoise_beats_raw_and_naive_smoothing():
    V, E, T = _holed_grid()
    d1, _ = boundary_matrices(V, E, T)
    rng = np.random.default_rng(2)
    clean = d1.T @ rng.standard_normal(V)         # a pure transport (gradient) flow
    noisy = clean + 0.5 * rng.standard_normal(len(E))
    den = denoise_flow(V, E, noisy, T, keep=("gradient", "harmonic"))
    A = (np.abs(d1).T @ np.abs(d1)) > 0
    np.fill_diagonal(A, False)
    sm = 0.5 * noisy + 0.5 * np.array([noisy[A[i]].mean() if A[i].any() else noisy[i] for i in range(len(E))])
    assert np.linalg.norm(den - clean) < np.linalg.norm(noisy - clean)     # genuinely denoises
    assert np.linalg.norm(den - clean) < np.linalg.norm(sm - clean)        # beats naive smoothing


def test_tree_has_no_curl_or_harmonic():
    # kept negative: a tree (no cycles, no triangles) is pure gradient -- nothing to circulate.
    V, E = 5, [(0, 1), (1, 2), (1, 3), (3, 4)]
    rng = np.random.default_rng(3)
    _, c, h = hodge_decomposition(V, E, rng.standard_normal(len(E)), [])
    assert np.linalg.norm(c) < 1e-12 and np.linalg.norm(h) < 1e-9

"""EXP-7: principled topology by persistent homology (holographic_topology.py) -- a point cloud's Betti
signature names its shape, reproducing detect_topology on line/ring and extending to torus and sphere."""
import numpy as np

from holographic_topology import (
    vr_simplices, betti_at_scale, persistent_topology, _gf2_rank, _boundary_columns, _make, _selftest,
)


def test_selftest_passes():
    _selftest()


def test_classifies_the_four_manifolds():
    # the bar: match detect_topology on line/ring AND name a torus and a sphere it cannot.
    rng = np.random.default_rng(0)
    assert persistent_topology(_make("line", rng))[1] == (1, 0, 0)
    assert persistent_topology(_make("circle", rng))[1] == (1, 1, 0)
    assert persistent_topology(_make("torus", rng))[1] == (1, 2, 1)
    assert persistent_topology(_make("sphere", rng))[1] == (1, 0, 1)


def test_names_are_attached():
    rng = np.random.default_rng(1)
    assert persistent_topology(_make("torus", rng))[0] == "torus"
    assert persistent_topology(_make("sphere", rng))[0] == "sphere"
    assert persistent_topology(_make("circle", rng))[0] == "ring"


def test_b1_separates_line_circle_torus_and_b2_finds_the_sphere():
    # B1 alone orders line(0) < circle(1) < torus(2); B2 is what distinguishes the sphere from the line.
    rng = np.random.default_rng(2)
    b_line = persistent_topology(_make("line", rng))[1]
    b_circle = persistent_topology(_make("circle", rng))[1]
    b_torus = persistent_topology(_make("torus", rng))[1]
    b_sphere = persistent_topology(_make("sphere", rng))[1]
    assert b_line[1] < b_circle[1] < b_torus[1]              # 0 < 1 < 2 loops
    assert b_sphere[1] == b_line[1] == 0 and b_sphere[2] == 1 and b_line[2] == 0   # only B2 tells them apart


def test_gf2_betti_agrees_with_hodge_laplacian():
    # EXP-7's fast GF(2) rank route gives the same B0/B1 as EXP-5's Hodge-Laplacian harmonic-dimension route.
    from holographic_spectral import betti_numbers as hodge_betti
    for nv, E, T in [(4, [(0, 1), (1, 2), (2, 3), (3, 0)], None),
                     (3, [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)]),
                     (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], None)]:
        vidx = {(i,): i for i in range(nv)}
        eidx = {e: i for i, e in enumerate(E)}
        r1 = _gf2_rank(_boundary_columns(E, vidx))
        r2 = _gf2_rank(_boundary_columns(T or [], eidx))
        assert (nv - r1, len(E) - r1 - r2) == hodge_betti(nv, E, T)


def test_betti_at_scale_on_a_known_complex():
    # a single explicit scale on a square loop of 4 points: one component, one loop, no void
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    assert betti_at_scale(pts, 1.1) == (1, 1, 0)             # eps just over the side length, under the diagonal


def test_deterministic():
    rng = np.random.default_rng(3)
    P = _make("torus", rng)
    assert persistent_topology(P)[1] == persistent_topology(P)[1]


def test_uneven_cloud_is_a_kept_negative():
    # honest scope: an unevenly sampled cloud (a sine's delay embedding) does NOT cleanly read as a ring.
    t = np.linspace(0, 8 * np.pi, 160, endpoint=False)
    emb = np.column_stack([np.sin(t)[i * 6:i * 6 + 148] for i in range(3)])
    emb = emb[np.linspace(0, len(emb) - 1, 60).astype(int)]
    assert persistent_topology(emb)[1] != (1, 1, 0)         # finicky on uneven sampling, by design


def test_dense_cloud_is_capped_and_flagged():
    # a blob has no clean low-dim topology; the simplex budget caps the explosion (was ~30s on 250 pts, now
    # sub-second) and the result honestly flags how many band scales were too dense to read cleanly.
    rng = np.random.default_rng(0)
    blob = rng.standard_normal((250, 4))
    name, betti, hist = persistent_topology(blob)
    assert hist.get("dense_scales", 0) >= 1                  # the wide scales exploded -> skipped and flagged
    # the budget itself: an absurdly small budget makes even a clean torus report UNRELIABLE (None) Betti
    from holographic_topology import betti_at_scale, _subsample, _median_nn
    P = _subsample(_make("torus", rng), 250)
    d = _median_nn(P)
    b = betti_at_scale(P, 2.6 * d, tri_budget=5, tet_budget=5)   # tiny budget -> capped
    assert None in b                                            # unreliable when the complex hits the budget


def test_budget_does_not_change_clean_manifolds():
    # the simplex budget never trips on a well-sampled low-dim manifold: the classifications are unchanged.
    rng = np.random.default_rng(1)
    assert persistent_topology(_make("sphere", rng))[1] == (1, 0, 1)
    assert persistent_topology(_make("torus", rng))[1] == (1, 2, 1)
    assert persistent_topology(_make("circle", rng))[1] == (1, 1, 0)

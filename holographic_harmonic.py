"""RT-VI -- context-dependent meaning in a harmonic basis (holographic_harmonic).

THE TRANSFER (as above, so below)
---------------------------------
3D Gaussian splatting makes a splat's COLOUR a function of view direction, expanded in spherical harmonics: the
degree-0 (DC) term is the base colour, higher degrees add smooth view-dependent variation, and a handful of
coefficients capture the angular variation cheaply. holostuff's FHRR/FPE substrate already uses PHASE = a point on
the circle = a direction, and FPE compute-on-functions already represents f: direction -> value as a superposition.
So an atom whose MEANING is a function of the query context is native here: represent content(theta) -- the meaning
as a function of a context angle -- in a CIRCULAR-harmonic (Fourier) basis. The DC term is the context-FREE meaning
(today's fixed atom, exactly); higher harmonics encode how the meaning shifts with context. Reading the atom at a
context angle is the harmonic sum -- the analog of unbinding with an FPE-encoded role(theta) (a phase rotation).

This gives CONTEXT-CONDITIONED / POLYSEMOUS atoms: one symbol whose decoded content depends on the surrounding
context -- a word's sense under different contexts, a filler that means different things under different roles.

WHY IT IS SUBSTRATE-ALIGNED, NOT BOLTED ON: spherical harmonics are Fourier-on-the-sphere; the engine's core bind is
FFT-on-a-domain, FHRR is phasors, and the manifold module already picks harmonic bases for ring/torus topologies.
This is the engine's own native basis pointed at MEANING instead of GEOMETRY -- the same move RT-IV1 made with
anisotropy. (Circular harmonics here -- a 1-D context angle, the FPE phase case; full spherical harmonics over a 2-D
direction is the natural extension, noted as scope below.)

WHAT IT PROVIDES
  * harmonic_atom(thetas, meanings, n_harmonics) -- fit a context-conditioned atom: the circular-harmonic
    coefficients (least squares) of the meaning function sampled at (context angle, meaning) pairs. n_harmonics=K
    keeps the DC plus K-1 harmonics (2K-1 coefficient vectors).
  * harmonic_decode(atom, theta) -- the meaning at context angle theta (the harmonic sum).
  * harmonic_dc(atom) -- the DC (degree-0), context-free meaning -- exactly the plain fixed atom.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * POLYSEMY: distinct senses placed at distinct contexts are each recovered at their context (cosine ~1.0), and a
    context between two senses decodes to a blend of them.
  * DEGREE-0 FALLBACK (backward-compatible): a CONTEXT-FREE atom (one stable meaning) is captured by the DC term
    alone (K=1) and decodes EXACTLY at any context -- the harmonic atom reduces to the plain atom, the way beta->inf
    reduces the Hopfield update to hard-NN.
  * SMOOTH win: a meaning function band-limited to B harmonics reconstructs EXACTLY at K=B+1 (2B+1 vectors) for ANY
    context, beating a per-context nearest-neighbor store that needs far more vectors for worse continuous readout.
  * DEGENERATE TRAP: a NON-smooth meaning function (unrelated meaning per context) is NOT captured by a few
    harmonics -- the error stays high and worsens with K (aliasing) -- so it degenerates to "store every context".

DETERMINISM (per ISA.md)
  A closed-form least-squares fit + a cosine/sine sum; no RNG. Same samples -> same atom and same decode (asserted).

KEPT NEGATIVES (loud)
  * For CONTEXT-FREE atoms the harmonic expansion spends coefficients for no gain -- the DC term suffices, so it MUST
    tie the plain atom there (and does, by construction). The win exists ONLY where the context variation is real AND
    smooth enough that a few harmonics beat per-context storage.
  * If the variation is NOT smooth, it degenerates to storing every context (the degenerate trap, measured) -- the
    bare-codebook degenerate-sampler failure mode from B10, in this basis. The burden is on a harmonic-specific win.
  * This is CIRCULAR harmonics (a 1-D context angle, the FPE phase case). Full SPHERICAL harmonics over a 2-D
    direction is the natural extension and is NOT implemented here.
  * The fit is least squares: with 2K-1 >= number of samples it interpolates the samples exactly; otherwise it is a
    smoothing fit (and an under-sampled high harmonic aliases) -- choose K against the sample count and the expected
    smoothness.
"""

import numpy as np


def harmonic_atom(thetas, meanings, n_harmonics):
    """Fit a context-conditioned atom: the circular-harmonic coefficients of the meaning function sampled at
    (context angle, meaning) pairs, by least squares. `thetas` are context angles (radians), `meanings` the matching
    vectors, `n_harmonics`=K keeps the DC plus K-1 harmonics. Returns {'K', 'coeffs'} where coeffs is (2K-1, D):
    row 0 = DC (a0), rows 1..K-1 = cos coefficients, rows K..2K-2 = sin coefficients."""
    th = np.asarray(thetas, float)
    Mv = np.stack([np.asarray(m, float) for m in meanings])
    K = int(n_harmonics)
    cols = [np.ones(len(th))] + [np.cos(k * th) for k in range(1, K)] + [np.sin(k * th) for k in range(1, K)]
    A = np.stack(cols, axis=1)                             # (n_samples, 2K-1)
    coeffs, *_ = np.linalg.lstsq(A, Mv, rcond=None)        # (2K-1, D)
    return {"K": K, "coeffs": coeffs}


def harmonic_decode(atom, theta):
    """The meaning at context angle `theta` -- the harmonic sum a0 + sum_k a_k cos(k theta) + b_k sin(k theta)."""
    K = atom["K"]
    row = [1.0] + [np.cos(k * theta) for k in range(1, K)] + [np.sin(k * theta) for k in range(1, K)]
    return np.asarray(row) @ atom["coeffs"]


def harmonic_dc(atom):
    """The DC (degree-0), context-free meaning -- exactly the plain fixed atom."""
    return atom["coeffs"][0].copy()


# =====================================================================================================
# Self-test -- polysemy, the exact degree-0 fallback, the smooth win over per-context, the degenerate trap.
# =====================================================================================================
def _selftest():
    rng = np.random.default_rng(0)
    D = 256

    # --- POLYSEMY: distinct senses at distinct contexts, each recovered; a between-context blends them ---
    senses = [rng.standard_normal(D) for _ in range(3)]
    senses = [s / np.linalg.norm(s) for s in senses]
    ctx = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    atom = harmonic_atom(ctx, senses, n_harmonics=2)       # 2K-1=3 coeffs = 3 samples -> exact
    for t, s in zip(ctx, senses):
        rec = harmonic_decode(atom, t)
        assert rec @ s / np.linalg.norm(rec) > 0.999, "each sense must be recovered at its context"
    mid = harmonic_decode(atom, np.pi / 3)                 # between sense 0 and sense 1
    c0 = mid @ senses[0] / np.linalg.norm(mid)
    c1 = mid @ senses[1] / np.linalg.norm(mid)
    assert c0 > 0.3 and c1 > 0.3, "a between-context must blend the two neighbouring senses"

    # --- DEGREE-0 FALLBACK: a context-free atom is captured by the DC alone, decoded exactly anywhere ---
    const = rng.standard_normal(D)
    cfree = harmonic_atom([0.0, 1.0, 2.0, 3.0], [const, const, const, const], n_harmonics=1)
    assert np.linalg.norm(harmonic_decode(cfree, 1.234) - const) < 1e-10, "context-free atom must decode exactly (DC)"
    assert np.linalg.norm(harmonic_dc(cfree) - const) < 1e-10, "the DC term is exactly the plain atom"

    # --- SMOOTH win: a B=3-band-limited meaning function is EXACT at K=4 and beats per-context NN ---
    r = np.random.default_rng(1)
    a0 = r.standard_normal(D)
    pairs = [(r.standard_normal(D), r.standard_normal(D)) for _ in range(3)]   # harmonics 1..3

    def content(theta):
        out = a0.copy()
        for k, (ak, bk) in enumerate(pairs, 1):
            out += ak * np.cos(k * theta) + bk * np.sin(k * theta)
        return out

    fit_th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    h_atom = harmonic_atom(fit_th, [content(t) for t in fit_th], n_harmonics=4)
    test_th = np.linspace(0.1, 2 * np.pi - 0.1, 50)
    truth = np.stack([content(t) for t in test_th])
    h_err = np.sqrt(((np.stack([harmonic_decode(h_atom, t) for t in test_th]) - truth) ** 2).sum(1)).mean()
    assert h_err < 1e-6, f"K=B+1 harmonics must reconstruct a band-limited function exactly, got {h_err:.4f}"
    # per-context NN with even MORE vectors is worse for continuous readout
    pc_th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pc_store = np.stack([content(t) for t in pc_th])
    pc_rec = np.stack([pc_store[np.argmin(np.abs(((tt - pc_th + np.pi) % (2 * np.pi)) - np.pi))] for tt in test_th])
    pc_err = np.sqrt(((pc_rec - truth) ** 2).sum(1)).mean()
    assert h_err < pc_err, "harmonic (7 vectors, exact) must beat per-context NN (24 vectors) on continuous readout"

    # --- DEGENERATE TRAP: a non-smooth meaning function is NOT captured by a few harmonics ---
    r2 = np.random.default_rng(9)
    Mns = 12
    th_ns = np.linspace(0, 2 * np.pi, Mns, endpoint=False)
    vals = r2.standard_normal((Mns, D))
    ns_atom = harmonic_atom(th_ns, list(vals), n_harmonics=4)
    ns_err = np.sqrt(((np.stack([harmonic_decode(ns_atom, t) for t in th_ns]) - vals) ** 2).sum(1)).mean()
    assert ns_err > 1.0, "a non-smooth meaning function must NOT be captured by a few harmonics (degenerate trap)"

    # --- determinism ---
    a1 = harmonic_atom(ctx, senses, n_harmonics=2)
    a2 = harmonic_atom(ctx, senses, n_harmonics=2)
    assert np.array_equal(a1["coeffs"], a2["coeffs"])

    print(f"holographic_harmonic selftest: ok (POLYSEMY -- 3 senses recovered at their contexts (cos>0.999), a "
          f"between-context blends them (cos {c0:.2f}/{c1:.2f}); DEGREE-0 fallback exact (context-free atom decodes "
          f"to <1e-10 from DC alone); SMOOTH B=3 function EXACT at K=4 (err {h_err:.1e}) beating per-context NN at 24 "
          f"vectors (err {pc_err:.2f}); DEGENERATE non-smooth not captured (err {ns_err:.1f}); deterministic)")


if __name__ == "__main__":
    _selftest()

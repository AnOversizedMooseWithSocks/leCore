"""Denoising as manifold projection + Plug-and-Play restoration (B7)."""
import numpy as np
import pytest
from holographic.rendering.holographic_denoise import fit_manifold, manifold_denoise, pnp_restore, codebook_denoise, nlm_denoise


def _low_rank_signals(n=400, dim=32, rank=5, seed=0):
    rng = np.random.default_rng(seed)
    B = np.linalg.svd(rng.standard_normal((rank, dim)), full_matrices=False)[2]
    coeffs = rng.standard_normal((n, rank))
    return coeffs @ B                          # genuinely rank-`rank` signals


def _snr(clean, est):
    return 10 * np.log10(np.sum(clean ** 2) / (np.sum((clean - est) ** 2) + 1e-12))


def test_manifold_projection_denoises_high_noise_low_rank_signal():
    rng = np.random.default_rng(1)
    X = _low_rank_signals()
    basis, mean = fit_manifold(X[:300], rank=6)
    raw = proj = 0.0
    for x in X[300:380]:
        noisy = x + 0.7 * rng.standard_normal(x.shape[0])
        raw += _snr(x, noisy); proj += _snr(x, manifold_denoise(noisy, basis, mean))
    assert proj > raw                          # projection helps when noise dominates off-manifold


def test_manifold_projection_does_not_help_random_data():
    # honest control: no low-rank manifold -> projecting onto a spurious subspace cannot denoise.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 32))
    basis, mean = fit_manifold(X[:300], rank=6)
    raw = proj = 0.0
    for x in X[300:380]:
        noisy = x + 0.7 * rng.standard_normal(32)
        raw += _snr(x, noisy); proj += _snr(x, manifold_denoise(noisy, basis, mean))
    assert proj < raw + 0.5                     # no real gain (and typically a loss)


def test_pnp_restore_recovers_an_inpainting_problem():
    # use the manifold denoiser as the prior to fill erased entries (A = a binary mask, A^T == A).
    rng = np.random.default_rng(3)
    X = _low_rank_signals(dim=40, rank=5)
    basis, mean = fit_manifold(X[:300], rank=6)
    x = X[350]
    mask = (rng.random(40) > 0.4).astype(float)        # keep ~60% of entries
    A = lambda v: mask * v
    y = A(x)
    den = lambda v: manifold_denoise(v, basis, mean)
    rec = pnp_restore(y, A, A, den, mu=0.8, steps=60)
    assert _snr(x, rec) > _snr(x, y)                   # restoration beats the masked measurement


def _motif_signal(M, R, D=24, sigma=0.6, seed=0):
    rng = np.random.default_rng(seed)
    motifs = rng.standard_normal((M, D))
    motifs /= np.linalg.norm(motifs, axis=1, keepdims=True)
    clean = np.repeat(motifs, R, axis=0)
    return clean, clean + sigma * rng.standard_normal(clean.shape)


def test_nlm_beats_projection_on_self_similar_signal():
    # repeated motifs -> NLM averages the near-duplicates and cancels noise; projection cannot.
    clean, noisy = _motif_signal(M=20, R=8)
    basis, mean = fit_manifold(noisy, rank=8)
    proj = np.stack([manifold_denoise(x, basis, mean) for x in noisy])
    nlm = nlm_denoise(noisy, k=8, use_forest=True)
    s = lambda A: np.mean([_snr(clean[i], A[i]) for i in range(len(clean))])
    assert s(nlm) > s(proj) and s(nlm) > s(noisy) + 3.0


def test_projection_beats_nlm_without_self_similarity():
    # KEPT NEGATIVE / complementarity: low-rank but every patch unique -> NLM has no duplicates to
    # average, projection captures the subspace and wins. The two denoisers cover different worlds.
    rng = np.random.default_rng(3)
    X = _low_rank_signals(n=400, dim=32, rank=5)        # all-unique low-rank patches
    noisy = X + 0.6 * rng.standard_normal(X.shape)
    basis, mean = fit_manifold(noisy, rank=6)
    proj = np.stack([manifold_denoise(x, basis, mean) for x in noisy])
    nlm = nlm_denoise(noisy, k=8, use_forest=True)
    s = lambda A: np.mean([_snr(X[i], A[i]) for i in range(len(X))])
    assert s(proj) > s(nlm)


def test_forest_recall_k_finds_near_duplicates():
    # the recall step: a duplicated patch's k-nearest should be dominated by its other copies.
    from holographic.misc.holographic_tree import HoloForest
    clean, noisy = _motif_signal(M=10, R=8, sigma=0.2, seed=5)
    f = HoloForest(noisy.shape[1], n_trees=4, leaf_size=8, seed=0).build(noisy)
    idx, sims = f.recall_k(noisy[0], k=6)
    assert len(idx) >= 1 and sims[0] >= sims[-1]        # ranked descending
    # the closest neighbours should come from the same motif block (the first 8 rows)
    assert np.mean(idx[:4] < 8) >= 0.5


# ---- trajectory denoise: lone-1-D-signal prior, promoted out of the pipeline (above/below sweep) -----

def test_trajectory_denoise_cleans_a_lone_1d_signal():
    """trajectory_denoise gives a LONE 1-D signal the prior it lacks, from its own sliding windows (SSA):
    a smooth/periodic signal's Hankel matrix is low-rank, so the windows project onto their own subspace and
    the signal rebuilds by anti-diagonal averaging. On such a signal the error drops well below the noisy
    input. (No free lunch: the prior IS the signal's structure, so a structureless signal has nothing to
    recover -- the method can only shrink it, not restore a signal that was never there.)"""
    from holographic.rendering.holographic_denoise import trajectory_denoise
    t = np.linspace(0, 1, 256)
    clean = np.sin(2 * np.pi * 3 * t) + 0.5 * t
    noisy = clean + 0.4 * np.random.default_rng(0).standard_normal(256)
    den = trajectory_denoise(noisy)
    assert np.linalg.norm(den - clean) < 0.7 * np.linalg.norm(noisy - clean)


def test_trajectory_method_is_the_pipeline_denoiser_promoted():
    """The denoise faculty exposes the trajectory denoiser as method='trajectory', and the pipeline's private
    _denoise_signal is now a thin delegate to it -- one shared implementation, bit-identical."""
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0)
    t = np.linspace(0, 1, 200)
    sig = 1 + 2 * t + 3 * t ** 2 + 0.3 * np.random.default_rng(0).standard_normal(200)
    faculty = np.asarray(m.denoise(sig, method="trajectory"))
    private = np.asarray(m._denoise_signal(sig))
    assert np.array_equal(faculty, private)


def test_denoise_gate_routes_but_is_opt_in():
    """The projection-denoise re-enable ROUTES on the residual ratio: clearly-noise-dominated -> project, else
    fall back to the no-op. NOTE (kept negative): this gate is OPT-IN, not an auto-default -- a fixed threshold
    can't robustly separate off-manifold detail from noise (measured harm-leak at strong detail). It's safe only
    when the caller knows the signal is low-rank. Here we test the mechanical routing + the fallback identity."""
    import numpy as np
    from holographic.rendering.holographic_denoise import fit_manifold, manifold_denoise, denoise_gated
    rng = np.random.default_rng(0); D, rank = 128, 8
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]; base, det = Q[:, :rank], Q[:, rank:rank+24]
    sc = lambda n: (rng.standard_normal((n, rank)) @ base.T) + 0.35 * (rng.standard_normal((n, 24)) @ det.T)
    basis, mean = fit_manifold(sc(300), rank=rank)
    lo = sc(1)[0] + 0.05 * rng.standard_normal(D)                 # low noise -> fallback (identity)
    hi = sc(1)[0] + 0.8 * rng.standard_normal(D)                  # high noise -> project
    r_lo, i_lo = denoise_gated(lo, basis, mean)
    r_hi, i_hi = denoise_gated(hi, basis, mean)
    assert i_lo["used"] == "fallback" and np.array_equal(r_lo, np.asarray(lo, float))   # safe no-op
    assert i_hi["used"] == "superior" and np.array_equal(r_hi, manifold_denoise(hi, basis, mean))

# ============================================================================================
# X2 -- SOFT CONSTRAINTS (Catto's Soft Step) folded into the projection family.
# The claim under test: a constraint stated as (hertz, zeta) means the SAME PHYSICS at any
# substep count, where a hand-tuned per-sweep `omega` does not. Baseline = fixed omega.
# ============================================================================================

def _pull_to(target):
    """A single hard constraint: snap x onto `target`. The simplest projection there is."""
    t = np.asarray(target, float)
    return lambda x: t.copy()


def test_hard_limit_is_bit_identical_to_the_old_omega_path():
    # BACKWARD COMPATIBILITY, pinned numerically. stiffness=(inf, zeta) must reproduce the pre-existing
    # omega=1.0 path EXACTLY -- not 'to 1e-12'. Twelve callers depend on that default not moving.
    from holographic.rendering.holographic_denoise import project_onto_constraints, soft_relaxation
    assert soft_relaxation(np.inf, 1.0, 1 / 60.0) == 1.0
    x0 = np.array([0.0, 3.0, -2.0])
    projs = [_pull_to([1.0, 1.0, 1.0]), lambda x: x * 0.5]
    a, na, _ = project_onto_constraints(x0, projs, iters=7, tol=None)                       # old path
    b, nb, _ = project_onto_constraints(x0, projs, iters=7, tol=None,
                                        stiffness=(np.inf, 1.0), dt=1 / 60.0)               # new path
    assert np.array_equal(a, b) and na == nb        # bit-identical, not merely close


def test_stiffness_is_substep_invariant_where_omega_is_not():
    # THE BAR. Fixed horizon T, rising substep count N. (hertz, zeta) converges to the continuous
    # 1 - exp(-lambda*T) at FIRST order; a fixed omega converges to the wrong number entirely.
    from holographic.rendering.holographic_denoise import project_onto_constraints
    hz, ze, T = 1.0, 2.0, 0.2
    lam = 2.0 * np.pi * hz / (2.0 * ze)
    exact = 1.0 - np.exp(-lam * T)
    x0, proj = np.array([0.0]), _pull_to([1.0])

    errs = []
    for N in (16, 32, 64, 128):
        x, _, _ = project_onto_constraints(x0, [proj], iters=N, stiffness=(hz, ze), dt=T / N, tol=None)
        errs.append(abs(float(x[0]) - exact))
    assert errs[0] < 3e-3                       # already close at N=16
    for a, b in zip(errs, errs[1:]):
        assert 1.8 < a / b < 2.2                # error halves as N doubles => first order

    # the HONEST BASELINE, in the original space: the same omega at different N is different physics
    xs = [float(project_onto_constraints(x0, [proj], iters=N, omega=0.30, tol=None)[0][0])
          for N in (8, 64)]
    assert abs(xs[0] - xs[1]) > 0.05            # omega drifts with N ...
    assert abs(xs[1] - exact) > 0.5             # ... and lands nowhere near the physical answer


def test_kept_negative_the_mass_scale_form_vanishes_with_substeps():
    # Pinned so a future session does not "restore" Catto's velocity coefficients into a position solver.
    # alpha = dt*bias_rate*mass_scale is O(dt^2), so the TOTAL relaxation over a fixed horizon -> 0.
    def wrong(hz, ze, dt):
        w = 2.0 * np.pi * hz
        a1 = 2.0 * ze + dt * w
        a2 = dt * w * a1
        return (dt * w / a1) * (a2 / (1.0 + a2))
    assert wrong(5.0, 1.0, 1 / 256.0) * 256 < wrong(5.0, 1.0, 1 / 8.0) * 8

    # ... and the shipped coefficient does NOT vanish: N*alpha stays O(1) as N rises
    from holographic.rendering.holographic_denoise import soft_relaxation
    assert soft_relaxation(5.0, 1.0, 1 / 256.0) * 256 > 0.5 * soft_relaxation(5.0, 1.0, 1 / 8.0) * 8


def test_degenerate_dials_and_refusal_to_guess():
    from holographic.rendering.holographic_denoise import project_onto_constraints, soft_relaxation
    assert soft_relaxation(0.0, 1.0, 0.01) == 0.0      # zero stiffness: inert constraint
    assert soft_relaxation(5.0, 0.0, 0.01) == 1.0      # zeta=0 degenerates to HARD, never to ringing
    x0, proj = np.array([0.0]), _pull_to([1.0])
    with pytest.raises(ValueError):                    # two names for one dial -- refuse to pick
        project_onto_constraints(x0, [proj], omega=0.5, stiffness=(5.0, 1.0), dt=0.01)
    with pytest.raises(ValueError):                    # stiffness needs dt to become a factor
        project_onto_constraints(x0, [proj], stiffness=(5.0, 1.0))
    with pytest.raises(ValueError):                    # dt alone is meaningless
        project_onto_constraints(x0, [proj], dt=0.01)


def test_softness_is_monotone_in_hertz():
    # The dial must actually be a dial: stiffer => closer to the constraint after a fixed horizon.
    from holographic.rendering.holographic_denoise import project_onto_constraints
    x0, proj = np.array([0.0]), _pull_to([1.0])
    got = [float(project_onto_constraints(x0, [proj], iters=32, stiffness=(hz, 1.0), dt=1 / 240.0)[0][0])
           for hz in (0.5, 2.0, 8.0, 32.0)]
    assert all(a < b for a, b in zip(got, got[1:])), got
    assert got[0] < 0.2 and got[-1] > 0.9


def test_cross_faculty_one_dial_serves_geometry_and_hypervectors():
    # INTEGRATION, and the point of X2: the projection engine is shared, so ONE stiffness dial reaches two
    # faculties that never import each other -- a PBD-style geometric distance constraint, and a cleanup
    # projection on a hypervector. The hard-lesson on record is that a shared kernel is not a shared
    # manifold, so this asserts behaviour in BOTH spaces, not just that the call returns.
    import lecore
    m = lecore.UnifiedMind(dim=512, seed=0)

    # (a) GEOMETRY: hold two particles at rest length 1.0 -- the PBD distance constraint.
    def distance_constraint(p):
        p = p.reshape(2, 2).copy()
        d = p[1] - p[0]
        n = np.linalg.norm(d)
        if n > 1e-12:
            corr = 0.5 * (n - 1.0) * (d / n)
            p[0] += corr
            p[1] -= corr
        return p.reshape(-1)

    start = np.array([0.0, 0.0, 3.0, 0.0])                     # stretched to length 3
    soft, _, _ = m.project_onto_constraints(start, [distance_constraint], iters=20,
                                            stiffness=(2.0, 1.0), dt=1 / 240.0)
    hard, _, _ = m.project_onto_constraints(start, [distance_constraint], iters=20,
                                            stiffness=(np.inf, 1.0), dt=1 / 240.0)
    len_soft = np.linalg.norm(soft[2:] - soft[:2])
    len_hard = np.linalg.norm(hard[2:] - hard[:2])
    assert abs(len_hard - 1.0) < 1e-9                          # rigid: rest length met
    assert len_soft > len_hard + 0.5                           # soft: still stretched, as a spring should be

    # (b) HYPERVECTORS: the same dial on a cleanup projection toward a stored atom.
    atom = np.asarray(m.hypervector("anchor").array, float)
    noisy = atom + 0.9 * np.asarray(m.hypervector("some other symbol").array, float)
    to_atom = lambda v: atom.copy()
    v_soft, _, _ = m.project_onto_constraints(noisy, [to_atom], iters=20, stiffness=(2.0, 1.0), dt=1 / 240.0)
    v_hard, _, _ = m.project_onto_constraints(noisy, [to_atom], iters=20, stiffness=(np.inf, 1.0), dt=1 / 240.0)
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(v_hard, atom) > 0.999999                        # rigid: snapped onto the atom
    assert cos(noisy, atom) < cos(v_soft, atom) < cos(v_hard, atom)   # soft: moved toward it, not onto it


def test_soft_relaxation_is_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    assert m.soft_relaxation(np.inf, 1.0, 1 / 60.0) == 1.0
    w = 2.0 * np.pi * 5.0
    assert abs(m.soft_relaxation(5.0, 1.0, 0.01) - 0.01 * w / (2.0 + 0.01 * w)) < 1e-15
    assert "Soft constraints" in str(m.find_capability("how stiff should my constraint be")[:3])

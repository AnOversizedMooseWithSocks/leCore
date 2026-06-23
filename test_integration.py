"""Integration tests: the recent modules are FACULTIES of UnifiedMind, proven end to end.

The integration plan's hardest lesson (its section 6): naive cross-module chaining REGRESSED once --
a denoiser fed a recall output dropped cosine, because a shared *kernel* is not a shared *manifold*.
So wiring is not proven by an import check; it is proven by running a cross-faculty pipeline THROUGH
UnifiedMind and asserting it actually works, with each hop's prior matched to its input.

This file covers the Tier 1 faculties -- decompose_signal / denoise / fit_function -- the DECOMPOSE /
DENOISE / FIT half of the loop. Each tier of wiring lands with at least one pipeline test here.
"""
import os
import tempfile

import numpy as np
import pytest

from holographic_unified import UnifiedMind
from holographic_symbolic import Formula


def _rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---- the faculties are actually on the mind (not a separate drawer of experiments) --------------
def test_unified_exposes_the_tier1_faculties():
    m = UnifiedMind(dim=256, seed=0)
    for name in ("decompose_signal", "denoise", "fit_function"):
        assert callable(getattr(m, name)), f"UnifiedMind is missing faculty {name!r}"


# ---- decompose_signal: foreign signal -> a savable, realizable generating law -------------------
def test_decompose_signal_recovers_periodic_law_and_seed_roundtrips():
    m = UnifiedMind(dim=256, seed=0)
    x = np.linspace(0, 4 * np.pi, 240)
    y = np.sin(x) + 0.3 * np.cos(2 * x)                 # a ring (periodic) law

    f, info = m.decompose_signal(x, y)
    assert info["topology"] == "ring"
    assert info["mode"] == "additive"
    assert info["n_terms"] <= 4 and info["resid_rms"] < 1e-2     # recovered the law tightly
    assert info["compression_ratio"] > 1.0                       # the seed is smaller than the samples

    # the Formula IS a realizable seed: save -> load -> generate must be BIT-identical
    seedpath = os.path.join(tempfile.mkdtemp(), "seed.json")
    f.save(seedpath)
    assert np.max(np.abs(Formula.load(seedpath).generate(x) - f.generate(x))) < 1e-12

    # and it extrapolates PERIODICALLY (bounded) instead of diverging the way a polynomial would
    xe = np.linspace(4 * np.pi, 6 * np.pi, 120)
    assert np.max(np.abs(f.generate(xe))) < 3.0


def test_decompose_signal_auto_selects_multiplicative_for_a_power_law():
    m = UnifiedMind(dim=256, seed=0)
    x = np.linspace(1.0, 6.0, 150)
    y = 2.0 * x ** 1.5                                  # a multiplicative law on a LINE domain (y > 0)

    f, info = m.decompose_signal(x, y)
    assert info["topology"] == "line" and info["mode"] == "multiplicative"
    assert info["resid_rms"] < 1e-6                     # log transform turns the power law additive

    # single-array shorthand: decompose_signal(y) fits the signal on a unit index grid and still runs
    f2, info2 = m.decompose_signal(y)
    assert set(info2) >= {"topology", "mode", "resid_rms", "compression_ratio"}
    assert np.isfinite(f2.generate(np.arange(len(y), dtype=float))).all()


# ---- THE PIPELINE the plan asks for: decompose -> save -> realize -> denoise, through ONE mind ---
def test_end_to_end_decompose_save_realize_denoise_pipeline():
    """detect topology -> decompose_signal -> seed.save -> realize (reload + generate) -> denoise.
    The denoise hop runs on a manifold that CONTAINS the regenerated signal (compatible prior), so the
    chain is honest, and the end result must IMPROVE on the noisy input -- no silent regression."""
    m = UnifiedMind(dim=256, seed=0)
    x = np.linspace(0, 4 * np.pi, 240)
    clean = np.sin(x) + 0.4 * np.sin(3 * x)             # only ODD harmonics -> an antiperiodic (mobius) law

    # decompose into a law and persist the seed. (This signal is purely odd-harmonic, so the topology
    # detector correctly calls it mobius -- it decomposes onto the ODD-harmonic basis. Either periodic
    # class extrapolates on its manifold; the pipeline is what we are testing, not the class.)
    f, info = m.decompose_signal(x, clean)
    assert info["topology"] in ("ring", "mobius") and info["resid_rms"] < 1e-2
    seedpath = os.path.join(tempfile.mkdtemp(), "law.json")
    f.save(seedpath)

    # realize: reload the seed and regenerate the signal it encodes
    regenerated = Formula.load(seedpath).generate(x)
    assert _rms(regenerated, clean) < 1e-2              # the law reproduces the signal

    # denoise a noisy copy on the signal's OWN harmonic family (a manifold that contains it)
    fam = np.stack([np.sin(x + p) + 0.4 * np.sin(3 * (x + p))
                    for p in np.linspace(0, 2 * np.pi, 64)])
    rng = np.random.default_rng(0)
    noisy = regenerated + 0.8 * rng.standard_normal(len(x))
    den = m.denoise(noisy, method="adaptive", samples=fam)

    assert _rms(den, clean) < _rms(noisy, clean) - 0.2  # the pipeline END materially helps (no regress)


# ---- denoise: routing, the honest "needs a prior" refusal, and the codebook map -----------------
def test_denoise_routes_and_requires_a_prior():
    m = UnifiedMind(dim=256, seed=0)

    # auto with no prior is an honest refusal -- a denoiser is a map of a manifold; a lone vector
    # has none, so there is no free lunch to silently take
    with pytest.raises(ValueError):
        m.denoise(np.zeros(32))

    # codebook route: a noisy atom should move TOWARD its codebook entry
    rng = np.random.default_rng(0)
    cb = rng.standard_normal((8, 64))
    cb /= np.linalg.norm(cb, axis=1, keepdims=True)
    noisy_atom = cb[3] + 0.5 * rng.standard_normal(64)
    cleaned = m.denoise(noisy_atom, method="codebook", codebook=cb)
    assert _cos(cleaned, cb[3]) > _cos(noisy_atom, cb[3])


def test_denoise_helps_at_high_noise_on_real_low_rank_data():
    """Grounding on real SOL price windows -- the same low-rank manifold the denoise module measured.
    At high noise, projecting onto the manifold removes off-manifold noise (the measured win)."""
    px = np.load("data/sol_5min.npz")["px"].astype(float)
    W, step = 64, 16
    wins = np.stack([px[i:i + W] for i in range(0, len(px) - W, step)])
    wins = (wins - wins.mean(1, keepdims=True)) / (wins.std(1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0); rng.shuffle(wins)
    train, test = wins[:600], wins[600:750]

    m = UnifiedMind(dim=256, seed=0)
    snr = lambda c, e: 10 * np.log10(np.var(c) / (np.mean((c - e) ** 2) + 1e-12))
    rng2 = np.random.default_rng(1)
    gains = []
    for c in test:
        n = c + 0.8 * rng2.standard_normal(len(c))
        d = m.denoise(n, method="adaptive", samples=train)
        gains.append(snr(c, d) - snr(c, n))
    assert np.mean(gains) > 1.5                          # solid denoising at high noise, on real data


# ---- fit_function: interpretable additive fit, with its interaction boundary kept on record ------
def test_fit_function_recovers_additive_parts_and_shows_interaction_limit():
    m = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1400, 2))
    g1 = lambda t: np.sin(2 * np.pi * t)
    g2 = lambda t: 4 * (t - 0.5) ** 2
    y = g1(X[:, 0]) + g2(X[:, 1]) + 0.02 * rng.standard_normal(1400)

    k = m.fit_function(X[:1000], y[:1000])
    r2 = 1 - np.sum((y[1000:] - k.predict(X[1000:])) ** 2) / np.sum((y[1000:] - y[1000:].mean()) ** 2)
    assert r2 > 0.98
    ts = np.linspace(0.05, 0.95, 40)
    assert abs(np.corrcoef(k.feature_function(0, ts), g1(ts))[0, 1]) > 0.95   # psi_1 recovers sin

    # KEPT NEGATIVE (surfaced, not hidden): the additive form cannot fit an interaction
    yprod = (2 * X[:, 0] - 1) * (2 * X[:, 1] - 1) + 0.02 * rng.standard_normal(1400)
    kp = m.fit_function(X[:1000], yprod[:1000])
    r2p = 1 - np.sum((yprod[1000:] - kp.predict(X[1000:])) ** 2) / np.sum(
        (yprod[1000:] - yprod[1000:].mean()) ** 2)
    assert r2p < 0.3                                     # the boundary, shown

    # 1-D feature shorthand: a lone feature vector is accepted as one column
    k1 = m.fit_function(X[:500, 0], y[:500])
    assert k1.predict(X[:5, 0].reshape(-1, 1)).shape == (5,)


# ============================ Tier 2 -- the factor_composite de-siloing ============================
# The integration plan's "real de-siloing": the higher-capacity SBC resonator gets a first-class mind
# faculty (decompose_structure), and factor_composite becomes ONE entry point that delegates to it for
# SBC problems while keeping -- and deprecating -- the legacy dense MAP path (a different algebra the SBC
# resonator cannot factor, so it is delegated-past, not deleted).

def test_decompose_structure_faculty_factors_and_verifies():
    from holographic_sbc import sbc_codebook, sbc_reconstruct
    m = UnifiedMind(dim=256, seed=0)
    B, L = 16, 16
    cbs = [sbc_codebook(B, L, 10, seed=k) for k in range(3)]
    true = (2, 5, 8)
    P = sbc_reconstruct(true, cbs, L)

    out = m.decompose_structure(P, cbs, L)
    assert tuple(out["picks"]) == true and out["verified"]
    assert np.array_equal(sbc_reconstruct(out["picks"], cbs, L), P)   # the recipe rebuilds the structure
    assert out["present"] == [True, True, True]                       # all three factors are present


def test_factor_composite_routes_to_sbc_and_matches_decompose_structure():
    from holographic_sbc import sbc_codebook, sbc_reconstruct, sbc_identity
    m = UnifiedMind(dim=256, seed=0)
    B, L = 16, 16
    cbs = [sbc_codebook(B, L, 10, seed=k) for k in range(3)]
    true = (2, 5, 8)
    P = sbc_reconstruct(true, cbs, L)

    # one entry point, given an L, takes the SBC path and agrees with the canonical faculty
    fc = m.factor_composite(P, cbs, L=L)
    assert fc["backend"] == "sbc"
    assert fc["solved"] and fc["factors"] == true
    assert fc["factors"] == tuple(m.decompose_structure(P, cbs, L)["picks"])

    # presence detection survives the routing: an identity factor is reported ABSENT
    cbs2 = [list(cbs[0]) + [sbc_identity(B)], cbs[1], cbs[2]]
    P_absent = sbc_reconstruct((len(cbs2[0]) - 1, 4, 6), cbs2, L)      # factor 0 = identity
    fa = m.factor_composite(P_absent, cbs2, L=L)
    assert fa["present"][0] is False and fa["present"][1] and fa["present"][2]


def test_factor_composite_dense_path_is_backward_compatible_and_deprecated():
    """The pinned legacy contract: a dense MAP/bipolar composite still factors through factor_composite
    (the SBC resonator cannot do this algebra, so the path is kept) -- and it now warns, steering new
    code to the SBC factorizer."""
    import warnings
    from holographic_resonator import map_codebook, map_bind
    m = UnifiedMind(dim=256, seed=0)
    books = [map_codebook(40, 1500, s) for s in range(3)]
    rng = np.random.default_rng(4)
    true = [int(rng.integers(40)) for _ in range(3)]
    c = map_bind(*[books[f][true[f]] for f in range(3)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = m.factor_composite(c, books, restarts=30)
        assert any(issubclass(x.category, DeprecationWarning) for x in caught)  # deprecation steers callers
    assert r["solved"] and r["factors"] == tuple(true) and r["backend"] == "dense"


# =================== Tier 2 (items 4 & 6) -- decode_structure (peel) + energy cleanup ===============
# Item 4: the B8 per-peel decode exposed as the inverse of the B7 chain typed structure, run through
# the mind (build -> realize -> decode). Item 6: the B1 dense-Hopfield cleanup wired as an opt-in flag
# on the core cleanup, pinned bit-for-bit to argmax at high beta.

def test_decode_structure_round_trips_a_chain_through_the_mind():
    m = UnifiedMind(dim=512, seed=1)
    recipe, nodes = m.chain_structure(16)
    M = m.realize(recipe)                                  # B7 forward: realize the chain memory
    n_correct = lambda seq: sum(1 for h, i in enumerate(seq) if i == h + 1)

    hard = m.decode_structure(M, nodes, cleanup="hard")
    soft = m.decode_structure(M, nodes, cleanup="soft")
    raw = m.decode_structure(M, nodes, cleanup=None)

    assert n_correct(hard) == 15                           # per-peel cleanup decodes every hop
    assert n_correct(soft) == 15                           # soft ties hard on discrete pointers (B1 negative)
    assert n_correct(raw) <= 3                              # KEPT NEGATIVE: raw decode craters (noise compounds)
    assert hard == list(range(1, 16))                      # the exact recovered sequence


def test_energy_cleanup_is_opt_in_and_matches_argmax_at_high_beta():
    from holographic_ai import Vocabulary, random_vector
    v = Vocabulary(512, seed=2)
    for nm in ("alpha", "beta", "gamma", "delta", "epsilon"):
        v.get(nm)
    rng = np.random.default_rng(0)
    noisy = v.get("gamma") + 0.7 * random_vector(512, rng)

    plain = v.cleanup(noisy)                               # default off -- the existing decision
    energy_hi = v.cleanup(noisy, energy=True, beta=1e6)    # beta->inf softmax is one-hot == argmax
    assert energy_hi[0] == plain[0]                        # bit-for-bit the same identity (B1 guarantee)

    # the flag is genuinely opt-in: a clean atom is recovered with or without it
    assert v.cleanup(v.get("delta"))[0] == "delta"
    assert v.cleanup(v.get("delta"), energy=True)[0] == "delta"


# ============================== Tier 3 -- search & dynamics faculties ===============================
# Min-cost search (a maze; a fragment assembly) and learned linear dynamics, as faculties of the mind.
# The assembly result comes back as a B7 typed structure the mind can realize; dynamics is a bind.

def test_solve_maze_faculty_finds_the_optimal_path_deterministically():
    from holographic_creature import GridWorld
    m = UnifiedMind(dim=256, seed=0)
    w = GridWorld(16, 16, maze=True, fixed_seed=7, braid=1.0)
    path, info = m.solve_maze(w)
    assert info["reached"] and info["deterministic"] is True
    assert info["extracted_len"] == info["optimal"]       # flow collapses onto the shortest tube
    # deterministic: a second solve gives the identical path
    path2, _ = m.solve_maze(GridWorld(16, 16, maze=True, fixed_seed=7, braid=1.0))
    assert path == path2


def test_assemble_faculty_is_optimal_and_a_realizable_typed_structure():
    from holographic_assembly import assemble_optimal_energy
    from holographic_typed import op_kinds
    m = UnifiedMind(dim=256, seed=0)
    target = "ABCABCABCA"
    full = sorted({target[p:p + 2] for p in range(len(target) - 1)})

    out = m.assemble(target, full)
    assert out["assembled"] == target and out["energy"] == 0

    # the result IS a B7 typed structure: only the typed op-kinds, and the mind can realize it
    assert op_kinds(out["recipe"]) <= {"atom", "bind", "bundle", "superpose", "permute", "raw", "normalize"}
    assert m.realize(out["recipe"]).shape == (256,)        # realizes to a hypervector at the mind's dim

    # forced to mismatch, the flow assembly still attains the GLOBAL optimum (matches the exact DP)
    lib = sorted((set(full) - {"CA"}) | {"AA", "BB", "CC"})
    o2 = m.assemble(target, lib)
    assert o2["energy"] == assemble_optimal_energy(target, lib, frag_len=2) and o2["energy"] > 0


def test_learn_dynamics_faculty_predicts_bind_shaped_and_round_trips():
    from holographic_ai import bind, cosine, random_vector
    rng = np.random.default_rng(0)
    U = random_vector(256, rng)
    s = random_vector(256, rng)
    traj = [s]
    for _ in range(400):
        s = bind(U, s) + 0.01 * rng.standard_normal(256)
        s /= np.linalg.norm(s)
        traj.append(s)
    traj = np.array(traj)

    m = UnifiedMind(dim=256, seed=0)
    prop = m.learn_dynamics(traj[:300])

    # step is LITERALLY a bind with the learned operator, and it predicts bind-shaped dynamics
    assert np.allclose(prop.step(traj[310]), bind(prop.U, traj[310]))
    pred = np.mean([cosine(prop.step(traj[300 + i]), traj[301 + i]) for i in range(80)])
    persist = np.mean([cosine(traj[300 + i], traj[301 + i]) for i in range(80)])
    assert pred > 0.9 and pred > persist + 0.2

    # the durable win: the trajectory is content-addressable -- forward k then back k returns the start
    x = traj[350]
    assert cosine(x, prop.recall_at(prop.rollout(x, 4)[-1], 4)) > 0.99


# =================== Tier 4 -- persistence (rd) + the generative faculties ==========================
# Item 9: the mind is persistable via the kernel save (so quant='rd' applies), round-tripping its
# LEARNED generalization (classify + decide identical). Item 10: vector generation as denoising-from-
# noise (B10), and a 2-D field as a superposition of Gaussian splats (B8).

def test_unified_mind_save_load_round_trips_classify_and_decide():
    import os
    import tempfile
    m = UnifiedMind(dim=256, seed=0, maintain="manual")
    rng = np.random.default_rng(0)
    for _ in range(20):                                    # two clean numeric classes
        m.learn(round(float(rng.uniform(0, 1)), 3), "small", modality="number")
        m.learn(round(float(rng.uniform(5, 6)), 3), "big", modality="number")
    m.actions(["left", "right"])                           # and a decision brain
    for _ in range(30):
        s = round(float(rng.uniform(0, 1)), 3)
        m.reinforce(s, "left" if s < 0.5 else "right", 1.0, modality="number")

    probes = [round(float(rng.uniform(0, 6)), 3) for _ in range(15)]
    before_cls = [m.classify(p, modality="number")[0] for p in probes]
    before_dec = [m.decide(p, modality="number") for p in probes]

    path = os.path.join(tempfile.mkdtemp(), "mind")
    m.save(path, quant="rd")                               # the B5 rate-distortion save level, via the mind
    m2 = UnifiedMind.load(path)

    assert [m2.classify(p, modality="number")[0] for p in probes] == before_cls  # classify identical
    assert [m2.decide(p, modality="number") for p in probes] == before_dec       # decide identical

    # documented boundary: the verbatim recall index of individuals is NOT persisted (re-learn for it)
    import pytest
    with pytest.raises(RuntimeError):
        m2.recall(0.5, modality="number")


def test_generate_vector_faculty_lands_on_the_manifold():
    from holographic_ai import random_vector, cosine
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    codebook = np.stack([random_vector(256, rng) for _ in range(8)])
    g = m.generate_vector(codebook, seed=3)
    # denoising from pure noise walks onto a stored pattern (B10); over a bare codebook that is a
    # stored atom -- the kept-negative degenerate regime, which is exactly what we assert here
    assert max(cosine(g, codebook[i]) for i in range(8)) > 0.99
    # deterministic in the seed
    assert np.allclose(g, m.generate_vector(codebook, seed=3))


def test_splat_field_faculty_reconstructs_and_denoises():
    from holographic_splat import psnr
    G = 48
    ys, xs = np.mgrid[0:G, 0:G]
    rng = np.random.default_rng(0)
    T = np.zeros((G, G))
    for _ in range(4):
        cy, cx, s, a = rng.uniform(8, G - 8, 2).tolist() + [rng.uniform(3, 7), rng.uniform(0.5, 1)]
        T += a * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * s * s))
    T /= T.max()

    splats, rendered = UnifiedMind(dim=128, seed=0).splat_field(T, k=40)
    assert len(splats) == 40 and psnr(T, rendered) > 25.0   # superposition of primitives reconstructs

    noisy = T + 0.10 * rng.standard_normal(T.shape)
    clean = UnifiedMind(dim=128, seed=0).splat_field(noisy, k=30, denoise=True)
    assert psnr(T, clean) > psnr(T, noisy) + 1.0            # the splat fit denoises (no capacity for noise)


# ============== Wiring-check follow-ups: axial perception + the splat-bundle archive ===============
# Boundary 1: holographic_mobius's AxialEncoder wired as the "axial" modality (theta == theta+pi).
# Boundary 2: a splat-bundle image archive beside the WHT plates, + the addendum's splat_bundle/recall_region.

def test_axial_modality_treats_theta_and_theta_plus_pi_as_the_same():
    import math
    m = UnifiedMind(dim=512, seed=0)
    t = 0.7
    assert m.axial_similarity(t, t + math.pi) > 0.99          # a pi flip is the SAME orientation
    assert m.axial_similarity(t, t + math.pi / 2) < 0.5       # an orthogonal orientation is dissimilar
    # decode is mod pi: theta and theta+pi both read back as the same angle in [0, pi)
    assert abs(m.decode_axial(m.perceive(1.2, "axial")) - 1.2) < 0.05
    assert abs(m.decode_axial(m.perceive(1.2 + math.pi, "axial")) - 1.2) < 0.05

    # learn / classify over orientations: an A reported as a pi flip still classifies as A
    rng = np.random.default_rng(0)
    for _ in range(15):
        m.learn(float(rng.uniform(0.0, 0.4)), "A", modality="axial")     # cluster near 0.2
        m.learn(float(rng.uniform(1.2, 1.6)), "B", modality="axial")     # cluster near 1.4
    assert m.classify(0.2 + math.pi, modality="axial")[0] == "A"         # flip-invariant classification


def test_splat_archive_reconstructs_refines_and_region_queries():
    from holographic_archive import _gallery
    from holographic_splat import psnr
    imgs = _gallery(48)
    K = 120
    m = UnifiedMind(dim=128, seed=0)
    arch = m.splat_archive((48, 48, 3), keep=K)
    for im in imgs:
        arch.add(im)

    # reconstruction is reasonable, and a PREFIX is a coarser-but-valid preview (progressive refinement)
    full = np.mean([psnr(imgs[i], arch.recover(i)) for i in range(arch.n)])
    quarter = np.mean([psnr(imgs[i], arch.recover(i, k=K // 4)) for i in range(arch.n)])
    assert full > 15.0 and full > quarter + 1.0               # quality rises with k (importance order)

    # EXACT region query: every returned splat's centre lies inside the requested box
    box = (0, 24, 0, 24)
    here, _patch = arch.region(0, box)
    for chan in here:
        assert all(0 <= cy < 24 and 0 <= cx < 24 for (cy, cx, _a, _s) in chan)

    # content recall on a noisy copy lands on the right image
    noisy = imgs[3] + 0.05 * np.random.default_rng(0).standard_normal(imgs[3].shape)
    assert arch.recall(noisy)[0] == 3


def test_splat_bundle_is_a_superposition_carrying_region_signal():
    from holographic_splat import splat_fit, splat_bundle, recall_region
    # a single blob in the TOP-LEFT, empty bottom-right
    G = 48
    ys, xs = np.mgrid[0:G, 0:G]
    T = np.exp(-((ys - 8) ** 2 + (xs - 8) ** 2) / (2 * 5.0 ** 2))
    T = T / T.max()
    splats = splat_fit(T, 10)

    scene, ctx = splat_bundle(splats, T.shape, dim=4096, grid=4)
    assert scene.shape == (4096,)
    # content-addressable region read: the OCCUPIED region recalls more energy than an EMPTY one.
    # (Only a coarse signal -- exact per-splat recovery is the archive's job; this is the VSA cliff.)
    occupied = recall_region(scene, (0, 0), ctx)              # top-left: has the blob
    empty = recall_region(scene, (3, 3), ctx)                 # bottom-right: empty
    assert occupied > empty

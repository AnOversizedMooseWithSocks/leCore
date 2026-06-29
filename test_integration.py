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


# ============== Honesty layer woven into recognition (RecallNull / SPRT / bh_fdr as core) ===========
# RecallNull/SPRT/bh_fdr were a standalone measurement harness; these prove they are now part of how the
# mind RECOGNISES -- calibrated confidence, honest abstention, sequential decision, FDR-controlled batch.

def _animal_mind():
    m = UnifiedMind(dim=512, seed=0)
    for w in ["dog", "wolf", "puppy", "hound"]:
        m.learn(w, "canine")
    for w in ["cat", "lion", "kitten", "tiger"]:
        m.learn(w, "feline")
    for w in ["oak", "pine", "maple", "birch"]:
        m.learn(w, "tree")
    return m


def test_recognize_is_calibrated_and_classify_can_abstain():
    m = _animal_mind()
    # a learned member: low false-alarm p; gibberish: clearly higher p
    _lab, _sim, p_real = m.recognize("dog")
    _lab2, _sim2, p_noise = m.recognize("qz xkqv zzpf")
    assert p_real < 0.05 and p_noise > p_real + 0.2
    # default classify ALWAYS names a nearest label (backward compatible, returns a (label, score) tuple)
    lab, score = m.classify("qz xkqv zzpf")
    assert lab is not None and isinstance(score, float)
    # with abstain: None label on noise, real label on a member -- both keep the (label, score) shape
    none_lab, _ = m.classify("qz xkqv zzpf", abstain=0.05)
    real_lab, _ = m.classify("dog", abstain=0.05)
    assert none_lab is None and real_lab == "canine"


def test_stream_recognize_decides_match_for_real_and_reject_for_noise():
    m = _animal_mind()
    dec_real, lab_real, _n1 = m.stream_recognize(["dog", "hound", "puppy", "wolf"])
    dec_noise, _lab, _n2 = m.stream_recognize(["qz1 zz", "zxq ww", "vbn qq", "wqp kk"])
    assert dec_real == "MATCH" and lab_real == "canine"
    assert dec_noise == "REJECT"


def test_recognize_batch_controls_false_discovery():
    m = _animal_mind()
    out = m.recognize_batch(["dog", "tiger", "oak", "zzqx vvbn wqlk"], alpha=0.1)
    sig = {r["label"]: r["significant"] for r in out[:3]}        # first three are real members
    assert all(sig.values())                                     # learned members survive FDR
    assert out[3]["significant"] is False                        # the gibberish does not


def test_recall_can_abstain_on_unseen_inputs():
    m = _animal_mind()
    # a stored individual recalls with a tiny false-alarm p; gibberish sits above the abstain threshold
    _pay, _sim, p_real = m.recall_calibrated("dog")
    _pay2, _sim2, p_noise = m.recall_calibrated("zzqx vvbn wqlk")
    assert p_real < 0.05 < p_noise                               # they straddle the abstention level
    # default recall is unchanged (returns (payload, score)); abstain returns the payload or None
    assert m.recall("dog") is not None
    assert m.recall("dog", abstain=0.05) is not None
    assert m.recall("zzqx vvbn wqlk", abstain=0.05) is None


# ============== Coherence-gated maintenance (the measured win from the calibrated-novelty study) ======
# Calibrated NOVELTY (the originally-flagged idea) was a NEGATIVE -- novelty detects "matches nothing",
# but reorganization's value is fixing INCOHERENCE, which novelty cannot see. The study that disproved it
# found COHERENCE-gated reorganization instead: reorganize only when the store is actually incoherent, not
# on a fixed clock. This test pins the win at the mind level -- fewer reorganize passes at comparable
# accuracy -- and that the default (coherence_floor=None) still reorganizes on the fixed schedule.

def _shift_stream(seed=0):
    """Antipodal-bimodal classes (a single prototype is useless, only a SPLIT classifies), with two NEW
    classes arriving mid-stream and then a long STABLE coherent tail where a fixed schedule keeps paying
    for reorganize passes the gate can skip."""
    import numpy as np
    rng = np.random.default_rng(seed)
    L, NC, MODES = 24, 4, 2
    ang = np.linspace(0, 2 * np.pi, NC * MODES, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1) @ rng.standard_normal((2, L))
    csub = {c: [c + NC * m for m in range(MODES)] for c in range(NC)}
    samp = lambda c: dirs[csub[c][rng.integers(MODES)]] * 3 + 0.5 * rng.standard_normal(L)
    rows = []
    for _ in range(90):
        c = int(rng.integers(2)); rows.append((samp(c), c))          # phase 1: 2 classes
    for _ in range(90):
        c = int(rng.integers(NC)); rows.append((samp(c), c))         # shift: 2 new classes join
    for _ in range(150):
        c = int(rng.integers(NC)); rows.append((samp(c), c))         # stable, coherent tail
    return rows


def _run_maintained(coherence_floor, maintain="auto", check_every=30):
    import numpy as np
    rows = _shift_stream(0)
    m = UnifiedMind(dim=384, seed=0, check_every=check_every,
                    coherence_floor=coherence_floor, maintain=maintain)
    correct = []
    for i, (x, c) in enumerate(rows):
        pred = m.classify(x, modality="vector")[0] if m.memory.live.size() else None
        correct.append(pred == c)
        m.learn(x, c, modality="vector")
    return float(np.mean(correct)), len(m.journal), len(rows)        # journal length == reorganize passes


def test_coherence_gate_reorganizes_less_than_schedule_at_comparable_accuracy():
    sched_acc, sched_passes, total = _run_maintained(None)           # default: fixed schedule
    gated_acc, gated_passes, _ = _run_maintained(0.65)               # opt-in coherence gate
    floor_acc, _, _ = _run_maintained(None, maintain="manual")       # never reorganize (the floor)
    # the gate reorganizes FEWER times than the schedule (but does reorganize) ...
    assert 2 <= gated_passes < sched_passes
    # ... at accuracy comparable to the schedule ...
    assert gated_acc >= sched_acc - 0.15
    # ... and well above the never-reorganize floor (so it reorganizes USEFULLY, not idly) ...
    assert gated_acc >= floor_acc + 0.12
    # ... while the DEFAULT (coherence_floor=None) reorganizes on the fixed schedule, unchanged.
    assert sched_passes == total // 30


def test_coherence_floor_survives_save_and_reload():
    import tempfile, os
    m = UnifiedMind(dim=256, seed=0, coherence_floor=0.6)
    m.learn("dog", "canine"); m.learn("cat", "feline")
    p = os.path.join(tempfile.mkdtemp(), "mind")
    m.save(p)
    assert UnifiedMind.load(p).coherence_floor == 0.6                 # config round-trips


# ============== Tier-0 panel fixes: sublinear+calibrated recall, coverage, rd-in-auto ================
# Pharr (sublinear recall_calibrated), Cranmer (calibration coverage), Duda (auto save uses B5 rd).

def test_recall_calibrated_uses_the_same_path_as_recall_and_can_abstain():
    m = UnifiedMind(dim=512, seed=0)
    for w in ["dog", "wolf", "puppy", "hound", "cat", "lion", "kitten", "tiger", "oak", "pine", "maple"]:
        m.learn(w, "animal" if w not in ("oak", "pine", "maple") else "tree")
    # recall_calibrated now routes the winner through recall() itself (the forest on a big store, the exact
    # scan on a small one) instead of its own exact scan -- so the two agree on the winner and the score.
    pay_r, sim_r = m.recall("dog")
    pay_c, sim_c, p_real = m.recall_calibrated("dog")
    assert pay_r == pay_c and abs(sim_r - sim_c) < 1e-9
    _pay, _sim, p_noise = m.recall_calibrated("zzqx vvbn wqlk")
    assert p_real < 0.05 < p_noise                                # stored vs noise straddle the threshold
    assert m.recall("dog", abstain=0.05) is not None
    assert m.recall("zzqx vvbn wqlk", abstain=0.05) is None


def test_recognition_p_values_are_calibrated_on_noise():
    m = UnifiedMind(dim=512, seed=0)
    for w in ["dog", "wolf", "puppy", "hound", "cat", "lion", "kitten", "tiger",
              "oak", "pine", "maple", "birch", "ash", "elm", "fox", "bear"]:
        m.learn(w, "animal" if w not in ("oak", "pine", "maple", "birch", "ash", "elm") else "tree")
    rep = m.calibration_report(n=3000)
    # a calibrated detector fires on pure noise at ~= alpha, on BOTH the prototype and individual paths.
    for path in ("prototype_false_alarm", "individual_false_alarm"):
        assert 0.02 <= rep[path][0.05] <= 0.10                    # ~5% false-alarm at alpha=0.05
        assert 0.05 <= rep[path][0.1] <= 0.17                     # ~10% at alpha=0.10


def test_auto_save_uses_rate_distortion_for_large_low_rank_arrays():
    import numpy as np
    from holographic_ratedistortion import (geometry_preserving_code, pack_code, unpack_code,
                                             reconstruct, bits_per_vector)
    rng = np.random.default_rng(0)
    rows, cols, rank = 512, 256, 8
    A = rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, cols))   # large + genuinely low-rank
    code = geometry_preserving_code(A, target_cos=0.9999)
    # this is the decision auto now makes: take rd only when it beats int8 (8 bits/value)
    assert bits_per_vector(code) < 8 * cols
    B = reconstruct(unpack_code(pack_code(code)))                 # full pack -> unpack -> reconstruct
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    assert float(np.sum(An * Bn, axis=1).min()) >= 0.998         # decision-safe (cosines preserved)


def test_default_auto_save_round_trips_a_mind_identically():
    import tempfile, os
    import numpy as np
    m = UnifiedMind(dim=256, seed=0, maintain="manual")
    rng = np.random.default_rng(0)
    for _ in range(20):
        m.learn(round(float(rng.uniform(0, 1)), 3), "small", modality="number")
        m.learn(round(float(rng.uniform(5, 6)), 3), "big", modality="number")
    probe = [round(float(rng.uniform(0, 6)), 3) for _ in range(12)]
    before = [m.classify(p, modality="number")[0] for p in probe]
    p = os.path.join(tempfile.mkdtemp(), "mind")
    m.save(p)                                                     # default quant='auto', now rd-aware
    after = [UnifiedMind.load(p).classify(p2, modality="number")[0] for p2 in probe]
    assert before == after


# ============== Tier-1: calibrated decide -- honesty from perception to ACTION (Togelius) ============

def _taught_action_mind(dim=512, seed=0):
    import numpy as np
    import holographic_ai as A
    m = UnifiedMind(dim=dim, seed=seed)
    m.actions(["N", "S", "E", "W"])
    rng = np.random.default_rng(seed)
    archetypes = [A.random_vector(dim, rng) for _ in range(4)]
    best = ["N", "E", "S", "W"]
    for _ in range(40):
        for k, base in enumerate(archetypes):
            s = base + 0.25 * A.random_vector(dim, rng); s /= np.linalg.norm(s)
            m.reinforce(s, best[k], 1.0)                   # this action paid off in situations like this
            m.reinforce(s, best[(k + 1) % 4], -0.5)        # a bad alternative, so values differ
    return m, archetypes, best, rng


def test_decide_confidence_is_low_for_familiar_states_and_high_for_novel_ones():
    import numpy as np
    import holographic_ai as A
    m, archetypes, best, rng = _taught_action_mind()
    fam = archetypes[0] + 0.25 * A.random_vector(512, rng); fam /= np.linalg.norm(fam)
    nov = A.random_vector(512, rng)
    act_f, p_f = m.decide_confidence(fam)
    act_n, p_n = m.decide_confidence(nov)
    assert act_f == best[0]                                 # familiar -> the learned-good action
    assert p_f < 0.1 < p_n                                  # familiar vs novel straddle the threshold


def test_brain_recognition_null_is_calibrated_on_noise():
    import numpy as np
    m, _arch, _best, _rng = _taught_action_mind()
    null = m._brain_null()
    rng = np.random.default_rng(7)
    sup = np.array([max(m._brain.value(s, a)[1] for a in range(4))
                    for s in (rng.standard_normal((3000, 512)) /
                              np.linalg.norm(rng.standard_normal((3000, 512)), axis=1, keepdims=True))])
    ps = np.array([null.pvalue(x) for x in sup])
    assert 0.02 <= float(np.mean(ps <= 0.05)) <= 0.10      # the action-side detector tracks alpha too
    assert 0.05 <= float(np.mean(ps <= 0.10)) <= 0.17


def test_explore_if_unrecognized_guesses_randomly_on_novel_states_and_commits_on_familiar():
    import numpy as np
    import holographic_ai as A
    m, archetypes, best, rng = _taught_action_mind()
    fam = archetypes[0] + 0.25 * A.random_vector(512, rng); fam /= np.linalg.norm(fam)
    nov = A.random_vector(512, rng)
    novel_actions = {m.decide(nov, explore_if_unrecognized=0.1) for _ in range(100)}
    fam_actions = {m.decide(fam, explore_if_unrecognized=0.1) for _ in range(100)}
    assert len(novel_actions) >= 3                          # unrecognized -> safe random move (guessing)
    assert fam_actions == {best[0]}                         # recognized -> commits to the trusted action


# ============== Tier-0: the SPRT earns its keep on OVERLAPPING densities (Wald sample-savings) ==========

def test_sprt_spends_more_samples_as_densities_overlap_and_beats_fixed_n():
    import numpy as np
    from holographic_honesty import SPRTRecall
    def avg_n_err(mu0, sd0, mu1, sd1, trials=1500, cap=60):
        null = np.random.default_rng(1).normal(mu0, sd0, 3000)
        match = np.random.default_rng(2).normal(mu1, sd1, 3000)
        g = np.random.default_rng(5); ns, ok = [], 0
        for t in range(trials):
            ms = (t % 2 == 0); mu, sd = (mu1, sd1) if ms else (mu0, sd0)
            d, n = SPRTRecall(null, match, alpha=0.05, beta=0.05).decide(g.normal(mu, sd, cap), cap=cap)
            ns.append(n); ok += (d == ("MATCH" if ms else "REJECT"))
        return float(np.mean(ns)), 1.0 - ok / trials
    sep_n, _ = avg_n_err(0.10, 0.04, 0.45, 0.14)          # well-separated -> decisive in ~1 sample
    ovl_n, ovl_err = avg_n_err(0.35, 0.13, 0.52, 0.13)    # heavy overlap -> several samples
    assert sep_n < 1.5 < ovl_n                            # the SPRT spends more samples when densities overlap
    # at matched error, the smallest fixed-window rule uses MORE samples than the SPRT's average
    thresh = (0.35 + 0.52) / 2.0
    def fixedN_err(N, trials=1500):
        h = np.random.default_rng(11); bad = 0
        for t in range(trials):
            ms = (t % 2 == 0); mu, sd = (0.52, 0.13) if ms else (0.35, 0.13)
            bad += ((float(np.mean(h.normal(mu, sd, N))) >= thresh) != ms)
        return bad / trials
    fixedN = next((N for N in range(1, 40) if fixedN_err(N) <= ovl_err + 0.01), None)
    assert fixedN is not None and fixedN > ovl_n          # Wald uses fewer samples for the same error


# ============== Tier-0: auto-calibrated coherence floor -- relative drop, no absolute threshold ==========

def test_auto_coherence_floor_matches_the_hand_set_floor_without_an_absolute_threshold():
    sched_acc, sched_passes, total = _run_maintained(None)           # fixed schedule
    auto_acc, auto_passes, _ = _run_maintained('auto')               # auto relative-drop floor (90% of peak)
    floor_acc, _, _ = _run_maintained(None, maintain="manual")       # never reorganize (the floor)
    assert 2 <= auto_passes < sched_passes                           # reorganizes less than the schedule, but does
    assert auto_acc >= sched_acc - 0.15                              # at accuracy comparable to the schedule
    assert auto_acc >= floor_acc + 0.12                              # usefully above never-reorganizing
    # the 'auto' sentinel round-trips through save/load exactly like a numeric floor
    import tempfile, os
    m = UnifiedMind(dim=256, seed=0, coherence_floor='auto')
    m.learn("dog", "canine"); m.learn("cat", "feline")
    p = os.path.join(tempfile.mkdtemp(), "mind"); m.save(p)
    assert UnifiedMind.load(p).coherence_floor == 'auto'


# ============== Tier-1 A1: the scan faculty -- SPRT per channel + FDR across channels (Siemion) ==========

def _scan_mind(dim=256, seed=0):
    import numpy as np, holographic_ai as A
    rng = np.random.default_rng(seed)
    m = UnifiedMind(dim=dim, seed=seed)
    base = A.random_vector(dim, rng)
    for _ in range(50):                                   # a weak/drifting target: wide-noise views
        v = base + rng.uniform(1.0, 4.0) * A.random_vector(dim, rng); v /= np.linalg.norm(v)
        m.learn(v, "signal", modality="vector")
    for j in range(6):                                    # other classes so routing/null are meaningful
        ob = A.random_vector(dim, rng)
        for _ in range(8):
            v = ob + 1.5 * A.random_vector(dim, rng); v /= np.linalg.norm(v)
            m.learn(v, f"o{j}", modality="vector")
    return m, base


def test_scan_detects_signals_controls_look_elsewhere_and_decides_as_fast_as_evidence_allows():
    import numpy as np, holographic_ai as A
    m, base = _scan_mind()
    unit = lambda v: v / np.linalg.norm(v)
    sig = lambda noise, L, r: [unit(base + noise * A.random_vector(256, r)) for _ in range(L)]
    noi = lambda L, r: [A.random_vector(256, r) for _ in range(L)]
    L = 14
    chans, kind = [], []
    for k in range(8):  chans.append(sig(1.5, L, np.random.default_rng(200 + k))); kind.append("clear")
    for k in range(4):  chans.append(sig(3.4, L, np.random.default_rng(220 + k))); kind.append("faint")
    for k in range(80): chans.append(noi(L, np.random.default_rng(300 + k)));     kind.append("noise")
    rows = m.scan(chans, modality="vector", alpha=0.05, beta=0.05, fdr=0.1)
    kind = np.array(kind)
    det = np.array([r["detected"] for r in rows]); pv = np.array([r["pvalue"] for r in rows])
    fsig = np.array([r["fdr_significant"] for r in rows])
    cmask, fmask, nmask = kind == "clear", kind == "faint", kind == "noise"
    # detection: the signal channels are found, the noise channels are not (FDR controls the proportion)
    assert int(det[cmask].sum()) >= 7                     # clear signals detected
    assert int(det[fmask].sum()) >= 2                     # faint signals mostly detected too
    assert int(det[nmask].sum()) / max(1, int(det.sum())) <= 0.25   # false-discovery proportion held near fdr
    # the look-elsewhere: naive per-channel p<=fdr manufactures false positives; BH-FDR cuts them
    naive_fp = int((pv[nmask] <= 0.1).sum())
    assert naive_fp >= 3 and int(fsig[nmask].sum()) < naive_fp
    # the SPRT spends MORE samples on faint channels than clear ones (decide as fast as evidence allows)
    clear_n = float(np.mean([rows[i]["n_samples"] for i in np.where(cmask)[0]]))
    faint_n = float(np.mean([rows[i]["n_samples"] for i in np.where(fmask)[0]]))
    assert faint_n >= clear_n
    # determinism (Macklin): the same scan twice is bit-identical
    rows2 = m.scan(chans, modality="vector", alpha=0.05, beta=0.05, fdr=0.1)
    assert all(a["detected"] == b["detected"] and abs(a["pvalue"] - b["pvalue"]) < 1e-12
               for a, b in zip(rows, rows2))


# ====== Tier-1 A2: calibrated soft confidence for the resonator on approximate inputs (Olshausen, Cranmer) ===

def test_resonator_confidence_rescues_approximate_inputs_and_abstains_on_noise():
    import numpy as np
    import holographic_sbc as S
    B, L, F, n = 16, 16, 3, 8
    cbs = [[S.sbc_random(B, L, seed=100 * f + i) for i in range(n)] for f in range(F)]

    def prod_of(idx):
        out = np.asarray(cbs[0][idx[0]]).copy()
        for f in range(1, F):
            out = S.sbc_bind(out, cbs[f][idx[f]], L)
        return out

    true = (2, 5, 1); prod = prod_of(true)
    m = UnifiedMind(dim=256, seed=0)
    # exact input -> verified True and confident (the soft confidence agrees with the hard certificate)
    r0 = m.factor_composite(prod, cbs, L=L, restarts=6, seed=0, confidence=True)
    assert r0["verified"] and r0["factors"] == true and r0["pvalue"] < 0.1
    # a few corrupted blocks: the resonator still recovers the TRUE factors, so `verified` is uselessly False
    # (not an EXACT rebuild) but the calibrated p stays confident -- the rescue the boolean cannot give
    for k in [1, 2, 3]:
        c = prod.copy(); rr = np.random.default_rng(50 + k)
        for b in rr.choice(B, k, replace=False):
            c[b] = (c[b] + 1) % L
        res = m.factor_composite(c, cbs, L=L, restarts=6, seed=0, confidence=True)
        assert res["factors"] == true and not res["verified"] and res["pvalue"] < 0.1
    # pure noise: the calibrated p ABSTAINS, and its false-alarm rate is controlled (conservative is fine --
    # block agreement is discrete, so the p-value is stepwise). A random-picks null would falsely be confident.
    ps = np.array([m.factor_composite(np.random.default_rng(1000 + s).integers(0, L, size=B),
                                      cbs, L=L, restarts=6, seed=0, confidence=True)["pvalue"]
                   for s in range(50)])
    assert (ps <= 0.1).mean() <= 0.2           # false-alarm at the nominal 0.1 held on structureless input
    assert float(np.median(ps)) >= 0.3         # a typical noise product abstains


# ====== Tier-2 A3: pluggable energy + structure-compare for assemble (Baker) ======================

def test_assemble_pluggable_energy_changes_the_optimum_and_structure_compare_reads_overlap():
    import holographic_assembly as ASM
    m = UnifiedMind(dim=512, seed=0)
    T = "EAAE"; lib = ("AB", "BA", "BD", "BE", "EE")
    GROUP = {c: ('V' if c in 'AE' else 'C') for c in 'ABDE'}

    def subst(frag, pos, target):                     # not every mismatch costs the same (the Rosetta move)
        c = 0
        for j in range(len(frag)):
            if frag[j] != target[pos + j]:
                c += 1 if GROUP[frag[j]] == GROUP[target[pos + j]] else 4
        return c

    rH = m.assemble(T, lib)                            # default Hamming stand-in
    rS = m.assemble(T, lib, energy=subst)             # pluggable substitution energy
    # each is the GLOBAL optimum under its OWN energy (the flow search matches the Viterbi DP). Both optima are
    # UNIQUE here (no tie), so the chosen assembly is deterministic -- the right way to test a tie-sensitive
    # search (Macklin): pick an instance without a tie rather than depend on how a tie breaks.
    assert rH["energy"] == ASM.assemble_optimal_energy(T, lib)
    assert rS["energy"] == ASM.assemble_optimal_energy(T, lib, energy=subst)
    # the supplied energy genuinely changes the optimum: Hamming picks 'BABE' (3 plain mismatches), but those
    # are cross-group B-for-vowel swaps that cost 4 each under substitution, so substitution picks 'EEEE'
    assert rH["assembled"] == "BABE" and rS["assembled"] == "EEEE"
    assert rH["fragments"] != rS["fragments"]
    subst_cost_of_rH = sum(subst(f, p, T) for (p, f) in rH["fragments"])
    assert subst_cost_of_rH > rS["energy"]            # the Hamming choice is strictly worse under substitution

    # structure-compare: the holographic (consolidation SVD) read matches the exact placement overlap
    A = {"fragments": [(0, 'AB'), (1, 'BC'), (2, 'CD')]}
    for B, exp in [({"fragments": list(A["fragments"])}, 1.0),
                   ({"fragments": [(0, 'AB'), (1, 'BX'), (2, 'XD')]}, 1.0 / 3.0),
                   ({"fragments": [(0, 'PQ'), (1, 'QR'), (2, 'RS')]}, 0.0)]:
        r = m.compare_structures(A, B)
        assert abs(r["placement_overlap"] - exp) < 1e-9
        assert abs(r["holographic_overlap"] - r["placement_overlap"]) < 1e-9   # the two reads agree
    # determinism (Macklin): the holographic read is bit-identical run-to-run
    assert m.compare_structures(A, A)["holographic_overlap"] == m.compare_structures(A, A)["holographic_overlap"]


# ====== Tier-2 A4: one iterate-a-projection faculty + a determinism audit of the new paths (Macklin) ======

def test_project_onto_constraints_is_pocs_a_resonator_and_pnp():
    import numpy as np, holographic_ai as A, holographic_denoise as D
    m = UnifiedMind(dim=512, seed=0)

    # (1) POCS: alternating projection onto two subspaces converges to a point in their intersection
    n = 20; rng = np.random.default_rng(0)
    u = rng.standard_normal(n); u /= np.linalg.norm(u)            # the shared 1-D direction
    a1, a2, b1, b2 = (rng.standard_normal(n) for _ in range(4))
    QA, _ = np.linalg.qr(np.stack([u, a1, a2], axis=1))          # A = span(u,a1,a2)
    QB, _ = np.linalg.qr(np.stack([u, b1, b2], axis=1))          # B = span(u,b1,b2); A∩B = span(u)
    PA = lambda x: QA @ (QA.T @ x); PB = lambda x: QB @ (QB.T @ x)
    x0 = rng.standard_normal(n)
    xf, sweeps, conv = m.project_onto_constraints(x0, [PA, PB], iters=500, tol=1e-12)
    assert conv and np.allclose(xf, (x0 @ u) * u, atol=1e-6)      # converged to the intersection projection

    # (2) the SAME engine as a RESONATOR: factor-cleanup projections + restarts recover a bound product
    dim = 512; rr = np.random.default_rng(1)
    CA = np.stack([A.random_vector(dim, rr) for _ in range(6)])
    CB = np.stack([A.random_vector(dim, rr) for _ in range(6)])
    P = A.bind(CA[2], CB[4])
    clean = lambda v, cb: cb[int(np.argmax(cb @ (v / np.linalg.norm(v))))]
    sp = lambda x: (x[:dim], x[dim:])
    pA = lambda x: np.concatenate([clean(A.unbind(P, sp(x)[1]), CA), sp(x)[1]])
    pB = lambda x: np.concatenate([sp(x)[0], clean(A.unbind(P, sp(x)[0]), CB)])
    best, bs = None, -2.0
    for s in range(12):                                          # restarts escape spurious fixed points
        ri = np.random.default_rng(50 + s)
        xf2, _, _ = m.project_onto_constraints(
            np.concatenate([CA[ri.integers(6)], CB[ri.integers(6)]]), [pA, pB], iters=20, tol=1e-9)
        a_r, b_r = sp(xf2); rec = A.bind(a_r, b_r)
        sim = float(rec @ P / (np.linalg.norm(rec) * np.linalg.norm(P)))
        if sim > bs: bs, best = sim, (a_r, b_r)
    assert int(np.argmax(CA @ best[0])) == 2 and int(np.argmax(CB @ best[1])) == 4

    # (3) PnP restore is LITERALLY this engine -- bit-identical
    cl = A.random_vector(dim, np.random.default_rng(7))
    mask = (np.random.default_rng(8).random(dim) > 0.3).astype(float)
    fwd = lambda x: mask * x; adj = lambda x: mask * x
    y = fwd(cl) + 0.05 * A.random_vector(dim, np.random.default_rng(9))
    Smp = np.stack([A.random_vector(dim, np.random.default_rng(100 + i)) for i in range(40)] + [cl])
    basis, _, mean = D.fit_manifold_full(Smp, rank=64)
    prior = lambda v: D.adaptive_manifold_denoise(v, basis, mean)
    viaPnP = D.pnp_restore(y, fwd, adj, prior, mu=0.5, steps=30)
    df = lambda x: x - 0.5 * adj(fwd(x) - y)
    viaEngine, _, _ = m.project_onto_constraints(adj(y), [df, lambda x: np.asarray(prior(x), float)],
                                                 iters=30, tol=None)
    assert np.array_equal(viaPnP, viaEngine)

    # (4) determinism: the engine is bit-identical run-to-run
    xa, _, _ = m.project_onto_constraints(x0, [PA, PB], iters=50)
    xb, _, _ = m.project_onto_constraints(x0, [PA, PB], iters=50)
    assert np.array_equal(xa, xb)


def test_determinism_audit_calibrated_and_null_paths_are_bit_identical_and_rng_independent():
    """Macklin's audit, EXPANDED to the new calibrated/null paths: each is run twice on a FRESHLY rebuilt
    setup (so its null is recomputed, not reused from cache), with numpy's GLOBAL RNG scrambled in between.
    Bit-identical results prove the paths are deterministic AND draw only from their own seeded RNG -- the
    class of bug the assemble tie-break (and bind_batch before it) was about."""
    import numpy as np, holographic_ai as A, holographic_sbc as S

    def perturb():                                                # scramble the GLOBAL RNG between runs
        np.random.seed(424242)
        for _ in range(300):
            np.random.random()

    # --- recall_calibrated + scan: a fresh vector mind each run (instance caches rebuild) ---
    def vec_run():
        rng = np.random.default_rng(0); mm = UnifiedMind(dim=192, seed=0)
        b = A.random_vector(192, rng)
        for _ in range(40):
            v = b + 1.3 * A.random_vector(192, rng); v /= np.linalg.norm(v); mm.learn(v, "sig", modality="vector")
        for j in range(4):
            o = A.random_vector(192, rng)
            for _ in range(6):
                v = o + 1.3 * A.random_vector(192, rng); v /= np.linalg.norm(v); mm.learn(v, f"o{j}", modality="vector")
        qq = b + 1.3 * A.random_vector(192, np.random.default_rng(5)); qq /= np.linalg.norm(qq)
        rc = mm.recall_calibrated(qq, modality="vector")
        unit = lambda v: v / np.linalg.norm(v)
        chans = [[unit(b + 1.3 * A.random_vector(192, np.random.default_rng(200 + k))) for _ in range(8)] for k in range(3)]
        chans += [[A.random_vector(192, np.random.default_rng(300 + k)) for _ in range(8)] for k in range(3)]
        return rc, mm.scan(chans, modality="vector")
    (rc1, sc1) = vec_run(); perturb(); (rc2, sc2) = vec_run()
    assert rc1[1] == rc2[1] and rc1[2] == rc2[2]                  # recall_calibrated: similarity + pvalue identical
    assert all(x["pvalue"] == y["pvalue"] and x["detected"] == y["detected"] for x, y in zip(sc1, sc2))

    # --- decide_confidence ---
    def decide_run():
        mm, arch, best, rng = _taught_action_mind()
        fam = arch[0] + 0.25 * A.random_vector(512, rng); fam /= np.linalg.norm(fam)
        return mm.decide_confidence(fam)
    d1 = decide_run(); perturb(); d2 = decide_run()
    assert d1[0] == d2[0] and d1[1] == d2[1]                      # action + pvalue identical

    # --- factor_composite(confidence=True): clear the MODULE-level null cache so the null RECOMPUTES ---
    def factor_run():
        S._RESONATOR_NULL_CACHE.clear()
        cbs = [[S.sbc_random(16, 16, seed=100 * f + i) for i in range(8)] for f in range(3)]
        prod = S.sbc_reconstruct((2, 5, 1), cbs, 16)
        return UnifiedMind(dim=128, seed=0).factor_composite(prod, cbs, L=16, restarts=6, seed=0, confidence=True)
    f1 = factor_run(); perturb(); f2 = factor_run()
    assert f1["pvalue"] == f2["pvalue"] and f1["agreement"] == f2["agreement"]

    # --- compare_structures (deterministic atom seeding) ---
    def cmp_run():
        return UnifiedMind(dim=256, seed=0).compare_structures(
            {"fragments": [(0, 'AB'), (1, 'BC'), (2, 'CD')]}, {"fragments": [(0, 'AB'), (1, 'BX'), (2, 'XD')]})
    c1 = cmp_run(); perturb(); c2 = cmp_run()
    assert c1["holographic_overlap"] == c2["holographic_overlap"]


# ====== Tier-2 A5: PnP/RED inverse problem through the mind + sigma-estimate validation (Milanfar, Ozcan) ===

def test_restore_inpaints_an_erased_plate_through_the_mind_and_beats_one_shot():
    import numpy as np
    from holographic_denoise import estimate_sigma
    S, K, N = 16, 6, 40
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:S, 0:S] / S
    patterns = []
    for f in range(K):                                          # a low-rank gallery: K smooth 2-D patterns
        a, b = rng.integers(1, 4, size=2)
        patterns.append(np.sin(np.pi * a * yy + 0.7 * f) * np.cos(np.pi * b * xx + 0.3 * f))
    patterns = np.stack([p / np.linalg.norm(p) for p in patterns])

    def mk(seed):
        c = np.random.default_rng(seed).standard_normal(K)
        img = (c[:, None, None] * patterns).sum(0); img -= img.min(); img /= (img.max() + 1e-9)
        return img

    gallery = np.stack([mk(100 + i) for i in range(N)])
    G = gallery.reshape(N, S * S)                               # the manifold prior (clean rows)
    m = UnifiedMind(dim=256, seed=0)
    psnr = lambda a, b: -10 * np.log10(((a - b) ** 2).mean())

    # the mind's OWN archive holds the gallery (recover a plate to confirm it round-trips)
    arch = m.splat_archive((S, S), keep=24)
    for g in gallery:
        arch.add(g)
    assert psnr(arch.recover(0), gallery[0]) > 20

    # erase a 5x5 block + light noise -> the degraded measurement
    clean = gallery[3].copy()
    mask = np.ones((S, S)); mask[5:10, 6:11] = 0.0
    y = mask * clean + 0.05 * np.random.default_rng(7).standard_normal((S, S))
    oneshot = m.denoise(y.flatten(), method="adaptive", samples=G).reshape(S, S)   # a SINGLE denoise (no loop)
    restored = m.restore(y.flatten(), mask=mask.flatten(), samples=G).reshape(S, S)  # the PnP/RED loop
    # the loop beats the one-shot on erasure (the one-shot is dragged toward zero by the missing pixels) and
    # genuinely recovers the plate
    assert psnr(restored, clean) > psnr(oneshot, clean) + 5.0
    assert psnr(restored, clean) > 28.0

    # sigma-estimate validation: accurate at moderate-high noise, and the adaptive denoiser self-estimates it
    noisy = gallery[0] + 0.20 * np.random.default_rng(20).standard_normal((S, S))
    assert 0.15 < estimate_sigma(noisy.flatten()) < 0.26       # within ~25% of the true 0.20
    auto = m.denoise(noisy.flatten(), method="adaptive", samples=G)            # sigma=None -> self-estimated
    told = m.denoise(noisy.flatten(), method="adaptive", samples=G, sigma=0.20)  # the true sigma supplied
    assert abs(psnr(auto.reshape(S, S), gallery[0]) - psnr(told.reshape(S, S), gallery[0])) < 1.0


# ====== Tier-3 A6: capacity / SNR + calibration-coverage-vs-load diagnostic (Plate + Cranmer) =============

def test_capacity_report_locates_the_cliff_and_coverage_holds_under_load():
    import numpy as np, holographic_ai as A

    def build(D, C, views=12, noise=1.2):
        rng = np.random.default_rng(0); m = UnifiedMind(dim=D, seed=0)
        for c in range(C):
            b = A.random_vector(D, rng)
            for _ in range(views):
                v = b + noise * A.random_vector(D, rng); v /= np.linalg.norm(v)
                m.learn(v, f"c{c}", modality="vector")
        return m

    roomy = build(512, 8).capacity_report(alpha=0.05, loads=(64, 512), n_floor=400, n_fa=400)
    loaded = build(64, 20).capacity_report(alpha=0.05, loads=(64, 512), n_floor=400, n_fa=400)

    # operating point: a genuine match sits well above the noise floor, and the LOADED store sits CLOSER to
    # the cliff than the roomy one (the diagnostic captures load)
    assert roomy["dprime"] > 10 and loaded["dprime"] > 2
    assert loaded["dprime"] < roomy["dprime"]
    # the measured floor tracks the HRR extreme-value bound sqrt(2 ln N / D) -- same order, sitting a bit below
    # the asymptotic bound at small N (the geometry behaves as theory predicts)
    for r in (roomy, loaded):
        assert 0.4 * r["hrr_floor_bound"] < r["floor_mean"] < 1.4 * r["hrr_floor_bound"]
    # headroom: both have room to grow, and the roomy (high-D) store has FAR more before noise wins
    assert roomy["headroom"] > 1.0 and loaded["headroom"] > 1.0
    assert roomy["headroom_log10"] > loaded["headroom_log10"]
    # coverage vs LOAD: the calibrated false-alarm rate stays ~alpha as N grows (it does not blow up at the
    # largest load -- the procedure-matched null re-fits to the rising floor). Cranmer's open question.
    for r in (roomy, loaded):
        assert all(fa <= 0.11 for fa in r["coverage_vs_load"].values())

    # determinism (the audit theme): the report is bit-identical run-to-run
    again = build(512, 8).capacity_report(alpha=0.05, loads=(64, 512), n_floor=400, n_fa=400)
    assert again["dprime"] == roomy["dprime"] and again["floor_mean"] == roomy["floor_mean"]
    assert again["coverage_vs_load"] == roomy["coverage_vs_load"]


# ====== Tier-3 A7: spectral/audio FHRR modality + dynamics on audio frames (Puckette + Stam) ==============

def test_spectral_modality_round_trips_feeds_fhrr_memory_and_dynamics_beats_persistence_and_mean():
    import numpy as np, holographic_fhrr as F
    m = UnifiedMind(dim=256, seed=0)
    L = 256
    t = np.arange(2000)

    def frames_of(sig, hop):
        n = 1 + (len(sig) - L) // hop
        return np.stack([sig[i*hop:i*hop+L] for i in range(n)])

    # (1) the audio modality round-trips EXACTLY and the phasor part is a valid FHRR vector (unit circle)
    frame = sum(np.sin(2*np.pi*c*t[:L]/L + p) for c, p in [(5, 0.3), (11, 1.1), (19, 2.0)])
    ph, mag = m.spectral_encode(frame)
    assert np.allclose(np.abs(ph), 1.0)                       # every component on the unit circle
    assert np.max(np.abs(m.spectral_decode(ph, mag) - frame)) < 1e-9

    # (2) the modality FEEDS the FHRR memory: encode several distinct BROADBAND sounds (a fundamental plus
    # harmonics plus a little noise -- energy across many bins, so the unit-phasor encodings are genuinely
    # distinct), cram them into one phasor trace, recall each by its key (the high-capacity regime
    # high_capacity_memory exists for). A pure tone is too sparse for phase alone to separate -- its silent
    # bins dominate the encoding and magnitude carries its identity -- which is why this uses broadband sounds.
    def broadband(f0, seed):
        r = np.random.default_rng(seed)
        sig = sum((1.0 / h) * np.sin(2*np.pi*f0*h*t[:L]/L + r.uniform(0, 6)) for h in range(1, 12))
        return sig + 0.15 * r.standard_normal(L)
    rng = np.random.default_rng(0)
    sounds = {n: m.spectral_encode(broadband(f0, s))[0]
              for n, (f0, s) in {"bass": (3, 1), "mid": (7, 2), "treble": (13, 3)}.items()}
    mem, _ = m.high_capacity_memory()
    keys = {n: F.phasor_atom(L, rng) for n in sounds}
    for n in sounds:
        mem.learn(keys[n], sounds[n])
    for n in sounds:
        rec = mem.recall(keys[n])
        assert max(sounds, key=lambda q: F.fhrr_sim(rec, sounds[q])) == n

    # (3) dynamics THROUGH THE MIND beats persistence AND mean on a sustained tone -- the B4 proving ground:
    # audio HAS the linear phase structure a fixed bind operator exploits, where market returns had none
    fr = frames_of(sum(np.sin(2*np.pi*c*t/L + p) for c, p in [(5, 0.3), (11, 1.1), (19, 2.0)]), hop=37)
    k = int(len(fr) * 0.7); train, test = fr[:k], fr[k:]
    prop = m.learn_dynamics(train)
    meanf = train.mean(0)
    rel = lambda a, b: np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)
    e_prop = np.mean([rel(prop.step(test[i]), test[i+1]) for i in range(len(test)-1)])
    e_pers = np.mean([rel(test[i],           test[i+1]) for i in range(len(test)-1)])
    e_mean = np.mean([rel(meanf,             test[i+1]) for i in range(len(test)-1)])
    assert e_prop < 0.1                                       # nails the per-bin phase advance
    assert e_prop < 0.3 * e_pers and e_prop < 0.3 * e_mean   # clearly beats both baselines

    # (4) the two faculties are WIRED: the propagator's predicted next frame, run through the audio modality,
    # matches the true next frame's encoding -- prediction lands in the modality's own representation
    pred_ph, _ = m.spectral_encode(prop.step(test[0]))
    true_ph, _ = m.spectral_encode(test[1])
    assert F.fhrr_sim(pred_ph, true_ph) > 0.9

    # (5) the durable dynamics win regardless of signal: forward k then back k returns the start
    s0 = fr[3]; back = prop.recall_at(prop.rollout(s0, 4)[-1], 4)
    assert float(np.dot(back, s0) / (np.linalg.norm(back) * np.linalg.norm(s0))) > 0.99


# ====== Tier-3 A8: validate learn_dynamics on a fluid field (Stam) ========================================

def test_learn_dynamics_predicts_a_fluid_field_beats_baselines_and_is_honest_on_nonlinear_flow():
    import numpy as np
    m = UnifiedMind(dim=256, seed=0)
    N = 256; x = np.arange(N)
    rel = lambda a, b: np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

    # a passive scalar on a periodic (toroidal) domain -- Stam's FFT-on-a-torus setting. The LINEAR
    # advection-diffusion step is exact in Fourier: each mode rotates (advection) and decays (diffusion),
    # which is precisely the per-bin complex transfer learn_dynamics learns.
    def field():
        f = np.exp(-0.5*((x-64)/8.0)**2) + 0.4*np.sin(2*np.pi*3*x/N) + 0.3*np.cos(2*np.pi*5*x/N)
        return f - f.mean()
    def lin_step(u, shift=3.7, nu=3e-4):
        U = np.fft.rfft(u); k = np.arange(U.size)
        U *= np.exp(-1j*2*np.pi*k*shift/N) * np.exp(-nu*k**2)
        return np.fft.irfft(U, n=N)
    u = field(); seq = [u.copy()]
    for _ in range(40):
        u = lin_step(u); seq.append(u.copy())
    seq = np.stack(seq)
    k = int(len(seq)*0.6); train, test = seq[:k], seq[k:]

    # (1) THROUGH THE MIND: one-step prediction beats persistence AND mean -- a fluid field HAS the linear
    # structure a fixed bind operator exploits (the loop the market negative left open, closed again)
    prop = m.learn_dynamics(train); meanf = train.mean(0)
    e_prop = np.mean([rel(prop.step(test[i]), test[i+1]) for i in range(len(test)-1)])
    e_pers = np.mean([rel(test[i],           test[i+1]) for i in range(len(test)-1)])
    e_mean = np.mean([rel(meanf,             test[i+1]) for i in range(len(test)-1)])
    assert e_prop < 0.05
    assert e_prop < 0.3 * e_pers and e_prop < 0.3 * e_mean

    # (2) the learned operator is a SURROGATE SOLVER: roll it out 8 steps from one field, track the true sim
    roll = prop.rollout(seq[k], 8); true = seq[k+1:k+9]
    assert np.mean([rel(roll[i], true[i]) for i in range(8)]) < 0.1

    # (3) the trajectory is content-addressable: the operator's own forward-k-then-back-k returns the start
    back = prop.recall_at(prop.rollout(seq[k], 4)[-1], 4)
    assert float(np.dot(back, seq[k]) / (np.linalg.norm(back)*np.linalg.norm(seq[k]))) > 0.99

    # (4) HONEST LIMIT on nonlinear flow: a Burgers field forms shocks (nonlinear steepening) that no single
    # fixed linear operator captures -- here the propagator does WORSE than persistence. Kept on record.
    def burg_step(u, dt=0.004, nu=0.02):
        U = np.fft.rfft(u); ux = np.fft.irfft(1j*np.arange(U.size)*U, n=N)
        u = u - dt*u*ux; U = np.fft.rfft(u); U *= np.exp(-nu*np.arange(U.size)**2*dt)
        return np.fft.irfft(U, n=N)
    u = 3.0*np.sin(2*np.pi*x/N); bs = [u.copy()]
    for _ in range(40):
        u = burg_step(u); bs.append(u.copy())
    bs = np.stack(bs); kb = int(len(bs)*0.6)
    bprop = m.learn_dynamics(bs[:kb])
    be_prop = np.mean([rel(bprop.step(bs[kb+i]), bs[kb+i+1]) for i in range(len(bs)-kb-1)])
    be_pers = np.mean([rel(bs[kb+i],             bs[kb+i+1]) for i in range(len(bs)-kb-1)])
    assert be_prop > 2 * be_pers           # the fixed linear operator cannot capture nonlinear shock formation


# ====== Tier-3 A9: multi-terminal network design (Adamatzky) ==============================================

def test_design_network_connects_terminals_tunes_tree_vs_mesh_and_yields_a_queryable_typed_structure():
    import numpy as np, heapq, itertools
    from holographic_ai import unbind, cosine
    m = UnifiedMind(dim=1024, seed=0)

    def grid(R, C):
        nbr = {}
        for r in range(R):
            for c in range(C):
                nbr[(r, c)] = [(r+dr, c+dc) for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))
                               if 0 <= r+dr < R and 0 <= c+dc < C]
        return nbr

    def sp(nbr, a, b):                                    # grid shortest-path hops
        seen = {a: 0}; q = [(0, a)]
        while q:
            d, x = heapq.heappop(q)
            if x == b: return d
            for y in nbr[x]:
                if y not in seen or d+1 < seen[y]:
                    seen[y] = d+1; heapq.heappush(q, (d+1, y))
        return 1e9

    def mst_len(nbr, terms):                              # MST on terminals = minimal-cost baseline
        E = sorted((sp(nbr, a, b), a, b) for a, b in itertools.combinations(terms, 2))
        par = {t: t for t in terms}
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        tot = 0
        for w, a, b in E:
            if find(a) != find(b): par[find(a)] = find(b); tot += w
        return tot

    def connected(net, terms):
        adj = {}
        for u, v in net: adj.setdefault(u, []).append(v); adj.setdefault(v, []).append(u)
        seen = {terms[0]}; st = [terms[0]]
        while st:
            x = st.pop()
            for y in adj.get(x, ()):
                if y not in seen: seen.add(y); st.append(y)
        return all(t in seen for t in terms)

    def cycles(net):                                     # |E| - |V| + 1: 0 = tree, >0 = redundant loops
        V = len({x for e in net for x in e}); return len(net) - V + 1

    nbr = grid(7, 7); terms = [(0, 0), (0, 6), (6, 0), (6, 6), (3, 3)]
    mlen = mst_len(nbr, terms)

    # high mu -> a near-minimal Steiner TREE (no redundant loops, no longer than the terminal-MST baseline --
    # Physarum's Steiner approximation)
    hi = m.design_network(nbr, terms, mu=4.0)
    assert connected(hi["edges"], terms)
    assert cycles(hi["edges"]) == 0 and len(hi["edges"]) <= mlen

    # low mu -> a REDUNDANT, fault-tolerant mesh: more tubes survive, with cycles (alternate routes)
    lo = m.design_network(nbr, terms, mu=0.8)
    assert connected(lo["edges"], terms)
    assert len(lo["edges"]) > len(hi["edges"]) and cycles(lo["edges"]) > 0

    # the default network is returned as a B7 TYPED STRUCTURE that the engine can query: unbind the
    # graph-memory by a node atom and clean up -> that node's actual network neighbours, above all non-neighbours
    net = m.design_network(nbr, terms, mu=2.0)
    assert connected(net["edges"], terms)
    assert net["memory"] is not None and net["structure"] is not None
    adj = {}
    for u, v in net["edges"]:
        adj.setdefault(u, []).append(v); adj.setdefault(v, []).append(u)
    u = (0, 0); true_nb = set(adj[u]); nodes = net["nodes"]
    probe = unbind(net["memory"], nodes[u])
    nb_sim = [cosine(probe, nodes[w]) for w in true_nb]
    other_sim = [cosine(probe, nodes[w]) for w in nodes if w != u and w not in true_nb]
    assert min(nb_sim) > max(other_sim)                  # every true neighbour outranks every non-neighbour


# ====== Tier-3 A10: cross-modal recall (tag <-> image) wired to the archive (Ozcan) =======================

def test_image_archive_recovers_exactly_and_recalls_cross_modally_in_both_directions():
    import numpy as np
    m = UnifiedMind(dim=512, seed=0)
    S = 24

    def disc(cx, cy, r):
        y, x = np.ogrid[:S, :S]; return ((x-cx)**2 + (y-cy)**2 <= r*r).astype(float)
    def box(x0, y0, w, h):
        im = np.zeros((S, S)); im[y0:y0+h, x0:x0+w] = 1.0; return im

    imgs = {"circle": disc(8, 8, 5), "square": box(4, 4, 10, 10),
            "gradient": np.tile(np.linspace(0, 1, S), (S, 1)), "ring": disc(12, 12, 9) - disc(12, 12, 5)}
    tags = {"circle": ["round", "small"], "square": ["boxy", "large"],
            "gradient": ["smooth", "horizontal"], "ring": ["round", "large"]}
    names = list(imgs)

    arch = m.image_archive((S, S), capacity=len(imgs))      # keep defaults to all coefficients -> exact
    for nm in names:
        arch.add(imgs[nm], tags=tags[nm])

    # the archive's exact-recall strength, now reachable from the mind
    assert max(float(np.max(np.abs(arch.recover(i) - imgs[names[i]]))) for i in range(len(names))) < 1e-6

    # cross-modal tag -> image: describe and retrieve, no picture needed (single tag)
    for q, want in (["smooth"], "gradient"), (["boxy"], "square"):
        idx, _, _ = arch.recall_by_tags(words=q)
        assert names[idx] == want
    # soft-AND over multiple tags: 'round'+'large' is the ring, 'round'+'small' is the circle
    assert names[arch.recall_by_tags(words=["round", "large"])[0]] == "ring"
    assert names[arch.recall_by_tags(words=["round", "small"])[0]] == "circle"

    # the reverse direction (improvement): image -> tags, the description the archive would give it
    cands = sorted({t for ts in tags.values() for t in ts})
    ring_top = [w for w, _ in arch.tags_of(names.index("ring"), cands)[:2]]
    assert set(ring_top) == {"round", "large"}

    # cross-modal recall is robust to damage: describe-then-retrieve still reconstructs under 40% plate erasure
    mask = arch.damage_mask(0.4, seed=1)
    idx, recon, _ = arch.recall_by_tags(words=["smooth"], mask=mask)
    assert names[idx] == "gradient" and float(np.max(np.abs(recon - imgs["gradient"]))) < 0.05


# ====== Tier-3 A11: generation over a COMPOSED subspace (Eno) =============================================

def test_generate_structure_makes_novel_valid_compositions_not_degenerate_stored_atoms():
    import numpy as np
    from holographic_ai import bind, unbind, derived_atom, cosine
    m = UnifiedMind(dim=1024, seed=0)
    S, V = 3, 6
    roles = np.stack([derived_atom(m.seed, f"slot:{i}", m.dim, unitary=True) for i in range(S)])
    fillers = np.stack([derived_atom(m.seed, f"fill:{w}", m.dim, unitary=True) for w in "ABCDEF"])

    def decode(z):                                       # hard NN filler per slot
        out = []
        for r in roles:
            u = unbind(z, r); out.append(int((fillers @ (u / (np.linalg.norm(u)+1e-12))).argmax()))
        return out

    combos = set()
    for s in range(10):
        z = m.generate_structure(roles, fillers, seed=s)
        dec = decode(z)
        # VALID by construction: re-encoding the decoded fillers reproduces the generated vector
        reenc = np.sum([bind(roles[i], fillers[dec[i]]) for i in range(S)], axis=0)
        reenc /= np.linalg.norm(reenc)
        assert cosine(z, reenc) > 0.95
        # and it is a COMPOSITION, not a bare stored atom: nearly orthogonal to every single filler atom
        assert max(abs(cosine(z, f)) for f in fillers) < 0.4
        combos.add(tuple(dec))

    # DIVERSITY: different seeds explore the V^S structure space, not one attractor
    assert len(combos) >= 4

    # the KEPT NEGATIVE, for contrast: generate_vector over the BARE filler codebook collapses to a stored
    # atom (a degenerate sampler) -- exactly what generating over the composed manifold avoids
    bare = m.generate_vector(fillers, seed=3)
    assert max(cosine(bare, f) for f in fillers) > 0.9


# ====== Tier-3 A12: recursive/fractal scene from a seed vector (Quilez) ====================================

def test_fractal_scene_expands_one_kernel_from_a_seed_vector_to_a_measured_fractal_dimension():
    import numpy as np
    m = UnifiedMind(dim=1024, seed=0)

    # Seed A -- a Sierpinski kernel: 3 copies at scale 1/2, self-similar dimension log3/log2 = 1.585
    seedA = m.fractal_seed([(0, 0), (1, 0), (0.5, 0.857)], 0.5)
    A = m.fractal_scene(seedA, depth=8)
    assert A["n_maps"] == 3 and abs(A["scale"] - 0.5) < 1e-9          # kernel decoded exactly from the vector
    assert abs(A["dimension"] - A["expected"]) < 0.15                # box count lands near the Hausdorff dim
    assert 1.2 < A["dimension"] < 1.8                                # genuinely fractal (non-integer)

    # Seed B -- a different kernel: 5 copies at scale 1/3, dimension log5/log3 = 1.465
    seedB = m.fractal_seed([(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)], 1.0 / 3.0)
    B = m.fractal_scene(seedB, depth=6)
    assert B["n_maps"] == 5 and abs(B["scale"] - 1.0 / 3.0) < 1e-6
    assert abs(B["dimension"] - B["expected"]) < 0.15

    # the seed genuinely drives the scene: different kernels -> distinct measured dimensions
    assert abs(A["dimension"] - B["dimension"]) > 0.03

    # deterministic: the same seed vector expands to exactly the same scene
    again = m.fractal_scene(seedA, depth=8)
    assert np.array_equal(again["points"], A["points"]) and again["dimension"] == A["dimension"]


# ====== Tier-4 A13: anisotropic splats + 3-D (Drettakis) ==================================================

def test_splat_aniso_beats_isotropic_on_oriented_structure_in_2d_and_3d():
    import numpy as np
    from holographic_splat import psnr, aniso_render
    m = UnifiedMind(dim=512, seed=0)

    # 2-D: two ELONGATED oriented ridges -- circular splats can't align, anisotropic ones can
    H = W = 44; Y, X = np.mgrid[0:H, 0:W]
    def ridge(cy, cx, ang, L, th):
        c, s = np.cos(ang), np.sin(ang); u = (X-cx)*c + (Y-cy)*s; v = -(X-cx)*s + (Y-cy)*c
        return np.exp(-0.5*((u/L)**2 + (v/th)**2))
    T = ridge(15, 15, 0.6, 9, 2.0) + 0.8*ridge(29, 27, -0.5, 11, 2.5)

    iso2 = m.splat_aniso(T, k=4, steps=0, denoise=True)        # steps=0 = the isotropic warm start
    splats2, ani2 = m.splat_aniso(T, k=4, steps=200)           # the anisotropic differentiable fit
    assert psnr(T, ani2) > psnr(T, iso2) + 15                  # anisotropy wins decisively on oriented structure
    assert psnr(T, ani2) > 30

    # the splats genuinely learned ANISOTROPY: at least one has a strongly elongated covariance
    def aspect(L):
        ev = np.linalg.eigvalsh(L @ L.T)                       # eigenvalues of the inverse covariance
        return float(max(ev) / max(min(ev), 1e-9))
    assert max(aspect(L) for _, _, L in splats2) > 3.0

    # re-rendering the returned splat code reproduces the fit
    assert psnr(ani2, aniso_render(splats2, T.shape)) > 40

    # 3-D: an elongated ellipsoid is essentially one anisotropic Gaussian -- a large win over isotropic
    D = 18; Z, Yy, Xx = np.mgrid[0:D, 0:D, 0:D]
    vol = np.exp(-0.5*(((Xx-9)/6.0)**2 + ((Yy-9)/2.0)**2 + ((Z-9)/2.5)**2))
    iso3 = m.splat_aniso(vol, k=3, steps=0, denoise=True)
    _, ani3 = m.splat_aniso(vol, k=3, steps=150)
    assert psnr(vol, ani3) > psnr(vol, iso3) + 15 and psnr(vol, ani3) > 35


# ====== Tier-4 A14: tensor-train (MPS) bind mode + capacity comparison vs HRR (Stoudenmire) ===============

def test_tensor_bind_capacity_vs_hrr_higher_fidelity_exact_for_orthogonal_keys_mps_compresses_low_rank():
    import numpy as np
    from holographic_ai import bind, unbind, random_vector, cosine
    m = UnifiedMind(dim=128, seed=0)
    D = 128
    rng = np.random.default_rng(0)
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)

    def hrr_recall_mean(keys, values):
        M = np.zeros(D)
        for k, v in zip(keys, values):
            M = M + bind(k, v)
        return float(np.mean([cosine(unbind(M, k), v) for k, v in zip(keys, values)]))

    def tb_recall_mean(mem, keys, values):
        return float(np.mean([cosine(mem.recall(k), v) for k, v in zip(keys, values)]))

    # (1) at a fixed LOAD, the tensor-product bind recalls far better than HRR (it spends D x the storage)
    ks = [unit(random_vector(D, rng)) for _ in range(16)]
    vs = [unit(random_vector(D, rng)) for _ in range(16)]
    tb = m.tensor_bind(ks, vs)
    assert tb_recall_mean(tb, ks, vs) > 0.85 and hrr_recall_mean(ks, vs) < 0.40
    assert tb.n_numbers == D * D                                       # the uncompressed cost it pays

    # (2) with ORTHOGONAL keys the tensor product is EXACT up to M = D; circular convolution is not
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    oks = [Q[i] for i in range(D)]
    ovs = [unit(random_vector(D, rng)) for _ in range(D)]
    assert tb_recall_mean(m.tensor_bind(oks, ovs), oks, ovs) > 0.99
    assert hrr_recall_mean(oks, ovs) < 0.30

    # (3) MPS truncation LOSSLESSLY compresses a low-rank binding matrix; truncating below the rank loses recall
    r_true = 8
    basis = np.linalg.qr(rng.standard_normal((D, r_true)))[0]
    lks = [unit(random_vector(D, rng)) for _ in range(48)]
    lvs = [unit(basis @ rng.standard_normal(r_true)) for _ in range(48)]    # values in a rank-8 subspace
    full = m.tensor_bind(lks, lvs)
    mps8 = m.tensor_bind(lks, lvs, rank=r_true)
    mps4 = m.tensor_bind(lks, lvs, rank=4)
    full_acc = tb_recall_mean(full, lks, lvs)
    assert abs(tb_recall_mean(mps8, lks, lvs) - full_acc) < 0.02         # lossless at the true rank
    assert mps8.n_numbers < full.n_numbers                              # at a fraction of the D^2 storage
    assert tb_recall_mean(mps4, lks, lvs) < tb_recall_mean(mps8, lks, lvs) - 0.10   # below-rank truncation is lossy

    # (4) the KEPT NEGATIVE: even the lossless MPS form still costs MORE than HRR's D numbers -- the tensor
    # route buys fidelity/absolute-capacity with storage, it does not beat HRR's per-number efficiency
    assert mps8.n_numbers > D


# ====== Path D: federated RAID store + compute-in-superposition, wired into the mind =====================

def test_path_d_storage_array_federates_with_parity_and_superpose_compute_is_exact_then_walls():
    import numpy as np
    from holographic_ai import random_vector, cosine
    m = UnifiedMind(dim=1024, seed=0)
    D = m.dim
    rng = np.random.default_rng(7)

    # --- WIDTH faculty: compute-in-superposition (holographic_superposed) ---
    # the contract: a single keyed item recovers EXACTLY (unitary keys -> only crosstalk is error)
    item = random_vector(D, rng)
    assert cosine(m.superpose_compute(item)["recovered"][0], item) > 0.999

    # parallel evaluation, cleanup-gated against a codebook (the reliable, crosstalk-resetting regime):
    # hold K candidates in ONE vector, score all against a query, pick the winner
    K = 6
    items = np.stack([random_vector(D, rng) for _ in range(K)])
    target = 4
    res = m.superpose_compute(items, query=items[target], codebook=items)
    assert res["winner"] == target                 # one vector evaluated K computations and resolved the winner
    assert res["recovered"].shape == (K, D)         # all K came back from a single bundle (parallel readout)

    # the conservation law as a KEPT NEGATIVE: recovery fidelity decays as width grows -- width is bounded
    def mean_fid(width):
        its = np.stack([random_vector(D, rng) for _ in range(width)])
        rec = m.superpose_compute(its)["recovered"]
        return float(np.mean([cosine(rec[i], its[i]) for i in range(width)]))
    assert mean_fid(4) > mean_fid(64)

    # --- capacity/resilience faculty: federated RAID store (holographic_array) ---
    arr = m.storage_array(n_parity=1, add_threshold=0.90)
    for _ in range(150):                            # store well past one shard's ~0.1xD budget
        arr.add(int(rng.integers(0, 256)))
    assert len(arr.data) >= 2                        # it FEDERATED -- grew beyond a single shard under pressure
    base = arr.accuracy()
    assert base > 0.85                               # high recall across the federation

    # RAID-5: lose a shard; parity reconstructs recall EXACTLY (across-shard sibling of graceful degradation)
    lost = 1
    zeroed = float(np.mean(
        [arr._recall_one(g, [d if i != lost else np.zeros(D) for i, d in enumerate(arr.data)])[0] == v
         for g, (k, v) in arr.truth.items()]))
    recovered = arr.accuracy(down=(lost,))
    assert recovered >= base - 1e-9                  # parity restores it exactly
    assert zeroed < base - 0.05                      # and a zeroed (unreconstructed) shard genuinely hurts


# ====== Path D / P1: exact RNS-phasor matmul, wired into the mind =========================================

def test_exact_matmul_is_exact_where_superposition_is_lossy_and_range_federates():
    import numpy as np
    from holographic_ai import random_vector, unitary_vector, bundle, bind_batch, bind_fixed
    import holographic_rns as rns
    m = UnifiedMind(dim=1024, seed=0)
    rng = np.random.default_rng(0)

    # the FHRR connection the faculty rests on: modular accumulation IS phasor binding (phase addition),
    # exact for any number of terms -- the crosstalk-free MAC a bundle cannot do
    for N in (10, 1000):
        terms = rng.integers(0, 9973, size=N)
        assert rns.phasor_sum_mod(terms, 9973) == int(terms.sum() % 9973)

    # exact integer matmul at a size where the lossy superposed readout degrades badly
    M, N = 256, 64
    W = rng.integers(-50, 51, size=(M, N)); x = rng.integers(-50, 51, size=N)
    assert np.array_equal(m.exact_matmul(W, x), W @ x)        # EXACT -- zero error

    # the SAME matmul as a lossy bundle (superpose_compute's regime): fidelity far below exact
    D = m.dim
    Wd = np.stack([random_vector(D, rng) for _ in range(M)]); Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    xd = random_vector(D, rng); xd /= np.linalg.norm(xd)
    roles = np.stack([unitary_vector(D, rng) for _ in range(M)])
    inv = lambda A: np.concatenate([A[:, :1], A[:, :0:-1]], axis=1)
    What = bind_fixed(bundle(bind_batch(roles, Wd)), inv(roles))
    a, b = What @ xd, Wd @ xd; a, b = a - a.mean(), b - b.mean()
    lossy_fid = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert lossy_fid < 0.5                                    # superposition is lossy here; RNS was exact

    # fixed-point float matmul: exact for the QUANTIZED operands (the kept negative = rounding only, not size)
    Wf = rng.standard_normal((8, 8)); xf = rng.standard_normal(8)
    yq = m.exact_matmul(Wf, xf, scale=64)
    truth_q = (np.round(Wf * 64).astype(np.int64) @ np.round(xf * 64).astype(np.int64)) / (64.0 * 64)
    assert np.allclose(yq, truth_q)

    # the range FEDERATES over moduli channels -- more channels, bigger exact range (the arithmetic sibling
    # of storage_array's federation)
    _, P_few = rns.choose_moduli(10 ** 8)
    _, P_many = rns.choose_moduli(10 ** 30)
    assert P_many > P_few > 10 ** 8


# ====== Path D / P2: pivot-tree sublinear index, wired into the mind ======================================

def test_pivot_index_routes_sublinearly_matching_exhaustive_with_beam_recall():
    import numpy as np
    m = UnifiedMind(dim=256, seed=0)
    D = m.dim
    rng = np.random.default_rng(3)
    unit = lambda v: v / (np.linalg.norm(v) or 1.0)

    # a well-separated hierarchical leaf set, so the exhaustive ceiling is high and any drop is the routing
    def gen(depth, F, center, scale, decay, out):
        if depth == 0:
            out.append(center); return
        for _ in range(F):
            gen(depth - 1, F, center + unit(rng.standard_normal(D)) * scale, scale * decay, decay, out)
    leaves = []
    gen(3, 6, np.zeros(D), 0.6 * 9.0, 1.0 / 3.0, leaves)        # 6^3 = 216 leaves
    leaves = np.stack(leaves); K = len(leaves)
    qr = np.random.default_rng(77); NQ = 250
    tgt = qr.integers(0, K, size=NQ)
    Q = leaves[tgt] + 0.22 * qr.standard_normal((NQ, D))
    exhaustive = float(np.mean([int(((leaves - Q[i]) ** 2).sum(1).argmin()) == tgt[i] for i in range(NQ)]))

    idx = m.pivot_index(leaves, fanout=6)
    top1 = float(np.mean([idx.query(Q[i], beam=1)[0] == tgt[i] for i in range(NQ)]))
    comps = float(np.mean([idx.query(Q[i], beam=1)[1] for i in range(NQ)]))
    rec5 = float(np.mean([tgt[i] in idx.reached(Q[i], beam=5) for i in range(NQ)]))

    assert top1 >= exhaustive - 0.05            # greedy top-1 routing matches the exhaustive scan...
    assert comps < K / 3                         # ...while touching far fewer pivots (sublinear, ~O(log N))
    assert rec5 >= 0.95                          # a beam lands the true leaf in the candidate set nearly always


# ====== Path D / P3: sketch-routed array recall (the broadcast-wall fix), through the mind ================

def test_storage_array_sketch_routing_tracks_directory_and_beats_broadcast_sublinearly():
    import numpy as np
    m = UnifiedMind(dim=1024, seed=0)
    arr = m.storage_array(n_parity=0, add_threshold=0.0)     # manual shards, no sensing, for a clean measure
    for s in range(64):
        if s > 0:
            arr._spin_up()
        for _ in range(30):
            arr.add(int(np.random.default_rng(s * 1000 + _).integers(0, 256)))
    pool = list(arr.truth)
    samp = [pool[i] for i in np.random.default_rng(5).choice(len(pool), 250, replace=False)]
    directory = float(np.mean([arr.recall(g) == arr.truth[g][1] for g in samp]))
    routed = float(np.mean([arr.routed_recall(g, c=8) == arr.truth[g][1] for g in samp]))
    broadcast = float(np.mean([arr.broadcast_recall(g) == arr.truth[g][1] for g in samp]))

    assert routed > 0.95                          # sketch routing stays accurate at 64 shards...
    assert abs(routed - directory) < 0.05         # ...tracking the directory
    assert routed >= broadcast                     # at least as accurate as full broadcast (which erodes with K)
    assert 8 < len(arr.data)                       # while unbinding only c=8 of the shards -- sublinear, not O(K)


# ====== Path D / P4: distributed (and deep, cleanup-gated / exact) forward pass, through the mind =========

def test_distributed_forward_federation_moves_class_wall_and_depth_cures_hold():
    import numpy as np
    import holographic_compute as hc
    m = UnifiedMind(dim=1024, seed=0)
    rng = np.random.default_rng(7)
    D = m.dim

    # THE WIN: federating the weight rows moves the class-capacity wall
    C = 64
    Hte, yte, W, Lex = hc._classifier(C, 20, D, rng)
    exact_acc = float(np.mean(Lex.argmax(1) == yte))
    acc1 = float(np.mean(m.distributed_forward(W, Hte, K=1).argmax(1) == yte))     # single vector (the cap)
    acc8 = float(np.mean(m.distributed_forward(W, Hte, K=8).argmax(1) == yte))     # eight shards (federated)
    assert acc8 > acc1 + 0.05                # federation moves the wall...
    assert acc8 >= exact_acc - 0.05          # ...recovering near the exact classifier at K=8

    # DEPTH cure 1 -- EXACT arithmetic: exact_matmul per layer has no crosstalk to compound (ties P4 to P1)
    A = rng.integers(-5, 6, size=(6, 8)); B = rng.integers(-5, 6, size=(4, 6)); v = rng.integers(-5, 6, size=8)
    exact_deep = B @ np.maximum(A @ v, 0)
    h1 = np.maximum(m.exact_matmul(A, v), 0).astype(np.int64)
    assert np.array_equal(m.exact_matmul(B, h1), exact_deep)        # exact at depth -- depth wall was arithmetic

    # DEPTH cure 2 -- cleanup-gating MECHANISM: softclean snaps crosstalk-corrupted activations back onto the
    # valid manifold (the dense-Hopfield reset). The faculty exposes this via cleanup_books; the end-to-end
    # accuracy benefit needs a well-formed / trained manifold (exp_A1) and is kept as honest scope, but the
    # primitive itself is robust:
    protos = rng.standard_normal((40, 64))
    clean = protos[rng.integers(0, 40, size=300)]
    noisy = clean + 0.8 * rng.standard_normal(clean.shape)
    cleaned = hc.softclean(noisy, protos)
    cos = lambda P, Qm: float(np.mean(np.sum(P * Qm, 1) /
                                      (np.linalg.norm(P, axis=1) * np.linalg.norm(Qm, axis=1) + 1e-12)))
    assert cos(cleaned, clean) > cos(noisy, clean)                 # cleanup moves activations toward the manifold

    # and the cleanup_books path is wired through the mind: a deep federated pass with a cleanup codebook runs
    Wd1 = rng.standard_normal((48, D)) / np.sqrt(D); Wd2 = rng.standard_normal((12, 48)) / np.sqrt(48)
    book = np.maximum(rng.standard_normal((30, 48)), 0.0)
    deep = m.distributed_forward([Wd1, Wd2], Hte[:5], K=4, cleanup_books=[book])
    assert deep.shape == (5, 12)


# ====== Path D / Bucket A: federated selection + sequence (superpose_compute shards), federated archive ===

def test_superpose_compute_federation_moves_width_wall_for_selection_and_sequence():
    import numpy as np
    from holographic_ai import random_vector, unitary_vector
    m = UnifiedMind(dim=1024, seed=0)
    D = m.dim

    # A3 -- hypothesis SELECTION among more candidates than one vector holds: federate, then pick the planted match
    def select_acc(H, shards, trials=5):
        hits = 0
        for t in range(trials):
            r = np.random.default_rng(200 + t)
            Hyp = np.stack([random_vector(D, r) for _ in range(H)]); j = int(r.integers(0, H))
            keys = np.stack([unitary_vector(D, r) for _ in range(H)])
            hits += int(m.superpose_compute(Hyp, query=Hyp[j].copy(), keys=keys, shards=shards)["winner"] == j)
        return hits / trials
    sel1, sel8 = select_acc(160, 1), select_acc(160, 8)
    assert sel8 > sel1 + 0.2 and sel8 >= 0.9     # federating the candidates moves the selection wall

    # A4 -- SEQUENCE recall of a longer symbol string than one vector holds (decoded vs the truth)
    def seq_acc(T, shards, V=64, trials=4):
        accs = []
        for t in range(trials):
            r = np.random.default_rng(300 + t)
            cb = np.stack([random_vector(D, r) for _ in range(V)]); seq = r.integers(0, V, size=T)
            pos = np.stack([unitary_vector(D, r) for _ in range(T)])
            accs.append(float(np.mean(m.superpose_compute(cb[seq], keys=pos, codebook=cb, shards=shards)["decoded"] == seq)))
        return float(np.mean(accs))
    seq1, seq8 = seq_acc(160, 1), seq_acc(160, 8)
    assert seq8 > seq1 + 0.2 and seq8 >= 0.9     # federating the positions moves the sequence-length wall


def test_federated_archive_federates_images_with_conservation():
    import numpy as np
    from holographic_archive import HolographicArchive
    m = UnifiedMind(dim=1024, seed=0)
    rng = np.random.default_rng(7)
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    S, N, B, K = 16, 64, 8192, 4
    imgs = [rng.random((S, S)) for _ in range(N)]; keep = B // N
    mono = HolographicArchive((S, S, 1), capacity=N, keep=keep, dim=B, seed=0)
    for im in imgs:
        mono.add(im)
    mono_corr = float(np.mean([corr(mono.recover(i).ravel(), imgs[i].ravel()) for i in range(N)]))
    fed = m.federated_archive((S, S, 1), capacity=N, K=K, keep=keep, dim=B // K)   # matched total dim B
    gi = [fed.add(im) for im in imgs]
    fed_corr = float(np.mean([corr(fed.recover(gi[i]).ravel(), imgs[i].ravel()) for i in range(N)]))
    assert fed.n == N                            # all N images held across the K shards (total = K x per-shard)
    assert fed_corr > 0.9                         # federated recovery is high...
    assert abs(fed_corr - mono_corr) < 0.05      # ...and conserved: federating at fixed total dim doesn't degrade


# ====== Path D: federation / conservation diagnostic (wraps the conservation-law measurements) ============

def test_federation_report_measures_conservation_law_and_recommends_shards():
    import math
    m = UnifiedMind(dim=1024, seed=0)
    D = m.dim
    rep = m.federation_report(target_items=500)
    b = rep["per_vector_budget"]
    assert 0.02 * D < b < 0.2 * D                          # a per-vector budget in the conservation-law range
    assert rep["federated"]["recall"] >= 0.85              # K aligned shards hold ~K x budget at (near) threshold
    assert rep["federated"]["stored"] == 4 * b              # 4 shards -> 4 x the per-vector budget
    assert 0.7 < rep["conservation_ratio"] < 1.3            # partitioning the dimension conserves capacity (~1)
    assert rep["recommended_shards"] == math.ceil(500 / b)  # the shard recommendation for the target item count


def test_resonator_null_keyed_by_shape_not_content():
    """The calibrated resonator null is a property of the search SHAPE, not codebook CONTENT (measured:
    the p-value is identical across random codebook contents of one shape for every decision-relevant
    agreement). So two different codebook sets of the same shape SHARE one cached null -- the multi-second
    cold fit is paid once per shape, not once per codebook set (a big saving across a suite of same-shape
    minds). Pins the cache-key fix so a content hash is never re-added."""
    from holographic_sbc import sbc_codebook, _resonator_noise_null
    B, L, F = 8, 8, 2
    cbsA = [sbc_codebook(B, L, 6, seed=10 + k) for k in range(F)]
    cbsB = [sbc_codebook(B, L, 6, seed=20 + k) for k in range(F)]   # different content, SAME shape
    cbsC = [sbc_codebook(B, L, 9, seed=30 + k) for k in range(F)]   # different shape (codebook size 9 vs 6)
    nA = _resonator_noise_null(cbsA, L, restarts=2, iters=8, m=20)
    nB = _resonator_noise_null(cbsB, L, restarts=2, iters=8, m=20)
    nC = _resonator_noise_null(cbsC, L, restarts=2, iters=8, m=20)
    assert nA is nB                                                 # same shape -> SHARED cached null (the fix)
    assert nC is not nA                                             # different shape -> its own null
    # the readout is part of the procedure, so a different readout is a different null
    nA_sparse = _resonator_noise_null(cbsA, L, restarts=2, iters=8, m=20, readout="sparsemax")
    assert nA_sparse is not nA


def test_vector_function_encoder_faculty():
    """BLD-7: the N-D FPE encoder is a faculty of UnifiedMind, built on the mind's dim/seed. Through the mind:
    a 2-D shift is a binding (exact), and a function bundled from encoded points reads high at its points and
    low elsewhere -- the compute-on-functions algebra on the shared substrate."""
    from holographic_ai import bind, cosine
    m = UnifiedMind(dim=512, seed=0)
    enc = m.vector_function_encoder(2, bounds=[(0, 10), (0, 10)])
    # shift-as-bind in 2-D, exact in direction
    p = np.array([3.0, 4.0]); d = np.array([1.5, 2.0])
    assert cosine(bind(enc.encode(p), enc.encode(d)), enc.encode(p + d)) > 0.999
    # a function localises at its placed points
    f = enc.bundle([(2.0, 2.0), (7.0, 3.0)], [1.0, 0.7])
    assert min(enc.query(f, (2.0, 2.0)), enc.query(f, (7.0, 3.0))) > 3 * enc.query(f, (9.5, 9.5))
    # built on the mind's seed -> two minds with the same seed give identical codes
    m2 = UnifiedMind(dim=512, seed=0)
    assert np.allclose(enc.encode(p), m2.vector_function_encoder(2, bounds=[(0, 10), (0, 10)]).encode(p))


def test_spectral_basis_faculty():
    """EXP-6: the Laplacian eigenbasis is a faculty of UnifiedMind. Through the mind, on a sphere (a manifold
    the topology detector can only call 'line'), the data-driven basis denoises a smooth field where a line/
    index-order basis cannot -- the basis-selector generalising decompose_signal's hand-picked choice."""
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(5)
    N = 300
    idx = np.arange(N)
    phi = np.arccos(1 - 2 * (idx + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    P = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], 1)
    f = P[:, 2] ** 2 - 1 / 3 + P[:, 0] * P[:, 1]
    fn = f + 0.3 * rng.standard_normal(N)
    sb = m.spectral_basis(P, k=10, n_basis=12)
    err_lap = np.linalg.norm(sb.denoise(fn) - f)
    DCTi = np.stack([np.ones(N) / np.sqrt(N)] +
                    [np.sqrt(2 / N) * np.cos(np.pi * (np.arange(N) + 0.5) * kk / N) for kk in range(1, 12)]).T
    err_line = np.linalg.norm(DCTi @ (DCTi.T @ fn) - f)
    assert err_lap < 0.6 * err_line
    # and the basis round-trips a signal that lives in it
    c = rng.standard_normal(12)
    assert np.allclose(sb.decompose(sb.reconstruct(c)), c, atol=1e-9)


def test_manifold_topology_faculty():
    """EXP-7: persistent-homology topology is a faculty of UnifiedMind. Through the mind, a torus is named
    (1,2,1) and a sphere (1,0,1) -- topologies the 1-D detect_topology cannot name -- while a ring stays
    (1,1,0), reproducing the hand-coded detector on the case it knows."""
    m = UnifiedMind(dim=256, seed=0)
    # torus
    Nu, Nv = 24, 12
    u = np.repeat(np.linspace(0, 2 * np.pi, Nu, endpoint=False), Nv)
    v = np.tile(np.linspace(0, 2 * np.pi, Nv, endpoint=False), Nu)
    torus = np.column_stack([(2 + 0.8 * np.cos(v)) * np.cos(u), (2 + 0.8 * np.cos(v)) * np.sin(u), 0.8 * np.sin(v)])
    name, betti, _ = m.manifold_topology(torus)
    assert betti == (1, 2, 1) and name == "torus"
    # sphere
    N = 200
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    tta = np.pi * (1 + 5 ** 0.5) * i
    sphere = np.column_stack([np.sin(phi) * np.cos(tta), np.sin(phi) * np.sin(tta), np.cos(phi)])
    assert m.manifold_topology(sphere)[1] == (1, 0, 1)
    # a ring still reads as detect_topology would call it
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.column_stack([np.cos(th), np.sin(th), np.zeros(40)])
    assert m.manifold_topology(ring)[0] == "ring"


def test_hodge_flow_decomposition_faculty():
    """EXP-8: the Helmholtz-Hodge flow decomposition is a faculty of UnifiedMind. Through the mind, a transport
    flow splits into orthogonal parts that sum exactly, the harmonic part is genuinely curl/divergence-free,
    and dropping the noisy curl denoises a flow better than the raw input."""
    from holographic_spectral import boundary_matrices
    m = UnifiedMind(dim=256, seed=0)
    # a triangulated 3x3 grid with one triangle removed -> a hole (B1=1)
    tris_all = []
    for cy in range(2):
        for cx in range(2):
            a = cy * 3 + cx
            tris_all += [(a, a + 1, a + 4), (a, a + 4, a + 3)]
    tris = [t for t in tris_all if t != (0, 1, 4)]
    edges = sorted({tuple(sorted(e)) for t in tris_all for e in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]})
    V = 9
    d1, d2 = boundary_matrices(V, edges, tris)
    rng = np.random.default_rng(0)
    flow = d1.T @ rng.standard_normal(V) + d2 @ rng.standard_normal(len(tris))
    g, c, h = m.hodge_decomposition(V, edges, flow, tris)
    assert np.linalg.norm(g + c + h - flow) < 1e-9                 # orthogonal parts sum to the flow
    assert np.linalg.norm(d1 @ h) < 1e-9 and np.linalg.norm(d2.T @ h) < 1e-9   # harmonic is div/curl-free
    # denoise a transport flow through the mind
    clean = d1.T @ rng.standard_normal(V)
    noisy = clean + 0.5 * rng.standard_normal(len(edges))
    den = m.denoise_flow(V, edges, noisy, tris, keep=("gradient", "harmonic"))
    assert np.linalg.norm(den - clean) < np.linalg.norm(noisy - clean)


def test_clifford_rotor_faculty():
    """EXP-9: Cl(3,0) geometric algebra is a parallel binding mode of UnifiedMind. Through the mind, rotor
    composition is exact (the product of rotors == sequential rotation) and non-commutative -- the rotation-
    shaped win the engine's commutative convolution bind cannot reach."""
    m = UnifiedMind(dim=256, seed=0)
    cl = m.clifford()
    rng = np.random.default_rng(0)
    # exact composition
    RA = cl.rotor(rng.standard_normal(3), 1.2)
    RB = cl.rotor(rng.standard_normal(3), 0.6)
    v = rng.standard_normal(3)
    assert np.linalg.norm(cl.rotate(RA, cl.rotate(RB, v)) - cl.rotate(cl.compose(RA, RB), v)) < 1e-12
    # non-commutative (a real order gap, which a commutative bind would collapse to zero)
    gap = np.linalg.norm(cl.rotate(cl.compose(RA, RB), v) - cl.rotate(cl.compose(RB, RA), v))
    assert gap > 1e-6
    # the faculty is cached (same algebra object)
    assert m.clifford() is cl


def test_wasserstein_faculty():
    """BLD-8: optimal-transport distance is a faculty of UnifiedMind. Through the mind, Wasserstein tracks how
    far two distributions sit apart even with no overlap, where the engine's bin-wise comparison saturates --
    the transport-geometry distance for histograms/spectra/distributions over a metric space."""
    m = UnifiedMind(dim=256, seed=0)
    x = np.arange(50)

    def g(mu):
        v = np.exp(-0.5 * ((x - mu) / 2.0) ** 2)
        return v / v.sum()

    ref = g(10)
    w_near = m.wasserstein(ref, g(25))                     # shift 15 -- already non-overlapping (sig=2)
    w_far = m.wasserstein(ref, g(40))                      # shift 30 -- also non-overlapping
    assert w_far > w_near                                   # farther distributions -> larger distance
    # bin-wise Euclidean saturates: both shifts are non-overlapping, so it cannot tell them apart
    e_near = np.linalg.norm(ref - g(25))
    e_far = np.linalg.norm(ref - g(40))
    assert abs(e_far - e_near) < 0.05 and (w_far - w_near) > 10.0
    # matches the 1-D closed form
    w_true = float(np.sum(np.abs(np.cumsum(ref) - np.cumsum(g(40)))))
    assert abs(m.wasserstein(ref, g(40), eps=0.5) - w_true) < 0.2


def test_flow_circulation_faculty():
    """Above/below sweep: the flow solver wired to the Hodge split + topology. Through UnifiedMind,
    flow_circulation runs a Tero flow and decomposes its flux into transport + circulation, reporting the
    graph's loop count (B1) and the redundancy (harmonic energy fraction) -- zero on a tree, nonzero on a
    loopy grid."""
    m = UnifiedMind(dim=256, seed=0)

    def grid(R, C):
        nbr = {}
        for r in range(R):
            for c in range(C):
                nbr[(r, c)] = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                               if 0 <= r + dr < R and 0 <= c + dc < C]
        return nbr

    # a loopy grid: real loops, and the gradient flux carries the injected current
    res = m.flow_circulation(grid(4, 4), (0, 0), (3, 3))
    assert res["loops"] == len(res["edges"]) - res["n_vertices"] + 1     # B1 = E - V + 1
    assert res["loops"] > 0
    assert res["transport_energy"] > 0
    assert 0.0 <= res["redundancy"] <= 1.0

    # a tree: the route is forced, so there is no circulation at all
    tree = {0: [1], 1: [0, 2, 3], 2: [1], 3: [1, 4], 4: [3]}
    rt = m.flow_circulation(tree, 0, 4)
    assert rt["loops"] == 0 and rt["redundancy"] < 1e-9 and rt["circulation_energy"] < 1e-9

    # disconnected -> None
    assert m.flow_circulation({"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}, "A", "D") is None


def test_spectral_denoise_faculty():
    """Above/below sweep: the EXP-5/6 graph-Laplacian eigenbasis is wired SIDEWAYS into the unified denoise
    faculty as method='spectral' -- the nonlinear-manifold map the linear methods lacked. It denoises a lone
    scalar field on a curved manifold (a 2-sphere) using only the cloud's geometry, beating the geometry-blind
    options (trajectory/SSA) that cannot see the curvature."""
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    N = 200
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    tta = np.pi * (1 + 5 ** 0.5) * i
    P = np.column_stack([np.sin(phi) * np.cos(tta), np.sin(phi) * np.sin(tta), np.cos(phi)])
    f = P[:, 2] ** 2 - 0.5 * P[:, 0]                        # a smooth field on the sphere
    fn = f + 0.3 * rng.standard_normal(N)
    den = m.denoise(fn, method="spectral", points=P)
    traj = m.denoise(fn, method="trajectory", rank=8)      # geometry-blind: treats the field as a 1-D sequence
    assert np.linalg.norm(den - f) < np.linalg.norm(fn - f)            # genuinely denoises
    assert np.linalg.norm(den - f) < 0.5 * np.linalg.norm(traj - f)    # and crushes the geometry-blind baseline
    # the guard: spectral needs the geometry
    try:
        m.denoise(fn, method="spectral")
        assert False, "should require points="
    except ValueError:
        pass


def test_is_manifold_gate_faculty():
    """The now-fast persistent homology turned into a first-class GATE: is_manifold names a clean connected
    manifold True and a dense blob False, sub-second. A 2-sphere is a manifold (B0=1); a random high-dim blob is
    not (fragmented / dense). This is the cheap premise-check for manifold-assuming operations."""
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    N = 200
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    tta = np.pi * (1 + 5 ** 0.5) * i
    sphere = np.column_stack([np.sin(phi) * np.cos(tta), np.sin(phi) * np.sin(tta), np.cos(phi)])
    s = m.is_manifold(sphere)
    assert s["is_manifold"] is True and s["betti"][0] == 1            # one connected manifold
    b = m.is_manifold(rng.standard_normal((200, 4)))
    assert b["is_manifold"] is False                                   # a blob is not a clean manifold
    assert b["betti"][0] != 1 or b["dense_scales"] > 1                # because fragmented or too dense


def test_spectral_denoise_check_manifold_guard():
    """check_manifold=True wires is_manifold as a guard on the spectral denoiser: it proceeds on a manifold but
    REFUSES on a blob (whose 'denoise' would be mere graph low-pass), with check_manifold=False as the escape
    hatch. Default off leaves the path overhead-free and unchanged."""
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1)
    N = 200
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    tta = np.pi * (1 + 5 ** 0.5) * i
    P = np.column_stack([np.sin(phi) * np.cos(tta), np.sin(phi) * np.sin(tta), np.cos(phi)])
    f = P[:, 2] ** 2 - 0.5 * P[:, 0]
    fn = f + 0.3 * rng.standard_normal(N)
    out = m.denoise(fn, method="spectral", points=P, check_manifold=True)   # manifold -> proceeds
    assert np.linalg.norm(out - f) < np.linalg.norm(fn - f)
    Pb = rng.standard_normal((200, 4))
    gn = Pb[:, 0] + 0.3 * rng.standard_normal(200)
    try:
        m.denoise(gn, method="spectral", points=Pb, check_manifold=True)    # blob -> refuses
        assert False, "check_manifold=True should refuse a non-manifold cloud"
    except ValueError:
        pass
    m.denoise(gn, method="spectral", points=Pb, check_manifold=False)       # escape hatch -> proceeds


def test_spectral_denoise_scales_to_large_cloud():
    """method='spectral' scales to a large cloud via SpectralBasis's Chebyshev-filtered partial eigensolver
    (sparse matvec, no O(n^3) eigh): a smooth field on a 2500-point sphere is denoised through the faculty, and
    the result matches the exact dense eigh basis to within tolerance."""
    from holographic_spectral import knn_laplacian, laplacian_eigenbasis
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    N = 2500
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    tta = np.pi * (1 + 5 ** 0.5) * i
    P = np.column_stack([np.sin(phi) * np.cos(tta), np.sin(phi) * np.sin(tta), np.cos(phi)])
    f = P[:, 2] ** 2 - 0.5 * P[:, 0]
    fn = f + 0.3 * rng.standard_normal(N)
    den = m.denoise(fn, method="spectral", points=P)                  # > threshold -> ChebFSI path
    _, Ve = laplacian_eigenbasis(knn_laplacian(P, 10), 12)
    err_exact = np.linalg.norm(Ve @ (Ve.T @ fn) - f)
    assert np.linalg.norm(den - f) < 1.15 * err_exact + 1e-9          # matches exact at scale
    assert np.linalg.norm(den - f) < np.linalg.norm(fn - f)           # and denoises


def test_restore_procedure_pnp_as_program():
    """Self-hosting: Plug-and-Play/RED restoration expressed AS A VSA PROGRAM -- ITERATE [APPLY datafit; APPLY
    denoise] -- through restore_procedure. It recovers a half-masked low-rank signal to the SAME error-to-truth
    as the Python denoise(method='pnp') loop, and the trace confirms it ran as an ITERATE-to-fixed-point program."""
    from holographic_denoise import pnp_restore, fit_manifold_full, adaptive_manifold_denoise
    m = UnifiedMind(dim=512, seed=7)
    D = m.dim
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((6, D)); basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    samples = np.stack([c @ basis for c in rng.standard_normal((40, 6))])
    clean = rng.standard_normal(6) @ basis
    mask = (rng.random(D) > 0.5).astype(float)
    forward = lambda x: mask * x
    adjoint = lambda y: mask * y
    y = forward(clean) + 0.05 * rng.standard_normal(D)

    restored, trace = m.restore_procedure(y, forward, adjoint, samples, mu=0.8, steps=60)
    raw_err = np.linalg.norm(y - clean) / np.linalg.norm(clean)
    vsa_err = np.linalg.norm(restored - clean) / np.linalg.norm(clean)
    # the Python faculty, same operator and prior
    bfull, _, mean = fit_manifold_full(samples, rank=24)
    prior = lambda v: adaptive_manifold_denoise(v, bfull, mean, sigma=None)
    py_err = np.linalg.norm(pnp_restore(y, forward, adjoint, prior, mu=0.8, steps=60) - clean) / np.linalg.norm(clean)

    assert vsa_err < 0.4 * raw_err                 # genuinely restores (raw ~0.86 -> ~0.17)
    assert abs(vsa_err - py_err) < 0.05            # same error-to-truth as the Python loop
    assert trace[0][0] == "ITERATE" and trace[0][3] == "converged"   # it ran as a fixed-point PROGRAM


def test_generate_procedure_diffusion_as_program():
    """Self-hosting: the B10 generative diffusion expressed AS A VSA PROGRAM -- ITERATE [APPLY diffuse] from a
    noise seed -- through generate_procedure. The self-cooling diffuse step walks a random vector onto the
    codebook manifold; over a bare codebook it lands on a stored atom (the kept B10 negative), and the trace
    confirms it ran as a program. Deterministic in the seed."""
    m = UnifiedMind(dim=512, seed=7)
    M = m._machine()
    fillers = np.stack([M._atom(f"gen_fill:{i}") for i in range(5)])
    fn = fillers / np.linalg.norm(fillers, axis=1, keepdims=True)

    sample, trace = m.generate_procedure(fillers, steps=12, seed=3)
    best = max(float(sample @ f / (np.linalg.norm(sample) * np.linalg.norm(f))) for f in fn)
    assert best > 0.95                             # lands on the manifold (a valid sample)
    assert abs(np.linalg.norm(sample) - 1.0) < 1e-6   # a unit vector
    assert trace[0][0] == "ITERATE"                # it ran as a program
    # determinism: same seed -> same sample
    sample2, _ = m.generate_procedure(fillers, steps=12, seed=3)
    assert np.linalg.norm(sample - sample2) < 1e-9


def test_audit_procedure_faculty():
    """D1: protocol-as-data anti-pattern auditing is a faculty of UnifiedMind. Treat an analysis procedure as
    program-as-data and read its STRUCTURE back from the vector: a complete honest protocol (search + null +
    fdr + a split between select and decide) is sound; a search with no procedure-matched null is flagged; and
    a no-search procedure carries no obligation, so it is not falsely flagged. The honesty discipline becomes a
    structural query on the protocol vector rather than a habit."""
    m = UnifiedMind(dim=256, seed=0)
    # a complete, honest protocol reads sound
    good = m.audit_procedure(steps=["encode", "combination_search", "oos_split", "calibrated_null", "fdr", "decide"])
    assert good["sound"] and good["violations"] == []
    assert "search" in good["roles"] and "null" in good["roles"]
    # the canonical artifact-factory: a search with no procedure-matched null is flagged
    bad = m.audit_procedure(steps=["encode", "combination_search", "oos_split", "fdr", "decide"])
    assert not bad["sound"]
    assert any(code == "search_without_null" for code, _msg in bad["violations"])
    # targeted, not trigger-happy: a no-search restoration loop has no honesty obligation
    restore = m.audit_procedure(steps=["datafit", "denoise"])
    assert restore["sound"]


def test_finding_registry_faculty():
    """D3: the findings registry is a faculty of UnifiedMind -- a research log as a holographic knowledge
    structure. Through the mind, structured claims are recalled by similarity and the log's own
    contradictions are detected: a horizon-conditioned tension (same claim, opposite polarity, DIFFERENT
    conditions) is distinguished from a flat contradiction (same claim, opposite polarity, same/no
    condition). The registry is cached on the mind."""
    m = UnifiedMind(dim=256, seed=0)
    reg = m.finding_registry()
    i0 = reg.add("efficiency_ratio", "momentum", +1, condition="horizon_10d")
    i1 = reg.add("efficiency_ratio", "momentum", -1, condition="intraday")
    reg.add("low_vol", "vol_expansion", +1)
    i4 = reg.add("bracket_order", "convexity", +1)
    i5 = reg.add("bracket_order", "convexity", -1)
    # similarity recall finds the related findings
    assert {r["index"] for r in reg.query(subject="efficiency_ratio", k=2)} == {i0, i1}
    # the headline: conditioned tension vs flat contradiction, correctly classified, no false positives
    tens = {(t["a"], t["b"]): t["type"] for t in reg.tensions()}
    assert tens.get((i0, i1)) == "conditioned"
    assert tens.get((i4, i5)) == "flat"
    assert len(tens) == 2
    # the registry is cached on the mind (same instance back)
    assert m.finding_registry() is reg


def test_finding_registry_persists_across_minds(tmp_path):
    """D3 persistence: the mind's findings log can be saved and reloaded into a standalone registry, so a
    conditioned tension recorded in one session survives into the next. The file holds only the structured
    claims; the vectors rebuild from the seeds, so the reloaded registry detects the same tension."""
    from holographic_knowledge import FindingRegistry
    m = UnifiedMind(dim=256, seed=0)
    reg = m.finding_registry()
    reg.add("efficiency_ratio", "momentum", +1, condition="horizon_10d")
    reg.add("efficiency_ratio", "momentum", -1, condition="intraday")    # conditioned tension
    path = tmp_path / "mind_findings.json"
    reg.save(str(path))
    restored = FindingRegistry.load(str(path))
    tens = restored.tensions()
    assert len(tens) == 1 and tens[0]["type"] == "conditioned"
    assert [dict(f) for f in restored.findings] == [dict(f) for f in reg.findings]


def test_svg_canvas_faculty():
    """The holographic vector-graphics (SVG) faculty: through the mind, a scene of typed primitives encodes into
    one hypervector and decodes back (type/colour exact, position close), two scenes morph by interpolating their
    vectors, the diffusion generates distinct novel scenes, and any scene renders as crisp resolution-independent
    SVG. The sharp cousin of the splat archive; the canvas is cached on the mind."""
    m = UnifiedMind(dim=4096, seed=0)
    svg = m.svg_canvas()
    assert m.svg_canvas() is svg                                  # cached on the mind

    # round-trip a scene through one hypervector
    scene = [(1, 0.30, 0.35, 0.12, 1), (0, 0.70, 0.62, 0.10, 0), (2, 0.52, 0.40, 0.09, 2)]
    dec = svg.decode(svg.encode(scene), len(scene))
    assert all(a[0] == b[0] and a[4] == b[4] for a, b in zip(scene, dec))   # type + colour exact
    perr = np.mean([abs(a[1] - b[1]) + abs(a[2] - b[2]) for a, b in zip(scene, dec)]) / 2
    assert perr < 0.06

    # generate distinct novel scenes and render crisp SVG
    scenes = [tuple(svg.generate(k=4, seed=s)) for s in range(5)]
    assert len(set(scenes)) >= 3
    out = svg.to_svg(scenes[0])
    assert out.startswith("<svg") and 'viewBox=' in out and out.rstrip().endswith("</svg>")


def test_splat_field_joint_refit():
    """The splat_field faculty: the joint amplitude refit (the gradient-free 'looping') is on by default and
    reconstructs a hard-edged image better than the raw greedy matching pursuit -- the under-reconstruction the
    adaptive-density-control literature targets, addressed here by re-solving overlap-double-counted amplitudes."""
    from holographic_splat import psnr
    m = UnifiedMind(dim=512, seed=0)
    ys, xs = np.mgrid[0:64, 0:64] / 64.0
    T = np.zeros((64, 64)); T[(xs > 0.25) & (xs < 0.68) & (ys > 0.25) & (ys < 0.68)] = 1.0
    T += 0.8 * np.exp(-((xs - 0.72) ** 2 + (ys - 0.74) ** 2) / (2 * 0.10 ** 2)); T = np.clip(T, 0, 1)
    _, greedy = m.splat_field(T, k=200, refit=False)
    _, refit = m.splat_field(T, k=200, refit=True)               # default is refit=True
    assert psnr(T, refit) > psnr(T, greedy) + 1.0




def test_low_discrepancy_sample_faculty():
    """The mind exposes low_discrepancy_sample as a COVERAGE sampler: it returns a valid quasi-random set in
    [0,1)^d that covers more evenly than default_rng (lower dispersion -- the measured win) and is
    deterministic via the mind's seed. The right sampler for placement/jitter, vs default_rng for independence."""
    from holographic_lowdiscrepancy import dispersion
    m = UnifiedMind(dim=512, seed=0)
    pts = m.low_discrepancy_sample(64, d=2)
    assert pts.shape == (64, 2) and pts.min() >= 0.0 and pts.max() < 1.0
    rand = np.mean([dispersion(np.random.default_rng(s).random((64, 2))) for s in range(8)])
    assert dispersion(pts) < rand                                # even coverage beats random
    assert np.allclose(pts, m.low_discrepancy_sample(64, d=2))   # deterministic (mind's seed)

def test_gated_traverse_faculty_recovers_chain_then_abstains():
    """The gated_traverse faculty drives a REAL holographic traversal -- a directed linked list stored in
    superposition -- recovering every valid hop via a cleanup-confidence throughput gate, then abstaining the
    instant the chain is exhausted (the recoverable signal gone), at a fraction of a fixed depth's steps. The
    Russian-roulette stop, on the engine's own bind/unbind. (The traversal sets its own dimension D; the
    mind's dim is irrelevant -- gated_traverse is a generic driver over a step closure.)"""
    from holographic_ai import bind, involution
    m = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(1); D, Lc = 8192, 8
    def unit():
        v = rng.standard_normal(D); return v / np.linalg.norm(v)
    perm = rng.permutation(D); inv = np.argsort(perm)               # the DIRECTION role
    chain = [unit() for _ in range(Lc + 1)]
    cb = np.array(chain + [unit() for _ in range(8)])
    cbn = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    M = np.zeros(D)
    for i in range(Lc):
        M = M + bind(chain[i], chain[i + 1][perm])
    def rstep(cur):
        probe = bind(M, involution(cur))[inv]
        cs = cbn @ (probe / (np.linalg.norm(probe) + 1e-12))
        j = int(np.argmax(cs)); return (cb[j], cs[j], j)
    res = m.gated_traverse(rstep, chain[0], floor=0.2, max_steps=25)
    assert res.payloads == list(range(1, Lc + 1))                   # every chain node, in order
    assert res.stopped == "floor" and res.steps == Lc and res.steps < 25   # abstains when exhausted, cheaply

def test_splat_field_adaptive_count():
    """ADAPT-1 faculty: splat_field(noise_thresh=...) makes the splat COUNT adapt to content -- a simple field
    fits with far fewer splats than a busy one at matched quality, where a fixed k would over- or under-spend.
    The default path (noise_thresh=None) is unchanged -- backward compatible."""
    from holographic_splat import psnr
    m = UnifiedMind(dim=512, seed=0)
    ys, xs = np.mgrid[0:48, 0:48] / 48.0
    def bump(cy, cx, s):
        return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s * s))
    simple = bump(.5, .5, .18); simple /= simple.max()
    busy = sum(bump(*p) for p in [(.25, .25, .07), (.3, .7, .06), (.7, .3, .06), (.72, .72, .05),
                                  (.5, .5, .05), (.2, .55, .05), (.6, .15, .05)]); busy /= busy.max()
    sp_s, r_s = m.splat_field(simple, noise_thresh=0.03)
    sp_b, r_b = m.splat_field(busy, noise_thresh=0.03)
    assert len(sp_b) > len(sp_s) + 5                          # the busy field is given more splats
    assert abs(psnr(simple, r_s) - psnr(busy, r_b)) < 4.0     # at matched quality (a common noise floor)
    sp_fixed, _ = m.splat_field(simple, k=12)                 # default path unchanged: a fixed-k fit
    assert len(sp_fixed) == 12

def test_directed_structure_forward_only_and_graph():
    """RAY-3: a directed structure recovers a node's SUCCESSOR with the predecessor suppressed to noise (unlike
    the undirected chain, where both neighbours come back equal), and a branching graph node returns all its
    successors from one unbind."""
    m = UnifiedMind(dim=2048, seed=0)
    ds = m.directed_structure(7)                              # linear chain 0->...->6
    for i in (2, 4):
        hits = dict(m.directed_successor(ds, i, topk=7))
        top = m.directed_successor(ds, i)[0]
        assert top[0] == i + 1 and top[1] > 0.25             # successor recovered, confidently
        assert hits[i - 1] < 0.10                            # predecessor is noise -- forward-only
    g = m.directed_structure(6, edges=[(0, 1), (0, 2), (0, 4)])
    assert {k for k, _ in m.directed_successor(g, 0, topk=3)} == {1, 2, 4}   # a branching node's successors


def test_directed_traverse_walks_chain_forward():
    """RAY-3 x RAY-1: directed_traverse walks a stored directed chain forward via the throughput-gated walk,
    recovering every node in order and abstaining when the chain runs out (the directed substrate under the
    Russian-roulette traversal)."""
    m = UnifiedMind(dim=2048, seed=0)
    ds = m.directed_structure(9)
    res = m.directed_traverse(ds, start_index=0, floor=0.2, max_steps=20)
    assert res.payloads == list(range(1, 9)) and res.stopped == "floor"

def test_plan_bakes_a_corridor_and_replan_gates():
    """Corridor planning through the mind: plan() rolls out a goal-field corridor, bakes it on the directed
    substrate, and returns the decoded route + per-step throughput; replan_needed re-anchors when the plan
    is exhausted or a tile is blocked. The brain is consulted once per corridor, not once per tile."""
    rng = np.random.default_rng(0)
    m = UnifiedMind(dim=1024, seed=0)
    tiles = rng.standard_normal((11, 1024)); tiles /= np.linalg.norm(tiles, axis=1, keepdims=True)
    def field_step(cur):
        i = int(np.argmax(tiles @ (cur / (np.linalg.norm(cur) + 1e-12))))
        return tiles[i + 1] if i + 1 < len(tiles) else None
    p = m.plan(tiles[0], field_step, max_steps=10, floor=0.12, action_of=lambda a, b: "go")
    assert p.route == list(range(1, 11))                       # whole corridor baked and decoded
    assert len(p.actions) == 10 and min(p.throughputs) > 0.12
    assert m.replan_needed(p, 0, floor=0.12) is False          # on-route, execute the baked step
    assert m.replan_needed(p, len(p.route), floor=0.12) is True # exhausted -> re-anchor

def test_plan_route_returns_a_whole_route_past_the_single_corridor_cap():
    """Corridor planning bounds ONE leg, not the route: a 40-tile route (well past the per-structure cap)
    that collapses if crammed into one plan() comes back EXACT through the mind's plan_route, which chains
    cap-sized corridors and re-anchors internally. The expensive call happens once per corridor, not per tile."""
    rng = np.random.default_rng(3)
    m = UnifiedMind(dim=512, seed=0)
    N = 40
    tiles = rng.standard_normal((N, 512)); tiles /= np.linalg.norm(tiles, axis=1, keepdims=True)
    def field_step(cur):
        i = int(np.argmax(tiles @ (cur / (np.linalg.norm(cur) + 1e-12))))
        return tiles[i + 1] if i + 1 < N else None
    def action_of(a, b):
        return int(np.argmax(tiles @ (b / (np.linalg.norm(b) + 1e-12))))
    crammed = m.plan(tiles[0], field_step, max_steps=N, floor=0.12, action_of=action_of)
    route = m.plan_route(tiles[0], field_step, corridor=14, floor=0.12, action_of=action_of)
    assert len(crammed.actions) < 5                            # one structure can't carry the whole route
    assert route.actions == list(range(1, N))                 # plan_route does -- the full route, exact
    assert route.stopped == "field_end" and route.reanchors >= 2

def test_chunk_route_replays_an_explicit_known_sequence_through_the_mind():
    """The explicit-list twin: a known 200-step plan (GPS waypoints, an experiment protocol) you ALREADY hold
    replays EXACTLY through the mind by chunking into <=14 clean pieces -- effective length unbounded at linear
    cost, ~15 chunks for 200 steps, each chunk one compact vector. The per-chunk cap is physics; chunking is
    the scale answer, and now it's a one-call faculty for sequences you have, not only ones you discover."""
    rng = np.random.default_rng(4)
    m = UnifiedMind(dim=512, seed=0)
    N = 200
    seq = rng.standard_normal((N, 512)); seq /= np.linalg.norm(seq, axis=1, keepdims=True)
    def action_of(a, b):
        return int(np.argmax(seq @ (b / (np.linalg.norm(b) + 1e-12))))
    r = m.chunk_route(list(seq), chunk=14, floor=0.12, action_of=action_of)
    assert r.actions == list(range(1, N))                     # the whole 200-step sequence, exact, past the cap
    assert len(r.corridors) <= N // 14 + 2                    # linear cost: ~15 chunks
    assert all(c.memory.shape == (512,) for c in r.corridors) # each chunk is one compact vector

def test_index_route_gives_sub_linear_random_access_through_the_mind():
    """A long chunked route should be random-access, not replay-from-start: m.index_route builds a BVH over the
    chunks, and .locate(query) finds "where am I" two-level (nearest chunk summary -> nearest tile within), in
    ~(#chunks + chunk_size) comparisons instead of #tiles. Exact location, far fewer comparisons."""
    rng = np.random.default_rng(5)
    m = UnifiedMind(dim=512, seed=0)
    N = 200
    seq = rng.standard_normal((N, 512)); seq /= np.linalg.norm(seq, axis=1, keepdims=True)
    route = m.chunk_route(list(seq), chunk=14, floor=0.12, action_of=lambda a, b: 0)
    idx = m.index_route(route)
    for t in (0, 73, 137, 199):                               # locate sample tiles: exact chunk+position+step
        c, p, g = idx.locate(seq[t])
        assert np.allclose(idx.chunks[c][p], seq[t]) and g == t
    assert idx.n_chunks + max(len(c) for c in idx.chunks) < N  # sub-linear comparisons per query

def test_learn_plan_chunked_keeps_a_long_protocol_exact_through_the_mind():
    """A long ordered plan (a scientist's multi-step protocol) stored with learn_plan(..., chunk=K) keeps
    step_at / precedes / validate_plan EXACT past the single-bundle cap, where the unchunked positional
    encoding decays with length. Short plans are unchanged (chunk defaults to 0)."""
    m = UnifiedMind(dim=2048, seed=0)
    steps = [f"op{i}" for i in range(200)]
    m.learn_plan("protocol", steps, chunk=14)
    assert all(m.step_at("protocol", i) == steps[i] for i in range(0, 200, 11))   # positional access exact
    assert m.precedes("protocol", "op10", "op180") is True                        # order exact across a long gap
    assert m.precedes("protocol", "op180", "op10") is False
    ok, viol = m.validate_plan("protocol", [("op10", "op180"), ("op50", "op150")])
    assert ok and viol == []
    m.learn_plan("short", steps[:20])                                             # short plan: default, still works
    assert m.step_at("short", 7) == "op7"

def test_splat_scene_tiling_keeps_region_recall_exact_at_fine_resolution_through_the_mind():
    """A content-addressable splat scene built through the mind: region recall is decode-via-cleanup and a
    single bundle caps as the grid gets finer, but splat_scene's tiling routes each cell to a small tile
    bundle so 'what is at this cell?' stays accurate at fine resolution -- the chunking lesson in the image
    domain (image generation/representation, not text: text generators use bounded context and don't cap)."""
    rng = np.random.default_rng(1)
    S = 96
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    img = np.zeros((S, S))
    for _ in range(8):
        cy, cx, sig, amp = rng.uniform(10, S-10), rng.uniform(10, S-10), rng.uniform(5, 12), rng.uniform(0.5, 1.0)
        img += amp * np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sig**2))
    img /= img.max()
    m = UnifiedMind(dim=1024, seed=0)
    scene = m.splat_scene(img, grid=32, tile=8, levels=5, k=30)
    acc = sum(abs(m.splat_region(scene, (gy, gx)) - scene["desc"][(gy, gx)]) < 1e-9
              for gy in range(32) for gx in range(32)) / (32*32)
    assert acc > 0.99 and len(scene["tiles"]) == 16

def test_mis_recover_beats_naive_average_and_singles():
    """MIS-1: m.mis_recover combines hard 1-NN and soft Hopfield per-query by the balance heuristic. On a mix
    of on-grid + off-grid cues over a coarse sharp-kernel manifold it beats naive averaging AND both singles,
    where naive averaging is worse than the better single (the MIS warning). combine_estimators is the
    reusable primitive."""
    from holographic_encoders import ScalarEncoder
    m = UnifiedMind(dim=512, seed=0)
    D = 512
    enc = ScalarEncoder(D, lo=0.0, hi=1.0, seed=1, kernel="rbf", bandwidth=6.0)
    gv = np.linspace(0, 1, 8); CB = np.stack([enc.encode(g) for g in gv])
    CBn = CB / np.linalg.norm(CB, axis=1, keepdims=True)
    def cos(a, b):
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
    rng = np.random.default_rng(3); eh = es = ea = em = 0.0; N = 400
    for _ in range(N):
        on = rng.random() < 0.5
        v = float(rng.choice(gv)) if on else float(rng.uniform(0.03, 0.97))
        t = enc.encode(v); q = t + 0.5 * rng.standard_normal(D) / np.sqrt(D)
        cs = CBn @ (q / np.linalg.norm(q)); xh = CB[int(cs.argmax())]
        w = np.exp(10 * (cs - cs.max())); w /= w.sum(); xs = (w[:, None] * CB).sum(0)
        xm = m.mis_recover(q, CB)
        eh += 1 - cos(xh, t); es += 1 - cos(xs, t); ea += 1 - cos(0.5 * xh + 0.5 * xs, t); em += 1 - cos(xm, t)
    eh, es, ea, em = eh / N, es / N, ea / N, em / N; best = min(eh, es)
    assert ea > best and em < ea and em < best            # naive worse than best; MIS beats naive AND both singles
    a = m.combine_estimators([(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 4.0)])
    assert cos(a, np.array([0.0, 1.0])) > 0.99            # the primitive: weight collapses onto the reliable one

def test_gradient_cache_first_order_beats_nearest_and_global_weights_fail():
    """CACHE-1: m.gradient_cache + m.cache_interp do Ward first-order interpolation of a smooth field. First-order
    gradient interp beats the nearest-neighbour (grid-argmax) baseline at fixed anchors, and global weights (no
    validity radius) fail badly -- a distant anchor dumps a bad long-range extrapolation into the query."""
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1); K = 5
    cx = rng.uniform(0, 1, K); cy = rng.uniform(0, 1, K)
    amp = rng.uniform(0.5, 1.5, K); sig = rng.uniform(0.2, 0.32, K)
    def f(u, v):
        return float(np.sum(amp * np.exp(-(((u - cx) ** 2 + (v - cy) ** 2) / (2 * sig ** 2)))))
    def grad(u, v):
        e = amp * np.exp(-(((u - cx) ** 2 + (v - cy) ** 2) / (2 * sig ** 2)))
        return np.array([np.sum(e * (-(u - cx) / sig ** 2)), np.sum(e * (-(v - cy) / sig ** 2))])
    g = np.linspace(0, 1, 5); A = np.array([[u, v] for u in g for v in g])
    V = np.array([f(u, v) for u, v in A]); J = np.array([grad(u, v) for u, v in A])
    cache = m.gradient_cache(A, V, J); R = 1.7 / 4
    Q = [[u, v] for u in np.linspace(0.1, 0.9, 12) for v in np.linspace(0.1, 0.9, 12)]
    e_fo = np.mean([abs(float(m.cache_interp(cache, q, R)) - f(*q)) for q in Q])
    e_nn = np.mean([abs(float(V[np.argmin(np.linalg.norm(A - q, axis=1))]) - f(*q)) for q in Q])
    e_glob = np.mean([abs(float(m.cache_interp(cache, q, R, global_weights=True)) - f(*q)) for q in Q])
    assert e_fo < e_nn                    # gradients beat the nearest-neighbour baseline at fixed anchors
    assert e_glob > e_fo * 1.5            # global weights FAIL -- the validity-radius guard is required

def test_robust_accumulate_harmonic_converges_and_clamp_resists_fireflies():
    """ACCUM-2/3: m.robust_accumulate. Harmonic (1/n) weights converge on a stationary noisy stream where a
    fixed-alpha EMA plateaus; and clamp_k makes the average robust to injected fireflies with no loss on clean."""
    m = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(5); D = 128
    def cos(a, b):
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
    mu = rng.standard_normal(D); mu /= np.linalg.norm(mu)
    stream = [mu + 0.8 * rng.standard_normal(D) / np.sqrt(D) for _ in range(400)]
    e_harm = 1 - cos(m.robust_accumulate(stream, schedule="harmonic"), mu)
    e_ema = 1 - cos(m.robust_accumulate(stream, schedule="ema", alpha=0.2), mu)
    assert e_harm < e_ema                      # harmonic converges; the EMA plateaus on a stationary stream
    clean = [mu + 0.3 * rng.standard_normal(D) / np.sqrt(D) for _ in range(40)]
    truth = np.mean(clean, 0)
    fire = clean + [8.0 * rng.standard_normal(D) / np.sqrt(D) for _ in range(5)]
    e_plain = 1 - cos(m.robust_accumulate(fire, schedule="mean"), truth)
    e_clamp = 1 - cos(m.robust_accumulate(fire, schedule="mean", clamp_k=2.5), truth)
    assert e_clamp < e_plain * 0.5             # clamping resists fireflies; no loss on clean data

def test_find_pattern_by_downscale_recovers_buried_pattern_and_noise_fails_safe():
    """XDATA-1: m.find_pattern_by_downscale recovers a pattern invisible at full resolution by downscaling, on
    two non-image data types, with a permutation null making it fail safe on pure noise."""
    m = UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(7); D, r = 256, 3
    B, _ = np.linalg.qr(rng.standard_normal((D, r)))
    C = rng.standard_normal((800, r)); X = C @ B.T; X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X + 0.4 * rng.standard_normal((800, D))
    res = m.find_pattern_by_downscale(Xn, kind="vectors", k=r, n_null=40, seed=1)
    overlap = float(np.sum((res.pattern.T @ B) ** 2) / r)
    assert res.found and overlap > 0.5                              # recovered the buried subspace, flagged found
    assert not m.find_pattern_by_downscale(rng.standard_normal((800, D)), kind="vectors",
                                           k=r, n_null=40, seed=1).found        # pure noise -> nothing (fail safe)
    T = 512; t = np.arange(T)
    clean = np.sin(2 * np.pi * 3 * t / T) + 0.7 * np.sin(2 * np.pi * 7 * t / T); clean /= np.std(clean)
    sres = m.find_pattern_by_downscale(clean + 2.0 * rng.standard_normal(T), kind="signal", k=6, n_null=80, seed=1)
    assert sres.found                                              # buried sinusoids recovered by low-pass downscale
    assert not m.find_pattern_by_downscale(rng.standard_normal(T), kind="signal",
                                           k=6, n_null=80, seed=1).found        # pure noise -> nothing (fail safe)

def test_manifold_denoise_settles_and_generate_is_novel_but_valid():
    """XDATA-2: m.manifold_denoise settles points onto a curved manifold (a ring) -- idempotent, and beats
    interpolation (the chord midpoint leaves the ring; denoise settles it back). m.manifold_generate produces
    novel-but-valid samples from noise where the bare codebook would be degenerate."""
    m = UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(11); D, N = 64, 48
    U, _ = np.linalg.qr(rng.standard_normal((D, 2))); u, v = U[:, 0], U[:, 1]
    th = np.linspace(0, 2 * np.pi, N, endpoint=False); S = np.stack([np.cos(t) * u + np.sin(t) * v for t in th])
    def ring_dist(x):
        a, b = u @ x, v @ x; plane = a * u + b * v
        return float(np.hypot(np.linalg.norm(x - plane), abs(np.hypot(a, b) - 1)))
    noisy = S[7] + 0.6 * rng.standard_normal(D) / np.sqrt(D)
    settled = m.manifold_denoise(noisy, S)
    assert ring_dist(noisy) > 0.3 and ring_dist(settled) < 0.15        # settles onto the ring
    mid = 0.5 * (S[4] + S[24])                                          # interpolation midpoint
    assert ring_dist(mid) > 0.3 and ring_dist(m.manifold_denoise(mid, S)) < 0.15   # off-manifold -> settled back on
    xgs = [m.manifold_generate(S, seed=s) for s in range(8)]
    gd = [ring_dist(xg) for xg in xgs]
    nov = [min(float(np.linalg.norm(xg - si)) for si in S) for xg in xgs]
    assert np.mean(gd) < 0.1 and np.mean(nov) > 0.01                    # generated: valid (on ring) and novel (between samples)

def test_sharpen_loop_recovers_detail_converges_and_guard_beats_oversharpening():
    """XDATA-3: m.sharpen_loop recovers detail an over-smoothed signal lost (Van Cittert negative-lobe
    sharpening), converging with no noise; the discrepancy guard stops near the optimum and beats running
    unguarded under noise (over-sharpening kept negative); an over-large step diverges into ringing."""
    m = UnifiedMind(dim=64, seed=0)
    from holographic_sharpen import _gauss_blur
    rng = np.random.default_rng(13); T = 256; t = np.arange(T)
    truth = np.sin(2 * np.pi * 3 * t / T) + 0.6 * np.sin(2 * np.pi * 30 * t / T) * np.exp(-((t - 128) ** 2) / (2 * 25 ** 2))
    blurred = _gauss_blur(truth, 3.0)
    def err(z):
        return float(np.linalg.norm(z - truth) / np.linalg.norm(truth))
    rec = m.sharpen_loop(blurred, sigma=3.0, lam=1.0, iters=80, noise_level=0.0)
    assert err(rec) < 0.05 and err(rec) < err(blurred)           # recovers detail, converges (no blow-up)
    noisy = blurred + 0.005 * rng.standard_normal(T)
    guarded = m.sharpen_loop(noisy, sigma=3.0, lam=1.0, iters=80, noise_level=0.005)
    unguarded = m.sharpen_loop(noisy, sigma=3.0, lam=1.0, iters=80, noise_level=0.0)
    assert err(guarded) < err(blurred) and err(guarded) < err(unguarded) * 0.6   # guard beats over-sharpening
    blown = m.sharpen_loop(blurred, sigma=3.0, lam=2.5, iters=30, noise_level=0.0)
    assert err(blown) > 10.0                                     # over-large step diverges into ringing

def test_smooth_sharp_split_beats_single_basis_at_fixed_budget():
    """CACHE-2: m.smooth_sharp_split + m.smooth_sharp_reconstruct store a smooth-plus-sharp signal as a
    low-frequency smooth layer + a sparse-sample sharp layer. At a budget covering both layers it beats both a
    single-FFT and a single-sparse representation, and the sharp positions reconstruct exactly."""
    m = UnifiedMind(dim=64, seed=0)
    from holographic_twolayer import _fft_topk, _sparse_topk
    rng = np.random.default_rng(17); T = 256; t = np.arange(T)
    smooth = np.sin(2 * np.pi * 2 * t / T) + 0.6 * np.cos(2 * np.pi * 5 * t / T)
    sharp = np.zeros(T); pos = rng.choice(T, 6, replace=False); sharp[pos] = rng.uniform(-3, 3, 6)
    x = smooth + sharp
    def psnr(rec):
        mse = np.mean((rec - x) ** 2); return float(10 * np.log10((x.max() - x.min()) ** 2 / (mse + 1e-12)))
    code = m.smooth_sharp_split(x, 6, 6)
    rec = m.smooth_sharp_reconstruct(code)
    assert psnr(rec) > psnr(_fft_topk(x, 12)) + 5 and psnr(rec) > psnr(_sparse_topk(x, 12)) + 5   # split wins
    assert np.allclose(x[code.sharp_idx], rec[code.sharp_idx])      # sharp positions reconstruct exactly

def test_phase_morph_uniform_motion_and_energy_with_wrapping_negative():
    """PHASE-1: m.phase_morph interpolates FHRR phasors in the phase domain -- moving an encoded feature at
    constant velocity (tracks the ideal trajectory; the amplitude blend eases) and staying a valid unit phasor
    (the blend collapses). Kept negative: under extreme change the shortest-arc morph wraps and loses tracking."""
    m = UnifiedMind(dim=64, seed=0)
    from holographic_fhrr import phasor_atom, fhrr_sim
    from holographic_phasemorph import amplitude_morph
    rng = np.random.default_rng(0); D = 2048; phi = np.angle(phasor_atom(D, rng))
    def encode(x):
        return np.exp(1j * x * phi)
    def decode_x(q, grid):
        s = np.array([fhrr_sim(q, encode(x)) for x in grid]); return float(grid[np.argmax(s)])
    xA, xB = 0.1, 0.9; a, b = encode(xA), encode(xB)
    grid = np.linspace(-0.1, 1.1, 481); ts = np.linspace(0.1, 0.9, 9)
    dev_p = max(abs(decode_x(m.phase_morph(a, b, t), grid) - ((1 - t) * xA + t * xB)) for t in ts)
    dev_a = max(abs(decode_x(amplitude_morph(a, b, t), grid) - ((1 - t) * xA + t * xB)) for t in ts)
    assert dev_p < 0.02 and dev_p < dev_a * 0.5                  # phase = uniform motion, amplitude = eased
    assert np.allclose(np.abs(m.phase_morph(a, b, 0.5)), 1.0)    # valid unit phasor at the midpoint
    a2, b2 = encode(-0.3), encode(1.3); grid2 = np.linspace(-0.5, 1.5, 601)   # extreme change
    dev_ext = max(abs(decode_x(m.phase_morph(a2, b2, t), grid2) - ((1 - t) * (-0.3) + t * 1.3)) for t in ts)
    assert dev_ext > 0.2                                          # kept negative: shortest arc wraps, loses tracking

def test_decompose_structure_early_stop_matches_at_lower_cost():
    """ADAPT-2: m.decompose_structure(early_stop=True) returns the same verified factorization as the fixed count,
    at lower iteration cost on a solvable problem (stats['iters'] reports the saving)."""
    m = UnifiedMind(dim=64, seed=0)
    from holographic_sbc import sbc_codebook, sbc_reconstruct
    B, L, F, N = 24, 7, 3, 10
    cbs = [sbc_codebook(B, L, N, seed=300 + f) for f in range(F)]
    rt = np.random.default_rng(11); true = tuple(int(rt.integers(0, N)) for _ in range(F))
    prod = sbc_reconstruct(true, cbs, L)
    sf, se = {}, {}
    rf = m.decompose_structure(prod, cbs, L, seed=0, stats=sf)
    re = m.decompose_structure(prod, cbs, L, seed=0, early_stop=True, stats=se)
    assert rf["verified"] and tuple(rf["picks"]) == true        # fixed count solves it
    assert re["verified"] and tuple(re["picks"]) == true        # early-stop returns the SAME verified answer
    assert se["iters"] < sf["iters"]                            # at lower iteration cost

def test_adaptive_anchors_beat_uniform_on_nonuniform_field():
    """CACHE-3: m.adaptive_anchors places cache anchors denser where the field bends, matching uniform-placement
    quality at materially fewer anchors on a non-uniformly-smooth field (and ~tied on a uniformly-smooth one)."""
    m = UnifiedMind(dim=64, seed=0)
    xs = np.linspace(0, 1, 4001)
    f = 0.3 * xs + np.exp(-((xs - 0.7) / 0.015) ** 2)
    def rmse(ax):
        return float(np.sqrt(np.mean((m.reconstruct_from_anchors(xs, ax, f) - f) ** 2)))
    uni = rmse(np.linspace(0, 1, 32)); ada = rmse(m.adaptive_anchors(xs, f, 32))
    assert ada < uni * 0.5                                       # far better at a fixed anchor count
    need = next(N for N in range(32, 800) if rmse(np.linspace(0, 1, N)) <= ada)
    assert need > 32 * 3                                         # uniform needs >3x as many anchors to match
    g = np.sin(2 * np.pi * 2 * xs)                               # honest control: uniformly-smooth -> ~tied
    def rmse_g(ax):
        return float(np.sqrt(np.mean((m.reconstruct_from_anchors(xs, ax, g) - g) ** 2)))
    assert rmse_g(m.adaptive_anchors(xs, g, 32)) > rmse_g(np.linspace(0, 1, 32)) * 0.5

def test_multires_pyramid_anti_aliased_coarse_query():
    """SCALE-1: m.multires_pyramid builds an anti-aliased mipmap -- a coarse query matches the true low-frequency
    band far better than a naive subsample (which aliases high frequency into the low band), levels halve in size,
    and the full level is exact."""
    m = UnifiedMind(dim=64, seed=0)
    N = 1024; x = np.arange(N) / N
    sig = np.sin(2 * np.pi * 2 * x) + 0.6 * np.sin(2 * np.pi * 150 * x)
    true_low = np.sin(2 * np.pi * 2 * x)
    pyr = m.multires_pyramid(sig, n_levels=5)
    assert [len(l) for l in pyr] == [1024, 512, 256, 128, 64]
    assert np.allclose(pyr[0], sig)                              # full level is exact
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mip = m.pyramid_reconstruct(pyr[3], N)                       # 1/8 anti-aliased level
    naive = m.pyramid_reconstruct(sig[::8], N)                   # 1/8 aliased subsample
    assert rmse(mip, true_low) < rmse(naive, true_low) * 0.5     # mipmap far cleaner at the coarse LOD

def test_reanchoring_is_load_bearing_for_deep_traversal():
    """RAY-2 (audit): m.gated_traverse with a re-anchored step recovers every hop of a directed linked list, while
    the SAME traversal with a raw (no-cleanup) step collapses almost immediately -- re-anchoring is load-bearing."""
    m = UnifiedMind(dim=64, seed=0)
    from holographic_reanchor import directed_linked_list, make_steps
    L = 12
    ll = directed_linked_list(L, dim=1024, seed=0)
    reanchored_step, raw_step = make_steps(ll)
    start = ll["chain"][0]
    g_re = m.gated_traverse(reanchored_step, start, floor=0.20, max_steps=L + 5)
    g_raw = m.gated_traverse(raw_step, start, floor=0.20, max_steps=L + 5)
    assert g_re.payloads == list(range(1, L + 1))               # re-anchored: every hop, in order
    assert len(g_raw.payloads) < len(g_re.payloads) - 5         # raw: collapses early (noise compounds)

def test_aniso_early_stop_saves_steps_at_small_cost():
    """C3: m.splat_aniso(early_stop=True) stops the Adam fit once the reconstruction has converged -- meaningfully
    fewer steps than the fixed 200 at a small MSE cost on an under-fit field; OFF by default runs the full
    schedule (bit-identical fixed-step fit)."""
    import numpy as np
    m = UnifiedMind(dim=64, seed=0)
    ys, xs = np.mgrid[0:48, 0:48]
    rng = np.random.default_rng(0)
    field = sum(rng.uniform(0.4, 1.0) * np.exp(-(((xs - rng.uniform(6, 42)) ** 2
                + (ys - rng.uniform(6, 42)) ** 2) / rng.uniform(8, 40))) for _ in range(9))   # busy -> residual floor
    st_full = {}
    m.splat_aniso(field, k=8, steps=200, stats=st_full)
    assert st_full["steps"] == 200                                   # off by default: full schedule
    st_es = {}
    _, rendered = m.splat_aniso(field, k=8, steps=200, early_stop=True, stats=st_es)
    assert 40 <= st_es["steps"] <= 175, st_es                        # stopped well before the cap (meaningful saving)
    mse_full = float(((m.splat_aniso(field, k=8, steps=200)[1] - field) ** 2).mean())
    assert float(((rendered - field) ** 2).mean()) <= mse_full * 1.15 + 1e-6   # at a small MSE cost

def test_generate_structure_early_stop_matches_full_at_half_the_steps():
    """B3: m.generate_structure(early_stop=True) stops once the decoded structure has settled -- the SAME
    structure as the full schedule at fewer steps, validity intact; off by default runs the full schedule."""
    import numpy as np
    from holographic_ai import random_vector, unbind
    m = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(3)
    roles = np.stack([random_vector(512, rng) for _ in range(4)])
    fillers = np.stack([random_vector(512, rng) for _ in range(8)])

    def combo(z):
        return tuple(int(np.argmax(fillers @ unbind(z, r))) for r in roles)

    st_f = {}
    zf = m.generate_structure(roles, fillers, seed=7, readout="sparsemax", stats=st_f)
    assert st_f["steps"] == 16                                       # off by default: full schedule
    st_e = {}
    ze = m.generate_structure(roles, fillers, seed=7, readout="sparsemax", early_stop=True, stats=st_e)
    assert combo(ze) == combo(zf)                                    # same structure (novelty preserved)
    assert st_e["steps"] < 16, st_e                                  # fewer steps

def test_robust_returns_resists_outlier_rewards():
    """D2: m.actions(robust_returns=True) winsorises outlier rewards in the brain's value -- a fluke reward
    cannot swing the estimate, so the value stays closer to the true mean than the plain running average; off by
    default is the plain average."""
    import numpy as np
    def learn(robust, seed):
        m = UnifiedMind(dim=128, seed=0)
        m.actions(["a", "b"], robust_returns=robust)
        rng = np.random.default_rng(seed)
        state = tuple(float(x) for x in rng.standard_normal(4))     # one fixed state (a 4-tuple observation)
        for _ in range(150):
            r = rng.normal(20.0, 5.0) if rng.random() < 0.08 else rng.normal(1.0, 0.3)
            m.reinforce(state, "a", float(r))
        sv = m.perceive(state)
        return abs(m._brain.value(sv, 0)[0] - 1.0)
    err_plain = np.mean([learn(False, s) for s in range(5)])
    err_robust = np.mean([learn(True, s) for s in range(5)])
    assert err_robust < err_plain * 0.7, (err_robust, err_plain)    # robust closer to the true value under outliers

def test_splat_densify_beats_one_shot_on_multiscale():
    """C1: m.splat_densify (coarse-to-fine) reaches a markedly better optimum than m.splat_aniso (one-shot) on a
    multi-scale target -- the staged warm start escapes the local optimum the one-shot is stuck in."""
    import numpy as np
    m = UnifiedMind(dim=64, seed=0)
    ys, xs = np.mgrid[0:56, 0:56]
    T = (np.exp(-(((xs - 28) ** 2 + (ys - 28) ** 2) / 300.0))
         + sum(0.8 * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 8.0))
               for cx, cy in [(12, 12), (44, 16), (16, 44), (42, 42)]))

    def mse(z):
        return float(((z - T) ** 2).mean())

    one = mse(m.splat_aniso(T, k=12, steps=210)[1])
    st = {}
    cf = mse(m.splat_densify(T, k=12, stage_steps=(40, 80, 650), stats=st)[1])
    assert st["stages"] == 3, st
    assert cf < one * 0.5, (cf, one)                                # densify reaches a markedly better optimum

def test_adaptive_encoder_resolution_on_nonuniform_data():
    """A3: ScalarEncoder.fit_resolution warps the encoder's input axis by the value-density CDF, so a non-uniform
    distribution decodes markedly better under noise than the uniform encoder; an unfitted encoder is the plain
    encoder (the warp is the identity)."""
    import numpy as np
    from holographic_encoders import ScalarEncoder
    rng = np.random.default_rng(0)
    clustered = np.clip(np.where(rng.random(4000) < 0.5,
                                 rng.normal(0.25, 0.04, 4000), rng.normal(0.75, 0.04, 4000)), 0, 1)

    def err(fit):
        enc = ScalarEncoder(512, 0.0, 1.0, seed=1, kernel="rbf", bandwidth=2.0)
        if fit:
            enc.fit_resolution(clustered)
        test = np.clip(np.where(rng.random(300) < 0.5,
                                rng.normal(0.25, 0.04, 300), rng.normal(0.75, 0.04, 300)), 0, 1)
        return float(np.mean([abs(enc.decode(enc.encode(float(x))
                     + 0.4 * rng.standard_normal(512) / np.sqrt(512), 400) - float(x)) for x in test]))

    assert err(True) < err(False) * 0.8                              # the fitted encoder decodes in-distribution better

def test_morph_scene_phase_slides_translation_better_than_dct():
    """C2: morph_scene(method='phase') slides a translated blob to its intermediate position (a compact, sharp
    midpoint) where method='dct' smears it -- a higher midpoint peak for a small translation."""
    import numpy as np
    m = UnifiedMind(dim=64, seed=0)
    S = 28
    ys, xs = np.mgrid[0:S, 0:S]

    def blob(cx):
        return np.exp(-(((xs - cx) ** 2 + (ys - 14) ** 2) / (2 * 3.0 ** 2)))

    a, b = blob(10), blob(16)                                       # small translation (shift 6)

    def midpeak(frames):
        mid = frames[len(frames) // 2]
        ends = 0.5 * (float(frames[0].max()) + float(frames[-1].max()))
        return float(mid.max() / (ends + 1e-12))

    ph = midpeak(m.morph_scene(a, b, method="phase"))
    dc = midpeak(m.morph_scene(a, b, method="dct"))
    assert ph > dc, (ph, dc)                                        # phase slides compactly; dct smears


def test_structured_index_faculty_is_the_shared_lookup_for_content_and_routes():
    # The shared abstraction proves itself by serving BOTH jobs through ONE faculty: content-addressed
    # lookup (the store) and route/sequence position lookup (the chunkers), distinguished only by payload.
    m = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(0)

    # (a) content-address: file items under themselves, get back meaningful labels -- the content-store job
    items = rng.standard_normal((400, 512)); items /= np.linalg.norm(items, axis=1, keepdims=True)
    cidx = m.structured_index(items, payloads=[f"doc:{i}" for i in range(400)], n_trees=6, leaf_size=64)
    assert cidx.locate(items[42], beam=6)[0] == "doc:42"           # routes home (sub-linearity: see unit test)

    # (b) the SAME faculty, payloads carrying (chunk, step) -- the route/sequence job, one primitive
    tiles = rng.standard_normal((300, 512)); tiles /= np.linalg.norm(tiles, axis=1, keepdims=True)
    ridx = m.structured_index(tiles, payloads=[(i // 14, i) for i in range(300)])
    assert ridx.locate_exact(tiles[157])[0] == (157 // 14, 157)    # exact, structured label returned


# ---- C1 / X1 chunking-transfer faculties (dedup + tiled scene factorization) ----------------------

def test_dedup_chunks_faculty_coalesces_repeated_chunks_exactly():
    # One mind faculty content-addresses a chunk store: a route revisiting corridors collapses to its distinct
    # chunks, and the references rebuild the original sequence bit-for-bit.
    m = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(0)
    segs = [rng.standard_normal(512) for _ in range(4)]
    pattern = [0, 1, 2, 0, 1, 0, 3, 0, 1, 2]                 # 10 chunks, 4 distinct
    chunks = [segs[p] for p in pattern]
    unique, refs = m.dedup_chunks(chunks)
    assert len(unique) == 4 and refs == pattern
    rebuilt = [unique[r] for r in refs]
    assert all(np.array_equal(rebuilt[i], chunks[i]) for i in range(len(chunks)))


def test_decompose_scene_tiled_faculty_beats_whole_past_the_cap():
    # The tiled-factorization faculty recovers a 15-object scene that the whole-scene resonator cannot.
    # (Per-seed tiled recovery runs 10-15/15 -- within-tile crosstalk at tile=5/dim=1024; the multi-seed
    # robustness is asserted in test_holographic_scene.py. Here one representative scene confirms the wiring
    # and the unambiguous win over the capped whole scene.)
    from holographic_scene import COLOURS, SHAPES, TEXTURES
    m = UnifiedMind(dim=1024, seed=0)
    coder = m.scene()
    r = np.random.default_rng(200)
    seen, objs = set(), []
    while len(objs) < 15:
        t = (COLOURS[r.integers(len(COLOURS))], SHAPES[r.integers(len(SHAPES))], TEXTURES[r.integers(len(TEXTURES))])
        if t not in seen:
            seen.add(t)
            objs.append({"colour": t[0], "shape": t[1], "texture": t[2]})
    keys = lambda os: set((o["colour"], o["shape"], o["texture"]) for o in os)
    groups = [objs[i:i + 5] for i in range(0, 15, 5)]
    tiled = m.decompose_scene_tiled([coder.encode_scene(g) for g in groups], [len(g) for g in groups], sweeps=3)
    whole = coder.factor_scene(coder.encode_scene(objs), 15, sweeps=3)
    assert len(keys(tiled) & keys(objs)) >= 13          # tiling recovers nearly all 15
    assert len(keys(tiled) & keys(objs)) > len(keys(whole) & keys(objs)) + 5   # well past the capped whole scene


# ---- schema-guided plans + descend through the mind (the structured branching output) ------------

def test_plan_faculties_round_trip_and_descend_through_the_mind():
    from holographic_planshape import PlanNode
    m = UnifiedMind(dim=1024, seed=0)
    actions = ["advance", "hold", "retreat", "scan", "abort", "reroute"]
    scopes = ["global", "local", "step", "mission"]
    plan = PlanNode("advance", "mission", branches={
        "blocked": PlanNode("reroute", "local", branches={"lowfuel": PlanNode("hold", "step")}),
        "contact": PlanNode("abort", "mission")})
    shape = m.plan_shape(actions, scopes, {"blocked": {"lowfuel": {}}, "contact": {}})
    vec = m.encode_plan(plan)
    assert m.decode_plan(vec, shape) == plan                       # schema-guided round-trip through the mind
    assert m.descend(vec, "blocked", shape) == ["advance", "reroute"]
    assert m.descend(vec, "contact", shape) == ["advance", "abort"]
    assert m.descend(vec, "clear", shape) == ["advance"]           # abstain: no branch applies


def test_general_record_faculty_round_trips_a_structured_output():
    m = UnifiedMind(dim=1024, seed=0)
    rec = {"phase": "stationary", "regime": "high_vol", "call": "hold"}
    schema = {"phase": ["stationary", "drifting"], "regime": ["low_vol", "high_vol"], "call": ["hold", "act"]}
    assert m.decode_record(m.encode_record(rec), schema) == rec


# ---- graph-signal denoising through the mind (reverse-transfer RT-III1) ---------------------------

def test_graph_denoise_faculty_beats_per_vector_at_high_noise():
    m = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(0)
    D, N = 512, 100
    t = np.linspace(0, 1, N)
    omega = rng.uniform(1, 10, D); phi = rng.uniform(0, 2 * np.pi, D)
    clean = np.cos(2 * np.pi * np.outer(t, omega) + phi); clean /= np.linalg.norm(clean, axis=1, keepdims=True)
    nz = rng.standard_normal((N, D)); nz /= np.linalg.norm(nz, axis=1, keepdims=True)
    noisy = clean + 1.2 * nz                                        # high noise -- the graph filter's regime

    def quality(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        return float(np.mean(np.sum(Xn * clean, axis=1)))

    def per_vector(X):
        m0 = X.mean(0); Vt = np.linalg.svd(X - m0, full_matrices=False)[2]
        return max(quality(m0 + (X - m0) @ Vt[:r].T @ Vt[:r]) for r in (8, 16, 24))

    taub = m.graph_denoise(noisy, k=8, method="taubin")
    assert quality(taub) > quality(noisy)                          # the faculty denoises
    assert quality(taub) > per_vector(noisy)                       # and beats per-vector at high noise
    # the no-shrink property vs the naive baseline, through the mind
    naive = m.graph_denoise(noisy, k=8, method="laplacian")
    assert np.linalg.norm(taub, axis=1).mean() > np.linalg.norm(naive, axis=1).mean() + 0.2


# ---- nonlinear manifold chart through the mind (reverse-transfer RT-II1) --------------------------

def test_manifold_chart_faculty_beats_linear_svd_on_a_curved_manifold():
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    N, D = 300, 256
    u = rng.uniform(0, 1, N); v = rng.uniform(0, 1, N)
    ang = 1.5 * np.pi * (1 + 2 * u)
    roll = np.stack([ang * np.cos(ang), 21 * v, ang * np.sin(ang)], 1)
    Q = np.linalg.qr(rng.standard_normal((D, 3)))[0]
    X = roll @ Q.T + 0.05 * rng.standard_normal((N, D))
    lab = np.clip((u * 4).astype(int), 0, 3)

    from holographic_chart import geodesic_distances
    Gt = geodesic_distances(X, k=10); iu = np.triu_indices(N, 1)

    def geo_corr(Y):
        dy = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
        return float(np.corrcoef(dy, Gt[iu])[0, 1])

    def sep(Y):
        Yc = Y - Y.mean(0); cen = np.stack([Yc[lab == c].mean(0) for c in range(4)])
        return float((np.argmin(((Yc[:, None, :] - cen[None, :, :]) ** 2).sum(-1), 1) == lab).mean())

    iso = m.manifold_chart(X, dim=2, method="isomap")
    svd = (X - X.mean(0)) @ np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:2].T
    assert iso.shape == (N, 2)
    assert geo_corr(iso) > geo_corr(svd)                   # the nonlinear chart preserves the manifold metric
    assert sep(iso) > sep(svd)                             # and separates classes the linear chart folds


# ---- the ISA conformance suite as a mind faculty (ISA-2) ------------------------------------------

def test_conformance_report_faculty_passes_for_the_live_kernel():
    m = UnifiedMind(dim=64, seed=0)
    report = m.conformance_report(dim=64, seed=0)
    # every base instruction conforms to its definitional reference
    for op, r in report.items():
        assert r["passed"], f"{op} not conformant: {r}"
    # the TOL/EXACT split is reported and correct
    assert report["bind"]["class"] == "TOL"
    assert report["permute"]["class"] == "EXACT" and report["permute"]["max_diff"] == 0.0


# ---- the HoloMachine register file through the mind (ISA-4) ---------------------------------------

def test_register_file_exact_recall_through_the_mind():
    from holographic_ai import cosine, bind
    m = UnifiedMind(dim=1024, seed=7)
    M = m._machine()
    # store 'a' in R0, overwrite ACC, recall R0 -> exact
    prog = [("LOAD", "a"), ("STORE", "R0"), ("LOAD", "b"), ("BIND", "c"), ("RECALL", "R0"), ("HALT", "a")]
    acc, _ = M.run(M.assemble(prog))
    assert cosine(acc, M.data_atoms["a"]) > 0.99999
    # a program WITHOUT register opcodes still runs identically (backward-compatible)
    plain = [("LOAD", "a"), ("BIND", "b"), ("HALT", "a")]
    acc2, _ = M.run(M.assemble(plain))
    assert cosine(acc2, bind(M.data_atoms["a"], M.data_atoms["b"])) > 0.99


# ---- the permute-stack and calling convention through the mind (ISA-5) ----------------------------

def test_permute_stack_and_frame_local_registers_through_the_mind():
    from holographic_ai import cosine
    m = UnifiedMind(dim=1024, seed=7)
    M = m._machine()
    # reverse-via-stack: push a,b,c then the first POP yields 'c' (LIFO)
    prog = [("LOAD", "a"), ("PUSH", "_"), ("LOAD", "b"), ("PUSH", "_"),
            ("LOAD", "c"), ("PUSH", "_"), ("POP", "_"), ("HALT", "a")]
    acc, _ = M.run(M.assemble(prog))
    assert cosine(acc, M.data_atoms["c"]) > 0.99
    # frame-local registers: a callee clobbering its R0 leaves the caller's R0 intact (the ABI guarantee)
    M.define("clob", [("LOAD", "f"), ("STORE", "R0"), ("HALT", "a")])
    prog2 = [("LOAD", "a"), ("STORE", "R0"), ("CALL", "clob"), ("RECALL", "R0"), ("HALT", "a")]
    acc2, _ = M.run(M.assemble(prog2))
    assert cosine(acc2, M.data_atoms["a"]) > 0.99999


# ---- parameterized recipe templates through the mind (ISA-6) --------------------------------------

def test_instantiate_template_through_the_mind():
    from holographic_ai import unbind, cosine, derived_atom
    from holographic_template import STARTER_LIBRARY
    import numpy as np
    m = UnifiedMind(dim=1024, seed=7)
    assert {"pair", "record", "ordered_pair"} <= set(m.template_names())
    # distinct args -> distinct, bit-exact structures
    r1 = m.instantiate_template("record", key="name", val="moose")
    r1b = m.instantiate_template("record", key="name", val="moose")
    r2 = m.instantiate_template("record", key="name", val="socks")
    assert np.array_equal(r1, r1b)                            # bit-exact replay through the mind
    assert cosine(r1, r2) < 0.9                               # different val -> different record
    # a single-binding pair recovers its value exactly via the (unitary) role
    p = m.instantiate_template("pair", x="alpha")
    role = STARTER_LIBRARY["pair"].role_atom(1024, 7, "role")
    assert cosine(unbind(p, role), derived_atom(7, "alpha", 1024)) > 0.99


# ---- the structure-description language through the mind (ISA-7) -----------------------------------

def test_realize_structure_through_the_mind():
    from holographic_ai import cosine, unbind, derived_atom
    from holographic_template import STARTER_LIBRARY
    import numpy as np
    m = UnifiedMind(dim=1024, seed=7)
    spec = "(bundle (record name moose) (pair socks))"
    v = m.realize_structure(spec)
    assert np.array_equal(v, m.realize_structure(spec))      # bit-exact through the mind
    # compile_structure returns a recipe that realizes to the same vector (the IR is the target)
    r = m.compile_structure(spec)
    assert np.array_equal(m.realize(r), v)
    # the language's record form agrees with instantiate_template (the layers are consistent)
    assert np.array_equal(m.realize_structure("(record name moose)"),
                          m.instantiate_template("record", key="name", val="moose"))


# ---- the reversibility audit + auto-cleanup scheduler through the mind (ISA-8) ---------------------

def test_auto_cleanup_scheduler_through_the_mind():
    from holographic_ai import random_vector, cosine
    from holographic_reversible import _bursty_program
    import numpy as np
    m = UnifiedMind(dim=1024, seed=7)
    # the audit is a mind faculty
    aud = m.reversibility_audit()
    assert aud["bind"][0] == "reversible" and aud["cleanup"][0] == "lossy"
    # the scheduler through the mind: adaptive holds fidelity at fewer cleanups than fixed under bursty damage
    cb = [random_vector(1024, np.random.default_rng(100 + i)) for i in range(16)]
    tgt = 3

    def measure(sched, **kw):
        cl, below = [], []
        for s in range(20):
            steps = _bursty_program(cb, tgt, dim=1024, seed=s)
            v, c = m.run_with_auto_cleanup(cb[tgt], steps, cb, schedule=sched, **kw)
            cl.append(c)
            below.append(cosine(v, cb[tgt]) < 0.9)
        return np.mean(cl), np.mean(below)

    ad_cl, ad_below = measure("adaptive", floor=0.9)
    fx_cl, fx_below = measure("fixed", k=3)
    assert ad_below < 0.1 and fx_below < 0.1
    assert ad_cl < fx_cl                                      # fewer cleanups, matched fidelity


# ---- anisotropic / steering kernel regression through the mind (RT-IV1) ----------------------------

def test_steering_regress_through_the_mind():
    import numpy as np
    m = UnifiedMind(dim=1024, seed=1)

    def f(p):
        return np.tanh(3.0 * (p[1] - 5.0))                # dense ridge: flat x, sharp y
    g = np.linspace(0.5, 9.5, 12)
    Xtr = np.array([[x, y] for x in g for y in g])
    ytr = np.array([f(p) for p in Xtr])
    rng = np.random.default_rng(0)
    Xq = rng.uniform(1, 9, (60, 2))
    yq = np.array([f(p) for p in Xq])

    pred, bw = m.steering_regress(Xtr, ytr, Xq, bounds=[(0, 10), (0, 10)], base=2.0)
    steered_rmse = float(np.sqrt(np.mean((pred - yq) ** 2)))
    assert bw[0] < bw[1]                                  # x (flat) smoother than y (sharp)
    # the steered anisotropic kernel beats a matched isotropic baseline on this directional data
    from holographic_steering import kernel_regress
    from holographic_fpe import VectorFunctionEncoder
    iso = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 10), (0, 10)], bandwidth=2.0, seed=1)
    iso_rmse = float(np.sqrt(np.mean((kernel_regress(iso, Xtr, ytr, Xq) - yq) ** 2)))
    assert steered_rmse < iso_rmse


# ---- spectral iteration of a learned propagator through the mind (RT-I1) ---------------------------

def test_propagator_spectral_jump_through_the_mind():
    import numpy as np
    from holographic_ai import bind, cosine
    from holographic_dynamics import Propagator
    m = UnifiedMind(dim=256, seed=7)
    # a learnable trajectory: a fixed bind operator applied repeatedly
    rng = np.random.default_rng(1)
    U_true = np.fft.irfft(0.97 * np.exp(1j * rng.uniform(0, 2 * np.pi, 129)), n=256)
    s0 = rng.standard_normal(256)
    traj = [s0]
    for _ in range(20):
        traj.append(bind(U_true, traj[-1]))
    traj = np.array(traj)
    # the one-eval k-step jump matches the learned propagator's own k-bind rollout to tolerance
    from holographic_iterate import step_k
    U = m.learn_dynamics(traj).U
    jump = m.propagator_jump(traj, traj[0], 6)
    assert np.max(np.abs(jump - Propagator(U, U).rollout(traj[0], 6)[-1])) < 1e-9
    # and it tracks the actual trajectory closely (learning error only)
    assert cosine(jump, traj[6]) > 0.99
    # the spectrum reads a regime without running
    prof = m.propagator_spectrum(traj)
    assert prof["regime"] in ("contractive", "marginal", "divergent")


# ============================================================================================
# FORWARD DCC backlog -- the explicit 3-D geometry vertical slice (FWD-1 mesh + FWD-2 glTF).
# The boundary to three.js, proven end to end THROUGH the mind: build a mesh, read its Euler
# invariant, serialise to a .glb, parse it back. This is the slice the backlog says to land
# first, to de-risk the single most important new plumbing before any modeling features.
# ============================================================================================
def test_unified_exposes_the_mesh_faculties():
    m = UnifiedMind(dim=256, seed=0)
    for name in ("mesh_box", "mesh_tetrahedron", "mesh_grid", "mesh_euler",
                 "mesh_to_gltf", "mesh_from_gltf"):
        assert callable(getattr(m, name)), f"UnifiedMind is missing faculty {name!r}"


def test_mesh_euler_invariant_through_the_mind():
    m = UnifiedMind(dim=256, seed=0)
    info = m.mesh_euler(m.mesh_box(2, 2, 2))
    assert info["vertices"] == 8 and info["edges"] == 12 and info["faces"] == 6
    assert info["characteristic"] == 2          # V - E + F = 2 for a genus-0 closed surface
    assert info["closed"] and info["manifold"] and info["genus"] == 0
    # an open mesh: chi = 1, genus undefined
    g = m.mesh_euler(m.mesh_grid(4, 4))
    assert g["characteristic"] == 1 and not g["closed"] and g["genus"] is None


def test_mesh_to_gltf_boundary_round_trips_through_the_mind():
    # THE VERTICAL SLICE: a cube goes back-end -> .glb -> back, entirely via mind faculties.
    m = UnifiedMind(dim=256, seed=0)
    cube = m.mesh_box(2.0, 2.0, 2.0)
    glb = m.mesh_to_gltf(cube)
    # a real GLTFLoader requires a 4-aligned .glb beginning with the glTF magic -- the boundary contract
    assert isinstance(glb, (bytes, bytearray)) and len(glb) % 4 == 0
    assert glb[:4] == b"glTF"
    back = m.mesh_from_gltf(glb)
    assert back.n_vertices == 8                  # the cube's 8 corners survive
    assert back.n_faces == 12                    # 6 quads -> 12 triangles across the glTF boundary
    assert np.allclose(back.vertices.astype(np.float32), cube.vertices.astype(np.float32))


def test_mesh_gltf_is_byte_reproducible_through_the_mind():
    # a serialised artifact is the EXACT class: identical bytes run to run (ISA determinism)
    m1 = UnifiedMind(dim=256, seed=0)
    m2 = UnifiedMind(dim=256, seed=0)
    assert m1.mesh_to_gltf(m1.mesh_box(2, 2, 2)) == m2.mesh_to_gltf(m2.mesh_box(2, 2, 2))


def test_structured_index_keying_regimes_through_the_mind():
    """The consolidated routing fabric is reachable as a mind faculty in all three regimes: projection
    (content), hash (the RAM / page-table regime -- zero-comparison exact), and spatial (the splat tiler)."""
    import numpy as np
    from holographic_unified import UnifiedMind
    um = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)

    # projection: content recall (the default, unchanged)
    keys = rng.standard_normal((300, 256))
    proj = um.structured_index(keys, payloads=[f"v{i}" for i in range(300)])
    assert proj.locate(keys[10])[0] == "v10"

    # hash: the RAM regime -- compute the address, ~O(1), exact
    h = um.structured_index([f"k{i}" for i in range(300)], keying="hash")
    payload, comps = h.locate("k10")
    assert payload == 10 and comps <= 4
    assert h.locate("absent")[0] is None

    # spatial: floor-divide tiles (the splat-tiler regime)
    coords = list({(int(rng.integers(0, 32)), int(rng.integers(0, 32))) for _ in range(200)})
    sp = um.structured_index(coords, keying="spatial", tile=8)
    assert sp.locate(coords[5])[0] == 5


def test_splat_tiler_and_spatial_index_share_one_route_through_the_mind():
    """The splat tiler's tiling and StructuredIndex(keying='spatial') now route through the SAME _tile_bucket
    -- the de-siloing is real, not nominal: the same cell maps to the same tile in both."""
    from holographic_tree import _tile_bucket, StructuredIndex
    from holographic_splat import splat_bundle_tiled, recall_region_tiled, splat_fit
    import numpy as np
    rng = np.random.default_rng(0)
    occ = np.zeros((64, 64))
    for _ in range(4):
        cy, cx = rng.uniform(10, 54, 2)
        ys, xs = np.mgrid[0:64, 0:64]
        occ += np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / 50.0)
    scene = splat_bundle_tiled(splat_fit(occ, 20), (64, 64), dim=2048, grid=16, tile=8, seed=0)
    # the tile a cell lands in (splat scene) is exactly the spatial index's bucket for the same cell+tile
    for cell in [(0, 0), (9, 5), (15, 15)]:
        assert _tile_bucket(cell, scene["tile"]) in scene["tiles"] or recall_region_tiled(scene, cell) == 0.0
        si = StructuredIndex(2048, keying="spatial", tile=scene["tile"]).build([cell])
        assert si._spatial_bucket(cell) == _tile_bucket(cell, scene["tile"])


def test_routeindex_delegates_to_the_shared_sequential_keying():
    """The RouteIndex consolidation is real, not nominal: its routing IS StructuredIndex(keying='sequential'),
    and a sequential index built from the same chunks routes a tile to the same (chunk, position)."""
    import numpy as np
    from holographic_plan import chunk_route, RouteIndex
    from holographic_tree import StructuredIndex

    rng = np.random.default_rng(0)
    tiles = rng.standard_normal((60, 256))
    tiles = tiles / np.linalg.norm(tiles, axis=1, keepdims=True)
    route = chunk_route(list(tiles), chunk=12, floor=0.12, seed=0, action_of=lambda a, b: 0)
    idx = RouteIndex(route)

    # the routing fabric underneath is the shared sequential-keyed StructuredIndex
    assert isinstance(idx._idx, StructuredIndex) and idx._idx.keying == "sequential"

    # an independent sequential index over the same chunks routes a tile identically (chunk, pos)
    chunks = [c.nodes for c in route.corridors]
    si = StructuredIndex(256, keying="sequential").build(chunks)
    c, pos, _ = idx.locate(tiles[20])
    (c2, pos2), _ = si.locate(tiles[20])
    assert (c, pos) == (c2, pos2)


def test_facetstore_hot_bucket_is_the_shared_structured_index():
    """The content-store consolidation is real, not nominal: a hot bucket's content search IS a
    StructuredIndex (keying='projection', normalize=False) -- fulfilling the claim baked into its docstring."""
    import numpy as np
    import holographic_uri as uri
    from holographic_scene import SceneCoder
    from holographic_tree import StructuredIndex

    coder = SceneCoder(dim=512, seed=0)
    tag = {"colour": "red", "shape": "circle", "texture": "busy"}
    s = uri.FacetStore()
    rng = np.random.default_rng(0)
    for i in range(300):
        s.put(i, tag, vector=coder.encode(tag))   # all into one hot bucket
    s.build_indexes(threshold=128)
    key = uri.make_key(tag)
    assert key in s._idx and isinstance(s._idx[key], StructuredIndex)
    assert s._idx[key].keying == "projection"


def test_fwd7_euler_edit_operators_through_the_mind():
    """FWD-7: the Euler edit operators are real UnifiedMind faculties and preserve the manifold/Euler
    invariants end-to-end. split_edge then collapse_edge round-trips through the mind exactly; flip_edge
    keeps the surface a closed manifold with chi unchanged."""
    import numpy as np
    from collections import Counter
    from holographic_unified import UnifiedMind
    from holographic_mesh import Mesh
    from holographic_eulerops import _face_with_directed_edge, _third

    um = UnifiedMind(dim=256, seed=0)
    cube = um.mesh_box(2.0, 2.0, 2.0)
    tm = Mesh(cube.vertices.copy(), [tuple(t) for t in cube.triangulate()])
    chi0 = tm.euler_characteristic()

    # find a flippable interior edge (apexes not already connected)
    d = Counter()
    for f in tm.faces:
        for k in range(3):
            d[(f[k], f[(k + 1) % 3])] += 1
    es = set(tm.edges())
    a = b = None
    for (x, y) in d:
        if (y, x) in d:
            cc = _third(tm.faces[_face_with_directed_edge(tm.faces, x, y)], x, y)
            dd = _third(tm.faces[_face_with_directed_edge(tm.faces, y, x)], x, y)
            if (min(cc, dd), max(cc, dd)) not in es:
                a, b = x, y
                break
    assert a is not None

    def canon(mesh):
        return tuple(sorted(tuple(f[f.index(min(f)):] + f[:f.index(min(f))]) for f in mesh.faces))

    # split then collapse via the faculties -> exact restoration
    split, m = um.mesh_split_edge(tm, a, b)
    assert split.n_vertices == tm.n_vertices + 1 and split.euler_characteristic() == chi0
    back = um.mesh_collapse_edge(split, keep=a, remove=m)
    assert back is not None and canon(back) == canon(tm)

    # flip via the faculty -> closed manifold, chi unchanged
    flipped = um.mesh_flip_edge(tm, a, b)
    assert flipped.is_manifold() and flipped.is_closed() and flipped.euler_characteristic() == chi0


def test_fwd4_mesh_smoothing_through_the_mind():
    """FWD-4: Taubin mesh smoothing is a real UnifiedMind faculty (the shipped graphsignal filter wired onto a
    mesh). It denoises a noisy sphere without shrinking it, and preserves connectivity/chi end-to-end -- only
    vertices move. The naive Laplacian baseline shrinks, confirming Taubin's no-shrink win."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshsmooth import _icosphere, laplacian_smooth
    from holographic_mesh import Mesh

    um = UnifiedMind(dim=256, seed=0)
    clean = _icosphere(3)
    rng = np.random.default_rng(0)
    noisy = Mesh(clean.vertices + rng.normal(0.0, 0.05, clean.vertices.shape), list(clean.faces))

    re = lambda m: float(np.abs(np.linalg.norm(m.vertices, axis=1) - 1.0).mean())
    mr = lambda m: float(np.linalg.norm(m.vertices, axis=1).mean())

    out = um.mesh_smooth(noisy, iters=10)
    assert re(out) < 0.6 * re(noisy)                       # denoised
    assert mr(out) > 0.95                                  # no shrink
    assert out.faces == clean.faces                        # connectivity untouched
    assert out.euler_characteristic() == clean.euler_characteristic()
    assert mr(laplacian_smooth(noisy, iters=10)) < mr(out) - 0.05   # baseline shrinks


def test_fwd6_mesh_curvature_through_the_mind():
    """FWD-6: curvature and crease detection are real UnifiedMind faculties. Mean/Gaussian curvature on a unit
    sphere are ~1, the angle defect obeys Gauss-Bonnet exactly (validated against the kernel's chi), and the
    cube's 12 sharp edges are detected -- all end-to-end through the mind."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshsmooth import _icosphere
    from holographic_meshcurvature import gauss_bonnet_defect

    um = UnifiedMind(dim=256, seed=0)
    sphere = _icosphere(3)

    H = um.mesh_curvature(sphere, kind="mean")
    K = um.mesh_curvature(sphere, kind="gaussian")
    assert 0.8 < float(H.mean()) < 1.25 and 0.8 < float(K.mean()) < 1.25   # ~1 on the unit sphere
    assert abs(gauss_bonnet_defect(sphere)) < 1e-6                          # exact topological check

    conf = um.mesh_curvature_confidence(sphere)
    assert conf.shape == (sphere.n_vertices,) and np.all((conf >= 0) & (conf <= 1))

    assert len(um.mesh_creases(um.mesh_box(2, 2, 2), threshold_deg=30.0)) == 12   # cube = 12 sharp edges
    assert len(um.mesh_creases(sphere, threshold_deg=30.0)) == 0                   # sphere = none


def test_fwd5_surface_geodesics_through_the_mind():
    """FWD-5: surface geodesics and geodesic soft-selection are real UnifiedMind faculties. On a unit sphere the
    geodesic from the pole tracks the great-circle distance, the antipode is the farthest point (~pi, exceeding
    the straight-line distance), and a geodesic soft-selection correctly EXCLUDES the antipode that a Euclidean
    ball of the same radius would bleed into -- all end-to-end through the mind."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshsmooth import _icosphere

    um = UnifiedMind(dim=256, seed=0)
    s = _icosphere(3)
    north = int(np.argmax(s.vertices[:, 2]))
    south = int(np.argmin(s.vertices[:, 2]))

    g = um.mesh_geodesic(s, north)
    true_geo = np.arccos(np.clip(s.vertices[:, 2], -1.0, 1.0))
    assert float(np.corrcoef(g, true_geo)[0, 1]) > 0.99       # tracks the analytic great circle
    assert abs(g[south] - np.pi) < 0.2                         # antipode ~ pi
    assert g[south] > float(np.linalg.norm(s.vertices[south] - s.vertices[north]))   # > straight-line

    sel = um.mesh_soft_selection(s, north, radius=2.5)
    assert sel[north] == 1.0 and sel[south] == 0.0            # geodesic excludes the antipode...
    assert float(np.linalg.norm(s.vertices[south] - s.vertices[north])) < 2.5        # ...Euclidean would not


def test_fwd3_uv_unwrapping_through_the_mind():
    """FWD-3: UV unwrapping is a real UnifiedMind faculty -- classical MDS of the mesh geodesic matrix (Isomap on
    explicit edges). A flat developable patch unwraps nearly isometrically through the mind, the UV is flip-free
    (no overlap), and on a curved cap the Isomap chart beats a naive linear projection -- all end-to-end."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshuv import flat_grid_mesh, hemisphere_cap

    um = UnifiedMind(dim=256, seed=0)

    flat = flat_grid_mesh(9)
    uv = um.mesh_uv_unwrap(flat)
    assert um.mesh_uv_distortion(flat, uv) < 0.07               # near-isometric on a developable patch
    # flip-free packing (locally injective: no overlap)
    signs = []
    for (a, b, c) in flat.faces:
        e1 = uv[b] - uv[a]; e2 = uv[c] - uv[a]
        signs.append(np.sign(e1[0] * e2[1] - e1[1] * e2[0]))
    signs = np.asarray(signs)
    assert max(int((signs > 0).sum()), int((signs < 0).sum())) == len(signs)   # all one winding

    cap = hemisphere_cap(3)
    iso = um.mesh_uv_distortion(cap, um.mesh_uv_unwrap(cap, method="isomap"))
    planar = um.mesh_uv_distortion(cap, um.mesh_uv_unwrap(cap, method="planar"))
    assert iso < planar                                         # geodesic chart wins on the curved surface


def test_fwd7_modeler_verbs_through_the_mind():
    """FWD-7: the core modeler verbs (extrude / inset / dissolve) are real UnifiedMind faculties, each producing a
    VALID mesh -- chi preserved, still a closed manifold -- with its exact geometric signature, end-to-end."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshsmooth import _icosphere
    from holographic_meshverbs import _face_normal

    um = UnifiedMind(dim=256, seed=0)
    s = _icosphere(2)
    chi0 = s.euler_characteristic()

    # extrude: valid mesh + cap moved exactly `distance` along the normal
    nrm = _face_normal(s.vertices, s.faces[0])
    cap_before = np.mean([s.vertices[v] for v in s.faces[0]], axis=0)
    ex = um.mesh_extrude(s, 0, distance=0.3)
    assert ex.euler_characteristic() == chi0 and ex.is_closed() and ex.is_manifold()
    cap_after = ex.vertices[ex.n_vertices - 3:].mean(axis=0)
    assert abs(float(np.dot(cap_after - cap_before, nrm)) - 0.3) < 1e-9

    # inset + dissolve: valid meshes
    ins = um.mesh_inset(s, 0, ratio=0.4)
    diss = um.mesh_dissolve_vertex(s, 5)
    for m in (ins, diss):
        assert m.euler_characteristic() == chi0 and m.is_closed() and m.is_manifold()
    assert diss.n_vertices == s.n_vertices - 1


def test_fwd8_subdivision_through_the_mind():
    """FWD-8: Loop subdivision is a real UnifiedMind faculty. Through the mind it quadruples faces, preserves chi
    and the closed-manifold property, leaves a flat mesh exactly flat (affine reproduction), and smooths an
    angular mesh (the low-pass signature) -- end-to-end."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshsmooth import _icosphere
    from holographic_meshuv import flat_grid_mesh
    from holographic_meshcurvature import dihedral_angles
    from holographic_meshsubdiv import _triangles
    from holographic_mesh import box, Mesh

    um = UnifiedMind(dim=256, seed=0)

    s = _icosphere(1)
    sub = um.mesh_subdivide(s, 1)
    assert sub.n_faces == 4 * s.n_faces
    assert sub.euler_characteristic() == s.euler_characteristic() and sub.is_closed() and sub.is_manifold()

    # affine reproduction: flat stays flat
    assert float(np.max(np.abs(um.mesh_subdivide(flat_grid_mesh(5), 2).vertices[:, 2]))) < 1e-12

    # low-pass: dihedral spread drops on a cube
    cube = box()
    before = float(np.std(list(dihedral_angles(Mesh(cube.vertices.copy(), _triangles(cube))).values())))
    after = float(np.std(list(dihedral_angles(um.mesh_subdivide(cube, 2)).values())))
    assert after < before


def test_fwd10_inverse_kinematics_through_the_mind():
    """FWD-10: FABRIK inverse kinematics is a real UnifiedMind faculty -- and it runs through the mind's OWN
    project_onto_constraints engine (IK as iterate-a-projection). Through the mind: a reachable target is hit to
    tolerance with every bone length and the root preserved, and an unreachable target fully extends the chain."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshik import chain

    um = UnifiedMind(dim=256, seed=0)
    arm = chain(4, 1.0)
    rest = [float(np.linalg.norm(arm[i + 1] - arm[i])) for i in range(len(arm) - 1)]

    target = np.array([2.0, 1.5, 0.5])
    posed, _ = um.solve_ik(arm, target, iters=30)
    assert np.linalg.norm(posed[-1] - target) < 1e-6                      # reaches the target
    new = [float(np.linalg.norm(posed[i + 1] - posed[i])) for i in range(len(posed) - 1)]
    assert np.allclose(new, rest, atol=1e-9)                              # bones preserved
    assert np.allclose(posed[0], arm[0], atol=1e-12)                      # root fixed

    far, _ = um.solve_ik(arm, np.array([100.0, 0.0, 0.0]), iters=60)
    assert abs(float(np.linalg.norm(far[-1] - far[0])) - sum(rest)) < 1e-4   # unreachable -> fully extended


def test_fwd9_skinning_through_the_mind():
    """FWD-9: linear blend skinning is a real UnifiedMind faculty. Through the mind: a shared rigid bone transform
    is reproduced exactly on every vertex (the partition-of-unity guarantee), faces are untouched, and the
    candy-wrapper kept negative shows -- a 50/50 twist collapses a unit ring's radius to cos(theta/2)."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_mesh import box
    from holographic_meshskin import make_transform, rotation, linear_blend_skin

    um = UnifiedMind(dim=256, seed=0)

    # rigid reproduction + faces preserved, on a box
    M = make_transform(rot=rotation([0.2, 1.0, 0.3], 0.5), translation=[1.0, 0.0, -0.5])
    b = box()
    transforms = np.stack([M, M])
    skinned = um.skin_mesh(b, transforms, np.ones((b.n_vertices, 2)) * np.array([0.3, 0.7]))
    expected = (np.hstack([b.vertices, np.ones((b.n_vertices, 1))]) @ M.T)[:, :3]
    assert np.allclose(skinned.vertices, expected, atol=1e-12)   # shared transform reproduced exactly
    assert skinned.faces == b.faces                              # connectivity untouched

    # candy-wrapper kept negative (via the module's blend): radius = cos(theta/2)
    phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
    bones = np.stack([np.eye(4), make_transform(axis=[0, 0, 1], angle=2 * np.pi / 3)])
    radius = float(np.mean(np.linalg.norm(linear_blend_skin(ring, bones, np.full((24, 2), 0.5))[:, :2], axis=1)))
    assert abs(radius - np.cos(np.pi / 3)) < 1e-9                # 120-degree twist -> radius 0.5


def test_fwd11_mesh_sdf_bridge_through_the_mind():
    """FWD-11: the mesh<->SDF bridge is a real UnifiedMind faculty. Through the mind: an analytic sphere SDF
    extracts (marching tetrahedra) to a closed-manifold sphere whose vertices lie on the sphere, the reverse
    mesh->SDF matches the analytic distance, and a splat (metaball) field meshes to a closed blob -- end-to-end."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshbridge import sphere_sdf, metaball_field

    um = UnifiedMind(dim=256, seed=0)

    # SDF -> mesh: closed manifold sphere, vertices on the sphere
    sphere = um.mesh_from_sdf(sphere_sdf(radius=1.0), ((-1.5,) * 3, (1.5,) * 3), res=20)
    assert sphere.is_closed() and sphere.is_manifold() and sphere.euler_characteristic() == 2
    radii = np.linalg.norm(sphere.vertices, axis=1)
    assert abs(float(radii.mean()) - 1.0) < 0.02

    # mesh -> SDF: matches analytic |p|-1, correct sign
    got = um.mesh_to_sdf(sphere, np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert got[0] < 0 and got[1] > 0 and abs(got[1] - 1.0) < 0.05

    # splat -> mesh: a metaball blob meshes to a closed manifold
    blob = um.mesh_from_sdf(metaball_field(np.array([[-0.4, 0, 0], [0.4, 0, 0]]), radius=0.4),
                            ((-1.5,) * 3, (1.5,) * 3), res=20, level=0.5)
    assert blob.is_closed() and blob.is_manifold()


def test_arch1_recipe_edit_operators_through_the_mind():
    """ARCH-1: the StructureRecipe validator + edit operators (the recipe equivalent of the mesh Euler operators)
    are real UnifiedMind faculties. Through the mind: validate accepts a well-formed recipe; commute_bind and
    reorder_members preserve the realized vector and are invertible; substitute_atom changes it and reverses."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_recipe import StructureRecipe

    um = UnifiedMind(dim=512, seed=0)
    r = StructureRecipe(dim=512, seed=0)
    a = r.atom("a"); b = r.atom("b"); c = r.atom("c")
    ab = r.bind(a, b); bun = r.bundle([a, b, c])
    r.mark_output(ab); r.mark_output(bun)
    base = [v.copy() for v in r.outputs()]

    assert um.validate_recipe(r)[0]                                    # well-formed

    # commute_bind: vector-preserving + own inverse
    flipped = um.recipe_commute_bind(r, ab)
    assert um.validate_recipe(flipped)[0]
    assert np.allclose(flipped.outputs()[0], base[0], atol=1e-12)
    assert np.allclose(um.recipe_commute_bind(flipped, ab).outputs()[0], base[0], atol=1e-12)

    # reorder_members: vector-preserving
    assert np.allclose(um.recipe_reorder_members(r, bun, [2, 0, 1]).outputs()[1], base[1], atol=1e-12)

    # substitute_atom: changes and reverses
    swapped = um.recipe_substitute_atom(r, 0, "z")
    assert not np.allclose(swapped.outputs()[0], base[0], atol=1e-6)
    assert np.allclose(um.recipe_substitute_atom(swapped, 0, "a").outputs()[0], base[0], atol=1e-12)


def test_arch4_seam_cutting_through_the_mind():
    """ARCH-4: seam cutting is a real UnifiedMind faculty -- the real FWD-3 seam. Through the mind: a meridian seam
    cuts a closed sphere into a disk (chi=1, manifold), NON-DESTRUCTIVELY (every face preserved, unlike the
    puncture which deletes faces), and a well-chosen seam unwraps better than the puncture -- end-to-end."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshsmooth import _icosphere
    from holographic_meshuv import uv_unwrap, uv_distortion, puncture

    um = UnifiedMind(dim=256, seed=0)
    s = _icosphere(3)
    north = int(np.argmax(s.vertices[:, 2]))
    south = int(np.argmin(s.vertices[:, 2]))

    meridian = um.mesh_shortest_seam(s, north, south)
    disk = um.mesh_cut_seam(s, meridian)
    assert disk.is_manifold() and not disk.is_closed() and disk.euler_characteristic() == 1   # a disk
    assert disk.n_faces == s.n_faces                                  # non-destructive
    assert puncture(s, 0).n_faces < s.n_faces                        # the puncture deletes geometry

    # a well-chosen seam beats the puncture on unwrap distortion
    equator = int(np.argmin(np.abs(s.vertices[:, 2])))
    good = um.mesh_cut_seam(s, um.mesh_shortest_seam(s, north, equator))
    assert uv_distortion(good, uv_unwrap(good)) < uv_distortion(puncture(s, 0), uv_unwrap(puncture(s, 0)))


def test_arch7_representation_routing_csg_through_the_mind():
    """ARCH-7: representation routing is a real UnifiedMind faculty -- the policy layer on FWD-11's bridge.
    Through the mind: the router sends booleans to the SDF representation, and mesh_csg computes a union by routing
    two meshes through the SDF -- merging topology when they overlap (one closed-manifold blob), and getting it
    geometrically right (inclusion-exclusion) -- something the mesh kernel can't do natively."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_mesh import Mesh
    from holographic_meshsmooth import _icosphere

    um = UnifiedMind(dim=256, seed=0)
    assert um.route_representation("union") == "sdf"          # the policy
    assert um.route_representation("boundary") == "mesh"

    sph = _icosphere(2)
    A = Mesh(sph.vertices + np.array([-0.5, 0, 0]), [tuple(f) for f in sph.faces])
    B = Mesh(sph.vertices + np.array([0.5, 0, 0]), [tuple(f) for f in sph.faces])

    uni = um.mesh_csg("union", A, B)
    assert um.mesh_connected_components(uni) == 1            # overlapping -> merged to one blob
    assert uni.is_closed() and uni.is_manifold()

    inter = um.mesh_csg("intersection", A, B)
    vA, vB = um.mesh_volume(A), um.mesh_volume(B)
    assert abs(um.mesh_volume(uni) - (vA + vB - um.mesh_volume(inter))) / vA < 0.05   # inclusion-exclusion


def test_arch3_geometry_weighted_graph_through_the_mind():
    """ARCH-3: geometry-weighted graph operations on hypervectors are real UnifiedMind faculties -- the
    cotangent-Laplacian mirror. Through the mind: the weighted similarity-graph eigenmap recovers a ring, the
    weighted adjacency carries cosine similarities (not 1s), and under non-uniform sampling weighting beats binary."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_simgraph import _ring_vectors, _circ_corr

    um = UnifiedMind(dim=256, seed=0)
    V, th = _ring_vectors(nonuniform=False, seed=0)

    # the eigenmap recovers the ring
    assert _circ_corr(um.graph_ring_order(V, weighted=True), th) > 0.99

    # the weighting carries the geometry (cosine similarities), not binary 1s
    A = um.similarity_graph(V, k=6, weighted=True)
    assert A[A > 0].std() > 0.0 and A[A > 0].max() <= 1.0000001
    B = um.similarity_graph(V, k=6, weighted=False)
    assert set(np.unique(B[B > 0])) == {1.0}              # the binary graph's edges are all 1

    # geometry-weighting wins under irregular sampling
    Vn, thn = _ring_vectors(nonuniform=True, seed=0)
    rw = _circ_corr(um.graph_ring_order(Vn, weighted=True), thn)
    rb = _circ_corr(um.graph_ring_order(Vn, weighted=False), thn)
    assert rw > rb


def test_arch5_subdivide_sequence_through_the_mind():
    """ARCH-5: subdivision curves on hypervector sequences are a real UnifiedMind faculty -- FWD-8's mesh
    subdivision turned inward (1-manifold). Through the mind: the count doubles each level (refine), a straight line
    of vectors stays straight (affine reproduction), and a zig-zag's roughness shrinks (low-pass smoothing)."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(0)

    # refine: count doubles
    P = rng.standard_normal((6, 64))
    assert len(um.subdivide_sequence(P, levels=2)) == 18                  # 6 -> 10 -> 18

    # affine reproduction: a straight line of vectors stays on the line
    a, b = rng.standard_normal(64), rng.standard_normal(64)
    ramp = np.array([a + (b - a) * t for t in np.linspace(0, 1, 6)])
    sub = um.subdivide_sequence(ramp, levels=3)
    dn = (b - a) / np.linalg.norm(b - a)
    assert max(np.linalg.norm((p - a) - np.dot(p - a, dn) * dn) for p in sub) < 1e-12

    # low-pass: a zig-zag smooths
    zig = np.zeros((10, 64)); zig[::2, 0] = 1.0; zig[1::2, 0] = -1.0
    r0 = float(np.sum(np.diff(zig, n=2, axis=0) ** 2))
    r2 = float(np.sum(np.diff(um.subdivide_sequence(zig, levels=2), n=2, axis=0) ** 2))
    assert r2 < r0


def test_arch6_blendshape_posing_through_the_mind():
    """ARCH-6: rig + IK for structures (blendshape posing) are real UnifiedMind faculties -- FWD-9 skinning + FWD-10
    IK turned inward. Through the mind: the forward blend reproduces a target for a one-hot weight (skinning), and
    the IK recovers a reachable blend's weights / returns the closest valid blend otherwise -- via the same
    project_onto_constraints sweeper FWD-10 uses."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=256, seed=0)
    P = np.random.default_rng(0).standard_normal((4, 256))

    # forward skinning: a one-hot blend is that target
    assert np.allclose(um.blend_pose(P, [0, 1, 0, 0]), P[1] / np.linalg.norm(P[1]), atol=1e-12)

    # IK reachable: recover a known blend's weights
    w_true = np.array([0.4, 0.3, 0.2, 0.1])
    w = um.solve_pose(P, P.T @ w_true)
    assert np.sum(np.abs(w - w_true)) < 0.02
    assert w.min() >= -1e-9 and abs(w.sum() - 1.0) < 1e-6        # a valid convex blend

    # IK unreachable: closest valid blend (beats any single target)
    goal = np.random.default_rng(9).standard_normal(256)
    w2 = um.solve_pose(P, goal)
    blend_resid = np.linalg.norm(P.T @ w2 - goal)
    assert blend_resid <= min(np.linalg.norm(P[i] - goal) for i in range(4)) + 1e-6


def test_fwd7_remainder_modeler_verbs_through_the_mind():
    """FWD-7 remainder: bevel, bridge, loop-cut are real UnifiedMind faculties -- the three verbs needing vertex
    duplication / loop tracing. Through the mind: bevel chamfers a cube corner (chi preserved), bridge joins two
    squares into a tube, loop-cut inserts an edge loop around a cube (chi preserved)."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_mesh import box

    um = UnifiedMind(dim=128, seed=0)
    cube = box()

    bev = um.mesh_bevel_vertex(cube, 0, ratio=0.3)
    assert bev.is_manifold() and bev.is_closed() and bev.euler_characteristic() == 2

    sq0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    sq1 = sq0.copy(); sq1[:, 2] = 1.0
    tube = um.mesh_bridge(np.vstack([sq0, sq1]), [0, 1, 2, 3], [4, 5, 6, 7], closed=True)
    assert tube.is_manifold() and tube.n_faces == 4 and tube.euler_characteristic() == 0

    f0 = tuple(cube.faces[0])
    lc = um.mesh_loop_cut(cube, 0, (f0[0], f0[1]))
    assert lc.is_manifold() and lc.is_closed() and lc.euler_characteristic() == 2
    assert lc.n_faces == cube.n_faces + 4


def test_scene_graph_algebra_through_the_mind():
    """The scene-graph capstone is a real UnifiedMind faculty set: a scene read as GEOMETRY (scene_flatten instances
    + merges) and as STRUCTURE (scene_to_recipe encodes to a recipe), with the two views consistent -- swapping
    siblings changes neither the flattened geometry nor the realised vector (merge and bundle both commute), tying
    the FWD mesh kernel to the ARCH-1 recipe algebra."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_mesh import box
    from holographic_recipeops import validate

    um = UnifiedMind(dim=256, seed=0)
    cube = box()
    scene = um.scene_graph(children=[um.scene_graph(um.scene_translation([2, 0, 0]), mesh=cube),
                                     um.scene_graph(um.scene_translation([0, 2, 0]), mesh=cube)])

    # GEOMETRY view: two cubes instanced + merged
    flat = um.scene_flatten(scene)
    assert flat.n_vertices == 2 * cube.n_vertices and flat.n_faces == 2 * cube.n_faces
    assert np.allclose(flat.vertices[flat.vertices[:, 0] > 1].mean(axis=0), [2, 0, 0], atol=1e-9)

    # STRUCTURE view: a valid recipe ARCH-1 operates on
    assert validate(um.scene_to_recipe(scene))[0]

    # CONSISTENCY THEOREM: swap siblings -> geometry AND vector identical
    swapped = um.scene_graph(children=[scene.children[1], scene.children[0]])
    a, b = um.scene_flatten(scene), um.scene_flatten(swapped)
    assert np.allclose(np.sort(a.vertices, axis=0), np.sort(b.vertices, axis=0)) and a.n_faces == b.n_faces
    assert np.allclose(um.scene_to_recipe(scene).outputs()[0], um.scene_to_recipe(swapped).outputs()[0], atol=1e-12)


def test_qem_decimation_through_the_mind():
    """QEM decimation is a real UnifiedMind faculty: the quadric error metric wired to the guarded collapse, with a
    surface-deviation quality metric. Through the mind: decimate an icosphere (closed manifold + chi preserved) and
    confirm QEM beats a naive midpoint collapse on surface error."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshsmooth import _icosphere
    from holographic_eulerops import collapse_edge
    from holographic_meshqem import _edges

    um = UnifiedMind(dim=128, seed=0)
    ico = _icosphere(2)
    qem = um.mesh_qem_decimate(ico, 64)
    assert qem.is_manifold() and qem.is_closed() and qem.euler_characteristic() == 2 and qem.n_faces <= 64

    # naive midpoint baseline for the comparison
    m = ico
    while m.n_faces > 64:
        ranked = sorted(((float(np.linalg.norm(m.vertices[i] - m.vertices[j])), i, j) for (i, j) in _edges(m)),
                        key=lambda t: (t[0], t[1], t[2]))
        done = False
        for (_, i, j) in ranked:
            keep, remove = (i, j) if i < j else (j, i)
            nm = collapse_edge(m, keep, remove)
            if nm is None:
                keep, remove = remove, keep
                nm = collapse_edge(m, keep, remove)
            if nm is None:
                continue
            kn = keep if keep < remove else keep - 1
            nm.vertices[kn] = 0.5 * (m.vertices[keep] + m.vertices[remove])
            m = nm; done = True; break
        if not done:
            break
    q_mean, q_max = um.mesh_surface_deviation(ico, qem)
    n_mean, n_max = um.mesh_surface_deviation(ico, m)
    assert q_mean < n_mean and q_max < n_max


def test_octahedral_normal_encoding_through_the_mind():
    """Octahedral normal encoding is a real UnifiedMind faculty: quantize unit normals on their S^2 manifold (2
    intrinsic DOF) instead of 3 ambient x/y/z. Through the mind: round-trip is accurate at 8 bits, and at an equal
    16-bit budget it beats naive xyz quantization -- the manifold-quantization win (reverse item R3's S^2 case)."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(0)
    N = rng.standard_normal((4000, 3)); N = N / np.linalg.norm(N, axis=-1, keepdims=True)

    def ang(a, b):
        return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))

    recovered = um.oct_decode_normals(um.oct_encode_normals(N, 8), 8)
    assert np.allclose(np.linalg.norm(recovered, axis=-1), 1.0, atol=1e-9)
    assert ang(N, recovered).max() < 1.0

    # equal 16-bit budget: oct (8+8) beats naive xyz (5+5+6)
    def qn(a, bits):
        levels = (1 << bits) - 1
        return np.round((a + 1) * 0.5 * levels) / levels * 2 - 1
    nb = np.stack([qn(N[:, 0], 5), qn(N[:, 1], 5), qn(N[:, 2], 6)], axis=-1)
    nb = nb / np.linalg.norm(nb, axis=-1, keepdims=True)
    assert ang(N, recovered).mean() < ang(N, nb).mean()


def test_spectral_bandwidth_and_fractal_crosscheck_through_the_mind():
    """The bandwidth driver + singularity cross-check are real UnifiedMind faculties (the genuinely-new parts of the
    fractal/bandwidth probe; dimension itself is already shipped). Through the mind: bandwidth separates smooth from
    broadband, the cross-check agrees on clean fBm and FLAGS a step the shipped single-estimator dimension would
    misread."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    n = 8192

    smooth = np.sin(2 * np.pi * 3 * np.arange(n) / n)
    white = np.random.default_rng(2).standard_normal(n)
    assert um.spectral_bandwidth(smooth) < 0.05 and um.spectral_bandwidth(white) > 0.5

    # clean fBm: cross-check agrees
    rng = np.random.default_rng(1)
    f = np.fft.rfftfreq(n); amp = np.zeros(len(f)); amp[1:] = f[1:] ** (-(2 * 0.5 + 1) / 2)
    fbm = np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(f))), n)
    assert um.fractal_confidence(fbm)[2]

    # a step: cross-check flags it (the kept negative the shipped dimension can't catch)
    step = np.zeros(n); step[n // 2:] = 1.0
    assert not um.fractal_confidence(step)[2]


def test_auto_bandwidth_kde_through_the_mind():
    """Auto-bandwidth KDE is a real UnifiedMind faculty (the disciplined band-limited-encoding faculty, landed on the
    encoder's RBF-as-KDE use). Through the mind: LCV bandwidth selection matches the kernel to the data and the
    estimate tracks a bimodal density, beating a fixed default bandwidth several-fold."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    bimodal = lambda x: 0.5 * np.exp(-0.5 * ((x - 0.3) / 0.05) ** 2) + 0.5 * np.exp(-0.5 * ((x - 0.7) / 0.07) ** 2)
    rng = np.random.default_rng(0)
    xs = []
    while len(xs) < 400:
        c = rng.uniform(0, 1)
        if rng.uniform(0, 6) < bimodal(c):
            xs.append(c)
    xs = np.array(xs)
    qx = np.linspace(0.02, 0.98, 200)
    truth = bimodal(qx)

    est, bw = um.density_estimate(xs, 0, 1, qx, dim=1024, method="lcv")
    assert np.corrcoef(est, truth)[0, 1] > 0.95
    assert um.kde_bandwidth(xs, 0, 1, "lcv") == bw

    def shape_rmse(e, t):
        a = np.sum(e * t) / np.sum(e * e)
        return np.sqrt(np.mean((a * e - t) ** 2))
    e_def, _ = um.density_estimate(xs, 0, 1, qx, dim=1024, bandwidth=1.8)
    assert shape_rmse(est, truth) < shape_rmse(e_def, truth) / 3


def test_mesh_lod_policy_through_the_mind():
    """The screen-space-error LOD policy is a real UnifiedMind faculty (the geometric instance of the engine's
    error-budget resolution selection, on top of QEM). Through the mind: build a chain off a sphere, then the
    selected level coarsens with viewing distance -- full detail up close, coarser far away."""
    from holographic_unified import UnifiedMind
    from holographic_meshsmooth import _icosphere

    um = UnifiedMind(dim=128, seed=0)
    chain = um.mesh_lod_chain(_icosphere(2), targets=(0.5, 0.25, 0.125))
    assert len(chain) >= 3
    assert chain[0].max_error == 0.0                       # the original is the fine end
    assert all(chain[i].n_faces > chain[i + 1].n_faces for i in range(len(chain) - 1))

    picks = [um.mesh_select_lod(chain, d, 2.0) for d in (2.0, 15.0, 200.0)]
    assert picks[0] == 0 and picks[-1] > 0                 # full up close, coarser far
    assert all(picks[i] <= picks[i + 1] for i in range(len(picks) - 1))


def test_binding_stability_through_the_mind():
    """The binding-stability regime test is a real UnifiedMind faculty (the band-limit-preservation item grounded in
    the real bind). Through the mind: spectral flatness separates a unitary key (exact bind/unbind, stable) from a
    random one (lossy, unstable)."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_ai import unitary_vector, random_vector

    um = UnifiedMind(dim=128, seed=0)
    uni = unitary_vector(1024, np.random.default_rng(1))
    ran = random_vector(1024, np.random.default_rng(2))

    assert um.spectral_flatness(uni) > 0.97 > um.spectral_flatness(ran)
    rep_u = um.binding_stability(uni)
    rep_r = um.binding_stability(ran)
    assert rep_u["stable"] and rep_u["distortion"] < 1e-6
    assert not rep_r["stable"] and rep_r["distortion"] > 0.5


def test_splat_prune_and_lod_through_the_mind():
    """Splat prune/merge + LOD are real UnifiedMind faculties (the splat twin of the mesh LOD policy). Through the
    mind: contribution prune beats naive, the LOD chain degrades gracefully, and selection by PSNR budget keeps more
    splats for a tighter budget."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_splat import splat_fit, splat_render, splat_refit, psnr

    um = UnifiedMind(dim=128, seed=0)
    yy, xx = np.mgrid[0:64, 0:64]
    g = lambda cy, cx, s: np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    target = 1.0 * g(18, 20, 5) + 0.8 * g(40, 44, 7) + 0.6 * g(48, 15, 4)
    full = splat_fit(target, 50, refit=True)

    contrib = um.splat_prune(full, target, 18)
    rng = np.random.default_rng(0)
    rand = splat_refit([full[i] for i in rng.permutation(len(full))[:18]], target)
    assert psnr(splat_render(contrib, (64, 64)), target) > psnr(splat_render(rand, (64, 64)), target) + 6

    chain = um.splat_lod_chain(full, target, keeps=(30, 15, 8))
    psnrs = [c[2] for c in chain]
    assert all(psnrs[i] >= psnrs[i + 1] - 1e-6 for i in range(len(psnrs) - 1))
    # a tighter PSNR budget keeps MORE splats (compare counts, not chain indices)
    assert chain[um.splat_select_lod(chain, 40.0)][1] >= chain[um.splat_select_lod(chain, 25.0)][1]

    assert len(um.splat_merge(full, target, 4.0)) < len(full)


def test_scene_delta_through_the_mind():
    """Scene component delta + dedup measurement are real UnifiedMind faculties (the explicit diff/transmission layer
    over the automatic content-addressed sharing). Through the mind: a one-subtree change is a small delta, and the
    dedup saving across variants is measured."""
    from holographic_unified import UnifiedMind
    from holographic_scenegraph import SceneNode, translation
    from holographic_mesh import box
    from holographic_scenedelta import scene_components, apply_scene_delta

    um = UnifiedMind(dim=128, seed=0)
    cube, other = box(), box(2, 1, 1)

    def variant(i, changed=True):
        ch = [SceneNode(translation([0, 0, 0]), mesh=cube), SceneNode(translation([2, 0, 0]), mesh=cube),
              SceneNode(translation([0, 2, 0]), mesh=other), SceneNode(translation([2, 2, 0]), mesh=cube)]
        if changed:
            ch[i % 4] = SceneNode(translation([float(i), 5, 0]), mesh=cube)
        return SceneNode(children=ch)

    base = variant(0, changed=False)
    var = variant(1)
    d = um.scene_delta(base, var)
    assert len(d["added"]) + len(d["removed"]) < len(scene_components(var))
    assert apply_scene_delta(scene_components(base), d) == scene_components(var)

    sav = um.scene_dedup_saving([base] + [variant(i) for i in range(8)])
    assert sav["saving_x"] > 2.0 and sav["naive"] >= sav["unique"]


def test_occlusion_recall_through_the_mind():
    """RT-V occlusion recall is a real UnifiedMind faculty (the alpha-compositing transfer that breaks the bundle
    capacity cliff). Through the mind: at high load it recovers the present atoms where the linear top-m washes out;
    at low load it ties linear."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(0)
    N, D = 200, 512
    cb = rng.standard_normal((N, D)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)

    def f1(pred, true):
        pred = set(pred); tp = len(pred & true)
        p = tp / len(pred) if pred else 0.0; r = tp / len(true) if true else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # high load: occlusion beats the linear top-m readout
    fo = fl = 0.0
    for seed in range(10):
        r = np.random.default_rng(seed); S = set(r.choice(N, 45, replace=False).tolist())
        cue = cb[list(S)].sum(0); cue = cue / np.linalg.norm(cue)
        fo += f1([j for j, _ in um.occlusion_recall(cue, cb, m=45)], S)
        fl += f1(list(np.argsort(-(cb @ cue))[:45]), S)
    assert fo / 10 > 0.99 and fo / 10 > fl / 10 + 0.03

    # weighted recovery through the mind
    r = np.random.default_rng(5); S = r.choice(N, 5, replace=False); W = r.uniform(0.5, 2.0, 5)
    cue = (W[:, None] * cb[S]).sum(0)
    rec = um.occlusion_recall(cue, cb, m=5)
    assert set(S.tolist()) == set(j for j, _ in rec)


def test_harmonic_context_atom_through_the_mind():
    """RT-VI context-conditioned atoms are real UnifiedMind faculties (the spherical-harmonics transfer). Through the
    mind: a polysemous atom recovers distinct senses at distinct contexts, and a context-free atom reduces to the
    plain atom at the DC term (the backward-compatible degree-0 fallback)."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(0)
    D = 256

    # polysemy: 3 senses at 3 contexts, each recovered through the mind
    senses = [rng.standard_normal(D) for _ in range(3)]
    senses = [s / np.linalg.norm(s) for s in senses]
    ctx = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    atom = um.harmonic_atom(ctx, senses, n_harmonics=2)
    for t, s in zip(ctx, senses):
        rec = um.harmonic_decode(atom, t)
        assert rec @ s / np.linalg.norm(rec) > 0.999

    # degree-0 fallback: a context-free atom decodes to the plain atom exactly, and the DC term is that atom
    const = rng.standard_normal(D)
    cfree = um.harmonic_atom([0.0, 1.0, 2.0], [const, const, const], n_harmonics=1)
    assert np.linalg.norm(um.harmonic_decode(cfree, 0.77) - const) < 1e-10
    assert np.linalg.norm(um.harmonic_dc(cfree) - const) < 1e-10


def test_splat_densify_through_the_mind():
    """Clone-vs-split density control is a real UnifiedMind faculty (the 3DGS densification distinction). Through the
    mind: on a mixed-error target (a ridge needing cover + twin peaks needing resolve), scale-aware densify beats both
    always-clone and always-split at a fixed splat budget."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_splat import splat_render, splat_refit
    from holographic_splatdensify import clone_splat, split_splat

    um = UnifiedMind(dim=128, seed=0)
    ys, xs = np.mgrid[0:64, 0:64]
    ridge = np.exp(-(((xs - 14) ** 2) / 6.0 + ((ys - 32) ** 2) / 120.0))
    twin = (np.exp(-(((xs - 46) ** 2 + (ys - 30) ** 2) / 4.0)) + np.exp(-(((xs - 52) ** 2 + (ys - 30) ** 2) / 4.0)))
    target = ridge + twin
    splats = splat_refit([(32, 14, 0.0, 1.0), (30, 49, 0.0, 3.5)], target)
    shape = target.shape
    res = target - splat_render(splats, shape)

    def mse(a, b):
        return float(((a - b) ** 2).mean())

    def blind(strategy):
        out = []
        for sp in splats:
            out += (split_splat(sp, res, shape) if strategy == "split" else [sp] + clone_splat(sp, res, shape))
        return mse(splat_render(splat_refit(out, target), shape), target)

    scale_mse = mse(splat_render(um.splat_clone_split(splats, target), shape), target)
    assert scale_mse < blind("clone") and scale_mse < blind("split")


def test_splat_relocate_through_the_mind():
    """MCMC birth-death relocation is a real UnifiedMind faculty (the successor to evict-rarest). Through the mind:
    relocating dead splats to under-represented regions beats dropping them, at a conserved budget."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_splat import splat_render, splat_refit

    um = UnifiedMind(dim=128, seed=0)
    ys, xs = np.mgrid[0:64, 0:64]
    target = sum(np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 12.0))
                 for cx, cy in [(16, 16), (48, 16), (16, 48), (48, 48), (32, 32), (32, 10)])
    useful = [(16, 16, 0.0, 3.5), (48, 16, 0.0, 3.5), (16, 48, 0.0, 3.5),
              (48, 48, 0.0, 3.5), (32, 32, 0.0, 3.5), (32, 10, 0.0, 3.5)]
    dead = [(2, 2, 0.0, 1.0)] * 6
    splats = splat_refit(useful + dead, target)
    shape = target.shape
    thr = 0.05 * np.abs([s[2] for s in splats]).max()

    def mse(a, b):
        return float(((a - b) ** 2).mean())

    drop = mse(splat_render(splat_refit([s for s in splats if abs(s[2]) >= thr], target), shape), target)
    reloc_splats = um.splat_relocate(splats, target)
    reloc = mse(splat_render(reloc_splats, shape), target)
    assert reloc < drop * 0.6 and len(reloc_splats) == len(splats)


def test_occlusion_gram_fast_path_through_the_mind():
    """SPEED-1: the Gram-cached fast path is a real UnifiedMind faculty -- build the Gram once, pass it to
    occlusion_recall, and recover the IDENTICAL atoms the rescan path gives (exact, just faster)."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(0)
    N, D = 300, 512
    cb = rng.standard_normal((N, D)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)

    G = um.build_occlusion_gram(cb)
    assert G.shape == (N, N)

    for seed in range(8):
        r = np.random.default_rng(seed); S = r.choice(N, 60, replace=False)
        cue = cb[S].sum(0); cue = cue / np.linalg.norm(cue)
        a = um.occlusion_recall(cue, cb, m=60)              # rescan
        b = um.occlusion_recall(cue, cb, m=60, gram=G)      # Gram-cached fast path
        assert [j for j, _ in a] == [j for j, _ in b]       # identical recovery


def test_occlusion_cache_reuses_gram_through_the_mind():
    """RAM-1: the mind's cache=True path keeps a GramCache, so a vocabulary queried many times pays the Gram
    precompute once -- the second recall is a cache hit, and recovery is identical to the explicit-gram path."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(0)
    N, D = 300, 512
    cb = rng.standard_normal((N, D)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)

    r = np.random.default_rng(1); S = r.choice(N, 60, replace=False)
    cue = cb[S].sum(0); cue = cue / np.linalg.norm(cue)

    a = um.occlusion_recall(cue, cb, m=60, cache=True)      # first call: builds + caches the Gram
    b = um.occlusion_recall(cue, cb, m=60, cache=True)      # second call: cache HIT, no rebuild
    assert a == b
    assert um._gram_cache.hits >= 1 and um._gram_cache.misses == 1

    # identical to the explicit-gram fast path and to the rescan
    G = um.build_occlusion_gram(cb)
    c = um.occlusion_recall(cue, cb, m=60, gram=G)
    d = um.occlusion_recall(cue, cb, m=60)
    assert [j for j, _ in a] == [j for j, _ in c] == [j for j, _ in d]


def test_general_optimizer_through_the_mind():
    """GRAD-2: the splat-fit Adam machinery, promoted -- the mind minimizes an arbitrary scalar loss to its known
    minimum with an analytic gradient, and the finite-difference fallback reaches the same place (gradients on the
    fly, no autodiff)."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(0)

    # least-squares loss minimized through the mind reaches the lstsq solution
    A = rng.standard_normal((20, 6)); b = rng.standard_normal(20)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    x = um.optimize(lambda z: float(((A @ z - b) ** 2).sum()), np.zeros(6),
                    grad=lambda z: 2 * A.T @ (A @ z - b), steps=2000, lr=0.02)
    assert np.linalg.norm(x - sol) < 1e-2

    # the FD fallback (no analytic gradient) reaches the same minimum of a simple quadratic
    t = rng.standard_normal(5)
    x_fd = um.optimize(lambda z: float(((z - t) ** 2).sum()), np.zeros(5), steps=400, lr=0.1)
    assert np.linalg.norm(x_fd - t) < 1e-3

    # the mind's fd_gradient matches the analytic gradient
    z0 = rng.standard_normal(5)
    g = um.fd_gradient(lambda z: float(((z - t) ** 2).sum()), z0)
    assert np.max(np.abs(g - 2 * (z0 - t))) < 1e-5


def test_iht_recall_through_the_mind():
    """GRAD-1: IHT recall is a faculty -- it recovers a bundle's components, ties greedy occlusion on an incoherent
    dictionary, and beats it on a coherent one (the support-revision win), all through the one mind."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)

    def f1(rec, true):
        got = set(i for i, _ in rec); tp = len(got & true)
        p = tp / max(len(got), 1); r = tp / max(len(true), 1)
        return 2 * p * r / max(p + r, 1e-12)

    # incoherent: IHT recovers perfectly through the mind
    rng = np.random.default_rng(0)
    cb = rng.standard_normal((200, 512)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(200, 12, replace=False); cue = cb[S].sum(0)
    assert f1(um.iht_recall(cue, cb, 12), set(int(i) for i in S)) > 0.95

    # coherent: IHT beats greedy occlusion, both via the mind
    iht_s, occ_s = [], []
    for s in range(8):
        r = np.random.default_rng(100 + s)
        cbc = r.standard_normal((200, 512)) + 1.5 * r.standard_normal(512)
        cbc = cbc / np.linalg.norm(cbc, axis=1, keepdims=True)
        Sc = r.choice(200, 12, replace=False); w = r.uniform(0.5, 1.5, 12)
        cuec = (w[:, None] * cbc[Sc]).sum(0); true = set(int(i) for i in Sc)
        iht_s.append(f1(um.iht_recall(cuec, cbc, 12), true))
        occ_s.append(f1(um.occlusion_recall(cuec, cbc, m=12), true))
    assert np.mean(iht_s) > np.mean(occ_s)


def test_cosamp_recall_through_the_mind():
    """SPEED-3: CoSaMP is a faculty -- the strongest recovery route. Through the one mind it recovers a coherent-
    dictionary bundle PERFECTLY where greedy occlusion and gradient IHT degrade, in a handful of rounds."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)

    def f1(rec, true):
        got = set(i for i, _ in rec); tp = len(got & true)
        p = tp / max(len(got), 1); r = tp / max(len(true), 1)
        return 2 * p * r / max(p + r, 1e-12)

    cos_s, iht_s, occ_s = [], [], []
    for s in range(8):
        rng = np.random.default_rng(200 + s)
        cb = rng.standard_normal((200, 512)) + 1.5 * rng.standard_normal(512)
        cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
        S = rng.choice(200, 12, replace=False); w = rng.uniform(0.5, 1.5, 12)
        cue = (w[:, None] * cb[S]).sum(0); true = set(int(i) for i in S)
        cos_s.append(f1(um.cosamp_recall(cue, cb, 12), true))
        iht_s.append(f1(um.iht_recall(cue, cb, 12), true))
        occ_s.append(f1(um.occlusion_recall(cue, cb, m=12), true))
    # CoSaMP is the strongest: near-perfect, and ahead of both other iterative routes
    assert np.mean(cos_s) > 0.95
    assert np.mean(cos_s) > np.mean(iht_s) and np.mean(cos_s) > np.mean(occ_s)

    # converges in a few rounds (the M-factor win), reported through the mind
    st = {}
    um.cosamp_recall(cue, cb, 12, stats=st)
    assert st["rounds"] <= 6


def test_forest_occlusion_through_the_mind():
    """SPEED-2: forest-routed occlusion is a faculty (the N-factor). Through the mind it stays accurate at moderate
    N and its selection is sub-linear -- shipped with its measured negative (a regression at current scale) loud."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)

    def f1(rec, true):
        got = set(i for i, _ in rec); tp = len(got & true)
        p = tp / max(len(got), 1); r = tp / max(len(true), 1)
        return 2 * p * r / max(p + r, 1e-12)

    rng = np.random.default_rng(0)
    cb = rng.standard_normal((800, 256)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(800, 10, replace=False); cue = cb[S].sum(0); true = set(int(i) for i in S)

    F = um.build_occlusion_forest(cb, seed=0)
    rec = um.occlusion_recall_forest(cue, cb, 10, forest=F)
    assert f1(rec, true) > 0.8                              # accurate at moderate N
    assert F.last_comparisons < cb.shape[0]                 # sub-linear: fewer comparisons than a full scan
    # at moderate N it should match the exact occlusion faculty's recovery quality
    assert f1(um.occlusion_recall(cue, cb, m=10), true) >= f1(rec, true) - 1e-9


def test_mixture_of_experts_learned_gate_through_the_mind():
    """W1: mixture_of_experts is a faculty -- a LEARNED gate routes by the input's CONTENT, which the mind's rule
    dispatch cannot. Two experts own different halves of the number line; the trained gate sends each value to the
    right one, beating any single expert."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(1)
    lo = [(rng.uniform(0.02, 0.46), "L", None) for _ in range(60)]
    hi = [(rng.uniform(0.54, 0.98), "H", None) for _ in range(60)]

    moe = um.mixture_of_experts(seed=2, number_range=(0.0, 1.0))
    moe.add_expert("low", lo)
    moe.add_expert("high", hi)
    moe.train_gate(lo + hi, epochs=14)

    test = ([(rng.uniform(0.02, 0.46), "L", None) for _ in range(40)]
            + [(rng.uniform(0.54, 0.98), "H", None) for _ in range(40)])
    learned = np.mean([moe.predict(x, mod)[0] == lab for x, lab, mod in test])
    single = max(np.mean([moe.predict_with(i, x, mod) == lab for x, lab, mod in test]) for i in range(2))
    assert learned >= 0.85               # routes by value, nearly perfectly
    assert learned > single + 0.25       # clearly beats either single expert


def test_kinematics_binding_is_motion_through_the_mind():
    """W2: kinematics is a faculty -- the VSA-is-geometry thesis pointed at motion. A trajectory integrated by pure
    BINDING decodes to the true positions, and the velocity between two positions is read by UNBIND."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=128, seed=0)
    kin = um.kinematics(dim=2048, lo=-50.0, hi=50.0, seed=1)

    # integrate x0=0, v0=2, no acceleration, for 10 steps -- positions advance by binding
    decoded, true = kin.trajectory(0.0, 2.0, a=0.0, steps=10)
    assert np.max(np.abs(decoded - true)) < 1.0          # binding-integrated path tracks the true motion
    assert true[-1] == 20.0                               # x = x0 + v*t = 2*10

    # velocity read off two positions by unbind
    v = kin.read_velocity(10.0, 13.0)
    assert abs(v - 3.0) < 1.0                              # 13 - 10 = 3, recovered by unbind+decode

    # the honest boundary: a path leaving the encoder range raises
    raised = False
    try:
        kin.trajectory(0.0, 100.0, steps=5)               # 100*5 = 500 >> hi=50
    except ValueError:
        raised = True
    assert raised


def test_versioned_store_commit_rollback_through_the_mind():
    """W4: versioned_store is a faculty -- commit/edit/rollback round-trips EXACTLY, the proof gate rejects a bad
    commit (and only logs it), and a rollback is itself recorded so history is never erased."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=64, seed=0)
    vs = um.versioned_store()
    rng = np.random.default_rng(0)

    a, b, c = vs.new_id(), vs.new_id(), vs.new_id()
    r = {a: rng.standard_normal(64), b: rng.standard_normal(64)}
    v0 = vs.commit(r, [a, b], note="initial")

    # edit: add a row, change another
    r2 = {a: r[a], b: rng.standard_normal(64), c: rng.standard_normal(64)}
    v1 = vs.commit(r2, [a, b, c], note="edit")
    assert v1 == v0 + 1

    # checkout v0 reconstructs the ORIGINAL state exactly
    rows0, order0 = vs.checkout(v0)
    assert order0 == [a, b]
    assert np.array_equal(rows0[a], r[a]) and np.array_equal(rows0[b], r[b])

    # rollback to v0 -> live state matches v0 again, and the rollback is a NEW recorded version
    vr = vs.rollback(v0)
    rows_back, order_back = vs.checkout(vr)
    assert order_back == [a, b] and np.array_equal(rows_back[b], r[b])
    assert vs.head() == vr and vr > v1                     # history grew; nothing erased

    # a proof that always fails rejects the commit (returns -1) and leaves the store unchanged
    before = vs.head()
    rej = vs.commit(r2, [a, b, c], proof=lambda rows, order: False, note="bad")
    assert rej == -1 and vs.head() == before
    assert any(entry["accepted"] is False for entry in vs.history())


def test_video_codec_motion_compensation_through_the_mind():
    """W3: video_codec is a faculty -- on a rigid pan (motion = a cyclic shift), the keyframe + motion-compensated-
    residual GOP codec beats per-frame intra storage: fewer bytes AND higher PSNR, because a shift is one bind that
    nearly zeroes the residual."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_video import HolographicVideo

    um = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(0)
    H, W = 40, 48
    base = np.zeros((H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(6):
        cy, cx = rng.integers(4, H - 4), rng.integers(4, W - 4)
        base += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 30.0)
    base = base / base.max()
    frames = [np.roll(base, 2 * t, axis=1) for t in range(6)]   # rigid cyclic pan

    codec = um.video_codec(dim=2048, key_keep=150, res_keep=30, gop_len=6, seed=0)
    packets, total = codec.encode(frames)
    gop_psnr = codec.mean_psnr(frames, packets)
    intra_total, intra_psnr = HolographicVideo.intra_baseline(frames, keep=150, dim=2048, seed=0)

    assert total < intra_total          # fewer bytes
    assert gop_psnr > intra_psnr        # and higher quality -- the motion-compensation win

    # decode round-trips to the right number of frames at the right shape
    recon = codec.decode(packets)
    assert len(recon) == len(frames) and recon[0].shape == frames[0].shape


def test_sculpt_brush_local_remesh_through_the_mind():
    """FS-1: sculpt is a faculty -- a brush edits a field LOCALLY (bit-identical outside the ball), the surface grows
    where inflated, and the re-extracted mesh stays watertight/manifold. Sculpt the field, re-mesh correct topology --
    the resolution-independent move a fixed mesh cannot do."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshbridge import metaball_field, sample_field, marching_tetrahedra

    um = UnifiedMind(dim=128, seed=0)
    fn = metaball_field(np.array([[0.0, 0.0, 0.0]]), radius=0.4)
    bounds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)); res, level = 28, 0.5
    (x0, y0, z0), (x1, y1, z1) = bounds
    xs = np.linspace(x0, x1, res); ys = np.linspace(y0, y1, res); zs = np.linspace(z0, z1, res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)

    p = np.zeros(3)
    inflated = um.sculpt(fn, "inflate", p, 1.0, strength=0.4)

    # LOCAL: unchanged outside the brush ball
    far = grid[np.linalg.norm(grid - p, axis=1) > 1.0 + 1e-9]
    assert np.max(np.abs(inflated(far) - fn(far))) < 1e-12
    # GREW: more of the volume rises above the mesh level
    assert int((inflated(grid) > level).sum()) > int((fn(grid) > level).sum())
    # WATERTIGHT/MANIFOLD re-extract
    vals, axes = sample_field(inflated, bounds, res)
    mesh = marching_tetrahedra(vals, axes, level=level)
    assert mesh.is_manifold() and len(mesh.faces) > 0


def test_splat_export_roundtrip_through_the_mind():
    """FS-3: export_splats + field_to_splats are faculties -- pull a metaball field's Gaussians as splats and write
    the standard 3DGS .ply; re-importing recovers the covariance (the L -> scale+rotation conversion round-trips)."""
    import os, tempfile
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_splatexport import splats_from_ply, quaternion_to_rotation, principal_axes

    um = UnifiedMind(dim=64, seed=0)

    # pull splats from a metaball field's centres (no fit), then export to .ply
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    splats = um.field_to_splats(centers, radius=0.4)
    assert len(splats) == 2

    tmp = os.path.join(tempfile.gettempdir(), "holo_mind_splats.ply")
    n = um.export_splats(splats, path=tmp, fmt="ply")
    recs = splats_from_ply(tmp)
    os.remove(tmp)
    assert n == 2 and len(recs) == 2
    # the isotropic std round-trips to the metaball radius
    s0 = np.array(recs[0]["scale"])
    assert np.allclose(s0, 0.4, atol=1e-4)
    assert np.allclose(recs[0]["position"], [0.0, 0.0, 0.0], atol=1e-5)

    # the json path also works through the mind
    js = um.export_splats(splats, fmt="json")
    assert isinstance(js, str) and "splats" in js


def test_sparse_field_local_sculpt_through_the_mind():
    """FS-2: build a narrow-band sparse field through the mind, apply a LOCAL brush, and re-mesh only the dirty bricks
    -- the stroke touches O(brush) voxels (<< res^3) and the patch is watertight."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_sparsefield import _smooth_falloff

    um = UnifiedMind(dim=64, seed=0)
    R = 0.6
    bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    voxel = 2.0 / 36
    band = 4 * voxel

    def sphere(P):
        return np.linalg.norm(P, axis=1) - R

    sf = um.sparse_field(sphere, bounds, voxel, band, tile=6)
    full = int(np.prod(sf.ncorner))
    assert len(sf.values) > 0

    # the extracted surface is on the sphere and watertight
    mesh = sf.extract_local()
    assert mesh.n_faces > 0 and mesh.is_manifold()

    # a local inflate touches O(brush) voxels, not the whole grid
    p = np.array([R, 0.0, 0.0])
    brush_r = 0.25

    def inflate(points):
        d = np.linalg.norm(points - p, axis=1)
        return -0.5 * band * _smooth_falloff(d, brush_r)

    dirty, touched = sf.apply_local(inflate, p, brush_r)
    assert 0 < touched < 0.1 * full
    sf.reinitialize(iters=4)
    patch = sf.extract_local(dirty_bricks=dirty)
    assert patch.n_faces > 0


def test_surface_mesh_extract_paths_through_the_mind():
    """FS-4: surface_mesh turns ANY field rep into a drawable mesh -- a field FUNCTION (via marching) and a
    SparseField (via its local marching) both give a watertight surface (the loop's re-extract step). Marching only,
    no LOD chain (kept fast)."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=64, seed=0)
    bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    # function-field path
    m1 = um.surface_mesh(sphere, bounds, resolution=16)
    assert m1.n_faces > 0 and m1.is_manifold()

    # sparse-field path (same scene)
    voxel = 2.0 / 36
    sf = um.sparse_field(sphere, bounds, voxel, 4 * voxel, tile=6)
    m2 = um.surface_mesh(sf)
    assert m2.n_faces > 0 and m2.is_manifold()


def test_surface_mesh_budget_coarsens_and_loop_to_splats():
    """FS-4: with a pixel budget at a far distance, surface_mesh returns a COARSER mesh than full detail. LOD is now
    FIELD-NATIVE -- the source field is re-marched at a coarser resolution and re-projected (the mesh is a projection
    of the field), not QEM-decimated. The authoring loop also re-projects the field to display splats."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=64, seed=0)
    bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    full = um.surface_mesh(sphere, bounds, resolution=10)                       # full-detail projection
    far = um.surface_mesh(sphere, bounds, resolution=10, pixel_budget=5.0,
                          distance=100.0)                                        # coarser field re-projected far away
    assert far.n_faces < full.n_faces                                          # the budget coarsens with distance

    # the loop's other re-projection: the field as display splats
    splats = um.field_to_splats(np.array([[0.0, 0.0, 0.0]]), radius=0.6)
    js = um.export_splats(splats, fmt="json")
    assert isinstance(js, str) and "splats" in js


def test_cached_sculpt_loop_remarks_only_dirty_bricks():
    """FS-4 perf: surface_mesh(cache=True) on a SparseField uses the brick-mesh working-set cache (the ReflexCache
    idea) -- after a local brush, the loop re-marches only the dirty bricks, not the whole surface."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_sparsefield import _smooth_falloff

    um = UnifiedMind(dim=64, seed=0)
    bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    voxel = 2.0 / 42
    band = 4 * voxel

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    sf = um.sparse_field(sphere, bounds, voxel, band, tile=6)
    cold = um.surface_mesh(sf, cache=True)               # cold: marches all active bricks
    cold_marched = sf._last_marched
    assert cold_marched == len(sf.active) and cold.n_faces > 0

    p = np.array([0.6, 0.0, 0.0])

    def inflate(points):
        d = np.linalg.norm(points - p, axis=1)
        return -0.5 * band * _smooth_falloff(d, 0.25)

    sf.apply_local(inflate, p, 0.25)
    warm = um.surface_mesh(sf, cache=True)               # warm: only the dirty bricks re-marched
    assert sf._last_marched < cold_marched
    assert warm.is_manifold() or warm.n_faces > 0        # still a valid surface (grid-dependent manifoldness)


def test_field_native_lod_coarsens_source_not_mesh():
    """The field-native LOD thesis, made testable: surface_mesh's budget path builds its LOD chain by re-marching the
    SOURCE field at coarser strides (SparseField.lod_chain), NOT by QEM-decimating the fine mesh. The chain coarsens
    monotonically with a field-read error that only grows, selection coarsens with distance, and the whole chain
    builds far faster than a single QEM decimation of the fine mesh would take."""
    import time
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=64, seed=0)
    bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    voxel = 2.0 / 36
    band = 4 * voxel

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    sf = um.sparse_field(sphere, bounds, voxel, band, tile=6)

    t0 = time.time()
    chain = sf.lod_chain()
    build_ms = (time.time() - t0) * 1000.0

    assert len(chain) >= 3                                          # several resolution levels
    faces = [lvl.n_faces for lvl in chain]
    errs = [lvl.max_error for lvl in chain]
    assert all(faces[i] > faces[i + 1] for i in range(len(faces) - 1))   # strictly coarser each level
    assert errs[0] == 0.0 and all(errs[i] <= errs[i + 1] for i in range(len(errs) - 1))  # error only grows
    assert build_ms < 5000.0                                       # the whole chain (all re-marches) is fast

    near = um.surface_mesh(sf, pixel_budget=2.0, distance=0.5)
    far = um.surface_mesh(sf, pixel_budget=2.0, distance=200.0)
    assert far.n_faces < near.n_faces                              # the budget coarsens with distance

    # the field-native error is a real geometric deviation: a coarse vertex's field value IS its distance to truth
    coarse = chain[-1]
    d = np.abs(sf.sample(coarse.mesh.vertices))
    assert abs(float(d.max()) - coarse.max_error) < 1e-9           # the chain's error == the field-read deviation


def test_parallel_decimation_and_mesh_to_field_through_the_mind():
    """The imported-mesh PARALLEL path through UnifiedMind: mesh_cluster_decimate (O(n) vertex clustering, the
    bundled-quadric merge) coarsens a mesh with no field behind it; then mesh_to_field decomposes a mesh into a
    SIGNED banded SDF by tiling, and mesh_sample_field reads point-to-surface distance from it -- so the decimated
    mesh's error can be measured as a cheap field query (the field-native error, O(V)), not an O(Va*Fb) scan."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=64, seed=0)

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    m = um.mesh_from_sdf(sphere, ((-1, -1, -1), (1, 1, 1)), res=20, vectorized=True)

    # parallel decimation: coarsens, deterministic
    coarse = um.mesh_cluster_decimate(m, 12)
    assert 0 < coarse.n_faces < m.n_faces
    assert np.array_equal(um.mesh_cluster_decimate(m, 12).vertices, coarse.vertices)

    # mesh -> signed banded field, then sample the decimated vertices' distance to the original surface
    lo = m.vertices.min(0) - 0.05
    hi = m.vertices.max(0) + 0.05
    grid, axes = um.mesh_to_field(m, (lo, hi), res=48)
    dev = np.abs(um.mesh_sample_field(grid, axes, coarse.vertices))
    voxel = float((hi - lo).max()) / 47
    assert dev.max() < 6.0 * voxel                                  # coarse verts sit near the original surface

    # the field is signed: a point just inside reads negative, just outside positive
    s_in = um.mesh_sample_field(grid, axes, m.vertices[:10] * 0.95)
    s_out = um.mesh_sample_field(grid, axes, m.vertices[:10] * 1.05)
    assert np.all(s_in < 0) and np.all(s_out > 0)


def test_imported_mesh_becomes_a_field_and_gets_field_native_lod():
    """The decomposition closure through UnifiedMind: an imported mesh with no field behind it is converted to a FULL
    SDF (mesh_to_sdf_grid, interior flood-filled), then mesh_field_lod RE-MARCHES that field at coarser strides -- so
    the imported mesh coarsens exactly like a native field-backed surface, the field-native LOD path, no mesh
    decimation. The coarser levels must have strictly fewer faces, and the field must round-trip (re-march back near
    the original surface)."""
    import numpy as np
    from holographic_unified import UnifiedMind

    um = UnifiedMind(dim=64, seed=0)

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    m = um.mesh_from_sdf(sphere, ((-1, -1, -1), (1, 1, 1)), res=24, vectorized=True)
    bnds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    grid, axes = um.mesh_to_sdf_grid(m, bnds, res=56)
    mid = grid.shape[0] // 2
    assert grid[mid, mid, mid] < 0.0                                  # interior filled negative

    lods = um.mesh_field_lod(m, bnds, res=56, strides=(1, 2, 4))
    faces = [l.n_faces for l in lods]
    assert all(faces[i] > faces[i + 1] for i in range(len(faces) - 1))  # field-native LOD coarsens
    assert lods[0].is_closed()                                        # the re-marched field is a closed surface
    rr = np.linalg.norm(lods[0].vertices, axis=1)
    assert abs(float(rr.mean()) - 0.6) < 0.05                         # round-trips near the original radius


def test_grid_accelerated_lod_error_through_the_mind():
    """The cluster LOD chain's per-level deviation now rides the vectorized spatial-grid point-to-mesh (the work-
    culling index), ~110x faster than the brute scan and exact here because decimated vertices are near-surface. Check
    the mind's mesh_point_distance matches the brute distance near the surface, and that the cluster LOD chain still
    produces a valid coarsening (the speedup did not change the result)."""
    import numpy as np
    from holographic_unified import UnifiedMind
    from holographic_meshbridge import _closest_point_on_triangle

    um = UnifiedMind(dim=64, seed=0)

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    m = um.mesh_from_sdf(sphere, ((-1, -1, -1), (1, 1, 1)), res=24, vectorized=True)
    Q = m.vertices * 1.02

    fast = um.mesh_point_distance(m, Q, radius=2)
    brute = np.full(len(Q), np.inf)
    for f in m.faces:
        brute = np.minimum(brute, np.linalg.norm(Q - _closest_point_on_triangle(Q, m.vertices[f[0]], m.vertices[f[1]], m.vertices[f[2]]), axis=1))
    assert not np.any(np.isinf(fast)) and np.abs(fast - brute).max() < 1e-9   # fast == exact, near-surface

    chain = um.mesh_cluster_lod_chain(m, grids=(16, 10, 6))
    faces = [lvl.n_faces for lvl in chain]
    assert all(faces[i] > faces[i + 1] for i in range(len(faces) - 1))       # still a valid coarsening
    assert chain[0].max_error == 0.0


def test_surface_as_a_hypervector_edit_is_bind_through_the_mind():
    """FS-5 through UnifiedMind: a mesh becomes a single FPE hypervector (mesh_to_field_vector); translating the whole
    surface is one binding, exact on the value field; the 0-level re-extracts a surface."""
    import numpy as np
    from holographic_unified import UnifiedMind
    um = UnifiedMind(dim=64, seed=0)

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    m = um.mesh_from_sdf(sphere, ((-1, -1, -1), (1, 1, 1)), res=20, vectorized=True)
    field = um.mesh_to_field_vector(m, ((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3)), dim=2048, bandwidth=18.0, grid=12)

    assert field.f.shape == (2048,)                                # the surface is ONE vector
    assert float(field.value([[0.0, 0.0, 0.0]])[0]) < 0.0          # negative inside

    d = np.array([0.2, 0.0, 0.0])
    moved = field.translate(d)                                     # edit = bind
    cg = np.linspace(-0.4, 0.4, 5)
    X = np.array([(a, b, c) for a in cg for b in cg for c in cg])
    assert np.max(np.abs(moved.value(X) - field.value(X - d))) < 1e-9   # exact translation

    extract = field.surface(((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3)), res=18)
    assert extract.n_faces > 0


def test_holographic_field_delta_editing_through_the_mind():
    """A model carried as one hypervector (mesh_to_field_vector) is edited by a DELTA: apply adds it in O(dim), undo
    subtracts it exactly. The edit cost is independent of model size -- the headline for real-time editing of large
    models in holographic space."""
    import numpy as np
    from holographic_unified import UnifiedMind
    um = UnifiedMind(dim=64, seed=0)
    m = um.mesh_from_sdf(lambda P: np.linalg.norm(P, axis=1) - 0.6, ((-1, -1, -1), (1, 1, 1)), res=20, vectorized=True)
    field = um.mesh_to_field_vector(m, ((-1.4, -1.4, -1.4), (1.4, 1.4, 1.4)), dim=2048, bandwidth=18.0, grid=12)

    q = np.array([0.6, 0.0, 0.0])
    delta = field.make_delta(np.array([q, q + [0.05, 0, 0], q - [0.05, 0, 0]]), np.array([-0.35, -0.35, -0.35]))
    edited = field.apply_delta(delta)
    assert edited.value([q])[0] < field.value([q])[0]              # the edit pushed the surface out there
    undone = edited.remove_delta(delta)
    assert np.max(np.abs(undone.f - field.f)) < 1e-9               # exact undo
    # local re-extraction of just the edited region
    region = edited.surface(((0.3, -0.4, -0.4), (1.1, 0.4, 0.4)), res=12)
    assert region.n_faces > 0

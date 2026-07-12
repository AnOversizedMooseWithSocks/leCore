"""Rate-distortion geometry-preserving code (B5): bit-exact rANS + transform coding."""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import cosine, random_vector, bundle
from holographic.misc.holographic_ratedistortion import rans_encode, rans_decode, make_freq, geometry_preserving_code, reconstruct, bits_per_vector


def test_rans_is_bit_exact():
    # the determinism gate: random symbol streams must round-trip EXACTLY.
    rng = np.random.default_rng(0)
    for _ in range(15):
        K = int(rng.integers(2, 48))
        n = int(rng.integers(50, 1500))
        p = rng.dirichlet(np.ones(K) * rng.uniform(0.2, 3.0))
        syms = rng.choice(K, size=n, p=p)
        prec = int(rng.integers(10, 16))
        freq = make_freq(np.bincount(syms, minlength=K).astype(np.float64) + 1e-9, prec)
        enc = rans_encode(syms, freq, prec)
        assert np.array_equal(rans_decode(enc, freq, prec, n), syms)


def test_rans_approaches_entropy():
    rng = np.random.default_rng(1)
    syms = rng.choice(8, size=4000, p=[0.5, 0.2, 0.1, 0.08, 0.05, 0.04, 0.02, 0.01])
    freq = make_freq(np.bincount(syms, minlength=8).astype(np.float64), 14)
    enc = rans_encode(syms, freq, 14)
    p = np.bincount(syms, minlength=8) / len(syms)
    H = -(p * np.log2(p + 1e-12)).sum()
    assert len(enc) * 8 / len(syms) < H * 1.05        # within 5% of entropy, far under int8's 8 bits


def _low_rank_states(D=256, K=16, N=600, seed=0):
    rng = np.random.default_rng(seed)
    senses = [random_vector(D, rng) for _ in range(K)]
    X = np.zeros((N, D))
    for i in range(N):
        pick = rng.choice(K, size=int(rng.integers(3, 7)), replace=False)
        X[i] = bundle([senses[j] for j in pick])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_geometry_code_preserves_cosines_on_low_rank_state():
    X = _low_rank_states()
    code = geometry_preserving_code(X, target_cos=0.999)
    Xh = reconstruct(code)
    cos = np.mean([cosine(X[i], Xh[i]) for i in range(len(X))])
    assert cos >= 0.999


def test_geometry_code_beats_int8_on_low_rank_state():
    # the rate-distortion win: fewer bits/vector than int8 at matched fidelity, on low-rank state.
    X = _low_rank_states()
    code = geometry_preserving_code(X, target_cos=0.999)
    assert bits_per_vector(code) < 8 * X.shape[1] * 0.5     # comfortably under half of int8's 8 bits/dim


def test_kept_negative_full_rank_data():
    # KEPT NEGATIVE: random full-rank data has no low-rank structure; the code cannot beat int8 there.
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, 128))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    code = geometry_preserving_code(X, target_cos=0.99)
    # to reach even 0.99 it must keep almost full rank, so bits/vector stays high (no big win)
    assert bits_per_vector(code) > 8 * X.shape[1] * 0.4


def test_rate_distortion_report_faculty_is_honest():
    # The wired faculty (mind.rate_distortion_report) must be discoverable AND honest: it reports bits/vector against
    # the float32 baseline, and its `pays` flag distinguishes compressible from incompressible input -- the kept
    # negative (random unit vectors do NOT pay) is visible in the return value, not hidden.
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)

    # discoverability: a user's phrasing surfaces it in the top-3
    names = [c.name for c in mind.find_capability("how many bits to store a vector at fidelity")[:3]]
    assert any(n.startswith("Rate-distortion report") for n in names), names

    rng = np.random.default_rng(0)

    # low-rank vectors -> compresses, pays=True, baseline reported, fidelity held (mean)
    B = rng.normal(size=(3, 64))
    low = [(np.array([1.0, 0.4, -0.2]) + 0.05 * rng.normal(size=3)) @ B for _ in range(12)]
    r = mind.rate_distortion_report(low, target_cos=0.999)
    assert r["pays"] is True and r["ratio"] > 1.0, r
    assert r["float32_bits_per_vector"] == 64 * 32
    assert r["bits_per_vector"] < r["float32_bits_per_vector"]
    assert r["achieved_cos_mean"] >= 0.999 - 5e-4, r["achieved_cos_mean"]

    # incompressible random unit vectors -> does NOT pay; the report says so rather than dressing it as a win
    rnd = [a / np.linalg.norm(a) for a in [rng.normal(size=48) for _ in range(6)]]
    r2 = mind.rate_distortion_report(rnd, target_cos=0.999)
    assert r2["pays"] is False and r2["ratio"] <= 1.0, r2

"""Binding-stability regime test: spectral flatness predicts binding distortion (holographic_flatness).

WHY THIS MODULE EXISTS (and what the investigation found)
--------------------------------------------------------
The fractal-optics backlog asked for a "band-limit-preservation regime test" in the Trefethen transient-growth /
pseudospectra spirit: do the engine's operations preserve the spectral band-limit, or do they amplify high
frequencies as you compose them? The investigation measured all three relevant operations on the real substrate and
found the surprising-but-clean answer:

  * The LINEAR ops preserve the band-limit. bind (circular convolution), bundle (superposition), and permute
    (cyclic shift) all map a white (flat) spectrum to a white spectrum -- a single atom, a bind, a bundle, and a
    roll all sit at high-frequency-energy fraction ~0.5. No spectral concentration. (Measured; in the self-test.)
  * The CLEANUP shows NO transient growth. Perturb a stored atom with a pure HIGH-FREQUENCY perturbation and
    iterate the dense-associative (modern-Hopfield) cleanup one step at a time: the error contracts MONOTONICALLY to
    zero (in one step at usable beta) -- it never overshoots. The non-normal transient amplification Trefethen's
    lens looks for does not appear here. (Measured; in the self-test.)
  * So the real stability axis is NOT transient growth -- it is a LINEAR property of the binding KEY: its SPECTRAL
    FLATNESS. unbind(bind(x, k), k) returns x convolved with |K|^2 (the key's power spectrum), which equals x only
    when |K| = 1 at every frequency -- a UNITARY key (flatness 1.0). A random Gaussian key has a peaky spectrum
    (flatness ~0.5) and so DISTORTS, and the distortion compounds catastrophically over a chain of binds.

The engine already ships the stable regime -- `unitary_vector` mints flat-spectrum atoms, and the array store and
the assembly roles already use them "for exact unbind." What was missing, and is the genuinely-new contribution
here, is the DIAGNOSTIC: a way to MEASURE where any vector sits on that stability spectrum, and the regime test that
confirms flatness predicts distortion. This answers, for an arbitrary key, "is it safe to bind and unbind this
repeatedly?" -- which is the band-limit-preservation question, grounded in the engine's actual bind.

WHAT IT PROVIDES
  * spectral_flatness(v) -- the Wiener entropy of v's power spectrum (geometric mean / arithmetic mean), in (0, 1]:
    1.0 = perfectly flat = a unitary, distortion-free binding key; lower = peakier = more lossy as a key.
  * binding_distortion(key, seed, trials) -- the measured single-round bind/unbind distortion of a key, averaged
    over random targets: ||unbind(bind(x, key), key) - x|| (0 for unitary, ~1 for random).
  * binding_stability(v, tol) -- {'flatness', 'distortion', 'stable'}: the regime diagnostic for a key, 'stable'
    iff its single-round distortion is within tol.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * flatness: a unitary_vector reads ~1.0; a random unit vector reads ~0.5.
  * a UNITARY key gives EXACT bind/unbind -- machine-epsilon distortion, holding over a 64-deep chain; a random key
    is lossy (~1.0 distortion) and compounds.
  * flatness PREDICTS distortion: across keys interpolated from unitary toward random, falling flatness gives rising
    distortion, monotonically.
  * the linear ops preserve a white spectrum; the cleanup contracts monotonically (no transient growth) -- the two
    supporting findings above.

DETERMINISM (per ISA.md)
  Flatness is a pure FFT statistic; the distortion measurement seeds its own RNG. Same input -> identical numbers
  (asserted).

KEPT NEGATIVES (loud)
  * The stable regime itself is NOT new -- `unitary_vector` already mints flat-spectrum keys and the engine already
    uses them where exact unbind matters. This module adds the MEASUREMENT, not the regime. And unitarity is a mint
    CHOICE, not a free default: the engine's own record notes a starved-maze bootstrap that went to zero under
    unitary atoms (their flatness removes a redundancy some paths rely on). Flatness tells you the binding cost; it
    does not tell you unitary is always the right call.
  * The Trefethen transient-growth framing, taken literally, came up EMPTY here -- the cleanup does not transiently
    amplify, and the linear ops are spectrally flat-preserving. The honest result is a linear-stability story (key
    flatness), not a non-normal-dynamics one. Reported as found.
  * Flatness governs binding (convolution) specifically; it says nothing about bundle capacity or cleanup
    confusability, which are separate axes the engine measures elsewhere.
"""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, unitary_vector, random_vector


def spectral_flatness(v):
    """Wiener entropy of v's power spectrum: geometric mean / arithmetic mean, in (0, 1]. 1.0 = perfectly flat
    spectrum = a unitary, distortion-free binding key; lower = peakier = more lossy as a key."""
    p = np.abs(np.fft.rfft(np.asarray(v, float))) ** 2 + 1e-30
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def binding_distortion(key, seed=0, trials=8):
    """The measured single-round bind/unbind distortion of `key`: average over random unit targets x of
    ||unbind(bind(x, key), key) - x||. 0 for a unitary key (exact), ~1 for a random key (lossy). This is the ground
    truth that spectral_flatness predicts."""
    key = np.asarray(key, float)
    D = key.shape[0]
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(trials):
        x = rng.standard_normal(D)
        x = x / (np.linalg.norm(x) + 1e-12)
        errs.append(float(np.linalg.norm(unbind(bind(x, key), key) - x)))
    return float(np.mean(errs))


def binding_stability(v, tol=0.05, seed=0):
    """Regime diagnostic for a binding key: {'flatness', 'distortion', 'stable'}, 'stable' iff the key's single-round
    bind/unbind distortion is within `tol` (i.e. it is effectively unitary and safe to bind/unbind repeatedly)."""
    flat = spectral_flatness(v)
    dist = binding_distortion(v, seed=seed)
    return {"flatness": flat, "distortion": dist, "stable": bool(dist < tol)}


# =====================================================================================================
# Self-test -- flatness separates unitary/random and PREDICTS distortion; linear ops & cleanup are stable.
# =====================================================================================================
def _selftest():
    from holographic.agents_and_reasoning.holographic_ai import bundle, involution
    from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup

    D = 1024
    rng = np.random.default_rng(0)
    uni = unitary_vector(D, np.random.default_rng(1))
    ran = random_vector(D, np.random.default_rng(2))

    # --- flatness separates unitary from random ---
    assert spectral_flatness(uni) > 0.97, f"unitary flatness should be ~1.0, got {spectral_flatness(uni):.3f}"
    assert spectral_flatness(ran) < 0.75, f"random flatness should be well below 1, got {spectral_flatness(ran):.3f}"

    # --- a unitary key is exact, a random key is lossy; the unitary chain stays at machine epsilon ---
    assert binding_distortion(uni) < 1e-6, "unitary key must give exact bind/unbind"
    assert binding_distortion(ran) > 0.5, "random key must be lossy"
    target = random_vector(D, np.random.default_rng(3))
    s = target.copy()
    for _ in range(64):
        k = unitary_vector(D, rng)
        s = unbind(bind(s, k), k)
    assert np.linalg.norm(s - target) < 1e-9, "a chain of unitary binds must stay exact"

    # --- flatness PREDICTS distortion: interpolate unitary -> random, distortion rises as flatness falls ---
    flats, dists = [], []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        ph = rng.uniform(-np.pi, np.pi, D); ph[0] = 0.0
        for k in range(1, D // 2 + 1):
            ph[D - k] = -ph[k]
        if D % 2 == 0:
            ph[D // 2] = 0.0
        mag = (1 - alpha) * np.ones(D // 2 + 1) + alpha * np.abs(rng.standard_normal(D // 2 + 1))
        key = np.fft.irfft(mag * np.exp(1j * ph[:D // 2 + 1]), D)
        key = key / np.linalg.norm(key)
        flats.append(spectral_flatness(key))
        dists.append(binding_distortion(key))
    # flatness decreases and distortion increases monotonically across the blend
    assert all(flats[i] >= flats[i + 1] - 1e-6 for i in range(len(flats) - 1)), f"flatness should fall: {flats}"
    assert all(dists[i] <= dists[i + 1] + 1e-6 for i in range(len(dists) - 1)), f"distortion should rise: {dists}"

    # --- supporting finding 1: the linear ops preserve a white spectrum (hf-energy fraction ~0.5) ---
    def hf_frac(x):
        s = np.abs(np.fft.rfft(x)) ** 2
        return float(s[len(s) // 2:].sum() / s.sum())
    a, b = random_vector(D, np.random.default_rng(4)), random_vector(D, np.random.default_rng(5))
    for x in (bind(a, b), bundle([random_vector(D, np.random.default_rng(i)) for i in range(5)]), np.roll(a, 7)):
        assert 0.4 < hf_frac(x) < 0.6, f"linear op should preserve a white spectrum, hf-fraction {hf_frac(x):.3f}"

    # --- supporting finding 2: the cleanup contracts monotonically under a HF perturbation (no transient growth) ---
    cb = np.array([random_vector(D, np.random.default_rng(100 + i)) for i in range(20)])
    clean = cb[7].copy()
    F = np.fft.rfft(rng.standard_normal(D)); F[:len(F) // 2] = 0
    hf = np.fft.irfft(F, D); hf = hf / np.linalg.norm(hf)
    q = clean + 1.0 * hf; q = q / np.linalg.norm(q)
    traj = [np.linalg.norm(q - clean)]
    s = q.copy()
    for _ in range(6):
        s = dense_cleanup(s, cb, beta=15.0, steps=1); s = s / np.linalg.norm(s)
        traj.append(np.linalg.norm(s - clean))
    assert all(traj[i + 1] <= traj[i] + 1e-9 for i in range(len(traj) - 1)), f"cleanup must not overshoot: {traj}"

    # --- determinism ---
    assert spectral_flatness(uni) == spectral_flatness(uni)
    assert binding_distortion(ran) == binding_distortion(ran)

    print(f"holographic_flatness selftest: ok (flatness unitary {spectral_flatness(uni):.3f} vs random "
          f"{spectral_flatness(ran):.3f}; unitary key exact (chain-64 error < 1e-9), random key distortion "
          f"{binding_distortion(ran):.2f}; flatness PREDICTS distortion -- flats {[round(f, 2) for f in flats]} -> "
          f"dists {[round(d, 2) for d in dists]}; linear ops preserve white spectrum; cleanup contracts monotonically "
          f"(no transient growth); deterministic)")


if __name__ == "__main__":
    _selftest()

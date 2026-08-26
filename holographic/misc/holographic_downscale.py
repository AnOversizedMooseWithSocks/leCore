"""Denoise-by-downscale -- find a pattern by projecting to a coarse representation where noise averages out.

WHY THIS EXISTS (the Group G through-line, entry point)
-------------------------------------------------------
"Patterns can be found by downscaling to eliminate noise." Downscaling an image pools neighbouring pixels so
independent noise averages out while the structure survives. That is NOT an image trick -- it is a manifold
operation, and the engine already owns its forms: `consolidation` (low-rank SVD) is downscaling for CORRELATED
VECTORS (pool across samples; the shared low-rank subspace reinforces while per-coordinate noise cancels), and
low-pass filtering is downscaling for SIGNALS (keep the low-frequency / strongest spectral components; broadband
noise is discarded). This module makes "downscale to find the pattern" a first-class, data-type-agnostic faculty
and points it at NON-image data.

The honest catch: keeping the top-k components ALWAYS concentrates a little, even on pure noise (you are selecting
the largest of many random components). So "a pattern was found" cannot be read off the concentration alone -- it
is decided against a PERMUTATION NULL (shuffle the data to destroy the cross-structure but keep the noise level,
recompute the score). A score far above that null means real structure; a score at the null means nothing was
found. This is the engine's permutation-test discipline, and it makes the faculty FAIL SAFE: on pure noise it
reports nothing, it does not hallucinate a pattern.

MEASURED (see `_selftest`):
  * a rank-3 subspace INVISIBLE in any single noisy vector (per-sample subspace energy ~0.03) is recovered by
    pooling many samples (subspace overlap ~0.8), and the score sits ~60 sigma above the permutation null.
  * a sum of slow sinusoids buried under 2x noise is recovered by keeping the strongest spectral components
    (correlation ~0.9 to the clean signal), ~14 sigma above the null.
  * CONTROL: pure noise (either type) scores AT the permutation null -> found = False, nothing recovered.
"""

from collections import namedtuple

import numpy as np

PatternResult = namedtuple("PatternResult", "pattern score null_mean null_std found")


def downscale_lowrank(data, k):
    """Downscale correlated vectors to their top-k subspace (consolidation / SVD). Returns the (D x k) subspace
    basis and the full singular-value spectrum. Pooling across the samples averages out independent noise; the
    shared low-rank structure survives in the leading directions."""
    data = np.asarray(data, float)
    Xc = data - data.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T, S


def downscale_lowfreq(signal, k):
    """Downscale a signal to its k strongest spectral components (low-pass / top-k FFT). Returns the recovered
    signal and the kept frequency indices. Broadband noise is spread thinly across all frequencies and discarded;
    a coherent low-frequency pattern concentrates in a few components and survives."""
    signal = np.asarray(signal, float)
    F = np.fft.rfft(signal - signal.mean())
    mag = np.abs(F)
    keep = np.argsort(mag)[::-1][:k]
    Fk = np.zeros_like(F); Fk[keep] = F[keep]
    rec = np.fft.irfft(Fk, n=len(signal)) + signal.mean()
    return rec, keep


def _lowrank_score(data, k):
    Xc = np.asarray(data, float) - np.asarray(data, float).mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return float(np.sum(s[:k] ** 2) / (np.sum(s ** 2) + 1e-12))   # captured-variance fraction in top-k


def _lowfreq_score(signal, k):
    mag = np.abs(np.fft.rfft(np.asarray(signal, float) - np.mean(signal)))
    idx = np.argsort(mag)[::-1][:k]
    return float(np.sum(mag[idx] ** 2) / (np.sum(mag ** 2) + 1e-12))


def _shuffle_cols(data, rng):
    return np.stack([rng.permutation(col) for col in data.T], axis=1)   # destroy the shared subspace, keep marginals


def find_pattern_by_downscale(data, kind="vectors", k=3, n_null=80, seed=0):
    """Find a pattern in noisy data by downscaling, deciding 'found' against a PERMUTATION NULL so it fails safe.
    `kind='vectors'` pools correlated vectors to a top-k subspace (SVD); `kind='signal'` keeps a signal's k
    strongest spectral components (FFT). Returns a PatternResult(pattern, score, null_mean, null_std, found):
    `pattern` is the recovered subspace (vectors) or recovered signal (signal); `found` is True iff the score is
    > 4 sigma above the permutation null (real structure, not selection-bias concentration on noise)."""
    data = np.asarray(data, float)
    rng = np.random.default_rng(seed)
    if kind == "vectors":
        pattern, _ = downscale_lowrank(data, k)
        score = _lowrank_score(data, k)
        null = np.array([_lowrank_score(_shuffle_cols(data, rng), k) for _ in range(n_null)])
    elif kind == "signal":
        pattern, _ = downscale_lowfreq(data, k)
        score = _lowfreq_score(data, k)
        null = np.array([_lowfreq_score(rng.permutation(data), k) for _ in range(n_null)])
    else:
        raise ValueError("kind must be 'vectors' or 'signal'")
    nm, ns = float(null.mean()), float(null.std())
    found = bool(score > nm + 4.0 * ns)
    return PatternResult(pattern, float(score), nm, ns, found)


def _selftest():
    """CI-fast: downscale recovers a buried pattern on two NON-image data types (low-rank vectors, low-frequency
    signal), with the permutation null reporting nothing on pure noise (fail-safe)."""
    rng = np.random.default_rng(0)
    D, r = 256, 3

    # low-rank: a rank-3 subspace invisible per-sample, recovered by pooling
    B, _ = np.linalg.qr(rng.standard_normal((D, r)))
    C = rng.standard_normal((800, r)); X = C @ B.T; X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X + 0.4 * rng.standard_normal((800, D))
    persample = np.mean([np.linalg.norm(B.T @ x) ** 2 / np.linalg.norm(x) ** 2 for x in Xn[:50]])
    assert persample < 0.1                                       # the pattern is invisible in any single vector
    res = find_pattern_by_downscale(Xn, kind="vectors", k=r, n_null=50, seed=1)
    overlap = float(np.sum((res.pattern.T @ B) ** 2) / r)
    assert res.found and overlap > 0.5, (res.found, overlap)    # recovered the real subspace, flagged as found
    noise_res = find_pattern_by_downscale(rng.standard_normal((800, D)), kind="vectors", k=r, n_null=50, seed=1)
    assert not noise_res.found                                  # pure noise -> nothing found (fail safe)

    # low-frequency signal: slow sinusoids buried under 2x noise, recovered by top-k FFT
    T = 512; t = np.arange(T)
    clean = np.sin(2 * np.pi * 3 * t / T) + 0.7 * np.sin(2 * np.pi * 7 * t / T) + 0.5 * np.cos(2 * np.pi * 5 * t / T)
    clean /= np.std(clean)
    noisy = clean + 2.0 * rng.standard_normal(T)
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
    sres = find_pattern_by_downscale(noisy, kind="signal", k=6, n_null=120, seed=1)
    assert sres.found and corr(sres.pattern, clean) > 0.7, (sres.found, corr(sres.pattern, clean))
    snoise = find_pattern_by_downscale(rng.standard_normal(T), kind="signal", k=6, n_null=120, seed=1)
    assert not snoise.found                                     # pure noise -> nothing found (fail safe)


if __name__ == "__main__":
    _selftest()
    print("holographic_downscale selftest passed")

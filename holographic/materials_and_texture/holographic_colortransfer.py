"""holographic_colortransfer.py -- COLOUR TRANSFER: grade toward a reference image's statistics (ST1).

Match a reference image's COLOUR STATISTICS onto another image -- the "make this render feel like that sunset
photo" knob (Reinhard et al. 2001). Pure statistics, no learned weights, a few lines. It moves COLOUR, not
content, and is exactly the mood-match IR4's analysis-by-synthesis loop wants.

Two forms, from simplest to fullest:

  * 'meanstd'    -- match each channel's MEAN and STD independently: (x - mean_x)*(std_ref/std_x) + mean_ref. The
                    simplest Reinhard form. It ignores cross-channel correlation, which is why the classic paper
                    first rotates into the decorrelated l-alpha-beta space; here the covariance mode below does
                    that decorrelation implicitly, so meanstd is offered as the cheap option.
  * 'covariance' -- match the full MEAN and 3x3 COVARIANCE by WHITENING the source and COLOURING by the reference:
                    whiten z = Cx^(-1/2) (x - mean_x)  (now identity covariance), then colour y = Cr^(1/2) z +
                    mean_ref (now the reference's mean and covariance). This is the Monge-Kantorovich linear
                    colour transfer -- it handles colour correlations (a teal-orange grade, say), where plain
                    per-channel matching cannot. The matrix square roots are done by eigendecomposition of the
                    (symmetric, positive-definite) covariances -- readable and exact.

`strength` in [0, 1] blends from the original (0) to the full transfer (1).

KEPT NEGATIVE (loud): GLOBAL statistics only. It moves colour, not content, and can WASH OUT when the two palettes
are very different (a linear map can't turn a green field into a red desert without destroying detail). Local /
histogram-matching variants fix this at more cost. NumPy + stdlib only; deterministic.
"""
import numpy as np


def _sym_sqrt(C, inverse=False):
    """The (inverse) square root of a symmetric positive-definite matrix, via its eigendecomposition:
    C = V diag(w) V^T, so C^(1/2) = V diag(sqrt(w)) V^T and C^(-1/2) = V diag(1/sqrt(w)) V^T. Clamps the
    eigenvalues to stay positive-definite on a near-degenerate (e.g. grayscale) covariance."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    d = 1.0 / np.sqrt(w) if inverse else np.sqrt(w)
    return (V * d) @ V.T


# Canonical mode name <- the spellings a caller might reasonably type. WHY this map exists: `mode` used to be a
# bare `if mode == "meanstd": ... else: covariance`, so ANY other string (a misspelling like "mean_std", a dead
# option) silently fell through to covariance -- a parameter that looked live but did nothing. We canonicalise the
# common spellings and REJECT the unknown, so a typo fails loudly at the call instead of shipping as a no-op.
_MODE_ALIASES = {
    "meanstd": "meanstd", "mean_std": "meanstd", "mean-std": "meanstd", "mean std": "meanstd",
    "covariance": "covariance", "cov": "covariance", "monge-kantorovich": "covariance", "mk": "covariance",
}


def color_transfer(img, reference, mode="covariance", strength=1.0, clip=True):
    """Grade `img` toward `reference`'s colour statistics. `mode` is 'meanstd' (per-channel mean+std) or
    'covariance' (full mean+covariance, whitening/colouring); 'mean_std'/'mean-std'/'cov'/'mk' are accepted
    aliases. An UNKNOWN mode raises ValueError with the valid list (it used to be silently ignored). `strength`
    blends 0->original, 1->full transfer. Returns an image the same shape as `img`."""
    key = str(mode).strip().lower()
    if key not in _MODE_ALIASES:                               # loud failure beats a silently dead parameter
        valid = sorted(set(_MODE_ALIASES.values()))
        raise ValueError(f"color_transfer: unknown mode {mode!r}; valid modes are {valid} "
                         f"(aliases: {sorted(_MODE_ALIASES)})")
    mode = _MODE_ALIASES[key]                                  # from here on `mode` is the canonical name
    img = np.asarray(img, float)
    ref = np.asarray(reference, float)
    shape = img.shape
    X = img.reshape(-1, shape[-1])
    R = ref.reshape(-1, ref.shape[-1])
    mu_x = X.mean(axis=0)
    mu_r = R.mean(axis=0)

    if mode == "meanstd":
        sd_x = X.std(axis=0) + 1e-8
        sd_r = R.std(axis=0)
        Y = (X - mu_x) * (sd_r / sd_x) + mu_r
    else:                                                    # full mean + covariance
        Cx = np.cov(X.T) + 1e-6 * np.eye(shape[-1])         # source covariance
        Cr = np.cov(R.T) + 1e-6 * np.eye(ref.shape[-1])     # reference covariance
        T = _sym_sqrt(Cr) @ _sym_sqrt(Cx, inverse=True)     # colour(ref) . whiten(src)
        Y = (X - mu_x) @ T.T + mu_r

    if strength != 1.0:
        Y = (1.0 - strength) * X + strength * Y             # blend original -> transferred
    if clip:
        Y = np.clip(Y, 0.0, 1.0)
    return Y.reshape(shape)


def _selftest():
    """Covariance transfer makes the output's mean AND covariance match the reference; meanstd matches per-channel
    mean/std; strength=0 is the identity, strength=1 is the full match; shape preserved; deterministic."""
    rng = np.random.default_rng(0)
    # a bluish source and a warm reference, each with cross-channel correlation
    src = np.clip(0.4 + 0.15 * rng.standard_normal((64, 64, 3)) @ np.array([[1, .3, 0], [.3, 1, .2], [0, .2, 1]]),
                  0, 1)
    ref = np.clip(0.6 + 0.20 * rng.standard_normal((50, 70, 3)) @ np.array([[1, .1, .4], [.1, 1, .1], [.4, .1, 1]]),
                  0, 1)

    # (1) COVARIANCE mode: the graded image's mean and covariance match the reference's (before clipping bias)
    out = color_transfer(src, ref, mode="covariance", strength=1.0, clip=False)
    Xo = out.reshape(-1, 3); Rr = ref.reshape(-1, 3)
    assert np.allclose(Xo.mean(0), Rr.mean(0), atol=1e-6)                      # mean matched
    assert np.allclose(np.cov(Xo.T), np.cov(Rr.T), atol=1e-3)                 # full covariance matched

    # (2) MEANSTD mode: per-channel mean and std match
    out2 = color_transfer(src, ref, mode="meanstd", strength=1.0, clip=False)
    Xo2 = out2.reshape(-1, 3)
    assert np.allclose(Xo2.mean(0), Rr.mean(0), atol=1e-6)
    assert np.allclose(Xo2.std(0), Rr.std(0), atol=1e-6)

    # (3) strength=0 is a no-op; strength=1 is the full transfer; 0.5 sits between
    assert np.allclose(color_transfer(src, ref, strength=0.0, clip=False), src)
    half = color_transfer(src, ref, strength=0.5, clip=False)
    full = color_transfer(src, ref, strength=1.0, clip=False)
    assert np.allclose(half, 0.5 * src + 0.5 * full)

    # (4) shape preserved + clipping keeps it in range
    assert color_transfer(src, ref).shape == src.shape
    assert color_transfer(src, ref).min() >= 0.0 and color_transfer(src, ref).max() <= 1.0

    # (5) deterministic
    assert np.array_equal(color_transfer(src, ref), color_transfer(src, ref))

    # (6) ITEM 3 -- mode validation + aliases. The alias 'mean_std' must produce the SAME result as 'meanstd'
    #     (the misspelling that silently shipped as a dead parameter now resolves correctly), and an unknown mode
    #     must RAISE with the valid list rather than falling through to covariance.
    assert np.array_equal(color_transfer(src, ref, mode="mean_std", clip=False),
                          color_transfer(src, ref, mode="meanstd", clip=False))
    assert np.array_equal(color_transfer(src, ref, mode="MeanStd", clip=False),   # case-insensitive
                          color_transfer(src, ref, mode="meanstd", clip=False))
    try:
        color_transfer(src, ref, mode="not_a_mode"); raised = False
    except ValueError:
        raised = True
    assert raised, "unknown mode must raise ValueError, not silently fall through to covariance"
    # KEPT NEGATIVE: before this, mode='mean_std' silently DID NOTHING (fell through to covariance) -- a live-looking
    # parameter with no effect. Loud rejection of the unknown is the fix; the two real modes are unchanged.

    print("holographic_colortransfer selftest OK: covariance transfer matches the reference's MEAN and full 3x3 "
          "COVARIANCE (a teal-orange grade a per-channel match can't do); meanstd matches per-channel mean/std; "
          "strength blends 0->original .. 1->full; shape preserved; clips to range; deterministic")


if __name__ == "__main__":
    _selftest()

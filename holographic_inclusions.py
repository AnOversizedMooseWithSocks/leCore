"""holographic_inclusions.py -- M3: IMPURITIES & INCLUSIONS as a material socket (bubbles in glass, carbon in
steel, veins in stone).

WHY THIS EXISTS (Material Structure backlog, item M3)
-----------------------------------------------------
The planet already scatters ore/mineral pockets into a host layer with a noise-threshold "blob" model
(holographic_matlib._DepositSDF: a point is in a pocket where a noise field exceeds a threshold). This module
lifts that same pattern out of the planet and scopes it to a MATERIAL: given a base material and a list of
inclusions, it returns an albedo socket f(points (M,3)) -> (M,3) rgb that shades the base everywhere EXCEPT in
noise-blob pockets, where the inclusion's colour shows through. That is exactly what a metallurgist's carbon
speckle, a glassblower's bubbles, or a stone's mineral veins look like.

THE ONE IDEA THAT MAKES IT HONEST: coverage is CALIBRATED, not hoped for
------------------------------------------------------------------------
A raw "noise > 0.5" threshold gives you *some* pockets but not a KNOWN fraction. The backlog's measurement bar
is "seed a base with a target inclusion fraction; the measured covered fraction matches within tolerance." So
for each inclusion we pick the threshold as the (1 - fraction) empirical QUANTILE of the noise over a fixed
calibration sample -- then, by construction, about `fraction` of space lands above it. Deterministic (the noise
is holographic_pattern.value_noise's integer-lattice hash, and the calibration sample uses a fixed seed).

The socket is volumetric (a solid texture over 3-D object space), so cutting the material shows the inclusions
continue through the body -- the same field-native property the wood grain and planet layers have.

HONEST SCOPE (kept negative): spatial + statistical impurities -- a believable, coverage-controlled speckle --
NOT a metallurgical solidification / phase-diagram model. The alloy fraction is a placement statistic here, not
a thermodynamic composition; a real `composition` data column on the material definitions is a separate,
additive step (backlog phase 2). NumPy + stdlib only; deterministic.
"""
import numpy as np
from holographic_pattern import value_noise


def _as_rgb(color):
    """A preset NAME (looked up in the matlib catalog) or an rgb triple -> an (3,) float array."""
    if isinstance(color, str):
        import holographic_matlib as _ml
        return _ml.albedo(color)
    return np.asarray(color, float)


def _calibrated_threshold(noise_fn, fraction, calib_n=4000, seed=0):
    """Pick the noise threshold so that ~`fraction` of space exceeds it -- the (1-fraction) quantile of the
    noise over a fixed calibration sample. This is what turns 'some pockets' into 'a known covered fraction',
    which is the module's measurement bar. Deterministic: the calibration points use a fixed seed."""
    frac = float(np.clip(fraction, 0.0, 1.0))
    if frac <= 0.0:
        return np.inf                                            # never exceeded -> no inclusion
    if frac >= 1.0:
        return -np.inf                                           # always exceeded -> full coverage
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-2.0, 2.0, (int(calib_n), 3))             # a neutral cube of calibration points
    vals = np.asarray(noise_fn(pts), float)
    # the value that (1-frac) of samples fall below == the value that `frac` of samples exceed
    return float(np.quantile(vals, 1.0 - frac))


def with_inclusions(base, inclusions, seed=0, calib_n=4000):
    """A material albedo socket f(points (M,3)) -> (M,3) rgb: the `base` colour everywhere EXCEPT inside
    noise-blob pockets, where an inclusion's colour shows. `inclusions` is a list of (material, fraction, scale):
      * material -- a matlib preset name or an rgb triple (the pocket colour);
      * fraction -- the target covered fraction in [0,1] (calibrated, see above);
      * scale    -- noise frequency (bigger = smaller, denser pockets).
    Inclusions are applied in list order, so a later one paints over an earlier one where they overlap.
    Deterministic; volumetric (a solid texture, so a cut face shows the inclusions inside)."""
    base_rgb = _as_rgb(base)
    prepared = []                                               # (noise_fn, threshold, colour) per inclusion
    for i, (material, fraction, scale) in enumerate(inclusions):
        nf = value_noise(scale=float(scale), seed=seed + 101 * (i + 1))   # its own independent noise field
        th = _calibrated_threshold(nf, fraction, calib_n=calib_n, seed=seed + 7 * (i + 1))
        prepared.append((nf, th, _as_rgb(material)))

    def _socket(points):
        P = np.atleast_2d(np.asarray(points, float))
        out = np.repeat(base_rgb[None, :], len(P), axis=0).astype(float)
        for nf, th, col in prepared:
            inside = np.asarray(nf(P), float) > th             # in this inclusion's pockets?
            out[inside] = col                                  # later inclusions override earlier (list order)
        return out
    return _socket


def inclusion_coverage(inclusion, seed=0, calib_n=4000, test_n=8000, test_seed=999):
    """Measured covered fraction for ONE (material, fraction, scale) inclusion, over an independent test sample
    -- the honest check that the calibration hit its target. Returns the measured fraction in [0,1]."""
    material, fraction, scale = inclusion
    nf = value_noise(scale=float(scale), seed=seed + 101)
    th = _calibrated_threshold(nf, fraction, calib_n=calib_n, seed=seed + 7)
    rng = np.random.default_rng(test_seed)
    pts = rng.uniform(-2.0, 2.0, (int(test_n), 3))             # a DIFFERENT sample than calibration used
    return float((np.asarray(nf(pts), float) > th).mean())


def _selftest():
    """Coverage lands near the target, the socket returns valid rgb, base shows where there are no pockets, and
    everything is deterministic."""
    # (1) calibrated coverage matches the requested fraction within tolerance (the module's bar)
    for frac in (0.1, 0.25, 0.4):
        got = inclusion_coverage(("gold_ore", frac, 6.0))
        assert abs(got - frac) < 0.05, (frac, got)

    # (2) the socket paints base + inclusions, all valid colours, and is deterministic
    sock = with_inclusions("steel", [("coal", 0.15, 7.0)], seed=1)   # carbon speckle in steel
    P = np.random.default_rng(0).uniform(-2, 2, (2000, 3))
    a = sock(P); b = sock(P)
    assert a.shape == (2000, 3) and a.min() >= 0 and a.max() <= 1
    assert np.array_equal(a, b)                                 # deterministic

    # (3) some points are base, some are the inclusion (it isn't all-or-nothing)
    import holographic_matlib as _ml
    base = _ml.albedo("steel"); incl = _ml.albedo("coal")
    is_incl = np.all(np.abs(a - incl) < 1e-9, axis=1)
    is_base = np.all(np.abs(a - base) < 1e-9, axis=1)
    assert is_incl.sum() > 0 and is_base.sum() > 0
    assert 0.10 < is_incl.mean() < 0.20                         # ~15% coverage as requested
    print("holographic_inclusions selftest OK: calibrated coverage hits target; socket is valid rgb, "
          "deterministic, base+inclusion both present (%.0f%% inclusion)" % (100 * is_incl.mean()))


if __name__ == "__main__":
    _selftest()

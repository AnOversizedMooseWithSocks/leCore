"""Jittered sub-pixel splat accumulation -- a KEPT NEGATIVE: it does not sharpen past the refit.

WHY THIS EXISTS (ACCUM-1)
-------------------------
TAA/DLSS supersample by jittering the camera sub-pixel across frames and accumulating, so samples land at sub-pixel
positions. The splat fit here places every splat at an INTEGER grid position (the residual peak) and the joint refit
keeps positions fixed, so the natural idea was: jitter the FIT at sub-pixel offsets across passes and accumulate,
letting splats land between grid points and sharpen sub-pixel edges. The honest question the backlog posed was
whether this sharpens PAST the joint refit -- and the measured answer is NO.

THE MEASUREMENT (a continuous target with a sharp SUB-PIXEL feature, fit with K splats, scored at high resolution):
  * REFIT-ONLY on the base grid: a baseline error -- grid-aligned splats cannot sit exactly on the sub-pixel feature.
  * JITTERED accumulation (fit K/j splats on j sub-pixel-shifted grids, accumulate, joint-refit amplitudes): better
    than the base refit -- BUT only because it SAMPLES the continuous target at sub-pixel offsets (it is supersampling,
    more samples), not because of anything special about jittering.
  * THE CONTROL THAT SETTLES IT: given the SAME sub-pixel samples, fitting DIRECTLY on the finer grid (an ordinary
    refit, just at higher resolution) is STRICTLY BETTER than the jittered accumulation -- a global greedy + joint
    refit over all sub-pixel positions beats fitting each shifted grid independently and summing.
  * AND with NO new information (the shifted grids interpolated from the base grid), jittering cannot manufacture
    sub-pixel detail that the base samples never contained.

THE NEGATIVE, STATED PLAINLY: jittered sub-pixel accumulation is NOT a sharpening tool. The only lever is the
SAMPLING RESOLUTION of the target -- if you have sub-pixel samples, fit directly on them (a finer-grid refit wins);
if you do not, jittering adds nothing. Pixel-aligned placement + joint refit, at a sufficient sampling resolution, is
already the right answer. (Consistent with the earlier no-op: supersampling a band-limited Gaussian sum has nothing
to anti-alias.) So nothing is wired -- this records the experiment and the negative.
"""

import numpy as np

_SCALES = (0.7, 1.0, 1.6, 2.5)


def _target(xs):
    """A continuous field with a sharp narrow feature at a SUB-PIXEL center (between integer grid points)."""
    return np.exp(-((xs - 32.37) / 0.9) ** 2) + 0.4 * np.exp(-((xs - 20.6) / 1.1) ** 2)


def _fit_on_grid(grid_x, K, scales=_SCALES):
    """Greedy matching pursuit: K splats, each placed at the residual peak (a position on `grid_x`), best scale, amp."""
    y = _target(grid_x); R = y.copy(); splats = []
    for _ in range(K):
        i = int(np.argmax(np.abs(R)))
        best = None
        for s in scales:
            g = np.exp(-((grid_x - grid_x[i]) / s) ** 2); a = float(g @ R / (g @ g + 1e-9))
            err = float(np.sum((R - a * g) ** 2))
            if best is None or err < best[0]:
                best = (err, s, a)
        _, s, a = best
        splats.append((float(grid_x[i]), s, a)); R = R - a * np.exp(-((grid_x - grid_x[i]) / s) ** 2)
    return splats


def _refit_amps(splats, grid_x):
    """Joint least-squares amplitude refit, positions and scales fixed (the splat_refit move)."""
    cols = np.stack([np.exp(-((grid_x - c) / s) ** 2) for c, s, _ in splats], axis=1)
    a, *_ = np.linalg.lstsq(cols, _target(grid_x), rcond=None)
    return [(splats[i][0], splats[i][1], float(a[i])) for i in range(len(splats))]


def _render(splats, xs):
    return np.sum([a * np.exp(-((xs - c) / s) ** 2) for c, s, a in splats], axis=0)


def jittered_accumulate(base, K, j, scales=_SCALES):
    """Fit K/j splats on each of j sub-pixel-shifted grids and accumulate, then joint-refit all amplitudes against
    the base-grid target -- the ACCUM-1 method. Returns the accumulated splat list."""
    per = K // j
    allsp = []
    for t in range(j):
        allsp += _fit_on_grid(base + t / j, per, scales)
    cols = np.stack([np.exp(-((base - c) / s) ** 2) for c, s, _ in allsp], axis=1)
    amp, *_ = np.linalg.lstsq(cols, _target(base), rcond=None)
    return [(allsp[i][0], allsp[i][1], float(amp[i])) for i in range(len(allsp))]


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _selftest():
    """CI-fast, recording the KEPT NEGATIVE: jittered sub-pixel accumulation beats the base refit only by sampling
    more (supersampling), and a finer-grid refit given the SAME sub-pixel samples is STRICTLY BETTER -- so jittering
    does not sharpen past the refit; the lever is sampling resolution."""
    N = 64
    base = np.arange(N).astype(float)
    hi = np.linspace(0, N - 1, 2048); truth = _target(hi)
    K, j = 8, 4

    refit_base = _refit_amps(_fit_on_grid(base, K), base)
    jit = jittered_accumulate(base, K, j)
    refit_fine = _refit_amps(_fit_on_grid(np.linspace(0, N - 1, N * 4), K), np.linspace(0, N - 1, N * 4))

    e_base = _rmse(_render(refit_base, hi), truth)
    e_jit = _rmse(_render(jit, hi), truth)
    e_fine = _rmse(_render(refit_fine, hi), truth)

    assert e_jit < e_base                                        # jittered DOES use the extra samples (supersampling)
    assert e_fine < e_jit                                        # but a finer-grid refit beats it -- jittering is not the win


if __name__ == "__main__":
    _selftest()
    print("holographic_jittersplat selftest passed (kept negative recorded)")

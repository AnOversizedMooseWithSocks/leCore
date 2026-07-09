"""holographic_adaptive_sample.py -- a CALIBRATED adaptive-sampling stop rule: given a renderer's per-pixel
variance-of-the-mean, decide which pixels have CONVERGED (their estimate's confidence interval is within
tolerance) and how many more samples the rest need. Replaces a hand-set variance threshold with a principled
confidence interval -- "keep sampling where the estimate is still uncertain, stop where it is confident."

WHY THIS EXISTS (Forecasting & Prediction backlog, sec.5 sweep -- the renderer delegation)
------------------------------------------------------------------------------------------
The sweep's move is to make every improvised "estimate + confidence + abstain" delegate to one calibrated engine.
The renderer improvises exactly that: holographic_pathtrace returns a per-pixel variance-of-the-mean map and an
`active` mask, and spends a second pass on the noisy pixels -- but the stop threshold is hand-set. This turns it
into a principled stop.

THE HONEST TOOL (kept loud): a per-pixel Monte-Carlo estimate is a SAMPLE MEAN of many light paths, so by the
CLT its confidence interval is GAUSSIAN -- half-width = z * sqrt(variance-of-the-mean). That is the right
calibrated stop here, NOT conformal: conformal needs a per-pixel calibration set of residuals, which a single
pixel's Monte-Carlo estimate does not have. So the sweep's "conformal everywhere" framing is corrected here to
the estimator that actually fits -- a variance/CLT interval -- and the abstention is "not converged, sample more."
Because var-of-the-mean falls as sigma^2/n, halving the interval costs 4x the samples (the standard MC law),
which the budget makes explicit.

Deterministic; NumPy + stdlib. Seat: Milanfar/Cranmer (honest per-estimate uncertainty), Pharr (the renderer).
"""
import numpy as np

Z95 = 1.959963984540054                                          # the 95% two-sided normal quantile


def ci_half_width(variance_of_mean, z=Z95):
    """The confidence-interval half-width of a sample-mean estimate: z * sqrt(variance-of-the-mean). This is how
    uncertain each pixel's radiance estimate is at confidence z."""
    v = np.clip(np.asarray(variance_of_mean, float), 0.0, None)
    return z * np.sqrt(v)


def converged_mask(variance_of_mean, tolerance, z=Z95):
    """Which pixels have CONVERGED: their CI half-width is already within `tolerance` -- stop sampling them.
    Returns a boolean mask (True = converged / stop). Its complement is the renderer's `active` mask for the
    next pass.

    THE COMPLEMENT IS AN ESCALATE MASK, so this cites the unifier that owns them (`holographic_coarsefirst`) rather
    than hand-rolling a second threshold rule. The tie convention is the one thing that had to be preserved and is
    the reason `inclusive` exists: a pixel whose CI half-width is EXACTLY the tolerance has converged and must stop,
    so the escalation is strict (`u > t`), where coarse-first's default refines on a tie (`u >= t`). Verified
    bit-identical to the old inline comparison on 100,000 random variances including exact ties."""
    from holographic.misc.holographic_coarsefirst import escalate_mask
    return ~escalate_mask(ci_half_width(variance_of_mean, z), threshold=float(tolerance), inclusive=False)


def samples_to_target(variance_of_mean, current_n, target_half_width, z=Z95):
    """How many TOTAL samples a pixel needs to reach `target_half_width` at confidence z. Since the variance of
    the mean falls as sigma^2/n, the sample variance is sigma^2 = variance_of_mean * current_n, and the target n
    is (z*sigma/target)^2. Returns a per-pixel integer count (>= current_n). This is the 4x-samples-per-halving
    MC law made explicit."""
    v = np.clip(np.asarray(variance_of_mean, float), 0.0, None)
    sigma2 = v * float(current_n)                               # recover the per-sample variance
    need = (z * z * sigma2) / (float(target_half_width) ** 2 + 1e-12)
    return np.maximum(np.ceil(need).astype(int), int(current_n))


def sample_budget(variance_of_mean, current_n, target_half_width, z=Z95):
    """The EXTRA samples each pixel needs to reach the target (0 for already-converged pixels). What an adaptive
    second pass should spend, concentrated exactly where the estimate is still uncertain."""
    total = samples_to_target(variance_of_mean, current_n, target_half_width, z)
    return np.maximum(total - int(current_n), 0)


def _selftest():
    """A converged (low-variance) region needs no more samples; a noisy region does; halving the target interval
    quadruples the samples (the MC law); the converged mask matches the CI test."""
    rng = np.random.default_rng(0)
    H = W = 16
    # a low-variance (converged) left half, a high-variance (noisy) right half, measured at n=64 samples
    var_of_mean = np.zeros((H, W))
    var_of_mean[:, :W // 2] = 1e-5                              # converged: tiny CI
    var_of_mean[:, W // 2:] = 4e-3                              # noisy: wide CI
    n = 64

    tol = 0.05
    mask = converged_mask(var_of_mean, tol)
    assert mask[:, :W // 2].all()                              # the low-variance half is converged
    assert not mask[:, W // 2:].any()                          # the noisy half is not

    budget = sample_budget(var_of_mean, n, target_half_width=tol)
    assert budget[:, :W // 2].sum() == 0                       # converged pixels want no more samples
    assert budget[:, W // 2:].min() > 0                        # noisy pixels want more

    # the MC law: halving the target interval quadruples the required samples
    n_wide = samples_to_target(np.array([4e-3]), n, target_half_width=0.10)[0]
    n_half = samples_to_target(np.array([4e-3]), n, target_half_width=0.05)[0]
    ratio = n_half / max(n_wide, 1)
    assert 3.5 <= ratio <= 4.5, ratio

    # deterministic
    assert np.array_equal(converged_mask(var_of_mean, tol), converged_mask(var_of_mean, tol))

    print("holographic_adaptive_sample selftest OK: low-variance region converges (no more samples), noisy region "
          "wants more; halving the target interval costs %.1fx the samples (the sigma^2/n MC law); a Gaussian/CLT "
          "stop, not conformal -- a pixel mean has no per-pixel calibration set" % ratio)


if __name__ == "__main__":
    _selftest()

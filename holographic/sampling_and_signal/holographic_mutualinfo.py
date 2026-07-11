"""holographic_mutualinfo.py -- mutual information between two signals (L11), with the shuffle-null discipline.

WHY THIS MODULE EXISTS
----------------------
The ladder's pipeline-assembly validation gate (§3i / L12) needs to ask "does this driven output carry
measurably more structure from the REAL input than from a time-shuffled one" -- otherwise ANY random projection
"works" and we have discovered nothing. That question is mutual information, and MI was used internally (block-
model context widths) but never surfaced as a discoverable faculty. This wires it as a first-class citizen.

THE MEASURE (classical, NumPy only)
  I(X;Y) = sum_xy p(x,y) log2( p(x,y) / (p(x) p(y)) ) -- bits of information X and Y share. Zero iff independent.
  Estimated from a 2-D histogram of paired samples. Continuous signals are binned; discrete ones use their
  symbols directly.

THE HONEST PART (the whole reason it exists here)
  MI is BIASED UPWARD by finite samples and fine bins: even two INDEPENDENT signals show apparent MI just from
  histogram noise (the plan's 'high-D noise has basins too', again). So the useful number is not raw MI but MI
  ABOVE A SHUFFLE NULL: shuffle one signal to destroy any real dependence, measure the residual apparent MI, and
  report how far the real MI exceeds it (in null standard deviations -- a z-score). A dependence counts only if it
  clears the null. Raw MI without its null is a Rorschach test.

NumPy only. Deterministic (seeded shuffles).
"""

import numpy as np


def _discretize(x, bins):
    """Map a signal to integer bin ids. Already-integer/small-cardinality signals are used as-is; float signals
    are quantile-binned (equal-occupancy) so the histogram is not dominated by empty bins."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer) or len(np.unique(x)) <= bins:
        # discrete: map distinct values to 0..k-1
        vals, inv = np.unique(x, return_inverse=True)
        return inv, len(vals)
    # continuous: equal-occupancy (quantile) bins -- robust to skew, unlike equal-width.
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    ids = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    return ids, bins


def mutual_information(x, y, bins=16):
    """Mutual information I(X;Y) in BITS between two equal-length signals `x`, `y` (discrete or continuous). Zero
    iff independent; higher = more shared information. Continuous signals are quantile-binned into `bins` bins.
    This is the RAW estimate -- for a significance-aware number use mutual_information_vs_null, because raw MI is
    biased upward by finite samples (even independent signals show some). Returns a float >= 0."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must be the same length; got %d and %d" % (len(x), len(y)))
    xi, kx = _discretize(x, bins)
    yi, ky = _discretize(y, bins)
    joint = np.zeros((kx, ky))
    np.add.at(joint, (xi, yi), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    # only nonzero joint cells contribute; the rest are 0*log0 = 0.
    nz = joint > 0
    denom = (px @ py)                                          # outer product p(x)p(y)
    return float(np.sum(joint[nz] * np.log2(joint[nz] / denom[nz])))


def mutual_information_vs_null(x, y, bins=16, n_shuffle=64, seed=0):
    """Mutual information ABOVE its shuffle null -- the honest dependence measure (the gate §3i needs). Computes
    raw I(X;Y), then a null distribution of MI values with `y` SHUFFLED (all real dependence destroyed, only the
    finite-sample/binning bias remains), and reports how far the real MI exceeds the null. Returns a dict:
    `mi` (raw bits), `null_mean`, `null_std`, `excess` (mi - null_mean), and `z` (excess in null std devs -- the
    significance). A dependence counts as REAL only when z clears a few sigma; raw MI without this is a Rorschach
    test (finite samples give ANY pair apparent MI). Deterministic given `seed`."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mi = mutual_information(x, y, bins=bins)
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        null[i] = mutual_information(x, y[rng.permutation(len(y))], bins=bins)
    null_mean = float(null.mean())
    null_std = float(null.std()) + 1e-12
    excess = mi - null_mean
    return {"mi": mi, "null_mean": null_mean, "null_std": null_std,
            "excess": excess, "z": excess / null_std}


def _selftest():
    """Contracts (the honest-MI properties, cross-seed):

    1. IDENTICAL signals share maximal MI (= the signal's own entropy); INDEPENDENT signals have MI ~ the null.
    2. A dependent pair (y = f(x) + noise) has POSITIVE MI well above its shuffle null (high z); an independent
       pair has z near 0 (excess consistent with noise).
    3. MI is symmetric: I(X;Y) == I(Y;X).
    4. Determinism: same inputs + seed -> same numbers.
    5. The shuffle null is NON-trivial: independent signals show apparent raw MI > 0 (the bias this guards).
    """
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)

    # (1) identical vs independent
    mi_self = mutual_information(x, x, bins=16)
    indep = rng.normal(size=n)
    mi_indep = mutual_information(x, indep, bins=16)
    assert mi_self > mi_indep * 3, (mi_self, mi_indep)         # sharing with itself >> with noise

    # (2) dependent pair clears the null; independent does not.
    y = np.sign(x) * np.abs(x) ** 0.5 + 0.3 * rng.normal(size=n)   # a real (nonlinear) dependence
    dep = mutual_information_vs_null(x, y, bins=16, n_shuffle=48, seed=1)
    ind = mutual_information_vs_null(x, indep, bins=16, n_shuffle=48, seed=1)
    assert dep["z"] > 5.0, dep                                 # real dependence is many sigma above null
    assert ind["z"] < 3.0, ind                                 # independent pair sits near the null

    # (3) symmetry
    assert abs(mutual_information(x, y, bins=16) - mutual_information(y, x, bins=16)) < 1e-9

    # (4) determinism
    a = mutual_information_vs_null(x, y, bins=16, n_shuffle=16, seed=7)
    b = mutual_information_vs_null(x, y, bins=16, n_shuffle=16, seed=7)
    assert a == b

    # (5) the null is real: independent signals have apparent raw MI > 0 (finite-sample bias this guards against)
    assert ind["null_mean"] > 0.0

    print("holographic_mutualinfo selftest OK (self MI %.2f >> indep %.3f bits; dependent pair z=%.1f clears null, "
          "independent z=%.1f does not; symmetric; deterministic; null_mean %.3f > 0 confirms the finite-sample "
          "bias this guards)" % (mi_self, mi_indep, dep["z"], ind["z"], ind["null_mean"]))


if __name__ == "__main__":
    _selftest()

"""The structure verifier, generalised beyond text: does a signal carry the
autocorrelation signature of real data, or is it noise / corruption?

The text verifier (holographic_structure) scored a word sequence by how closely its
LAG-COHERENCE PROFILE -- the similarity of each item to the item k positions back --
matched a band calibrated on real text. That idea is not specific to words. For a
CONTINUOUS sequence the same profile is the lag-AUTOCORRELATION; for an IMAGE it is
the SPATIAL autocorrelation across pixel offsets. This module carries the verifier
across those domains and reports, honestly, where it transfers cleanly and where it
does not.

WHAT WAS MEASURED (and the honest verdict per domain):

  * IMAGES -- TRANSFERS CLEANLY. Natural images have stable spatial coherence:
    neighbouring pixels correlate, with a smooth falloff over distance. Scored
    against a band calibrated on real patches, a natural patch sits near 0 (~ -0.6)
    while white noise and pixel-shuffled versions crash to ~ -14 -- the same clean
    separation the text verifier gives. Spatial coherence is as stable as the
    lag-profile of prose, so the method works as-is.

  * MARKET / RETURNS -- TRANSFERS ONLY WITH THE RIGHT SIGNATURE, AND NEEDS DATA.
    Raw returns are nearly uncorrelated (efficient-market-like), so a symmetric
    profile-deviation does NOT separate them from a shuffle -- in fact a flat
    shuffle can score HIGHER, because real volatility structure is intermittent and
    widens the band. The structure that DOES distinguish real returns is volatility
    CLUSTERING: |returns| is positively autocorrelated while a shuffle destroys
    that. The honest statistic is therefore directional -- lag-1 autocorrelation of
    |returns|, tested against a shuffled control -- not a symmetric band match. On a
    long synthetic GARCH series the clustering is unmistakable; on the short real
    DAI/WETH sample here (~100 returns) it is present but only ~1 sigma above the
    shuffle -- too little data to call, exactly the regime where honest measurement
    says 'not enough signal yet' rather than inventing one.

THE GENERAL LESSON, kept: the verifier idea -- compare a signal's autocorrelation
signature to a band of real data -- is genuinely cross-domain, but the SIGNATURE has
to match the domain's actual structure (stable spatial coherence for images;
intermittent volatility clustering for returns; an even lag-profile for text). The
machinery transfers; the choice of what to autocorrelate is the domain knowledge.

Needs: numpy.
"""
import numpy as np


def lag_autocorr_profile(x, lags=range(1, 9)):
    """The continuous analogue of the text lag-coherence profile: the
    autocorrelation of a 1-D signal at each lag. The fingerprint of temporal
    structure."""
    x = np.asarray(x, float)
    x = x - x.mean()
    v = x.var() + 1e-12
    return np.array([float(np.mean(x[k:] * x[:-k]) / v) if len(x) > k else 0.0
                     for k in lags])


def spatial_coherence_profile(image, lags=range(1, 9)):
    """The 2-D analogue: mean correlation between a pixel and the pixel k columns to
    the right. Natural images fall off smoothly; noise is flat near zero."""
    g = np.asarray(image, float)
    g = g - g.mean()
    v = g.var() + 1e-12
    return np.array([float(np.mean(g[:, k:] * g[:, :-k]) / v) if g.shape[1] > k else 0.0
                     for k in lags])


class SignalStructureVerifier:
    """Calibrate the autocorrelation profile of real data (1-D series or 2-D image
    patches), then score any sample by how closely its profile matches -- the same
    proof-of-structure the text verifier gives, for continuous signals."""

    def __init__(self, kind="series", lags=range(1, 9)):
        if kind not in ("series", "image"):
            raise ValueError("kind must be 'series' or 'image'")
        self.kind = kind
        self.lags = lags
        self.ref_mean = None
        self.ref_std = None
        self.threshold = None

    def _profile(self, sample):
        if self.kind == "series":
            return lag_autocorr_profile(sample, self.lags)
        return spatial_coherence_profile(sample, self.lags)

    def calibrate(self, real_samples, z_floor=3.0):
        """real_samples: a list of real windows (1-D arrays) or real patches (2-D
        arrays). Learns the reference band and a verdict threshold."""
        P = np.stack([self._profile(s) for s in real_samples])
        self.ref_mean = P.mean(0)
        self.ref_std = P.std(0) + 1e-9
        scores = np.array([self._raw(s) for s in real_samples])
        self.threshold = float(scores.mean() - z_floor * (scores.std() + 1e-9))
        return self

    def _raw(self, sample):
        z = np.abs((self._profile(sample) - self.ref_mean) / self.ref_std)
        return float(-z.mean())

    def structure_score(self, sample):
        """How well a sample's autocorrelation profile matches real data. 0 =
        typical, more negative = anomalous. Requires calibrate."""
        if self.ref_mean is None:
            raise RuntimeError("call calibrate(real_samples) first")
        return self._raw(sample)

    def is_structured(self, sample):
        """Verdict: does this sample carry the real-data structure signature?"""
        return self.structure_score(sample) >= self.threshold


def volatility_clustering(returns):
    """The directional signature that actually distinguishes real returns: the
    lag-1 autocorrelation of |returns| (volatility clustering). Positive for real
    series, ~0 for a shuffle. Use against a shuffled control, not a symmetric band
    (see module docstring for why)."""
    a = np.abs(np.asarray(returns, float))
    a = a - a.mean()
    return float(np.mean(a[1:] * a[:-1]) / (a.var() + 1e-12))


def clustering_zscore(returns, n_shuffle=50, seed=0):
    """How many sigma the real volatility clustering exceeds shuffled controls. >2
    is meaningful structure; near 0 means none (or too little data to tell)."""
    real = volatility_clustering(returns)
    rng = np.random.default_rng(seed)
    sh = []
    r = np.asarray(returns, float)
    for _ in range(n_shuffle):
        rr = r.copy()
        rng.shuffle(rr)
        sh.append(volatility_clustering(rr))
    return float((real - np.mean(sh)) / (np.std(sh) + 1e-9))

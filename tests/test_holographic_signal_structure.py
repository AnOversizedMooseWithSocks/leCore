"""The structure verifier generalised to continuous signals and images: the
autocorrelation-profile idea transfers cleanly to images (stable spatial
coherence) and to returns via the volatility-clustering signature (with enough
data)."""
import numpy as np

from holographic.misc.holographic_signal_structure import SignalStructureVerifier, lag_autocorr_profile, spatial_coherence_profile, volatility_clustering, clustering_zscore


def test_image_structure_separates_natural_from_noise():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96]
    nat = np.sin(xx / 12.0) + np.cos(yy / 9.0) + 0.3 * rng.standard_normal((96, 96))
    patches = [nat[i:i + 32, j:j + 32] for i in range(0, 64, 16) for j in range(0, 64, 16)]
    v = SignalStructureVerifier("image").calibrate(patches)
    assert v.structure_score(nat[64:, 64:]) > v.structure_score(rng.standard_normal((96, 96)))
    # natural patch passes the verdict; noise fails it
    assert v.is_structured(nat[64:, 64:])
    assert not v.is_structured(rng.standard_normal((96, 96)))


def test_spatial_profile_falls_off_for_natural_image():
    # a smooth image has high near-lag coherence; noise is flat near zero
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:64, 0:64]
    smooth = np.sin(xx / 10.0) + np.cos(yy / 8.0)
    noise = rng.standard_normal((64, 64))
    assert spatial_coherence_profile(smooth)[0] > spatial_coherence_profile(noise)[0]


def test_volatility_clustering_detected_in_garch():
    # synthetic GARCH returns show strong volatility clustering vs a shuffle
    rng = np.random.default_rng(0)
    n = 3000
    r = np.zeros(n); s = np.ones(n) * 0.01
    for t in range(1, n):
        s[t] = np.sqrt(1e-5 + 0.1 * r[t - 1] ** 2 + 0.85 * s[t - 1] ** 2)
        r[t] = s[t] * rng.standard_normal()
    assert clustering_zscore(r) > 2.0                 # clear structure
    # an iid series has none
    assert clustering_zscore(rng.standard_normal(3000)) < 2.0


def test_lag_autocorr_profile_shape():
    x = np.sin(np.arange(200) / 5.0)
    p = lag_autocorr_profile(x, range(1, 9))
    assert len(p) == 8
    assert p[0] > 0.5                                  # a sine is strongly autocorrelated


def test_volatility_clustering_zero_for_iid():
    rng = np.random.default_rng(3)
    assert abs(volatility_clustering(rng.standard_normal(5000))) < 0.1


def test_brain_verify_image_structure():
    # the cross-domain image verifier, wired into the mind
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96]
    nat = np.sin(xx / 12.0) + np.cos(yy / 9.0) + 0.3 * rng.standard_normal((96, 96))
    m = UnifiedMind(dim=256, seed=0)
    ext = [nat[i:i + 32, j:j + 32] for i in range(0, 64, 32) for j in range(0, 64, 32)]
    nat_score = m.verify_image_structure(nat[64:, 64:], real_patches=ext)["score"]
    noise_score = m.verify_image_structure(rng.standard_normal((96, 96)), real_patches=ext)["score"]
    assert nat_score > noise_score


def test_brain_volatility_structure():
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    n = 3000
    r = np.zeros(n); s = np.ones(n) * 0.01
    for t in range(1, n):
        s[t] = np.sqrt(1e-5 + 0.1 * r[t - 1] ** 2 + 0.85 * s[t - 1] ** 2)
        r[t] = s[t] * rng.standard_normal()
    m = UnifiedMind(dim=256, seed=0)
    assert m.volatility_structure(r)["zscore"] > 2.0

"""D2 through the mind: the envelope forecaster's coverage, its mandatory constant-band baseline, and the
composition the layer exists for -- envelope skill on a series whose DIRECTION simultaneously fails the honest
gates (null_persistence), so the two claims stay separated in one test."""
import numpy as np

import lecore


def _clustered(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.where((np.arange(n) // 250) % 2 == 0, 0.5, 2.5)
    return np.cumsum(rng.normal(size=n) * scale)


def test_envelope_holds_coverage_and_beats_the_constant_band():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = _clustered()
    e = mind.envelope_forecast(x, window=20, alpha=0.1)
    assert abs(e["coverage_holdout"] - 0.9) < 0.05
    cmp_ = mind.envelope_vs_constant(x, window=20, alpha=0.1)
    assert cmp_["sharper"] and cmp_["width_ratio"] < 0.9


def test_iid_noise_earns_no_sharpness():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(1).normal(size=4000))
    cmp_ = mind.envelope_vs_constant(x, window=20, alpha=0.1)
    assert not cmp_["sharper"]                                     # the win is the data's, not the method's


def test_scale_skill_coexists_with_directional_chance():
    """The campaign's asymmetry as one assertion: the SAME series yields a sharp, covering envelope while the
    SIGNS of its moves carry ~zero information about the next sign (F1's bits discipline on the direction
    channel directly). NOTE, learned refactoring this test: null_persistence's surrogates act on the LEVEL
    series, so ANY vol clustering separates from an iid level-shuffle -- that readout mixes scale structure
    into the direction question. The clean directional test is MI between consecutive move signs; the clean
    scale test is MI between consecutive move magnitudes. Same series, two channels, opposite answers."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = _clustered(seed=2)
    e = mind.envelope_forecast(x, window=20, alpha=0.1)
    assert mind.envelope_vs_constant(x, window=20)["sharper"]       # scale: real skill
    d = np.diff(x)
    sgn = np.sign(d)
    direction = mind.mutual_information_vs_null(sgn[:-1], sgn[1:], n_shuffle=48)
    magnitude = mind.mutual_information_vs_null(np.abs(d)[:-1], np.abs(d)[1:], n_shuffle=48)
    assert direction["z"] < 3.0 and direction["excess"] < 0.005     # signs: nothing, in bits
    assert magnitude["z"] > 10.0 and magnitude["excess"] > 0.05     # magnitudes: the clustering, in bits
    assert "ZERO directional bits" in e["directional_bits_note"]

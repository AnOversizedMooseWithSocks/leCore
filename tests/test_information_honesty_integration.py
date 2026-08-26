"""The F-group information-honesty layer through the mind: dpi_guard + holdout_auc (F2) and the bits-headline
discipline on mutual information (F1), plus the composition the campaign actually ran -- propose a feature,
check novelty, then check its target value IN BITS."""
import numpy as np

import lecore


def test_dpi_guard_separates_transform_from_novel_through_the_mind():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    F = rng.normal(size=(1500, 4))
    transform = np.tanh(0.5 * F[:, 0] + 0.3 * F[:, 1] * F[:, 2])
    novel = rng.normal(size=1500)
    r_t, r_n = mind.dpi_guard(F, transform), mind.dpi_guard(F, novel)
    assert r_t["r2_holdout"] > 0.9 and r_t["verdict"].startswith("TRANSFORM")
    assert r_n["novel_frac"] > 0.9 and r_n["verdict"].startswith("NOVEL")


def test_the_full_feature_vetting_chain_novelty_then_bits():
    """The two-step gauntlet: a feature must be (1) novel under dpi_guard AND (2) worth something against the
    target in BITS. Three candidates cover the quadrant that matters:
      - a re-dressed old feature: fails (1) even though it correlates with the target;
      - novel noise: passes (1), fails (2) -- novelty is necessary, not sufficient;
      - a genuinely new informative channel: passes both."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1)
    n = 3000
    F = rng.normal(size=(n, 3))
    hidden = rng.normal(size=n)                       # a channel the existing features do not carry
    target = np.sign(F[:, 0] + hidden)

    candidates = {
        "re_dressed": np.tanh(F[:, 0]),               # correlates with target but is a transform of F
        "novel_noise": rng.normal(size=n),
        "new_channel": hidden + 0.3 * rng.normal(size=n),
    }
    verdicts = {}
    for name, g in candidates.items():
        novelty = mind.dpi_guard(F, g)
        value = mind.mutual_information_vs_null(g, target, n_shuffle=48)
        # the gate: reproducibly novel AND carries real bits about the target (excess is the headline; the z
        # support keeps a lucky binning artifact out).
        verdicts[name] = (novelty["novel_frac"] > 0.5) and (value["excess"] > 0.02) and (value["z"] > 3.0)
    assert verdicts == {"re_dressed": False, "novel_noise": False, "new_channel": True}, verdicts


def test_bits_headline_outranks_sample_inflated_z():
    """F1's rule as a ranking check: a strong-small dependence must outrank a weak-huge one on excess bits,
    even when z says otherwise."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    xw = rng.normal(size=48000)
    weak_huge = mind.mutual_information_vs_null(xw, 0.12 * xw + rng.normal(size=48000), n_shuffle=48)
    xs = rng.normal(size=800)
    strong_small = mind.mutual_information_vs_null(xs, 0.9 * xs + 0.5 * rng.normal(size=800), n_shuffle=48)
    assert weak_huge["z"] > 20.0                                     # z calls the weak one a monster
    assert strong_small["excess"] > 20 * weak_huge["excess"]         # bits know better


def test_holdout_auc_pairs_are_inseparable_and_exact():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    h = mind.holdout_auc([1, 2, 3, 4], [0, 0, 1, 1], [4, 3, 2, 1], [0, 0, 1, 1])
    assert h["auc_train"] == 1.0 and h["auc_test"] == 0.0 and h["gap"] == 1.0
    assert mind.holdout_auc(np.zeros(40), np.arange(40) % 2, np.zeros(40), np.arange(40) % 2)["auc_train"] == 0.5

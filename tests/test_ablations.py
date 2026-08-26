"""Ablation verdicts (G2): pin the honest per-subsystem findings -- which subsystems are
VSA-load-bearing, which are uniformity, which the baseline wins. The verdict() helper is
tested hermetically; the real-data ablations are NLTK-gated and run with a few seeds for
speed (the full table is `python holographic_ablate.py`).
"""
import numpy as np
import pytest

from holographic.misc.holographic_ablate import verdict, topic_classify, language_id, segmentation, key_value_noisy, recall_index


def _nltk(*names):
    try:
        from nltk import corpus
        for n in names:
            getattr(corpus, n).fileids()
        return True
    except Exception:
        return False


def _stats(mean, lo, hi):
    return {"mean": mean, "std": 0.0, "ci": (lo, hi), "n": 5}


def test_verdict_reads_the_confidence_intervals():
    # holo clearly above -> load-bearing; baseline clearly above -> baseline wins;
    # overlapping -> uniformity. The whole judgement is "do the CIs separate?".
    assert verdict(_stats(0.83, 0.79, 0.86), _stats(0.61, 0.56, 0.66)) == "VSA load-bearing"
    assert verdict(_stats(0.82, 0.80, 0.84), _stats(1.00, 1.00, 1.00)) == "baseline wins"
    assert verdict(_stats(0.99, 0.98, 1.00), _stats(0.99, 0.99, 1.00)) == "uniformity"


def test_key_value_noisy_is_vsa_load_bearing():
    # The sharpest win: an exact dict scores 0 on perturbed keys; VSA cleanup recovers
    # the value. This needs no corpus -- it is the core algebra under approximation.
    h, b, _ = key_value_noisy(seeds=range(4))
    assert verdict(h, b) == "VSA load-bearing"
    assert b["mean"] == 0.0                              # the dict genuinely cannot do it
    assert h["mean"] > 0.6


def test_recall_forest_loses_recall_but_wins_comparisons():
    # The forest is honestly behind exact scan on recall, but reaches it sublinearly.
    h, b, _ = recall_index(seeds=range(4))
    assert verdict(h, b) == "baseline wins"              # exact scan is trivially perfect
    assert h["comparison_fraction"] < 0.6                # ...but the forest scans far less
    assert h["mean"] > 0.6                               # while still recalling most


@pytest.mark.skipif(not _nltk("reuters"), reason="NLTK reuters unavailable")
def test_topic_classify_is_vsa_load_bearing_on_reuters():
    # A real ~0.22 accuracy win over bag-of-words: superposition earns its place here.
    h, b, _ = topic_classify(seeds=range(4))
    assert verdict(h, b) == "VSA load-bearing"
    assert h["mean"] > b["mean"] + 0.1


@pytest.mark.skipif(not _nltk("udhr"), reason="NLTK udhr unavailable")
def test_language_id_is_uniformity_on_udhr():
    # The bag-of-trigrams baseline ties (or marginally beats) the holographic profiles:
    # the trigram idea works, not the VSA encoding of it.
    h, b, _ = language_id(seeds=range(4))
    assert verdict(h, b) == "uniformity"


@pytest.mark.skipif(not _nltk("brown"), reason="NLTK brown unavailable")
def test_segmentation_is_not_vsa_load_bearing_on_brown():
    # Exact count-based branching entropy ties or marginally beats the holographic
    # estimate: the entropy IDEA finds boundaries, not the VSA encoding. Either verdict
    # (uniformity, or a razor-thin exact-baseline win) makes the same point -- VSA is not
    # load-bearing here -- and the exact estimator is at least as good, as it should be.
    h, b, _ = segmentation(seeds=range(4))
    assert verdict(h, b) in ("uniformity", "baseline wins")
    assert verdict(h, b) != "VSA load-bearing"
    assert b["mean"] >= h["mean"] - 0.02                 # exact is at least as good


def test_fdr_pass_controls_the_ablation_family():
    # The FDR pass over the whole table: a paired permutation p-value per subsystem, then
    # bh_fdr across the family. A unanimous holo>base win clears; ties/losses do not; and
    # a fabricated family of mostly-null rows must not over-declare discoveries.
    import numpy as np
    from holographic.misc.holographic_ablate import _paired_perm_p, fdr_verdicts

    win = (np.array([0.80, 0.82, 0.79, 0.81, 0.83, 0.80]),
           np.array([0.70, 0.71, 0.69, 0.72, 0.70, 0.71]))
    tie = (win[1], win[1])
    assert _paired_perm_p(*win) < 0.05            # unanimous win -> small p
    assert _paired_perm_p(*tie) == 1.0            # identical arms -> p = 1
    # unequal seed counts must not crash (falls back to a two-sample permutation test)
    assert 0.0 <= _paired_perm_p(np.array([0.8, 0.82, 0.79]), np.array([0.7, 0.71])) <= 1.0

    # a synthetic table: one genuine, unanimous load-bearing win among nulls. BH-Yekutieli is
    # honestly conservative, so the win needs enough seeds for its p to clear the top-rank bar
    # of the whole family (6 seeds only reaches 1/64 -- too coarse; 10 reaches ~1e-3).
    def stat(scores):
        a = np.asarray(scores, float)
        return {"mean": float(a.mean()), "std": float(a.std()), "ci": (0, 0), "n": len(a), "scores": a}
    real = (np.array([0.80, 0.82, 0.79, 0.81, 0.83, 0.80, 0.81, 0.79, 0.82, 0.80]),
            np.array([0.70, 0.71, 0.69, 0.72, 0.70, 0.71, 0.70, 0.69, 0.71, 0.70]))
    rows = [("real", stat(real[0]), stat(real[1]), "base", "VSA load-bearing")]
    for k in range(5):
        s = 0.7 + 0.01 * np.random.default_rng(k).standard_normal(6)
        rows.append((f"null{k}", stat(s), stat(s.copy()), "base", "uniformity"))
    aug, n_lb, n_surv = fdr_verdicts(rows, alpha=0.1)
    assert n_lb == 1 and n_surv == 1               # the one real win survives, nulls don't inflate

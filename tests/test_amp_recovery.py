"""Regression traps for holographic_amp -- the fifth bundle-recovery member.

The headline here is a CROSSOVER, not a win, and the first draft of this module claimed the opposite.
So both directions are pinned: AMP must keep beating CoSaMP at heavy load on incoherent dictionaries,
AND it must keep losing badly on coherent ones. A one-sided assertion is what produced the wrong
docstring in the first place.
"""
import numpy as np
import pytest

import lecore
from holographic.sampling_and_signal.holographic_amp import (amp_recall, measure_vs_cosamp,
                                                             soft_threshold)
from holographic.sampling_and_signal.holographic_cosamp import cosamp_recall


def _codebook(rng, n, d):
    a = rng.standard_normal((n, d))
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _coherent_codebook(rng, n, d, strength):
    """Atoms pulled toward 8 shared directions -- the regime state evolution is not valid in."""
    base = rng.standard_normal((8, d))
    a = rng.standard_normal((n, d)) + strength * base[rng.integers(0, 8, size=n)]
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _f1(got, want):
    got, want = set(got), set(want)
    if not got or not want:
        return 0.0
    tp = len(got & want)
    if not tp:
        return 0.0
    p, r = tp / len(got), tp / len(want)
    return 2 * p * r / (p + r)


def test_amp_recovers_exact_support_at_light_load():
    rng = np.random.default_rng(0)
    cb = _codebook(rng, 1024, 256)
    for M in (4, 8, 16):
        true = rng.choice(1024, size=M, replace=False)
        got = [i for i, _ in amp_recall(cb[true].sum(axis=0), cb, K=M)]
        assert set(got) == set(true.tolist())


def test_amp_recovers_without_being_told_k():
    # THE CAPABILITY THAT IS ACTUALLY NEW. occlusion, IHT and CoSaMP all require the sparsity; state
    # evolution means AMP does not. The support must be TIGHT as well as inclusive -- a permissive
    # threshold returns ~150 atoms for a true 8, which is not a recovery.
    rng = np.random.default_rng(1)
    cb = _codebook(rng, 1024, 256)
    true = rng.choice(1024, size=8, replace=False)
    got = [i for i, _ in amp_recall(cb[true].sum(axis=0), cb, K=None)]
    assert set(true.tolist()) <= set(got)
    assert len(got) <= 24


def test_the_onsager_term_is_load_bearing():
    # Without the memory term AMP degenerates to a badly-tuned IHT. Pinned numerically so the one line
    # that separates the two algorithms cannot be dropped as "redundant".
    rng = np.random.default_rng(2)
    dim, n = 256, 1024
    cb = _codebook(rng, n, dim)
    true = rng.choice(n, size=32, replace=False)
    cue = cb[true].sum(axis=0)

    def no_onsager(iters=30, alpha=1.5):
        w, z = np.zeros(n), cue.copy()
        for _ in range(iters):
            tau = alpha * np.linalg.norm(z) / np.sqrt(dim)
            w = soft_threshold(w + cb @ z, tau)
            z = cue - cb.T @ w
        return [int(i) for i in np.argsort(-np.abs(w))[:32]]

    with_term = len(set(i for i, _ in amp_recall(cue, cb, K=32)) & set(true.tolist()))
    without = len(set(no_onsager()) & set(true.tolist()))
    assert with_term >= without


def test_soft_threshold_is_the_exact_shrinkage():
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    assert np.allclose(soft_threshold(x, 1.0), [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_amp_is_deterministic():
    rng = np.random.default_rng(3)
    cb = _codebook(rng, 512, 128)
    true = rng.choice(512, size=8, replace=False)
    cue = cb[true].sum(axis=0)
    assert amp_recall(cue, cb, K=8) == amp_recall(cue, cb, K=8)


def test_shape_guards_fail_loudly():
    rng = np.random.default_rng(4)
    cb = _codebook(rng, 64, 32)
    for cue, book in ((np.zeros(5), cb), (np.zeros(32), np.zeros(32))):
        with pytest.raises(ValueError):
            amp_recall(cue, book, K=2)


# --------------------------------------------------------------------------------------
# THE CROSSOVER -- both directions, because the first draft got it one-sided and wrong.
# --------------------------------------------------------------------------------------

def test_amp_beats_cosamp_past_the_phase_transition():
    # The measured win: 0.558 vs 0.167 at M/D=0.33 on an incoherent dictionary. If this stops holding,
    # the module's central claim is wrong and must be re-measured -- do not relax the test.
    rows = measure_vs_cosamp(dim=256, n_atoms=1024, loads=(86,), seeds=4)
    assert rows[0]["amp_f1"] >= rows[0]["cosamp_f1"]


def test_amp_loses_badly_on_a_coherent_dictionary():
    # THE OTHER HALF, and the reason AMP must never silently replace CoSaMP. State evolution assumes a
    # roughly i.i.d. design; a coherent dictionary violates it and AMP collapses (0.052 vs 1.000 at
    # coherence 0.5). Neither method dominates, and the guidance depends on this staying true.
    rng = np.random.default_rng(11)
    cb = _coherent_codebook(rng, 1024, 256, strength=1.0)
    true = rng.choice(1024, size=32, replace=False)
    cue = cb[true].sum(axis=0)
    amp_f1 = _f1([i for i, _ in amp_recall(cue, cb, K=32)], true)
    cos_f1 = _f1([i for i, _ in cosamp_recall(cue, cb, 32)], true)
    assert amp_f1 < cos_f1


def test_cosamp_baseline_is_not_a_strawman_at_default_iterations():
    # Checked before the win was claimed: CoSaMP at 4x its default iteration budget does NOT close the
    # gap at heavy load, so the comparison is not an artefact of an under-run baseline.
    rng = np.random.default_rng(5)
    dim, n, M = 256, 1024, 86
    cb = _codebook(rng, n, dim)
    true = rng.choice(n, size=M, replace=False)
    cue = cb[true].sum(axis=0)
    cheap = _f1([i for i, _ in cosamp_recall(cue, cb, M, iters=15)], true)
    rich = _f1([i for i, _ in cosamp_recall(cue, cb, M, iters=60)], true)
    assert abs(rich - cheap) < 0.15, "CoSaMP is iteration-starved at the default; the comparison is unfair"


def test_amp_cost_is_flat_in_load_while_cosamp_grows():
    # The other genuine advantage: AMP has no per-round least-squares, so its cost does not track M.
    rows = measure_vs_cosamp(dim=256, n_atoms=1024, loads=(16, 86), seeds=3)
    light, heavy = rows[0], rows[1]
    assert heavy["amp_ms"] < 3 * light["amp_ms"], "AMP cost is no longer flat in load"
    assert heavy["cosamp_ms"] > light["cosamp_ms"], "CoSaMP cost no longer grows with load"


# --------------------------------------------------------------------------------------
# CROSS-FACULTY
# --------------------------------------------------------------------------------------

def test_amp_is_wired_and_discoverable():
    mind = lecore.UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(6)
    cb = _codebook(rng, 512, 128)
    true = rng.choice(512, size=6, replace=False)
    assert set(i for i, _ in mind.amp_recall(cb[true].sum(axis=0), cb, K=6)) == set(true.tolist())
    for query in ("recover a bundle without knowing how many items are in it",
                  "approximate message passing", "unmix a heavily loaded bundle"):
        top = str(mind.find_capability(query)[:3]).lower()
        assert "amp" in top or "bundle recovery" in top, "%r surfaces neither AMP nor the family" % query


def test_the_family_entry_names_all_five_members():
    # The one-door invariant: a stranger finding the family must be told AMP exists, or it is a fifth
    # member nobody will ever reach.
    mind = lecore.UnifiedMind(dim=128, seed=0)
    cap = [c for c in mind.find_capability("recover many items from one bundle")[:3]
           if "Bundle recovery" in getattr(c, "name", "")][0]
    for member in ("linear", "occlusion", "iht", "cosamp", "amp"):
        assert member in cap.does.lower(), "the family entry no longer names %s" % member

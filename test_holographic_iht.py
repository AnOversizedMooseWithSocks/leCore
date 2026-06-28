"""Tests for GRAD-1 Iterative Hard Thresholding recovery (holographic_iht): the gradient-native sparse-recovery member,
built on the GRAD-2 optimizer (a gradient step on the reconstruction loss + a K-sparse projection). Ties greedy
occlusion when incoherent, beats it when coherent, reduces to lstsq at K=N."""

import numpy as np

from holographic_iht import iht_recall, hard_threshold


def _f1(rec, true_set):
    got = set(i for i, _ in rec)
    tp = len(got & true_set)
    prec = tp / max(len(got), 1)
    rc = tp / max(len(true_set), 1)
    return 2 * prec * rc / max(prec + rc, 1e-12)


def _occlusion(cue, cb, m):
    cb = np.asarray(cb, float)
    resid = np.asarray(cue, float).copy()
    out = []
    for _ in range(m):
        a = cb @ resid
        j = int(np.argmax(a))
        w = float(a[j])
        out.append((j, w))
        resid = resid - w * cb[j]
    return out


def _trial(coherence, seed, N=200, D=512, M=12):
    rng = np.random.default_rng(seed)
    cb = rng.standard_normal((N, D))
    if coherence > 0:
        cb = cb + coherence * rng.standard_normal(D)
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(N, M, replace=False)
    w = rng.uniform(0.5, 1.5, M)
    cue = (w[:, None] * cb[S]).sum(0)
    true = set(int(i) for i in S)
    return cue, cb, M, true


def test_iht_ties_occlusion_when_incoherent():
    iht = np.mean([_f1(iht_recall(*(_trial(0.0, s)[:3])), _trial(0.0, s)[3]) for s in range(8)])
    assert iht > 0.99


def test_iht_beats_occlusion_when_coherent():
    iht_scores, occ_scores = [], []
    for s in range(12):
        cue, cb, M, true = _trial(1.5, s)
        iht_scores.append(_f1(iht_recall(cue, cb, M), true))
        occ_scores.append(_f1(_occlusion(cue, cb, M), true))
    assert np.mean(iht_scores) > np.mean(occ_scores) + 0.05


def test_iht_kn_reduces_to_lstsq():
    rng = np.random.default_rng(0)
    cb = rng.standard_normal((30, 64))
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(30, 5, replace=False)
    w = rng.uniform(0.5, 1.5, 5)
    cue = (w[:, None] * cb[S]).sum(0)
    rec = iht_recall(cue, cb, 30, steps=5000)  # K=N -> plain gradient descent
    cvec = np.zeros(30)
    for i, wv in rec:
        cvec[i] = wv
    sol = np.linalg.lstsq(cb.T, cue, rcond=None)[0]
    assert np.linalg.norm(cvec - sol) < 1e-6


def test_hard_threshold_keeps_k_largest():
    v = np.array([0.1, -3.0, 2.0, -0.5, 1.0])
    ht = hard_threshold(v, 2)
    assert np.count_nonzero(ht) == 2
    assert ht[1] == -3.0 and ht[2] == 2.0
    # K >= len is the identity
    assert np.array_equal(hard_threshold(v, 5), v)
    assert np.array_equal(hard_threshold(v, 99), v)


def test_iht_returns_descending_by_magnitude():
    rng = np.random.default_rng(1)
    cb = rng.standard_normal((50, 128))
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(50, 6, replace=False)
    cue = cb[S].sum(0)
    rec = iht_recall(cue, cb, 6)
    mags = [abs(w) for _, w in rec]
    assert mags == sorted(mags, reverse=True)


def test_iht_deterministic():
    rng = np.random.default_rng(2)
    cb = rng.standard_normal((40, 64))
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    cue = cb[[1, 5, 9]].sum(0)
    assert iht_recall(cue, cb, 3) == iht_recall(cue, cb, 3)

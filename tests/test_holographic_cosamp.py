"""Tests for SPEED-3 CoSaMP batch-selection recovery (holographic_cosamp): the strongest recovery-family member --
batch atom selection with a least-squares solve each round. Recovers perfectly across dictionary coherence where
greedy occlusion and gradient-step IHT degrade, with exact coefficients, in a few rounds -- and falls off honestly at
the underdetermined phase transition."""

import numpy as np

from holographic.sampling_and_signal.holographic_cosamp import cosamp_recall


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


def _make(coherence, seed, N=200, D=512, M=12):
    rng = np.random.default_rng(seed)
    cb = rng.standard_normal((N, D))
    if coherence > 0:
        cb = cb + coherence * rng.standard_normal(D)
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(N, M, replace=False)
    w = rng.uniform(0.5, 1.5, M)
    cue = (w[:, None] * cb[S]).sum(0)
    return cue, cb, M, set(int(i) for i in S), dict(zip(S.tolist(), w.tolist()))


def test_cosamp_perfect_across_coherence():
    for coh in (0.0, 1.0, 1.5):
        score = np.mean([_f1(cosamp_recall(*_make(coh, s)[:3]), _make(coh, s)[3]) for s in range(8)])
        assert score > 0.99


def test_cosamp_beats_occlusion_when_coherent():
    cos, occ = [], []
    for s in range(8):
        cue, cb, M, true, _w = _make(1.5, s)
        cos.append(_f1(cosamp_recall(cue, cb, M), true))
        occ.append(_f1(_occlusion(cue, cb, M), true))
    assert np.mean(cos) > np.mean(occ) + 0.2


def test_cosamp_coefficients_are_exact():
    cue, cb, M, _true, wmap = _make(0.0, 3)
    rec = dict(cosamp_recall(cue, cb, M))
    rmse = np.sqrt(np.mean([(rec.get(i, 0.0) - wmap[i]) ** 2 for i in wmap]))
    assert rmse < 1e-6


def test_cosamp_converges_in_few_rounds():
    st = {}
    cue, cb, M, _true, _w = _make(1.0, 0)
    cosamp_recall(cue, cb, M, stats=st)
    assert st["rounds"] <= 6


def test_cosamp_falls_off_at_phase_transition():
    # below the transition (M/D small) -> perfect; above it (M ~ D/2) -> degraded. The honest cliff.
    def cliff(M, D=128, N=300):
        sc = []
        for s in range(8):
            rng = np.random.default_rng(s)
            cb = rng.standard_normal((N, D)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
            S = rng.choice(N, M, replace=False); w = rng.uniform(0.5, 1.5, M)
            cue = (w[:, None] * cb[S]).sum(0)
            sc.append(_f1(cosamp_recall(cue, cb, M), set(int(i) for i in S)))
        return np.mean(sc)
    assert cliff(20) > 0.95
    assert cliff(80) < 0.8


def test_cosamp_returns_descending_by_magnitude():
    cue, cb, M, _true, _w = _make(0.0, 1)
    rec = cosamp_recall(cue, cb, M)
    mags = [abs(w) for _, w in rec]
    assert mags == sorted(mags, reverse=True)


def test_cosamp_deterministic():
    cue, cb, M, _true, _w = _make(0.0, 0)
    assert cosamp_recall(cue, cb, M) == cosamp_recall(cue, cb, M)

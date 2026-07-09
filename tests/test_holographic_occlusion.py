"""Tests for RT-V occlusion recall (holographic_occlusion): the alpha-compositing transfer -- an ordered, saturating
front-to-back readout that breaks the linear-bundle capacity cliff for multi-component recall, distinct from (and
measured against) the order-free Hopfield softmax/TopK readouts, tying them at low load."""

import numpy as np

from holographic.rendering.holographic_occlusion import occlusion_recall

_D, _N = 512, 200
_RNG = np.random.default_rng(0)
_CB = _RNG.standard_normal((_N, _D))
_CB = _CB / np.linalg.norm(_CB, axis=1, keepdims=True)


def _make_cue(M, seed):
    r = np.random.default_rng(seed)
    S = r.choice(_N, M, replace=False)
    cue = _CB[S].sum(0)
    return cue / np.linalg.norm(cue), set(S.tolist())


def _f1(pred, true):
    pred = set(pred)
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    rec = tp / len(true) if true else 0.0
    return 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0


def _linear_topm(cue, m):
    return list(np.argsort(-(_CB @ cue))[:m])


def _softmax_topm(cue, m, beta=20.0):
    s = _CB @ cue
    w = np.exp(beta * (s - s.max()))
    return list(np.argsort(-w)[:m])


def test_high_load_occlusion_beats_linear():
    M = 50
    fo = fl = 0.0
    for seed in range(20):
        cue, S = _make_cue(M, seed)
        fo += _f1([j for j, _ in occlusion_recall(cue, _CB, m=M)], S)
        fl += _f1(_linear_topm(cue, M), S)
    assert fo / 20 > 0.99 and fo / 20 > fl / 20 + 0.04


def test_softmax_and_topk_reduce_to_linear_for_recovery():
    # the doc's point: order-free readouts re-rank the same cosines, so top-m is identical -> they wash out together
    fl = fs = 0.0
    for seed in range(20):
        cue, S = _make_cue(50, seed)
        fl += _f1(_linear_topm(cue, 50), S)
        fs += _f1(_softmax_topm(cue, 50), S)
    assert abs(fl - fs) < 1e-6


def test_low_load_ties_linear():
    fo = fl = 0.0
    for seed in range(20):
        cue, S = _make_cue(4, seed)
        fo += _f1([j for j, _ in occlusion_recall(cue, _CB, m=4)], S)
        fl += _f1(_linear_topm(cue, 4), S)
    assert fo / 20 == fl / 20 == 1.0


def test_recovers_all_present_at_high_load():
    cue, S = _make_cue(40, 3)
    rec = set(j for j, _ in occlusion_recall(cue, _CB, m=40))
    assert rec == S


def test_weighted_recovery():
    r = np.random.default_rng(5)
    S = r.choice(_N, 6, replace=False)
    W = r.uniform(0.5, 2.0, 6)
    cue = (W[:, None] * _CB[S]).sum(0)
    rec = occlusion_recall(cue, _CB, m=6)                  # fixed count -> exact recovery
    got = dict(rec)
    assert set(S.tolist()) == set(j for j, _ in rec)
    assert np.mean([abs(got[s] - w) for s, w in zip(S, W)]) < 0.05


def test_front_to_back_heaviest_first():
    # the robust front-to-back property: the heaviest atom is recovered FIRST (full monotonicity is not guaranteed
    # under overlap -- subtracting one atom can raise an overlapping atom's residual score)
    for seed in (5, 7, 11, 13):
        r = np.random.default_rng(seed)
        S = r.choice(_N, 6, replace=False)
        W = r.uniform(0.5, 2.0, 6)
        cue = (W[:, None] * _CB[S]).sum(0)
        rec = occlusion_recall(cue, _CB, m=6)
        assert rec[0][0] == int(S[np.argmax(W)])


def test_threshold_stopping_recovers_about_right_count():
    cnts = []
    for seed in range(15):
        cue, _ = _make_cue(10, seed)
        cnts.append(len(occlusion_recall(cue, _CB, min_share=0.15)))
    assert 9 <= np.mean(cnts) <= 12


def test_deterministic():
    cue, _ = _make_cue(20, 1)
    assert occlusion_recall(cue, _CB, m=20) == occlusion_recall(cue, _CB, m=20)


def test_build_gram():
    from holographic.rendering.holographic_occlusion import build_gram
    G = build_gram(_CB)
    assert G.shape == (_N, _N)
    assert np.allclose(G, _CB @ _CB.T)


def test_gram_path_identical_atoms_and_order():
    from holographic.rendering.holographic_occlusion import build_gram
    G = build_gram(_CB)
    for seed in range(10):
        cue, _ = _make_cue(40, seed)
        a = occlusion_recall(cue, _CB, m=40)
        b = occlusion_recall(cue, _CB, m=40, gram=G)
        assert [j for j, _ in a] == [j for j, _ in b]


def test_gram_path_weights_match_to_epsilon():
    from holographic.rendering.holographic_occlusion import build_gram
    G = build_gram(_CB)
    cue, _ = _make_cue(50, 3)
    a = occlusion_recall(cue, _CB, m=50)
    b = occlusion_recall(cue, _CB, m=50, gram=G)
    assert max(abs(wa - wb) for (_, wa), (_, wb) in zip(a, b)) < 1e-9


def test_gram_path_threshold_mode_matches():
    from holographic.rendering.holographic_occlusion import build_gram
    G = build_gram(_CB)
    cue, _ = _make_cue(30, 5)
    a = occlusion_recall(cue, _CB, min_share=0.15)
    b = occlusion_recall(cue, _CB, min_share=0.15, gram=G)
    assert [j for j, _ in a] == [j for j, _ in b]


def test_gram_none_is_backward_compatible():
    # gram=None must be the original rescan path, unchanged
    cue, _ = _make_cue(25, 2)
    assert occlusion_recall(cue, _CB, m=25) == occlusion_recall(cue, _CB, m=25, gram=None)


def test_gram_cache_hit_reuses_same_object():
    from holographic.rendering.holographic_occlusion import GramCache
    gc = GramCache()
    g1 = gc.gram(_CB)
    g2 = gc.gram(_CB)
    assert g1 is g2 and gc.hits == 1 and gc.misses == 1


def test_gram_cache_identical_recovery():
    from holographic.rendering.holographic_occlusion import GramCache, build_gram
    gc = GramCache()
    G = build_gram(_CB)
    cue, _ = _make_cue(40, 1)
    a = occlusion_recall(cue, _CB, m=40, gram=gc.gram(_CB))
    b = occlusion_recall(cue, _CB, m=40, gram=G)
    assert [j for j, _ in a] == [j for j, _ in b]


def test_gram_cache_lru_bounded():
    from holographic.rendering.holographic_occlusion import GramCache
    gc = GramCache(max_entries=2)
    cbs = [_RNG.standard_normal((50, 64)) for _ in range(4)]
    for cb in cbs:
        gc.gram(cb)
    assert len(gc) <= 2


def test_gram_cache_clear():
    from holographic.rendering.holographic_occlusion import GramCache
    gc = GramCache()
    gc.gram(_CB)
    assert len(gc) == 1
    gc.clear()
    assert len(gc) == 0


def test_gram_cache_gc_invalidates():
    import gc as _gc
    from holographic.rendering.holographic_occlusion import GramCache
    cache = GramCache()
    arr = _RNG.standard_normal((40, 64))
    cache.gram(arr)
    assert len(cache) == 1
    del arr
    _gc.collect()
    # the weakref callback should have dropped the entry once the codebook was collected
    assert len(cache) == 0


def _f1f(rec, true_set):
    got = set(i for i, _ in rec); tp = len(got & true_set)
    p = tp / max(len(got), 1); r = tp / max(len(true_set), 1)
    return 2 * p * r / max(p + r, 1e-12)


def _occlusion_exact(cue, cb, m):
    cb = np.asarray(cb, float); resid = np.asarray(cue, float).copy(); out = []; sel = set()
    for _ in range(m):
        a = cb @ resid; j = int(np.argmax(a))
        if j in sel:
            break
        w = float(a[j]); out.append((j, w)); sel.add(j); resid = resid - w * cb[j]
    return out


def test_forest_occlusion_sublinear_comparisons():
    from holographic.rendering.holographic_occlusion import build_occlusion_forest
    rng = np.random.default_rng(0)
    cb = rng.standard_normal((2000, 256)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    F = build_occlusion_forest(cb, seed=0)
    F.recall_k(cb[5] + cb[10], k=4, beam=4)
    assert F.last_comparisons < cb.shape[0]  # the N-factor: fewer comparisons than a full scan


def test_forest_occlusion_accurate_at_moderate_n():
    from holographic.rendering.holographic_occlusion import occlusion_recall_forest, build_occlusion_forest
    rng = np.random.default_rng(0)
    cb = rng.standard_normal((800, 256)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(800, 10, replace=False); cue = cb[S].sum(0); true = set(int(i) for i in S)
    F = build_occlusion_forest(cb, seed=0)
    assert _f1f(occlusion_recall_forest(cue, cb, 10, forest=F), true) > 0.8


def test_forest_occlusion_never_beats_exact():
    # the kept negative: approximate selection is at best as accurate as the exact scan
    from holographic.rendering.holographic_occlusion import occlusion_recall_forest, build_occlusion_forest
    rng = np.random.default_rng(1)
    cb = rng.standard_normal((3000, 256)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(3000, 12, replace=False); cue = cb[S].sum(0); true = set(int(i) for i in S)
    F = build_occlusion_forest(cb, seed=1)
    f_forest = _f1f(occlusion_recall_forest(cue, cb, 12, forest=F), true)
    f_exact = _f1f(_occlusion_exact(cue, cb, 12), true)
    assert f_forest <= f_exact + 1e-9


def test_forest_occlusion_deterministic():
    from holographic.rendering.holographic_occlusion import occlusion_recall_forest, build_occlusion_forest
    rng = np.random.default_rng(2)
    cb = rng.standard_normal((600, 128)); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    cue = cb[[1, 5, 9, 20]].sum(0)
    F = build_occlusion_forest(cb, seed=2)
    assert occlusion_recall_forest(cue, cb, 4, forest=F) == occlusion_recall_forest(cue, cb, 4, forest=F)

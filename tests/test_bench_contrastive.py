"""Smoke test for tools/bench_contrastive.py -- the contrastive backlog's one harness (E0.1 / E0.4).

No dataset needed (CLINC150 / Banking77 are not vendored): tiny synthetic and hand-written inputs pin the parts every
later gate depends on -- the E0.4 gate-protocol arithmetic, bootstrap determinism, the sha256 hash splits, the
encoder's chunk-invariance -- plus the one contract that makes the harness honest: its InfoNCE arm IS the shared
ProtoStore rule, and as tau -> 0 that rule makes the same decisions as the AdaptHD baseline. Well under 5 s.
"""
import numpy as np

from tools import bench_contrastive as bc


def _brute_threshold(sig, ok, is_in, P, prior=bc.PRIOR):
    """w4 p2b's gate_thr, verbatim (a Python loop over distinct scores): the reference the vectorised form must equal."""
    N_IN, N_OOS = prior
    best = np.inf
    for thr in np.unique(sig)[::-1]:
        s = sig >= thr
        right = (ok & s & is_in).sum() / is_in.sum()
        served_in = (s & is_in).sum() / is_in.sum()
        oos = (s & ~is_in).sum() / max((~is_in).sum(), 1)
        if right * N_IN / max(served_in * N_IN + oos * N_OOS, 1e-9) >= P:
            best = thr
    return best


def test_calibrated_threshold_equals_the_panel_loop():
    rng = np.random.default_rng(0)
    for trial in range(5):
        n = 300
        is_in = rng.random(n) < 0.8
        sig = np.round(rng.random(n), 2)                      # rounded: plenty of ties, the case a vectorised
        ok = is_in & (rng.random(n) < 0.3 + 0.7 * sig)        # scan most easily gets wrong
        for P in (0.9, 0.95, 0.97):
            assert bc.calibrate_threshold(sig, ok, is_in, target=P) == _brute_threshold(sig, ok, is_in, P)


def test_gate_protocol_hand_example():
    # report side: 4 in-scope (3 right), 2 out-of-scope; calibration side mirrors it
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.85, 0.1])
    ok = np.array([True, True, False, True, False, False])
    oos = np.array([False, False, False, False, True, True])
    scores = np.concatenate([s, s])
    correct = np.concatenate([ok, ok])
    is_oos = np.concatenate([oos, oos])
    is_cal = np.concatenate([np.ones(6, bool), np.zeros(6, bool)])
    g = bc.gate_protocol(scores, correct, is_oos, is_cal, targets=(0.95,), prior=(4.0, 2.0))
    assert g["n"] == {"cal_in": 4, "cal_oos": 2, "rep_in": 4, "rep_oos": 2}
    assert g["top1"] == 0.75
    # at threshold 0.9 only the top in-scope item is served: precision 1 -> the lowest passing threshold is 0.9
    # (0.85 would serve an out-of-scope item: precision 1*4/(1*4... ) drops below 0.95)
    c = g["calibrated"]["P0.95"]
    assert c["thr"] == 0.9 and c["precision"] == 1.0 and c["coverage"] == 0.25 and c["oos_served"] == 0.0
    # prior weighting: with prior 4:2 and 2 oos in the split, each oos weighs (2/2)*(4/4) = 1 -> plain AURC
    o = np.argsort(-s, kind="stable")
    w_ok = (ok & ~oos)[o].astype(float)
    prec = np.cumsum(w_ok) / np.arange(1, 7)
    assert abs(g["aurc"] - float(np.mean(1 - prec))) < 1e-12
    assert "oracle" in g and g["oracle"]["P0.95"]["coverage"] == 0.25


def test_precision_is_under_the_prior_not_the_split_mix():
    # duplicating every out-of-scope item changes the split's mix but not the prior-weighted precision / threshold
    rng = np.random.default_rng(1)
    sig = rng.random(200)
    is_in = rng.random(200) < 0.7
    ok = is_in & (rng.random(200) < sig)
    t1 = bc.calibrate_threshold(sig, ok, is_in, target=0.9)
    sig2 = np.concatenate([sig, sig[~is_in]])
    ok2 = np.concatenate([ok, ok[~is_in]])
    in2 = np.concatenate([is_in, is_in[~is_in]])
    assert bc.calibrate_threshold(sig2, ok2, in2, target=0.9) == t1
    r1, r2 = bc.realised(sig, ok, is_in, t1), bc.realised(sig2, ok2, in2, t1)
    assert abs(r1["precision"] - r2["precision"]) < 1e-12
    # (AURC is prior-weighted too, but splitting one item into two half-weight copies moves the intermediate
    # prefix by half a step, so it is equal only approximately -- asserted loosely)
    assert abs(bc.aurc(sig, ok, is_in) - bc.aurc(sig2, ok2, in2)) < 0.01


def test_bootstrap_is_deterministic_and_paired():
    rng = np.random.default_rng(2)
    a = rng.random(500) < 0.7
    b = a | (rng.random(500) < 0.1)                        # b = a plus some fixes: a paired improvement
    r1, r2 = bc.paired_bootstrap(a, b, n=300, seed=0), bc.paired_bootstrap(a, b, n=300, seed=0)
    assert r1 == r2
    assert r1["lo"] <= r1["diff"] <= r1["hi"] and r1["lo"] > 0 and r1["a_only"] == 0
    x = rng.random(500)                                    # continuous values: a new seed must move the interval
    y = x + 0.05 * rng.standard_normal(500)
    c0, c1 = bc.paired_bootstrap(x, y, n=300, seed=0), bc.paired_bootstrap(x, y, n=300, seed=1)
    assert (c0["lo"], c0["hi"]) != (c1["lo"], c1["hi"]) and c0["diff"] == c1["diff"]
    s1 = bc.paired_bootstrap_stat(lambda i: a[i].mean(), lambda i: b[i].mean(), 500, n=200, seed=3)
    s2 = bc.paired_bootstrap_stat(lambda i: a[i].mean(), lambda i: b[i].mean(), 500, n=200, seed=3)
    assert s1 == s2 and abs(s1["diff"] - r1["diff"]) < 1e-12


def test_hash_splits_are_pinned_and_order_free():
    # pinned values (sha256, never hash()): these may only change with a deliberate salt/version bump
    assert bc.oos_half_a(["what is my balance", "play some jazz", "hello there", "book a flight to paris"]).tolist() \
        == [True, False, False, True]
    assert abs(bc.hash_fraction("play some jazz", bc.BANKING_VAL_SALT) - 0.011648981538724491) < 1e-15
    train = {"b_intent": ["play some jazz", "x %d" % 1] + ["b %d" % i for i in range(200)],
             "a_intent": ["play some jazz"] + ["a %d" % i for i in range(200)]}
    rest, val = bc.hash_val_split(train)
    rest2, val2 = bc.hash_val_split(dict(reversed(list(train.items()))))
    assert val == val2 and rest == rest2                  # dict order does not matter
    assert ("play some jazz", "a_intent") in val and ("play some jazz", "b_intent") in val   # duplicates: one side
    assert 0.05 < len(val) / 403.0 < 0.15


def test_encoder_is_chunk_invariant_and_is_systemones():
    from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
    texts = ["I lost my card", "card arrived?", "top up failed", "I lost my card"]
    H1 = bc.encode_texts(texts, dim=64, chunk=1)
    H2 = bc.encode_texts(texts, dim=64, chunk=100)
    assert np.array_equal(H1, H2)
    enc = hashed_ngram_encode(dim=64)
    for t, h in zip(texts, H1):
        v = enc(t)
        assert np.allclose(h, v / (np.linalg.norm(v) + 1e-12), atol=1e-12)


def test_infonce_arm_is_the_shared_rule_and_its_tau0_limit_is_adapthd():
    rng = np.random.default_rng(4)
    C, d = 4, 32
    cen = rng.standard_normal((C, d))
    lab = ["a", "b", "c", "d"]

    def draw(n):
        y = rng.integers(0, C, n)
        H = cen[y] + 1.5 * rng.standard_normal((n, d))
        return H / np.linalg.norm(H, axis=1, keepdims=True), y
    Hi, yi = draw(12)
    for c in range(C):
        yi[c] = c                                           # every row seeded
    Hs, ys = draw(80)
    r_inf = bc.run_arm({"rule": "infonce", "tau": 1e-9, "lr": 0.5}, lab, Hi, yi, Hs, ys)
    r_ad = bc.run_arm({"rule": "adapthd", "lr": 0.5}, lab, Hi, yi, Hs, ys)
    assert r_inf["store"] is not None and r_inf["store"].n_updates == len(ys)   # ProtoStore did the learning
    assert np.array_equal(r_inf["preds"], r_ad["preds"])
    assert np.allclose(r_inf["P"], r_ad["P"], atol=1e-9)


def test_auroc_ties_match_the_pairwise_definition():
    rng = np.random.default_rng(5)
    sc = np.round(rng.random(120), 1)                       # heavy ties
    lab = rng.random(120) < 0.4
    pos, neg = sc[lab], sc[~lab]
    ref = np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])
    assert abs(bc.auroc(sc, lab) - ref) < 1e-12


def test_match_uses_the_panels_printed_digits():
    assert bc.match(0.8263, "0.826") and not bc.match(0.8266, "0.826")
    assert bc.match(0.2007 * 100, "20.1") and bc.match(-6.04, "-6.0")

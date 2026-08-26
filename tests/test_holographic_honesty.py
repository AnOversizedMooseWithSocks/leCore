"""The honesty instrument must pass its own audit: a planted edge clears the bar and its
shuffle control collapses, pure noise does not clear it, and the FDR control rejects real
discoveries while sparing nulls."""
import numpy as np

from holographic.agents_and_reasoning.holographic_honesty import walk_forward_recall, bh_fdr


def test_walk_forward_recall_clears_a_planted_signal_and_its_shuffle_collapses():
    rng = np.random.default_rng(0)
    N, dim = 1500, 256
    states = rng.standard_normal((N, dim))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    # the first coordinate leaks the next sign -> a recall vote should find it
    outcomes = np.sign(states[:, 0]) * np.abs(rng.standard_normal(N)) * 50
    res = walk_forward_recall(states, outcomes, R=25, cost=10)
    assert res["beats_chance"]
    assert res["acc_shuffled"] < 0.5 + res["chance_band"]      # harness is not leaking


def test_walk_forward_recall_does_not_clear_pure_noise():
    rng = np.random.default_rng(0)
    N, dim = 1500, 256
    states = rng.standard_normal((N, dim))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    noise = rng.standard_normal(N) * 50
    assert not walk_forward_recall(states, noise, R=25)["beats_chance"]


def test_bh_fdr_rejects_real_discoveries_and_spares_nulls():
    rng = np.random.default_rng(0)
    # five genuine effects (tiny p) buried in 95 nulls (uniform p)
    p = np.concatenate([rng.uniform(0, 1e-4, 5), rng.uniform(0, 1, 95)])
    rej, k = bh_fdr(p, alpha=0.1, dependent=True)
    assert rej[:5].all()                                       # the planted five survive
    # an all-null family should yield essentially no discoveries
    _, k_null = bh_fdr(rng.uniform(0, 1, 200), alpha=0.1, dependent=True)
    assert k_null <= 2


def test_recall_null_calibrates_confidence_and_is_well_calibrated():
    # A raw cosine means nothing until compared to how high noise reaches against THIS
    # codebook. RecallNull turns it into an honest false-alarm probability: a clean match
    # gets p ~ 0, and a random query's p is ~uniform (so a p<=alpha gate has false-alarm
    # rate ~alpha). The signal-vs-noise discipline, callable per recall.
    import numpy as np
    from holographic.agents_and_reasoning.holographic_honesty import RecallNull
    from holographic.agents_and_reasoning.holographic_ai import Vocabulary, random_vector

    rng = np.random.default_rng(0)
    voc = Vocabulary(512, seed=1)
    for i in range(400):
        voc.get(f"sym{i}")
    _, mat = voc._matrix()
    cal = RecallNull().fit(mat, n_null=2000, seed=0)

    # clean stored atom -> confident (p at the resolution floor)
    assert cal.calibrated_recall(mat[3], mat)[2] < 0.01
    # random queries -> calibrated: the p-values track their nominal false-alarm rate
    rand_p = np.array([cal.calibrated_recall(random_vector(512, rng), mat)[2]
                       for _ in range(1000)])
    assert abs(np.mean(rand_p <= 0.05) - 0.05) < 0.03
    assert abs(np.mean(rand_p <= 0.20) - 0.20) < 0.05
    # calibrated_recall returns the right atom for a clean query
    idx, score, p = cal.calibrated_recall(mat[42], mat)
    assert idx == 42 and score > 0.99 and p < 0.01


def test_sprt_decides_clear_streams():
    # mechanics: a stream drawn from the match density decides MATCH; from the null density, REJECT.
    import numpy as np
    from holographic.agents_and_reasoning.holographic_honesty import SPRTRecall
    rng = np.random.default_rng(0)
    null = rng.normal(0.13, 0.016, 2000)
    match = rng.normal(0.20, 0.016, 2000)            # clearly separated
    sprt = SPRTRecall(null, match, alpha=0.02, beta=0.02)
    d_match, _ = sprt.decide(rng.normal(0.20, 0.016, 50), cap=50)
    d_null, _ = sprt.decide(rng.normal(0.13, 0.016, 50), cap=50)
    assert d_match == "MATCH" and d_null == "REJECT"


def test_sprt_uses_fewer_samples_than_fixed_n_at_matched_error():
    # the Wald optimality bar: reach a target error pair in FEWER expected samples than any fixed-N
    # rule. Overlapping densities so a single cue is ambiguous and streaming earns its keep.
    import numpy as np
    from holographic.agents_and_reasoning.holographic_honesty import SPRTRecall
    rng = np.random.default_rng(1)
    mu0, mu1, sd = 0.13, 0.165, 0.018               # ~2-std separation -> single-cue ambiguity
    null = rng.normal(mu0, sd, 4000); match = rng.normal(mu1, sd, 4000)
    alpha = beta = 0.02
    sprt = SPRTRecall(null, match, alpha=alpha, beta=beta)

    def run_sprt(true_match, trials=2000, cap=200):
        err = 0; samples = 0
        for _ in range(trials):
            stream = rng.normal(mu1 if true_match else mu0, sd, cap)
            d, n = sprt.decide(stream, cap=cap)
            err += ((d == "MATCH") != true_match); samples += n
        return err / trials, samples / trials

    e1, n1 = run_sprt(True); e0, n0 = run_sprt(False)
    asn = 0.5 * (n1 + n0); realized = max(e1, e0)

    def fixed_err(N, trials=2000):
        e = 0
        for tm in (True, False):
            mu = mu1 if tm else mu0
            for _ in range(trials // 2):
                s = rng.normal(mu, sd, N)
                llr = (SPRTRecall._loglik(s, mu1, sd) - SPRTRecall._loglik(s, mu0, sd)).sum()
                e += ((llr > 0) != tm)
        return e / trials

    n_fixed = next((N for N in range(1, 200) if fixed_err(N) <= realized), 200)
    assert asn < n_fixed                              # SPRT decides in fewer expected samples
    assert realized <= 0.06                           # and stays near the target error pair


def test_permutation_null_primitive_through_mind():
    """P2 (Tarter/Siemion/Cranmer): the shuffled-null discipline is exposed as ONE composable faculty
    (mind.permutation_null) so any capability can get honest measurement -- calibrated (random datum flags at
    ~alpha), has power (a true match collapses the null), discoverable, and deterministic."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)

    # discoverable by a user phrasing
    top3 = [c.name for c in m.find_capability("score against a shuffled null")[:3]]
    assert any(n.startswith("Shuffled-null test") for n in top3), top3

    rng0 = np.random.default_rng(2)
    cb = rng0.standard_normal((25, 128)); cb /= np.linalg.norm(cb, axis=1, keepdims=True) + 1e-12
    score = lambda q: float(np.max(cb @ (q / (np.linalg.norm(q) + 1e-12))))
    resample = lambda r: r.standard_normal(128)

    # POWER: a true codebook entry collapses the null
    r = m.permutation_null(score(cb[7]), score, resample, n_null=300, seed=4)
    assert r["collapsed"] and r["p"] < 0.02, r
    assert 0.0 < r["p"] <= 1.0, "the +1 plug keeps p in (0,1]"
    assert r["null_ci"][0] <= r["null_mean"] <= r["null_ci"][1]

    # CALIBRATION: a random query does NOT reliably collapse the null (its p is not tiny)
    q = np.random.default_rng(99).standard_normal(128)
    rr = m.permutation_null(score(q), score, resample, n_null=300, seed=4)
    assert rr["p"] > 0.02, ("a random query should not look like a strong match", rr["p"])

    # DETERMINISM
    assert (m.permutation_null(0.4, score, resample, n_null=200, seed=1)["p"]
            == m.permutation_null(0.4, score, resample, n_null=200, seed=1)["p"])

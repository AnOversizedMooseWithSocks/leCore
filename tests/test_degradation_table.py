"""The measurement behind the README's degradation table (work plan item 6.3).

A headline number in a README is a promise, so it gets a test. This one also exists because the table as
originally drafted did NOT reproduce: it reported 100.0% at 80% damage and 87.5% at 90%, and 87.5% is
exactly 14/16 while 100.0% is 16/16 -- both consistent with a SINGLE DRAW of 16 items rather than an
average. Averaged over 40 seeded trials the honest figures are 97.0% and 75.6%. A single draw of 16 items
reports in steps of 6.25% and will always look tidier than the truth.
"""
import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_ai import bind, random_vector, unbind

D, N, TRIALS = 1024, 16, 40


def _stores():
    """The holographic store and the contiguous one, holding the SAME items in the SAME 1024 floats.
    That equal-budget framing is the whole comparison -- given more space the contiguous store would simply
    win, and the claim is about what distribution buys at a FIXED size."""
    rng = np.random.default_rng(0)
    keys = [random_vector(D, rng) for _ in range(N)]
    vals = [random_vector(D, rng) for _ in range(N)]
    holo = np.sum([bind(k, v) for k, v in zip(keys, vals)], axis=0)
    seg = D // N
    contig = np.concatenate([v[:seg] for v in vals])
    return keys, vals, holo, contig, seg


def _measure(frac):
    keys, vals, holo, contig, seg = _stores()
    V = np.stack(vals)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    rng = np.random.default_rng(7)
    hits = intact = 0
    for _ in range(TRIALS):
        mask = np.ones(D, bool)
        mask[rng.choice(D, int(frac * D), replace=False)] = False
        damaged = holo * mask
        for i, k in enumerate(keys):
            got = unbind(damaged, k)
            norm = np.linalg.norm(got) or 1.0
            hits += int(np.argmax(Vn @ (got / norm))) == i
        for i in range(N):
            intact += bool(mask[i * seg:(i + 1) * seg].all())
    return 100.0 * hits / (TRIALS * N), 100.0 * intact / (TRIALS * N)


@pytest.mark.slow                       # ~40 trials x 4 rows of full recall sweeps
def test_the_readme_degradation_table_reproduces():
    expected = {0.10: (100.0, 1.0), 0.40: (100.0, 0.5), 0.80: (95.0, 0.5), 0.90: (72.0, 0.5)}
    for frac, (min_recall, max_intact) in expected.items():
        recall, intact = _measure(frac)
        assert recall >= min_recall, "recall at %.0f%% damage fell to %.1f%%" % (frac * 100, recall)
        assert intact <= max_intact, "the contiguous baseline survived %.1f%% -- check the equal-budget setup" % intact


def test_the_contiguous_store_is_already_gone_at_ten_percent():
    # The left column is only interesting because of the right one. An item needs all 64 of its own floats,
    # so P(survive) = 0.9^64 ~ 0.001 -- the baseline is not "worse", it is effectively destroyed.
    _, intact = _measure(0.10)
    assert intact < 2.0
    assert 0.9 ** (D // N) < 0.01


def test_holographic_recall_is_perfect_at_forty_percent_loss():
    # The headline claim, and the cheapest row to run.
    recall, _ = _measure(0.40)
    assert recall == 100.0, "recall at 40%% damage is %.1f%%; the README says 100.0%%" % recall

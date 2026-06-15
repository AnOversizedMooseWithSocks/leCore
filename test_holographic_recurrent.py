"""Reservoir (Echo State Network) recurrent layer: gradient-free dynamics, ridge
readout, and the honest verdict that it loses to the existing n-gram / bag baselines on
real corpora while winning a pure-order control. Real-data A/Bs are NLTK-gated; the
mechanism and API tests are hermetic."""
import numpy as np
import pytest

from holographic_recurrent import (EchoStateNetwork, VSAReservoir,
                                    ReservoirCharModel, ReservoirSequenceClassifier,
                                    bag_vs_reservoir, vsa_reservoir_step)


def _nltk_text():
    try:
        from nltk.corpus import gutenberg, udhr
        gutenberg.fileids(); udhr.fileids()
        return True
    except Exception:
        return False


def test_esn_is_gradient_free_ridge_solve_and_fits_a_linear_map():
    # The ESN trains ONLY a linear readout by one ridge solve (no epochs/gradients).
    # On a task its fixed reservoir + linear readout can represent -- echo a delayed
    # input -- it should fit well, proving the readout learns.
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((400, 4))
    targets = np.roll(seq, 3, axis=0)               # predict the input from 3 steps ago
    esn = EchoStateNetwork(n_in=4, n_res=200, seed=0)
    esn.fit(seq, targets)
    pred = esn.predict(seq)
    # correlation between predicted and true delayed signal, well above zero
    corr = np.corrcoef(pred[10:, 0], targets[10:, 0])[0, 1]
    assert esn.W_out is not None
    assert corr > 0.5


def test_echo_state_property_state_forgets_initial_condition():
    # With spectral radius < 1, two runs from DIFFERENT initial states converge to the
    # same trajectory under the same input (the echo state property -- a usable fading
    # memory rather than dependence on where it started).
    rng = np.random.default_rng(1)
    esn = EchoStateNetwork(n_in=3, n_res=150, spectral_radius=0.9, seed=1)
    u = rng.standard_normal((150, 3))
    a = esn.run(u, x0=np.ones(esn.n_res))
    b = esn.run(u, x0=-np.ones(esn.n_res))
    assert np.linalg.norm(a[-1] - b[-1]) < 1e-3     # forgot the initial condition


def test_vsa_reservoir_step_stays_on_unit_sphere():
    # The native reservoir keeps its state normalised (lives in the hypervector space).
    rng = np.random.default_rng(2)
    x = rng.standard_normal(256); x /= np.linalg.norm(x)
    u = rng.standard_normal(256); u /= np.linalg.norm(u)
    for _ in range(20):
        x = vsa_reservoir_step(x, u)
    assert abs(np.linalg.norm(x) - 1.0) < 1e-6


def test_reservoir_wins_the_order_only_control():
    # CONTROL (not a real-task claim): two classes with the SAME multiset in opposite
    # order. A bag is structurally blind (chance); the reservoir carries order and
    # separates them. Proves the mechanism, nothing more.
    train = [("abcd" * 6, 0) for _ in range(15)] + [("dcba" * 6, 1) for _ in range(15)]
    test = [("abcd" * 6, 0)] * 8 + [("dcba" * 6, 1)] * 8
    clf = ReservoirSequenceClassifier(dim=64, n_res=200, seed=0).fit(train)
    acc = np.mean([clf.classify(s) == l for s, l in test])
    assert acc > 0.9                                 # order alone -> the reservoir nails it


@pytest.mark.skipif(not _nltk_text(), reason="NLTK gutenberg/udhr not available")
def test_ngram_beats_reservoir_on_real_alice_generation():
    # KEPT NEGATIVE on real data: next-char accuracy on Gutenberg's Alice favours the
    # existing n-gram over both reservoir flavours.
    import re
    from nltk.corpus import gutenberg
    from holographic_recurrent import compare_to_ngram
    alice = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ",
                   gutenberg.raw("carroll-alice.txt").lower()))
    r = compare_to_ngram(alice[:40000], cut=0.85, n=6, dim=256, n_res=400)
    assert r["ngram"] > r["esn"]                     # n-gram wins generation
    assert r["ngram"] > r["vsa_reservoir"]


@pytest.mark.skipif(not _nltk_text(), reason="NLTK gutenberg/udhr not available")
def test_bag_beats_reservoir_on_real_language_id():
    # KEPT NEGATIVE on real data: on UDHR language ID, a bag-of-trigrams beats the
    # reservoir's final-state classifier -- order adds nothing over symbol statistics.
    import re
    from nltk.corpus import udhr
    langs = ["English-Latin1", "French_Francais-Latin1", "German_Deutsch-Latin1",
             "Spanish_Espanol-Latin1", "Italian_Italiano-Latin1"]
    data = []
    for f in langs:
        try:
            raw = re.sub(r"[^a-z ]+", " ", udhr.raw(f).lower())
        except Exception:
            continue
        lab = f.split("-")[0].split("_")[0]
        for i in range(0, len(raw) - 60, 60):
            ch = raw[i:i + 60]
            if len(ch.strip()) > 40:
                data.append((ch, lab))
    rng = np.random.default_rng(0); rng.shuffle(data)
    cut = int(len(data) * 0.7)
    bag_acc, res_acc = bag_vs_reservoir(data[:cut], data[cut:], ngram=3, dim=128, n_res=300)
    assert bag_acc > res_acc                         # the bag wins on real language ID
    assert bag_acc > 0.8                             # and it wins decisively

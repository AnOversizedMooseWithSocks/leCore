"""The predictive loop: anticipation by resonance (generalises past exact
contexts), surprise as a signal, error-gated learning, free-energy convergence,
generation by anticipation, and the brain wiring."""
import numpy as np

from holographic_predictive import PredictiveMemory, zread


def test_learns_periodic_pattern_to_zero_surprise():
    # On a repeating stream the model should anticipate perfectly within one
    # period: surprise falls to ~0 and accuracy reaches 1.0.
    pm = PredictiveMemory(dim=1024, order=2, seed=0)
    steps = pm.learn_sequence(["a", "b", "c", "d"] * 30)
    assert np.mean([max(0, s.surprise) for s in steps[-8:]]) < 0.05
    assert pm.predict_accuracy(["a", "b", "c", "d"] * 30) == 1.0


def test_free_energy_converges():
    # Free energy (smoothed prediction error) falls toward 0 as the model learns
    # to anticipate its input -- the model becoming a fixed point of the stream.
    pm = PredictiveMemory(dim=1024, order=2, seed=0)
    steps = pm.learn_sequence(["x", "y", "z"] * 40)
    assert steps[-1].self_free_energy < 0.1
    assert steps[0].self_free_energy > steps[-1].self_free_energy


def test_generalises_to_unseen_context():
    # Prediction by resonance: a context never seen exactly still predicts
    # sensibly when a similar one was seen (shared recent symbol). Exact n-gram
    # backoff cannot do this.
    pm = PredictiveMemory(dim=2048, order=2, seed=0)
    for seq in (["the", "cat", "sat"], ["a", "cat", "sat"], ["the", "cat", "sat"]):
        pm.learn_sequence(seq)
    sym, conf = pm.predict(["my", "cat"])      # 'my cat' never seen
    assert sym == "sat"                         # generalised from the cat-contexts
    assert 0.2 < conf < 0.95                    # confident-ish but not a memorised hit


def test_error_gated_learning_actions():
    # Re-seeing a learned transition should reinforce (no create); a novel
    # transition should create. The write is gated by surprise.
    pm = PredictiveMemory(dim=1024, order=2, seed=0)
    pm.learn_sequence(["a", "b"])
    n_after_first = len(pm._ctx)
    pm.step(["a"], "b")                         # familiar -> reinforce, no new entry
    assert len(pm._ctx) == n_after_first
    pm.step(["a"], "q")                         # novel next -> create
    assert len(pm._ctx) > n_after_first


def test_generation_by_anticipation():
    pm = PredictiveMemory(dim=1024, order=2, seed=0)
    pm.learn_sequence(["a", "b", "c", "d"] * 20)
    out = pm.generate(["a", "b"], length=6)
    assert out[:4] == ["c", "d", "a", "b"]      # follows the learned cycle


def test_zread_is_gated_and_weighted():
    # ZREAD blends only entries above the participation gate, weighted by coupling.
    rng = np.random.default_rng(0)
    q = rng.standard_normal(256); q /= np.linalg.norm(q)
    near = q + 0.05 * rng.standard_normal(256)
    far = rng.standard_normal(256)
    vals = [np.ones(256), -np.ones(256)]
    out = zread(q, np.stack([near, far]), vals, t_min=0.5)
    # the near (high-coupling) entry dominates the blend
    assert np.dot(out, vals[0]) > np.dot(out, vals[1])


def test_brain_predictive_wiring():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=1024, seed=0).build_predictor(order=2)
    m.observe_sequence(["a", "b", "c"] * 20)
    assert m.anticipate(["a", "b"])[0] == "c"
    assert m.generate_predictive(["a", "b"], 4)[:2] == ["c", "a"]
    rep = m.prediction_report(["a", "b", "c"] * 20)
    assert rep["accuracy"] > 0.8 and rep["free_energy"] < 0.2

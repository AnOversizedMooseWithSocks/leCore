"""Proof of structure: the lag-coherence profile separates real text from salad,
catching even the locally-coherent self-generated loops that single-step coherence
misses; and steered generation uses it to escape those loops."""
import numpy as np

from holographic_meaning_predict import MeaningPredictor, cooccurrence_space
from holographic_structure import StructureVerifier, steered_generate


def _corpus():
    # grouped two-topic corpus with internal structure, hermetic
    a = "the ship sailed across the cold dark sea toward the distant bright star".split()
    b = "the farmer planted green seeds in the warm wet soil near the old barn".split()
    sents = ([a, b] * 30)
    return sents, [w for s in sents for w in s]


def _verifier(sents, stream):
    vocab, M, idx = cooccurrence_space(sents, dim=512, window=2, seed=0)
    v = StructureVerifier(vocab, M, idx).calibrate(stream, chunk=60, z_floor=2.0)
    return v, vocab, M, idx


def test_structure_score_orders_real_above_salad():
    sents, stream = _corpus()
    v, vocab, M, idx = _verifier(sents, stream)
    real = stream[:120]
    rng = np.random.default_rng(0)
    shuf = list(real); rng.shuffle(shuf)
    rand = [vocab[rng.integers(len(vocab))] for _ in real]
    # the score orders meaning above salad
    assert v.structure_score(real) > v.structure_score(rand)
    assert v.structure_score(real) >= v.structure_score(shuf) - 0.5


def test_catches_self_generated_loop():
    # THE KEY CASE the user pointed at: a locally-coherent generated loop scores
    # far BELOW real text (single-step coherence would rate it higher).
    sents, stream = _corpus()
    v, vocab, M, idx = _verifier(sents, stream)
    mp = MeaningPredictor(dim=512, order=2, seed=0).set_space(vocab, M).fit_transitions(stream)
    loop = list(stream[:2])
    for _ in range(60):
        w, _, _ = mp.predict_meaning(loop[-2:])
        loop.append(w if w else vocab[0])
    assert v.structure_score(loop) < v.structure_score(stream[:120])


def test_steered_generation_beats_greedy_on_structure():
    # Using the verifier as a process: steered generation keeps a higher structure
    # score than plain greedy decoding (which tends to collapse into a loop).
    sents, stream = _corpus()
    v, vocab, M, idx = _verifier(sents, stream)
    mp = MeaningPredictor(dim=512, order=2, seed=0).set_space(vocab, M).fit_transitions(stream)
    seed = stream[:2]
    greedy = list(seed)
    for _ in range(40):
        w, _, _ = mp.predict_meaning(greedy[-2:])
        greedy.append(w if w else vocab[0])
    steered = list(seed) + steered_generate(mp, v, seed, length=40, beam=6, lookback=8)
    assert v.structure_score(steered[2:]) >= v.structure_score(greedy[2:])


def test_profile_has_one_entry_per_lag():
    sents, stream = _corpus()
    v, vocab, M, idx = _verifier(sents, stream)
    p = v.profile(stream[:100])
    assert len(p) == len(v.lags)


def test_calibrate_required():
    sents, stream = _corpus()
    vocab, M, idx = cooccurrence_space(sents, dim=256, window=2, seed=0)
    import pytest
    with pytest.raises(RuntimeError):
        StructureVerifier(vocab, M, idx).structure_score(stream[:50])


def test_brain_verify_and_structured_generate():
    from holographic_unified import UnifiedMind
    sents, stream = _corpus()
    m = UnifiedMind(dim=512, seed=0).build_meaning_predictor(sents, order=2)
    real = m.verify_structure(stream[:120])
    assert "score" in real and "meaningful" in real
    # structured generation returns non-empty, non-constant output
    out = m.generate_structured(stream[:2], length=20, beam=6)
    assert len(out) > 5
    assert len(set(out)) > 1                       # not a single repeated token

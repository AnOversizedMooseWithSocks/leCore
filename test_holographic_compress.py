"""Better structure -> better compression, made literal: a predictor used as a
rank coder spends fewer bits on structured text than on shuffled or random
controls, and the structure score predicts the compression ratio."""
import numpy as np

from holographic_meaning_predict import MeaningPredictor, cooccurrence_space
from holographic_structure import StructureVerifier
from holographic_compress import PredictiveCompressor, structure_compression_correlation


def _setup():
    a = "the president led the government and the senate passed a national law today".split()
    b = "the school taught young children to read good books in the bright classroom".split()
    c = "the team won the football game at the stadium before the loud happy crowd".split()
    sents = ([a, b, c] * 25)
    stream = [w for s in sents for w in s]
    vocab, M, idx = cooccurrence_space(sents, dim=512, window=2, seed=0)
    mp = MeaningPredictor(dim=512, order=2, seed=0).set_space(vocab, M).fit_transitions(stream)
    v = StructureVerifier(vocab, M, idx).calibrate(stream, chunk=60, z_floor=2.0)
    return PredictiveCompressor(mp), v, vocab, stream


def test_structure_compresses_below_baseline():
    comp, v, vocab, stream = _setup()
    r = comp.encode_cost(stream[:200])
    assert r["ratio"] < 0.95                          # structured text beats uniform
    assert r["bits_per_symbol"] < r["baseline_bits_per_symbol"]


def test_shuffle_compresses_worse_than_real():
    comp, v, vocab, stream = _setup()
    real = stream[:300]
    rng = np.random.default_rng(0)
    shuf = list(real); rng.shuffle(shuf)
    assert comp.compressibility(real) < comp.compressibility(shuf)


def test_random_compresses_worst():
    comp, v, vocab, stream = _setup()
    rng = np.random.default_rng(0)
    randw = [vocab[rng.integers(len(vocab))] for _ in range(300)]
    assert comp.compressibility(stream[:300]) < comp.compressibility(randw)


def test_structure_predicts_compression():
    # across windows of varying structure, more structure -> lower ratio (negative
    # correlation between structure score and compression ratio).
    comp, v, vocab, stream = _setup()
    rng = np.random.default_rng(0)
    windows = [stream[i:i + 80] for i in range(0, 600, 80)]
    # add some shuffled windows to widen the structure range
    for i in range(0, 300, 80):
        w = list(stream[i:i + 80]); rng.shuffle(w); windows.append(w)
    corr = structure_compression_correlation(v, comp, windows)
    assert corr < 0.0                                  # more structure -> fewer bits


def test_brain_compress_cost():
    from holographic_unified import UnifiedMind
    a = "the president led the government and the senate passed a national law".split()
    b = "the school taught young children to read good books in the classroom".split()
    sents = [a, b] * 25
    m = UnifiedMind(dim=512, seed=0).build_meaning_predictor(sents, order=2)
    stream = [w for s in sents for w in s]
    r = m.compress_cost(stream[:150])
    assert r["ratio"] < 1.0 and r["n"] > 0

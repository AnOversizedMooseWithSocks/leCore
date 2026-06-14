"""Self-discovery of structure: branching entropy on the substrate recovers unit
boundaries from an unsegmented stream with no labels, and the discovered chunks
compress better than single symbols."""
import numpy as np

from holographic_segment import (Segmenter, boundary_f1, chunk_compression)


def _repeated_words_stream():
    # a stream built from a small word set with NO separators; boundaries are known
    words = ["cat", "dog", "bird", "fish", "lion", "bear"]
    rng = np.random.default_rng(0)
    seq = [words[rng.integers(len(words))] for _ in range(400)]
    stream = "".join(seq)
    truth = set()
    pos = -1
    for w in seq:
        pos += len(w)
        truth.add(pos)                                # boundary after each word's last char
    return stream, truth


def test_discovers_boundaries_above_random():
    stream, truth = _repeated_words_stream()
    seg = Segmenter(dim=512, order=3, seed=0).fit(stream)
    pred = seg.boundaries(stream, percentile=60)
    f = boundary_f1(pred, truth)["f1"]
    rng = np.random.default_rng(1)
    rand = set(rng.choice(len(stream), len(pred), replace=False).tolist())
    rf = boundary_f1(rand, truth)["f1"]
    assert f > rf * 1.5                              # genuinely above chance


def test_branching_entropy_peaks_at_boundaries():
    # entropy should be higher at true boundaries than mid-unit on average
    stream, truth = _repeated_words_stream()
    seg = Segmenter(dim=512, order=3, seed=0).fit(stream)
    H = seg.branching_entropy(stream)
    at_b = np.mean([H[i] for i in truth if i < len(H)])
    mid = np.mean([H[i] for i in range(len(H)) if i not in truth])
    assert at_b > mid


def test_discovered_chunks_compress_better():
    stream, truth = _repeated_words_stream()
    seg = Segmenter(dim=512, order=3, seed=0).fit(stream)
    chunks = seg.segment(stream, percentile=60)
    cb, sb = chunk_compression(stream, chunks)
    assert cb < sb                                  # chunks beat single-symbol coding


def test_segment_reconstructs_stream():
    stream, truth = _repeated_words_stream()
    seg = Segmenter(dim=256, order=3, seed=0).fit(stream)
    chunks = seg.segment(stream, percentile=60)
    assert "".join("".join(c) for c in chunks) == stream    # lossless partition


def test_resonance_smearing_is_avoided():
    # branching uses exact contexts: two different contexts produce independent
    # entropies (no blending). Construct a context with one successor (low entropy)
    # and one with many (high entropy).
    seg = Segmenter(dim=512, order=2, seed=0)
    # 'ab' always followed by 'c'; 'xy' followed by many different chars
    s = "abc" * 30 + "".join("xy" + ch for ch in "defghijklmno")
    seg.fit(s)
    H = seg.branching_entropy(s)
    # entropy right after 'ab' (predicting c) should be low; after 'xy' (many) high
    # find an index where context is 'ab'
    low = [H[i] for i in range(2, 90) if s[i-2:i] == "ab"]
    high = [H[i] for i in range(90, len(s)) if s[i-2:i] == "xy"]
    assert np.mean(high) > np.mean(low)


def test_brain_discover_units():
    from holographic_unified import UnifiedMind
    stream, truth = _repeated_words_stream()
    m = UnifiedMind(dim=512, seed=0)
    r = m.discover_units(stream, order=3, percentile=60)
    assert len(r["chunks"]) > 1
    assert r["chunk_bits"] < r["symbol_bits"]       # the compression payoff

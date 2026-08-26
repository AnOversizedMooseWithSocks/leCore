"""Procedural generation (four modalities), each measured against the dumbest honest
baseline. Image/audio tests are hermetic; the text coherence A/B is NLTK-gated (real
Alice). The bar is the same as everywhere else: beat the baseline on an honest metric.
"""
import os
import re
import tempfile
import wave

import numpy as np
import pytest

from holographic.misc.holographic_generate import morph_images, crossfade_images, ghosting, morph_video, nucleus_sample, generate_text, real_word_fraction, distinct_ngram_fraction, sequence_to_pitches, sonify


def _nltk(*names):
    try:
        from nltk import corpus
        for n in names:
            getattr(corpus, n).fileids()
        return True
    except Exception:
        return False


def _archive_with_gallery():
    from holographic.misc.holographic_archive import HolographicArchive, _gallery
    imgs = _gallery(S=64)
    arch = HolographicArchive(shape=imgs[0].shape, capacity=len(imgs),
                              keep=600, dim=16384, seed=0)
    for im in imgs:
        arch.add(im)
    return arch


# ----------------------------- image / video ------------------------------

def test_morph_endpoints_are_exact_and_count_is_right():
    arch = _archive_with_gallery()
    A, B = arch.recover(0), arch.recover(3)
    frames = morph_images(arch.M, A, B, steps=15)
    assert len(frames) == 15
    assert np.sqrt(np.mean((frames[0] - A) ** 2)) < 1e-6     # frame[0] == A
    assert np.sqrt(np.mean((frames[-1] - B) ** 2)) < 1e-6    # frame[-1] == B
    for f in frames:
        assert f.min() >= 0.0 and f.max() <= 1.0             # every frame a valid image


def test_coeff_morph_beats_crossfade_on_ghosting():
    # The honest win: a coefficient-domain morph midpoint sits AWAY from the literal
    # double-exposure, while a pixel crossfade midpoint IS the double-exposure (0).
    arch = _archive_with_gallery()
    import itertools
    morph_g, cross_g = [], []
    for i, j in itertools.combinations(range(arch.n), 2):
        A, B = arch.recover(i), arch.recover(j)
        morph_g.append(ghosting(morph_images(arch.M, A, B, steps=21)[10], A, B))
        cross_g.append(ghosting(crossfade_images(A, B, steps=21)[10], A, B))
    assert np.mean(cross_g) < 1e-6                            # crossfade is exactly the ghost
    assert np.mean(morph_g) > 10 * (np.mean(cross_g) + 1e-9)  # morph clearly less ghosted
    assert np.mean(morph_g) > 0.02


def test_morph_video_threads_keyframes():
    arch = _archive_with_gallery()
    keys = [arch.recover(i) for i in range(4)]
    frames = morph_video(arch.M, keys, steps_between=8)
    # 3 transitions of 8 frames; the first two drop their shared last frame: 7 + 7 + 8 = 22
    assert len(frames) == 22
    assert np.sqrt(np.mean((frames[0] - keys[0]) ** 2)) < 1e-6
    assert np.sqrt(np.mean((frames[-1] - keys[-1]) ** 2)) < 1e-6


# ----------------------------- text (nucleus) -----------------------------

def test_nucleus_sample_respects_top_p_support():
    # With a peaked distribution and a small top_p, sampling only ever returns symbols
    # from the nucleus (the few most-likely), never the long tail.
    rng = np.random.default_rng(0)
    dist = {"a": 0.6, "b": 0.3, "c": 0.05, "d": 0.03, "e": 0.02}
    seen = {nucleus_sample(dist, rng, temperature=1.0, top_p=0.9) for _ in range(200)}
    assert seen <= {"a", "b", "c"}                           # tail (d,e) is trimmed
    assert "a" in seen and "b" in seen


def test_repetition_penalty_downweights_recent_symbols():
    # With a flat distribution and a strong penalty, a symbol filling the recent window
    # is chosen far less than its unpenalised share.
    rng = np.random.default_rng(0)
    dist = {"x": 1.0, "y": 1.0}
    picks = [nucleus_sample(dist, rng, temperature=1.0, top_p=1.0,
                            recent="x" * 12, rep_penalty=0.9) for _ in range(300)]
    assert picks.count("x") < picks.count("y")               # 'x' suppressed


@pytest.mark.skipif(not _nltk("gutenberg"), reason="NLTK gutenberg unavailable")
def test_nucleus_is_more_coherent_than_temperature_on_alice():
    # The real top-p win on real text: cutting the unlikely tail raises the real-word
    # fraction (coherence) over plain temperature sampling, the model and seed held fixed.
    from holographic.misc.holographic_text import HolographicNGram
    from nltk.corpus import gutenberg
    alice = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ",
                   gutenberg.raw("carroll-alice.txt").lower()))
    ng = HolographicNGram(dim=1024, n=6, seed=0).fit(alice[:80000])
    vocab = set(alice.split())
    nuc, tmp = [], []
    for s in range(4):
        g_n = generate_text(ng, "alice was ", length=350, temperature=0.6, top_p=0.85,
                            rng=np.random.default_rng(s))
        g_t = ng.generate("alice was ", length=350, temperature=0.6,
                          rng=np.random.default_rng(s))
        nuc.append(real_word_fraction(g_n, vocab))
        tmp.append(real_word_fraction(g_t, vocab))
    assert np.mean(nuc) > np.mean(tmp) + 0.05                # clearly more coherent


# ----------------------------- audio (sonify) -----------------------------

def test_sequence_to_pitches_is_faithful_and_deterministic():
    # Distinct symbols map to distinct pitches; same symbol always the same pitch; and
    # the mapping is order-independent (depends on the symbol, not where it appears).
    pitches, table = sequence_to_pitches("abca")
    assert pitches[0] == pitches[3]                          # both 'a' -> same pitch
    assert len({round(p, 4) for p in pitches}) == 3         # a,b,c -> 3 distinct pitches
    again, _ = sequence_to_pitches("caab")
    assert again[1] == pitches[0]                            # 'a' is 'a' regardless of order


def test_sonify_writes_a_valid_wav_of_expected_length():
    seq = "abcabc"
    path = os.path.join(tempfile.gettempdir(), "holo_sonify_test.wav")
    sonify(seq, path, note_seconds=0.1, sample_rate=8000)
    with wave.open(path, "r") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 8000
        # one 0.1s note per symbol at 8 kHz = 800 frames each
        assert w.getnframes() == len(seq) * 800
    os.remove(path)

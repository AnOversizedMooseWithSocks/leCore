"""Procedural generation, four modalities, driving decoders the engine already has.

This is the "low-hanging generation" bundle: it turns the engine from analysis-and-memory
into something that PRODUCES output, without learning a distribution or crossing the
gradient line. Each generator reuses an existing decoder and is measured against the
dumbest honest baseline, the same bar as everything else -- it beats the baseline on an
honest metric or it is reported as not doing so.

  image / video  : slerp morph between two stored images in the DCT-coefficient domain,
                   then inverse-transform each interpolated frame. The win over a pixel
                   crossfade is measurable: a crossfade midpoint IS a double-exposure of
                   the two pictures (ghosting); a coefficient-domain morph blends
                   STRUCTURE, so its midpoint sits measurably away from the double image.
  text           : nucleus (top-p) decoding with an optional repetition penalty over the
                   existing holographic n-gram distribution. Cutting the unlikely tail
                   measurably raises the real-word fraction (coherence) versus plain
                   temperature sampling, at a modest diversity cost -- the real top-p
                   tradeoff, reported honestly.
  audio          : sonify an existing symbolic sequence -- map each symbol to a pitch and
                   render a short tone, writing a real WAV. This is rendering, not a
                   learned synthesiser; the honest claim is faithfulness (distinct symbols
                   -> distinguishable, repeatable pitches), which is asserted, not the
                   vague "it sounds good".

Nothing here learns a distribution; it drives stored structure forward. The heavier
generative path (run the resonator forward to compose NEW scenes) builds on this.
"""
import struct
import wave

import numpy as np

from holographic_ai import slerp


# ===========================================================================
# 1. IMAGE / VIDEO -- coefficient-domain slerp morph
# ===========================================================================

def _dct_channels(M, img):
    """Per-channel 2D DCT of an image (H,W) or (H,W,C)."""
    if img.ndim == 2:
        return (M @ img @ M.T)[..., None]
    return np.stack([M @ img[..., c] @ M.T for c in range(img.shape[-1])], -1)


def _idct_channels(M, coeff):
    """Inverse of _dct_channels, clipped back to a valid image in [0,1]."""
    chans = [M.T @ coeff[..., c] @ M for c in range(coeff.shape[-1])]
    img = np.stack(chans, -1) if coeff.shape[-1] > 1 else chans[0]
    return np.clip(img, 0, 1)


def morph_images(M, img_a, img_b, steps=21):
    """A morph from img_a to img_b by spherical interpolation in the DCT-coefficient
    domain. Interpolating the coefficient DIRECTION (slerp) and magnitude separately
    blends image STRUCTURE -- low frequencies (layout) and high frequencies (detail)
    cross over smoothly -- rather than fading one picture out while the other fades in.
    Returns `steps` frames, frame[0]==img_a and frame[-1]==img_b. M is a DCT basis
    matrix (e.g. an archive's `.M`)."""
    ca, cb = _dct_channels(M, np.asarray(img_a, float)), _dct_channels(M, np.asarray(img_b, float))
    fa, fb = ca.ravel(), cb.ravel()
    na, nb = float(np.linalg.norm(fa)), float(np.linalg.norm(fb))
    ua, ub = (fa / na if na else fa), (fb / nb if nb else fb)
    frames = []
    for t in np.linspace(0.0, 1.0, steps):
        mag = (1 - t) * na + t * nb
        v = slerp(ua, ub, float(t)) * mag
        frames.append(_idct_channels(M, v.reshape(ca.shape)))
    return frames


def crossfade_images(img_a, img_b, steps=21):
    """The honest baseline: a straight pixel-space linear crossfade. Its midpoint is the
    exact double-exposure 0.5*a + 0.5*b -- both pictures visible at once (ghosting)."""
    a, b = np.asarray(img_a, float), np.asarray(img_b, float)
    return [np.clip((1 - t) * a + t * b, 0, 1) for t in np.linspace(0.0, 1.0, steps)]


def ghosting(mid_frame, img_a, img_b):
    """How far a morph's midpoint sits from the literal double-exposure of the two
    endpoints. A pixel crossfade scores 0 (it IS the double-exposure); a structural
    morph scores higher because it blends content instead of overlaying it."""
    return float(np.sqrt(np.mean((np.asarray(mid_frame, float)
                                  - 0.5 * (np.asarray(img_a, float) + np.asarray(img_b, float))) ** 2)))


def morph_video(M, keyframes, steps_between=12):
    """A procedural video: morph through a list of stored keyframes in turn, in the
    coefficient domain. Returns the concatenated frame list (each transition contributes
    `steps_between` frames; shared endpoints are de-duplicated)."""
    frames = []
    for i in range(len(keyframes) - 1):
        seg = morph_images(M, keyframes[i], keyframes[i + 1], steps=steps_between)
        frames.extend(seg if i == len(keyframes) - 2 else seg[:-1])
    return frames


# ===========================================================================
# 2. TEXT -- nucleus (top-p) decoding with an optional repetition penalty
# ===========================================================================

def nucleus_sample(dist, rng, temperature=0.6, top_p=0.9, recent="", rep_penalty=0.0):
    """Sample one symbol from a {symbol: score} distribution by NUCLEUS (top-p) decoding:
    keep the smallest set of most-likely symbols whose probability sums to top_p, and
    sample within it. This trims the unlikely tail that produces garbage while leaving
    real diversity intact. An optional repetition penalty downweights symbols seen in the
    `recent` window before sampling (off by default -- this n-gram does not loop, so it
    is a knob, not a fix). Returns the chosen symbol."""
    syms = list(dist)
    w = np.clip(np.array([dist[s] for s in syms], dtype=float), 0, None)
    if rep_penalty > 0 and recent:
        for k, s in enumerate(syms):
            if s in recent:
                w[k] *= (1.0 - rep_penalty)
    w = w ** (1.0 / max(temperature, 1e-6))
    tot = w.sum()
    if tot <= 0:
        return syms[int(rng.integers(len(syms)))]
    p = w / tot
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    keep = order[:max(1, int(np.searchsorted(cum, top_p)) + 1)]
    masked = np.zeros_like(p)
    masked[keep] = p[keep]
    masked /= masked.sum()
    return syms[int(rng.choice(len(syms), p=masked))]


def generate_text(ngram, seed_text, length=300, temperature=0.6, top_p=0.9,
                  rep_penalty=0.0, rng=None):
    """Generate text from a fitted HolographicNGram using nucleus decoding. Drives the
    model's own holographic next-character distribution; only the DECODING differs from
    `ngram.generate` (which samples the full tempered distribution)."""
    rng = rng or np.random.default_rng(0)
    out = seed_text.lower() if getattr(ngram, "fold_case", True) else seed_text
    for _ in range(length):
        dist = ngram._distribution(out[-ngram.n:])
        out += nucleus_sample(dist, rng, temperature, top_p,
                              recent=out[-12:], rep_penalty=rep_penalty)
    return out


def real_word_fraction(text, vocabulary):
    """Fraction of whitespace tokens that are words in `vocabulary` -- the coherence
    metric. Nucleus decoding raises this over plain temperature sampling by cutting the
    tail characters that break words."""
    ws = [w for w in text.split() if w]
    return float(np.mean([w in vocabulary for w in ws])) if ws else 0.0


def distinct_ngram_fraction(text, n=4):
    """Fraction of distinct n-grams -- a diversity metric. Nucleus trades a little of
    this for the coherence gain above; the tradeoff is reported, not hidden."""
    grams = [text[i:i + n] for i in range(len(text) - n)]
    return len(set(grams)) / max(1, len(grams))


# ===========================================================================
# 3. AUDIO -- sonify an existing symbolic sequence to a real WAV
# ===========================================================================

def sequence_to_pitches(symbols, base_hz=220.0, semitone_span=24):
    """Map each distinct symbol in a sequence to a fixed pitch on a chromatic scale,
    deterministically (sorted symbols -> ascending semitones, wrapping at semitone_span).
    Returns a list of frequencies, one per symbol in the sequence. Deterministic and
    repeatable: the same symbol always sonifies to the same pitch."""
    vocab = sorted(set(symbols))
    table = {s: base_hz * (2.0 ** ((i % semitone_span) / 12.0)) for i, s in enumerate(vocab)}
    return [table[s] for s in symbols], table


def sonify(symbols, path, note_seconds=0.18, sample_rate=16000, base_hz=220.0):
    """Render a symbolic sequence to a real mono 16-bit WAV at `path`: each symbol becomes
    a short sine tone at its mapped pitch, with a tiny fade to avoid clicks. This is
    RENDERING, not synthesis from a learned model -- the honest claim is faithfulness
    (distinct symbols -> distinguishable, repeatable pitches), which the tests assert.
    Returns the pitch table used."""
    pitches, table = sequence_to_pitches(symbols, base_hz=base_hz)
    n = int(note_seconds * sample_rate)
    fade = max(1, n // 20)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    samples = []
    for f in pitches:
        t = np.arange(n) / sample_rate
        samples.append(0.6 * np.sin(2 * np.pi * f * t) * env)
    wav = np.concatenate(samples) if samples else np.zeros(0)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<%dh" % len(pcm), *pcm.tolist()))
    return table


def _demo():
    import re
    print("PROCEDURAL GENERATION (four modalities, driving existing decoders)\n")

    # --- image morph vs crossfade: the anti-ghosting win ---
    try:
        from holographic_archive import HolographicArchive, _gallery
        imgs = _gallery(S=64)
        arch = HolographicArchive(shape=imgs[0].shape, capacity=len(imgs),
                                  keep=600, dim=16384, seed=0)
        for im in imgs:
            arch.add(im)
        A, B = arch.recover(0), arch.recover(3)
        mid_morph = morph_images(arch.M, A, B, steps=21)[10]
        mid_x = crossfade_images(A, B, steps=21)[10]
        print(f"image morph:  midpoint distance from double-exposure -- "
              f"coeff slerp {ghosting(mid_morph, A, B):.3f} vs crossfade {ghosting(mid_x, A, B):.3f} "
              f"(higher = less ghosting; crossfade IS the ghost)")
    except Exception as e:
        print("image morph: (skipped:", e, ")")

    # --- text: nucleus coherence vs temperature ---
    try:
        from holographic_text import HolographicNGram
        from nltk.corpus import gutenberg
        alice = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ",
                       gutenberg.raw("carroll-alice.txt").lower()))
        ng = HolographicNGram(dim=1024, n=6, seed=0).fit(alice[:80000])
        vocab = set(alice.split())
        rng = np.random.default_rng(0)
        nuc = generate_text(ng, "alice was ", length=400, temperature=0.6, top_p=0.85, rng=rng)
        tmp = ng.generate("alice was ", length=400, temperature=0.6, rng=np.random.default_rng(0))
        print(f"text nucleus: real-word fraction -- nucleus {real_word_fraction(nuc, vocab):.3f} "
              f"vs temperature {real_word_fraction(tmp, vocab):.3f} "
              f"(distinct-4gram {distinct_ngram_fraction(nuc):.2f} vs {distinct_ngram_fraction(tmp):.2f}: "
              f"more coherent for a little less variety)")
    except Exception as e:
        print("text nucleus: (skipped:", e, ")")

    # --- audio: sonify a sequence to a WAV ---
    try:
        import tempfile, os
        seq = "abcacbacabacbcab"
        path = os.path.join(tempfile.gettempdir(), "holo_sonify_demo.wav")
        table = sonify(seq, path)
        print(f"audio sonify: wrote {path} -- {len(set(seq))} distinct symbols -> "
              f"{len(set(round(v, 2) for v in table.values()))} distinct pitches "
              f"({min(table.values()):.0f}-{max(table.values()):.0f} Hz)")
    except Exception as e:
        print("audio sonify: (skipped:", e, ")")


if __name__ == "__main__":
    _demo()

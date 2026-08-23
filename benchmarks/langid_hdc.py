"""B2 -- the Rahimi/Kanerva 21-language identification benchmark, run deterministically.

THE PUBLISHED NUMBER, and it is the most-cited HDC result there is: Joshi, Halseth & Kanerva
(arXiv:1412.7026) report 97.8% on 21,000 short sentences across 21 European languages with
tetragrams; Rahimi, Kanerva & Rabaey (ISLPED 2016) report 96.7% for the hardware trigram classifier.

WHY leCORE RUNS IT AT ALL. The task is already zero-learned-weight -- n-grams in hyperspace, one
bundled prototype per language -- so accuracy is not the axis we win on. What we add is the axis
nobody else reports: BIT-REPRODUCIBILITY. Every letter hypervector is derived from hashlib, not
from a PRNG seeded by wall-clock or by Python's salted hash(), so the same corpus yields the same
prototypes and the same predictions on any machine, forever. The published implementations use
MATLAB's rand and are reproducible only within a session.

THE ENCODING, exactly as the papers describe it: an n-gram is the letter vectors bound by
PERMUTATION-then-multiply (rho^(n-1)(l1) * rho^(n-2)(l2) * ... * ln), a text is the SUM of its
n-gram vectors, and a language prototype is the sum over its training text. Classification is
nearest prototype by cosine. Nothing is learned; everything is counted.

KEPT NEGATIVE: the bundle is a SUM, not a normalised mean, and the prototypes are compared by
COSINE -- normalising per n-gram first would throw away the frequency information that carries the
signal, which is the whole reason a sum works here.
"""

import hashlib
import pathlib
import sys

import numpy as np

# The 21 languages of the benchmark, by the ISO prefix the test filenames use, mapped to the
# training file that carries that language. Both live in the repo; nothing is downloaded twice.
LANGS = {
    "bg": "bulgarian", "cs": "czech", "da": "danish", "de": "german", "el": "greek",
    "en": "english", "es": "spanish", "et": "estonian", "fi": "finnish", "fr": "french",
    "hu": "hungarian", "it": "italian", "lt": "lit", "lv": "lav", "nl": "dutch",
    "pl": "polish", "pt": "portuguese", "ro": "romanian", "sk": "slovak", "sl": "slovenian",
    "sv": "swedish",
}

ALPHABET = "abcdefghijklmnopqrstuvwxyz "


def letter_vectors(dim, seed=0):
    """One deterministic bipolar hypervector per letter, derived from hashlib.

    WHY hashlib AND NOT default_rng: the point of this run is that the vectors are a FUNCTION OF
    THE LETTER, reproducible across machines, processes and languages -- not merely of a seed held
    in one process. sha256(letter|index|seed) is the same everywhere, forever.
    """
    out = {}
    for ch in ALPHABET:
        bits = np.empty(dim, dtype=np.int8)
        need = (dim + 255) // 256
        raw = b"".join(hashlib.sha256(("%s|%d|%d" % (ch, i, seed)).encode()).digest()
                       for i in range(need))
        arr = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:dim]
        bits[:] = np.where(arr == 1, 1, -1)
        out[ch] = bits.astype(np.float32)
    return out


def encode_text(text, lv, dim, n=4):
    """Sum of n-gram vectors; an n-gram is the permuted-and-multiplied letter product.

    Vectorised over the whole text at once: rolling each letter's vector by its position in the
    n-gram is exactly the permutation the papers apply, and multiplying the n shifted planes is the
    binding. Doing it as one array product rather than a Python loop is what makes 21,000 test
    sentences take seconds instead of an hour.
    """
    idx = [ALPHABET.index(c) for c in text if c in ALPHABET]
    if len(idx) < n:
        return np.zeros(dim, dtype=np.float32)
    # The letter TABLE is (27, dim); the text is a list of indices into it. Never materialise
    # the text as (L, dim) -- that was 4.6 GB for one file and is the same OOM in a new costume.
    T = np.stack([lv[c] for c in ALPHABET])               # (27, dim), tiny and reused
    idx = np.asarray(idx, dtype=np.int32)
    ng = len(idx) - n + 1
    # CHUNKED, and the chunking is EXACT: the text vector is a SUM over n-grams and addition is
    # associative, so summing chunk by chunk is bit-identical to summing all at once (up to float
    # ordering, which is fixed here because the chunk boundaries are deterministic). The full-array
    # form allocated (ng, dim) floats -- 7 GB for one 175 KB training file at dim 10000, which the
    # OOM killer found before any measurement did. Peak memory is now the chunk, not the corpus.
    out = np.zeros(dim, dtype=np.float32)
    step = max(1, (1 << 22) // max(dim, 1))               # ~16 MB of float32 per chunk
    for s0 in range(0, ng, step):
        s1 = min(s0 + step, ng)
        acc = np.ones((s1 - s0, dim), dtype=np.float32)
        for k in range(n):
            acc *= np.roll(T[idx[s0 + k:s1 + k]], n - 1 - k, axis=1)
        out += acc.sum(axis=0)
    return out


def run(dim=10000, n=4, seed=0, root="/tmp/hdc_lang/HDC-Language-Recognition-master"):
    """Build one prototype per language from the training texts, classify every test sentence."""
    root = pathlib.Path(root)
    lv = letter_vectors(dim, seed)

    protos, names = [], []
    for code, fname in sorted(LANGS.items()):
        p = root / "training_texts" / (fname + ".txt")
        if not p.exists():
            print("MISSING training text for %s (%s)" % (code, fname))
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore").lower()
        v = encode_text(txt, lv, dim, n)
        protos.append(v / (np.linalg.norm(v) or 1.0))
        names.append(code)
    P = np.stack(protos)
    print("built %d language prototypes at dim=%d, n=%d" % (len(names), dim, n))

    files = sorted((root / "testing_texts").glob("*_p.txt"))
    files = [f for f in files if f.name.split("_")[0] in LANGS]
    correct = total = 0
    per = {c: [0, 0] for c in names}
    for f in files:
        gold = f.name.split("_")[0]
        if gold not in per:
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore").lower()
        v = encode_text(txt, lv, dim, n)
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        pred = names[int(np.argmax(P @ (v / nv)))]
        ok = (pred == gold)
        correct += ok
        total += 1
        per[gold][0] += ok
        per[gold][1] += 1
    acc = correct / max(total, 1)
    print("ACCURACY %.4f on %d test sentences (%d languages)" % (acc, total, len(names)))
    worst = sorted(((v[0] / max(v[1], 1), k) for k, v in per.items()))[:4]
    print("weakest: " + "  ".join("%s %.3f" % (k, a) for a, k in worst))
    # the differentiator: the run is a function of the corpus, not of a session
    h = hashlib.sha256(P.round(6).tobytes()).hexdigest()[:32]
    print("prototype SHA-256 (first 32): %s" % h)
    return acc, total, h


def _selftest():
    """A tiny end-to-end run: prototypes must separate two languages far better than chance."""
    lv = letter_vectors(1024, 0)
    a = encode_text("the quick brown fox jumps over the lazy dog " * 20, lv, 1024, 4)
    b = encode_text("der schnelle braune fuchs springt ueber den faulen hund " * 20, lv, 1024, 4)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    same = float(a @ a)
    cross = float(a @ b)
    assert same > 0.99, same
    assert cross < 0.35, "distinct languages must not collide: %.3f" % cross
    print("holographic langid selftest OK (self %.3f, cross %.3f)" % (same, cross))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        d = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
        ng = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        run(dim=d, n=ng)

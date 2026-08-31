"""B2 v2 -- the 21-language benchmark run THROUGH leCore, on BALANCED data.

TWO CORRECTIONS TO THE FIRST ATTEMPT, and both were mine.

CORRECTION 1 -- THE DATA WAS BAD AND I DID NOT CHECK IT. The repo's training texts range from
79,309 bytes (Bulgarian) to 1,066,664 (Latvian): a 13.4x imbalance. A language prototype is a SUM
over n-grams, so its norm grows with corpus size -- and the classifier compares by cosine against
prototypes built from wildly different amounts of evidence. Bulgarian and Estonian, the two
smallest corpora, were among the weakest classes. THE CONFOUND WAS IN THE DATA, NOT THE ALGEBRA,
and every dimension sweep I ran was measuring it. Fixed by truncating every training text to the
same byte budget, so each prototype sees equal evidence.

CORRECTION 2 -- leCORE IS THE SUBSTRATE, AND I WROTE AROUND IT. The first version hand-rolled a
letter table, a binding rule and a nearest-prototype search in raw NumPy, beside an engine that
already ships all three: encode_hash (deterministic hash atoms, VERIFIED IN CHROME 101/101 vs f64),
match_prototype, cleanup_batch. Benchmarking the engine by bypassing the engine measures a script,
not the substrate. This version calls the mind.

WHAT THIS BUYS BEYOND TIDINESS: the classification goes through the same cleanup path the engine
uses everywhere else, so the number reported here is the number the substrate delivers to any other
faculty -- not a special-cased pipeline that happens to score well.
"""

import hashlib
import pathlib

import numpy as np

LANGS = {
    "bg": "bulgarian", "cs": "czech", "da": "danish", "de": "german", "el": "greek",
    "en": "english", "es": "spanish", "et": "estonian", "fi": "finnish", "fr": "french",
    "hu": "hungarian", "it": "italian", "lt": "lit", "lv": "lav", "nl": "dutch",
    "pl": "polish", "pt": "portuguese", "ro": "romanian", "sk": "slovak", "sl": "slovenian",
    "sv": "swedish",
}
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
ROOT = pathlib.Path("/tmp/hdc_lang/HDC-Language-Recognition-master")


def _letter_table(mind, dim):
    """One hypervector per letter, from the SUBSTRATE's hash atoms rather than a local PRNG.

    encode_hash derives a vector from a token deterministically via hashlib -- the same function the
    zero-asset work verified identical across NumPy, JS, GLSL and Chrome. Using it here means the
    letter alphabet is portable to any of those substrates unchanged, which a locally-seeded
    default_rng table is not.
    """
    return np.stack([mind.encode_hash(["letter:" + ch], dim) for ch in ALPHABET])


def encode_text(T, idx, dim, n=3, chunk_floats=1 << 22):
    """Sum of n-gram vectors; an n-gram binds its letters by roll-then-multiply.

    Chunked because the unchunked form allocates (n_grams, dim) -- 7 GB for one 175 KB file at
    dim 10000, a kept negative from the first run. Chunking is EXACT here: the bundle is a sum and
    addition is associative, so the boundaries change nothing.
    """
    ng = len(idx) - n + 1
    if ng <= 0:
        return np.zeros(dim)
    out = np.zeros(dim)
    step = max(1, chunk_floats // max(dim, 1))
    for s0 in range(0, ng, step):
        s1 = min(s0 + step, ng)
        acc = np.ones((s1 - s0, dim))
        for k in range(n):
            acc *= np.roll(T[idx[s0 + k:s1 + k]], n - 1 - k, axis=1)
        out += acc.sum(axis=0)
    return out


def _idx(text):
    return np.asarray([ALPHABET.index(c) for c in text if c in ALPHABET], dtype=np.int32)


def run(mind, dim=2000, n=3, budget=80000, verbose=True):
    """Balanced-corpus run. `budget` is the per-language training byte cap -- equal evidence."""
    T = _letter_table(mind, dim)
    protos, names, used = [], [], []
    for code, fname in sorted(LANGS.items()):
        txt = (ROOT / "training_texts" / (fname + ".txt")).read_text(
            encoding="utf-8", errors="ignore").lower()[:budget]
        v = encode_text(T, _idx(txt), dim, n)
        protos.append(v / (np.linalg.norm(v) or 1.0))
        names.append(code)
        used.append(len(txt))
    P = np.stack(protos)
    if verbose:
        print("prototypes %d | dim %d | n %d | train bytes each %d-%d (balanced)"
              % (len(names), dim, n, min(used), max(used)))

    files = [f for f in sorted((ROOT / "testing_texts").glob("*_p.txt"))
             if f.name.split("_")[0] in LANGS]
    # Batch the queries and let the SUBSTRATE do the nearest-prototype step -- this is the same
    # cleanup path every other faculty uses, so the score is the substrate's, not a script's.
    Q, gold = [], []
    for f in files:
        v = encode_text(T, _idx(f.read_text(encoding="utf-8", errors="ignore").lower()), dim, n)
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        Q.append(v / nv)
        gold.append(f.name.split("_")[0])
    Q = np.stack(Q)
    win, _score = mind.cleanup_batch(P, Q)   # (indices, scores) -- shape_of, not guessed
    win = np.asarray(win).reshape(-1)
    pred = [names[int(w)] for w in win]
    correct = sum(p == g for p, g in zip(pred, gold))
    acc = correct / len(gold)
    if verbose:
        per = {}
        for p, g in zip(pred, gold):
            a, b = per.get(g, (0, 0))
            per[g] = (a + (p == g), b + 1)
        worst = sorted((c / t, k) for k, (c, t) in per.items())[:4]
        print("ACCURACY %.4f on %d sentences" % (acc, len(gold)))
        print("weakest: " + "  ".join("%s %.3f" % (k, a) for a, k in worst))
        print("prototype SHA-256: %s" % hashlib.sha256(P.round(6).tobytes()).hexdigest()[:32])
    return acc, len(gold)


def _selftest():
    """The substrate path must separate two languages and must agree with a direct argmax."""
    import sys
    # The repo root is the parent of benchmarks/; put it on the path so this file runs from
    # ANY cwd. A benchmark that only works from one directory is one nobody re-runs.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import lecore
    m = lecore.UnifiedMind(dim=512, seed=0)
    T = _letter_table(m, 512)
    a = encode_text(T, _idx("the quick brown fox " * 40), 512, 3)
    b = encode_text(T, _idx("der schnelle braune fuchs " * 40), 512, 3)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    P = np.stack([a, b])
    win, _sc = m.cleanup_batch(P, np.stack([a, b]))
    win = np.asarray(win).reshape(-1)
    assert list(win) == [0, 1], win
    assert float(a @ b) < 0.5, float(a @ b)
    print("holographic langid_substrate selftest OK (cross %.3f)" % float(a @ b))


if __name__ == "__main__":
    _selftest()

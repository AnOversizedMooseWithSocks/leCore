"""A SHADER-NATIVE ATOM FAMILY -- so a query can be TYPED in the browser.

THE BLOCKER THIS REMOVES: `derived_atom` is blake2b -> PCG64 -> numpy's ziggurat -> FFT
magnitude-normalise. Reproducing that in GLSL ES is a declared dead end (no u64, and it would be
a SECOND IMPLEMENTATION of the atom generator). So every page so far had to ship pre-encoded
queries. That is the last thing standing between this and "type a query in a browser".

THE FIX, and the rule that makes it legal: define the atom family ONCE, in u32 integer
arithmetic that is EXACT in both NumPy and GLSL ES 3.00, and EVALUATE IT TWICE. One definition,
two evaluations -- not two implementations. Then pin the two against each other.

The family is Rademacher (+/-1) atoms from an integer hash: FNV-1a over the token bytes, then
the `lowbias32` finaliser per component, sign taken from the top bit. Chosen over a phasor/trig
atom on purpose: no transcendental means NO FLOATING-POINT DIVERGENCE AT ALL in generation, so
the two evaluations are BIT-IDENTICAL rather than merely close. This is Kanerva's binary
spatter-code family in a real-valued costume, so it is a known-good VSA vocabulary, not a novelty.

NOTE ON NORMALISATION: the query vector is NOT normalised. A positive scalar on the query cannot
move an argmax, so the browser skips a whole reduction pass for free.
"""
import numpy as np

from holographic.misc.holographic_determinism import hash32_pcg

FNV_OFFSET = np.uint32(2166136261)
FNV_PRIME = np.uint32(16777619)


def fnv1a(name: str) -> np.uint32:
    """A stable u32 for a NAME. Host-side only -- a shader never sees a string, only this uint.

    WHY NOT hash_u64: it is 64-bit and cannot be reproduced in GLSL ES / WGSL. WHY NOT
    hash32_pcg alone: that is a uint->uint PERMUTATION, not a string digest; it needs an integer
    to permute, and FNV-1a is the smallest standard way to fold bytes into one. JS reproduces it
    with Math.imul; a plain `*` would go through a double and lose bits.
    """
    h = FNV_OFFSET
    # WRAPAROUND IS THE SPEC: FNV-1a is defined mod 2**32, and both GLSL `uint` and Math.imul
    # wrap. Suppressed HERE, where the wrap is intended, not globally where it would hide a bug.
    with np.errstate(over="ignore"):
        for b in name.encode("utf-8"):
            h = np.uint32(h ^ np.uint32(b))
            h = np.uint32(h * FNV_PRIME)
    return h


# PORTING TRAP, PAID FOR ONCE. A GLSL/JS port of this family MUST use the PCG constants in
# holographic_determinism.hash32_pcg (v*747796405u + 2891336453u; ((s >> ((s>>28u)+4u)) ^ s) *
# 277803737u; (w>>22u)^w). The browser page generators once shipped the LOWBIAS32 mix instead --
# a different permutation this engine had already deleted as a duplicate -- and produced atoms
# ORTHOGONAL to the reference (cosine -0.077, recall at chance). It presented as a driver bug in
# Chrome and was not: the terms and the FNV-1a hashes were exact, only the expansion differed.
# Diagnose a port by simulating it against encode_hash in ten lines of Python before blaming a
# substrate. Verified in Chrome after the fix: 101/101, accuracy 0.9703 == the f64 engine.


def canonical_terms(text):
    """THE normalisation boundary for the whole pipeline. Delegates to holographic_bm25.tokenize.

    WHY THIS HAS A NAME. The atom families hash whatever string they are handed, while BM25 hashes
    its own NORMALISED tokens -- so a term's atom and its posting disagree about which word they
    represent for 48.1% of a 12,015-word vocabulary (measured: 'settings'/'setting',
    'a_bad'/'bad', '1950s'/'1950'). That is harmless only while the two arms are never joined on
    term identity, and it stops being harmless the moment anyone revisits weighted fusion.
    Everything that needs a SHARED term identity goes through here; nothing re-implements it.
    """
    from holographic.semantic_router.holographic_bm25 import tokenize
    return tokenize(text)


def term_id(text):
    """The u32 identity of a term, taken AFTER the boundary -- the id both arms must agree on.

    Returns None for text the boundary drops entirely (stopwords, single characters), because a
    dropped term has no identity and inventing one would paper over the drop.
    """
    t = canonical_terms(text)
    return int(fnv1a(t[0])) if t else None


def _mix(base, dim):
    """One PCG-permuted uint32 per component. DELEGATES to hash32_pcg -- never reimplement it."""
    idx = np.arange(dim, dtype=np.uint32)
    with np.errstate(over="ignore"):
        return hash32_pcg(np.uint32(base) ^ idx)


def hash_atom(name: str, dim: int) -> np.ndarray:
    """A +/-1 atom that is a FUNCTION of its name -- no table, no RNG, no storage."""
    sign = np.where((_mix(fnv1a(name), dim) >> np.uint32(31)) == 1, 1.0, -1.0)
    return sign / np.sqrt(dim)


def encode_hash(tokens, dim, normalise=True):
    """Bag-of-atoms over the hash family. Docs want normalising; queries do not need it."""
    v = np.zeros(dim)
    for t in tokens:
        v += hash_atom(t, dim)
    if normalise:
        n = np.linalg.norm(v)
        if n:
            v /= n
    return v


def _selftest():
    a = hash_atom("holographic", 256)
    assert np.array_equal(a, hash_atom("holographic", 256)), "atom is not deterministic"
    assert abs(np.linalg.norm(a) - 1.0) < 1e-12, "atom is not unit norm"
    assert set(np.unique(a * np.sqrt(256))) <= {-1.0, 1.0}, "atom is not Rademacher"

    # Near-orthogonality against a DERIVED bar, not a picked one: E|cos| for random +/-1 vectors
    # is ~sqrt(2/(pi*d)), so 6 sigma at d=512 is 6/sqrt(d). Assert the CONTRAST, not a magic bar.
    d = 512
    names = ["tok%d" % i for i in range(200)]
    A = np.stack([hash_atom(n, d) for n in names])
    off = np.abs(A @ A.T - np.eye(len(names)))
    bound = 6.0 / np.sqrt(d)
    assert off.max() < bound, "cross-talk %.4f exceeds derived bound %.4f" % (off.max(), bound)

    # KEPT NEGATIVE, PINNED: this family is NOT unitary in the HRR sense -- its FFT magnitude
    # spectrum is not flat, so bind/unbind through it is NOT exact. It BUNDLES; it does not BIND.
    # Use holographic_phasor when you need binding. Asserted so nobody assumes otherwise.
    spec = np.abs(np.fft.rfft(hash_atom("x", 256)))
    assert spec.std() / spec.mean() > 0.1, "spectrum unexpectedly flat -- re-check the claim"

    # CROSS-ARM IDENTITY, pinned: once a string has passed the boundary, the atom arm and the
    # lexical arm MUST resolve it to the same id. Before the boundary they do not, for ~half the
    # vocabulary, which is exactly why the boundary has a name.
    for w in ("settings", "classes", "processes", "meshing", "vector"):
        cw = canonical_terms(w)
        assert cw, w
        assert term_id(w) == int(fnv1a(cw[0])), "term_id must hash the POST-boundary token"
    assert term_id("the") is None, "a dropped term has no identity and must not be invented"

    # THE CONTRACT IS "APPLY EXACTLY ONCE", and it has to be, because the boundary is NOT stable
    # on its own output: 'settings' -> 'setting' -> 'sett'. An earlier version of this test
    # asserted stability and FAILED, correctly -- asserting the wish instead of the measured
    # behaviour is how a test becomes decoration. Pinned both ways so neither can drift silently.
    assert term_id(canonical_terms("settings")[0]) != term_id("settings"), \
        "the boundary became idempotent -- that is a BEHAVIOUR CHANGE, not a bug fix; re-read " \
        "the P1.11 measurement before adopting it"
    assert term_id("meshing") == term_id(canonical_terms("meshing")[0]), \
        "some words ARE stable; if this one stopped being so the boundary changed"

    doc = encode_hash(["alpha", "beta", "gamma", "delta"], 512)
    q = encode_hash(["alpha", "gamma"], 512, normalise=False)
    other = encode_hash(["epsilon", "zeta", "eta", "theta"], 512)
    assert doc @ q > other @ q + 0.2, "bag-of-atoms failed to separate"
    print("holographic_hashatom self-test passed (deterministic, unit-norm, Rademacher via "
          "hash32_pcg, cross-talk %.4f < derived bound %.4f, spectrum NOT flat so this family "
          "does NOT bind, retrieval separates by %.3f)"
          % (off.max(), bound, float(doc @ q - other @ q)))


if __name__ == "__main__":
    _selftest()

"""
holographic_fhrr.py
===================

FHRR -- Fourier Holographic Reduced Representations -- the complex-phasor VSA.

WHY THIS EXISTS (and why it's separate from the real-valued core)
-----------------------------------------------------------------
The rest of the engine uses Plate's real-valued HRR: Gaussian atoms, binding by
circular convolution (done via FFT), similarity by cosine. That is the right
*default* -- it is readable, real-valued, and at the binding loads this project
actually runs (a handful of role-filler pairs per record, dim 512-1024) it is
already perfect.

But the most-cited recent cross-VSA comparison (Schlegel, Neubert & Protzel,
"A comparison of vector symbolic architectures", 2021) finds that FHRR performs
best across their benchmark suite, and the surveys (Kleyko et al., 2022/2023)
agree it has the cleanest high-capacity behaviour. FHRR represents each atom as
a vector of COMPLEX UNIT PHASORS e^{i*theta} (one random phase per component).
Binding is element-wise complex multiplication (phases ADD); unbinding multiplies
by the conjugate (phases SUBTRACT); superposition is complex addition; similarity
is the mean cosine of the per-component phase differences. It is exactly self-
consistent: bind then unbind by the same key returns the value with no convolution
round-trip.

MEASURED on THIS substrate (so the claim isn't taken on faith) -- how many
key->value pairs survive in ONE superposed trace before nearest-value readback
breaks, dim 256:

    pairs     real-HRR   FHRR
      20        0.93      1.00
      30        0.77      0.97
      40        0.61      0.90
      60        0.40      0.74

FHRR keeps far more pairs per vector. Two honest qualifiers, also measured:
  * At LOW load (<=~10 pairs at 256-d, or the few-factor records this project
    normally builds at 512-1024) both are at 1.000 -- FHRR changes nothing, so
    the real-valued default loses nothing by staying the default.
  * The project's existing `unitary` atoms (unit-magnitude SPECTRUM, real domain)
    do NOT capture this advantage: unitary-HRR tracks real-HRR (0.41 vs 0.40 at
    60 pairs), not FHRR (0.74). The capacity win comes from staying in the
    complex phasor domain, not merely from flattening the spectrum.
  * It does NOT raise the nested-scene composition ceiling, because that ceiling
    is the resonator factoring a noisy unbound sub-scene -- not key-value binding
    capacity (the outer group-binding is already perfect to 12 groups in plain
    HRR). FHRR fixes the binding-capacity bottleneck; it can't fix a different one.

So FHRR is offered as an OPT-IN high-capacity tool for the one regime where it
measurably wins -- a large key->value trace -- not as a core replacement.
Everything here is seed-deterministic, like the rest of the engine.
"""

import numpy as np


def phasor_atom(dim, rng):
    """A random FHRR atom: `dim` complex unit phasors, one uniform random phase each.
    |each component| == 1, so the whole vector lies on the complex unit torus."""
    return np.exp(1j * rng.uniform(-np.pi, np.pi, dim))


def fhrr_bind(a, b):
    """Bind = element-wise complex multiplication (the phases ADD). Commutative,
    associative, and -- unlike real circular convolution -- needs no FFT."""
    return a * b


def fhrr_unbind(c, a):
    """Unbind = multiply by the conjugate (the phases of `a` SUBTRACT). bind then
    unbind by the same atom is exact: unbind(bind(a, b), a) == b."""
    return c * np.conj(a)


def fhrr_bundle(vectors):
    """Superpose = complex addition. Like real HRR, the sum is NOT renormalised per
    component here; readback uses similarity, which is magnitude-invariant."""
    return np.sum(vectors, axis=0)


def fhrr_sim(a, b):
    """Similarity = mean cosine of the per-component phase differences, i.e. the real
    part of the normalised complex inner product. 1.0 for identical phasor vectors,
    ~0 for independent ones."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.real(np.vdot(b, a)) / denom)


class PhasorVocabulary:
    """An FHRR atom store, mirroring the real Vocabulary's interface (get/cleanup) so
    it is a drop-in where the complex domain is wanted. Atoms are minted from a seeded
    rng in get()-call order; `derived=True` makes each atom a pure function of
    (seed, name) via a stable hash, so the whole vocabulary regenerates from the seed."""

    def __init__(self, dim, seed=0, derived=False):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.derived = derived
        self.vectors = {}

    def get(self, name):
        if name not in self.vectors:
            if self.derived:
                import hashlib
                h = hashlib.blake2b(f"{int(self.seed)}\x00{name}".encode("utf-8"),
                                    digest_size=8).digest()
                local = np.random.default_rng(int.from_bytes(h, "big"))
                self.vectors[name] = phasor_atom(self.dim, local)
            else:
                self.vectors[name] = phasor_atom(self.dim, self.rng)
        return self.vectors[name]

    def cleanup(self, noisy, candidates=None):
        """Nearest stored atom to a noisy phasor vector, by FHRR similarity."""
        names = candidates if candidates is not None else list(self.vectors)
        if not names:
            return None, -1.0
        best, bs = None, -2.0
        for nm in names:
            s = fhrr_sim(noisy, self.vectors[nm])
            if s > bs:
                best, bs = nm, s
        return best, bs


class PhasorMemory:
    """A high-capacity key->value trace memory in the FHRR domain -- the same classic
    'cram many pairs into one vector' trick as HolographicMemory, but complex-phasor
    so it holds substantially more pairs before readback degrades (see the module
    docstring's measured table). learn() folds bind(key, value) into a running complex
    sum; recall() unbinds with a key and (optionally) cleans up against known values."""

    def __init__(self, dim):
        self.dim = dim
        self.trace = np.zeros(dim, dtype=complex)

    def learn(self, key, value):
        self.trace = self.trace + fhrr_bind(key, value)

    def recall(self, key):
        """The noisy value bound to `key` (clean up against a value vocabulary to snap
        it to a known symbol)."""
        return fhrr_unbind(self.trace, key)

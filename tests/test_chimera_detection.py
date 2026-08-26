"""A4: chimera detection -- and WHY the obvious detector cannot work.

A chimera is the VSA failure mode the literature calls "variable confusion": a decode that
cross-pairs fillers from two different stored facts. Our own fact_capacity measurement found
it to be the D-independent failure mode of a bundled fact base.

THE NEGATIVE FIRST, because it is the load-bearing result: verification-by-reconstruction
(re-encode the decoded atom, check its overlap with the trace) CANNOT detect chimeras, and
not because the threshold is hard to tune -- because THE EVIDENCE IS NOT IN THE ENCODING.
encode_atom binds each argument to its ROLE and never to the other arguments, so a chimera's
slots are both genuinely present in the superposition. Measured: 0.7194 genuine vs 0.7080
chimera at load 2, a gap of 0.011 that shrinks with load.

THE FIX IS AN ENCODING CHANGE, not a smarter test: bind a per-fact NONCE into every slot, so
slots from different facts carry different nonces and a cross-pairing no longer reconstructs.
Measured: 0.705 vs 0.359 -- a 2x separation, at every load tested.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import derived_atom, bind, bundle

D = 2048


def _sym(n):
    return derived_atom(0, "a4:" + n, D)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _enc(pred, args, nonce=None):
    parts = []
    for i, a in enumerate(args):
        slot = bind(_sym("role_%d" % i), _sym(a))
        if nonce is not None:
            slot = bind(slot, _sym("nonce_%d" % nonce))
        parts.append(slot)
    return bind(_sym("pred_" + pred), bundle(np.stack(parts)))


def _trial(load, nonce, seed):
    rng = np.random.default_rng(seed)
    S = ["s%d" % i for i in range(24)]
    facts = [(S[int(rng.integers(24))], S[int(rng.integers(24))]) for _ in range(load)]
    T = np.sum([_enc("p", f, nonce=(i if nonce else None))
                for i, f in enumerate(facts)], axis=0)
    gen, chi = [], []
    for i, f in enumerate(facts):
        gen.append(_cos(_enc("p", f, nonce=(i if nonce else None)), T))
    for i in range(load):
        ch = (facts[i][0], facts[(i + 1) % load][1])
        if ch in facts:
            continue
        chi.append(_cos(_enc("p", ch, nonce=(i if nonce else None)), T))
    return gen, chi


def test_reconstruction_alone_cannot_detect_chimeras():
    """THE KEPT NEGATIVE. Pinned so nobody 'fixes' the threshold: the separation is absent
    because the encoding carries no evidence, not because the test is badly calibrated."""
    gen, chi = [], []
    for t in range(12):
        g, c = _trial(4, nonce=False, seed=700 + t)
        gen += g
        chi += c
    assert abs(np.mean(gen) - np.mean(chi)) < 0.05, "separation appeared -- investigate"


def test_a_per_fact_nonce_makes_chimeras_detectable():
    """The fix is an ENCODING change. Binding a per-fact nonce into every slot means a
    cross-pairing no longer reconstructs, and the gap becomes thresholdable at every load."""
    for load in (2, 4, 8):
        gen, chi = [], []
        for t in range(12):
            g, c = _trial(load, nonce=True, seed=700 + t)
            gen += g
            chi += c
        assert np.mean(gen) > 1.7 * np.mean(chi), (load, np.mean(gen), np.mean(chi))

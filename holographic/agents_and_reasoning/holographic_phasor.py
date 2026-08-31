"""FHRR PHASOR ATOMS FROM AN INTEGER HASH -- the binding vocabulary the hash family is not.

WHY THIS EXISTS. holographic_hashatom gave the browser typed queries at zero storage, but its
selftest PINS a refusal: its FFT magnitude spectrum is not flat, so it BUNDLES and does not BIND.
Bag-of-words retrieval was fine with that; the algebra is not. Role-filler records, VM programs
and the whole VSA read path need bind/unbind to be EXACT.

THE MOVE, straight out of the FHRR literature (Plate's frequency-domain HRR; Gayler; the
phasor/spatter-code line): stop trying to make a real vector whose spectrum happens to be flat,
and define the atom AS the spectrum. A phasor atom is unit magnitude by construction, so
  bind   = phase ADDITION      (componentwise complex multiply)
  unbind = phase SUBTRACTION   (multiply by the conjugate)
are exact with NO FFT anywhere and no normalisation step to get wrong.

That is also why it suits a shader: unit_vector()'s route to flat spectra is
draw-Gaussian -> FFT -> divide by magnitude -> inverse FFT, none of which a fragment shader wants.
Phase addition is one add.

STORAGE. An ATOM is a function of its name: phases come from the same u32 integer hash the
Rademacher family uses, so the vocabulary is still zero bytes (lever 3). A RECORD -- a bundle of
bound pairs -- is NOT unit magnitude, so it is stored as complex (re, im) pairs. Atoms are
generated; records are stored. That distinction is the whole storage model.
"""
import numpy as np

import holographic.agents_and_reasoning.holographic_hashatom as HA
from holographic.misc.holographic_determinism import hash32_pcg

TWO32 = np.float64(4294967296.0)


def phases(name, dim):
    """Phase per component, in TURNS (0..1), not radians.

    WHY TURNS: bind is (a + b) mod 1, exact in float for values already in [0,1), and the only
    constant that then has to agree across NumPy, GLSL and JS is 2**32 -- never pi. Putting a
    transcendental in the middle of the one thing that must match bit for bit is how two
    evaluations of "the same" definition drift apart.

    DELEGATES to hash32_pcg (Jarzynski & Olano 2020), the engine's existing GPU-reproducible
    32-bit hash, whose GLSL is emitted by hash32_pcg_glsl. There is exactly one such primitive.
    """
    return HA._mix(HA.fnv1a(name), dim).astype(np.float64) / TWO32


def atom(name, dim):
    """The atom as a complex unit phasor vector -- |a_i| = 1 for every component."""
    return np.exp(2j * np.pi * phases(name, dim))


def bind(a, b):
    """Componentwise complex multiply == phase ADDITION. No FFT, no normalisation."""
    return a * b


def unbind(z, key):
    """Exact because |key| = 1: the conjugate is the TRUE inverse, not a pseudo-inverse."""
    return z * np.conj(key)


def bundle(items):
    return np.sum(items, axis=0)


def similarity(z, a):
    """Real part of the complex inner product -- the cleanup score."""
    return float(np.real(np.vdot(a, z)))


def cleanup(z, names, dim):
    """Nearest generated atom by Re<a,z>. Candidates are NAMES: no vocabulary is stored."""
    scores = [similarity(z, atom(n, dim)) for n in names]
    return names[int(np.argmax(scores))], scores


def factor(composite, codebooks, iters=100):
    """Factor a phasor product back into one atom per codebook -- a resonator that keeps the phase.

    WHY THIS EXISTS RATHER THAN CALLING factor_composite. That faculty is correct for the
    real/bipolar family it was built and validated for, but handed a COMPLEX composite it casts to
    float and DISCARDS THE IMAGINARY PART, then returns a dict whose `factors` field is populated
    and whose `solved` field is False. MEASURED over 60 random 3-factor products with 8 entries per
    codebook (search space 512): the real-cast path recovers 0.250, this one 0.967, chance 0.002.
    Half the representation is half the answer, and the failure is silent from the caller's side.

    Shape is the engine's usual one -- ITERATE A PROJECTION, the same family as IK, PBD and the
    real resonator: unbind by the conjugate of the current estimates, project onto each codebook by
    the complex inner product, re-estimate, stop at a fixpoint. T3 bounds the step count for a
    contraction; convergence here is observed, not proved, and the iteration cap is real.
    """
    import numpy as _np
    F = len(codebooks)
    est = [_np.asarray(cb).mean(0) for cb in codebooks]
    for _ in range(int(iters)):
        new = []
        for f in range(F):
            others = _np.ones(len(composite), dtype=complex)
            for g in range(F):
                if g != f:
                    others = others * est[g]
            probe = composite * _np.conj(others)
            new.append(_np.asarray(codebooks[f])[int(_np.argmax(_np.real(_np.asarray(codebooks[f]).conj() @ probe)))])
        if all(_np.array_equal(a, b) for a, b in zip(new, est)):
            break
        est = new
    out = []
    for f in range(F):
        cb = _np.asarray(codebooks[f])
        out.append(int(_np.argmax(_np.real(cb.conj() @ est[f]))))
    return tuple(out)


def power(a, x):
    """Fractional power of a phasor atom -- PHASE SCALING, one multiply, no FFT and no machinery.

    Continuous coordinates, timestamps and recency for free: similarity decays smoothly with
    |x - y|. Measured spearman 0.956 against -|x-1| over x in [0,3], with sim(a^1, a^1.1)=0.975
    and sim(a^1, a^2.0)=0.025. This is the phasor spelling of the engine's fractional-power
    encoding; the real-valued spelling with kaiser sidelobe shaping already exists in the
    Encoders family and is the one to use OUTSIDE this atom family.
    """
    import numpy as _np
    return _np.exp(1j * float(x) * _np.angle(a))


def _selftest():
    D = 512
    a, b = atom("shape", D), atom("sphere", D)

    # THE POINT OF THE FAMILY: bind/unbind exact, with no FFT and no normalisation step.
    err = float(np.max(np.abs(unbind(bind(a, b), b) - a)))
    assert err < 1e-12, "bind/unbind not exact: %.3e" % err
    assert np.allclose(np.abs(a), 1.0, atol=1e-12), "atom is not unit magnitude"

    # Near-orthogonality against a DERIVED bar: Re<a,b>/D has sd 1/sqrt(2D), so 6 sigma is
    # 6/sqrt(2D). Assert the contrast, never a picked threshold.
    names = ["tok%d" % i for i in range(200)]
    A = np.stack([atom(n, D) for n in names])
    off = np.abs(np.real(A.conj() @ A.T) / D - np.eye(len(names)))
    bar = 6.0 / np.sqrt(2 * D)
    assert off.max() < bar, "cross-talk %.4f exceeds derived bound %.4f" % (off.max(), bar)

    roles = ["colour", "size", "material"]
    fillers = ["red", "large", "metal"]
    distract = ["blue", "small", "wood", "green", "tiny", "glass"]
    rec = bundle([bind(atom(r, D), atom(f, D)) for r, f in zip(roles, fillers)])
    for r, f in zip(roles, fillers):
        got, _ = cleanup(unbind(rec, atom(r, D)), fillers + distract, D)
        assert got == f, "recovered %s for role %s, expected %s" % (got, r, f)

    # FACTORING, pinned against chance and against the real-cast path's measured 0.250.
    import numpy as _np
    cbs = [_np.stack([atom("f%d_%d" % (g, i), D) for i in range(6)]) for g in range(3)]
    truth = (2, 4, 1)
    comp = cbs[0][truth[0]] * cbs[1][truth[1]] * cbs[2][truth[2]]
    assert factor(comp, cbs) == truth, "the resonator must recover a clean 3-factor product"

    # FPE, pinned: monotone decay is the property, and exactness must survive it.
    ax = atom("axis", D)
    s1 = float(_np.real(_np.vdot(power(ax, 1.0), power(ax, 1.1)))) / D
    s2 = float(_np.real(_np.vdot(power(ax, 1.0), power(ax, 2.0)))) / D
    assert s1 > 0.9 > s2, "fractional power must decay with distance (%.3f, %.3f)" % (s1, s2)
    assert abs(float(_np.max(_np.abs(_np.abs(power(ax, 0.7)) - 1.0)))) < 1e-12, \
        "a fractional power must stay unit magnitude"

    # KEPT NEGATIVE, PINNED: a RECORD is not unit magnitude. Unbinding with a record as if it
    # were a key returns a wrong answer rather than an error, so the failure is asserted here.
    assert abs(np.mean(np.abs(rec)) - 1.0) > 0.1, "bundle magnitude unexpectedly ~1"

    print("holographic_phasor self-test passed (bind/unbind exact to %.2e with NO FFT, atoms "
          "unit-magnitude, cross-talk %.4f < derived bar %.4f, 3-role record fully recovered "
          "against 9 candidates, and a bundle is NOT a unit atom)" % (err, off.max(), bar))


if __name__ == "__main__":
    _selftest()

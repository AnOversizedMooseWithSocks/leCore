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
import json

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


def factor(composite, codebooks, iters=100, tolerant=False, restarts=20, patience=8, accept_p=0.01, m_null=100,
           seed=0):
    """Factor a phasor product back into one atom per codebook -- a resonator that keeps the phase.

    tolerant=False (the default) is the ORIGINAL path, bit for bit: one deterministic run from the codebook means,
    the hard-snap update, a fixpoint exit, a tuple of indices. tolerant=True returns a DICT with an agreement exit
    and a procedure-matched null p-value -- see factor_tolerant (the CLM backlog's resonator fix, wave 2).

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
    if tolerant:
        return factor_tolerant(composite, codebooks, iters=iters, restarts=restarts, patience=patience,
                               accept_p=accept_p, m_null=m_null, seed=seed)
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


# ---------------------------------------------------------------------------------------------------------------
# THE TOLERANT PHASOR RESONATOR (CLM backlog, wave 2 -- the pattern of holographic_resonator.factor(tolerant=True)
# and holographic_sbc.resonator_confidence, carried to the complex family).
#
# THE DEFECT, MEASURED: the default path above solves 0 of 20 EXACT products at 20 x 50 x 50 (D=2048, a search
# space of 50,000) -- one deterministic run from the codebook means, and the hard snap (each factor jumps to ONE
# codebook entry per sweep) parks on a wrong fixpoint and stops. Giving that same snap rule 20 random restarts and
# the agreement exit still solved only 1 of 20 (KEPT NEGATIVE: the exits are not the fix on their own). What
# solves it is the classic FHRR resonator update (Frady et al. 2020): project onto the codebook by SUPERPOSITION --
# v = X^T (X^H probe), every entry weighted by its similarity -- and phase-normalise v to unit magnitude. Then:
#   * AGREEMENT  Re<product of the picked atoms, composite> / (sqrt(D) |composite|): exactly 1.0 for an exact
#                product, ~cos of the phase noise for a noisy one, ~N(0, 1/sqrt(2D)) for a random composite.
#   * EXITS      exact (agreement 1 - 1e-9: the certificate), ACCEPT (the best agreement beats what a structureless
#                composite reaches under this same procedure at the caller's false-alarm level accept_p), STALL
#                (no strict improvement for `patience` sweeps -- this is what makes a refusal cheap), budget.
#   * RESTARTS   restart 0 starts where the default path starts (the codebook means); restart r >= 1 from random
#                unit phasors (default_rng([seed, r])). Best-of-restarts, ties keep the earliest.
#   * p_value    the procedure-matched null: the same search (accept exit off) on m_null random unit-phasor
#                composites, cached per codebook SHAPE; p = (1 + #null >= agreement) / (m_null + 1).
# MEASURED (PYTHONHASHSEED=0 python -m holographic.agents_and_reasoning.holographic_phasor --bench: 20 x 50 x 50,
# D=2048, 20 trials per row, restarts 20 x 100 iters, patience 8, m_null 100, accept_p 0.01; 2 cores at load ~7):
#     phase noise s   before (tolerant=False)      after (tolerant=True)
#     0.0              0/20 right,  75 ms           20/20 right, 20 accepted, 0 wrong accepted,   60 ms
#     0.3              2/20 right,  86 ms           20/20 right, 20 accepted, 0 wrong accepted,   59 ms
#     0.6              0/20 right,  70 ms           20/20 right, 20 accepted, 0 wrong accepted,  189 ms
#     random composite  (returns a guess, no p)     0/20 accepted, median p 0.45, 1,229 ms to refuse
# KEPT NEGATIVES: (1) the null costs m_null full searches ONCE per codebook shape -- 126 s at this shape under
# load 7 (cached per process), the price of a procedure-matched p; (2) a refusal exhausts every restart to its
# stall (1.2 s) where the default path returned a wrong tuple in 70 ms; (3) the snap rule + these exits (no
# superposition) solved 1/20 -- the update rule, not the exit, was the defect at this shape.
# ---------------------------------------------------------------------------------------------------------------
_TOLERANT_NULL_CACHE = {}


def agreement(composite, codebooks, idx):
    """How well the picked atoms re-bind the composite: Re<prod_f codebooks[f][idx[f]], c> / (sqrt(D) |c|).
    1.0 for an exact product; ~N(0, 1/sqrt(2D)) for a composite with no structure."""
    c = np.asarray(composite, dtype=complex)
    rec = np.ones(len(c), dtype=complex)
    for f, i in enumerate(idx):
        rec = rec * np.asarray(codebooks[f])[int(i)]
    den = np.sqrt(len(c)) * float(np.linalg.norm(c))
    return float(np.real(np.vdot(rec, c)) / den) if den > 0 else 0.0


def _tolerant_run(c, cbs, init, iters, patience, accept_thr):
    """One restart of the superposition update with the tolerant exits -> (best idx, best agreement, t, exit)."""
    D, F = len(c), len(cbs)
    est = [np.asarray(e, dtype=complex) for e in init]
    best_a, best_idx, since = -2.0, None, 0
    for t in range(int(iters)):
        new = []
        for f in range(F):
            others = np.ones(D, dtype=complex)
            for g in range(F):
                if g != f:
                    others = others * est[g]
            v = cbs[f].T @ (cbs[f].conj() @ (c * np.conj(others)))      # superposition, weighted by similarity
            mag = np.abs(v)
            new.append(np.where(mag > 0, v / np.maximum(mag, 1e-300), 1.0 + 0j))   # back to unit phasors
        idx = tuple(int(np.argmax(np.real(cbs[f].conj() @ new[f]))) for f in range(F))   # ties -> lowest index
        a = agreement(c, cbs, idx)
        if a >= 1.0 - 1e-9:
            return idx, a, t, "exact"
        if a > best_a + 1e-12:                     # strict improvement only: a plateau counts toward the stall
            best_a, best_idx, since = a, idx, 0
        else:
            since += 1
        if accept_thr is not None and best_a > accept_thr:
            return best_idx, best_a, t, "accept"
        if since >= int(patience):
            return best_idx, best_a, t, "stall"
        est = new
    return best_idx, best_a, int(iters), "budget"


def _tolerant_search(c, cbs, restarts, iters, patience, accept_thr, seed):
    """Best-of-restarts -> (idx, agreement, restarts used, iterations of the returned run, exit)."""
    D = len(c)
    best = (None, -2.0, 0, 0, "budget")
    for r in range(int(restarts)):
        if r == 0:
            init = [cb.mean(0) for cb in cbs]           # the default path's own start
        else:
            rng = np.random.default_rng([int(seed), r])
            init = [np.exp(2j * np.pi * rng.random(D)) for _ in cbs]
        idx, a, t, ex = _tolerant_run(c, cbs, init, iters, patience, accept_thr)
        if ex == "exact":
            return idx, a, r + 1, t, "exact"
        if a > best[1] + 1e-12:                        # ties keep the EARLIEST restart (one stated tie rule)
            best = (idx, a, r + 1, t, ex)
        if ex == "accept":
            return idx, a, r + 1, t, "accept"
    return best[0], best[1], int(restarts), best[3], "budget"


def tolerant_null(codebooks, restarts=20, iters=100, patience=8, m=100, seed=0):
    """The procedure-matched null: the best agreement the tolerant search (accept exit OFF) reaches on m random
    unit-phasor composites, sorted. Cached per codebook SHAPE (D, sizes) and procedure -- the keying
    holographic_sbc / holographic_resonator measured content-independent for random atoms."""
    cbs = [np.asarray(cb, dtype=complex) for cb in codebooks]
    D = int(cbs[0].shape[1])
    sig = (D, tuple(int(cb.shape[0]) for cb in cbs), int(restarts), int(iters), int(patience), int(m), int(seed))
    if sig not in _TOLERANT_NULL_CACHE:
        rng = np.random.default_rng(int(seed) + 7919)
        out = np.empty(int(m))
        for i in range(int(m)):
            junk = np.exp(2j * np.pi * rng.random(D))
            out[i] = _tolerant_search(junk, cbs, restarts, iters, patience, None, seed)[1]
        _TOLERANT_NULL_CACHE[sig] = np.sort(out)
    return _TOLERANT_NULL_CACHE[sig]


def factor_tolerant(composite, codebooks, iters=100, restarts=20, patience=8, accept_p=0.01, m_null=100, seed=0):
    """Factor a phasor composite that may be NOISY: the superposition resonator with agreement exits and a p-value.

    Returns {factors, agreement, p_value, accepted (p_value <= accept_p), exit ('exact' | 'accept' | 'budget'),
    solved (exact reconstruction only -- its old meaning), restarts, iterations, search_space, null_max}. A
    composite with no structure comes back with a large p_value and accepted False: abstain on it."""
    c = np.asarray(composite, dtype=complex)
    cbs = [np.asarray(cb, dtype=complex) for cb in codebooks]
    null = tolerant_null(cbs, restarts=restarts, iters=iters, patience=patience, m=m_null, seed=seed)
    n = len(null)
    k_max = int(np.floor(float(accept_p) * (n + 1) - 1 + 1e-9))   # how many null values may reach the threshold
    thr = None if k_max < 0 else (-2.0 if k_max >= n else float(null[n - 1 - k_max]))
    idx, a, used, t, ex = _tolerant_search(c, cbs, restarts, iters, patience, thr, seed)
    p = float((1 + int((null >= a - 1e-12).sum())) / (n + 1))
    space = 1
    for cb in cbs:
        space *= int(cb.shape[0])
    return {"factors": idx, "agreement": float(a), "p_value": p, "accepted": bool(p <= float(accept_p)), "exit": ex,
            "solved": ex == "exact", "restarts": used, "iterations": t, "search_space": space,
            "null_max": float(null[-1])}


def _bench(trials=20, D=2048, sizes=(20, 50, 50), noises=(0.0, 0.3, 0.6)):
    """Before / after on the panel's shape (20 x 50 x 50, D=2048). Phase noise s: every component's phase is
    jittered by N(0, (s * pi / 2)^2). Prints and returns the rows (right, accepted, wrong-accepted, ms per call)."""
    import os
    import time as _time
    cbs = [np.stack([atom("f%d_%d" % (g, i), D) for i in range(n)]) for g, n in enumerate(sizes)]
    rng = np.random.default_rng(0)
    t0 = _time.time()
    tolerant_null(cbs)
    rows = {"null_build_s": round(_time.time() - t0, 1),
            "loadavg": open("/proc/loadavg").read().split()[:3] if os.path.exists("/proc/loadavg") else None}
    for s in noises:
        cases = []
        for _ in range(trials):
            tru = tuple(int(rng.integers(n)) for n in sizes)
            c = cbs[0][tru[0]] * cbs[1][tru[1]] * cbs[2][tru[2]]
            if s:
                c = c * np.exp(1j * s * np.pi * rng.standard_normal(D) / 2)
            cases.append((tru, c))
        t1 = _time.time()
        before = sum(factor(c, cbs) == tru for tru, c in cases)
        tb = (_time.time() - t1) / trials * 1000
        t1 = _time.time()
        got = [(tru, factor(c, cbs, tolerant=True)) for tru, c in cases]
        ta = (_time.time() - t1) / trials * 1000
        rows["noise_%.1f" % s] = {"before_right": int(before), "before_ms": round(tb, 1),
                                  "after_right": sum(g["factors"] == tru for tru, g in got),
                                  "after_accepted": sum(g["accepted"] for _, g in got),
                                  "after_wrong_accepted": sum(g["accepted"] and g["factors"] != tru for tru, g in got),
                                  "after_ms": round(ta, 1)}
    junk = [np.exp(2j * np.pi * rng.random(D)) for _ in range(trials)]
    t1 = _time.time()
    got = [factor(c, cbs, tolerant=True) for c in junk]
    rows["random"] = {"accepted": sum(g["accepted"] for g in got), "median_p": float(np.median([g["p_value"] for g in got])),
                      "after_ms": round((_time.time() - t1) / trials * 1000, 1)}
    print(json.dumps(rows, indent=1))
    return rows


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
    # THE TOLERANT PATH (wave 2): exact -> the certificate; a phase-jittered product -> right AND significant; a
    # structureless composite -> not accepted. A small null (m 19) keeps the selftest fast; accept_p 0.05 fits it.
    r = factor(comp, cbs, tolerant=True, m_null=19, accept_p=0.05)
    assert r["factors"] == truth and r["exit"] == "exact" and r["agreement"] > 0.999, r
    jit = comp * _np.exp(1j * 0.5 * _np.random.default_rng(1).standard_normal(D))
    r = factor(jit, cbs, tolerant=True, m_null=19, accept_p=0.05)
    assert r["factors"] == truth and r["accepted"] and r["p_value"] <= 0.05, r
    junk = _np.exp(2j * _np.pi * _np.random.default_rng(2).random(D))
    assert not factor(junk, cbs, tolerant=True, m_null=19, accept_p=0.05)["accepted"]

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
    import sys as _sys
    if "--bench" in _sys.argv:
        _bench()
    else:
        _selftest()

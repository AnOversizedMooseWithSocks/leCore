"""qFHRR-1 -- QUANTIZED integer-phase FHRR (holographic_qfhrr).

WHAT THIS IS
------------
A storage-and-exactness tier bolted onto the engine's existing complex FHRR (holographic_fhrr), not a new
algebra. The FHRR atom is a vector of complex unit phasors e^{i*theta}; all the information is in the
PHASES, and complex128 spends 128 bits per dimension carrying two floats of which one is redundant (the
magnitude is always 1). qFHRR stores the phase as an integer INDEX q in {0..K-1} instead:

    bind    = (q_a + q_b) mod K        exact integer addition
    unbind  = (q_c - q_a) mod K        exact integer subtraction
    similar = cosine lookup table + integer accumulation

At K=16 that is 4 bits per dimension against 128 -- a 96.9% reduction -- and binding becomes exact modular
integer arithmetic with no floating point anywhere. Following Snyder, Poursiami & Parsa (arXiv:2604.25939,
a PREPRINT, not peer-reviewed).

THE TWO PROPERTIES WORTH KNOWING, BOTH MEASURED HERE
----------------------------------------------------
1. UNBIND IS EXACT, WHICH THE REAL-VALUED PATH IS NOT. In real HRR the involution is a QUASI-inverse:
   unbind(bind(a,b),a) recovers b at cosine ~0.70 (measured in holographic_ntt, and unchanged by computing
   the convolution exactly). Here phases subtract exactly, so bind-then-unbind returns the ORIGINAL PHASE
   INDICES, bit for bit, at every K. That is a genuine algebraic difference between the two representations,
   not a numerical one -- and it is the strongest exactness result in the engine.

2. BUNDLING IS NOT CLOSED, AND THIS IS THE LOAD-BEARING CAVEAT. There is no integer operation on phase
   indices that superposes them. Superposition must LEAVE the representation: dequantize to Cartesian, sum
   the complex vectors, then re-quantize with atan2 and a round(). So the tier is exact for bind/unbind and
   APPROXIMATE for bundle, and `round()` at a bin boundary IS ITSELF A TIE-ARBITRATION POINT.

   THEREFORE, AND THIS IS THE REFUTATION THIS WHOLE RESEARCH LINE KEEPS ARRIVING AT: quantized VSA does NOT
   let the engine delete its tie-arbitration machinery. Exactness holds for binding and unbinding only.
   Anything that bundles or cleans up still needs the tie discipline (holographic_determinism.argmax_tiebreak).

3. THE PREPRINT'S FIDELITY TABLE IS REPRODUCIBLE -- BUT ONLY AGAINST A PHASE-ONLY REFERENCE, AND THAT
   MATTERS. The consolidation flagged those figures as unverified from the abstract. Re-measured here at
   D=1024 over 8 seeds, BIND fidelity matches almost exactly (paper 0.9497 / 0.9872 / 0.9999 at K=8/16/256;
   measured 0.9498 / 0.9875 / 0.9999). BUNDLE fidelity did NOT match -- until the reference was changed:

       K        vs COMPLEX bundle    vs PHASE of that bundle    paper
       8            0.8471                  0.9156              0.9147
       16           0.8808                  0.9738              0.9731
       256          0.8923                  0.9998              0.9997
       4096         0.8923                  1.0000                --

   So the paper's bundle column is measured PHASE-TO-PHASE. Against the actual complex bundle, fidelity
   SATURATES AT ~0.892 AND MORE PHASE LEVELS DO NOT HELP -- the ceiling is discarding the MAGNITUDE, which
   is independent of K. An engineer replacing a complex FHRR bundle with this tier loses ~11% similarity
   PERMANENTLY, and the preprint's 0.9997 does not describe that situation. Both numbers are reported by
   measure_fidelity so the distinction cannot quietly collapse back into the optimistic one.

WHERE IT PAYS
  The natural home is a path that binds and unbinds but never bundles -- the VM decode loop is exactly that
  shape. It is a STORAGE and DETERMINISM tier, not a capacity or speed one. Opt-in: the complex FHRR path
  and every real-valued default are untouched.

DETERMINISM
  Quantization is a fixed round-half-away-from-zero on a fixed grid, seeded RNG only for atom generation.
  No CORDIC: numpy's atan2 is used directly, which is deterministic on a given machine but IS floating
  point -- so the BUNDLE projection is the one place this tier does not inherit machine-independence.
  Bind, unbind and similarity are pure integer and are machine-independent.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_fhrr import phasor_atom

#: Default number of phase levels. 16 levels = 4 bits/dim, the knee of the measured fidelity curve
#: (bind fidelity ~0.987 against full complex FHRR; see measure_fidelity).
DEFAULT_LEVELS = 16


def quantize_phases(v, levels=DEFAULT_LEVELS):
    """Complex phasor vector -> integer phase indices in {0..levels-1}. The phase angle is mapped onto a
    uniform grid of `levels` bins. This is the lossy step and the ONLY lossy step in the representation:
    everything downstream of it is exact integer arithmetic."""
    v = np.asarray(v)
    if levels < 2:
        raise ValueError("need at least 2 phase levels, got %d" % levels)
    ang = np.angle(v)                                   # (-pi, pi]
    q = np.rint(ang * levels / (2.0 * np.pi)).astype(np.int64) % levels
    return q


def dequantize_phases(q, levels=DEFAULT_LEVELS):
    """Integer phase indices -> complex unit phasors, the inverse of `quantize_phases` up to the bin width
    that quantization already discarded. Used to hand a quantized vector back to the complex FHRR path."""
    q = np.asarray(q, dtype=np.int64)
    return np.exp(1j * (2.0 * np.pi * q / levels))


def qfhrr_atom(dim, rng, levels=DEFAULT_LEVELS):
    """A random quantized atom: a complex FHRR atom from the existing module, quantized. Delegates to
    holographic_fhrr.phasor_atom rather than minting its own randomness, so the two tiers draw from the
    same distribution and a quantized experiment is comparable with its complex baseline."""
    return quantize_phases(phasor_atom(dim, rng), levels)


def qfhrr_bind(qa, qb, levels=DEFAULT_LEVELS):
    """Bind two quantized atoms EXACTLY: phases add, so indices add mod `levels`. Pure integer, no floating
    point, identical on every machine. This is the operation the whole tier exists for."""
    return (np.asarray(qa, dtype=np.int64) + np.asarray(qb, dtype=np.int64)) % levels


def qfhrr_unbind(qc, qa, levels=DEFAULT_LEVELS):
    """Unbind EXACTLY: phases subtract, so indices subtract mod `levels`.

    Unlike the real-valued HRR path, this is a TRUE inverse, not a quasi-inverse -- qfhrr_unbind(qfhrr_bind(
    a, b), a) returns b's indices bit for bit, at every K. Real HRR recovers b at cosine ~0.70 and needs
    cleanup; this needs none. It is the strongest exactness guarantee in the engine, and it is bought by
    giving up closed bundling (see qfhrr_bundle)."""
    return (np.asarray(qc, dtype=np.int64) - np.asarray(qa, dtype=np.int64)) % levels


def qfhrr_sim(qa, qb, levels=DEFAULT_LEVELS):
    """Similarity as the mean cosine of per-component phase DIFFERENCES -- a cosine lookup table indexed by
    an integer difference, then an average. The table is built once per call from `levels` entries, so the
    hot path is an integer subtract and a gather, never a trig call per component."""
    table = np.cos(2.0 * np.pi * np.arange(levels) / levels)
    d = (np.asarray(qa, dtype=np.int64) - np.asarray(qb, dtype=np.int64)) % levels
    return float(np.mean(table[d]))


def qfhrr_bundle(qs, levels=DEFAULT_LEVELS):
    """Superpose quantized atoms -- BY LEAVING THE REPRESENTATION, because there is no closed integer
    operation that does this.

    Dequantize to Cartesian, sum the complex vectors, re-quantize with atan2 + round. THE ROUND AT A BIN
    BOUNDARY IS A TIE-ARBITRATION POINT, and the atan2 is floating point, so this single function is where
    the tier's exactness and machine-independence both stop. Everything else here is pure integer.

    That is not an implementation shortcut -- it is a property of the representation, and it is why
    quantized VSA does NOT remove the engine's need for tie discipline."""
    qs = [np.asarray(q, dtype=np.int64) for q in qs]
    if not qs:
        raise ValueError("qfhrr_bundle needs at least one vector")
    total = np.sum([dequantize_phases(q, levels) for q in qs], axis=0)
    return quantize_phases(total, levels)


def bits_per_dim(levels=DEFAULT_LEVELS):
    """Bits needed per dimension at this many phase levels -- the storage claim, computed rather than
    quoted. Compare against 128 for the complex128 FHRR atom this tier replaces."""
    return int(np.ceil(np.log2(levels)))


def measure_fidelity(dim=1024, levels_list=(4, 8, 16, 32, 64, 256), bundle_n=16, seeds=8, seed0=0):
    """Re-measure the quantized-vs-complex fidelity table ON THIS SUBSTRATE, with variance.

    WHY THIS EXISTS: RESEARCH_CONSOLIDATED.md quotes the qFHRR preprint's table (bind 0.9497 at K=8, 0.9872
    at K=16, 0.9999 at K=256) and explicitly flags that those figures COULD NOT BE RE-VERIFIED from the
    abstract and must be confirmed before being quoted as measured. This function is that confirmation, and
    it stays runnable so the numbers never become folklore.

    TWO BUNDLE METRICS ARE REPORTED, AND THE DIFFERENCE BETWEEN THEM IS THE FINDING. `bundle_fid` compares
    the quantized bundle against the TRUE COMPLEX bundle (magnitudes included); `bundle_fid_phase` compares
    it against the PHASE of that bundle. The paper's table is the second one -- re-measuring both is what
    identified that, and the first is what an engineer replacing a complex bundle actually gets.

    Returns dicts with levels, bits, bind_fid, bundle_fid, bundle_fid_phase, unbind_exact (fraction of
    components recovered EXACTLY), and size_reduction against complex128."""
    from holographic.sampling_and_signal.holographic_fhrr import fhrr_bind, fhrr_bundle, fhrr_sim

    out = []
    for levels in levels_list:
        bf, uf, bnf, bnp = [], [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + s)
            a, b = phasor_atom(dim, rng), phasor_atom(dim, rng)
            qa, qb = quantize_phases(a, levels), quantize_phases(b, levels)
            # bind fidelity: the quantized bind, dequantized, against the true complex bind
            bf.append(float(fhrr_sim(dequantize_phases(qfhrr_bind(qa, qb, levels), levels), fhrr_bind(a, b))))
            # unbind EXACTNESS: not a similarity -- the fraction of indices recovered bit for bit
            uf.append(float(np.mean(qfhrr_unbind(qfhrr_bind(qa, qb, levels), qa, levels) == qb)))
            # bundle fidelity: the non-closed operation, against the complex bundle
            vs = [phasor_atom(dim, rng) for _ in range(bundle_n)]
            ref = fhrr_bundle(vs)
            deq = dequantize_phases(qfhrr_bundle([quantize_phases(v, levels) for v in vs], levels), levels)
            bnf.append(float(fhrr_sim(deq, ref)))
            # ... and against the PHASE of the same bundle. Stripping the reference's magnitude is what
            # the paper's table does; keeping it is what an engineer actually experiences.
            bnp.append(float(fhrr_sim(deq, np.exp(1j * np.angle(ref)))))
        out.append({"levels": levels, "bits": bits_per_dim(levels),
                    "bind_fid": float(np.mean(bf)), "bind_sd": float(np.std(bf)),
                    "bundle_fid": float(np.mean(bnf)), "bundle_sd": float(np.std(bnf)),
                    "bundle_fid_phase": float(np.mean(bnp)),
                    "unbind_exact": float(np.mean(uf)),
                    "size_reduction": 1.0 - bits_per_dim(levels) / 128.0})
    return out


def _selftest():
    rng = np.random.default_rng(0)
    dim = 512

    # 1. THE HEADLINE: unbind is EXACT, at every K. Not "high fidelity" -- array_equal on the indices.
    #    This is the property that distinguishes the quantized tier from the real-valued path, where
    #    unbind is a quasi-inverse recovering ~0.70 cosine.
    for levels in (4, 8, 16, 256):
        qa, qb = qfhrr_atom(dim, rng, levels), qfhrr_atom(dim, rng, levels)
        assert np.array_equal(qfhrr_unbind(qfhrr_bind(qa, qb, levels), qa, levels), qb), \
            "unbind is not exact at K=%d -- the tier's central claim" % levels

    # 2. BIND IS PURE INTEGER. If a float ever leaks in, machine-independence is gone silently.
    qa, qb = qfhrr_atom(dim, rng), qfhrr_atom(dim, rng)
    assert np.issubdtype(qfhrr_bind(qa, qb).dtype, np.integer)
    assert np.issubdtype(qfhrr_unbind(qa, qb).dtype, np.integer)

    # 3. BIND IS COMMUTATIVE and self-similarity is exactly 1 (the cosine table at difference 0).
    assert np.array_equal(qfhrr_bind(qa, qb), qfhrr_bind(qb, qa))
    assert abs(qfhrr_sim(qa, qa) - 1.0) < 1e-12

    # 4. BUNDLING IS NOT CLOSED -- asserted, not merely documented. Bundling then unbundling must LOSE
    #    information, because the projection re-quantizes. If this ever becomes lossless, the caveat that
    #    governs this entire research line is wrong and must be rewritten.
    levels = 16
    atoms = [qfhrr_atom(dim, rng, levels) for _ in range(8)]
    bundled = qfhrr_bundle(atoms, levels)
    sims = [qfhrr_sim(bundled, a, levels) for a in atoms]
    assert max(sims) < 0.999, "qfhrr_bundle became lossless -- the not-closed caveat must be rewritten"
    assert min(sims) > 0.0, "bundle lost every member (min sim %.3f)" % min(sims)

    # 5. QUANTIZATION IS THE ONLY LOSSY STEP: a round trip through quantize/dequantize/quantize must be a
    #    FIXED POINT. If it is not, the grid is inconsistent and every stored vector drifts on re-read.
    v = phasor_atom(dim, rng)
    q1 = quantize_phases(v, 16)
    assert np.array_equal(quantize_phases(dequantize_phases(q1, 16), 16), q1), "quantization is not idempotent"

    # 6. STORAGE CLAIM, computed: 16 levels is 4 bits, a 96.9% cut against complex128.
    assert bits_per_dim(16) == 4 and bits_per_dim(256) == 8
    rows = measure_fidelity(dim=256, levels_list=(16,), seeds=3)
    assert abs(rows[0]["size_reduction"] - (1 - 4 / 128)) < 1e-12

    # 7. FIDELITY RISES WITH K, and K=16 clears 0.95 -- the knee that justifies the default.
    rows = measure_fidelity(dim=512, levels_list=(4, 16, 256), seeds=4)
    assert rows[0]["bind_fid"] < rows[1]["bind_fid"] < rows[2]["bind_fid"], "fidelity is not monotone in K"
    assert rows[1]["bind_fid"] > 0.95, "K=16 bind fidelity fell below 0.95 (%.4f)" % rows[1]["bind_fid"]
    assert rows[1]["unbind_exact"] == 1.0, "unbind stopped being exact"

    # 8. GUARDS.
    for bad in (0, 1):
        try:
            quantize_phases(np.array([1 + 0j]), bad)
            raise AssertionError("accepted %d phase levels" % bad)
        except ValueError:
            pass

    print("holographic_qfhrr: all selftests passed (exact unbind, integer bind, NOT-closed bundling, storage)")


if __name__ == "__main__":
    _selftest()

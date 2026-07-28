"""Regression traps for holographic_qfhrr -- the quantized integer-phase tier.

Two things are pinned hardest. First, unbind is EXACT (array_equal on indices) -- the strongest exactness
guarantee in the engine, and the one property that distinguishes this representation from the real-valued
path. Second, bundling is NOT closed and the magnitude ceiling is real: those are the caveats that keep
this research line honest, and both are asserted so they cannot quietly disappear.
"""
import numpy as np
import pytest

import lecore
from holographic.sampling_and_signal.holographic_fhrr import fhrr_bundle, fhrr_sim, phasor_atom
from holographic.sampling_and_signal.holographic_qfhrr import (bits_per_dim, dequantize_phases,
                                                               measure_fidelity, qfhrr_atom,
                                                               qfhrr_bind, qfhrr_bundle, qfhrr_sim,
                                                               qfhrr_unbind, quantize_phases)


def test_unbind_is_exact_at_every_level_count():
    # THE HEADLINE. Not "high fidelity" -- bit-for-bit equality of the phase indices, at every K. Real HRR
    # recovers b at cosine ~0.70 and needs cleanup; this needs none.
    rng = np.random.default_rng(0)
    for levels in (4, 8, 16, 256):
        qa, qb = qfhrr_atom(512, rng, levels), qfhrr_atom(512, rng, levels)
        assert np.array_equal(qfhrr_unbind(qfhrr_bind(qa, qb, levels), qa, levels), qb)


def test_bind_and_unbind_stay_pure_integer():
    # If a float ever leaks in, machine-independence is gone silently.
    rng = np.random.default_rng(1)
    qa, qb = qfhrr_atom(256, rng), qfhrr_atom(256, rng)
    assert np.issubdtype(qfhrr_bind(qa, qb).dtype, np.integer)
    assert np.issubdtype(qfhrr_unbind(qa, qb).dtype, np.integer)


def test_bind_is_commutative_and_self_similarity_is_exactly_one():
    rng = np.random.default_rng(2)
    qa, qb = qfhrr_atom(256, rng), qfhrr_atom(256, rng)
    assert np.array_equal(qfhrr_bind(qa, qb), qfhrr_bind(qb, qa))
    assert abs(qfhrr_sim(qa, qa) - 1.0) < 1e-12


def test_quantization_is_idempotent():
    # A round trip through quantize/dequantize/quantize must be a FIXED POINT, or every stored vector
    # drifts each time it is re-read.
    v = phasor_atom(512, np.random.default_rng(3))
    q = quantize_phases(v, 16)
    assert np.array_equal(quantize_phases(dequantize_phases(q, 16), 16), q)


def test_bundling_is_not_closed_kept_negative():
    # THE CAVEAT THAT GOVERNS THIS RESEARCH LINE. Bundling must leave the representation and re-quantize,
    # so it LOSES information. If it ever becomes lossless, the claim that quantized VSA cannot delete
    # tie-arbitration is wrong and must be rewritten -- do not relax this.
    rng = np.random.default_rng(4)
    levels = 16
    atoms = [qfhrr_atom(512, rng, levels) for _ in range(8)]
    sims = [qfhrr_sim(qfhrr_bundle(atoms, levels), a, levels) for a in atoms]
    assert max(sims) < 0.999
    assert min(sims) > 0.0


def test_bundle_fidelity_saturates_because_magnitude_is_discarded():
    # THE FINDING THE PREPRINT'S TABLE HIDES. Against the TRUE complex bundle, fidelity plateaus around
    # 0.89 and MORE PHASE LEVELS DO NOT HELP -- the ceiling is discarding the magnitude, independent of K.
    # Pinned so the optimistic phase-referenced number can never silently stand in for this one.
    rows = measure_fidelity(dim=512, levels_list=(64, 256), bundle_n=16, seeds=4)
    coarse, fine = rows[0], rows[1]
    assert fine["bundle_fid"] < 0.95, "the magnitude ceiling disappeared (%.4f)" % fine["bundle_fid"]
    assert abs(fine["bundle_fid"] - coarse["bundle_fid"]) < 0.02, "bundle_fid is no longer saturated in K"
    # ... while the PHASE-referenced metric does keep climbing toward 1.0. Both must stay reported.
    assert fine["bundle_fid_phase"] > 0.99
    assert fine["bundle_fid_phase"] > fine["bundle_fid"] + 0.05


def test_bind_fidelity_reproduces_the_preprint():
    # The consolidation flagged these figures as unverifiable from the abstract. Measured here they match:
    # paper 0.9497 / 0.9872 / 0.9999 at K=8/16/256. Pinned as a claim about THIS substrate.
    rows = {r["levels"]: r for r in measure_fidelity(dim=1024, levels_list=(8, 16, 256), seeds=4)}
    assert abs(rows[8]["bind_fid"] - 0.9497) < 0.01
    assert abs(rows[16]["bind_fid"] - 0.9872) < 0.01
    assert abs(rows[256]["bind_fid"] - 0.9999) < 0.01
    for r in rows.values():
        assert r["unbind_exact"] == 1.0


def test_fidelity_is_monotone_in_levels_and_storage_claim_holds():
    rows = measure_fidelity(dim=512, levels_list=(4, 16, 256), seeds=4)
    assert rows[0]["bind_fid"] < rows[1]["bind_fid"] < rows[2]["bind_fid"]
    assert bits_per_dim(16) == 4 and bits_per_dim(256) == 8
    assert abs(rows[1]["size_reduction"] - (1 - 4 / 128)) < 1e-12


def test_level_count_guards():
    for bad in (0, 1):
        with pytest.raises(ValueError):
            quantize_phases(np.array([1 + 0j]), bad)


# --------------------------------------------------------------------------------------
# CROSS-FACULTY
# --------------------------------------------------------------------------------------

def test_qfhrr_interoperates_with_the_complex_fhrr_tier():
    # The tier is bolted ONTO holographic_fhrr, so a quantized vector must round-trip back into the complex
    # path and still be recognisable. This is the pairing that breaks first if the phase grid convention
    # drifts, and neither module's selftest can see it.
    rng = np.random.default_rng(5)
    v = phasor_atom(1024, rng)
    back = dequantize_phases(quantize_phases(v, 256), 256)
    assert fhrr_sim(back, v) > 0.999
    vs = [phasor_atom(1024, rng) for _ in range(4)]
    assert fhrr_sim(fhrr_bundle(vs), fhrr_bundle(vs)) > 0.999


def test_exact_unbind_beats_the_real_valued_path_end_to_end():
    # THE CONTRAST THAT JUSTIFIES THE TIER, measured against the engine's own real-valued binding rather
    # than asserted. qFHRR unbind is exact; the real HRR path recovers direction only (~0.7).
    mind = lecore.UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(6)
    qa, qb = qfhrr_atom(512, rng), qfhrr_atom(512, rng)
    assert np.array_equal(mind.qfhrr_unbind(mind.qfhrr_bind(qa, qb), qa), qb)

    a, b = rng.integers(-1, 2, size=512), rng.integers(-1, 2, size=512)
    got = mind.ntt_unbind(mind.ntt_bind(a, b), a).astype(float)
    cos = float(got @ b / (np.linalg.norm(got) * np.linalg.norm(b)))
    assert cos < 0.95, "the real-valued path became exact; the contrast claim needs rewriting"


def test_qfhrr_is_discoverable_by_stranger_phrasing():
    mind = lecore.UnifiedMind(dim=128, seed=0)
    for query in ("store a hypervector at three or four bits per dimension",
                  "quantize phase angles to integers", "bind by adding phase indices modulo k",
                  "shrink hypervector memory footprint"):
        assert "qFHRR" in str(mind.find_capability(query)[:3]), "%r no longer surfaces qFHRR" % query

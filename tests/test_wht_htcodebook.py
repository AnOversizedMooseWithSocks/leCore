"""Regression traps for the exact-transform work: holographic_wht + holographic_htcodebook.

These are TRAPS, not smoke tests. Each pins an exact numeric or structural contract that, if it
silently drifted, would change a stored artefact or a cleanup DECISION rather than merely slowing
something down. Two of them exist because a claim made during the build turned out to be wrong when
measured, and a third pins a delegation that every stored archive plate depends on.
"""
import numpy as np
import pytest

import lecore
from holographic.caching_and_storage.holographic_htcodebook import HadamardCodebook
from holographic.sampling_and_signal.holographic_wht import (fwht, fwht_exact, hadamard_matrix,
                                                             ifwht, measure_wht_vs_fft)


# --------------------------------------------------------------------------------------
# holographic_wht -- exactness is the product, so it is asserted exactly (array_equal).
# --------------------------------------------------------------------------------------

def test_fwht_equals_the_dense_hadamard_matrix_exactly():
    # The fast path must equal an INDEPENDENT construction (Sylvester), in integer arithmetic,
    # with no tolerance. allclose would hide a butterfly that is subtly wrong at one index.
    for m in (1, 2, 5, 7):
        d = 2 ** m
        x = np.random.default_rng(m).integers(-9, 10, size=d)
        assert np.array_equal(fwht(x), hadamard_matrix(d) @ x)


def test_integer_wht_is_bit_exact_and_involutive():
    # The guarantee the FFT cannot make: integer in, integer out, double transform EXACTLY D*x.
    # This is what makes an integer cleanup decision machine-independent.
    x = np.random.default_rng(0).integers(-1, 2, size=1024)
    y = fwht_exact(x)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.array_equal(fwht_exact(y), 1024 * x.astype(np.int64))


def test_fwht_exact_refuses_float_so_the_guarantee_is_enforced():
    # A guarantee that lives only in a docstring is a guarantee somebody will violate.
    with pytest.raises(TypeError):
        fwht_exact(np.zeros(8, dtype=np.float64))


def test_fwht_rejects_bad_shapes_loudly():
    # Silent truncation of a non-power-of-two would corrupt data rather than fail.
    for bad in (np.zeros(3), np.zeros((4, 4))):
        with pytest.raises(ValueError):
            fwht(bad)


def test_image_fwht_delegation_stays_bit_identical():
    # holographic_image._fwht now delegates to holographic_wht. Every HolographicArchive plate ever
    # stored was encoded through it, so a divergence here silently re-interprets stored data. Note
    # the float64 cast: the image module ALWAYS coerced, while fwht preserves integer dtypes, and
    # the delegation deliberately keeps the old coercion.
    from holographic.io_and_interop.holographic_image import _fwht as image_fwht
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for d in (8, 512, 4096):
            for x in (rng.standard_normal(d), rng.integers(-5, 6, size=d)):
                a, b = image_fwht(x), fwht(np.asarray(x, dtype=np.float64))
                assert a.dtype == b.dtype and np.array_equal(a, b)


def test_the_wht_is_slower_than_the_fft_kept_negative():
    # THE NEGATIVE, PINNED. The research shortlist floats the WHT as a possible FFT speedup; on this
    # codebase it is 4-9x slower (Python loop vs C pocketfft). The bound is loose because a timing
    # assertion is a machine-load detector, not a numeric contract -- but if this ever fails, the
    # module docstring is wrong and must be rewritten rather than the test relaxed.
    rows = measure_wht_vs_fft(sizes=(1024,), repeats=120)
    assert rows[0]["ratio"] > 1.5


def test_measure_vs_fft_reports_median_and_mean_so_skew_stays_visible():
    # The first version of this harness took a bare mean with no warmup and asserted the WRONG
    # direction off first-call noise (rfft's spread at D=256 was +/-91 us against an 11 us mean).
    # Both statistics must survive, or that failure mode is invisible again.
    row = measure_wht_vs_fft(sizes=(256,), repeats=40)[0]
    assert {"wht_us", "wht_mean_us", "fft_us", "fft_mean_us"} <= set(row)


# --------------------------------------------------------------------------------------
# holographic_htcodebook -- the fast path must DECIDE identically to the honest scan.
# --------------------------------------------------------------------------------------

def test_transform_cleanup_decides_identically_to_the_matmul_scan():
    # The correctness contract. If the transform and the scan ever disagree, "cleanup" is decoding a
    # different codebook than the one atoms() reports -- a silent wrong answer, not a slow one.
    for d in (8, 64, 256):
        cb = HadamardCodebook(d, seed=1)
        A = cb.atoms()
        rng = np.random.default_rng(d)
        for _ in range(25):
            cue = rng.integers(-7, 8, size=d)
            assert np.array_equal(cb.correlations(cue), A @ cue)
            assert cb.cleanup(cue)[0] == int(np.argmax(A @ cue))


def test_hadamard_atoms_are_exactly_orthogonal():
    # Zero crosstalk is WHY argmax-of-correlation is the maximum-likelihood decode. Assert the Gram
    # matrix is exactly D*I -- not approximately.
    cb = HadamardCodebook(64, seed=2)
    R = cb.atoms()[:64]
    assert np.array_equal(R @ R.T, 64 * np.eye(64, dtype=np.int64))


def test_every_atom_recalls_itself_including_the_signed_half():
    cb = HadamardCodebook(32, seed=3)
    assert cb.K == 64
    for i in range(cb.K):
        assert cb.cleanup(cb.atom(i))[0] == i


def test_cleanup_survives_half_amplitude_noise_with_zero_errors():
    # A distribution bound, not one lucky draw.
    cb = HadamardCodebook(256, seed=4)
    rng = np.random.default_rng(5)
    wrong = 0
    for _ in range(200):
        i = int(rng.integers(cb.K))
        wrong += cb.cleanup(cb.atom(i) + 0.5 * rng.standard_normal(256))[0] != i
    assert wrong == 0


def test_the_codebook_size_cap_is_real():
    # THE KEPT NEGATIVE THAT CORRECTS THE BACKLOG ITEM. RESEARCH_CONSOLIDATED.md proposed replacing a
    # K=16384 scan at D=1024 with one transform. A Hadamard construction holds at most 2*D atoms, so
    # that codebook cannot exist at D=1024 -- it needs D=8192. Pinned so the wrong trade is never
    # quoted back as available.
    assert HadamardCodebook(1024).K == 2048
    assert HadamardCodebook(8192).K == 16384
    for bad in (0, 3, 100):
        with pytest.raises(ValueError):
            HadamardCodebook(bad)


def test_codebook_is_deterministic_for_a_seed():
    a, b = HadamardCodebook(128, seed=7), HadamardCodebook(128, seed=7)
    assert np.array_equal(a.atoms(), b.atoms())


# --------------------------------------------------------------------------------------
# CROSS-FACULTY: the hard lesson on record is that a shared kernel is not a shared manifold,
# so the new faculties are exercised THROUGH the mind and against neighbouring faculties.
# --------------------------------------------------------------------------------------

def test_faculties_round_trip_through_the_mind():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = np.array([1, -1, 1, 1, -1, -1, 1, -1])
    assert np.array_equal(mind.wht_exact(mind.wht_exact(x)), 8 * x)
    assert np.max(np.abs(mind.wht_inverse(mind.wht(x.astype(float))) - x)) < 1e-12
    with pytest.raises(TypeError):
        mind.wht_exact(np.zeros(8))


def _normed_atoms(cb):
    a = cb.atoms().astype(np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_unsigned_hadamard_codebook_feeds_the_bundle_recovery_family():
    # THE CROSS-FACULTY TRAP, and it earned its keep on the first run. An UNSIGNED Hadamard codebook is
    # mutually orthogonal (coherence exactly 0), which is the ideal regime for sparse recovery, so CoSaMP
    # must recover the exact support. Neither module's own selftest could see this pairing.
    mind = lecore.UnifiedMind(dim=256, seed=0)
    cb = mind.hadamard_codebook(256, seed=0, signed=False)
    atoms = _normed_atoms(cb)
    rng = np.random.default_rng(1)
    for M in (4, 16, 32):
        true = rng.choice(cb.K, size=M, replace=False)
        cue = atoms[true].sum(axis=0)
        got = {i for i, _ in mind.cosamp_recall(cue, atoms, M)}
        assert got == set(true.tolist()), "CoSaMP failed on an orthogonal codebook at M=%d" % M


def test_signed_codebook_is_a_degenerate_recovery_dictionary_kept_negative():
    # THE NEGATIVE THIS TEST DISCOVERED, now pinned. signed=True doubles K by adding each atom's exact
    # NEGATION, so the dictionary contains +v and -v pairs and its coherence is exactly 1.000. Sparse
    # recovery cannot distinguish index i from index i+D -- both explain the cue equally with opposite
    # signs -- and CoSaMP duly returns a wrong support (measured 0/3 vs 3/3 unsigned). The signed half is
    # correct and useful for CLEANUP, where the sign is part of the answer; it must NOT be handed to the
    # recovery family as a dictionary. A shared kernel is not a shared manifold.
    mind = lecore.UnifiedMind(dim=256, seed=0)
    signed, unsigned = mind.hadamard_codebook(256, seed=0), mind.hadamard_codebook(256, seed=0, signed=False)
    for cb, wanted in ((signed, 1.0), (unsigned, 0.0)):
        g = _normed_atoms(cb) @ _normed_atoms(cb).T
        np.fill_diagonal(g, 0.0)
        assert abs(float(np.abs(g).max()) - wanted) < 1e-9, "coherence moved for K=%d" % cb.K
    atoms = _normed_atoms(signed)
    rng = np.random.default_rng(1)
    true = rng.choice(signed.K, size=8, replace=False)
    got = {i for i, _ in mind.cosamp_recall(atoms[true].sum(axis=0), atoms, 8)}
    assert got != set(true.tolist()), "signed codebook is no longer degenerate -- rewrite the kept negative"


def test_new_capabilities_are_discoverable_by_stranger_phrasing():
    # The governing rule: a capability find_capability cannot surface does not exist. These exact
    # phrasings measured 0/N before this work landed, which is how a research sweep concluded the
    # recovery family was unwired when it had been wired all along.
    mind = lecore.UnifiedMind(dim=128, seed=0)
    expect = {
        "recover many items from one bundle": "Bundle recovery",
        "what went into this bundle": "Bundle recovery",
        "greedy solver for a mixture of atoms": "Bundle recovery",
        "fast walsh hadamard transform": "Walsh-Hadamard",
        "transform without rounding error": "Walsh-Hadamard",
        "cleanup without comparing against every codebook entry": "Hadamard codebook",
        "nearest codeword in log time": "Hadamard codebook",
    }
    for query, wanted in expect.items():
        top = str(mind.find_capability(query)[:3])
        assert wanted in top, "%r no longer surfaces %r" % (query, wanted)

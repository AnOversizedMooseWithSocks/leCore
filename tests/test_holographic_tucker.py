"""A7/M1 -- multi-way tensor compression: Tucker (HOSVD) and Tensor Train, with a rank gate.

The claim under test is not "it compresses" -- anything compresses. It is:
  * it beats the honest baseline (per-slice SVD, which sees structure WITHIN a slice but none ACROSS slices);
  * the TT error tracks its tolerance, and tol -> 0 recovers exactly;
  * and on data with NO low-rank structure the gate says so, instead of destroying it.
"""
import numpy as np
import pytest

from holographic.caching_and_storage.holographic_tucker import (
    per_slice_svd_size, rank_gate, rel_error, tt_compress, tt_reconstruct, tt_size,
    tucker_compress, tucker_reconstruct, tucker_size, unfold)


def _diffusing_field(n=20, frames=16):
    """A REAL tensor from the engine: a heat field evolving in time -- smooth in x, y AND t."""
    from holographic.simulation_and_physics.holographic_laplacian import diffuse_spectral
    xs = np.arange(n) / n
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    T0 = np.exp(-40 * ((X - 0.5) ** 2 + (Y - 0.35) ** 2)) + 0.6 * np.sin(4 * np.pi * X)
    return np.stack([diffuse_spectral(T0, alpha=0.002, t=t, dx=1.0 / n) for t in np.linspace(0, 4, frames)])


def test_unfold_shapes():
    X = np.zeros((3, 4, 5))
    assert unfold(X, 0).shape == (3, 20)
    assert unfold(X, 2).shape == (5, 12)


def test_rank_gate_finds_structure_in_a_real_field():
    field = _diffusing_field()
    ranks, kept = rank_gate(field, energy=0.999)
    assert kept >= 0.999
    assert min(ranks) < min(field.shape)


def test_rank_gate_reports_full_rank_on_white_noise():
    """The honest negative: no structure -> full rank -> 'do not compress'. And Tucker would cost MORE than raw."""
    noise = np.random.default_rng(0).standard_normal((12, 12, 12))
    ranks, _ = rank_gate(noise, energy=0.99)
    assert min(ranks) >= 10
    assert tucker_size(tucker_compress(noise, energy=0.99)) > noise.size


def test_tucker_beats_the_per_slice_svd_baseline():
    field = _diffusing_field()
    code = tucker_compress(field, energy=0.999)
    err = rel_error(field, tucker_reconstruct(code))
    ratio = field.size / tucker_size(code)
    baseline = field.size / per_slice_svd_size(field, energy=0.999)
    assert err < 1e-2
    assert ratio > 5.0 * baseline                       # measured ~57x vs ~5.9x


def test_tucker_is_exact_at_full_rank():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 6, 7))
    code = tucker_compress(X, ranks=(5, 6, 7))
    assert rel_error(X, tucker_reconstruct(code)) < 1e-12


def test_tt_error_tracks_its_tolerance_and_is_exact_at_full_rank():
    field = _diffusing_field()
    prev = None
    for tol in (1e-2, 1e-4, 1e-6):
        code = tt_compress(field, tol=tol)
        err = rel_error(field, tt_reconstruct(code))
        assert err < 10.0 * tol                          # honours the budget
        if prev is not None:
            assert err < prev                            # tighter tol -> smaller error (this was once NOT true)
        prev = err
        assert tt_size(code) < field.size
    assert rel_error(field, tt_reconstruct(tt_compress(field, tol=1e-14))) < 1e-12


def test_tt_sign_pinning_does_not_break_reconstruction():
    """Regression: pinning singular-vector signs for determinism must flip the paired Vt row too, or U @ C no longer
    reconstructs. Tucker is immune (its factors appear twice, so signs cancel); TT is not, and this floored the TT
    error at 6.6e-2 whatever the tolerance."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 5, 6))
    a = tt_compress(X, tol=1e-14)
    assert rel_error(X, tt_reconstruct(a)) < 1e-12
    b = tt_compress(X, tol=1e-14)
    assert all(np.array_equal(p, q) for p, q in zip(a["cores"], b["cores"]))   # deterministic


def test_through_the_mind_and_cp_is_refused():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    X = _diffusing_field(n=12, frames=8)
    code = m.compress_tensor(X, energy=0.9999)
    assert rel_error(X, m.decompress_tensor(code)) < 1e-2
    tt = m.compress_tensor(X, method="tt", tol=1e-5)
    assert rel_error(X, m.decompress_tensor(tt)) < 1e-3
    with pytest.raises(ValueError):
        m.compress_tensor(X, method="cp")               # a best rank-R CP approximation may not exist


# ---------------------------------------------------------------------------------------------------------------
# Tier 5 -- the codec as a DENOISER (Milanfar: a denoiser is a map of the manifold clean signals live on).
# ---------------------------------------------------------------------------------------------------------------
def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10.0 * np.log10((a.max() - a.min()) ** 2 / max(mse, 1e-18))


def test_tensor_denoise_beats_the_per_slice_baseline_at_every_noise_level():
    from holographic.caching_and_storage.holographic_tucker import per_slice_svd_denoise, tucker_denoise
    clean = _diffusing_field(n=20, frames=16)
    rng = np.random.default_rng(0)
    for sigma in (0.02, 0.05, 0.10):
        noisy = clean + sigma * rng.standard_normal(clean.shape)
        den, _ranks, sig = tucker_denoise(noisy)
        base = per_slice_svd_denoise(noisy, sig)
        assert _psnr(clean, den) > _psnr(clean, noisy) + 8.0     # a real restoration
        assert _psnr(clean, den) > _psnr(clean, base) + 4.0      # ~7 dB over the baseline, measured


def test_tensor_denoise_estimates_sigma_from_the_data():
    from holographic.caching_and_storage.holographic_tucker import tucker_denoise
    clean = _diffusing_field(n=16, frames=12)
    noisy = clean + 0.05 * np.random.default_rng(1).standard_normal(clean.shape)
    _den, _ranks, sigma = tucker_denoise(noisy)                  # no sigma handed in
    assert abs(sigma - 0.05) < 0.02


def test_low_rank_prior_destroys_a_full_rank_signal():
    """THE KEPT NEGATIVE, pinned. A low-rank prior is a CLAIM about the signal; where it is false, this denoiser
    throws the signal away with the noise (measured 43.1 dB -> 17.1 dB). It cannot detect that from the inside --
    check the rank gate first."""
    from holographic.caching_and_storage.holographic_tucker import tucker_denoise
    rng = np.random.default_rng(2)
    full = rng.standard_normal((14, 14, 14))                     # no low-rank structure anywhere
    noisy = full + 0.05 * rng.standard_normal(full.shape)
    den, _r, _s = tucker_denoise(noisy)
    assert _psnr(full, den) < _psnr(full, noisy) - 10.0


def test_noise_ranks_recovers_the_true_rank_of_a_low_rank_signal():
    from holographic.caching_and_storage.holographic_tucker import noise_ranks
    rng = np.random.default_rng(3)
    U = [np.linalg.qr(rng.standard_normal((16, 3)))[0] for _ in range(3)]
    core = rng.standard_normal((3, 3, 3))
    X = np.einsum("abc,ia,jb,kc->ijk", core, *U)
    noisy = X + 0.05 * rng.standard_normal(X.shape)
    assert noise_ranks(noisy, 0.05) == (3, 3, 3)                 # exactly the true multilinear rank


def test_denoise_tensor_through_the_mind():
    """NOTE the honest size dependence, measured: the gain grows with the data because a bigger tensor has more
    redundancy for the prior to exploit -- 7.6 dB at (10,14,14), 8.4 at (12,16,16), 11.9 at (16,20,20). The bar
    here is set for the small case."""
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    clean = _diffusing_field(n=14, frames=10)
    noisy = clean + 0.05 * np.random.default_rng(4).standard_normal(clean.shape)
    den, ranks, sigma = m.denoise_tensor(noisy)
    assert _psnr(clean, den) > _psnr(clean, noisy) + 6.0
    assert len(ranks) == 3 and sigma > 0


def test_sigma_estimate_is_biased_high_by_signal_high_frequencies():
    """Kept, because it explains the ranks: Donoho's MAD-of-differences assumes the clean signal is smoother than
    the noise. This field carries a sin(4*pi*x) term, so some SIGNAL lands in the finest detail band and sigma comes
    back a little high (0.06 for a true 0.05) -- which makes the rank rule slightly conservative. That is the safe
    direction (under-fit rather than keep noise), but it is a bias, not a coincidence."""
    from holographic.caching_and_storage.holographic_tucker import tucker_denoise
    clean = _diffusing_field(n=20, frames=16)
    noisy = clean + 0.05 * np.random.default_rng(4).standard_normal(clean.shape)
    _d, _r, sigma = tucker_denoise(noisy)
    assert 0.05 < sigma < 0.08


# ---------------------------------------------------------------------------------------------------------------
# Tier 5 -- the codec as STORAGE. The bar is int8 (1 byte/element), not raw float64: any array can be stored at
# 1 byte/element, so beating float64 proves nothing.
# ---------------------------------------------------------------------------------------------------------------
def test_pack_tt_round_trips():
    from holographic.caching_and_storage.holographic_tucker import pack_tt, unpack_tt, tt_bytes
    field = _diffusing_field(n=20, frames=16)
    code = tt_compress(field, tol=1e-4)
    back = unpack_tt(pack_tt(code))
    assert back["shape"] == code["shape"]
    assert rel_error(field, tt_reconstruct(back)) < 1e-3      # float32 cores: truncation error still dominates
    assert tt_bytes(code) < field.size                        # smaller than int8


def test_save_tensor_beats_int8_on_size_and_fidelity():
    import os
    import tempfile
    from holographic.caching_and_storage.holographic_tucker import load_tensor, save_tensor
    field = _diffusing_field(n=24, frames=20)
    path = os.path.join(tempfile.mkdtemp(), "f.tt")
    info = save_tensor(field, path)
    back = load_tensor(path)
    assert info["tt"]
    assert os.path.getsize(path) < field.size                 # beats int8's 1 byte/element
    tt_err = rel_error(field, back)
    peak = float(np.abs(field).max())
    i8 = np.round(field / (peak / 127.0)).astype(np.int8).astype(float) * (peak / 127.0)
    assert tt_err < rel_error(field, i8)                      # ...and is MORE accurate, not less


def test_save_tensor_refuses_and_stores_raw_when_tt_does_not_pay():
    """White noise has no cross-mode structure: the TT code is bigger than int8, so it must be refused and the
    array stored raw (exactly). This was a real bug -- the size bar compared against float64, and noise 'won'."""
    import os
    import tempfile
    from holographic.caching_and_storage.holographic_tucker import load_tensor, save_tensor
    noise = np.random.default_rng(0).standard_normal((16, 20, 20))
    path = os.path.join(tempfile.mkdtemp(), "n.tt")
    info = save_tensor(noise, path)
    assert not info["tt"]
    assert np.array_equal(noise, load_tensor(path))           # raw -> exact


def test_core_try_tt_decisions():
    """The hook core.save() uses. HONEST NOTE: no object the engine currently persists holds a 3-D array, so this
    branch is ready but unexercised by real state -- which is exactly why the decision function is tested directly."""
    import holographic.misc.holographic_core as core
    field = _diffusing_field(n=24, frames=20)
    packed = core._try_tt(field)
    assert packed is not None and len(packed) < field.size                    # accepted, beats int8
    assert core._try_tt(np.random.default_rng(0).standard_normal((16, 20, 20))) is None   # noise: refused
    assert core._try_tt(np.ones((300, 64))) is None                           # 2-D belongs to the rd/KLT code
    assert core._try_tt(np.ones((3, 3, 3))) is None                           # too small to bother


# ---------------------------------------------------------------------------------------------------------------
# The AREA-LAW diagnostic: will a factorisation pay, before you pay to find out?
# (The checkable shadow of a structural idea: a capacity is a COUNT across a boundary, not a tuned parameter.)
# ---------------------------------------------------------------------------------------------------------------
def test_band_limited_signal_has_constant_bond_rank():
    """The purest area law: a sine reshaped into 12 binary modes keeps rank 2 at EVERY cut -- the information
    crossing a boundary does not grow with the volume it encloses."""
    from holographic.caching_and_storage.holographic_tucker import bond_ranks
    n = 12
    x = np.linspace(0, 1, 2 ** n)
    ranks = bond_ranks(np.sin(2 * np.pi * 3 * x).reshape([2] * n), tol=1e-8)
    assert max(ranks) <= 3, ranks


def test_white_noise_saturates_the_volume_law_bound():
    from holographic.caching_and_storage.holographic_tucker import structure_verdict
    noise = np.random.default_rng(0).standard_normal((16, 20, 20))
    v = structure_verdict(noise, tol=1e-4)
    assert v["saturation"] > 0.95 and v["verdict"] == "volume-law"
    assert v["ranks"] == v["bound"]                  # every cut carries the maximum it could


def test_verdict_predicts_whether_tt_beats_int8():
    """The diagnostic's whole purpose: it decides BEFORE the encode, and it agrees with the bytes afterwards."""
    from holographic.caching_and_storage.holographic_tucker import structure_verdict, tt_bytes, tt_compress
    field = _diffusing_field(n=24, frames=20)
    noise = np.random.default_rng(1).standard_normal(field.shape)
    for A, expect_area in ((field, True), (noise, False)):
        v = structure_verdict(A, tol=1e-4)
        tt_wins = tt_bytes(tt_compress(A, tol=1e-4)) < A.size      # A.size == int8 bytes
        assert (v["verdict"] == "area-law") == expect_area
        assert tt_wins == expect_area                              # prediction matches the outcome


def test_tensor_structure_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    v = m.tensor_structure(_diffusing_field(n=16, frames=12), tol=1e-4)
    assert v["verdict"] == "area-law" and 0.0 < v["saturation"] < 0.5
    assert len(v["ranks"]) == len(v["bound"]) == 2                 # a 3-mode tensor has 2 bonds

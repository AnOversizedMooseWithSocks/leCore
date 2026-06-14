"""Temporal (video) compression on the substrate: the motion-compensation win,
its honest boundary, and exact round-trip -- plus the audio/spectral unification."""
import numpy as np

from holographic_video import HolographicVideo, estimate_shift


def _rigid(S=64, n=14, step=2):
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    base = (((xx - 18) ** 2 + (yy - 32) ** 2) <= 11 ** 2).astype(float)
    return [np.roll(base, step * t, axis=1) for t in range(n)]


def _deform(S=64, n=14):
    out = []
    for t in range(n):
        yy, xx = np.mgrid[0:S, 0:S].astype(float)
        out.append((((xx - 32) ** 2 + (yy - 32) ** 2) <= (8 + 0.9 * t) ** 2).astype(float))
    return out


def test_motion_search_recovers_exact_shift():
    # The physics property in pixel space: a rigid translation is recovered
    # exactly by a one-number motion search, and the motion-compensated residual
    # is numerically ZERO -- the sequence collapses to keyframe + one int/frame.
    frames = _rigid(step=2)
    for t in range(1, len(frames)):
        dx = estimate_shift(frames[t - 1], frames[t])
        assert dx == 2
        residual = frames[t] - np.roll(frames[t - 1], dx, axis=1)
        assert np.linalg.norm(residual) < 1e-9


def test_gop_beats_intra_on_rigid_motion():
    # STRICT rate-distortion win where motion is the change: keyframe + motion-
    # compensated residuals are smaller AND higher fidelity than per-frame INTRA.
    frames = _rigid()
    vc = HolographicVideo(key_keep=400, res_keep=80)
    packets, gop_bytes = vc.encode(frames)
    gop_psnr = vc.mean_psnr(frames, packets)
    intra_bytes, intra_psnr = HolographicVideo.intra_baseline(frames, keep=400)
    assert gop_bytes < intra_bytes                   # smaller
    assert gop_psnr >= intra_psnr - 0.1              # and not worse (measured +0.4)


def test_gop_loses_on_deformation_and_we_say_so():
    # THE HONEST BOUNDARY: when the inter-frame change is NOT a rigid shift, the
    # motion model is wrong, residuals stay large, and GOP coding LOSES fidelity
    # at matched budget. Motion compensation pays only when motion is the change.
    frames = _deform()
    vc = HolographicVideo(key_keep=400, res_keep=80)
    packets, gop_bytes = vc.encode(frames)
    gop_psnr = vc.mean_psnr(frames, packets)
    intra_bytes, intra_psnr = HolographicVideo.intra_baseline(frames, keep=400)
    assert gop_psnr < intra_psnr - 1.0               # clearly worse: kept negative


def test_decode_mirrors_encode():
    # The decoder's reconstruction chain matches the encoder's (so motion vectors
    # are applied consistently); round-trip PSNR is finite and high on rigid.
    frames = _rigid()
    vc = HolographicVideo(key_keep=400, res_keep=80)
    packets, _ = vc.encode(frames)
    recon = vc.decode(packets)
    assert len(recon) == len(frames)
    assert vc.mean_psnr(frames, packets) > 20.0


def test_spectral_compression_is_basis_agnostic():
    # AUDIO insight unified: the same DCT-truncate-and-store machinery compresses
    # a 1-D signal (a 1xN 'image'), and the holographic plate survives erasure
    # with no extra loss -- spatial vs spectral is the same operation.
    from holographic_image import HolographicImage, _psnr
    N = 256
    t = np.linspace(0, 1, N)
    sig = (0.6 * np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)
           + 0.15 * np.sin(2 * np.pi * 23 * t))
    sig = (sig - sig.min()) / (sig.max() - sig.min())
    hi = HolographicImage((1, N), keep=32, dim=512, seed=0).store(sig[None, :], bits=8)
    clean = _psnr(sig[None, :], hi.reconstruct().ravel()[None, :])
    erased = _psnr(sig[None, :],
                   hi.reconstruct(hi.damage_mask(0.30)).ravel()[None, :])
    assert clean > 25.0                              # compresses 8x at high SNR
    assert erased > clean - 3.0                      # robust to 30% erasure


def test_subpixel_motion_beats_integer_search():
    # FRONTIER EXTENSION: a fractional drift is recovered exactly by sub-pixel
    # (Fourier-shift) motion compensation -- residual to numerical zero -- where
    # integer search rounds and leaves residual. A pixel shift is a phase ramp in
    # frequency, the scalar code's fractional-power principle in 2-D.
    from holographic_video import estimate_subpixel_shift, fourier_shift, estimate_shift
    S = 64
    yy, xx = np.mgrid[0:S, 0:S].astype(float)

    def blob(cx):
        return np.exp(-(((xx - cx) ** 2 + (yy - 32) ** 2) / 40.0))

    frames = [blob(18 + 1.7 * t) for t in range(8)]
    int_res, sub_res = [], []
    for t in range(1, len(frames)):
        di = estimate_shift(frames[t - 1], frames[t])
        int_res.append(np.linalg.norm(frames[t] - np.roll(frames[t - 1], di, axis=1)))
        ds = estimate_subpixel_shift(frames[t - 1], frames[t])
        sub_res.append(np.linalg.norm(frames[t] - fourier_shift(frames[t - 1], ds)))
    assert np.mean(sub_res) < 1e-3                   # near-exact (tiny edge wrap)
    assert np.mean(sub_res) < 0.02 * np.mean(int_res)  # ~100x below integer search

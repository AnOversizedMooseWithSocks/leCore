"""Inverse-rendering IR13: checkerboard/sparse rendering -- shade ~half, reconstruct the rest (masked recovery)."""
import numpy as np
from holographic_render import Camera
from holographic_sdf import box
from holographic_checkerboard import (checkerboard_mask, reconstruct_checkerboard, render_checkerboard,
                                     _shade_all, _row_halved, _psnr)

CAM = Camera(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)


def test_mask_is_half_and_parity_flips():
    m0 = checkerboard_mask(40, 40, 0)
    assert abs(m0.mean() - 0.5) < 0.02 and np.all(m0 == ~checkerboard_mask(40, 40, 1))


def test_reconstruction_matches_full():
    full = _shade_all(box(1, 0.7, 0.5), CAM, 80, 80)
    recon = reconstruct_checkerboard(full, checkerboard_mask(80, 80, 0))
    assert _psnr(recon, full) > 30.0


def test_reconstruction_beats_nofill_and_rowhalving():
    full = _shade_all(box(1, 0.7, 0.5), CAM, 80, 80)
    mask = checkerboard_mask(80, 80, 0)
    recon = reconstruct_checkerboard(full, mask)
    nofill = full * mask[..., None]
    assert _psnr(recon, full) > _psnr(nofill, full)            # beats leaving gaps
    assert _psnr(recon, full) > _psnr(_row_halved(full), full)  # beats matched-cost row-halving (2D > 1D spread)


def test_render_checkerboard_traces_half_and_reconstructs():
    full = _shade_all(box(1, 0.7, 0.5), CAM, 80, 80)
    ck, mask = render_checkerboard(box(1, 0.7, 0.5), CAM, 80, 80)
    assert abs(mask.mean() - 0.5) < 0.02                       # ~half the rays traced
    assert _psnr(ck, full) > 30.0                              # reconstructed frame ~ full shade


def test_reconstruction_keeps_shaded_pixels_exact():
    full = _shade_all(box(1, 0.7, 0.5), CAM, 60, 60)
    mask = checkerboard_mask(60, 60, 0)
    recon = reconstruct_checkerboard(full, mask)
    assert np.allclose(recon[mask], full[mask])                # shaded pixels are untouched


def test_parity_flips_fill_other_half():
    _, m0 = render_checkerboard(box(1, 0.7, 0.5), CAM, 60, 60, parity=0)
    _, m1 = render_checkerboard(box(1, 0.7, 0.5), CAM, 60, 60, parity=1)
    assert np.all(m1 == ~m0)


def test_deterministic():
    a = render_checkerboard(box(1, 0.7, 0.5), CAM, 40, 40)[0]
    b = render_checkerboard(box(1, 0.7, 0.5), CAM, 40, 40)[0]
    assert np.array_equal(a, b)

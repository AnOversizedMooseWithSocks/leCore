"""Tests for the Nystrom low-rank RE-ENABLE (apply_kernel_gated) -- Nystrom when low-rank, exact fallback otherwise."""
import numpy as np
from holographic_nystrom import apply_kernel_gated, exact_kernel_apply, nystrom_probe_error


def _scene(N=600, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, 2)), rng.standard_normal((N, 2)), rng.standard_normal(N)


def _rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def test_low_rank_uses_nystrom_and_is_accurate():
    pts, src, w = _scene()
    ref = exact_kernel_apply(pts, src, w, 1.5)
    field, info = apply_kernel_gated(pts, src, w, 1.5, m=50)
    assert info["method"] == "nystrom" and _rel(field, ref) < 0.05        # smooth kernel -> Nystrom, accurate


def test_high_rank_falls_back_to_exact():
    pts, src, w = _scene()
    ref = exact_kernel_apply(pts, src, w, 0.15)
    field, info = apply_kernel_gated(pts, src, w, 0.15, m=50)
    assert info["method"] == "exact" and _rel(field, ref) < 1e-9          # sharp kernel -> exact, byte-correct


def test_probe_error_tracks_the_regime():
    pts, src, w = _scene()
    lo = nystrom_probe_error(pts, src, w, 1.5, m=50)
    hi = nystrom_probe_error(pts, src, w, 0.15, m=50)
    assert lo < 0.05 and hi > 0.3                                          # detector separates low/high rank


def test_result_always_accurate_across_sigma():
    # whichever path is taken, the result is close to exact (Nystrom in-regime, or exact itself)
    pts, src, w = _scene()
    for sigma in [2.0, 1.0, 0.5, 0.25, 0.1]:
        ref = exact_kernel_apply(pts, src, w, sigma)
        field, info = apply_kernel_gated(pts, src, w, sigma, m=50, threshold=0.1)
        assert _rel(field, ref) < 0.12                                     # never badly wrong (exact fallback)


def test_info_reports_score_and_threshold():
    pts, src, w = _scene()
    _, info = apply_kernel_gated(pts, src, w, 1.5, m=50)
    assert "score" in info and info["threshold"] == 0.1 and info["method"] in ("nystrom", "exact")

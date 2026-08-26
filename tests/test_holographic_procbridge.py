"""Tests for S3 the procedural bridges (holographic_procbridge): the MEASURED connections from the
SDF/procedural layer to the rest of the stack. C1 (compression/complexity) and C4 (the shared soft
operator) are wins; C2 (the FPE field as a denoiser) is a KEPT NEGATIVE locked in as a test."""

import numpy as np

from holographic.mesh_and_geometry.holographic_sdf import sphere, menger
from holographic.io_and_interop.holographic_procbridge import procedural_compression, soft_min, fpe_smooth, _selftest


def _snr(clean, est):
    return 10 * np.log10(np.var(clean) / (np.var(clean - est) + 1e-12))


def test_c1_generator_size_is_constant_in_output_complexity():
    sizes = [procedural_compression(menger(d, 1.0), res=36) for d in (1, 2, 3)]
    dsl_bytes = {s["dsl_bytes"] for s in sizes}
    assert len(dsl_bytes) == 1                                  # constant generator size
    assert sizes[2]["mesh_faces"] > sizes[0]["mesh_faces"] * 1.3  # growing output complexity


def test_c1_compression_ratio_is_large():
    m = procedural_compression(menger(2, 1.0), res=40)
    assert m["ratio"] > 1000 and m["dsl_bytes"] < 64


def test_c4_soft_min_temperature_limit():
    a, b = -0.25, 0.10
    assert abs(soft_min(a, b, 0.001) - min(a, b)) < 1e-3       # k->0 is hard min
    assert soft_min(a, b, 0.5) < min(a, b)                     # finite k rounds below


def test_c4_smooth_union_temperature_limit():
    a = sphere(1.0); c = sphere(1.0).translate([1.5, 0, 0])
    P = np.array([[0.75, 0, 0.0]])
    hard = float(np.minimum(a.eval(P), c.eval(P))[0])
    gaps = [abs(float(a.smooth_union(c, k).eval(P)[0]) - hard) for k in (0.5, 0.1, 0.01)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 0.01      # -> hard union as k->0


def test_c2_fpe_smooth_is_a_kept_negative_on_uniform_sampling():
    rng = np.random.default_rng(0)
    N = 120; x = np.linspace(0, 1, N); clean = np.sin(2 * np.pi * 2 * x)
    noisy = clean + rng.normal(0, 0.3, N)
    # the honest, repeatable finding: the FPE kernel smoother does NOT beat the noisy signal here
    assert _snr(clean, fpe_smooth(x, noisy, bandwidth=6.0)) < _snr(clean, noisy)


def test_selftest_runs():
    _selftest()

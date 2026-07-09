"""Möbius / non-orientable encoders: axial (theta==theta+pi) and sign-flipping data."""
import numpy as np
from holographic.mesh_and_geometry.holographic_mobius import AxialEncoder, antiperiodic_fraction, antiperiodic_split


def test_axial_identifies_theta_and_theta_plus_pi():
    # the whole point: an orientation and its pi-flip are the SAME state -> similarity ~ +1.
    enc = AxialEncoder(256, seed=0)
    sims = [enc.similarity(t, t + np.pi) for t in np.linspace(0, np.pi, 9, endpoint=False)]
    assert min(sims) > 0.999


def test_naive_circle_would_disagree():
    # contrast: encoding the angle WITHOUT doubling (a plain circle) makes theta and theta+pi opposite.
    enc = AxialEncoder(256, seed=0)
    naive = lambda t: np.exp(1j * enc.freqs * t)            # single-angle == ordinary circle
    a, b = naive(0.7), naive(0.7 + np.pi)
    sim = float(np.real(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert sim < 0.5                                        # circle treats them as far apart -- wrong for axial


def test_axial_recovers_orientation_despite_pi_flips():
    # values reported as theta OR theta+pi at random are still recovered mod pi.
    enc = AxialEncoder(256, seed=1)
    rng = np.random.default_rng(2)
    true = rng.uniform(0, np.pi, 60)
    obs = true + rng.integers(0, 2, 60) * np.pi
    err = [min(abs(enc.decode(enc.encode(o)) - t), np.pi - abs(enc.decode(enc.encode(o)) - t))
           for o, t in zip(obs, true)]
    assert np.mean(err) < 0.05


def test_axial_merges_halfturn_on_purpose():
    # KEPT-NEGATIVE / scope: the encoder DELIBERATELY discards the directed distinction.
    enc = AxialEncoder(128, seed=0)
    assert np.allclose(enc.encode(0.9), enc.encode(0.9 + np.pi))


def test_antiperiodic_fraction_detects_sign_flip():
    T = 32
    t = np.arange(2 * T)
    flip = np.sin(np.pi * t / T) + 0.5 * np.sin(3 * np.pi * t / T)   # f(t+T) = -f(t)
    peri = np.sin(2 * np.pi * t / T)                                  # f(t+T) =  f(t)
    assert antiperiodic_fraction(flip) > 0.99
    assert antiperiodic_fraction(peri) < 0.01


def test_antiperiodic_split_orthogonal_reconstruction():
    x = np.arange(20.0)
    p, a = antiperiodic_split(x)
    assert np.allclose(p + a, x[:10]) and np.allclose(p - a, x[10:20])

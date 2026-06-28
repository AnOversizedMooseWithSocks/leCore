"""Tests for holographic_flatness: the band-limit-preservation regime test grounded in the real bind. Spectral
flatness predicts binding distortion; the Trefethen transient-growth concern does not materialize (linear ops
preserve a white spectrum, the cleanup contracts monotonically)."""

import numpy as np

from holographic_ai import bind, unbind, bundle, unitary_vector, random_vector
from holographic_hopfield import dense_cleanup
from holographic_flatness import spectral_flatness, binding_distortion, binding_stability

D = 1024


def _blend_key(alpha, rng):
    """A key whose spectrum interpolates from flat (alpha=0, unitary) to random-magnitude (alpha=1)."""
    ph = rng.uniform(-np.pi, np.pi, D); ph[0] = 0.0
    for k in range(1, D // 2 + 1):
        ph[D - k] = -ph[k]
    if D % 2 == 0:
        ph[D // 2] = 0.0
    mag = (1 - alpha) * np.ones(D // 2 + 1) + alpha * np.abs(rng.standard_normal(D // 2 + 1))
    key = np.fft.irfft(mag * np.exp(1j * ph[:D // 2 + 1]), D)
    return key / np.linalg.norm(key)


def test_flatness_separates_unitary_and_random():
    assert spectral_flatness(unitary_vector(D, np.random.default_rng(1))) > 0.97
    assert spectral_flatness(random_vector(D, np.random.default_rng(2))) < 0.75


def test_unitary_key_is_exact():
    assert binding_distortion(unitary_vector(D, np.random.default_rng(1))) < 1e-6


def test_random_key_is_lossy():
    assert binding_distortion(random_vector(D, np.random.default_rng(2))) > 0.5


def test_unitary_chain_stays_exact():
    rng = np.random.default_rng(0)
    target = random_vector(D, np.random.default_rng(3))
    s = target.copy()
    for _ in range(64):
        s = unbind(bind(s, unitary_vector(D, rng)), unitary_vector(D, rng))
    # note: each round uses two fresh unitary keys; bind then unbind by the same key is exact, the second key cancels
    # so re-derive with matched keys
    s = target.copy()
    for _ in range(64):
        k = unitary_vector(D, rng)
        s = unbind(bind(s, k), k)
    assert np.linalg.norm(s - target) < 1e-9


def test_flatness_predicts_distortion_monotonically():
    rng = np.random.default_rng(7)
    flats, dists = [], []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        key = _blend_key(alpha, rng)
        flats.append(spectral_flatness(key))
        dists.append(binding_distortion(key))
    assert all(flats[i] >= flats[i + 1] - 1e-6 for i in range(len(flats) - 1))
    assert all(dists[i] <= dists[i + 1] + 1e-6 for i in range(len(dists) - 1))


def test_linear_ops_preserve_white_spectrum():
    def hf(x):
        s = np.abs(np.fft.rfft(x)) ** 2
        return s[len(s) // 2:].sum() / s.sum()
    a, b = random_vector(D, np.random.default_rng(4)), random_vector(D, np.random.default_rng(5))
    assert 0.4 < hf(bind(a, b)) < 0.6
    assert 0.4 < hf(bundle([random_vector(D, np.random.default_rng(i)) for i in range(5)])) < 0.6
    assert 0.4 < hf(np.roll(a, 7)) < 0.6


def test_cleanup_contracts_monotonically_under_hf_perturbation():
    rng = np.random.default_rng(0)
    cb = np.array([random_vector(D, np.random.default_rng(100 + i)) for i in range(20)])
    clean = cb[7].copy()
    F = np.fft.rfft(rng.standard_normal(D)); F[:len(F) // 2] = 0
    hf = np.fft.irfft(F, D); hf = hf / np.linalg.norm(hf)
    q = clean + 1.0 * hf; q = q / np.linalg.norm(q)
    traj = [np.linalg.norm(q - clean)]
    s = q.copy()
    for _ in range(6):
        s = dense_cleanup(s, cb, beta=15.0, steps=1); s = s / np.linalg.norm(s)
        traj.append(np.linalg.norm(s - clean))
    assert all(traj[i + 1] <= traj[i] + 1e-9 for i in range(len(traj) - 1))


def test_binding_stability_report():
    rep_u = binding_stability(unitary_vector(D, np.random.default_rng(1)))
    rep_r = binding_stability(random_vector(D, np.random.default_rng(2)))
    assert rep_u["stable"] and rep_u["flatness"] > 0.97
    assert not rep_r["stable"] and rep_r["distortion"] > 0.5


def test_deterministic():
    uni = unitary_vector(D, np.random.default_rng(1))
    ran = random_vector(D, np.random.default_rng(2))
    assert spectral_flatness(uni) == spectral_flatness(uni)
    assert binding_distortion(ran) == binding_distortion(ran)

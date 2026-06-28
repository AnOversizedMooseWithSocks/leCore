"""Tests for RT-VI context-dependent meaning in a harmonic basis (holographic_harmonic): the spherical-harmonics
transfer -- a context-conditioned / polysemous atom whose decoded meaning is a function of a context angle, with the
DC term as the exact context-free fallback (backward-compatible), a smooth-variation win over per-context storage, and
the non-smooth degenerate trap kept loud."""

import numpy as np

from holographic_harmonic import harmonic_atom, harmonic_decode, harmonic_dc

_D = 256


def test_polysemy_each_sense_recovered():
    rng = np.random.default_rng(0)
    senses = [rng.standard_normal(_D) for _ in range(3)]
    senses = [s / np.linalg.norm(s) for s in senses]
    ctx = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    atom = harmonic_atom(ctx, senses, n_harmonics=2)
    for t, s in zip(ctx, senses):
        rec = harmonic_decode(atom, t)
        assert rec @ s / np.linalg.norm(rec) > 0.999


def test_between_context_blends_senses():
    rng = np.random.default_rng(0)
    senses = [rng.standard_normal(_D) for _ in range(3)]
    senses = [s / np.linalg.norm(s) for s in senses]
    ctx = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    atom = harmonic_atom(ctx, senses, n_harmonics=2)
    mid = harmonic_decode(atom, np.pi / 3)
    c0 = mid @ senses[0] / np.linalg.norm(mid)
    c1 = mid @ senses[1] / np.linalg.norm(mid)
    assert c0 > 0.3 and c1 > 0.3


def test_degree0_fallback_exact():
    # a context-free atom is captured by the DC alone and decodes exactly at any context (backward-compatible)
    rng = np.random.default_rng(2)
    const = rng.standard_normal(_D)
    atom = harmonic_atom([0.0, 1.0, 2.0, 3.0], [const, const, const, const], n_harmonics=1)
    assert np.linalg.norm(harmonic_decode(atom, 1.234) - const) < 1e-10
    assert np.linalg.norm(harmonic_dc(atom) - const) < 1e-10


def test_dc_is_the_context_free_mean():
    rng = np.random.default_rng(3)
    senses = [rng.standard_normal(_D) for _ in range(4)]
    th = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    atom = harmonic_atom(th, senses, n_harmonics=2)
    assert np.linalg.norm(harmonic_dc(atom) - np.mean(senses, axis=0)) < 1e-9


def test_smooth_band_limited_exact_at_K_equals_B_plus_1():
    r = np.random.default_rng(1)
    a0 = r.standard_normal(_D)
    pairs = [(r.standard_normal(_D), r.standard_normal(_D)) for _ in range(3)]  # B=3

    def content(theta):
        out = a0.copy()
        for k, (ak, bk) in enumerate(pairs, 1):
            out += ak * np.cos(k * theta) + bk * np.sin(k * theta)
        return out

    fit_th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    atom = harmonic_atom(fit_th, [content(t) for t in fit_th], n_harmonics=4)
    test_th = np.linspace(0.1, 2 * np.pi - 0.1, 50)
    truth = np.stack([content(t) for t in test_th])
    err = np.sqrt(((np.stack([harmonic_decode(atom, t) for t in test_th]) - truth) ** 2).sum(1)).mean()
    assert err < 1e-6


def test_beats_per_context_nearest_neighbor():
    r = np.random.default_rng(1)
    a0 = r.standard_normal(_D)
    pairs = [(r.standard_normal(_D), r.standard_normal(_D)) for _ in range(3)]

    def content(theta):
        out = a0.copy()
        for k, (ak, bk) in enumerate(pairs, 1):
            out += ak * np.cos(k * theta) + bk * np.sin(k * theta)
        return out

    test_th = np.linspace(0.1, 2 * np.pi - 0.1, 50)
    truth = np.stack([content(t) for t in test_th])
    fit_th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    atom = harmonic_atom(fit_th, [content(t) for t in fit_th], n_harmonics=4)  # 7 vectors
    h_err = np.sqrt(((np.stack([harmonic_decode(atom, t) for t in test_th]) - truth) ** 2).sum(1)).mean()
    pc_th = np.linspace(0, 2 * np.pi, 24, endpoint=False)                       # 24 vectors
    pc_store = np.stack([content(t) for t in pc_th])
    pc_rec = np.stack([pc_store[np.argmin(np.abs(((tt - pc_th + np.pi) % (2 * np.pi)) - np.pi))] for tt in test_th])
    pc_err = np.sqrt(((pc_rec - truth) ** 2).sum(1)).mean()
    assert h_err < pc_err


def test_non_smooth_degenerate_trap():
    r = np.random.default_rng(9)
    M = 12
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    vals = r.standard_normal((M, _D))
    atom = harmonic_atom(th, list(vals), n_harmonics=4)
    err = np.sqrt(((np.stack([harmonic_decode(atom, t) for t in th]) - vals) ** 2).sum(1)).mean()
    assert err > 1.0


def test_deterministic():
    rng = np.random.default_rng(0)
    senses = [rng.standard_normal(_D) for _ in range(3)]
    ctx = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    a1 = harmonic_atom(ctx, senses, n_harmonics=2)
    a2 = harmonic_atom(ctx, senses, n_harmonics=2)
    assert np.array_equal(a1["coeffs"], a2["coeffs"])

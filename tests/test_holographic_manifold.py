"""Tests for B9: topology detection (line/ring/mobius/torus), the matched basis extrapolating where
the flat-line assumption diverges, and the documented torus window requirement."""

import numpy as np

from holographic.agents_and_reasoning.holographic_symbolic import symbolic_regress
from holographic.mesh_and_geometry.holographic_manifold import detect_topology, line_dictionary, manifold_dictionary, decompose_on_manifold

W0 = 2 * np.pi / 5.0                         # period 5 -- OFF the elementary fixed-freq grid
X = np.linspace(0, 10, 400)                  # two periods
XE = np.linspace(10, 15, 200)               # extrapolate one more period


def test_detects_line_ring_mobius():
    assert detect_topology(X, 0.3 * X + 0.05 * X ** 2)[0] == "line"
    assert detect_topology(X, np.sin(W0 * X) + 0.5 * np.cos(2 * W0 * X))[0] == "ring"
    assert detect_topology(X, np.sin(W0 * X) + np.sin(3 * W0 * X))[0] == "mobius"


def test_detection_survives_noise():
    rng = np.random.default_rng(0)
    for s in range(3):
        n = 0.05 * rng.standard_normal(len(X))
        assert detect_topology(X, np.sin(W0 * X) + 0.5 * np.cos(2 * W0 * X) + n)[0] == "ring"
        assert detect_topology(X, np.sin(W0 * X) + np.sin(3 * W0 * X) + n)[0] == "mobius"


def test_detected_period_is_accurate():
    _, P = detect_topology(X, np.sin(W0 * X) + 0.5 * np.cos(2 * W0 * X))
    assert abs(P - 5.0) < 0.2                 # recovers the off-grid period ~5


def test_matched_basis_extrapolates_flat_line_diverges():
    y = np.sin(W0 * X) + 0.5 * np.cos(2 * W0 * X)
    true_e = np.sin(W0 * XE) + 0.5 * np.cos(2 * W0 * XE)
    fm, _ = decompose_on_manifold(X, y)
    fl, _ = symbolic_regress(X, y, dictionary=line_dictionary())
    em = np.sqrt(np.mean((fm.generate(XE) - true_e) ** 2))
    el = np.sqrt(np.mean((fl.generate(XE) - true_e) ** 2))
    assert em < 0.2 and el > 0.5              # matched extrapolates periodically; flat-line fails


def test_manifold_dictionary_shapes():
    ring = manifold_dictionary("ring", 5.0, n_harmonics=4)
    mob = manifold_dictionary("mobius", 5.0, n_harmonics=4)
    line = line_dictionary()
    assert all(k in ("cos", "sin") for k, _ in ring)
    # mobius uses only ODD harmonics: every freq is an odd multiple of the fundamental
    w = 2 * np.pi / 5.0
    assert all(round((p / w)) % 2 == 1 for _, p in mob)
    assert all(k == "pow" for k, _ in line)


def test_decompose_on_manifold_records_topology_and_recovers():
    y = np.sin(W0 * X) + np.sin(3 * W0 * X)
    f, info = decompose_on_manifold(X, y)
    assert info["topology"] == "mobius" and abs(info["period"] - 5.0) < 0.2
    assert np.sqrt(np.mean((f.generate(X) - y) ** 2)) < 0.1   # fits in-window


def test_torus_needs_a_long_enough_window():
    # two incommensurate tones: unresolvable on a short window (Rayleigh), resolved on a long one
    xs = np.linspace(0, 12, 480)
    xl = np.linspace(0, 60, 2400)
    short = detect_topology(xs, np.sin(xs) + np.sin(np.sqrt(2) * xs))[0]
    long = detect_topology(xl, np.sin(xl) + np.sin(np.sqrt(2) * xl))[0]
    assert short != "torus"                   # falls back rather than guessing
    assert long == "torus"                     # resolves when the window is long enough


def test_line_signal_is_not_forced_onto_a_ring():
    # a non-periodic ramp+quadratic must classify as line, not a spurious ring
    assert detect_topology(X, 1.0 + 0.4 * X + 0.03 * X ** 2)[0] == "line"

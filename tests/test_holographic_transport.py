"""BLD-8: the Wasserstein (earth-mover's) distance by Sinkhorn iteration (holographic_transport.py) -- a
transport-geometry distance that beats bin-wise Euclidean/cosine when WHERE the mass sits matters."""
import numpy as np

from holographic.misc.holographic_transport import wasserstein, _default_cost, _selftest


def _gauss(x, mu, sig=2.0):
    g = np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return g / g.sum()


def test_selftest_passes():
    _selftest()


def test_matches_1d_closed_form():
    # 1-D Wasserstein-1 has the closed form W1 = sum_i |CDF_a(i) - CDF_b(i)|; Sinkhorn matches it.
    n = 50
    x = np.arange(n)
    C = np.abs(x[:, None] - x[None, :]).astype(float)
    a, b = _gauss(x, 12), _gauss(x, 30)
    w_true = float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))
    assert abs(wasserstein(a, b, C, eps=0.5) - w_true) < 0.1


def test_tracks_shift_where_binwise_metrics_saturate():
    # THE WIN: W grows with the shift even after support stops overlapping; Euclidean/cosine cannot.
    n = 60
    x = np.arange(n)
    C = np.abs(x[:, None] - x[None, :]).astype(float)
    ref = _gauss(x, 15)
    W, E, K = [], [], []
    for shift in (5, 10, 20):
        s = _gauss(x, 15 + shift)
        W.append(wasserstein(ref, s, C, eps=0.5))
        E.append(np.linalg.norm(ref - s))
        K.append(float((ref @ s) / (np.linalg.norm(ref) * np.linalg.norm(s))))
    assert W[0] < W[1] < W[2] and W[2] / W[0] > 3.5          # W spans the full 4x
    assert (E[2] - E[1]) < 0.05                              # Euclidean has saturated (blind to distance)
    assert K[1] < 0.01 and K[2] < 0.01                      # cosine has collapsed to ~0 for both


def test_eps_knob_is_a_kept_negative():
    n = 50
    x = np.arange(n)
    C = np.abs(x[:, None] - x[None, :]).astype(float)
    # large eps blurs the distance high (toward the independent-coupling cost)
    a, b = _gauss(x, 25, 1.5), _gauss(x, 25, 6.0)
    w_true = float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))
    assert wasserstein(a, b, C, eps=50.0) > 1.2 * w_true
    # tiny eps underflows the kernel between separated supports -> a broken (non-finite or wildly wrong) answer
    p, q = _gauss(x, 15), _gauss(x, 25)
    w_tiny = wasserstein(p, q, C, eps=0.01)
    assert (not np.isfinite(w_tiny)) or abs(w_tiny - 10.0) > 2.0


def test_custom_cost_matrix():
    # a non-default cost: bins on a RING (cyclic distance) -- mass can wrap around, shrinking the distance.
    n = 12
    x = np.arange(n)
    lin = np.abs(x[:, None] - x[None, :])
    ring = np.minimum(lin, n - lin).astype(float)           # cyclic ground distance
    a = (x == 0).astype(float)
    b = (x == 11).astype(float)                              # one step away on the ring (11 on a line)
    assert wasserstein(a, b, ring, eps=0.1) < wasserstein(a, b, lin.astype(float), eps=0.1)


def test_self_distance_is_small_relative_to_cross():
    n = 50
    x = np.arange(n)
    C = np.abs(x[:, None] - x[None, :]).astype(float)
    a, b = _gauss(x, 15), _gauss(x, 30)
    assert wasserstein(a, a, C, eps=0.5) < 0.15 * wasserstein(a, b, C, eps=0.5)


def test_default_cost_and_eps():
    # with neither cost nor eps given, the 1-D default cost and the eps rule produce a sensible distance.
    n = 50
    x = np.arange(n)
    a, b = _gauss(x, 15), _gauss(x, 25)
    w_true = float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))
    assert abs(wasserstein(a, b) - w_true) < 0.3
    assert _default_cost(3, 3).shape == (3, 3)


def test_deterministic():
    n = 30
    x = np.arange(n)
    a, b = _gauss(x, 8), _gauss(x, 20)
    assert wasserstein(a, b, eps=0.5) == wasserstein(a, b, eps=0.5)

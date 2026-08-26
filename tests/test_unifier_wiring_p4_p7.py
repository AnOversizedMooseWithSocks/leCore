"""P4/P5/P7 -- the promote-and-wire results, pinned with the measurements that justified them."""
import numpy as np
import pytest


def test_poc_simultaneous_sweep_reproduces_block_update_exactly():
    """P7: the resonator's Jacobi block update IS project_onto_constraints' 'simultaneous' sweep on disjoint blocks."""
    from holographic.rendering.holographic_denoise import project_onto_constraints as poc
    x = np.zeros(6)
    p0 = lambda v: np.concatenate([np.ones(3), v[3:]])
    p1 = lambda v: np.concatenate([v[:3], np.full(3, 2.0)])
    out, _, _ = poc(x, [p0, p1], iters=1, sweep="simultaneous")
    assert np.allclose(out, [1, 1, 1, 2, 2, 2])                 # disjoint blocks: moves sum, no double-count
    avg, _, _ = poc(x, [p0, p1], iters=1, sweep="simultaneous", average=True)
    assert np.allclose(avg, [0.5, 0.5, 0.5, 1, 1, 1])           # Cimmino averaging halves both moves
    with pytest.raises(ValueError):
        poc(x, [p0], sweep="bogus")


def test_resonator_still_factors_after_delegating():
    from holographic.misc.holographic_resonator import ResonatorNetwork
    rng = np.random.default_rng(0)
    dim, F, L = 512, 3, 8
    books = [np.sign(rng.standard_normal((L, dim))) for _ in range(F)]
    r = ResonatorNetwork(books)
    truth = (2, 5, 1)
    c = np.ones(dim)
    for f, i in enumerate(truth):
        c = c * books[f][i]
    assert r.factor(c)["factors"] == truth


def test_dynamics_step_is_not_a_projection():
    """The measurement behind the P7 retraction: a projection is idempotent; advancing a state is not."""
    from holographic.agents_and_reasoning.holographic_ai import bind
    rng = np.random.default_rng(0)
    dim = 128
    U = rng.standard_normal(dim) / np.sqrt(dim)
    x = rng.standard_normal(dim)
    assert not np.allclose(bind(U, bind(U, x)), bind(U, x))


def test_prt_delegates_to_the_lowdiscrepancy_home():
    """P4: one spherical-Fibonacci lattice, in the low-discrepancy home."""
    from holographic.misc.holographic_prt import _sphere_dirs
    from holographic.sampling_and_signal.holographic_lowdiscrepancy import sphere_directions
    a, b = _sphere_dirs(256), sphere_directions(256)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0)          # unit directions
    assert np.abs(b.mean(0)).max() < 0.01                       # near-uniform over the sphere


def test_randomized_qmc_hemisphere_lowers_variance_and_default_is_unchanged():
    """P4/L2': QMC on the low-dimensional smooth primary integral, RANDOMIZED so variance stays estimable."""
    from holographic.sampling_and_signal.holographic_samplinghome import Sampling
    N = np.array([[0.0, 0.0, 1.0]])

    def est(n, ld, seed):
        d = Sampling.cosine_hemisphere(N, n, seed=seed, low_discrepancy=ld)[0]
        return float(np.mean(d[:, 2] ** 2))

    white = [est(256, False, s) for s in range(16)]
    qmc = [est(256, True, s) for s in range(16)]
    assert np.var(qmc) < np.var(white) / 10.0                   # measured ~121x at n=256
    assert np.var(qmc) > 0.0                                    # randomized -> still an unbiased, measurable estimator
    # the default path is untouched
    assert np.allclose(Sampling.cosine_hemisphere(N, 8, seed=0),
                       Sampling.cosine_hemisphere(N, 8, seed=0, low_discrepancy=False))


def test_accumulate_exact_is_order_independent():
    """P5: bit-identical across bucket orders; ema refuses (it is order-dependent by design)."""
    from holographic.misc.holographic_accumulate import robust_accumulate
    rng = np.random.default_rng(0)
    S = [rng.standard_normal(16) * 10.0 ** rng.integers(-6, 6) for _ in range(20)]
    idx = list(range(20)); rng.shuffle(idx)
    S2 = [S[i] for i in idx]
    assert not np.array_equal(robust_accumulate(S, "mean"), robust_accumulate(S2, "mean"))       # float: order matters
    assert np.array_equal(robust_accumulate(S, "mean", exact=True),
                          robust_accumulate(S2, "mean", exact=True))                             # exact: it does not
    with pytest.raises(ValueError):
        robust_accumulate(S, "ema", exact=True)

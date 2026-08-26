"""X8 -- the tuning bank: M variants of one island, advanced in one pass.

The backlog proposed superposing M (friction, stiffness) variants into ONE hypervector under distinct keys and
unbinding to read any of them, "budget M <= D/256". Two independent things are wrong with that, and both are
pinned here:

  (1) the capacity law is already RETRACTED (holographic_shader's H7 note): unbinding one of M keyed items returns
      the item plus M-1 random vectors, so fidelity follows 1/sqrt(M), and sqrt(M/D) is a different quantity.
  (2) worse, the object does not superpose. s_k = A^k s0 + G(A,k) b is LINEAR in the forcing b and NONLINEAR in
      the operator A. Stiffness and friction live in A.

What works is B7's other half -- data-oriented SoA -- because numpy.linalg.eig is vectorised over a stack of
matrices. M variants, one batched eigendecomposition, any horizon, exact, no capacity budget.
"""

import numpy as np
import pytest

from holographic.misc.holographic_iterate import (
    affine_step_k, affine_transfer, affine_step_k_batch, affine_transfer_batch)
from holographic.simulation_and_physics.holographic_modal import (
    soft_chain_matrices, soft_chain_bank, advance_bank, blend_forcings)


def _substep_batch(As, bs, S0, n):
    S = np.asarray(S0, float).copy()
    for _ in range(int(n)):
        S = np.einsum("mij,mj->mi", As, S) + bs
    return S


def test_the_bank_builds_m_variants_and_broadcasts_its_dials():
    As, bs, h = soft_chain_bank(6, hertz=np.linspace(5.0, 40.0, 8), zeta=0.7, substeps=32)
    assert As.shape == (8, 12, 12) and bs.shape == (8, 12)

    # scalar hertz with a vector zeta broadcasts, and vice versa
    A2, b2, _ = soft_chain_bank(6, hertz=15.0, zeta=np.linspace(0.3, 1.5, 5), substeps=32)
    assert A2.shape[0] == 5
    A3, b3, _ = soft_chain_bank(6, hertz=15.0, zeta=0.7, substeps=32)
    assert A3.shape[0] == 1                                       # a scalar dial is a bank of one

    # variant m must equal the single-variant builder for the same dials -- bit-identically
    single, sb, _ = soft_chain_matrices(6, hertz=40.0, zeta=0.7, substeps=32)
    assert np.array_equal(As[-1], single) and np.array_equal(bs[-1], sb)


def test_the_batched_jump_matches_substepping_every_variant():
    M, N = 8, 600
    As, bs, _ = soft_chain_bank(6, hertz=np.linspace(5.0, 40.0, M), zeta=0.7, substeps=32)
    S0 = np.zeros((M, 12))
    assert np.abs(advance_bank(S0, As, bs, N) - _substep_batch(As, bs, S0, N)).max() < 1e-10
    assert np.abs(advance_bank(S0, As, bs, N)).max() > 1e-3       # the chains actually sagged


def test_each_bank_row_equals_the_single_variant_solve():
    M = 5
    hz = np.linspace(5.0, 40.0, M)
    As, bs, _ = soft_chain_bank(6, hertz=hz, zeta=0.7, substeps=32)
    S = advance_bank(np.zeros((M, 12)), As, bs, 400)
    for m in range(M):
        one = affine_step_k(np.zeros(12), As[m], bs[m], 400)
        assert np.abs(S[m] - one).max() < 1e-10                   # the batch is not a different computation


def test_the_horizon_is_free_for_the_whole_bank():
    M = 6
    As, bs, _ = soft_chain_bank(6, hertz=np.linspace(5.0, 40.0, M), zeta=0.7, substeps=32)
    tr = affine_transfer_batch(As)                                # ONE batched eigendecomposition ...
    S0 = np.zeros((M, 12))
    for k in (100, 1000, 10_000):                                 # ... reused across horizons
        assert np.abs(affine_step_k_batch(S0, As, bs, k, transfer=tr) - _substep_batch(As, bs, S0, k)).max() < 1e-9


def test_batch_edge_cases_and_validation():
    As, bs, _ = soft_chain_bank(4, hertz=np.array([10.0, 20.0]), zeta=0.7, substeps=16)
    S0 = np.zeros((2, 8))
    assert np.array_equal(affine_step_k_batch(S0, As, bs, 0), S0)  # k=0 is the identity
    with pytest.raises(ValueError):
        affine_step_k_batch(S0, As, bs, -1)
    with pytest.raises(ValueError):
        affine_transfer_batch(np.zeros((3, 4)))                    # not a stack of square matrices


def test_a_defective_variant_is_refused_by_name_not_silently_poisoning_one_row():
    good, _, _ = soft_chain_matrices(4, hertz=15.0, zeta=0.7, substeps=16)
    d = good.shape[0]
    jordan = np.eye(d) + np.eye(d, k=1) * 0.01                     # a Jordan block: no eigenbasis
    stack = np.stack([good, jordan])
    with pytest.raises(ValueError) as ei:
        affine_transfer_batch(stack)
    assert "1" in str(ei.value)                                    # it names the offending variant


# ---------------------------------------------------------------------------------------------------------
# THE DIVIDING LINE: linear in the forcing, nonlinear in the operator.
# ---------------------------------------------------------------------------------------------------------

def test_forcing_variants_superpose_exactly():
    # s_k = A^k s0 + G(A,k) b is LINEAR in b. So a blend of forcings IS the blend of trajectories -- one
    # eigendecomposition serves every forcing variant and every blend of them (gravity, wind, a load dial).
    A, b, _ = soft_chain_matrices(6, hertz=15.0, zeta=0.7, substeps=32)
    tr = affine_transfer(A)
    f1 = b.copy()
    f2 = b * 0.25
    w = np.array([0.3, 0.7])
    blended = blend_forcings(np.zeros(12), A, np.stack([f1, f2]), w, 600, transfer=tr)
    separate = (w[0] * affine_step_k(np.zeros(12), A, f1, 600, transfer=tr)
                + w[1] * affine_step_k(np.zeros(12), A, f2, 600, transfer=tr))
    assert np.abs(blended - separate).max() < 1e-12                # exact, to machine precision

    # ... and blending is ONE solve, not M: the weights enter before the recurrence
    assert np.abs(blend_forcings(np.zeros(12), A, np.stack([f1, f2]), np.array([1.0, 0.0]), 600, transfer=tr)
                  - affine_step_k(np.zeros(12), A, f1, 600, transfer=tr)).max() < 1e-12


def test_blend_forcings_validates_its_weights():
    A, b, _ = soft_chain_matrices(4, hertz=15.0, zeta=0.7, substeps=16)
    with pytest.raises(ValueError):
        blend_forcings(np.zeros(8), A, np.stack([b, b]), np.array([1.0]), 10)


def test_kept_negative_operator_variants_do_not_superpose_at_any_dimension():
    # THE BACKLOG'S PREMISE, falsified. Stiffness lives in A, and the map is not linear in A.
    A1, b, _ = soft_chain_matrices(6, hertz=5.0, zeta=0.7, substeps=32)
    A2, _, _ = soft_chain_matrices(6, hertz=40.0, zeta=0.7, substeps=32)
    mixed = 0.5 * A1 + 0.5 * A2
    lhs = affine_step_k(np.zeros(12), mixed, b, 600)               # solve the blended operator
    rhs = 0.5 * affine_step_k(np.zeros(12), A1, b, 600) + 0.5 * affine_step_k(np.zeros(12), A2, b, 600)
    assert np.abs(lhs - rhs).max() > 1e-3, "blending operators is supposed to be nonsense"

    # ... and the dimension is irrelevant: it is not a capacity problem, it is an algebra problem.
    for n in (4, 6, 12):
        Aa, bb, _ = soft_chain_matrices(n, hertz=5.0, zeta=0.7, substeps=32)
        Ab, _, _ = soft_chain_matrices(n, hertz=40.0, zeta=0.7, substeps=32)
        z = np.zeros(2 * n)
        l = affine_step_k(z, 0.5 * Aa + 0.5 * Ab, bb, 400)
        r = 0.5 * affine_step_k(z, Aa, bb, 400) + 0.5 * affine_step_k(z, Ab, bb, 400)
        assert np.abs(l - r).max() > 1e-4, n


def _unitary_atom(rng, D):
    """A UNITARY hypervector (|FFT| == 1), so unbind is exact and the only error is crosstalk. Non-unitary keys
    make deconvolution amplify noise and confound the measurement -- that mistake cost me a red test."""
    ph = rng.uniform(0.0, 2.0 * np.pi, D // 2 + 1)
    ph[0] = 0.0
    if D % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.exp(1j * ph), n=D)


def test_kept_negative_the_keyed_bundle_fidelity_is_one_over_sqrt_m_not_sqrt_m_over_d():
    # The retracted capacity law, re-measured with the ENGINE'S OWN bind/unbind so X8's docstring cannot drift
    # back to "M <= D/256". Unbinding one of M keyed items returns it plus M-1 random vectors -> cos ~ 1/sqrt(M).
    # Measured at D=8192: M=2 -> 0.720, M=8 -> 0.359, M=32 -> 0.173. sqrt(M/D) is 0.0156 .. 0.0625: a DIFFERENT
    # quantity (the cosine with a WRONG item), and using it as a capacity budget is the error being pinned.
    import lecore
    D = 8192
    rng = np.random.default_rng(0)
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    for M in (2, 4, 8, 16, 32):
        items = rng.normal(size=(M, D))
        keys = np.stack([_unitary_atom(rng, D) for _ in range(M)])
        bundle = sum(lecore.bind(k, v) for k, v in zip(keys, items))
        c = cos(lecore.unbind(bundle, keys[0]), items[0])
        assert abs(c - 1.0 / np.sqrt(M)) < 0.02, (M, c)            # follows 1/sqrt(M), tightly
        assert c > 2.5 * np.sqrt(M / D)                            # ... and sqrt(M/D) is nowhere near it


def test_the_bank_is_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    As, bs, h = m.soft_chain_bank(6, np.linspace(5.0, 40.0, 8), 0.7, substeps=32)
    S = m.advance_bank(np.zeros((8, 12)), As, bs, 600)
    assert S.shape == (8, 12)
    assert np.abs(S - _substep_batch(np.asarray(As), np.asarray(bs), np.zeros((8, 12)), 600)).max() < 1e-10

    A, b, _ = m.soft_chain_matrices(6, hertz=15.0, zeta=0.7, substeps=32)
    got = m.blend_forcings(np.zeros(12), A, np.stack([b, b * 0.25]), np.array([0.3, 0.7]), 600)
    assert got.shape == (12,)

    assert "Modal jump" in str(m.find_capability("sweep friction and stiffness settings at once")[:3])
    # and "how many variants fit in one vector" must route to the entry that OWNS the crosstalk negative,
    # not to the tuning bank -- because the honest answer there is "none; they do not superpose".
    assert "Blend M shader variants" in str(m.find_capability("how many variants fit in one vector")[:3])

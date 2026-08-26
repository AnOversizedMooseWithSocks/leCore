"""Sparse readout in the SBC resonator (the session's finding applied to factorization).

Pins: (1) softmax default unchanged; (2) the MEASURED capacity win (sparse recovers factorizations
softmax cannot at high alphabet, no regression at low); (3) the confidence null is matched to the
readout (the project's procedure-matched-null rule); (4) ABOVE/BELOW -- the mind delegates to the
SBC factorizer and threads the readout, no reimplementation.
"""
import numpy as np
from holographic.misc.holographic_sbc import sbc_codebook, sbc_reconstruct, sbc_resonator, decompose_structure, _resonator_noise_null
from holographic.misc.holographic_unified import UnifiedMind


def _cap(N, readout, trials=30, F=3, B=16, L=16):
    """Fraction of problems where ALL factors are recovered, at alphabet N."""
    ok = 0
    for s in range(trials):
        rng = np.random.default_rng(s)
        cbs = [sbc_codebook(B, L, N, seed=1000 + f + s * 13) for f in range(F)]
        true = tuple(int(rng.integers(N)) for _ in range(F))
        prod = sbc_reconstruct(true, cbs, L)
        picks, _ = sbc_resonator(prod, cbs, L, restarts=6, iters=50, seed=s, readout=readout)
        ok += int(tuple(picks) == true)
    return ok / trials


def test_softmax_default_unchanged():
    # default readout is softmax and gives identical picks to the explicit softmax path (backward compatible)
    rng = np.random.default_rng(0)
    cbs = [sbc_codebook(16, 16, 20, seed=f) for f in range(3)]
    true = tuple(int(rng.integers(20)) for _ in range(3))
    prod = sbc_reconstruct(true, cbs, 16)
    a, va = sbc_resonator(prod, cbs, 16, seed=7)
    b, vb = sbc_resonator(prod, cbs, 16, seed=7, readout="softmax")
    assert a == b and va == vb


def test_sparse_raises_capacity():
    # the measured win: sparse recovers factorizations softmax cannot at high alphabet, and never regresses
    assert _cap(25, "sparsemax") > _cap(25, "softmax") + 0.05       # clear win in the mid regime
    assert _cap(50, "sparsemax") > _cap(50, "softmax")             # recovers where softmax collapses to ~0
    assert _cap(10, "sparsemax") >= _cap(10, "softmax") - 1e-9      # no regression where both work


def test_confidence_null_matched_to_readout():
    # the procedure-matched-null rule: the noise floor is recomputed under the chosen readout, not shared
    cbs = [sbc_codebook(8, 8, 8, seed=f) for f in range(3)]
    n_soft = _resonator_noise_null(cbs, 8, restarts=4, iters=30, m=60, readout="softmax")
    n_sparse = _resonator_noise_null(cbs, 8, restarts=4, iters=30, m=60, readout="sparsemax")
    assert not np.array_equal(n_soft, n_sparse)                    # separate nulls -> readout is in the key


def test_mind_decompose_threads_readout_and_delegates():
    # ABOVE/BELOW: the mind's decompose_structure IS the SBC factorizer with the readout passed through
    rng = np.random.default_rng(2)
    cbs = [sbc_codebook(16, 16, 18, seed=10 + f) for f in range(3)]
    true = tuple(int(rng.integers(18)) for _ in range(3))
    prod = sbc_reconstruct(true, cbs, 16)
    mind = UnifiedMind(dim=256, seed=5)
    for ro in ("softmax", "sparsemax"):
        m = mind.decompose_structure(prod, cbs, 16, restarts=6, iters=50, seed=3, readout=ro)
        d = decompose_structure(np.asarray(prod), cbs, 16, restarts=6, iters=50, seed=3, readout=ro)
        assert tuple(m["picks"]) == tuple(d["picks"])              # delegation, not reimplementation
    # a clean, easy product (small alphabet) is recovered + verified through the mind with sparsemax
    cbs_e = [sbc_codebook(16, 16, 6, seed=30 + f) for f in range(3)]
    true_e = (2, 4, 1)
    prod_e = sbc_reconstruct(true_e, cbs_e, 16)
    res = mind.decompose_structure(prod_e, cbs_e, 16, restarts=6, iters=50, seed=0, readout="sparsemax")
    assert tuple(res["picks"]) == true_e and res["verified"]


def test_factor_composite_threads_readout():
    # factor_composite (the one-entry factorizer) also accepts the readout and solves a clean SBC product
    rng = np.random.default_rng(4)
    cbs = [sbc_codebook(16, 16, 15, seed=20 + f) for f in range(3)]
    true = tuple(int(rng.integers(15)) for _ in range(3))
    prod = sbc_reconstruct(true, cbs, 16)
    mind = UnifiedMind(dim=256, seed=0)
    out = mind.factor_composite(prod, cbs, L=16, restarts=6, iters=50, seed=1, readout="sparsemax")
    assert out["backend"] == "sbc" and tuple(out["factors"]) == true and out["solved"]

"""Sparse block codes + scaled resonator (B2): exact block algebra, validated factorization."""
import numpy as np
from holographic.misc.holographic_sbc import sbc_random, sbc_codebook, sbc_bind, sbc_unbind, sbc_reconstruct, sbc_resonator


def test_block_bind_unbind_is_exact():
    a = sbc_random(16, 16, 1); b = sbc_random(16, 16, 2)
    assert np.array_equal(sbc_unbind(sbc_bind(a, b, 16), a, 16), b)   # modular arithmetic is exact


def test_resonator_factors_a_clean_product():
    B, L = 16, 16
    cbs = [sbc_codebook(B, L, 10, seed=s) for s in (0, 1, 2)]
    true = (3, 7, 1)
    P = sbc_reconstruct(true, cbs, L)
    picks, validated = sbc_resonator(P, cbs, L, seed=0)
    assert picks == true and validated


def test_confidence_validates_correctness():
    """validated=True must mean the picks actually reconstruct the product (precision ~1.0)."""
    B, L = 16, 16
    for s in range(8):
        cbs = [sbc_codebook(B, L, 12, seed=10 * s + k) for k in range(3)]
        rng = np.random.default_rng(s)
        true = tuple(int(rng.integers(0, 12)) for _ in range(3))
        P = sbc_reconstruct(true, cbs, L)
        picks, validated = sbc_resonator(P, cbs, L, seed=s)
        if validated:
            assert np.array_equal(sbc_reconstruct(picks, cbs, L), P)   # verified => correct factorization


def test_resonator_high_coverage_on_small_alphabet():
    B, L = 16, 16
    hits = 0
    for s in range(10):
        cbs = [sbc_codebook(B, L, 10, seed=100 * s + k) for k in range(3)]
        rng = np.random.default_rng(s)
        true = tuple(int(rng.integers(0, 10)) for _ in range(3))
        P = sbc_reconstruct(true, cbs, L)
        picks, validated = sbc_resonator(P, cbs, L, seed=s)
        hits += (validated and picks == true)
    assert hits >= 8                                        # ~>=0.8 coverage at N=10


def test_abstains_on_corrupted_product():
    """A corrupted product has no exact factorization, so the resonator must NOT claim validation."""
    B, L = 16, 16
    cbs = [sbc_codebook(B, L, 10, seed=k) for k in range(3)]
    P = sbc_reconstruct((2, 4, 6), cbs, L).copy()
    P[0] = (P[0] + 3) % L; P[5] = (P[5] + 7) % L            # corrupt two blocks
    _, validated = sbc_resonator(P, cbs, L, seed=1)
    assert not validated                                    # honest abstention under corruption


def test_decompose_structure_recovers_and_verifies():
    from holographic.misc.holographic_sbc import decompose_structure
    B, L = 16, 16
    cbs = [sbc_codebook(B, L, 10, seed=k) for k in range(3)]
    true = (2, 5, 8)
    P = sbc_reconstruct(true, cbs, L)
    out = decompose_structure(P, cbs, L, seed=0)
    assert out["picks"] == true and out["verified"]
    assert np.array_equal(sbc_reconstruct(out["picks"], cbs, L), P)   # recipe rebuilds the structure


def test_decompose_detects_absent_factor():
    from holographic.misc.holographic_sbc import decompose_structure, sbc_identity
    B, L = 16, 16
    cbs = [list(sbc_codebook(B, L, 8, seed=10 + k)) + [sbc_identity(B)] for k in range(3)]
    factors = [cbs[0][3], cbs[1][6], sbc_identity(B)]                 # third factor absent
    P = factors[0].copy()
    for f in (1, 2):
        P = sbc_bind(P, factors[f], L)
    out = decompose_structure(P, cbs, L, seed=1)
    assert out["verified"] and out["present"] == [True, True, False]  # presence detection


def test_factors_cannot_be_read_off_a_product_naively():
    """A bound product is dissimilar to its factors: per-factor readout is ~chance; the resonator isn't."""
    from holographic.misc.holographic_sbc import sbc_onehot, decompose_structure
    B, L = 16, 16
    cbs = [sbc_codebook(B, L, 10, seed=k) for k in range(3)]
    true = (2, 5, 8)
    P = sbc_reconstruct(true, cbs, L); Po = sbc_onehot(P, L)
    naive = tuple(int(max(range(10), key=lambda i: float((sbc_onehot(cbs[f][i], L) * Po).sum())))
                  for f in range(3))
    out = decompose_structure(P, cbs, L, seed=0)
    assert naive != true                                             # naive fails (deconfounding needed)
    assert out["picks"] == true and out["verified"]                  # the joint verified search succeeds


def test_topk_readout_recovers_and_verifies():
    """The TopK readout (Gao et al. 2024) -- the high-load option in the same energy family as
    softmax/sparsemax -- runs through the resonator, recovers a VERIFIED factorization, and its
    calibrated confidence null is matched to the readout+k (its measured win at very high N is in
    NOTES_concepts)."""
    import numpy as np
    from holographic.misc.holographic_sbc import decompose_structure, sbc_reconstruct
    L, B, F, N = 64, 16, 3, 40
    r = np.random.default_rng(0)
    cbs = [[tuple(r.integers(0, L, size=B)) for _ in range(N)] for _ in range(F)]
    true = tuple(int(r.integers(N)) for _ in range(F))
    P = np.asarray(sbc_reconstruct(true, cbs, L))
    out = decompose_structure(P, cbs, L, readout="topk", k=8, confidence=True)
    assert out["picks"] == true and out["verified"]       # topk recovers and verifies
    assert out["pvalue"] < 0.05                            # calibrated confidence (null matched to topk)

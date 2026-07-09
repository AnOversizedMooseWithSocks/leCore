"""Tests for self-verifying storage (BLD-1): the holographic Merkle tree (bind + bundle)."""
import math
import numpy as np
from holographic.misc.holographic_verify import CompositionTree, _selftest
from holographic.misc.holographic_unified import UnifiedMind


def test_holographic_verify_selftest():
    # the module's own measured guarantees: the bar + the kept negatives, all deterministic
    assert _selftest()


def test_detect_and_localize_single_tamper_in_log_checks():
    rng = np.random.default_rng(1)
    D, n = 512, 32
    items = [rng.standard_normal(D) for _ in range(n)]
    tree = CompositionTree(items, seed=3)
    for _ in range(20):
        j = int(rng.integers(n))
        it = list(items); it[j] = rng.standard_normal(D)
        idx, checks = tree.locate(it)
        assert idx == j                                  # localised the exact tampered slot
        assert checks <= int(math.log2(n)) + 1           # in <= log2(n)+1 composite comparisons
    assert tree.locate(items)[0] is None                 # clean store: no tamper reported


def test_position_binding_catches_reordering():
    # a plain bundle is commutative; binding position into each leaf makes a swap detectable + localisable
    rng = np.random.default_rng(2)
    D, n = 256, 16
    items = [rng.standard_normal(D) for _ in range(n)]
    tree = CompositionTree(items, seed=0)
    sw = list(items); sw[2], sw[11] = sw[11], sw[2]
    assert tree.locate(sw)[0] is not None                # the swap is caught (would slip a plain bundle)


def test_linear_collision_is_constructible_kept_negative():
    # THE kept negative: the root is linear, so a key-aware adversary cancels a change by deconvolution and
    # leaves the root ~unchanged -- corruption-evidence, NOT cryptographic tamper-proofing.
    from holographic.agents_and_reasoning.holographic_ai import bind, cosine
    rng = np.random.default_rng(4)
    D, n = 512, 32
    items = [rng.standard_normal(D) for _ in range(n)]
    tree = CompositionTree(items, seed=5)
    a, b = 7, 20
    da = rng.standard_normal(D)
    db = np.fft.irfft(np.fft.rfft(-bind(tree.positions[a], da)) / np.fft.rfft(tree.positions[b]), n=D)
    forged = list(items); forged[a] = items[a] + da; forged[b] = items[b] + db
    fr = sum(bind(tree.positions[i], np.asarray(forged[i], float)) for i in range(n))
    u = lambda v: v / (np.linalg.norm(v) + 1e-12)
    assert cosine(u(fr), u(tree.root())) > 0.999         # the forged store passes -- an invisible collision


def test_verify_store_faculty_round_trips():
    # the UnifiedMind faculty: commit, verify clean, detect + localise a tamper
    m = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(7)
    items = [rng.standard_normal(512) for _ in range(20)]
    tree = m.verify_store(items)
    assert tree.verify(items)                            # the committed items verify
    tampered = list(items); tampered[13] = rng.standard_normal(512)
    assert not tree.verify(tampered)                     # a change is detected
    assert tree.locate(tampered)[0] == 13                # and localised

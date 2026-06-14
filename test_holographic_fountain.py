"""LT fountain codes: exact rateless erasure recovery (collect any k(1+eps)
droplets), the information floor below k (the honest cliff), order/erasure
blindness, and the ~20% overhead tax."""
import numpy as np
import random

from holographic_fountain import Fountain, robust_soliton, recovery_curve


def test_soliton_is_a_distribution():
    mu = robust_soliton(200)
    assert abs(mu.sum() - 1.0) < 1e-9
    assert mu[0] == 0.0 and mu[1] > 0          # degree-1 spike present to seed peeling


def test_exact_recovery_above_threshold():
    rng = np.random.default_rng(0)
    blocks = [rng.integers(0, 256, size=16, dtype=np.uint8) for _ in range(300)]
    f = Fountain(blocks)
    drops = f.droplets(int(300 * 1.4), seed=1)
    rec = Fountain.decode(drops, 300)
    assert all(r is not None and np.array_equal(r, b) for r, b in zip(rec, blocks))


def test_information_floor_below_k():
    # You cannot recover k blocks from fewer than k droplets -- the honest cliff,
    # not a flaw. Decode from exactly k droplets should not fully succeed.
    rng = np.random.default_rng(0)
    blocks = [rng.integers(0, 256, size=8, dtype=np.uint8) for _ in range(200)]
    f = Fountain(blocks)
    drops = f.droplets(int(200 * 0.95), seed=1)      # fewer than k
    rec = Fountain.decode(drops, 200)
    assert any(r is None for r in rec)               # cannot be fully solved


def test_erasure_and_order_blind():
    # Collect ANY sufficient subset, in ANY order, whichever survived -- recover
    # exactly. Shuffle and drop droplets, then decode.
    data = b"the same water no matter which drops you catch " * 200
    f = Fountain.from_bytes(data, block_size=32)        # k ~ 290, large enough for the asymptotics
    # provision for the loss: send enough that ~30% loss still leaves ~1.5k
    drops = f.droplets(int(f.k * 2.2), seed=2)
    random.seed(0)
    random.shuffle(drops)
    survivors = [d for d in drops if random.random() > 0.3]    # lose ~30%
    out = f.decode_bytes(survivors, f.orig_len)
    assert out == data


def test_recovery_curve_has_the_cliff():
    # Reliability rises with overhead: near-zero below ~1.1k, reliable by ~1.35k.
    curve = recovery_curve(300, overheads=(1.0, 1.1, 1.35, 1.5), trials=6, seed=0)
    assert curve[1.0] == 0.0
    assert curve[1.35] >= 0.8
    assert curve[1.5] >= curve[1.1]


def test_byte_roundtrip_when_complete():
    data = b"exact bit-for-bit recovery of an arbitrary blob \x00\x01\x02\xff" * 20
    f = Fountain.from_bytes(data, block_size=48)
    drops = f.droplets(int(f.k * 1.6), seed=5)
    assert f.decode_bytes(drops, f.orig_len) == data

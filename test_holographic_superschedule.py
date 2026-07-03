"""Fill 3: auto-superposition + spill -- gated recall under the dial, spill beats cram over it."""
import numpy as np
from holographic_superschedule import (pack_capacity, superpose_batch, recover_batch, apply_in_superposition)
from holographic_superposed import pack, recover_all, resolve
from holographic_ai import bind, cosine


def _units(rng, k, d):
    v = rng.standard_normal((k, d)); return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_gated_recall_under_dial():
    rng = np.random.default_rng(0); D = 512; cap = pack_capacity(D, True)
    K = cap // 3
    keys = _units(rng, K, D); items = _units(rng, K, D)
    packed, buckets = superpose_batch(keys, items, gated=True)
    assert len(packed) == 1
    rec = recover_batch(packed, buckets, keys)
    acc = np.mean([resolve(rec[i], items)[0] == i for i in range(K)])
    assert acc > 0.85


def test_spill_beats_cram():
    rng = np.random.default_rng(1); D = 512; cap = pack_capacity(D, True)
    N = cap * 2
    keys = _units(rng, N, D); items = _units(rng, N, D)
    packed, buckets = superpose_batch(keys, items, gated=True)
    assert len(buckets) == 2
    rec_spill = recover_batch(packed, buckets, keys)
    acc_spill = np.mean([resolve(rec_spill[i], items)[0] == i for i in range(N)])
    rec_cram = recover_all(pack(keys, items), keys)
    acc_cram = np.mean([resolve(rec_cram[i], items)[0] == i for i in range(N)])
    assert acc_spill > acc_cram + 0.2


def test_apply_in_superposition():
    rng = np.random.default_rng(2); D = 512
    K = max(3, pack_capacity(D, gated=False) // 2)
    keys = _units(rng, K, D); items = _units(rng, K, D); op = _units(rng, 1, D)[0]
    out = apply_in_superposition(keys, items, op, gated=True)
    truth = np.stack([bind(items[i], op) for i in range(K)])
    assert np.mean([cosine(out[i], truth[i]) for i in range(K)]) > 0.3


def test_continuous_dial_smaller_and_deterministic():
    D = 512
    assert pack_capacity(D, gated=False) < pack_capacity(D, gated=True)
    rng = np.random.default_rng(3)
    keys = _units(rng, 10, D); items = _units(rng, 10, D)
    a = recover_batch(*superpose_batch(keys, items), keys)
    b = recover_batch(*superpose_batch(keys, items), keys)
    assert np.array_equal(a, b)

"""Tests for the chunked delta chain with hash-chain + Merkle integrity (DELTA-1)."""

import numpy as np
from holographic.agents_and_reasoning.holographic_deltachain import DeltaChain, IntegrityError, merkle_root


def _drift(n=10, N=120, D=16, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((N, D))
    chain = DeltaChain(base)
    cur = base.copy(); originals = [base.copy()]
    for _ in range(n):
        cur = cur.copy(); cur[rng.choice(N, 4, replace=False)] = rng.standard_normal((4, D))
        chain.append(cur); originals.append(cur.copy())
    return chain, originals


def test_reconstruction_is_bit_exact():
    chain, originals = _drift()
    assert all(np.array_equal(chain.get(i), originals[i + 1]) for i in range(10))


def test_drifting_sequence_prefers_prior_deltas():
    chain, _ = _drift()
    assert sum(d["ref"] == "prior" for d in chain._deltas) >= 8     # incremental edits stay small vs the prior


def test_near_base_sequence_prefers_base_deltas():
    rng = np.random.default_rng(1); N, D = 120, 16
    base = rng.standard_normal((N, D)); chain = DeltaChain(base)
    for _ in range(10):
        v = base.copy(); v[rng.choice(N, 2, replace=False)] = rng.standard_normal((2, D)); chain.append(v)
    assert sum(d["ref"] == "base" for d in chain._deltas) >= 8      # each chunk is closest to the base


def test_corruption_is_detected():
    chain, _ = _drift()
    assert chain.verify()
    chain._deltas[2]["lit"][0, 0] += 1.0                           # tamper with a stored delta
    raised = False
    try:
        chain.get(2)
    except IntegrityError:
        raised = True
    assert raised


def test_merkle_root_is_deterministic_and_sensitive():
    a, _ = _drift(seed=0)
    b, _ = _drift(seed=0)
    c, _ = _drift(seed=1)
    assert a.root() == b.root()                                    # same sequence -> same proof
    assert a.root() != c.root()                                    # any change -> different proof


def test_codebook_compression_is_lossless():
    rng = np.random.default_rng(2); N, D = 120, 64
    cb = rng.standard_normal((24, D)); base = rng.standard_normal((N, D))
    cc = DeltaChain(base, codebook=cb); lit = DeltaChain(base)
    cur = base.copy()
    for _ in range(8):
        cur = cur.copy(); cur[rng.choice(N, 5, replace=False)] = cb[rng.choice(24, 5)]
        cc.append(cur); lit.append(cur)
    assert all(np.array_equal(cc.get(i), lit.get(i)) for i in range(8))   # same reconstruction
    assert cc.memory_bytes() < lit.memory_bytes()                 # but smaller (atom-rows -> indices)


def test_memory_saving_over_full():
    chain, _ = _drift()
    assert chain.memory_bytes() < chain.full_bytes()              # deltas beat storing every chunk full


def test_merkle_root_helper():
    leaves = [bytes([i]) * 32 for i in range(5)]
    assert merkle_root(leaves) == merkle_root(leaves)             # deterministic
    assert len(merkle_root(leaves)) == 32

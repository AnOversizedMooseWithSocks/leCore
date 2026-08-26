"""Tests for the reversible / error-correction model (ISA-8): the reversibility audit (verified empirically),
the oracle-free health signal, and the auto-cleanup scheduler's measured win over a fixed cadence."""

import numpy as np
import pytest

from holographic.misc.holographic_reversible import reversibility_class, reversibility_audit, health, snap, auto_cleanup_run, _bursty_program
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, permute, involution, cosine, random_vector, derived_atom

DIM, SEED = 1024, 0


def test_audit_classifies_each_instruction():
    rev = {op for op, (c, _) in reversibility_audit().items() if c == "reversible"}
    lossy = {op for op, (c, _) in reversibility_audit().items() if c == "lossy"}
    assert {"bind", "unbind", "permute", "involution"} <= rev
    assert {"bundle", "superpose", "cleanup"} <= lossy
    with pytest.raises(ValueError):
        reversibility_class("nonsense")


def test_reversible_instructions_actually_round_trip():
    rng = np.random.default_rng(SEED)
    a = random_vector(DIM, rng)
    ku = derived_atom(SEED, "key", DIM, unitary=True)        # a unitary key -> bind/unbind is exact
    assert cosine(unbind(bind(a, ku), ku), a) > 0.999
    assert np.allclose(permute(permute(a, 7), -7), a)        # permute inverse is exact
    assert np.allclose(involution(involution(a)), a)         # involution is self-inverse


def test_lossy_instructions_destroy_information():
    rng = np.random.default_rng(SEED)
    a, b = random_vector(DIM, rng), random_vector(DIM, rng)
    mix = bundle([a, b])
    # the bundle is not either summand (information mixed) -- there is no exact inverse
    assert cosine(mix, a) < 0.95 and cosine(mix, b) < 0.95
    # cleanup is a projection: applying it twice changes nothing more (idempotent)
    cb = [a, b]
    s1, _ = snap(mix, cb)
    s2, _ = snap(s1, cb)
    assert np.array_equal(s1, s2)


def test_health_signal_tracks_drift():
    rng = np.random.default_rng(SEED)
    cb = [random_vector(DIM, rng) for _ in range(16)]
    h0, idx0 = health(cb[0], cb)
    assert h0 > 0.999 and idx0 == 0                          # a clean atom has health ~1.0
    drifted = cb[0] + 0.6 * (random_vector(DIM, rng))
    drifted /= np.linalg.norm(drifted)
    h1, _ = health(drifted, cb)
    assert h1 < h0                                           # drift lowers the health signal


def _measure(schedule, floor=0.9, k=3, seeds=40):
    cl, below = [], []
    for s in range(seeds):
        cb = [random_vector(DIM, np.random.default_rng(1000 + s)) for _ in range(16)]
        tgt = int(np.random.default_rng(2000 + s).integers(16))
        steps = _bursty_program(cb, tgt, dim=DIM, seed=s)
        v, c = auto_cleanup_run(cb[tgt], steps, cb, floor=floor, schedule=schedule, k=k)
        cl.append(c)
        below.append(cosine(v, cb[tgt]) < 0.9)
    return np.mean(cl), np.mean(below)


def test_adaptive_scheduler_beats_fixed_at_matched_fidelity():
    # THE BAR: the adaptive scheduler holds the program above the fidelity threshold at FEWER cleanups than the
    # fixed cadence that matches it -- echoing the coherence-gate's ~1/3 passes.
    ad_cl, ad_below = _measure("adaptive", floor=0.9)
    fx_cl, fx_below = _measure("fixed", k=3)
    assert ad_below < 0.1 and fx_below < 0.1                 # both hold final fidelity
    assert ad_cl < 0.6 * fx_cl                               # adaptive uses far fewer cleanups (measured ~5 vs ~16)


def test_no_cleanup_degrades():
    # the negative control: without cleanup the bursty program drifts well below threshold
    _, below = _measure("none")
    assert below > 0.5

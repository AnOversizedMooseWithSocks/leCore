"""Tests for holographic_computehome -- the Compute home (H7: fuse/schedule/machine, stay VSA-native)."""
import numpy as np
from holographic.misc.holographic_computehome import Compute, compute_levers


def _record(n=16, d=512, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(d) for _ in range(n)], [rng.standard_normal(d) for _ in range(n)]


def test_fuse_record_uses_fewer_ffts_measured():
    keys, values = _record(16)
    Compute.reset_fft_counts()
    Compute.fuse_record(keys, values)
    c = Compute.fft_counts()
    total = c["rfft"] + c["irfft"]
    assert total == 2 * 16 + 2                                    # the module's stated 2*len(keys)+2
    assert total < 3 * 16                                         # measurably fewer than the op-by-op ~3*len


def test_fuse_record_agrees_with_op_by_op():
    from holographic.agents_and_reasoning.holographic_ai import bind, bundle
    keys, values = _record(12)
    fused = Compute.fuse_record(keys, values)
    ref = bundle(np.stack([bind(keys[i], values[i]) for i in range(12)]))
    assert np.allclose(fused, ref, atol=1e-9)


def test_fuse_expression_agrees_with_eager():
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind
    rng = np.random.default_rng(1); d = 256
    a = rng.standard_normal(d); b = rng.standard_normal(d)
    expr = Compute.unbind(Compute.bind(Compute.leaf(a), Compute.leaf(b)), Compute.leaf(a))
    assert np.allclose(Compute.fuse(expr), unbind(bind(a, b), a), atol=1e-9)


def test_fuse_record_matches_mind_faculty():
    from holographic.misc.holographic_unified import UnifiedMind
    keys, values = _record(8)
    m = UnifiedMind(dim=64, seed=0)
    assert np.array_equal(m.fuse_record(keys, values), Compute.fuse_record(keys, values))


def test_machine_reachable():
    mach = Compute.machine(dim=256, seed=1)
    assert hasattr(mach, "run")


def test_levers_listed():
    assert set(compute_levers()) == {"fuse", "schedule", "width", "program"}

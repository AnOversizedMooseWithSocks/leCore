"""Tests for holographic_loadmemory -- the load-gated role->filler memory (FHRR-at-high-load re-enable)."""
from holographic_loadmemory import AdaptiveRoleFillerMemory, choose_backend


def test_gate_picks_backend_by_load():
    assert choose_backend(5, 512) == "hrr"        # low load -> cheap real-HRR
    assert choose_backend(80, 512) == "fhrr"      # high load -> FHRR
    # dim-relative: the knee scales with dim
    assert choose_backend(40, 2048) == "hrr"      # 40 is low load for a big space
    assert choose_backend(40, 256) == "fhrr"      # 40 is high load for a small space


def test_low_load_uses_hrr_and_recalls_perfectly():
    m = AdaptiveRoleFillerMemory(dim=512, expected_pairs=6, seed=0)
    assert m.backend == "hrr"
    pairs = [("color", "red"), ("shape", "round"), ("size", "big"), ("taste", "sweet")]
    for r, f in pairs:
        m.add(r, f)
    assert all(m.recall(r) == f for r, f in pairs)


def test_high_load_uses_fhrr_and_beats_hrr():
    N = 90
    fhrr = AdaptiveRoleFillerMemory(dim=512, expected_pairs=N, seed=1)
    hrr = AdaptiveRoleFillerMemory(dim=512, expected_pairs=1, seed=1)   # forced hrr, same seed/atoms
    assert fhrr.backend == "fhrr" and hrr.backend == "hrr"
    for i in range(N):
        fhrr.add(f"r{i}", f"f{i}"); hrr.add(f"r{i}", f"f{i}")
    fhrr_hits = sum(fhrr.recall(f"r{i}") == f"f{i}" for i in range(N))
    hrr_hits = sum(hrr.recall(f"r{i}") == f"f{i}" for i in range(N))
    assert fhrr_hits > hrr_hits                    # the win the gate captures at high load


def test_uniform_interface_regardless_of_backend():
    for pairs, expect in [(5, "hrr"), (80, "fhrr")]:
        m = AdaptiveRoleFillerMemory(dim=512, expected_pairs=pairs, seed=2)
        assert m.backend == expect
        m.add("k", "v")
        assert m.recall("k") == "v"                 # same add/recall both ways


def test_mind_faculty():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    rec = m.adaptive_record(expected_pairs=80)
    assert rec.backend == "fhrr"
    rec.add("color", "red")
    assert rec.recall("color") == "red"


def test_exact_need_uses_tensor_and_is_exact():
    NX = 200
    m = AdaptiveRoleFillerMemory(dim=256, expected_pairs=NX, exact=True, seed=2)
    assert m.backend == "tensor"
    for i in range(NX):
        m.add(f"r{i}", f"f{i}")
    assert sum(m.recall(f"r{i}") == f"f{i}" for i in range(NX)) == NX     # EXACT recall to M~D


def test_tensor_gate_respects_memory_budget():
    # exact requested but D*D storage over budget -> must NOT pick tensor
    m = AdaptiveRoleFillerMemory(dim=256, expected_pairs=200, exact=True, max_numbers=1000, seed=2)
    assert m.backend != "tensor"
    assert choose_backend(200, 256, exact=True, max_numbers=1000) == "fhrr"
    assert choose_backend(200, 256, exact=True, max_numbers=100000) == "tensor"


def test_exact_beats_fhrr_at_high_load():
    N = 200
    tensor = AdaptiveRoleFillerMemory(dim=256, expected_pairs=N, exact=True, seed=3)
    fhrr = AdaptiveRoleFillerMemory(dim=256, expected_pairs=N, seed=3)     # no exact -> fhrr at this load
    assert tensor.backend == "tensor" and fhrr.backend == "fhrr"
    for i in range(N):
        tensor.add(f"r{i}", f"f{i}"); fhrr.add(f"r{i}", f"f{i}")
    t_hits = sum(tensor.recall(f"r{i}") == f"f{i}" for i in range(N))
    f_hits = sum(fhrr.recall(f"r{i}") == f"f{i}" for i in range(N))
    assert t_hits == N and t_hits > f_hits                                # tensor exact, beats fhrr at high load

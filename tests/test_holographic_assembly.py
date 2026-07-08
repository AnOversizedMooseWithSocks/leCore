"""Tests for B6 part 2: fragment assembly as min-energy flow search, matching the DP optimum, with the
assembly returned as a B7 typed structure."""

from holographic.simulation_and_physics.holographic_assembly import assemble, assemble_optimal_energy
from holographic.misc.holographic_typed import op_kinds

TARGET = "ABCABCABCA"
FULL = sorted({TARGET[p:p + 2] for p in range(len(TARGET) - 1)})


def test_assembles_target_exactly_with_complete_library():
    out = assemble(TARGET, FULL, frag_len=2)
    assert out["assembled"] == TARGET and out["energy"] == 0


def test_matches_dp_optimum_when_forced_to_mismatch():
    lib = sorted((set(FULL) - {"CA"}) | {"AA", "BB", "CC"})    # missing a true fragment -> mismatches
    out = assemble(TARGET, lib, frag_len=2)
    opt = assemble_optimal_energy(TARGET, lib, frag_len=2)
    assert out["energy"] == opt and opt > 0                     # global optimum, not greedy


def test_assembly_is_a_typed_structure():
    out = assemble(TARGET, FULL, frag_len=2)
    r = out["recipe"]
    assert op_kinds(r) <= {"atom", "bind", "bundle", "superpose", "permute", "raw", "normalize"}
    assert len(out["fragments"]) == len(TARGET) - 1            # one fragment per trellis position
    assert r.get(r._outputs[0]).shape == (1024,)              # realizes to a hypervector


def test_optimal_reference_is_a_lower_bound():
    lib = sorted((set(FULL) - {"BC"}) | {"BD", "DC"})
    out = assemble(TARGET, lib, frag_len=2)
    opt = assemble_optimal_energy(TARGET, lib, frag_len=2)
    assert out["energy"] >= opt                                # flow never beats the exact optimum
    assert out["energy"] == opt                                # and here it attains it

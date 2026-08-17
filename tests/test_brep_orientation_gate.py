"""B1's completed gate: Euler-Poincare + orientation together catch all five mutation classes.

The measurement that motivated this: brep_validate alone is 4/5 -- it catches deleted faces,
duplicated vertices, dropped vertices and dangling faces 40/40, and is BLIND to a flipped
face (0/40), because reversing a loop preserves V, E, F, R and S exactly. That is a property
of the invariant, not a bug. The complementary-winding check is the standard companion (each
DIRECTED edge traversed exactly once), and leCore already shipped it as mesh_is_oriented --
so the gap closes by REUSE, with no new code.
"""
import numpy as np
from holographic.mesh_and_geometry.holographic_brep import BFace


def _box():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    b = m.brep_box()
    return m, [tuple(map(float, v)) for v in b.vertices], [list(f.outer) for f in b.faces]


def _valid(m, vs, fs):
    """The COMBINED gate: Euler-Poincare validity AND consistent orientation."""
    try:
        r = m.brep_validate(m.brep_from_faces(vs, [BFace(f) for f in fs]))
        if not (r["closed_manifold"] and r["euler_ok"]):
            return False
    except Exception:
        return False
    return bool(m.mesh_is_oriented(fs))


def test_clean_box_passes_both_checks():
    m, V, F = _box()
    assert _valid(m, V, F)


def test_combined_gate_catches_all_five_mutation_classes():
    m, V, F = _box()
    rng = np.random.default_rng(0)
    kinds = ["delete_face", "flip_face", "dup_vertex_in_face",
             "drop_vertex_from_face", "add_dangling_face"]
    caught = {k: 0 for k in kinds}
    per = 8
    for k in kinds:
        for _ in range(per):
            fs = [list(f) for f in F]
            vs = list(V)
            i = int(rng.integers(len(fs)))
            if k == "delete_face":
                fs.pop(i)
            elif k == "flip_face":
                fs[i] = list(reversed(fs[i]))
            elif k == "dup_vertex_in_face":
                fs[i] = fs[i] + [fs[i][0]]
            elif k == "drop_vertex_from_face":
                fs[i] = fs[i][:-1]
            else:
                vs = vs + [(9.0, 9.0, 9.0)]
                fs = fs + [[0, 1, len(vs) - 1]]
            caught[k] += int(not _valid(m, vs, fs))
    assert all(caught[k] == per for k in kinds), caught


def test_euler_alone_is_blind_to_flips_and_that_is_a_property():
    """Pinned as a FINDING, not a bug: no Euler-style invariant can see a reversed loop,
    because V, E, F, R and S are all unchanged. If this ever starts failing, the validator
    grew an orientation check and this note should move, not be deleted."""
    m, V, F = _box()
    fs = [list(f) for f in F]
    fs[0] = list(reversed(fs[0]))
    r = m.brep_validate(m.brep_from_faces(V, [BFace(f) for f in fs]))
    assert r["closed_manifold"] and r["euler_ok"]      # Euler sees nothing wrong
    assert not m.mesh_is_oriented(fs)                  # orientation does

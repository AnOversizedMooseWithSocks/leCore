"""C3 -- canonical element + delta chain, and the family that decides everything.

Instancing generalised: not "these two objects are the same mesh" but "these two objects are the same ANYTHING,
modulo a recognised delta." Store the canonical element once, one delta per instance.

    family        classes (200 tris, 5 bases)   ratio vs raw
    rigid                 200                      0.56x   <- UNDER-fits: scale is not in the family
    similarity              5                      1.09x   <- exactly the generating family
    affine                  5                      0.98x   <- collapses SHAPE; delta costs what the triangle costs

TWO FACTS WORTH HAVING:

  * `affine` gives 5 and not 1 because whitening a triangle's hull makes it exactly EQUILATERAL (all three sides
    sqrt(6)). The shape really is collapsed; what remains is the in-hull ROTATION, which an unordered point set
    cannot pin. *"Every non-degenerate triangle is affinely the same" is a statement about ORDERED triangles.*
  * A TRIANGLE CAN NEVER PAY. Its hull is rank 2, so an affine delta is 3*2 + 3 = 9 floats -- exactly the triangle.
    The dividend scales with the ELEMENT against an O(1) delta: 0.75x at 3 vertices, 143x at 2000. Per-triangle
    canonicalisation is a RECOGNISER, and its dividend is C1's compute cache.
"""

import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_canonmesh import (
    FAMILIES, apply_delta, canonical_form, class_key, rebuild, recognize, storage_report)


def _rot(rng):
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    th = rng.uniform(0.2, 2.0)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _scene(n_bases=5, per_base=40, seed=0):
    """5 base triangles, each instanced under a random rotation + translation + UNIFORM SCALE. The generating
    family is `similarity`, and that is the point of the fixture: only one of the three families matches it."""
    rng = np.random.default_rng(seed)
    bases = [rng.normal(size=(3, 3)) for _ in range(n_bases)]
    els = []
    for base in bases:
        for _ in range(per_base):
            els.append((rng.uniform(0.5, 2.0) * base) @ _rot(rng).T + rng.normal(size=3))
    return bases, els


def test_selftest_runs():
    from holographic.mesh_and_geometry import holographic_canonmesh as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# reconstruction is exact
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family", list(FAMILIES))
def test_reconstruction_is_exact(family):
    rng = np.random.default_rng(1)
    for n in (3, 4, 12, 60):
        V = rng.normal(size=(n, 3))
        canon, delta = canonical_form(V, family=family)
        assert np.abs(apply_delta(canon, delta) - V).max() < 1e-12, (family, n)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_whole_scene_rebuilds_exactly(family):
    _bases, els = _scene(n_bases=3, per_base=5)
    rec = recognize(els, family=family)
    for got, want in zip(rebuild(rec), els):
        assert np.abs(got - want).max() < 1e-10


def test_canonicalisation_is_deterministic_and_so_is_the_class_key():
    rng = np.random.default_rng(2)
    V = rng.normal(size=(9, 3))
    a, _ = canonical_form(V, "similarity")
    b, _ = canonical_form(V, "similarity")
    assert np.array_equal(a, b)
    assert class_key(a) == class_key(b)


def test_the_null_axis_sign_does_not_leak_into_a_real_axis():
    # A planar element's third singular vector has a zero singular value; its sign is noise. Forcing det = +1 on a
    # frame that includes it pushed that noise onto a REAL axis and split each base shape into four spurious
    # classes (5 bases came out as 20). Two instances of the SAME base must canonicalise identically.
    rng = np.random.default_rng(3)
    base = rng.normal(size=(3, 3))
    a, _ = canonical_form(base @ _rot(rng).T + rng.normal(size=3), "similarity")
    b, _ = canonical_form(base @ _rot(rng).T + rng.normal(size=3), "similarity")
    assert np.abs(a - b).max() < 1e-9
    assert class_key(a) == class_key(b)


# ---------------------------------------------------------------------------------------------------------
# THE FAMILY DECIDES THE CLASS COUNT
# ---------------------------------------------------------------------------------------------------------

def test_rigid_under_fits_when_the_scene_contains_scales():
    bases, els = _scene()
    rep = storage_report(els, "rigid")
    assert rep["classes"] == len(els)                # nothing matches anything
    assert rep["ratio"] < 1.0                        # ... and it costs more than raw


def test_similarity_matches_the_generating_family_exactly():
    bases, els = _scene()
    rep = storage_report(els, "similarity")
    assert rep["classes"] == len(bases)
    assert 1.0 < rep["ratio"] < 1.3                  # the RIGHT family, and it wins by almost nothing


def test_affine_whitening_makes_every_triangle_equilateral():
    # THE FACT. So the shape really is collapsed, and what remains is the in-hull rotation.
    rng = np.random.default_rng(4)
    for _ in range(6):
        canon, _d = canonical_form(rng.normal(size=(3, 3)), "affine")
        sides = sorted(float(np.linalg.norm(canon[i] - canon[j])) for i, j in ((0, 1), (1, 2), (2, 0)))
        assert max(abs(s - np.sqrt(6.0)) for s in sides) < 1e-9


def test_affine_leaves_the_in_hull_rotation_so_it_gives_five_classes_not_one():
    bases, els = _scene()
    assert storage_report(els, "affine")["classes"] == len(bases)
    # ... which is why "every non-degenerate triangle is affinely the same" is about ORDERED triangles


def test_every_triangle_is_planar_so_its_affine_hull_has_rank_two():
    rng = np.random.default_rng(5)
    _c, d = canonical_form(rng.normal(size=(3, 3)), "affine")
    assert d["rank"] == 2

    collinear = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    _c2, d2 = canonical_form(collinear, "affine")
    assert d2["rank"] == 1                            # still recognisable, in its own 1-D hull
    assert np.abs(apply_delta(_c2, d2) - collinear).max() < 1e-12


# ---------------------------------------------------------------------------------------------------------
# KEPT NEGATIVES: where it pays and where it cannot
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_a_triangle_can_never_pay_an_affine_delta():
    # 9 floats of triangle, 3*rank+3 = 9 floats of delta. Break-even before storing one canonical element.
    _bases, els = _scene()
    assert storage_report(els, "affine")["ratio"] < 1.0


def test_the_dividend_scales_with_the_element_against_an_o1_delta():
    # NOTE THE FAMILY. A rigid delta is 7 floats, so a 9-float triangle pays 1.23x -- barely, and only because 7 < 9.
    # An AFFINE delta on a triangle is 3*rank + 3 = 9 floats, exactly the triangle, and cannot pay. Asserting
    # "a triangle always loses" would have been true of the affine family and false of the rigid one.
    rng = np.random.default_rng(6)
    ratios = []
    for n_verts in (3, 12, 100):
        base = rng.normal(size=(n_verts, 3))
        els = [base @ _rot(rng).T + rng.normal(size=3) for _ in range(30)]
        rep = storage_report(els, "rigid")
        assert rep["classes"] == 1                    # rigid instances of ONE element
        ratios.append(rep["ratio"])
    assert ratios[0] < 1.5                            # a triangle barely pays, even under the cheapest delta
    assert ratios == sorted(ratios)                   # ... and it improves monotonically with element size
    assert ratios[-1] > 10.0
    assert ratios[-1] > 10.0 * ratios[0]              # the dividend is in the ELEMENT, not the family


def test_it_beats_zlib_on_large_elements_unlike_the_same_idea_on_code():
    # float64 coordinates are high-entropy: zlib manages ~1.04x. The same canonical+delta idea applied to SOURCE
    # CODE came out 1.12x LARGER than zlib. A mesh's delta is O(1); a statement's delta is O(statement).
    rng = np.random.default_rng(7)
    base = rng.normal(size=(200, 3))
    els = [base @ _rot(rng).T + rng.normal(size=3) for _ in range(30)]
    rep = storage_report(els, "rigid")
    assert rep["zlib_ratio"] < 1.5                    # zlib barely touches it
    assert rep["ratio"] > 10.0
    assert rep["beats_zlib"] is True


def test_the_report_carries_its_own_baseline():
    _bases, els = _scene(n_bases=2, per_base=4)
    rep = storage_report(els, "similarity")
    for k in ("classes", "raw_floats", "canonical_floats", "delta_floats", "total_floats",
              "ratio", "zlib_ratio", "beats_zlib"):
        assert k in rep
    assert rep["total_floats"] == rep["canonical_floats"] + rep["delta_floats"]


# ---------------------------------------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------------------------------------

def test_degenerate_and_malformed_inputs_raise():
    with pytest.raises(ValueError):
        canonical_form(np.zeros((4, 3)), "affine")            # zero extent: no frame exists
    with pytest.raises(ValueError):
        canonical_form(np.zeros((4, 3)), "similarity")        # no scale to recover
    with pytest.raises(ValueError):
        canonical_form(np.zeros((4, 2)), "rigid")             # not (n, 3)
    with pytest.raises(ValueError):
        canonical_form(np.random.default_rng(0).normal(size=(4, 3)), "projective")


def test_class_key_is_a_content_hash_and_normalises_negative_zero():
    a = np.array([[0.0, 1.0, 2.0]])
    b = np.array([[-0.0, 1.0, 2.0]])
    assert class_key(a) == class_key(b)
    assert class_key(a) != class_key(a + 1.0)


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    rng = np.random.default_rng(8)
    base = rng.normal(size=(50, 3))
    els = [base @ _rot(rng).T + rng.normal(size=3) for _ in range(30)]

    canon, delta = m.canonical_form(els[0], "rigid")
    assert np.abs(apply_delta(canon, delta) - els[0]).max() < 1e-12

    rec = m.recognize_elements(els, "rigid")
    assert len(rec["canonicals"]) == 1 and len(rec["instances"]) == 30

    rep = m.canon_storage_report(els, "rigid")
    assert rep["ratio"] > 5.0 and rep["beats_zlib"] is True

    assert "Canonical element" in str(m.find_capability("instancing generalized")[:3])

"""C2 -- the equivariance table, measured, plus Part C's claims A/B/C checked directly.

Part C's whole cache policy is one table: for each (operator, transform) pair, which of INVARIANT / EQUIVARIANT /
ADJOINT / RECOMPUTE applies. This suite pins the table, the two cells that were WRONG until the laws were derived,
and the adjoint statement that is true only for rotations.

**RECOMPUTE must mean NO LAW EXISTS, not "I did not write one down."** A table that says recompute where a law
exists is a cache that never fires -- and it looks exactly like a table that is merely honest.
"""

import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_equivariance import (
    OPERATORS, TRANSFORMS, apply_affine, area, cache_policy, centroid, classify, equivariance_table, inertia,
    max_x, normal, sample_transform, shade, shade_adjoint)


def test_selftest_runs():
    from holographic.mesh_and_geometry import holographic_equivariance as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------------------------------------

def test_the_table_is_measured_and_stable_across_seeds():
    a = equivariance_table(seed=0)
    b = equivariance_table(seed=7, trials=20)
    assert a == b                                        # the verdicts are facts, not draws


def test_the_strongest_claims_hold():
    t = equivariance_table()
    assert t["area"]["translate"] == "invariant"
    assert t["area"]["rotate"] == "invariant"
    assert t["area"]["reflect"] == "invariant"           # a reflection preserves area, flips winding
    assert t["normal"]["translate"] == "invariant"
    assert t["normal"]["uniform_scale"] == "invariant"   # a uniform scale does not tilt a plane
    assert t["inertia"]["translate"] == "invariant"


def test_a_translation_moves_the_centroid_so_it_is_not_invariant():
    # The easy mistake: "translation is harmless." It is not, for a quantity that lives at a position.
    t = equivariance_table()
    assert t["centroid"]["translate"] == "equivariant"
    tri = np.random.default_rng(0).normal(size=(3, 3))
    assert not np.allclose(centroid(tri), centroid(apply_affine(np.eye(3), np.ones(3), tri)))


def test_no_geometric_cell_is_recompute():
    # THE FINDING. Two of these read `recompute` until the laws were derived properly.
    t = equivariance_table()
    for op in ("area", "centroid", "normal", "inertia"):
        for kind in TRANSFORMS:
            assert t[op][kind] != "recompute", (op, kind)


@pytest.mark.parametrize("kind", list(TRANSFORMS))
def test_the_derived_area_law_is_exact_for_every_affine_family(kind):
    # area(A x) = |det A| * ||A^-T n|| * area(x). My first law used |det A|^(2/3), which is wrong off the
    # uniform-scale diagonal, and the cell read `recompute` as a result.
    rng = np.random.default_rng(1)
    for _ in range(10):
        tri = rng.normal(size=(3, 3))
        A, b = sample_transform(kind, rng)
        pred = abs(np.linalg.det(A)) * np.linalg.norm(np.linalg.inv(A).T @ normal(tri)) * area(tri)
        assert abs(pred - area(apply_affine(A, b, tri))) < 1e-12


@pytest.mark.parametrize("kind", list(TRANSFORMS))
def test_the_derived_normal_law_is_exact_and_a_reflection_flips_the_winding(kind):
    rng = np.random.default_rng(2)
    for _ in range(10):
        tri = rng.normal(size=(3, 3))
        A, b = sample_transform(kind, rng)
        v = np.linalg.inv(A).T @ normal(tri)
        pred = np.sign(np.linalg.det(A)) * v / np.linalg.norm(v)
        assert np.abs(pred - normal(apply_affine(A, b, tri))).max() < 1e-12

    # the sign is not decoration: a reflection reverses orientation
    A, b = sample_transform("reflect", np.random.default_rng(0))
    assert np.linalg.det(A) < 0


def test_inertia_transforms_as_a_tensor():
    rng = np.random.default_rng(3)
    for kind in TRANSFORMS:
        tri = rng.normal(size=(3, 3))
        A, b = sample_transform(kind, rng)
        assert np.abs(A @ inertia(tri) @ A.T - inertia(apply_affine(A, b, tri))).max() < 1e-12


# ---------------------------------------------------------------------------------------------------------
# a GENUINE recompute, so the negative branch is exercised by something real
# ---------------------------------------------------------------------------------------------------------

def test_max_x_is_the_real_recompute_case():
    t = equivariance_table()
    assert t["max_x"]["rotate"] == "recompute"
    assert t["max_x"]["shear"] == "recompute"
    # ... and it is NOT recompute where a law does exist
    assert t["max_x"]["translate"] == "equivariant"
    assert t["max_x"]["uniform_scale"] == "equivariant"
    assert t["max_x"]["reflect"] == "invariant"          # a z-reflection never touches x


def test_why_max_x_has_no_law_under_rotation():
    # WHICH vertex attained the maximum is information the scalar threw away. Two triangles with the SAME max_x
    # rotate to different max_x -- so no function of max_x alone can predict the rotated value.
    #
    # (A first attempt used two triangles differing by a sign flip, and a random rotation kept the SAME vertex
    # winning in both -- the counterexample has to make the winner CHANGE, which is the whole point.)
    th = np.pi / 6.0
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    t1 = np.array([[1.0, 0.0, 0.0], [0.9, 0.5, 0.0], [0.0, 0.0, 1.0]])
    t2 = np.array([[1.0, 0.0, 0.0], [0.9, -0.5, 0.0], [0.0, 0.0, 1.0]])
    assert max_x(t1) == max_x(t2) == 1.0

    r1, r2 = max_x(apply_affine(R, 0, t1)), max_x(apply_affine(R, 0, t2))
    assert abs(r1 - r2) > 0.1, (r1, r2)              # same input value, different output: no law can exist
    assert np.argmax(apply_affine(R, 0, t1)[:, 0]) != np.argmax(apply_affine(R, 0, t2)[:, 0])


# ---------------------------------------------------------------------------------------------------------
# THE ADJOINT: Part C's claim C, and where it is wrong
# ---------------------------------------------------------------------------------------------------------

def test_unrotate_the_light_is_exact_for_a_rotation():
    rng = np.random.default_rng(6)
    for _ in range(12):
        tri = rng.normal(size=(3, 3))
        L = rng.normal(size=3)
        L /= np.linalg.norm(L)
        R, _b = sample_transform("rotate", rng)
        assert abs(shade(apply_affine(R, 0, tri), L) - shade(tri, np.linalg.inv(R) @ L)) < 1e-12


def test_kept_negative_unrotate_the_light_fails_for_a_uniform_scale():
    # The naive adjoint drops a factor. Even a UNIFORM scale breaks it, by 0.38 -- because the normal is
    # renormalised and the scale does not cancel. This is the cell most likely to be assumed rather than measured.
    rng = np.random.default_rng(7)
    S = 1.7 * np.eye(3)
    worst = 0.0
    for _ in range(12):
        tri = rng.normal(size=(3, 3))
        L = rng.normal(size=3)
        L /= np.linalg.norm(L)
        worst = max(worst, abs(shade(apply_affine(S, 0, tri), L) - shade(tri, np.linalg.inv(S) @ L)))
    assert worst > 0.1


@pytest.mark.parametrize("kind", ["rotate", "uniform_scale", "nonuniform_scale", "shear"])
def test_the_corrected_adjoint_holds_for_every_affine_and_reads_the_normal(kind):
    rng = np.random.default_rng(8)
    for _ in range(10):
        tri = rng.normal(size=(3, 3))
        L = rng.normal(size=3)
        L /= np.linalg.norm(L)
        A, _b = sample_transform(kind, rng)
        assert abs(shade(apply_affine(A, 0, tri), L) - shade_adjoint(tri, L, A)) < 1e-12


# ---------------------------------------------------------------------------------------------------------
# the cache policy, and Part C's claim B
# ---------------------------------------------------------------------------------------------------------

def test_the_cache_policy_follows_from_the_verdict():
    assert cache_policy("area", "rotate")["key_includes_delta"] is False
    assert cache_policy("max_x", "rotate")["key_includes_delta"] is True
    assert set(cache_policy("area", "shear")["read_set"]) == {"area", "normal"}
    assert cache_policy("centroid", "shear")["read_set"] == ["centroid"]


def test_the_read_set_is_the_point_every_non_rigid_law_reads_the_normal():
    # Part C: "filtered to the deltas the computation reads." The measurement sharpens it: the filter is over the
    # quantities the LAW reads, and for area, normal and shade under a non-rigid delta that includes the NORMAL.
    assert "normal" in OPERATORS["area"]["read_set"]
    assert "normal" in OPERATORS["normal"]["read_set"]
    assert "normal" not in OPERATORS["centroid"]["read_set"]     # the one law that reads only its own value


def test_part_c_claim_B_the_material_delta_drops_out_of_the_key():
    # 400 triangles = 64 shape classes x 8 materials. An expensive GEOMETRIC integral keyed on the shape class only
    # computes 64 times, not 400, and the results are bit-identical.
    rng = np.random.default_rng(9)
    shapes = [rng.normal(size=(3, 3)) for _ in range(64)]
    materials = list(range(8))
    calls = {"n": 0}

    def expensive_geometric(tri):
        calls["n"] += 1
        return area(tri) * float(np.linalg.norm(inertia(tri)))   # reads geometry only, never the material

    cache = {}
    brute, cached = [], []
    for s_id, tri in [(i, shapes[i]) for i in range(64) for _m in materials]:
        brute.append(expensive_geometric(tri))
    n_brute = calls["n"]

    calls["n"] = 0
    for s_id in range(64):
        for _m in materials:
            if s_id not in cache:                                 # the MATERIAL never enters the key
                cache[s_id] = expensive_geometric(shapes[s_id])
            cached.append(cache[s_id])
    n_cached = calls["n"]

    assert n_brute == 512 and n_cached == 64
    assert n_brute / n_cached == 8.0
    assert np.array_equal(np.array(brute), np.array(cached))      # bit-identical, not merely close


# ---------------------------------------------------------------------------------------------------------
# guards + wiring
# ---------------------------------------------------------------------------------------------------------

def test_unknown_names_raise():
    with pytest.raises(ValueError):
        classify("nonsense", "rotate")
    with pytest.raises(ValueError):
        sample_transform("wobble", np.random.default_rng(0))


def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    t = m.equivariance_table()
    assert t["area"]["rotate"] == "invariant"
    assert m.classify_equivariance("max_x", "rotate") == "recompute"
    assert m.cache_policy("area", "shear")["read_set"] == ["area", "normal"]

    rng = np.random.default_rng(0)
    tri = rng.normal(size=(3, 3))
    L = np.array([0.0, 0.0, 1.0])
    A = 1.7 * np.eye(3)
    assert abs(m.shade_adjoint(tri, L, A) - shade(apply_affine(A, 0, tri), L)) < 1e-12

    assert "Equivariance table" in str(m.find_capability("does the delta drop out of the cache key")[:3])

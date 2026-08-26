"""The transform tower: patterns -> translation -> rotation/shear -> scale, as a group, MEASURED.

    scale                central -- commutes with the whole linear part
      ^
    rotation, shear      the sl(n) part -- non-commuting peers
      ^
    translation          the abelian ideal -- the content
      ^
    hypervectors         the atoms

This is the Levi decomposition of `Aff(n) = GL(n) |x| R^n`. It is not a picture; it makes predictions, and every
one is a test here.

**THE IDEAL IS NORMAL** -- `A T(t) A^-1 == T(A t)` -- and that single line is three things this engine already
found: `shade_adjoint`'s "push the delta onto the other operand", DL11's group closure, and the shape of the
equivariance table.

**ONLY THE IDEAL IS DIAGONALISABLE.** A convolution algebra is commutative, so a `TransformBank` is a
representation of the abelian ideal, not a cache of transforms -- and its refusal to hold a scale is the tower
speaking. A layer you cannot diagonalise, you RELOCATE: on a log axis a dilation is a translation.
"""

import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_grouptower import (
    TOWER, classify_transform, commutator, commutator_table, diagonalisable, hypervector_layer,
    mellin_promotes_scale, rotation2, rotation3, scale, semidirect_law, shear2, translation)


def test_selftest_runs():
    from holographic.mesh_and_geometry import holographic_grouptower as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the ideal
# ---------------------------------------------------------------------------------------------------------

def test_the_translations_are_an_abelian_ideal():
    assert commutator(translation([0.3, -0.7]), translation([1.1, 0.2])) == 0.0
    assert commutator_table()["[T,T'] ideal is abelian"] == 0.0


@pytest.mark.parametrize("A", [rotation2(0.5), shear2(0.4), scale(1.6), rotation2(-1.2) @ shear2(0.3)])
def test_the_ideal_is_normal_which_is_the_whole_mechanism(A):
    # A T(t) A^-1 == T(A t). Conjugation. This is `shade_adjoint` and DL11's closure, in one line.
    assert semidirect_law(A, [0.3, -0.7]) < 1e-12


def test_conjugation_is_exactly_the_adjoint_move():
    # `holographic_equivariance.shade_adjoint` pushes a delta onto the other operand. That IS conjugation, and the
    # tower says it works precisely because the translations are normal.
    A = rotation2(0.6)
    t = np.array([0.4, -0.25])
    lhs = A @ translation(t) @ np.linalg.inv(A)
    rhs = translation(A[:2, :2] @ t)
    assert np.abs(lhs - rhs).max() < 1e-12


# ---------------------------------------------------------------------------------------------------------
# the peers, and the centre
# ---------------------------------------------------------------------------------------------------------

def test_scale_is_central_in_the_linear_part():
    tab = commutator_table()
    assert tab["[S,R] scale central in GL"] == 0.0
    assert tab["[S,Sh] ... the whole linear part"] == 0.0
    assert tab["[S,S'] scales commute"] == 0.0


def test_scale_is_NOT_central_in_the_affine_group():
    # s(x + t) = sx + st, not sx + t. Scale acts ON the ideal; it does not commute past it. The diagram says
    # "commutes with the whole LINEAR part", and the qualifier is load-bearing.
    assert commutator_table()["[S,T] NOT central in Aff"] > 0.1
    assert commutator(scale(1.7), translation([0.3, -0.7])) > 0.1


def test_the_peers_do_not_commute_and_in_2d_that_means_rotation_vs_shear():
    tab = commutator_table()
    assert tab["[R,Sh] non-commuting peers"] > 0.1
    assert tab["[R,R'] SO(2) is abelian"] < 1e-12          # ... two 2-D rotations DO commute
    assert tab["[Rx,Ry] SO(3) is not"] > 0.1               # ... and two 3-D rotations do not
    assert commutator(rotation3(0, 0.7), rotation3(1, 0.9)) > 0.1


def test_rotation_does_not_commute_with_translation():
    assert commutator_table()["[R,T] the semidirect action"] > 0.1


# ---------------------------------------------------------------------------------------------------------
# ONLY THE IDEAL IS DIAGONALISABLE
# ---------------------------------------------------------------------------------------------------------

def test_a_translation_is_a_bind_and_a_rotation_and_scale_are_not():
    # A bank entry is ONE spectrum applied to ANY vector. Fit it on one encoded point, test on another.
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder

    enc = VectorFunctionEncoder(3, dim=512, bounds=[(-1, 1)] * 3, seed=0)
    t = np.array([0.1, -0.2, 0.15])
    R = rotation3(2, np.pi / 2)[:3, :3]

    assert diagonalisable(lambda x: x + t, enc) < 1e-12        # generalises: it is a bind
    assert diagonalisable(lambda x: R @ x, enc) > 0.05         # does not
    assert diagonalisable(lambda x: 1.5 * x, enc) > 0.05       # does not


def test_the_fpe_law_says_why_translation_is_the_group_operation():
    from holographic.agents_and_reasoning.holographic_ai import bind
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder

    enc = VectorFunctionEncoder(3, dim=512, bounds=[(-1, 1)] * 3, seed=0)
    x = np.random.default_rng(0).uniform(-0.4, 0.4, 3)
    t = np.array([0.1, -0.2, 0.15])
    assert np.abs(bind(enc.encode(x), enc.encode(t)) - enc.encode(x + t)).max() < 1e-12


def test_the_transform_bank_is_a_representation_of_the_ideal():
    # A convolution algebra is COMMUTATIVE, so a bank can only represent an abelian group. Its composition is
    # necessarily order-independent, and it has no `add_scale` -- the tower forbids one.
    from holographic.caching_and_storage.holographic_transformbank import TransformBank

    b = TransformBank(256, seed=0)
    b.add_random_unitary("a")
    b.add_random_unitary("c")
    b.add_rotation("r5", 5)
    assert np.abs(b.compose(["a", "c", "r5"]) - b.compose(["r5", "c", "a"])).max() < 1e-12
    assert not hasattr(b, "add_scale")


def test_the_banks_rotation_is_a_translation_in_index_space():
    # The name was the bug. A cyclic shift is an element of the IDEAL, not of the tower's rotation layer -- which
    # is exactly why it composes commutatively with everything else in the bank.
    from holographic.caching_and_storage.holographic_transformbank import TransformBank

    b = TransformBank(128, seed=0)
    b.add_rotation("r3", 3)
    b.add_rotation("r5", 5)
    v = np.random.default_rng(0).normal(size=128)
    assert np.abs(b.apply_chain(["r3", "r5"], v) - np.roll(v, 8)).max() < 1e-10   # shifts ADD


# ---------------------------------------------------------------------------------------------------------
# a layer you cannot diagonalise, you RELOCATE
# ---------------------------------------------------------------------------------------------------------

def test_the_mellin_lift_promotes_scale_into_the_ideal():
    mel = mellin_promotes_scale()
    assert mel["log_axis"] < 1e-9                            # a dilation is a SHIFT on a log axis: a bind
    assert mel["linear_axis"] > 0.05                         # ... and is not one on the linear axis
    assert mel["linear_axis"] > 1e6 * mel["log_axis"]


def test_the_lift_is_what_mellin_scale_already_does():
    # `registration.mellin_scale` recovers a dilation by turning it into a shift. The tower explains why that is
    # the only move available: scale cannot be diagonalised where it stands.
    from holographic.sampling_and_signal.holographic_registration import mellin_scale, resample_affine

    x = np.linspace(0, 1, 2048)
    f = (np.sin(2 * np.pi * (20 * x + 60 * x ** 2)) * np.exp(-((x - 0.5) ** 2) / 0.06)
         + 0.5 * np.sin(2 * np.pi * 180 * x) * np.exp(-((x - 0.3) ** 2) / 0.005))
    assert abs(mellin_scale(f, resample_affine(f, 1.5, 17.0)) - 1.5) < 0.08


# ---------------------------------------------------------------------------------------------------------
# the declared tower
# ---------------------------------------------------------------------------------------------------------

def test_the_declared_tower_matches_the_measurements():
    assert [lv["name"] for lv in TOWER] == ["hypervectors", "translation", "rotation, shear", "scale"]
    assert TOWER[1]["diagonalisable"] is True
    assert TOWER[2]["diagonalisable"] is False
    assert TOWER[3]["diagonalisable"] is False
    assert [lv["level"] for lv in TOWER] == [0, 1, 2, 3]


def test_wired_to_the_mind_and_discoverable():
    import lecore

    m = lecore.UnifiedMind(dim=256, seed=0)
    tab = m.commutator_table()
    assert tab["[T,T'] ideal is abelian"] == 0.0
    assert tab["[S,T] NOT central in Aff"] > 0.1
    assert m.semidirect_law(rotation2(0.5), [0.3, -0.7]) < 1e-12
    assert m.is_diagonalisable(lambda x: x + np.array([0.1, 0.0, 0.0])) < 1e-12
    assert m.mellin_promotes_scale()["log_axis"] < 1e-9
    assert "transform tower" in str(m.find_capability("transform hierarchy")[:3])


# ===========================================================================================
# NOT BURIED: one entry point, on the main class, at the top level.
# ===========================================================================================

def _R():
    return rotation3(2, 0.6)[:3, :3]


def _Sh():
    S = np.eye(3)
    S[0, 1] = 0.6
    return S


@pytest.mark.parametrize("fn,layer,name", [
    (lambda x: x + np.array([0.1, -0.2, 0.05]), 1, "translation"),
    (lambda x: _R() @ x, 2, "rotation / shear"),
    (lambda x: _Sh() @ x, 2, "rotation / shear"),
    (lambda x: 1.7 * x, 3, "scale"),
    (lambda x: _R() @ x + np.array([0.1, 0.0, 0.0]), 2, "rotation / shear"),
    (lambda x: x / (1.0 + 0.3 * x[2]), 4, "beyond the affine ceiling"),
    (lambda x: x + x ** 2, 4, "beyond the affine ceiling"),
])
def test_classify_transform_names_the_floor(fn, layer, name):
    rep = classify_transform(fn)
    assert rep["layer"] == layer and rep["name"] == name
    assert isinstance(rep["why"], str) and len(rep["why"]) > 20


def test_delta_pushable_is_the_question_the_tower_exists_to_answer():
    # It is `shade_adjoint`'s licence, DL11's closure, and the equivariance table's shape, in one boolean.
    assert classify_transform(lambda x: x + np.array([0.1, 0, 0]))["delta_pushable"] is True
    assert classify_transform(lambda x: _R() @ x)["delta_pushable"] is True       # peers: normal ideal below
    assert classify_transform(lambda x: 1.7 * x)["delta_pushable"] is True        # centre: still affine
    assert classify_transform(lambda x: x / (1.0 + 0.3 * x[2]))["delta_pushable"] is False   # the ceiling


def test_only_the_ideal_is_bankable():
    assert classify_transform(lambda x: x + np.array([0.1, 0, 0]))["bankable"] is True
    for fn in (lambda x: _R() @ x, lambda x: 1.7 * x, lambda x: x / (1.0 + 0.3 * x[2])):
        assert classify_transform(fn)["bankable"] is False


# ---------------------------------------------------------------------------------------------------------
# on the MAIN CLASS
# ---------------------------------------------------------------------------------------------------------

def test_a_hypervector_operator_is_always_the_abelian_ideal():
    from holographic.agents_and_reasoning.holographic_ai import unitary_vector
    from holographic.sampling_and_signal.holographic_hypervector import Hypervector

    rng = np.random.default_rng(0)
    v = Hypervector.wrap(unitary_vector(256, rng))
    lay = v.transform_layer()
    assert lay["level"] == 1 and lay["name"] == "translation" and lay["diagonalisable"] is True
    assert lay == hypervector_layer()
    assert "commutative" in lay["reason"]


def test_the_hypervector_algebra_satisfies_every_abelian_group_axiom():
    # closed, associative, COMMUTATIVE, identity, inverses -- which is why no hypervector operator can ever be a
    # rotation or a shear. The algebra forbids it.
    from holographic.agents_and_reasoning.holographic_ai import unitary_vector
    from holographic.sampling_and_signal.holographic_hypervector import Hypervector

    D = 256
    rng = np.random.default_rng(0)
    a, b = Hypervector.wrap(unitary_vector(D, rng)), Hypervector.wrap(unitary_vector(D, rng))
    v = Hypervector.wrap(rng.normal(size=D))

    assert v.commutes_with(a) < 1e-12                                   # commutative
    assert np.abs(a.bind(b).array - b.bind(a).array).max() < 1e-12
    assert np.abs(v.bind(a).bind(b).array - v.bind(a.bind(b)).array).max() < 1e-10   # associative

    d0 = np.zeros(D)
    d0[0] = 1.0
    assert np.abs(v.bind(Hypervector.wrap(d0)).array - v.array).max() < 1e-10        # identity

    inv = np.fft.irfft(np.conj(np.fft.rfft(a.array)), n=D)
    assert np.abs(v.bind(a).bind(Hypervector.wrap(inv)).array - v.array).max() < 1e-10   # inverses


def test_permute_is_a_translation_in_index_space():
    from holographic.sampling_and_signal.holographic_hypervector import Hypervector

    v = Hypervector.wrap(np.random.default_rng(0).normal(size=128))
    assert np.abs(v.permute(3).permute(5).array - v.permute(8).array).max() == 0.0   # shifts ADD, exactly


def test_the_transform_bank_reports_its_own_floor():
    from holographic.caching_and_storage.holographic_transformbank import TransformBank

    b = TransformBank(128, seed=0)
    b.add_rotation("r3", 3)
    assert b.tower_layer()["level"] == 1
    assert b.layer_of("r3")["name"] == "translation"           # even the entry called "rotation"
    with pytest.raises(KeyError):
        b.layer_of("nope")


# ---------------------------------------------------------------------------------------------------------
# at the TOP LEVEL
# ---------------------------------------------------------------------------------------------------------

def test_the_tower_is_exported_from_lecore():
    import lecore

    for name in ("TOWER", "classify_transform", "commutator_table", "semidirect_law", "hypervector_layer",
                 "affine_normality", "is_affine", "texture_projection_error"):
        assert hasattr(lecore, name), name

    assert lecore.classify_transform(lambda x: 1.7 * x)["name"] == "scale"
    assert lecore.hypervector_layer()["level"] == 1


def test_the_entry_point_is_on_the_mind_too():
    import lecore

    m = lecore.UnifiedMind(dim=256, seed=0)
    assert m.classify_transform(lambda x: x + np.array([0.1, 0, 0]))["name"] == "translation"
    assert m.hypervector_layer()["level"] == 1
    for phrase in ("which floor is this transform on", "can I push a delta through this", "classify a transform"):
        assert "transform tower" in str(m.find_capability(phrase)[:3]), phrase


def test_the_modules_that_consult_the_tower_now_name_it():
    # It was a fact you had to already know to find. Now the modules whose shape it explains point at it.
    import holographic.caching_and_storage.holographic_transformbank as bank
    import holographic.mesh_and_geometry.holographic_equivariance as equiv
    import holographic.sampling_and_signal.holographic_registration as reg

    for mod in (equiv, reg, bank):
        assert "grouptower" in (mod.__doc__ or ""), mod.__name__


# ===========================================================================================
# AUDIT PASS: a matrix is data; a callable is not
# ===========================================================================================

def test_classify_transform_accepts_a_matrix():
    from holographic.mesh_and_geometry.holographic_grouptower import as_transform
    from holographic.mesh_and_geometry.holographic_projectivetower import projective

    T4 = translation([0.1, -0.2, 0.05])
    R4 = np.eye(4)
    R4[:3, :3] = _R()
    S4 = np.eye(4)
    S4[:3, :3] = 1.7 * np.eye(3)
    P4 = projective([0.15, -0.05, 0.25])

    assert classify_transform(T4)["name"] == "translation"
    assert classify_transform(R4)["name"] == "rotation / shear"
    assert classify_transform(S4)["name"] == "scale"
    assert classify_transform(P4)["name"] == "beyond the affine ceiling"   # the divide is applied

    assert classify_transform(_R())["name"] == "rotation / shear"          # a bare (3,3) linear part
    assert as_transform(lambda x: x) is not None

    with pytest.raises(ValueError, match="expected a callable or a"):
        classify_transform(np.zeros((2, 5)))


def test_the_matrix_door_is_callable_over_http():
    import json

    from holographic.mesh_and_geometry.holographic_projectivetower import projective
    from holographic_service import Service

    app = Service()
    invoke = app._routes[("POST", "/invoke")]
    res = json.loads(json.dumps(invoke({"name": "classify_transform",
                                        "args": {"fn": projective([0.15, -0.05, 0.25]).tolist()}})))["result"]
    assert res["layer"] == 4 and res["delta_pushable"] is False

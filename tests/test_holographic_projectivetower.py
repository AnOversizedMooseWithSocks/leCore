"""The projective ceiling: where the transform tower stops, and where the "word" analogy breaks.

Compose any chain of transform generators and you get ONE 4x4, exactly. So the whole transform IS the composed
group element. **But a group is not a language.** In a language a word is not a letter; in a group, the composition
of generators is another group element from the SAME set. That is CLOSURE, and it is why DL11's edit chain collapses
to a single group element rather than a sequence.

So the hierarchy is real, and it is not letters -> words -> sentences. It is a chain of subgroups ordered by
NORMALITY:

    translations  <|  Aff(3)  <  PGL(4)
    (normal in Aff)   (NOT normal in PGL)

"Which layer am I on" is the question **"can I push a delta through?"** -- yes exactly when the layer below is
normal. Conjugating a translation by a perspective leaves the affine group entirely, and the tower's mechanism
stops there.

TEXTURE PROJECTION IS THAT CEILING IN A RENDERER: affine UV interpolation is wrong by 0.3310 -- a third of the
texture -- where the homogeneous (u/w, v/w, 1/w) divide is exact.
"""

import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_grouptower import rotation3, scale, translation
from holographic.mesh_and_geometry.holographic_projectivetower import (
    affine_normality, apply_point, compose_word, conjugate, is_affine, projective, texture_projection_error,
    uv_affine, uv_perspective_correct, word_equals_chain)


def _rot4(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4)
    M[:2, :2] = [[c, -s], [s, c]]
    return M


def test_selftest_runs():
    from holographic.mesh_and_geometry import holographic_projectivetower as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# affine is a statement about the plane at infinity
# ---------------------------------------------------------------------------------------------------------

def test_affine_is_a_boolean_about_the_bottom_row():
    assert is_affine(translation([0.3, -0.7, 0.2]))
    assert is_affine(_rot4(0.6))
    assert is_affine(np.eye(4))
    assert not is_affine(projective([0.15, -0.05, 0.25]))


def test_a_nearly_affine_perspective_is_still_not_affine():
    # A matrix is affine or it is not. A perspective that is *nearly* affine still divides, and the divide is the
    # whole difference. `is_affine` is not a tolerance on a distance.
    assert not is_affine(projective([1e-6, 0.0, 0.0]))
    assert is_affine(projective([0.0, 0.0, 0.0]))


def test_the_divide_is_the_whole_map():
    P = projective([0.3, 0.0, 0.0])
    x = np.array([2.0, 1.0, 0.0])
    linear_only = (P @ np.append(x, 1.0))[:3]                 # what you get if you forget to divide
    assert not np.allclose(apply_point(P, x), linear_only)

    T = translation([0.5, 0.0, 0.0])
    assert np.allclose(apply_point(T, x), (T @ np.append(x, 1.0))[:3])   # affine: the divide is by 1


# ---------------------------------------------------------------------------------------------------------
# A WORD IS A LETTER: the group is closed
# ---------------------------------------------------------------------------------------------------------

def test_a_chain_composes_to_one_matrix_exactly():
    rng = np.random.default_rng(0)
    chain = [translation(rng.normal(size=3) * 0.2), _rot4(0.4),
             projective(rng.normal(size=3) * 0.1), scale(1.3, n=3)]
    x = np.array([0.4, -0.2, 0.3])
    assert word_equals_chain(chain, x) < 1e-12


def test_the_affine_subgroup_is_closed_and_one_projective_letter_infects_the_word():
    affine_word = compose_word([translation([0.1, 0.2, 0.3]), _rot4(0.5), translation([-0.4, 0.0, 0.1])])
    assert is_affine(affine_word)                              # affine letters spell an affine word

    mixed = compose_word([translation([0.1, 0.2, 0.3]), projective([0.1, 0.0, 0.0])])
    assert not is_affine(mixed)                                # one projective letter, and the word is projective


def test_the_empty_word_is_the_identity():
    assert np.allclose(compose_word([]), np.eye(4))


def test_composition_is_associative_but_not_commutative():
    A, B = _rot4(0.4), translation([0.3, 0.0, 0.0])
    assert np.allclose(compose_word([A, B]) @ np.eye(4), (B @ A))
    assert not np.allclose(compose_word([A, B]), compose_word([B, A]))   # order is spelling; it matters


# ---------------------------------------------------------------------------------------------------------
# THE CEILING: Aff is not normal in PGL
# ---------------------------------------------------------------------------------------------------------

def test_the_ideal_is_normal_in_the_affine_group():
    rep = affine_normality()
    assert rep["in_affine"] < 1e-12                            # A T(t) A^-1 == T(A t)


def test_conjugating_a_translation_by_a_perspective_leaves_the_affine_group():
    # THE FINDING. The tower's mechanism -- push the delta onto the other operand -- rests on normality, and stops.
    rep = affine_normality()
    assert rep["conjugate_is_affine"] is False
    assert rep["in_projective"] > 1e-3                         # its linear part is not the identity either


@pytest.mark.parametrize("A", [_rot4(0.5), scale(1.6, n=3), rotation3(1, 0.8)])
def test_every_affine_conjugator_keeps_a_translation_a_translation(A):
    C = conjugate(np.asarray(A, float), translation([0.3, -0.7, 0.2]))
    assert is_affine(C)
    assert np.abs(C[:3, :3] - np.eye(3)).max() < 1e-12         # still a pure translation


def test_a_perspective_conjugator_does_not():
    C = conjugate(projective([0.2, -0.1, 0.15]), translation([0.3, -0.7, 0.2]))
    assert not is_affine(C)
    assert np.abs(C[:3, :3] - np.eye(3)).max() > 1e-3          # not even a translation


# ---------------------------------------------------------------------------------------------------------
# the ceiling, in a renderer
# ---------------------------------------------------------------------------------------------------------

def test_affine_uv_interpolation_is_wrong_by_a_third_of_the_texture():
    rep = texture_projection_error()
    assert rep["affine_max"] > 0.2                             # measured 0.3310
    assert rep["affine_mean"] > 0.05
    assert rep["n"] > 100


def test_the_homogeneous_divide_is_exact():
    rep = texture_projection_error()
    assert rep["perspective_max"] < 1e-12                      # 2.2e-16


def test_without_perspective_the_affine_map_is_exact():
    # The ceiling only bites under perspective. Equal depths -> the map really is affine, and interpolating (u,v)
    # in screen space is correct.
    flat = texture_projection_error(depths=(2.0, 2.0, 2.0))
    assert flat["affine_max"] < 1e-12


def test_the_two_interpolators_agree_when_the_depths_agree():
    b = np.array([0.3, 0.5, 0.2])
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    w = np.array([2.0, 2.0, 2.0])
    assert np.abs(uv_affine(b, uv) - uv_perspective_correct(b, uv, w)).max() < 1e-12

    w2 = np.array([1.0, 4.0, 1.5])
    assert np.abs(uv_affine(b, uv) - uv_perspective_correct(b, uv, w2)).max() > 0.05


# ---------------------------------------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------------------------------------

def test_there_is_no_nearest_affine():
    # Projecting a perspective onto the affine subgroup throws away the only thing that made it perspective.
    import holographic.mesh_and_geometry.holographic_projectivetower as mod
    assert not hasattr(mod, "nearest_affine")


def test_wired_to_the_mind_and_discoverable():
    import lecore

    m = lecore.UnifiedMind(dim=256, seed=0)
    assert m.is_affine_matrix(translation([0.1, 0.2, 0.3])) is True
    assert m.is_affine_matrix(projective([0.1, 0.0, 0.0])) is False
    assert m.affine_normality()["conjugate_is_affine"] is False
    assert m.texture_projection_error()["affine_max"] > 0.2
    assert not m.is_affine_matrix(m.compose_word([translation([0.1, 0.2, 0.3]), projective([0.1, 0, 0])]))
    assert "projective ceiling" in str(m.find_capability("texture projection")[:3])

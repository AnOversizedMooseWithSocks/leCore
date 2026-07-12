"""holographic_projectivetower.py -- the ceiling of the transform tower, and where the "word" analogy breaks.

Moose asked whether a 4-D transform is a *word* built from transform *letters* -- and whether a texture-projection
parameter is another letter. The mathematics answers precisely, and the answer sharpens the analogy in one place
and refuses it in another.

WHAT IS A "LETTER" AND WHAT IS A "WORD"

`holographic_grouptower` established the affine tower: translation (the abelian ideal), rotation and shear (the
non-commuting peers), scale (central in the linear part). Compose any chain of those generators and you get **one
4x4 matrix**, exactly (verified to 3.3e-16 against applying the chain step by step). So yes: **the whole transform
is the composed group element, and the generators are what it is composed from.**

**BUT A GROUP IS NOT A LANGUAGE, AND THIS IS THE INTERESTING PART.** In a language a word is not a letter -- "cat"
is not a member of the alphabet. In a group, the composition of generators **is another group element**, drawn from
the very same set. Words and letters live in one alphabet. That is exactly what CLOSURE means, and it is why DL11's
edit chain collapses to a single `(S, T)` instead of needing a sequence: *the recoverable object is the group
element, not the spelling.*

So the hierarchy is real, and it is **not** letters -> words -> sentences. It is a chain of subgroups ordered by
**normality**:

        translations   <|   Aff(3)   <   PGL(4)
        (normal in Aff)      (NOT normal in PGL)

"Which layer am I standing on" is not a question about length. It is the question **"can I push a delta through?"**
-- and the answer is yes exactly when the layer below is normal.

THE CEILING, MEASURED. A 4x4 is AFFINE when its bottom row is `[0, 0, 0, 1]` -- equivalently, when it fixes the
plane at infinity. Give the bottom row three free numbers and you have a PROJECTIVE map. Then:

    A T(t) A^-1 == T(A t)                        A a rotation, shear or scale:   1.1e-16    the law HOLDS
    P T(t) P^-1  is not a translation            P a perspective:                           the law BREAKS

Conjugating a translation by a perspective produces a matrix that **is not even affine** -- its bottom row comes
back non-zero. **`Aff` is a subgroup of `PGL`, but not a normal one.** The tower's whole mechanism -- push the delta
onto the other operand, collapse the chain, read the equivariance table -- rests on normality, and it stops here.

TEXTURE PROJECTION IS THAT CEILING, IN A RENDERER. Interpolating `(u, v)` linearly across a triangle in screen space
assumes the map from the triangle to the texture is affine. Under perspective it is not. Measured, on a triangle
whose vertices have depths `w = 1, 4, 1.5`:

    affine UV (interpolate u, v)                        max error 0.3310   -- 33% of the texture
    perspective-correct (interpolate u/w, v/w, 1/w)     max error 2.2e-16  -- exact

**The extra parameter is not another letter in the same alphabet. It is an extra COORDINATE**, carried through the
transform and divided out at the end -- the `q` of a homogeneous `(u, v, q)` texture coordinate. It enlarges the
space the alphabet acts on, and by doing so it *breaks the affine group's normality*. That is why the fix is a
divide and not a matrix.

HONEST SCOPE: **I am not asserting SketchUp's API from memory.** Homogeneous texture coordinates and projective
material mapping are the standard mechanism for exactly this problem, and this module measures the mathematics.
What any particular tool calls its parameter, and how it stores it, is a question for that tool's documentation.

KEPT NEGATIVE -- **a projective map is not "affine plus a bit."** It is linear on a HIGHER-dimensional homogeneous
space, whose shadow on the affine chart is nonlinear. `is_affine` is a boolean about the bottom row, not a
tolerance on a distance -- and `nearest_affine` does not exist, because projecting a perspective onto the affine
subgroup throws away the only thing that made it perspective.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_grouptower import translation


def projective(v):
    """A projective 4x4 with bottom row `[v0, v1, v2, 1]`. **The letter the affine alphabet does not have** -- it
    moves the plane at infinity, which is exactly what an affine map may not do."""
    M = np.eye(4)
    M[3, :3] = np.asarray(v, float)
    return M


def is_affine(M, tol=1e-12):
    """Does `M` fix the plane at infinity? That is, is its bottom row `[0, 0, 0, *]`?

    A BOOLEAN, not a tolerance on a distance. A matrix is affine or it is not; a perspective that is *nearly*
    affine still divides, and the divide is the whole difference."""
    M = np.asarray(M, float)
    return bool(np.abs(M[3, :3]).max() < float(tol))


def apply_point(M, x):
    """Apply a 4x4 to a 3-vector through homogeneous coordinates, **with the divide**. For an affine `M` the divide
    is by 1 and changes nothing; for a projective one it is the whole map."""
    y = np.asarray(M, float) @ np.append(np.asarray(x, float), 1.0)
    return y[:3] / y[3]


def compose_word(chain):
    """Compose a chain of transforms, applied left to right, into ONE 4x4.

    **This is the "word".** And the point is that it is drawn from the same set as its letters: a group is closed,
    so a word IS a letter. Verified exact (3.3e-16) against applying the chain step by step."""
    W = np.eye(4)
    for M in chain:
        W = np.asarray(M, float) @ W
    return W


def word_equals_chain(chain, x):
    """`max |compose_word(chain) applied to x  -  the chain applied step by step|`. Zero, because the group is
    closed. The whole reason DL11's edit history collapses to one group element rather than a sequence."""
    x = np.asarray(x, float)
    step = np.append(x, 1.0)
    for M in chain:
        step = np.asarray(M, float) @ step
    return float(np.abs(apply_point(compose_word(chain), x) - step[:3] / step[3]).max())


def conjugate(A, B):
    """`A B A^-1`, normalised so the homogeneous scale is fixed. Conjugation is the tower's engine: it is
    `shade_adjoint`'s "push the delta onto the other operand"."""
    A = np.asarray(A, float)
    C = A @ np.asarray(B, float) @ np.linalg.inv(A)
    return C / C[3, 3]


def affine_normality(t=(0.3, -0.7, 0.2), seed=0):
    """**Where the tower's mechanism stops.** Returns `{in_affine, in_projective, conjugate_is_affine}`.

    `in_affine` -- conjugating a translation by a ROTATION gives `T(A t)`, to 1.1e-16: the ideal is normal in Aff.
    `in_projective` -- conjugating the same translation by a PERSPECTIVE gives a matrix that is not a translation
    and **not even affine**. So `Aff` is a subgroup of `PGL` and not a normal one, and no delta can be pushed
    through a perspective the way it is pushed through a rotation."""
    rng = np.random.default_rng(int(seed))
    t = np.asarray(t, float)
    T = translation(t)

    c, s = np.cos(0.6), np.sin(0.6)
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    lhs = conjugate(R, T)
    rhs = translation(R[:3, :3] @ t)
    in_affine = float(np.abs(lhs - rhs).max())

    P = projective(rng.normal(size=3) * 0.15)
    C = conjugate(P, T)
    return {"in_affine": in_affine,
            "in_projective": float(np.abs(C[:3, :3] - np.eye(3)).max()),
            "conjugate_is_affine": is_affine(C)}


# ---------------------------------------------------------------------------------------------------------
# texture projection: the ceiling, in a renderer
# ---------------------------------------------------------------------------------------------------------

def _barycentric(p, tri2d):
    A, B, C = tri2d
    M = np.array([[A[0] - C[0], B[0] - C[0]], [A[1] - C[1], B[1] - C[1]]])
    lam = np.linalg.solve(M, np.asarray(p, float) - C)
    return np.array([lam[0], lam[1], 1.0 - lam[0] - lam[1]])


def uv_affine(bary, uv):
    """Interpolate `(u, v)` linearly in SCREEN space. **Wrong under perspective**, by up to 33% of the texture."""
    return np.asarray(bary, float) @ np.asarray(uv, float)


def uv_perspective_correct(bary, uv, w):
    """Interpolate `(u/w, v/w)` and `1/w`, then divide. Exact (2.2e-16).

    **The `q` of a homogeneous `(u, v, q)` texture coordinate is that `1/w`** -- an extra coordinate carried through
    the transform and divided out at the end. It is not another letter in the alphabet; it enlarges the space the
    alphabet acts on."""
    bary = np.asarray(bary, float)
    uv = np.asarray(uv, float)
    w = np.asarray(w, float)
    inv_w = bary @ (1.0 / w)
    return (bary @ (uv / w[:, None])) / inv_w


def texture_projection_error(depths=(1.0, 4.0, 1.5), n=2000, seed=0):
    """MEASURE the ceiling in a renderer: `{affine_max, affine_mean, perspective_max}`.

    A textured triangle whose vertices sit at different depths. Interpolating UV linearly in screen space assumes
    the triangle-to-texture map is affine; under perspective it is not. Measured at depths (1, 4, 1.5):
    affine max error **0.3310**, perspective-correct **2.2e-16**."""
    w = np.asarray(depths, float)
    clip = np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.2]])
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    screen = clip / w[:, None]

    rng = np.random.default_rng(int(seed))
    aff, pc = [], []
    for _ in range(int(n)):
        lam = rng.dirichlet(np.ones(3))
        b = _barycentric(lam @ screen, screen)
        if (b < 0).any():
            continue
        # the truth: barycentric weights in 3-D are b/w, renormalised
        true_w = b / w
        true_w = true_w / true_w.sum()
        truth = true_w @ uv
        aff.append(np.abs(uv_affine(b, uv) - truth).max())
        pc.append(np.abs(uv_perspective_correct(b, uv, w) - truth).max())
    return {"affine_max": float(np.max(aff)), "affine_mean": float(np.mean(aff)),
            "perspective_max": float(np.max(pc)), "n": len(aff)}


def _selftest():
    """Regression trap: a word IS a letter (the group is closed); the affine ceiling is a boolean about the bottom
    row; conjugating a translation by a perspective leaves the affine group entirely; and affine texture mapping is
    wrong by a third of the texture while the homogeneous divide is exact."""
    rng = np.random.default_rng(0)

    # 1. AFFINE is a statement about the bottom row -- about fixing the plane at infinity
    c, s = np.cos(0.6), np.sin(0.6)
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    assert is_affine(translation([0.3, -0.7, 0.2])) and is_affine(R)
    assert not is_affine(projective([0.15, -0.05, 0.25]))

    # 2. A WORD IS A LETTER: the group is closed, so a chain composes to one element of the same set
    chain = [translation(rng.normal(size=3) * 0.2), R, projective(rng.normal(size=3) * 0.1),
             translation(rng.normal(size=3) * 0.2)]
    x = np.array([0.4, -0.2, 0.3])
    assert word_equals_chain(chain, x) < 1e-12
    W = compose_word(chain)
    assert W.shape == (4, 4)
    assert not is_affine(W)                                    # one projective letter makes the whole word projective

    # ... and a word of purely affine letters stays affine: the subgroup is closed under composition
    assert is_affine(compose_word([translation([0.1, 0.2, 0.3]), R, translation([-0.4, 0.0, 0.1])]))

    # 3. THE CEILING: the ideal is normal in Aff, and Aff is NOT normal in PGL
    rep = affine_normality()
    assert rep["in_affine"] < 1e-12                            # A T A^-1 == T(A t)
    assert rep["conjugate_is_affine"] is False                 # P T P^-1 is not even affine
    assert rep["in_projective"] > 1e-3                         # ... its linear part is not the identity

    # 4. TEXTURE PROJECTION is that ceiling in a renderer
    tex = texture_projection_error()
    assert tex["affine_max"] > 0.2, tex                        # measured 0.3310 -- a third of the texture
    assert tex["perspective_max"] < 1e-12, tex                 # the homogeneous divide is exact
    assert tex["n"] > 100

    # ... and with no perspective (equal depths) the affine map is exact: the ceiling only bites under perspective
    flat = texture_projection_error(depths=(2.0, 2.0, 2.0))
    assert flat["affine_max"] < 1e-12

    # 5. `nearest_affine` does not exist, and the refusal is deliberate
    import holographic.mesh_and_geometry.holographic_projectivetower as _mod
    assert not hasattr(_mod, "nearest_affine")

    print("OK: holographic_projectivetower self-test passed (a chain of transforms composes to ONE 4x4 to %.1e -- a "
          "group is CLOSED, so a 'word' IS a 'letter', which is exactly why DL11's edit chain collapses to a group "
          "element rather than a sequence; AFFINE is a boolean about the bottom row -- fixing the plane at infinity "
          "-- and conjugating a translation by a PERSPECTIVE leaves the affine group entirely (the conjugate is not "
          "affine), so Aff is a subgroup of PGL and NOT a normal one and the tower's push-the-delta mechanism stops "
          "there; and that ceiling is visible in a renderer: affine UV interpolation is wrong by %.4f -- a third of "
          "the texture -- where the homogeneous (u/w, v/w, 1/w) divide is exact to %.1e)"
          % (word_equals_chain(chain, x), tex["affine_max"], tex["perspective_max"]))


if __name__ == "__main__":
    _selftest()

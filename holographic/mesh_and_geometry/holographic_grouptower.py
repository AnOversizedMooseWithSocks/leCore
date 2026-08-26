"""holographic_grouptower.py -- the transform hierarchy, as a group, MEASURED.

Moose's tower, stated as a hierarchy the way letters -> words -> sentences -> document is one:

    scale                central -- commutes with the whole linear part
      ^  (dilates)
    rotation, shear      the sl(2) part -- non-commuting peers
      ^  (rotates / distorts)
    translation          the abelian ideal -- the content
      ^
    hypervectors         the atoms

This is the **Levi decomposition of the affine group**: `Aff(n) = GL(n) |x| R^n`, with the translations a normal
abelian subgroup (the ideal), `GL(n) = center x SL(n)` -- scale the center, rotation and shear the semisimple part.
It is not a picture. It makes predictions, and every one of them is checked here rather than asserted.

THE COMMUTATOR TABLE, measured on homogeneous affine matrices (`max |AB - BA|`):

    [T, T']   translations with each other            0.00e+00    the ideal is ABELIAN
    [S, R]    scale with rotation                     0.00e+00    scale is CENTRAL in GL ...
    [S, Sh]   scale with shear                        0.00e+00    ... the WHOLE linear part
    [R, Sh]   rotation with shear                     2.34e-01    non-commuting PEERS
    [Rx, Ry]  two rotations, in 3-D                   5.05e-01    (in 2-D, SO(2) is abelian -- see below)
    [S, T]    scale with TRANSLATION                  4.90e-01    central in GL, NOT in Aff
    [R, T]    rotation with translation               2.49e-01    the semidirect action

**TWO PRECISIONS THE DIAGRAM EARNS.** *Scale is central in the LINEAR part, exactly as the diagram says -- and not
in the affine group*: `[S, T] = 0.49`, because `s(x + t) = sx + st`, not `sx + t`. Scale acts on the ideal, it does
not commute past it. And in TWO dimensions the rotations commute with each other (SO(2) is abelian), so "non-
commuting peers" is really rotation-vs-SHEAR there; it becomes rotation-vs-rotation only in 3-D and above.

THE IDEAL IS NORMAL, AND THAT IS THE WHOLE MECHANISM:

    A T(t) A^-1 == T(A t)      for every linear A -- verified to 1.1e-16 for rotation, shear and scale

**That single line is three things this engine already found, wearing three costumes:**
  * it is `holographic_equivariance.shade_adjoint` -- "push the delta onto the other operand" is conjugation;
  * it is DL11's group closure -- `x -> s2(s1 x + t1) + t2` collapses because the ideal is normal;
  * it is why the equivariance table has the shape it has: an operator's law under a delta is a statement about
    which layer of this tower the delta lives in.

WHICH LAYER CAN A TRANSFORM BANK HOLD? **Exactly the ideal, and nothing above it.** A bank entry is one fixed
spectrum applied to any vector; a convolution algebra is COMMUTATIVE, so it can only represent an abelian group.
Fit the best spectrum for a transform on one encoded point and apply it to another:

    translation  x -> x + t      relative error 3.8e-16     GENERALISES: it is a bind
    rotation     x -> R x        relative error 5.4e-01     does not
    scale        x -> 1.5 x      relative error 1.3e-01     does not

And the FPE law says why: `bind(encode(x), encode(t)) == encode(x + t)` to 3.3e-16. **Translation IS the group
operation of the encoding.** So `holographic_transformbank` is not a cache of arbitrary transforms -- it is a
*representation of the abelian ideal*, and its refusal to hold a scale is the tower speaking.

*(And the bank's own "rotation" -- a cyclic shift of the index axis -- is a TRANSLATION in index space. It was never
the tower's rotation layer. The name was the bug, again.)*

HOW SCALE GETS IN: **change the axis, not the algebra.** On a LOG axis a dilation becomes a translation, so it joins
the ideal and becomes a bind. That is Reddy-Chatterji's Fourier-Mellin lift, and `registration.mellin_scale` is it.
Measured here: the fitted spectrum of a dilation generalises to a second signal with relative error 1.4e-15 on the
log axis, against 1.3e-01 on the linear one. **A layer you cannot diagonalise, you relocate.**
"""

import numpy as np

#: The tower, bottom to top. Each level names what it is in the group, and whether a single Fourier spectrum can
#: represent it (i.e. whether it can live in a `TransformBank`).
TOWER = (
    {"level": 0, "name": "hypervectors", "role": "the atoms", "diagonalisable": None},
    {"level": 1, "name": "translation", "role": "the abelian ideal -- the content", "diagonalisable": True},
    {"level": 2, "name": "rotation, shear", "role": "the sl(n) part -- non-commuting peers", "diagonalisable": False},
    {"level": 3, "name": "scale", "role": "central -- commutes with the whole linear part", "diagonalisable": False},
)


def homogeneous(A=None, t=None, n=2):
    """An affine map as an `(n+1, n+1)` homogeneous matrix: linear part `A`, translation `t`."""
    M = np.eye(n + 1)
    if A is not None:
        M[:n, :n] = np.asarray(A, float)
    if t is not None:
        M[:n, n] = np.asarray(t, float)
    return M


def translation(t):
    """A pure translation, as a homogeneous matrix. **The abelian ideal** -- these commute with each other, and they
    are the only layer a Fourier spectrum (a bind) can represent."""
    t = np.asarray(t, float)
    return homogeneous(t=t, n=len(t))


def rotation2(a):
    """A 2-D rotation by `a` radians. NOTE: SO(2) is ABELIAN -- two of these commute. The tower's "non-commuting
    peers" is rotation-vs-SHEAR in the plane, and only becomes rotation-vs-rotation in 3-D."""
    c, s = np.cos(a), np.sin(a)
    return homogeneous(A=[[c, -s], [s, c]], n=2)


def rotation3(axis, angle):
    """A 3-D rotation about a coordinate `axis`. SO(3) is NOT abelian: `[Rx, Ry] = 0.505`, so the tower's peers
    genuinely do not commute above two dimensions."""
    v = np.zeros(3)
    v[int(axis)] = 1.0
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return homogeneous(A=np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K, n=3)


def shear2(k):
    """A 2-D shear. A non-commuting PEER of rotation: `[R, Sh] = 0.234`."""
    return homogeneous(A=[[1.0, float(k)], [0.0, 1.0]], n=2)


def scale(s, n=2):
    """A uniform dilation. **Central in the LINEAR part** (`[S, R] = [S, Sh] = 0`) and NOT in the affine group
    (`[S, T] = 0.49`), because `s(x + t) = sx + st`: scale acts ON the ideal rather than commuting past it."""
    return homogeneous(A=float(s) * np.eye(n), n=n)


def commutator(A, B):
    """`max |AB - BA|`. Zero means they commute; the number is how far from commuting they are."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return float(np.abs(A @ B - B @ A).max())


def commutator_table():
    """Every claim in the tower, as a number. Regenerate it; a table that cannot re-measure itself is a rumour."""
    t1, t2 = translation([0.3, -0.7]), translation([1.1, 0.2])
    r1, r2 = rotation2(0.4), rotation2(1.1)
    sh, s1, s2 = shear2(0.6), scale(1.7), scale(0.6)
    rx, ry = rotation3(0, 0.7), rotation3(1, 0.9)
    return {
        "[T,T'] ideal is abelian": commutator(t1, t2),
        "[R,R'] SO(2) is abelian": commutator(r1, r2),
        "[Rx,Ry] SO(3) is not": commutator(rx, ry),
        "[R,Sh] non-commuting peers": commutator(r1, sh),
        "[S,R] scale central in GL": commutator(s1, r1),
        "[S,Sh] ... the whole linear part": commutator(s1, sh),
        "[S,S'] scales commute": commutator(s1, s2),
        "[S,T] NOT central in Aff": commutator(s1, t1),
        "[R,T] the semidirect action": commutator(r1, t1),
    }


def semidirect_law(linear, t):
    """`max |A T(t) A^-1 - T(A t)|`. **Zero, because the ideal is NORMAL.**

    This is conjugation, and conjugation is `shade_adjoint`'s "push the delta onto the other operand", and it is
    why DL11's affine chain collapses to one group element. One line, three costumes."""
    A = np.asarray(linear, float)
    n = A.shape[0] - 1
    lhs = A @ translation(t) @ np.linalg.inv(A)
    rhs = translation(A[:n, :n] @ np.asarray(t, float))
    return float(np.abs(lhs - rhs).max())


def diagonalisable(transform, encoder=None, points=None, seed=0):
    """**Can a single Fourier spectrum represent `transform`?** Fit the best spectrum on one encoded point, apply it
    to another, return the relative error.

    Near zero -> the transform is a BIND, and belongs in a `TransformBank`. Large -> no spectrum represents it, and
    the bank must refuse it. Measured: translation 3.8e-16, rotation 5.4e-01, scale 1.3e-01.

    A convolution algebra is commutative, so **only an abelian group can be diagonalised** -- and the tower's only
    abelian layer is the ideal. This function is the tower's prediction, made checkable."""
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder

    transform = as_transform(transform, dim=3)                 # a matrix is data; a callable is not
    if encoder is None:
        encoder = VectorFunctionEncoder(3, dim=1024, bounds=[(-1, 1)] * 3, seed=int(seed))
    if points is None:
        rng = np.random.default_rng(int(seed))
        points = rng.uniform(-0.4, 0.4, (2, 3))               # kept well inside the encoder's range: bad data lies
    p1, p2 = np.asarray(points, float)[:2]

    dim = encoder.dim
    fitted = np.fft.rfft(encoder.encode(transform(p1))) / np.fft.rfft(encoder.encode(p1))
    pred = np.fft.irfft(np.fft.rfft(encoder.encode(p2)) * fitted, n=dim)
    want = encoder.encode(transform(p2))
    return float(np.abs(pred - want).max() / max(float(np.abs(want).max()), 1e-30))


def mellin_promotes_scale(n=1024, s=1.4, seed=0):
    """**A layer you cannot diagonalise, you relocate.** On a LOG axis a dilation becomes a translation, so it joins
    the abelian ideal and becomes a bind.

    Returns `{linear_axis, log_axis}` -- the `diagonalisable` relative error for a dilation on each. Measured: 0.13
    on the linear axis, ~1e-15 on the log one. That is Reddy-Chatterji, and `registration.mellin_scale` is the
    engine's use of it."""
    rng = np.random.default_rng(int(seed))
    u = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)      # a LOG-axis coordinate

    def _fit_transfer(sig1, out1, sig2, out2):
        S = np.fft.rfft(out1) / np.fft.rfft(sig1)
        pred = np.fft.irfft(np.fft.rfft(sig2) * S, n=n)
        return float(np.abs(pred - out2).max() / max(float(np.abs(out2).max()), 1e-30))

    # ON THE LOG AXIS: a dilation by s is a SHIFT by log(s). Shifts are circulant -> one spectrum, any signal.
    k = int(round(np.log(s) / (u[1] - u[0]))) or 1
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    log_err = _fit_transfer(f1, np.roll(f1, k), f2, np.roll(f2, k))

    # ON THE LINEAR AXIS: a dilation resamples the index. Not shift-invariant.
    def _dilate(x):
        idx = np.arange(n) / float(s)
        i = np.floor(idx).astype(int)
        fr = idx - i
        return x[i % n] * (1 - fr) + x[(i + 1) % n] * fr

    lin_err = _fit_transfer(f1, _dilate(f1), f2, _dilate(f2))
    return {"linear_axis": lin_err, "log_axis": log_err}


def as_transform(fn, dim=3):
    """Coerce to a callable on points. Accepts a callable, or a **MATRIX** -- `(dim, dim)` linear, `(dim, dim+1)`
    affine, or `(dim+1, dim+1)` homogeneous (applied with the divide, so a perspective classifies as one).

    **A matrix is data; a callable is not.** A transform an agent can POST is a transform it can ask about, and a
    capability an agent cannot call does not exist."""
    if callable(fn):
        return fn
    A = np.asarray(fn, float)
    d = int(dim)
    if A.shape == (d, d):
        return lambda x, _A=A: _A @ np.asarray(x, float)
    if A.shape == (d, d + 1):
        return lambda x, _A=A: _A[:, :d] @ np.asarray(x, float) + _A[:, d]
    if A.shape == (d + 1, d + 1):
        def _homog(x, _A=A):
            y = _A @ np.append(np.asarray(x, float), 1.0)
            return y[:d] / y[d]                                # WITH the divide: a perspective stays a perspective
        return _homog
    raise ValueError("expected a callable or a (%d,%d) / (%d,%d) / (%d,%d) matrix; got %r"
                     % (d, d, d, d + 1, d + 1, d + 1, A.shape))


def classify_transform(fn, dim=3, n=64, tol=1e-9, seed=0):
    """**THE ONE ENTRY POINT: which floor of the tower is this transform standing on?**

    Give it any callable on points. It measures, and returns
    `{layer, name, diagonalisable, bankable, delta_pushable, why}`.

    The classification is a decision procedure, not a lookup:

      1. Is one Fourier spectrum enough? -> the **abelian ideal**. It is a bind, it goes in a `TransformBank`, and a
         delta pushes through it trivially because the layer commutes with itself.
      2. Otherwise, does a linear fit `x -> A x + b` explain it exactly? Then it is **affine**, and the linear part
         says which floor: `A` a multiple of the identity is the **centre** (scale); anything else is the
         **sl(n) peers** (rotation, shear). A delta still pushes through, because the translations are NORMAL in
         `Aff` -- that is `A T(t) A^-1 = T(A t)`.
      3. Otherwise it is **beyond the affine ceiling** -- projective, or not a group action at all. No delta pushes
         through, because `Aff` is not normal in `PGL`. See `holographic_projectivetower`.

    `delta_pushable` is the question the tower exists to answer. It is `shade_adjoint`'s licence, DL11's closure,
    and the equivariance table's shape, in one boolean."""
    fn = as_transform(fn, dim=int(dim))
    rng = np.random.default_rng(int(seed))
    P = rng.uniform(-0.4, 0.4, (int(n), int(dim)))

    # 1. the ideal: is it a bind?
    if int(dim) == 3:
        try:
            if diagonalisable(fn, seed=int(seed)) < float(tol):
                return {"layer": 1, "name": "translation", "diagonalisable": True, "bankable": True,
                        "delta_pushable": True,
                        "why": "one Fourier spectrum represents it: it is a bind, and the abelian ideal is what a "
                               "convolution algebra can hold"}
        except Exception:                                     # a transform the encoder cannot see; fall through
            pass

    # 2. affine? fit x -> A x + b exactly, by least squares on a design matrix with a constant column
    Y = np.stack([np.asarray(fn(p), float) for p in P])
    G = np.hstack([P, np.ones((len(P), 1))])
    sol, *_ = np.linalg.lstsq(G, Y, rcond=None)
    resid = float(np.abs(G @ sol - Y).max())
    if resid > 1e-8:
        return {"layer": 4, "name": "beyond the affine ceiling", "diagonalisable": False, "bankable": False,
                "delta_pushable": False,
                "why": "no exact affine fit (residual %.2e): projective, or not a group action. Aff is a subgroup "
                       "of PGL and NOT a normal one, so no delta pushes through -- see holographic_projectivetower"
                       % resid}

    A = sol[:int(dim)].T
    b = sol[int(dim)]
    if np.abs(A - np.eye(int(dim))).max() < 1e-8:
        return {"layer": 1, "name": "translation", "diagonalisable": True, "bankable": True, "delta_pushable": True,
                "why": "linear part is the identity: a pure translation, the abelian ideal"}

    s = float(np.trace(A)) / int(dim)
    if np.abs(A - s * np.eye(int(dim))).max() < 1e-8:
        return {"layer": 3, "name": "scale", "diagonalisable": False, "bankable": False, "delta_pushable": True,
                "why": "linear part is a multiple of the identity: the CENTRE of GL. It commutes with the whole "
                       "linear part and NOT with translation ([S,T] != 0), because s(x+t) = sx + st"}

    return {"layer": 2, "name": "rotation / shear", "diagonalisable": False, "bankable": False,
            "delta_pushable": True,
            "why": "an exact affine fit with a non-scalar linear part: the sl(n) peers. Not a bind (no single "
                   "spectrum), but the ideal is NORMAL in Aff, so a delta still pushes through by conjugation: "
                   "A T(t) A^-1 = T(A t)"}


def hypervector_layer():
    """**Which floor does a HYPERVECTOR operator stand on? Always the abelian ideal. The algebra forbids anything
    else.**

    `bind` is a circular convolution, so the set of hypervector operators is closed, associative, COMMUTATIVE, has
    an identity (the delta at 0) and has inverses for unitary atoms -- every axiom of an abelian group, verified in
    `Hypervector`'s own tests. A convolution algebra can only represent an abelian group, so **no hypervector
    operator can ever be a rotation or a shear.** `permute` is not an exception: it is a translation in INDEX space,
    and two permutes compose by ADDING their shifts, exactly.

    This is why `TransformBank` refuses a scale, why `iterate.step_k` can jump a million steps, and why DL11's edit
    chain collapses. One fact, at the bottom of the tower."""
    return dict(TOWER[1], reason="bind is a circular convolution, hence commutative; a convolution algebra can only "
                                 "represent an abelian group")


def _selftest():
    """Regression trap: every claim of the tower, as a number. The ideal is abelian and normal; scale is central in
    the LINEAR part and not in the affine group; only the ideal is diagonalisable; and the Mellin lift promotes
    scale into it."""
    tab = commutator_table()

    # 1. the ideal is ABELIAN
    assert tab["[T,T'] ideal is abelian"] == 0.0

    # 2. scale is CENTRAL in the linear part -- exactly what the diagram says
    assert tab["[S,R] scale central in GL"] == 0.0
    assert tab["[S,Sh] ... the whole linear part"] == 0.0
    assert tab["[S,S'] scales commute"] == 0.0

    # 3. ... and NOT central in the affine group: scale acts on the ideal
    assert tab["[S,T] NOT central in Aff"] > 0.1

    # 4. the peers do not commute -- but in 2-D that is rotation vs SHEAR, because SO(2) is abelian
    assert tab["[R,Sh] non-commuting peers"] > 0.1
    assert tab["[R,R'] SO(2) is abelian"] < 1e-12
    assert tab["[Rx,Ry] SO(3) is not"] > 0.1
    assert tab["[R,T] the semidirect action"] > 0.1

    # 5. THE IDEAL IS NORMAL: A T(t) A^-1 == T(A t). Conjugation. The adjoint move. DL11's closure.
    for A in (rotation2(0.5), shear2(0.4), scale(1.6)):
        assert semidirect_law(A, [0.3, -0.7]) < 1e-12

    # 6. ONLY THE IDEAL IS DIAGONALISABLE -- the tower's prediction, on the engine's own encoding
    from holographic.agents_and_reasoning.holographic_ai import bind
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder

    enc = VectorFunctionEncoder(3, dim=1024, bounds=[(-1, 1)] * 3, seed=0)
    t = np.array([0.1, -0.2, 0.15])
    R3 = rotation3(2, np.pi / 2)[:3, :3]

    err_t = diagonalisable(lambda x: x + t, enc)
    err_r = diagonalisable(lambda x: R3 @ x, enc)
    err_s = diagonalisable(lambda x: 1.5 * x, enc)
    assert err_t < 1e-12, err_t                                # a bind
    assert err_r > 0.05, err_r                                 # not
    assert err_s > 0.05, err_s                                 # not

    # ... and the FPE law says why: translation IS the group operation of the encoding
    x = np.random.default_rng(0).uniform(-0.4, 0.4, 3)
    assert np.abs(bind(enc.encode(x), enc.encode(t)) - enc.encode(x + t)).max() < 1e-12

    # 7. the TransformBank is a representation of the ideal, and refuses everything above it
    from holographic.caching_and_storage.holographic_transformbank import TransformBank

    bank = TransformBank(256, seed=0)
    assert not hasattr(bank, "add_scale")
    bank.add_random_unitary("a")
    bank.add_random_unitary("b")
    assert np.abs(bank.compose(["a", "b"]) - bank.compose(["b", "a"])).max() < 1e-12   # abelian, necessarily

    # 8. a layer you cannot diagonalise, you RELOCATE: the Mellin lift puts scale in the ideal
    mel = mellin_promotes_scale()
    assert mel["log_axis"] < 1e-9, mel
    assert mel["linear_axis"] > 0.05, mel
    assert mel["linear_axis"] > 1e6 * mel["log_axis"]

    # 9. the tower is declared, and its `diagonalisable` flags match the measurements
    assert [lv["name"] for lv in TOWER] == ["hypervectors", "translation", "rotation, shear", "scale"]
    assert TOWER[1]["diagonalisable"] is True
    assert TOWER[2]["diagonalisable"] is False and TOWER[3]["diagonalisable"] is False

    print("OK: holographic_grouptower self-test passed (the ideal is abelian [T,T'] = 0 and NORMAL -- "
          "A T(t) A^-1 = T(At) to 1e-12, which is `shade_adjoint` and DL11's closure in one line; scale is central "
          "in the LINEAR part ([S,R] = [S,Sh] = 0) and NOT in the affine group ([S,T] = %.2f), because it acts on "
          "the ideal; the peers do not commute ([R,Sh] = %.2f, and in 2-D that is rotation-vs-SHEAR because SO(2) "
          "is abelian -- [Rx,Ry] = %.2f only in 3-D). And only the IDEAL is diagonalisable: a single spectrum "
          "represents a translation to %.1e and a rotation to %.2f, so a TransformBank is a representation of the "
          "abelian ideal, not a cache of transforms. A layer you cannot diagonalise, you RELOCATE: on a log axis a "
          "dilation is a translation, %.1e against %.2f on the linear one)"
          % (tab["[S,T] NOT central in Aff"], tab["[R,Sh] non-commuting peers"], tab["[Rx,Ry] SO(3) is not"],
             err_t, err_r, mel["log_axis"], mel["linear_axis"]))


if __name__ == "__main__":
    _selftest()

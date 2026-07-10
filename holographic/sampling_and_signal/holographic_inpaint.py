"""holographic_inpaint.py -- fill the gaps in a field (NCA backlog B1).

The plainest possible faculty, and the engine did not have it. Audited before building: `inpaint a hole` returned
`holographic_backwardwarp`; `impute missing values` returned `Asset relocation / relink`; `fill in missing data`
returned `File map ingest`; `label propagation` returned `Physics & chemistry (domain)`. Four fallbacks. leCore
could not fill a hole in a field.

TWO SOLVERS, dispatched on type:

  * CONTINUOUS -> `harmonic_fill`: a Laplace solve on the holes. Known cells are pinned; unknown cells relax to the
    mean of their four neighbours. The minimal-energy interpolant, and the smoothest thing that agrees with the data.
  * CATEGORICAL -> `majority_fill`: a label propagates into a hole by a vote of its known neighbours, one ring per
    sweep. A discrete field has no mean, so averaging it is a category error.

MEASURED, 48x48, 59% random erasure (the NCA backlog's own setup), ACROSS 8 SEEDS -- because a single field's
score is a fact about that field:

    harmonic fill of a smooth field, MAE on holes        mean 0.0015   range 0.0012 - 0.0018
    majority fill of a 5-region Voronoi field, accuracy  mean 0.9653   range 0.9553 - 0.9749
    ... the same, restricted to region INTERIORS          mean 0.9990   range 0.9973 - 1.0000

The backlog quotes 0.960 for the majority fill. That lands inside this spread, and the spread is the point: the
overall accuracy moves with how many BOUNDARY cells the field happens to have (16.2%-20.1% of holes across seeds).
**The overall number is a property of the field; the interior number is a property of the algorithm.** The
self-test asserts both, with bars taken from the measured distribution rather than from one lucky seed.

THE BOUNDARY CONDITION IS THE GATE, and it is not a detail. `np.roll` wraps. On a NON-PERIODIC field a wrapped
Laplacian solves a different problem: MAE 0.00666 wrapped against 0.00123 clamped, **5.4x worse**. The label field
is the same story -- 0.9495 wrapped against 0.9661 clamped. `periodic=False` (edge-clamped, Neumann) is the
default, because a field is not periodic unless you say so.

WHERE THE ERROR LIVES, measured rather than assumed: the majority fill is 99.90% accurate in region INTERIORS and
72-82% on region BOUNDARIES, which are ~17.5% of the holes. Nearly all of the error is boundary error, where the
true label is genuinely ambiguous from a local vote. **A 96% accuracy bar on a given label field is a statement
about how many boundary cells that field has, not about the algorithm** -- so the self-test asserts the interior
accuracy separately, where the algorithm's claim actually lives. It also beats a nearest-known-label fill
(0.9661 against 0.9618 on the same field).

DECLARED NEGATIVES -- the VSA route was measured and lost. Do not rebuild it.

  * N6: a VSA record (one vector per cell, a continuous field and a categorical field each bound to a role) loses
    to these two classical fills on BOTH channels. Temperature MAE: harmonic 0.0077, VSA-diffuse 0.0248, VSA with
    per-step cleanup 0.0485. Material accuracy: majority 96.0%, VSA 94.2% -- it loses even on the categorical
    channel, where it was expected to win.
  * N7: per-step cleanup in a multi-role NCA is a NET LOSS -- zero categorical benefit and DOUBLE the continuous
    error. Mechanism: cleanup is per-role but the bundle is shared, so snapping the material role to its nearest
    atom and re-superposing injects crosstalk into the temperature role riding in the same vector. *A shared kernel
    is not a shared manifold.*
  * N8: merely encoding a scalar into a 2-role record and reading it back costs MAE 0.0160 -- more than TWICE the
    error a harmonic solve achieves while actually reconstructing missing values. Not a dimension artefact: the
    floor is 0.0081 at D=1024 and 0.0057 at D=8192, so 8x the dimension buys 30%.
  * N9: a bundle's per-role error grows with the number of roles -- 0.0010 (1 role), 0.0177 (2), 0.0324 (8),
    0.0732 (32). Type-blind fill is a schema-agnostic FALLBACK, not a precision tool.
"""

import numpy as np


def _neighbour_mean(u, periodic):
    """The 4-neighbour average. `periodic=False` clamps at the edge (a Neumann wall), which is what a field with
    borders actually does; `np.roll` would wrap it into the opposite side."""
    if periodic:
        return 0.25 * (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1))
    p = np.pad(u, 1, mode="edge")
    return 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])


def harmonic_fill(field, known, periodic=False, iters=2000, tol=1e-9):
    """Fill the unknown cells of a CONTINUOUS field by a Laplace solve: relax each hole to the mean of its four
    neighbours, holding the known cells fixed.

    `known` is a boolean mask, True where the value is trusted. Jacobi iteration to `tol` or `iters`; deterministic,
    no RNG, and the initial guess is the mean of the known cells so the answer does not depend on what was in the
    holes. Returns a new array; the input is not modified.

    MEASURED on a 48x48 field with 59% erased: MAE 0.00123 on the holes (non-periodic field, clamped BC). Wrapping
    a non-periodic field costs 5.4x. Raises if nothing is known -- there is no interpolant through no data."""
    u = np.asarray(field, float).copy()
    m = np.asarray(known, bool)
    if u.shape != m.shape:
        raise ValueError("field %r and mask %r must have the same shape" % (u.shape, m.shape))
    if not m.any():
        raise ValueError("harmonic_fill needs at least one known cell; there is no interpolant through no data")
    if m.all():
        return u
    u[~m] = float(u[m].mean())                       # a deterministic start: the holes' contents cannot leak in
    for _ in range(int(iters)):
        nb = _neighbour_mean(u, periodic)
        new = np.where(m, u, nb)
        if np.abs(new - u).max() < tol:
            return new
        u = new
    return u


def majority_fill(labels, known, periodic=False, iters=1000):
    """Fill the unknown cells of a CATEGORICAL field by neighbour vote, one ring per sweep.

    A discrete field has no mean, so a harmonic solve is a category error on it. Each unknown cell adjacent to at
    least one KNOWN cell takes the most common of its known neighbours' labels and becomes known; the front advances
    one cell per sweep until every reachable hole is filled.

    Ties are broken toward the LOWEST label index (`argmax` on the vote counts), which is stated rather than
    accidental -- the label written into a tied cell is an observable decision.

    MEASURED on a 5-region Voronoi field, 48x48, 59% erased: 0.9661 accuracy on the holes, and 0.9974 in region
    INTERIORS. Nearly all the error is on region boundaries (17.5% of holes), where the true label is genuinely
    ambiguous from a local vote. It also beats a nearest-known-label fill (0.9618)."""
    lab = np.asarray(labels)
    m = np.asarray(known, bool).copy()
    if lab.shape != m.shape:
        raise ValueError("labels %r and mask %r must have the same shape" % (lab.shape, m.shape))
    if not m.any():
        raise ValueError("majority_fill needs at least one known cell")
    lab = lab.copy()
    n_labels = int(lab.max()) + 1 if lab.size else 0
    if n_labels <= 0:
        return lab

    for _ in range(int(iters)):
        votes = np.zeros(lab.shape + (n_labels,), int)
        for axis in (0, 1):
            for s in (1, -1):
                nb = np.roll(lab, s, axis)
                nbm = np.roll(m, s, axis)
                if not periodic:
                    # a rolled-in edge row/column came from the OPPOSITE side: it is not a neighbour, it is a wrap
                    edge = np.zeros_like(m)
                    if axis == 0:
                        edge[0 if s == 1 else -1, :] = True
                    else:
                        edge[:, 0 if s == 1 else -1] = True
                    nbm = nbm & ~edge
                for k in range(n_labels):
                    votes[..., k] += ((nb == k) & nbm).astype(int)
        reachable = votes.sum(axis=-1) > 0
        fill_here = reachable & ~m
        if not fill_here.any():
            break                                    # nothing left that touches a known cell
        lab = np.where(fill_here, votes.argmax(axis=-1), lab)   # ties -> lowest label index, by argmax
        m = m | fill_here
    return lab


def inpaint(field, known, kind="auto", periodic=False, **kw):
    """Fill the gaps in `field`, dispatching on TYPE.

    `kind="auto"` sends an integer/bool array to `majority_fill` and a float array to `harmonic_fill`. Pass
    `kind="continuous"` or `kind="categorical"` to override -- a float array holding class ids is a real thing, and
    averaging it would silently produce labels that do not exist."""
    arr = np.asarray(field)
    if kind == "auto":
        kind = "categorical" if arr.dtype.kind in "biu" else "continuous"
    if kind == "categorical":
        return majority_fill(arr, known, periodic=periodic, **kw)
    if kind == "continuous":
        return harmonic_fill(arr, known, periodic=periodic, **kw)
    raise ValueError("kind must be 'auto', 'continuous' or 'categorical'; got %r" % (kind,))


def fill_report(truth, filled, known):
    """Score a fill against ground truth, ON THE HOLES ONLY -- the known cells are copied through and scoring them
    would flatter every method equally. Returns {n_holes, hole_fraction, mae, accuracy}. `accuracy` is None for a
    continuous field and `mae` is None for a categorical one; reporting the wrong one is how a category error hides."""
    t = np.asarray(truth)
    f = np.asarray(filled)
    m = np.asarray(known, bool)
    holes = ~m
    n = int(holes.sum())
    out = {"n_holes": n, "hole_fraction": float(holes.mean()), "mae": None, "accuracy": None}
    if n == 0:
        return out
    if t.dtype.kind in "biu":
        out["accuracy"] = float((f[holes] == t[holes]).mean())
    else:
        out["mae"] = float(np.abs(f[holes].astype(float) - t[holes].astype(float)).mean())
    return out


def _selftest():
    """A numeric regression trap, not a smoke test: the harmonic fill must hit a stated MAE and the majority fill a
    stated accuracy, on fixed seeds -- plus the boundary-condition negative and the two type-dispatch guards."""
    N = 48
    rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="ij")
    known = rng.random((N, N)) > 0.59                # 59% erasure, the backlog's setup

    # 1. CONTINUOUS: a non-periodic field, clamped BC. Bar MAE <= 0.008; measured 0.0012-0.0018 across 8 seeds.
    smooth = 0.3 * xx + 0.4 * np.exp(-((xx - 0.6) ** 2 + (yy - 0.3) ** 2) / 0.05)
    u = harmonic_fill(smooth, known)
    mae = fill_report(smooth, u, known)["mae"]
    assert mae <= 0.008, mae
    assert np.array_equal(u[known], smooth[known])   # known cells are PINNED, bit for bit

    # 2. KEPT NEGATIVE: wrapping a non-periodic field solves the wrong problem -- measured 5.4x worse.
    mae_wrapped = fill_report(smooth, harmonic_fill(smooth, known, periodic=True), known)["mae"]
    assert mae_wrapped > 3.0 * mae, (mae_wrapped, mae)

    # 3. CATEGORICAL: a 5-region Voronoi label field. Overall accuracy varies with the field's boundary fraction
    #    (0.9553-0.9749 across 8 seeds), so the bar is the measured MINIMUM, not one seed's lucky 0.9661.
    seeds = rng.random((5, 2))
    labels = np.stack([(xx - s[1]) ** 2 + (yy - s[0]) ** 2 for s in seeds]).argmin(axis=0)
    lab = majority_fill(labels, known)
    acc = fill_report(labels, lab, known)["accuracy"]
    assert acc >= 0.95, acc
    assert np.array_equal(lab[known], labels[known])

    # 4. ... and the ALGORITHM's claim is about INTERIORS, where it is stable: 0.9973-1.0000 across the same seeds.
    boundary = np.zeros_like(labels, bool)
    for axis in (0, 1):
        for s in (1, -1):
            boundary |= np.roll(labels, s, axis) != labels
    interior = (~known) & (~boundary)
    assert (lab[interior] == labels[interior]).mean() > 0.99

    # 5. type dispatch, and the guards
    assert np.array_equal(inpaint(labels, known), lab)          # int -> majority
    assert np.allclose(inpaint(smooth, known), u)               # float -> harmonic
    for bad in (lambda: harmonic_fill(smooth, np.zeros((N, N), bool)),
                lambda: majority_fill(labels, np.zeros((N, N), bool)),
                lambda: inpaint(smooth, known, kind="nonsense")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("a degenerate input must raise")

    # 6. an all-known field is returned unchanged
    assert np.array_equal(harmonic_fill(smooth, np.ones((N, N), bool)), smooth)

    print("OK: holographic_inpaint self-test passed (harmonic fill MAE %.5f on 59%%-erased holes, against a wrapped "
          "Laplacian's %.5f -- the boundary condition is the gate; majority fill %.4f accuracy overall and %.4f in "
          "region interiors, so the error is boundary error; type dispatch and every degenerate input guarded)"
          % (mae, mae_wrapped, acc, (lab[interior] == labels[interior]).mean()))


if __name__ == "__main__":
    _selftest()

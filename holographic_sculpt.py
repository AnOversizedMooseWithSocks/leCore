"""FS-1 -- implicit-field sculpt brushes (holographic_sculpt).

WHAT THIS IS
------------
The field atoms packaged as SCULPT BRUSHES. A surface is carried as a FIELD -- a function point -> scalar whose
level-set is the surface (holographic_meshbridge meshes it with marching_tetrahedra). A brush is a LOCAL,
falloff-weighted edit of that field in a ball around a point: inflate, carve, smooth, grab, flatten, pinch. Each
takes a field function and returns a NEW field function that differs from the original ONLY inside the ball -- so the
surface changes only where you brushed, and the result re-extracts to a clean watertight mesh at any resolution. This
is the resolution-independent "sculpt the field, then re-mesh correct topology" move (DynaMesh / Sculptris territory)
that a fixed-mesh pipeline cannot do.

WORKS ON ANY FIELD, NOT JUST THE SURFACE SDF. A brush operates on a field FUNCTION (vectorized: P of shape (N, 3) ->
(N,) values), so the same operator does:
  * surface sculpting on the meshable SDF / metaball field (the primary use),
  * REWARD SHAPING on the creature's value landscape (holographic_field.landscape IS a value field -- inflate a
    region to make it more rewarding, carve to make it less),
  * local strengthening of a density / memory landscape (the radius + falloff the bare `reinforce` lacks).
One operator, many fields -- the only thing that changes per domain is the distance metric (Euclidean here for 3-D
geometry; the hypersphere `Field` would use angular distance, the same construction).

THE FALLOFF matches the engine's shipped soft-selection brush (holographic_meshgeodesic.geodesic_soft_selection):
'smooth' is the smoothstep 1 - (3t^2 - 2t^3), 'linear' is 1 - t, and BOTH are exactly 0 beyond the radius -- which is
what guarantees the edit is local (the field outside the ball is bit-identical).

WHAT IT PROVIDES
  * falloff(d, r, kind) -- the radial weight (vectorized), matching the shipped soft-selection shapes.
  * brush_inflate / brush_carve -- raise / lower the field in the ball (grow / shrink a high-inside surface).
  * brush_smooth -- blend the field toward its local average in the ball (Laplacian smoothing of the field).
  * brush_grab -- drag the field's domain by a vector inside the ball (pull the surface along).
  * brush_flatten -- pull the field toward a target level in the ball (flatten the surface to a plane value).
  * brush_pinch -- drag the domain toward the brush centre in the ball (cinch the surface in).
  * apply_brush(fn, kind, p, r, s, **kw) -- dispatch by name.

DONE-WHEN (checked in the self-test, on a metaball field meshed with marching_tetrahedra):
  * LOCAL: the brushed field is bit-identical to the original at every point OUTSIDE the ball (the surface there does
    not move), guaranteed because the falloff is exactly 0 past the radius.
  * CORRECT SIGN: inflate GROWS the surface (more of the volume rises above the mesh level), carve SHRINKS it -- the
    expected signed move in the band.
  * STILL WATERTIGHT/MANIFOLD: the re-extracted mesh after a brush is manifold (marching_tetrahedra keeps it
    watertight for any field).

DETERMINISM (per ISA.md): pure functions of the field and the brush parameters -- no RNG. Same field + same brush
give the same edited field (asserted).

KEPT HONEST: on a DENSE field the re-extract after each stroke is still O(res^3) -- FS-2 (the narrow band) is what
makes a stroke cost O(brush); FS-1 is the brush math, not yet the fast representation. A grab/pinch that drags past
the band can fold the level set; keep the drag within the radius (FS-2's reinitialize re-distances after such edits).
"""

import numpy as np


def falloff(d, r, kind="smooth"):
    """Radial brush weight at distance(s) `d` for radius `r`: 1 at the centre, falling to 0 at `r`, exactly 0 beyond
    (vectorized). Matches the shipped soft-selection brush: 'smooth' = smoothstep 1 - (3t^2 - 2t^3), 'linear' = 1 - t.
    The exact-0-beyond-radius is what makes a brush LOCAL."""
    d = np.asarray(d, float)
    t = np.clip(d / max(r, 1e-12), 0.0, 1.0)
    if kind == "linear":
        w = 1.0 - t
    elif kind == "smooth":
        w = 1.0 - (3.0 * t ** 2 - 2.0 * t ** 3)            # smoothstep, identical to geodesic_soft_selection
    else:
        raise ValueError(f"falloff kind must be 'smooth' or 'linear', got {kind!r}")
    return w


def _ball_weights(P, p, r, kind):
    """Per-point brush weights for the points `P` (N, 3) under a ball at `p` of radius `r`. Zero outside the ball."""
    P = np.asarray(P, float)
    p = np.asarray(p, float)
    d = np.linalg.norm(P - p, axis=1)
    return falloff(d, r, kind)


def brush_inflate(fn, p, r, s=0.3, kind="smooth"):
    """RAISE the field by `s` (falloff-weighted) inside the ball -- grows a high-inside surface outward."""
    p = np.asarray(p, float)
    return lambda P: fn(P) + s * _ball_weights(P, p, r, kind)


def brush_carve(fn, p, r, s=0.3, kind="smooth"):
    """LOWER the field by `s` (falloff-weighted) inside the ball -- shrinks/dents a high-inside surface inward."""
    p = np.asarray(p, float)
    return lambda P: fn(P) - s * _ball_weights(P, p, r, kind)


def brush_smooth(fn, p, r, s=0.6, kind="smooth", eps=None):
    """SMOOTH the field toward its local average inside the ball (Laplacian smoothing of the field). The local
    average is the mean of the field at +/- `eps` along each axis (6 samples); `s` in [0,1] is the blend strength."""
    p = np.asarray(p, float)
    h = float(eps) if eps is not None else 0.15 * r
    offs = np.array([[h, 0, 0], [-h, 0, 0], [0, h, 0], [0, -h, 0], [0, 0, h], [0, 0, -h]], float)

    def g(P):
        P = np.asarray(P, float)
        base = fn(P)
        avg = sum(fn(P + o) for o in offs) / len(offs)     # local mean -> the field's smoothed value
        w = _ball_weights(P, p, r, kind)
        return base + w * s * (avg - base)
    return g


def brush_grab(fn, p, r, drag, kind="smooth"):
    """DRAG the field's domain by the vector `drag` inside the ball -- pulls the surface along. Points are displaced
    by -w*drag before sampling, so the feature at the centre moves by `drag`."""
    p = np.asarray(p, float)
    drag = np.asarray(drag, float)
    return lambda P: fn(np.asarray(P, float) - _ball_weights(P, p, r, kind)[:, None] * drag)


def brush_flatten(fn, p, r, level, s=0.6, kind="smooth"):
    """PULL the field toward the target `level` inside the ball -- flattens the surface toward a plane value."""
    p = np.asarray(p, float)
    return lambda P: fn(P) + _ball_weights(P, p, r, kind) * s * (level - fn(P))


def brush_pinch(fn, p, r, s=0.5, kind="smooth"):
    """DRAG the domain toward the brush centre inside the ball -- cinches the surface in (the opposite domain move
    to grab-away)."""
    p = np.asarray(p, float)

    def g(P):
        P = np.asarray(P, float)
        w = _ball_weights(P, p, r, kind)
        return fn(P + (w * s)[:, None] * (p - P))
    return g


_BRUSHES = {
    "inflate": brush_inflate, "carve": brush_carve, "smooth": brush_smooth,
    "grab": brush_grab, "flatten": brush_flatten, "pinch": brush_pinch,
}


def apply_brush(fn, kind, p, r, s=0.3, **kw):
    """Dispatch a sculpt brush by name onto the field function `fn`, returning the edited field function. `kind` is
    one of inflate, carve, smooth, grab, flatten, pinch. `grab` needs drag=..., `flatten` needs level=...; `s` is the
    strength (ignored by grab, which uses drag)."""
    if kind not in _BRUSHES:
        raise ValueError(f"unknown brush {kind!r}; choose from {sorted(_BRUSHES)}")
    if kind == "grab":
        return brush_grab(fn, p, r, kw.pop("drag"), kind=kw.pop("kind", "smooth"))
    if kind == "flatten":
        return brush_flatten(fn, p, r, kw.pop("level"), s=s, kind=kw.pop("kind", "smooth"))
    return _BRUSHES[kind](fn, p, r, s=s, **kw)


# =====================================================================================================
# Self-test -- local edits, correct sign, watertight re-extract (on a metaball field).
# =====================================================================================================
def _selftest():
    from holographic_meshbridge import metaball_field, sample_field, marching_tetrahedra

    centers = np.array([[0.0, 0.0, 0.0]])
    fn = metaball_field(centers, radius=0.4)               # high INSIDE the blob; surface (level 0.5) at d~0.47
    bounds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))         # (min corner, max corner)
    res = 28
    level = 0.5
    p = np.array([0.0, 0.0, 0.0])                          # brush at the blob centre
    r = 1.0                                                # radius spans the surface, so the falloff bites in the band

    # grid for counting "inside" cells (above the mesh level)
    (x0, y0, z0), (x1, y1, z1) = bounds
    xs = np.linspace(x0, x1, res); ys = np.linspace(y0, y1, res); zs = np.linspace(z0, z1, res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)

    def inside_count(g):
        return int((g(grid) > level).sum())

    base_in = inside_count(fn)

    # --- INFLATE grows; CARVE shrinks (correct sign in the band) ---
    inf = apply_brush(fn, "inflate", p, r, s=0.4)
    car = apply_brush(fn, "carve", p, r, s=0.4)
    assert inside_count(inf) > base_in, "inflate must GROW the surface"
    assert inside_count(car) < base_in, "carve must SHRINK the surface"

    # --- LOCAL: every brush leaves the field bit-identical OUTSIDE the ball ---
    far = grid[np.linalg.norm(grid - p, axis=1) > r + 1e-9]
    for kind, edited in (("inflate", inf), ("carve", car),
                         ("smooth", apply_brush(fn, "smooth", p, r, s=0.6)),
                         ("grab", apply_brush(fn, "grab", p, r, drag=np.array([0.2, 0.0, 0.0]))),
                         ("flatten", apply_brush(fn, "flatten", p, r, level=0.7, s=0.5)),
                         ("pinch", apply_brush(fn, "pinch", p, r, s=0.5))):
        d = np.max(np.abs(edited(far) - fn(far)))
        assert d < 1e-12, f"{kind} changed the field OUTSIDE the ball by {d:.2e} (must be local)"

    # --- STILL WATERTIGHT/MANIFOLD after a brush (re-extract) ---
    vals, axes = sample_field(inf, bounds, res)
    mesh = marching_tetrahedra(vals, axes, level=level)
    assert mesh.is_manifold(), "the re-extracted mesh after a brush must stay manifold"
    assert len(mesh.faces) > 0

    # --- WORKS ON ANY FIELD: a brush reshapes a value/density landscape the same way ---
    def value_field(P):                                    # a simple reward landscape: high near a 'good' point
        P = np.asarray(P, float)
        return np.exp(-np.sum((P - np.array([0.5, 0.5, 0.5])) ** 2, axis=1))
    shaped = apply_brush(value_field, "inflate", np.array([0.5, 0.5, 0.5]), 0.4, s=1.0)  # make that region MORE rewarding
    q = np.array([[0.5, 0.5, 0.5]])
    assert shaped(q)[0] > value_field(q)[0], "reward shaping: inflate must raise the value landscape locally"

    # --- determinism ---
    a = apply_brush(fn, "inflate", p, r, s=0.4)(grid)
    b = apply_brush(fn, "inflate", p, r, s=0.4)(grid)
    assert np.array_equal(a, b)

    print(f"holographic_sculpt selftest: ok (inflate grows {base_in}->{inside_count(inf)} cells, carve shrinks to "
          f"{inside_count(car)}; ALL six brushes change the field by <1e-12 outside the ball (local); re-extracted "
          f"mesh stays manifold with {len(mesh.faces)} faces; the same inflate brush reshapes a value landscape "
          f"(reward shaping); deterministic. One falloff-weighted operator, any field)")


if __name__ == "__main__":
    _selftest()

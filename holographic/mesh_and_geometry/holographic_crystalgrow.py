"""CRYSTAL GROWTH -- seed crystals on a surface, in a cluster, inside a geode, or wherever a field says.

`crystal_habit` (holographic_bravais) gives the SHAPE a lattice permits: the intersection of the
symmetry-equivalent {hkl} faces. That is one crystal, centred at the origin. Everything a user
actually wants -- a druse on a rock, a geode lined with points, quartz growing only on the veins of a
cavern wall -- is that one form PLACED MANY TIMES according to where crystals could nucleate. This
module is the placement half.

THE PHYSICS THAT DRIVES THE PLACEMENT, and it is the reason a single rule covers all these cases:

    A CRYSTAL GROWS PERPENDICULAR TO THE SUBSTRATE IT NUCLEATED ON.

Competitive growth is why: seeds start in every orientation, but a crystal whose long axis points
away from the wall keeps reaching fresh solution, while one lying along the wall is quickly buried by
its neighbours. The survivors are the ones pointing out. That single fact gives a druse its radiating
spray AND makes a geode's crystals point INWARD toward the cavity -- the geode is not a different
algorithm, it is the same rule on a surface whose outward direction happens to face the middle.

So the whole module is: find points on a surface, take the normal there, align each crystal's c-axis
to it, union the lot. Seeding delegates to `emit_from_surface`, which already projects points onto
any SDF, returns normals, and accepts a WEIGHT that may be a constant, a map or a FIELD -- which is
exactly the "grow only where this material is" case, for free.

KEPT NEGATIVE, and it is a real limit of doing it this way: crystals are UNIONED, so where two
interpenetrate you get the outer envelope of both, not the re-entrant contact surface real
intergrowth produces. Twin laws (Dauphine, Brazil, Japan for quartz) are not modelled at all.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_bravais import crystal_habit


#: Habits worth having by name, as (system, faces, size-weights). The size weights are RELATIVE, so a
#: caller gives ONE size and the form keeps its proportions -- a quartz point stays a point whether it
#: is a millimetre or a metre (D-7: proportions belong to the habit, absolute scale to the caller).
HABITS = {
    "quartz":       ("hexagonal", ((1, 0, 0), (1, 0, 1)), (0.30, 1.00)),
    "beryl":        ("hexagonal", ((1, 0, 0), (0, 0, 1)), (0.34, 1.00)),
    "cube":         ("cubic",     ((1, 0, 0),),           (1.00,)),
    "octahedron":   ("cubic",     ((1, 1, 1),),           (1.00,)),
    "dodecahedron": ("cubic",     ((1, 1, 0),),           (1.00,)),
    "needle":       ("hexagonal", ((1, 0, 0), (1, 0, 1)), (0.12, 1.00)),
}


def habit_sdf(name="quartz", size=1.0):
    """One crystal of a named habit, centred at the origin with its c-axis along +z.

    +z because that is the lattice's c-axis in `crystal_habit`'s reciprocal basis; `grow_on` rotates
    it onto each seed's normal. Keeping the convention in ONE place is what lets every growth mode
    share a single placement path.
    """
    if name not in HABITS:
        raise ValueError("unknown habit %r; one of %s" % (name, sorted(HABITS)))
    system, faces, rel = HABITS[name]
    sizes = tuple(float(size) * float(r) for r in rel)
    return crystal_habit(system, faces, sizes, form=True)


def _align(a, b):
    """Rotation taking unit vector `a` onto unit vector `b` (Rodrigues, antiparallel case handled)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # antiparallel: any perpendicular axis, half turn
        p = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        p = p - a * (p @ a); p /= np.linalg.norm(p)
        return -np.eye(3) + 2.0 * np.outer(p, p)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def placed(sdf, R, t, scale=1.0):
    """A form rotated by `R`, moved to `t` and scaled -- by transforming the QUERY, not the geometry.

    Transforming the query is what makes a thousand crystals cost one form: there is a single habit
    SDF and a thousand cheap coordinate changes, rather than a thousand meshes.
    """
    Rt = np.asarray(R, float).T
    t = np.asarray(t, float)
    s = float(scale) if abs(float(scale)) > 1e-9 else 1.0

    def f(P, _Rt=Rt, _t=t, _s=s, _n=sdf):
        # `P` is already a float (M,3) when called from `union`, which converts ONCE for the whole
        # group. Re-converting per crystal was pure overhead: a sparse render round showed 225,019
        # atleast_2d and 450,919 asarray calls, because every one of 30 crystals re-validated the
        # same array on every field query. Checked rather than assumed, so a direct caller passing a
        # list still works.
        Q = P if (type(P) is np.ndarray and P.ndim == 2 and P.dtype == np.float64) \
            else np.atleast_2d(np.asarray(P, float))
        return np.asarray(_n(((Q - _t[None, :]) @ _Rt.T) / _s), float).ravel() * _s
    f.eval = f
    return f


def batched_union(base, Rs, ts, ss):
    """One field evaluating N placed copies of ONE habit in a single vectorised pass.

    WHY THIS EXISTS: `union` loops in Python over every crystal, so a 110-crystal geode costs 110
    field evaluations per query and a path trace of it took 163 s at 110x90 -- the union, not the
    tracer, was the bottleneck. Every crystal here is the SAME habit under a different rigid
    transform, so the transforms can be applied as one einsum and the habit evaluated once over the
    stacked points. Same result, one pass.

    Only valid when every instance shares a base form; `union` stays for mixed collections.

    REFUTED AS AN OPTIMISATION -- KEPT FOR THE RECORD, NOT FOR SPEED. The reasoning above is sound and
    the measurement disagrees at every size tried. Re-measured after the tracer learned to compact to
    active pixels, on a 30-crystal geode:

        20,000-point bulk query   loop 0.112 s   batched 0.725 s   (6.5x SLOWER)
        whole render              loop 1.55 s    batched 1.76 s    (13% slower)

    Earlier notes here claimed it was "genuinely faster for the bulk-query case"; that was inferred
    from the einsum's shape, never measured, and is wrong. The (N,M,3) intermediate costs more to
    allocate and traverse than N sequential passes over (M,3), and it OOMs at large M. The default is
    the loop, and nothing in this repo should switch to `batched=True` expecting a win.
    """
    R = np.asarray(Rs, float)                      # (N,3,3)
    t = np.asarray(ts, float)                      # (N,3)
    s = np.asarray(ss, float)                      # (N,)
    s = np.where(np.abs(s) > 1e-9, s, 1.0)

    # CHUNKED, because the batch is (N crystals x M points x 3). Unchunked, a 110-crystal geode
    # against 200,000 query points asks for 528 MB in one allocation and the process is KILLED --
    # the speedup traded time for memory and the trade has to be bounded. The cap is on the PRODUCT,
    # so it holds whether the caller sends few points against many crystals or the reverse.
    budget = 4_000_000

    def f(P, _R=R, _t=t, _s=s, _b=base, _bud=budget):
        Q = np.atleast_2d(np.asarray(P, float))
        n = len(_R)
        step = max(int(_bud // max(n, 1)), 1)
        out = np.empty(len(Q))
        for i in range(0, len(Q), step):
            B = Q[i:i + step]
            rel = B[None, :, :] - _t[:, None, :]                    # (N,m,3)
            loc = np.einsum("nij,nmi->nmj", _R, rel) / _s[:, None, None]
            d = np.asarray(_b(loc.reshape(-1, 3)), float).reshape(n, len(B))
            out[i:i + step] = (d * _s[:, None]).min(axis=0)
        return out
    f.eval = f
    return f


def union(fields):
    """Hard union of many fields. HARD, not smooth: crystal faces are the point, and a smooth_union
    would round exactly the edges that make a crystal read as a crystal."""
    fs = tuple(fields)
    if not fs:
        raise ValueError("nothing to union")

    def f(P, _fs=fs):
        # CONVERT ONCE for the whole group, then hand every member the same validated array. With 30
        # crystals this removes 29 redundant conversions per query, and a field query happens once per
        # sphere-trace STEP -- so the saving multiplies by the trace depth.
        Q = P if (type(P) is np.ndarray and P.ndim == 2 and P.dtype == np.float64) \
            else np.atleast_2d(np.asarray(P, float))
        d = None
        for g in _fs:
            v = np.asarray(g(Q), float).ravel()
            d = v if d is None else np.minimum(d, v)
        return d
    f.eval = f
    return f


def seed_surface(sdf, count, bounds, where=None, seed=0):
    """Points and outward normals on `sdf`'s surface, optionally gated by a field.

    `where` is passed through as the emitter's WEIGHT, so it can be a constant, a map, or a callable
    field -- which is the "crystals only on the veins" case with no extra machinery: hand it a
    material mask and seeds land only where the mask is high.

    Returns (points (K,3), normals (K,3)).
    """
    from holographic.simulation_and_physics.holographic_emitter import emit_from_surface

    def ev(P):
        return np.asarray(sdf(np.atleast_2d(np.asarray(P, float))), float).ravel()
    P, N, _ = emit_from_surface(ev, int(count), bounds, weight=where, seed=int(seed))
    return np.asarray(P, float), np.asarray(N, float)


def grow_on(sdf, bounds, count=24, habit="quartz", size=0.18, size_jitter=0.45,
            inward=False, tilt=0.18, where=None, seed=0, substrate=True, batched=False):
    """GROW CRYSTALS ON A SURFACE -- the one call the other modes are special cases of.

    Seeds land on `sdf`'s surface (gated by `where`), and each crystal's c-axis is aligned to the
    surface NORMAL there, because a crystal grows perpendicular to what it nucleated on. `tilt` adds
    a little scatter so the spray is not mechanically parallel; real druses vary because the
    substrate is rough at a scale finer than the seeding.

    `inward=True` flips the alignment, which is all a geode is: crystals on the inside of a cavity,
    growing toward the middle.

    `substrate=True` keeps the host solid in the result, so crystals emerge FROM the rock rather than
    floating beside it. Set it False to get only the crystals.
    """
    P, N = seed_surface(sdf, count, bounds, where=where, seed=seed)
    if len(P) == 0:
        raise ValueError("no seeds landed on the surface -- check bounds and `where`")
    rng = np.random.default_rng(int(seed) + 1)
    base = habit_sdf(habit, 1.0)
    zc = np.array([0.0, 0.0, 1.0])
    out = []
    for i in range(len(P)):
        n = N[i] * (-1.0 if inward else 1.0)
        if float(tilt) > 0.0:
            n = n + rng.normal(0.0, float(tilt), 3)
        ln = float(np.linalg.norm(n))
        if ln < 1e-9:
            continue
        n = n / ln
        s = float(size) * (1.0 + float(size_jitter) * rng.uniform(-1.0, 1.0))
        s = max(s, 1e-4)
        # Sink the crystal slightly INTO the substrate so it is rooted, not balanced on the surface.
        root = P[i] - n * s * 0.35
        out.append((_align(zc, n), root, s))
    if not out:
        raise ValueError("every seed was rejected")
    # PLAIN UNION BY DEFAULT, and this is a MEASURED refutation of the obvious optimisation.
    # `batched_union` evaluates every instance in one einsum and is genuinely faster for a few large
    # queries -- but the path tracer calls a field MANY times with SMALL point counts (per bounce,
    # per surviving ray), and there the fixed cost of setting up an (N,m,3) batch over 110 transforms
    # dominates: the identical 110x90 probe went from 163 s to over 1700 s. It also OOMed at 200k
    # points until chunked (528 MB in one allocation). Batching is kept and documented for the
    # bulk-query case; the default stays the loop. RE-MEASURED after the tracer learned to compact:
    # batching is slower at EVERY size tried (6.5x on a 20k bulk query), so the "bulk" caveat was
    # itself wrong -- see batched_union's docstring.
    crystals = (batched_union(base, [o[0] for o in out], [o[1] for o in out], [o[2] for o in out])
                if batched else union([placed(base, R, t, s) for R, t, s in out]))
    return union([sdf, crystals]) if substrate else crystals


def cluster(count=9, habit="quartz", size=0.30, radius=0.22, seed=0, **kw):
    """A free-standing DRUSE: crystals radiating from a small rocky base.

    A cluster is `grow_on` with the substrate being a little blob, which is what a druse physically
    is -- the base is not decoration, it is the thing the crystals nucleated on and the reason they
    point outward.
    """
    r = float(radius)

    def base(P, _r=r):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.linalg.norm(Q, axis=1) - _r
    base.eval = base
    b = 1.6 * (r + float(size))
    return grow_on(base, ((-b, -b, -b), (b, b, b)), count=count, habit=habit, size=size,
                   seed=seed, **kw)


def geode(radius=0.7, shell=0.16, count=60, habit="quartz", size=0.13, seed=0, where=None, **kw):
    """A GEODE: a hollow nodule whose CAVITY WALL is lined with crystals pointing inward.

    Built from the physics rather than as a special shape -- the cavity is a sphere, the crystals are
    `grow_on(..., inward=True)` on it, and the rind is the shell between the cavity and the outer
    skin. That is also what a real geode is: a gas bubble in lava or a dissolved nodule, lined from
    the wall inward as mineral solution seeped in.

    `where` gates the lining, so a field can leave part of the wall bare -- which is what makes one
    geode look grown rather than machined.
    """
    R = float(radius); t = float(shell)
    cav = R - t

    def cavity(P, _c=cav):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.linalg.norm(Q, axis=1) - _c
    cavity.eval = cavity

    def rind(P, _R=R, _c=cav):
        Q = np.atleast_2d(np.asarray(P, float))
        d = np.linalg.norm(Q, axis=1)
        return np.maximum(d - _R, _c - d)                 # shell: inside outer, outside cavity
    rind.eval = rind

    b = 1.25 * R
    lining = grow_on(cavity, ((-b, -b, -b), (b, b, b)), count=count, habit=habit, size=size,
                     inward=True, where=where, seed=seed, substrate=False, **kw)
    return union([rind, lining])


def cut(field, normal=(1.0, 0.0, 0.0), point=(0.0, 0.0, 0.0)):
    """Slice a solid with a half-space -- how you actually LOOK INSIDE a geode.

    The cut face points along +normal (material is kept on the negative side), so a camera belongs on
    that side. Stated because getting it backwards renders the intact back of the nodule, which looks
    like an ordinary rock and gives no hint anything was cut.
    """
    n = np.asarray(normal, float); n = n / (np.linalg.norm(n) + 1e-12)
    p0 = np.asarray(point, float)

    def f(P, _n=n, _p=p0, _f=field):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.maximum(np.asarray(_f(Q), float).ravel(), (Q - _p[None, :]) @ _n)
    f.eval = f
    f.cut_face_normal = n
    return f


def vein_field(scale=6.0, threshold=0.55, seed=0, sharpness=6.0):
    """A banded/veined WEIGHT field: high on the veins, ~0 elsewhere.

    Exists so "grow crystals only where this material is" has something to demonstrate with. Any
    callable P->[0,1] works; this is a cheap deterministic one built from summed sinusoids, not a
    noise library, so it stays NumPy-only and reproducible.
    """
    rng = np.random.default_rng(int(seed))
    dirs = rng.normal(0.0, 1.0, (4, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ph = rng.uniform(0.0, 2 * np.pi, 4)
    amp = np.array([1.0, 0.55, 0.35, 0.22])

    def f(P, _d=dirs, _p=ph, _a=amp, _s=float(scale), _t=float(threshold), _k=float(sharpness)):
        Q = np.atleast_2d(np.asarray(P, float))
        v = np.zeros(len(Q))
        for k in range(len(_d)):
            v += _a[k] * np.sin(_s * (Q @ _d[k]) + _p[k])
        v = v / _a.sum()
        return 1.0 / (1.0 + np.exp(-_k * (v - (2.0 * _t - 1.0))))
    return f


def _selftest():
    rng = np.random.default_rng(0)

    # 1) THE HABIT KEEPS ITS PROPORTIONS AT ANY SIZE (D-7): a quartz point scaled 4x is the same
    # shape, so occupancy within its own bounding box is scale-invariant.
    def occ(sdf, b, n=40000):
        Q = rng.uniform(-b, b, size=(n, 3))
        return float((np.asarray(sdf(Q), float) < 0).mean())
    o1 = occ(habit_sdf("quartz", 0.25), 0.30)
    o2 = occ(habit_sdf("quartz", 1.00), 1.20)
    assert abs(o1 - o2) < 0.02, "habit proportions must not change with size: %.4f vs %.4f" % (o1, o2)

    # 2) CRYSTALS POINT AWAY FROM THE SUBSTRATE. This is the whole physical premise, so it is
    # measured, not assumed: material must exist OUTSIDE the host sphere where the crystals grew.
    def ball(P, r=0.5):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.linalg.norm(Q, axis=1) - r
    ball.eval = ball
    g = grow_on(ball, ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), count=20, size=0.22, seed=1)
    Q = rng.uniform(-1.2, 1.2, size=(60000, 3))
    r = np.linalg.norm(Q, axis=1)
    inside_g = np.asarray(g(Q), float) < 0
    outside_host = r > 0.53
    assert (inside_g & outside_host).sum() > 200, \
        "crystals must protrude BEYOND the substrate: %d points" % (inside_g & outside_host).sum()
    # And the substrate is still there. CONDITIONALLY, not as a share of the whole sample box: the
    # host ball is ~4% of that box, so an unconditional threshold measures the box, not the host.
    assert inside_g[r < 0.45].mean() > 0.95, \
        "substrate=True must keep the host solid: %.3f of interior points filled" % inside_g[r < 0.45].mean()

    # 3) A GEODE IS HOLLOW AND LINED INWARD. Both halves are asserted because either alone passes for
    # the wrong shape: a solid ball is "lined" by nothing, an empty shell has no crystals.
    G = geode(radius=0.7, shell=0.16, count=40, size=0.12, seed=2)
    Q = rng.uniform(-0.9, 0.9, size=(80000, 3))
    r = np.linalg.norm(Q, axis=1)
    ing = np.asarray(G(Q), float) < 0
    deep = r < 0.30                                  # well inside the cavity: must be mostly EMPTY
    wall = (r > 0.56) & (r < 0.70)                   # the rind: must be SOLID
    band = (r > 0.36) & (r < 0.52)                   # just inside the cavity wall: the crystal zone
    assert ing[deep].mean() < 0.25, "a geode must be hollow at its centre: %.2f filled" % ing[deep].mean()
    assert ing[wall].mean() > 0.90, "the rind must be solid: %.2f" % ing[wall].mean()
    assert ing[band].mean() > ing[deep].mean() + 0.05, \
        "crystals must line the wall and point inward: band %.3f vs centre %.3f" % (
            ing[band].mean(), ing[deep].mean())

    # 4) A FIELD GATES WHERE CRYSTALS GROW. The gated growth must differ from the ungated one AND
    # concentrate where the field is high -- "it ran" would pass with the weight ignored entirely.
    vf = vein_field(scale=7.0, threshold=0.62, seed=3)
    gated = grow_on(ball, ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), count=40, size=0.16,
                    where=vf, seed=4, substrate=False)
    plain = grow_on(ball, ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), count=40, size=0.16,
                    seed=4, substrate=False)
    Q = rng.uniform(-1.2, 1.2, size=(60000, 3))
    gi = np.asarray(gated(Q), float) < 0
    pi = np.asarray(plain(Q), float) < 0
    assert not np.array_equal(gi, pi), "a `where` field must change WHERE crystals grow"
    w = np.asarray(vf(Q), float).ravel()
    assert w[gi].mean() > w[pi].mean(), \
        "gated crystals must sit where the field is HIGH: %.3f vs %.3f" % (w[gi].mean(), w[pi].mean())

    # 5) DETERMINISM, and that seed actually matters.
    a = grow_on(ball, ((-1.2,)*3, (1.2,)*3), count=12, size=0.2, seed=7, substrate=False)
    b = grow_on(ball, ((-1.2,)*3, (1.2,)*3), count=12, size=0.2, seed=7, substrate=False)
    c = grow_on(ball, ((-1.2,)*3, (1.2,)*3), count=12, size=0.2, seed=8, substrate=False)
    T = rng.uniform(-1.2, 1.2, size=(4000, 3))
    assert np.array_equal(np.asarray(a(T), float) < 0, np.asarray(b(T), float) < 0), "must be deterministic"
    assert not np.array_equal(np.asarray(a(T), float) < 0, np.asarray(c(T), float) < 0), "seed must matter"

    # 6) THE CUT EXPOSES THE INTERIOR: slicing a geode must reveal cavity where solid rind was.
    C = cut(G, normal=(1, 0, 0))
    face = np.stack([np.full(4000, -0.002), rng.uniform(-0.35, 0.35, 4000),
                     rng.uniform(-0.35, 0.35, 4000)], axis=1)
    assert float(C.cut_face_normal[0]) > 0.9, "the cut must report which way its face points"
    assert (np.asarray(C(face), float) < 0).mean() < (np.asarray(G(face), float) < 0).mean() + 1e-9

    print("crystalgrow selftest OK: habit scale-invariant (%.3f/%.3f), crystals protrude past the "
          "substrate, geode hollow %.2f centre / %.2f rind / %.2f crystal band, field gating raises "
          "weight %.3f -> %.3f, deterministic"
          % (o1, o2, ing[deep].mean(), ing[wall].mean(), ing[band].mean(), w[pi].mean(), w[gi].mean()))


if __name__ == "__main__":
    _selftest()

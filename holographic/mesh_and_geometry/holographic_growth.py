"""Staged growth + scrubbing (organics backlog G-1): every grower exposes a progress axis t in [0,1].

WHY THIS MODULE EXISTS
----------------------
You cannot verify growth you cannot step through. The audit found the PLAYBACK half fully shipped and
wired -- `timeline` (keyframes), `transport` (play/scrub), `frame_cache` (sparse tiered DELTA cache,
built precisely for scrub playback), `render_animation` (frames -> GIF). The gap was entirely on the
GENERATOR side: every grower in the engine returns only its END state and discards the intermediates
it already computed. So this module adds no playback machinery at all -- it adds the missing staging
contract and hands the stages to the shipped players.

THE CONTRACT
    grow_stages(kind, spec, n_stages) -> [stage_0 ... stage_n]     discrete checkpoints
    grow_at(kind, spec, t)            -> stage                     continuous t in [0,1]

Four growers, each staged along the axis it ALREADY steps along internally:
  * "plant"   L-system: stage = iteration count. Continuous t interpolates BETWEEN iterations by
              scaling the newest generation's segments from 0 -> full length (buds extend, then branch).
  * "tree"    space colonization (T-1): the algorithm IS a step loop; a stage is a PREFIX of the
              growth-ordered segment list. Free -- the order is the algorithm's own output.
              (Now live: holographic_tree3d, registered below.)
  * "crystal" Bravais lattice: reveal sites in NUCLEATION ORDER (distance from a seed, hashlib ties),
              the physically honest picture of accretion outward from a nucleus.
  * "dendrite" DLA / dielectric breakdown: threshold the sim's OWN per-cell growth-order field.
              The shipped DielectricBreakdown already records `.order` ("for animating / age", says
              its comment) -- so staging is one run plus a comparison, not n re-runs. This replaced an
              earlier checkpointing version of this function; see _dendrite_order for the note.

VERIFICATION IS THE POINT (assert, do not eyeball)
    `growth_report` checks the two properties that make a scrub trustworthy, and RETURNS them as data:
      * PURITY      grow_at(spec, t) is a pure function of t -- same t, same bytes, no matter what was
                    scrubbed before. A grower with hidden playback state fails here.
      * MONOTONE    stage k's geometry is contained in stage k+1's (nothing retracts or teleports).
    Measured, not assumed: monotonicity is a property of the RULE, not of staging in general. An
    L-system whose production shortens its step, or a rule that reorients the axiom, legitimately
    breaks containment -- so the report MEASURES it per spec and reports `monotone: False` rather than
    asserting a universal that is not true. A False here is information, not necessarily a bug.

KEPT NEGATIVES (loud)
  * t is GROWTH PROGRESS, not physical time. No claim that equal steps in t are equal steps in
    seconds, sap flow, or crystal accretion rate. Do not label a scrub axis "time" in a UI.
  * The plant grower's continuous interpolation scales the NEWEST generation only. Interior branches
    do not thicken with age (no secondary growth / cambium). Structural preview, not botany.
  * `grow_stages` recomputes each stage from the spec rather than incrementally advancing one state.
    That is O(n_stages x cost) and deliberately so: recompute is what makes PURITY true. If a grower
    ever becomes expensive enough that this hurts, cache the stages -- do not make the growers stateful.
"""

import numpy as np


def _stage_points(stage):
    """The (N,3) point set that represents a stage, whatever shape the grower returned it in. One
    place, so every checker below compares growers on equal terms: segments -> their endpoints,
    a point set -> itself, a mask -> the coordinates of its set cells."""
    if isinstance(stage, dict):
        for k in ("points", "sites", "segments", "cluster"):
            if k in stage:
                stage = stage[k]
                break
    if isinstance(stage, np.ndarray) and stage.ndim == 2 and stage.shape[1] == 3:
        return stage
    if isinstance(stage, np.ndarray) and stage.dtype == bool:
        idx = np.argwhere(stage)                              # a DLA cluster mask -> occupied cells
        return np.column_stack([idx, np.zeros(len(idx))]) if idx.shape[1] == 2 else idx.astype(float)
    segs = list(stage)
    if not segs:
        return np.zeros((0, 3))
    return np.asarray([np.asarray(s[1], float) for s in segs], float)   # segment endpoints


def grow_plant_stages(spec, n_stages=None):
    """L-system plant growth, staged by iteration with CONTINUOUS interpolation between iterations.

    `spec`: {axiom, productions, iterations, angle_deg, step, stochastic, rng_seed}. Returns a list of
    segment lists. Reuses the shipped holographic_grammar.LSystem / turtle_to_segments -- no rewriting
    of the turtle, which is why a staged plant is guaranteed to match the unstaged one at t=1.
    """
    from holographic.agents_and_reasoning.holographic_grammar import LSystem, turtle_to_segments
    spec = dict(spec)
    iters = int(spec.get("iterations", 3))
    ls = LSystem(spec.get("axiom", "F"), spec.get("productions", {"F": "F[+F]F[-F]F"}),
                 stochastic=spec.get("stochastic"), rng_seed=int(spec.get("rng_seed", 0)))
    kw = {"angle_deg": float(spec.get("angle_deg", 25.0)), "step": float(spec.get("step", 1.0))}
    n_stages = iters if n_stages is None else int(n_stages)
    out = []
    for k in range(n_stages + 1):
        it = int(round(k * iters / max(n_stages, 1)))
        out.append(turtle_to_segments(ls.expand(it), **kw))
    return out


def grow_plant_at(spec, t):
    """A plant at continuous progress t in [0,1]: the last completed iteration, plus the NEXT
    generation's new segments scaled from their start point by the fractional part. So a bud visibly
    extends before it branches, instead of a whole generation popping into existence."""
    from holographic.agents_and_reasoning.holographic_grammar import LSystem, turtle_to_segments
    spec = dict(spec)
    iters = int(spec.get("iterations", 3))
    ls = LSystem(spec.get("axiom", "F"), spec.get("productions", {"F": "F[+F]F[-F]F"}),
                 stochastic=spec.get("stochastic"), rng_seed=int(spec.get("rng_seed", 0)))
    kw = {"angle_deg": float(spec.get("angle_deg", 25.0)), "step": float(spec.get("step", 1.0))}
    t = float(np.clip(t, 0.0, 1.0))
    x = t * iters
    lo = int(np.floor(x + 1e-12)); frac = x - lo
    base = turtle_to_segments(ls.expand(min(lo, iters)), **kw)
    if lo >= iters or frac <= 1e-12:
        return base
    nxt = turtle_to_segments(ls.expand(lo + 1), **kw)
    old = {_seg_key(s) for s in base}
    grown = []
    for s in nxt:
        if _seg_key(s) in old:
            grown.append(s)                                   # an existing branch: unchanged
        else:
            a, b = np.asarray(s[0], float), np.asarray(s[1], float)
            grown.append((a, a + (b - a) * frac))             # a new bud: extending
    return grown


def _seg_key(s):
    """A rounded, hashable identity for a segment -- so 'is this the same branch as last generation'
    is a set membership test rather than an O(n^2) float comparison."""
    return (tuple(np.round(np.asarray(s[0], float), 6)), tuple(np.round(np.asarray(s[1], float), 6)))


def grow_crystal_stages(spec, n_stages=8):
    """Crystal growth staged by NUCLEATION ORDER: stage k reveals the first k/n of the lattice sites,
    sorted by distance from the nucleus. Reuses holographic_bravais entirely (lattice_points +
    nucleation_order); this function only slices."""
    from holographic.mesh_and_geometry.holographic_bravais import lattice_basis, lattice_points, nucleation_order
    spec = dict(spec)
    basis, motif = lattice_basis(spec.get("system", "cubic"), a=float(spec.get("a", 1.0)),
                                 b=spec.get("b"), c=spec.get("c"),
                                 alpha=float(spec.get("alpha", 90.0)), beta=float(spec.get("beta", 90.0)),
                                 gamma=float(spec.get("gamma", 90.0)), centring=spec.get("centring", "P"))
    pts = lattice_points(basis, motif, extent=int(spec.get("extent", 2)))
    order = nucleation_order(pts, seed=int(spec.get("seed", 0)))
    n = len(order)
    return [pts[order[:int(round(k * n / max(n_stages, 1)))]] for k in range(int(n_stages) + 1)]


def grow_crystal_at(spec, t):
    """A crystal at progress t: the first t*N sites in nucleation order. Pure -- no dependence on
    what was scrubbed before, which is the property `growth_report` verifies."""
    from holographic.mesh_and_geometry.holographic_bravais import lattice_basis, lattice_points, nucleation_order
    spec = dict(spec)
    basis, motif = lattice_basis(spec.get("system", "cubic"), a=float(spec.get("a", 1.0)),
                                 b=spec.get("b"), c=spec.get("c"),
                                 alpha=float(spec.get("alpha", 90.0)), beta=float(spec.get("beta", 90.0)),
                                 gamma=float(spec.get("gamma", 90.0)), centring=spec.get("centring", "P"))
    pts = lattice_points(basis, motif, extent=int(spec.get("extent", 2)))
    order = nucleation_order(pts, seed=int(spec.get("seed", 0)))
    return pts[order[:int(round(float(np.clip(t, 0, 1)) * len(order)))]]


def _tree_stages(spec, n_stages=8):
    """Space-colonization tree growth, staged. Delegates to holographic_tree3d, which emits its
    segments in growth order -- so a stage is a PREFIX and staging costs nothing extra."""
    import holographic.mesh_and_geometry.holographic_tree3d as _t3
    return _t3.tree_stages(spec, n_stages)


def _tree_at(spec, t):
    """A tree at continuous progress t. See holographic_tree3d.tree_at."""
    import holographic.mesh_and_geometry.holographic_tree3d as _t3
    return _t3.tree_at(spec, t)


def _dendrite_order(spec):
    """Run the shipped DLA/dielectric-breakdown sim ONCE and hand back its per-cell growth ORDER field.

    REUSE FOUND BY AUDIT, not by writing more code: DielectricBreakdown already records `.order` (the
    step index at which each cell joined the cluster) and its own comment says it is "for animating /
    age". An earlier version of this module CHECKPOINTED the sim -- re-running it n_stages times and
    copying the mask -- which was both slower and less exact. The order field gives every stage from a
    single run by thresholding, so staging costs one sim, not n.
    """
    from holographic.misc.holographic_dendrite import DielectricBreakdown
    spec = dict(spec)
    shape = tuple(spec.get("shape", (61, 61)))
    dbm = DielectricBreakdown(shape, eta=float(spec.get("eta", 1.0)), seed=int(spec.get("seed", 0)))
    dbm.seed_point(*spec.get("seed_point", (shape[0] // 2, shape[1] // 2)))
    dbm.set_source_border()
    dbm.grow(int(spec.get("steps", 120)))
    return dbm.order


def grow_dendrite_stages(spec, n_stages=8):
    """DLA / dielectric-breakdown growth (ice, frost, lightning), staged by THRESHOLDING the sim's own
    growth-order field. Returns a list of boolean cluster masks. The sim itself is untouched."""
    order = _dendrite_order(spec)
    top = int(order.max())
    return [(order >= 0) & (order <= int(round(k * top / max(int(n_stages), 1))))
            for k in range(int(n_stages) + 1)]


def grow_dendrite_at(spec, t):
    """A dendrite at continuous progress t: every cell whose growth order is within the first t of the
    run. Continuous and exact, because the order field is per-cell rather than per-checkpoint."""
    order = _dendrite_order(spec)
    k = int(round(float(np.clip(t, 0.0, 1.0)) * int(order.max())))
    return (order >= 0) & (order <= k)


#: The staged growers, by kind. A dict rather than an if-chain so `grow_kinds()` can list them and a
#: new grower is one entry, not a new branch in three functions.
_GROWERS = {
    "plant":    (grow_plant_stages, grow_plant_at),
    "crystal":  (grow_crystal_stages, grow_crystal_at),
    "tree":     (_tree_stages, _tree_at),
    "dendrite": (grow_dendrite_stages, grow_dendrite_at),
}


def variant(spec, seed, jitter=0.25, keys=None):
    """T-3: a deterministic VARIATION of a grower spec -- the permutations a scattered field needs.

    Perturbs the numeric parameters of `spec` (or just `keys`) by up to +-jitter, fractionally, from a
    hashlib-derived stream so variant 7 of a spec is the same everywhere, forever, with nothing stored.
    Integer fields stay integers (an L-system cannot run 3.4 iterations) and stay >= 1.

    WHY SPEC-LEVEL AND NOT MESH-LEVEL: `realize_scatter` already draws from a POOL of finished meshes,
    which gives variety but only among meshes you already built. Varying the SPEC means the pool is
    generated, so "20 different ferns" is a loop over seeds rather than 20 authored assets -- and
    because it is a pure function of (spec, seed), the pool never has to be stored either. That is the
    determinism-instead-of-storage lever, applied to art assets.
    """
    import hashlib
    out = dict(spec)
    for k in sorted(out) if keys is None else list(keys):
        v = out.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        h = hashlib.sha256(("variant:%d:%s" % (int(seed), k)).encode()).digest()
        u = int.from_bytes(h[:8], "little") / 2 ** 64                      # deterministic u in [0,1)
        f = 1.0 + float(jitter) * (2.0 * u - 1.0)
        out[k] = max(1, int(round(v * f))) if isinstance(v, int) else v * f
    return out


def variant_pool(spec, n, jitter=0.25, keys=None, base_seed=0):
    """`n` distinct variants of a spec -- the generated pool that feeds realize_scatter's `variants`.
    Variant 0 is the UNCHANGED spec, so a pool always contains the thing the user actually authored."""
    return [dict(spec)] + [variant(spec, base_seed + i, jitter, keys) for i in range(1, int(n))]


def grow_kinds():
    """The growers that support staging -- so a caller (or an agent) can discover the axis names
    instead of guessing them."""
    return sorted(_GROWERS)


def grow_stages(kind, spec, n_stages=8):
    """Discrete growth checkpoints for `kind` (see grow_kinds()): a list of n_stages+1 stages, from
    nothing to the finished form. Feed straight to the shipped frame_cache / timeline for scrubbing --
    growth is append-only, so consecutive-stage deltas are maximally sparse, which is exactly the
    shape frame_cache's delta encoding wants."""
    if kind not in _GROWERS:
        raise ValueError("unknown grower %r; one of %s" % (kind, grow_kinds()))
    return _GROWERS[kind][0](spec, n_stages)


def grow_at(kind, spec, t):
    """The stage at continuous progress t in [0,1]. PURE: the same (kind, spec, t) always returns the
    same bytes, independent of any earlier call -- so scrubbing backwards is safe and a UI slider
    needs no reset. Growers without a continuous form fall back to the nearest discrete stage."""
    if kind not in _GROWERS:
        raise ValueError("unknown grower %r; one of %s" % (kind, grow_kinds()))
    stages_fn, at_fn = _GROWERS[kind]
    if at_fn is not None:
        return at_fn(spec, t)
    stages = stages_fn(spec, 8)                               # no continuous form: nearest checkpoint
    return stages[int(round(float(np.clip(t, 0, 1)) * (len(stages) - 1)))]


def growth_report(kind, spec, n_stages=6, tol=1e-6):
    """THE VERIFICATION TOOL -- is this growth being done correctly? Returns a dict:

        purity      grow_at(t) called twice (and out of order) gives byte-identical results
        monotone    every stage's points are contained in the next stage's (nothing retracts)
        counts      points per stage -- should be non-decreasing
        first_break the stage index where containment first failed, or None

    MEASURED, NOT ASSUMED: monotone=False is a legitimate answer for a rule that rescales or
    reorients as it grows, and this reports it rather than asserting a universal that is not true.
    What it does catch, hard, is the two real bugs: hidden playback state (purity False) and
    geometry that vanishes mid-growth (monotone False with a shrinking count).
    """
    stages = grow_stages(kind, spec, n_stages)
    pts = [_stage_points(s) for s in stages]
    counts = [len(p) for p in pts]

    # PURITY: probe t out of order, twice -- a stateful grower gives itself away here.
    ts = [0.5, 0.2, 1.0, 0.2, 0.5]
    seen, purity = {}, True
    for t in ts:
        a = _stage_points(grow_at(kind, spec, t))
        if t in seen:
            purity &= (a.shape == seen[t].shape and np.allclose(a, seen[t], atol=0, rtol=0))
        seen[t] = a

    # MONOTONE: set containment on rounded coordinates -- the honest test of "nothing retracted".
    monotone, first_break = True, None
    for k in range(len(pts) - 1):
        A = {tuple(np.round(p, 6)) for p in pts[k]}
        B = {tuple(np.round(p, 6)) for p in pts[k + 1]}
        if not A <= B:
            monotone = False
            if first_break is None:
                first_break = k
    return {"kind": kind, "purity": bool(purity), "monotone": bool(monotone), "counts": counts,
            "first_break": first_break, "n_stages": int(n_stages),
            "non_decreasing": bool(all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)))}


def _selftest():
    """Numeric contracts: staging must END where the unstaged grower ends, purity must hold for all
    three growers, crystal growth must be strictly monotone, and the report must CATCH a violation."""
    # 1) PLANT: staged growth ends exactly where the shipped grow_plant ends (staging changed nothing).
    from holographic.agents_and_reasoning.holographic_grammar import LSystem, turtle_to_segments
    spec = {"axiom": "F", "productions": {"F": "F[+F]F[-F]F"}, "iterations": 3}
    st = grow_stages("plant", spec, 3)
    ref = turtle_to_segments(LSystem("F", {"F": "F[+F]F[-F]F"}).expand(3))
    assert len(st[-1]) == len(ref) == 125, "final stage must equal the unstaged grower: %d vs %d" % (
        len(st[-1]), len(ref))
    assert len(st[0]) == 1, "stage 0 is the axiom"

    # 2) PLANT continuous: a bud must be SHORTER mid-interpolation and full at the boundary.
    mid = grow_plant_at(spec, 0.5 / 3)                        # halfway into the first iteration
    full = grow_plant_at(spec, 1.0 / 3)
    lm = sum(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) for a, b in mid)
    lf = sum(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) for a, b in full)
    assert lm < lf, "a growing bud must be shorter than a finished one (%.3f vs %.3f)" % (lm, lf)
    assert len(grow_plant_at(spec, 1.0)) == 125, "t=1 must be the finished plant"

    # 3) CRYSTAL: strictly monotone accretion, exact counts, ends at the full lattice.
    cs = {"system": "cubic", "centring": "F", "extent": 1}
    rep = growth_report("crystal", cs, n_stages=5)
    assert rep["purity"], "crystal growth must be pure"
    assert rep["monotone"], "crystal accretion must never retract: broke at %s" % rep["first_break"]
    assert rep["non_decreasing"] and rep["counts"][0] == 0
    assert rep["counts"][-1] == 4 * 27, "F-centred 3x3x3 cells = 108 sites, got %d" % rep["counts"][-1]

    # 4) PLANT report: purity must hold. Monotonicity is MEASURED, not assumed -- record what it says.
    prep = growth_report("plant", spec, n_stages=3)
    assert prep["purity"], "plant growth must be pure"
    assert prep["non_decreasing"], "plant segment count must not shrink"

    # 5) THE REPORT MUST CATCH A REAL VIOLATION -- otherwise it is decoration. A rule that HALVES its
    #    step each generation genuinely retracts (old segments move), and monotone must go False.
    shrink = {"axiom": "F", "productions": {"F": "FF"}, "iterations": 2, "step": 1.0}
    _orig = _GROWERS["plant"]
    def _shrinking_stages(sp, n):                             # a deliberately non-monotone grower
        return [turtle_to_segments(LSystem("F", {"F": "FF"}).expand(k), step=1.0 / (k + 1))
                for k in range(int(n) + 1)]
    _GROWERS["plant"] = (_shrinking_stages, _orig[1])
    try:
        bad = growth_report("plant", shrink, n_stages=2)
        assert not bad["monotone"], "the report FAILED to catch a retracting grower -- it is decoration"
        assert bad["first_break"] is not None
    finally:
        _GROWERS["plant"] = _orig                             # never leave the registry mutated

    # 6) grow_at clamps and never raises outside [0,1].
    assert len(_stage_points(grow_at("crystal", cs, -5.0))) == 0
    assert len(_stage_points(grow_at("crystal", cs, 99.0))) == 108

    # 7) VARIANTS (T-3): deterministic, actually different, structurally still valid, ints stay ints.
    vs = variant_pool({"axiom": "F", "productions": {"F": "F[+F]F[-F]F"}, "iterations": 3,
                       "angle_deg": 25.0, "step": 1.0}, 6, jitter=0.3)
    assert len(vs) == 6 and vs[0]["iterations"] == 3, "variant 0 must be the authored spec"
    assert vs[1] == variant({"axiom": "F", "productions": {"F": "F[+F]F[-F]F"}, "iterations": 3,
                             "angle_deg": 25.0, "step": 1.0}, 1, 0.3), "variants must be pure"
    angles = {round(v["angle_deg"], 6) for v in vs}
    assert len(angles) >= 5, "a pool of 6 must actually differ, got %d distinct angles" % len(angles)
    assert all(isinstance(v["iterations"], int) and v["iterations"] >= 1 for v in vs), \
        "an L-system cannot run a fractional number of iterations"
    for v in vs:                                              # every variant must still GROW
        assert len(grow_stages("plant", v, 1)[-1]) > 0

    print("growth selftest OK: plant 1->125 staged, crystal 0->108 monotone accretion, purity held "
          "for both, %d distinct variants, and the report CAUGHT a deliberately retracting grower"
          % len(angles))


if __name__ == "__main__":
    _selftest()

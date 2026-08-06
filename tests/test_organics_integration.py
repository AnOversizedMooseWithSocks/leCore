"""Cross-faculty integration tests for the ORGANICS backlog (crystals / grass / plants / growth / idle).

WHY THESE ARE INTEGRATION TESTS AND NOT MORE SELFTESTS
------------------------------------------------------
Each module's `_selftest` already pins its own numeric contract. What those cannot catch is the failure
mode this repo has on record: a faculty that works alone and breaks when fed another faculty's output
(the denoiser fed a recall output, cosine 0.13 -> -0.06 -- a shared kernel is not a shared manifold).
So every test here CROSSES a boundary: lattice -> mesher, plant -> skinner, sampler -> instancer,
rig -> animator, generator -> scrubber.

They also pin the REUSE claims made in the backlog. If C-2 ("crystals need no new meshing code") or
T-2 ("plant limbs reuse the shipped B-Mesh") ever stops being true, these fail rather than the claim
quietly rotting in a document.
"""

import numpy as np
import pytest

import lecore
from holographic.mesh_and_geometry.holographic_mesh import Mesh


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


@pytest.fixture(scope="module")
def ground():
    """A flat 6x6 quad -- an exact, known surface area (36) so area-dependent claims are checkable."""
    V = np.array([[0., 0, 0], [6, 0, 0], [6, 6, 0], [0, 6, 0]])
    return Mesh(V, np.array([[0, 1, 2], [0, 2, 3]]))


# ----------------------------------------------------------------- crystals (C-1 / C-2 / C-3) --

def test_lattice_feeds_the_shipped_meshers(mind):
    """C-2's whole claim: a lattice needs NO new meshing code -- the point set drops into metaball_mesh
    and its bonds into sweep_tube. If either mesher stops accepting it, the claim is dead."""
    pts = mind.lattice_sites("cubic", centring="F", extent=1)
    assert len(pts) == 108
    bonds, dmin = mind.lattice_bonds(pts)
    assert abs(dmin - np.sqrt(0.5)) < 1e-9                   # FCC NN distance is a/sqrt(2)
    blob = mind.metaball_mesh(pts[:14], radius=0.35, resolution=24)
    assert len(np.asarray(blob.vertices)) > 0
    i, j = bonds[0]
    tv, tf = mind.sweep_tube([pts[i], pts[j]], radius=0.05)
    assert len(np.asarray(tv)) > 0 and len(np.asarray(tf)) > 0


def test_crystal_habit_marches_through_the_shipped_sdf_path(mind):
    """C-3: the faceted habit is an SDF, so mesh_from_sdf must mesh it with no special-casing."""
    sdf = mind.crystal_habit("cubic", [(1, 1, 1), (1, 0, 0)], [0.55, 0.5])
    assert sdf(np.zeros((1, 3)))[0] < 0                       # the origin is inside the crystal
    mesh = mind.mesh_from_sdf(sdf, bounds=((-1, -1, -1), (1, 1, 1)), res=24)
    assert len(np.asarray(mesh.vertices)) > 100


def test_only_fourteen_bravais_lattices(mind):
    """The premise correction, pinned: 7 systems, 14 lattices, and an illegal pairing is REFUSED
    rather than silently producing a lattice that does not exist."""
    systems = mind.crystal_systems()
    assert len(systems) == 7
    assert sum(len(cents) for _, cents in systems.values()) == 14
    with pytest.raises(ValueError):
        mind.lattice_basis("cubic", centring="C")             # not one of the 14


# --------------------------------------------------------------------- grass / scatter (S-1 / S-2) --

def test_scatter_lands_on_the_surface_and_conserves_geometry(mind, ground):
    """S-1 + S-2 across the boundary: every blade must sit ON the lawn, and merging must conserve
    counts exactly (n instances x the source's own vertex count -- no silent dropping or welding)."""
    blade = mind.grass_blade()
    nv = len(np.asarray(blade.vertices))
    out = mind.scatter_mesh(ground, blade, 200, seed=1)
    assert out["count"] == 200
    V = np.asarray(out["geometry"].vertices)
    assert len(V) == 200 * nv
    assert V[:, 2].min() >= -1e-9                            # nothing sank through the ground plane
    assert np.asarray(out["transforms"])[:, 2, 3].max() < 1e-9   # every ROOT is exactly on z=0


def test_instanced_mode_actually_shares_definitions(mind, ground):
    """The instancing claim is a MEMORY claim: n placements, one definition. If this ever becomes n
    definitions the mode is pointless and the kept negative about merge cost becomes the only option."""
    sc = mind.scatter_mesh(ground, mind.grass_blade(), 300, seed=2, mode="instanced")["geometry"]
    assert len(sc.instances) == 300
    assert len(sc.definitions()) == 1
    variants = [mind.grass_blade(height=0.2), mind.grass_blade(height=0.5), mind.grass_blade(height=0.35)]
    pv = mind.scatter_mesh(ground, None, 120, seed=3, variants=variants, mode="instanced")["geometry"]
    assert len(pv.definitions()) == 3 and len(pv.instances) == 120


def test_scatter_is_deterministic_and_density_is_respected(mind, ground):
    """Determinism is the engine's hard constraint; a lawn must be bit-identical across runs. And the
    density mask must EXCLUDE, not merely thin -- a half-plane mask leaves nothing on the far side."""
    a = mind.scatter_mesh(ground, mind.grass_blade(), 80, seed=7)
    b = mind.scatter_mesh(ground, mind.grass_blade(), 80, seed=7)
    assert np.array_equal(np.asarray(a["geometry"].vertices), np.asarray(b["geometry"].vertices))
    masked = mind.sample_mesh_surface(ground, 600, density=lambda P: (P[:, 0] < 3.0).astype(float), seed=4)
    assert masked["points"][:, 0].max() <= 3.0 + 1e-9


def test_scatter_accepts_both_mesh_shapes(mind, ground):
    """The engine hands meshes around as BOTH a Mesh object and a bare (V,F) tuple (sweep_tube returns
    the tuple, metaball_mesh returns the object). A scatter that took only one would break half its
    callers -- this pins that both work as a source."""
    tv, tf = mind.sweep_tube([[0., 0, 0], [0, 0, 0.4]], radius=0.03)
    out = mind.scatter_mesh(ground, (tv, tf), 25, seed=5)
    assert len(np.asarray(out["geometry"].vertices)) == 25 * len(np.asarray(tv))


# ---------------------------------------------------------------------- plants / trees (O-1 / T-2) --

def test_promoted_lsystem_faculties_are_callable_and_compose(mind):
    """O-1: these were shipped but had NO faculty, so they were undiscoverable and uncallable through
    the mind. Pin the whole promoted chain -- expand, turtle, scene, mesh."""
    ls = mind.lsystem("F", {"F": "F[+F]F[-F]F"})
    segs = mind.turtle_to_segments(ls.expand(2))
    assert len(segs) == 25
    scene = mind.segments_to_scene(segs, radius=0.03)
    assert scene is not None
    mesh, segs3, _ = mind.grow_plant(ls, 3)
    assert len(segs3) == 125 and len(np.asarray(mesh.vertices)) > 0


def test_plant_skeleton_reuses_the_shipped_skinner(mind):
    """T-2's claim: plant limbs need no new mesher -- the branch skeleton feeds skin_skeleton (B-Mesh)
    directly. This crosses grammar -> geometry, the exact boundary a selftest cannot see."""
    ls = mind.lsystem("F", {"F": "F[+F]F"})
    _, segs, _ = mind.grow_plant(ls, 2)
    nodes = sorted({tuple(np.round(p, 6)) for s in segs for p in s})
    idx = {p: i for i, p in enumerate(nodes)}
    edges = [(idx[tuple(np.round(a, 6))], idx[tuple(np.round(b, 6))]) for a, b in segs]
    skinned = mind.skin_skeleton(np.array(nodes), edges, np.full(len(nodes), 0.05), resolution=24)
    assert len(np.asarray(skinned.vertices)) > 0


# -------------------------------------------------------------------------- growth scrubbing (G-1) --

@pytest.mark.parametrize("kind,spec", [
    ("plant", {"axiom": "F", "productions": {"F": "F[+F]F[-F]F"}, "iterations": 3}),
    ("crystal", {"system": "cubic", "centring": "F", "extent": 1}),
    ("dendrite", {"shape": (41, 41), "steps": 40}),
])
def test_every_grower_is_pure_and_never_retracts(mind, kind, spec):
    """THE SCRUB CONTRACT, on every grower. purity = no hidden playback state (so scrubbing backwards
    is safe); monotone = nothing vanishes mid-growth. These are the two bugs a visual scrub is meant
    to reveal, asserted numerically so they are caught before anyone has to spot them by eye."""
    rep = mind.growth_report(kind, spec, n_stages=4)
    assert rep["purity"], "%s growth has hidden state" % kind
    assert rep["monotone"], "%s retracted at stage %s" % (kind, rep["first_break"])
    assert rep["non_decreasing"]


def test_grow_at_is_stable_under_a_backwards_scrub(mind):
    """The property a UI slider depends on: dragging forward then back must return the SAME bytes.
    A grower that advanced internal state would pass a forward-only test and fail here."""
    spec = {"system": "cubic", "centring": "F", "extent": 1}
    first = mind.grow_at("crystal", spec, 0.4)
    for t in (0.9, 0.1, 1.0, 0.0):                            # scrub around, out of order
        mind.grow_at("crystal", spec, t)
    assert np.array_equal(first, mind.grow_at("crystal", spec, 0.4))


def test_staged_growth_ends_where_the_unstaged_grower_ends(mind):
    """Staging must not CHANGE the thing being grown -- the final stage has to equal what the plain
    grower produces, or the scrub is showing a different plant than the one you ship."""
    spec = {"axiom": "F", "productions": {"F": "F[+F]F[-F]F"}, "iterations": 3}
    stages = mind.grow_stages("plant", spec, 3)
    ls = mind.lsystem("F", {"F": "F[+F]F[-F]F"})
    _, segs, _ = mind.grow_plant(ls, 3)
    assert len(stages[-1]) == len(segs) == 125
    assert len(stages[0]) == 1                                # stage 0 is the bare axiom


def test_growth_stages_are_scrubbable_through_the_shipped_frame_cache(mind):
    """G-1's reuse claim: growth is append-only, so consecutive stages are exactly the sparse deltas
    the shipped FrameCache wants. Crossing generator -> player is the point of the whole item."""
    stages = mind.grow_stages("crystal", {"system": "cubic", "centring": "P", "extent": 1}, 4)
    counts = [len(s) for s in stages]
    assert counts == sorted(counts) and counts[0] == 0
    cache = mind.frame_cache(np.asarray(stages[-1], float))
    for i, s in enumerate(stages):
        padded = np.zeros_like(np.asarray(stages[-1], float))
        padded[:len(s)] = np.asarray(s, float)
        cache.put(i, padded)
    assert np.allclose(cache.get(2)[:counts[2]], np.asarray(stages[2], float))


# ------------------------------------------------------------------- creature idle animation (R-10) --

def test_idle_never_violates_the_rig_it_came_from(mind):
    """R-10's central promise: the stored joint LIMIT is what drives the motion, so the animation
    physically cannot show a bend the rig forbids. Negative headroom would mean the idle lies about
    the rig -- the single most important assertion in this file."""
    c = mind.creature(mind.quadruped_spec(), skin=False)
    rep = mind.creature_idle_report(c, n_frames=16)
    assert rep["limit_headroom"] >= -1e-12
    assert rep["bone_length_error"] < 1e-9                    # rotation only: nothing may stretch
    assert rep["moved"] == 1.0                                # a motionless idle would show nothing
    assert min(rep["max_flex_deg"].values()) > 5.0            # and the flex must be readable


def test_idle_is_pure_and_does_not_mutate_the_creature(mind):
    """Scrubbing an idle back and forth must be safe, and asking for a pose must not silently repose
    the creature -- otherwise a preview would corrupt the asset it is previewing."""
    c = mind.creature(mind.quadruped_spec(), skin=False)
    rest = {k: np.asarray(v).copy() for k, v in c.joints.items()}
    a = mind.creature_idle(c, 0.3)
    mind.creature_idle(c, 0.9)
    b = mind.creature_idle(c, 0.3)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert all(np.allclose(rest[k], np.asarray(c.joints[k])) for k in rest)


def test_idle_cycle_loops_seamlessly(mind):
    """A looping preview with a visible seam is worse than no preview: t=0 and t=period must agree."""
    c = mind.creature(mind.quadruped_spec(), skin=False)
    p0 = mind.creature_idle(c, 0.0, period=2.0)
    pT = mind.creature_idle(c, 2.0, period=2.0)
    assert max(float(np.abs(p0[k] - pT[k]).max()) for k in p0) < 1e-9
    assert len(mind.creature_idle_frames(c, 12)) == 12


def test_idle_composes_with_the_posed_rig(mind):
    """Cross-faculty: idling a creature that has already been POSED by the IK solver must still respect
    bone lengths. The animator reads the rig's CURRENT state, so a posed rig is a different input than
    a rest rig -- exactly the shared-kernel-is-not-a-shared-manifold trap."""
    c = mind.creature(mind.quadruped_spec(), skin=False)
    name = sorted(c.chains)[0]
    tip = np.asarray(c.joints[c.chains[name][-1]], float)
    c.pose_limb(name, tip + np.array([0.05, 0.0, -0.05]), mind=mind)
    rep = mind.creature_idle_report(c, n_frames=8)
    assert rep["bone_length_error"] < 1e-9
    assert rep["limit_headroom"] >= -1e-12


# --------------------------------------------------------------------------- discoverability (step 4) --

@pytest.mark.parametrize("phrasing,expect", [
    ("cubic lattice", "Crystal lattices"),
    ("salt crystal structure", "Crystal lattices"),
    ("put grass on my terrain mesh", "Scatter meshes"),
    ("cover a surface in plants", "Scatter meshes"),
    ("make a bush", "Procedural plants"),
    ("vegetation generator", "Procedural plants"),
    ("watch it grow step by step", "Scrub through growth"),
    ("make the creature move a little", "Creature idle"),
    ("show me where the joints bend", "reature idle"),
])
def test_a_stranger_can_find_it(mind, phrasing, expect):
    """The governing rule of this repo: a capability find_capability cannot surface does not exist.
    EVERY phrasing here was MEASURED failing before these catalog entries were written -- they are the
    words that actually missed, not ones that sounded plausible to the implementer."""
    assert expect in str(mind.find_capability(phrasing)[:3]), \
        "%r no longer finds %s -- the capability has gone dark" % (phrasing, expect)


# =========================== SECOND PASS: trees (T-1/T-2/T-4), creature skin + spine (R-1/R-2) ===========================

def test_space_colonization_defaults_grow_a_finished_tree(mind):
    """The defaults must TERMINATE. The original ones did not -- step 0.08 / influence 0.55 / kill 0.14
    produced 2943 nodes, capped at max_iters, with 14 attractors never reached. The termination rule
    is roughly kill >= 2*step; this pins the fixed defaults so nobody re-tunes one of the three alone."""
    A = mind.crown_attractors(n=200, seed=1)
    t = mind.grow_tree(A)
    assert t["terminated"] == "attractors_consumed", "defaults must finish, got %s" % t["terminated"]
    assert t["consumed"] == t["n_attractors"]
    assert 50 < len(t["nodes"]) < 600


def test_tree_growth_order_makes_every_prefix_a_valid_tree(mind):
    """Why scrubbing a tree is free: segments come out in growth order, so a child never precedes its
    parent and any prefix is a connected younger tree."""
    t = mind.grow_tree(mind.crown_attractors(n=150, seed=2))
    seen = {0}
    for a, b in t["segments"]:
        assert a in seen, "segment %d->%d grows from a node that does not exist yet" % (a, b)
        seen.add(b)
    rep = mind.growth_report("tree", {"n_attractors": 120, "seed": 1}, n_stages=4)
    assert rep["purity"] and rep["monotone"] and rep["non_decreasing"]


def test_da_vinci_taper_holds_at_every_fork(mind):
    """T-2's allometry, exact: a parent's radius^2 equals the SUM of its children's radius^2. An
    earlier version seeded every node with the tip radius AND summed its children, double-counting
    at each fork (0.12024 vs 0.120204) -- caught only because this is asserted numerically."""
    t = mind.grow_tree(mind.crown_attractors(n=180, seed=3))
    r = mind.taper_radii(t, tip_radius=0.006, exponent=2.0)
    parent = np.asarray(t["parent"], int)
    kids = {}
    for i in range(1, len(parent)):
        kids.setdefault(int(parent[i]), []).append(i)
    forks = 0
    for p, ch in kids.items():
        if len(ch) >= 2:
            forks += 1
            assert abs(r[p] ** 2 - sum(r[c] ** 2 for c in ch)) < 1e-9
    assert forks > 0 and r[0] == r.max()


def test_tree_meshes_both_ways_within_their_measured_limits(mind):
    """T-2 reuse AND its ceiling. skin_skeleton nests one SDF node per edge and recurses, dying
    between 200 and 300 edges (measured: 199 OK, 299 RecursionError) -- so a full tree needs the
    per-branch sweep path. Both halves pinned: the small case blends, the big case scales."""
    from holographic.mesh_and_geometry.holographic_tree3d import tree_edges
    small = mind.grow_tree(mind.crown_attractors(n=25, seed=5), max_iters=40)
    assert len(tree_edges(small)) < 200
    blended = mind.skin_skeleton(small["nodes"], tree_edges(small), mind.taper_radii(small), resolution=20)
    assert len(np.asarray(blended.vertices)) > 0

    big = mind.grow_tree(mind.crown_attractors(n=250, seed=6))
    mesh = mind.tree_mesh(big)
    assert len(np.asarray(mesh.vertices)) == len(big["segments"]) * 12


def test_leaves_instance_onto_a_tree_through_the_scatter_keystone(mind):
    """T-4 crossing into S-2: phyllotaxis emits (m,4,4) frames, and realize_scatter turns them into
    geometry -- the SAME keystone that places grass. If these two ever stop composing, the 'build it
    once, four callers' claim is dead."""
    t = mind.grow_tree(mind.crown_attractors(n=200, seed=7))
    frames = mind.phyllotaxis_frames(t, per_node=1, size=0.05)
    assert len(frames) > 10
    R = frames[0, :3, :3] / 0.05
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9, "leaf frames must be orthonormal"
    leaf = mind.grass_blade(height=0.6, width=0.3, segments=2)
    out = mind.realize_scatter(leaf, frames[:40], mode="instanced")
    assert len(out.instances) == 40 and len(out.definitions()) == 1


# --------------------------------------------------------- creature skin + spine editing (R-1/R-2) --

def test_metaball_spacing_adds_balls_when_a_bone_stretches(mind):
    """R-1's auto-density property, which is what makes a spine editable: ball count derives from
    length / (radius * spacing), so stretching ADDS balls rather than stretching one shape -- and a
    THINNER bone needs MORE of them to stay smooth."""
    from holographic.mesh_and_geometry.holographic_creatureskin import ball_chain
    short, _ = ball_chain([0, 0, 0], [0, 0, 1.0], 0.1, 0.1)
    long_, _ = ball_chain([0, 0, 0], [0, 0, 2.0], 0.1, 0.1)
    thin, _ = ball_chain([0, 0, 0], [0, 0, 1.0], 0.05, 0.05)
    assert len(long_) >= 2 * len(short) - 2
    assert len(thin) > len(short)


def test_spine_profile_reaches_the_skin_and_scalar_specs_are_unchanged(mind):
    """R-1 across the boundary spec -> rig -> skin, plus the ADDITIVE constraint: a profile must
    visibly vary the balls, and a plain scalar spec must still produce a uniform spine."""
    spec = mind.spine_profile(mind.quadruped_spec(), [0.06, 0.16, 0.20, 0.16, 0.06])
    c = mind.creature(spec, skin=False)
    _, radii, bone_of = mind.creature_metaballs(c, spec)
    spine_r = np.asarray([r for r, b in zip(radii, bone_of) if b.startswith("spine")])
    assert spine_r.max() > 2.5 * spine_r.min()
    assert len(bone_of) == len(radii), "bone_of must label every ball (R-7 will need it)"

    plain = mind.quadruped_spec()
    _, pr, _ = mind.creature_metaballs(mind.creature(plain, skin=False), plain)
    assert abs(pr.max() - pr.min()) < 0.2, "a scalar-radius creature must stay uniform"


def test_creature_skin_mesh_marches_through_the_shipped_sdf_path(mind):
    """The blended skin must actually mesh. Note it does NOT go through the shipped metaball_mesh,
    which takes one radius for ALL centres and so cannot express a fat torso with thin wrists --
    hence the per-ball field. Crossing skin -> marcher is where that choice gets checked."""
    spec = mind.spine_profile(mind.quadruped_spec(), [0.07, 0.15, 0.18, 0.13, 0.07])
    mesh = mind.creature_skin_mesh(mind.creature(spec, skin=False), spec, resolution=28)
    assert len(np.asarray(mesh.vertices)) > 200


@pytest.mark.parametrize("edit", ["extend", "insert", "radius", "reshape"])
def test_spine_edits_are_pure_and_leave_a_buildable_spec(mind, edit):
    """R-2: every edit returns a NEW spec and must not touch the input -- that is what gives an editor
    undo for free and keeps a preview from corrupting the asset it previews. The result must also
    still build a Creature, or the edit produced a spec that only looks valid."""
    import copy
    base = mind.quadruped_spec()
    frozen = copy.deepcopy(base)
    out = {"extend": lambda: mind.extend_spine(base, 2),
           "insert": lambda: mind.insert_spine_node(base, 0.5),
           "radius": lambda: mind.set_spine_radius(base, 0.5, 0.25, falloff=0.3),
           "reshape": lambda: mind.reshape_spine(base, curve=0.3)}[edit]()
    assert base == frozen, "%s mutated the input spec" % edit
    mind.creature(out, skin=False)


def test_extending_the_spine_keeps_segment_length_and_limb_positions(mind):
    """The 'drag the tail out' behaviour: extending makes the spine LONGER at the same resolution.
    And limbs must not drift -- `at` is a fraction that Creature snaps to the nearest node, so adding
    a segment would silently slide every limb along the back if the fractions were not preserved."""
    base = mind.quadruped_spec()
    seg = base["spine"]["length"] / base["spine"]["segments"]
    out = mind.extend_spine(base, 2)
    assert out["spine"]["segments"] == base["spine"]["segments"] + 2
    assert abs(out["spine"]["length"] / out["spine"]["segments"] - seg) < 1e-12
    assert len(out.get("limbs", [])) == len(base.get("limbs", []))


def test_idle_still_works_on_an_edited_creature(mind):
    """Cross-faculty, and the trap this repo has on record: R-10 reads the rig's CURRENT state, so a
    creature whose spine was extended and re-thickened is a different input than a fresh one. Bone
    lengths and joint limits must still hold."""
    spec = mind.set_spine_radius(mind.extend_spine(mind.quadruped_spec(), 2), 0.5, 0.2, falloff=0.4)
    c = mind.creature(spec, skin=False)
    rep = mind.creature_idle_report(c, n_frames=8)
    assert rep["bone_length_error"] < 1e-9
    assert rep["limit_headroom"] >= -1e-12


@pytest.mark.parametrize("phrasing,expect", [
    ("leaves on a tree", "Trees by space"),
    ("oak tree", "Trees by space"),
    ("how thick should a branch be", "Trees by space"),
    ("make the belly fatter", "Spore-style"),
    ("fat torso thin neck", "Spore-style"),
    ("sculpt the torso", "Spore-style"),
])
def test_a_stranger_can_find_the_second_pass_too(mind, phrasing, expect):
    """Each of these returned an unrelated fallback before its catalog entry existed -- "oak tree"
    returned file_tree, "make the belly fatter" returned nothing at all."""
    assert expect in str(mind.find_capability(phrasing)[:3]), \
        "%r no longer finds %s -- the capability has gone dark" % (phrasing, expect)


# ================== SCALING: the "ceiling" that was a left fold, and the instancing win ==================

def test_deep_sdf_fold_evaluates_and_is_bit_identical(mind):
    """THE CORRECTION, pinned. skin_skeleton used to raise RecursionError between 200 and 300 edges,
    and this file previously recorded that as a hard ceiling. It was not: the union chain is built by
    a LEFT FOLD, so `_eval` recursed once per part. Unrolling that spine iteratively removed the limit.

    The assertion that matters is BIT-IDENTITY, not merely "it runs": smooth_union is NOT associative
    (rebalancing the tree moves the surface ~3e-3), so a restructuring fix would have silently changed
    every existing skinned mesh. Unrolling the EVALUATION changes nothing, and this proves it against
    an independent reference at depths either side of the iterative path's threshold.
    """
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    rng = np.random.default_rng(1)
    P = rng.normal(size=(200, 3)) * 0.5

    def reference(parts, kk):
        """The old recursive semantics, written out independently -- left to right, no shortcuts."""
        acc = parts[0].eval(P)
        for q in parts[1:]:
            b = q.eval(P)
            h = np.clip(0.5 + 0.5 * (b - acc) / kk, 0.0, 1.0)
            acc = b * (1 - h) + acc * h - kk * h * (1 - h)
        return acc

    for n in (10, 63, 64, 65, 500, 2000):                     # straddle the 64-deep switch point
        parts = [sphere(0.05 + 0.001 * i).translate([i * 0.01, 0, 0]) for i in range(n)]
        f = parts[0]
        for q in parts[1:]:
            f = f.smooth_union(q, 0.02)
        assert np.array_equal(f.eval(P), reference(parts, 0.02)), \
            "depth %d: the iterative path must be BIT-identical, not merely close" % n


def test_smooth_union_is_not_associative_so_the_tree_must_not_be_rebalanced(mind):
    """The measurement that ruled OUT the obvious fix, kept as a test so nobody "optimizes" the fold
    into a balanced tree later. A balanced reduction would cut depth to log2(N) -- and move the
    surface by thousands of ULP, flipping every previously-emitted mesh."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    P = np.random.default_rng(0).normal(size=(400, 3)) * 0.8
    a, b, c = sphere(0.5), sphere(0.4).translate([0.3, 0, 0]), sphere(0.45).translate([0, 0.3, 0])
    left = a.smooth_union(b, 0.1).smooth_union(c, 0.1)
    right = a.smooth_union(b.smooth_union(c, 0.1), 0.1)
    assert not np.array_equal(left.eval(P), right.eval(P)), \
        "if smooth_union ever becomes associative this test should be revisited, not deleted"
    # plain union IS a true commutative monoid (it is min), and stays exactly associative
    assert np.array_equal(a.union(b).union(c).eval(P), a.union(b.union(c)).eval(P))


def test_a_full_size_tree_now_skins_through_the_blended_path(mind):
    """The payoff: a tree far past the old ~250-edge limit blends through skin_skeleton."""
    from holographic.mesh_and_geometry.holographic_tree3d import tree_edges
    t = mind.grow_tree(mind.crown_attractors(n=250, seed=6))
    assert len(tree_edges(t)) > 250, "this test is pointless unless the tree exceeds the old ceiling"
    mesh = mind.skin_skeleton(t["nodes"], tree_edges(t), mind.taper_radii(t), resolution=16)
    assert len(np.asarray(mesh.vertices)) > 0


def test_instanced_tree_stores_one_definition_regardless_of_branch_count(mind):
    """Instancing, with a BASELINE rather than an adjective: the merged mesh stores one copy of the
    tube per branch, the instanced scene stores one tube total. Measured at 420 branches: 5040 verts
    vs 12. The claim under test is O(1) geometry in branch count, so it is checked at two sizes."""
    small = mind.grow_tree(mind.crown_attractors(n=120, seed=2))
    big = mind.grow_tree(mind.crown_attractors(n=400, seed=1))
    for t in (small, big):
        sc = mind.tree_instanced(t)
        stored = sum(len(np.asarray(d.geometry.vertices)) for d in sc.definitions())
        merged = len(np.asarray(mind.tree_mesh(t).vertices))
        assert len(sc.definitions()) == 1 and stored == 12
        assert len(sc.instances) <= len(t["segments"])
        assert merged > 20 * stored, "instancing must be a real win, not a rounding difference"


# ============ THIRD PASS: the HOLOGRAPHIC half -- parts (R-3), symmetry (R-4), weights (R-7), T-3, S-3 ============

def test_part_assembly_is_recallable_from_the_vector_not_the_dict(mind):
    """R-3: the point of encoding the layout as a bound record is that the VECTOR answers questions.
    Attach parts, then recall each socket by unbinding -- no dict lookup involved in the assertion."""
    lib = mind.part_library(dim=1024, seed=0)
    for p in ["horn", "eye", "jaw", "fin", "claw"]:
        lib.define(p, handles={"length": (0.5, 2.0)})
    a, v = mind.attach_part({}, "left_shoulder", "horn", lib)
    a, v = mind.attach_part(a, "jaw", "claw", lib)
    for sock, want in a.items():
        got, cos = mind.what_is_at(v, sock, lib)
        assert got == want and cos > 0.2, "%s -> %s (%.3f)" % (sock, got, cos)


def test_bundle_capacity_is_measured_and_actually_degrades(mind):
    """The honest half of R-3. A bundle fades as it loads, and a capacity claim nobody can falsify is
    decoration -- so this asserts BOTH directions: clean recall at a sane load, and genuine failure
    when overloaded. If the overload case ever passes, the encoding is not really superposing."""
    lib = mind.part_library(dim=1024, seed=0)
    for p in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        lib.define(p)
    ok = {"s%d" % i: sorted(lib.parts)[i % 8] for i in range(20)}
    rep = mind.assembly_report(ok, lib)
    assert rep["accuracy"] == 1.0 and rep["min_margin"] > 0

    tiny = mind.part_library(dim=64, seed=1)
    for p in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        tiny.define(p)
    over = {"s%d" % i: sorted(tiny.parts)[i % 8] for i in range(60)}
    assert mind.assembly_report(over, tiny)["accuracy"] < 1.0

    few = mind.assembly_report({k: ok[k] for k in list(ok)[:3]}, lib)
    assert few["mean_cosine"] > rep["mean_cosine"], "recall MUST weaken as the bundle fills"


@pytest.mark.parametrize("kind,n,count", [("none", 1, 1), ("bilateral", 2, 2), ("radial", 5, 5)])
def test_symmetry_groups_generalise_the_single_mirror_plane(mind, kind, n, count):
    """R-4: bilateral was the rig's only symmetry. As a GROUP, radial-5 is the same code path -- and
    the transforms must be geometrically valid, with a mirror carrying det -1 so transform_mesh knows
    to repair the winding."""
    T = mind.symmetry_transforms(kind, n)
    assert len(T) == count
    for M in T:
        assert np.abs(np.abs(np.linalg.det(M)) - 1.0) < 1e-9
        assert np.abs(M @ M.T - np.eye(3)).max() < 1e-9
    if kind == "bilateral":
        assert np.linalg.det(T[1]) < 0, "a mirror must reflect, not rotate"


def test_symmetric_attach_places_the_part_on_every_generated_socket(mind):
    lib = mind.part_library(dim=1024, seed=2)
    lib.define("fin")
    a, v = mind.attach_part({}, "flank", "fin", lib, symmetry="radial", n=5)
    assert len(a) == 5
    assert all(mind.what_is_at(v, s, lib)[0] == "fin" for s in a)


def test_skin_weights_follow_the_bones_that_made_the_balls(mind):
    """R-7 across the boundary creature -> metaballs -> weights. `creature_metaballs` returns bone_of
    precisely so this needs no re-derivation; the assertion is that provenance actually controls the
    weights, plus partition of unity and no NaN for a vertex outside every ball."""
    spec = mind.spine_profile(mind.quadruped_spec(), [0.07, 0.15, 0.18, 0.13, 0.07])
    c = mind.creature(spec, skin=False)
    C, R, bones = mind.creature_metaballs(c, spec)
    verts = np.vstack([C[:20], np.array([[50.0, 50.0, 50.0]])])
    idx, w, names, book = mind.skin_weights_from_balls(verts, C, R, bones, dim=256)
    assert np.allclose(w.sum(1), 1.0) and (w >= 0).all()
    assert np.isfinite(w).all(), "a vertex outside every ball must fall back, not produce NaN"
    # a vertex sitting exactly on a ball must be dominated by the bone that made that ball
    assert names[idx[0, 0]] == bones[0]


def test_scatter_layer_is_region_addressable(mind, ground):
    """The regression this pass fixed: the shipped ScatterLayer bound placements to region codes and
    bundled them, and the mesh-scatter path dropped that -- making the newer, more general path LESS
    capable. An occupied region must read far above an empty one."""
    g = mind.scatter_mesh(ground, mind.grass_blade(), 80, seed=11,
                          holographic=True, dim=2048, cell_size=1.0)
    assert "layer" in g and "instance" in g
    # Tested as the STATISTICAL read it is. An earlier version of this assert compared ONE occupied
    # point to ONE empty point and demanded an 8x ratio; it failed at 6.8x -- not because the encoding
    # was wrong, but because a single empty probe is a draw from a zero-mean noise distribution and its
    # sign is luck. Measured floor instead: occupied 0.165 vs empty -0.0008 +- 0.031, 5.4 sigma.
    rng_far = np.random.default_rng(0)
    far = [mind.region_occupancy(g["layer"], g["instance"], rng_far.normal(size=3) * 1000 + 2000,
                                 dim=2048, cell_size=1.0, seed=11) for _ in range(60)]
    occ = [mind.region_occupancy(g["layer"], g["instance"], g["transforms"][k, :3, 3],
                                 dim=2048, cell_size=1.0, seed=11)
           for k in range(0, len(g["transforms"]), 5)]
    assert (np.mean(occ) - np.mean(far)) / max(np.std(far), 1e-12) > 4.0
    assert abs(np.mean(far)) < 0.02, "empty regions must be zero-MEAN"
    # and it stays OFF by default -- additive, nobody pays for what they did not ask for
    assert "layer" not in mind.scatter_mesh(ground, mind.grass_blade(), 20, seed=1)


def test_spec_variants_are_pure_distinct_and_still_growable(mind):
    """T-3: variety without stored assets. Pure in (spec, seed), genuinely different, structurally
    valid -- and integer fields stay integers, because an L-system cannot run 3.4 iterations."""
    base = {"axiom": "F", "productions": {"F": "F[+F]F[-F]F"}, "iterations": 3,
            "angle_deg": 25.0, "step": 1.0}
    pool = mind.spec_variant_pool(base, 8, jitter=0.3)
    assert pool[0]["iterations"] == 3, "variant 0 must be the authored spec"
    assert mind.spec_variant(base, 3, 0.3) == mind.spec_variant(base, 3, 0.3), "variants must be pure"
    assert len({round(v["angle_deg"], 6) for v in pool}) >= 6
    for v in pool:
        assert isinstance(v["iterations"], int) and v["iterations"] >= 1
        assert len(mind.grow_stages("plant", v, 1)[-1]) > 0


def test_groom_strands_become_ribbon_grass(mind):
    """S-3, and a backlog correction: this was filed as "add profile='ribbon' to build_strand_body",
    but that function builds a PBD SoftBody -- the groom layer had no meshing path at all. With
    ribbons, grass inherits the whole shipped groom pipeline (root, simulate, wind) instead of being
    a static card. Crossing groom -> geometry is exactly the boundary a selftest cannot see."""
    strands = mind.groom_hair(lambda P: np.linalg.norm(P, axis=-1) - 1.0, 12,
                              ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), length=0.4, n_pts=6, seed=0)
    pts = [np.asarray(s.points, float) for s in strands] if hasattr(strands[0], "points") else strands
    rib = mind.strand_ribbons(pts, width=0.03)
    assert len(np.asarray(rib.vertices)) == sum(2 * len(p) for p in pts)
    assert len(np.asarray(rib.faces)) == sum(2 * (len(p) - 1) for p in pts)


@pytest.mark.parametrize("phrasing,expect", [
    ("snap parts onto a creature", "Creature parts"),
    ("library of body parts", "Creature parts"),
    ("which bone controls this vertex", "Skin weights"),
    ("is anything scattered here", "scatter layer"),
    ("twenty different ferns", "variant"),
])
def test_a_stranger_can_find_the_holographic_half(mind, phrasing, expect):
    assert expect.lower() in str(mind.find_capability(phrasing)[:3]).lower(), \
        "%r no longer finds %s" % (phrasing, expect)


# ==================== FINAL: paint mode (R-9) and scatter bake / LOD (S-4) ====================

def test_paint_follows_the_rig_not_world_space(mind):
    """R-9's central claim, crossing weights -> colour. A vertex fully weighted to one bone must take
    exactly that bone's hue, a 50/50 vertex must be the exact midpoint, and the whole thing must be
    deterministic. World-space noise cannot do the first two -- that is the difference."""
    names = ["spine0", "armL"]
    idx = np.array([[0, 1], [1, 0], [0, 1]])
    w = np.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    C = mind.bone_tint(idx, w, names, seed=0)
    assert not np.allclose(C[0], C[1]), "different bones must get different hues"
    assert np.allclose(C[2], 0.5 * (C[0] + C[1])), "a 50/50 vertex is the exact midpoint"
    assert np.array_equal(C, mind.bone_tint(idx, w, names, seed=0))


def test_paint_composes_with_a_real_creature(mind):
    """Cross-faculty end to end: creature -> metaballs -> weights -> colours, one vertex per row and
    everything in range. This is the chain R-1, R-7 and R-9 form, so it is checked as a chain."""
    spec = mind.spine_profile(mind.quadruped_spec(), [0.07, 0.15, 0.18, 0.13, 0.07])
    c = mind.creature(spec, skin=False)
    C, R, bones = mind.creature_metaballs(c, spec)
    mesh = mind.creature_skin_mesh(c, spec, resolution=20)
    V = np.asarray(mesh.vertices, float)
    idx, w, names, _ = mind.skin_weights_from_balls(V, C, R, bones, dim=256)
    cols = mind.paint_creature(V, idx, w, names, pattern="stripes")
    assert cols.shape == (len(V), 3)
    assert cols.min() >= 0.0 and cols.max() <= 1.0
    assert cols.std(0).max() > 1e-6, "a painted creature must not come out a flat colour"


def test_lod_thinning_is_nested_so_blades_never_flicker(mind):
    """S-4's usability property, and the one naive random thinning fails. The set kept at a FARTHER
    distance must be a strict subset of the set kept nearer, so dollying out only removes blades and
    dollying back restores the same ones."""
    M = mind.placement_frames(np.random.default_rng(1).uniform(0, 10, size=(400, 3)),
                              np.tile([0., 0, 1], (400, 1)), seed=1)
    sets, fracs = [], []
    for d in (0.0, 15.0, 30.0, 45.0, 80.0):
        kept, frac = mind.scatter_lod(M, d, seed=2)
        sets.append({tuple(np.round(t[:3, 3], 9)) for t in kept})
        fracs.append(frac)
    assert fracs[0] == 1.0 and fracs[-1] <= 0.10
    assert all(fracs[i] >= fracs[i + 1] - 1e-12 for i in range(len(fracs) - 1))
    for i in range(len(sets) - 1):
        assert sets[i + 1] <= sets[i], "LOD level %d is not nested inside %d -- blades would flicker" % (i + 1, i)


def test_scatter_bake_reports_exact_savings_against_a_real_baseline(mind):
    """The measurement discipline the backlog demanded. The baseline is the STRONGEST honest one --
    every instance at full resolution, what you would ship without LOD -- and triangle counts are
    exact, so they are asserted as integers with no error bars."""
    M = mind.placement_frames(np.random.default_rng(3).uniform(0, 10, size=(400, 3)),
                              np.tile([0., 0, 1], (400, 1)), seed=3)
    blade = mind.grass_blade(segments=3)
    bake = mind.scatter_bake(M, blade, seed=3)
    rep = bake.report()
    assert rep["baseline_tris"] == 400 * len(np.asarray(blade.faces))
    ratios = [r["ratio"] for r in rep["rows"]]
    assert ratios[0] == 1.0, "at distance 0 nothing may be dropped"
    assert all(ratios[i] >= ratios[i + 1] - 1e-12 for i in range(len(ratios) - 1))
    assert ratios[-1] < 0.15


def test_lod_levels_do_not_silently_fall_back(mind):
    """A bare try/except around decimation would let a failure look like a working LOD chain: thinning
    alone would still show a 'saving' while every level quietly drew the full-resolution blade. The
    report must name what happened per level, and at least one level must genuinely be smaller."""
    M = mind.placement_frames(np.random.default_rng(4).uniform(0, 10, size=(100, 3)),
                              np.tile([0., 0, 1], (100, 1)), seed=4)
    rep = mind.scatter_bake(M, mind.grass_blade(segments=3), seed=4).report()
    assert len(rep["level_note"]) == len(rep["level_faces"])
    assert not any(n.startswith("FALLBACK") for n in rep["level_note"]), rep["level_note"]
    assert rep["decimated_levels"] >= 1, \
        "if no level actually decimates, only the thinning is real: %s" % rep["level_faces"]


def test_the_bake_is_a_cache_not_a_rescatter(mind):
    """bake-once-sample-O(1): the same distance queried twice must return identical placements."""
    M = mind.placement_frames(np.random.default_rng(5).uniform(0, 10, size=(200, 3)),
                              np.tile([0., 0, 1], (200, 1)), seed=5)
    bake = mind.scatter_bake(M, mind.grass_blade(segments=3), seed=5)
    a, ma, fa, la = bake.at(30.0)
    b, mb, fb, lb = bake.at(30.0)
    assert np.array_equal(a, b) and fa == fb and la == lb


@pytest.mark.parametrize("phrasing,expect", [
    ("skin markings", "Paint a creature"),
    ("colour the body", "Paint a creature"),
    ("thin distant grass", "Scatter bake"),
    ("too many blades", "Scatter bake"),
])
def test_a_stranger_can_find_the_final_pass(mind, phrasing, expect):
    assert expect.lower() in str(mind.find_capability(phrasing)[:3]).lower(), \
        "%r no longer finds %s" % (phrasing, expect)


# ==================== ORGANIC CREATURE MATERIALS (layered anatomy + taxon skins) ====================

@pytest.mark.parametrize("taxon", ["reptile", "fish", "amphibian", "insect", "worm", "mammal"])
def test_every_taxon_produces_varying_channels(mind, taxon):
    """A flat channel means the structure field silently did nothing and the creature renders as a
    blob -- which is exactly what a constant-0.5 fallback pattern would produce."""
    P = np.random.default_rng(0).normal(size=(1500, 3)) * 0.4
    s = mind.creature_material(taxon, seed=1)
    c, r, f = s["colour"](P), s["roughness"](P), s["reflect"](P)
    assert c.shape == (len(P), 3) and 0.0 <= c.min() and c.max() <= 1.0
    assert c.std() > 1e-3 and r.std() > 1e-4
    assert 0.0 <= f.min() and f.max() <= 1.0


def test_the_families_are_measurably_different(mind):
    """If two taxa produced near-identical structure fields the recipes would be decoration. Compared
    pairwise rather than asserted by naming."""
    from holographic.materials_and_texture.holographic_creaturematerial import structure_field, TAXA
    P = np.random.default_rng(0).normal(size=(1500, 3)) * 0.4
    fields = {n: structure_field(n, seed=1)(P) for n in TAXA}
    for a in TAXA:
        for b in TAXA:
            if a < b:
                assert abs(float(np.corrcoef(fields[a], fields[b])[0, 1])) < 0.95, "%s ~ %s" % (a, b)


def test_coat_gloss_follows_the_anatomy(mind):
    """Wetness is the whole visual difference between a frog and a mouse, so the ordering is pinned:
    chitin > keratin > sebum, and a wet amphibian beats dry mammal skin."""
    P = np.random.default_rng(0).normal(size=(800, 3)) * 0.4
    refl = {t: mind.creature_material(t, seed=1)["reflect"](P).mean()
            for t in ("insect", "reptile", "mammal", "amphibian")}
    assert refl["insect"] > refl["reptile"] > refl["mammal"]
    assert refl["amphibian"] > refl["mammal"]


def test_structure_is_body_aligned_not_world_aligned(mind):
    """Scales that follow world axes swim across a flank and shear at a bend. Rotating the body axis
    must rotate the pattern -- if it does not, the field is world-locked."""
    from holographic.materials_and_texture.holographic_creaturematerial import structure_field
    P = np.random.default_rng(0).normal(size=(1200, 3)) * 0.4
    fz = structure_field("reptile", axis=(0, 0, 1), seed=1)(P)
    fx = structure_field("reptile", axis=(1, 0, 0), seed=1)(P)
    assert abs(float(np.corrcoef(fz, fx)[0, 1])) < 0.9


def test_insects_refuse_a_skeleton(mind):
    """The anatomy is ENFORCED, not merely documented: an arthropod's rigid structure is its
    exoskeleton, so stacking a bone layer under one would model an animal that does not exist."""
    with pytest.raises(ValueError, match="endoskeleton"):
        mind.anatomy_stack("insect", with_bone=True)
    ins = mind.anatomy_stack("insect")
    assert not ins["endoskeleton"]
    assert not any(l.startswith("skeleton") for l in ins["layers"])
    assert any(l.startswith("coat:chitin") for l in ins["layers"])
    rep = mind.anatomy_stack("reptile")
    assert rep["endoskeleton"] and rep["layers"][0].startswith("skeleton")


def test_interior_visibility_is_honest(mind):
    """Bone and organ only tint what re-emerges through translucent skin. A translucent frog shows
    more than an opaque insect, and switching the organ off makes it exactly zero -- not 'a bit'."""
    assert mind.anatomy_stack("amphibian")["interior_visible"] > \
           mind.anatomy_stack("insect")["interior_visible"]
    assert mind.anatomy_stack("mammal", with_organ=False)["interior_visible"] == 0.0


def test_the_render_field_is_a_real_distance_field(mind):
    """THE BUG THIS PINS: metaball_field returns a DENSITY (fine for marching cubes, which only reads
    the sign) whose gradient measured 0.0 to 26.1 on a real creature. A sphere tracer needs a
    Lipschitz bound, so the first quality render came back as pure background. The distance form must
    stay ~1."""
    spec = mind.spine_profile(mind.quadruped_spec(), [0.07, 0.15, 0.18, 0.13, 0.07])
    cr = mind.creature(spec, skin=False)
    fld = mind.creature_skin_field(cr, spec, spacing=0.9)
    P = np.random.default_rng(0).normal(size=(300, 3)) * 0.5 + np.array([0, 0, 0.5])
    e = 1e-3
    g = np.stack([(fld(P + d) - fld(P - d)) / (2 * e)
                  for d in (np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e]))], 1)
    gm = np.linalg.norm(g, axis=1)
    assert 0.75 < gm.mean() < 1.25 and gm.max() < 3.0
    # and it satisfies BOTH consumer contracts from one object
    assert callable(fld) and fld.ids(P).shape == (300,)
    assert len(fld.bounds()) == 2


def test_relief_keeps_the_field_conservative(mind):
    """Displacing structure into the surface breaks the Lipschitz bound (the structure's cells are
    small, so its gradient is large). The Lipschitz rescale must keep |grad| <= 1, or the tracer
    overshoots and punches holes in the skin."""
    spec = mind.spine_profile(mind.quadruped_spec(), [0.07, 0.15, 0.18, 0.13, 0.07])
    cr = mind.creature(spec, skin=False)
    fld = mind.creature_skin_field(cr, spec, spacing=0.9)
    struct = mind.creature_material("reptile", seed=2)["structure"]
    P = np.random.default_rng(0).normal(size=(250, 3)) * 0.5 + np.array([0, 0, 0.5])
    e = 1e-3
    for amp in (0.004, 0.010):
        r = fld.with_relief(struct, amplitude=amp)
        g = np.stack([(r(P + d) - r(P - d)) / (2 * e)
                      for d in (np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e]))], 1)
        assert np.linalg.norm(g, axis=1).max() <= 1.05, "relief amp=%.3f broke the bound" % amp
        assert r.relief_lipschitz >= 1.0


def test_tint_recolours_without_changing_the_family(mind):
    """A green frog and a blue one are one recipe with two tints, not two recipes -- so the structure
    must be identical while the colour differs."""
    from holographic.materials_and_texture.holographic_creaturematerial import structure_field
    P = np.random.default_rng(0).normal(size=(600, 3)) * 0.4
    a = mind.creature_material("amphibian", seed=3, tint=(0.2, 0.5, 0.3))
    b = mind.creature_material("amphibian", seed=3, tint=(0.2, 0.3, 0.6))
    assert np.array_equal(a["structure"](P), b["structure"](P))
    assert not np.allclose(a["colour"](P), b["colour"](P))


@pytest.mark.parametrize("phrasing,expect", [
    ("scales for a lizard", "Creature skin"),
    ("chitin", "Creature skin"),
    ("frog skin", "Creature skin"),
    ("exoskeleton", "Layered creature anatomy"),
    ("what is my creature made of", "Layered creature anatomy"),
])
def test_a_stranger_can_find_the_materials(mind, phrasing, expect):
    """'chitin' and 'exoskeleton' returned NOTHING before these entries existed."""
    assert expect.lower() in str(mind.find_capability(phrasing)[:3]).lower(), phrasing


# ============ VIEW-DEPENDENT SOCKETS + REAL THIN-FILM IRIDESCENCE (a defect, corrected) ============

def test_view_socket_leaves_existing_materials_bit_identical():
    """THE BACKWARD-COMPAT GUARANTEE for a change to shipped rendering code. `resolve` gained optional
    `normals`/`view_dirs`; every existing material must resolve to the SAME BYTES with or without
    them, and the reference render must not move at all."""
    from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial, render_surface
    from holographic.misc.holographic_param import Param
    from holographic.misc.holographic_pattern import make_pattern, field_lerp
    from holographic.rendering.holographic_render import Camera

    col = field_lerp(make_pattern("checker", scale=2.5), (0.85, 0.2, 0.15), (0.95, 0.9, 0.85))
    m0 = SurfaceMaterial(color=Param(field=col), roughness=0.35)
    P = np.random.default_rng(0).normal(size=(50, 3))
    N = np.tile([0.0, 0.0, 1.0], (50, 1)); V = np.tile([0.0, 0.0, -1.0], (50, 1))
    a, b = m0.resolve(P), m0.resolve(P, N, V)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not m0.is_view_dependent()

    class Balls:
        cs = np.array([[0.0, 0, 0], [1.9, 0, 0]])
        def eval(s, Q): return np.min(np.stack([np.linalg.norm(Q - c, axis=1) - 0.85 for c in s.cs]), 0)
        def ids(s, Q): return np.argmin(np.stack([np.linalg.norm(Q - c, axis=1) for c in s.cs]), 0)

    m1 = SurfaceMaterial.from_name("metal", color=(0.8, 0.8, 0.85)); m1.opacity = 0.55
    img = render_surface(Balls(), Camera(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52),
                         72, 72, {0: m0, 1: m1})
    assert abs(float(img.mean()) - 0.50964675) < 1e-8, "the reference render moved: %.8f" % img.mean()


def test_a_view_socket_refuses_to_resolve_without_geometry():
    """A material that quietly renders as the wrong thing is worse than one that refuses. Iridescence
    cannot be resolved from a hit point alone, so asking it to must raise."""
    from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial, ViewSocket
    vs = SurfaceMaterial(color=ViewSocket(lambda p, n, v: np.tile([1.0, 0, 0], (len(p), 1))))
    P = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(ValueError, match="view-dependent"):
        vs.resolve(P)
    got = vs.resolve(P, np.tile([0.0, 0, 1.0], (20, 1)), np.tile([0.0, 0, -1.0], (20, 1)))["color"]
    # and the (M,3) result must survive intact -- resolve_param reads such an array as a LOOKUP MAP
    # and collapses it to (M,), which silently turned an iridescent red into flat white once.
    assert np.allclose(got[0], [1.0, 0.0, 0.0]), "a resolved (M,3) colour must not be collapsed"
    assert vs.is_view_dependent()


def test_iridescence_is_live_not_dead_data(mind):
    """THE DEFECT THIS PINS. `iridescence` was set for fish and insect, returned in the dict, and read
    by nothing -- while a kept negative described the limitations of that non-existent code. A number
    nobody consumes must never again pass for a feature."""
    from holographic.mesh_and_geometry.holographic_surface import ViewSocket
    for taxon in ("fish", "insect"):
        s = mind.creature_material(taxon, seed=1)
        assert s["iridescence"] > 0
        assert isinstance(s["colour_socket"], ViewSocket), \
            "%s advertises iridescence but exposes no view-dependent socket" % taxon
    for taxon in ("reptile", "amphibian", "worm", "mammal"):
        assert not isinstance(mind.creature_material(taxon, seed=1)["colour_socket"], ViewSocket)
    assert not isinstance(mind.creature_material("insect", seed=1, iridescence=0.0)["colour_socket"],
                          ViewSocket)


def test_iridescence_shifts_hue_across_viewing_angle(mind):
    """Swept at a FIXED normal, because averaging over scattered normals washes the angular response
    out -- the first version of this measurement reported a 0.002 shift and the effect is 50x that."""
    s = mind.creature_material("insect", seed=1)
    ang = np.radians(np.linspace(0.0, 85.0, 9))
    views = np.stack([np.sin(ang), np.zeros(9), -np.cos(ang)], axis=1)
    cols = s["colour_socket"](np.zeros((9, 3)), np.tile([0.0, 0, 1.0], (9, 1)), views)
    rb = cols[:, 0] - cols[:, 2]
    assert cols.min() >= 0.0 and cols.max() <= 1.0
    assert rb.max() - rb.min() > 0.10, "hue barely moves (%.4f) -- a static tint, not iridescence" % (rb.max() - rb.min())


def test_thin_film_physics_actually_reverses():
    """Interference is not a ramp: past a half-wavelength of path difference the constructive and
    destructive colours SWAP. Asserted on the shipped physics, which is the layer the claim belongs
    to -- the creature's final colour also carries its base pigment and would not isolate this."""
    from holographic.rendering.holographic_thinfilm import thin_film_tint
    cos_sweep = np.cos(np.radians(np.linspace(0.0, 85.0, 12)))
    tint = np.asarray(thin_film_tint(np.full_like(cos_sweep, 340.0), cos_sweep, n_film=1.56), float)
    rb = tint[:, 0] - tint[:, 2]
    assert rb.min() < 0 < rb.max(), "a film must reverse its colour across angle (%.3f..%.3f)" % (rb.min(), rb.max())


# ==================== THE CREATURE EDITOR LOOP (sockets, picking, session) ====================

@pytest.fixture
def lib(mind):
    from holographic.mesh_and_geometry.holographic_creaturesocket import _unit_cone
    L = mind.part_library(dim=256, seed=0)
    L.define("horn", handles={"length": (0.5, 2.0)}, geometry=_unit_cone())
    return L


def test_a_socket_stays_on_the_skin_through_body_edits(mind, lib):
    """THE PROPERTY SOCKETS EXIST FOR. Anatomy coordinates (t, theta) survive reshaping; a stored
    world position would leave the horn floating beside the body after the first spine edit."""
    ed = mind.creature_editor(part_library=lib)
    before = mind.resolve_socket(ed.creature(), ed.field(), 0.5, 0.0)
    assert before["hit"] and abs(float(ed.field()(before["point"][None, :])[0])) < 1e-6
    ed.set_thickness(0.5, 0.26, falloff=0.4)
    after = mind.resolve_socket(ed.creature(), ed.field(), 0.5, 0.0)
    assert after["hit"] and abs(float(ed.field()(after["point"][None, :])[0])) < 1e-6
    assert after["depth"] > before["depth"] + 1e-3, "a fatter body must push the socket outward"


def test_pick_and_place_round_trip(mind):
    """If picking and placing disagreed, a part would jump the instant the user released the mouse."""
    ed = mind.creature_editor()
    cr, fld = ed.creature(), ed.field()
    for t, th in [(0.3, 0.0), (0.5, 1.2), (0.7, -2.0)]:
        pt = mind.resolve_socket(cr, fld, t, th)
        assert pt["hit"]
        back = mind.socket_at_point(cr, pt["point"])
        again = mind.resolve_socket(cr, fld, back["t"], back["theta"])
        assert float(np.linalg.norm(again["point"] - pt["point"])) < 0.02


def test_viewport_pick_hits_the_body_and_misses_empty_space(mind):
    ed = mind.creature_editor()
    cr, fld = ed.creature(), ed.field()
    tgt = mind.resolve_socket(cr, fld, 0.5, 0.0)
    eye = tgt["point"] + tgt["normal"] * 2.0
    hit = mind.pick_socket(cr, fld, eye, -tgt["normal"])
    assert hit is not None and float(np.linalg.norm(hit["point"] - tgt["point"])) < 1e-3
    assert mind.pick_socket(cr, fld, eye + np.array([50.0, 50.0, 50.0]),
                            np.array([0.0, 0.0, 1.0])) is None


def test_placed_parts_sit_on_the_surface_not_inside_it(mind, lib):
    """Part geometry must land ON the skin. Buried parts are the failure mode a socket system exists
    to prevent, and it is invisible until you render."""
    ed = mind.creature_editor(part_library=lib)
    cr, fld = ed.creature(), ed.field()
    out = mind.place_parts(cr, fld, [{"t": 0.5, "theta": 0.9, "part": "horn",
                                      "symmetry": "bilateral"}], lib)
    assert len(out["placements"]) == 2 and not out["missed"]
    d = np.asarray(fld(np.asarray(out["geometry"].vertices, float)), float)
    assert d.min() > -0.05, "part geometry is buried in the body (%.3f)" % d.min()
    # ...and the two mirrored parts must be in DIFFERENT places (the transform-convention trap again)
    pts = np.array([p["point"] for p in out["placements"]])
    assert float(np.linalg.norm(pts[0] - pts[1])) > 1e-3, "mirrored parts must not coincide"

    # A part ON THE MIRROR PLANE mirrors onto itself, so it must be placed ONCE. Two coincident horns
    # would z-fight and cost double -- found by this test expecting two distinct points at theta=0.
    mid = mind.place_parts(cr, fld, [{"t": 0.5, "theta": 0.0, "part": "horn",
                                      "symmetry": "bilateral"}], lib)
    assert len(mid["placements"]) == 1, "a midline part must not be duplicated"


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_undo_walks_back_exactly_n_steps(mind, lib, steps):
    """Each edit is separately undoable, in order, and redo re-applies."""
    ed = mind.creature_editor(part_library=lib)
    base = ed.spec["spine"]["segments"]
    ed.extend_spine(2).set_thickness(0.5, 0.22).add_part("horn", 0.5, 0.0)
    for _ in range(steps):
        ed.undo()
    if steps >= 1:
        assert len(ed.spec["sockets"]) == 0
    if steps >= 3:
        assert ed.spec["spine"]["segments"] == base and not ed.can_undo()
    assert ed.can_redo()
    ed.redo()
    assert ed.can_undo()


def test_save_load_round_trips_byte_identical(mind, lib):
    """A saved creature must reload as the SAME creature. The document is kept canonically
    JSON-shaped at all times so this is an identity, not an approximation."""
    ed = mind.creature_editor(part_library=lib)
    ed.extend_spine(1).set_thickness(0.4, 0.18).add_part("horn", 0.6, 1.1, symmetry="radial", n=3)
    text = ed.to_json()
    back = mind.load_creature(text, part_library=lib)
    assert back.spec == ed.spec
    assert back.to_json() == text
    with pytest.raises(ValueError):
        mind.load_creature('{"format": "not-a-creature"}')


def test_validation_catches_real_breakage(mind, lib):
    ed = mind.creature_editor(part_library=lib)
    assert ed.validate()["ok"]
    off = mind.creature_editor(part_library=lib)
    off.spec["limbs"][0]["at"] = 5.0
    assert not off.validate()["ok"]
    unknown = mind.creature_editor(part_library=lib)
    unknown.add_part("no_such_part", 0.5, 0.0)
    v = unknown.validate()
    assert not v["ok"] and any("library" in e for e in v["errors"])


def test_complexity_budget_counts_symmetry(mind, lib):
    """A radial-5 part costs five parts, not one -- the budget must count what is actually built."""
    ed = mind.creature_editor(part_library=lib)
    assert ed.complexity(cap=100)["within"]
    for i in range(12):
        ed.add_part("horn", 0.1 + 0.05 * i, 0.0, symmetry="radial", n=5)
    c = ed.complexity(cap=100)
    assert c["parts"] == 60 and not c["within"]


def test_the_document_is_never_mutated_in_place(mind):
    """A held reference must not change under the caller -- this is what makes undo and preview safe."""
    ed = mind.creature_editor()
    held = ed.spec
    base = held["spine"]["segments"]
    ed.extend_spine(3).set_thickness(0.5, 0.2)
    assert held["spine"]["segments"] == base


def test_build_produces_skin_and_parts_together(mind, lib):
    ed = mind.creature_editor(part_library=lib)
    ed.add_part("horn", 0.5, 0.9, symmetry="bilateral")
    out = ed.build(resolution=24)
    assert len(np.asarray(out["skin"].vertices)) > 100
    assert out["parts"] is not None and len(out["placements"]) == 2 and not out["missed"]


def test_click_to_place_stores_anatomy_coordinates(mind, lib):
    """The click path must store (t, theta), not the world point it was clicked at -- otherwise the
    part stops surviving edits, which is the whole point of the socket system."""
    ed = mind.creature_editor(part_library=lib)
    cr, fld = ed.creature(), ed.field()
    p = mind.resolve_socket(cr, fld, 0.42, 0.8)["point"]
    ed.add_part_at_point("horn", p, creature=cr, symmetry="none")
    sock = ed.spec["sockets"][0]
    assert set(sock) >= {"t", "theta", "part"}
    assert 0.0 <= sock["t"] <= 1.0
    again = mind.resolve_socket(cr, fld, sock["t"], sock["theta"])
    assert float(np.linalg.norm(again["point"] - p)) < 0.02


@pytest.mark.parametrize("phrasing,expect", [
    ("undo my last change", "Creature editor"),
    ("save a creature to a file", "Creature editor"),
    ("dna points", "Creature editor"),
    ("stick a horn on the back", "Creature sockets"),
    ("where did i click on the model", "Creature sockets"),
])
def test_a_stranger_can_find_the_editor(mind, phrasing, expect):
    assert expect.lower() in str(mind.find_capability(phrasing)[:3]).lower(), phrasing


# ==================== THE PARAMETRIC PART LIBRARY ====================

@pytest.mark.parametrize("name", ["eye", "mouth", "foot", "hand", "claw", "horn", "spike",
                                  "fin", "antenna", "ear", "digit"])
def test_every_part_builds_and_is_well_formed(mind, name):
    """A part must be non-degenerate, finite, index-safe, and oriented the way a socket frame
    expects. An aperture (a mouth) legitimately straddles the surface instead of standing on it, so
    each kind is held to its own contract rather than one loose rule that fits neither."""
    from holographic.mesh_and_geometry.holographic_creaturepartlib import part_origin
    m = mind.build_part(name)
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, int)
    assert len(V) > 3 and len(F) > 3 and np.isfinite(V).all()
    assert F.max() < len(V) and F.min() >= 0
    assert V[:, 2].max() > 0.01
    if part_origin(name) == "standing":
        assert V[:, 2].min() > -0.03
    else:
        assert V[:, 2].min() < 0.0 < V[:, 2].max()


def test_digit_count_changes_topology_not_just_scale(mind):
    """The handle that justifies parametric parts over deformed authored meshes: a five-toed foot has
    MORE geometry than a three-toed one. No amount of scaling a mesh produces another toe."""
    f3 = np.asarray(mind.build_part("foot", digits=3).faces)
    f5 = np.asarray(mind.build_part("foot", digits=5).faces)
    assert len(f5) > len(f3)
    # ...and the count clamps to the authored range rather than accepting nonsense
    assert len(np.asarray(mind.build_part("foot", digits=99).faces)) == \
           len(np.asarray(mind.build_part("foot", digits=6).faces))


def test_handles_change_shape_measurably(mind):
    """Each authored handle must do what its name says, measured rather than assumed."""
    def ext(m, ax):
        v = np.asarray(m.vertices, float)[:, ax]
        return float(v.max() - v.min())
    assert np.asarray(mind.build_part("horn", length=2.0).vertices)[:, 2].max() > \
           1.6 * np.asarray(mind.build_part("horn", length=1.0).vertices)[:, 2].max()
    straight = np.asarray(mind.build_part("horn", curl=0.0).vertices, float)[:, 0].max()
    curled = np.asarray(mind.build_part("horn", curl=1.4).vertices, float)[:, 0].max()
    assert curled > straight + 0.02, "curl must sweep the horn forward"
    assert np.asarray(mind.build_part("eye", stalk=2.0).vertices)[:, 2].max() > \
           np.asarray(mind.build_part("eye", stalk=0.0).vertices)[:, 2].max() + 0.05
    assert ext(mind.build_part("fin", span=2.0), 0) > 1.6 * ext(mind.build_part("fin", span=1.0), 0)


def test_tapered_sweep_actually_tapers(mind):
    """The workhorse every part is built on. `sweep_tube` takes one profile for the whole tube and
    cannot do this, which is why it exists."""
    P = np.stack([np.zeros(5), np.zeros(5), np.linspace(0, 1, 5)], axis=1)
    s = mind.sweep_profile(P, np.linspace(0.1, 0.02, 5), sides=8)
    V = np.asarray(s.vertices, float)
    assert len(V) == 5 * 8 + 2, "5 rings of 8 plus two cap centres"
    r0 = np.linalg.norm(V[:8, :2], axis=1).mean()
    r1 = np.linalg.norm(V[32:40, :2], axis=1).mean()
    assert r0 > 4 * r1, "the sweep must taper along the path"


def test_the_library_registers_every_part_with_clamping_handles(mind):
    lib = mind.creature_parts(dim=256, seed=0)
    assert set(lib.parts) == set(mind.part_names())
    assert all(lib.parts[n]["geometry"] is not None for n in mind.part_names())
    assert abs(lib.clamp("horn", "length", 99.0) - 3.0) < 1e-9
    assert abs(lib.clamp("horn", "length", 0.0) - 0.4) < 1e-9


def test_real_parts_attach_to_a_real_creature(mind):
    """The whole stack in one test: parametric parts -> library -> editor -> sockets -> geometry.
    Every socket must land exactly on the skin and every placement be in a distinct spot."""
    lib = mind.creature_parts(dim=256, seed=0)
    ed = mind.creature_editor(part_library=lib)
    ed.extend_spine(2).set_profile([0.055, 0.105, 0.155, 0.175, 0.145, 0.095, 0.055])
    ed.add_part("eye", 0.955, 0.75, symmetry="bilateral")
    ed.add_part("mouth", 0.90, 0.0, symmetry="none")
    ed.add_part("horn", 0.90, 0.45, symmetry="bilateral")
    for i in range(3):
        ed.add_part("fin", 0.40 + 0.11 * i, 0.0, symmetry="none")
    assert ed.validate()["ok"]
    out = ed.build(resolution=30)
    assert len(out["placements"]) == 8 and not out["missed"]
    fld = ed.field()
    for p in out["placements"]:
        assert abs(float(fld(p["point"][None, :])[0])) < 1e-6, "%s is off the skin" % p["part"]
    pts = np.array([p["point"] for p in out["placements"]])
    assert len({tuple(np.round(q, 6)) for q in pts}) == len(pts), "placements must be distinct"
    # a document with real parts must still round-trip
    assert mind.load_creature(ed.to_json(), part_library=lib).spec == ed.spec


@pytest.mark.parametrize("phrasing,expect", [
    ("give it eyes", "Creature body parts"),
    ("three toed foot", "Creature body parts"),
    ("how many fingers", "Creature body parts"),
    ("tapered tube", "Tapered sweep"),
])
def test_a_stranger_can_find_the_parts(mind, phrasing, expect):
    assert expect.lower() in str(mind.find_capability(phrasing)[:3]).lower(), phrasing


# ==================== GAIT: locomotion, and foot slip as the honest metric ====================

def _stance_spec(mind, n_pairs=2, length=0.55):
    """A body in a WALKING stance: spine horizontal, limbs pointing down.

    The shipped `quadruped_spec` has a VERTICAL spine with limbs radiating sideways, so its lower
    pair are legs and its upper pair are arms -- which `analyze_rig` correctly reports as 2 legs.
    This helper exists so the tests exercise a real quadruped rather than the classifier being
    loosened to call arms legs.
    """
    s = mind.quadruped_spec()
    s["spine"] = {"length": 1.2, "segments": 4, "axis": [0.0, 1.0, 0.0], "curve": 0.05, "radius": 0.1}
    s["limbs"] = [{"at": 0.2 + (0.6 / max(n_pairs - 1, 1)) * i if n_pairs > 1 else 0.5,
                   "dir": [1.0, 0.0, -2.2], "segments": 3, "length": length,
                   "radius": 0.05, "mirror": True} for i in range(n_pairs)]
    return s


def test_legs_are_found_by_measurement_not_by_name(mind):
    """Morphology-independence starts here: a limb is a leg because it REACHES THE GROUND, not
    because anything labelled it.

    THE SECOND ASSERT USED TO EXPECT 2 ON THE SHIPPED QUADRUPED, with a docstring explaining that
    this was "the correct reading of that body". It was not. `analyze_rig` tested axis 2 for ground
    contact, but on quadruped_spec z is the SPINE'S LENGTH axis and y is vertical -- all four feet
    sit at y = -0.376 while their z values are 0.300 and 0.900, so it selected the two FRONT legs.
    Down is now measured from the body, and this test says four. The lesson is in the old docstring:
    a surprising number was met with prose reconciling us to it rather than an investigation."""
    quad = mind.creature(_stance_spec(mind, 2), skin=False)
    rig = mind.analyze_rig(quad)
    assert rig["n_legs"] == 4 and rig["stride"] > 0.0
    shipped = mind.analyze_rig(mind.creature(mind.quadruped_spec(), skin=False))
    assert shipped["n_legs"] == 4, "the shipped quadruped has four legs: %r" % shipped["legs"]
    # ...and the gait's answer must equal the rig's own role inference. They silently disagreed about
    # which way is down for as long as both existed, and each looked right in isolation.
    cr = mind.creature(mind.quadruped_spec(), skin=False)
    assert len(mind.rig_roles(cr).find_by_role("foot")) == shipped["n_legs"]


def test_speed_is_derived_so_planted_feet_cannot_slide(mind):
    """The design decision the whole module rests on: speed = stride / (duty * period). Take speed as
    an independent input and the feet slide by whatever the mismatch happens to be."""
    from holographic.mesh_and_geometry.holographic_gait import speed_for
    pat = mind.gait_pattern(4, "walk")
    v = speed_for(0.30, 1.0, pat["duty"])
    assert abs(v * pat["duty"] * 1.0 - 0.30) < 1e-12


def test_the_feet_actually_step_before_slip_is_believed(mind):
    """THE CONTROL. A foot that never moves cannot slip, so a broken IK scores a PERFECT 0.00%. That
    is precisely what happened while building this: a tuple return was unpacked as a list, every leg
    was declared unreachable, and the metric read zero while measuring nothing. These assertions are
    what make the slip figure a result rather than an artifact."""
    cr = mind.creature(_stance_spec(mind, 2), skin=False)
    rig = mind.analyze_rig(cr)
    rep = mind.gait_report(cr, gait="walk", period=1.0, n_frames=48)
    assert rep["unreachable"] == [], "legs %s could not be posed" % rep["unreachable"]
    frames = [mind.gait_pose(cr, float(t), period=1.0, rig=rig)
              for t in np.linspace(0.0, 1.0, 48, endpoint=False)]
    tip = cr.chains[rig["legs"][0]][-1]
    pts = np.array([f["joints"][tip] for f in frames])
    assert float(np.linalg.norm(pts - pts[0], axis=1).max()) > 0.5 * rig["stride"], "feet must STEP"
    assert float(pts[:, 2].max() - pts[:, 2].min()) > 0.02, "feet must LIFT during swing"
    assert rep["distance"] > 0.5 * rep["stride"], "the body must travel"


@pytest.mark.parametrize("gait", ["walk", "trot", "pace", "bound", "gallop"])
def test_every_tetrapod_gait_keeps_planted_feet_near_still(mind, gait):
    """Slip is NOT zero -- the derived speed removes the systematic component, and what remains is IK
    residual within the joint limits plus frame discretisation. The bar sits just above where it
    lands (measured ~6.8%), so a regression shows rather than being absorbed by a loose threshold."""
    cr = mind.creature(_stance_spec(mind, 2), skin=False)
    r = mind.gait_report(cr, gait=gait, period=1.0, n_frames=32)
    assert r["unreachable"] == [], "%s left %s unposed" % (gait, r["unreachable"])
    assert r["slip_ratio"] < 0.14, "%s slips %.1f%% of a stride" % (gait, 100 * r["slip_ratio"])


def test_duty_measured_matches_the_gait_diagram(mind):
    """Feet must be down for about the fraction the named gait claims, or the pattern is decorative."""
    cr = mind.creature(_stance_spec(mind, 2), skin=False)
    rep = mind.gait_report(cr, gait="walk", period=1.0, n_frames=48)
    for leg, frac in rep["duty_measured"].items():
        assert abs(frac - rep["duty_nominal"]) < 0.12, "%s planted %.2f, gait says %.2f" % (
            leg, frac, rep["duty_nominal"])


def test_named_gaits_are_actually_different_animals(mind):
    """A trot is not a walk with another name: lower duty, different phase pairing."""
    walk, trot = mind.gait_pattern(4, "walk"), mind.gait_pattern(4, "trot")
    assert trot["duty"] < walk["duty"] and walk["phases"] != trot["phases"]
    assert set(mind.gait_names(4)) == {"walk", "trot", "pace", "bound", "gallop"}
    assert set(mind.gait_names(2)) == {"walk", "run"}


def test_a_hexapod_walks_from_the_same_code(mind):
    """The morphology-independence claim, checked on a body the gait tables say nothing about: six
    legs get an evenly spread metachronal wave, which is what many-legged animals use."""
    c6 = mind.creature(_stance_spec(mind, 3), skin=False)
    assert mind.analyze_rig(c6)["n_legs"] == 6
    p6 = mind.gait_pattern(6)
    assert len(p6["phases"]) == 6 and p6["kind"] == "wave"
    r6 = mind.gait_report(c6, period=1.0, n_frames=32)
    assert r6["unreachable"] == [] and r6["slip_ratio"] < 0.14


def test_gait_poses_are_pure_and_do_not_mutate_the_rig(mind):
    """Scrubbing a walk back and forth must be safe, and asking for a pose must not repose the asset."""
    cr = mind.creature(_stance_spec(mind, 2), skin=False)
    before = {k: np.asarray(v).copy() for k, v in cr.joints.items()}
    a = mind.gait_pose(cr, 0.3)["joints"]
    mind.gait_pose(cr, 0.7)
    b = mind.gait_pose(cr, 0.3)["joints"]
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert all(np.allclose(before[k], cr.joints[k]) for k in before)
    assert len(mind.gait_frames(cr, n_frames=16)) == 16


def test_a_legless_creature_is_handled_not_crashed(mind):
    """A snake is a legitimate document; it simply does not walk."""
    s = _stance_spec(mind, 2); s["limbs"] = []
    assert mind.gait_report(mind.creature(s, skin=False))["legs"] == 0


def test_gait_composes_with_the_editor(mind):
    """Cross-faculty: a creature built through the editor -- spine extended, thickness edited -- must
    still walk, because the gait reads the CURRENT rig rather than an authored assumption."""
    ed = mind.creature_editor(_stance_spec(mind, 2))
    ed.extend_spine(1).set_thickness(0.5, 0.16, falloff=0.3)
    cr = ed.creature()
    r = mind.gait_report(cr, gait="trot", period=1.0, n_frames=32)
    assert r["legs"] == 4 and r["unreachable"] == [] and r["slip_ratio"] < 0.14


@pytest.mark.parametrize("phrasing,expect", [
    ("make it walk", "Creature gait"),
    ("trot gallop", "Creature gait"),
    ("moonwalk", "Foot slip"),
    ("does my creature walk properly", "Foot slip"),
])
def test_a_stranger_can_find_the_gait(mind, phrasing, expect):
    assert expect.lower() in str(mind.find_capability(phrasing)[:3]).lower(), phrasing


# ============ BODY SHAPING, MESH QUALITY, PART FUSION (the three visual defects) ============

def test_an_unwarped_body_is_circular_and_a_warped_one_is_not(mind):
    """Metaballs are spheres, so a creature's cross section is a CIRCLE until space is warped -- a
    fat belly is otherwise still a round belly. Measured on the body's OWN axes: probing world x/y
    reads the wrong directions entirely and made the first measurement look inverted."""
    from holographic.mesh_and_geometry.holographic_curves import rotation_minimizing_frame
    ed = mind.creature_editor()
    ed.extend_spine(1).set_profile([0.10, 0.19, 0.24, 0.23, 0.17, 0.11])
    cr = ed.creature()
    nodes = np.array([np.asarray(cr.joints[n], float) for n in cr.spine_nodes])
    T, N, B = rotation_minimizing_frame(nodes)
    k = len(nodes) // 2
    mid, nax, bax = nodes[k], np.asarray(N)[k], np.asarray(B)[k]

    def hw(f, d):
        rs = np.linspace(0.0, 1.2, 1200)
        v = f(mid[None, :] + rs[:, None] * np.asarray(d, float)[None, :])
        c = np.where(np.diff(np.sign(v)) != 0)[0]
        return float(rs[c[0]]) if len(c) else 0.0

    plain = mind.creature_skin_field(cr, ed.spec, spacing=0.9)
    bn, bb = hw(plain, nax), hw(plain, bax)
    assert abs(bn / bb - 1.0) < 0.15, "an unwarped body must be circular in cross-section"
    shaped = mind.creature_skin_field(cr, ed.spec, spacing=0.9,
                                      warp=mind.section_warp(cr, width=1.6, depth=0.7))
    assert abs(hw(shaped, nax) / bn - 1.6) < 0.12
    assert abs(hw(shaped, bax) / bb - 0.7) < 0.12


def test_a_warped_body_is_still_a_valid_distance_field(mind):
    """THE DEFECT THIS PINS: taking the NEAREST spine station makes the frame jump at the midpoint
    between nodes, so the warped field's gradient spiked to 2.11 where a distance field must stay at
    1.0 -- a sphere tracer would punch through the surface. Marching cubes never noticed, because it
    reads only the SIGN, which is exactly how such a defect ships looking fine. Frames are now
    blended across stations."""
    ed = mind.creature_editor()
    ed.extend_spine(1).set_profile([0.10, 0.19, 0.24, 0.23, 0.17, 0.11])
    cr = ed.creature()
    f = mind.creature_skin_field(cr, ed.spec, spacing=0.9,
                                 warp=mind.section_warp(cr, width=1.45, depth=0.78, belly=0.3))
    P = np.random.default_rng(3).normal(size=(400, 3)) * 0.5
    e = 1e-3
    g = np.stack([(f(P + d) - f(P - d)) / (2 * e)
                  for d in (np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e]))], 1)
    assert float(np.linalg.norm(g, axis=1).max()) <= 1.05


def test_warp_none_is_bit_identical(mind):
    """Additive constraint: shaping must not change any existing creature."""
    ed = mind.creature_editor()
    cr = ed.creature()
    P = np.random.default_rng(0).normal(size=(300, 3)) * 0.5
    a = mind.creature_skin_field(cr, ed.spec, spacing=0.9)
    b = mind.creature_skin_field(cr, ed.spec, spacing=0.9, warp=None)
    assert np.array_equal(a(P), b(P))


def test_the_quality_guard_catches_a_lumpy_resolution(mind):
    """The lumpy-limb bug: a limb 2.3 marching cells across beads up. The guard must call it, and
    must clear the resolution it recommends -- an advisor that only ever complains is useless."""
    ed = mind.creature_editor()
    ed.extend_spine(2).set_profile([0.055, 0.105, 0.155, 0.175, 0.145, 0.095, 0.055])
    cr = ed.creature()
    bad = mind.skin_quality(cr, ed.spec, spacing=0.9, resolution=104)
    assert not bad["ok"] and bad["cells_across"] < 4.0 and "LUMPY" in bad["verdict"]
    good = mind.skin_quality(cr, ed.spec, spacing=0.9, resolution=bad["recommended_resolution"])
    assert good["ok"] and good["cells_across"] >= 4.0


def test_fusion_makes_a_part_genuinely_part_of_the_skin(mind):
    """"Attached" means one implicit surface, not geometry resting against another. A point above the
    plain skin must read OUTSIDE it and INSIDE the fused field."""
    lib = mind.creature_parts(dim=256, seed=0)
    ed = mind.creature_editor(part_library=lib)
    ed.extend_spine(2).set_profile([0.055, 0.105, 0.155, 0.175, 0.145, 0.095, 0.055])
    ed.add_part("horn", 0.90, 0.5, symmetry="bilateral", handles={"length": 1.6})
    ed.add_part("fin", 0.5, 0.0, symmetry="none")
    cr = ed.creature()
    plain = mind.creature_skin_field(cr, ed.spec, spacing=0.9)
    fused, names, unfused = mind.fused_field(cr, ed.spec, ed.spec["sockets"], lib, spacing=0.9)
    assert "horn" in names, "a horn is a tapered tube and must fuse"
    assert "fin" in unfused, "a fin is a membrane -- fusing it would make it a blob"
    r = mind.resolve_socket(cr, plain, 0.90, 0.5)
    probe = (r["point"] + r["normal"] * 0.12)[None, :]
    assert float(plain(probe)[0]) > 0.0, "the probe must be outside the plain skin"
    assert float(fused(probe)[0]) < 0.0, "and inside the fused one -- otherwise it is not attached"


@pytest.mark.parametrize("phrasing,expect", [
    ("barrel chested", "body shape"),
    ("why is my creature lumpy", "Mesh quality"),
    ("parts look glued on", "Fuse parts"),
])
def test_a_stranger_can_find_the_shaping_fixes(mind, phrasing, expect):
    assert expect.lower() in str(mind.find_capability(phrasing)[:3]).lower(), phrasing


# ============ LIMB SOCKETS, AUTO FEET, CEL SHADING ============

def test_a_foot_lands_on_the_end_of_a_leg(mind):
    """CLOSES A KEPT NEGATIVE: sockets were spine-relative, so anything on a limb used the nearest
    SPINE station and was documented as approximate -- which is why feet could not go where feet go.
    A limb socket uses the limb's OWN axis, so a foot lands at the tip, on the surface."""
    lib = mind.creature_parts(dim=256, seed=0)
    ed = mind.creature_editor(part_library=lib)
    ed.extend_spine(1).set_profile([0.10, 0.20, 0.25, 0.235, 0.175, 0.11])
    ed.spec["limbs"] = [{"at": 0.30, "dir": [1.0, 0.0, -1.8], "segments": 3, "length": 0.54,
                         "radius": 0.088, "mirror": True},
                        {"at": 0.72, "dir": [1.0, 0.0, -1.8], "segments": 3, "length": 0.52,
                         "radius": 0.088, "mirror": True}]
    cr, fld = ed.creature(), ed.field()
    feet = mind.auto_feet(cr, fld, part="foot", scale=1.1)
    assert feet and all(f["along_axis"] and f["u"] == 1.0 for f in feet)
    out = mind.place_parts(cr, fld, feet, lib)
    assert len(out["placements"]) == len(feet) and not out["missed"]
    for p in out["placements"]:
        assert abs(float(fld(p["point"][None, :])[0])) < 1e-6, "the foot must sit ON the skin"
        tip = np.asarray(cr.joints[cr.chains[p["limb"]][-1]], float)
        assert float(np.linalg.norm(p["point"] - tip)) < 0.35, "and near the leg's TIP"


def test_auto_feet_does_not_put_feet_on_arms(mind):
    """Legs are identified by reaching the ground, the same measurement the gait uses -- so a body
    whose upper limbs are arms gets feet only on the lower pair, with no authoring."""
    ed = mind.creature_editor()
    cr = ed.creature()
    legs = mind.analyze_rig(cr)["legs"]
    feet = mind.auto_feet(cr, ed.field())
    # The COUNT is not the claim -- it moved from 2 to 4 when the ground axis stopped being hard-coded
    # to z (see test_legs_are_found_by_measurement_not_by_name). The claim is AGREEMENT: auto_feet
    # must put a foot on exactly the limbs the gait calls legs, whatever that set turns out to be.
    assert len(feet) == len(legs) > 0
    assert {f["limb"] for f in feet} == set(legs)


@pytest.mark.parametrize("bands", [1, 2, 3, 5, 12])
def test_cel_shading_quantises_into_exactly_n_bands(mind, bands):
    """The banding must be exact. Measured with rim=0, because the rim term is CONTINUOUS and adds
    levels regardless of the band count -- measuring them together made the first reading look
    broken when it was the probe that was confounded."""
    sph = lambda P: np.linalg.norm(P, axis=-1) - 1.0
    mesh = mind.mesh_from_sdf(sph, bounds=((-1.4, -1.4, -1.4), (1.4, 1.4, 1.4)), res=32)
    V = np.asarray(mesh.vertices, float)
    base = np.tile(np.array([0.55, 0.62, 0.36]), (len(V), 1))
    cols = mind.toon_shade(mesh, base, np.array([0.0, -4.0, 1.5]), bands=bands, rim=0.0)
    assert len(np.unique(np.round(cols.mean(1), 4))) == bands


def test_the_rim_darkens_the_silhouette_not_the_facing_surface(mind):
    """An outline depicts the surface turning away from the eye. Checked geometrically: vertices
    near-perpendicular to the view must come out darker than vertices facing it."""
    import holographic.mesh_and_geometry.holographic_paintlod as pl
    sph = lambda P: np.linalg.norm(P, axis=-1) - 1.0
    mesh = mind.mesh_from_sdf(sph, bounds=((-1.4, -1.4, -1.4), (1.4, 1.4, 1.4)), res=32)
    V = np.asarray(mesh.vertices, float)
    eye = np.array([0.0, -4.0, 1.5])
    base = np.tile(np.array([0.55, 0.62, 0.36]), (len(V), 1))
    cols = mind.toon_shade(mesh, base, eye, bands=3, rim=0.5)
    N = pl._vertex_normals(mesh)
    Vw = eye[None, :] - V
    Vw = Vw / (np.linalg.norm(Vw, axis=1, keepdims=True) + 1e-12)
    ndv = np.abs((N * Vw).sum(1))
    assert cols[ndv < 0.25].mean() < cols[ndv > 0.9].mean()
    # rim=0 must leave the silhouette alone, or the parameter does nothing
    flat = mind.toon_shade(mesh, base, eye, bands=3, rim=0.0)
    assert flat[ndv < 0.25].mean() > cols[ndv < 0.25].mean()


@pytest.mark.parametrize("phrasing,expect", [
    ("put feet on the legs", "Feet on the legs"),
    ("flat cartoon look", "Cel shading"),
    ("comic book look", "Cel shading"),
])
def test_a_stranger_can_find_feet_and_toon(mind, phrasing, expect):
    assert expect.lower() in str(mind.find_capability(phrasing)[:3]).lower(), phrasing

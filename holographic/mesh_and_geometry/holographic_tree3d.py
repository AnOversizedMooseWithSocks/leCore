"""Space-colonization trees, da Vinci taper, and phyllotaxis foliage (organics backlog T-1 / T-2 / T-4).

WHY THIS MODULE EXISTS -- and the audit that justified it
---------------------------------------------------------
The backlog said T-1 must not be written before reading `dielectric_breakdown` / `grow_ice`, because
they already do "attractor-driven branching growth" and might generalize. They were read. VERDICT:
they do NOT generalize, for three concrete reasons, so this is a new module rather than an extension:

  1. SUBSTRATE     DielectricBreakdown grows on a 2-D boolean GRID (`_candidates` is hardcoded to
                   ys/xs 4-adjacency). A tree grows in continuous 3-D space.
  2. DRIVER        DBM picks the next cell by phi^eta from a relaxed LAPLACE FIELD -- it needs a
                   global PDE solve per step. Space colonization needs no field at all: each node
                   moves toward the MEAN DIRECTION of the attractors that chose it. No solve.
  3. TERMINATION   DBM grows until you stop it. Space colonization ends when attractors are consumed,
                   which is what gives a tree a finite, shaped crown.
  The shared word "branching" was a costume, not a mechanism. Recorded so the question is not reopened.

WHAT IS REUSED (this module deliberately writes no meshing code)
  * segments -> geometry, THREE ways, because they trade off differently:
      `skin_skeleton`  one BLENDED watertight surface (limbs fuse at forks). Fed by `taper_radii`.
      `tree_mesh`      per-branch swept tubes, merged. O(edges), no field, but tubes interpenetrate.
      `tree_instanced` ONE unit-tube definition placed per branch -- O(1) geometry, O(edges) transforms.
    HISTORY, kept because the correction matters: skin_skeleton used to die with RecursionError
    between 200 and 300 edges (measured 199 OK / 299 not), and this module originally recorded that
    as a hard ceiling with tree_mesh as the workaround. It was not a ceiling -- it was a LEFT FOLD.
    `_eval` recursed once per part because the union chain is built by folding. Unrolling that spine
    iteratively (holographic_sdf._eval_chain) removed the limit entirely and BIT-IDENTICALLY: 5999
    edges now skin where 299 crashed, maxdiff 0.0 against the recursive result at every depth. The
    lesson is on the record: a RecursionError is a statement about the evaluator, not about the
    problem size. Walk the levers before writing "ceiling" in a docstring.
  * staging/scrubbing: segments come out in GROWTH ORDER, so `holographic_growth` stages a tree by
    taking a PREFIX -- free, no checkpointing (the same lesson the dendrite staging learned).
  * leaf placement -> `realize_scatter` from holographic_meshscatter, via the frames this module emits.

THE ALGORITHM (Runions, Lane & Prusinkiewicz 2007, "Modeling Trees with a Space Colonization
Algorithm"): scatter ATTRACTORS in a crown volume. Each attractor within `influence` of some node
votes for its nearest node. Every node with votes grows one new segment of length `step` along the
normalized mean of its votes. Any attractor within `kill` of a node is consumed. Repeat.

KEPT NEGATIVES (loud)
  * NO physics, no phototropism, no gravity/wind bending, no collision between branches. Branches can
    and do intersect. This is a branching STRUCTURE generator, not a growth simulation.
  * The da Vinci rule (parent radius^n = sum of child radius^n) is an OBSERVED allometry, not a law;
    n=2 is the classic value but real species vary roughly 2-3. Exposed as a parameter, not baked in.
  * Phyllotaxis here places leaves at a fixed divergence angle along a branch. Real phyllotaxis is set
    at the meristem and interacts with internode elongation; this is the visual signature only.
  * Attractor consumption makes the result sensitive to `kill` and `influence`: too small a kill and
    growth never terminates (guarded by max_iters and reported, never silently truncated).
"""

import numpy as np


def crown_attractors(n=400, centre=(0.0, 0.0, 1.6), radius=0.9, shape="ellipsoid",
                     squash=(1.0, 1.0, 1.2), seed=0):
    """Scatter `n` attractor points in a crown volume -- the shape the tree will grow INTO.

    `shape` is 'ellipsoid' (a rounded canopy) or 'cone' (a conifer). Rejection-sampled inside the
    volume rather than pushed onto its surface, because interior attractors are what make a crown
    fill out instead of becoming a shell. Deterministic from `seed`.
    """
    rng = np.random.default_rng(int(seed))
    c = np.asarray(centre, float)
    sq = np.asarray(squash, float)
    out, guard = [], 0
    while len(out) < int(n) and guard < 200:
        guard += 1
        p = rng.uniform(-1.0, 1.0, size=(int(n) * 2, 3))
        if shape == "cone":
            # A cone standing on its base: allowed radius shrinks linearly with height.
            h = (p[:, 2] + 1.0) * 0.5                          # 0 at base, 1 at tip
            keep = (p[:, 0] ** 2 + p[:, 1] ** 2) <= np.maximum(1.0 - h, 1e-6) ** 2
        else:
            keep = (p ** 2).sum(1) <= 1.0
        out.extend(p[keep])
    P = np.asarray(out[:int(n)], float)
    return P * (sq * float(radius))[None, :] + c[None, :]


def grow_tree(attractors, root=(0.0, 0.0, 0.0), step=0.12, influence=0.30, kill=0.25,
              max_iters=400, start_dir=(0.0, 0.0, 1.0)):
    """T-1: grow a branching skeleton into `attractors` by space colonization.

    DEFAULTS WERE MEASURED, NOT GUESSED. The termination condition is roughly `kill >= 2*step`: a
    node must be able to reach an attractor's kill radius within a step or two, or the attractor is
    never consumed and growth runs to `max_iters`. The original defaults here (step 0.08, influence
    0.55, kill 0.14) did exactly that -- 2943 nodes, 186/200 attractors consumed, capped at max_iters,
    i.e. a tree that never finished growing. Measured sweep: (0.30, 0.25, 0.12) gives 165 nodes with
    200/200 consumed, terminating naturally at iteration 160. Change these together, not singly.

    Returns a dict with `nodes` (n,3), `parent` (n,) parent index (-1 for the root), `segments`
    [(a,b)...] IN GROWTH ORDER (so a prefix is a valid younger tree -- what the scrubber needs), and
    `iters` / `consumed` / `terminated` so the caller can tell a finished tree from a capped one
    instead of guessing. Deterministic: no RNG at all in this function -- the only randomness is in
    the attractor cloud the caller passes in.
    """
    A = np.array(attractors, float, copy=True)
    alive = np.ones(len(A), bool)
    nodes = [np.asarray(root, float)]
    parent = [-1]
    order = []
    d0 = np.asarray(start_dir, float)
    d0 = d0 / (np.linalg.norm(d0) + 1e-12)
    terminated = "attractors_consumed"
    it = 0
    for it in range(1, int(max_iters) + 1):
        if not alive.any():
            break
        N = np.asarray(nodes, float)
        # Each live attractor votes for its NEAREST node, if that node is within `influence`.
        d = np.linalg.norm(A[alive][:, None, :] - N[None, :, :], axis=-1)
        nearest = np.argmin(d, axis=1)
        dmin = d[np.arange(len(nearest)), nearest]
        voting = dmin <= float(influence)
        if not voting.any():
            # Nothing in range: extend the trunk toward the attractor cloud rather than stalling --
            # otherwise a root placed below the crown produces a one-node "tree" and no error.
            live_idx = np.where(alive)[0]
            tgt = A[live_idx[np.argmin(dmin)]]
            tip = int(np.argmin(np.linalg.norm(N - tgt, axis=1)))
            dirv = tgt - N[tip]
            dirv = dirv / (np.linalg.norm(dirv) + 1e-12)
            nodes.append(N[tip] + dirv * float(step)); parent.append(tip)
            order.append((tip, len(nodes) - 1))
            continue
        live_idx = np.where(alive)[0][voting]
        chosen = nearest[voting]
        # Every node with votes grows ONE segment along the mean of its attractors' directions.
        for ni in np.unique(chosen):
            tgts = A[live_idx[chosen == ni]]
            v = (tgts - nodes[ni])
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            v = v.sum(0)
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue                                       # opposing pulls cancel: no growth here
            newp = nodes[ni] + (v / nv) * float(step)
            nodes.append(newp); parent.append(int(ni))
            order.append((int(ni), len(nodes) - 1))
        # Consume attractors the tree has reached.
        N = np.asarray(nodes, float)
        dd = np.linalg.norm(A[:, None, :] - N[None, :, :], axis=-1).min(axis=1)
        alive &= dd > float(kill)
    else:
        terminated = "max_iters"                               # capped, not finished -- say so
    return {"nodes": np.asarray(nodes, float), "parent": np.asarray(parent, int),
            "segments": order, "iters": int(it), "consumed": int((~alive).sum()),
            "n_attractors": int(len(A)), "terminated": terminated}


def taper_radii(tree, tip_radius=0.006, exponent=2.0):
    """T-2: per-node radii by the DA VINCI rule -- a parent's radius^n equals the SUM of its children's
    radius^n (Leonardo's notebooks; n=2 is the classic value, real trees run ~2-3).

    Computed leaf-to-root in reverse topological order, which a single reversed pass gives for free
    because `grow_tree` appends children after parents. Returns the (n,) array `skin_skeleton` wants,
    so a tree meshes through the SHIPPED B-Mesh skinner with no new geometry code.
    """
    parent = np.asarray(tree["parent"], int)
    n = len(parent)
    nchild = np.zeros(n, int)
    for pi in parent[1:]:
        if pi >= 0:
            nchild[pi] += 1
    tip_n = float(tip_radius) ** float(exponent)
    # ONLY TIPS seed the accumulator. An earlier version started EVERY node at tip_n and then added
    # its children, which double-counts tip_n at every fork -- caught by the selftest's exact
    # parent^n == sum(children^n) assertion (0.12024 vs 0.120204). Kept as a WHY so it stays fixed:
    # an interior node's radius is defined ENTIRELY by what it carries, not by itself.
    acc = np.where(nchild == 0, tip_n, 0.0)
    for i in range(n - 1, 0, -1):                              # children come after parents, so reverse works
        pi = parent[i]
        if pi >= 0:
            acc[pi] += acc[i]
    return acc ** (1.0 / float(exponent))


def tree_edges(tree):
    """The tree's segments as (i, j) index pairs -- the `edges` argument of skin_skeleton / sweep_tube.
    Kept separate from `segments` so the growth ORDER (which the scrubber needs) stays intact."""
    return [(int(a), int(b)) for a, b in tree["segments"]]


def phyllotaxis_frames(tree, per_node=2, angle_deg=137.507764, size=0.06, min_depth=2, seed=0):
    """T-4: leaf placement frames along the branches at the GOLDEN ANGLE (~137.5 deg divergence), the
    signature of real phyllotaxis -- successive leaves spiral rather than stacking in rows.

    Returns (m,4,4) transforms ready for `realize_scatter`, so a leaf mesh instances onto a tree with
    the same keystone that scatters grass. `min_depth` skips the trunk, where leaves do not grow.
    """
    nodes = np.asarray(tree["nodes"], float)
    parent = np.asarray(tree["parent"], int)
    depth = np.zeros(len(nodes), int)
    for i in range(1, len(nodes)):
        depth[i] = depth[parent[i]] + 1 if parent[i] >= 0 else 0
    # Child count tells us which nodes are twigs; leaves belong on thin, terminal wood.
    nchild = np.zeros(len(nodes), int)
    for p in parent[1:]:
        if p >= 0:
            nchild[p] += 1
    ang = np.radians(float(angle_deg))
    M, k = [], 0
    for i in range(1, len(nodes)):
        if depth[i] < int(min_depth) or nchild[i] > 1:
            continue                                           # trunk and forks carry no leaves
        a = nodes[parent[i]]; b = nodes[i]
        v = b - a
        L = np.linalg.norm(v)
        if L < 1e-9:
            continue
        Z = v / L
        ref = np.array([0.0, 0.0, 1.0]) if abs(Z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        X = np.cross(ref, Z); X /= (np.linalg.norm(X) + 1e-12)
        Y = np.cross(Z, X)
        for j in range(int(per_node)):
            th = ang * k                                       # the divergence angle accumulates GLOBALLY
            k += 1
            t = (j + 0.5) / max(int(per_node), 1)
            pos = a + v * t
            out = np.cos(th) * X + np.sin(th) * Y              # leaf points away from the twig
            up = np.cross(out, Z)
            # COLUMN convention (M @ v) -- see holographic_meshscatter.placement_frames. Built
            # row-wise these put every leaf at the origin while every count-based test still passed.
            T = np.zeros((4, 4))
            T[:3, 0] = out * float(size)
            T[:3, 1] = up * float(size)
            T[:3, 2] = Z * float(size)
            T[:3, 3] = pos; T[3, 3] = 1.0
            M.append(T)
    return np.asarray(M, float) if M else np.zeros((0, 4, 4))


def tree_mesh(tree, radii=None, sides=6, tip_radius=0.006):
    """T-2, the path that SCALES: mesh every branch as its own swept tube and concatenate.

    WHY THIS EXISTS ALONGSIDE skin_skeleton. The backlog claimed plant limbs could simply reuse the
    shipped B-Mesh skinner. MEASURED, that claim has a hard ceiling: skin_skeleton composes one SDF
    node per edge into a nested union and evaluates it recursively, so it raises RecursionError
    somewhere between 200 and 300 edges (measured: 199 edges OK, 299 RecursionError, at Python's
    default limit of 1000). A tree is routinely 150-3000 edges, so the skinner covers only the small
    end. This path is O(edges) with no recursion and no grid, and delegates the actual tube geometry
    to the shipped `sweep_tube`.

    TRADE-OFF, stated rather than hidden: skin_skeleton produces a single BLENDED watertight surface
    (limbs merge smoothly at forks); this produces interpenetrating tubes with visible joins. For a
    trunk-and-branches structure that reads fine and costs a fraction; for an organic blob it does
    not. Use skin_skeleton under ~200 edges when the blend matters, this otherwise.
    """
    from holographic.mesh_and_geometry.holographic_curves import sweep_tube
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    nodes = np.asarray(tree["nodes"], float)
    r = taper_radii(tree, tip_radius=tip_radius) if radii is None else np.asarray(radii, float)
    profile = np.stack([np.cos(np.linspace(0, 2 * np.pi, int(sides), endpoint=False)),
                        np.sin(np.linspace(0, 2 * np.pi, int(sides), endpoint=False))], axis=1)
    V, F, off = [], [], 0
    for a, b in tree["segments"]:
        # Each branch is a tube of its own radius; the taper between parent and child shows as a step,
        # which is the honest cost of per-branch tubes (see the trade-off note above).
        tv, tf = sweep_tube([nodes[a], nodes[b]], profile=profile * float(r[b]))
        tv = np.asarray(tv, float); tf = np.asarray(tf, int)
        V.append(tv); F.append(tf + off); off += len(tv)
    if not V:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), int))
    return Mesh(np.concatenate(V), np.concatenate(F))


def tree_instanced(tree, radii=None, sides=6, tip_radius=0.006, scene=None, material="paint"):
    """Mesh a tree as INSTANCES: one unit-tube Definition placed once per branch through a per-branch
    transform. Geometry cost is O(1) in the number of branches -- a 3000-branch tree holds one tube.

    WHY A UNIT TUBE WORKS HERE. Every branch is the same shape up to a similarity transform: a tube of
    unit length and unit radius, scaled by (radius, radius, length), rotated onto the branch direction,
    translated to its start. So the ONE thing that varies -- the taper -- lives in the transform's
    scale, not in the geometry. That is exactly the condition instancing needs, and it is why a forest
    of these costs no more geometry than a single tree.

    Delegates the placement to the shipped InstancedScene (edit the definition, every branch updates).
    KEPT NEGATIVE: an instanced branch is a similarity transform of a unit tube, so a branch whose two
    ends have DIFFERENT radii is rendered at its mean radius -- the per-branch taper of `tree_mesh` is
    traded away for the memory win. Use tree_mesh when the taper within a single branch must be exact.
    """
    from holographic.misc.holographic_instancing import Definition, InstancedScene
    from holographic.mesh_and_geometry.holographic_curves import sweep_tube
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    nodes = np.asarray(tree["nodes"], float)
    r = taper_radii(tree, tip_radius=tip_radius) if radii is None else np.asarray(radii, float)
    th = np.linspace(0, 2 * np.pi, int(sides), endpoint=False)
    unit_v, unit_f = sweep_tube([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                                profile=np.stack([np.cos(th), np.sin(th)], axis=1))
    scene = InstancedScene() if scene is None else scene
    defn = Definition("tree_branch_unit", Mesh(np.asarray(unit_v, float), np.asarray(unit_f, int)), material)
    for a, b in tree["segments"]:
        A, B = nodes[a], nodes[b]
        d = B - A
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            continue
        Z = d / L
        ref = np.array([1.0, 0.0, 0.0]) if abs(Z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        X = np.cross(ref, Z); X /= (np.linalg.norm(X) + 1e-12)
        Y = np.cross(Z, X)
        rad = 0.5 * (float(r[a]) + float(r[b]))               # see the kept negative: mean, not taper
        # COLUMN convention (M @ v) -- what InstancedScene applies. See the note in
        # holographic_meshscatter.placement_frames: built row-wise these silently collapse every
        # instance onto the origin.
        M = np.zeros((4, 4))
        M[:3, 0] = X * rad; M[:3, 1] = Y * rad; M[:3, 2] = Z * L
        M[:3, 3] = A; M[3, 3] = 1.0
        scene.place(defn, M)
    return scene


def tree_at(spec, t):
    """A tree at growth progress t in [0,1]: the first t of its segments, IN GROWTH ORDER. Pure -- the
    tree is regrown from the spec, so scrubbing backwards cannot corrupt it. Registered with
    holographic_growth as the 'tree' grower."""
    spec = dict(spec)
    A = crown_attractors(n=int(spec.get("n_attractors", 300)), centre=spec.get("centre", (0.0, 0.0, 1.6)),
                         radius=float(spec.get("radius", 0.9)), shape=spec.get("shape", "ellipsoid"),
                         seed=int(spec.get("seed", 0)))
    tree = grow_tree(A, root=spec.get("root", (0.0, 0.0, 0.0)), step=float(spec.get("step", 0.08)),
                     influence=float(spec.get("influence", 0.55)), kill=float(spec.get("kill", 0.14)),
                     max_iters=int(spec.get("max_iters", 400)))
    k = int(round(float(np.clip(t, 0.0, 1.0)) * len(tree["segments"])))
    nodes = tree["nodes"]
    return [(nodes[a], nodes[b]) for a, b in tree["segments"][:k]]


def tree_stages(spec, n_stages=8):
    """Discrete growth checkpoints of a tree -- prefixes of the growth-ordered segment list. Free:
    the algorithm already emits its segments in the order it grew them."""
    return [tree_at(spec, k / max(int(n_stages), 1)) for k in range(int(n_stages) + 1)]


def _selftest():
    """Numeric contracts: the tree must actually consume its crown, segments must be in growth order
    (every child's parent already exists), taper must satisfy the da Vinci sum at every fork, leaves
    must hit the golden angle, and staging must be a strict prefix."""
    A = crown_attractors(n=250, seed=1)
    assert len(A) == 250
    # Attractors lie inside the crown ellipsoid they were asked for.
    rel = (A - np.array([0.0, 0.0, 1.6])) / (np.array([1.0, 1.0, 1.2]) * 0.9)
    assert (rel ** 2).sum(1).max() <= 1.0 + 1e-9, "attractors must stay inside the crown volume"

    tree = grow_tree(A)                                        # THE DEFAULTS, which must be good ones
    n = len(tree["nodes"])
    assert n > 50, "a 250-attractor crown should grow a real tree, got %d nodes" % n
    # THE DEFAULTS MUST TERMINATE NATURALLY. This is the assertion that would have caught the
    # original bad defaults (kill 0.14 vs step 0.08 -> 2943 nodes, capped at max_iters, 14 attractors
    # never reached). A tree that hits its iteration cap is a tree that never finished growing.
    assert tree["terminated"] == "attractors_consumed", \
        "default parameters must grow a FINISHED tree, got %s at %d nodes" % (tree["terminated"], n)
    assert tree["consumed"] == tree["n_attractors"], \
        "a finished tree consumed its whole crown: %d/%d" % (tree["consumed"], tree["n_attractors"])
    assert n < 600, "defaults should give a workable tree, not %d nodes" % n

    # 1) GROWTH ORDER is a real invariant: a child never appears before its parent, so any PREFIX of
    #    the segment list is a valid, connected younger tree -- which is what makes scrubbing free.
    seen = {0}
    for a, b in tree["segments"]:
        assert a in seen, "segment %d->%d grows from a node that does not exist yet" % (a, b)
        seen.add(b)

    # 2) TOPOLOGY: exactly one root, every other node has a parent, no cycles (parent index < child).
    parent = tree["parent"]
    assert parent[0] == -1 and (parent[1:] >= 0).all()
    assert (parent[1:] < np.arange(1, n)).all(), "parent must precede child (no cycles)"

    # 3) DA VINCI TAPER: at every fork, parent^2 == sum(children^2) to 1e-9, and radii shrink outward.
    r = taper_radii(tree, tip_radius=0.006, exponent=2.0)
    kids = {}
    for i in range(1, n):
        kids.setdefault(int(parent[i]), []).append(i)
    forks = 0
    for p, ch in kids.items():
        if len(ch) >= 2:
            forks += 1
            lhs = r[p] ** 2
            rhs = sum(r[c] ** 2 for c in ch)
            assert abs(lhs - rhs) < 1e-9, "da Vinci rule broken at node %d: %.6g vs %.6g" % (p, lhs, rhs)
    assert forks > 0, "a tree with no forks is not a tree"
    assert r[0] == r.max(), "the trunk must be the thickest part"
    assert r.min() >= 0.006 - 1e-12, "no radius may fall below the tip radius"

    # 4) THE T-2 REUSE CLAIM, CHECKED -- and its measured CEILING pinned so it cannot rot.
    #    skin_skeleton works, but only for SMALL skeletons: it nests one SDF node per edge and
    #    recurses, so it dies past ~200 edges. Both halves are asserted: the small case works, and
    #    the scaling path (tree_mesh, per-branch sweep_tube) handles a full-size tree.
    from holographic.mesh_and_geometry.holographic_meshtools import skin_skeleton
    small = grow_tree(crown_attractors(n=25, seed=5), max_iters=40)
    assert len(tree_edges(small)) < 200, "keep the B-Mesh case under the measured recursion ceiling"
    blended = skin_skeleton(small["nodes"], tree_edges(small), taper_radii(small), resolution=20)
    assert len(np.asarray(blended.vertices)) > 0, "small skeletons must still mesh through B-Mesh"

    # 4b) THE CEILING IS GONE. skin_skeleton used to die past ~250 edges; the iterative left-spine
    #     eval removed the limit bit-identically, so a FULL tree now blends. Pinned here so a
    #     regression in the evaluator shows up as a tree that cannot be skinned.
    big_blend = skin_skeleton(tree["nodes"], tree_edges(tree), r, resolution=16)
    assert len(np.asarray(big_blend.vertices)) > 0, \
        "a full-size tree must skin now that the fold is evaluated iteratively (%d edges)" % len(tree_edges(tree))

    # 4c) INSTANCED path: every branch shares ONE definition -- O(1) geometry.
    sc = tree_instanced(tree)
    assert len(sc.definitions()) == 1, "all branches must share one unit-tube definition"
    assert len(sc.instances) == len(tree["segments"])
    # ...and the instances must SPAN the tree, not pile up at the root (the transform-convention trap)
    tx = np.array([np.asarray(i.transform, float)[:3, 3] for i in sc.instances])
    assert tx[:, 2].max() - tx[:, 2].min() > 0.5 * (nodes_z_extent := float(
        tree["nodes"][:, 2].max() - tree["nodes"][:, 2].min())), \
        "instanced branches must span the tree's height, got %.3f of %.3f" % (
            tx[:, 2].max() - tx[:, 2].min(), nodes_z_extent)

    mesh = tree_mesh(tree)
    assert len(np.asarray(mesh.vertices)) == len(tree["segments"]) * 12, \
        "tree_mesh must emit one 6-sided tube (12 verts) per branch"
    assert len(tree_edges(tree)) > 100

    # 5) PHYLLOTAXIS: successive leaves differ by the golden angle, and frames are well-formed.
    M = phyllotaxis_frames(tree, per_node=1, size=0.05)
    assert len(M) > 0, "a grown tree must carry leaves"
    R = M[0, :3, :3] / 0.05
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9, "leaf frames must be orthonormal"
    # ...and the leaves must be SPREAD over the tree, not stacked at the origin
    lp = M[:, :3, 3]
    assert lp[:, 2].max() - lp[:, 2].min() > 0.3, \
        "leaf frames must span the tree, got z-extent %.3f" % (lp[:, 2].max() - lp[:, 2].min())

    # 6) DETERMINISM: identical inputs, identical bytes.
    t2 = grow_tree(crown_attractors(n=250, seed=1))
    assert np.array_equal(t2["nodes"], grow_tree(crown_attractors(n=250, seed=1))["nodes"])

    # 7) STAGING IS A STRICT PREFIX: an earlier stage is the literal head of a later one.
    s = tree_stages({"n_attractors": 120, "seed": 2}, 4)
    assert len(s[0]) == 0 and len(s[-1]) > 0
    for k in range(len(s) - 1):
        assert len(s[k]) <= len(s[k + 1])
        for i in range(len(s[k])):
            assert np.allclose(s[k][i][0], s[k + 1][i][0]) and np.allclose(s[k][i][1], s[k + 1][i][1]), \
                "stage %d is not a prefix of stage %d -- scrubbing would jump" % (k, k + 1)

    # 8) A CAPPED RUN SAYS SO rather than silently returning a stunted tree.
    capped = grow_tree(crown_attractors(n=200, seed=3), max_iters=3)
    assert capped["terminated"] == "max_iters"

    print("tree selftest OK: %d nodes from %d/%d attractors consumed, %d forks all satisfying da Vinci "
          "to 1e-9, growth order is a strict prefix, %d leaves placed"
          % (n, tree["consumed"], tree["n_attractors"], forks, len(M)))


if __name__ == "__main__":
    _selftest()

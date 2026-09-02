"""Part 14 of UnifiedMind's faculty surface -- ORGANICS: crystals, grass/scatter, plants, growth scrubbing, idle.

NOT A STANDALONE MODULE. This is one slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which is still the only import path anyone uses.

WHY THIS PART EXISTS
--------------------
The organics backlog (crystals / grass / trees / plants / creatures) found that most of the work was
PROMOTION, not invention: `grow_plant`, `turtle_to_segments`, `segments_to_scene`, `greeble_panel`,
`procedural_object`, `object_to_mesh`, `greeble_mesh` and `scatter_on_terrain` were all shipped and
tested but had NO faculty, so `find_capability("procedural tree")` returned Texture graph and
fit_shape -- a pure discoverability failure with working code sitting behind it. This part wires them,
plus the six genuinely new modules the audit justified (bravais, meshscatter, growth, creatureidle,
tree3d, creatureskin).

Every method here DELEGATES; none reimplements. Each is default-off in the sense that it is a new name
-- no existing faculty's behaviour changes, and no existing emitted bytes flip.

KEPT NEGATIVE, so nobody "tidies" it: these part classes are NOT a public API and must never be
imported or subclassed directly. They carry no `__init__` and assume the state UnifiedMind.__init__
builds; instantiated alone they would fail on the first attribute access.
"""

from holographic.unified import check_part


class _UnifiedPart14:

    # ------------------------------------------------------------------ crystals (C-1 / C-3) --
    def lattice_basis(self, system, a=1.0, b=None, c=None, alpha=90.0, beta=90.0, gamma=90.0, centring="P"):
        """The (3,3) cell basis + centring motif for one of the 14 BRAVAIS LATTICES: pick one of the 7
        crystal systems (cubic, tetragonal, orthorhombic, hexagonal, trigonal, monoclinic, triclinic)
        and a centring (P/I/F/C). The system fills in its own constraints; an illegal system+centring
        pair is refused rather than silently built. See holographic_bravais.lattice_basis."""
        import holographic.mesh_and_geometry.holographic_bravais as _bv
        return _bv.lattice_basis(system, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, centring=centring)

    def lattice_sites(self, system="cubic", centring="P", a=1.0, extent=2, **cell):
        """The (N,3) atom SITES of a crystal lattice -- the point set every downstream faculty eats:
        metaball_mesh(centers=...) for ball-and-stick, sweep_tube along lattice_bonds for the bonds,
        scatter_mesh for instanced unit cells. See holographic_bravais.lattice_points."""
        import holographic.mesh_and_geometry.holographic_bravais as _bv
        basis, motif = _bv.lattice_basis(system, a=a, centring=centring, **cell)
        return _bv.lattice_points(basis, motif, extent=extent)

    def lattice_bonds(self, points, tol=1e-6):
        """The nearest-neighbour BOND pairs of a lattice: [(i,j)...] at the minimum inter-site distance,
        plus that distance. Feed to sweep_tube for a ball-and-stick model. O(N^2) by design -- exact and
        honest at display sizes. See holographic_bravais.neighbor_pairs."""
        import holographic.mesh_and_geometry.holographic_bravais as _bv
        return _bv.neighbor_pairs(points, tol=tol)

    def crystal_habit(self, system="cubic", miller_faces=((1, 1, 1),), sizes=0.5, **cell):
        """A faceted crystal FORM as an SDF: the intersection of half-spaces along Miller-index plane
        normals (computed in the reciprocal basis, which is what makes non-cubic cells come out right).
        Returns sdf(P)->distance, ready for mesh_from_sdf or the raymarcher. See
        holographic_bravais.crystal_habit."""
        import holographic.mesh_and_geometry.holographic_bravais as _bv
        return _bv.crystal_habit(system, miller_faces, sizes, **cell)

    def crystal_systems(self):
        """The 7 crystal systems, their cell-parameter constraints, and the centrings each allows --
        the grouping that makes 14 Bravais lattices out of 7 systems. See holographic_bravais.SYSTEMS."""
        import holographic.mesh_and_geometry.holographic_bravais as _bv
        return dict(_bv.SYSTEMS)

    # ------------------------------------------------------- grass / scatter on a mesh (S-1 / S-2) --
    def scatter_mesh(self, surface, source, count, seed=0, scale=1.0, scale_jitter=0.0, align=1.0,
                     density=None, relax=False, radius=None, mode="merge", variants=None,
                     holographic=False, dim=1024, cell_size=0.25):
        """GRASS ON A LAWN, one call: scatter `source` (a mesh) over the SURFACE AREA of `surface`
        `count` times and get real geometry back. Area-weighted so big triangles get proportionally
        more; mode='merge' bakes one mesh, mode='instanced' shares one definition across n transforms.
        `variants` scatters a POOL of meshes (plant permutations). `holographic=True` also returns a
        content-addressable layer vector, so the field can be queried by region with no spatial index.
        See holographic_meshscatter.scatter_mesh."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.scatter_mesh(surface, source, count, seed=seed, scale=scale, scale_jitter=scale_jitter,
                                align=align, density=density, relax=relax, radius=radius, mode=mode,
                                variants=variants, holographic=holographic, dim=dim, cell_size=cell_size)

    def sample_mesh_surface(self, mesh, count, density=None, seed=0, relax=False, radius=None):
        """Sample points on a MESH surface, area-weighted, with per-point normals and tangents --
        the scatter's first stage, exposed because 'where would things land' is its own question.
        `relax=True` blue-noise-thins the result (returns FEWER points, never pads). See
        holographic_meshscatter.sample_mesh_surface."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.sample_mesh_surface(mesh, count, density=density, seed=seed, relax=relax, radius=radius)

    def placement_frames(self, points, normals, tangents=None, scale=1.0, scale_jitter=0.0,
                         yaw_jitter=True, align=1.0, seed=0):
        """Surface samples -> (n,4,4) instance TRANSFORMS: up along the surface normal (blend toward
        world-up with `align`), random yaw, jittered scale. See holographic_meshscatter.placement_frames."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.placement_frames(points, normals, tangents=tangents, scale=scale,
                                    scale_jitter=scale_jitter, yaw_jitter=yaw_jitter, align=align, seed=seed)

    def realize_scatter(self, source, transforms, mode="merge", scene=None, variants=None, seed=0,
                        material="paint"):
        """PLACEMENTS -> GEOMETRY, the keystone: given a source mesh and (n,4,4) transforms, return one
        merged Mesh (mode='merge') or an InstancedScene sharing one definition (mode='instanced').
        Serves grass, plant permutations, rocks and crystal unit cells alike. Kept negative: merge cost
        is LINEAR in instances. See holographic_meshscatter.realize_scatter."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.realize_scatter(source, transforms, mode=mode, scene=scene, variants=variants,
                                   seed=seed, material=material)

    def grass_blade(self, height=0.3, width=0.02, segments=4, bend=0.35, taper=0.15):
        """One RIBBON grass blade: a tapered, drooping quad strip, deliberately tiny (2*segments tris)
        because a lawn multiplies its cost by n. Use as the `source` for scatter_mesh, or build several
        with different height/bend as `variants`. See holographic_meshscatter.grass_blade."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.grass_blade(height=height, width=width, segments=segments, bend=bend, taper=taper)

    def triangle_areas(self, mesh):
        """Per-triangle area of a mesh -- the weight that makes surface scattering uniform over AREA
        rather than over the face list. See holographic_meshscatter.triangle_areas."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.triangle_areas(mesh)

    # ------------------------------------------- L-systems: plants & trees, PROMOTED (backlog O-1) --
    def grow_plant(self, lsystem, iterations=3, angle_deg=25.0, step=1.0, radius=0.04):
        """PROCEDURAL TREE / PLANT from an L-system: expand the rules, walk them with a 3-D turtle,
        assemble a scenegraph and flatten to a Mesh. Returns (mesh, segments, scene). Was shipped but
        unwired -- this is the door. See holographic_grammar.grow_plant."""
        import holographic.agents_and_reasoning.holographic_grammar as _g
        return _g.grow_plant(lsystem, iterations, angle_deg=angle_deg, step=step, radius=radius)

    def turtle_to_segments(self, symbols, angle_deg=25.0, step=1.0):
        """Interpret an L-system string with a 3-D TURTLE -> [(start, end)...] branch segments (the
        skeleton, before any meshing). F draw, f move, +/- yaw, &/^ pitch, \\ and / roll, [ ] branch.
        See holographic_grammar.turtle_to_segments."""
        import holographic.agents_and_reasoning.holographic_grammar as _g
        return _g.turtle_to_segments(symbols, angle_deg=angle_deg, step=step)

    def segments_to_scene(self, segments, radius=0.04):
        """Branch segments -> a scenegraph of tapered limbs, ready to flatten to a Mesh. The step
        between a plant SKELETON and plant GEOMETRY. See holographic_grammar.segments_to_scene."""
        import holographic.agents_and_reasoning.holographic_grammar as _g
        return _g.segments_to_scene(segments, radius=radius)

    def greeble_panel(self, *args, **kw):
        """Recursive GREEBLE detail on a panel (the demoscene/sci-fi surface-detail grammar) -- the
        non-botanical half of the procedural grammar. See holographic_grammar.greeble_panel."""
        import holographic.agents_and_reasoning.holographic_grammar as _g
        return _g.greeble_panel(*args, **kw)

    def procedural_object(self, seed=0, complexity=3):
        """A whole 3-D OBJECT from a single integer seed: the demoscene composition of SDF algebra,
        grammar and noise. Returns an SDF node -- pair with object_to_mesh. See
        holographic_procgen.procedural_object."""
        import holographic.io_and_interop.holographic_procgen as _pg
        return _pg.procedural_object(seed=seed, complexity=complexity)

    def object_to_mesh(self, sdf_node, bounds=((-2, -2, -2), (2, 2, 2)), res=40):
        """March a procedural SDF object into a Mesh. See holographic_procgen.object_to_mesh."""
        import holographic.io_and_interop.holographic_procgen as _pg
        return _pg.object_to_mesh(sdf_node, bounds=bounds, res=res)

    def greeble_mesh(self, base_mesh, seed=0, density=0.7, max_height=0.15, footprint=0.5):
        """Encrust a mesh with procedural GREEBLES (panels, boxes, vents) -- the sci-fi detail pass.
        See holographic_procgen.greeble_mesh."""
        import holographic.io_and_interop.holographic_procgen as _pg
        return _pg.greeble_mesh(base_mesh, seed=seed, density=density, max_height=max_height,
                                footprint=footprint)

    def scatter_on_terrain(self, terrain, instance_fn, count=12, seed=0, scale_range=(0.5, 1.0),
                           jitter_yaw=True):
        """Scatter instances across a TERRAIN height field (the vegetated-landscape path). For an
        arbitrary mesh surface use scatter_mesh instead. See holographic_procgen.scatter_on_terrain."""
        import holographic.io_and_interop.holographic_procgen as _pg
        return _pg.scatter_on_terrain(terrain, instance_fn, count=count, seed=seed,
                                      scale_range=scale_range, jitter_yaw=jitter_yaw)

    # ------------------------------------------------------- growth staging & scrubbing (G-1) --
    def grow_stages(self, kind, spec, n_stages=8):
        """SCRUB GROWTH: discrete checkpoints of a grower ('plant', 'crystal', 'dendrite') from nothing
        to finished form -- a list of n_stages+1 stages. Growth is append-only, so consecutive deltas
        are maximally sparse: feed straight to frame_cache / timeline. See holographic_growth.grow_stages."""
        import holographic.mesh_and_geometry.holographic_growth as _gr
        return _gr.grow_stages(kind, spec, n_stages=n_stages)

    def grow_at(self, kind, spec, t):
        """The state of a grower at continuous progress t in [0,1]. PURE -- the same (kind, spec, t)
        always gives the same bytes regardless of what was scrubbed before, so a UI slider needs no
        reset and scrubbing backwards is safe. Kept negative: t is growth PROGRESS, not physical time.
        See holographic_growth.grow_at."""
        import holographic.mesh_and_geometry.holographic_growth as _gr
        return _gr.grow_at(kind, spec, t)

    def growth_report(self, kind, spec, n_stages=6, tol=1e-6):
        """IS THIS GROWTH BEING DONE CORRECTLY? Checks the two properties that make a scrub trustworthy
        and returns them as data: `purity` (no hidden playback state) and `monotone` (nothing retracts
        or teleports between stages), plus per-stage counts and the first stage where containment
        broke. monotone=False is information, not automatically a bug -- a rule that rescales as it
        grows legitimately breaks containment. See holographic_growth.growth_report."""
        import holographic.mesh_and_geometry.holographic_growth as _gr
        return _gr.growth_report(kind, spec, n_stages=n_stages, tol=tol)

    def grow_kinds(self):
        """The growers that support staged scrubbing, by name. See holographic_growth.grow_kinds."""
        import holographic.mesh_and_geometry.holographic_growth as _gr
        return _gr.grow_kinds()

    # ------------------------------------------ space-colonization trees + foliage (T-1 / T-2 / T-4) --
    def crown_attractors(self, n=400, centre=(0.0, 0.0, 1.6), radius=0.9, shape="ellipsoid",
                         squash=(1.0, 1.0, 1.2), seed=0):
        """The attractor cloud a tree grows INTO -- the crown VOLUME, 'ellipsoid' (canopy) or 'cone'
        (conifer). Interior points, not a shell, so the crown fills out. See
        holographic_tree3d.crown_attractors."""
        import holographic.mesh_and_geometry.holographic_tree3d as _t3
        return _t3.crown_attractors(n=n, centre=centre, radius=radius, shape=shape, squash=squash, seed=seed)

    def grow_tree(self, attractors, root=(0.0, 0.0, 0.0), step=0.12, influence=0.30, kill=0.25,
                  max_iters=400, start_dir=(0.0, 0.0, 1.0)):
        """PROCEDURAL TREE by SPACE COLONIZATION (Runions et al. 2007): branches grow toward an
        attractor cloud and consume it, which gives a natural limb distribution and a shaped crown.
        Segments come out IN GROWTH ORDER, so a prefix is a valid younger tree (free scrubbing).
        Check `terminated`: 'max_iters' means the tree never finished. See holographic_tree3d.grow_tree."""
        import holographic.mesh_and_geometry.holographic_tree3d as _t3
        return _t3.grow_tree(attractors, root=root, step=step, influence=influence, kill=kill,
                             max_iters=max_iters, start_dir=start_dir)

    def taper_radii(self, tree, tip_radius=0.006, exponent=2.0):
        """Per-node branch radii by the DA VINCI rule: a parent's radius^n equals the SUM of its
        children's radius^n (n=2 classic; real trees ~2-3). Returns the array skin_skeleton wants.
        See holographic_tree3d.taper_radii."""
        import holographic.mesh_and_geometry.holographic_tree3d as _t3
        return _t3.taper_radii(tree, tip_radius=tip_radius, exponent=exponent)

    def tree_mesh(self, tree, radii=None, sides=6, tip_radius=0.006):
        """Mesh a tree as per-branch swept tubes -- the path that SCALES. Kept negative: the shipped
        skin_skeleton gives a smoother BLENDED surface but recurses per edge and dies between 200 and
        300 edges (measured), while trees are routinely larger. See holographic_tree3d.tree_mesh."""
        import holographic.mesh_and_geometry.holographic_tree3d as _t3
        return _t3.tree_mesh(tree, radii=radii, sides=sides, tip_radius=tip_radius)

    def tree_instanced(self, tree, radii=None, sides=6, tip_radius=0.006, scene=None, material="paint"):
        """Mesh a tree as INSTANCES: one unit-tube definition placed per branch, so geometry cost is
        O(1) in branch count (measured: 420 branches, 5040 merged verts vs 12 stored -- 420x less).
        Kept negative: a branch is a similarity transform of a unit tube, so its two ends render at
        their MEAN radius; use tree_mesh when within-branch taper must be exact. See
        holographic_tree3d.tree_instanced."""
        import holographic.mesh_and_geometry.holographic_tree3d as _t3
        return _t3.tree_instanced(tree, radii=radii, sides=sides, tip_radius=tip_radius,
                                  scene=scene, material=material)

    def phyllotaxis_frames(self, tree, per_node=2, angle_deg=137.507764, size=0.06, min_depth=2, seed=0):
        """LEAF placement frames along the branches at the GOLDEN ANGLE (~137.5 deg) -- real
        phyllotaxis spirals rather than stacking in rows. Returns (m,4,4) transforms for
        realize_scatter, so leaves instance onto a tree with the same keystone that scatters grass.
        Skips the trunk and forks. See holographic_tree3d.phyllotaxis_frames."""
        import holographic.mesh_and_geometry.holographic_tree3d as _t3
        return _t3.phyllotaxis_frames(tree, per_node=per_node, angle_deg=angle_deg, size=size,
                                      min_depth=min_depth, seed=seed)

    # --------------------------------- Spore-style creature skin + spine editing (R-1 / R-2) --
    def creature_metaballs(self, creature, spec=None, spacing=1.0, limb_taper=0.6, head=True):
        """The Spore-style METABALL SKIN of a creature: (centers, radii, bone_of). Balls are spaced
        from their own radius, so stretching a spine or limb ADDS balls instead of stretching a shape.
        `bone_of` says which bone made each ball -- what skin-weight binding needs. See
        holographic_creatureskin.creature_metaballs."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.creature_metaballs(creature, spec, spacing=spacing, limb_taper=limb_taper, head=head)

    def creature_skin_mesh(self, creature, spec=None, spacing=1.0, resolution=40, pad=0.35, blend=1.0, warn=True):
        """A creature -> a smooth BLENDED metaball skin mesh: limbs FLOW into the torso instead of
        intersecting it, the visual difference between a capsule union and the Spore-style skin.
        Handles per-ball radii, which the shipped metaball_mesh (one radius for all) cannot. See
        holographic_creatureskin.creature_metaball_mesh."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.creature_metaball_mesh(creature, spec, spacing=spacing, resolution=resolution,
                                          pad=pad, blend=blend, warn=warn)

    def spine_profile(self, spec, radii):
        """ADJUST THICKNESS ALONG THE SPINE: replace a spec's scalar spine radius with a per-node
        radius profile (array or callable f(t)->r) -- a fat belly and a thin neck on one creature.
        Returns a NEW spec; scalar specs keep working unchanged. See
        holographic_creatureskin.spine_profile."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.spine_profile(spec, radii)

    def extend_spine(self, spec, n=1, keep_segment_length=True):
        """EXTEND THE SPINE by `n` segments (the 'drag the tail out' edit). Returns a NEW spec, so an
        editor gets undo for free and the result stays serialisable. Limb positions are preserved
        across the resolution change. See holographic_creatureskin.extend_spine."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.extend_spine(spec, n=n, keep_segment_length=keep_segment_length)

    def insert_spine_node(self, spec, at=0.5):
        """SUBDIVIDE the spine at fraction `at`: more resolution, same length and shape. Any radius
        profile is resampled so thickness survives. See holographic_creatureskin.insert_node."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.insert_node(spec, at=at)

    def set_spine_radius(self, spec, at, radius, falloff=0.0):
        """THICKEN OR THIN the spine at a fraction along it; `falloff` > 0 blends into neighbours for
        a smooth belly rather than one bulging ring. See holographic_creatureskin.set_radius."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.set_radius(spec, at, radius, falloff=falloff)

    def reshape_spine(self, spec, curve=None, length=None, axis=None):
        """RESHAPE the spine as a whole: its arch (`curve`), `length`, or `axis`. Kept negative:
        per-node free positioning is deliberately not offered -- the spine is GENERATED from these
        parameters, so an arbitrary node offset would make the spec stop describing its own curve.
        See holographic_creatureskin.move_node."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.move_node(spec, curve=curve, length=length, axis=axis)

    # ------------- HOLOGRAPHIC creature parts, symmetry, skin weights (R-3 / R-4 / R-7) --
    def part_library(self, dim=None, seed=0, expect_sockets=32, alpha=0.90):
        """A RIGBLOCK library: named parts, each an atom in a codebook, each carrying its own authored
        deformation handles so a part deforms only within ranges its author sanctioned. Parts are
        derived from their names, so the same name is the same vector in every session with no stored
        state. See `dim=None` PRICES the dimension from `expect_sockets` via the capacity law rather than
        guessing -- a part assembly is a superposed pair memory, so the closed form applies. See
        holographic_creatureparts.PartLibrary."""
        import holographic.mesh_and_geometry.holographic_creatureparts as _cp
        return _cp.PartLibrary(dim=dim, seed=seed, expect_sockets=expect_sockets, alpha=alpha)

    def stocked_part_library(self, sockets, dim=None, seed=0, params=None):
        """Build geometry for every part the `sockets` ask for and DEFINE it into a fresh library
        (holographic_creaturepartlib.stock_for_sockets) -- the missing half of the render path.
        part_library() returns an EMPTY codebook by design (define(name, geometry=) is how geometry
        gets in), so place_parts against an unstocked library places NOTHING and reports missed: []
        -- placement succeeded; there was nothing to place. `sockets` is
        build_creature(..., parts=True)['sockets']. Postchecks each part: a 0-vertex build is
        reported in 'empty', an unknown name in 'missed' (never raised -- one exotic socket must not
        cost the creature its feet). Returns {"library", "stocked", "missed", "empty"}.
        KEPT NEG: stocking the WHOLE palette for a 3-part creature pays ~10x the geometry build for
        vectors that sit unread; the sockets are the demand signal."""
        import holographic.mesh_and_geometry.holographic_creaturepartlib as _pl
        return _pl.stock_for_sockets(sockets, dim=dim, seed=seed, params=params)

    def attach_part(self, assembly, socket, part_name, library, symmetry=None, n=2):
        """ATTACH A PART to a socket, holographically: the layout becomes ONE vector,
        bundle_bind(socket_roles, part_atoms), so it is queryable (what_is_at), comparable (cosine
        between two creatures) and composable with anything else that eats a hypervector. With
        `symmetry` ('bilateral' or 'radial') the group generates every mirrored/rotated socket at once.
        Returns (assembly_dict, vector). See holographic_creatureparts.attach."""
        import holographic.mesh_and_geometry.holographic_creatureparts as _cp
        if symmetry:
            return _cp.attach_symmetric(assembly, socket, part_name, library, symmetry, n)
        return _cp.attach(assembly, socket, part_name, library)

    def what_is_at(self, vec, socket, library):
        """Query an assembly VECTOR: unbind the socket role and clean up to the nearest real part.
        Returns (part_name, cosine) -- the cosine is returned, not hidden, so an overloaded bundle is
        visible rather than silently wrong. See holographic_creatureparts.what_is_at."""
        import holographic.mesh_and_geometry.holographic_creatureparts as _cp
        return _cp.what_is_at(vec, socket, library)

    def assembly_report(self, assembly, library):
        """MEASURE the part bundle: recall every socket from the vector and check it against what was
        authored. Returns accuracy, mean cosine and min margin -- the number that predicts when the
        next attachment starts returning garbage. A bundle degrades silently; this makes it loud.
        See holographic_creatureparts.assembly_report."""
        import holographic.mesh_and_geometry.holographic_creatureparts as _cp
        return _cp.assembly_report(assembly, library)

    def symmetry_transforms(self, kind="bilateral", n=2, axis=(0.0, 0.0, 1.0)):
        """The geometric transforms of a symmetry GROUP: a reflection for bilateral, n rotations for
        radial. Feed to transform_mesh, which repairs face winding under a reflection. Generalises the
        creature rig's single mirror plane. See holographic_creatureparts.symmetry_transforms."""
        import holographic.mesh_and_geometry.holographic_creatureparts as _cp
        return _cp.symmetry_transforms(kind=kind, n=n, axis=axis)

    def skin_weights_from_balls(self, vertices, centers, radii, bone_of, dim=256, seed=0,
                                falloff=2.0, max_bones=4):
        """SKIN WEIGHTS from metaball provenance -- the soft mixture of experts skinning already is.
        Each vertex bundles the atoms of the balls near it; a bone's weight is its share of that
        bundle. `creature_metaballs` returns `bone_of` for exactly this. Returns (idx, w, names, book)
        in the compact form linear_blend_skin_indexed consumes. Kept negative: distance-based, not
        geodesic -- touching limbs bleed weight, which is why Spore's fat torsos sheared. See
        holographic_creatureparts.skin_weights."""
        import holographic.mesh_and_geometry.holographic_creatureparts as _cp
        return _cp.skin_weights(vertices, centers, radii, bone_of, dim=dim, seed=seed,
                                falloff=falloff, max_bones=max_bones)

    # -------------------------------------- holographic scatter layer + variants + ribbons --
    def scatter_layer_vector(self, transforms, source_name="instance", dim=1024, cell_size=0.25, seed=0):
        """The CONTENT-ADDRESSABLE form of a scatter: each placement bound to its region code, all
        bundled into one layer vector -- ask "is anything scattered near here?" with no spatial index.
        See holographic_meshscatter.scatter_layer_vector."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.scatter_layer_vector(transforms, source_name=source_name, dim=dim,
                                        cell_size=cell_size, seed=seed)

    def region_occupancy(self, layer, instance, point, dim=1024, cell_size=0.25, seed=0):
        """Query a scatter layer vector at a point: unbind the region code, read the cosine against
        the instance atom. Kept negative: a coarse cell hash, so region-addressable, not per-instance
        recall. See holographic_meshscatter.region_occupancy."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.region_occupancy(layer, instance, point, dim=dim, cell_size=cell_size, seed=seed)

    def spec_variant(self, spec, seed, jitter=0.25, keys=None):
        """A deterministic VARIATION of a grower spec -- 'twenty different ferns' as a loop over seeds
        rather than twenty authored assets, and never stored because it is a pure function of
        (spec, seed). See holographic_growth.variant."""
        import holographic.mesh_and_geometry.holographic_growth as _gr
        return _gr.variant(spec, seed, jitter=jitter, keys=keys)

    def spec_variant_pool(self, spec, n, jitter=0.25, keys=None, base_seed=0):
        """`n` variants of a spec, the generated pool that feeds realize_scatter's `variants`. Variant
        0 is the unchanged spec. See holographic_growth.variant_pool."""
        import holographic.mesh_and_geometry.holographic_growth as _gr
        return _gr.variant_pool(spec, n, jitter=jitter, keys=keys, base_seed=base_seed)

    def strand_ribbons(self, strands, width=0.02, taper=0.15, twist=0.0):
        """Turn STRAND chains (as groom_hair / simulate_hair produce) into RIBBON geometry -- so grass
        inherits the whole shipped groom pipeline: root on a surface, simulate with PBD, blow with
        curl-noise wind, then ribbon. See holographic_meshscatter.strand_ribbons."""
        import holographic.mesh_and_geometry.holographic_meshscatter as _ms
        return _ms.strand_ribbons(strands, width=width, taper=taper, twist=twist)

    # ------------------------------------------- procedural paint + scatter bake/LOD (R-9 / S-4) --
    def paint_creature(self, vertices, idx=None, w=None, names=None, pattern=None, pattern_scale=6.0,
                       palette=None, base=(0.55, 0.45, 0.35), accent=(0.15, 0.12, 0.10),
                       seed=0, bone_mix=0.6):
        """PAINT MODE: procedural per-vertex colours mixing a BONE tint (anatomy -- markings follow the
        rig, so they travel with a pose instead of swimming through world-space noise) with a PATTERN
        (stripes/dots/checker/noise, via the shipped pattern_field). Works with or without weights.
        Kept negative: per-vertex, so detail finer than the mesh's vertex spacing needs a texture bake.
        See holographic_paintlod.paint_creature."""
        import holographic.mesh_and_geometry.holographic_paintlod as _pl
        return _pl.paint_creature(vertices, idx=idx, w=w, names=names, pattern=pattern,
                                  pattern_scale=pattern_scale, palette=palette, base=base,
                                  accent=accent, seed=seed, bone_mix=bone_mix)

    def bone_tint(self, idx, w, names, palette=None, seed=0):
        """Per-vertex colour straight from the skin-weight bundle: each bone gets a deterministic hue
        and a vertex is the weighted mix, so a limb is automatically its own colour region. See
        holographic_paintlod.bone_tint."""
        import holographic.mesh_and_geometry.holographic_paintlod as _pl
        return _pl.bone_tint(idx, w, names, palette=palette, seed=seed)

    def scatter_lod(self, transforms, distance, near=8.0, far=60.0, min_keep=0.05, seed=0):
        """Thin a scattered population by DISTANCE, deterministically: the set kept far away is a
        strict SUBSET of the set kept near, so a camera dollying in and out reveals and hides the same
        blades instead of making them flicker. Returns (kept, keep_fraction). Kept negative: this
        REMOVES geometry -- it is a quality trade, not a free win. See holographic_paintlod.scatter_lod."""
        import holographic.mesh_and_geometry.holographic_paintlod as _pl
        return _pl.scatter_lod(transforms, distance, near=near, far=far, min_keep=min_keep, seed=seed)

    def scatter_bake(self, transforms, source, lod_targets=(0.5, 0.25), seed=0):
        """BAKE a scatter once and serve any distance from the cache: placements plus a decimated LOD
        chain, so a distance query costs a hash compare and a level index rather than a re-scatter.
        `.report()` gives EXACT triangle counts against the full-resolution baseline, and names any
        level that failed to decimate instead of letting thinning masquerade as mesh LOD. Measured
        (3000 blades, 5 seeds): 0.0879s baseline vs 0.0058s at distance 60, CIs disjoint. See
        holographic_paintlod.ScatterBake."""
        import holographic.mesh_and_geometry.holographic_paintlod as _pl
        return _pl.ScatterBake(transforms, source, lod_targets=lod_targets, seed=seed)

    # ------------------------------------------- organic creature MATERIALS (layered anatomy) --
    def creature_material(self, taxon, axis=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), seed=0,
                          tint=None, structure_strength=1.0, wetness=1.0, iridescence=1.0,
                          film_nm=340.0, n_film=1.56, body_length=None, cells_across=None):
        """SKIN for a creature by taxon -- 'reptile'/'fish' (scales), 'amphibian' (glands, wet),
        'insect' (chitin plates), 'worm' (annuli), 'mammal' (pores). Returns channel FIELDS
        (colour/roughness/reflect/structure) evaluated in the creature's own BODY frame, so scale rows
        elongate down the body and travel with it instead of swimming through world space. For fish and insect the returned `colour_socket` carries REAL thin-film interference
        iridescence (a ViewSocket, since it needs the eye-to-surface angle); `iridescence=0` disables
        it. See holographic_creaturematerial.creature_material."""
        import holographic.materials_and_texture.holographic_creaturematerial as _cm
        return _cm.creature_material(taxon, axis=axis, origin=origin, seed=seed, tint=tint,
                                     structure_strength=structure_strength, wetness=wetness,
                                     iridescence=iridescence, film_nm=film_nm, n_film=n_film, body_length=body_length, cells_across=cells_across)

    def anatomy_stack(self, taxon, with_bone=None, with_organ=True, seed=0, **kw):
        """The LAYERED anatomy of an integument, bottom to top: [bone] -> [organ] -> dermis ->
        epidermis -> coat, composed through the shipped LayeredMaterial so the base<diffuse<specular<
        coat order is enforced at compose time. An INSECT refuses a bone layer -- its rigid structure
        is the exoskeleton, so stacking a skeleton under one would model an animal that does not
        exist. Kept negative: interior layers only show through TRANSLUCENCY (`interior_visible`) --
        this is not an x-ray. See holographic_creaturematerial.anatomy_stack."""
        import holographic.materials_and_texture.holographic_creaturematerial as _cm
        return _cm.anatomy_stack(taxon, with_bone=with_bone, with_organ=with_organ, seed=seed, **kw)

    def creature_surface_material(self, taxon, axis=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0),
                                  seed=0, **kw):
        """A SurfaceMaterial whose colour/roughness/reflect sockets are a taxon's fields -- ready for
        render_surface, which resolves each socket per hit, so the scale pattern becomes a true solid
        3-D texture that wraps the body with no UV unwrap. See
        holographic_creaturematerial.surface_material_for."""
        import holographic.materials_and_texture.holographic_creaturematerial as _cm
        return _cm.surface_material_for(taxon, axis=axis, origin=origin, seed=seed, **kw)

    def creature_taxa(self):
        """The available integument families with their structure, coat and whether they even HAVE a
        skeleton. See holographic_creaturematerial.taxa."""
        import holographic.materials_and_texture.holographic_creaturematerial as _cm
        return _cm.taxa()

    def creature_skin_field(self, creature, spec=None, spacing=1.0, blend=1.0, distance=True,
                            smooth_k=0.06, warp=None):
        """The creature's metaball skin as an SDF callable f(P)->distance -- the input the
        high-quality renderers (render_surface / path_trace) take. Previously this field was built
        only INSIDE the mesher, so the quality render path had no door to it. The returned CreatureField is
        callable AND carries .eval/.ids/.bounds, so one object feeds the mesher, the raymarcher and
        the path tracer. See holographic_creatureskin.creature_field."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.creature_field(creature, spec, spacing=spacing, blend=blend,
                                  distance=distance, smooth_k=smooth_k, warp=warp)

    # ------------------------------------------- body SHAPING, mesh quality, and part fusion --
    def section_warp(self, creature, width=1.0, depth=1.0, ridge=0.0, belly=0.0):
        """SHAPE THE BODY'S CROSS-SECTION -- the difference between a tapered tube and an animal.
        Metaballs are spheres, so every cross section is a CIRCLE and no profile edit changes that.
        This warps SPACE in the body frame instead of using ellipsoid primitives, so every ball
        becomes the same ellipse for free: `width` broadens across, `depth` front-to-back, `ridge`
        raises a spinal crest, `belly` flattens the underside. MEASURED: no evaluation cost over
        spheres (0.2306s vs 0.2433s / 40k pts), which refutes the received wisdom that
        non-spherical metaballs are slower -- true of ellipsoid PRIMITIVES, false of a space warp.
        See holographic_creatureskin.section_warp."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.section_warp(creature, width=width, depth=depth, ridge=ridge, belly=belly)

    def skin_quality(self, creature, spec=None, spacing=1.0, resolution=48, pad=0.35):
        """IS THIS RESOLUTION ENOUGH? Reports how many marching cells span the THINNEST feature, and
        the resolution that would fix it. A feature needs >=4 cells to mesh smoothly; below 2 it beads
        into visible LUMPS -- which is what a thin limb does inside the bounding box of a much larger
        body, because cell size is set by the whole body and the limb has no say. See
        holographic_creatureskin.skin_quality."""
        import holographic.mesh_and_geometry.holographic_creatureskin as _cs
        return _cs.skin_quality(creature, spec=spec, spacing=spacing, resolution=resolution, pad=pad)

    def fused_field(self, creature, spec=None, sockets=None, library=None, spacing=1.0,
                    smooth_k=0.06, fuse=True):
        """FUSE parts into the skin so they look GROWN, not glued: parts become metaballs in the SAME
        implicit surface as the body, giving one continuous mesh with a smooth fillet at the join
        rather than a hard seam. Only tapered-tube parts survive being reduced to a ball chain
        (horn/claw/spike/antenna/ear/eye); fins, hands and mouths would become blobs and are returned
        as `unfused` for placing as geometry. See holographic_creaturesocket.fused_field."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        return _sk.fused_field(creature, spec=spec, sockets=sockets, library=library,
                               spacing=spacing, smooth_k=smooth_k, fuse=fuse)

    # --------------------------------------------- limb sockets, auto feet, and cel shading --
    def resolve_limb_socket(self, creature, field, limb, u, theta=0.0, along_axis=False,
                            max_radius=3.0, steps=192):
        """Where a part lands on a LIMB -- at fraction `u` along it (0 mount, 1 tip), angle `theta`
        around it. `along_axis=True` casts down the limb's own axis, which is what a FOOT needs
        because a foot goes on the END of a leg, not on its side. Closes the kept negative that
        sockets were spine-relative and therefore approximate on limbs. See
        holographic_creaturesocket.resolve_limb_socket."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        return _sk.resolve_limb_socket(creature, field, limb, u, theta=theta,
                                       along_axis=along_axis, max_radius=max_radius, steps=steps)

    def auto_feet(self, creature, field, part="foot", scale=1.0, ground_frac=0.35, handles=None):
        """PUT FEET ON THE LEGS, automatically. Legs are identified the way the gait identifies them
        -- by which limbs reach the ground -- so this needs no authoring and works on any body plan,
        and an ARM correctly gets nothing. Returns socket dicts ready to append to a document. See
        holographic_creaturesocket.auto_feet."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        return _sk.auto_feet(creature, field, part=part, scale=scale, ground_frac=ground_frac,
                             handles=handles)

    def toon_shade(self, mesh, colours, eye, light_dir=(-0.55, 0.6, -0.55), bands=3, rim=0.30,
                   rim_power=2.5, ambient=0.45, band_floor=0.55):
        """CEL SHADING: quantise the light into flat BANDS and darken the silhouette, per vertex, so
        it composes with vertex paint and the existing rasteriser rather than needing a new render
        path. The silhouette is computed from GEOMETRY (the surface turning away, N.V -> 0) rather
        than by filtering the image, which would trace colour edges on a flat belly just as happily.
        Kept negative: darkens the silhouette only -- it does not stroke interior creases or part
        boundaries, which needs an image pass with depth and ids. See holographic_paintlod.toon_shade."""
        import holographic.mesh_and_geometry.holographic_paintlod as _pl
        return _pl.toon_shade(mesh, colours, eye, light_dir=light_dir, bands=bands, rim=rim,
                              rim_power=rim_power, ambient=ambient, band_floor=band_floor)

    def accelerate_convergence(self, step, x0, max_iters=200, tol=1e-12, r2_floor=0.99, probe=3):
        """JUMP TO AN ITERATIVE SOLVER'S LIMIT when its convergence is lawful, or decline. `step(x)->x`
        is any fixed-point iteration -- a relaxation sweep, a physics settle, an IK pass. A convergence
        sequence is a STREAM, so the ladder's question applies to it: does it have a generator? When it
        does, three iterates give the limit in closed form (measured: 7 iterations where plain needed
        70, to machine precision). A jump is taken ONLY if it VALIDATES -- one more step must move it
        less than it moves the plain iterate -- because naive extrapolation on a multi-mode solve
        measured 250x WORSE than simply iterating. See holographic_hrnn.accelerate_convergence."""
        import holographic.agents_and_reasoning.holographic_hrnn as _h
        return _h.accelerate_convergence(step, x0, max_iters=max_iters, tol=tol,
                                         r2_floor=r2_floor, probe=probe)

    def fleet_signature(self, streams, dim=None, seed=0):
        """ONE hypervector summarising how a whole COHORT of streams behaves structurally, plus the
        cohort's own calibrated `floor`. Does NOT grow with the number of streams and retains none of
        the raw data. See holographic_hrnn.fleet_signature."""
        import holographic.agents_and_reasoning.holographic_hrnn as _h
        return _h.fleet_signature(streams, dim=dim or self.dim, seed=seed)

    def fleet_anomaly(self, x, signature, dim=None, seed=0):
        """Is this stream behaving unlike its cohort? Compares STRUCTURE, not values -- EXACTLY
        invariant to scale, offset and sign (measured identical across 15 orders of magnitude), so a
        pressure sensor and a temperature sensor are directly comparable with no normalisation and no
        per-sensor calibration. Catches DRIFT, which amplitude and spectral baselines miss. KEPT
        NEGATIVE: a flatline is NOT caught -- a constant IS a generator -- so pair it with an
        amplitude check. See holographic_hrnn.fleet_anomaly."""
        import holographic.agents_and_reasoning.holographic_hrnn as _h
        return _h.fleet_anomaly(x, signature, dim=dim or self.dim, seed=seed)

    def explain_stream(self, x, dim=None, seed=0, alpha=0.9, n_tones=4):
        """ONE CALL, PLAIN ENGLISH -- hand it a stream, get what it IS and what to DO about it.
        Returns {headline, what_it_is, what_to_do, wont_do, confidence, predict, record, verdict}:
        a sentence a non-specialist can act on, the recommended next call written as runnable code,
        and the HONEST refusal saying what the verdict does not license. `predict` is the callable
        when a generator was found; `record` is the verdict as one hypervector for composing into a
        VSA application. This is the front door for the whole HRNN ladder -- everything else is
        reachable through it. See holographic_hrnn.explain_stream."""
        import holographic.agents_and_reasoning.holographic_hrnn as _h
        return _h.explain_stream(x, dim=dim or self.dim, seed=seed, alpha=alpha, n_tones=n_tones)

    def fit_multitone(self, x, n_tones=4, r2_floor=0.95, stages=1):
        """Fit a signal as a sum of INDEPENDENT sinusoids -- the generator class `fit_harmonics`
        cannot express, since it fits harmonics of ONE fundamental and honestly refuses on
        incommensurate tones (beating oscillators, two-rotor vibration, tidal constituents). Measured
        8.9-10.3x better there, with no regression on a harmonic stack. Greedy matching pursuit with
        off-grid refinement, deliberately NOT a sparse solve over a frequency dictionary -- a dense
        one is coherent and CoSaMP over it spans NRMSE 3.45e-01 to 2.24e+04 across density. See
        holographic_hrnn.fit_multitone."""
        import holographic.agents_and_reasoning.holographic_hrnn as _h
        return _h.fit_multitone(x, n_tones=n_tones, r2_floor=r2_floor, stages=stages)

    def certify_cycle(self, frames, tol=1e-6, pmax=None, hint=0, flatten=None):
        """DOES THIS SEQUENCE REPEAT? The smallest period p at which every recent frame matches the
        one p back, certified at a numeric tolerance -- or certified=False, never a best guess.
        Promoted out of run_until_settled's oscillatory branch, where it was reachable only by
        running a simulation. Use it on any sequence: a REGIME STREAM (HRNN verdicts over windows) is
        a square wave that a harmonic fit rings on (NRMSE 0.584) and a cycle certificate replays at
        0.037. See holographic_statedemand.certify_cycle."""
        import holographic.sampling_and_signal.holographic_statedemand as _sd
        return _sd.certify_cycle(frames, tol=tol, pmax=pmax, hint=hint, flatten=flatten)

    # ------------------------------------- HRNN verdicts as hypervectors (VSA-native) --
    def verdict_to_record(self, verdict, dim=None, seed=0):
        """Carry an HRNN stream VERDICT as ONE hypervector -- each field bound to its role, bundled.
        Turns a plain dict into something the VSA layer can actually use: compare two streams by
        COSINE, bundle a whole workspace of them into one vector, unbind a field and clean it up.
        Kept negative: the record carries the verdict, NOT the model -- the `predict` closure has no
        hypervector form and a lossy encode of the coefficients would predict the wrong thing. See
        holographic_hrnn.verdict_to_record."""
        import holographic.agents_and_reasoning.holographic_hrnn as _h
        return _h.verdict_to_record(verdict, dim=dim or self.dim, seed=seed)

    def verdict_from_record(self, record, dim=None, seed=0, vocabulary=None):
        """Recover a verdict's fields from its hypervector: unbind each role, clean up against the
        codebook. Returns the fields plus `_confidence` per field, so a degraded bundle is VISIBLE
        rather than confidently wrong. See holographic_hrnn.verdict_from_record."""
        import holographic.agents_and_reasoning.holographic_hrnn as _h
        return _h.verdict_from_record(record, dim=dim or self.dim, seed=seed, vocabulary=vocabulary)

    def verdict_vocabulary(self, regimes=None, mechanisms=None, dim=None, seed=0):
        """The codebook a recalled verdict field is cleaned up against. An unbound role gives a NOISY
        vector; reading it without cleanup is the mistake this engine keeps a negative about. See
        holographic_hrnn.verdict_vocabulary."""
        import holographic.agents_and_reasoning.holographic_hrnn as _h
        kw = {}
        if regimes is not None:
            kw["regimes"] = regimes
        if mechanisms is not None:
            kw["mechanisms"] = mechanisms
        return _h.verdict_vocabulary(dim=dim or self.dim, seed=seed, **kw)

    # ----------------------------------------------------------------- GAIT (locomotion) --
    def analyze_rig(self, creature, ground_frac=0.35):
        """Work out which limbs are LEGS -- by asking which reach the ground, so it is MEASURED, not
        named -- plus ground height, per-leg reach and a derived stride. Everything the gait needs is
        taken off the rig, which is what makes locomotion morphology-independent. See
        holographic_gait.analyze_rig."""
        import holographic.mesh_and_geometry.holographic_gait as _g
        return _g.analyze_rig(creature, ground_frac=ground_frac)

    def gait_pattern(self, n_legs, kind="walk"):
        """Phase offsets and duty factor for a gait. Bipeds and tetrapods get the classic diagrams
        (walk/trot/pace/bound/gallop); any other leg count gets an evenly spread metachronal wave.
        See holographic_gait.gait_pattern."""
        import holographic.mesh_and_geometry.holographic_gait as _g
        return _g.gait_pattern(n_legs, kind=kind)

    def gait_pose(self, creature, t, gait="walk", period=1.0, lift=None, forward=(0.0, 1.0, 0.0),
                  rig=None, iters=24):
        """MAKE IT WALK: the creature posed at time `t`, feet planted where the gait says. Body speed
        is DERIVED (stride / (duty * period)), never a free parameter -- take it as an input and the
        planted feet slide. Legs are posed through the rig's own limit-constrained IK, so a gait
        cannot violate a joint limit the rig declares. See holographic_gait.gait_pose."""
        import holographic.mesh_and_geometry.holographic_gait as _g
        return _g.gait_pose(creature, t, gait=gait, period=period, lift=lift, forward=forward,
                            rig=rig, mind=self, iters=iters)

    def gait_frames(self, creature, gait="walk", period=1.0, n_frames=24, forward=(0.0, 1.0, 0.0)):
        """A full walk cycle as poses, ready for the shipped timeline / render_animation. See
        holographic_gait.gait_frames."""
        import holographic.mesh_and_geometry.holographic_gait as _g
        return _g.gait_frames(creature, gait=gait, period=period, n_frames=n_frames,
                              forward=forward, mind=self)

    def gait_report(self, creature, gait="walk", period=1.0, n_frames=48, forward=(0.0, 1.0, 0.0)):
        """MEASURE the walk: FOOT SLIP -- how far a planted foot slides in world space, absolutely and
        as a fraction of stride. That is the moonwalk artifact, and it is a number rather than an
        opinion. Also reports distance travelled, measured duty per foot, and any leg the IK could not
        place. Kept negative: a foot that never moves cannot slip, so check `unreachable` is empty
        before trusting a low score. See holographic_gait.gait_report."""
        import holographic.mesh_and_geometry.holographic_gait as _g
        return _g.gait_report(creature, gait=gait, period=period, n_frames=n_frames,
                              forward=forward, mind=self)

    def gait_names(self, n_legs=4):
        """The gaits available for a leg count -- what an app's gait picker enumerates. See
        holographic_gait.gait_names."""
        import holographic.mesh_and_geometry.holographic_gait as _g
        return _g.gait_names(n_legs=n_legs)

    # ------------------------------------------------ the PARAMETRIC part library (rigblocks) --
    def creature_parts(self, dim=1024, seed=0, params=None):
        """A PartLibrary pre-loaded with every parametric part (eye, mouth, foot, hand, claw, horn,
        spike, fin, antenna, ear, digit), each with its AUTHORED handle ranges and default geometry.
        One call gets an app a working part palette. See holographic_creaturepartlib.library."""
        import holographic.mesh_and_geometry.holographic_creaturepartlib as _pl
        return _pl.library(dim=dim, seed=seed, params=params)

    def build_part(self, name, **params):
        """Build one body part by name and parameters -- procedural, so `digits=3` versus `digits=5`
        is a genuinely different foot rather than one mesh scaled, which is what a rigblock handle is
        supposed to mean. See holographic_creaturepartlib.build_part."""
        import holographic.mesh_and_geometry.holographic_creaturepartlib as _pl
        return _pl.build_part(name, **params)

    def part_names(self):
        """Every part the library can build -- what an app's part picker enumerates. See
        holographic_creaturepartlib.part_names."""
        import holographic.mesh_and_geometry.holographic_creaturepartlib as _pl
        return _pl.part_names()

    def sweep_profile(self, path, radii, sides=10, cap_start=True, cap_end=True):
        """Sweep a circular cross-section of VARYING radius along a 3-D path -- the tapered sweep that
        every organic appendage needs and that `sweep_tube` (one profile for the whole tube) cannot
        do. Rotation-minimizing, so the ring does not spin on a curved path. See
        holographic_creaturepartlib.sweep_profile."""
        import holographic.mesh_and_geometry.holographic_creaturepartlib as _pl
        return _pl.sweep_profile(path, radii, sides=sides, cap_start=cap_start, cap_end=cap_end)

    # -------------------------------- creature EDITOR: sockets, picking, session (Spore loop) --
    def creature_editor(self, spec=None, max_depth=256, part_library=None):
        """A live creature EDITING SESSION -- the object an app drives. Holds the document, records
        every change so it can be undone, saves and loads as JSON, validates, meters a complexity
        budget, and builds geometry. Edits return `self`, so a UI can chain and still undo each step
        separately. See holographic_creatureeditor.CreatureEditor."""
        import holographic.mesh_and_geometry.holographic_creatureeditor as _ce
        return _ce.CreatureEditor(spec=spec, max_depth=max_depth, part_library=part_library)

    def load_creature(self, text, part_library=None):
        """Load a saved creature document (JSON) back into an editing session. Round-trips exactly.
        See holographic_creatureeditor.CreatureEditor.from_json."""
        import holographic.mesh_and_geometry.holographic_creatureeditor as _ce
        return _ce.CreatureEditor.from_json(text, part_library=part_library)

    def resolve_socket(self, creature, field, t, theta, max_radius=3.0, steps=192):
        """WHERE A PART LANDS: given anatomy coordinates (t along the spine, theta around the body),
        march the creature's own skin field outward and return the surface point, normal and a (4,4)
        placement frame. Because a socket stores (t, theta) rather than a world position, the part
        rides the skin through every later spine and thickness edit. Reports `hit` rather than
        inventing a placement when the ray misses. See holographic_creaturesocket.resolve_socket."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        return _sk.resolve_socket(creature, field, t, theta, max_radius=max_radius, steps=steps)

    def socket_at_point(self, creature, point, samples=257):
        """The INVERSE: a world point on the body -> the socket (t, theta) that names it. This is what
        turns a mouse click into an editable, edit-proof attachment. Round-trips with resolve_socket.
        See holographic_creaturesocket.socket_at_point."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        return _sk.socket_at_point(creature, point, samples=samples)

    def pick_socket(self, creature, field, origin, direction, max_t=20.0, steps=512):
        """VIEWPORT PICK: cast a ray at the creature and return the socket it lands on, or None on a
        miss. Marches the skin FIELD rather than a tessellated proxy, so parts land exactly on the
        surface the user sees. See holographic_creaturesocket.pick_socket."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        return _sk.pick_socket(creature, field, origin, direction, max_t=max_t, steps=steps)

    def place_parts(self, creature, field, sockets, library=None, mode="merge", scene=None,
                    material="paint"):
        """Resolve a list of sockets and PLACE their part geometry on the body -- merged into one mesh
        or instanced through one Definition per part. Symmetry groups are expanded here, so bilateral
        and radial parts come from one code path. Returns the geometry plus any sockets that MISSED,
        so an app can flag them. See holographic_creaturesocket.place_parts."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        return _sk.place_parts(creature, field, sockets, library=library, mode=mode, scene=scene,
                               material=material)

    def spine_frames(self, creature):
        """The body's own coordinate system: spine node positions plus a stable rotation-minimizing
        frame at each, which is what makes `theta` mean the same thing along a curved body instead of
        twisting at inflections. See holographic_creaturesocket.spine_frames."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        return _sk.spine_frames(creature)

    # ------------------------------------------------------------- creature idle animation (R-10) --
    def creature_idle(self, creature, t, amplitude=0.35, period=2.0, seed=0, chains=None):
        """A simple IDLE POSE for a creature at time `t`: every joint flexes within its OWN stored
        limit, so knees/elbows/hips visibly show WHERE they are and WHICH WAY they bend. The limit is
        the driver, so an impossible bend cannot be displayed. Returns {joint: position}; does not
        mutate the creature. Kept negative: a limits demo, not locomotion -- no gait, no ground
        contact, no balance. See holographic_creatureidle.idle_pose."""
        import holographic.mesh_and_geometry.holographic_creatureidle as _ci
        return _ci.idle_pose(creature, t, amplitude=amplitude, period=period, seed=seed, chains=chains)

    def creature_idle_frames(self, creature, n_frames=24, amplitude=0.35, period=2.0, seed=0, loop=True):
        """A full, seamlessly looping idle CYCLE: `n_frames` poses over one period, ready for the
        shipped timeline / render_animation. See holographic_creatureidle.idle_frames."""
        import holographic.mesh_and_geometry.holographic_creatureidle as _ci
        return _ci.idle_frames(creature, n_frames=n_frames, amplitude=amplitude, period=period,
                               seed=seed, loop=loop)

    def creature_idle_report(self, creature, n_frames=16, amplitude=0.35, period=2.0, seed=0):
        """VERIFY an idle against the rig it came from: max bone-length error across the cycle (must be
        ~0 -- this is a rotation, nothing may stretch), peak flex per limb in degrees, the smallest
        headroom left against any stored limit (negative would mean the animation lies about the rig),
        and the fraction of limb joints that actually move. See holographic_creatureidle.idle_report."""
        import holographic.mesh_and_geometry.holographic_creatureidle as _ci
        return _ci.idle_report(creature, n_frames=n_frames, amplitude=amplitude, period=period, seed=seed)

    def tiered_memory(self, hot_capacity=64, half_life=32.0, vocab=256, dim=None, seed=0,
                      policy="exact"):
        """Adaptive SHORT/LONG-term key->value memory: exact bounded hot dict, constant-size
        superposed LT trace + compressed exact spill, importance-driven demotion (recency-window
        veto) and access-driven promotion. Low overhead for what matters, low disk for what
        doesn't. See holographic_tieredmemory.TieredMemory."""
        import holographic.caching_and_storage.holographic_tieredmemory as _tm
        return _tm.TieredMemory(self, hot_capacity=hot_capacity, half_life=half_life,
                                vocab=vocab, dim=dim, seed=seed, policy=policy)

    def celled_memory(self, dim=4096, vocab=8192, seed=0, cell_pairs=None, keep_warm=8):
        """Unbounded pairs over BOUNDED superposed cells -- Quilez domain repetition (opRep)
        applied to the capacity law: the measured limit IS the tile size. One shared seed-derived
        codebook; cells of n* pairs; warm/cold cell tiers; exact key->cell directory. MEASURED:
        one memory 70x past the law recalls at 0.007; celled recalls 1.000. See
        holographic_cellmemory.CelledMemory (kept negative there: a holographic directory would
        re-buy the interference the cells escape)."""
        import holographic.caching_and_storage.holographic_cellmemory as _cm
        return _cm.CelledMemory(self, dim=dim, vocab=vocab, seed=seed,
                                cell_pairs=cell_pairs, keep_warm=keep_warm)

    def superposed_memory(self, dim=None, vocab=256, seed=0, precision="f64", codebook="dense"):
        """One-vector key-value store (memory = sum of bind(key, value)) with a closed-form
        capacity law, decision-free int8, and a load-GATED resonator-style PIC decoder that
        refuses past its phase transition instead of silently degrading. See
        holographic_superposed.SuperposedMemory; law/allocator: mind.memory_capacity_law /
        mind.allocate_memory_dim."""
        from holographic.caching_and_storage.holographic_supermemory import SuperposedMemory
        return SuperposedMemory(dim or self.dim, vocab, seed=seed, precision=precision, codebook=codebook)

    def memory_capacity_law(self, dim=None, vocab=256, alpha=0.90):
        """Predicted one-shot capacity n* of a superposed pair memory -- closed form with the
        V-dependence measured on this tree (n* ~ C*D/(2*Qinv((1-alpha)/V)^2)). See
        holographic_superposed.capacity_law."""
        from holographic.caching_and_storage.holographic_supermemory import capacity_law
        return capacity_law(dim or self.dim, vocab, alpha=alpha)

    def allocate_memory_dim(self, n_pairs, vocab=256, alpha=0.90, decoder="one-shot"):
        """Dimension needed BEFORE storing n_pairs at accuracy alpha -- the capacity law
        inverted, with margin (price the demand, then spend). decoder='pic' allocates
        against the iterative transition instead of the matched-filter wall. See
        holographic_superposed.allocate."""
        from holographic.caching_and_storage.holographic_supermemory import allocate
        return allocate(n_pairs, vocab, alpha=alpha, decoder=decoder)

    def state_demand(self, x, k=4, length=6, seed=0):
        """How much state does this stream DEMAND -- TT-SVD bond dimensions of its block
        distribution, thresholded against the stream's own shuffled null (iid reads 1, not
        sampling noise). Returns {ranks, demand_bits, floor}; feed demand into
        allocate_memory_dim. See holographic_statedemand.tt_state_demand."""
        from holographic.sampling_and_signal.holographic_statedemand import (
            quantize_stream, tt_state_demand)
        import numpy as _np
        arr = _np.asarray(x)
        sym = arr.astype(_np.int64) if arr.dtype.kind in "iu" else quantize_stream(arr, k)
        return tt_state_demand(sym, k=k, length=length, seed=seed)

    def entropy_rate(self, x, k=4):
        """Entropy rate h and excess entropy E from block-entropy scaling, dense-regime
        guarded (refuses block lengths the sample count cannot support -- the measured
        silent-low-bias failure). h ~ 0 marks the deterministic regime. See
        holographic_statedemand.entropy_rate_report."""
        from holographic.sampling_and_signal.holographic_statedemand import (
            quantize_stream, entropy_rate_report)
        import numpy as _np
        arr = _np.asarray(x)
        sym = arr.astype(_np.int64) if arr.dtype.kind in "iu" else quantize_stream(arr, k)
        return entropy_rate_report(sym, k=k)

    def compressibility_check(self, x, k=4, h_max=0.5, n_null=20, alpha=0.05, use_fit=True):
        """Two-stage 'does a generator exist' gate with a MANDATORY horizon field (the same
        process honestly earns different verdicts at different windows -- measured). Stage 1
        rejects on entropy rate (the nulls surrogates cannot catch); stage 2 calibrates
        fit_deterministic's correlation against phase-randomised surrogates of the SAME
        signal. See holographic_statedemand.compressibility_gate."""
        from holographic.sampling_and_signal.holographic_statedemand import (
            compressibility_gate)
        import numpy as _np
        score_fn = None
        surrogate_fn = None
        if use_fit:
            def score_fn(v):
                f = self.fit_deterministic(_np.asarray(v, dtype=float).ravel())
                c = f.get("correlation")
                return 0.0 if c is None else float(_np.ravel(c)[0])

# UNIFIED: this was an inline copy of holographic_surrogate.phase_randomize. All four copies
            # forced the DC phase to 0.0, which FLIPS THE SIGN OF THE MEAN for a negative-mean signal
            # (measured -2.933 -> +2.933). The canonical one preserves angle(F[0]). Delegate, never re-inline.
            def surrogate_fn(v, rng):
                """Phase-randomised surrogate. See holographic_surrogate.phase_randomize."""
                from holographic.sampling_and_signal.holographic_surrogate import phase_randomize
                return phase_randomize(v, rng=rng)
        return compressibility_gate(_np.asarray(x, dtype=float).ravel(), k=k, h_max=h_max,
                                    score_fn=score_fn, surrogate_fn=surrogate_fn,
                                    n_null=n_null, alpha=alpha)

    def holographic_rnn(self, dim=None, seed=0, alpha=0.90, use_fit_deterministic=True):
        """The HOLOGRAPHIC RNN (HRNN-1): a sequence engine that measures before it models.
        process_stream(x) walks the abstention ladder -- calibrated compressibility gate ->
        identify the generator (all three triangle corners, THIS horizon only) | price the
        state demand and route to associative()/classifier() | refuse with an allocator
        quote. classifier() reads trajectories with BOTH measured invariances (arrival-time
        traps + Levy areas); associative(n_pairs=..) allocates dimension from the capacity
        law before storing. Every verdict carries {regime, mechanism, h, horizon, why}.
        See holographic_hrnn.HolographicRNN."""
        from holographic.agents_and_reasoning.holographic_hrnn import HolographicRNN
        import numpy as _np
        gen_fit = None
        if use_fit_deterministic:
            def gen_fit(v):
                # WHY a wrapper: the ladder wants {r2, ok, predict}; fit_deterministic
                # speaks {correlation, verdict} and extend_generator speaks forecasts
                # from the end of the record. predict() therefore serves indices at or
                # past len(v) via extend_generator and refuses interior queries -- the
                # generator rung only ever extends, never interpolates history.
                v = _np.asarray(v, dtype=float).ravel()
                f = self.fit_deterministic(v)
                c = f.get("correlation")
                r2 = 0.0 if c is None else float(_np.ravel(c)[0])
                ok = f.get("family") is not None and str(f.get("verdict")) != "refused"

                def predict(idx):
                    idx = _np.asarray(idx, dtype=int)
                    if idx.size == 0 or idx.min() < len(v):
                        raise ValueError("generator rung extends past the record only")
                    ext = self.extend_generator(f, int(idx.max()) - len(v) + 1, len(v))
                    fc = _np.asarray(ext["forecast"])
                    return fc[idx - len(v)]

                return {"ok": ok, "r2": r2, "fit": f, "predict": predict}
        return HolographicRNN(dim=dim or self.dim, seed=seed, alpha=alpha,
                              generator_fit=gen_fit)

    def stream_sentinel(self, window=512, hop=256, h_jump=0.5, k=4, seed=0, use_mind_engine=True):
        """Watch a stream through the HRNN ladder: watch(x) segments by regime and raises
        change events WITH both windows' evidence attached; record(x) stores each window
        at its cheapest FAITHFUL form (generator params ~30 floats | quantile symbols |
        raw -- noise is never fake-compressed) and replay(tape) reconstructs with the
        certificates intact. Generator claims are prefix-fit/suffix-certified
        (predictive compression), immune to the pure-tone surrogate degeneracy. See
        holographic_sentinel.StreamSentinel."""
        from holographic.sampling_and_signal.holographic_sentinel import StreamSentinel
        eng = self.holographic_rnn(seed=seed) if use_mind_engine else None
        return StreamSentinel(window=window, hop=hop, h_jump=h_jump, k=k,
                              engine=eng, seed=seed)

    def triage_cascade(self, feature_fn=None, full_fn=None, safety=1.0):
        """A little trained model that AMORTISES an expensive predicate -- the codebook
        idea applied to computation itself. Contract: the fast path may only REJECT;
        every accept runs the full machinery (a cheap surrogate that can say yes is a
        false-certification machine; one that can only say no fails, at worst, into the
        price you were already paying). Defaults front the compressibility gate with
        cheap spectral/entropy features. Train with .fit(streams); trained heads
        save()/load() like the other models. See holographic_triage.TriageCascade."""
        from holographic.agents_and_reasoning.holographic_triage import (
            TriageCascade, gate_features)
        import numpy as _np
        if feature_fn is None:
            feature_fn = gate_features
        if full_fn is None:
            def full_fn(x):
                return bool(self.compressibility_check(_np.asarray(x, dtype=float)
                                                       .ravel())["passed"])
        return TriageCascade(feature_fn, full_fn, safety=safety)

    def train_model(self, examples, labels=None, task="auto", dim=None, seed=0, alpha=0.90):
        """ONE front door for training: sequences+labels -> trajectory classifier
        (REFUSES to call an underdetermined ridge 'trained' -- the measured learning-
        curve knee enforced at the API); (keys, values) -> pair memory with dimension
        allocated from the capacity law BEFORE storing; a bare stream -> the HRNN
        ladder (generator model with predict(), or an honest verdict). Every result
        carries {kind, trained, why}; every model save()s. Uses fit_deterministic for
        the generator rung. See holographic_modeltrain.train_model."""
        from holographic.agents_and_reasoning.holographic_modeltrain import train_model
        gen = self.holographic_rnn(dim=dim, seed=seed, alpha=alpha).generator_fit
        return train_model(examples, labels=labels, task=task, dim=dim or self.dim,
                           seed=seed, alpha=alpha, generator_fit=gen)

    def structure_fingerprint(self, x, k=4):
        """Tiny structural signature of any stream -- {h, E, ranks, demand_bits,
        horizon} -- memoised underneath, cheap enough to log per artifact per release.
        Compare two with mind.structure_drift. See holographic_modeltrain.fingerprint."""
        from holographic.agents_and_reasoning.holographic_modeltrain import fingerprint
        return fingerprint(x, k=k)

    def structure_drift(self, fp_a, fp_b, h_tol=0.25, rank_tol=1):
        """Did the structure change between two fingerprints? Returns {changed, why} in
        measured units (entropy rate moved / state demand moved), with tolerances set
        from this tree's own observed spreads. The regression detector for pipelines:
        same artifact fingerprints the same; a structure change moves h or the ranks
        before it breaks a unit test. See holographic_modeltrain.drift."""
        from holographic.agents_and_reasoning.holographic_modeltrain import drift
        return drift(fp_a, fp_b, h_tol=h_tol, rank_tol=rank_tol)

    def nested_memory(self, n_bases=8, facts_per_base=64, vocab=1024, seed=0,
                      precision="f64", dim=None):
        """A LIBRARY of knowledge bases in ONE vector, any fact from any base in a
        SINGLE unbind: query(base, key) = cleanup(unbind(library, name(*)key)) --
        bind's associativity makes two-level lookup cost one operation, and capacity is
        the flat law at the product load (allocated here BEFORE the first fact unless
        dim is given). Bases can be built in place (add) or shelved from existing
        trained pair memories (shelve). Exports at the declared precision -- a whole
        library at one bit per dimension. See holographic_nested.NestedMemoryLibrary."""
        from holographic.caching_and_storage.holographic_nested import NestedMemoryLibrary
        from holographic.caching_and_storage.holographic_supermemory import allocate
        D = dim or allocate(n_bases * facts_per_base, vocab)
        return NestedMemoryLibrary(D, vocab=vocab, max_bases=max(64, n_bases),
                                   seed=seed, precision=precision)

    def easy_model(self, examples, labels=None, task="auto", seed=0, alpha=0.90):
        """The three-verb handle: train anything, ask it anything, save/load anywhere.
        m = mind.easy_model(data); m.ask(query); m.save(path); reload with
        mind.load_easy_model(path). ask() dispatches by kind so users never need to
        know whether they got a memory, a classifier, or a generator. Built on
        train_model, so the honesty guards (underdetermination, allocation-before-
        storing, horizon) all still apply. See holographic_modeltrain.Model."""
        from holographic.agents_and_reasoning.holographic_modeltrain import Model, train_model
        gen = self.holographic_rnn(dim=None, seed=seed, alpha=alpha).generator_fit
        return Model(train_model(examples, labels=labels, task=task, dim=self.dim,
                                 seed=seed, alpha=alpha, generator_fit=gen))

    def load_easy_model(self, path):
        """Reload a saved easy model (any kind) ready to ask(). See
        holographic_modeltrain.Model.load."""
        from holographic.agents_and_reasoning.holographic_modeltrain import Model
        return Model.load(path)

    def hrnn_recipes(self, topic=None):
        """Domain front door for the HRNN family: recipes(topic) returns a working call
        sequence for forecasting, market analysis, scientific study, data processing,
        text, or audio -- each with what it does, how to call it, and the HONEST scope
        of what it will not do (the refusals are the product). No topic -> the index.
        See holographic_modeltrain.recipes."""
        from holographic.agents_and_reasoning.holographic_modeltrain import recipes
        return recipes(topic)

    def behavior_meter(self, actions, rewards=None, prev=None, n_actions=None):
        """Creature/agent learning instrument: entropy rate of the action stream
        measures policy FORMATION, rewards measure CORRECTNESS, and formation
        advancing while reward does not fires the WRONG-HABIT ALARM (measured live:
        a real CreatureMind crystallised to h 1.96->0.97 at policy-correct 0.25 --
        structured, wrong, and invisible to either meter alone). Log per creature per
        epoch; feed each result back as prev. See holographic_modeltrain.behavior_meter."""
        from holographic.agents_and_reasoning.holographic_modeltrain import behavior_meter
        return behavior_meter(actions, rewards=rewards, prev=prev, n_actions=n_actions)

    def synthesize_model(self, examples, labels=None, seed=0, alpha=0.90):
        """SYN-1: measure the data, EMIT the pipeline as an inspectable JSON recipe
        (each stage choice with the measurement that justified it), then train it.
        Recipes are artifacts like stored VM programs: diffable, versionable,
        replayable. See holographic_modeltrain.synthesize."""
        from holographic.agents_and_reasoning.holographic_modeltrain import synthesize
        return synthesize(examples, labels=labels, seed=seed, alpha=alpha)

    def make_surrogate(self, fn, sample_inputs, seed=0, alpha=0.90):
        """SUR-1: the certified surrogate layer -- run fn once over samples; serve
        certified extension where the ladder certifies a generator (measured 9078x on
        the oscillator), exact hash-replay on seen inputs, the real computation
        (memoised) otherwise. Never fabricates; .provenance states the contract in
        force. See holographic_modeltrain.make_surrogate."""
        from holographic.agents_and_reasoning.holographic_modeltrain import make_surrogate
        return make_surrogate(fn, sample_inputs, seed=seed, alpha=alpha)

    def big_pair_memory(self, dim=None, vocab=65536, seed=0, n_pairs=None):
        """CTX-1 substrate: SuperposedMemory for vocabularies whose codebooks cannot
        be materialised -- atoms regenerated from seeds in chunks (the MQAR pattern,
        proven at V=8192 recall 1.000). Batch store/recall; the state is ONE vector.
        n_pairs allocates dim from the capacity law. See
        holographic_superposed.BigPairMemory."""
        from holographic.caching_and_storage.holographic_supermemory import (
            BigPairMemory, allocate)
        d = dim or (allocate(n_pairs, vocab) if n_pairs else self.dim)
        return BigPairMemory(d, vocab, seed=seed)

    def find_capability_enriched(self, problem, k=3):
        """CTX-1 hook: retrieval-augmented routing. Words the catalog does not know
        are looked up in the in-tree 144k dictionary and their definition tokens
        (with suffix families) are appended before searching, so exotic phrasings
        reach the capabilities their plain synonyms name. Returns {results,
        expansions, enriched} -- the expansions are reported, never silent. Additive
        by construction: tokens are only added, so a raw hit can never be lost. See
        holographic_modeltrain.enrich_query."""
        from holographic.agents_and_reasoning.holographic_modeltrain import enrich_query
        eq, exp = enrich_query(problem)
        return {"results": self.find_capability(eq if exp else problem, k=k),
                "expansions": exp, "enriched": bool(exp)}

    def replay_model_recipe(self, recipe, examples, labels=None):
        """Replay a stored synthesis recipe and ASSERT the stage choices reproduce --
        a recipe is a contract; drift raises with the diff. See
        holographic_modeltrain.replay_recipe."""
        from holographic.agents_and_reasoning.holographic_modeltrain import replay_recipe
        return replay_recipe(recipe, examples, labels=labels)

    def lincode_codebooks(self, dim=None, n_factors=3, n_entries=24, seed=0):
        """Build codebooks whose phases are a LINEAR image of their index bits (Raviv 2024 form).

        Returns (codebooks, basis). The basis IS the structure that makes factorization a SOLVE
        instead of a search -- keep it, or factor_exact has nothing to work with.
        See holographic_lincode.build_codebooks.
        """
        import holographic.agents_and_reasoning.holographic_lincode as _LC
        return _LC.build_codebooks(dim or self.dim, n_factors, n_entries, seed=seed)

    def factor_exact(self, composite, basis, n_factors, n_entries):
        """Recover F indices from a bound product by SOLVING for index bits -- exact, not searched.

        MEASURED head-to-head against phasor_factor at D=1024, 12 trials each: resonator
        2/12 -> this 12/12 at F=3 M=24; 0/12 -> 12/12 at F=4 M=24; 0/12 -> 12/12 at F=6 M=8;
        and 5.4ms -> 0.3ms. KEPT NEGATIVE: this is NOT a better resonator. It requires codebooks
        built by lincode_codebooks (they carry the structure being solved) and REFUSES an
        underdetermined system rather than guessing. The resonator factors codebooks it did not
        create; this one is exact on codebooks it did. See holographic_lincode.factor_exact.
        """
        import holographic.agents_and_reasoning.holographic_lincode as _LC
        return _LC.factor_exact(composite, basis, n_factors, n_entries)

    def capacity_gate(self, **kw):
        """Consult every measured capacity law and ROUTE to the escape (proceed/reroute/abstain).

        Takes the same keywords as advise_scale. A law that is measured and never consulted is
        worth nothing, and every failing law here has a MEASURED escape that is not "more dim":
        pair-capacity -> celled_memory (recall 0.007 -> 1.000), nesting depth ->
        encode_tree_carrier (leaf share 0.00044 at d7 -> recovery 0.94-1.00 at d7-32).
        See holographic_capacitygate.capacity_gate.
        """
        import holographic.caching_and_storage.holographic_capacitygate as _cg
        return _cg.capacity_gate(self.advise_scale(**kw))

    def advise_scale(self, n_pairs=None, vocab=None, dim=None, bundle_k=None,
                     depth=None, factors=None, alpha=0.90, decoder="one-shot",
                     fix=False, codebook=None):
        """Every measured capacity/depth law in one checkpoint, consulted BEFORE the
        wall: margins per law, the BINDING constraint, and a concrete prescription
        (exact dim, decoder switch at the PIC transition, or the named lever --
        partition / sparse decoder / factor-group tiling). fix=True also returns the
        corrected spec. Empirical knobs route to mind.auto_scale by name. See
        holographic_superposed.advise_scale."""
        from holographic.caching_and_storage.holographic_supermemory import advise_scale
        return advise_scale(n_pairs=n_pairs, vocab=vocab, dim=dim or self.dim,
                            bundle_k=bundle_k, depth=depth, factors=factors,
                            alpha=alpha, decoder=decoder, fix=fix, codebook=codebook)

    def depth_probe(self, depth, dim=None, n=8, seed=0):
        """Measure the nesting-depth wall for typed trees: worst-case cosine between
        trees differing only at the deepest leaf (1.0 = that level is gone). The
        measured law it carries: the collapse is DIM-INDEPENDENT, so dim is not the
        lever -- elevate levels onto carriers or anchor coarse/fine. See
        holographic_superposed.depth_probe."""
        from holographic.caching_and_storage.holographic_supermemory import depth_probe
        return depth_probe(depth, dim or self.dim, n=n, seed=seed)

    def compute_plan(self, n, calls_expected=1, repeat_fraction=0.0, stream=None,
                     zig_n=None, use_gpu_row=None):
        """The unified compute router: memo and certified-surrogate tiers consulted
        BEFORE the backend race, then the real zig policy (measured 2-5x regime) and
        the real gpu crossover row when hardware has measured one -- an unmeasured
        device is NAMED blocked, never guessed. Where a GPU wins on throughput, the
        winning move is often to shrink the work, not race it. See
        holographic_modeltrain.plan_compute."""
        from holographic.agents_and_reasoning.holographic_modeltrain import plan_compute
        zig = None
        try:
            zig = self.zig_dispatch_policy(zig_n or n, calls_expected)
        except Exception:
            pass
        return plan_compute(n, calls_expected=calls_expected,
                            repeat_fraction=repeat_fraction, stream=stream,
                            zig=zig, gpu=use_gpu_row)

    def convergence_guard(self, increments, max_lag=64):
        """Guard the CLT adaptive-sampling stop with the assumption it silently
        makes: the variance interval is right for i.i.d. increments and a lie for a
        pixel still drifting (a caustic being discovered) or sampling correlated.
        Measured trap: two streams with near-identical CLT half-widths, one with
        10x its claimed error. Consult per pixel/tile BEFORE honouring
        adaptive_sample_budget's stop. See holographic_statedemand.convergence_guard."""
        from holographic.sampling_and_signal.holographic_statedemand import convergence_guard
        return convergence_guard(increments, max_lag=max_lag)

    def run_until_settled(self, step, state, steps, residual=None, window=96,
                          check_every=16, cycle_handoff=False, cycle_tol=1e-6,
                          settle_tol=1e-2, max_lag=48):
        """Settle-gated simulation runner: pay for dynamics, not equilibrium. Runs
        step(state)->state until the residual stream passes convergence_guard
        (i.i.d.), then serves remaining frames from the settled state. Measured on
        the real fluid solver: 600-frame decaying shot settled at step 96 -- 504
        frames served, 4.7x wall-clock, final-frame error vs the full simulation
        0.00e+00; a driven flow keeps ORDER in its residuals and every frame is
        honestly simulated. See holographic_statedemand.run_until_settled."""
        from holographic.sampling_and_signal.holographic_statedemand import run_until_settled
        return run_until_settled(step, state, steps, residual=residual,
                                 window=window, check_every=check_every,
                                 cycle_handoff=cycle_handoff, cycle_tol=cycle_tol,
                                 settle_tol=settle_tol, max_lag=max_lag)

    def behavior_pool(self, window=64, alpha=0.98):
        """Behavior LOD for agent populations: agents whose output stream certifies
        as an exact cycle are DEMOTED to served cycles at near-zero cost; external
        input PROMOTES back to live instantly; agents that never certify (driven,
        chaotic, learning) are NEVER demoted. Born from the 50k-NPC single-box
        budget: settled behavior costs what its information content costs. See
        holographic_modeltrain.BehaviorPool."""
        from holographic.agents_and_reasoning.holographic_modeltrain import BehaviorPool
        return BehaviorPool(window=window, alpha=alpha)

    def stream_meter(self, window=256, max_lag=48):
        """Online meters for live data: push(x) per sample/block, verdict() gives
        the convergence guard on the current window (bit-identical to the batch
        guard on the same bytes -- selftest-pinned), entropy() the live rate report.
        For audio blocks, sim residuals, agent action streams -- the instruments
        where the data is born. Returns the meter object (HTTP: via the handle
        registry). See holographic_statedemand.StreamMeter."""
        from holographic.sampling_and_signal.holographic_statedemand import StreamMeter
        return StreamMeter(window=window, max_lag=max_lag)

    def encode_tree_carrier(self, tree, dim=None, seed=0):
        """Carrier-elevated tree encoding: each level rides an explicit carrier, so
        depth contribution is LINEAR not geometric. Measured against the flat
        encoder's dim-independent wall: separable at d7 (flat dead at d5), deep-leaf
        readout via carrier unbind 0.94-1.00 at depths 7-32 where flat carries zero
        bits. Trade-off stated: depth-addressability for the flat encoder's holistic
        nesting algebra -- both costumes kept. See
        holographic_supermemory.encode_tree_carrier / depth_probe_carrier."""
        from holographic.caching_and_storage.holographic_supermemory import encode_tree_carrier
        return encode_tree_carrier(tree, dim or self.dim, seed=seed)

    def bundle_decode(self, m, codebook, k, method="omp"):
        """Recover the k members of a direct bundle. method='omp' (greedy
        subtraction) holds 4.0x the linear top-k ceiling at equal set-recovery
        (measured D=512, V=1024); refit adds nothing (kept negative); the gain does
        NOT transfer to convolutive pair recall (the 37.5% channel wall binds
        there). See holographic_supermemory.bundle_decode."""
        from holographic.caching_and_storage.holographic_supermemory import bundle_decode
        return bundle_decode(m, codebook, k, method=method)

    def _scale_tap(self, message):
        """Advisory side channel for capacity walls (item 12): the measured laws
        fire WHERE the spec is visible -- at the encode/bundle call -- as a
        stdlib warning plus a stored note, never a changed return type or a
        refusal. warnings.warn is suppressible, testable, and standard; the
        default behaviour of every tapped faculty is byte-identical."""
        import warnings
        self._last_scale_advice = message
        warnings.warn(message, stacklevel=3)

    def last_advice(self):
        """The most recent capacity advisory this mind issued (or None). The
        taps at tree_structure (flat depth wall, measured d5) and
        superpose_batch (bundle readout k* ~ 0.13*D) write here."""
        return getattr(self, "_last_scale_advice", None)

    def path_trace_adaptive(self, sdf, camera, width=96, height=96, tol=0.02,
                        block=8, max_spp=256, min_spp=16, seed=0, **kw):
        """CI-driven adaptive path tracing: each pixel stops when its CLT 95%
        half-width is under tol*scale (MC samples are iid by construction, so
        the interval is valid -- the guard's stopping rule per pixel). Measured:
        84% of a flat 128-spp render's samples avoided at 7x under tolerance,
        spp 16-112 spatially adaptive. See holographic_pathtrace.render_adaptive."""
        from holographic.rendering.holographic_pathtrace import render_adaptive
        return render_adaptive(sdf, camera, width=width, height=height, tol=tol,
                               block=block, max_spp=max_spp, min_spp=min_spp,
                               seed=seed, **kw)


    def creature_webbing_report(self, creature, spec=None, field=None, samples=9, margin=0.25,
                                level=0.0, tolerance=None):
        """THE READABILITY GATE (backlog M-2): for every pair of NON-ADJACENT rig segments, is there
        material in the corridor between them where there should be a gap? This is Hecker's
        flying-squirrel bug -- one global implicit surface lets independent limbs web together --
        and it is the number a field rebuild must drive toward zero. Pass `field=` to score an
        alternative field against the same rig. See holographic_creaturereport.webbing_report."""
        import holographic.mesh_and_geometry.holographic_creaturereport as _cr
        return _cr.webbing_report(creature, spec=spec, field=field, samples=samples,
                                  margin=margin, level=level, tolerance=tolerance)

    def creature_silhouette_report(self, creature, spec=None, field=None, axis=0, res=96, level=0.0):
        """READABILITY AS NEGATIVE SPACE (backlog M-3): count the ENCLOSED holes in the creature's
        orthographic silhouette -- the gaps between legs and under the body that make a shape read as
        an animal. A blob scores ~0 holes. See holographic_creaturereport.silhouette_report."""
        import holographic.mesh_and_geometry.holographic_creaturereport as _cr
        return _cr.silhouette_report(creature, spec=spec, field=field, axis=axis, res=res, level=level)

    def rig(self, source):
        """ONE RIG TYPE over any skeleton (backlog D-1/R-3): the shared joints + per-segment bones +
        chains view, with canonical `"<chain>#<index>"` segment tags that JOIN to skin provenance.
        Takes a Creature, a Humanoid, or anything else carrying joints/bones/chains, so downstream
        code never branches on which kind of body it got -- which is what makes a hybrid a spec
        rather than a code path. See holographic_rig.rig_of."""
        import holographic.mesh_and_geometry.holographic_rig as _rg
        return _rg.rig_of(source)

    def rig_invariant(self, source):
        """PIN THE RIG (backlog R-2/D-2): a bone is one rigid segment between two joints, 1:1 with
        the rig, and cannot bend mid-shaft. Returns {segments, tags, joints, chains, degenerate,
        reference_length} and RAISES on violation. Holds on creature AND humanoid -- an invariant
        enforced on one rig is a convention, not a contract. See holographic_rig.rig_invariant."""
        import holographic.mesh_and_geometry.holographic_rig as _rg
        return _rg.rig_invariant(source)

    def rig_roles(self, source, ground_frac=0.35):
        """CAPABILITY TAGS FROM GEOMETRY (backlog R-5): label segments `foot` / `tip` / `torso` with
        no authoring, on any body plan, so gait and animation find parts by ROLE rather than by name.
        Returns the Rig; query it with .find_by_role('foot') (exact, authoritative) or
        .find_by_role_holographic (the VSA unbind path, measured cliff in its docstring).
        See holographic_rig.auto_roles."""
        import holographic.mesh_and_geometry.holographic_rig as _rg
        return _rg.auto_roles(source, ground_frac=ground_frac)

    def creature_tree(self, source, blend=None, groups=True, group_blend=0.0, taper=0.6,
                      spine_radius=None, limb_radius=None, head=True, radii=None,
                      op="smooth", blend_rel=0.5, tip_inset=0.0, mount_flare=0.0):
        """THE SKIN AS A COMPOSITION TREE, not one global summed field (backlog F-1/F-2/F-4) -- the
        fix for limbs melting into the torso. Parent-child segments blend at their shared joint;
        everything else HARD-unions, so webbing between unrelated limbs is not reduced but made
        UNEXPRESSIBLE (Hecker's metaball groups). MEASURED on the shipped quadruped: webbing_pairs
        76 -> 0, silhouette negative space 0.130 -> 0.443. The joint blend is RELATIVE by default
        (`blend_rel` = a multiple of the thinner segment's radius, clamped at 1.0) because an
        ABSOLUTE blend webs at large values under BOTH operators; pass `blend=` to force one. Returns an SDF, so it meshes, raymarches
        and emits a Shadertoy shader through the existing machinery (NOT the 4-dialect WGSL emitter:
        bones are capsules and that table declares capsule unemittable -- verified, not assumed). DEFAULT-OFF: creature_field is untouched.
        `groups=False` gives the softer variant that mounts limbs with a blend instead of a union.
        See holographic_creaturetree.creature_tree / creature_tree_grouped."""
        import holographic.mesh_and_geometry.holographic_creaturetree as _ct
        # tip_inset / mount_flare go in the kw DICT, not on one call: the grouped branch is the
        # DEFAULT (groups=True) and forwards **kw, so adding them to the ungrouped call alone would
        # have made them a silent no-op for most callers -- a worse bug than the drift being fixed.
        kw = dict(taper=taper, spine_radius=spine_radius, limb_radius=limb_radius,
                  head=head, radii=radii, op=op, blend_rel=blend_rel,
                  tip_inset=tip_inset, mount_flare=mount_flare)
        if groups:
            return _ct.creature_tree_grouped(source, group_blend=group_blend, blend=blend, **kw)
        return _ct.creature_tree(source, blend=blend, **kw)

    def tissue_fields(self, source, body=None, blend_rel=0.5, organs=True, **kw):
        """VOLUMETRIC ANATOMY (backlog T-1/T-2/T-3): one nested SDF per tissue -- {'bone','muscle',
        'fat','skin'} plus 'organ' viscera in the cavity -- grown OUTWARD from bone, so you set muscle and fat per bone and the skin falls out
        (D-3, the inside-out control that makes one skeleton a whippet or a bulldog). Each layer is
        compiled by the same creature_tree, so hiding and cutting compose for free. Pass the shipped
        body_params block as `body`. See holographic_creaturetissue.tissue_fields."""
        import holographic.mesh_and_geometry.holographic_creaturetissue as _ct
        return _ct.tissue_fields(source, body=body, blend_rel=blend_rel, organs=organs, **kw)

    def organ_field(self, source, organs=None, body=None, blend_rel=0.35, fields=None, clearance=1.02):
        """VISCERA AS METABALLS (backlog T-3) -- and this is the ONE place metaballs are right: a liver
        genuinely IS a smooth blob and neighbouring organs genuinely DO press against each other,
        which is the same property that ruins a limb. Placed in ANATOMY SPACE so they ride a spine
        bend or a thickness edit, shrunk to fit inside the muscle envelope, with the bone field
        SUBTRACTED so an organ cannot occupy a vertebra. Included in mind.tissue_fields as 'organ'.
        See holographic_creaturetissue.organ_field."""
        import holographic.mesh_and_geometry.holographic_creaturetissue as _ct
        kw = {} if organs is None else {"organs": organs}
        return _ct.organ_field(source, body=body, blend_rel=blend_rel, fields=fields,
                               clearance=clearance, **kw)

    def centaur_spec(self, body=None):
        """THE HYBRID REGRESSION SPEC (backlog D-1 / Tier 9): a horse body with a humanoid torso and
        arms mounted ON that torso. Nothing in the engine knows what a centaur is -- it is a spec, not
        a code path, which is the whole claim of D-1. Feed it to mind.creature like any other. Proves
        one rig, one skin compiler, one tissue stack and one gait handle a body with no special case.
        See holographic_creature.centaur_spec."""
        import holographic.mesh_and_geometry.holographic_creature as _c
        return _c.centaur_spec(body=body)

    def rotation_invariance_probe(self, build, measure, axes=((0, 0, 1), (1, 0, 0), (0, 0, -1), (1, 0, 1)),
                                  tol=0.0):
        """DOES THIS SURVIVE A CHANGE OF BODY ORIENTATION? -- the directional twin of
        scale_invariance_probe. That one exists because the same absolute-vs-relative bug hit
        DISTANCES four times; this one because it hit DIRECTIONS three times (mirror plane, limb dir,
        spine arch). `build(axis)` points the body that way, `measure(obj)` returns what should not
        depend on orientation. CANNOT know which quantities SHOULD rotate, and that is the point: a
        reared body genuinely has fewer feet on the ground, because gravity is a WORLD quantity while
        shape is a BODY one. See holographic_rig.rotation_invariance_probe."""
        import holographic.mesh_and_geometry.holographic_rig as _rg
        return _rg.rotation_invariance_probe(build, measure, axes=axes, tol=tol)

    def scale_invariance_probe(self, build, reference_length, measure, factor=3.0, tol=0.02):
        """DOES THIS SURVIVE A CHANGE OF BODY SIZE (backlog X-3/D-7)? The generalised form of a bug
        this codebase has now found FOUR separate times -- texture cell_scale, marching resolution,
        joint blend radius and organ blend were each an ABSOLUTE length where a body-relative one
        belonged. `build(scale)` makes the thing at a body scale, `measure(obj, L)` reads the quantity
        that must stay constant. Reports rather than raises, because seeing the broken AND fixed
        numbers side by side is the evidence. See holographic_rig.scale_invariance_probe."""
        import holographic.mesh_and_geometry.holographic_rig as _rg
        return _rg.scale_invariance_probe(build, reference_length, measure, factor=factor, tol=tol)

    def spine_station(self, creature, t):
        """ANATOMY SPACE (p, tangent, normal, binormal) at fraction `t` along the backbone. Anything
        placed in these coordinates RIDES body edits -- bend the spine or change its profile and the
        placement follows. Sockets, scales, rig-bound paint, limb sockets and organ placement all use
        it. See holographic_creaturesocket.spine_station."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as _cs
        return _cs.spine_station(creature, t)

    def tissue_at(self, points, fields, level=0.0):
        """WHICH TISSUE IS AT THIS POINT (backlog T-4): 'bone'|'muscle'|'fat'|'skin'|'air'. Tissue is
        a volumetric MATERIAL channel, not separate geometry -- this is the primitive the cutaway and
        the layer toggles rest on. See holographic_creaturetissue.tissue_at."""
        import holographic.mesh_and_geometry.holographic_creaturetissue as _ct
        return _ct.tissue_at(points, fields, level=level)

    def anatomy_report(self, source, fields=None, body=None, samples=6000, seed=0, level=0.0):
        """IS THE ANATOMY ACTUALLY NESTED (backlog M-4)? Asserts bone within muscle within fat within
        skin at sampled points and reports each tissue's share of interior volume, so a pretty cut
        face cannot hide geometry that is simply wrong. See holographic_creaturetissue.anatomy_report."""
        import holographic.mesh_and_geometry.holographic_creaturetissue as _ct
        return _ct.anatomy_report(source, fields=fields, body=body, samples=samples, seed=seed, level=level)

    def tissue_visible_field(self, fields, hide=(), cut=None, level=0.0):
        """SEE INSIDE (backlog V-1/V-2/V-4): hide tissue layers and/or cut with a plane, returning ONE
        field. Hide the skin and the ray passes to muscle; hide muscle and fat and the whole skeleton
        is visible IN PLACE -- a better verifier than a cross-section, which only shows the bones its
        plane happens to cross. `cut` is (point, normal). Hiding and cutting are orthogonal and
        compose. See holographic_creaturetissue.visible_field."""
        import holographic.mesh_and_geometry.holographic_creaturetissue as _ct
        return _ct.visible_field(fields, hide=hide, cut=cut, level=level)

    def tissue_weights(self, source, points, body=None, falloff=1.5, max_bones=4):
        """SKIN WEIGHTS FROM ANATOMY (backlog T-5): tissue formed around bone B belongs to bone B, so
        weights come from distance to the bone AXIS rather than from which metaball happened to land
        nearest. Retires the provenance approximation behind the fat-torso shear. Ties resolve to the
        lowest index (bilateral symmetry ties exactly). See holographic_creaturetissue.tissue_weights."""
        import holographic.mesh_and_geometry.holographic_creaturetissue as _ct
        return _ct.tissue_weights(source, points, body=body, falloff=falloff, max_bones=max_bones)

    def integument_stack(self, taxon, with_bone=None, with_organ=True, seed=0, **kw):
        """The INTEGUMENT shading stack (dermis/epidermis/coat, with bone and organ as TINTS under a
        translucent dermis). The honest name for what `anatomy_stack` always was -- it is shading, not
        anatomy, and no bone exists in space in it (backlog B-4). For real volumetric anatomy use
        mind.tissue_fields. `anatomy_stack` still works. See holographic_creaturematerial."""
        import holographic.materials_and_texture.holographic_creaturematerial as _cm
        return _cm.integument_stack(taxon, with_bone=with_bone, with_organ=with_organ, seed=seed, **kw)

    def build_creature(self, spec, parts=True, ground=True, cage_res=40, library=None,
                       quads=True, lods=None, body=None, subdiv=1, mount_flare=0.55,
                       fuse_parts=False, fuse_blend=0.012, foot_size=1.0, surface="sdf",
                       pose=None, gait="walk", period=1.0, **kw):
        """MAKE A CREATURE -- one call: spec -> body -> skin -> mesh -> parts -> grounded, scored.
        The entry point that was MISSING: find_capability('make a creature') used to return the parts
        library, the body-shape module and the editor -- everything except how to make one. Returns
        {creature, rig, field, mesh, parts, sockets, ground, score, quads, lods}. `quads=True` runs
        the field-guided tris-to-quads retopo over the scaffold (measured 78% quads, 15,136 -> 8,491
        faces, vertices UNMOVED at 1.32e-16) -- quads following the limb are what clean deformation
        needs. `lods=(0.5, 0.25)` builds a SILHOUETTE-GUARDED decimation chain, so a level that saves
        faces by eating a limb is refused rather than shipped. KEPT NEGATIVE: attaching a foot
        does not yet make a leg READ as having a foot (the limb's own capsule already caps that
        space; measured 0.58% of pixels change). See holographic_creaturetree.build_creature."""
        import holographic.mesh_and_geometry.holographic_creaturetree as _ct
        return _ct.build_creature(spec, parts=parts, ground=ground, cage_res=cage_res,
                                  library=library, quads=quads, lods=lods, body=body,
                                  subdiv=subdiv, mount_flare=mount_flare, fuse_parts=fuse_parts,
                                  fuse_blend=fuse_blend, foot_size=foot_size, surface=surface,
                                  pose=pose, gait=gait, period=period, mind=self, **kw)

    def measured_ratio(self, numerator, denominator, of, note=""):
        """A RATIO THAT CARRIES ITS DENOMINATOR -- the measurement twin of D-7's reference length.
        `of` names the denominator in words and is required. Exists because "parts change 0.58% of
        pixels, so parts do not read" was one mistake reported as two findings: 0.58% was of the whole
        IMAGE (~95% background); against the BODY the same parts add 11% of silhouette. A percentage
        without its denominator measures the framing, not the subject.
        See holographic_creaturereport.ratio."""
        import holographic.mesh_and_geometry.holographic_creaturereport as _cr
        return _cr.ratio(numerator, denominator, of, note=note)

    def render_specimen(self, sdf, eye, target, material, sky, width=240, height=180, fov_deg=42.0,
                        tol=0.02, min_spp=16, max_spp=64, max_bounce=5, seed=0, denoise=True,
                        albedo_fn=None, prefer="colour", far=12.0, budget_s=None, bake=False):
        """ONE CALL from an SDF to a finished image: ADAPTIVE trace -> G-buffer -> SVGF denoise ->
        firefly clamp -> searched exposure. `tol` replaces a sample count -- state the quality, and
        pixels that have converged stop being sampled (measured 83% of a flat 48-spp render's samples
        avoided at equal mean radiance). Composes capabilities that already existed separately and
        were hand-wired for every render, which is how the newest one kept getting missed.
        CAUTION, measured: at min_spp=8 the block CI is optimistic and declares convergence
        everywhere (spp 8-8 flat); 16 escalates properly. See holographic_gemrender."""
        import holographic.rendering.holographic_gemrender as _gr
        return _gr.render_specimen(self, sdf, eye, target, material, sky, width=width, height=height,
                                   fov_deg=fov_deg, tol=tol, min_spp=min_spp, max_spp=max_spp,
                                   max_bounce=max_bounce, seed=seed, denoise=denoise,
                                   albedo_fn=albedo_fn, prefer=prefer, far=far,
                                   budget_s=budget_s, bake=bake)

    def render_plan(self, sdf, eye, target, material, sky, width=240, height=180, fov_deg=42.0,
                    budget_s=None, min_spp=16, max_spp=64, max_bounce=5, probe=(64, 52), seed=0,
                    safety=0.55):
        """MEASURE a render's cost on a tiny tile, then say what fits. Four overruns in this repo's
        history came from extrapolating LINEARLY off a cheap probe -- on a transmissive scene that
        always understates, because more samples means more rays surviving deep enough to enter glass
        and march through interiors. The estimate is deliberately pessimistic and the measurement is
        reported beside it. `budget_s` returns fits/suggest; render_specimen(budget_s=) resizes to fit
        rather than overrunning. See holographic_gemrender.render_plan."""
        import holographic.rendering.holographic_gemrender as _gr
        return _gr.render_plan(self, sdf, eye, target, material, sky, width=width, height=height,
                               fov_deg=fov_deg, budget_s=budget_s, min_spp=min_spp, max_spp=max_spp,
                               max_bounce=max_bounce, probe=probe, seed=seed, safety=safety)

    def render_grade(self, img, prefer="colour"):
        """Tone map with a SEARCHED exposure rather than a guessed one -- the white point is swept and
        scored on saturation and contrast. The obvious choice is measurably wrong: at white=p99 a gem
        render came back with 0.2% highlights and no dielectric sparkle at all.
        See holographic_gemrender.grade."""
        import holographic.rendering.holographic_gemrender as _gr
        return _gr.grade(img, prefer=prefer)

    def render_gbuffer(self, sdf, eye, target, width, height, fov_deg=42.0, far=12.0, albedo_fn=None):
        """Depth, normal and albedo by sphere tracing -- the inputs svgf_denoise needs and the path
        tracer does not return. The shipped sdf_depth_cpu fixes its own camera orientation and cannot
        match an arbitrary view, which is why this exists. See holographic_gemrender.gbuffer."""
        import holographic.rendering.holographic_gemrender as _gr
        e, d = _gr.camera_rays(eye, target, width, height, fov_deg)
        return _gr.gbuffer(sdf, e, d, far=far, albedo_fn=albedo_fn)

    def crystal_flawed_material(self, gem, cloud=None, incl=None, phan=None, frac=None,
                                incl_color=(0.20, 0.14, 0.10), absorb_scale=1.0):
        """A gem material PLUS its imperfections, as the path tracer's 8-channel callback. A perfect
        crystal reads as glass; specimens read as mineral because they are flawed in structured ways.
        Each field modifies OPTICS, not shape: cloudiness raises absorption NEUTRALLY across RGB
        (milkiness is scattering, so it whitens rather than saturating -- measured [1.92,1.92,1.92]),
        inclusions swap in a dark albedo and absorb ~8x harder, phantoms veil a concentric shell,
        fractures brighten thin sheets. See holographic_crystalflaw.flawed_material."""
        import holographic.materials_and_texture.holographic_crystalflaw as _cf
        return _cf.flawed_material(gem, cloud=cloud, incl=incl, phan=phan, frac=frac,
                                   incl_color=incl_color, absorb_scale=absorb_scale)

    def crystal_cloudiness(self, strength=1.0, freq=9.0, seed=0, threshold=0.45, sharp=5.0):
        """MILKY zones as a field P -> [0,1]. Milky quartz is not a different mineral, it is quartz
        full of sub-micron fluid inclusions that SCATTER -- optically neutral absorption plus interior
        roughness. See holographic_crystalflaw.cloudiness."""
        import holographic.materials_and_texture.holographic_crystalflaw as _cf
        return _cf.cloudiness(strength=strength, freq=freq, seed=seed, threshold=threshold, sharp=sharp)

    def crystal_inclusions(self, count=40, radius=0.035, extent=1.0, seed=0, elongation=1.0):
        """Suspended foreign bodies -- blebs, or NEEDLES at elongation>1 (rutilated quartz). A FIELD,
        not geometry: an inclusion is seen THROUGH the host, so what matters is that light crossing it
        is absorbed and coloured differently. See holographic_crystalflaw.inclusions."""
        import holographic.materials_and_texture.holographic_crystalflaw as _cf
        return _cf.inclusions(count=count, radius=radius, extent=extent, seed=seed, elongation=elongation)

    def crystal_phantom(self, habit_sdf, fractions=(0.45, 0.72), width=0.035, strength=1.0):
        """PHANTOMS: ghosts of the crystal at earlier, smaller sizes. A growth pause dusts the
        surface and later growth buries it, so the ghost is the SAME habit scaled down -- which is why
        phantoms are concentric with their crystal and never arbitrary blobs. Measured: the veil hugs
        the scaled habit surface to |d| 0.011. See holographic_crystalflaw.phantom."""
        import holographic.materials_and_texture.holographic_crystalflaw as _cf
        return _cf.phantom(habit_sdf, fractions=fractions, width=width, strength=strength)

    def crystal_fractures(self, count=6, seed=0, width=0.02, extent=1.0):
        """Internal cleavage planes -- thin bright sheets that catch light inside the crystal.
        See holographic_crystalflaw.fractures."""
        import holographic.materials_and_texture.holographic_crystalflaw as _cf
        return _cf.fractures(count=count, seed=seed, width=width, extent=extent)

    def crystal_chipped(self, sdf, count=10, radius=0.06, extent=0.6, seed=0):
        """Knock chips off a crystal -- the one imperfection that changes GEOMETRY. Real specimens are
        damaged, and mathematically perfect edges are what read as CGI. Subtracted spheres are placed
        near the surface, so corners and edges take the damage as they do in life. Pinned to only ever
        REMOVE material. See holographic_crystalflaw.chipped."""
        import holographic.materials_and_texture.holographic_crystalflaw as _cf
        return _cf.chipped(sdf, count=count, radius=radius, extent=extent, seed=seed)

    def material_trace_channels(self, name, scale=1.0):
        """A named material as the PATH TRACER's channel callback -- albedo, metallic, roughness,
        emission, IOR, subsurface, iridescence AND Beer-Lambert ABSORPTION, in one call. Use this
        instead of hand-building the tuple: callers that assembled it by hand silently dropped every
        channel added after they were written, which is exactly how the gem renders ran an entire arc
        with absorption unset. `scale` multiplies sigma, which is in 1/scene-unit.
        See holographic_matlib.trace_channels."""
        import holographic.materials_and_texture.holographic_matlib as _ml
        return _ml.trace_channels(name, scale=scale)

    def material_absorption(self, name):
        """The Beer-Lambert ABSORPTION coefficient (sigma per RGB, 1/scene-unit) of a named material.
        This is what gives a transmissive solid DEPTH -- light is attenuated by the distance it
        travelled INSIDE it, so a thick gem is darker and more saturated than a thin edge, where an
        albedo tint makes them identical. Amethyst (1.10, 2.30, 0.55), diamond ~0.01 (nearly clear).
        See holographic_matlib._ABSORB."""
        import holographic.materials_and_texture.holographic_matlib as _ml
        return tuple(_ml.material(name).absorption)

    def crystal_grow_on(self, sdf, bounds, count=24, habit="quartz", size=0.18, size_jitter=0.45,
                        inward=False, tilt=0.18, where=None, seed=0, substrate=True, batched=False):
        """GROW CRYSTALS ON ANY SURFACE. Seeds land on the SDF and each crystal's c-axis aligns to the
        surface NORMAL, because a crystal grows perpendicular to what it nucleated on -- which is why
        a druse radiates and (with inward=True) why a geode points at its own middle. `where` is a
        weight FIELD, so crystals grow only where a material says: measured, gating raised the mean
        field value under the crystals from 0.313 to 0.577. See holographic_crystalgrow.grow_on."""
        import holographic.mesh_and_geometry.holographic_crystalgrow as _cg
        return _cg.grow_on(sdf, bounds, count=count, habit=habit, size=size,
                           size_jitter=size_jitter, inward=inward, tilt=tilt, where=where,
                           seed=seed, substrate=substrate, batched=batched)

    def crystal_cluster(self, count=9, habit="quartz", size=0.30, radius=0.22, seed=0, **kw):
        """A free-standing DRUSE -- crystals radiating from a small rocky base. This is grow_on with
        the substrate being a blob, which is what a druse physically IS: the base is the thing they
        nucleated on and the reason they point outward. See holographic_crystalgrow.cluster."""
        import holographic.mesh_and_geometry.holographic_crystalgrow as _cg
        return _cg.cluster(count=count, habit=habit, size=size, radius=radius, seed=seed, **kw)

    def crystal_geode(self, radius=0.7, shell=0.16, count=60, habit="quartz", size=0.13,
                      seed=0, where=None, **kw):
        """A GEODE: a hollow nodule whose cavity wall is lined with INWARD-pointing crystals, built
        from the physics rather than as a special shape. MEASURED hollow: 0.00 filled at the centre,
        1.00 in the rind, with a distinct crystal band between. `where` can leave part of the wall
        bare. Slice it with crystal_cut to look inside. See holographic_crystalgrow.geode."""
        import holographic.mesh_and_geometry.holographic_crystalgrow as _cg
        return _cg.geode(radius=radius, shell=shell, count=count, habit=habit, size=size,
                         seed=seed, where=where, **kw)

    def crystal_cut(self, field, normal=(1.0, 0.0, 0.0), point=(0.0, 0.0, 0.0)):
        """Slice a solid with a half-space -- how you LOOK INSIDE a geode. Material is kept on the
        NEGATIVE side so the cut face points along +normal, and the returned field says so via
        `cut_face_normal`: a camera on the wrong side sees the intact back of the nodule and no hint
        anything was cut. See holographic_crystalgrow.cut."""
        import holographic.mesh_and_geometry.holographic_crystalgrow as _cg
        return _cg.cut(field, normal=normal, point=point)

    def crystal_habits(self):
        """The named crystal habits (quartz, beryl, cube, octahedron, dodecahedron, needle) with
        their systems and face forms. Sizes inside a habit are RELATIVE, so one `size` scales the
        crystal without changing its proportions. See holographic_crystalgrow.HABITS."""
        import holographic.mesh_and_geometry.holographic_crystalgrow as _cg
        return dict(_cg.HABITS)

    def crystal_vein_field(self, scale=6.0, threshold=0.55, seed=0, sharpness=6.0):
        """A veined WEIGHT field (P -> [0,1]) for gating where crystals grow -- the demonstrator for
        'crystals only where this material is'. Any callable works; this one is deterministic and
        NumPy-only. See holographic_crystalgrow.vein_field."""
        import holographic.mesh_and_geometry.holographic_crystalgrow as _cg
        return _cg.vein_field(scale=scale, threshold=threshold, seed=seed, sharpness=sharpness)

    def crystal_form_faces(self, system, hkl):
        """EXPAND A MILLER FORM {hkl} into its symmetry-equivalent faces. In crystallography the
        braces mean the FORM -- {100} cubic is SIX faces (a cube), {111} is EIGHT (an octahedron).
        `crystal_habit` takes explicit faces and adds only the centrosymmetric pair, so asking for
        {100} and expecting a cube silently yields a SLAB (measured volume 5.76 vs a cube's 1.00, and
        not invariant under a 90-degree turn). Pass `form=True` to crystal_habit, or use this.
        See holographic_bravais.form_faces."""
        import holographic.mesh_and_geometry.holographic_bravais as _bv
        return _bv.form_faces(system, hkl)

    def convolution_field(self, segments, iso=0.35, samples=24, kernel=2.2):
        """CONVOLUTION SURFACE over a CONTIGUOUS skeleton (Bloomenthal & Shoemake 1991). Sum the
        convolution of every segment: because of the superposition property, contiguous primitives
        produce NO BULGE at their joints -- unlike a smooth_union, where the blend IS the bulge.
        MEASURED on a right-angle joint: smooth-union corner 0.1065 vs convolution 0.0788, 26% less.
        Segments are (a, b, radius) or (a, b, radius, aniso) where `aniso` gives the ellipsoidal
        normal section of Fuentes Suarez, Hubert & Zanni 2019 -- (along, wide, flat), which is how a
        sole gets to be flat without extra primitives. See holographic_creatureconv."""
        import holographic.mesh_and_geometry.holographic_creatureconv as _cv
        return _cv.convolution_field(segments, iso=iso, samples=samples, kernel=kernel)

    def convolution_groups(self, groups, iso=0.35, samples=20, kernel=2.2):
        """CONVOLUTION GROUPS: contiguity kills the bulge WITHIN a group, a hard union between groups
        kills the blending BETWEEN them. That answers the open problem Bloomenthal names in his own
        hand paper -- "when two fingers approach each other, they should not blend". MEASURED on a
        3-toe fan: summed convolution shows 1 solid run (all toes webbed into one blob, his defect
        reproduced), grouped shows 3. See holographic_creatureconv.grouped_field."""
        import holographic.mesh_and_geometry.holographic_creatureconv as _cv
        return _cv.grouped_field(groups, iso=iso, samples=samples, kernel=kernel)

    def foot_skeleton(self, size=1.0, digits=3, spread=0.7, toe_len=0.9, sole_flat=0.45):
        """A FOOT AS A SKELETON, the way the convolution-surface literature builds one: a contiguous
        heel-arch-ball-ankle chain carrying a flat anisotropic section, with each toe its own
        contiguous chain in its own group. Returns (groups, sole_aniso) for convolution_groups.
        See holographic_creatureconv.foot_skeleton."""
        import holographic.mesh_and_geometry.holographic_creatureconv as _cv
        return _cv.foot_skeleton(size=size, digits=digits, spread=spread, toe_len=toe_len,
                                 sole_flat=sole_flat)

    def creature_auto_sockets(self, source, field=None, feet=True, head_parts=True, hands=True,
                              ears=False, horns=False, spikes=False, part_scale=2.4, foot_frac=0.13):
        """WHERE PARTS GO, from the rig's ROLE TAGS (backlog R-5/P-1). A ground-touching tip gets a
        FOOT; a LATERAL non-ground tip gets a HAND; the head gets eyes and a mouth, optionally ears,
        horns and a dorsal spike ridge. MEASURED across body plans from ONE rule set, no per-plan
        table: quadruped 4 feet / 0 hands, centaur 4 feet + 2 hands, humanoid 2 + 2. Medial tips are
        excluded because a neck or tail is a tip too -- the centaur got a hand on its neck until that
        was fixed. See holographic_creaturetree.auto_sockets."""
        import holographic.mesh_and_geometry.holographic_creaturetree as _ct
        return _ct.auto_sockets(source, field=field, feet=feet, head_parts=head_parts, hands=hands,
                                ears=ears, horns=horns, spikes=spikes, part_scale=part_scale,
                                foot_frac=foot_frac)

    def creature_readability_score(self, source, field=None, res=64, samples=4000, seed=0,
                                   dominance_target=0.70, dominance_weight=1.0):
        """SCORE HOW WELL A CREATURE READS (backlog A-1/A-2) -- negative space PLUS one-dominant-mass,
        with webbing as a hard feasibility gate. Two terms because one is degenerate: negative space
        alone is monotone in limb thickness (0.470 -> 0.332 as limbs thicken), so maximising it alone
        produces a spindly wisp. Dominance (spine share of body volume) runs the other way, so the
        pair has an interior optimum. See holographic_creatureproportion.readability_score."""
        import holographic.mesh_and_geometry.holographic_creatureproportion as _cp
        return _cp.readability_score(source, field=field, res=res, samples=samples, seed=seed,
                                     dominance_target=dominance_target,
                                     dominance_weight=dominance_weight)

    def creature_proportion_search(self, spec, knobs=None, steps=3, res=56, samples=3000, seed=0):
        """SEARCH A SPEC FOR READABILITY (backlog A-1/A-2/A-3), per Togelius et al.'s search-based PCG:
        define an evaluation function and search it rather than hand-coding proportion rules. Sweeps
        declared candidate values (limb thickness, spine curve = line of action), keeps the best
        FEASIBLE one, and returns the full trace so the search's own evidence is inspectable.
        See holographic_creatureproportion.proportion_search."""
        import holographic.mesh_and_geometry.holographic_creatureproportion as _cp
        return _cp.proportion_search(spec, knobs=knobs, steps=steps, res=res, samples=samples, seed=seed)

    def creature_ground(self, source, field=None, res=48):
        """STAND IT ON THE GROUND (backlog A-4): the translation putting the body's lowest MATERIAL at
        y=0, measured from the field rather than the joints (a foot's flesh reaches below its bone, so
        a rig-only answer floats the creature by a limb radius). Reports `supported` -- at least three
        ground contacts, the minimum for static support. A creature floating in abstract space reads
        as an object; planted, it reads as an animal. See holographic_creatureproportion.ground_creature."""
        import holographic.mesh_and_geometry.holographic_creatureproportion as _cp
        return _cp.ground_creature(source, field=field, res=res)

    def creature_head_definition(self, source, field=None, samples=400, reach=0.6):
        """DOES THE HEAD READ AS A HEAD (backlog A-1)? Returns the half-thickness profile along the
        spine plus head_ratio, neck_ratio and has_neck. FOUND BY LOOKING at an ASCII profile while
        every score was green: the shipped quadruped's head IS 1.43x its body, but the profile is
        MONOTONE (0.104, 0.114, 0.113, 0.114, 0.163) so nothing separates head from torso and it reads
        as a tube that thickens. A big/medium/small hierarchy needs the PINCH, not just the size.
        See holographic_creatureproportion.head_definition."""
        import holographic.mesh_and_geometry.holographic_creatureproportion as _cp
        return _cp.head_definition(source, field=field, samples=samples, reach=reach)

    def creature_mass_dominance(self, source, field=None, samples=6000, seed=0):
        """WHAT SHARE OF THE BODY IS THE TORSO (backlog A-1's big-shape/small-shape hierarchy)?
        Ownership is nearest-bone-axis, the same rule tissue_weights uses, so it cannot disagree with
        the skinning about which part a point belongs to. Limbs with similar visual weight to the
        torso score near 0.5. See holographic_creatureproportion.mass_dominance."""
        import holographic.mesh_and_geometry.holographic_creatureproportion as _cp
        return _cp.mass_dominance(source, field=field, samples=samples, seed=seed)

    def creature_scaffold_mesh(self, source, field=None, cage_res=40, iters=10, factor=1.0,
                               radii=None, **kw):
        """SCAFFOLD MESHING (backlog M-5): build a coarse cage AROUND THE SKELETON and project it onto
        the field, instead of marching one global grid. A global grid is sized for the whole body, so
        a thin limb gets a few cells across it and comes out LUMPY; a scaffold's density follows the
        skeleton. MEASURED on the shipped quadruped's thinnest segment: radial ripple 25.4% (marching,
        res 40) -> 1.6% (scaffold), with FEWER vertices (7,570 vs 10,754), and the verts land on the
        isosurface to 1e-16. Composition of skin_skeleton + shrinkwrap_field + creature_tree, all
        three already shipped. See holographic_creaturetree.scaffold_mesh."""
        import holographic.mesh_and_geometry.holographic_creaturetree as _ct
        return _ct.scaffold_mesh(source, field=field, cage_res=cage_res, iters=iters,
                                 factor=factor, radii=radii, **kw)

    def shrinkwrap_field(self, mesh, field, factor=1.0, iters=8, level=0.0, eps=None, max_step=None):
        """SNAP A MESH ONTO AN IMPLICIT SURFACE -- shrinkwrap generalised from a mesh target to a
        FIELD one. A distance field's value is the distance and its gradient the direction, so the
        closest point is Newton iteration, no search structure needed. The step is clamped, because a
        smooth-union tree can exceed the Lipschitz bound locally and an unclamped step overshoots to
        the far side of a limb. See holographic_meshtools.shrinkwrap_field."""
        import holographic.mesh_and_geometry.holographic_meshtools as _mt
        return _mt.shrinkwrap_field(mesh, field, factor=factor, iters=iters, level=level,
                                    eps=eps, max_step=max_step)

    def bone_capsule(self, a, b, r):
        """One rigid bone as an exact capsule SDF between joints `a` and `b` (a sphere if degenerate)
        -- the shared limb primitive every rig builds from. Promoted out of the humanoid, which now
        delegates to it. See holographic_creaturetree.bone_capsule."""
        import holographic.mesh_and_geometry.holographic_creaturetree as _ct
        return _ct.bone_capsule(a, b, r)

    def rig_from_primitives(self, fit, min_length=1e-6):
        """CLOSE THE LOOP (backlog L-1/D-6): turn fit_primitives output into the SHARED Rig type, so
        an observed body and a generated one are the same kind of thing. A fitted CAPSULE is a bone
        segment; spheres are skipped rather than faked into zero-length bones. The result feeds
        creature_tree / tissue_fields / the reports with no new code path.
        See holographic_rig.rig_from_primitives."""
        import holographic.mesh_and_geometry.holographic_rig as _rg
        return _rg.rig_from_primitives(fit, min_length=min_length)

    def rig_from_mesh(self, mesh, res=28, pad=0.1, nbins=12, min_length=1e-6):
        """RECOVER A SPINED RIG FROM A MESH (backlog L-2): the medial-axis centerline becomes a
        `spine` chain with per-segment MEDIAL RADIUS -- the shape's own measurement of how thick it
        is. Unlike rig_from_primitives (segments, no backbone) this gives a PARENTED chain, so joint
        blending and anatomy space work on an observed body. Returns (rig, thickness). KEPT NEGATIVE:
        single-branch -- it recovers the TORSO of a limbed creature, not its limbs.
        See holographic_rig.rig_from_mesh."""
        import holographic.mesh_and_geometry.holographic_rig as _rg
        return _rg.rig_from_mesh(mesh, res=res, pad=pad, nbins=nbins, min_length=min_length)

    def infer_tissue_fractions(self, rig, thickness, bone_frac=0.45, skin_frac=0.06):
        """INFER TISSUE FROM AN OBSERVED SKIN (backlog L-3): muscle and fat from the gap between the
        fitted bone and the measured surface, exactly as the anatomy literature derives it. Returns
        {'radii', 'soft_fraction', 'body'} where `body` is body_params-shaped -- so an INFERRED body
        drives tissue_fields through the same control surface an authored one does. KEPT NEGATIVE:
        the muscle/fat SPLIT is not observable from a silhouette; only the total is.
        See holographic_rig.infer_tissue_fractions."""
        import holographic.mesh_and_geometry.holographic_rig as _rg
        return _rg.infer_tissue_fractions(rig, thickness, bone_frac=bone_frac, skin_frac=skin_frac)

    def creature_regression_report(self, seed=0, res=64, samples=3000):
        """RUN THE THREE REGRESSION SPECS through one pipeline (backlog Tier 9): a quadruped
        (baseline), a CENTAUR (hybrid -- proves hybrids are specs, not code paths) and a rig FITTED
        from a point cloud (proves observe and generate produce the same rig). Reports segments,
        webbing, negative space and anatomy nesting per body. Any body plan needing a special case
        would have to put it here, in the open. See holographic_creaturereport.regression_report."""
        import holographic.mesh_and_geometry.holographic_creaturereport as _cr
        return _cr.regression_report(seed=seed, res=res, samples=samples)

    def creature_part_ids(self, creature, spec=None, points=None, spacing=1.0):
        """WHICH RIG SEGMENT OWNS EACH POINT (backlog M-1) -- the flat per-part colour test. A hard
        colour seam at a limb/torso junction means two parts meeting; NO seam confirms one global
        blended field. Per-segment since B-1. See holographic_creaturereport.part_colour_ids."""
        import holographic.mesh_and_geometry.holographic_creaturereport as _cr
        return _cr.part_colour_ids(creature, spec=spec, points=points, spacing=spacing)

    def split_plan(self, paths, contended=True):
        """Dispatch work PROPORTIONALLY across contending paths instead of picking one winner
        (holographic_splitplan). `paths` is [{"name", "throughput": items/sec, "shares_bus": bool}] with
        throughput MEASURED, never a spec-sheet number.
        WHY (FreeToken, arXiv 2608.16157): an MoE expert miss can be served over PCIe on the GPU or on the
        CPU where it already lives, and BOTH READ THE SAME SYSTEM MEMORY -- they compete for one pool
        rather than adding. Engines that pick one strategy at load time miss most of what the router asks
        for; FreeToken measures both bandwidths on the actual machine and splits in proportion. Two
        machines with the same GPU can want opposite strategies, and none of it is on a spec sheet.
        THIS IS leCore'S OWN FINDING FROM THE OTHER SIDE: machine_spec_sheet's founding negative is that a
        latency-ordered hierarchy is the wrong frame because "every one is a BATCH unit whose per-access
        cost collapses with N". Units contending for one resource do not form a ladder.
        `contended=True` caps bus-sharing paths at the fastest of them rather than summing -- deliberately
        CONSERVATIVE, because over-claiming a split's benefit is how a win becomes a production regression.
        ADDITIVE: argmax(weights) is exactly the tier a picker would choose, so `compute_plan` callers are
        unaffected. `split_gain` is 0.0 when one path dominates -- the honest answer, and the case where
        picking was right."""
        from holographic.agents_and_reasoning.holographic_splitplan import split_plan
        return split_plan(paths, contended=contended)

    def render_plate(self, sdf, eye, target, material, sky, width=340, height=255, fov_deg=36.0,
                     tol=0.012, min_spp=16, max_spp=128, max_bounce=2, seed=0, denoise=True,
                     white=None, white_pct=99.0, gamma=2.2, exposure=1.0, far=12.0, albedo_fn=None,
                     budget_s=None, upsample=1):
        """render_specimen's pipeline with a FIXED white point instead of a searched one
        (holographic_plate). Identical trace -> denoise -> clamp; only the last step differs.
        WHY: render_specimen ends in grade(), which SEARCHES an exposure and RE-NORMALISES its input --
        so scaling the lights cannot change the result (three sessions were spent lowering `gain` against
        a blowout that was structurally immune to it), and on a mostly-BRIGHT subject like an anatomical
        section (fat 0.90, bone 0.87) it has no dark reference and drives the pale tissues to clipping.
        A plate also needs REPRODUCIBLE tone: a searched exposure means two renders of the same subject
        under the same light are not comparable, which defeats the point of a plate.
        `white=None` measures one at `white_pct` WITH HEADROOM and reports it -- pin that number for a
        series. Headroom is load-bearing: extended Reinhard maps L == W to exactly 1.0, so a white point
        set AT the percentile clips the top by construction (measured: 100% of a flat bright fixture).
        Report carries white / highlight_fraction / grain_raw / grain_denoised / sample_saving, so "is it
        blown out" is a MEASUREMENT rather than an opinion.
        KEPT NEG: a fixed white below the scene's real range WILL clip -- reproducible and your
        responsibility, versus adaptive and unpredictable.
        `budget_s` MEASURES rather than times out: render_plan probes the cost on a tiny tile and shrinks
        resolution/spp to fit, and the plan is reported so you see what was traded instead of silently
        receiving a smaller image. Omitting it is how a 16-minute render with no output happens.
        `upsample=N` traces COLOUR at 1/N size and upscales guided by a FULL-RES G-buffer. The asymmetry is
        the point: a G-buffer is one sphere-trace per pixel while a path trace is min_spp..max_spp bounced
        paths, so full-res geometry is nearly free and full-res colour is the whole cost. PREFER THIS OVER
        bake, whose 9.6x brings QUANTISED NORMALS and a ~0.2 radiance error that does NOT converge with
        grid resolution -- an approximation in the SHADING, where this one approximates only colour BETWEEN
        known edges. KEPT NEG: it invents plausible detail, not true detail; it is not a high-res trace."""
        from holographic.rendering.holographic_plate import render_plate
        return render_plate(self, sdf, eye, target, material, sky, width=width, height=height,
                            fov_deg=fov_deg, tol=tol, min_spp=min_spp, max_spp=max_spp,
                            max_bounce=max_bounce, seed=seed, denoise=denoise, white=white,
                            white_pct=white_pct, gamma=gamma, exposure=exposure, far=far, albedo_fn=albedo_fn,
                            budget_s=budget_s, upsample=upsample)

    def material_preview(self, sdf, eye, target, material, width=160, height=120, fov_deg=36.0, far=12.0):
        """See what a material callback ACTUALLY paints, in ~1s, WITHOUT path tracing
        (holographic_matpreview). One sphere trace gives surface positions -- the entire input to a
        material -- so albedo is nearly free while a lit render is 50-140s on a composed SDF.
        WHY: a material was only ever called BY the path tracer, so "did my change reach the pixels" cost
        a full render, which is long enough that guessing beats measuring. It did: eyes authored at radius
        0.017 that are two pixels wide, a fur term that is ~1 everywhere and flattens, a coat paler than
        its albedo -- each diagnosed only after a render.
        Report gives hit_fraction, n_unique_colours and luma_span, so a FLAT callback and a BROKEN one --
        indistinguishable in a render, two minutes each -- are told apart instantly.
        KEPT NEG: this is ALBEDO, not radiance. It cannot say the render is too bright, dark or noisy;
        those are transport and exposure questions and belong to render_plate's report."""
        from holographic.rendering.holographic_matpreview import preview_material
        return preview_material(self, sdf, eye, target, material, width, height, fov_deg, far)

    def feature_coverage(self, sdf, eye, target, features, width=160, height=120, fov_deg=36.0, far=12.0):
        """For each named (centre, radius) feature, the fraction of VISIBLE pixels it occupies
        (holographic_matpreview). `features` is [(name, centre_xyz, radius), ...].
        THE NUMBER THAT WAS MISSING FOR FOUR SESSIONS. An eye at radius 0.017 on a body 1.6 units long is
        not "small" in any way a person can judge from the spec -- it is a pixel count, and 0.0000 says so
        in one second. Each entry also reports `nearest`, the distance from the feature centre to the
        closest visible surface point, so a MISPLACED feature (large nearest) and a TOO-SMALL one (nearest
        ~0, pixels 0) are distinguished -- they need opposite fixes and both read as 'it is not there'."""
        from holographic.rendering.holographic_matpreview import feature_coverage
        return feature_coverage(self, sdf, eye, target, features, width, height, fov_deg, far)

    def verify_render_stages(self, sdf, eye, target, material, sky, width=96, height=72, budget_s=60):
        """Assert each render STAGE's contract in one pass (holographic_stagecheck) -- because four defects
        shipped in this arc and every one was a stage never measured in isolation:
        (1) tonemap_fixed had a white point and NO exposure (doubling white moved the mean 0.444 -> 0.424);
        (2) render_plate had no budget_s (16+ minutes, no output); (3) render_plan was costing the OUTPUT
        size while upsample traces at 1/N, losing 4x resolution; (4) gbuffer(albedo_fn=None) gave SVGF a
        FLAT guide, so it smoothed across every texture -- four sessions of "fur does not work".
        All four passed end-to-end testing, because end-to-end only asks whether an image came out.
        Each check asserts the EFFECT of a stage, never its implementation -- reading the code back is how
        all four survived. Small by default; a verifier that costs a full render is one nobody runs.
        KEPT NEG: verifies MECHANISM, not beauty. Every check can pass on a render that looks wrong --
        that question belongs to an eye and to material_preview at 0.044s, not to an assertion."""
        from holographic.rendering.holographic_stagecheck import verify_render_stages
        return verify_render_stages(self, sdf, eye, target, material, sky,
                                    width=width, height=height, budget_s=budget_s)

def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p14_organics", "_UnifiedPart14")
    # item 12: the capacity taps fire at the sites, silently for safe specs.
    import warnings as _w
    import numpy as np
    from lecore import UnifiedMind as _UM
    _m2 = _UM(dim=128, seed=0)
    deep = "x"
    for _i in range(6):
        deep = ("op%d" % _i, deep, "pad")
    with _w.catch_warnings(record=True) as _rec:
        _w.simplefilter("always")
        _m2.tree_structure(("add", "a", "b"))
        assert len(_rec) == 0, "shallow tree must not tap"
        _m2.tree_structure(deep)
        assert len(_rec) == 1 and "carrier" in str(_rec[0].message)
        _rk = np.random.default_rng(0).standard_normal((40, 128))
        _ri = np.random.default_rng(1).standard_normal((40, 128))
        _m2.superpose_batch(_rk, _ri, gated=False)         # keys are VECTORS
        assert len(_rec) == 2 and "0.13" in str(_rec[1].message)
    assert _m2.last_advice() is not None

    print("holographic_unified_p14_organics selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

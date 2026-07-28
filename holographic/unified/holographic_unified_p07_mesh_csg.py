"""Part 07 of UnifiedMind's faculty surface -- 90 methods, mesh_csg .. render_scene_document.

NOT A STANDALONE MODULE. This is one slice of the single `UnifiedMind` class, which grew to 17.4k lines
in one file and went past the 1 MB cap an agent can read in a single pass -- so the engine could no
longer read its own central nervous system. The class is assembled from these parts by
holographic/misc/holographic_unified.py, which is still the only import path anyone uses.

Every method here is a real attribute of UnifiedMind at runtime (mixin, not delegation), so `mind.x()`,
`dir(mind)`, the doc generators and the service's tool introspection all behave exactly as before. The
bodies were moved by line range, not regenerated, so they are byte-identical to the originals.

KEPT NEGATIVE, so nobody "tidies" it: these part classes are NOT a public API and must never be
imported or subclassed directly. They carry no `__init__` and assume the state UnifiedMind.__init__
builds; instantiated alone they would fail on the first attribute access. The leading underscore on
the class name says so, and the reachability audit reads them as referenced-by-unified, not as
standalone capabilities.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _Index
from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
from holographic.misc.holographic_creature import HolographicMind
from holographic.unified import check_part


class _UnifiedPart07:

    def mesh_csg(self, operation, mesh_a, mesh_b, res=28, bounds=None):
        """Boolean of two solids by ROUTING through the SDF (holographic_route, ARCH-7's flagship): `operation` in
        {"union","intersection","difference"}. The mesh kernel has no native booleans; this routes mesh_a, mesh_b
        -> SDF (mesh_to_sdf), combines the fields (min / max / max-with-negation), and extracts back to a mesh
        (marching tetrahedra). The result can CHANGE TOPOLOGY -- overlapping solids merge to one component, separate
        ones stay two -- which a mesh cannot do to itself. Geometrically correct (satisfies inclusion-exclusion).
        Returns a Mesh. Kept negative: resolution is the grid's (sharp seams round); the input meshes' SDF sign must
        be reliable (convex-ish), per FWD-11."""
        from holographic.scene_and_pipeline.holographic_route import route_csg
        return route_csg(operation, mesh_a, mesh_b, res=res, bounds=bounds)

    def mesh_volume(self, mesh):
        """The enclosed volume of a closed mesh (holographic_route, ARCH-7; divergence theorem over outward-wound
        triangles). Used to verify CSG booleans are geometrically -- not just topologically -- correct."""
        from holographic.scene_and_pipeline.holographic_route import mesh_volume
        return mesh_volume(mesh)

    def cvt_remesh(self, mesh, n_sites=500, iterations=6, shrink=True, silhouette=0.95):
        """CVT (Lloyd-relaxed) remeshing after CWF (Xu et al., SIGGRAPH 2024): k-means on the surface -- the
        engine's codebook move in a mesh costume -- with cluster_decimate's bundled-quadric representatives as
        the QEM term. MEASURED vs the fixed grid at equal budget on a scanned mantis: min-angle median 22.8 ->
        43.1 deg, slivers 14% -> 1%, components 41 -> 9, non-manifold 211 -> 82. NOT provably manifold: gate
        the result with m.topology_gate. silhouette=0.95 holds a worst-view IoU floor by walking n_sites up
        (None disables). Returns (Mesh, report). See holographic_meshqem.cvt_remesh."""
        from holographic.mesh_and_geometry.holographic_meshqem import cvt_remesh as _cvt
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guarded as _sg
        msh = self._as_mesh(mesh)
        if silhouette is None:
            return _cvt(msh, n_sites=n_sites, iterations=iterations, shrink=shrink)
        # DEFAULT-ON silhouette guard, like every other mesh-reducing faculty (the institutional lesson the
        # guard-coverage pin enforces): the knob is n_sites -- more sites = finer -- and the guard walks it
        # up until the worst-view IoU holds. The primitive stays unguarded for callers who gate differently.
        q, g = _sg(msh, lambda k: _cvt(msh, n_sites=int(k), iterations=iterations, shrink=shrink)[0],
                   knob=int(n_sites), min_iou=float(silhouette))
        return q, {"sites_requested": int(n_sites), "guard": g}

    def gabor_cloud_render(self, field, O, D, L, sun_dir, ceiling, view_steps=32, sigma_t=6.0,
                           sigma_s=0.95, g=0.5):
        """GAB-CLOUD: render a fitted GaborField as a single-scattered VOLUMETRIC CLOUD through the engine's
        cloud renderer -- the field satisfies the density protocol (.density + finite-segment .optical_depth,
        verified 1e-6 vs quadrature), so cloud_single_scatter's CLOSED-FORM shadow rays work on it unchanged
        (measured 49x fewer evals at 8e-5 error, same as on FPE volumes). Returns (radiance, density_evals).
        field.lod(cutoff) before this call gives a cheaper coarse cloud with no refit. See holographic_cloud."""
        from holographic.rendering.holographic_cloud import single_scatter
        return single_scatter(field, O, D, L, sun_dir, ceiling, view_steps=view_steps,
                              sigma_t=sigma_t, sigma_s=sigma_s, g=g)

    def gabor_volume(self, rho, K=48, n_freqs=3, bounds=(0.0, 1.0), anisotropic=False):
        """GABOR FIELD fit of a density grid (Condor et al., SIGGRAPH 2026): a mixture of Gaussian-envelope x
        cosine-wave primitives where Gaussians (w=0) and oriented Gabors COMPETE for every slot. MEASURED on
        this engine: +14.4 dB over an equal-count Gaussian-only fit on wispy (oriented) content at K=24; ray
        integrals are CLOSED FORM (verified 2e-16 vs quadrature) so transmittance costs one call per ray, no
        marching; LOD is FREE -- field.lod(cutoff) prunes kernels above a frequency, no refit, no mipmaps
        (Gaussian base always survives). Returns (GaborField, report) -- field.eval(P), field.ray_integral(o,d),
        field.transmittance(o,d,extinction), field.lod(cut), field.render_ortho(). KEPT NEGATIVES: isotropic
        cost is the price (greedy pursuit, paid once per asset). GAB-ANISO: anisotropic=True fits ORIENTED
        ellipsoid envelopes (per-kernel SPD Q from the local residual covariance) -- MEASURED +2 to +6 dB over
        isotropic on thin filament content at equal count, breaking the isotropic PSNR plateau (round envelopes
        cannot elongate); slightly WORSE on blobby content (kept negative), so it is opt-in. See
        holographic_gaborfield."""
        from holographic.rendering.holographic_gaborfield import fit_gabor_field
        import numpy as np
        return fit_gabor_field(np.asarray(rho, float), K=K, n_freqs=n_freqs, bounds=bounds, anisotropic=anisotropic)

    def manifold_cleanup(self, mesh, keep_largest=True):
        """R3: make a retopo result STRICTLY MANIFOLD (so QEM decimate / half-edge builds accept it) and
        REPORT the topological cost. MEASURED on a scan retopo: 142 non-manifold 'fin' edges -> 0, 1 component
        preserved, 24 small holes introduced, 93%% faces kept, QEM then accepts. Four local surgeries were
        tried and all traded the defect for holes or fragments (kept negatives on record); a lossless fix
        needs a manifold-guaranteeing extraction (R3-proper). Returns (mesh, report) with the gate verdict and
        faces_kept_frac. See holographic_meshtools.manifold_cleanup."""
        from holographic.mesh_and_geometry.holographic_meshtools import manifold_cleanup as _mc
        return _mc(self._as_mesh(mesh), keep_largest=keep_largest)

    def topology_report(self, mesh):
        """PER-COMPONENT topology invariants (R1): for each connected component V/E/F, euler chi, boundary-loop
        count, and genus, plus per-loop fingerprints (centroid/length) and non-manifold edge count. The numbers
        that distinguish an INTENDED hole (a boundary loop present in the input) from mesh DESTRUCTION (a new
        loop, a new component, a genus change). See holographic_meshtools.topology_report."""
        from holographic.mesh_and_geometry.holographic_meshtools import topology_report as _tr
        return _tr(self._as_mesh(mesh))

    def topology_gate(self, before, after, loop_match_tol=0.25):
        """ACCEPT/REJECT a remesh by topology invariants (R1): passes iff component count is preserved, no
        non-manifold edges appear, per-component genus holds, and every output boundary loop matches an input
        loop (intended holes kept; punched holes rejected). Returns (passed, report) with violations NAMED so
        a pipeline can retry finer instead of amputating with keep_largest -- the measured motive: silent 11%
        amputation of a scanned mantis. See holographic_meshtools.topology_gate."""
        from holographic.mesh_and_geometry.holographic_meshtools import topology_gate as _tg
        return _tg(self._as_mesh(before) if not isinstance(before, dict) else before,
                   self._as_mesh(after) if not isinstance(after, dict) else after, loop_match_tol=loop_match_tol)

    def spatial_recall(self, points, queries, payloads=None, k=1, dim=256, bandwidth=8.0, seed=0):
        """H5 -- every closest-point is a RECALL: encode `points` (n, d) as position hypervectors (fractional
        power encoding: nearby points -> similar vectors, spearman 0.967) and recall the nearest stored points
        at `queries` by argmax cosine -- one matmul, no spatial hash. Measured 4.1x vs brute at scan scale
        (18k pts / 20k queries, dim 256), recalled points within 1%% of true-nearest distance at p95. With
        `payloads` (one row per point) also returns the resonant top-k weighted readout (soft nearest-neighbour
        gather -- 0.034 RGB reading scan colour, better than a volumetric bake). Returns (indices, payload_out,
        report). For MANY query batches against one source, build holographic_spatialmem.SpatialMemory once and
        reuse it -- the amortised source encoding is where the speed lives. KEPT NEGATIVE: no single-bundle
        scene mode -- FPE keys are correlated by design, and correlated keys cross-talk in a superposition
        (33%% recall at K=128); proximity-preservation and bundle-capacity are in direct tension. See
        holographic_spatialmem.spatial_recall."""
        from holographic.sampling_and_signal.holographic_spatialmem import spatial_recall as _sr
        return _sr(points, queries, payloads=payloads, k=k, dim=dim, bandwidth=bandwidth, seed=seed)

    def graph_connected_components(self, n_nodes, edges):
        """Partition n_nodes (0..n_nodes-1) into connected components under an undirected edges list of (u,v)
        pairs -- the GENERIC graph flood fill under every 'island' in the engine (physics constraint graphs,
        mesh edge adjacency, conflict graphs, DDM subdomains). Returns a list of sorted index lists, ordered by
        smallest member (deterministic, edge-order independent). Isolated nodes are singletons. This is the
        reusable primitive mesh_connected_components and route both build on. See
        holographic_island.connected_components."""
        from holographic.simulation_and_physics.holographic_island import connected_components as _cc
        return _cc(n_nodes, edges)

    def process_scan(self, mesh, uv=None, texture=None, retopo=True, lod=None, density=1.0,
                     bake_size=1024, bake_margin=2, keep_shards=False, silhouette=0.95, bake_method="project",
                     retopo_fast=False, retopo_snap=False, retopo_sized=False, bake_normal_aware=False, manifold=False):
        """ONE WORKFLOW: repair a scan and reduce its polys, keeping the texture -- in the correct order:
        repair the ORIGINAL -> retopo the repaired mesh -> LOD (a coarser retopo when retopo=True, measured:
        decimating a quad retopo re-shatters it; QEM decimation when retopo=False) -> shard cleanup -> fresh
        per-face atlas + reproject the original texture (rebake, never a transfer of the scan's fragmented
        uvs). Four workflows via the flags: retopo+lod, retopo only, lod only, repair only. lod is a fraction
        (<1) or a face count. bake_method="scatter" uses the holographic scatter/gather fast path for the
        rebake (~1500x on dense scans). retopo_fast=True uses position_field's vectorised-inner fast path
        (~3.5x on the retopo, bit-identical extraction). Returns (mesh, uv, image, report); uv/image None in
        geometry-only mode. See holographic_meshtools.process_scan."""
        from holographic.mesh_and_geometry.holographic_meshtools import process_scan as _ps
        return _ps(self._as_mesh(mesh), uv=uv, texture=texture, retopo=retopo, lod=lod, density=density,
                   bake_size=bake_size, bake_margin=bake_margin, keep_shards=keep_shards,
                   silhouette=silhouette, bake_method=bake_method, retopo_fast=retopo_fast, retopo_snap=retopo_snap, retopo_sized=retopo_sized,
                   bake_normal_aware=bake_normal_aware, manifold=manifold)

    def mesh_drop_small_components(self, mesh, min_faces=None, min_fraction=0.0, keep_largest=False):
        """Remove disconnected surface COMPONENTS that are too small -- the cleanup a field-guided retopo needs
        (extracting quads from a scan drops isolated cells: a mantis retopo shattered into 88 components, one
        body + ~75 shards). keep_largest=True keeps only the biggest; min_faces / min_fraction keep components
        above a threshold. Re-indexes verts, carries uvs/normals. Returns (cleaned_mesh, report). Built on the
        shared graph flood. See holographic_meshtools.drop_small_components."""
        from holographic.mesh_and_geometry.holographic_meshtools import drop_small_components as _dsc
        return _dsc(self._as_mesh(mesh), min_faces=min_faces, min_fraction=min_fraction, keep_largest=keep_largest)

    def mesh_connected_components(self, mesh):
        """The number of connected components of a mesh (holographic_route, ARCH-7; flood fill over edge adjacency).
        A CSG union of overlapping solids has 1; of separate solids, 2."""
        from holographic.scene_and_pipeline.holographic_route import connected_components
        return connected_components(mesh)

    # ---- the SEARCH & DYNAMICS faculties (integration plan, Tier 3) -----------------------------
    # Min-cost search on a graph or a trellis (a maze; a fragment assembly) and learned linear
    # dynamics -- the last modules built beside the mind, now faculties of it. Where the structure is
    # natural the search returns a B7 typed structure (assemble); dynamics is, literally, an algebra
    # of binds.

    def design_network(self, nbr, terminals, steps=200, mu=2.0, dt=0.2, keep=0.25):
        """Multi-terminal network design by the Tero/Physarum flow model -- the 'Tokyo rail' experiment, the
        multi-terminal generalisation of `solve_maze`. Given a graph `nbr` (adjacency {node: [neighbours]})
        and a set of TERMINALS (food sources), grow tubes that thicken with flux until an efficient network
        connecting every terminal emerges (holographic_flow.tero_network). `mu` tunes the trade-off Physarum
        is famous for: HIGH mu -> a near-minimal Steiner TREE, LOW mu -> a REDUNDANT, fault-tolerant mesh.

        The network is returned BOTH as raw edges and as a B7 TYPED STRUCTURE: a graph-memory recipe
        M = superpose over edges of bind(node_u, node_v) -- the same construction `chain_structure` uses, so
        it realize()s to one hypervector and the engine's unbind+cleanup recalls a node's neighbours
        (unbind M by a node atom, snap the result to the node codebook). Returns a dict with 'edges', the
        typed 'structure' (recipe), the 'memory' vector, and 'nodes' (name -> unitary atom for cleanup)."""
        from holographic.misc.holographic_flow import tero_network
        from holographic.agents_and_reasoning.holographic_ai import derived_atom
        edges, _D = tero_network(nbr, terminals, steps=steps, mu=mu, dt=dt, keep=keep)
        rec = self.typed_structure()
        names = sorted({x for e in edges for x in e}, key=str)
        handles = {nd: rec.atom(f"node:{nd}", unitary=True) for nd in names}        # symbolic build-graph nodes
        nodes = {nd: derived_atom(self.seed, f"node:{nd}", self.dim, unitary=True)  # matching codebook (cleanup)
                 for nd in names}
        memory = None
        if edges:                                                                  # M = superpose bind(u, v)
            rec.mark_output(rec.superpose([rec.bind(handles[u], handles[v]) for (u, v) in edges]))
            memory = self.realize(rec)
        return {"edges": edges, "structure": rec, "memory": memory, "nodes": nodes}

    def solve_maze(self, world, steps=200, mu=1.5, dt=0.2):
        """Solve a GridWorld maze by the DETERMINISTIC Tero flow-conductance model (holographic_flow):
        Physarum-style tubes thicken with flux (Poiseuille conductance) until the network collapses onto
        the shortest path. Same (path, info) interface as the stochastic slime solver, but deterministic
        and ~100x faster, and it lands EXACTLY on the optimum on braided mazes. info reports
        reached / optimal / extracted_len / cells / deterministic. Returns (path, info)."""
        from holographic.misc.holographic_flow import solve_maze_flow
        return solve_maze_flow(world, steps=steps, mu=mu, dt=dt)

    def flow_circulation(self, nbr, start, goal, steps=200, mu=1.5, dt=0.2):
        """Decompose a solved Tero/Physarum flow into TRANSPORT and CIRCULATION -- the analysis layer the flow
        solver lacked, wiring it to the Helmholtz-Hodge split (EXP-8) and the graph's topology (the B1 of
        EXP-5/7). Runs the flow on the adjacency graph `nbr` (the same one `solve_maze`/`tero_solve` take),
        reads the CONVERGED signed edge flux (`holographic_flow.tero_flux` -- the quantity the solver computes
        and discards), then Hodge-splits it: the GRADIENT part is the net source->goal transport (its
        divergence is the injected current), and the HARMONIC part is circulation around the graph's loops (its
        dimension is the graph's B1). A maze graph has no filled triangles, so there is no curl. Returns a dict
        {loops, redundancy, transport_energy, circulation_energy, flux, edges, n_vertices}: `loops` = B1 (the
        graph's independent cycles), and `redundancy` = the harmonic fraction of the flux energy -- how much of
        the converged flow CIRCULATES rather than transports. It is exactly 0 on a tree (the route is forced),
        and on a loopy graph it varies with mu (a previously-hidden property of the flow: at high mu competing
        thick tubes leave more circulating flux). Returns None if start and goal are disconnected."""
        from holographic.misc.holographic_flow import tero_flux
        from holographic.sampling_and_signal.holographic_spectral import hodge_decomposition, betti_numbers
        res = tero_flux(nbr, start, goal, steps=steps, mu=mu, dt=dt)
        if res is None:
            return None
        n, edges, flux = res
        gradient, _curl, harmonic = hodge_decomposition(n, edges, flux, None)   # graph -> no triangles -> no curl
        _, b1 = betti_numbers(n, edges, None)
        circ = float(np.dot(harmonic, harmonic))
        total = float(np.dot(flux, flux)) + 1e-30
        return {
            "loops": b1,
            "redundancy": circ / total,
            "transport_energy": float(np.dot(gradient, gradient)),
            "circulation_energy": circ,
            "flux": flux,
            "edges": edges,
            "n_vertices": n,
        }

    def assemble(self, target, library, frag_len=2, steps=300, mu=1.5, dt=0.2, energy=None):
        """Assemble `target` from a `library` of overlapping fragments by MIN-ENERGY flow search
        (holographic_assembly) -- Rosetta-style fragment assembly (choose a fragment per position to
        minimise a placement energy, consecutive fragments overlap-agreeing) cast as the SAME min-cost-
        path flow the maze solver runs, on a (position x fragment) trellis. Returns a dict: the assembled
        string, its total energy, the chosen (position, fragment) list, and a B7 StructureRecipe binding
        each fragment to its position -- the assembly AS a typed holographic structure (realize() it,
        save() it). Built at this mind's dim/seed.

        It finds the GLOBAL optimum (it matches the exact Viterbi DP), not a greedy one. `energy` is an
        optional callable energy(frag, pos, target) -> non-negative cost (default: Hamming mismatch). A
        SUPPLIED energy is the Rosetta move -- not every mismatch costs the same (a substitution matrix, a
        cleanup-based score) -- and the search stays globally optimal under it. The default stand-in is the
        combinatorial core, not a protein force field (the kept negative); a real energy is now pluggable."""
        from holographic.simulation_and_physics.holographic_assembly import assemble as _assemble
        return _assemble(target, library, frag_len=frag_len, steps=steps, mu=mu, dt=dt,
                         dim=self.dim, seed=self.seed, energy=energy)

    def compare_structures(self, a, b, dim=None, seed=None, tol=0.1):
        """Superpose two assembled structures (assemble() outputs) and read their OVERLAP -- the Baker seat's
        compare-two-folds. Returns {placement_overlap, holographic_overlap, shared}: the exact shared-motif
        fraction (overlap coefficient of the (pos, fragment) sets), the SAME overlap read holographically from
        the superposed role-bound vectors via consolidation (so you can compare structures you only hold as
        hypervectors -- a recalled fold -- and it degrades gracefully under noise), and the shared placements.
        On clean structures the two overlaps agree, the holographic read validated against the exact count.
        Built at this mind's dim/seed by default."""
        from holographic.simulation_and_physics.holographic_assembly import compare_structures as _cmp
        return _cmp(a, b, dim=(self.dim if dim is None else dim),
                    seed=(self.seed if seed is None else seed), tol=tol)

    def wasserstein(self, a, b, cost=None, eps=None):
        """The Wasserstein (earth-mover's) distance between two distributions by Sinkhorn iteration
        (holographic_transport, BLD-8) -- the least work to MOVE one onto the other, mass times the ground
        distance it travels. Unlike a bin-wise metric (Euclidean/cosine), it keeps GROWING as two distributions
        move apart even after they stop overlapping (a peak at bin 12 vs bin 20 reads farther from bin 10 than
        bin 12 does), which is the right answer when the bins carry geometry (position, time, frequency).
        `cost` is the ground-distance matrix (default |i-j|); `eps` is the entropic regularisation (default
        scales to the cost). KEPT NEGATIVES: the eps knob (too large blurs the distance high, too small
        underflows the kernel) and O(n*m) per iteration. Returns the distance."""
        from holographic.misc.holographic_transport import wasserstein as _w
        return _w(a, b, cost=cost, eps=eps)

    def learn_dynamics(self, states, ridge=1e-3):
        """Learn a fixed dynamics operator U so that state(t+1) ~ bind(U, state(t)) -- dynamics as an
        ALGEBRA OF BINDS (holographic_dynamics). In HRR's Fourier domain a learned bind is a per-frequency
        complex transfer, i.e. the Koopman/DMD operator in Fourier coordinates (the object Stam's FFT
        fluid step and Puckette's phase vocoder also manipulate). Returns a Propagator with .step(state)
        (one-step prediction = a SINGLE bind), .rollout(state, k), and .recall_at(state, k) -- recover the
        state k steps BEFORE one now, so the trajectory is content-addressable, not just forward-runnable.

        `states` is a sequence of state rows (T, dim). KEPT NEGATIVE on real market RETURNS: prediction
        only TIES a trivial mean predictor -- near-efficient-market returns have almost no linear structure
        for a fixed operator to exploit (the correct, expected result, kept on record). It SHINES where the
        dynamics ARE linear, now measured on two such regimes: AUDIO frames of a sustained tone (the per-bin
        phase advance; one-step error 0.001 vs persistence 1.64 / mean 1.00) and a FLUID field on a torus
        (linear advection-diffusion, each mode a rotation + decay; error 0.011 vs persistence 0.34 / mean 1.12,
        and the learned operator rolls out as a surrogate solver tracking the true sim to ~3% over 8 steps).
        Its HONEST LIMIT, also measured: a NONLINEAR Burgers field forms shocks no single fixed linear operator
        captures, where it does worse than persistence. The CONTENT-ADDRESSABLE round-trip (the operator's own
        forward k then back k returns the start at cosine ~1.0) is the durable win regardless of regime."""
        from holographic.simulation_and_physics.holographic_dynamics import Propagator
        return Propagator.learn(np.asarray(states, float), ridge=ridge)

    def kinematics(self, dim=None, lo=-50.0, hi=50.0, seed=1):
        """CLOSED-FORM KINEMATICS on the substrate (holographic_physics, Kinematics) -- physics as an algebra of
        binds. Position advances by velocity as ONE binding (x += v is bind(state_x, state_v)), acceleration advances
        velocity the same way, and the velocity BETWEEN two observed positions is read by UNBIND (bind x_b with the
        involution of x_a, decode). The direct embodiment of the engine's core thesis, 'binding is a rigid shift,'
        pointed at motion -- the Stam/Macklin seats' territory. This is the CLOSED-FORM twin of learn_dynamics
        (Propagator), which LEARNS its operator from data; here the operator is the encoder's own shift, exact by
        construction rather than fitted. Returns a Kinematics over [lo, hi] (the mind's dim by default): state(x),
        step(S_x, S_v), trajectory(x0, v0, a, steps) -- integrate by pure binding and decode each position, which
        RAISES if the true path leaves the encoder's range (the honest boundary) -- and read_velocity(x_a, x_b).
        Delegates to holographic_physics."""
        from holographic.simulation_and_physics.holographic_physics import Kinematics
        return Kinematics(dim=dim or self.dim, lo=lo, hi=hi, seed=seed)

    # ---- the GENERATIVE faculties (integration plan, Tier 4) -----------------------------------
    # Generation is denoising run backwards, and a splat scene is a bundle -- so the last two modules
    # built beside the mind reconcile straight into it: generate a vector by the cleanup-attractor
    # diffusion, and represent a 2-D field as a superposition of Gaussian primitives.

    def low_discrepancy_sample(self, n, d=2, seed=None):
        """`n` low-discrepancy (quasi-random) points in [0, 1)^d -- even coverage of a domain. The right
        sampler wherever you PLACE points to COVER (generation seeds, codebook / anchor placement, sub-pixel
        jitter) rather than to draw an INDEPENDENT sample. Roberts' generalised golden-ratio sequence
        (holographic_lowdiscrepancy): deterministic, progressive (any prefix is well-distributed), and
        measurably tighter coverage than default_rng -- a quasi-Monte-Carlo integrator with far lower error
        than plain MC at equal count. Use default_rng where genuine independence is wanted (these points are
        correlated by construction). `seed` defaults to the mind's seed."""
        from holographic.sampling_and_signal.holographic_samplinghome import Sampling                 # via the Sampling home  consolidation R4
        return Sampling.low_discrepancy(n, d=d, seed=self.seed if seed is None else seed)

    def generate_vector(self, codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=None,
                        readout="softmax"):
        """GENERATE a hypervector by denoising FROM PURE NOISE (B10) -- the cleanup attractor as a tiny
        holographic diffusion (holographic_hopfield.generate): start from a random unit vector, anneal
        beta UP (vague -> sharp) and injected noise DOWN across `steps`, and walk onto the codebook
        manifold. Generation and denoising are the SAME operation in different regimes -- this is the
        vector-level twin of the text generate(), pointed at the B10 diffusion sampler. Returns a unit
        vector, deterministic in `seed` (this mind's seed by default).

        KEPT NEGATIVE: over a BARE codebook this converges to a stored atom (a degenerate sampler) --
        feed it a COMPOSED or continuous manifold for novel-but-valid samples. `readout='sparsemax'` is
        available but does NOT change the bare/continuous-codebook behaviour (measured: both readouts snap
        to a stored atom, novelty ~0) -- the sparse readout's win is on generate_structure (below), where it
        cures generative mode collapse."""
        from holographic.agents_and_reasoning.holographic_hopfield import generate as _generate
        return _generate(np.asarray(codebook, float), steps=steps, beta0=beta0, beta1=beta1,
                         noise0=noise0, seed=self.seed if seed is None else seed, readout=readout)

    def generate_structure(self, roles, fillers, steps=16, beta0=4.0, beta1=60.0, noise0=0.5, seed=None,
                           readout="softmax", early_stop=False, min_steps=None, stats=None):
        """GENERATE a novel-but-valid COMPOSED structure by denoising from noise over the composition manifold
        (B10 + the Eno reframe) -- the composed-subspace answer to `generate_vector`'s kept negative. Same
        annealed diffusion, but the denoiser is a slot-wise projection (unbind each role, snap the filler to
        the vocabulary, rebind, bundle), so the walk lands on the manifold of role-filler STRUCTURES rather
        than collapsing to a stored atom. `roles` is (S, dim) unitary role atoms, `fillers` is (V, dim) the
        filler vocabulary; the result is a unit vector whose every slot unbinds to a vocabulary atom -- a NEW
        combination (one of V^S), valid by construction (re-encoding the decoded fillers reproduces it).
        Different seeds give different structures; `generate_vector` over a bare codebook returns a stored
        atom (the degenerate case this fixes). Deterministic in `seed` (this mind's seed by default).

        `readout='sparsemax'` switches the slot-wise blend from softmax to the sparse readout. MEASURED: both
        readouts produce perfectly VALID structures (the generated vector reencodes its decoded combination at
        cosine 1.000), but softmax generation MODE-COLLAPSES -- many random seeds settle into the same few
        structures (diversity as low as 0.03-0.5) because the softmax blend's wide metastable basins funnel
        them together -- while sparsemax stays DIVERSE (0.6-1.0, nearly every seed a distinct valid structure).
        It is the same metastable-mixing fix as the cleanup/resonator readout, here curing generative mode
        collapse at no validity cost. Default stays softmax for backward-compatibility; sparsemax is the
        recommended setting when sampling for variety.

        ADAPTIVE STOP (B3, opt-in early_stop=True; pass stats={} to read stats['steps']): the decoded structure
        settles well before the fixed schedule ends, so stop once it has been STABLE for a few steps past a
        floor (Eno's condition: stability, not first-convergence -- novelty preserved). Measured ~50% fewer
        steps, the SAME structure as the full run on every seed, and a final crisp snap restores validity to
        1.000 -- essentially FREE (the hard decoded combination is an effective certificate, unlike the splat
        fit's soft plateau). Off by default (bit-identical to the fixed schedule)."""
        from holographic.agents_and_reasoning.holographic_hopfield import generate_structure as _gs
        return _gs(np.asarray(roles, float), np.asarray(fillers, float), steps=steps, beta0=beta0,
                   beta1=beta1, noise0=noise0, seed=self.seed if seed is None else seed, readout=readout,
                   early_stop=early_stop, min_steps=min_steps, stats=stats)

    def svg_canvas(self):
        """The holographic vector-graphics (SVG) faculty (holographic_svg.HolographicSVG) -- the sharp,
        resolution-INDEPENDENT cousin of splat_archive. Encode a scene of typed primitives (rect/circle/triangle,
        each with a continuous position, size, and palette colour) into ONE hypervector, decode it back, MORPH two
        scenes by interpolating their vectors (vector arithmetic that tracks a parameter lerp), GENERATE novel
        scenes via the composed-manifold diffusion (generate_structure), and render any scene as crisp SVG. A
        vector <rect>/<circle> has analytically exact edges at any zoom, so this sidesteps the Gaussian-basis blur
        the splat work had to fight with smaller splats or supersampling; SVG emission is pure string formatting,
        no new dependency. Cached on the mind, built at this mind's dim/seed -- round-trip fidelity scales with
        dimension (a few primitives are faithful at 2048+; a crowded scene wants more, the bundle's honest
        capacity limit). MEASURED: round-trip type/colour exact and position within ~0.03 on [0,1]."""
        if getattr(self, "_svg_canvas", None) is None:
            from holographic.io_and_interop.holographic_svg import HolographicSVG
            self._svg_canvas = HolographicSVG(dim=self.dim, seed=self.seed)
        return self._svg_canvas

    def _fractal_codebooks(self, G=8):
        """Deterministic codebooks for the fractal-kernel seed: a G*G grid of offset POSITIONS in the unit
        square (each with a unitary atom), a position role, a scale role, and a small set of candidate scales
        (each with an atom). Built from this mind's seed, so encode and decode agree by construction."""
        from holographic.agents_and_reasoning.holographic_ai import derived_atom
        cells = [(i / (G - 1), j / (G - 1)) for i in range(G) for j in range(G)]
        pos_atoms = np.stack([derived_atom(self.seed, f"fpos:{i}:{j}", self.dim, unitary=True)
                              for i in range(G) for j in range(G)])
        pos_role = derived_atom(self.seed, "f:pos_role", self.dim, unitary=True)
        scale_role = derived_atom(self.seed, "f:scale_role", self.dim, unitary=True)
        scales = [0.5, 1.0 / 3.0, 0.25]
        scale_atoms = np.stack([derived_atom(self.seed, f"fscale:{k}", self.dim, unitary=True)
                                for k in range(len(scales))])
        return cells, pos_atoms, pos_role, scale_role, scales, scale_atoms

    def fractal_seed(self, offsets, scale, G=8):
        """Encode a fractal KERNEL into ONE seed hypervector (Quilez: one kernel, a single seed). The kernel is
        N copies of the plane, each contracted by `scale` and translated to an (x, y) in `offsets` (snapped to
        the grid codebook). The seed is the holographic bundle
            seed = sum_k bind(pos_role, grid_atom[offset_k])  +  bind(scale_role, scale_atom[scale])
        -- the kernel carried in the geometry of one vector. Decode and expand it with `fractal_scene`.
        Returns a unit seed vector; different kernels give different seeds."""
        from holographic.agents_and_reasoning.holographic_ai import bind
        cells, pos_atoms, pos_role, scale_role, scales, scale_atoms = self._fractal_codebooks(G)
        parts = []
        for (ox, oy) in offsets:                                  # snap each offset to the nearest grid atom
            k = int(np.argmin([(ox - cx) ** 2 + (oy - cy) ** 2 for cx, cy in cells]))
            parts.append(bind(pos_role, pos_atoms[k]))
        s = min(scales, key=lambda z: abs(z - scale))             # nearest candidate scale
        parts.append(bind(scale_role, scale_atoms[scales.index(s)]))
        v = np.sum(parts, axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    def fractal_scene(self, seed, depth=8, G=8, max_points=60000):
        """Decode the kernel from a seed hypervector and EXPAND it to a self-similar scene by domain repetition
        of that one kernel to `depth` (the scene-of-scenes-of-scenes Quilez asked for -- each level places a
        contracted copy of the whole), then report its fractal dimension. Decoding is pure VSA: unbind the
        position role and threshold the grid atoms to recover WHICH cells are offsets, unbind the scale role
        and clean up to recover the scale. Returns {'offsets', 'scale', 'n_maps', 'points', 'dimension',
        'expected'} where expected = log(N)/log(1/scale) is the self-similar (Hausdorff) dimension the box
        count should land near. Deterministic in the seed."""
        from holographic.agents_and_reasoning.holographic_ai import unbind
        from holographic.misc.holographic_fractal import box_counting_dimension
        cells, pos_atoms, pos_role, scale_role, scales, scale_atoms = self._fractal_codebooks(G)
        pq = unbind(seed, pos_role); pn = pq / (np.linalg.norm(pq) + 1e-12)
        sims = pos_atoms @ pn                                     # which grid cells are present in the seed?
        offsets = [cells[k] for k in range(len(cells)) if sims[k] > max(0.5 * float(sims.max()), 0.12)]
        sq = unbind(seed, scale_role); sn = sq / (np.linalg.norm(sq) + 1e-12)
        s = scales[int(np.argmax(scale_atoms @ sn))]             # which candidate scale?
        N = len(offsets)
        d = depth
        while N > 1 and N ** d > max_points and d > 1:           # keep N^depth bounded
            d -= 1
        pts = np.zeros((1, 2))
        for _ in range(d):                                       # one kernel, repeated to depth d
            pts = np.vstack([pts * s + np.array(o) for o in offsets]) if offsets else pts
        dim = float(box_counting_dimension(pts)) if N > 1 else 0.0
        expected = float(np.log(N) / np.log(1.0 / s)) if (N > 1 and s < 1) else float("nan")
        return {"offsets": offsets, "scale": s, "n_maps": N, "points": pts,
                "dimension": dim, "expected": expected, "depth": d}

    def splat_field(self, target, k=20, denoise=False, refit=True, noise_thresh=None, k_min=4, k_max=200,
                    basis="gaussian"):
        """Represent a 2-D field/image as a SUPERPOSITION of K Gaussian primitives (holographic_splat) --
        the structural twin of bundle (a Gaussian-splat scene IS a bundle, and the RBF ScalarEncoder is
        already a Gaussian splat in hypervector space). Fits the splats by matching pursuit (greedy
        superposition); returns (splats, rendered) where `splats` is a compact (cy, cx, amplitude, sigma)
        code and `rendered` is their sum. With denoise=True returns just the rendered field, which is a
        DENOISER -- a few smooth Gaussians have no capacity for high-frequency noise.

        refit=True (default) re-solves the amplitudes JOINTLY after placement (`splat_refit`) -- greedy
        matching pursuit double-counts overlapping splats, and one least-squares solve removes that for
        ~2-4 dB (the gain grows with k). It is closed-form and gradient-FREE.

        noise_thresh (default None) switches the COUNT from fixed to ADAPTIVE (`adaptive_fit`, V-Ray's
        adaptive sampler): placement runs until the residual is below noise_thresh*range, bounded to [k_min,
        k_max], so a simple field uses few splats and a busy one uses more at MATCHED quality. Orthogonal to
        refit -- the count is WHERE the splats go, refit is HOW STRONG they are. None keeps the fixed-k path
        unchanged. (Meaningful only for fields the smooth Gaussian basis can represent: a hard edge runs to k_max.)

        basis="gabor" (H8, default off) lifts each primitive with a FREQUENCY, ORIENTATION and PHASE -- a Gabor
        atom, seven numbers instead of four -- and returns 7-tuples. It is a BANDPASS primitive, so it buys you
        exactly the band it is tuned to. Measured at equal PARAMETER budget against this same jointly-refit Gaussian
        fit: +7.0 dB on a narrowband oriented grating, +0.2 dB on a sharp broadband edge, +0.1 dB on noise-like
        texture. It costs 89x the fitting time (a 196-atom dictionary per placement against 4). Reach for it when
        the content is oscillatory, and only when the budget is large -- seven numbers per atom is a levy paid up
        front, so on the same grating the win grows from +0.6 dB at a 224-number budget to +7.5 dB at 1,344.
        KEPT NEGATIVE, against the backlog's prediction: it does NOT dissolve the `splatsharpen` negative, which
        was recorded on a sharp edge. An edge is not a band; it is every band at once.

        KEPT NEGATIVE / SCOPE: isotropic splats and a fixed scale set (the honest matching-pursuit
        baseline); the *amplitude* refit is the gradient-free half of 'looping', but the gradient
        optimisation of positions/scales and anisotropic covariances (full 3DGS) needs autodiff and stays
        out of scope. Storing a whole gallery AS splat codes is now splat_archive() (holographic_splat_archive)."""
        from holographic.rendering.holographic_splat import splat_fit, splat_render
        target = np.asarray(target, float)
        if basis == "gabor":                                 # H8: the frequency-lifted basis, default off
            if noise_thresh is not None:
                raise ValueError("splat_field: the adaptive count (noise_thresh) is only defined for the Gaussian "
                                 "basis -- its stopping rule is calibrated against the Gaussian residual.")
            from holographic.rendering.holographic_splat import gabor_fit, gabor_render
            atoms = gabor_fit(target, k, refit=refit)
            rendered = gabor_render(atoms, target.shape)
            return rendered if denoise else (atoms, rendered)
        if basis != "gaussian":
            raise ValueError("splat_field: basis must be 'gaussian' or 'gabor', got %r" % (basis,))
        if noise_thresh is not None:                          # ADAPT-1: let the COUNT adapt to content
            from holographic.rendering.holographic_splat import adaptive_fit
            splats, _ = adaptive_fit(target, noise_thresh=noise_thresh, k_min=k_min, k_max=k_max, refit=refit)
        else:
            splats = splat_fit(target, k, refit=refit)
        rendered = splat_render(splats, target.shape)
        return rendered if denoise else (splats, rendered)

    def spectral_detail(self, img, cutoff=0.5):
        """The fraction of an image's spectral energy above `cutoff` x Nyquist -- "is the sharpness actually STORED?"

        PSNR is dominated by low frequencies, because that is where the energy lives, so a fit can match a target's
        PSNR while holding almost none of its detail. This is the number the `splatsharpen` kept negative is really
        about, and the one to report next to PSNR whenever a basis claims to capture detail.
        See holographic_splat.spectral_energy_fraction."""
        from holographic.rendering.holographic_splat import spectral_energy_fraction
        return spectral_energy_fraction(img, cutoff=cutoff)

    def export_splats(self, splats, path=None, fmt="ply", colors=None):
        """EXPORT Gaussian splats to a browser-renderer format (holographic_splatexport, FS-3) -- so a field/scene
        can be DISPLAYED as splats (the GPU's job; the engine stays the authoring brain). `splats` is a list of
        (center, amplitude, L), L the Cholesky of the inverse covariance (aniso_fit's native format; field_to_splats
        produces it from a metaball field). fmt='ply' writes the STANDARD 3D-Gaussian-Splatting .ply to `path` (opens
        in any 3DGS viewer) and returns the count; fmt='json' returns a compact JSON string for a three.js
        Gaussian-billboard shader. The core conversion is L -> scale + rotation quaternion by eigen-decomposing the
        precision (principal_axes). KEPT HONEST: base colour only -- holostuff splats carry no view-dependent
        spherical-harmonic colour (a further add, not faked); a degenerate/flat covariance is RAISED, not garbage.
        Delegates to holographic_splatexport."""
        from holographic.rendering.holographic_splatexport import splats_to_ply, splats_to_json
        if fmt == "json":
            return splats_to_json(splats, colors=colors)
        if path is None:
            raise ValueError("fmt='ply' needs a path to write to")
        return splats_to_ply(splats, path, colors=colors)

    def field_to_splats(self, centers, radius=0.5, amp=1.0):
        """Pull a metaball FIELD's Gaussians directly as exportable splats (holographic_splatexport, FS-3) -- no fit:
        the centres ARE the splat positions and the metaball `radius` IS the isotropic standard deviation. Returns a
        list of (center, amp, L) with L = (1/radius) I, ready for export_splats. For an already-fitted anisotropic
        field, pass aniso_fit's (center, amp, L) to export_splats directly."""
        from holographic.rendering.holographic_splatexport import field_to_splats
        return field_to_splats(centers, radius=radius, amp=amp)

    def export_splats_2d(self, splats2d, path=None, fmt="ply", colors=None, z=0.0, pixel_scale=1.0):
        """Export 2-D image splats (splat_fit's (cy, cx, amp, sigma)) to the standard 3DGS .ply by lifting them
        to the z=`z` plane (splats_2d_to_records: center=(cx,cy,z), isotropic L=1/sigma), so 2-D and 3-D splats
        export through one path. fmt='ply' writes to `path`; fmt='json' returns the three.js string. See
        holographic_splatexport.splats_2d_to_records."""
        from holographic.rendering.holographic_splatexport import splats_2d_to_records, splats_to_ply, splats_to_json
        recs = splats_2d_to_records(splats2d, z=z, pixel_scale=pixel_scale)
        if fmt == "json":
            return splats_to_json(recs, colors=colors)
        if path is None:
            raise ValueError("fmt='ply' needs a path to write to")
        return splats_to_ply(recs, path, colors=colors)

    def distributed_forward(self, layers, x, K=1, cleanup_books=None, relu=True):
        """A federated (and optionally deep, cleanup-gated) forward pass in the holographic space -- Path D's
        compute win: the storage array's federation applied to the MATMUL. A linear layer's weight rows stored
        in ONE bundled vector cap at ~0.02 x D classes (crosstalk on the continuous logit, with no cleanup to
        absorb it); FEDERATING the rows across K weight-memory shards (row c in shard c mod K) moves the wall to
        ~K x 0.02 x D -- measured: 16 classes faithful on one vector -> 96 on eight shards (~6x), tracking the
        exact classifier far past where a single vector collapses. For DEPTH, pass `cleanup_books` (a codebook of
        valid hidden activations per hidden layer) and each layer's output is snapped onto that manifold (a soft
        dense-Hopfield) so crosstalk resets between layers instead of compounding; or compute each layer exactly
        with `exact_matmul`, which has no crosstalk to compound at all.

        `layers` -- one (out, in) weight matrix or a list of them (deep). `x` -- (in,) or (N, in). Returns the
        final-layer logits, (N, out_last). KEPT NEGATIVE: federation buys FIDELITY / capacity, not fewer FLOPs
        (total unbinds are still C, grouped into K vectors; the K shards parallelise on neuromorphic hardware),
        and WITHOUT a depth cure a deep federated pass decays with depth -- the decay that cleanup (or exact
        arithmetic) removes. Delegates to holographic_compute."""
        import holographic.misc.holographic_compute as hc
        return hc.distributed_forward(layers, x, K=K, seed=self.seed, cleanup_books=cleanup_books, relu=relu)

    def pivot_index(self, items, fanout=7, seed=None):
        """A recursive pivot-tree index for SUBLINEAR nearest-item recall (Path D, the forest/data-structure
        seat). A naive index that summarizes items upward into a bundle hits the capacity wall; a B-tree holds
        PIVOTS explicitly instead, so the wall never bites. Here each node is a small cleanup memory of
        (pivot -> child) and routing is a nearest-pivot decision applied RECURSIVELY -- the same `cleanup`
        primitive the mind already uses, one level per hop, inception as the addressing fabric. Returns a
        holographic_pivot.PivotIndex: `.query(q, beam)` -> (nearest item index, pivot comparisons used),
        `.reached(q, beam)` -> the candidate leaf set.

        Greedy top-1 routing (beam=1) matches an exhaustive scan while touching only ~O(log N) pivots; a wider
        beam buys near-perfect recall of the true leaf into the candidate set, after which an exact key-unbind
        finishes. KEPT NEGATIVE: each hop is an approximate nearest-pivot decision, so a wrong turn at beam=1 can
        lose a query on overlapping data -- the beam is the honest knob that recovers recall; the build cost is
        the recursive k-means (NumPy only, no sklearn -- the minimal-frameworks rule)."""
        from holographic.misc.holographic_pivot import PivotIndex
        return PivotIndex(np.asarray(items, float), fanout=fanout, seed=(self.seed if seed is None else seed))

    def exact_matmul(self, W, x, scale=None, moduli=None):
        """Exact integer / fixed-point matmul carried over the FHRR phasor algebra -- Path D's arithmetic lever.
        General matmul in a lossy superposition dies as the matrix grows (crosstalk: the bundled rows interfere
        on readout). This instead carries each number as residues over coprime moduli (a Residue Number System)
        and does every multiply-accumulate as EXACT phasor-binding modular arithmetic -- a product of unit
        phasors adds their phases, so the modular sum is exact for ANY number of terms, with no crosstalk --
        then recomposes the integer with the Chinese Remainder Theorem. The dynamic range FEDERATES over moduli
        channels (more channels -> bigger exact range), the arithmetic sibling of `storage_array`'s federation.

        Integer W, x -> exact y = W @ x. Float W, x (with `scale`, or auto when the dtype is float) -> a
        fixed-point exact-arithmetic matmul (delegates to holographic_rns). KEPT NEGATIVE / scope: exact for
        integer / fixed-point operands within range -- a float is QUANTIZED first and the only error is that
        rounding (set by `scale`), a bit-depth question, not the crosstalk wall (it does not grow with size); and
        the FLOPs are real -- the parallelism is per-modulus / per-output, native on phasor / RNS hardware."""
        import holographic.misc.holographic_rns as rns
        W = np.asarray(W)
        x = np.asarray(x)
        if scale is not None or W.dtype.kind == "f" or x.dtype.kind == "f":
            return rns.rns_matmul_float(W, x, scale=(scale if scale is not None else 64), moduli=moduli)
        return rns.rns_matmul(W, x, moduli=moduli)

    def storage_array(self, n_parity=1, n_vals=256, add_threshold=0.90):
        """A federated, RAID-style symbol store -- the capacity/resilience faculty from the Path D
        'as above, so below' arc. One D-vector holds only ~0.1 x D symbols faithfully, and that budget is
        CONSERVED: to store more you FEDERATE across aligned shards coordinated by a thin layer
        (align/place/grow/protect), which is the within-vector graceful-degradation move applied one rung up,
        between shards. Returns a holographic_array.HoloArray on the mind's own dim and seed -- `.add(value_index)`
        stores a symbol (auto-growing a shard under capacity pressure), `.recall(g, down=...)` recalls it
        (reconstructing any lost shards from parity by subtraction -- the real-valued sibling of a fountain
        droplet), and `.accuracy(...)` measures recall across the federation.

        KEPT NEGATIVE / information floor: `n_parity` parity shards survive at most `n_parity` simultaneous
        shard losses -- it cannot recover more losses than it has parity, mirroring the fountain's 'too few
        droplets -> nothing'. The coordinator delegates to the same bind/unbind/derived_atom kernel the mind's
        own memory uses; it adds coordination, never a new algebra. The array also supports three recall modes:
        `.recall(g)` (directory-routed, O(1)), `.broadcast_recall(g)` (routerless, O(shards)), and
        `.routed_recall(g, c)` -- content-addressable SKETCH ROUTING that matches a key against per-shard
        key-sketches and unbinds only the top-c candidate shards, staying accurate where broadcast erodes."""
        from holographic.misc.holographic_array import HoloArray
        return HoloArray(self.dim, seed=self.seed, n_parity=n_parity, add_threshold=add_threshold, n_vals=n_vals)

    def superpose_compute(self, items, query=None, codebook=None, keys=None, shards=1):
        """The WIDTH faculty: evaluate K computations at once inside ONE vector (Kanerva / Kleyko 'computing in
        superposition') -- the parallel-readout complement to the mind's DEPTH side (recursive structure:
        `encode_tree`, `peel` traversal, the measured inception depth law). Bundles the keyed items into a single
        vector (holographic_superposed.pack), recovers them all with one batched unbind, and -- given a `query` --
        scores them in parallel and returns the winner, cleanup-gated against `codebook` when supplied (the
        discrete decision that resets crosstalk). Returns a dict: `packed`, `keys`, `recovered` ((n, D) noisy
        readout), `decoded` (per-item cleanup indices, when a `codebook` is given), and -- with a `query` --
        `scores` + `winner` (+ `winner_score`).

        `shards > 1` FEDERATES the items across that many vectors (item i -> shard i mod shards), recovering each
        shard separately, so the width wall moves ~shards-fold -- the storage array's federation applied to the
        readout. That lets one call serve the Bucket-A tasks the single vector capped: hypothesis SELECTION among
        more candidates than one vector holds (pass a `query`) and SEQUENCE recall of a longer symbol string (pass
        position-atom `keys` + a symbol `codebook`, read `decoded`).

        Keys default to unitary atoms derived from the mind's seed, so a single keyed item recovers EXACTLY and
        the only readback error is superposition crosstalk. KEPT NEGATIVE / the conservation law: one D-vector
        holds only ~0.1-0.2 x D items under cleanup-gated recall and ~0.02 x D when recovered values feed
        continuous math with no cleanup -- width is bounded per vector; you buy more by spending DEPTH (recurse,
        cleanup-gate each level) or by FEDERATING across shards, not by widening one flat bundle."""
        import holographic.misc.holographic_superposed as hs
        from holographic.agents_and_reasoning.holographic_ai import unitary_vector
        items = np.asarray(items, float)
        if items.ndim == 1:
            items = items[None, :]
        n = items.shape[0]
        if keys is None:
            rng = np.random.default_rng(self.seed)
            keys = np.stack([unitary_vector(self.dim, rng) for _ in range(n)])
        else:
            keys = np.asarray(keys, float)
        if shards <= 1:
            packed = hs.pack(keys, items)
            recovered = hs.recover_all(packed, keys)
        else:
            recovered = np.zeros_like(items)
            packed = []
            for k in range(shards):                              # federate items across shards (i mod shards)
                idx = np.arange(n)[np.arange(n) % shards == k]
                if len(idx) == 0:
                    continue
                Sk = hs.pack(keys[idx], items[idx])
                packed.append(Sk)
                recovered[idx] = hs.recover_all(Sk, keys[idx])   # recover only this shard's items
            packed = np.stack(packed) if packed else np.zeros((0, self.dim))
        out = {"packed": packed, "keys": keys, "recovered": recovered}
        if codebook is not None:
            cb = np.asarray(codebook, float)
            decoded = np.array([hs.resolve(r, cb)[0] for r in recovered])   # cleanup each item to the codebook
            out["decoded"] = decoded
        if query is not None:
            q = np.asarray(query, float)
            scores = (cb[decoded] if codebook is not None else recovered) @ q
            win = int(np.argmax(scores))
            out.update(winner=win, winner_score=float(scores[win]), scores=scores)
        return out

    def reservoir(self, n_in=1, rho=0.95, leak=0.3, in_scale=0.6, recurrence="shift"):
        """Gradient-free SEQUENCE learning -- the substrate-native Echo-State Network, the truly
        derivative-free corner of the learning program. The recurrent reservoir is FIXED: holostuff's
        `permute` (a cyclic shift) is norm-preserving (orthogonal), which is exactly the echo-state property,
        so the engine's own sequence operator IS a near-optimal reservoir; only a linear READOUT is trained,
        by one closed-form ridge regression -- no gradients, no backprop-through-time, fully deterministic.
        Returns a holographic_reservoir.HolographicESN on the mind's dim/seed: `.fit(U, Y)` solves the readout,
        `.predict(U)`, and `.generate(n, warm_U, feedback)` runs closed-loop autoregressive generation.

        Measured: NARMA10 to a literature-grade NRMSE ~0.37 (the reservoir features carry it -- it beats a
        linear-on-raw baseline), and gradient-free LEARNED text generation. KEPT NEGATIVE: chaotic free-running
        prediction MUST diverge pointwise after ~one Lyapunov time (the 'climate' is learnable, the 'weather'
        is not), and the readout learns a linear map of FIXED reservoir features, not new internal features.
        Delegates to holographic_reservoir."""
        from holographic.rendering.holographic_reservoir import HolographicESN
        return HolographicESN(n_in, dim=self.dim, rho=rho, leak=leak, in_scale=in_scale,
                              seed=self.seed, recurrence=recurrence)

    def mixture_of_experts(self, dim=None, seed=0, number_range=(-4.0, 4.0)):
        """MIXTURE OF EXPERTS with a LEARNED GATE (holographic_moe, GatedMixture) -- a bank of specialists plus a
        trained holographic gate (itself a creature brain) that routes each input to ONE expert, learned from reward.
        This is the genuinely distinct routing the mind's own dispatch is NOT: `decide`/`classify`/`recognize` route
        by RULE (which verb you called, what type the input is), whereas the MoE gate is TRAINED -- so it routes by
        the input's CONTENT, which a type check cannot do (two experts owning different halves of the number line; the
        gate sends each value to the right one). Build it: `add_expert(name, examples)` / `add_linear_expert(...)` for
        specialists, `train_gate(examples, epochs)` to learn the routing from outcomes, then `predict(x, modality)`.
        Returns a GatedMixture on the mind's dim/seed. Measured: the learned gate beats any single expert by a wide
        margin and approaches the oracle; it also beats CONFIDENCE routing when a specialist can be confidently wrong
        (the outcome-trained gate is not fooled). Serves the Olshausen/Togelius seats -- learned, interpretable
        routing. Delegates to holographic_moe."""
        from holographic.agents_and_reasoning.holographic_moe import GatedMixture
        return GatedMixture(dim=dim or self.dim, seed=seed, number_range=number_range)

    def prototype_classifier(self, levels=32, bandwidth=3.5):
        """Gradient-free CLASSIFICATION -- the HDC/VSA prototype learner, the other truly derivative-free
        method. Stage 1: encode each example (bind a feature-id atom with a ScalarEncoder level, bundle over
        features) and BUNDLE a class's examples into one prototype -- a one-pass centroid. Stage 2: perceptron
        retraining -- on a misclassified example, pull the correct prototype toward it and push the wrongly
        predicted one away (add/subtract on bundled vectors, no gradients). Returns a
        holographic_classifier.HolographicClassifier on the mind's dim/seed: `.fit(X, y, epochs)`, `.predict(X)`.

        Measured (test accuracy): digits 0.90 one-shot -> 0.95 retrained, breast_cancer 0.93 -> 0.95, and the
        encoding lifts a centroid model dramatically (wine raw-centroid 0.67 -> HDC 0.98). KEPT NEGATIVE (the
        field's own verdict): retraining beats the one-shot centroid, but the classifier lands just BELOW a
        tuned linear model (logistic regression) -- traded for a dead-simple gradient-free rule. Delegates to
        holographic_classifier."""
        from holographic.agents_and_reasoning.holographic_classifier import HolographicClassifier
        return HolographicClassifier(dim=self.dim, levels=levels, bandwidth=bandwidth, seed=self.seed)

    def equilibrium_net(self, n_in, n_hidden=64, n_out=2, beta=0.35, dt=0.4, t_free=40, t_nudge=12):
        """Equilibrium Propagation -- the LEARNING RULE for the energy-based (Hopfield) memory the engine
        uses as a fixed cleanup. Where the dense-Hopfield cleanup relaxes a query to a FIXED stored
        attractor, EP LEARNS the weights of a continuous Hopfield net so its energy minima encode a task. No
        backprop: a free relaxation (clamp the input -> equilibrium = the prediction) and symmetric nudged
        relaxations (+/- beta * loss on the output) whose difference is a contrastive Hebbian update that
        estimates the loss gradient (Scellier & Bengio 2017; Laborieux 2021, symmetric nudging). Returns a
        holographic_equilibrium.EquilibriumNet: `.fit(X, y_onehot, epochs)`, `.predict(X)`, plus
        `.ep_gradient_Who` / `.fd_gradient_Who` for the gradient-matching correctness check.

        This is the LOCAL-GRADIENT corner of the learning program -- NOT derivative-free like `reservoir`
        and `prototype_classifier`; EP estimates a gradient, just with relaxations rather than a backward
        pass. Its payoff over those two: it learns the HIDDEN weights, so it fits a NONLINEAR task a linear
        readout cannot -- on two interleaving moons it reaches ~0.92 vs a linear model's ~0.85, and its
        symmetric update matches the true gradient to cosine ~1.0. KEPT NEGATIVE / scope: needs SYMMETRIC
        weights; costs several relaxations per update (far more than a one-shot rule); the finite-beta
        estimate is still biased (it lands below exact backprop, which reaches ~1.0 here); and it is
        validated at small / moderate scale, not frontier scale. Delegates to holographic_equilibrium."""
        from holographic.simulation_and_physics.holographic_equilibrium import EquilibriumNet
        return EquilibriumNet(n_in, n_hidden=n_hidden, n_out=n_out, beta=beta, dt=dt,
                              t_free=t_free, t_nudge=t_nudge, seed=self.seed)

    def forward_forward(self, n_in, layer_sizes=(64, 64), n_classes=2, theta=0.05, label_scale=3.0):
        """The Forward-Forward algorithm -- backprop-free, settling-free DEPTH from purely LOCAL objectives
        (Hinton 2022). A stack of layers, each trained by its own goodness objective via TWO forward passes
        (positive data -> high goodness; data with a WRONG label embedded -> low goodness), each layer's
        weights moving by the gradient of THAT layer's local logistic alone, with L2-normalization between
        layers so a later layer can't read the length an earlier one already separated. Classification is
        label-embedded: prepend a one-hot label, and at test pick the label whose accumulated goodness is
        highest. Returns a holographic_forward.ForwardForwardNet: `.fit(X, y, epochs)`, `.predict(X)`.

        LOCAL-GRADIENT, not derivative-free (like `equilibrium_net`): each layer follows its own local
        gradient; there is just no backward pass linking the layers, and no relaxation. Its niche over EP is
        arbitrary DEPTH with a cheap closed-form local update per layer. KEPT NEGATIVE (measured, loud): at
        the small scale here this compact FF is a WORKING but WEAK classifier -- it TRAILS a plain linear /
        logistic model on every task tried (two-moons ~0.88 tie; overlapping 4-class blobs 0.95 vs 0.99;
        sklearn digits 0.88 vs logistic 0.97), beating linear only on a radial task where linear provably
        fails, and even then weakly. FF's published accuracy (Hinton's ~1.4% MNIST error) needs the full
        MNIST-scale recipe; what this demonstrates is the MECHANISM -- positive goodness provably separates
        from negative -- a conceptual route to backprop-free depth, not a competitive number. The stronger
        Mono-Forward (2025) refinement (per-layer local supervised heads) is the natural next step, not built
        here. Delegates to holographic_forward."""
        from holographic.misc.holographic_forward import ForwardForwardNet
        return ForwardForwardNet(n_in, layer_sizes=layer_sizes, n_classes=n_classes,
                                 theta=theta, label_scale=label_scale, seed=self.seed)

    def learn_chaos(self, states, dim=600, rho=0.9, leak=1.0, in_scale=0.5,
                    ridge=1e-6, washout=200, noise=1e-2):
        """Learn a NONLINEAR dynamics operator -- the companion to `learn_dynamics`. Where
        `learn_dynamics` fits ONE per-frequency complex transfer (the linear Koopman/DMD operator, exact
        for linearisable flow but pinned at the persistence floor on a state-dependent or chaotic system),
        this fits the reservoir's FIXED nonlinear expansion plus a TRAINED linear readout to the one-step
        evolution map -- the learned lift the dynamics negative called for. Returns a
        holographic_chaos.NonlinearPropagator on the mind's seed: `.predict_sequence(states)` for one-step-
        ahead forecasts, `.free_run(warmup, k)` for closed-loop rollout. It delegates to the reservoir
        faculty (it does not re-implement a learner).

        Measured (Lorenz '63, the canonical reservoir-computing test): the nonlinear one-step prediction
        lands ~0.0014 relative error -- about 40x better than the BEST linear map (full DMD) and ~50x
        better than persistence, where every linear operator sits at the chaos floor. Deterministic.
        KEPT NEGATIVES (loud): (1) closed-loop free-run tracks only ~ONE Lyapunov time -- far short of what
        the one-step error implies, because the autonomous reservoir has the well-known free-run STABILITY
        problem; noise=1e-2 is the sweet spot and more hurts, bigger reservoirs help only modestly, and the
        recurrence mixing (shift / perm / unitary-bind) is NOT the lever. (2) HIGH-dimensional PDE fields
        are out of reach for a single global reservoir (a 48-D Burgers field forecasts at ~0.27, worse than
        persistence; the literature needs local/parallel reservoirs, and EP is weak at high-D field
        regression too). (3) On mild dissipative flow persistence is a punishing baseline regardless of
        learner. The win is a genuine LOW-dimensional nonlinear-dynamics result, honestly bounded.
        Delegates to holographic_chaos."""
        from holographic.misc.holographic_chaos import NonlinearPropagator
        return NonlinearPropagator.learn(states, dim=dim, rho=rho, leak=leak, in_scale=in_scale,
                                         ridge=ridge, washout=washout, noise=noise, seed=self.seed)

    def learn_cleanup(self, patterns, noise=0.30, n_hidden=None, epochs=80, beta=0.5):
        """Learn a cleanup's ATTRACTORS instead of storing them -- a LEARNED energy memory. Every cleanup
        in the engine is fixed: the classical one snaps to a stored atom, and the modern-Hopfield energy
        cleanup (`cleanup(..., energy=True)` / dense_cleanup) relaxes against a FIXED codebook. This trains
        an energy (via Equilibrium Propagation -- it delegates to `equilibrium_net`, not a new learner)
        whose attractors form a LEARNED manifold, so a noisy query is projected onto the manifold rather
        than snapped to the nearest stored sample. `patterns` are manifold samples in [0,1]; the returned
        holographic_energy.LearnedEnergyMemory exposes `.cleanup(x)` (single vector or batch). This is the
        natural learned prior for a Plug-and-Play / RED restoration loop.

        Measured: on a continuous manifold the learned energy beats the fixed SOFT energy cleanup at every
        codebook size, and on a manifold of dimension >= 2 it beats a matched-memory codebook of random
        samples (2-D: ~0.43 vs ~0.49) -- learning's compactness beats the curse of dimensionality that
        makes tiling a manifold with samples cost ~grid^d points. Deterministic.
        KEPT NEGATIVES (loud): (1) for DISCRETE atoms recovered from isotropic noise the HARD 1-NN cleanup
        returns the EXACT atom (~0.02) and is unbeatable -- a learned approximate energy cannot beat exact
        recovery (B1's tie, sharpened to a loss); use the existing cleanup for discrete recall. (2) In 1-D
        the curse does not bite, so a matched-memory codebook wins (~0.27 vs ~0.33) -- the advantage over
        storing data requires manifold dimension >= 2. (3) The win over a codebook is at MATCHED memory,
        not unbounded, and EP inherits its weakness at very high output dimension (moderate D, low
        intrinsic-dim manifolds). Delegates to holographic_energy."""
        from holographic.simulation_and_physics.holographic_energy import LearnedEnergyMemory
        return LearnedEnergyMemory.learn(patterns, noise=noise, n_hidden=n_hidden,
                                         epochs=epochs, beta=beta, seed=self.seed)

    def procedural_noise(self, n_dims=2, dim=1024, bounds=None, octaves=4, lacunarity=2.0,
                         gain=0.5, base_bandwidth=2.0, seed=None):
        """G1 -- holographic band-limited procedural noise as a FIELD; fBm as an octave BUNDLE.

        Returns a holographic_noise.FractalNoise: `.query(point)` evaluates the amplitude-weighted sum of
        per-octave band fields, each a single hypervector (an FPE bundle of random-weighted RBF kernels).
        Frequency is bandwidth (base_bandwidth * lacunarity^o), amplitude is gain^o. Seats: Stam (spectral
        noise), Berry (fBm/band-limit), Quilez (noise as a procedural-SDF primitive).
        KEPT NEGATIVE: band-limited/smooth by construction (no sharp/discontinuous noise), and FFT-bound --
        each kernel is one encode, so deep fully-filled fBm is expensive (kernels are capped per octave)."""
        from holographic.sampling_and_signal.holographic_noise import FractalNoise
        return FractalNoise(n_dims, dim=dim, bounds=bounds, octaves=octaves, lacunarity=lacunarity,
                            gain=gain, base_bandwidth=base_bandwidth,
                            seed=self.seed if seed is None else seed)

    def material(self, channels=None, dim=1024, bandwidth=3.0, bounds=None):
        """G2 -- a PBR material as a role-filler HRR record; textures as FPE functions over UV.

        `channels` maps a name (albedo, roughness, normal, height, ...) to (uv_points, values); each becomes
        a texture field over the UV square, and the material binds them under per-name role atoms into one
        record = sum_r bind(role_r, channel_r). Returns a holographic_material.Material: sample() is exact
        (stored field), the record composes/blends/transmits as one vector, and transform_uv re-UVs every
        channel with a single bind. Seats: Plate (the record), Pharr (PBR channels), Drettakis (per-primitive
        material). KEPT NEGATIVE: band-limited (smooth textures; sharp masks stay raster) and the bare-record
        channel recovery carries ~sqrt(n)/sqrt(dim) crosstalk (raise dim to buy capacity)."""
        from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
        from holographic.materials_and_texture.holographic_material import Material, texture_field
        if bounds is None:
            bounds = [(0.0, 1.0), (0.0, 1.0)]
        enc = VectorFunctionEncoder(2, dim=dim, bounds=bounds, bandwidth=bandwidth, seed=self.seed)
        mat = Material(enc)
        if channels:
            for name, (pts, vals) in channels.items():
                mat.add(name, texture_field(enc, pts, vals))
        return mat

    def displace(self, target, scalar_fn, amount, use_uv=False):
        """G3 -- displace a surface along its normal by amount*scalar_fn. Dispatches on the target type.

        On a HolographicField the offset is a field DELTA (apply_delta of -amount*scalar at the field points),
        O(edit) with EXACT remove_delta undo; returns (displaced_field, delta). On a Mesh each vertex moves
        along its normal; returns the displaced Mesh. Seats: Pharr (displacement), Quilez (SDF displace = add
        to distance). KEPT NEGATIVE: the SDF path is the near-surface shader approximation (exact only where
        |grad sdf|=1); mesh displacement can self-intersect for large amounts on concavities."""
        from holographic.misc.holographic_displace import displace_sdf, displace_mesh
        from holographic.sampling_and_signal.holographic_fpefield import HolographicField
        if isinstance(target, HolographicField):
            return displace_sdf(target, scalar_fn, amount)
        return displace_mesh(target, scalar_fn, amount, use_uv=use_uv)

    def bump(self, mesh, scalar_fn, amount, eps=1e-3):
        """G3 -- bump mapping: perturb a mesh's shading normals from a scalar field's slope, NO vertices move.
        Returns an (V,3) array of tilted unit normals. The cheap fake-detail path (silhouette unchanged)."""
        from holographic.misc.holographic_displace import bump_normals
        return bump_normals(mesh, scalar_fn, amount, eps=eps)

    def terrain(self, bounds=None, octaves=5, lacunarity=2.0, gain=0.5, base_bandwidth=2.0,
                dim=1024, seed=None):
        """G4 -- a holographic fBm heightfield, liftable to a displaced-grid mesh or a heightfield SDF.

        Returns a holographic_terrain.Terrain (`.height(xy)`, `.heightmap(res)`); use holographic_terrain's
        terrain_to_mesh / terrain_to_sdf to lift it. A composition of G1 (height) and the displacement idea,
        so LOD is just re-sampling. Seats: Stam, Berry. KEPT NEGATIVE: no erosion (pure fBm statistics), and
        the heightfield SDF (z - height) is sign-correct but not a true Euclidean distance where it is steep."""
        from holographic.mesh_and_geometry.holographic_terrain import Terrain
        return Terrain(bounds=bounds, octaves=octaves, lacunarity=lacunarity, gain=gain,
                       base_bandwidth=base_bandwidth, dim=dim, seed=self.seed if seed is None else seed)

    def lsystem(self, axiom, productions, stochastic=None):
        """G5 -- a context-free L-system grammar; productions are a holographic record, output is a scenegraph.

        Returns a holographic_grammar.LSystem (`.expand(n)` parallel-rewrites the string); interpret it with
        turtle_to_segments and assemble with segments_to_scene / grow_plant (each segment instanced through a
        transform -- a recursive bundle that scene_to_recipe turns back into a holographic recipe). The one
        genuinely new geometry capability. Seat: Plate (HRR productions). KEPT NEGATIVE: recursive composition,
        not a biological growth simulation (no tropism/competition); deterministic context-free (optionally
        seeded-stochastic)."""
        from holographic.agents_and_reasoning.holographic_grammar import LSystem
        return LSystem(axiom, productions, stochastic=stochastic, rng_seed=self.seed)

    def attribute_field(self, encoder, points, values, weights=None):
        """G6 -- a per-vertex/texel attribute as a RESOLUTION-INDEPENDENT field over the surface domain.

        Same construction as a texture (an FPE function), but the intent is a data channel: bake it to a
        coarse mesh, subdivide, re-bake, and shared points keep their values because the field never changed
        -- only the sample points densified. Returns the field vector; sample with holographic_attributes.
        sample_attribute. A light raster store (attach_attribute/get_attribute) coexists for hard masks.
        KEPT NEGATIVE: band-limited (smooth attributes interpolate; hard 0/1 masks come back smoothed)."""
        from holographic.misc.holographic_attributes import attribute_field
        return attribute_field(encoder, points, values, weights=weights)

    def sdf_object(self, seed=None, complexity=3):
        """S2 -- a procedurally generated 3D OBJECT as an SDF tree, from a seed (the demoscene seed->world).

        Returns a holographic_sdf.SDF: a few transformed primitives combined by CSG / smooth-union with an
        occasional round/twist. Render with sdf_render, emit a shader with sdf_shader, represent it with
        .to_tree() (a holographic recipe). Seat: Quilez (tiny seed, structured world). KEPT NEGATIVE: a
        generator, not an art director -- a random tree can subtract most of itself away or leave a
        disconnected surface; marching rounds sub-cell features."""
        from holographic.io_and_interop.holographic_procgen import procedural_object
        return procedural_object(self.seed if seed is None else seed, complexity=complexity)

    def sdf_render(self, sdf_node, bounds=((-2, -2, -2), (2, 2, 2)), res=40):
        """S1 -- march an SDF tree to a triangle Mesh through the engine's existing marching bridge."""
        from holographic.io_and_interop.holographic_procgen import object_to_mesh
        return object_to_mesh(sdf_node, bounds=bounds, res=res)

    def sdf_shader(self, sdf_node, name="map", camera="fixed"):
        """S1 -- emit a complete Shadertoy-ready GLSL fragment shader (map() + raymarch + normals + light)
        for an SDF tree -- the demoscene OUTPUT. The shader embeds its own DSL in a header comment, so it
        round-trips back to a tree. camera="fixed" (default) is the classic head-on view (byte-identical to the
        historic output); camera="uniforms" emits an ORBIT camera driven by host-bound uAngle/uHeight/uDist
        uniforms, so a WebGL2 host can spin/zoom by setting uniforms instead of string-splicing a new camera into
        the source. Seat: Quilez (raymarched SDFs). KEPT NEGATIVE: twist/displace are domain warps (not exact
        distances) -- the emitter flags them and the raymarcher must shorten steps."""
        return sdf_node.to_glsl(name=name, camera=camera)

    def sdf_parse(self, dsl_text):
        """S1 -- parse a compact SDF DSL string back into an SDF tree -- the INPUT side of shader I/O.
        (kind p0 p1 ... child0 child1 ...); the inverse of node.to_dsl()."""
        from holographic.mesh_and_geometry.holographic_sdf import parse_dsl
        return parse_dsl(dsl_text)

    def shape(self, kind="sphere", position=None, scale=None, rotate=None, **kw):
        """Build a 3-D primitive by NAME, optionally placed -- the first call when you are making a scene.

        `kind` is a word you would actually type: 'cube'/'box', 'ball'/'sphere', 'floor'/'ground'/'plane',
        'donut'/'ring'/'torus', 'cylinder', 'cone', 'capsule', 'ellipsoid', 'octahedron', plus the fractals
        ('menger', 'mandelbulb'). Size parameters pass through (r, bx/by/bz, h, R, ...); an unknown kind
        raises with the full list rather than a bare KeyError.

        PLACEMENT IS BUILT IN, in the order scale -> rotate -> translate, so it cannot be got wrong: rotating
        after translating swings the object around the world ORIGIN instead of spinning it in place, which
        reads as "my object jumped" and is invisible in a single frame. `rotate` is (ax, ay, az, radians).

        The result is an SDF you can hand straight to scene.add(geometry=...), render_sdf, or combine with
        .union / .subtract / .intersect / .smooth_union. WHY THIS EXISTS: every primitive here was reachable
        only by import -- asked for a sphere, this mind used to return a Lipschitz bound.
        See holographic_sdf.make_sdf_shape."""
        from holographic.mesh_and_geometry.holographic_sdf import make_sdf_shape
        return make_sdf_shape(kind=kind, position=position, scale=scale, rotate=rotate, **kw)

    def sdf_grammar(self):
        """The SDF DSL described well enough to WRITE one: every node kind, what its numbers mean, an example.

        sdf_parse has always accepted a compact s-expression for a whole shape tree, and the node names and
        their parameter counts lived in a module-level dict nothing surfaced -- a grammar you could only use
        if you already knew it. Returns {syntax, nodes: [{kind, params, children, does}], example}, sorted
        primitives -> modifiers -> combinators, which is the order you build in.
        See holographic_sdf.dsl_grammar."""
        from holographic.mesh_and_geometry.holographic_sdf import dsl_grammar
        return dsl_grammar()

    def menger_fractal(self, iterations=3, size=1.0):
        """S1 -- the canonical Menger-sponge FRACTAL model as an SDF (a box minus recursive crosses). Evals,
        marches to a mesh, AND emits a GLSL loop -- the demoscene fractal. Seat: Quilez."""
        from holographic.mesh_and_geometry.holographic_sdf import menger
        return menger(iterations, size)

    def fold_fractal(self, iterations=12, scale=2.0, min_radius=0.5, fold_limit=1.0):
        """The KALEIDOSCOPIC-IFS / MANDELBOX distance-estimator SDF -- the general FOLD ENGINE behind the fractal-
        forums 3D fractals and the Yohei-Nishitsuji tweet-shader look (holographic_sdf). Iterates box-fold (conditional
        reflection) + sphere-fold (inversion through nested spheres) + scale/translate, tracking the derivative for a
        usable distance estimate. `scale` is the Mandelbox constant, `min_radius` sets where the sphere fold bites,
        `fold_limit` the box-fold extent. All conformal transforms, so it raymarches and orbit-traps cleanly with the
        existing renderer. A four-float recipe that regenerates megabytes of deterministic self-similar structure.
        Returns an SDF. Kept negative: INEXACT (a distance ESTIMATE) -- the in-engine raymarcher steps conservatively,
        but the GLSL emitter refuses it (a shader consumer must hand-tune the step size)."""
        from holographic.mesh_and_geometry.holographic_sdf import fold_fractal
        return fold_fractal(iterations, scale, min_radius, fold_limit)

    def mandelbulb(self, power=8.0, iterations=8, bailout=2.0):
        """The MANDELBULB distance-estimator SDF (holographic_sdf) -- White & Nylander's polar-power fractal, the 3D
        Mandelbrot analogue. Iterates z -> z^power + c in spherical coords with the analytic DE 0.5*log(r)*r/dr.
        power=8 is the classic bulb. Unlike fold_fractal (a Mandelbox FOLD engine), this is the ESCAPE-TIME family in
        3D (the z^n+c that draws the Mandelbrot set, lifted to a triplex algebra). Raymarches + orbit-traps with the
        existing renderer. Returns an SDF. Kept negative: INEXACT (a distance ESTIMATE); the GLSL emitter refuses it."""
        from holographic.mesh_and_geometry.holographic_sdf import mandelbulb
        return mandelbulb(power, iterations, bailout)

    def escape_time(self, width=256, height=256, center=(-0.5, 0.0), span=3.0, max_iter=100,
                    power=2.0, julia_c=None):
        """The 2D ESCAPE-TIME fractal FIELD (holographic_sdf) -- Mandelbrot (julia_c=None) or Julia (julia_c=(re,im)),
        the classic z -> z^power + c iteration in the complex plane. Returns a (height,width) float array of SMOOTH
        (continuous) escape counts in [0, max_iter], ready to feed a palette. The 2D sibling of mandelbulb: same
        z^n+c recurrence read as a field. center/span frame the view. Vectorised, deterministic."""
        from holographic.mesh_and_geometry.holographic_sdf import escape_time
        return escape_time(width=width, height=height, center=center, span=span, max_iter=max_iter,
                           power=power, julia_c=julia_c)

    def fold_fit(self, target, iterations=10, coarse=6, refine_steps=40):
        """INFER a fold RECIPE from an observed structure (holographic_foldfit) -- the inverse of fold_fractal.
        Recover the (scale, min_radius, fold_limit) whose Mandelbox fractal best fits the `target` (M,3) point cloud:
        a deterministic coarse grid over recipe space, then a local refine with this mind's own `optimize` (composed).
        Returns {recipe, loss, baseline, improved}. The pattern-recognition payoff -- self-similarity detection as
        parameter estimation. Kept negative: the loss (mean distance-to-surface) is NECESSARY not SUFFICIENT (the DE
        lower bound can score an over-large fractal that merely contains the points) -- so the baseline-improvement
        RATIO, not the absolute loss, is the discriminative signal. Recovers A recipe consistent with the cloud."""
        from holographic.mesh_and_geometry.holographic_foldfit import fold_fit
        return fold_fit(target, iterations=iterations, coarse=coarse, refine_steps=refine_steps, mind=self)

    def fit_shape(self, target, **kw):
        """CLOSEST-FIT to a procedural formula + its SHADERTOY / GLSL (holographic_fitshape) -- the capstone. Given a
        `target`, fit the closest procedural representation and return it with runnable code. Dispatches on the target:
        an (M,3) POINT CLOUD -> a FRACTAL SDF recipe via fold_fit, reconstructed and emitted as a complete Shadertoy
        raymarch shader (the strong path: good for self-similar/fractal 3-D structure, quality = fold_fit's baseline-
        improvement ratio, >~3x is a real fit); a 2-D IMAGE / HEIGHT / TEXTURE -> a PROCEDURAL fBm matched to the
        target's statistical signature + a GLSL fbm snippet. Returns a dict with `kind`, the fit, a measured `quality`
        vs `baseline`, the emitted code (`shadertoy` or `glsl`), and a `note`. Honest by construction: KEPT NEGATIVE
        -- the texture path is a family match (matched roughness+detail), NOT parameter recovery or a pixel match; it
        does not fit arbitrary meshes or L-systems (a fern's true generator) -- those are scoped next fitters."""
        from holographic.mesh_and_geometry.holographic_fitshape import fit_shape
        return fit_shape(target, mind=self, **kw)

    def to_shadertoy(self, sdf, camera="fixed"):
        """Emit a complete, runnable SHADERTOY fragment shader for an SDF (holographic_sdf.sdf_shader) -- map() +
        raymarch + normals + lighting + a mainImage entry point, ready to paste into shadertoy.com. Works for the
        fractal SDFs too (fold_fractal, mandelbulb, menger): they emit their fold/polar-power loop with a header note
        that a distance ESTIMATE needs conservative ray steps. camera="fixed" (default) is the classic head-on view;
        camera="uniforms" emits an ORBIT camera controlled by host-bound uAngle/uHeight/uDist uniforms (for a WebGL2
        host that spins/zooms the scene without re-emitting or string-splicing the shader). The 'get the Shadertoy
        code for it' primitive. Alias of sdf_shader with the conventional name."""
        return self.sdf_shader(sdf, camera=camera)

    def ifs_generate(self, name_or_ifs="barnsley_fern", n=20000, seed=0):
        """Generate a plant/fractal point cloud from an AFFINE IFS via the chaos game (holographic_ifs). Pass a named
        system ('barnsley_fern', 'culcita_fern', 'sierpinski', 'fractal_tree', 'dragon_curve') or an AffineIFS object;
        get an (n,2) attractor point cloud. A fern/tree/sierpinski from a handful of 6-number affine maps -- the
        botanical/branching model that fold_fractal (a Mandelbox fold) is not. Mesh it via sdf_from_points ->
        sdf_to_mesh for geometry. Returns (n,2) points. Deterministic given `seed`."""
        from holographic.mesh_and_geometry.holographic_ifs import ifs_library, AffineIFS
        ifs = name_or_ifs
        if isinstance(name_or_ifs, str):
            lib = ifs_library()
            if name_or_ifs not in lib:
                raise ValueError("unknown IFS %r; known: %s" % (name_or_ifs, sorted(lib)))
            ifs = lib[name_or_ifs]
        return ifs.generate(n=n, seed=seed)

    def ifs_fit(self, target, bins=28):
        """Match a 2-D `target` point cloud to the CLOSEST NAMED affine-IFS system (holographic_ifs) -- the honest
        'fit a fern/tree' -- by occupancy signature. Returns {name, ifs, distance, quality, baseline, ranking, note}.
        `quality` beats `baseline` (the library mean) when the target really resembles a known system; a cloud unlike
        anything scores near baseline. The botanical companion to fold_fit (Mandelbox) and fit_shape. Kept negative:
        snaps to a LIBRARY, does not recover arbitrary IFS maps; not rotation-invariant."""
        from holographic.mesh_and_geometry.holographic_ifs import ifs_fit
        return ifs_fit(target, bins=bins)

    def fit_primitives(self, target, k=6, auto_k=False, k_max=16, tol=0.05, primitives=("sphere", "box", "capsule")):
        """Approximate a (M,3) point cloud with a UNION of PRIMITIVES, best-fit per cluster (holographic_primfit) --
        the honest model for a HARD-SURFACE or NON-FRACTAL organic shape (a 'creature', a part) that fold_fractal and
        the affine-IFS library can't represent. Clusters the points deterministically, then per cluster fits a SPHERE
        (round parts), an ORIENTED BOX (blocky parts, via PCA), and a CAPSULE (elongated/rounded limbs) and keeps the
        best-fitting one. All are EXACT SDFs, so the union raymarches / sdf_to_mesh's / to_shadertoy's. Returns {sdf,
        parts, kinds, quality, baseline, residual, k}: `kinds` counts each primitive type chosen; `quality` is how
        many times better than a single bounding sphere. `primitives` restricts the palette (('sphere',) = the old
        sphere-only behaviour). auto_k=True grows K to the elbow. Kept negative: a cluster spanning two oriented parts
        is fit by one loose primitive (raise K); approximates the surface, not a minimal CSG tree."""
        from holographic.mesh_and_geometry.holographic_primfit import fit_primitives
        return fit_primitives(target, k=k, auto_k=auto_k, k_max=k_max, tol=tol, primitives=primitives)

    def humanoid(self, targets=None, scale=1.0, iters=30, skin=True, body=None,
                 limb_radius=0.06, head_radius=0.11, torso_radius=0.10):
        """Build a parametric biped HUMANOID with automatic IK rigging and CHARACTER-EDITOR body morphs
        (holographic_humanoid). Starts in a T-pose; if `targets` (end_effector -> (x,y,z)) is given, each limb is posed
        by IK (FABRIK) keeping bone lengths. `body` is a character-editor parameter block (see body_params /
        default_body): global weight/muscle/fat sliders that distribute across the body by region, per-segment
        muscle/fat/length overrides, and optional breast geometry (size/sag/separation/nipple_diameter/nipple_depth).
        Returns the posed Humanoid; if `skin=True` also its morphed primitive-skin SDF (meshes, emits Shadertoy). All
        morphs default to 0 -> the base build is byte-identical to the un-morphed figure. End-effectors: l_wrist,
        r_wrist, l_ankle, r_ankle, head."""
        from holographic.mesh_and_geometry.holographic_humanoid import Humanoid
        h = Humanoid(scale=scale, body=body)
        if targets:
            h.pose_to(targets, iters=iters, mind=self)
        if skin:
            return h, h.skin(limb_radius=limb_radius, head_radius=head_radius, torso_radius=torso_radius)
        return h

    def body_params(self):
        """The neutral character-editor parameter block for humanoid() (holographic_humanoid.default_body) -- every
        slider at 0. Copy and adjust: global weight/muscle/fat in [-1,1]; segments[name] = {muscle, fat, length} for
        torso/neck/shoulder/upper_arm/forearm/hip/thigh/shin; breasts = None or {size, sag, separation,
        nipple_diameter, nipple_depth}. Pass the result as humanoid(body=...)."""
        from holographic.mesh_and_geometry.holographic_humanoid import default_body
        return default_body()

    def fit_pose(self, keypoints, camera=None, iters=30, scale=1.0):
        """Fit a HUMANOID rig to KEYPOINTS -- the honest 'approximate a pose' (holographic_humanoid). Pass 3-D
        keypoints (a dict joint_name -> (x,y,z), e.g. from mocap) for a direct IK fit, OR 2-D image keypoints (dict
        joint_name -> (u,v)) WITH a `camera` (needs .ray(uv) and .project(pts)) for a bone-length-constrained lift +
        IK. Returns the posed Humanoid (2-D also returns the lifted keypoints). KEPT NEGATIVE, loud: this fits
        KEYPOINTS, it does NOT detect them in an image (that needs a learned model, which the engine forbids); and a
        monocular 2-D lift is depth-ambiguous, so it recovers A plausible pose, not THE unique one."""
        from holographic.mesh_and_geometry.holographic_humanoid import fit_pose_3d, fit_pose_2d
        if camera is not None:
            return fit_pose_2d(keypoints, camera, iters=iters, mind=self, scale=scale)
        return fit_pose_3d(keypoints, iters=iters, mind=self, scale=scale)

    def solve_ik_limited(self, joints, target, limits, iters=20, root_ref=(0.0, 1.0, 0.0)):
        """CONSTRAINED inverse kinematics (holographic_iklimit): reach `target` while keeping each joint within an
        anatomical limit -- no hyperextended elbows/knees, ball joints within their cones. `joints` is (n+1,3);
        `limits` is a list of len n, each None (free) or {'type':'hinge','axis','lo','hi'} / {'type':'cone','half',
        'ref'?} in RADIANS. Alternates a FABRIK reach (solve_ik) with a root->tip limit projection (constrained
        FABRIK, Aristidou-Lasenby). Returns (joints, reach_error) -- error>0 when the limits correctly prevent
        reaching an out-of-range target. Bone lengths preserved; never returns an out-of-limit pose. Kept negative:
        angle limits only, no self-collision."""
        from holographic.mesh_and_geometry.holographic_iklimit import solve_ik_limited
        return solve_ik_limited(joints, target, limits, iters=iters, root_ref=root_ref, mind=self)

    def creature(self, spec, skin=True):
        """Build a Spore-style non-humanoid CREATURE from a body-plan spec (holographic_creature) -- a spine with limbs
        attached at fractional positions, bilateral symmetry, and generic organic joint constraints (a cone at each
        limb mount, no-hyperextension hinges along it). `spec`: {spine:{length,segments,axis,curve,radius}, limbs:[{at,
        dir,segments,length,radius,mirror,cone_deg,hinge_deg}], head:{at,radius}, body:<morph block>}. Returns the
        Creature; if skin=True also its morph-aware primitive-skin SDF (meshes, emits Shadertoy). Pass mind.creature
        (mind.quadruped_spec()) for a ready quadruped. Generalises the humanoid to arbitrary body plans."""
        from holographic.mesh_and_geometry.holographic_creature import Creature
        cre = Creature(spec)
        if skin:
            return cre, cre.skin()
        return cre

    def creature_pose(self, spec, targets, iters=30):
        """Build a creature from `spec` and pose its limbs to `targets` ({chain_name: (x,y,z)}) via CONSTRAINED IK in
        one deterministic call (holographic_creature). Chain names are 'L0','L0m','L1',... (m = the mirrored twin).
        Returns (Creature, skin_sdf). Joint limits (and their muscle/fat tightening) are enforced, so limbs never
        hyperextend."""
        from holographic.mesh_and_geometry.holographic_creature import Creature
        cre = Creature(spec)
        cre.pose(targets, iters=iters, mind=self)
        return cre, cre.skin()

    def quadruped_spec(self, body=None):
        """A ready-made creature body plan -- a quadruped (spine + two mirrored leg pairs + head)
        (holographic_creature.quadruped_spec). A concrete starting spec to build on; pass to creature()."""
        from holographic.mesh_and_geometry.holographic_creature import quadruped_spec
        return quadruped_spec(body=body)

    def greeble(self, base_mesh, seed=None, density=0.7, max_height=0.15, footprint=0.5):
        """S2 -- cover a base mesh's faces with extruded greeble boxes (the G5 panel idea on any surface) ->
        a merged Mesh of mechanical hull detail. Seat: Quilez/demoscene. KEPT NEGATIVE: instancing, not CSG
        -- greebles can intersect the hull (which is how greebling actually looks)."""
        from holographic.io_and_interop.holographic_procgen import greeble_mesh
        return greeble_mesh(base_mesh, seed=self.seed if seed is None else seed,
                            density=density, max_height=max_height, footprint=footprint)

    def vegetated_terrain(self, seed=None, n_plants=10, plant_iterations=3, terrain_kwargs=None):
        """S2 -- a fBm terrain (G4) with L-system plants (G5) scattered on its surface -> one scenegraph.
        Returns (scene_node, terrain). Composes terrain + grammar + scatter; flatten_scene gives the mesh.
        KEPT NEGATIVE: a deterministic scatter at the terrain height, not an ecology (no collision/clustering)."""
        from holographic.io_and_interop.holographic_procgen import vegetated_terrain
        return vegetated_terrain(self.seed if seed is None else seed, n_plants=n_plants,
                                 plant_iterations=plant_iterations, terrain_kwargs=terrain_kwargs)

    def procedural_compression(self, sdf_node, bounds=((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), res=48):
        """S3/C1 -- measure procedural representation AS compression: the tiny generator (DSL) vs the
        expanded geometry it marches to. Returns {dsl, dsl_bytes, mesh_faces, mesh_bytes, ratio}. The
        finding it makes concrete: a generator's size is CONSTANT in its output's complexity (a Menger
        DSL is 12 bytes at any depth), so storing the LAW escapes the capacity/complexity wall -- the same
        MDL principle as symbolic_regress/compress_signal, for geometry. KEPT NEGATIVE: only compressible
        content has a short generator (an arbitrary mesh does not); procedural compression is lossy and
        content-restricted, not a universal codec."""
        from holographic.io_and_interop.holographic_procbridge import procedural_compression
        return procedural_compression(sdf_node, bounds=bounds, res=res)

    def rate_distortion_report(self, arrays, target_cos=0.9999):
        """Duda's question, answered honestly: what is the CHEAPEST bit budget that stores these vectors while
        keeping their GEOMETRY (pairwise similarity), not just their bits? Runs the geometry-preserving rate-
        distortion code (auto KLT rank + coarsest quantization step meeting `target_cos` MEAN reconstruction cosine,
        rANS-entropy-coded -- Duda's ANS) and reports it AGAINST the float32 baseline so the answer is never dressed
        up as a free win. Returns {bits_per_vector, float32_bits_per_vector, ratio, achieved_cos_mean,
        achieved_cos_min, rank, pays}. `pays` is True only when ratio > 1 (it compresses); KEPT NEGATIVE, kept loud:
        incompressible near-orthogonal vectors do NOT pay -- the RD code can be LARGER than float32, and this report
        will say so (pays=False). Wires holographic_ratedistortion, which was import-only. See
        holographic_ratedistortion.geometry_preserving_code / bits_per_vector."""
        import numpy as _np
        from holographic.misc.holographic_ratedistortion import geometry_preserving_code, bits_per_vector, reconstruct
        A = _np.asarray(arrays, float)
        if A.ndim == 1:
            A = A[None, :]
        code = geometry_preserving_code(A, target_cos=target_cos)
        recon = _np.asarray(reconstruct(code))
        cos = []
        for o, r in zip(A, recon):
            no = _np.linalg.norm(o) * _np.linalg.norm(r)
            cos.append(float(_np.dot(o, r) / no) if no > 1e-12 else 1.0)
        bpv = float(bits_per_vector(code))
        base = float(A.shape[1] * 32)                       # float32 baseline, bits per vector
        ratio = base / bpv if bpv > 0 else float("inf")
        return {"bits_per_vector": bpv, "float32_bits_per_vector": base, "ratio": ratio,
                "achieved_cos_mean": float(_np.mean(cos)), "achieved_cos_min": float(min(cos)),
                "rank": int(code["B"].shape[0]) if hasattr(code.get("B"), "shape") else None,
                "pays": bool(ratio > 1.0)}

    def soft_min(self, a, b, k):
        """S3/C4 -- the log-sum-exp soft minimum, -k*log(exp(-a/k)+exp(-b/k)); k->0 gives min(a,b). This is
        the SAME log-sum-exp the modern-Hopfield/softmax cleanup uses (softmax is a soft-arg-MAX; this is a
        soft-arg-MIN over distances): a smooth CSG union of geometry and a soft recall of a memory are one
        temperature-controlled operator, with k playing the role of 1/beta. The SDF's smooth_union and the
        memory cleanup are the same math seen in two domains."""
        from holographic.io_and_interop.holographic_procbridge import soft_min
        return soft_min(a, b, k)

    def evolving_atom(self, n_harmonics=3, dim=None, forgetting=1.0, delta=1e-6):
        """SUBSTRATE EVOLUTION -- a context-conditioned atom that updates its OWN harmonic coefficients as
        (context angle, meaning) pairs stream in, via online Recursive Least Squares (the batch
        harmonic_atom fit made autonomous). Returns a holographic_harmonic.OnlineHarmonicAtom: `.observe(
        theta, meaning)` folds in one observation by a rank-1 Sherman-Morrison step, `.decode(theta)` reads
        the current meaning. forgetting=1.0 converges to the batch least-squares fit; forgetting<1.0 turns
        the codebook into a dynamical system that TRACKS a drifting meaning function. No RNG, no autodiff --
        the engine's own least squares, online. KEPT NEGATIVE: forgetting<1 trades steady-state accuracy on
        a stationary function for tracking a non-stationary one."""
        from holographic.sampling_and_signal.holographic_harmonic import OnlineHarmonicAtom
        return OnlineHarmonicAtom(n_harmonics, self.dim if dim is None else dim,
                                  forgetting=forgetting, delta=delta)

    def optimize_toolchain(self, tool_vecs, goal_sig, length, steps=200, lr=0.5):
        """DIFFERENTIABLE ORCHESTRATION -- optimize a whole tool-chain JOINTLY against a chain-level
        structural score, instead of scoring tools independently. `tool_vecs` (N, D) are the registry's
        tool vectors (tools already live in hyperspace); `goal_sig` is the desired composed chain signature
        (the order-encoded superposition a working chain would produce). Optimizes a soft selection over
        tools per step by gradient ASCENT on cosine(chain_signature, goal_sig) -- the gradient derived
        analytically through cosine/superposition/permute/softmax in numpy, NO autodiff (the same
        gradient-without-a-framework method as holographic_optimize). Returns (indices, score). KEPT
        NEGATIVE: a local optimum of a non-convex landscape; on orthogonal/easy tool sets per-position
        greedy already wins -- the gain is on CORRELATED tool sets where independent scoring is misled by
        cross-talk between positions."""
        from holographic.scene_and_pipeline.holographic_orchestrator import optimize_toolchain
        return optimize_toolchain(tool_vecs, goal_sig, length, steps=steps, lr=lr)

    def synthesize_program(self, library, goal_sig, max_length=4, threshold=0.85, steps=200, lr=0.5):
        """Fill a VOID CAPABILITY GAP (SYNTH-1): when no registered tool chain reaches a goal, SYNTHESISE one in
        the latent space instead of failing. Optimises a chain over `library` toward `goal_sig` (growing the
        length if a short program won't reach -- the structural 're-bundle'), VERIFIES the discrete chain's
        coherence (never trusts the soft optimum), and GATES: returns status 'synthesized' with the chain if
        coherence >= `threshold`, else 'abstain' (the gap genuinely could not be filled -- it declines rather than
        running an incoherent program). Measured: 20/20 reachable goals synthesized (coh ~1.0), 20/20 unreachable
        abstained (best coh ~0.19) -- the gate cleanly separates fillable from void gaps. The latent ascent is a
        hand-derived analytic gradient (numpy, NO autodiff); abstention is the load-bearing safety property. See
        holographic_voidsynth."""
        from holographic.misc.holographic_voidsynth import synthesize_for_goal
        return synthesize_for_goal(library, goal_sig, max_length=max_length, threshold=threshold, steps=steps, lr=lr)

    def blend_programs(self, sig_a, sig_b, weights=(1.0, 1.0)):
        """BLEND two program signatures into one by bundling -- composition in the shared substrate. The blend
        stays coherent to BOTH source goals (measured ~0.72/0.74 for a graphics + an audio program), so one
        vector can carry two intents: a program + a program, or a program + data. Because every domain's tools
        live in the SAME space, this cross-domain blend is what 'synesthesia across domains' actually is -- the
        project's one-algebra thesis, not a new sense. See holographic_voidsynth.blend_programs."""
        from holographic.misc.holographic_voidsynth import blend_programs
        return blend_programs(sig_a, sig_b, weights=weights)

    def fill_capability_gap(self, library, goal_sig, registry_hit=None, threshold=0.85, max_length=4, steps=200):
        """The orchestration: if a registered tool/chain already reaches the goal (`registry_hit` >= threshold),
        use it (status 'registry', no gap); otherwise synthesise and gate/abstain. The bridge from the
        orchestrator's plan()='gap' to verified latent synthesis. See holographic_voidsynth.fill_capability_gap."""
        from holographic.misc.holographic_voidsynth import fill_capability_gap
        return fill_capability_gap(library, goal_sig, registry_hit=registry_hit, threshold=threshold,
                                   max_length=max_length, steps=steps)

    def agent(self, actions, dim=512, seed=0, value_floor=0.25, pain_reflex=0.6, synth_threshold=0.8):
        """An upgraded creature agent (AGENT-1): an action LIBRARY as VSA atoms, reward AND pain affect, a
        pain-avoidance REFLEX (one painful experience blocks an action, faster than value learning), and VOID-GAP
        ACTION SYNTHESIS -- when no learned single action fits the situation, it synthesises a multi-step action
        program toward a goal and gates it (commit if coherent, else abstain to a safe default), reusing the
        SYNTH-1 loop. Self-explaining (decide returns WHY) and deterministic. Because actions are atoms, a plan has
        a composed signature embeddable in / blendable with other VSA programs -- the agent can DRIVE a program.
        Build agents from this for roles beyond a maze NPC; the bespoke per-action value engine still lives in
        HolographicMind. See holographic_agent.Agent."""
        from holographic.agents_and_reasoning.holographic_agent import Agent
        return Agent(actions, dim=dim, seed=seed, value_floor=value_floor, pain_reflex=pain_reflex,
                     synth_threshold=synth_threshold)

    def drive_system(self, weights=None):
        """A set of homeostatic DRIVES (DRIVE-1): internal needs (clarity, understanding, coverage, energy) that
        decide which faculty an agent should apply next -- the most under-satisfied applicable need wins. The
        mechanism that lets the agent DRIVE denoising / pattern recognition / descent decisions through a deeply
        nested process where the schedule is too large to hand-script. See holographic_drives.DriveSystem."""
        from holographic.misc.holographic_drives import DriveSystem
        return DriveSystem(weights=weights)

    def drive_process(self, root, codebook, drives=None, energy=24, recog_thresh=0.5, policy="drive", seed=0):
        """Walk a NESTED/fractal process under homeostatic drives, choosing at each node whether to DENOISE,
        RECOGNISE, or DESCEND by which need is most starved (policy='drive'); 'denoise'/'recognize'/'descend' are
        fixed-priority baselines and 'random' is the naive control. Faculties are real (codebook cleanup + cosine
        recognition; recognition only succeeds on a cleaned signal). MEASURED: the drive schedule matches the best
        fixed priority WITHOUT being told it (~0.46 vs ~0.45 worst-served need) and beats naive scheduling 2-4x --
        an adaptive default for processes too nested to schedule by hand. See holographic_drives.drive_process and
        make_nested_process."""
        from holographic.misc.holographic_drives import drive_process
        return drive_process(root, codebook, drives=drives, energy=energy, recog_thresh=recog_thresh,
                             policy=policy, seed=seed)

    def abstract_program(self, examples, name=None, max_depth=2, threshold=0.9):
        """Abstract a reusable PROGRAM from a TRACE -- a set of (input_vec, output_vec) examples demonstrating one
        transform (a recorded behaviour, a demonstration, the in/out of a creature's moves). Synthesise a
        procedure that reproduces the FIRST example, then VERIFY it reproduces ALL the others: the abstraction is
        the program CONSISTENT ACROSS examples, not a fit to one. If it generalises, optionally store it by `name`
        (callable later). Returns dict{program, generalizes, fit (mean cosine over examples), worst}. WHY this
        beats raw prototypes (the 'transfers better' claim): a stored prototype only matches near-identical
        states; an abstracted program captures the TRANSFORM itself, so it transfers to inputs never seen --
        measured in the tests against a prototype-nearest-neighbour baseline. HONEST: it only abstracts transforms
        expressible in the VM's ops within max_depth (BIND/BUNDLE/PERMUTE), and returns generalizes=False (rather
        than a wrong program) when the examples don't share one such transform."""
        import numpy as _np
        from holographic.agents_and_reasoning.holographic_ai import cosine as _cos
        examples = list(examples)
        prog = self.synthesize_procedure(examples[0][0], examples[0][1], max_depth=max_depth, threshold=threshold)
        if prog is None:
            return {"program": None, "generalizes": False, "fit": 0.0, "worst": 0.0}
        fits = []
        for inp, outp in examples:                           # VERIFY the synthesised program on EVERY example
            out, _ = self.run_procedure(prog, init_acc=inp)
            fits.append(float(_cos(out, outp)))
        worst = float(min(fits))
        generalizes = worst >= threshold                     # consistent across all examples -> a real abstraction
        if generalizes and name is not None:
            self.learn_procedure(name, prog)
        return {"program": prog, "generalizes": generalizes, "fit": float(_np.mean(fits)), "worst": worst}

    def path_trace(self, sdf, camera, width=96, height=96, spp=16, max_bounce=4, material=None, sky=None, seed=0):
        """Monte-Carlo PATH TRACER for true multi-bounce global illumination -- the core of V-Ray/Redshift/Arnold.
        Solves the full rendering equation over an SDF scene by following random light paths and averaging:
        BRDF importance sampling (cosine + GGX), Russian roulette, vectorised over rays. Indirect light (color
        bleeding, soft GI in concavities) falls out for free, unlike the engine's single-bounce irradiance cache.
        Returns an (H,W,3) HDR image. MEASURED: unbiased (white-furnace -> albedo), noise ~1/sqrt(spp), color
        bleed reproduced; 128^2/96spp ~13-16s (OFFLINE NumPy brain, NOT GPU-realtime). KEPT NEGATIVE: no
        next-event estimation, so light is gathered only when a bounce hits the emissive environment -- great for
        a big sky, very noisy for small emitters (NEE/MIS is the next step). See holographic_pathtrace.path_trace."""
        from holographic.rendering.holographic_pathtrace import path_trace
        return path_trace(sdf, camera, width=width, height=height, spp=spp, max_bounce=max_bounce,
                          material=material, sky=sky, seed=seed)

    def render_auto(self, sdf, camera, width=96, height=96, material=None, sky=None, quality="high",
                    max_bounce=4, seed=0, return_stats=False, **kw):
        """AUTO-CALIBRATING render -- one quality knob, no per-scene spp or denoise tuning. It wires together
        machinery the engine already had but never connected into a render loop: it samples in PASSES and, after
        each, asks the calibrated stop rule (holographic_adaptive_sample.converged_mask) which pixels have reached
        the target confidence interval -- those STOP, the rest keep sampling (path_trace's `active` mask) -- so
        hard pixels (glass, silhouettes, grazing reflections) automatically get more samples than flat regions.
        It then denoises with a VARIANCE-GUIDED SVGF whose per-pixel strength is set by the variance the sampler
        measured, so residual grain is smoothed and converged detail is preserved. `quality` is a target CI
        half-width (name 'draft'/'medium'/'high'/'ultra' or a float). MEASURED: at equal average sample budget it
        beats a raw path trace at draft/medium and ties near convergence at high (denoise stops mattering once a
        pixel is already converged -- the documented crossover). See holographic_gbuffer.render_auto."""
        from holographic.rendering.holographic_gbuffer import render_auto
        return render_auto(sdf, camera, width, height, material, sky=sky, quality=quality,
                           max_bounce=max_bounce, seed=seed, return_stats=return_stats, **kw)

    def render_scene_document(self, scene, camera, width=96, height=72, quality="medium", max_bounce=4,
                              seed=0, sky=None, default_material="matte_gray", return_stats=False, sss_dir=None,
                              sss_depth=0.6, sss_sigma=4.0, lights=None, dome_cache=False, demodulate=False, soft_light_cache=False,
                              indirect_cache=False, view=None, affine=False):
        """Render the canonical SCENE DOCUMENT (holographic_scene_doc.Scene) -- the 'a modeling app builds a
        document, then renders it' path. The document is a table of objects (each a stable handle + transform +
        SDF geometry + library material); this flattens it to ONE scene SDF (nearest-object distance) plus a
        material_fn that shades each hit with its owning object's material, then renders with render_auto. So the
        renderer consumes the authoritative scene instead of a hand-built Python class per scene (backlog H7).
        `sss_dir` (a light direction) turns on the SUBSURFACE glow for translucent materials (wax/jade/skin).
        `dome_cache` (default off) serves any DomeLight via the cheap cached-dome pass (holographic_domecache)
        instead of ray-traced ambient occlusion. `demodulate` (default off) denoises by dividing the albedo out
        (holographic_modulate, M4) -- cleaner on textured diffuse surfaces. See
        `view` (default None = the raw scene-referred buffer, unchanged) applies a DISPLAY transform on the
        way out: the tracer emits linear radiance with no upper bound, and MEASURED on a dome + area-light
        still life 15.5% of pixels left it above 1.0 and clipped flat when saved. view="display" is the
        correctness step (metered auto-exposure -> ACES -> gamma: 0.0000 clipped, 0.0000 crushed);
        view="graded" adds the look (bloom/vignette/grain); or pass a PostChain. It stays OFF by default
        because a caller measuring radiance or diffing two renders needs the linear buffer and would be
        silently wrong if a tone curve appeared under it. See
        `affine` (default False = shipped behaviour) also applies the object's ROTATION. Off by default
        because turning it on changes the picture of every scene containing a rotated object -- the current
        picture is wrong, but shipped output does not move without an explicit decision. mind.place() writes
        transforms that expect affine=True. See holographic_scene_render.render_scene_document."""
        from holographic.rendering.holographic_scene_render import render_scene_document
        return render_scene_document(scene, camera, width=width, height=height, quality=quality,
                                     max_bounce=max_bounce, seed=seed, sky=sky,
                                     default_material=default_material, return_stats=return_stats, sss_dir=sss_dir,
                                     sss_depth=sss_depth, sss_sigma=sss_sigma, lights=lights, dome_cache=dome_cache,
                                     demodulate=demodulate, soft_light_cache=soft_light_cache,
                                     indirect_cache=indirect_cache, view=view, affine=affine)

    def render_preview(self, scene, camera, width=240, height=180, scale=0.5, max_bounce=1,
                       quality="draft", seed=0, sky=None, lights=None, view="display", **kw):
        """A FAST, deliberately rough look at a Scene document -- the 'is it roughly right?' pass.

        MEASURED against render_scene_document at the SAME 240x180 output, same scene/lights/seed:
            preview 3.81s (sd 0.02)   full 45.85s (sd 0.06)   -> 12.0x, mean abs error 0.0159
        Use it for the see->fix loop, where eight looks beat one render; use render_scene_document for
        anything you will keep.

        THE OBVIOUS PLAN WAS WRONG AND THE MEASUREMENT SAID SO. "Render small and upscale" buys under 2x:
        the tracer is DISPATCH-bound at preview sizes (16x the pixels cost 2.8x the time, log-log slope
        ~0.3), so pixels are nearly free and the cost is a fixed number of numpy passes. The win is in
        PASSES -- max_bounce=1 is 2.76x and quality='draft' another 1.72x. Upscaling stays in the path as
        an OUTPUT-SIZE lever, not a speed one.

        The trade is exactly what one bounce costs: indirect light. A preview is flatter, with darker
        shadows, than the final. See holographic_scene_render.render_preview for the full measurements and
        for why bake_sdf is NOT used here (measured 0.5-0.6x on scenes like this)."""
        from holographic.rendering.holographic_scene_render import render_preview
        return render_preview(scene, camera, width=width, height=height, scale=scale,
                              max_bounce=max_bounce, quality=quality, seed=seed, sky=sky,
                              lights=lights, view=view, **kw)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p07_mesh_csg", "_UnifiedPart07")
    print("holographic_unified_p07_mesh_csg selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

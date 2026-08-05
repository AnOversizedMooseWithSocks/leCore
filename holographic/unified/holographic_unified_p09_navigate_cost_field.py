"""Part 09 of UnifiedMind's faculty surface -- 158 methods, navigate_cost_field .. photo_to_3d.

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


class _UnifiedPart09:

    def navigate_cost_field(self, cost, shape, start, goal, blocked=None, lo=None, hi=None):
        """NAVIGATE a known N-D COST FIELD: discretize it to a grid, weight edges by the field, and return the
        LEAST-COST path start->goal. The one primitive for routing through smoke density (volumetrics), a potential
        (physics), or any resistance/terrain (particles); the uniform maze is the special case. `cost` is a callable
        f(points)->cost or an array of shape `shape`. See holographic_ndfield. (Distinct from navigate_field, which is
        the gravity/attractor field navigator.)"""
        from holographic.misc.holographic_ndfield import navigate_field
        return navigate_field(cost, shape, start, goal, blocked=blocked, lo=lo, hi=hi)

    def path_cost(self, path, cost, shape, lo=None, hi=None):
        """Total field cost accumulated along a path -- for comparing a navigated route to a naive straight shot."""
        from holographic.misc.holographic_ndfield import path_cost
        return path_cost(path, cost, shape, lo=lo, hi=hi)

    def cost_to_go_field(self, cost, shape, goal, blocked=None, lo=None, hi=None):
        """SOLVE THE WHOLE VALUE FIELD ONCE, then route from ANYWHERE for free. One Dijkstra sweep from the goal yields
        V (cost-to-go at every cell) and a next-step field; after that, routing from any start is an O(path) descent, no
        re-search -- the 'precompute once, read out anywhere' pattern (the SDF bake, PRT) carried into navigation. The
        win grows with the number of agents/queries to that goal (measured ~5x at 8 starts, and each extra route is
        ~free). V is also a POTENTIAL / VALUE FUNCTION: its negative gradient is a physics force, and descent on it is an
        optimal policy -- the same object as a distance field, a physics potential, and an RL value. Returns
        (V, nxt, route) where route(start) -> the cell path to the goal. See holographic_ndfield."""
        from holographic.misc.holographic_ndfield import field_weighted_graph, cost_to_go, route_from
        nbr, edge_cost = field_weighted_graph(shape, cost, blocked=blocked, lo=lo, hi=hi)
        V, nxt = cost_to_go(nbr, edge_cost, goal)
        return V, nxt, (lambda start: route_from(nxt, start, goal))

    def straight_line_cells(self, start, goal):
        """The grid cells a straight line start->goal crosses -- the tie-break-independent baseline a naive shot pays."""
        from holographic.misc.holographic_ndfield import straight_line_cells
        return straight_line_cells(start, goal)

    def navigate_scene(self, sdf_eval, lo, hi, shape, start_world, goal_world, clearance=0.25):
        """Route an agent through a LIVE SCENE using its own SDF as the cost field: inside geometry is impassable,
        near a surface is costly, so the path threads free space around objects. One structure for drawing AND moving.
        `sdf_eval` is a callable P->signed distance. Returns world-space waypoints. See holographic_ndfield."""
        from holographic.misc.holographic_ndfield import navigate_scene
        return navigate_scene(sdf_eval, lo, hi, shape, start_world, goal_world, clearance=clearance)

    def encode_path(self, path, dim=2048, seed=0):
        """A navigated path -> ONE hypervector, COMPOSABLE in a VSA program (bind to a label, bundle several routes,
        query order). Returns (vector, SequenceMemory, keys). See holographic_ndfield."""
        from holographic.misc.holographic_ndfield import encode_path
        return encode_path(path, dim=dim, seed=seed)

    def decode_path_step(self, vec, sm, keys, i):
        """Read waypoint i back out of a path hypervector -- the route survives as composable VSA data."""
        from holographic.misc.holographic_ndfield import decode_path_step
        return decode_path_step(vec, sm, keys, i)

    def emit_from_surface(self, sdf_eval, n, bounds, speed=1.0, weight=None, seed=0):
        """Spawn particles ON a surface to DRIVE a particle system: samples the zero level-set of `sdf_eval`, returns
        (positions, normals, velocities) with velocity along the outward normal. `speed` and `weight` (emission
        density on the surface) each take a constant OR a map / field / wired output. See holographic_emitter."""
        from holographic.simulation_and_physics.holographic_emitter import emit_from_surface
        return emit_from_surface(sdf_eval, n, bounds, speed=speed, weight=weight, seed=seed)

    def advance_particles(self, pos, vel, force=None, dt=0.05, damping=0.0, wrap_to=None):
        """One integration step for an N-D particle set (gravity / attractor / sampled-field force). See holographic_emitter."""
        from holographic.simulation_and_physics.holographic_emitter import advance
        return advance(pos, vel, force=force, dt=dt, damping=damping, wrap_to=wrap_to)

    def param(self, value=None, field=None, map=None, domain=None, source=None, default=0.0):
        """Make a connectable parameter SOCKET: a value that is a constant OR wired to a map / field / named output --
        the 'choose a map instead of a number' affordance. Pass the result anywhere a faculty resolves parameters
        (region reflect/roughness, emit speed/weight, a field cost). See holographic_param."""
        from holographic.misc.holographic_param import Param
        return Param(value=value, field=field, map=map, domain=domain, source=source, default=default)

    def resolve_param(self, p, points=None, ctx=None, n=None):
        """Resolve any parameter (constant / map / field / socket) to concrete values at `points`. See holographic_param."""
        from holographic.misc.holographic_param import resolve_param
        return resolve_param(p, points=points, ctx=ctx, n=n)

    def collide_sdf(self, X, sdf_eval, radius=0.0):
        """ENVIRONMENT collision: push every point inside `sdf_eval` (signed distance < radius) out to the surface --
        keep particles / cloth outside scene geometry. The positional contact resolve behind SoftBody.step(collider=...).
        See holographic_collide."""
        from holographic.simulation_and_physics.holographic_collide import resolve_sdf_collision
        return resolve_sdf_collision(X, sdf_eval, radius=radius)

    def sdf_collision_projection(self, sdf_eval, N, D, radius=0.0):
        """A collision PROJECTION callable for project_onto_constraints -- so 'stay outside this surface' is just one
        more constraint in the SAME unified sweep as distance/bend/denoise/resonator (Macklin's one-solver-many-uses).
        See holographic_collide."""
        from holographic.simulation_and_physics.holographic_collide import sdf_collision_projection
        return sdf_collision_projection(sdf_eval, N, D, radius=radius)

    def dirty_field(self, shape, lo=None, hi=None, base=0.0):
        """A navigation / physics cost field with DIRTY-FLAG deltas: add movable colliders, then `move` one and only
        its footprint is re-evaluated (O(footprint), grid-size-independent), staying bit-identical to a full rebuild.
        The 'recompute only what changed' render discipline, carried into physics/nav. Returns a DirtyField whose
        `cost_grid()` feeds navigate_field. See holographic_dirtyfield."""
        from holographic.misc.holographic_dirtyfield import DirtyField
        return DirtyField(shape, lo=lo, hi=hi, base=base)

    def bake_sdf(self, sdf, lo, hi, res):
        """PRECOMPUTE a scene SDF (anything with `.eval`, and optionally `.ids`) onto a grid, then sample it O(1) --
        the realtime distance-field shortcut. Cost of a sample is independent of the number of primitives, so the ONE
        baked grid speeds every SDF consumer at once (the shader's trace/shadows/AO/reflections, navigation, collision,
        emission). Amortises over many rays/frames/queries. Returns a GridSDF (a drop-in union). See holographic_sdfbake."""
        from holographic.mesh_and_geometry.holographic_sdfbake import GridSDF
        return GridSDF.bake(sdf, lo, hi, res)

    def dispatch_methods(self, x, tags, ops, default=None):
        """COMPOSABILITY OF CALCULATION METHODS -- apply a DIFFERENT operator to different elements of one structure,
        chosen per-element by `tags`, and recombine. The same "part fluid, part static, by a field" idea the engine
        uses for DATA, now applied to WHICH COMPUTATION runs where: trace the first hit, then per bounce dispatch to
        collapse (a PRT dot product) on diffuse, trace on a mirror, a glossy bundle on a rough patch -- switching on the
        fly. It is the per-element generalization of the whole-signal method selection `denoise(method='auto')` and
        `decompose_signal` already do. `ops` is {label: fn(sub_x)->sub_y}. See holographic_dispatch."""
        from holographic.scene_and_pipeline.holographic_dispatch import dispatch_field
        return dispatch_field(x, tags, ops, default=default)

    def render_dispatch(self, sdf, camera, width, height, methods, colors, light, order=3, n=400):
        """RENDER by dispatching each hit to its best method and get a RELIGHT handle -- the pipeline form of "collapse
        on diffuse, trace on a mirror, switch on the fly". `methods` maps object id -> 'collapse' (PRT dot product) or
        'trace' (a mirror bounce whose diffuse hits themselves collapse). Returns (frame, relight, info): `relight(new
        light)` re-shades the collapsed parts for free, and `info` reports the dispatch counts. This is how PRT and the
        method dispatch are USED in a real render, not just measured. See holographic_dispatch.render_dispatch."""
        from holographic.scene_and_pipeline.holographic_dispatch import render_dispatch
        return render_dispatch(sdf, camera, width, height, methods, colors, light, order=order, n=n)

    def bake_scene(self, sdf, camera, width, height, methods, colors, order=3, n=400):
        """PRECOMPUTE / BAKE a scene BEFORE any render, so the first render is already a relight, not a cold trace. Call
        this once at scene-load: it traces primary visibility, dispatches each hit to its method, and precomputes the
        PRT transfer for every diffuse hit and every diffuse surface behind a mirror bounce. Returns a BakedScene to hand
        to `render_baked(scene, light)` -- interactive relighting is then a dot product from frame one. See
        holographic_dispatch.bake_scene."""
        from holographic.scene_and_pipeline.holographic_dispatch import bake_scene
        return bake_scene(sdf, camera, width, height, methods, colors, order=order, n=n)

    def render_baked(self, scene, light):
        """Relight a BakedScene (from bake_scene) -- shade every pixel from its precomputed transfer, no tracing. Every
        frame, including the first, is this cheap dot-product relight. Returns a (H,W,3) frame. See
        holographic_dispatch.render_baked."""
        from holographic.scene_and_pipeline.holographic_dispatch import render_baked
        return render_baked(scene, light)

    def render_adaptive(self, objects, camera, width=256, height=256, frames=1, relight=False, light=None,
                        sun="bright", sky="clear", post=None, **kw):
        """ONE render call that ADAPTS -- it looks at the scene and the workload and picks the methods itself instead of
        you choosing bake/relax/collapse/trace by hand. Grounded in the MEASURED break-evens: it bakes the SDF only when
        primitives or frames make it pay, keeps the exact active-only marcher (over-relaxation stays a manual opt-in),
        and -- when relighting -- COLLAPSES diffuse surfaces (PRT, free relight) while TRACING reflective ones, deriving
        the per-surface method from each material's reflectivity. Returns (frame, relight, plan); `plan['reasons']`
        explains every choice so the automation stays legible. The separate options (render_scene bake=/relax=,
        render_dispatch, radiance_transfer) remain for manual control. See holographic_adaptive.render_adaptive."""
        from holographic.misc.holographic_adaptive import render_adaptive
        return render_adaptive(objects, camera, width=width, height=height, frames=frames, relight=relight,
                               light=light, sun=sun, sky=sky, post=post, **kw)

    def plan_render(self, objects, frames=1, relight=False):
        """The DECISION LAYER of the adaptive pipeline, on its own: given a scene and workload, return the plan (bake
        resolution, relax factor, path, per-surface methods) with a human-readable reason for each choice -- so you can
        see what render_adaptive WOULD do, and why, without rendering. See holographic_adaptive.plan_render."""
        from holographic.misc.holographic_adaptive import plan_render
        return plan_render(objects, frames=frames, relight=relight)

    def distribute_compute(self, buckets, worker, reduce="sum", cache=None, backend=None,
                           est_ms_per_bucket=None, n_bytes=None, flops_per_byte=None):
        """DISTRIBUTED COMPUTATION the holostuff way -- decompose a job into buckets, hand every bucket the same shared
        read-only `cache` (the "GI cache on the main node"), run `worker(bucket, cache)` on each, and reassemble with a
        COMMUTATIVE monoid so the result is independent of bucket order (=> the buckets could run on separate machines /
        VMs with no stitch pass). `reduce` selects the reassembly operator that matches the computation: 'sum' (linear
        superposition -- forces, fields, radiance, densities), 'min' (SDF union), 'max' (occupancy), 'bundle' (VSA
        scene/memory), or a callable. Runs in-process here (no speedup claimed); it builds the STRUCTURE that makes
        distribution correct. Returns (result, info). See holographic_distribute."""
        from holographic.misc.holographic_scalehome import Scale                      # the Scale home  consolidation H3
        if backend == "auto":
            # AUTOMATIC DECISION, NOT AUTOMATIC SPAWN. The pool is created only when the gate says it pays,
            # and it is CLOSED again before returning -- so "auto" never leaves worker processes alive after
            # the call. That costs one pool setup per call, which is precisely why auto is not the default:
            # a caller making many calls should hold ONE pool via local_pool() and pass it, rather than
            # paying setup every time. A hidden persistent pool would be faster and would also mean a
            # library call silently left interpreters running, which is not a library's decision to make.
            # ONE ORACLE, NOT TWO. This used to call should_pool directly, which meant `auto` could only ever
            # answer "pool or not" -- it could not see the device at all, and it ignored the resource policy
            # unless should_pool happened to be handed it. place_work already composes all three (unit, pool,
            # device) with the policy veto first, so asking IT keeps one decision in one place. A second
            # copy of the routing logic here would be the "three unrelated switches" problem re-created
            # inside the thing built to fix it.
            decision = self.place_work(n_buckets=len(buckets),
                                       est_ms_per_bucket=est_ms_per_bucket,
                                       n_bytes=n_bytes, flops_per_byte=flops_per_byte)
            self._last_placement = decision          # readable via last_placement() -- the WHY is auditable
            ok = decision["placement"] == "pool"
            if not ok:
                # A 'device' verdict does NOT dispatch here: this seam partitions BUCKETS across workers,
                # and a device kernel is a different shape entirely (one dispatch over an array, not N
                # independent Python callables). Reporting it and running on CPU is honest; silently doing
                # nothing about it would not be. Callers with a device-shaped kernel use wgsl_* directly.
                backend = None                       # today's exact behaviour, no processes
            else:
                pool = self.local_pool()
                try:
                    return Scale.map_reduce(buckets, worker, reduce=reduce, cache=cache, backend=pool)
                finally:
                    pool.close()
        return Scale.map_reduce(buckets, worker, reduce=reduce, cache=cache, backend=backend)

    def resource_policy(self, **kwargs):
        """SET OR READ WHAT THIS PROCESS IS ALLOWED TO USE (holographic_policy, POLICY-1).
        With keywords, sets the policy: cpu_cores (int cap), pool ('allow'|'deny'), gpu
        ('auto'|'on'|'off'), device_memory_mb. With no arguments, returns the EFFECTIVE policy plus the
        SOURCE of every value (policy / env / default) -- provenance, so a recorded run says what it was
        allowed to use, not just what it did.
        WHY THIS EXISTS: cpu_budget() answers what is PHYSICALLY AVAILABLE, which is not the same question
        as what this process MAY TAKE. On a shared box, a CI runner, or a machine running leOS beside the
        user's real work, the operator's answer is smaller and the engine cannot infer it.
        A POLICY CAPS, IT DOES NOT COMMAND: cpu_cores=4 means never more than 4, and the measured gates
        still decide inside the cap (should_pool may still refuse). Asking for more cores than exist gives
        you what exists -- permission is not hardware.
        NUMERICS: cpu_cores and pool are PERFORMANCE-ONLY (the pooled path is verified bit-identical), but
        gpu is NOT -- GPU matches NumPy only to a tolerance. describe() flags which is which and reports
        bit_exact, so nobody flips GPU on globally and silently changes their results."""
        if kwargs:
            from holographic.scene_and_pipeline.holographic_policy import ResourcePolicy
            self._policy = ResourcePolicy(**kwargs)
            return self._policy.describe()
        return self._resource_policy().describe()

    def _resource_policy(self):
        """The active policy, defaulting to a permissive one. Private: callers use resource_policy()."""
        policy = getattr(self, "_policy", None)
        if policy is None:
            from holographic.scene_and_pipeline.holographic_policy import ResourcePolicy
            policy = self._policy = ResourcePolicy()
        return policy

    def place_work(self, n_bytes=None, flops_per_byte=None, n_buckets=None, est_ms_per_bucket=None,
                   baseline_ns=None, n_calls=1, unit=None):
        """WHERE SHOULD THIS WORK RUN -- one decision over CPU / process pool / device / machine-model unit
        (holographic_placement, PLACE-1). Returns {placement, why, considered, provisional}.
        WHY IT EXISTS: three oracles already answered three placement questions and NONE knew about the
        others -- machine_place_unit for units, should_pool for processes, should_offload for the device --
        so a caller had to consult all three and reconcile them by hand, and nothing reconciled them with
        resource_policy at all. An oracle could recommend a device the operator had FORBIDDEN.
        IT COMPOSES, IT DOES NOT REIMPLEMENT: every verdict comes from the existing oracle for that question.
        This contributes the ORDER, the policy veto, and one honest report.
        THE ORDER IS THE ARGUMENT. (1) The POLICY VETO comes first -- no arithmetic makes a forbidden device
        faster. (2) CHEAPEST-CORRECT WINS TIES: unit, then pool, then device, because a pool costs a process
        and a device costs a transfer AND CHANGES THE NUMBERS (GPU matches NumPy only to a tolerance, while
        the pooled path is verified bit-identical). Those are not equivalent risks.
        A candidate with missing inputs is reported as NOT EVALUATED rather than skipped -- a placement
        nobody costed and a placement that lost are different facts. A device recommendation comes back
        marked `provisional`, because should_offload's thresholds are arithmetic from PCIe bandwidth and no
        host<->device crossover has ever been measured here."""
        from holographic.scene_and_pipeline.holographic_placement import place_work
        return place_work(n_bytes=n_bytes, flops_per_byte=flops_per_byte, n_buckets=n_buckets,
                          est_ms_per_bucket=est_ms_per_bucket, baseline_ns=baseline_ns, n_calls=n_calls,
                          unit=unit, policy=self._resource_policy(), mind=self)

    def cpu_budget(self):
        """HOW MANY CORES THIS PROCESS MAY ACTUALLY USE (holographic_coordinator.cpu_budget), >= 1.
        NOT os.cpu_count(), WHICH LIES IN A CONTAINER: it reports the HOST's cores and ignores cgroup quota
        and CPU affinity, so `docker run --cpus=2` on a 64-core box answers 64 -- and a pool sized from that
        spawns 64 interpreters to time-share 2 cores, slower than sequential and 64x the memory. On an engine
        meant for small devices that is a memory-bloat bug, not just a speed one.
        Takes the MINIMUM of sched_getaffinity, cgroup v2 cpu.max, cgroup v1 cfs quota, and os.cpu_count();
        anything unreadable is skipped rather than guessed."""
        # Honours the resource policy: this reports what may be USED, not merely what exists, so every
        # caller that sizes work from it inherits the operator's cap for free.
        return self._resource_policy().cores()

    def should_pool(self, n_buckets, est_ms_per_bucket, cores=None, margin=4.0):
        """WOULD A PROCESS POOL PAY FOR THIS JOB (holographic_coordinator.should_pool) -> (verdict, why).
        Refuses on three independent grounds: fewer than 2 usable CORES (a pool cannot add speed, only
        overhead and memory), fewer than 2 BUCKETS (nothing to parallelise), or WORK PER BUCKET below
        margin x the ~0.2 ms dispatch cost (dispatch would dominate). The default margin of 4 is
        deliberately conservative -- near break-even a pool costs memory and process lifetime for no gain,
        so 'roughly equal' should decline.
        Same shape as the machine model's placement oracle, and the same lesson: the answer depends on the
        CALLER'S numbers, so it is computed rather than assumed."""
        from holographic.scene_and_pipeline.holographic_coordinator import should_pool
        policy = self._resource_policy()
        if not policy.pool_allowed():
            return False, "the resource policy forbids process pools (pool='deny')"
        return should_pool(n_buckets, est_ms_per_bucket,
                           cores=policy.cores() if cores is None else cores, margin=margin)

    def last_placement(self):
        """WHY did the last backend='auto' call route the way it did? Returns the full place_work decision
        (placement, why, every candidate considered, provisional) or None if `auto` has not run.
        AN AUTOMATIC DECISION THAT CANNOT BE INSPECTED IS INDISTINGUISHABLE FROM A BUG. The routing gate can
        decline for four different reasons -- one core, too few buckets, work below the dispatch floor, or a
        policy veto -- and a caller seeing 'it stayed on CPU' deserves to know which."""
        return getattr(self, "_last_placement", None)

    def local_pool(self, n=None):
        """SPIN UP LOCAL WORKER PROCESSES (holographic_coordinator.LocalPool) -- a PERSISTENT process pool
        (not spawn-per-task) where each worker is its own interpreter with its own GIL, so GIL-bound NumPy
        and Python work actually runs in parallel on one machine. A large read-only cache is published ONCE
        into shared_memory (zero-copy, mapped by every worker) rather than pickled per bucket.
        THIS IS THE ANSWER TO 'spin up another instance' ON ONE BOX. `farm` is its cross-machine sibling and
        does NOT provision anything -- it consumes hosts you already started. LocalPool is the one that
        actually creates workers.
        Pass n= for the worker count (default: one per core). Hand the result to
        distribute_compute(backend=...) or Coordinator(backend=...). The worker must be a TOP-LEVEL,
        PICKLABLE function -- a lambda or a closure cannot cross a process boundary.
        CLOSE IT when done (pool.close()) or use it as a context manager: the pool is persistent BY DESIGN,
        so nothing reclaims it for you.

        THE DEFAULT REMAINS SINGLE-PROCESS, and the honest reason is that the break-even HAS NOT BEEN
        MEASURED ON REPRESENTATIVE HARDWARE. Measured here: bit-identical to in-process (verified), and
        0.09x on light buckets rising only to ~1.00x at 200 ms/bucket -- but THIS MACHINE HAS ONE CORE
        (os.sched_getaffinity -> 1), so a process pool cannot win here by construction and those numbers
        say nothing about a real box. They are recorded as a CONFOUND, not as a result.
        What IS established regardless of core count: dispatch overhead is roughly 0.2 ms per bucket, so
        buckets doing less work than that can never pay. Measure `local_pool` on YOUR hardware before making
        it a default -- and note the constraint that decides it is not speed but DETERMINISM: the reduce is a
        commutative monoid and the pooled path is bit-identical, so parallelism here is safe to opt into."""
        from holographic.scene_and_pipeline.holographic_coordinator import LocalPool
        return LocalPool(n=n)

    def partition_domain(self, n, k, costs=None):
        """Decompose a domain of n items into k disjoint buckets for distribution. With `costs` (a per-item work
        estimate) it LOAD-BALANCES -- heaviest-first onto the lightest bucket -- so the slowest bucket, which bounds a
        farm's wall-time, is minimised (adaptive bucket sizing). Returns a list of index arrays. See
        holographic_distribute.partition / adaptive_partition."""
        from holographic.misc.holographic_scalehome import Scale                      # the Scale home  consolidation H3
        return Scale.partition(n, k, costs=costs)

    def partition_grid(self, shape, blocks):
        """Decompose a 2D image/field (shape=(H,W)) into TILES or a 3D volume/grid (shape=(X,Y,Z)) into BRICKS -- the
        render-farm bucket layout generalised to 2D and 3D. `blocks` is an int or a per-axis tuple. Returns a list of
        slice-tuples covering the domain disjointly; each is an independent bucket (a separate VM/node), and a 3D brick
        with no surface can be skipped (sparse volumes). Also the cache-blocking layout -- a tile/brick sized to a
        working budget streams through a fast cache level. See holographic_distribute.partition_2d / partition_3d."""
        from holographic.misc.holographic_scalehome import Scale                      # the Scale home  consolidation H3
        return Scale.tiles(shape, blocks)

    def distribute_bricks(self, out_shape, regions, worker, cache=None, fill=0.0, skip=None):
        """Run `worker(region, cache)` on each tile/brick and PLACE its result at that region -- disjoint, so
        order-independent and seamless (the shared read-only cache makes borders agree). `skip(region)->bool` drops
        EMPTY bricks (sparse volumes: most of a volume is empty space -- the real speed win of bricking 3D, beyond
        parallelism). Returns (out, info) with the ran/skipped counts. See holographic_distribute.distribute_bricks."""
        from holographic.misc.holographic_scalehome import Scale                      # the Scale home  consolidation H3
        return Scale.bricks(out_shape, regions, worker, cache=cache, fill=fill, skip=skip)

    def surface_material(self, name=None, color=(0.7, 0.7, 0.7), **channels):
        """The FIRST-CLASS render material: every channel (color, roughness, reflect, emission, opacity) is a Param
        SOCKET -- a constant, a `Param`, a callable field (e.g. a `pattern_field`), or a map array -- resolved PER HIT
        by `render_surface`. With `name`, channels start from the ONE canonical MATERIAL_RENDER table (no more
        per-demo copies) and your overrides apply on top. This is the object that ties param -> pattern -> material ->
        render together. See holographic_surface.SurfaceMaterial."""
        from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial
        m = SurfaceMaterial.from_name(name, color=color) if name is not None else SurfaceMaterial(color=color)
        for k, v in channels.items():
            setattr(m, k, v)
        return m

    def realize_recipe_fused(self, recipe, spectrum_cache=None):
        """Fill 4 integration at Layer 4: realize a StructureRecipe's outputs through the SCHEDULER, fusing its
        straight-line bind/bundle/permute runs so a long structure build does fewer FFTs than the op-by-op
        `recipe.build()`. Returns (outputs, stats). THROUGHPUT path -- fusion is ~1e-15, so this does NOT keep the
        recipe's bit-exact-replay guarantee; `recipe.build()`/`realize` stay the exact default. See
        holographic_schedule.run_recipe."""
        from holographic.scene_and_pipeline.holographic_schedule import run_recipe
        return run_recipe(recipe, fused=True, spectrum_cache=spectrum_cache)

    def spectrum_cache(self, max_items=4096, key="content"):
        """Fill 1 (residency): a cache of atom -> rfft(atom), so binds/unbinds against KNOWN atoms skip the
        forward transform. Bit-identical to recompute. Pass it to `fuse`/`fuse_record`/`run_scheduled` to make
        their leaf transforms free for known atoms.

        USE key="identity" WHEN THE SAME ATOM OBJECTS RECUR -- which is the normal case for a codebook or a role
        table. The historical `key="content"` default hashes the whole atom on every lookup, and a sha256 of D
        floats costs more than the rfft it is avoiding (D=1024: 21.5us hash vs 13.0us transform), so the content
        default measures 0.50x-0.82x -- a cache that is slower than no cache. Identity keying measures 2.4x-2.6x
        scalar and 3.7x-4.3x inside fuse_record, bit-identical. Content keying stays the default (never-flip) and
        is genuinely required when byte-identical arrays arrive as distinct objects.
        See holographic_residency.SpectrumCache."""
        from holographic.simulation_and_physics.holographic_memoryhome import Memory                # the Memory home  consolidation H6
        return Memory.spectrum_cache(max_items=max_items, key=key)

    def fuse_record(self, keys, values, spectrum_cache=None):
        """Fill 2 (spectral fusion): build a role/filler record -- bundle([bind(k_i, v_i)]) -- in ONE fused FFT
        pass (leaves+1 transforms instead of ~3*len), equal to the op-by-op result to ~1e-15. THROUGHPUT path:
        tie-sensitive encoders (the maze-rescue path) must NOT use this. See holographic_fuse."""
        from holographic.misc.holographic_computehome import Compute                  # the Compute home  consolidation H7
        return Compute.fuse_record(keys, values, spectrum_cache=spectrum_cache)

    def fuse_expression(self, expr, spectrum_cache=None):
        """Fill 2: evaluate a five-op (bind/unbind/bundle/permute) expression tree in the FFT domain -- one
        transform per leaf, one out. Build `expr` with holographic_fuse.{leaf,fbind,funbind,fbundle,fpermute}."""
        from holographic.misc.holographic_computehome import Compute                  # the Compute home  consolidation H7
        return Compute.fuse(expr, spectrum_cache=spectrum_cache)

    def superpose_batch(self, keys, items, gated=True):
        """Fill 3 (auto-superposition + spill): pack N independent keyed items into the FEWEST superposed vectors
        that keep each bucket under the capacity dial -- one vector if it fits, else SPILL across buckets (not
        abstain). Returns (packed_vectors, buckets). `apply_in_superposition` does one op on each bundle at once.
        See holographic_superschedule."""
        from holographic.scene_and_pipeline.holographic_superschedule import superpose_batch
        if not gated and len(items) > 0:
            import numpy as _np
            _d = int(_np.asarray(items[0]).size)
            if len(items) > 0.13 * _d:
                # gated=True spills correctly on its own; this tap fires only on
                # the MANUAL override crossing the measured readout law.
                self._scale_tap(
                    "ungated superposition of %d items in dim %d exceeds the "
                    "measured linear readout law k* ~ 0.13*D (~%d): recovery "
                    "will degrade. Use gated=True (spills into buckets) or "
                    "decode with mind.bundle_decode(method='omp') (4x the "
                    "linear ceiling)." % (len(items), _d, int(0.13 * _d)))
        return superpose_batch(keys, items, gated=gated)

    def apply_in_superposition(self, keys, items, op, gated=True):
        """Fill 3: the latency-hiding move -- hold items in superposition and apply ONE bind by `op` to each
        bucket's whole bundle at once (transforming every item in flight), then recover. Spills past the dial."""
        from holographic.scene_and_pipeline.holographic_superschedule import apply_in_superposition
        return apply_in_superposition(keys, items, op, gated=gated)

    def schedule_program(self, ops, min_run=2, spectrum_cache=None, sequential=False):
        """Fill 4 (the scheduler capstone): run a VSA program DAG (built with holographic_schedule.{leaf,op_bind,
        op_unbind,op_bundle,op_permute,op_cleanup}) with the linear runs FUSED, tie-sensitive runs kept op-by-op
        and bit-exact, and Python crossings only at the cleanups. Returns (values, stats) where stats reports the
        FFT count, kernel-op calls, and crossings. `sequential=True` runs the op-by-op baseline for comparison.
        See holographic_schedule."""
        from holographic.scene_and_pipeline.holographic_schedule import run_scheduled, run_sequential
        if sequential:
            return run_sequential(ops)
        return run_scheduled(ops, min_run=min_run, spectrum_cache=spectrum_cache)

    def measure_area(self, mesh):
        """Modeling-app backlog (measurement + units): total surface area of a mesh, as a dimensioned [m^2]
        Quantity measured from the geometry. See holographic_metrology.surface_area."""
        from holographic.misc.holographic_metrology import surface_area
        return surface_area(mesh)

    def measure_volume(self, mesh):
        """Modeling-app backlog: enclosed volume of a CLOSED mesh (divergence theorem), a [m^3] Quantity. See
        holographic_metrology.volume."""
        from holographic.misc.holographic_metrology import volume
        return volume(mesh)

    def measure_bbox(self, mesh):
        """Modeling-app backlog: the axis-aligned bounding box of a mesh (extents + diagonal as [m] Quantities).
        See holographic_metrology.bounding_box."""
        from holographic.misc.holographic_metrology import bounding_box
        return bounding_box(mesh)

    def measure_distance(self, p, q):
        """Modeling-app backlog: the length between two points, as a dimensioned [m] Quantity (convert with
        .to('ft') etc). See holographic_metrology.distance."""
        from holographic.misc.holographic_metrology import distance
        return distance(p, q)

    def guided_upsample(self, low_color, guide_normal, guide_albedo=None, guide_depth=None, levels=4, sigma_color=2.0):
        """Inverse-rendering ST3: guided (joint-bilateral) super-resolution -- render colour SMALL, then upscale it
        steered by the full-res G-buffer (normal/depth/albedo, which render_channels exposes), so colour edges snap to
        the geometry the cheap render already knows at full res. Reuses the shipped SVGF bilateral. Invents plausible,
        not true, detail (below learned SR). See holographic_superres.guided_upsample."""
        from holographic.rendering.holographic_superres import guided_upsample
        return guided_upsample(low_color, guide_normal, guide_albedo=guide_albedo, guide_depth=guide_depth,
                               levels=levels, sigma_color=sigma_color)

    def synthesize_texture(self, sample, out_h, out_w, psize=24, overlap=6, seed=0, seam="mincut"):
        """Inverse-rendering ST2: grow a larger texture from a small sample by Image Quilting -- lay overlapping
        patches chosen by a patch search (HoloForest recall_k), stitched along min-cut seams. For material synthesis
        and feeding IR1 auto-bump with tileable maps. Patch-copying (can repeat/seam), best for texture/material, not
        free-form restyle. See holographic_texturesynth.synthesize_texture."""
        from holographic.materials_and_texture.holographic_texturesynth import synthesize_texture
        return synthesize_texture(sample, out_h, out_w, psize=psize, overlap=overlap, seed=seed, seam=seam)

    def complete_object(self, archive, front, match_floor=0.85):
        """Inverse-rendering IR11: given a partial FRONT view of an object and an ObjectArchive of complete objects,
        recall the nearest stored WHOLE object (including the unobserved back) by its view fingerprint, or ABSTAIN
        when nothing in the library matches. Retrieval, not hallucination -- the archive's 'recover the whole from a
        partial measurement' move, one dimension up. See holographic_objectarchive.ObjectArchive."""
        return archive.complete_from_front(front, match_floor=match_floor)

    def render_checkerboard(self, sdf, camera, width, height, parity=0, **kw):
        """Inverse-rendering IR13: checkerboard/sparse render -- shade only ~50% of the pixels (a 2x2 pattern) and
        reconstruct the rest as masked recovery (the unshaded pixels are 'damage'; their four cross-neighbours are
        all shaded). Roughly halves the shading cost for a near-full-resolution result. Flip `parity` per frame to
        fill the other half over time. Returns (image, mask). See holographic_checkerboard.render_checkerboard."""
        from holographic.rendering.holographic_checkerboard import render_checkerboard
        return render_checkerboard(sdf, camera, width, height, parity=parity, **kw)

    def upscale(self, image, scale=2.0, sharpness=0.4):
        """Inverse-rendering IR12: FSR1-style spatial upscale -- EASU (edge-adaptive Lanczos with anti-ringing) then
        RCAS (the shipped noise-aware sharpen). Take a low-res render up to display resolution edge-adaptively; beats
        plain bilinear on PSNR and edge sharpness. Reconstructs, cannot invent absent detail. See holographic_fsr."""
        from holographic.rendering.holographic_fsr import fsr_upscale
        return fsr_upscale(image, scale=scale, sharpness=sharpness)

    def render_channels(self, sdf, camera, want=None, width=32, height=32, objects=None, **render_kw):
        """Inverse-rendering IR14: render selectable, separate AOV channels (depth/normal/position/mask G-buffer,
        per-object Cryptomatte mattes), each with its own alpha, for compositing/science/debug. A channel is an
        UNBIND; the scene is a bundle at every level. Default (no selection) = the beauty pass, bit-identical to
        render_sdf. Lighting passes need trace-time accumulation (not in v1). See holographic_renderchannels."""
        from holographic.rendering.holographic_renderchannels import render_channels
        return render_channels(sdf, camera, want=want, width=width, height=height, objects=objects, **render_kw)

    def scene_hypothesis(self, image, k=4):
        """Inverse-rendering IR3: an archetype-level scene READING of an image -- dominant palette, the horizon row
        (sky/ground split), and a coarse sun direction. The perception seed that warm-starts the IR4 loop. A gist,
        not a segmentation (abstain-worthy outside its vocabulary). See holographic_perception.scene_hypothesis."""
        from holographic.agents_and_reasoning.holographic_perception import scene_hypothesis
        return scene_hypothesis(image, k=k)

    def estimate_light_direction(self, image, power=2.0):
        """Inverse-rendering IR3: a COARSE sun-direction estimate (azimuth, elevation) from an image's brightest
        region -- a warm-start cue for IR4 to refine, not a measurement. See holographic_perception."""
        from holographic.agents_and_reasoning.holographic_perception import estimate_light_direction
        return estimate_light_direction(image, power=power)

    def recover_scene(self, sdf, target_img, init_params, accept_threshold=None, **kw):
        """Inverse-rendering IR4 (the headline): analysis-by-synthesis. Given a TARGET image, gradient-free-search
        the camera + sun-direction parameters whose render best matches it (perceptual distance, not MSE), from a
        warm-start guess, with an optional conformal accept/abstain gate. The measurable milestone is self-recovery:
        render a known scene, recover its camera + light within tolerance. See holographic_inverserender."""
        from holographic.rendering.holographic_inverserender import recover_scene
        return recover_scene(sdf, target_img, init_params, accept_threshold=accept_threshold, **kw)

    def compare_images(self, x, y, w_struct=0.5, w_color=0.3, w_edge=0.2):
        """Inverse-rendering IR4: a PERCEPTUAL render-vs-target similarity in [0,1] (1 = identical) -- multi-scale
        SSIM + colour-histogram agreement + edge alignment. Shift/lighting-tolerant, unlike raw pixel MSE. This is
        the compare step of the analysis-by-synthesis loop. See holographic_imagecompare.perceptual_similarity."""
        from holographic.io_and_interop.holographic_imagecompare import perceptual_similarity
        return perceptual_similarity(x, y, w_struct=w_struct, w_color=w_color, w_edge=w_edge)

    def image_distance(self, x, y, **kw):
        """Inverse-rendering IR4: 1 - compare_images -- the objective the analysis-by-synthesis loop MINIMIZES
        (0 = a perfect match). See holographic_imagecompare.perceptual_distance."""
        from holographic.io_and_interop.holographic_imagecompare import perceptual_distance
        return perceptual_distance(x, y, **kw)

    def sharpen_image(self, x, blur=None, sigma=3.0, lam=1.0, iters=60, noise_level=0.0):
        """IMAGE MANIPULATION: deblur / sharpen a signal or image by iterating a deconvolution loop toward the
        deblurred signal (Group G, the sharpen half). `sigma` is the assumed blur width, `lam` the regularisation,
        `iters` the loop count. Returns the sharpened array. See holographic_sharpen.sharpen_loop."""
        from holographic.rendering.holographic_sharpen import sharpen_loop
        return sharpen_loop(x, blur=blur, sigma=sigma, lam=lam, iters=iters, noise_level=noise_level)

    def warp_gather(self, values, positions, query_source_positions):
        """IMAGE/FIELD WARP: resample `values` (defined at `positions`) at `query_source_positions` by BACKWARD
        gather -- the artifact-free way to warp an image/field (every output pixel pulls from where it came from,
        vs forward scatter which leaves holes). Returns the gathered values. See holographic_backwardwarp."""
        from holographic.misc.holographic_backwardwarp import backward_gather
        return backward_gather(values, positions, query_source_positions)

    def splat_points(self, points, camera, width, height, colors=None, radius_px=2.0, intensity=1.0,
                     depth_fade=None, background=(0.0, 0.0, 0.0)):
        """IMAGE GENERATION from geometry: render an (N,3) point cloud to an (image (H,W,3), alpha (H,W)) by
        splatting each point as a soft disc of `radius_px`. The cheap raster path for point clouds / particle
        sims (no marching). `colors` optional per-point rgb; `depth_fade` fades far points. See
        holographic_pointsplat.splat_points."""
        from holographic.rendering.holographic_pointsplat import splat_points
        return splat_points(points, camera, width, height, colors=colors, radius_px=radius_px,
                            intensity=intensity, depth_fade=depth_fade, background=background)

    def slime_solve_maze(self, world, dim=2048, ants=24, rounds=50, decay=0.80, seed=0, q=2.0,
                         use_compass=True, record=False, elite=0.0):
        """SIMULATION: solve a maze with a colony of slime-mold walkers laying pheromone into ONE holographic
        field, then reading the path back -- emergent pathfinding, deterministic given the seed. Returns
        (path, info). See holographic_slime.solve_maze."""
        from holographic.simulation_and_physics.holographic_slime import solve_maze
        return solve_maze(world, dim=dim, ants=ants, rounds=rounds, decay=decay, seed=seed, q=q,
                          use_compass=use_compass, record=record, elite=elite)

    def iridescent_tint(self, thickness_nm=320.0, cos_theta=1.0, n_film=1.33, phase_flip=True):
        """MATERIAL: the iridescent RGB tint of a thin film (soap-bubble / oil-slick sheen) of `thickness_nm` seen
        at angle `cos_theta` (1.0 = head-on). Multiply a surface's reflected colour by this for iridescence; sweeping
        the angle or thickness cycles the tint through the spectrum. n_film 1.33 = soapy water, 1.45 = oil. Returns
        an (...,3) sRGB tint. See holographic_thinfilm.thin_film_tint (iridescent_socket builds the full shader socket)."""
        from holographic.rendering.holographic_thinfilm import thin_film_tint
        return thin_film_tint(thickness_nm, cos_theta, n_film=n_film, phase_flip=phase_flip)


    def compare_image_files(self, path_a, path_b, w_struct=0.5, w_color=0.3, w_edge=0.2):
        """Perceptual similarity in [0,1] (1 = identical) between two images given as FILE PATHS (e.g. two
        rendered PNGs) -- the on-disk companion to compare_images, which takes arrays. This is the call an agent
        makes to check 'did my render change / match the target?' when the images are files on disk. Returns
        {similarity, distance, shape_a, shape_b}.

        PNG IS READ WITH THE STDLIB DECODER, no dependency. This faculty used to open both files with Pillow --
        an unguarded third-party import in a core that promises NumPy/Flask/stdlib/hashlib, sitting in the one
        method whose own docstring calls it the check an agent runs after a render. On a clean install it raised
        ImportError. Other formats still fall back to Pillow, and now say so instead of assuming it is there.

        `b` is resized to `a`'s shape when they differ, by bilinear resample rather than PIL's Lanczos. KEPT
        NEGATIVE: bilinear is softer, so a mismatched-size comparison scores slightly differently than it did
        under Lanczos -- compare like-sized renders if the absolute number matters. Same-size images, which is
        the case an agent actually hits, take no resample at all and are unaffected.
        See holographic_render.load_png / holographic_imagecompare.perceptual_similarity."""
        import numpy as _np
        from holographic.io_and_interop.holographic_imagecompare import perceptual_similarity
        from holographic.rendering.holographic_render import load_png

        def _read(path):
            if str(path).lower().endswith(".png"):
                return load_png(path)                      # stdlib, always available, deterministic
            try:
                from PIL import Image
            except ImportError:
                raise RuntimeError("reading %r needs Pillow (opt-in, like every accelerator): "
                                   "pip install pillow   (or the `images` extra). PNG needs nothing." % path)
            return _np.asarray(Image.open(path).convert("RGB"), float) / 255.0

        a = _read(path_a)
        b = _read(path_b)
        if b.shape[:2] != a.shape[:2]:
            from holographic.rendering.holographic_postfx import resample
            b = _np.asarray(resample(b, float(a.shape[0]) / b.shape[0]), float)
            b = b[:a.shape[0], :a.shape[1]]                # trim the rounding slack so the shapes match exactly
            if b.shape[:2] != a.shape[:2]:                 # ...or pad, if the resample landed short
                pad = ((0, a.shape[0] - b.shape[0]), (0, a.shape[1] - b.shape[1]), (0, 0))
                b = _np.pad(b, pad, mode="edge")
        sim = perceptual_similarity(a, b, w_struct=w_struct, w_color=w_color, w_edge=w_edge)
        return {"similarity": float(sim), "distance": float(1.0 - sim),
                "shape_a": list(a.shape), "shape_b": list(b.shape)}

    def recolor_image(self, image, reference, mode="covariance", strength=1.0):
        """2D EDIT -- grade an image toward another image's COLOUR statistics (a colour-transfer / recolour). `mode`
        'meanstd' matches per-channel mean+std, 'covariance' does full whitening/colouring; `strength` blends
        0->original .. 1->full transfer. Returns an image the same shape as the input. See holographic_colortransfer."""
        from holographic.materials_and_texture.holographic_colortransfer import color_transfer
        return color_transfer(image, reference, mode=mode, strength=strength)

    def blend_images(self, image_a, image_b, steps=21):
        """2D GENERATE -- a crossfade/morph sequence between two images (the midpoint is the 0.5*a+0.5*b double
        exposure). Returns `steps` frames from a to b. See holographic_generate.crossfade_images."""
        from holographic.misc.holographic_generate import crossfade_images
        return crossfade_images(image_a, image_b, steps=steps)

    def image_edges(self, rgb, quantile=0.85):
        """Boolean edge map of an image (classic CV, holographic_vision): Sobel gradient magnitude thresholded at
        `quantile` of its own distribution -- self-calibrating, no magic pixel threshold. Accepts RGB (converted to
        perceptual luma internally) or an already-gray 2-D array. See holographic_vision.edges."""
        import numpy as np
        from holographic.misc.holographic_vision import edges, to_gray
        a = np.asarray(rgb, float)
        return edges(to_gray(a) if a.ndim == 3 else a, quantile=quantile)

    def image_corners(self, rgb, n=12, rel=0.05, min_dist=4):
        """Top-n Harris corners of an image as (x, y) points, greedily spaced at least `min_dist` apart
        (holographic_vision): corners are where the local gradient varies in TWO directions -- the classic
        interest-point detector. Accepts RGB or gray. See holographic_vision.corners."""
        import numpy as np
        from holographic.misc.holographic_vision import corners, to_gray
        a = np.asarray(rgb, float)
        return corners(to_gray(a) if a.ndim == 3 else a, n=n, rel=rel, min_dist=min_dist)

    def image_lines(self, rgb, top=5, quantile=0.85, ntheta=180, nms=10):
        """Dominant straight lines in an image by the classic Hough transform (holographic_vision): every edge
        pixel votes for the (theta, rho) lines through it; the peaks are the lines. Chains the self-calibrating
        edge detector internally, so the input is just the image. Returns the `top` (theta, rho, votes) peaks.
        See holographic_vision.hough_lines."""
        import numpy as np
        from holographic.misc.holographic_vision import edges, hough_lines, to_gray
        a = np.asarray(rgb, float)
        return hough_lines(edges(to_gray(a) if a.ndim == 3 else a, quantile=quantile), ntheta=ntheta, top=top, nms=nms)

    def image_colours(self, rgb, k=4, seed=0, as_float=False):
        """The k most common colours in an image (holographic_vision): k-means++ clustering over a pixel sample --
        the palette / dominant-colour readout. Deterministic per seed. Returns (palette[k,3], weights[k]).
        as_float=False (default) keeps the legacy uint8 0-255 palette; as_float=True returns float 0-1 to match the
        rest of the image ecosystem with no range conversion (recommended for image pipelines). Default is OFF only
        to avoid flipping existing callers. See holographic_vision.dominant_colours."""
        from holographic.misc.holographic_vision import dominant_colours
        return dominant_colours(rgb, k=k, seed=seed, as_float=as_float)

    def segment_image(self, rgb, k=5, seed=0, spatial_weight=0.35, split_components=True, min_fraction=0.006,
                      max_dim=None):
        """DEMUX a photo into per-object REGIONS by colour (+ weak spatial coherence) -- the segmentation front end
        of the photo->3D pipeline. Returns a list of region dicts largest-first: id, mask (H,W bool), area, fraction,
        bbox, centroid, mean_color, shape (circle/rectangle/line/triangle), circularity/extent/aspect. Deterministic;
        splits on APPEARANCE not semantics (a shadow can split a floor) -- the per-region stats are the coarse guess
        the primitive-fit stage refines. Cost scales with pixel count: set max_dim to bound the resolution the sweep
        runs at (the input is box-downsampled to that longest side, segmented, and masks nearest-upsampled back to
        full size with all stats recomputed on the original image) -- the interactive-latency knob, in the engine
        instead of every caller. max_dim=None (default) runs at full resolution, byte-identical to before. See
        holographic_vision.segment_image."""
        from holographic.misc.holographic_vision import segment_image
        return segment_image(rgb, k=k, seed=seed, spatial_weight=spatial_weight,
                             split_components=split_components, min_fraction=min_fraction, max_dim=max_dim)


    def tighten_selection(self, alpha, bbox=None, threshold=0.0):
        """Shrink a rectangular raster selection to its NON-TRANSPARENT content -- the auto-shrink-to-opaque-pixels
        Photoshop/GIMP do so a later rotate/scale pivots about the DRAWING's centre, not the loose marquee's centre
        (a marquee whose centre lands in empty space spins the drawing around a point outside it). Pass the layer
        alpha (H,W in 0..1 or 0..255), an (H,W,4) RGBA image, or a boolean mask, and optionally the dragged marquee
        `bbox`=(r0,c0,r1,c1) inclusive. Returns {empty, bbox, centre, area}: `bbox` is the tight content box and
        `centre` is the (row,col) pivot a transform should use. `empty=True` (blank marquee) means keep the original
        selection rather than collapse it. Non-destructive; deterministic. See holographic_vision.tighten_selection.
        """
        from holographic.misc.holographic_vision import tighten_selection
        return tighten_selection(alpha, bbox=bbox, threshold=threshold)


    def image_signature(self, rgb):
        """One fixed-length feature vector describing an image (holographic_vision.describe): colour histogram +
        edge-orientation histogram + coarse layout, concatenated -- the classic-CV descriptor behind
        image_classes. Two visually similar images get nearby signatures; use it for retrieval, dedup, or as a
        cheap perceptual distance. See holographic_vision.describe."""
        from holographic.misc.holographic_vision import describe
        return describe(rgb)

    def image_classes(self, images, k, seed=0, standardize=False):
        """Cluster a set of images into k visual classes with NO labels (holographic_vision.emergent_classes):
        describe() every image, k-means the descriptors -- classes EMERGE from appearance. Returns
        (labels_per_image, class_centroids). Deterministic per seed. See holographic_vision.emergent_classes."""
        from holographic.misc.holographic_vision import emergent_classes
        return emergent_classes(images, k, seed=seed, standardize=standardize)

    def ascii_view(self, image, width=80, mode="ramp", ansi=None, ramp=None, gamma=1.0,
                   invert=False, cell_aspect=0.5):
        """Render any image to TEXT at `width` characters -- the terminal/log/SSH projection backend
        (holographic_ascii). Modes by detail-per-character: 'ramp' (luminance glyphs), 'edge' (oriented | / - \\
        glyphs on strong gradients), 'braille' (2x4 dots = 8 pixels per char, Bayer-dithered -- the max-detail
        mode), 'half' (2 full-color pixels per char; requires ansi). ansi='256'|'truecolor' colors any mode (the
        256 path uses a baked colour codebook -- ~13x faster than per-cell formatting). ramp='short'|'long'|
        'blocks'|'dots' or a custom glyph string picks the luminance codebook. gamma=2.2 brightens a linear
        render; invert flips the ramp. Deterministic to the byte; fully vectorised (240^2 to 100-wide braille
        ~5 ms). See holographic_ascii.ascii_render."""
        from holographic.rendering.holographic_ascii import ascii_render
        return ascii_render(image, width=width, mode=mode, ansi=ansi, ramp=ramp, gamma=gamma,
                            invert=invert, cell_aspect=cell_aspect)

    def ascii_sdf(self, sdf, width=80, mode="ramp", z=4.0, fov=0.8, camera=None, lit=True,
                  ansi=None, ramp=None, cell_aspect=0.5):
        """Preview a 3-D SDF scene as TEXT -- raymarch + shade + ASCII in one call (holographic_ascii.ascii_sdf),
        the 'see my SDF over SSH' path with no manual render loop. `sdf` is a live SDF, a domain-warped scene, or
        its DSL text. Default camera looks down -z from `z`; pass (origin, forward) as `camera` to override. lit
        adds a lambert term. mode/ansi/ramp as ascii_view. Small by design (a preview); for a full frame use the
        raymarcher and pass its image to ascii_view. See holographic_ascii.ascii_sdf."""
        from holographic.rendering.holographic_ascii import ascii_sdf
        return ascii_sdf(sdf, camera=camera, width=width, mode=mode, z=z, fov=fov, lit=lit,
                         ansi=ansi, ramp=ramp, cell_aspect=cell_aspect)

    def ascii_field(self, field, bounds=(-1.0, 1.0), res=None, width=80, mode="ramp",
                    ansi=None, ramp=None, cell_aspect=0.5):
        """Project a 2-D scalar FIELD sampler straight to TEXT (holographic_ascii.ascii_field) -- composability
        past finished images. `field` is any callable f(P:(N,2))->(N,) (a bake_nd slice, a noise function, a
        heightmap); it is sampled over [bounds]^2, self-normalised, and projected. This is the seam that lets the
        ASCII backend consume the engine's native fields. mode/ansi/ramp as ascii_view. See
        holographic_ascii.ascii_field."""
        from holographic.rendering.holographic_ascii import ascii_field
        return ascii_field(field, bounds=bounds, res=res, width=width, mode=mode,
                           ansi=ansi, ramp=ramp, cell_aspect=cell_aspect)

    def ascii_animate(self, frame, n, width=80, mode="ramp", ansi=None, ramp=None, cell_aspect=0.5, **kw):
        """Render an ASCII ANIMATION to a list of `n` text frames (holographic_ascii.ascii_frames) -- the
        demoscene 'kaleidoscope tunnel in a terminal' as data. `frame` is a callable frame(i, u) or frame(u)
        (u = i/n normalised time) returning what to draw each frame: an image array (-> ascii_render), an SDF
        node or DSL text (-> ascii_sdf, raymarched), or a 2-D field sampler f(P) (-> ascii_field). Returns a list
        of strings (pure, deterministic -- diff them, write a .txt reel, or drive your own loop). For live
        in-terminal playback with timing, call holographic_ascii.ascii_play directly (it does stdout I/O).
        mode/ansi/ramp as ascii_view. See holographic_ascii.ascii_frames."""
        from holographic.rendering.holographic_ascii import ascii_frames
        return ascii_frames(frame, n, width=width, mode=mode, ansi=ansi, ramp=ramp,
                            cell_aspect=cell_aspect, **kw)

    def auto_displace(self, mesh, rgb, amount=0.1, sigma=4.0, min_confidence=0.02):
        """Inverse-rendering IR5: promote an auto-bump height (IR1) from a shading bump to REAL geometry -- move a
        mesh's vertices along their normals by the derived height, but ONLY if the bump-confidence clears a
        (stricter) geometry threshold; otherwise ABSTAIN and return the mesh unchanged. Returns (mesh, info). See
        holographic_autodisplace.auto_displace."""
        from holographic.mesh_and_geometry.holographic_autodisplace import auto_displace
        return auto_displace(mesh, rgb, amount=amount, sigma=sigma, min_confidence=min_confidence)

    def color_transfer(self, img, reference, mode="covariance", strength=1.0, clip=True):
        """Inverse-rendering ST1: grade an image toward a REFERENCE image's colour statistics (Reinhard 2001) --
        the 'match the sunset's mood' knob. mode='meanstd' (per-channel) or 'covariance' (full mean+covariance,
        whitening/colouring); 'mean_std'/'mean-std'/'cov'/'mk' are accepted aliases and an unknown mode raises
        ValueError (it used to be silently ignored). Moves colour, not content. See
        holographic_colortransfer.color_transfer."""
        from holographic.materials_and_texture.holographic_colortransfer import color_transfer
        return color_transfer(img, reference, mode=mode, strength=strength, clip=clip)

    def integrate_normals(self, nmap):
        """Inverse-rendering IR7: integrate a tangent-space normal map into a single-valued, CONSISTENT height
        field by FFT (Frankot-Chellappa) -- the inverse of auto_bump's normal-from-height. Drift-free and
        seamlessly TILEABLE (periodic boundary). See holographic_surfaceint.height_from_normals."""
        from holographic.mesh_and_geometry.holographic_surfaceint import height_from_normals
        return height_from_normals(nmap)

    def auto_bump(self, rgb, strength=2.0, sigma=4.0, abstain_below=0.005):
        """Inverse-rendering IR1: derive a plausible tangent-space normal map (and height) from an albedo image
        alone -- 'auto bump' when no bump/normal map is supplied. Grayscale -> high-pass -> normal, with an honest
        confidence gate that ABSTAINS to flat when there is too little fine detail. Returns a dict with the normal
        map, height, confidence, and whether it abstained. See holographic_autobump.auto_bump."""
        from holographic.mesh_and_geometry.holographic_autobump import auto_bump
        return auto_bump(rgb, strength=strength, sigma=sigma, abstain_below=abstain_below)

    def sampler(self, shape, target, mode="point", radius=1.0, falloff="smooth", weight=None):
        """Modeling-app backlog (capstone): a placeable read-probe -- the read-dual of a FieldEffect. Reads a
        field/material at a point, surface patch, or volume region with a falloff weighting, and handles overlap
        with a labeled bundle. See holographic_sampler.Sampler."""
        from holographic.sampling_and_signal.holographic_sampler import Sampler
        return Sampler(shape, target, mode=mode, radius=radius, falloff=falloff, weight=weight)

    def place_sampler(self, scene, sampler, transform=None, name="Sampler"):
        """Modeling-app backlog (capstone): drop a Sampler into the Scene as an object (handle + transform), so it
        is placed/moved/animated like anything else. See holographic_sampler.place_sampler."""
        from holographic.sampling_and_signal.holographic_sampler import place_sampler
        return place_sampler(scene, sampler, transform=transform, name=name)

    def resolve_override(self, scene, handle, prop, defaults=None, default=None):
        """Modeling-app feature layer: resolve a render property for an object -- its own override, else its
        material's, else the scene defaults, else a bare default (a bound role with fallback). See
        holographic_overrides.resolve."""
        from holographic.misc.holographic_overrides import resolve
        return resolve(scene, handle, prop, defaults=defaults, default=default)

    def set_override(self, scene, handle, prop, value):
        """Modeling-app feature layer: bind a render override on an object (undoable). See
        holographic_overrides.set_override."""
        from holographic.misc.holographic_overrides import set_override
        set_override(scene, handle, prop, value)

    def snapper(self, grid=None, vertices=None, tol=0.25):
        """Modeling-app feature layer: a Snapper that snaps a dragged point to the nearest grid node or vertex
        within a tolerance (snapping = cleanup). See holographic_snap.Snapper."""
        from holographic.caching_and_storage.holographic_snap import Snapper
        return Snapper(grid=grid, vertices=vertices, tol=tol)

    def group_objects(self, scene, handles, name="Group"):
        """Modeling-app feature layer: group objects under a null parent (grouping = a bundle); one undo step.
        See holographic_grouping.group_objects."""
        from holographic.misc.holographic_grouping import group_objects
        return group_objects(scene, handles, name=name)

    def instance(self, scene, source, transform=None, name=None):
        """Modeling-app feature layer: create an instance sharing a source's geometry with its own transform
        (instancing = a bind); editing the source updates all instances. See holographic_grouping.instance."""
        from holographic.misc.holographic_grouping import instance
        return instance(scene, source, transform=transform, name=name)

    # -- GEOMETRY KERNEL faculties (tolerance authority + exact predicates + curve/surface intersection) --------
    def model_tolerance(self, abs_tol=1e-9, rel_tol=1e-12, ang_tol=1e-9):
        """The document's single tolerance authority (K11): the abs/rel/angular tolerances a boolean, a snap, and
        an intersection all consult so they agree on 'equal'. See holographic_geomkernel.ModelTolerance."""
        from holographic.mesh_and_geometry.holographic_geomkernel import ModelTolerance
        return ModelTolerance(abs_tol=abs_tol, rel_tol=rel_tol, ang_tol=ang_tol)

    def orient2d(self, a, b, c):
        """Exact-sign 2D orientation predicate: +1 if c is left of a->b (ccw), -1 right, 0 exactly collinear
        (decided by exact Fraction arithmetic, not a fuzzy epsilon). The primitive under robust intersection.
        See holographic_geomkernel.orient2d."""
        from holographic.mesh_and_geometry.holographic_geomkernel import orient2d
        return orient2d(a, b, c)

    def orient3d(self, a, b, c, d):
        """Exact-sign 3D orientation predicate: +1 if d is above the plane a-b-c, -1 below, 0 exactly coplanar.
        See holographic_geomkernel.orient3d."""
        from holographic.mesh_and_geometry.holographic_geomkernel import orient3d
        return orient3d(a, b, c, d)

    def curve_intersect(self, A, B, tol=None):
        """CURVE-CURVE intersection (K1): all crossings of two polylines A,B as records {point,i,j,t,u} (segment
        indices + in-segment parameters), crossings decided by the exact orient2d so a near-tangency is not
        swallowed. Curves arrive as sampled polylines. See holographic_curveint.intersect_polylines."""
        from holographic.mesh_and_geometry.holographic_curveint import intersect_polylines
        return intersect_polylines(A, B, tol=tol)

    def curve_self_intersect(self, A, tol=None):
        """Self-intersections of one polyline (what an offset curve must clean up). See
        holographic_curveint.self_intersections."""
        from holographic.mesh_and_geometry.holographic_curveint import self_intersections
        return self_intersections(A, tol=tol)

    def surface_intersect(self, f, g, lo, hi, res=24, step=None, tol=None):
        """SURFACE-SURFACE intersection (K2, the keystone): trace the intersection curve(s) of two implicit
        surfaces f=0, g=0 over the box [lo,hi] by a predict-correct FIELD MARCH (tangent = grad f x grad g,
        corrector = Newton projection onto both). Returns a list of (n,3) polylines. Fit a NURBS to a result for
        a trim loop. See holographic_surfint.surface_surface_intersect."""
        from holographic.mesh_and_geometry.holographic_surfint import surface_surface_intersect
        return surface_surface_intersect(f, g, lo, hi, res=res, step=step, tol=tol)

    def trimmed_surface(self, surf_uv, outer, holes=None, u_range=(0.0, 1.0), v_range=(0.0, 1.0)):
        """TRIMMED SURFACE (K3): a surface surf_uv(u,v)->(x,y,z) restricted to trim loops in parameter space (inside
        `outer`, outside `holes`) -- how Rhino represents a trimmed face. .is_inside(u,v), .tessellate(nu,nv),
        .area_fraction(). See holographic_trimsurf.TrimmedSurface."""
        from holographic.mesh_and_geometry.holographic_trimsurf import TrimmedSurface
        return TrimmedSurface(surf_uv, outer, holes=holes, u_range=u_range, v_range=v_range)

    def trim_loop_from_curve(self, surf_uv, curve3d, res=40):
        """Project a 3-D trimming curve (e.g. a K2 SSI polyline) to a (u,v) trim loop -- the K2->K3 bridge. See
        holographic_trimsurf.trim_loop_from_curve."""
        from holographic.mesh_and_geometry.holographic_trimsurf import trim_loop_from_curve
        return trim_loop_from_curve(surf_uv, curve3d, res=res)

    def region_boolean_area(self, A, B, op, res=240):
        """2D REGION BOOLEAN (K4): the area of union/difference/intersection of two closed polygonal regions, by
        exact even-odd membership quadrature (the robust scalar). Use region_membership for the predicate. See
        holographic_region2d.region_boolean_area."""
        from holographic.mesh_and_geometry.holographic_region2d import region_boolean_area
        return region_boolean_area(A, B, op, res=res)

    def region_membership(self, A, B, op):
        """A predicate (x,y)->bool for the boolean op {union,difference,intersection} of two 2D regions -- the
        always-correct membership core. See holographic_region2d.region_membership."""
        from holographic.mesh_and_geometry.holographic_region2d import region_membership
        return region_membership(A, B, op)

    def offset_curve(self, poly, dist, closed=True, tol=None):
        """CURVE OFFSET (K4): a parallel polyline at distance `dist` (positive grows a CCW loop outward), with the
        loops a concave offset folds in removed via self-intersection cleanup. See
        holographic_region2d.offset_polyline."""
        from holographic.mesh_and_geometry.holographic_region2d import offset_polyline
        return offset_polyline(poly, dist, closed=closed, tol=tol)

    def sketch2d(self, tol=None):
        """2D CONSTRAINT SKETCH (K8): a parametric sketch solved by iterated projection. Add points, declare
        constraints (fix/coincident/horizontal/vertical/distance/parallel/perpendicular/point_on_line), then
        .solve(); .dof() reports under/well/over-constrained. See holographic_sketch2d.Sketch2D."""
        from holographic.mesh_and_geometry.holographic_sketch2d import Sketch2D
        return Sketch2D(tol=tol)

    def mesh_to_stl(self, vertices, faces, name="lecore"):
        """CAD EXPORT (K7): an ASCII STL string for a mesh (tris/quads/ngons; per-facet normals from the winding).
        See holographic_cadexport.mesh_to_stl."""
        from holographic.io_and_interop.holographic_cadexport import mesh_to_stl
        return mesh_to_stl(vertices, faces, name=name)

    def polylines_to_dxf(self, polylines, closed=None, layer="0"):
        """CAD EXPORT (K7): a minimal DXF R12 ASCII string for 2-D polylines (POLYLINE/VERTEX; closed loops flagged).
        The 2-D drawing exchange format Rhino/AutoCAD read. See holographic_cadexport.polylines_to_dxf."""
        from holographic.io_and_interop.holographic_cadexport import polylines_to_dxf
        return polylines_to_dxf(polylines, closed=closed, layer=layer)

    def surface_curvature(self, surf_uv, u, v, h=1e-4):
        """SURFACE ANALYSIS (K9): Gaussian, mean, and principal curvatures at (u,v) on a parametric surface
        surf_uv(u,v)->(x,y,z), via the first/second fundamental forms. Returns {gaussian, mean, k1, k2}. See
        holographic_surfanalysis."""
        from holographic.mesh_and_geometry import holographic_surfanalysis as SA
        k1, k2 = SA.principal_curvatures(surf_uv, u, v, h)
        return {"gaussian": SA.gaussian_curvature(surf_uv, u, v, h),
                "mean": SA.mean_curvature(surf_uv, u, v, h), "k1": k1, "k2": k2}

    def draft_angle(self, surf_uv, u, v, pull_dir=(0.0, 0.0, 1.0), flip_normal=False):
        """SURFACE ANALYSIS (K9): the moldability draft angle (deg) at (u,v) for a mold pull direction -- positive
        drafts cleanly, ~0 is a vertical wall, negative is an undercut. Sign follows the surface normal; flip_normal
        for the caller's outward orientation. See holographic_surfanalysis.draft_angle."""
        from holographic.mesh_and_geometry.holographic_surfanalysis import draft_angle
        return draft_angle(surf_uv, u, v, pull_dir=pull_dir, flip_normal=flip_normal)

    def mass_properties(self, mesh, density=1.0):
        """CAD MASS PROPERTIES: volume, surface area, centre of mass, and the full inertia tensor (principal
        moments + axes) of a closed triangle mesh, by exact signed-tetrahedron integration (Tonon 2004
        covariance -- shipped once, correctly, so nobody re-derives negative moments). See
        holographic_meshtools.mass_properties."""
        from holographic.mesh_and_geometry.holographic_meshtools import mass_properties
        return mass_properties(mesh, density=density)

    def mesh_section(self, mesh, plane_point=(0.0, 0.0, 0.0), plane_normal=(0.0, 0.0, 1.0)):
        """EXACT planar CROSS-SECTION of a triangle mesh: polylines + area + perimeter + contour count, from the
        triangle/plane intersections themselves (no rasterising, no field sampling). See
        holographic_meshtools.section."""
        from holographic.mesh_and_geometry.holographic_meshtools import section
        return section(mesh, plane_point=plane_point, plane_normal=plane_normal)

    def draft_report(self, mesh, pull_dir=(0.0, 0.0, 1.0), min_draft_deg=2.0):
        """READ-ONLY draft-angle / MOLDABILITY report for a triangle mesh vs a pull direction: area-weighted
        moldable / parting / undercut fractions + the per-face angle distribution (numbers, not painted faces).
        See holographic_meshtools.draft_report; per-point parametric-surface draft is draft_angle."""
        from holographic.mesh_and_geometry.holographic_meshtools import draft_report
        return draft_report(mesh, pull_dir=pull_dir, min_draft_deg=min_draft_deg)

    def oriented_bbox(self, points, refine_steps=24, refine_span_deg=20.0):
        """Minimal-volume ORIENTED bounding box of a point set (PCA seed + coarse-to-fine rotation refinement,
        with an AABB fallback so it is NEVER worse than the axis-aligned box). Returns center/axes/half_extents/
        volume. See holographic_fitshape.oriented_bbox."""
        from holographic.mesh_and_geometry.holographic_fitshape import oriented_bbox
        return oriented_bbox(points, refine_steps=refine_steps, refine_span_deg=refine_span_deg)

    def terrain_erode(self, height, droplets=2000, steps=30, seed=0, **kw):
        """HYDRAULIC EROSION of a height grid: deterministic droplet simulation that carves drainage channels
        and softens peaks; additive (returns an eroded copy). See holographic_terrain.erode for all knobs."""
        from holographic.mesh_and_geometry.holographic_terrain import erode
        return erode(height, droplets=droplets, steps=steps, seed=seed, **kw)

    def camera_from_vanishing_points(self, vp1, vp2, principal_point):
        """CAMERA CALIBRATION from two vanishing points of orthogonal line families: focal length (pixels) +
        rotation (Caprile-Torre / Hartley-Zisserman). Consumes VP coordinates (from vanishing_point() detection
        or user clicks). See holographic_hazedepth.camera_from_vanishing_points."""
        from holographic.rendering.holographic_hazedepth import camera_from_vanishing_points
        return camera_from_vanishing_points(vp1, vp2, principal_point)

    def c_batch_eval(self, kernel, arrays, dtype="f64", opt="fast"):
        """Native BATCH KERNEL via the system C COMPILER -- the fallback twin of zig_batch_eval for containers
        with cc/gcc/clang but no Zig. Same emitted IR, same SoA harness, content-addressed cache; f64 is
        bit-identical to the Python kernel. Refuses loudly when no compiler exists. See holographic_ccrun."""
        from holographic.io_and_interop.holographic_ccrun import CKernel
        return CKernel(kernel, dtype=dtype, opt=opt)(*arrays)

    def snap_to_midpoints(self, point, vertices, edges, max_dist=None):
        """OBJECT SNAP (K10): snap a point to the nearest EDGE MIDPOINT ({edge, position, distance}). See
        holographic_snap.snap_to_midpoints."""
        from holographic.mesh_and_geometry.holographic_snap import snap_to_midpoints
        return snap_to_midpoints(point, vertices, edges, max_dist=max_dist)

    def snap_to_intersections(self, point, polylines, max_dist=None, tol=None):
        """OBJECT SNAP (K10): snap a point to the nearest INTERSECTION of 2-D polylines ({position, distance}),
        crossings found by the robust curve intersector. See holographic_snap.snap_to_intersections."""
        from holographic.mesh_and_geometry.holographic_snap import snap_to_intersections
        return snap_to_intersections(point, polylines, max_dist=max_dist, tol=tol)

    def fillet_union(self, f, g, r):
        """EDGE FILLET (K5): the union of two implicit surfaces f,g with the convex crease rounded to an EXACT
        constant radius r (iq opUnionRound) -- a true radius-r circular arc, unlike smooth_union's soft blend.
        Returns an SDF callable that raymarches/meshes/emits. See holographic_fillet.fillet_union."""
        from holographic.mesh_and_geometry.holographic_fillet import fillet_union
        return fillet_union(f, g, r)

    def fillet_intersection(self, f, g, r):
        """EDGE FILLET (K5): the intersection of f,g with the concave (pocket) crease rounded to radius r
        (opIntersectionRound). Returns an SDF callable. See holographic_fillet.fillet_intersection."""
        from holographic.mesh_and_geometry.holographic_fillet import fillet_intersection
        return fillet_intersection(f, g, r)

    def fillet_difference(self, f, g, r):
        """EDGE FILLET (K5): f minus g with the resulting edge rounded to radius r. Returns an SDF callable. See
        holographic_fillet.fillet_difference."""
        from holographic.mesh_and_geometry.holographic_fillet import fillet_difference
        return fillet_difference(f, g, r)

    def chamfer_union(self, f, g, r):
        """EDGE CHAMFER (K5): the union of f,g with a flat 45-degree chamfer of size r at the crease (the straight-
        bevel alternative to a fillet). Returns an SDF callable. See holographic_fillet.chamfer_union."""
        from holographic.mesh_and_geometry.holographic_fillet import chamfer_union
        return chamfer_union(f, g, r)

    def brep_box(self, lo=(-1.0, -1.0, -1.0), hi=(1.0, 1.0, 1.0)):
        """B-REP (K6): construct a valid closed cube boundary representation (8 verts, 12 edges, 6 faces, genus 0).
        The canonical smallest valid solid. See holographic_brep.box_brep."""
        from holographic.mesh_and_geometry.holographic_brep import box_brep
        return box_brep(lo=lo, hi=hi)

    def brep_validate(self, brep, expected_genus=0):
        """B-REP (K6): validity report for a boundary representation -- {closed_manifold, V,E,F,R,S, genus,
        euler_ok} via the Euler-Poincare law V-E+F-R = 2(S-H). See holographic_brep.Brep.validate."""
        return brep.validate(expected_genus=expected_genus)

    def brep_from_faces(self, vertices, faces, shells=None):
        """B-REP (K6): build a boundary representation from vertices + BFace loops (topology derived and checkable).
        See holographic_brep.Brep."""
        from holographic.mesh_and_geometry.holographic_brep import Brep
        return Brep(vertices, faces, shells=shells)

    def node_registry(self):
        """NODE EDITOR backend: the default node-type palette wired to real leCore compute (SDF primitives + CSG +
        K5 fillet, texture const/mix, scalar). Register more types on it. See holographic_nodegraph.default_registry."""
        from holographic.scene_and_pipeline.holographic_nodegraph import default_registry
        return default_registry()

    def node_graph(self, registry=None):
        """NODE EDITOR backend: a heterogeneous typed node graph an editor binds to -- add nodes, connect sockets
        (type-checked + cycle-checked), evaluate (topological, memoized), set_param (dirty-propagating), and
        to_dict/from_dict serialize. Delegates each node's compute to the existing subsystem. Pass a registry or
        the default is used. See holographic_nodegraph.NodeGraph."""
        from holographic.scene_and_pipeline.holographic_nodegraph import NodeGraph, default_registry
        return NodeGraph(registry if registry is not None else default_registry())

    def point_in_brep(self, brep, points, thresh=0.5):
        """B-REP MEMBERSHIP (toward K6 booleans): is each point inside the solid? Delegates to the generalized
        winding number of the B-rep triangulated boundary. Returns a boolean array. See
        holographic_brepbool.point_in_brep."""
        from holographic.mesh_and_geometry.holographic_brepbool import point_in_brep
        return point_in_brep(brep, points, mind=self, thresh=thresh)

    def brep_boolean_faces(self, brep_a, brep_b, op):
        """B-REP BOOLEAN CLASSIFICATION (toward K6 booleans): which whole FACES of A survive a boolean with B
        (keep outside-B faces for union/difference, inside-B for intersection); faces that STRADDLE the B boundary
        are flagged as needing a K2-SSI split. Returns {keep, straddle}. See
        holographic_brepbool.brep_boolean_faces."""
        from holographic.mesh_and_geometry.holographic_brepbool import brep_boolean_faces
        return brep_boolean_faces(brep_a, brep_b, op, mind=self)

    def brep_boolean(self, brep_a, brep_b, op, res=48, bounds=None, analytic=False):
        """The FINISHED B-REP BOOLEAN (union/difference/intersection): the SSI-driven re-stitch that turns two
        solids into one watertight B-rep. Routes the two solids through the SDF (the K2 surface-surface intersection
        seam + field combine + marching, reusing route_csg), wraps the watertight result as a B-rep, and VALIDATES
        it with K6 (closed 2-manifold, Euler, volume vs inclusion-exclusion; see result._boolean_report). analytic=True recovers POLYGONAL faces (merge coplanar triangles). Kept
        negative: result faces are the marching triangulation, not the inputs' analytic faces (that is the
        refinement); resolution is the grid's. See holographic_brepbool.brep_boolean."""
        from holographic.mesh_and_geometry.holographic_brepbool import brep_boolean
        return brep_boolean(brep_a, brep_b, op, res=res, bounds=bounds, analytic=analytic)

    def camera_controller(self, eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
        """Modeling-app feature layer: a viewport camera controller -- orbit/pan/dolly/zoom/frame around a target.
        See holographic_camera.CameraController."""
        from holographic.rendering.holographic_camera import CameraController
        return CameraController(eye=eye, target=target, up=up)

    def selection(self, scene):
        """Modeling-app feature layer: a Selection helper bound to a Scene -- query objects into a set of handles,
        save named sets, do set algebra (union/intersect/minus/invert), and push the current selection. See
        holographic_scene_query.Selection."""
        from holographic.agents_and_reasoning.holographic_scene_query import Selection
        return Selection(scene)

    def select_objects(self, scene, **predicates):
        """Modeling-app feature layer: select object handles from a Scene by exact predicates (name/material/tag/
        substring/where). For semantic 'select the metal-ish parts' with confidence, use
        holographic_scene_query.select_fuzzy. See holographic_scene_query.select."""
        from holographic.agents_and_reasoning.holographic_scene_query import select
        return select(scene, **predicates)

    def look_at(self, eye, target, up=(0.0, 1.0, 0.0)):
        """Modeling-app backlog (item G): an OpenGL view matrix for a camera at `eye` looking at `target`. See
        holographic_transform.look_at."""
        from holographic.misc.holographic_transform import look_at
        return look_at(eye, target, up)

    def decompose_transform(self, M):
        """Modeling-app backlog (item G): split a 4x4 transform into (translate, rotation quaternion, scale) --
        what a move/rotate/scale gizmo reads off a matrix. See holographic_transform.decompose."""
        from holographic.misc.holographic_transform import decompose
        return decompose(M)

    def cancel_token(self):
        """Modeling-app backlog (item F): a cooperative CancelToken to pass as should_stop= to a long render/sim,
        so it can be stopped mid-run and return a partial result. See holographic_cancel.CancelToken."""
        from holographic.misc.holographic_cancel import CancelToken
        return CancelToken()

    def modifier_stack(self, base):
        """Modeling-app backlog (item C): a per-object MODIFIER STACK + dependency graph over any payload (mesh /
        field / vector) -- an ordered, non-destructive op chain with stable handles that re-evaluates O(change)
        (only downstream of a changed parameter). See holographic_modifier.ModifierStack."""
        from holographic.misc.holographic_modifier import ModifierStack
        return ModifierStack(base)

    def new_scene(self, dim=None, seed=0):
        """Modeling-app backlog (item 0): a fresh canonical Scene document -- the single source of truth a modeling
        app is built around (a table of object records + hierarchy, owning selection and undo history, firing
        change events, with STABLE identity handles that survive edits). See holographic_scene_doc.Scene."""
        from holographic.scene_and_pipeline.holographic_scene_doc import Scene
        return Scene(dim=dim if dim is not None else self.dim, seed=seed)

    def scene_info(self, scene, verbose=True):
        """WHAT IS IN THIS SCENE -- the first call to make, before adding to it or rendering it.

        The document could be built (new_scene/add) and rendered (render_scene_document) and could NOT be
        read: an agent that added four objects had no way to confirm it, recall what it named them, or spot
        a mistake before paying for a trace. Returns JSON-safe types only, because this crosses /invoke.

        {n_objects, empty, objects[handle,name,geometry,material,position,scale,rotated,parent,tags],
         cameras, lights, selection, materials, problems}

        `problems` is a PRE-FLIGHT check, and it is why the call is worth making rather than a nicety. In
        milliseconds it catches the three failures that otherwise cost minutes or go unnoticed entirely: a
        material name absent from the library (which raises at RENDER time, after the whole scene is built);
        an object with no geometry; and a ROTATED transform, which scene_to_render silently drops -- so the
        picture disagrees with the document and nothing says so.

        KEPT NEGATIVE: no bounding box. An SDF is a function, not an extent (a plane is infinite), so the
        honest answer is position + geometry kind. Mesh the object first if you need a real extent.
        See holographic_scene_doc.scene_info."""
        from holographic.scene_and_pipeline.holographic_scene_doc import scene_info
        return scene_info(scene, verbose=verbose)

    # ---- SCENE MUTATION over a faculty surface (J-3D-24) -------------------------------------------------
    # WHY THESE EXIST AND ARE NOT "just call scene.add()". The Scene document's whole mutation API lives on
    # the OBJECT (scene.add / .edit / .remove / .undo), and object methods are invisible to GET /tools and
    # uncallable by POST /invoke. MEASURED: with object handles working, an HTTP agent could mint a Scene,
    # parse three SDFs, and then had NO WAY TO PUT ONE IN THE OTHER -- the authoring path dead-ended one
    # step past "new_scene". Four thin delegators, ONE catalog entry: four registrations would cost four
    # times the catalog budget (already at 80% of the read cap) for one workflow.

    def scene_add(self, scene, name=None, geometry=None, material=None, transform=None,
                  tags=None, params=None, parent=None):
        """Add an object to a Scene document and return its STABLE handle (survives every later edit).

        The handle is what selections, materials and edits refer to -- keep it. Validation is deliberately
        NOT done here: a half-built scene mid-edit is normal, so a bad material is reported by scene_info's
        pre-flight rather than refused at the point of the add. See holographic_scene_doc.Scene.add."""
        return scene.add(name=name, geometry=geometry, material=material, transform=transform,
                         tags=tags, params=params, parent=parent)

    def scene_edit(self, scene, handle, **changes):
        """Change an object's fields in place (name/transform/geometry/material/tags/params).

        THE HANDLE DOES NOT CHANGE -- identity survives the edit, which is the keystone that lets a
        selection or a material assignment keep pointing at the object. Records an undo entry and fires a
        change event, both for free. See holographic_scene_doc.Scene.edit."""
        return scene.edit(handle, **changes)

    def scene_remove(self, scene, handle):
        """Remove an object from the document. Undoable like any other edit. See Scene.remove."""
        return scene.remove(handle)

    def place(self, scene, handle, position=None, rotation=None, scale=None, degrees=True):
        """MOVE / ROTATE / SCALE an object -- the transform verb, instead of hand-building a 4x4.

        Every argument is optional and each REPLACES that component, leaving the others as they are, so
        `place(s, h, rotation=(0, 45, 0))` turns an object without also moving it back to the origin.
          position  (x, y, z) world position.
          rotation  (rx, ry, rz) Euler angles applied X then Y then Z, degrees by default -- pass
                    degrees=False for radians. A (3, 3) matrix or an (axis, angle) pair also works.
          scale     a single number. UNIFORM ONLY, and that is a real limit, not laziness: a non-uniform
                    scale breaks the distance-field property an SDF sphere-trace depends on, so a
                    (2, 1, 1) stretch would make the tracer overshoot and punch through surfaces.

        Records one undo entry and fires one change event, like any other scene edit.

        IMPORTANT AND EASY TO TRIP OVER: a rotation written here is only RENDERED when you pass
        affine=True to render_scene_document (or render_preview). The default drops it, because turning
        that on moves the picture of every existing scene that has a rotated object in it and shipped
        output does not move without an explicit decision. scene_info's pre-flight says so per object."""
        import numpy as _np
        obj = scene.get(handle)
        T = _np.asarray(obj.transform, float).copy()
        if T.shape != (4, 4):
            T = _np.eye(4)
        lengths = _np.linalg.norm(T[:3, :3], axis=0)
        cur_scale = float(_np.mean(lengths)) if _np.all(lengths > 1e-9) else 1.0
        R = (T[:3, :3] / lengths) if _np.all(lengths > 1e-9) else _np.eye(3)

        if rotation is not None:
            R = self._rotation_matrix(rotation, degrees=degrees)
        if scale is not None:
            cur_scale = float(scale)
        T[:3, :3] = R * cur_scale
        if position is not None:
            T[:3, 3] = _np.asarray(position, float)
        return scene.edit(handle, transform=T)

    @staticmethod
    def _rotation_matrix(rotation, degrees=True):
        """(rx, ry, rz) Euler angles, an (axis, angle) pair, or a 3x3 matrix -> a 3x3 rotation matrix.

        Three accepted spellings because callers genuinely arrive with all three: a person says '45 degrees
        about Y', a tool hands over an axis and an angle, and a file format stores a matrix. Rejecting two
        of them would just push the conversion into every caller."""
        import numpy as _np
        r = _np.asarray(rotation[0], float) if (isinstance(rotation, (tuple, list)) and len(rotation) == 2
                                                and _np.size(rotation[0]) == 3
                                                and _np.size(rotation[1]) == 1) else None
        if r is not None:                                        # (axis, angle)
            axis = r / max(_np.linalg.norm(r), 1e-12)
            ang = float(rotation[1]) * (_np.pi / 180.0 if degrees else 1.0)
            K = _np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            return _np.eye(3) + _np.sin(ang) * K + (1 - _np.cos(ang)) * (K @ K)
        arr = _np.asarray(rotation, float)
        if arr.shape == (3, 3):
            return arr
        rx, ry, rz = (arr.ravel() * (_np.pi / 180.0 if degrees else 1.0))
        cx, sx, cy, sy, cz, sz = (_np.cos(rx), _np.sin(rx), _np.cos(ry),
                                  _np.sin(ry), _np.cos(rz), _np.sin(rz))
        Rx = _np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = _np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = _np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx                                      # X, then Y, then Z

    def scene_undo(self, scene, redo=False):
        """Undo (or with redo=True, re-apply) the last scene edit. Returns True if anything moved.

        The document owns its own history, so this works across every tool that edited it -- that is the
        point of having one source of truth rather than a per-tool copy. See Scene.undo / Scene.redo."""
        return scene.redo() if redo else scene.undo()

    def scene_set_texture(self, scene, handle, texture, scale=4.0, seed=0, colors=None, **params):
        """Texture a Scene-document object BY NAME -- 'wood', 'marble', 'checker', an (H,W,3) image, or None
        to remove. JSON-safe end to end, which is the entire reason this exists.

        THE MECHANISM WAS ALREADY THERE AND A REMOTE AGENT COULD NOT REACH IT. scene_to_render honours an
        `albedo_socket` override -- a callable f(P (M,3))->rgb sampled at world hit points -- and
        proc_texture() builds exactly that callable from a name. Verified live before writing this: setting
        a socket moves the render by 0.094 mean abs. But a CALLABLE cannot cross POST /invoke, so over HTTP
        texturing was impossible even though every part of it worked in-process -- the same
        reachable-in-process/dead-at-the-boundary failure this arc keeps finding, one layer down. This
        faculty takes JSON (a texture NAME + numbers, or a plain nested-list image) and builds the callable
        on the server side, where callables are allowed to live.

        `texture`: one of proc_texture's names (noise, fbm, white, voronoi, musgrave, wave, marble, wood,
        brick, magic, checker, stripes, gradient, dots), OR an (H,W,3) array/nested list (mapped by world
        XZ -- a floor decal / ground texture projection), OR None to remove the texture.
        `colors`: optional (rgb_low, rgb_high) pair; a scalar field lerps between them, so 'wood' can be
        oak-coloured rather than greyscale. Extra **params go to proc_texture (kind=, octaves=, ...).

        The texture is SOLID (evaluated at 3-D world points), so it carves through the object like real wood
        grain rather than wallpapering the surface -- no UVs required, which for SDF objects is the honest
        choice, since an SDF has no intrinsic parameterisation to unwrap.

        KEPT NEGATIVES: albedo only -- roughness/metallic stay the library material's (a full material-
        socket system is its own design, not a texture patch). Image mapping is world-XZ planar only:
        triplanar needs the surface NORMAL, and the socket contract is f(P) without normals -- extending
        that contract touches every socket consumer and wants its own item. And the override participates in
        undo like any other edit, because it goes through set_override rather than poking the record."""
        import numpy as np

        if texture is None:
            return self.set_override(scene, handle, "albedo_socket", None)
        if isinstance(texture, str):
            from holographic.materials_and_texture.holographic_proctex import proc_texture
            # not every texture takes every knob -- checker has no seed, gradient no octaves. Retrying
            # without `seed` beats making every caller memorise which of 14 names is deterministic-by-nature;
            # any OTHER unexpected keyword still raises, because a silently dropped octaves= is a lie.
            try:
                field = proc_texture(texture, scale=scale, seed=seed, **params)
            except TypeError:
                field = proc_texture(texture, scale=scale, **params)

            if colors is not None:
                lo = np.asarray(colors[0], float)
                hi = np.asarray(colors[1], float)

                def socket(P, _f=field, _lo=lo, _hi=hi):
                    v = np.asarray(_f(np.atleast_2d(P)), float)
                    if v.ndim == 1:                              # scalar field -> lerp the two colours
                        return _lo + np.clip(v, 0.0, 1.0)[:, None] * (_hi - _lo)
                    return v                                     # already rgb: colours ignored, field wins
            else:
                def socket(P, _f=field):
                    v = np.asarray(_f(np.atleast_2d(P)), float)
                    return np.repeat(v[:, None], 3, axis=1) if v.ndim == 1 else v
        else:
            img = np.asarray(texture, float)
            if img.ndim != 3 or img.shape[2] < 3:
                raise ValueError("an image texture must be (H,W,3); got %s -- for a named procedural "
                                 "texture pass a string like 'wood'" % (img.shape,))
            from holographic.materials_and_texture.holographic_proctex import sample_image

            def socket(P, _img=img[..., :3], _s=float(scale)):
                P = np.atleast_2d(P)
                uv = (P[:, [0, 2]] / _s) % 1.0                   # world XZ, tiled every `scale` units
                return np.asarray(sample_image(_img, uv), float)

        return self.set_override(scene, handle, "albedo_socket", socket)

    def scatter_to_grid(self, points, values, shape, kernel="bilinear", periodic=False):
        """The shared kernel SCATTER = a BUNDLE: deposit each point's value onto a grid through a kernel (bilinear
        or B-spline) -- the superposition that MPM's P2G, a fluid deposit, and a splat all are. `points` (N,D) in
        grid-cell units, `values` (N,) or (N,C). See holographic_transfer.scatter."""
        from holographic.misc.holographic_transfer import scatter
        return scatter(points, values, shape, kernel=kernel, periodic=periodic)

    def gather_from_grid(self, field, points, kernel="bilinear", periodic=False):
        """The shared kernel GATHER = the READOUT: read a grid back at each point through the same kernel -- the
        adjoint of scatter, and what MPM's G2P, field sampling, and a texture lookup all are. See
        holographic_transfer.gather."""
        from holographic.misc.holographic_transfer import gather
        return gather(field, points, kernel=kernel, periodic=periodic)

    def snow_mpm(self, grid=48, dx=1.0, E=140.0, nu=0.2, gravity=9.81, seed=0):
        """Physics backlog (#8B, rung 4): a SNOW solver by the Material Point Method (Stomakhin 2013). Seed it
        (.seed_block) and .run(). Thinking holographically: its P2G scatter IS bundling (the grid is a
        superposition of kernel-weighted particle contributions -- verified equal to a bundle of splats) and G2P is
        the readout; only the elasto-plastic grid update is grid-native. See holographic_mpm.MPMSnow."""
        from holographic.simulation_and_physics.holographic_mpm import MPMSnow
        return MPMSnow(grid=grid, dx=dx, E=E, nu=nu, gravity=gravity, seed=seed)

    def simulation(self, solver, step_fn, field_fn, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0), name="sim"):
        """The SIMULATION SCAFFOLD (R9): wrap ANY time-stepped solver in ONE step loop, without flattening the
        solvers into one (that would destroy the differences that make each correct). `step_fn(solver, dt)` calls
        the solver's REAL step; `field_fn(solver)` returns its current grid. The returned Simulation gives it a
        uniform step(dt) / run(steps, dt), exposes its field for the volumetric renderer, and plugs into the render
        Pipeline's sim stage. Factories exist for the common solvers -- see Simulation.for_fluid / for_automaton.

        This returns a LIVE object (it holds the solver and a step adapter), so it is for IN-PROCESS use. Over an
        HTTP /invoke boundary use `run_simulation`, the stateless twin. See holographic_simulationhome.Simulation."""
        from holographic.misc.holographic_simulationhome import Simulation
        return Simulation(solver, step_fn, field_fn, lo=lo, hi=hi, name=name)

    def run_simulation(self, kind, steps, dt=1.0 / 60.0, grid=48, seed=0, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0),
                       **solver_kwargs):
        """Build a known solver, run it `steps` times through the shared loop, and return its final field grid --
        all in one call, from plain arguments to a plain array. The stateless twin of `simulation`, callable with
        nothing but JSON.

        `kind` is a registered strategy: 'fluid' (StableFluid, advect+project, a 3D `grid`-cubed smoke box seeded
        with a density blob and a little upward flow) or 'automaton' (HyperCA reaction-diffusion, a genuinely
        different algorithm). Returns the field the renderer would draw. `run_simulation('fluid', 30)` steps a fresh
        fluid 30 times and hands back its density grid.

        WHY A TWIN: `simulation` returns a live solver+adapter that does not survive JSON serialization -- the same
        reason `gather_samples` exists beside the bake/gather pair. This re-builds the solver every call (no reuse),
        so drive a long simulation in process; this is for one-shot agent-facing use. See holographic_simulationhome."""
        from holographic.misc.holographic_simulationhome import Simulation
        g = int(grid)
        k = str(kind).lower()
        if k == "fluid":
            from holographic.simulation_and_physics.holographic_fluid import StableFluid
            fluid = StableFluid((g, g, g), **solver_kwargs)
            lo_i = g // 3
            fluid.density[lo_i:lo_i + max(g // 4, 1), 1:max(g // 3, 2), lo_i:lo_i + max(g // 4, 1)] = 1.0
            fluid.vel[1, :, :max(g // 6, 1), :] = 1.0                # a little upward flow so something happens
            sim = Simulation.for_fluid(fluid, lo=lo, hi=hi)
        elif k == "automaton":
            from holographic.misc.holographic_automaton import HyperCA
            sim = Simulation.for_automaton(HyperCA(size=g, seed=seed, **solver_kwargs), lo=lo, hi=hi)
        else:
            from holographic.misc.holographic_simulationhome import known_solver_strategies
            raise ValueError("run_simulation: unknown kind %r; known: %s (or use mind.simulation() with your own "
                             "step/field closures)" % (kind, [x.replace("for_", "") for x in known_solver_strategies()]))
        return sim.run(int(steps), dt).grid()

    def simulate_snow(self, cx=24, cy=12, w=10, h=8, n=400, grid=48, gravity=9.81, dt=2e-3, steps=600, seed=0):
        """Physics backlog (#8B): seed a snow block and run it -- it falls, piles, and compresses plastically.
        Returns the settled MPMSnow. See holographic_mpm.MPMSnow."""
        from holographic.simulation_and_physics.holographic_mpm import MPMSnow
        snow = MPMSnow(grid=grid, gravity=gravity, seed=seed).seed_block(cx=cx, cy=cy, w=w, h=h, n=n)
        return snow.run(dt=dt, steps=steps)

    def free_surface(self, g=9.81, ground=0.0, damping=0.3):
        """Physics backlog (#8, rung 4): the OVERTURNING free-surface solver -- particles that can fold the water
        surface over itself (a breaking wave), which a height field fundamentally cannot. Seed it and call
        .advance(). See holographic_freesurface.FreeSurface."""
        from holographic.mesh_and_geometry.holographic_freesurface import FreeSurface
        return FreeSurface(g=g, ground=ground, damping=damping)

    def break_wave(self, length=10.0, n=40, crest_speed=8.0, phase_speed=3.0, height=4.0, dt=0.05, steps=20):
        """Physics backlog (#8): set up and run a PLUNGING BREAKER -- a crest whose tip outruns the wave, throwing
        it forward until the surface folds (overturns) into a multi-valued sheet. Returns the FreeSurface mid-plunge
        (query .is_overturning() / .is_multivalued()). See holographic_freesurface.seed_breaking_crest."""
        from holographic.mesh_and_geometry.holographic_freesurface import FreeSurface, seed_breaking_crest
        fs = FreeSurface()
        seed_breaking_crest(fs, length=length, n=n, crest_speed=crest_speed, phase_speed=phase_speed, height=height)
        fs.advance(dt, steps=steps)
        return fs

    def grow_ice(self, shape=(81, 81), eta=1.0, steps=200, seed=0):
        """Physics backlog (#7): grow an ICE / frost dendrite by diffusion-limited branching (the dielectric-
        breakdown model) -- a cluster racing into the steepest gradient of a Laplace field, branching. Returns a
        DielectricBreakdown; read its .cluster mask. See holographic_dendrite.ice_dendrite."""
        from holographic.misc.holographic_dendrite import ice_dendrite
        return ice_dendrite(shape=shape, eta=eta, steps=steps, seed=seed)

    def grow_lightning(self, shape=(81, 81), eta=3.0, steps=120, seed=0):
        """Physics backlog (#7): grow a LIGHTNING bolt -- the SAME diffusion-limited branching engine as the ice
        dendrite (N11: build once, get frost and bolts), only the seed (the cloud) and the source (the ground it
        reaches toward) differ. See holographic_dendrite.lightning."""
        from holographic.misc.holographic_dendrite import lightning
        return lightning(shape=shape, eta=eta, steps=steps, seed=seed)

    def dielectric_breakdown(self, shape, eta=1.0, seed=0):
        """Physics backlog (#7): the raw diffusion-limited branching engine (Niemeyer-Pietronero-Wiesmann). Seed
        it (seed_point / seed_line), set a source boundary (set_source_border), and grow(). eta tunes the shape
        (bushy -> fractal -> stringy). See holographic_dendrite.DielectricBreakdown."""
        from holographic.misc.holographic_dendrite import DielectricBreakdown
        return DielectricBreakdown(shape, eta=eta, seed=seed)

    def lorentz_force(self, q, E, v, B):
        """Physics backlog (#6): the Lorentz force F = q(E + v x B) on a charge q moving at v through fields E, B.
        See holographic_em.lorentz_force."""
        from holographic.simulation_and_physics.holographic_em import lorentz_force
        return lorentz_force(q, E, v, B)

    def push_charge(self, pos, vel, q, m, E, B, dt, steps):
        """Physics backlog (#6): integrate a charged particle through uniform fields E, B with the Boris pusher
        (energy-conserving) -- a cyclotron orbit in a magnetic field, an E-cross-B drift in crossed fields.
        Returns (trajectory, final_velocity). See holographic_em.push_particle."""
        from holographic.simulation_and_physics.holographic_em import push_particle
        return push_particle(pos, vel, q, m, E, B, dt, steps)

    def maxwell_field(self, n, dx=1.0, eps=1.0, mu=1.0):
        """Physics backlog (#6): a 1-D coupled Maxwell field (Yee/FDTD) -- Ez and Hy feed each other so a pulse
        propagates at c = 1/sqrt(mu*eps). Set .Ez, call .step(). This is the genuine E<->B coupling the spectral
        backbone's single-component em_field doesn't have. See holographic_em.Maxwell1D."""
        from holographic.simulation_and_physics.holographic_em import Maxwell1D
        return Maxwell1D(n, dx=dx, eps=eps, mu=mu)

    def plan_waves(self, height, depth=None, obstacles=None, dx=1.0, tile=8):
        """Physics backlog (#5, the AdaptiveSolver): the DECISION LAYER for the ocean stack -- per tile, pick the
        wave method (fft_ocean / wave_packets / shallow_water / free_surface) from the local regime and say WHY,
        exactly as plan_render picks bake/analytic/trace. No solving; the plan is inspectable before running, and
        deterministic (breaking > shallow > obstacle > open). Pair with solve_waves. See
        holographic_waveadaptive.plan_waves / plan_cost."""
        from holographic.simulation_and_physics.holographic_waveadaptive import plan_waves
        return plan_waves(height, depth=depth, obstacles=obstacles, dx=dx, tile=tile)

    def solve_waves(self, plan, field, dt=1.0, methods=None, halo=2):
        """Physics backlog (#5): EXECUTE a wave plan -- run each tile's chosen method on the shared surface field
        and blend the tile borders (overlap-add, no seam). The dear grid solver runs only where the plan marked a
        breaking tile; the cheap spectral path runs everywhere else. See holographic_waveadaptive.solve_waves."""
        from holographic.simulation_and_physics.holographic_waveadaptive import solve_waves
        return solve_waves(plan, field, dt=dt, methods=methods, halo=halo)

    def wave_packets(self, size=64.0, g=9.81, envelope=6.0, seed=0):
        """Physics backlog (N8): a water surface as localized WAVE PACKETS -- each a Gaussian-enveloped wave train
        that lives at a place, so unlike the global FFT ocean it can REFLECT off walls, SHOAL over depth changes,
        and diffract. A packet is a role-bound record and the surface is a bundle (content-addressable). Add
        packets with .add_packet, advance with .advance, read the surface with .render. See
        holographic_wavepacket.WavePacketField."""
        from holographic.simulation_and_physics.holographic_wavepacket import WavePacketField
        return WavePacketField(size=size, g=g, envelope=envelope, seed=seed)

    def spectral_pde(self, field, velocity=None, order="parabolic", rate=None, omega=None, dx=1.0):
        """Physics backbone (Part 3 #1): a linear field advanced in FOURIER space by a per-frequency transfer --
        (named spectral_pde to avoid colliding with the fractal-volume spectral_field synthesizer above.)
        'advancing time is one bind, any t in closed form.' order='parabolic' (diffusion, decay rate(|k|)) or
        'hyperbolic' (waves, oscillation omega(|k|), carries velocity). Superposition is add_source (bundle); a
        calibrated trigger_mask fires where a potential crosses threshold. See holographic_spectralfield."""
        from holographic.sampling_and_signal.holographic_spectralfield import SpectralField
        return SpectralField(field, velocity=velocity, order=order, rate=rate, omega=omega, dx=dx)

    def spectral_diffusion(self, field, D, dx=1.0):
        """A diffusion/heat/gas field as a SpectralField: rate(|k|) = -D|k|^2, closed-form any t. Beats the grid
        diffuse_heat baseline (machine-precision exact in one eval vs accumulated step error). See
        holographic_spectralfield.diffusion_field."""
        from holographic.sampling_and_signal.holographic_spectralfield import diffusion_field
        return diffusion_field(field, D, dx=dx)

    def spectral_wave(self, field, velocity=None, c=1.0, dx=1.0):
        """A wave/acoustic/EM(vacuum) field as a SpectralField: omega(|k|) = c|k|, a pulse propagates at speed c;
        closed-form any t. See holographic_spectralfield.wave_field."""
        from holographic.sampling_and_signal.holographic_spectralfield import wave_field
        return wave_field(field, velocity=velocity, c=c, dx=dx)

    def spectral_ocean(self, height, velocity=None, g=9.81, dx=1.0):
        """A deep-water ocean surface as a SpectralField: the dispersive omega(|k|) = sqrt(g|k|) (long swells
        outrun short chop). Seed the height with phillips_spectrum for a real sea state. See
        holographic_spectralfield.ocean_field / phillips_spectrum."""
        from holographic.sampling_and_signal.holographic_spectralfield import ocean_field
        return ocean_field(height, velocity=velocity, g=g, dx=dx)

    def electrostatic_potential(self, source, dx=1.0, eps0=1.0):
        """The electrostatic potential of a charge distribution in ONE spectral step: phi_hat = source_hat /
        (eps0|k|^2) -- the closed-form steady (t->inf) limit of the diffusion field (Thesis A: electrostatics is
        the limit). See holographic_spectralfield.poisson_solve."""
        from holographic.sampling_and_signal.holographic_spectralfield import poisson_solve
        return poisson_solve(source, dx=dx, eps0=eps0)

    def depth_to_mesh(self, depth, colour=None, fx=None, fy=None, cx=None, cy=None,
                      depth_scale=1.0, discontinuity=0.08, smooth_iters=0):
        """DEPTH MAP -> a CLEAN triangulated HEIGHT-FIELD MESH (the mesh-cleanup path for single-view photo-to-3D):
        every pixel a vertex at its unprojected 3-D position, each 2x2 block two triangles EXCEPT where depth jumps
        > discontinuity (dropped, so the near foreground is not welded to the far background -- the melted-mesh
        artifact). Regular-grid surface = ZERO non-manifold edges (unlike the dual-contour points_to_mesh path),
        directly smoothable/textured. Accepts ANY depth (1=near): hand it fuse_depth for hazy/DoF photos. Returns
        (mesh, vertex_colours). See holographic_photo3d.depth_to_mesh."""
        from holographic.rendering.holographic_photo3d import depth_to_mesh as _dtm
        return _dtm(depth, colour=colour, fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=depth_scale,
                    discontinuity=discontinuity, smooth_iters=smooth_iters)

    def image_to_mesh(self, image, light=None, res=48, depth_scale=1.0, smooth=1.0, repair=False):
        """END-TO-END image -> MESH: estimate depth by shape-from-shading, unproject to points, derive oriented
        normals from the depth, and reconstruct a surface (points_to_mesh / dual-contour). Returns (verts, quads,
        field, grids). HONEST: single-view + relative depth (shape-from-shading is ill-posed), so this meshes the
        VISIBLE FRONT as a height-field surface, not a watertight solid object -- the back is unobserved. For per-pixel
        splats instead of a mesh, use image_to_3d. Chains depth_from_image + unproject + normal_from_height +
        points_to_mesh.

        repair=True runs mesh_repair (weld near-dups + drop degenerate/unreferenced) on the result before returning it
        -- the standard cleanup for a downstream consumer. DEFAULT-OFF and byte-identical when off. MEASURED KEPT
        NEGATIVE: on shape-from-shading output the dual-contour extractor emits genuine NON-MANIFOLD edges (edges shared
        by >2 faces) that weld/fill CANNOT fix, so repair=True is a modest cleanup, NOT a guarantee of a manifold /
        retopo-ready mesh -- that needs a manifold-guaranteeing extractor or non-manifold-edge splitting (deferred)."""
        import numpy as _np
        from holographic.mesh_and_geometry.holographic_autobump import normal_from_height
        img = _np.asarray(image, float)
        H, W = img.shape[:2]
        fx = fy = 0.9 * W; cx, cy = W / 2.0, H / 2.0
        depth = self.depth_from_image(img, light=light, smooth=smooth)
        z = (1.5 - depth) * depth_scale
        pts = self.unproject_depth(z, fx, fy, cx, cy).reshape(-1, 3)     # (H*W, 3)
        # normals from the depth height-field (the existing autobump wheel), flattened to match the points.
        nmap = normal_from_height(z)                                     # (H, W, 3)
        nrm = nmap.reshape(-1, 3)
        lo = pts.min(axis=0) - 0.05; hi = pts.max(axis=0) + 0.05
        verts, quads, field, grids = self.points_to_mesh(pts, nrm, lo.tolist(), hi.tolist(), int(res))
        if repair:                                                      # opt-in standard cleanup (see kept negative)
            from holographic.mesh_and_geometry.holographic_mesh import Mesh
            rm, _report = self.mesh_repair(Mesh(verts, [tuple(int(i) for i in q) for q in quads]))
            verts, quads = rm.vertices, rm.faces
        return verts, quads, field, grids

    def depth_from_image(self, image, light=None, albedo=None, smooth=1.0):
        """Estimate a relative DEPTH MAP from a single image by classical SHAPE FROM SHADING (C1 of photo-to-3D) --
        no learned weights. Returns depth (H,W) normalised to [0,1] (1=nearest). This is the missing FRONT END for
        photo_to_3d / unproject_depth, which both need a depth map. HONEST: shape-from-shading is ill-posed
        (bas-relief ambiguity) so this is a PLAUSIBLE RELATIVE surface, not metric depth -- the confidence map in
        photo_to_3d abstains where it is weak. Pass `light` (a 3-vector) if you know it. See
        holographic_shapefromshading.shape_from_shading."""
        from holographic.rendering.holographic_shapefromshading import shape_from_shading
        return shape_from_shading(image, light=light, albedo=albedo, smooth=smooth)



    def haze_depth(self, image, p=0.95, sky_guard=True, return_extras=False):

        """RELATIVE DEPTH from a single HAZY/FOGGY image via the atmospheric scattering model (Tarel-Hautiere veil

        inference) -- the classical no-weights fix for scenes where shape_from_shading inverts the depth (fog reads

        as near). Returns depth (H,W) in [0,1], 1=nearest, ordering correct for hazy scenes. sky_guard clamps

        bright low-saturation upper-frame pixels to far. See holographic_hazedepth.haze_depth."""

        from holographic.rendering.holographic_hazedepth import haze_depth as _hd

        return _hd(image, p=p, sky_guard=sky_guard, return_extras=return_extras)



    def sharpness_depth(self, image, radius=6, gamma=0.6):

        """DEPTH-OF-FIELD DEPTH from a single image via LOCAL SHARPNESS (in-focus foreground = near, blurred

        background = far). Robust for heavily textured scenes. Returns depth (H,W) in [0,1], 1=nearest. See

        holographic_hazedepth.sharpness_depth."""

        from holographic.rendering.holographic_hazedepth import sharpness_depth as _sd

        return _sd(image, radius=radius, gamma=gamma)



    def guided_filter(self, guide, src, radius=8, eps=1e-3):
        """EDGE-AWARE MAP REFINER (He/Sun/Tang guided filter, O(N)): smooth a scalar map WHERE the guide image is
        smooth and keep its edges WHERE the guide has edges -- a local linear fit, no matting solve. Built for
        hazedepth's transmission map, but the call is general: it refines ANY (H,W) map against ANY (H,W) guide --
        AO, soft shadow, matte/alpha, upsampled normals-z, SSS thickness, a coarse mask that must snap to object
        boundaries. MEASURED (vs a same-support box blur, the honest baseline): on a guide-aligned AO map RMSE
        0.062 -> 0.017 with the edge step kept (0.55 true, 0.53 guided, 0.04 box -- the box destroys it); on a
        matte, box is WORSE than the noisy input (0.125 vs 0.105) while guided reaches 0.059.
        KEPT NEGATIVE (loud): the guide must actually EXPLAIN the map's structure. On a map whose edges IGNORE the
        guide, guided is NOT better than a box blur (0.030 vs 0.027) and injects a spurious edge from the guide.
        REGIME (do not confuse -- both are correct, in different regimes): this needs only a GUIDE IMAGE. For
        render denoising with a full G-buffer, use `denoise_svgf`/`guided_upsample`, whose variance-guided a-trous
        bilateral takes normal/albedo/depth. Deterministic. See holographic_hazedepth.guided_filter."""
        from holographic.rendering.holographic_hazedepth import guided_filter as _gf
        return _gf(guide, src, radius=radius, eps=eps)

    def fuse_depth(self, image, weights=(0.55, 0.45), use_haze=True, use_defocus=True, sky_guard=True):

        """FUSE classical depth cues (HAZE aerial-perspective + SHARPNESS depth-of-field) into one relative depth

        map -- the robust front end for HAZY or shallow-DoF photos where shape_from_shading fails. Returns depth

        (H,W) in [0,1], 1=nearest. On the foggy-forest test this more than doubled the near/far separation vs

        shape-from-shading. Hand to photo_to_3d/unproject. See holographic_hazedepth.fuse_depth."""

        from holographic.rendering.holographic_hazedepth import fuse_depth as _fd

        return _fd(image, weights=weights, use_haze=use_haze, use_defocus=use_defocus, sky_guard=sky_guard)




    def vanishing_point(self, image, top_lines=14, return_confidence=False):

        """Estimate the dominant VANISHING POINT of a scene's linear perspective from its strong OBLIQUE Hough

        lines (rails, walls, a corridor), as (vx,vy) in pixels (may lie outside the image), or None if no clear

        perspective. With return_confidence, also a [0,1] confidence (how tightly the oblique lines agree). See

        holographic_hazedepth.vanishing_point."""

        from holographic.rendering.holographic_hazedepth import vanishing_point as _vp

        return _vp(image, top_lines=top_lines, return_confidence=return_confidence)



    def ground_plane_depth(self, image, vp=None, horizon_softness=0.05):

        """GROUND-PLANE DEPTH from linear perspective: for a forward-looking camera (road, railway, hallway) the

        ground recedes toward the horizon, so depth increases with height up to the vanishing point's row. THE cue

        that captures a track/road recession when haze and defocus are both weak (mostly-in-focus scene with only

        distant mist). Returns depth (H,W) in [0,1], 1=nearest (frame bottom). vp auto-detected if None. See

        holographic_hazedepth.ground_plane_depth."""

        from holographic.rendering.holographic_hazedepth import ground_plane_depth as _gpd

        return _gpd(image, vp=vp, horizon_softness=horizon_softness)


    def auto_fuse_depth(self, image, sky_guard=None, return_weights=False):

        """AUTO-WEIGHTED depth fusion: combine the HAZE and SHARPNESS cues, each weighted by how well it AGREES

        with the scene's LINEAR PERSPECTIVE (the vanishing-point depth prior) -- so the cue actually tracking depth

        for THIS image dominates and an inverted cue is down-weighted, removing per-image hand-tuning. Falls back to

        the fixed 55/45 blend when no confident vanishing point is found. Auto sky-guard. Returns depth (H,W) in

        [0,1], 1=nearest; with return_weights also (haze_w, sharp_w, vp). See holographic_hazedepth.auto_fuse_depth."""

        from holographic.rendering.holographic_hazedepth import auto_fuse_depth as _afd

        return _afd(image, sky_guard=sky_guard, return_weights=return_weights)


    def image_to_3d(self, image, fx=None, fy=None, cx=None, cy=None, light=None, depth_scale=1.0,
                    confidence_floor=0.3, smooth=1.0):
        """END-TO-END PHOTO-TO-3D from a single image (C1->C2->C3): estimate depth by shape-from-shading, unproject
        to camera-space points, and fit per-pixel 3-D GAUSSIANS on the confident front-facing pixels (abstaining on
        edges / grazing / the unobserved back). Returns the photo_to_3d result dict (positions, colours, radii,
        confidences, abstain mask, coverage). `fx,fy,cx,cy` default to a reasonable pinhole for the image size;
        `depth_scale` stretches the relative depth into camera Z. HONEST: the depth is relative (shape-from-shading
        is ill-posed) and single-view, so this reconstructs the VISIBLE FRONT, not a watertight object. Chains
        depth_from_image + unproject_depth + photo_to_3d."""
        import numpy as _np
        img = _np.asarray(image, float)
        H, W = img.shape[:2]
        # sensible default intrinsics: ~50 deg horizontal FOV, principal point at centre.
        if fx is None:
            fx = 0.9 * W
        if fy is None:
            fy = 0.9 * W
        if cx is None:
            cx = W / 2.0
        if cy is None:
            cy = H / 2.0
        depth = self.depth_from_image(img, light=light, smooth=smooth)
        # map normalised [0,1] (1=near) to a positive camera distance (near = small Z), scaled.
        z = (1.5 - depth) * depth_scale                          # near pixels ~0.5, far ~1.5 (times scale)
        colour = img if img.ndim == 3 else _np.stack([img] * 3, axis=2)
        return self.photo_to_3d(z, colour, fx, fy, cx, cy, confidence_floor=confidence_floor)

    def photo_to_3d(self, depth, colour, fx, fy, cx, cy, confidence_floor=0.3):
        """Forecasting sweep (sec.5, depth delegation) / photo-to-3D: lift a depth map + image into per-pixel 3D
        Gaussians, but ONLY where the reconstruction is observed -- unproject the CONFIDENT front-facing,
        continuous pixels and ABSTAIN on invalid depth, occlusion edges (where unprojecting stretches fake
        geometry), grazing surfaces, and -- loudest -- the unobserved BACK of every object. A single view
        reconstructs the visible front, not a watertight guess. Returns positions/colours/radii/confidences + an
        abstain mask + coverage. See holographic_photo3d."""
        from holographic.rendering.holographic_photo3d import photo_to_gaussians
        return photo_to_gaussians(depth, colour, fx, fy, cx, cy, confidence_floor=confidence_floor)


    def sky_model(self, hour=12.0, clouds=(), stars_seed=None, star_density=0.9985, moon=None,
                  sun_intensity=18.0, cloud_seed=0, time_s=0.0, wind=(0.05, 0.02), evolve=0.10):
        """A PARAMETRIC sky -- time of day, sun arc, moon, deterministic stars, HIGH cloud layers -- as one
        radiance callable f(dirs)->rgb, pluggable anywhere sky_dome is (the tracer's sky=, a dome light's
        color=). hour drives a keyed gradient palette AND the sun's position; clouds=[(kind, coverage)] with SEVEN kinds -- cirrus
        (streaks), cirrostratus (veil), cirrocumulus (mackerel sky), altocumulus (clumps), altostratus
        (milky sun), stratocumulus (broken deck), nimbostratus (rain blanket) -- composing as Beer-Lambert
        shells with per-KIND extinction and threshold sharpness (cellular kinds keep real GAPS, base
        shading keeps texture in even a full deck);
        stars_seed makes a hash-of-direction starfield (same seed = same sky forever) that fades by daylight
        and by cloud; moon=True auto-places opposite the sun. MEASURED under one fixed transform: noon
        linear mean 0.529, sunset+cirrus 0.260, midnight+stars 0.019 -- a 28x day/night range the
        auto-exposing display view will happily hide, so compare skies with view=None. LOW puffy clouds are
        deliberately refused toward cloud_scene (the volumetric stack -- real depth and self-shadowing);
        this model owns only what reads as a textured dome from the ground. See holographic_skymodel."""
        from holographic.rendering.holographic_skymodel import sky_model
        return sky_model(hour=hour, clouds=clouds, stars_seed=stars_seed, star_density=star_density,
                         moon=moon, sun_intensity=sun_intensity, cloud_seed=cloud_seed,
                         time_s=time_s, wind=tuple(wind), evolve=evolve)

    def render_animation(self, scene, camera, keys, n_frames=24, fps=12.0, width=160, height=120,
                         gif=None, interp="smooth", seed=0, lights=None, sky=None, sky_keys=None,
                         **render_kw):
        """ANIMATE the Scene document and render it -- keyframes in, frames (and optionally a GIF) out.

        NOTHING HERE IS NEW MACHINERY, and that is the point. The keyframe Timeline (key/sample, easing,
        vectorised) existed. place() existed. render_preview existed. save_gif was the one missing writer.
        What did not exist was any path that COMPOSES them -- 'animate an object in my scene document'
        returned scene_info, 'render frames over time' returned a cache -- and no JSON client could build
        the composition itself, because a Timeline object cannot cross POST /invoke. This is the bridge, in
        the same sense scene_set_texture is: JSON-safe values in, the callables live server-side.

        `keys` (JSON-safe): {handle: {property: [[t, value], ...]}} with property one of
        'position' ([x,y,z]), 'rotation' (Euler degrees [rx,ry,rz]), 'scale' (a number). Times are in
        SECONDS; the animation spans [0, n_frames/fps]. `interp` is the Timeline's easing for every key
        ('linear', 'step', 'smooth', 'ease_in', 'ease_out').

        Returns the list of rendered frames ((H,W,3) float each); with `gif=` set, also writes an animated
        GIF there -- the see->fix loop for MOTION.

        KEPT NEGATIVES, stated rather than discovered later:
          * Frames render at PREVIEW quality (draft, 1 bounce) -- measured at 12x the full path. An
            N-frame final-quality animation is N full renders and wants the job system, not a loop that
            holds an HTTP request open for an hour.
          * Edits go through place(), so the LAST FRAME'S transforms persist on the document afterwards --
            deliberate (undo works, and 'where did it end up' is a real question), but a caller re-rendering
            stills afterwards should place() things back or undo.
          * Rotation keys interpolate EULER ANGLES componentwise. Fine for turntables and tilts; a path
            that swings past gimbal territory wants quaternions, which the Timeline does not speak. That is
            a real limitation of composing existing parts and it is documented instead of hidden."""
        import numpy as np
        from holographic.rendering.holographic_scene_render import render_preview

        tl = self.timeline()
        tracks = []                                            # (handle, property, channel_name)
        for handle, props in keys.items():
            for prop, kvs in props.items():
                if prop not in ("position", "rotation", "scale"):
                    raise ValueError("unknown animatable property %r -- position, rotation, scale "
                                     "(what place() can apply)" % (prop,))
                chan = "%s.%s" % (handle, prop)
                for t, value in kvs:
                    tl.key(chan, float(t), np.asarray(value, float) if prop != "scale" else float(value),
                           interp=interp)
                tracks.append((handle, prop, chan))

        # sky_keys: the TIMELAPSE half, default off (additive). {'hour': [[t, hour], ...]} plus any
        # static sky_model kwargs ('clouds', 'stars_seed', 'moon', ...). Discovery already routed
        # 'day to night timelapse' at sky_model + render_animation -- but neither could DO it: keys move
        # objects, and the sky was frozen per call. The hour rides the SAME Timeline as the object keys
        # (delegation, per the hand-roll audit), and the sky closure is rebuilt per frame -- closure
        # construction only, the texture fields inside are built per call by sky_model as before.
        sky_tl = None
        sky_static = {}
        if sky_keys is not None:
            if sky is not None:
                raise ValueError("pass sky= (fixed) OR sky_keys= (animated), not both -- one sky per frame")
            sky_static = {k: v for k, v in sky_keys.items() if k != "hour"}
            if "hour" not in sky_keys:
                raise ValueError("sky_keys needs 'hour': [[t, hour], ...] -- the keyed part; everything "
                                 "else in sky_keys is passed to sky_model unchanged")
            sky_tl = self.timeline()
            for tt, hh in sky_keys["hour"]:
                sky_tl.key("hour", float(tt), float(hh), interp=interp)

        frames = []
        for i in range(int(n_frames)):
            t = i / float(fps)
            for handle, prop, chan in tracks:
                self.place(scene, handle, **{prop: tl.sample(chan, t)})
            frame_sky = sky
            frame_lights = lights
            if sky_tl is not None:
                # clouds MOVE during a timelapse (review: "changing shape and moving naturally"): the
                # frame time feeds sky_model's wind-drift + solid-noise evolution unless the caller pinned
                # time_s themselves. A timelapse compresses hours into seconds, so the animation time is
                # scaled up (x240: one real second of animation ~ four minutes of sky) -- override with an
                # explicit 'time_scale' in sky_keys, or freeze the clouds with 'time_s': 0.
                if "time_s" not in sky_static:
                    sky_static_frame = dict(sky_static)
                    scale_t = float(sky_static_frame.pop("time_scale", 240.0))
                    sky_static_frame["time_s"] = t * scale_t
                else:
                    sky_static_frame = {k: v for k, v in sky_static.items() if k != "time_scale"}
                frame_sky = self.sky_model(hour=float(sky_tl.sample("hour", t)), **sky_static_frame)
                # a timelapse whose LIGHTING ignores the sky is two different times of day in one frame;
                # if the caller gave no lights, the animated sky drives a dome so the ground follows the sky
                if lights is None:
                    frame_lights = [self.scene_light("dome", color=frame_sky, intensity=1.0)]
            frames.append(np.asarray(render_preview(scene, camera, width=width, height=height,
                                                    seed=seed, lights=frame_lights, sky=frame_sky,
                                                    **render_kw), float))
        if gif is not None:
            from holographic.rendering.holographic_render import save_gif
            save_gif(gif, frames, fps=fps)
        return frames

    def describe_to_scene(self, text, scene=None):
        """Words -> the CANONICAL Scene document: 'a red cube on the left and a green sphere on the right'
        becomes real, named, handled objects you can then texture, place, keyframe, and path-trace.

        WHY A BRIDGE, when build_scene already exists. leCore grew TWO scene systems that could not talk:
        build_scene -> a SemanticScene (parses text, resolves 'on'/'beside'/'inside' into positions, renders
        itself, adjusts by sentence) and new_scene -> the Scene DOCUMENT (handles, undo, selection -- the
        thing scene_set_texture, place, render_animation, render_scene_document and the whole HTTP surface
        operate on). Every parity capability of this arc landed on the document side, so an agent that
        started from words was cut off from all of it: 8/8 audit phrasings for this conversion returned
        nothing relevant. The parser and the layout heuristics are REUSED (interpret_description +
        realize_scene), not reimplemented -- this is a join, and both halves keep their own behaviour.

        Returns {scene, handles: {object name: handle}, unknown: [words the parser could not ground],
        suggestions: [...]} -- unknown words are REPORTED rather than silently dropped, because 'a wooden
        gnome' quietly becoming an empty scene is the kind of no-op that sends an agent debugging its
        camera. Pass `scene=` to add into an existing document instead of a fresh one.

        Parsed colours become per-object PBRMaterials (base colour + modest roughness), so 'red' survives
        into the path tracer without the caller mapping words to library names.

        KEPT NEGATIVES, inherited honestly from the halves rather than papered over:
          * realize_scene's own limitation stands: rotation is not modelled -- 'diagonal' is an offset +
            stretch, not a tilt. Fix it there if it matters; a bridge is the wrong layer.
          * The realized SDFs arrive PRE-PLACED (position baked into the geometry), so each object's
            document transform starts as identity. place()/keyframes still work -- they COMPOSE on top --
            but scene_info reports position [0,0,0] for a freshly described object. Re-deriving the baked
            offset to normalise it would mean probing the SDF for its own centre: guessy, and wrong for
            'inside'. Reported as-is instead.
          * The controlled vocabulary is the parser's (SHAPES/COLORS/RELATIONS in scene_semantic). This
            bridge adds no words; `unknown` tells you what fell through."""
        from holographic.simulation_and_physics.holographic_scene_semantic import (interpret_description,
                                                                                   realize_scene)
        import holographic.materials_and_texture.holographic_matlib as ML

        parsed = interpret_description(text)
        renderables = realize_scene(parsed["objects"]) if parsed["objects"] else []
        sc = scene if scene is not None else self.new_scene()
        handles = {}
        for r in renderables:
            col = tuple(r.get("color", (0.7, 0.7, 0.7)))
            mat = r.get("mat_name") or ML.PBRMaterial(name="described:%s" % r["name"],
                                                      base_color=col + (1.0,), roughness=0.6)
            handles[r["name"]] = sc.add(name=r["name"], geometry=r["sdf"], material=mat)
        return {"scene": sc, "handles": handles,
                "unknown": list(parsed.get("unknown", ())),
                "suggestions": list(parsed.get("suggestions", ()))}

    def refine_scene(self, scene, target_image, max_steps=4, apply=True, geometry=False, focus=None,
                     width=96, height=72):
        """CLOSE THE LOOP: hand a described scene a TARGET IMAGE and let the engine improve itself toward
        it -- the capability that goes past screenshot-and-hope. The reference Blender integration can show
        an agent its render; it cannot score candidate edits against a goal and apply the best one. This
        can, deterministically, with the trail on record.

        `scene` is a SemanticScene (from build_scene / describe-side work); `target_image` is (H,W,3) --
        MUST match `width` x `height`, because the critic compares like with like (a size mismatch used to
        surface as a broadcast error from deep inside SSIM; it is checked HERE now, with the fix named).
        apply=True runs the bounded greedy driver (refine_to_target) and returns {applied,
        start_distance, final_distance, steps, history}; apply=False only SCORES (propose_edits) and
        returns the ranked candidates without touching the scene -- the read-only critic an agent can
        consult before deciding. geometry=True lets it also move/scale objects; `focus` scores a subject
        region only.

        Verified live before wiring: from 'a red sphere' toward a night-time target, distance 0.2625 ->
        0.0000 in one applied edit -- the loop rediscovered 'make it night' by itself.

        KEPT NEGATIVES: the edit vocabulary is the semantic scene's (lighting/brightness/material/colour,
        coarse move/scale with geometry=True) -- it proposes from what it can say, not from arbitrary
        parameter space; and it operates on the SEMANTIC scene, not the Scene document, because candidate
        edits are sentences. Use describe_to_scene afterwards to promote the refined result. See
        holographic_scene_semantic.SemanticScene.refine_to_target / propose_edits."""
        import numpy as np
        tgt = np.asarray(target_image, float)
        if tgt.shape[:2] != (height, width):
            raise ValueError("target_image is %dx%d but the critic renders %dx%d -- pass a target at the "
                             "loop's own size (width=/height=), or set width/height to match the target"
                             % (tgt.shape[1], tgt.shape[0], width, height))
        if apply:
            return scene.refine_to_target(tgt, max_steps=max_steps, geometry=geometry, focus=focus,
                                          width=width, height=height)
        return scene.propose_edits(tgt, geometry=geometry, width=width, height=height)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p09_navigate_cost_field", "_UnifiedPart09")
    print("holographic_unified_p09_navigate_cost_field selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

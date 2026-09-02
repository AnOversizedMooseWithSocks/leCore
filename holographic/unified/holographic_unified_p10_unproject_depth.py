"""Part 10 of UnifiedMind's faculty surface -- 140 methods, unproject_depth .. _encyclopedia_faculty.

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


class _UnifiedPart10:

    def unproject_depth(self, depth, fx, fy, cx, cy):
        """Turn a depth map into 3D points in camera space (the pinhole unprojection). Returns (H, W, 3). See
        holographic_photo3d.unproject."""
        from holographic.rendering.holographic_photo3d import unproject
        return unproject(depth, fx, fy, cx, cy)

    def make_scene(self, objects, dim=2048, seed=0):
        """Query Interface (Phase 4): build a SCENE from a list of nested objects -- each object is encoded as a
        nested VSA record (a nested field is bind(role, sub_record)). Pair with `query_scene`. See
        holographic_graphql.Scene."""
        from holographic.io_and_interop.holographic_graphql import Scene
        return Scene(objects, dim=dim, seed=seed)

    def query_scene(self, graphql, scene):
        """Query Interface (Phase 4): run a GraphQL query over a scene -- 'ask for exactly the nested fields you
        want,' which maps onto unbinding exactly those roles. Filters objects by a `where` arg and returns only
        the requested (possibly nested) fields per object. GraphQL is the natural fit for the nested scene where
        SQL fits the flat tables. See holographic_graphql.resolve."""
        from holographic.io_and_interop.holographic_graphql import resolve
        return resolve(scene, graphql)

    def table_analyze(self, table, column, tasks=("regimes", "forecast"), min_seg=16):
        """A table COLUMN is a SERIES in a different costume (the sweep-89 database check):
        the analyst stack existed and the database existed, with NO bridge -- a user had
        to hand-extract floats and separately discover five faculties. One door: pass a
        UserTable (db.resolve('ns.table')), a Table, or a list of row dicts, name a
        numeric column, pick tasks from {'demux','regimes','forecast','formula','drift'}
        (the same contract as the MCP series_analyze door). Non-numeric columns fail
        HERE, naming the offending value, not five calls deep."""
        import numpy as np
        # MEASURED (sweep 89): UserTable.rows is the list of ROW DICTS while .records is
        # the (n, dim) HYPERVECTOR matrix -- the names invite exactly the wrong guess.
        # Prefer dict rows; fall back to a callable records() (plain Table); accept a
        # bare list of dicts as itself.
        rows = getattr(table, "rows", None)
        rows = rows() if callable(rows) else rows
        if not (isinstance(rows, list) and (not rows or isinstance(rows[0], dict))):
            rec = getattr(table, "records", None)
            rows = rec() if callable(rec) else (table if isinstance(table, list) else [])
        vals = []
        for i, r in enumerate(rows):
            v = r.get(column) if isinstance(r, dict) else getattr(r, column, None)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                raise ValueError("column %r is not numeric at row %d (%r) -- "
                                 "table_analyze needs a numeric series" % (column, i, v))
        if len(vals) < 8:
            raise ValueError("column %r has %d numeric values; need >= 8"
                             % (column, len(vals)))
        arr = np.asarray(vals, float)
        want = set(tasks)
        out = {"column": str(column), "n": int(len(arr))}
        if "demux" in want:
            out["demux"] = self.demux_series(arr)
        if "regimes" in want:
            out["regimes"] = self.detect_regimes(arr, min_seg=int(min_seg))
        if "forecast" in want:
            out["forecast"] = self.envelope_forecast(arr)
        if "formula" in want:
            fml, rep = self.decompose_signal(arr)
            out["formula"] = {"formula": str(fml), **{k: rep[k] for k in
                              ("resid_rms", "n_terms", "mdl_bits", "mode") if k in rep}}
        if "drift" in want:
            half = len(arr) // 2
            out["drift"] = self.structure_drift(self.structure_fingerprint(arr[:half]),
                                                self.structure_fingerprint(arr[half:]))
        return out

    def database(self, dim=2048, seed=0):
        """Query Interface (Phases 9-13): a DATABASE you OWN -- user namespaces over a read-only 'system'
        namespace. The mind's capability registry is published as `system.actions` out of the box, so you can
        SELECT from it, CREATE your own databases/tables beside it, INSERT rows, bookmark system rows
        (`insert_select`), define live views (`create_view`), and persist by replay (`to_state`/`from_state`) --
        but never write to system.* (the wall). See holographic_query.Database."""
        from holographic.agents_and_reasoning.holographic_query import Database, capability_registry
        db = Database()
        db.register_system("actions", capability_registry(self, dim=dim, seed=seed))
        return db

    def db_query(self, sql, db):
        """Run a SQL statement over a Database: CREATE DATABASE, CREATE TABLE ns.t (cols), INSERT INTO ns.t (cols)
        VALUES (...), or SELECT ... FROM ns.table. Writes to system.* are refused by the wall. Bookmarks and views
        use the object API (db.insert_select / db.create_view). See holographic_query.run_db_sql."""
        from holographic.agents_and_reasoning.holographic_query import run_db_sql
        return run_db_sql(sql, db)

    def capabilities(self, dim=2048, seed=0):
        """Query Interface (Phase 6): introspect this mind into a capability REGISTRY -- a VSA table with one row
        per public faculty (name, a heuristic domain, its one-line doc). 'What can this mind do?' then becomes an
        ordinary data query: `mind.query("SELECT name FROM actions WHERE domain = 'render'", mind.capabilities())`
        or a GROUP BY domain census. See holographic_query.capability_registry."""
        from holographic.agents_and_reasoning.holographic_query import capability_registry
        return capability_registry(self, dim=dim, seed=seed)

    def explain_program(self, machine, program_vec, init_acc=None):
        """Query Interface (Phase 7): EXPLAIN a program WITHOUT running it. A DRY RUN with no handlers -- every
        APPLY is a no-op so the heavy work is skipped, but the machine walks the whole program, so the trace names
        which faculties it WOULD call and how many steps it takes. The program-level twin of the pipeline's
        plan()->EXPLAIN. See holographic_query.explain_program."""
        from holographic.agents_and_reasoning.holographic_query import explain_program
        return explain_program(machine, program_vec, init_acc=init_acc)

    def user_table(self, name, columns, dim=1024, seed=0):
        """A WRITABLE query table you own (sweep 123 -- UserTable had no constructor door;
        table_vacuum / table_analyze / table_history all consumed one that callers could only
        build by import). CREATE with its columns as role vectors, then .insert(row) /
        .update / .delete on the handle; run_sql over it; commit versions with
        table_history. See holographic_query.UserTable."""
        from holographic.agents_and_reasoning.holographic_query import UserTable
        return UserTable(str(name), list(columns), dim=int(dim), seed=int(seed))

    def make_table(self, rows, roles, dim=1024, seed=0):
        """Query Interface (Phase 1): ingest tabular data (a list of {column: value} dicts) into a VSA Table --
        each row becomes a role-bound record, with the exact values kept beside the vectors. Pair with `query`.
        See holographic_query.from_rows."""
        from holographic.agents_and_reasoning.holographic_query import from_rows
        return from_rows(rows, roles, dim=dim, seed=seed)

    def query(self, sql, table):
        """Query Interface (Phases 2-3): run a small SQL subset (SELECT/FROM/WHERE/ORDER BY/LIMIT) over a VSA
        Table. Exact predicates (=, >, <) run on the stored props; the FUZZY predicate (~) ranks rows by semantic
        cosine and returns a per-row `_confidence` -- the two things a plain database can't do natively. See
        holographic_query.run_sql."""
        from holographic.agents_and_reasoning.holographic_query import run_sql
        return run_sql(sql, table)

    def svgf_denoise(self, image, normal, albedo, depth, levels=5, **kw):
        """Interactive Render Speed (technique E): edge-aware denoise a noisy (1-spp-style) image the engine's
        way -- a holographic bilateral filter whose edge-stopping is a cosine in the bound (normal, albedo, depth)
        feature space, run coarse-to-fine over the a-trous hierarchy. Similar surfaces blend, edges don't; it
        beats a plain blur measurably (kept negative: it denoises, it can't add detail). The sibling render pieces
        already exist -- robust_accumulate (firefly clamp), SPRTRecall (adaptive sampling), TemporalReuse
        (reproject). See holographic_svgf."""
        from holographic.rendering.holographic_denoisehome import Denoise                    # the Denoise home  consolidation R5
        return Denoise.image(image, normal, albedo, depth, method="svgf", levels=levels, **kw)

    def forecast(self, series, d=20, alpha=0.1, abstain_width=None, seed=0):
        """Forecasting backlog (F3): the "forecast any data" door. Routes a 1-D series to the producer that
        calibrates tightest (linear AR vs analog recall), wraps it in a calibrated conformal interval, and
        abstains when uncertain. Returns a RoutedForecaster -- `.predict(last_window)` gives {point, interval,
        coverage, abstain, producer}. A misroute fails SAFE (wide interval), never a confident wrong answer. See
        holographic_forecast."""
        from holographic.misc.holographic_forecast import route_and_forecast
        rf, info = route_and_forecast(series, d=d, alpha=alpha, abstain_width=abstain_width, seed=seed)
        return rf

    def analog_forecaster(self, contexts, successors, sim_floor=0.5, seed=0):
        """Forecasting backlog (F4): analog forecasting -- "find the past that looks like now, return what
        followed." Pure VSA recall (sublinear via HoloForest); yields a DISTRIBUTION over outcomes natively and
        ABSTAINS when no near analog exists. Use holographic_analog.delay_embed to build (context, successor)
        pairs from a series. See holographic_analog.AnalogForecaster."""
        from holographic.misc.holographic_analog import AnalogForecaster
        return AnalogForecaster(sim_floor=sim_floor, seed=seed).fit(contexts, successors)

    def multi_horizon_forecaster(self, rollout_fn, alpha=0.1, kind="scalar"):
        """Forecasting backlog (F6): multi-horizon forecast with a TRUSTED-HORIZON gate -- calibrate a per-step
        interval that widens with the horizon, and report how far ahead a closed-loop `rollout_fn(state, H)` can
        be trusted so a sim/renderer substitutes a cheap forecast up to there and recomputes beyond it. Kept loud:
        chaotic systems have a short trusted horizon by nature (Lyapunov time). See holographic_horizon."""
        from holographic.misc.holographic_horizon import MultiHorizonForecaster
        return MultiHorizonForecaster(rollout_fn, alpha=alpha, kind=kind)

    def generate_gated(self, codebook, confidence_floor=0.6, steps=12, seed=None, **kw):
        """Forecasting backlog (F5): confidence-gated generation -- generate a vector, then score how VALID it is
        (cosine to the nearest codebook atom) and ACCEPT or flag it. Turns open-ended generation into calibrated
        generation: a low-confidence sample is flagged so a caller can resample or abstain. Kept scoped: for
        open-ended generation this is a filter/abstention aid, not a correctness guarantee."""
        import numpy as _np
        from holographic.agents_and_reasoning.holographic_ai import cosine as _cos
        v = self.generate_vector(codebook, steps=steps, seed=seed, **kw)
        if isinstance(v, tuple):
            v = v[0]
        cb = _np.asarray(codebook, float)
        vv = _np.asarray(v, float)
        sims = cb @ vv / (_np.linalg.norm(cb, axis=1) * (_np.linalg.norm(vv) + 1e-12) + 1e-12)
        conf = float(sims.max())
        return {"vector": vv, "confidence": conf, "accepted": conf >= confidence_floor,
                "nearest": int(sims.argmax())}

    def recurrent_forecaster(self, kind="esn", n_in=1, n_res=600, dim=1024, seed=0):
        """Forecasting backlog (F7, de-silo): a gradient-free sequence producer, now reachable through the mind.
        `kind='esn'` -> EchoStateNetwork (nonlinear-lift readout); `kind='vsa'` -> VSAReservoir (permute IS the
        recurrence). `.fit(inputs, targets)` then `.predict(inputs)`. Pairs with `forecast`/conformal for a
        calibrated interval. See holographic_recurrent."""
        from holographic.agents_and_reasoning.holographic_recurrent import EchoStateNetwork, VSAReservoir
        return EchoStateNetwork(n_in, n_res=n_res, seed=seed) if kind == "esn" else VSAReservoir(dim=dim, seed=seed)

    def market_projector(self, dim=512, K=5, H=3, R=80, seed=1):
        """Forecasting backlog (F7, de-silo): the RayProjector time-series study -- casts rays into a data field
        and reads held-out quantiles -- now reachable through the mind. `.fit(moves, burst)` then `.project(row)`.
        See holographic_market.RayProjector."""
        from holographic.misc.holographic_market import RayProjector
        return RayProjector(dim=dim, K=K, H=H, R=R, seed=seed)

    def adaptive_sample_budget(self, variance_of_mean, current_n, target_half_width, z=1.959963984540054):
        """Forecasting sweep (sec.5, renderer delegation): a CALIBRATED adaptive-sampling stop. Given a renderer's
        per-pixel variance-of-the-mean at `current_n` samples, return the EXTRA samples each pixel needs to reach
        `target_half_width` at confidence z (0 where already converged) -- "sample where the estimate is still
        uncertain, stop where it is confident," replacing a hand-set threshold. Honest: a pixel mean's interval is
        Gaussian/CLT (var falls as sigma^2/n; halving the interval costs 4x samples), NOT conformal -- a single
        pixel has no calibration set. See holographic_adaptive_sample."""
        from holographic.sampling_and_signal.holographic_adaptive_sample import sample_budget
        return sample_budget(variance_of_mean, current_n, target_half_width, z=z)

    def scheduler_capacity(self, dim=None, gated=True, target_recall=0.9, seed=0):
        """Forecasting sweep (sec.5.5): the scheduler's cost model IS a forecaster. Instead of assuming the
        theoretical packing wall (~0.10*D), MEASURE it -- probe growing superposition loads, measure gated cleanup
        recall, and return the largest load whose recall stays >= target_recall (a CALIBRATED capacity) plus the
        recall curve, so the scheduler packs as many as it is confident it can. `should_superpose` (in
        holographic_superschedule) gates a batch on the measured wall. Honest finding: the measured wall is often
        BELOW the theoretical dial at a strict target -- assuming overpacks. See holographic_superschedule."""
        from holographic.scene_and_pipeline.holographic_superschedule import calibrated_capacity, pack_capacity
        d = dim if dim is not None else self.dim
        cap, curve = calibrated_capacity(d, gated=gated, target_recall=target_recall, seed=seed)
        return {"capacity": cap, "curve": curve, "theoretical": pack_capacity(d, gated=gated),
                "target_recall": target_recall, "gated": gated}

    def calibrate_forecast(self, preds, actuals, alpha=0.1, kind="scalar", abstain_width=None):
        """Forecasting backlog (F1): wrap ANY producer's forecasts in a CALIBRATED prediction interval that
        abstains when too wide to trust. Fit on a held-out set of (prediction, truth) pairs -- scalar (scored by
        |error|) or vector (scored by 1 - cosine, the engine's own metric) -- and get a ConformalForecaster whose
        `.predict(point)` returns {point, interval/cosine_radius, coverage, abstain}. Distribution-free, no learned
        weights -- the forecasting twin of RecallNull. See holographic_conformal."""
        from holographic.mesh_and_geometry.holographic_conformal import ConformalForecaster
        cf = ConformalForecaster(alpha=alpha, kind=kind, abstain_width=abstain_width)
        cf.calibrate(list(preds), list(actuals))
        return cf

    def adaptive_conformal(self, alpha=0.1, gamma=0.05, window=200):
        """Forecasting backlog (F2): temporal conformal for time series (which break exchangeability). Adaptive
        Conformal Inference holds LONG-RUN coverage at 1-alpha under drift by widening after a miss / narrowing
        after a hit -- `.step(residual)` per observation, `.realized_coverage()` reads the held rate. Kept loud:
        under a fundamental regime change (~0% overlap) no feedback rule recovers coverage -- abstain and flag
        drift. See holographic_conformal.AdaptiveConformal."""
        from holographic.mesh_and_geometry.holographic_conformal import AdaptiveConformal
        return AdaptiveConformal(alpha=alpha, gamma=gamma, window=window)

    def forecast_coverage_report(self, residuals_calib, residuals_test, alphas=(0.01, 0.05, 0.1, 0.2)):
        """Forecasting backlog (F8): the coverage instrument -- for each alpha, confirm the empirical coverage on
        held-out residuals tracks the nominal 1-alpha. The forecasting twin of calibration_report; an interval you
        cannot verify is one you cannot trust. See holographic_conformal.coverage_report."""
        from holographic.mesh_and_geometry.holographic_conformal import coverage_report
        return coverage_report(residuals_calib, residuals_test, alphas=alphas)

    def forecast_crps(self, samples, actual):
        """Forecasting backlog (F8): CRPS -- the proper score for a probabilistic (sample) forecast; coverage says
        the interval is wide enough, CRPS says the forecast is GOOD (rewards accuracy AND sharpness). Lower is
        better; a sharp-and-accurate forecast scores strictly below a vague one. See holographic_conformal."""
        from holographic.mesh_and_geometry.holographic_conformal import crps_sample
        return crps_sample(samples, actual)

    def render_pipeline(self, preset="preview", **overrides):
        """Render/Sim Pipeline (Phases 1-2): build a configured, validated render+sim pipeline. `preset` is
        'preview' / 'final' / 'interactive', or a holographic_pipeline.PipelineConfig; `**overrides` tweak
        individual flags. Returns a Pipeline -- call `.plan()` to see exactly which stages will run and WHY
        (without rendering), or `.run(scene, seed, renderer=...)` to execute a frame. The builder auto-includes
        prerequisites (ask for SVGF, get the G-buffer) and rejects impossible combos up front with a clear
        message. See holographic_pipeline."""
        from holographic.scene_and_pipeline.holographic_pipeline import PipelineConfig, build_pipeline
        if isinstance(preset, str):
            cfg = {"preview": PipelineConfig.preview, "final": PipelineConfig.final,
                   "interactive": PipelineConfig.interactive}[preset]()
        else:
            cfg = preset
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return build_pipeline(cfg)

    def field_effect(self, sdf, effect, radius=1.0, falloff="smooth", strength=1.0, texture=None):
        """Render/Sim Pipeline (Part 4): a shaped zone of influence -- the SDF is the shape, its distance is the
        falloff, `effect(points, weight)` is what it does (attractor/wind/drag/density). Compose several with
        holographic_fieldeffect.FieldGroup (they add); attach one to a moving node with AttachedFieldEffect.
        Ready effects: attract_to / repel_from / uniform_force. See holographic_fieldeffect."""
        from holographic.misc.holographic_fieldeffect import FieldEffect
        return FieldEffect(sdf, effect, radius=radius, falloff=falloff, strength=strength, texture=texture)

    def particle_sim(self, pos, vel, force_fn, integrator="symplectic"):
        """Render/Sim Pipeline (Phase 0 / G2): a point-mass sim advanced by the shared, energy-stable symplectic
        integrator -- exactly what a FieldEffect's summed forces drive. `force_fn(pos, vel) -> accelerations`.
        `.advance(dt)` steps it. See holographic_integrate.ParticleSim / SimStep."""
        from holographic.misc.holographic_integrate import ParticleSim
        return ParticleSim(pos, vel, force_fn, integrator=integrator)

    def near_surface_to_sdf(self, near_surface, h=1.0, threshold=0.0):
        """Sweep 3 local completion (photo-to-3D): turn a NEAR-SURFACE signed field (accurate only in a thin band
        around the surface -- e.g. from depth unprojection) into a FULL, globally-consistent signed distance
        field. The band's sign is all we trust away from the surface, so we threshold it to an inside/outside
        occupancy and run the existing fast-sweeping eikonal (`signed_distance_field`) to redistance everywhere.
        Honest reuse: the eikonal solver already does the extension; this just prepares its input from a band."""
        import numpy as _np
        inside = _np.asarray(near_surface, float) < threshold        # sign is the reliable part away from the surface
        return self.signed_distance_field_3d(inside, h=h) if inside.ndim == 3 else self.signed_distance_field(inside, h=h)

    def texture_map(self, image, wrap="repeat"):
        """Sweep 3 local completion: an image-based TEXTURE MAP sampled by UV with bilinear interpolation -- the
        per-texel detail a factor-level PBRMaterial couldn't carry. Pass one to a PBRMaterial's `*_map` argument
        (base_color_map/metallic_map/roughness_map/emissive_map) and call material.sample(u, v) for the effective
        shaded values. See holographic_materialio.TextureMap."""
        from holographic.materials_and_texture.holographic_materialio import TextureMap
        return TextureMap(image, wrap=wrap)

    def graph_namespace(self, branching=6, beam=1, seed=0):
        """Sweep 3 local completion (fit-correct): a hierarchical namespace/navigation tree over labelled vectors
        (GraphMemory) -- observe_vector(v, label) grows the tree, classify_vector(v) routes a query to its region.
        Use it for HIERARCHY / NAMESPACE / NAVIGATION (a DB namespace tree, a region index), NOT for exact recall
        -- its recall accuracy is a documented negative that collapses at scale, which is why the sweep re-homed
        it here. See holographic_graph_memory.GraphMemory."""
        from holographic.simulation_and_physics.holographic_graph_memory import GraphMemory
        return GraphMemory(self.dim, branching=branching, beam=beam, seed=seed)

    def directional_field(self, dirs, values, order=3):
        """Sweep 3 item 8: project a DIRECTIONAL function (sampled `values` at unit directions `dirs`) onto
        spherical-harmonic coefficients -- the SAME primitive for directional LIGHT (PRT radiance transfer, splat
        view-dependent colour) and directional SOUND (ambisonic encoding). `sample_directional` reconstructs.
        Reuses prt's SH basis (no fork). See holographic_spharm."""
        from holographic.sampling_and_signal.holographic_spharm import sh_project
        return sh_project(dirs, values, order=order)

    def sample_directional(self, coeffs, dirs, order=3):
        """Sweep 3 item 8: reconstruct a directional function from its spherical-harmonic coefficients at `dirs`
        (radiance toward a view direction, or ambisonic gain toward a listening direction). Inverse of
        directional_field. See holographic_spharm."""
        from holographic.sampling_and_signal.holographic_spharm import sh_reconstruct
        return sh_reconstruct(coeffs, dirs, order=order)

    def conditional_propagator(self, transitions, ridge=1e-3):
        """Sweep 3 item 9: a CONDITIONAL Propagator -- one learned dynamics operator per ACTION, so predict is a
        bind of that action's transform onto the state (dynamics' 'predict = bind a transform to a state', made
        action-conditional). Unifies lookahead's per-action forward model with the dynamics module and gives
        model-based planning (`.plan(state, actions, codebook)` re-anchors each hop). `transitions` maps each
        action to its (state, next_state) pairs. See holographic_condprop.ConditionalPropagator."""
        from holographic.misc.holographic_condprop import ConditionalPropagator
        return ConditionalPropagator.learn(transitions, ridge=ridge)

    def storage_spine(self, block_size=32):
        """Sweep 3 item 7: one content-addressed, deduplicated, erasure-robust byte store -- uri KEYS a record,
        a content hash DEDUPS identical payloads, a fountain code makes retrieval robust to lost droplets. The
        shared spine for the query DBs, texture atlases, scene deltas, and the compile cache. `.put(tags, bytes)`
        / `.get(key, loss=...)`. See holographic_storage.StorageSpine."""
        from holographic.caching_and_storage.holographic_storage import StorageSpine
        return StorageSpine(block_size=block_size)

    def faculties(self):
        """The capability table (Sweep 3 item 1): the sorted names of every faculty currently callable from a VSA
        program as `APPLY <name>` -- built-ins (cleanup/denoise/...) plus anything `register_apply_handler` added
        (an octree query, an agent behaviour, a fitted embedding). This is the SAME live handler set the machine's
        APPLY uses, so introspection, a drives scheduler, or an moe gate all read one registry -- the convergence
        point the sweep flagged. Registration is `register_apply_handler(name, fn)`; this is the read side."""
        return sorted(self._procedure_handlers().keys())

    def spatial_index(self, points, cell_size):
        """Sweep 3 item 2: ONE shared uniform-grid spatial index over a point set -- radius / knn / closest-point
        queries in O(1)-ish (nearby cells only), byte-identical to a brute-force scan. The widest-fanout mechanism
        (cull, navigation, collision broadphase, sampling, Walk-on-Spheres closest-point all ask the same query).
        See holographic_spatial.SpatialGrid."""
        from holographic.misc.holographic_spatial import SpatialGrid
        return SpatialGrid(points, cell_size)

    def reaction_diffusion(self, size=64, dim=48, steps=40, seed=0, **kw):
        """Sweep 3 item 3: a reaction-diffusion cellular automaton (HyperCA) -- a local update rule over a
        hypervector field from which global patterns emerge (spots/stripes/fronts). One solver, many domains
        (patina/weathering, procedural texture, fur/skin patterns, crystal growth, erosion). Runs `steps` and
        returns the stepped HyperCA (read `.grid`). See holographic_automaton.HyperCA."""
        from holographic.misc.holographic_automaton import HyperCA
        ca = HyperCA(size=size, dim=dim, seed=seed, **kw)
        for _ in range(int(steps)):
            ca.step()
        return ca

    def emergent_concepts(self, vigilance=0.45, commit=3.0, prune=0.5, seed=0, **kw):
        """Sweep 3 item 4: online, label-free concept growth -- watch a stream, grow concepts with no fixed
        category count, and COMMIT one (via the double-diffusion staircase, which pulls in the diffusion
        mechanism) when its slowly-integrating support survives. An online GROUP BY / clustering reusable across
        agent situations, vision classes, market regimes, or the knowledge registry. Feed it with `.perceive(x)`.
        See holographic_emergence.EmergentConcepts."""
        from holographic.simulation_and_physics.holographic_emergence import EmergentConcepts
        return EmergentConcepts(vigilance=vigilance, commit=commit, prune=prune, seed=seed, **kw)

    def temporal_reuse(self):
        """Sweep 3 item 5: the temporal-reuse loop -- reuse last frame's per-cell result, reproject it
        (backward-warp), and re-solve ONLY the dirty region, optionally accumulating (running average) for noisy
        estimators. The render/solve SPEED discipline (path tracer, Walk-on-Spheres, fluid/wave). Call
        `.solve(solve_fn, n, dirty=..., reproject=..., accumulate=...)`. See holographic_temporal.TemporalReuse."""
        from holographic.simulation_and_physics.holographic_temporal import TemporalReuse
        return TemporalReuse()

    def cosserat_strand(self, points_or_strand, bend_stiffness=0.5, shape_stiffness=0.6):
        """A hair as a COSSERAT ROD (H2b): each segment carries an orientation frame, so the strand HOLDS its
        curl under gravity and can carry a TWIST -- the quality upgrade over plain bend springs. Accepts an
        (n,3) point array or a groom Strand. `.step()/.settle()` simulate it; `.set_root_twist(a)` twists it;
        `.curl_amount()`/`.twist_of(i)` read it out. See holographic_cosserat.CosseratStrand."""
        from holographic.simulation_and_physics.holographic_cosserat import CosseratStrand, from_strand
        if hasattr(points_or_strand, "points"):
            return from_strand(points_or_strand, bend_stiffness=bend_stiffness, shape_stiffness=shape_stiffness)
        return CosseratStrand(points_or_strand, bend_stiffness=bend_stiffness, shape_stiffness=shape_stiffness)

    def groom_hair(self, surface_sdf, n_strands, bounds, length=1.0, n_pts=8, curl=0.0, lean=0.0,
                   width=0.02, seed=0, length_jitter=0.0):
        """HAIR GROOM (H1): grow `n_strands` rooted on an SDF surface, each along its outward normal (+ optional
        lean), straight or curly, smoothed for rendering. `bounds`=(lo_vec,hi_vec). Returns a list of Strand.
        See holographic_groom.groom."""
        from holographic.mesh_and_geometry.holographic_groom import groom
        return groom(surface_sdf, n_strands, bounds, length=length, n_pts=n_pts, curl=curl, lean=lean,
                     width=width, seed=seed, length_jitter=length_jitter)

    def simulate_hair(self, strands, steps=60, dt=1.0 / 60.0, gravity=(0.0, -9.8, 0.0), wind=None,
                      body_sdf=None, collide_radius=0.0, bend_compliance=1e-3, ftl=True, damping=0.02):
        """HAIR DYNAMICS (H2): simulate strands as PBD chains (root pinned, inextensible with Follow-The-Leader,
        bend springs for stiffness) under gravity, optional wind force, and body collision. Returns new strands.
        See holographic_groom.simulate_strands."""
        from holographic.mesh_and_geometry.holographic_groom import simulate_strands
        return simulate_strands(strands, steps=steps, dt=dt, gravity=gravity, wind=wind, body_sdf=body_sdf,
                                collide_radius=collide_radius, bend_compliance=bend_compliance, ftl=ftl, damping=damping)

    def interpolate_hair(self, guides, render_roots, k=3, clump=0.4):
        """GUIDE INTERPOLATION (H3): make many render strands from a few simulated guide strands by blending the
        k nearest guides and clumping toward one -- what makes full fur affordable. See holographic_groom."""
        from holographic.mesh_and_geometry.holographic_groom import interpolate_strands
        return interpolate_strands(guides, render_roots, k=k, clump=clump)

    def hair_wind(self, strength=2.0, res=24, bounds=((-2, 2), (-2, 2), (-2, 2)), octaves=3, seed=0, base=(1.0, 0.0, 0.0)):
        """CURL-NOISE WIND (H7): a divergence-free (volume-preserving) turbulent wind field; call `.force(strand)`
        for the per-point force to pass to simulate_hair. Fur ripples without ballooning. See holographic_groom.CurlWind."""
        from holographic.mesh_and_geometry.holographic_groom import CurlWind
        return CurlWind(strength=strength, res=res, bounds=bounds, octaves=octaves, seed=seed, base=base)

    def render_hair(self, strands, camera, light_dir=(0.3, 0.6, 0.6), width=400, height=400,
                    shader="kajiya", hair_color=(0.55, 0.35, 0.15), smooth_levels=2, lod_stride=1,
                    specular_tint=0.0, specular_strength=1.0):
        """RENDER HAIR (H4/H5/H6): project each strand's smoothed centerline and shade its segments by their
        TANGENT -- `shader`='kajiya' (anisotropic sheen) or 'marschner' (physical R/TT/TRT with a colored
        secondary highlight). Returns an (H,W,3) image.

        DARK HAIR NEEDS `specular_tint`. Kajiya-Kay adds its specular lobe WHITE at full amplitude regardless
        of hair colour, so dark hair renders silver -- MEASURED over 4,000 strand orientations at
        hair_color=(0.075,0.048,0.034): 61% of strands brighter than the hair colour and 17.5% reading as
        white, peaking at 1.079 (a 14x overshoot). Marschner 2003 measured that the secondary highlight is
        COLOURED by the fibre; `specular_tint` (0=white, 1=fully hair-tinted) and `specular_strength` apply
        that. Defaults reproduce the published model bit-for-bit, so no existing render changes. Dark hair
        wants tint~0.7, strength~0.35: white-reading strands drop 17.5% -> 0.00%.
        See holographic_hairshade.render_hair."""
        from holographic.mesh_and_geometry.holographic_hairshade import render_hair
        return render_hair(strands, camera, light_dir=light_dir, width=width, height=height, shader=shader,
                           hair_color=hair_color, smooth_levels=smooth_levels, lod_stride=lod_stride,
                           specular_tint=specular_tint, specular_strength=specular_strength)

    def solve_pde(self, sdf, boundary_value, points, source=None, n_walks=256, eps=1e-3, seed=0, max_steps=256):
        """WALK ON SPHERES: solve Laplace (Delta u = 0) or Poisson (-Delta u = source) on the interior of an SDF,
        with NO meshing, by random walks that step by the distance-to-boundary (one SDF eval) until they hit the
        boundary and read `boundary_value` there. Returns (solution, standard_error) at each query point. Works on
        any shape you can write an SDF for. SIGGRAPH list #7. See holographic_wos.solve_on_sdf."""
        from holographic.misc.holographic_wos import solve_on_sdf
        return solve_on_sdf(sdf, boundary_value, points, source=source, n_walks=n_walks, eps=eps, seed=seed, max_steps=max_steps)

    def steady_heat(self, sdf, boundary_temperature, points, n_walks=256, eps=1e-3, seed=0):
        """STEADY-STATE heat on any SDF shape: hold the boundary at `boundary_temperature(point)` and find the
        equilibrium temperature at interior `points`. This is Laplace's equation via Walk on Spheres -- the
        grid-free, mesh-free steady complement to the transient `holographic_heat` diffusion and the `wave`
        field. Returns (temperature, standard_error). See holographic_wos."""
        from holographic.misc.holographic_wos import solve_on_sdf
        return solve_on_sdf(sdf, boundary_temperature, points, n_walks=n_walks, eps=eps, seed=seed)

    def curl_noise(self, res=64, bounds=((0.0, 8.0), (0.0, 8.0)), octaves=4, seed=0, obstacle_sdf=None, ramp=1.0, dx=1.0):
        """CURL NOISE: divergence-free procedural turbulence (u, v) on a grid -- the curl of an fBm streamfunction,
        so it never compresses (no sources/sinks). Optional `obstacle_sdf` makes the flow go AROUND a shape. Cheap
        wind/smoke detail with no fluid solve. SIGGRAPH list #1. See holographic_curlnoise.curl_noise."""
        from holographic.mesh_and_geometry.holographic_curlnoise import curl_noise
        return curl_noise(res, bounds=bounds, octaves=octaves, seed=seed, obstacle_sdf=obstacle_sdf, ramp=ramp, dx=dx)

    def tearable_cloth(self, rows=12, cols=12, spacing=1.0, compliance=2e-3, material="paper",
                       tear_strain=None, pin="top"):
        """A TEARABLE thin sheet: a PBD cloth whose links SNAP when stretched past the material's tear strength,
        so it rips and separates into pieces when yanked. `.step(pull=..., gravity=...)` advances and tears;
        `.connected_components()` / `.piece_sizes()` report the split. SIGGRAPH list #2 (fracture -- a new
        capability). See holographic_tear.TearableCloth."""
        from holographic.mesh_and_geometry.holographic_tear import TearableCloth
        return TearableCloth(rows=rows, cols=cols, spacing=spacing, compliance=compliance, material=material,
                             tear_strain=tear_strain, pin=pin)

    def levitation_chamber(self, height=0.10, wavelength=0.0086, amplitude=4000.0, n_beads=40,
                           gravity=9.81, bead_radius=1e-3, bead_density=25.0, seed=0, mass_scale=3e-07):
        """ACOUSTIC LEVITATION: beads in a vertical standing wave feel the Gor'kov radiation force and are trapped
        at the pressure NODES (spaced lambda/2) against gravity. `.settle(field_on=True)` holds them aloft;
        `field_on=False` lets them fall. The 'sound moves objects' showpiece. Acoustics A7 -- reuses the standing
        field idea (A3), the particle system, and gravity. See holographic_levitate.LevitationChamber."""
        from holographic.simulation_and_physics.holographic_levitate import LevitationChamber
        return LevitationChamber(height=height, wavelength=wavelength, amplitude=amplitude, n_beads=n_beads,
                                 gravity=gravity, bead_radius=bead_radius, bead_density=bead_density, seed=seed, mass_scale=mass_scale)

    def room_acoustics(self, size=(5.0, 4.0, 3.0), material="plaster", absorption=None, c=343.0):
        """GEOMETRIC ROOM ACOUSTICS: how a room echoes. `.rt60()` is the reverberation time (Sabine), `.reflections
        (source, listener)` the early echoes via the image-source method (arrival = path/c, level from the wall
        reflectance), `.impulse_response(...)` the sampled room response. Hard rooms ring, soft rooms are dead.
        Acoustics A6 -- the acoustic twin of the path tracer, reusing A2's reflectance. See
        holographic_roomacoustic.ShoeboxRoom."""
        from holographic.misc.holographic_roomacoustic import ShoeboxRoom
        return ShoeboxRoom(size=size, material=material, absorption=absorption, c=c)

    def power_law_viscosity(self, shear_rate, K=1.0, n=1.8, eta_min=1e-4, eta_max=1e4):
        """The non-Newtonian power-law viscosity eta = K * shear_rate^(n-1): a viscosity that DEPENDS on how fast
        the fluid is sheared. n>1 thickens under shear (cornstarch/oobleck), n<1 thins (ketchup, paint), n=1 is
        Newtonian. Returns a per-cell viscosity field. See holographic_nonnewtonian.power_law_viscosity."""
        from holographic.simulation_and_physics.holographic_nonnewtonian import power_law_viscosity
        return power_law_viscosity(shear_rate, K, n, eta_min=eta_min, eta_max=eta_max)

    def nonnewtonian_fluid(self, shape, power_law_n=1.8, consistency_K=1.0, **kwargs):
        """A fluid solver with NON-NEWTONIAN rheology: its viscosity is the shear-rate-dependent power law, so it
        can carry cornstarch (n>1, shear-thickening -- stiffens where you shear it hard) or a shear-thinning fluid
        (n<1). A StableFluid in its power-law mode; n=1 would be ordinary Newtonian. See holographic_fluid.StableFluid
        (power_law_n / consistency_K) and holographic_nonnewtonian."""
        from holographic.simulation_and_physics.holographic_fluid import StableFluid
        return StableFluid(shape, power_law_n=power_law_n, consistency_K=consistency_K, **kwargs)

    def wave_field(self, shape, c=343.0, dx=1.0, damping=0.0, absorb_border=0):
        """A scalar acoustic PRESSURE field that PROPAGATES (the compressible wave the incompressible fluid can't
        carry): d2p/dt2 = c^2 grad^2 p by leapfrog. `.pulse(center)` taps it, `.step(dt)` advances it (auto-
        subdivided to stay CFL-stable), an `absorb_border` sponge stops edge reflections. `c` may be a per-cell
        field from material sound_speed. Acoustics A3 -- the low-frequency wave complement to ray acoustics, and
        the standing field levitation (A7) will use. See holographic_wave.WaveField."""
        from holographic.simulation_and_physics.holographic_wave import WaveField
        return WaveField(shape, c=c, dx=dx, damping=damping, absorb_border=absorb_border)

    # -- QUANTUM faculties (complex wavefunction stack: field / solver / current / dot / interferometer) ---------
    def quantum_field(self, shape, dx=1.0, mass=1.0, hbar=1.0, q=1.0):
        """A COMPLEX wavefunction psi on a grid -- the central quantum object. `.gaussian_packet(center, sigma, k0)`
        launches a wave packet, `.set_potential(V)` installs a well/wall, `.set_vector_potential([Ax,Ay])` threads
        magnetic flux, `.probability_density()` is |psi|^2, `.normalize()` makes it integrate to 1. Hand it to
        `quantum_solver(...)` to evolve. The quantum complement to wave_field (which carries a REAL acoustic field).
        See holographic_quantum_field.QuantumField."""
        from holographic.simulation_and_physics.holographic_quantum_field import QuantumField
        return QuantumField(shape, dx=dx, mass=mass, hbar=hbar, q=q)

    def quantum_solver(self, field, absorb_border=0):
        """A split-operator (split-step Fourier) integrator for the time-dependent Schrodinger equation on a
        QuantumField. `.step(dt)` / `.run(n, dt)` evolve psi in place, UNITARILY (norm conserved to machine
        precision -- the kinetic step is spectral, the analytic continuation of the heat propagator). An
        `absorb_border` sponge opens the boundary for scattering. Explicit Euler is unstable and is NOT used (a
        recorded negative). See holographic_schrodinger.SplitStepSchrodinger."""
        from holographic.simulation_and_physics.holographic_schrodinger import SplitStepSchrodinger
        return SplitStepSchrodinger(field, absorb_border=absorb_border)

    def probability_current(self, psi, A=None, mass=1.0, hbar=1.0, q=1.0, dx=1.0, bc="periodic"):
        """The probability current j = (hbar/m) Im(psi* grad psi) - (q/m) A |psi|^2 of a wavefunction, as [jx, jy].
        The flow of |psi|^2 -- streamlines of j are the glowing threads in an interferometer, and a loop with
        circulation is a probability vortex. Returns real arrays. See holographic_probability_current.probability_current."""
        from holographic.simulation_and_physics.holographic_probability_current import probability_current
        return probability_current(psi, A=A, mass=mass, hbar=hbar, q=q, dx=dx, bc=bc)

    def quantum_velocity(self, psi, A=None, mass=1.0, hbar=1.0, q=1.0, dx=1.0, bc="periodic", eps=1e-12):
        """The probability VELOCITY field v = j/|psi|^2 -- hand it straight to advect_field to carry glowing tracers
        along the quantum flow (the SIDEWAYS reuse: the quantum current drives the existing advection). Returns
        [vx, vy]. See holographic_probability_current.velocity_field."""
        from holographic.simulation_and_physics.holographic_probability_current import velocity_field
        return velocity_field(psi, A=A, mass=mass, hbar=hbar, q=q, dx=dx, bc=bc, eps=eps)

    def barrier_wall(self, shape, axis, position, thickness, height, gap=None):
        """A high-V WALL across the grid, perpendicular to `axis` -- the other potential builder for a
        quantum simulation, and the sibling of the already-wired quantum_dot_well.
        WHY IT IS HERE AT ALL (sweep 134): the catalog has carried a `barrier_wall` card since it was
        written, naming a module function no agent could /invoke -- the "present below, unreachable
        above" class the derived above/below matrix found 18 of. Its sibling well was wired and it was
        not, so an agent could build the dot and not the barrier it tunnels through.
        `gap` (the delegate's own default: None) opens a slit in the wall -- a barrier with a hole in
        it is the double slit, and leaving it unreachable would have shipped the wall without the
        experiment. delegation_drift --gate caught that omission on the very sweep that wired this
        verb, which is the audit set doing its job on its own author.
        See holographic_quantum_dot.barrier_wall."""
        from holographic.simulation_and_physics.holographic_quantum_dot import barrier_wall
        return barrier_wall(shape, axis, position, thickness, height, gap=gap)

    def gas_pressure(self, density, temp_K, name="air"):
        """P = rho R_specific T -- the pressure (Pa) of a gas at a given density and temperature.
        THE INVERSE DIRECTION, which is why this is not redundant with ideal_gas. A parcel from
        mind.ideal_gas() is CONSTRUCTED from a pressure and exposes .P and .density(); nothing went
        the other way, so "what pressure does this density imply" had no door on the mind at all.
        See holographic_gas.gas_pressure."""
        from holographic.simulation_and_physics.holographic_gas import gas_pressure
        return gas_pressure(density, temp_K, name=name)

    def is_flammable(self, material):
        """Does this material have combustion data -- CAN it burn? A predicate, and deliberately not
        an action: mind.burn_object() commits you to an ignition, and asking first is the cheap half.
        See holographic_combustion.is_flammable."""
        from holographic.simulation_and_physics.holographic_combustion import is_flammable
        return is_flammable(material)

    def quantum_dot_well(self, shape, center, depth, width):
        """A narrow Gaussian potential well (the quantum dot) -- hand it to a QuantumField.set_potential. A negative
        `depth` makes a repulsive barrier (a tunnelling scatterer). See holographic_quantum_dot.gaussian_well."""
        from holographic.simulation_and_physics.holographic_quantum_dot import gaussian_well
        return gaussian_well(shape, center, depth, width)

    def quantum_transmission(self, k0, dot_V=None, shape=(256, 128), dx=0.2, sigma=10.0, x_start=40,
                             cut=None, steps=600, dt=0.02, absorb_border=20):
        """MEASURE the fraction of a packet (carrier k0) that crosses a cut plane past an optional dot potential --
        the transmission. Sweep k0 with and without the dot to see the resonance/tunnelling emerge (measured, not
        painted). See holographic_quantum_dot.measure_transmission."""
        from holographic.simulation_and_physics.holographic_quantum_dot import measure_transmission
        return measure_transmission(k0, dot_V=dot_V, shape=shape, dx=dx, sigma=sigma, x_start=x_start,
                                    cut=cut, steps=steps, dt=dt, absorb_border=absorb_border)

    def quantum_solenoid_A(self, shape, center, flux, core=1.0):
        """The azimuthal vector potential of a thin solenoid carrying `flux` -- thread it through a ring with
        QuantumField.set_vector_potential to get the Aharonov-Bohm phase. See holographic_quantum_scene.solenoid_vector_potential."""
        from holographic.simulation_and_physics.holographic_quantum_scene import solenoid_vector_potential
        return solenoid_vector_potential(shape, center, flux, core=core)

    def aharonov_bohm_phase(self, flux, shape=(128, 128), dx=0.2, q=1.0, ring_radius=24, steps=1):
        """MEASURE the relative phase the two arms of a ring accumulate from enclosed magnetic `flux` -- the
        Aharonov-Bohm interference shift, which equals q*Phi/hbar even though the field is zero on the arms. See
        holographic_quantum_scene.measure_two_arm_phase."""
        from holographic.simulation_and_physics.holographic_quantum_scene import measure_two_arm_phase
        return measure_two_arm_phase(flux, shape=shape, dx=dx, q=q, ring_radius=ring_radius, steps=steps)

    def quantum_two_slit(self, shape=(256, 256), dx=0.2, slit_axis=0, slit_pos=None, slit_gap=6, slit_sep=40,
                         wall_height=400.0):
        """Build a two-slit interferometer: a high-potential wall with two openings. Returns (QuantumField, V).
        Launch a packet at it (quantum_solver) and the two slits become coherent sources -> an interference
        pattern downstream. The canonical warm-up before the Aharonov-Bohm ring. See holographic_quantum_scene.two_slit."""
        from holographic.simulation_and_physics.holographic_quantum_scene import two_slit
        return two_slit(shape=shape, dx=dx, slit_axis=slit_axis, slit_pos=slit_pos, slit_gap=slit_gap,
                        slit_sep=slit_sep, wall_height=wall_height)

    def read_wav(self, path):
        """Read a PCM WAV file -> (samples in [-1,1] mono, sample_rate). The front door for driving acoustics/
        cymatics from a real sound. Acoustics A1. See holographic_audio.read_wav."""
        from holographic.misc.holographic_audio import read_wav
        return read_wav(path)

    def audio_spectrum(self, samples, rate, k=6):
        """The `k` dominant frequencies (Hz) and their amplitudes in a signal -- the tones that drive a plate or
        fluid. Acoustics A1. See holographic_audio.dominant_frequencies (and `spectrum`, `frames` there)."""
        from holographic.misc.holographic_audio import dominant_frequencies
        return dominant_frequencies(samples, rate, k=k)

    def audio_param_bus(self, samples, rate, hop=1024, size=2048, bands=None, smooth=2):
        """Build an audio -> parameter BUS: per-frame band-energy envelopes (bass/low-mid/high-mid/treble by
        default, normalised 0..1) plus an onset/beat signal -- the wire that drives scene parameters from music
        (W5'). Returns a ParamBus; in a render loop call `bus.subscribe(band, lo, hi, frame)` to map a band onto
        a parameter range (e.g. metaball viscosity from the bass), or `bus.at(frame)` for all bands, or
        `bus.onset` for beats. Reuses the existing STFT (audio_spectrum's holographic_audio.frames + spectrum) --
        only the band binning is new. `bands` is a tuple of (lo_hz, hi_hz) pairs; `smooth` de-jitters the
        envelopes. See holographic_parambus.param_bus."""
        from holographic.misc.holographic_parambus import param_bus, DEFAULT_BANDS
        return param_bus(samples, rate, hop=hop, size=size,
                         bands=bands if bands is not None else DEFAULT_BANDS, smooth=smooth)

    def milk_parse(self, text):
        """PARSE a Milkdrop `.milk` preset's TEXT into a MilkPreset (holographic_milkdrop) -- settings +
        per_frame_init/per_frame/per_pixel equation families (compiled) + the captured warp/comp HLSL shaders. Then
        `preset.initial_state()` and `preset.run_frame(state, audio={bass,mid,treb}, time, frame)` evaluate the
        per-frame equations deterministically, driving the motion vars (q1..q32, zoom, rot, ...) from the audio
        envelopes (pair with audio_param_bus). The EQUATION layer -- the per-pixel warp mesh + pixel shaders are
        parsed and stored but run by the renderer, not here. Returns a MilkPreset."""
        from holographic.io_and_interop.holographic_milkdrop import parse_milk
        return parse_milk(text)

    def milk_eval(self, expr, vars=None):
        """Evaluate ONE ns-eel2 expression string (Milkdrop's equation language) against a variable dict
        (holographic_milkdrop) -- SAFE (a whitelisted recursive-descent grammar, never Python eval) and
        deterministic. Unknown vars read as 0; divide-by-zero is 0; an unsupported function raises. Returns a float;
        `vars` is updated in place for assignments. The building block milk_parse compiles per equation."""
        from holographic.io_and_interop.holographic_milkdrop import eval_expr
        return eval_expr(expr, vars)

    def acoustic_impedance(self, material):
        """Characteristic acoustic impedance Z = rho * c (rayl) of a material, reused from its density x speed of
        sound -- the acoustic twin of a refractive index. Acoustics A2. See holographic_acoustic.impedance."""
        from holographic.misc.holographic_acoustic import impedance
        return impedance(material)

    def acoustic_interface(self, mat_a, mat_b):
        """Sound crossing from material A into B: (R, T) = fractions of energy reflected and transmitted, from the
        impedance mismatch (a big mismatch like air/steel reflects nearly all). Energy conserved. Acoustics A2. See
        holographic_acoustic.interface; `wall_absorption` for a surface's absorbed fraction."""
        from holographic.misc.holographic_acoustic import interface
        return interface(mat_a, mat_b)

    def chladni_plate(self, shape="square", grid=40, medium="sand", n_modes=48, base_hz=200.0,
                      n_grains=6000, seed=0):
        """A vibrating plate whose CYMATIC figures are its Laplacian eigenmodes: `.drive(freqs, amps)` (or
        `.drive_mode(k)`) sets the displacement from a sound's spectrum, `.step_medium(dt)`/`.settle()` drift sand
        to the nodes, `.render()` shows the figure. The headline acoustics demo, reusing the spectral eigenmodes.
        Acoustics A4. See holographic_cymatics.ChladniPlate."""
        from holographic.simulation_and_physics.holographic_cymatics import ChladniPlate
        return ChladniPlate(shape=shape, grid=grid, medium=medium, n_modes=n_modes, base_hz=base_hz,
                            n_grains=n_grains, seed=seed)

    def oxidation_field(self, shape, exposure=None, moisture=1.0, seed=None):
        """A CORROSION front over a surface grid: rust/patina that NUCLEATES at exposed/wet faces (default: the
        border) and SPREADS inward as a reaction-diffusion front. `.step(material, dt)` advances it, `.albedo(
        material)` gives the per-cell base->oxide colour blend (steel->rust, copper->patina). Process M4. See
        holographic_oxidation.OxidationField; `oxide_color` for a single whole-object sample."""
        from holographic.simulation_and_physics.holographic_oxidation import OxidationField
        return OxidationField(shape, exposure=exposure, moisture=moisture, seed=seed)

    def oxide_color(self, material, ox_fraction):
        """The blended colour of a material at oxidation fraction 0..1 -- pristine base to full oxide (rust orange,
        patina green). The weathering interpolation for a single sample. See holographic_oxidation.oxide_color."""
        from holographic.simulation_and_physics.holographic_oxidation import oxide_color
        return oxide_color(material, ox_fraction)

    def burn_object(self, material, mass_kg, temp_K=293.15):
        """An object BEING CONSUMED by fire: `.light()` ignites it, `.step(dt)` advances the burn -- it loses mass
        (drives an M6 Fire), its appearance marches base->char->ash, and it emits that material's smoke, ending as
        ash. Process M7, tying M6 (combustion) and, via `evaporate`, M5 (a puddle drying up). See
        holographic_burn.BurningObject; `char_color` for the appearance blend at a burn fraction."""
        from holographic.misc.holographic_burn import BurningObject
        return BurningObject(material, mass_kg, temp_K=temp_K)

    def char_color(self, material, burn_fraction):
        """The surface colour at burn fraction 0..1: pristine base -> char (blackened) -> ash (grey). The burning
        interpolation. See holographic_burn.char_color."""
        from holographic.misc.holographic_burn import char_color
        return char_color(material, burn_fraction)

    def element(self, symbol):
        """A chemical ELEMENT's engine-relevant properties by symbol: name, atomic number, atomic mass, density,
        melt/boil points, flame-test colour (or None), category. The atomic ingredients materials are made of. See
        holographic_elements.element."""
        from holographic.simulation_and_physics.holographic_elements import element
        return element(symbol)

    def identify_element(self, props, margin=0.1):
        """Identify the element(s) whose categorical fingerprint {category, state} best matches `props`, e.g.
        {'category':'noble_gas','state':'gas'} -> the noble gases. The REVERSE of element(): element() looks up
        properties by name, this queries the table BY attribute via match_record + decide_or_abstain. Returns
        {'ranked', 'best', 'confident', 'score'}; confident is False when several elements share the fingerprint
        (the honest 'under-determined' answer). Categorical only -- mass/number excluded. See
        holographic_elements.identify_element."""
        from holographic.simulation_and_physics.holographic_elements import identify_element
        return identify_element(props, mind=self, margin=margin)

    def material_elemental(self, name):
        """A material's ELEMENTAL makeup and everything derived from it: {composition (element:count/ratio),
        molar_mass, flame_color, mass_fractions}. molar_mass feeds the gas law (T1); flame_color is the
        ratio-weighted BLEND of the constituents' flame-test colours (the emission-line colour a copper compound
        burns green, which the blackbody continuum alone can't give -- feeds M6). None if no composition on file.
        This is a material referencing its elements + ratio, feeding simulation. See holographic_elements."""
        from holographic.simulation_and_physics.holographic_elements import material_elemental
        return material_elemental(name)

    def fire(self, material, fuel_kg, temp_K=293.15):
        """A material-aware BURNING body: `.step(dt)` checks ignition against the material's autoignition point,
        consumes fuel at its burn rate once lit (a fire latches and sustains until the fuel runs out), and reports
        the smoke it makes (that material's colour + soot) and the flame colour (blackbody at its temperature) --
        so wood smoke and plastic smoke genuinely differ. Process M6, standing on the heat model (T4) and blackbody
        (T3). See holographic_combustion.Fire; `ignites` gates it; couplings configure_fluid / emit_smoke feed the
        fluid solver and surface emitter."""
        from holographic.simulation_and_physics.holographic_combustion import Fire
        return Fire(material, fuel_kg, temp_K=temp_K)

    def ignites(self, material, temperature_K):
        """True if `material` is at or above its autoignition temperature (hot enough to catch fire); False below
        it, or if the material is not flammable. The honest ignition gate. See holographic_combustion.ignites."""
        from holographic.simulation_and_physics.holographic_combustion import ignites
        return ignites(material, temperature_K)

    def phase_state(self, material, mass_kg, temp_K=293.15, pressure_Pa=101325.0):
        """A parcel of `material` with mass across solid/liquid/gas at one temperature: `.add_heat(Q)` warms it and
        drives melt/boil/freeze/condense, HOLDING temperature flat during a transition while the latent heat is
        paid (the boiling plateau). Boiling point tracks pressure (from the gas model T1). Process M5, standing on
        the heat model (T4). See holographic_phase.PhaseState / boiling_point_at."""
        from holographic.misc.holographic_phase import PhaseState
        return PhaseState(material, mass_kg, temp_K=temp_K, pressure_Pa=pressure_Pa)

    def blackbody_color(self, temp_K, normalize="hue"):
        """The sRGB colour a blackbody GLOWS at temperature `temp_K` -- red ember (~900 K), orange flame, white
        filament (~2800 K), blue-white star (~12000 K) -- from Planck's law integrated against the CIE curves.
        Thermodynamics T3; the ember/flame/glowing-char colour the combustion & burn processes (M6/M7) will paint
        by temperature. `normalize='hue'` gives the pure hue at full value; 'none' keeps the luminance ratio. See
        holographic_blackbody.blackbody_rgb."""
        from holographic.misc.holographic_blackbody import blackbody_rgb
        return blackbody_rgb(temp_K, normalize=normalize)

    def diffuse_heat(self, temp_field, alpha, dx=1.0, dt=None, steps=1):
        """Spread heat through a temperature FIELD by Fourier conduction (dT/dt = alpha*laplacian(T)) for `steps`
        steps, auto-substepped to stay stable at any dt, insulated boundaries (total heat conserved). `alpha` =
        thermal diffusivity k/(rho c) (see thermal_diffusivity / material_thermal). Thermodynamics T4 -- the
        temperature source the phase-change / combustion / decay processes read. See holographic_heat.diffuse_heat."""
        from holographic.simulation_and_physics.holographic_heat import diffuse_heat
        return diffuse_heat(temp_field, alpha, dx=dx, dt=dt, steps=steps)

    def heat_body(self, material, mass_kg, temp_K=293.15):
        """A lumped body of a named `material` at a uniform temperature: `.add_energy(Q)` raises it by Q/(m c),
        `.newton_cool(ambient, hA, dt)` relaxes it toward ambient. Specific heat is pulled from the material
        definition. Thermodynamics T4. See holographic_heat.HeatBody / material_thermal."""
        from holographic.simulation_and_physics.holographic_heat import HeatBody, material_thermal
        c = material_thermal(material)["specific_heat"]
        return HeatBody(mass_kg, c, temp_K=temp_K)

    def material_thermal(self, material):
        """Thermal properties of a named material -- {'density','specific_heat','thermal_conductivity'} in SI,
        reusing the definition library and the enrichment data (conductivity), with thermal_diffusivity easily
        derived. See holographic_heat.material_thermal."""
        from holographic.simulation_and_physics.holographic_heat import material_thermal
        return material_thermal(material)

    def ideal_gas(self, name="air", temp_K=293.15, pressure_Pa=101325.0):
        """A parcel of gas in a definite state: `.density()`, `.sound_speed()`, `.adiabatic_change(V2/V1)`. The
        ideal gas law P V = m R_specific T. Thermodynamics T1 -- and its speed of sound cross-checks the
        definitions' tabulated value. See holographic_gas.IdealGas."""
        from holographic.simulation_and_physics.holographic_gas import IdealGas
        return IdealGas(name=name, temp_K=temp_K, pressure_Pa=pressure_Pa)

    def boiling_point(self, pressure_Pa, material="water"):
        """The boiling temperature (K) at a given pressure, from Clausius-Clapeyron (lower pressure -> boils
        cooler, the mountain effect). Defaults to water. Thermodynamics T1 -- the fact phase change (M5) consumes
        to know WHEN a liquid turns to vapour. See holographic_gas.boiling_point."""
        from holographic.simulation_and_physics.holographic_gas import boiling_point
        return boiling_point(pressure_Pa)

    def grain_material(self, axis=(0, 1, 0), light=(0.72, 0.52, 0.32), dark=(0.40, 0.26, 0.14),
                       ring_scale=8.0, fibre=0.35, warp=0.6, seed=0, center=(0.0, 0.0, 0.0)):
        """A WOOD-GRAIN colour socket f(points)->(M,3): concentric rings along an axis + lengthwise fibre streaks +
        an fBm domain-warp that bends rings into knots. Volumetric in object space, so a cut board shows the rings
        continue. Drop it into a channel: `surface_material(color=Param(field=mind.grain_material(...)))`. Structure
        primitive M1. See holographic_grainmat.wood_albedo (also `substrate_layers` there for plywood/strata)."""
        from holographic.materials_and_texture.holographic_grainmat import wood_albedo
        return wood_albedo(axis=axis, light=light, dark=dark, ring_scale=ring_scale, fibre=fibre, warp=warp,
                           seed=seed, center=center)

    def material_inclusions(self, base, inclusions, seed=0):
        """An IMPURITY/INCLUSION colour socket f(points)->(M,3): the `base` material everywhere except in
        noise-blob pockets where each inclusion shows (carbon in steel, bubbles in glass, veins in stone).
        `inclusions` is [(material, fraction, scale), ...] with fraction the CALIBRATED covered fraction. base and
        materials are matlib preset names or rgb triples. Volumetric, deterministic. Structure primitive M3 (the
        planet's ore-deposit noise-threshold pattern, scoped to a material). See holographic_inclusions."""
        from holographic.misc.holographic_inclusions import with_inclusions
        return with_inclusions(base, inclusions, seed=seed)

    def crystal_material(self, n_seeds=32, bounds=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), seed=0, jitter=1.0,
                         base=(0.55, 0.57, 0.62), spread=0.18, crack=(0.05, 0.05, 0.06), crack_width=0.03):
        """A POLYCRYSTALLINE colour socket f(points)->(M,3): a Worley/Voronoi partition where each grain is a
        slightly different facet colour, darkened along the cell boundaries (cracks). Volumetric, deterministic.
        Structure primitive M2. Returns (cells, socket) so you can also query cells.ids / cells.edge_distance (e.g.
        for a crack roughness channel via holographic_cellular.crack_mask). See holographic_cellular."""
        from holographic.simulation_and_physics.holographic_cellular import VoronoiCells, cell_albedo
        cells = VoronoiCells(n_seeds=n_seeds, bounds=bounds, seed=seed, jitter=jitter)
        return cells, cell_albedo(cells, base=base, spread=spread, crack=crack, crack_width=crack_width, seed=seed)

    def render_material(self, name, color=None):
        """A PHYSICALLY-PLAUSIBLE render material from the library (holographic_matlib) as a first-class
        SurfaceMaterial -- so any of its ~130 glTF-PBR presets (metals, woods, stones, gems, biomes, planetary layers,
        ore deposits) drives preview / path_trace / RenderSession directly. This is the fork's physical definitions
        plugged into our render pipeline: data-driven materials, not hand-set demo colours. See
        SurfaceMaterial.from_matlib and holographic_matlib.material."""
        from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial
        return SurfaceMaterial.from_matlib(name, color=color)

    def material_catalog(self):
        """The whole render-material library grouped by class (diffuse/metal/wood/stone/glass/gem/emissive/biome/
        layer/deposit/liquid/fabric/organic) -- for a UI picker. See holographic_matlib.catalog."""
        import holographic.materials_and_texture.holographic_matlib as _ml
        return _ml.catalog()

    def fractal_planet(self, radius=1.0, seed=0, dim=256, octaves=4, relief=0.10, **kw):
        """A data-driven, physically-plausible PLANET as one region field: a fBm-displaced sphere painted by a
        Whittaker biome classifier (elevation/temperature/moisture -> ocean/desert/forest/ice), wrapped around
        interior shells (crust/mantle/core) with ore-deposit pockets. `.cross_section()` slices it; `.material_at()`
        colours any point; `.biome_histogram()` reports the surface mix. 'As above, so below' -- a planet is region
        composition all the way down. See holographic_matlib.fractal_planet."""
        import holographic.materials_and_texture.holographic_matlib as _ml
        return _ml.fractal_planet(radius=radius, seed=seed, dim=dim, octaves=octaves, relief=relief, **kw)

    def physical_material(self, name):
        """The PHYSICAL properties of a named material (density kg/m3, viscosity, Young's modulus, refractive index,
        sound speed, specific heat, phase) from the definition library -- the numbers a SOLVER needs, so a simulation
        can be data-driven by real materials instead of raw parameters. Raises KeyError with near matches if unknown.
        See holographic_definitions.MATERIALS."""
        from holographic.misc.holographic_definitions import MATERIALS
        if name not in MATERIALS:
            near = [n for n in MATERIALS if name.split("_")[0] in n][:6]
            raise KeyError("unknown material %r%s" % (name, (" -- did you mean: %s" % near) if near else ""))
        return dict(MATERIALS[name])

    def material_info(self, name):
        """EVERYTHING the engine knows about a named material in one view: its RENDER appearance (PBR: base colour,
        metallic, roughness, ior, class -- if it's a preset) AND its PHYSICAL properties (density, refractive index,
        viscosity, Young's modulus, sound speed, specific heat, phase -- if defined). This bridges the two material
        libraries so 'tell me about gold' returns both how it LOOKS (for rendering) and how it BEHAVES (for science).
        See holographic_materialindex.material_info."""
        import holographic.materials_and_texture.holographic_materialindex as _mi
        return _mi.material_info(name)

    def find_materials(self, query, k=10):
        """Discover materials across BOTH libraries (render presets + physical definitions) by plain-English keywords
        -- name, render class (metal/gem/liquid/...), or phase. Returns matches with which library each lives in.
        See holographic_materialindex.find_materials."""
        import holographic.materials_and_texture.holographic_materialindex as _mi
        return _mi.find_materials(query, k=k)

    def material_data(self, name=None, category=None):
        """REAL physics for named materials (holographic_materialdata: 116 measured
        materials, 12 categories -- density, Young's modulus, sound speed, thermal
        properties, melting point, with UNITS). Three asks in one door: material_data
        ('copper') -> that record + units; material_data(category='metal') -> the
        category roster; material_data() -> all categories with counts. This is the
        LOOKUP door the DB never had -- the roster door (materials) lists RENDER
        libraries, a different question. Unknown names return the near misses instead
        of a KeyError: a typo is a query, not a crash."""
        from holographic.materials_and_texture import holographic_materialdata as _md
        if name is not None:
            key = str(name).lower().strip()
            rec = _md.PHYSICAL_MATERIALS.get(key)
            if rec is None:
                # difflib, not prefix matching: measured, 'coper' missed 'copper'
                # under a 4-char prefix rule (the double letter breaks the prefix)
                import difflib
                near = difflib.get_close_matches(key, list(_md.PHYSICAL_MATERIALS),
                                                 n=6, cutoff=0.6)
                return {"found": False, "name": key, "near": near,
                        "categories": _md.categories()}
            out = dict(rec)
            out.update({"found": True, "name": key,
                        "units": {k: _md.UNITS[k][0] for k in rec if k in _md.UNITS}})
            return out
        if category is not None:
            names = _md.by_category(str(category).lower().strip())
            return {"category": str(category), "count": len(names), "materials": names}
        cats = _md.categories()
        return {"categories": {c: len(_md.by_category(c)) for c in cats},
                "total": len(_md.PHYSICAL_MATERIALS)}

    def materials(self):
        """The whole material roster: every material with which library it lives in, plus a summary (counts by render
        class, physical count, overlap). The 'what materials do we have?' entry point. See holographic_materialindex."""
        import holographic.materials_and_texture.holographic_materialindex as _mi
        return {"summary": _mi.summary(), "materials": _mi.all_materials()}

    def material_units(self):
        """The units of the physical properties (density -> kg/m^3, youngs -> GPa, ...) -- so a returned value is
        self-describing for a scientist. See holographic_materialindex.physical_units."""
        import holographic.materials_and_texture.holographic_materialindex as _mi
        return _mi.physical_units()

    def materials_by_category(self, category):
        """The physical materials in a category (metal / liquid / gas / polymer / ceramic / glass / mineral / stone /
        wood / biological / building / semiconductor). See holographic_materialindex.physical_by_category."""
        import holographic.materials_and_texture.holographic_materialindex as _mi
        return _mi.physical_by_category(category)

    def validate_materials(self):
        """Plausibility-check the physical material database (units, ranges, category/phase). Empty list = clean --
        the honest self-audit of the library. See holographic_materialindex.validate_physical."""
        import holographic.materials_and_texture.holographic_materialindex as _mi
        return _mi.validate_physical()

    def resolve_scenario(self, description):
        """Turn a physical description ('a block of wood floating in water', 'a steel ball sinking in oil') into a
        VALIDATED, parameterised Scenario: it grounds each named thing to its physical properties, checks the physics
        (wood floats, steel sinks -- `.consistent`), and emits `.build_spec()` -- the phenomenon + solver family +
        per-body masses/volumes a shipped solver consumes. This is how a description becomes a physically-accurate
        simulation. See holographic_definitions.resolve_scenario."""
        from holographic.misc.holographic_definitions import resolve_scenario
        return resolve_scenario(description, dim=self.dim, seed=self.seed)

    def quantity(self, value, unit, uncertainty=0.0, source=None):
        """A physical QUANTITY = value + unit + uncertainty + source, in the dimensional GRAMMAR (holographic_
        quantities): multiplication composes dimensions (density * volume -> mass), addition requires matching
        dimensions (a length plus a mass is refused as the grammar error it is), conversion is one call (`.to('ft')`),
        and uncertainty propagates. Extensible via register_unit. See holographic_quantities.Quantity."""
        from holographic.misc.holographic_quantities import Quantity
        return Quantity(value, unit, uncertainty=uncertainty, source=source)

    def estimate_bill(self, bill, price_per_kg=None, carbon_factor=None):
        """'Render' the mass, cost, and embodied carbon of a bill of materials [(material, volume_m3), ...] by
        composing recipes over the definition library's densities (reused, never duplicated): mass = Sigma density*vol,
        cost = Sigma mass*price, carbon = Sigma mass*carbon_factor -- each dimensionally checked by the grammar.
        Returns {'mass','cost','carbon','missing'}; SAMPLE price/carbon tables are used if none supplied (flagged,
        pending a real USGS/ICE ingest). See holographic_quantities.bill_mass/bill_cost/bill_embodied_carbon."""
        from holographic.misc.holographic_quantities import bill_mass, bill_cost, bill_embodied_carbon, SAMPLE_PRICE_USD_PER_KG, SAMPLE_CARBON_KG_PER_KG
        from holographic.misc.holographic_definitions import build_standard_library
        lib = build_standard_library(dim=256, seed=0)              # small dim: only the density table is used
        price = price_per_kg if price_per_kg is not None else SAMPLE_PRICE_USD_PER_KG
        carbon = carbon_factor if carbon_factor is not None else SAMPLE_CARBON_KG_PER_KG
        mass, miss_m = bill_mass(lib, bill)
        cost, miss_c = bill_cost(lib, bill, price)
        co2, miss_k = bill_embodied_carbon(lib, bill, carbon)
        return {"mass": mass, "cost": cost, "carbon": co2,
                "missing": sorted(set(miss_m) | set(miss_c) | set(miss_k))}

    def render_session(self, sdf, materials, camera, width=256, height=256, bounds=None):
        """Open a RENDER SESSION over one scene -- the object that ties the renderers together so a preview and a
        photoreal final can't drift apart. Holds an SDF + a SurfaceMaterial per object id (or one for the whole SDF) +
        a camera, and derives every output from that SINGLE scene: `.preview()` (fast render_surface), `.render_final(
        spp, on_progress=)` (progressive path_trace), `.to_splats()` (a browser splat proxy), and `.edit_channel(id,
        channel, value)` (a live material edit that shows in both). This is what a demo page drives instead of
        re-wiring render_surface / path_trace / splat export by hand. See holographic_session.RenderSession."""
        from holographic.scene_and_pipeline.holographic_session import RenderSession
        return RenderSession(sdf, materials, camera, width=width, height=height, bounds=bounds)

    def sdf_surface_points(self, sdf, bounds, n=2000, seed=0, eps=0.02):
        """Sample points that lie ON an SDF's surface (random points + one Newton step onto the zero level, kept where
        |sdf|<eps) -- the front half of the SDF->splat bridge, so any SDF scene becomes splat-viewable via
        field_to_splats. Deterministic. See holographic_session.sdf_surface_points."""
        from holographic.scene_and_pipeline.holographic_session import sdf_surface_points
        return sdf_surface_points(sdf, bounds, n=n, seed=seed, eps=eps)

    def render_surface(self, sdf, camera, width, height, materials, **kw):
        """Render an SDF scene resolving every material channel PER HIT from its socket -- so a procedural pattern on
        any channel is a solid 3-D texture (wraps curved surfaces, no UV unwrap), and opacity alpha-composites one
        transparency layer. `materials` maps object id -> SurfaceMaterial (or one material for the whole SDF). Honest
        scope: environment reflection only (use render_dispatch / render_scene for object-object mirrors). See
        holographic_surface.render_surface."""
        from holographic.mesh_and_geometry.holographic_surface import render_surface
        return render_surface(sdf, camera, width, height, materials, **kw)

    def pattern_image(self, name="fbm", width=256, height=256, span=4.0, normalize=True,
                      **params):
        """A procedural pattern as PIXELS, one call -- the z=0-slice dance that every 2D
        consumer of pattern_field re-writes (measured across three sweeps: the MCP
        image_tool, the chart backgrounds, the 2D dogfood all repeat it): pattern_field
        returns a 3-D FIELD FUNCTION over (N,3) points, so an image is a sampled z=0
        slice over a (span x span*h/w) window, min-max normalised. Returns (H,W) floats
        in [0,1]; stack or tint for RGB. Deterministic for fixed seed params."""
        import numpy as np
        f = self.pattern_field(name, **params)
        w, h = int(width), int(height)
        xs = np.linspace(0.0, float(span), w)
        ys = np.linspace(0.0, float(span) * h / max(w, 1), h)
        X, Y = np.meshgrid(xs, ys)
        P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
        v = np.asarray(f(P), float).reshape(h, w)
        if normalize:
            v = (v - v.min()) / (v.max() - v.min() + 1e-12)
        return v

    def pattern_field(self, name, **params):
        """A named deterministic procedural pattern FIELD f(points)->[0,1] (checker, stripes, gradient, dots, noise,
        fbm) that plugs into ANY Param socket -- a material channel, a region field, an emitter rate. Deterministic by
        integer-lattice hash (PYTHONHASHSEED-independent). Use holographic_pattern.field_lerp to drive a channel
        lo..hi by the pattern. See holographic_pattern."""
        from holographic.misc.holographic_pattern import make_pattern
        return make_pattern(name, **params)

    def warped_noise(self, scale=2.0, octaves=4, seed=0, warp=0.4, warp_scale=1.0, gain=0.5, lacunarity=2.0):
        """DOMAIN-WARPED fBm (W11, iq's warped noise / dFBM): fbm sampled at a point displaced by a vector of
        other fbm fields -- the swirling, flowing, marbled look plain fbm cannot make (smoke, magma, wood grain).
        Returns f(points)->[0,1]. `warp` is the displacement strength (0 = plain fbm). The single most
        demoscene-recognisable noise. See holographic_pattern.domain_warped_fbm."""
        from holographic.misc.holographic_pattern import domain_warped_fbm
        return domain_warped_fbm(scale=scale, octaves=octaves, seed=seed, warp=warp,
                                 warp_scale=warp_scale, gain=gain, lacunarity=lacunarity)

    def domain_repeat(self, sdf, period, limit=None):
        """Tile an SDF (or any .eval field) into an INFINITE lattice by folding the query domain (iq's opRep) --
        one shape becomes a whole crystal, at O(1) cost, no stored copies. `period` is a scalar or per-axis
        spacing; a 0/negative axis is not repeated (so [2,0,2] tiles a floor plane, leaving height alone).
        `limit=(lo,hi)` (per-axis integer bounds) makes it FINITE -- an lo..hi block of copies instead of an
        endless field (iq's opRepLim). Returns an object with .eval that raymarches straight away. KEPT NOTE:
        exact distance only for a shape bounded inside its cell (a shape near the period wraps into itself).
        See holographic_domain.repeat / repeat_limited / wrap_sdf."""
        from holographic.mesh_and_geometry.holographic_domain import repeat, repeat_limited, wrap_sdf
        if limit is None:
            return wrap_sdf(sdf, lambda P: repeat(P, period))
        lo, hi = limit
        return wrap_sdf(sdf, lambda P: repeat_limited(P, period, lo, hi))

    def domain_fold(self, sdf, axes=None, plane=0.0):
        """Fold an SDF's domain into mirror symmetry (kaleidoscope from one abs(), iq's opMirror). Folding one
        axis reflects space across a plane; folding all axes maps the whole world into one octant -- an instant
        8-fold crystal from a single asymmetric primitive. `axes` selects which to fold (default: all). Returns
        an object with .eval. Mirroring is an isometry, so the distance stays exact. See
        holographic_domain.fold / wrap_sdf."""
        from holographic.mesh_and_geometry.holographic_domain import fold, wrap_sdf
        return wrap_sdf(sdf, lambda P: fold(P, axes=axes, plane=plane))

    def domain_twist(self, sdf, k, axis=2, dist_scale=0.7):
        """Twist an SDF's domain into a helix (iq's opTwist): rotate space by `k` radians per unit along `axis`,
        turning a bar into a screw or a column into a spiral. `dist_scale` (<1) shrinks the reported distance to
        keep a raymarcher from oversteping the stretched space (0.7 is a safe default; lower for stronger
        twists). Returns an object with .eval. See holographic_domain.twist / wrap_sdf."""
        from holographic.mesh_and_geometry.holographic_domain import domain_twist, wrap_sdf
        return wrap_sdf(sdf, lambda P: domain_twist(P, k, axis=axis), dist_scale=dist_scale)

    def domain_bend(self, sdf, k, axis=0, dist_scale=0.7):
        """Bend an SDF's domain into an arc (iq's opCheapBend): curl a straight beam by `k` radians per unit
        along `axis`. `dist_scale` (<1) keeps the march safe through the stretched domain. Returns an object
        with .eval. KEPT NOTE: the cheap bend warps distances slightly (fine for silhouettes/shading, which is
        what it is for). See holographic_domain.bend / wrap_sdf."""
        from holographic.mesh_and_geometry.holographic_domain import domain_bend, wrap_sdf
        return wrap_sdf(sdf, lambda P: domain_bend(P, k, axis=axis), dist_scale=dist_scale)

    def smooth_min(self, a, b, k=0.1):
        """The polynomial smooth-minimum (iq's smin): a soft min(a,b) that MELTS two distance fields together
        over width `k` instead of creasing at a seam -- what turns two SDFs into one organic metaball blob. Its
        partners: smooth_max(a,b,k) for a smooth intersection, smooth_max(a,-b,k) for a smooth subtraction. k->0
        recovers the hard min. Vectorised over scalars or arrays. See holographic_domain.smin."""
        from holographic.mesh_and_geometry.holographic_domain import smin
        return smin(a, b, k=k)

    def smooth_max(self, a, b, k=0.1):
        """The smooth-maximum partner of smooth_min (iq): smooth_max(f,g,k) is a crease-free INTERSECTION of two
        SDFs, smooth_max(f,-g,k) a crease-free SUBTRACTION. Same width `k`. See holographic_domain.smax."""
        from holographic.mesh_and_geometry.holographic_domain import smax
        return smax(a, b, k=k)

    def cosine_palette(self, t, a=(0.5, 0.5, 0.5), b=(0.5, 0.5, 0.5),
                       c=(1.0, 1.0, 1.0), d=(0.0, 0.33, 0.67)):
        """iq's cosine gradient palette: turn a scalar `t` (a distance, an iteration count, an orbit trap) into
        a smooth harmonious RGB colour via a + b*cos(2*pi*(c*t + d)) per channel -- no banding, no harsh clip.
        a=base, b=contrast, c=cycles-per-channel, d=per-channel phase (the hue). Defaults are iq's rainbow;
        `t` is a scalar or array, returns (*t.shape, 3) in [0,1]. Pair with random_palette for a seed-driven
        scheme. See holographic_domain.cosine_palette."""
        from holographic.mesh_and_geometry.holographic_domain import cosine_palette
        return cosine_palette(t, a=a, b=b, c=c, d=d)

    def pattern_to_glsl(self, name, fn_name="pattern", **params):
        """Compile a CLOSED-FORM procedural pattern (checker/stripes/gradient/dots) to a GLSL `float <fn_name>(vec3 p)`
        function -- matches the numpy pattern field per-point to float precision, so a procedural background renders
        client-side and composes with the SDF/postfx emitters into GPU-resident looks. noise/fbm raise (their int64-
        lattice hash is not reproducible in GLSL ES 3.00). For a 2-D background call `<fn_name>(vec3(uv, 0.0))`. See
        holographic_pattern.pattern_to_glsl."""
        from holographic.misc.holographic_pattern import pattern_to_glsl
        return pattern_to_glsl(name, fn_name=fn_name, **params)

    def pattern_to_wgsl(self, name, fn_name="pattern", **params):
        """Compile a CLOSED-FORM pattern (checker/stripes/gradient/dots) to a WGSL `fn <fn_name>(p: vec3<f32>) -> f32`
        for WebGPU -- the WGSL counterpart of pattern_to_glsl, matching the numpy field per-point to f32. WGSL is
        emitted from the same math (no `mod`, `select` for the ternary, vec3<f32>/let), not translated from the GLSL.
        noise/fbm/noise32/fbm32 are deferred WGSL scope and raise. See holographic_pattern.pattern_to_wgsl."""
        from holographic.misc.holographic_pattern import pattern_to_wgsl
        return pattern_to_wgsl(name, fn_name=fn_name, **params)

    def cosine_palette_to_glsl(self, a=(0.5, 0.5, 0.5), b=(0.5, 0.5, 0.5), c=(1.0, 1.0, 1.0),
                               d=(0.0, 0.33, 0.67), fn_name="palette"):
        """Compile iq's cosine palette to a GLSL `vec3 <fn_name>(float t)` function -- a + b*cos(2*pi*(c*t+d)),
        clamped -- matching cosine_palette per-point to float precision. Feed it an orbit trap / iteration count /
        distance in a shader for a demoscene colouring; pair with random_palette for a seed-driven scheme
        (m.cosine_palette_to_glsl(*m.random_palette(seed))). See holographic_domain.cosine_palette_to_glsl."""
        from holographic.mesh_and_geometry.holographic_domain import cosine_palette_to_glsl
        return cosine_palette_to_glsl(a=a, b=b, c=c, d=d, fn_name=fn_name)

    def random_palette(self, seed=0, contrast=0.5):
        """A random-but-harmonious cosine palette from a seed -- the 'regenerate from seeds' lever for COLOUR
        (iq). Returns the (a,b,c,d) tuple ready for cosine_palette, sampling the TASTEFUL subspace (mid base,
        bounded amplitude, low frequencies, free phase) so a random seed gives a pleasing scheme, not noise.
        Deterministic per seed. See holographic_domain.random_palette."""
        from holographic.mesh_and_geometry.holographic_domain import random_palette
        return random_palette(seed=seed, contrast=contrast)

    def palette_stops(self, seed=0, n=8, contrast=0.5, coeffs=None):
        """Sample a cosine palette into `n` plottable RGB colour STOPS -> array (n,3) in [0,1] -- the colours-you-
        can-plot companion to random_palette (which returns cosine COEFFICIENTS a,b,c,d, not colours). Use this for
        a swatch strip, a gradient ramp, or a legend; pass coeffs=(a,b,c,d) to sample a KNOWN palette instead of a
        seeded one. Pure composition of random_palette + cosine_palette, so the stops ARE the palette's colours.
        Deterministic per seed. See holographic_domain.palette_stops."""
        from holographic.mesh_and_geometry.holographic_domain import palette_stops
        return palette_stops(seed=seed, n=n, contrast=contrast, coeffs=coeffs)

    def radiance_transfer(self, sdf, points, normals, order=3, n=512):
        """PRECOMPUTED RADIANCE TRANSFER -- collapse the light-transport integral into a per-point transfer vector once,
        then RELIGHT with a dot product (no rays). For a STATIC scene the way a surface point turns incident lighting
        into outgoing radiance (including its own soft self-shadowing) depends only on geometry, so it is precomputed in
        a spherical-harmonic basis; runtime shading is `shade_prt(transfer, project_env_to_sh(light))`. The 'don't
        path-trace, just read out' idea: expensive to precompute, ~free to relight -- wins when the light changes often
        over fixed geometry. Returns the transfer matrix (len(points), order^2). See holographic_prt."""
        from holographic.misc.holographic_prt import precompute_transfer
        return precompute_transfer(sdf, points, normals, order=order, n=n)

    def holographic_radiance_field(self, points, rgb, bounds=None, grid=14, dim=768, bandwidth=None, halo=1, seed=0):
        """Bake scene RADIANCE (colour leaving each surface point) into a TILED holographic field: space is split into
        a deterministic grid of bricks, each a small FPE radiance field within capacity, only occupied bricks stored
        (holographic_radiance.TiledRadianceField). `query(points) -> (rgb, coverage)` reads the kernel-weighted colour
        at any point (Nadaraya-Watson, self-normalising -- no calibration); coverage ~0 marks empty space, known from
        the field. This is the capacity answer to the single-vector wall (the HoloOctree move, for radiance): refine
        `grid` and the wall moves (measured 15.6 dB single -> 28.9 dB tiled). Changes to a region rebuild only its
        bricks (rebuild_cells -- an O(change) delta). Pairs with holographic_fog_volume so a render becomes a QUERY of
        geometry + density + radiance fields. HONEST: stores radiance (a solver bakes it), view-independent/diffuse-ish
        from one view, RBF-smooth. See holographic_radiance."""
        import numpy as _np
        from holographic.rendering.holographic_radiance import TiledRadianceField as _TRF
        P = _np.atleast_2d(_np.asarray(points, float))
        if bounds is None:
            lo = P.min(0) - 0.5; hi = P.max(0) + 0.5
            bounds = list(zip(lo.tolist(), hi.tolist()))
        bw = float(bandwidth) if bandwidth is not None else 2.2 * grid    # sharp kernel scales with grid resolution
        return _TRF(bounds, grid=grid, dim=dim, bandwidth=bw, halo=halo, seed=seed).bake(P, rgb)

    def holographic_fog_volume(self, centers, weights=None, bounds=None, dim=2048, bandwidth=1.1, seed=0):
        """Encode a volumetric DENSITY field (fog/atmosphere) as ONE hypervector via Fractional Power Encoding, and
        return a HolographicVolume whose `optical_depth(O, D, L)` integrates the density along any ray in CLOSED FORM
        -- one inner product per ray, no marching (holographic_volint). Unlike a marcher, the field is a property of
        ALL space: empty regions read ~0 optical depth without being discovered. `centers` are fog-blob positions in
        R^3; `bounds` defaults to the centres' extent padded. Use render_fog (or the returned .optical_depth) to
        composite atmospheric fog over a rendered frame using its depth buffer. Closed-form integral verified exact
        vs a marched reference; ~steps-fold faster than marching the same field. See holographic_volint."""
        import numpy as _np
        from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder as _VFE
        from holographic.misc.holographic_volint import HolographicVolume as _HV
        C = _np.atleast_2d(_np.asarray(centers, float))
        if bounds is None:
            lo = C.min(0) - 2.0; hi = C.max(0) + 2.0
            bounds = list(zip(lo.tolist(), hi.tolist()))
        enc = _VFE(C.shape[1], dim=dim, bounds=bounds, kernel="rbf", bandwidth=bandwidth, seed=seed)
        return _HV.from_blobs(enc, [tuple(c) for c in C], weights)

    def render_volume(self, field, camera, bounds, width=256, height=256, steps=96, mode="smoke",
                      sigma=12.0, emission_color=None, albedo=(0.9, 0.9, 0.95), lights=None,
                      background=(0.0, 0.0, 0.0)):
        """Volumetrically render a density FIELD (smoke/fire/water/particles) by marching camera rays and
        accumulating the volume integral -- vectorised over all pixels (the field IS the volume, so this is
        field-native). mode='smoke' (absorption), 'fire' (emission/blackbody ramp), 'density' (raw). Returns
        (RGB image, alpha). See holographic_render.volume_render."""
        from holographic.rendering.holographic_render import volume_render
        return volume_render(field, camera, bounds, width=width, height=height, steps=steps, mode=mode,
                             sigma=sigma, emission_color=emission_color, albedo=albedo, lights=lights,
                             background=background)

    def save_render(self, path, rgb01):
        """Write a render (an (H,W,3) image in [0,1]), routed by extension: .png via the stdlib encoder
        (deterministic, always available); .jpg/.webp/... via Pillow when installed, otherwise a refusal naming
        the install command (`pip install pillow`, extra: [images]) -- the standard opt-in contract.
        See holographic_render.save_image."""
        from holographic.rendering.holographic_render import save_image
        return save_image(path, rgb01)

    def make_cloud(self, center=(0.0, 0.0, 0.0), radius=1.0, camera=None, width=384, height=384,
                   density=6.0, seed=0, grid=56, sky=(0.20, 0.42, 0.74), sun_dir=(-0.45, -0.55, -0.6),
                   steps=160, field=None):
        """Render a convincing volumetric CLOUD in ONE call and get back an (H,W,3) image in [0,1].

        This is the low-level shortcut behind `build_scene('a fluffy cloud')`: it builds a multi-lobe cumulus
        density (cloud_field -- fBm-eroded, smooth-unioned puffs with a flat base) and renders it with a
        physically-motivated lighting model (see volume_render): self-shadowing (bright crown, dark base),
        Henyey-Greenstein forward scattering for a silver-lining glow, the Beer-Powder term so thick faces read as
        round rather than flat, and a cheap multi-scatter approximation so shadowed regions aren't pitch black.
        A default 3/4 camera is used if none is given.

        `field` (opt-in, default None = the cloud_field path, unchanged): a callable points(N,3)->density>=0 that
        REPLACES the built-in cumulus density -- the seam that lets any density source (a proc_texture volume, a
        simulation's smoke, a baked grid) borrow this exact lighting rig. cloud_scene(texture=...) rides this.

        Cost note: the fBm density is baked on a grid^3 lattice once (grid=32 ~60 s; see cloud_field); the
        lighting march is a further ~1-2 min at the defaults. Save it with mind.save_render(path, img).
        """
        import numpy as _np
        from holographic.simulation_and_physics.holographic_semantic import cloud_field
        from holographic.rendering.holographic_render import Camera, Light, volume_render
        c = _np.asarray(center, float); r = float(radius)
        if camera is None:
            camera = Camera(eye=(c[0] + 2.5 * r, c[1] + 0.45 * r, c[2] + 3.4 * r),
                            target=tuple(c), up=(0, 1, 0), fov_deg=40, aspect=width / max(height, 1))
        field = field if field is not None else cloud_field(c, r * 1.3, density=density, seed=seed, grid=grid)
        # tight, cloud-SHAPED bounds (not a naive symmetric cube): the multi-lobe body in cloud_field sits low
        # (flat base near -0.42 in its own radius units) and rises higher than it sits below centre, so a snug
        # asymmetric box keeps the fixed `steps` count sampling the cloud, not surrounding empty air -- a loose
        # box halves the effective resolution through the cloud for no benefit.
        bounds = (c + r * _np.array([-2.2, -1.0, -2.2]), c + r * _np.array([2.2, 1.6, 2.2]))
        # albedo doubles as the (white, slightly >1) SUN colour here so the lit crown reads brilliant white;
        # a blue ambient fills the shadow side, and 5 multi-scatter octaves keep the shadow bright (clouds stay
        # white even in shade). sigma is high enough to read as a solid body but still lets light penetrate.
        img, alpha = volume_render(field, camera, bounds, width=width, height=height, steps=steps,
                                   mode="smoke", sigma=3.0, albedo=(1.32, 1.30, 1.24),
                                   lights=[Light("directional", direction=tuple(sun_dir))],
                                   self_shadow=True, shadow_steps=20, shadow_sigma=8.0, phase_g=0.35,
                                   powder=False, multi_scatter=5,
                                   ambient=(0.48, 0.60, 0.82), background=sky)
        # composite over a DEEP-blue vertical sky gradient (rich blue up top, paler at the horizon)
        yy = _np.linspace(0, 1, height)[:, None, None]
        top = _np.asarray(sky); bot = _np.clip(_np.asarray(sky) * 1.7 + 0.30, 0, 1)
        grad = top * (1 - yy) + bot * yy
        a = alpha[..., None]
        return _np.clip(img * a + grad * (1 - a), 0, 1)

    def cloud_scene(self, preset="cumulus", quality="fast", center=(0.0, 0.0, 0.0), radius=1.0,
                    seed=0, camera=None, texture=None, texture_params=None, erode=0.30, **overrides):
        """The ONE-WORD cloud tool: presets x quality tiers over make_cloud, so 'good clouds, fast' is a
        single call instead of tuning density/grid/steps/sun by hand.

        preset -- the cloud's CHARACTER (each is a tuned parameter set over make_cloud):
          'cumulus'  the classic fluffy fair-weather cloud (default)
          'wispy'    thin, eroded, translucent (low density)
          'storm'    dense, tall, dark-based under a moodier sky
          'sunset'   cumulus lit low and warm against an orange-pink sky
        quality -- the SPEED/QUALITY trade, measured on this box:
          'fast'     ~6 s   (grid 28, 64 march steps, 192px)  -- iteration speed
          'balanced' ~20 s  (grid 40, 100 steps, 288px)       -- everyday
          'final'    ~2 min (grid 56, 160 steps, 384px)       -- the make_cloud defaults
        texture -- OPT-IN density source from the procedural texture MENU instead of the built-in cumulus:
          cloud_scene(texture='musgrave', texture_params={'kind':'ridged'}) shapes the cloud from ridged
          multifractal (streaky/wispy), 'voronoi' gives cellular clump clusters, 'fbm' classic billow. The
          texture is windowed by a soft spherical falloff and ERODED (density = falloff * max(tex - erode, 0))
          so it reads as a blob in the sky, not a textured cube; the full lighting rig (self-shadow, silver
          lining, multi-scatter) applies unchanged. Direct evaluation -- no grid bake, so texture clouds skip
          the ~60 s lattice cost.
        Any make_cloud keyword can be overridden explicitly (density, sun_dir, sky, width, ...).
        Deterministic in seed. Returns the (H,W,3) image in [0,1]. See make_cloud for the density model and
        holographic_render.volume_render for the lighting rig; holographic_proctex supplies texture= fields."""
        import numpy as _np
        presets = {
            "cumulus": dict(density=6.0),
            "wispy":   dict(density=2.4),
            "storm":   dict(density=10.0, sky=(0.16, 0.20, 0.30), sun_dir=(-0.7, -0.25, -0.6)),
            "sunset":  dict(density=6.5, sky=(0.42, 0.26, 0.30), sun_dir=(-0.85, -0.12, -0.5)),
        }
        tiers = {
            "fast":     dict(grid=28, steps=64,  width=192, height=192),
            "balanced": dict(grid=40, steps=100, width=288, height=288),
            "final":    dict(grid=56, steps=160, width=384, height=384),
        }
        if preset not in presets:
            raise ValueError("preset must be one of %s, got %r" % (sorted(presets), preset))
        if quality not in tiers:
            raise ValueError("quality must be one of %s, got %r" % (sorted(tiers), quality))
        kw = dict(presets[preset]); kw.update(tiers[quality]); kw.update(overrides)
        if texture is not None:
            from holographic.materials_and_texture.holographic_proctex import proc_texture
            tp = dict(texture_params or {})
            tp.setdefault("seed", seed)
            tex = proc_texture(texture, **tp)
            c = _np.asarray(center, float); r = float(radius)
            dens = float(kw.pop("density", 6.0))
            ero = float(erode)

            def field(P):
                P = _np.atleast_2d(_np.asarray(P, float))
                q = (P - c[None, :]) / r
                q = q * _np.array([1.0, 1.6, 1.0])            # squash y: clouds are wider than tall
                d2 = _np.sum(q * q, axis=1)
                fall = _np.clip(1.0 - d2 / 1.9, 0.0, 1.0) ** 2   # soft spherical window -> a blob, not a cube
                return dens * fall * _np.clip(_np.asarray(tex(q + 0.5)) - ero, 0.0, None)
            kw["field"] = field
        return self.make_cloud(center=center, radius=radius, seed=seed, camera=camera, **kw)

    def water_body(self, container=None, level=0.72, preset="ocean", size=1.0, extent=40.0, res=192,
                   t=0.0, seed=0, ripple=0.35, material="water", **wave_overrides):
        """The CONTAINER-FIRST water tool: everything between 'I want water' and pixels. container=None ->
        OPEN water (ocean/pond over `extent` m); 'glass'/'pool'/'bowl' -> a stock vessel filled to `level`
        with real Gerstner RIPPLES on top (scaled to the vessel, animated by t); any SDF -> the cavity the
        water fills. `material` picks the liquid from the material library (water/water_deep/oil/honey ...:
        colour from matlib, IOR from the library -- oil really refracts at 1.47). Waves adjustable at every
        scale via preset + gerstner_waves keywords (choppiness, wind_heading, wavelength_range ...).
        Returns a WaterBody: .render('fast'|'final') with pre-balanced lighting (open: shaded-mesh raster
        ~2 s / hi-res; contained: refractive path trace ~60 s @ 20 spp / 64 spp), .camera(), .at_time(t)
        for coherent animation, plus .mesh/.surface (open) or .scene_sdf/.material_fn (contained) for
        custom pipelines. See holographic_ocean.water_body."""
        from holographic.simulation_and_physics.holographic_ocean import water_body as _wb
        return _wb(container=container, level=level, preset=preset, size=size, extent=extent, res=res,
                   t=t, seed=seed, ripple=ripple, material=material, **wave_overrides)

    # -- background job control (start/pause/resume/cancel/monitor a slow render) --------------------------
    # A shared JobManager, lazily built (same pattern as .mind on the HTTP service): every job_* method below
    # operates on it, so a job started in one call is visible to job_status/job_pause/etc. in the next.
    @property
    def _job_manager(self):
        if getattr(self, "_jobmgr", None) is None:
            from holographic.scene_and_pipeline.holographic_jobs import JobManager
            from holographic.scene_and_pipeline.holographic_coordinator import InProcessBackend
            from holographic.scene_and_pipeline.holographic_renderjobs import WORKER_NAME, _noise_bake_slice_worker
            self._jobmgr = JobManager(InProcessBackend(), store_dir=".lecore_jobs")
            self._jobmgr.register_worker(WORKER_NAME, _noise_bake_slice_worker)
            # C10: the GENERIC worker -- run any public faculty as a job. It closes over THIS mind, which is why
            # it is registered here rather than living at module scope like the noise worker: a faculty call needs
            # the mind, and a mind cannot be put in a JSON checkpoint. Re-registered on every reopen (this property
            # is lazy), so a resumed job finds its worker again by name exactly as the noise job does.
            self._jobmgr.register_worker("invoke_faculty",
                                         lambda bucket, cache: self.invoke(cache["name"], cache["args"]))
        return self._jobmgr

    def job_submit(self, name, args=None, job_id=None, background=True):
        """Run ANY public faculty as a background JOB: returns a job_id you poll with job_status(id) and read with
        job_result(id) once it is 'done'. `m.job_submit("render_mesh", {...})` -- the generic twin of
        bake_cloud_job, which could only background ITS OWN bake.

        WHY: job_list/status/result/cancel/pause/resume all existed and worked, but nothing could START an
        arbitrary faculty -- background=True was a kwarg only bake_cloud_job happened to accept, so a client's
        "run this async" toggle worked for exactly one method. The job machinery is a checkpointed monoid fold, so
        this is plumbing: one bucket, the `first` (identity) reduce, a worker that calls mind.invoke.

        HONEST LIMITS, because a job that lies about what it can do is worse than a blocking call:
          * ATOMIC. One bucket, so progress is 0 then 1, and job_pause/job_resume cannot split the call -- they
            act at bucket boundaries and there is only one. Poll job_status; do not expect a partial render.
          * `args` should be JSON-safe if you want the job to survive a process restart (the checkpoint is JSON).
            A live object works in-process -- coercion (holographic_coerce) means dicts are usually enough --
            but it will not persist. This method does not silently drop persistence: it runs either way, and the
            restart-survival property is the caller's to want.
          * Dispatch rules are mind.invoke's: public names only, ValueError on private/unknown -- raised HERE, at
            submit, not swallowed into a failed job you have to poll to discover.
        See holographic_jobs.JobManager and holographic_distribute.reduce_first."""
        if not name or not isinstance(name, str) or name.startswith("_"):
            raise ValueError("invalid or private faculty name: %r" % (name,))
        if not callable(getattr(self, name, None)):
            raise ValueError("no such faculty: %r" % (name,))
        import uuid
        job_id = job_id or ("invoke-%s-%s" % (name, uuid.uuid4().hex[:8]))
        mgr = self._job_manager
        mgr.create(job_id, buckets=[0], worker="invoke_faculty", reduce="first",
                   cache={"name": name, "args": args or {}}, meta={"faculty": name})
        mgr.start(job_id, background=background)
        return job_id

    def bake_cloud_job(self, center=(0.0, 0.0, 0.0), radius=1.0, seed=0, grid=32, octaves=4, gain=0.58,
                       n_buckets=8, job_id=None, background=True):
        """Start the SLOW part of make_cloud (the fBm noise bake -- grid=32 ~60s) as a real background JOB you can
        monitor, pause, and resume -- rather than a blocking call. Returns the job_id; poll with job_status(id),
        pause with job_pause(id), continue with job_resume(id), and once status is 'done', job_result(id) is the
        baked (grid,grid,grid) array -- feed it to cloud_field(..., noise_grid=that_array) to render without
        re-baking. Checkpoints are written to .lecore_jobs/, so the job survives even a process restart: reopen a
        mind, call job_resume(id), and it continues from where it paused. See holographic_renderjobs for the
        bucketing scheme (independent z-slice bands, assembled by a sum-reduce)."""
        from holographic.scene_and_pipeline.holographic_renderjobs import make_noise_bake_job
        r_field = radius * 1.3
        bounds = [(center[i] - r_field * 1.8, center[i] + r_field * 1.8) for i in range(3)]
        _, job_id = make_noise_bake_job(bounds, grid=grid, octaves=octaves, gain=gain, seed=seed,
                                        n_buckets=n_buckets, job_id=job_id, manager=self._job_manager)
        self._job_manager.start(job_id, background=background)
        return job_id

    def job_status(self, job_id):
        """{id, status: created/running/paused/done/cancelled/failed, progress in [0,1], done, total, error} for
        any job started with bake_cloud_job (or any job on the shared manager)."""
        return self._job_manager.status(job_id)

    def job_pause(self, job_id):
        """Ask a running job to pause at its next checkpoint boundary (waits for it to actually stop). Safe to
        call from another thread/process context; the job's progress is preserved and resumable."""
        self._job_manager.pause(job_id)
        return self._job_manager.status(job_id)

    def job_resume(self, job_id, background=True):
        """Continue a paused job from where it left off (only the remaining buckets run)."""
        self._job_manager.resume(job_id, background=background)
        return self._job_manager.status(job_id)

    def job_cancel(self, job_id):
        """Stop a job for good (not resumable) at its next checkpoint boundary."""
        self._job_manager.cancel(job_id)
        return self._job_manager.status(job_id)

    def job_result(self, job_id):
        """The finished job's result (for bake_cloud_job: the baked (grid,grid,grid) noise array). Raises if the
        job isn't done yet -- check job_status(job_id)['status'] == 'done' first."""
        return self._job_manager.result(job_id)

    def job_list(self):
        """{job_id: status} for every job on the shared manager (this process's, plus any reopened from
        .lecore_jobs/ via job_status on a known id)."""
        return {jid: self._job_manager.status(jid) for jid in self._job_manager.jobs}

    # -- the NAVIGATOR: the creature, repurposed to search the data tree with a LEARNED adaptive budget ---------
    # State lives on the mind (a trained agent is expensive to build), and every method takes/returns plain data.
    def train_navigator(self, items, queries=1500, leaf_size=48, max_regions=16, noise=0.5, hot_size=48, seed=0):
        """Train a learned navigator over `items` ((N, D) vectors) and hold it on this mind. Returns the world's
        shape: {items, regions, depth, train_queries}.

        WHY A LEARNED NAVIGATOR instead of the tree's built-in routing: a fixed beam spends the SAME effort on every
        query -- b regions read whether the cue lands cleanly in one or straddles a boundary -- so it must be set
        wide enough for the hard minority and overpays on the easy majority. The navigator reads a region, senses
        how confident the answer looks, and decides arrive-or-keep-moving.

        MEASURED against the tree's own fixed-beam curve (1,200 items, 32 regions, 250 eval queries): the navigator
        reaches 98.0% recall at 173 comparisons. The cheapest fixed beam that matches that recall is beam 12, at 450
        comparisons -- 2.6x more. And at the navigator's own budget the best fixed beam (4) reaches only 81.6%. Both
        readings agree, which is the point of quoting both. See holographic_navigator."""
        import numpy as _np
        from holographic.agents_and_reasoning.holographic_navigator import DataWorld, Navigator, train
        from holographic.misc.holographic_creature import CreatureEncoder, HolographicMind
        arr = _np.asarray(items, float)
        dim = int(arr.shape[1])
        world = DataWorld(arr, leaf_size=int(leaf_size), seed=int(seed), max_regions=int(max_regions),
                          noise=float(noise))
        enc = CreatureEncoder(dim, seed=seed + 1)
        agent = HolographicMind(dim, DataWorld.ACTIONS, k=12, epsilon=0.3, novelty_bonus=0.1,
                                memory_cap=4000, seed=seed + 3)
        train(world, enc, agent, queries=int(queries), seed=seed + 1)
        self._navigator_obj = Navigator(world, enc, agent, hot_size=int(hot_size))
        st = world.tree.stats()
        return {"items": int(arr.shape[0]), "regions": int(st["leaves"]), "depth": int(st["depth"]),
                "train_queries": int(queries)}

    def _require_navigator(self):
        nav = getattr(self, "_navigator_obj", None)
        if nav is None:
            raise ValueError("no navigator: call mind.train_navigator(items) first (training is expensive, so it "
                             "is an explicit step rather than a lazy one)")
        return nav

    def navigator_find(self, cue, explain=False):
        """Search the trained navigator's data tree for `cue` (a (D,) vector): {index, comparisons, trace}.
        A ReflexCache fronts it, so a FAMILIAR query is recognised instantly and only unfamiliar ones pay for the
        deeper search -- it gets faster at whatever you ask for most. See holographic_navigator.Navigator.find."""
        import numpy as _np
        idx, comps, trace = self._require_navigator().find(_np.asarray(cue, float), explain=explain)
        return {"index": int(idx), "comparisons": int(comps), "trace": list(trace)}

    def navigator_benchmark(self, queries=250, beams=(1, 2, 4, 8, 12, 16), seed=999):
        """The honest comparison, agent-callable: {recall, comparisons, fixed_beams:[{beam, recall, comparisons}]}.
        The fixed-beam curve is the tree's OWN routing at a range of budgets -- the strongest baseline in the
        original space, not a strawman. See holographic_navigator.evaluate / fixed_beam_curve."""
        from holographic.agents_and_reasoning.holographic_navigator import evaluate, fixed_beam_curve
        nav = self._require_navigator()
        rec, comps = evaluate(nav.world, nav.encoder, nav.mind, queries=int(queries), seed=int(seed))
        rows = fixed_beam_curve(nav.world, beams=tuple(beams), queries=int(queries), seed=int(seed))
        return {"recall": float(rec), "comparisons": float(comps),
                "fixed_beams": [{"beam": int(r["beam"]), "recall": float(r["recall"]),
                                 "comparisons": float(r["comparisons"])} for r in rows]}

    # -- the ENCYCLOPEDIA: relational knowledge over concepts (is_a taxonomy + has parts) --------------------
    # A lazily-built Encyclopedia held ON THE MIND. It stores hypervectors, so it is a LIVE object -- but unlike a
    # bake or a Simulation it does not need a stateless twin, because the state lives here and every method below
    # takes and returns PLAIN DATA (strings, floats, lists). A long-lived service therefore accumulates knowledge
    # across /invoke calls, which is exactly what a knowledge layer is for. Build it once with encyclopedia_reset()
    # if you want a different dim/seed.
    #
    # NAMED `_encyclopedia_faculty`, NOT `_encyclopedia` -- and the name is load-bearing. The curriculum layer
    # (learn_encyclopedia, above) had already claimed the plain attribute `self._encyclopedia` for its
    # {concept: {role: filler}} dict, and answer() uses that attribute's mere EXISTENCE as its "an encyclopedia
    # was taught" flag. A lazily-instantiating property of the same name broke BOTH sides at once:
    # learn_encyclopedia raised AttributeError (a property has no setter), and -- the quiet half -- the flag was
    # pinned permanently True, so a mind that had learned NOTHING answered "is a dog an animal?" with a
    # confident is_a=False at throughput 1.0 instead of abstaining. The fabricated answer is the worse bug,
    # because the crash is the one that gets fixed. Two faculties, two names. (Kept negative: do NOT "fix" this
    # by adding a setter -- that silences the crash and leaves the fabrication.)
    @property
    def _encyclopedia_faculty(self):
        if getattr(self, "_encyclopedia_obj", None) is None:
            from holographic.agents_and_reasoning.holographic_encyclopedia import Encyclopedia
            self._encyclopedia_obj = Encyclopedia(dim=self.dim, seed=self.seed)
        return self._encyclopedia_obj


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p10_unproject_depth", "_UnifiedPart10")
    print("holographic_unified_p10_unproject_depth selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

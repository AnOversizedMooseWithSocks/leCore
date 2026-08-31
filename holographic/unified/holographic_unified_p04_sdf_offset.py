"""Part 04 of UnifiedMind's faculty surface -- 126 methods, sdf_offset .. triage_code.

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


class _UnifiedPart04:

    # -- CCD: speculative margins + conservative advancement, both free from the SDF (Box3D B4 / backlog X4) --
    def sdf_offset(self, sdf_eval, margin):
        """The SPECULATIVE CONTACT MARGIN, and it costs one subtraction: enlarging a collider by `margin` is just
        `sdf(P) - margin`. Returns the offset callable. Kept negative: a margin DETECTS proximity, it does not
        PREVENT tunnelling -- a body that crossed a thin wall in one step is resolved out the WRONG side, because a
        point sample has no memory of the swept path. Use time_of_impact for that. See holographic_collide.sdf_offset."""
        from holographic.simulation_and_physics.holographic_collide import sdf_offset as _so
        return _so(sdf_eval, margin)

    def time_of_impact(self, X, V, dt, sdf_eval, radius=0.0):
        """CONTINUOUS COLLISION DETECTION by conservative advancement: (hit, toi, contact) for points X moving at V
        over dt against `sdf_eval` offset by `radius`. The core query -- 'how far can I move without hitting
        anything?' -- IS the SDF value, so this is sphere tracing, and it DELEGATES to the renderer's
        raymarch.sphere_trace. Same march that renders a pixel; same distance query Walk-on-Spheres steps by. There
        is no dedicated CCD pass. See holographic_collide.time_of_impact."""
        from holographic.simulation_and_physics.holographic_collide import time_of_impact as _toi
        return _toi(X, V, dt, sdf_eval, radius=radius)

    def advance_ccd(self, X, V, dt, sdf_eval, radius=0.0, restitution=0.0):
        """Advance points one step WITHOUT tunnelling: sweep to first contact, stop there, cancel the into-surface
        velocity (`restitution` bounces). Measured: a 30 m/s body stepping 0.5 m per frame passes clean through a
        0.1 m wall under discrete resolution and is stopped exactly on it here. See holographic_collide.advance_ccd."""
        from holographic.simulation_and_physics.holographic_collide import advance_ccd as _acc
        return _acc(X, V, dt, sdf_eval, radius=radius, restitution=restitution)

    def classify_contact(self, overlap, velocity, restitution, margin=0.1, **bins):
        """Name a contact's TYPE (bounce / slide / rest_contact / penetration / jam) from its {overlap,
        velocity, restitution}: bins the continuous scalars to categories, then match_record against the
        contact-type records + decide_or_abstain. A LABELING/DISPATCH layer over advance_ccd's numerics (not a
        replacement) -- so a solver can pick a per-type response and log a self-explaining reason. Returns
        {'type', 'confident', 'record', 'ranked'}. See holographic_collide.classify_contact."""
        from holographic.simulation_and_physics.holographic_collide import classify_contact
        return classify_contact(overlap, velocity, restitution, mind=self, margin=margin, **bins)

    # -- PURITY & EFFECT ANALYSIS from the stdlib (backlog K6): the gate a shape-keyed cache needs -----------
    def function_purity(self, source, name):
        """Is the module-level function `name` in `source` provably PURE? CONSERVATIVE: unknown means False. A
        wrong 'impure' costs a cache miss; a wrong 'pure' silently corrupts a cache and everything downstream, so
        an unresolved callee, an unrecognised method and an attribute write are all impure.
        See holographic_pycontext.is_pure."""
        from holographic.io_and_interop.holographic_pycontext import is_pure
        return bool(is_pure(str(source), str(name)))

    def purity_report(self, source):
        """Purity verdicts for every module-level function in `source`, closed over the CALL GRAPH: a function that
        calls an impure function is impure, however clean its own body. Returns {pure, impure, total, fraction,
        local_only_fraction, verdicts, reasons}.

        `local_only_fraction` is what a rule that IGNORES calls would have reported -- carried beside the sound
        number so the flattering half cannot be quoted alone. Measured on this tree: local 54.3%, sound 32.1% (the
        backlog's 76% is a local-rule figure, and a local purity rule is unsound for a cache).
        See holographic_pycontext.purity_report."""
        from holographic.io_and_interop.holographic_pycontext import purity_report as _pr
        return _pr(str(source))

    def purity_scan(self, root="holographic"):
        """Run the purity analysis over a whole tree, merging every module's functions into ONE call graph so a
        call to another module's pure helper resolves. Same shape as purity_report.
        See holographic_pycontext.scan_tree."""
        from holographic.io_and_interop.holographic_pycontext import scan_tree
        return scan_tree(str(root))

    # -- RECURSIVE FACTORING over learned chunk levels (backlog R2; R3's second consumer of the codebook) -----
    def map_codebook(self, n_codes, dim, seed=0):
        """A deterministic codebook of `n_codes` random bipolar (+/-1) vectors of length `dim`. Regenerated from
        the seed, so an agent ships a SEED rather than a megabyte of vectors -- determinism instead of storage.
        See holographic_resonator.map_codebook."""
        from holographic.misc.holographic_resonator import map_codebook as _mc
        return _mc(int(n_codes), int(dim), int(seed))

    def map_bind(self, *vectors):
        """MAP binding: elementwise product. Commutative AND self-inverse (bind(x, x) is the all-ones vector),
        which is what makes stable factorization possible -- and what makes the recoverable object a multiset
        modulo cancelling pairs. See holographic_resonator.map_bind."""
        from holographic.misc.holographic_resonator import map_bind as _mb
        return _mb(*[np.asarray(v, float) for v in vectors])

    def climb_ladder(self, corpus, lens="sequence", max_depth=8, min_gain=0.02):
        """Climb a CORPUS into a TOWER of abstraction levels (the abstraction ladder): consolidate -> find
        patterns -> promote to a new alphabet -> repeat, STOPPING when the MDL gain drops below `min_gain` (a
        fraction of current bits). Returns the tower (level dicts with depth, alphabet size, bits, gain, stable
        hashlib atom ids, and the terminal refusal reason). The generic form of the seven-step loop run by hand
        (letters->words, verts->parts->scene, transforms->grammar). `lens` picks adjacency: 'sequence' = position
        (ordered streams), 'structure' = shared group (order-INDEPENDENT set promotion -- recurring
        sub-assemblies in any order, the instanced-scene case). The lens is the choice of what counts as adjacent,
        never guessed: shuffled parts are structureless to 'sequence' but highly compressible to 'structure'. A
        ladder that tops out shallow is a RESULT (most data does), logged loudly. See holographic_ladder.climb."""
        from holographic.agents_and_reasoning.holographic_ladder import climb
        # INPUT GUARD (defect 5.4). `climb` wants list[list[int]] -- a corpus of token-id sequences. Handed a
        # list of STRINGS it died 100 lines deep in a private helper with
        # "TypeError: unsupported operand type(s) for -: 'str' and 'str'", which names neither the argument
        # nor the expectation. This faculty is reachable from /invoke, where free-form data is the DEFAULT
        # case, so the guard belongs at the boundary rather than in the algorithm.
        if not isinstance(corpus, (list, tuple)) or not corpus:
            raise TypeError("climb_ladder(corpus) wants a non-empty list of sequences, got %r"
                            % type(corpus).__name__)
        first = corpus[0]
        if isinstance(first, str) or not isinstance(first, (list, tuple)):
            raise TypeError(
                "climb_ladder(corpus) wants list[list[int]] -- a corpus of TOKEN-ID sequences -- but the "
                "first item is %r. Map your symbols to integer ids first (e.g. "
                "[[vocab[w] for w in doc] for doc in docs]); a list of strings is the common mistake."
                % type(first).__name__)
        return climb(corpus, lens=lens, max_depth=max_depth, min_gain=min_gain)

    def ladder_summary(self, tower):
        """A compact human-readable summary of a climbed tower (from climb_ladder): per level its depth, alphabet
        size, bits, and gain over the previous level, plus the terminal refusal reason. For logging a climb result
        honestly. See holographic_ladder.tower_summary."""
        from holographic.agents_and_reasoning.holographic_ladder import tower_summary
        return tower_summary(tower)

    def identify_level(self, corpus, max_merges=100, min_count=2):
        """'What am I looking at?' -- classify a CORPUS by which ladder operations pay on it, returning a
        signature of MEASUREMENTS (not a label): `compressible` (is there a level above?), `gain_over_null`
        (compression that survives a shuffle -- high-D noise has basins too, so only this counts), `lens`
        (sequence vs structure, PICKED not guessed by which adjacency compresses more), `regime`
        (repetitive / nested-structured / irreducible -- Wolfram's taxonomy as a measurement), and
        `compress_sensitivity` (Quilez Q6: how hard the structure resists, free from two passes). The step-0
        question of a climb, and a faculty of its own -- agents ask 'what is this' constantly. See
        holographic_ladder.identify_level."""
        from holographic.agents_and_reasoning.holographic_ladder import identify_level
        return identify_level(corpus, max_merges=max_merges, min_count=min_count)

    def ladder_kit_report(self, level):
        """Report a ladder LEVEL's invariant-kit slots (from a climb_ladder tower) as WIRED, DECLARED-NEGATIVE, or
        SILENT-GAP. Returns {'wired', 'declared_negative', 'silent_gaps'}. Every level should have 0 silent gaps --
        each of delta/chunk/scale/decompose/cleanup/bake/seed/tile/lod/canonical/cache/fuse/superpose/guide is
        either wired to a primitive or a declared negative with a reason. See holographic_ladder.kit_report."""
        from holographic.agents_and_reasoning.holographic_ladder import kit_report
        return kit_report(level)

    def adaptive_pipeline(self, corpus, null_margin=0.05, max_depth=8, min_gain=0.02):
        """MEASUREMENT-DRIVEN adaptive dispatcher: run identify_level, then route the data to the method its
        REGIME names instead of hard-coding one. ABSTAINS on null-indistinguishable input (the SETI gate -- never
        'clean' noise into a fabricated signal); FOLDS repetitive data (Quilez -- cheap, never pay for a climb);
        CLIMBS nested structure with the lens picked per-signal (Puckette -- the lens is the analysis window).
        Returns {method: abstain|fold|climb|store_raw, regime, lens, gain_over_null, reason, tower/dominant}. A
        readable, refusable dispatch on numbers already computed -- no new learner. See
        holographic_ladder.adaptive_pipeline."""
        from holographic.agents_and_reasoning.holographic_ladder import adaptive_pipeline
        return adaptive_pipeline(corpus, null_margin=null_margin, max_depth=max_depth, min_gain=min_gain)

    def sweep_directions(self, corpus, max_depth=6, min_gain=0.02):
        """The UP / DOWN / SIDEWAYS completeness sweep (holographic_ladder): does the ladder's structure-finding
        hold in all three directions, or only one? DOWN -- does structure survive DECOMPOSITION (are the parts
        themselves analyzable)? UP -- does it survive EMBEDDING in a larger corpus? SIDEWAYS -- which lens COSTUMES
        (sequence/structure) does the data wear? Returns a per-direction ok flag with the measurement behind it,
        plus `gaps` (failed directions) and `complete`. Null-aware: an irreducible corpus flags all three, never
        fabricating structure. A capability that works in only one direction is an INCOMPLETE faculty -- this makes
        that check runnable, like kit_gaps did for the invariant kit. See holographic_ladder.sweep_directions."""
        from holographic.agents_and_reasoning.holographic_ladder import sweep_directions
        return sweep_directions(corpus, max_depth=max_depth, min_gain=min_gain)

    def ladder_predict(self, history, order=2, max_merges=200, min_count=2, null_margin=0.02, seed=0):
        """Predict what comes NEXT after `history` using the ladder's learned HIERARCHICAL alphabet (the
        compression<->prediction duality -- a good compressor is a good predictor). Learns a chunk codebook over
        the history, predicts the next CHUNK by matching the context against continuation counts over the promoted
        alphabet, and decodes it to raw symbols -- so one step can emit a whole learned pattern, not one flat
        symbol. THE GATE (SETI, on the time axis): abstains to the persistence baseline ('next = last') when the
        ladder does not beat persistence on held-out continuations -- a forecast that cannot beat 'same as last' is
        a null result, said loudly. Returns {prediction, confidence, method, beats_persistence, reason}. See
        holographic_ladder.ladder_predict."""
        from holographic.agents_and_reasoning.holographic_ladder import ladder_predict
        return ladder_predict(history, order=order, max_merges=max_merges, min_count=min_count,
                              null_margin=null_margin, seed=seed)

    def ladder_forecast_calibrated(self, series, order=2, alpha=0.1, min_history=12, seed=0):
        """Forecast the next value of a numeric `series` with the ladder predictor, wrapped in a CALIBRATED
        prediction interval (an uncalibrated forecast is not a measurement). Rolls ladder_predict over the series
        to gather residuals on data it did not fit, calibrates a conformal forecaster on them, and returns the
        next-step POINT forecast plus an interval with MEASURED coverage -- not an assumed one. Returns {point,
        interval, half_width, coverage, empirical_coverage, n_calibration, reason}. Falls back to point-only when
        the history is too short to calibrate honestly. See holographic_ladder.ladder_forecast_calibrated."""
        from holographic.agents_and_reasoning.holographic_ladder import ladder_forecast_calibrated
        return ladder_forecast_calibrated(series, order=order, alpha=alpha, min_history=min_history, seed=seed)

    def extend_generator(self, fit_result, n_ahead, original_length):
        """FORECAST by playing a fitted generator PAST its data (store the formula, play the future). Given a
        fit_deterministic result and the original data length, regenerate `n_ahead` samples beyond the end.
        Returns {forecast, t_range, valid}; `valid` is False (samples still returned) when extrapolating far past
        the fit's validated window -- a generator fit on [0,1] evaluated at t=100 is confident nonsense, so it
        refuses beyond where it was validated. A refused fit cannot be extended. See
        holographic_fitgen.extend_generator."""
        from holographic.agents_and_reasoning.holographic_fitgen import extend_generator
        return extend_generator(fit_result, n_ahead, original_length)

    def reconstruct_tower(self, tower):
        """Expand a climbed ladder TOWER back to its ORIGINAL corpus of base symbols -- the inverse of
        climb_ladder (found by the down-sweep: a tower you cannot decompress is useless). For a SEQUENCE-lens
        tower this is LOSSLESS: reconstruct_tower(climb_ladder(corpus)) == corpus exactly. For a STRUCTURE-lens
        tower it recovers the SET of base part-types per group (order and duplicate counts are dropped by design --
        the structure lens's premise). See holographic_ladder.reconstruct."""
        from holographic.agents_and_reasoning.holographic_ladder import reconstruct_corpus
        return reconstruct_corpus(tower)

    def chart_space(self, alphabet, rays=64, steps=24, margin=0.1, band_keep=0.5, seed=0):
        """Chart a holographic ALPHABET as a measured atlas -- march rays between atoms and record where they
        enter cleanup BASINS (nearest atom distinctively nearer than the runner-up by `margin`). Returns
        hit_fraction, basins_per_ray, coverage (dead atoms never win), and the honest verdict structure_over_null
        (basin coverage MINUS a band-limited random-alphabet null, Quilez Q8). A region is structure only ABOVE
        the matched null -- high-D noise has basins too, so an atlas without its null is a Rorschach test. Uses for
        the number: capacity forecasting (shrinking coverage predicts an auto_scale trigger) and codebook
        placement (put new atoms in dead zones). See holographic_ladder.chart_space."""
        from holographic.agents_and_reasoning.holographic_ladder import chart_space
        return chart_space(alphabet, rays=rays, steps=steps, margin=margin, band_keep=band_keep, seed=seed)

    def bank_or_formula(self, eval_cost_us, hit_rate, n_entries, bytes_per_entry, lookup_cost_us=0.5,
                        regen_from_seed=True):
        """Decide whether to BANK computed values or keep the FORMULA and regenerate on demand (Quilez Q1, 'store
        the formula not the samples'). The demoscene economy as a MEASURED gate: banking pays iff
        hit_rate*eval_cost - lookup_cost > 0 (a miss must BUILD the entry, so only reused evals amortize the bake;
        break-even hit_rate = lookup/eval). A bank of things a cheap formula gives you for free is NEGATIVE
        storage. Returns {bank, saving_us_per_query, storage_bytes, reason}. The reusable wheel the transform bank,
        the generator bank, and any cache should consult before banking by reflex. See
        holographic_ladder.bank_or_formula."""
        from holographic.agents_and_reasoning.holographic_ladder import bank_or_formula
        return bank_or_formula(eval_cost_us, hit_rate, n_entries, bytes_per_entry,
                               lookup_cost_us=lookup_cost_us, regen_from_seed=regen_from_seed)

    def browse_capabilities(self, prefix="", by="location"):
        """Browse the capability namespace like a CONTEXT MENU. `by='location'` (default) walks the physical family
        tree: browse('') -> families, 'mesh_and_geometry/' -> modules, 'mesh_and_geometry/sdf/' -> leaf functions
        (the name IS the hierarchy, so it can't drift from the code). `by='semantic'` walks the File->Export->PNG
        VERB tree instead (select/ transform/ create/ ...) -- grouped by what a user DOES, not where the code lives;
        only capabilities with a semantic tag appear (see SEMANTIC_TAXONOMY.md). Returns {branch: leaf_count}. See
        holographic_capuri.browse / browse_semantic."""
        if by == "semantic":
            # Pass THIS MIND's catalog, not the module-level default. The faculty used to call browse_semantic(prefix)
            # bare, which falls back to default_catalog() -- the ~400 CURATED entries only. So the verb menu could
            # never show a mind's own auto-registered faculties no matter how well tagged they were: the tags lived
            # on one catalog and the menu read another. Found while raising tag coverage from 5.2% -> 31%; the new
            # tags were invisible until this line changed.
            from holographic.caching_and_storage.holographic_capuri import browse_semantic
            return browse_semantic(prefix, catalog=self._capability_catalog())
        from holographic.caching_and_storage.holographic_capuri import browse
        return browse(prefix)

    def invoke(self, name, args=None):
        """Call one PUBLIC faculty BY NAME with a dict of args -- the dispatch every non-HTTP client was
        re-implementing. `m.invoke("double", {"x": 21}) -> 42`. A private name (leading underscore) or an unknown
        or non-callable name raises ValueError rather than returning something a caller might mistake for a
        result. `args` may be a dict (kwargs) or a list/tuple (positional); None means no arguments.

        WHY IT EXISTS: this logic lived ONLY inside holographic_service.Service._invoke, so every other client --
        a node pack, a harness, an agent -- copied it and each copy could drift from ours. A downstream audit
        reported exactly that (their runtime.invoke mirrored our semantics by hand and already checks
        hasattr(mind, "invoke") to hand the job back the moment this landed). Returns the RAW result: JSON
        coercion is the service's boundary concern, not this method's, so in-process callers keep real objects."""
        if not name or not isinstance(name, str) or name.startswith("_"):
            raise ValueError("invalid or private faculty name: %r" % (name,))
        fn = getattr(self, name, None)
        if not callable(fn):
            raise ValueError("no such faculty: %r" % (name,))
        if args is None:
            return fn()
        if isinstance(args, dict):
            return fn(**args)
        if isinstance(args, (list, tuple)):
            return fn(*args)
        raise ValueError("args must be a dict, a list/tuple, or None -- got %r" % type(args).__name__)

    def features(self, names=None):
        """Which faculties THIS build has: `{name: bool}`. `m.features(["pipeline_map", "io_kinds"])` answers a
        preflight in one call; `m.features()` (no argument) returns every public faculty mapped to True, i.e. the
        whole callable surface as data.

        WHY: a downstream node pack hardcoded a list of faculty names to preflight and noted "that list will rot".
        It will -- and worse, it rots SILENTLY, because a missing faculty and a renamed one look identical from
        the outside (both are just an absent attribute). Asking the engine is the fix; the answer cannot go stale
        because it is computed from the live object.

        Pair it with `mind.version()` for the build identity. Names starting with '_' are always False: private
        faculties are not part of the contract, and a client that discovers one has found a footgun, not a
        feature."""
        if names is None:
            return {n: True for n in sorted(dir(self))
                    if not n.startswith("_") and callable(getattr(type(self), n, None))}
        if isinstance(names, str):
            names = [names]
        return {n: (not str(n).startswith("_")) and callable(getattr(self, n, None)) for n in names}

    def version(self):
        """This build's identity as data: `{engine, capabilities_schema, dim, seed}`. The companion to
        features() -- features() says WHAT is here, this says WHICH BUILD it is, so a client can log or gate on a
        version instead of sniffing for methods. `capabilities_schema` is the contract version of
        capabilities.json / describe_skill records; it moves only when that shape changes."""
        import lecore as _lc
        # 1.1: THE STEP AND MANIFEST FORMATS MOVED. A schema version that never
        # changes is a field clients learn to ignore, and this one sat at "1.0"
        # through +900 capabilities, the `method` field, `primary`/`params` and
        # memoisation. It moves now because two CONTRACTS changed in ways a
        # client can see:
        #   suggest_pipeline steps gained `method` -- they were
        #     {consumes, name, produces} and a planner client could not execute
        #     one without re-deriving the mapping;
        #   render edges now declare their SECOND input, so consumes can carry
        #     ("mesh", "camera") where a client may have assumed one kind.
        # Additive both times, which is why this is 1.1 and not 2.0 -- an old
        # client that ignores `method` and reads consumes[0] still works.
        return {"engine": getattr(_lc, "__version__", None), "capabilities_schema": "1.1",
                "dim": int(self.dim), "seed": int(getattr(self, "seed", 0))}

    def semantic_tag_coverage(self):
        """How much of THIS mind's action menu is visible: {'total','tagged','untagged','pct'}. browse_capabilities
        (by='semantic') omits untagged capabilities, so this is the number that decides whether the File->Export->PNG
        verb tree can see the engine at all. It was 108/2095 (5.2%) -- every auto-registered faculty arrived
        untagged -- until the tag was DERIVED at registration (see holographic_semantictag.infer_semantic).
        The remainder is not a to-do list: module-name entries abstain BY DESIGN (a module is not an action), and
        noun-named faculties (svg_canvas, planet_field) have no verb to file under. See holographic_semantictag."""
        from holographic.caching_and_storage.holographic_semantictag import coverage
        return coverage(self._capability_catalog())

    def infer_semantic_tag(self, name, doc=""):
        """Infer the taxonomy tag ('root/sub') a capability name would file under, or None if no verb matches --
        the same deterministic rule seed_from_mind applies at registration. Use it when adding a capability to see
        which menu branch it will land in. ABSTAINS rather than guess: a wrong branch files a capability under a
        verb nobody looks for and, unlike a missing tag, looks done. See holographic_semantictag.infer_semantic."""
        from holographic.caching_and_storage.holographic_semantictag import infer_semantic
        return infer_semantic(name, doc)

    def capability_collisions(self):
        """Every bare function name that resolves to more than one capability URI -- the semantic collisions, each
        with its disambiguating full paths. The name_collisions audit re-read through the URI lens: a collision is
        not a hazard to forbid but a name whose PATH you supply. See holographic_capuri.collisions."""
        from holographic.caching_and_storage.holographic_capuri import collisions
        return collisions()

    def resolve_capability_uri(self, name):
        """URI-ONLY -- returns [] for a plain FACULTY name like 'render_mesh'; use find_capability or
        describe_skill for those. (The obvious reading is "resolve any name", and a downstream integrator read it
        that way; this resolves LOCATION URIs -- module paths -- not method names.)
        Resolve a bare capability name or a partial path to the FULL capability URI(s) that match
        (holographic_capuri) -- 'rotation' -> ['mesh_and_geometry/meshskin/rotation',
        'scene_and_pipeline/scenegraph/rotation']; 'sdf/sphere' narrows to one. The disambiguation step when a name
        collides: supply more of the path. Pairs with browse_capabilities (the menu) and capability_collisions (the
        collision list). See holographic_capuri.resolve_uri."""
        from holographic.caching_and_storage.holographic_capuri import resolve_uri
        return resolve_uri(name)

    def chunk_levels(self, codebook, vocab):
        """The chunk depths a learned codebook can factor against, DEEPEST FIRST -- the ladder recursive_factor
        descends. See holographic_resonator.available_levels."""
        from holographic.misc.holographic_resonator import available_levels
        return [int(d) for d in available_levels(self.chunk_codebook(codebook), np.asarray(vocab, float))]

    def recursive_factor(self, composite, codebook, vocab, arity=2, restarts=10, iters=300):
        """Factor a DEEP composite by solving a SHALLOW problem over composed chunks, then expanding by lookup.
        Tries each chunk level deepest-first, verifies each candidate by RE-COMPOSITION, and falls back one level
        on failure -- so the answer is verified correct or reported unsolved, never a silent guess.

        MEASURED (D=4096, 32 symbols, MAP): the flat resonator falls off a cliff -- 60% at depth 4, 0% at depth 5
        and beyond. With PROMOTED chunks (62 pairs -> 64 quads) a depth-8 composite factors at 90.0% here versus
        0.0% flat, and 3x faster besides. Below the cliff recursion is a modest gain at 5x the cost (depth 4:
        93.3% vs 86.7% flat), so use it past the cliff. The condition is R1's: no structure, no dividend --
        mind.structure_score(stream) measures it first. `codebook` is a plain-data chunk codebook from
        mind.learn_chunks: R3's one codebook family, second consumer. See holographic_resonator.recursive_factor."""
        from holographic.misc.holographic_resonator import recursive_factor as _rf
        return _rf(np.asarray(composite, float), self.chunk_codebook(codebook), np.asarray(vocab, float),
                   arity=int(arity), restarts=int(restarts), iters=int(iters))

    def reduce_involution(self, leaves):
        """Reduce a leaf multiset modulo MAP's self-inverse binding: a leaf appearing twice CANCELS. Measured --
        factoring bind(v3, v7) against a pair codebook legitimately returns [0, 0, 3, 7], because
        bind(v0,v3)*bind(v0,v7) == v3*v7. Correct and non-minimal are different things.
        See holographic_resonator.reduce_involution."""
        from holographic.misc.holographic_resonator import reduce_involution as _ri
        return _ri(list(leaves))

    # -- W4: information-rate rendering -- shade the news, reproject the rest -------------------------------
    def refresh_renderer(self, first_frame, budget=0.20):
        """The reproject-and-refresh loop as a render mode: warp the previous frame forward and shade only the
        disocclusion border plus the OLDEST k pixels. MEASURED on a parallax-free procedural scene, 12 frames, 20%
        budget: 57.5 dB mean / 55.9 dB worst with a KNOWN camera shift -- five times fewer shader evaluations at
        visually-indistinguishable quality, tail slope +0.22 dB (stable). Pass `known_shift=` to `step`; recovering
        it from pixels costs a measured 10.5 dB and turns the tail slope to -9.52 (decay), because the loop warps
        its own output. See holographic_refresh.RefreshRenderer."""
        from holographic.rendering.holographic_refresh import RefreshRenderer
        return RefreshRenderer(np.asarray(first_frame, float), budget=float(budget))

    def exact_k_oldest(self, age, k):
        """Select EXACTLY k pixels with the greatest age, ties broken deterministically on the flat index. The
        threshold rule it replaces ("age >= the k-th largest") selects the WHOLE FRAME when ages are tied -- which
        they are on frame 0 -- reporting a perfect PSNR because it shaded everything. `argmax_tiebreak`'s lesson in
        a new place: a selection rule is an observable decision and its ties must be named.
        See holographic_refresh.exact_k_oldest."""
        from holographic.rendering.holographic_refresh import exact_k_oldest as _e
        return _e(np.asarray(age, float), int(k))

    def refresh_report(self, shade_at, n_frames=12, budget=0.20, known_shift=None):
        """Run the refresh loop and report {shaded_fraction, psnr_mean, psnr_worst, psnr_first, psnr_last,
        tail_slope, psnrs}. STABILITY IS `tail_slope`, not first-minus-last: frame 1 warps a PERFECT frame 0 and
        scores ~80 dB for free. See holographic_refresh.refresh_report."""
        from holographic.rendering.holographic_refresh import refresh_report as _r
        return _r(shade_at, n_frames=int(n_frames), budget=float(budget), known_shift=known_shift)

    # -- the brain/muscle contract, realised: the SCENE's SDF, emitted --------------------------------------
    def four_surface_demo(self, sdf_node, camera=None):
        """ONE KERNEL, FOUR SURFACES (W19): given a single SDF scene (node or DSL text), return its FOUR
        backend representations proving they are the same field -- {'glsl': Shadertoy source, 'wgsl': browser-GPU
        source, 'ascii': a braille raymarch, 'dsl': the canonical scene text}. The PNG path uses the same CPU eval
        the ascii path marches, and sdf_validate_c proves the emitted C matches that eval, so all four agree. The
        demo that explains the engine in one screen: author once, render everywhere. See sdf_dialect / to_glsl /
        ascii_sdf.

        `camera` is (eye (3,), forward (3,)); if omitted, an angled view from OUTSIDE the origin is used. WHY not
        the ascii default camera: a scene with `repeat` is an INFINITE lattice, and a camera embedded in it stares
        at a uniform wall (constant depth -> a flat, featureless ascii grid that does NOT show the scene -- the
        'camera inside an infinite lattice' trap). The default here sits back and looks in, so the ascii actually
        depicts the geometry."""
        import numpy as _np
        from holographic.mesh_and_geometry.holographic_sdf import parse_dsl, SDF
        from holographic.mesh_and_geometry.holographic_sdfemit import sdf_dialect as _sd
        node = sdf_node if isinstance(sdf_node, SDF) else parse_dsl(sdf_node)
        dsl = node.to_dsl()
        if camera is None:
            eye = _np.array([1.6, 1.2, 2.6])                    # outside, angled -- reads as a clear silhouette
            camera = (eye, -eye / _np.linalg.norm(eye))
        # WHY ramp, not braille: braille packs 2x4 dots per cell, which turns a busy scene into high-frequency
        # noise a human cannot read. A tonal `ramp` (light->dark chars) shows large-scale SHAPE, which is what an
        # ASCII surface is for. (The PNG carries fine detail; the ASCII carries silhouette.)
        return {"dsl": dsl,
                "glsl": node.to_glsl(),
                "wgsl": _sd(dsl, "wgsl"),
                "ascii": self.ascii_sdf(dsl, width=48, mode="ramp", camera=camera, fov=0.85)}

    def scene_cost(self, sdf_node):
        """Estimate the per-ray evaluation COST of an SDF scene (W2) -- an ALU/machine-model annotation for
        deciding if a scene raymarches in real time. Accepts an SDF node or DSL text. Returns a dict with `alu`
        (approx arithmetic ops per map() call), `nodes`, `depth`, `iterative` (contains a fractal/tiling loop),
        and a plain-language `verdict`. Know the price before you ship the scene. See holographic_sdf.SDF.cost."""
        from holographic.mesh_and_geometry.holographic_sdf import parse_dsl, SDF
        node = sdf_node if isinstance(sdf_node, SDF) else parse_dsl(sdf_node)
        return node.cost()

    def sdf_dialect(self, sdf_node, dialect="wgsl"):
        """Emit the SDF TREE's own `map(p) -> distance` in `wgsl` | `glsl` | `c_f64` | `c_f32` -- so the browser
        runs a PROJECTION of the authoritative Python scene, not a hand-written shader about a scene the engine
        never saw. `sdf.to_glsl()` already shipped and `emit_kernel` already shipped; THE TWO NEVER MET, so
        `payload('shader')` used to carry whatever text the caller passed. See holographic_sdfemit.sdf_dialect."""
        from holographic.mesh_and_geometry.holographic_sdfemit import sdf_dialect as _sd
        return _sd(sdf_node, dialect=str(dialect))

    def sdf_validate_c(self, sdf_node, points, dialect="c_f64"):
        """Compile the emitted C `map()` with `cc`, RUN it, and compare to the Python `_eval`. MEASURED on a compound
        tree over 200 points: c_f64 agrees to 6.7e-16 -- machine epsilon, and NOT bit-identical, because
        `np.linalg.norm` rescales to avoid overflow and sums in a different order than `sqrt(x*x+y*y+z*z)`; c_f32
        differs by 2.3e-07, which IS the tolerance a WGSL port is judged against.
        See holographic_sdfemit.validate_c."""
        from holographic.mesh_and_geometry.holographic_sdfemit import validate_c
        return validate_c(sdf_node, np.asarray(points, float), dialect=str(dialect))

    def sdf_emit_coverage(self):
        """Which SDF node kinds the dialect emitter handles and which it refuses. `emitted + refused == every kind`
        -- a gap here is a shader that silently omits geometry. `menger` and `repeat` fold the domain iteratively;
        `twist` and `displace` are inexact distance warps. See holographic_sdfemit.coverage."""
        from holographic.mesh_and_geometry.holographic_sdfemit import coverage
        return coverage()

    # -- realtime: draft frames, a refine pass, and a multi-format payload ----------------------------------
    def realtime_session(self, session, budget=0.20, shade_kwargs=None):
        """A reprojecting viewport over a `RenderSession`: `frame()` is a DRAFT that shades only the news,
        `refine()` traces every pixel, `payload(kinds)` pushes the same scene as pixels / mesh / splats / shader /
        lod, all JSON-safe. MEASURED: a 20% mask shades 3.2x faster and is BIT-IDENTICAL on the pixels it touches.
        KEPT NEGATIVE: pass `known_shift=(dy, dx)` -- recovering it from the pixels costs 2,280 extra traces, 3.7 dB,
        and a **-4.52 dB tail slope** (the loop warps its own output). See holographic_realtime.RealtimeSession."""
        from holographic.scene_and_pipeline.holographic_realtime import RealtimeSession
        return RealtimeSession(session, budget=float(budget), shade_kwargs=shade_kwargs)

    def frame_budget_controller(self, target_fps=60, ladder=None, headroom=0.15, start_level=None,
                                climb_after=8, climb_margin=0.6):
        """Build a FRAME-BUDGET CONTROLLER (holographic_framebudget) -- the one knob from a target FPS to concrete
        render + sim quality, held closed-loop against MEASURED frame time. Each frame call `.current()` for the
        quality preset to render/simulate with, then `.report(frame_ms)` with the measured time; the controller
        DROPS a quality level on a budget miss (react fast) and CLIMBS only after a streak of comfortable frames
        (hysteresis, so quality doesn't oscillate). This is the missing conductor tying render_adaptive,
        draft_vs_refine_simulation, and the LOD chains to a target frame rate for a front-end client. The ladder
        exposes render and sim quality SEPARATELY -- a coarse render is a draft of the fine one, but a coarse
        chaotic sim is a DIFFERENT trajectory (see draft_vs_refine_simulation), so you may hold the sim fixed and
        trade only render quality. See holographic_framebudget.FrameBudgetController."""
        from holographic.scene_and_pipeline.holographic_framebudget import FrameBudgetController
        return FrameBudgetController(target_fps=target_fps, ladder=ladder, headroom=headroom,
                                     start_level=start_level, climb_after=climb_after, climb_margin=climb_margin)

    def frame_budget_ms(self, target_fps, headroom=0.15):
        """Convert a target frame rate to a per-frame millisecond BUDGET, minus a headroom fraction (a frame that
        exactly fills 1/fps has already missed the vsync interval). 60 fps -> ~14.2 ms usable, 30 fps -> ~28.3 ms.
        See holographic_framebudget.frame_budget_ms."""
        from holographic.scene_and_pipeline.holographic_framebudget import frame_budget_ms
        return frame_budget_ms(target_fps, headroom=headroom)

    def workspace_manager(self):
        """A WORKSPACE MANAGER (holographic_workspace) over this mind's database -- durable user data coexisting
        with transient 3D/sim SCENES, each in its own namespace so they don't step on each other. Use it to SAVE
        and LOAD a workspace/scene: new_workspace(name), switch_workspace(name), export_workspace(name) -> a JSON
        blob, import_workspace(blob) -> rebuilds it BYTE-IDENTICALLY (the seed fixes every atom), combine_workspaces,
        reset_to_default (drop transient scenes, keep the durable tier). The persistence layer for a real-time
        client's scene. See holographic_workspace.WorkspaceManager."""
        from holographic.scene_and_pipeline.holographic_workspace import WorkspaceManager
        if getattr(self, "_workspace_manager", None) is None:
            self._workspace_manager = WorkspaceManager(self.db)
        return self._workspace_manager

    def save_container(self, sections, meta=None, compress=True):
        """Serialise a list of typed SECTIONS into one app-neutral CONTAINER FILE -> bytes (holographic_container).
        A section is {"kind", "id", "meta", "arrays": {name: ndarray}}; `meta` is optional top-level file metadata.
        The point: a section whose `kind` a reader does not understand round-trips UNTOUCHED, so an image editor, a
        3D app, and a video editor can all share ONE forward-compatible file -- each registers its own kinds, none
        owns the format. Numeric arrays only (no pickle, for safety); save->load->save is byte-identical. NOT the
        workspace_manager (that checkpoints a live DB by replay) -- this is the file format for shipping typed data
        between apps. See holographic_container.save_container."""
        from holographic.io_and_interop.holographic_container import save_container
        return save_container(sections, meta=meta, compress=compress)

    def load_container(self, data):
        """Inverse of save_container: container bytes -> {"meta", "sections": [{kind, id, meta, arrays}, ...]}
        (holographic_container). Sections come back in saved order with numeric arrays reconstructed; a kind this
        caller does not understand comes back exactly as stored, so a file survives a round-trip through an app that
        understands only SOME of its kinds -- the mechanism behind sharing one workspace across apps. Loads no
        pickle (runs no code). See holographic_container.load_container."""
        from holographic.io_and_interop.holographic_container import load_container
        return load_container(data)

    def frame_server(self, ladder=None, headroom=0.15):
        """Build a FRAME SERVER (holographic_framebudget) -- server-side real-time frame serving for front-end
        clients that PULL frames (the request/response form of a frame stream). Keeps one frame-budget controller
        PER SESSION; call `.next_frame(session, target_fps, last_frame_ms)` each frame to get the quality preset to
        render/simulate with, holding each client's target fps closed-loop. Two clients can run at different rates
        (a phone at 30, a desktop at 60). This is what the HTTP service's POST /frame endpoint delegates to. See
        holographic_framebudget.FrameServer."""
        from holographic.scene_and_pipeline.holographic_framebudget import FrameServer
        return FrameServer(ladder=ladder, headroom=headroom)

    def obs_capture_profile(self, base_url="http://127.0.0.1:5050/", preset="1080p", fps=30,
                            transparent=False, headroom=0.15):
        """The settings a streamer pastes into OBS to capture this canvas as a BROWSER SOURCE -- the realistic,
        in-constitution way to put leOS on a stream (OBS does the video encoding; the engine serves the page + the
        frames via /frame and /frame/stream). preset is '720p'/'1080p'/'1440p'/'4k' (match your OBS canvas so OBS
        does no scaling); transparent=True advertises a transparent background + the OBS custom CSS so the canvas
        composites over other sources. Returns {url, width, height, fps, frame_budget_ms, transparent, custom_css,
        obs_steps, note}. NOT an RTMP/NDI/virtual-camera encoder -- that needs ffmpeg/OS video I/O, outside the
        NumPy-only core. See holographic_framebudget.obs_capture_profile."""
        from holographic.scene_and_pipeline.holographic_framebudget import obs_capture_profile
        return obs_capture_profile(base_url=base_url, preset=preset, fps=fps,
                                   transparent=transparent, headroom=headroom)

    def synthetic_frame_source(self, kind="clock", size=(64, 64), frames=120, seed=0):
        """Build a pure-NumPy, DECODER-FREE reference FrameSource -- a deterministic synthetic clip (holographic_
        framesource) for demos/tests of the temporal seam without any cv2/ffmpeg. kinds: 'clock'/'gradient'/'bars'.
        seekable + pausable; advance()/seek()/pause() step it; get() -> (frame, seq). A host's real video source
        honours the SAME contract. See holographic_framesource.SyntheticFrameSource."""
        from holographic.io_and_interop.holographic_framesource import SyntheticFrameSource
        return SyntheticFrameSource(kind=kind, size=size, frames=frames, seed=seed)

    def map_frames(self, source, fn, cache=None):
        """The temporal DOOR: pull the current frame from a host-provided FrameSource (any object with get() ->
        (frame, seq)), apply fn(frame), and MEMOISE by seq so per-frame work runs once per distinct frame. Returns
        (out, seq). Pass a plain dict as `cache` (kept between calls) for the skip-recompute behaviour. Imports NO
        decoder -- cv2/ffmpeg live in the host's FrameSource; this is the seam for temporal NCA / optical flow /
        video colour transfer. See holographic_framesource.map_frames."""
        from holographic.io_and_interop.holographic_framesource import map_frames
        return map_frames(source, fn, cache=cache)

    def frame_key(self, source, prefix=""):
        """A deterministic (hashlib) cache key for a FrameSource's CURRENT frame -- `prefix` + the current seq. The
        signature-based invalidation seam: a per-frame result is valid exactly while this key is unchanged. See
        holographic_framesource.frame_key."""
        from holographic.io_and_interop.holographic_framesource import frame_key
        return frame_key(source, prefix=prefix)

    def is_frame_source(self, obj):
        """Duck-check whether `obj` honours the FrameSource contract (a callable get() returning a 2-tuple) -- so a
        HOST source that never imported leCore still qualifies. See holographic_framesource.is_frame_source."""
        from holographic.io_and_interop.holographic_framesource import is_frame_source
        return is_frame_source(obj)

    def pick_element(self, wireframe, screen_u, screen_v, want="vertex", cam_z=3.0, fov_scale=1.6):
        """VIEWPORT PICKING for a 3D-modeling app (holographic_framebudget): given a wireframe cage and a screen
        coordinate (screen_u, screen_v in -1..1 under the cursor), return which element the user is pointing at --
        {kind, index, distance, position/vertices} for the nearest 'vertex', 'edge', or 'face'. Projects the cage's
        own verts to the screen and finds the closest (exact, deterministic, no GPU pick buffer needed). The select
        step a modeling app needs before editing a vert/edge/face. See holographic_framebudget.pick_element."""
        from holographic.scene_and_pipeline.holographic_framebudget import pick_element
        return pick_element(wireframe, screen_u, screen_v, want=want, cam_z=cam_z, fov_scale=fov_scale)

    def mesh_selection(self, mesh, mode="vertex", indices=None):
        """A sub-object MESH SELECTION (holographic_meshselect) -- a persistent set of VERTS / EDGES / FACES with a
        mode and set algebra (add/remove/toggle/union/intersect/invert/select_all) plus mode CONVERSION
        (face->verts, verts->faces, ...). This is the edit-mode selection a modeling app operates every edit on,
        complementary to the object-level `selection` (which picks whole objects). Bind to a mesh {vertices, faces}
        so indices are validated. See holographic_meshselect.MeshSelection."""
        from holographic.mesh_and_geometry.holographic_meshselect import MeshSelection
        return MeshSelection(mesh, mode=mode, indices=indices)

    def select_edge_loop(self, mesh, seed_edge):
        """Select the EDGE LOOP through `seed_edge` (holographic_meshselect) -- the ring of edges continuing
        'straight' across quads, the Alt-click primitive users expect from Blender/Maya. Walks both ways from the
        seed, stopping at a pole or boundary (honest -- loops are only well-defined on quads). Returns an edge-mode
        MeshSelection. See holographic_meshselect.select_edge_loop."""
        from holographic.mesh_and_geometry.holographic_meshselect import select_edge_loop
        return select_edge_loop(mesh, seed_edge)

    def select_face_ring(self, mesh, seed_face):
        """Select the FACE RING from `seed_face` (holographic_meshselect) -- the band of quads a loop cut runs
        through, walking quad to quad across shared edges. Terminates at a non-quad or boundary. Returns a
        face-mode MeshSelection. See holographic_meshselect.select_face_ring."""
        from holographic.mesh_and_geometry.holographic_meshselect import select_face_ring
        return select_face_ring(mesh, seed_face)

    def select_boundary_loops(self, mesh):
        """Select the OPEN BOUNDARY edges of a mesh (holographic_meshselect) -- the edges used by exactly one face
        (a hole rim or open-surface border), the 'select the hole' step before filling or bridging. Returns an
        edge-mode MeshSelection. See holographic_meshselect.select_boundary_loops."""
        from holographic.mesh_and_geometry.holographic_meshselect import select_boundary_loops
        return select_boundary_loops(mesh)

    def soft_selection_weights(self, mesh, selection, radius, falloff="smooth"):
        """SOFT SELECTION as a reusable per-vertex WEIGHT FIELD (holographic_meshselect) -- 1 on the selection,
        falling off to 0 at `radius` along the surface (multi-source geodesic). This is proportional editing: a
        transform reads these weights and moves each vertex by weight*delta, dragging neighbours smoothly. Takes a
        MeshSelection or a raw vertex-index list; falloff is 'linear'/'smooth'/'sharp'. See
        holographic_meshselect.soft_selection_weights."""
        from holographic.mesh_and_geometry.holographic_meshselect import soft_selection_weights
        return soft_selection_weights(mesh, selection, radius, falloff=falloff)

    def proportional_edit(self, mesh, selection, translate, radius, falloff="smooth"):
        """PROPORTIONAL EDIT (Blender O + G): move the selected vertices by `translate` and drag their neighbours
        with a geodesic falloff -- one grab reshapes a whole region smoothly instead of moving every ring by hand.
        `selection` is a vertex-index list (or MeshSelection); `radius` sets the falloff reach along the surface;
        `falloff` is linear/smooth/sharp. Returns a new Mesh (topology unchanged). See
        holographic_meshselect.proportional_edit."""
        from holographic.mesh_and_geometry.holographic_meshselect import proportional_edit as _pe
        return _pe(self._as_mesh(mesh), selection, translate, radius, falloff=falloff)


    def select_symmetric(self, mesh, selection, axis=0, tol=1e-4):
        """SYMMETRY SELECTION (holographic_meshselect) -- add a selection's mirror-image elements across a world
        axis plane (axis 0/1/2 = x/y/z=0), so a symmetric edit can be applied to both sides. The selection-level
        complement to mirror_mesh (which mirrors GEOMETRY); here nothing is created, we find the counterpart
        elements that already exist, paired by reflected position within tol. See holographic_meshselect.select_symmetric."""
        from holographic.mesh_and_geometry.holographic_meshselect import select_symmetric
        return select_symmetric(mesh, selection, axis=axis, tol=tol)

    def select_in_box(self, mesh, lo, hi, mode="vertex", project=None):
        """REGION SELECT (holographic_meshselect) -- select every element inside the axis-aligned box [lo,hi], the
        box/rubber-band select of a viewport. Edge/face modes select if ANY vertex is in (inclusive, matching
        to_mode). Pass `project` (a view-projection matrix or a pt->(u,v) callable) to test in SCREEN coords instead
        -- that is frustum/rectangle select from the camera. Returns a MeshSelection. See
        holographic_meshselect.select_in_box."""
        from holographic.mesh_and_geometry.holographic_meshselect import select_in_box
        return select_in_box(mesh, lo, hi, mode=mode, project=project)

    def ray_mesh_intersect(self, mesh, origin, direction, cull_backface=False):
        """RAY-VS-MESH picking (holographic_raypick) -- cast a ray at a mesh {vertices, faces} and return the
        NEAREST hit {face, position, distance, barycentric, triangle} or None. Moller-Trumbore per triangle with an
        AABB broad phase; quads/n-gons are fan-triangulated but report the ORIGINAL face. This is how viewport
        picking hits a user's real geometry (not just the demo cage). See holographic_raypick.ray_mesh_intersect."""
        from holographic.mesh_and_geometry.holographic_raypick import ray_mesh_intersect
        return ray_mesh_intersect(mesh, origin, direction, cull_backface=cull_backface)

    def ray_sdf_intersect(self, sdf_fn, origin, direction, max_dist=50.0, max_steps=128, eps=1e-3):
        """RAY-VS-SDF picking (holographic_raypick) -- sphere-trace a ray into an SDF (any sdf_fn(pt)->distance) and
        return the hit {position, distance, normal, steps} or None. The native pick for the field/procedural half
        of a scene -- exact to the field, no triangulation. See holographic_raypick.ray_sdf_intersect."""
        from holographic.mesh_and_geometry.holographic_raypick import ray_sdf_intersect
        return ray_sdf_intersect(sdf_fn, origin, direction, max_dist=max_dist, max_steps=max_steps, eps=eps)

    def screen_ray(self, screen_u, screen_v, cam_eye=(0.0, 0.0, 3.0), cam_z=-1.6):
        """Build a world-space RAY from a normalized screen coordinate (holographic_raypick) -- (screen_u, screen_v)
        in -1..1 under the cursor -> (origin, direction) for the intersect functions, so 'the user clicked here'
        becomes a geometry query. See holographic_raypick.screen_ray."""
        from holographic.mesh_and_geometry.holographic_raypick import screen_ray
        return screen_ray(screen_u, screen_v, cam_eye=cam_eye, cam_z=cam_z)

    def pick_mesh(self, mesh, screen_u, screen_v, cam_eye=(0.0, 0.0, 3.0), cam_z=-1.6, want="face"):
        """VIEWPORT PICK on a REAL mesh (holographic_raypick) -- from a cursor (screen_u, screen_v in -1..1), build
        the ray and return the nearest 'face' or 'vertex' the user clicked, as {kind, index, position, distance} or
        {index:None} on a miss. The generalization of pick_element (which works on the demo cage) onto a user's
        arbitrary geometry -- one call from 'clicked here' to 'selected this'. See holographic_raypick.pick_mesh."""
        from holographic.mesh_and_geometry.holographic_raypick import pick_mesh
        return pick_mesh(mesh, screen_u, screen_v, cam_eye=cam_eye, cam_z=cam_z, want=want)

    def transform_selection(self, points, selection_idx, translate=None, rotate=None, scale=None,
                            pivot="median", space="world", constraint=(1, 1, 1),
                            cursor=None, active=None, view_matrix=None, local_matrix=None, weights=None):
        """The GIZMO BACKEND (holographic_transform_space): transform selected vertices about a PIVOT
        (median/active/cursor/bbox), in a SPACE (world/local/view), under an axis CONSTRAINT mask -- the triple that
        turns a raw matrix into the move/rotate/scale a modeler expects. `translate` (masked 3-vector), `rotate`
        (axis, angle), `scale` (scalar or 3-vector) about the pivot. Pass `weights` (e.g. soft_selection_weights) for
        PROPORTIONAL editing -- neighbours drag by their falloff. Non-destructive (returns new points). See
        holographic_transform_space.transform_selection."""
        from holographic.mesh_and_geometry.holographic_transform_space import transform_selection
        return transform_selection(points, selection_idx, translate=translate, rotate=rotate, scale=scale,
                                   pivot=pivot, space=space, constraint=constraint, cursor=cursor, active=active,
                                   view_matrix=view_matrix, local_matrix=local_matrix, weights=weights)

    def pivot_point(self, points, selection_idx, mode="median", cursor=None, active=None):
        """Resolve the PIVOT for a transform (holographic_transform_space) -- 'median' (centroid), 'bbox' (box
        centre), 'cursor' (a given point), or 'active' (a chosen vertex). The point a rotate/scale turns around.
        See holographic_transform_space.pivot_point."""
        from holographic.mesh_and_geometry.holographic_transform_space import pivot_point
        return pivot_point(points, selection_idx, mode=mode, cursor=cursor, active=active)

    def snap_to_grid(self, point, increment=1.0, origin=(0.0, 0.0, 0.0)):
        """GEOMETRIC grid snap (caching_and_storage/holographic_snap) -- snap a 3-D point to the nearest grid node
        of spacing `increment` (scalar OR per-axis; an axis with spacing <= 0 is left alone). The 'snap to grid' a
        modeler holds Ctrl for. Distinct from guide_snap (which is VSA codebook cleanup). See
        holographic_snap.snap_to_grid (the canonical geometric snap: 'snapping IS cleanup')."""
        from holographic.caching_and_storage.holographic_snap import snap_to_grid
        r = snap_to_grid(point, increment, origin)
        return r.tolist() if hasattr(r, "tolist") else r

    def snap_to_vertices(self, point, vertices, max_dist=None):
        """Snap a point to the NEAREST vertex (holographic_snap) -- returns {index, position, distance} or None if
        beyond max_dist. The vertex-snap that makes two verts coincide exactly. See
        holographic_snap.snap_to_vertices."""
        from holographic.mesh_and_geometry.holographic_snap import snap_to_vertices
        return snap_to_vertices(point, vertices, max_dist=max_dist)

    def snap_transform_delta(self, delta, target="grid", increment=1.0, moved_point=None, vertices=None,
                             edges=None, origin=(0.0, 0.0, 0.0), max_dist=None):
        """Snap a TRANSFORM DELTA so the dragged point lands on a target (holographic_snap) -- target
        'grid'/'vertex'/'edge'; returns {delta (corrected), snapped_to}. The form the gizmo uses: it has a raw delta
        and the point being dragged, and wants the delta adjusted so that point snaps. Keeps transform and snap
        layers separate -- transform_selection just adds the returned delta. See holographic_snap.snap_transform_delta."""
        from holographic.mesh_and_geometry.holographic_snap import snap_transform_delta
        return snap_transform_delta(delta, target=target, increment=increment, moved_point=moved_point,
                                    vertices=vertices, edges=edges, origin=origin, max_dist=max_dist)

    def sdf_scene(self, parts, bounds=None):
        """Build an SDF SCENE from parts (holographic_sdfscene) -- 'a scene is a set of SDF parts'. Pass a list of
        (sdf_fn, material_name) and optional (center, radius) bounds; get back a scene with .eval (nearest-surface
        distance = min over parts, what a ray-marcher calls), .part_ids / .material_at (argmin, for material
        lookup), and .parts_near (spatial cull). The SDF-scene state model for a modeling app, composing parts the
        way a splat scene bundles primitives. See holographic_sdfscene.SDFScene.from_parts."""
        from holographic.mesh_and_geometry.holographic_sdfscene import SDFScene
        return SDFScene.from_parts(parts, bounds=bounds)

    def residue_system(self, moduli=(7, 11, 13), dim=2048, seed=0):
        """Exact integer arithmetic in vectors via a RESIDUE NUMBER SYSTEM (holographic_extras) -- encode integers
        in [0,M) as CRT residues carried in hypervectors, then add/subtract/scale with vector ops that are exact
        (no floating error), decoding back to the integer. The number-theoretic view of VSA bundling. See
        holographic_extras.ResidueSystem."""
        from holographic.misc.holographic_extras import ResidueSystem
        return ResidueSystem(moduli=moduli, dim=dim, seed=seed)

    def vsa_region(self, center, radius):
        """A REGION of space as a signed-distance ball with boolean algebra (holographic_extras) -- union /
        intersect / subtract / complement of spherical regions, plus contains() and steer(). The set-algebra
        complement to sdf_scene: compose regions of interest for selection or routing. See holographic_extras.ball
        (and Region for the operators)."""
        from holographic.misc.holographic_extras import ball
        return ball(center, radius)

    def predictive_filter(self, momentum=0.5, base_decay=0.9, k=4.0, warmup=6):
        """A SURPRISE filter (holographic_extras) -- observe(vec) returns (is_novel, surprise); slow drift is
        absorbed by a moving prediction while an abrupt change fires once. Pass only surprising observations
        downstream, stay quiet on predictable ones -- an event gate for a stream. See
        holographic_extras.PredictiveFilter."""
        from holographic.misc.holographic_extras import PredictiveFilter
        return PredictiveFilter(momentum=momentum, base_decay=base_decay, k=k, warmup=warmup)

    def edit_history(self, max_depth=256):
        """The UNDO/REDO log for an interactive edit session (holographic_edithistory) -- an EditHistory you thread
        scene state through: `h.do(state, cmd)` applies and records, `h.undo(state)` / `h.redo(state)` walk it, all
        bit-identical (tie-safe replay). Build commands with `vertex_move_command` / `capture_edit_command`. This is
        what makes a modeling session undoable. See holographic_edithistory.EditHistory."""
        from holographic.mesh_and_geometry.holographic_edithistory import EditHistory
        return EditHistory(max_depth=max_depth)

    def vertex_move_command(self, indices, delta, name="move"):
        """A reversible VERTEX MOVE command (holographic_edithistory) for the undo log -- apply adds `delta` to the
        given vertices, invert subtracts it (closed-form inverse, O(edit) memory). Feed to EditHistory.do. See
        holographic_edithistory.vertex_move."""
        from holographic.mesh_and_geometry.holographic_edithistory import vertex_move
        return vertex_move(indices, delta, name=name)

    def capture_edit_command(self, indices, new_positions, prev_positions, name="edit"):
        """Wrap an ARBITRARY geometry edit into a reversible command (holographic_edithistory) by snapshotting the
        before/after positions of just the touched vertices -- O(edit) memory, for edits with no cheap algebraic
        inverse (a bevel, a smooth). Feed to EditHistory.do. See holographic_edithistory.capture_inverse."""
        from holographic.mesh_and_geometry.holographic_edithistory import capture_inverse
        return capture_inverse(indices, new_positions, prev_positions, name=name)

    def draft_vs_refine_simulation(self, kind="fluid", steps=30, draft_grid=16, refine_grid=32, seed=0):
        """MEASURE whether a coarse simulation is a draft of the fine one. Returns {draft_ms, refine_ms, speedup,
        rel_error, converges}. **`converges` is False for a chaotic solver** -- `fluid` at grid 32 against 48 has
        relative error 1.000 and at grid 24 has 0.669, NON-MONOTONIC. A draft render converges to its refinement; a
        draft simulation does not. Refining a chaotic solve replaces it rather than sharpening it.
        See holographic_realtime.draft_vs_refine_simulation."""
        from holographic.scene_and_pipeline.holographic_realtime import draft_vs_refine_simulation as _d
        return _d(self, kind=str(kind), steps=int(steps), draft_grid=int(draft_grid),
                  refine_grid=int(refine_grid), seed=int(seed))

    # -- agent-facing: a live Mesh handle does not survive JSON, so accept its buffers ----------------------
    @staticmethod
    def _as_mesh(mesh):
        """Coerce `mesh` to a live `Mesh`. Accepts one already, or the plain data an agent can POST:
        `{"vertices": [[x,y,z],...], "faces": [[a,b,c],...]}` or the `(vertices, faces)` pair.

        THE LESSON, RE-LEARNED: a live object handle does not survive JSON serialisation, so a faculty that only
        takes one cannot be called over `/invoke` -- and by this engine's own rule a capability an agent cannot call
        does not exist. `emit_kernel` learned this first (the kernel is text); the mesh faculties learn it here (a
        mesh is buffers)."""
        import numpy as _np

        from holographic.mesh_and_geometry.holographic_mesh import Mesh
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            return mesh
        if isinstance(mesh, dict):
            return Mesh(_np.asarray(mesh["vertices"], float), _np.asarray(mesh["faces"], int))
        if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
            return Mesh(_np.asarray(mesh[0], float), _np.asarray(mesh[1], int))
        raise ValueError("expected a Mesh, {vertices, faces}, or (vertices, faces); got %r" % (type(mesh),))

    @staticmethod
    def _as_operator(op):
        """Coerce `op` to a callable. A NAME resolves against the equivariance table's registered operators, which
        is how an agent names an operator over JSON -- a function cannot cross the wire, and guessing one would be
        worse than refusing."""
        from holographic.mesh_and_geometry.holographic_equivariance import OPERATORS
        if callable(op):
            return op
        if isinstance(op, str) and op in OPERATORS:
            return OPERATORS[op]["fn"]
        raise ValueError("op must be a callable, or the name of a registered operator (%s); got %r"
                         % (", ".join(sorted(OPERATORS)), op))

    # -- F2: the smoothest 4-RoSy cross field, and the bar that was vacuous --------------------------------
    def solve_linear_cg(self, A, b, x0=None, iters=250, tol=1e-13):
        """Solve A x = b for a Hermitian positive-definite A by conjugate gradient (the promoted shared solver,
        holographic_numerics.cg -- ledger P1). Accepts a dense matrix (JSON-drivable) or any object with a
        matmul; complex-Hermitian systems welcome (conjugated inner products); `x0` warm-starts. For the
        matvec-closure form (operators too big to materialise), import holographic_numerics.cg directly --
        closures do not cross the JSON boundary. Returns x."""
        import numpy as _np
        from holographic.misc.holographic_numerics import cg
        A = _np.asarray(A)
        return cg(lambda v: A @ v, _np.asarray(b), x0=None if x0 is None else _np.asarray(x0),
                  iters=iters, tol=tol)

    def _topology_check(self, opname, src, out, topology):
        """Shared M13 gate plumbing: compute the topology delta, and refuse only when asked. One place, five
        faculties -- the rule (report by default, refuse opt-in, skip on False/None) must not be re-typed per
        faculty or the conventions will drift."""
        if topology in (None, False):
            return None
        from holographic.mesh_and_geometry.holographic_meshtools import topology_delta
        d = topology_delta(src, out)
        if str(topology) == "refuse" and not d["preserved"]:
            raise ValueError("%s changed topology (islands_created=%s holes_created=%s holes_filled=%s "
                             "nonmanifold_added=%s); pass topology=True to report instead of refusing"
                             % (opname, d["islands_created"], d["holes_created"], d["holes_filled"],
                                d["nonmanifold_added"]))
        return d

    def smallest_eigenpair(self, matvec, n, c, seed=0, dtype=complex, on_matvec=None):
        """Smallest eigenpair of a Hermitian PSD operator given only its matvec (no matrix materialised) --
        the two-phase shifted-inverse-iteration solver behind cross_field's sparse path, promoted so any
        operator can use it (spectral field design, graph spectra, modal analysis). c = the caller's
        Gershgorin/upper bound on the spectrum; on_matvec keeps the matvec count caller-side. Returns
        (eigenvector, lambda_min, matvecs). See holographic_numerics.smallest_eigenpair."""
        from holographic.misc.holographic_numerics import smallest_eigenpair as _se
        return _se(matvec, n, c, seed=seed, dtype=dtype, on_matvec=on_matvec)

    def bisect_to_budget(self, probe, target, lo, hi, midpoint="arith", max_iters=20, tol=None,
                         cmp=None, key=None, bracket=False, on_probe=None):
        """Bisect a MONOTONE probe(knob) to hit a target budget -- the shared engine behind decimate_to (grid
        -> face count) and ratedistortion (scale -> cosine). midpoint "arith" ((lo+hi)//2) or "geom"
        (sqrt(lo*hi)); tol=None does a fixed max_iters sweep returning the final knob, a float best-tracks the
        closest within tol; key turns a probed object into its budget number; the caller keeps its own iter
        count via on_probe. See holographic_numerics.bisect_to_budget."""
        from holographic.misc.holographic_numerics import bisect_to_budget as _b2b
        return _b2b(probe, target, lo, hi, midpoint=midpoint, max_iters=max_iters, tol=tol, cmp=cmp,
                    key=key, bracket=bracket, on_probe=on_probe)

    def mesh_closest_point(self, mesh, points, cell_scale=1.0):
        """Closest point on `mesh` to each of `points` -- the shared correspondence machine behind uv/attribute
        transfer and the high-to-low bakes (M14). Returns a list of (face_index, barycentric, distance) so the
        caller reads whatever channel it needs (position, normal, uv, attribute) off one projection. Builds the
        spatial hash once and reuses it across all query points. See holographic_meshtools.build_face_grid /
        closest_face_point."""
        import numpy as _np
        from holographic.mesh_and_geometry.holographic_meshtools import build_face_grid, closest_face_point
        msh = self._as_mesh(mesh)
        V = _np.asarray(msh.vertices, float)
        F = [tuple(int(i) for i in f[:3]) for f in msh.faces]
        grid, tri, lo, cell = build_face_grid(V, F, cell_scale=cell_scale)
        pts = _np.atleast_2d(_np.asarray(points, float))
        out = []
        for p in pts:
            fi, bc, d2 = closest_face_point(p, grid, tri, lo, cell, F)
            out.append((fi, bc, float(d2) ** 0.5))
        return out

    def graded_levels(self, mesh, target_edge, rho0, k_min=0, k_max=6):
        """Per-vertex power-of-two size LEVELS from a per-vertex target edge length, 2:1-BALANCED so the level
        jump across any edge is at most 1 (M1 increment 1: the graded size field for adaptive retopo). Feed it
        target_edge = clamp(rho0 / (1 + curvature)) to refine where the surface bends. Returns (levels, rho =
        rho0*2^levels). See holographic_crossfield.graded_levels."""
        from holographic.mesh_and_geometry.holographic_crossfield import graded_levels as _gl
        return _gl(self._as_mesh(mesh), target_edge, rho0, k_min=k_min, k_max=k_max)

    def mesh_parts(self, mesh, band_factor=4.0, min_part_frac=0.015):
        """M9: segment a mesh into LIMBS AND BODY via the Reeb graph of geodesic distance -- computed on the
        SURFACE so thin limbs survive (the voxel ridge measured 45 points on a mantis's legs; this found 14
        clean parts in <1 s, every part one connected blob, aspect splitting limbs 7.5-13.4 from core 1.2).
        Weld scans with mesh_repair first (needs a connected surface). Returns (labels, report) with
        part_sizes and part_aspect. See holographic_skeleton.mesh_parts."""
        from holographic.mesh_and_geometry.holographic_skeleton import mesh_parts as _mp
        return _mp(self._as_mesh(mesh), band_factor=band_factor, min_part_frac=min_part_frac)

    def match_symmetric_parts(self, labels, report, vertices, axis=None, tol=0.35):
        """M9: pair parts that are mutual mirror images (a creature's left/right limbs) across the estimated
        bilateral plane, by size + aspect + mirrored-centroid agreement. Feed it mesh_parts' output. Returns
        [(part_a, part_b)]. See holographic_skeleton.match_symmetric_parts."""
        from holographic.mesh_and_geometry.holographic_skeleton import match_symmetric_parts as _ms
        import numpy as _np
        return _ms(_np.asarray(labels), report, _np.asarray(vertices, float), axis=axis, tol=tol)

    def fpe_lattice_resonator(self, bound, bases, ranges, iters=80):
        """R6 (gated): factor a BOUND PRODUCT of fractional-power-encoded integer coordinates back into its
        integers via a Fourier-HRR resonator network -- for the HOLISTIC-ONLY regime where the coordinates are
        never observed directly, only the single bound product (VERIFIED 200/200 at 0.6 rad phase noise, where
        rounding is undefined). KEPT NEGATIVE: for direct noisy coords np.round dominates -- do not use this
        there. Returns (coords, report). See holographic_fpe.fpe_lattice_resonator."""
        from holographic.sampling_and_signal.holographic_fpe import fpe_lattice_resonator as _r
        return _r(bound, bases, ranges, iters=iters)

    def low_eigenvectors(self, matvec, n, c, k=8, seed=0, dtype=float, **kw):
        """The k lowest eigenvectors of a Hermitian PSD operator from its matvec alone, no scipy -- the band a
        spectral analysis needs (mesh eigenmaps, Fiedler order, modal shapes). Block shifted inverse iteration
        on the shared cg; VERIFIED against dense eigh (l=1 residual ~1e-11 on a sphere). Returns
        (eigenvalues, eigenvectors). See holographic_numerics.low_eigenvectors."""
        from holographic.misc.holographic_numerics import low_eigenvectors as _le
        return _le(matvec, n, c, k=k, seed=seed, dtype=dtype, **kw)

    def mesh_fiedler_order(self, mesh):
        """A stable linear ORDER of a mesh's vertices from its Fiedler vector (2nd cotan-Laplacian
        eigenfunction) -- the spectral-seriation order a mesh-as-sequence encoding wants; connectivity-adjacent
        vertices land near each other. Sign-canonicalised -> deterministic. Returns an int index array.
        See holographic_crossfield.mesh_fiedler_order."""
        from holographic.mesh_and_geometry.holographic_crossfield import mesh_fiedler_order as _f
        return _f(self._as_mesh(mesh))

    def mesh_to_tokens(self, mesh, order="morton", bits=8):
        """SATO-SEQ: serialise a mesh to a stable token stream -- order the vertices (morton|zyx|fiedler),
        quantise coords to `bits` bits, emit 3 tokens/vertex. Returns (tokens, order, grid) for dequantising.
        The mesh-as-sequence a hypervector or autoregressive consumer wants. Clean-room (not the GPL SATO
        code). See holographic_meshseq.mesh_to_tokens."""
        from holographic.mesh_and_geometry.holographic_meshseq import mesh_to_tokens as _mt
        return _mt(self._as_mesh(mesh), order=order, bits=bits)

    def seq_encode(self, tokens, dim=1024, seed=0, vocab_size=256, chunk=None):
        """Encode an integer token sequence into one FHRR hypervector by permutation-power binding (or a list
        of block vectors past the ~dim/8 capacity cliff). Round-trips with seq_decode at the same
        dim/seed/vocab_size. See holographic_meshseq.seq_encode."""
        from holographic.mesh_and_geometry.holographic_meshseq import seq_encode as _se
        return _se(tokens, dim=dim, seed=seed, vocab_size=vocab_size, chunk=chunk)

    def seq_decode(self, H, length, dim=1024, seed=0, vocab_size=256, chunk=None):
        """Decode `length` tokens from a permutation-power hypervector (or block list) made by seq_encode.
        Cleanup memory over the seeded phasor vocab. See holographic_meshseq.seq_decode."""
        from holographic.mesh_and_geometry.holographic_meshseq import seq_decode as _sd
        return _sd(H, length, dim=dim, seed=seed, vocab_size=vocab_size, chunk=chunk)

    def worst_view(self, metric, mode="direct", maximize=True, max_evals=4000, eps=1e-4, lipschitz=None):
        """M16: find the GLOBAL worst view over S^2 without a dense sweep. mode="direct" (default) is
        Lipschitz-constant-free (safe when the metric jumps at occlusion); mode="certified" is Piyavskii
        branch-and-bound returning an optimality certificate (needs a Lipschitz bound). `metric` is a pure
        fn of a unit direction. Returns (best_dir, best_value, report). See holographic_worstview.worst_view."""
        from holographic.mesh_and_geometry.holographic_worstview import worst_view as _wv
        return _wv(metric, mode=mode, maximize=maximize, max_evals=max_evals, eps=eps, lipschitz=lipschitz)

    def stripe_pattern(self, mesh, direction_field, frequency=20.0):
        """Knoppel-Crane STRIPE PATTERNS: evenly-spaced stripes that follow a per-vertex tangent direction
        field -- the co-oriented iso-lines a quad layout, texture alignment, or hatching wants. ONE smallest-
        eigenvector problem (reuses the shipped matvec-only eigensolver). MEASURED: phase follows the field to
        a 0.006 rad median edge residual on a sphere. Stripes = level sets of numpy.angle(psi); crisp mask
        (numpy.cos(numpy.angle(psi))>0). Returns (psi complex, report). See holographic_crossfield.stripe_pattern."""
        from holographic.mesh_and_geometry.holographic_crossfield import stripe_pattern as _sp
        import numpy as _np
        return _sp(self._as_mesh(mesh), _np.asarray(direction_field, float), frequency=frequency)

    def mesh_laplacian_eigenmaps(self, mesh, k=8):
        """The low SPECTRUM of a mesh's cotan (Laplace-Beltrami) Laplacian -- the eigenfunctions a spectral
        analysis is built on (spectral segmentation, R6 quadrangulation, Morse layout). VALIDATED: on a sphere
        the eigenvalues cluster at l(l+1) and the first eigenspace recovers x,y,z at R2=1.000. Distinct from
        the crossfield CONNECTION Laplacian (faces + frame); this is the SCALAR vertex operator. Returns
        (eigenvalues (k,), eigenfunctions (n_verts,k)); index 0 is the constant. See holographic_crossfield.mesh_laplacian_eigenmaps."""
        from holographic.mesh_and_geometry.holographic_crossfield import mesh_laplacian_eigenmaps as _e
        return _e(self._as_mesh(mesh), k=k)

    def morse_critical_points(self, mesh, scalar):
        """Count + classify the CRITICAL POINTS (minima/maxima/saddles) of a scalar field on a mesh -- the
        singularity structure a Morse-Smale complex is built from. Discrete lower-star test on each 1-ring;
        obeys Euler-Poincare (min - saddle + max = chi), asserted on a sphere. Returns
        {minima, maxima, saddles, indices}. See holographic_crossfield.morse_critical_points."""
        from holographic.mesh_and_geometry.holographic_crossfield import morse_critical_points as _c
        import numpy as _np
        return _c(self._as_mesh(mesh), _np.asarray(scalar, float))

    def mesh_skeleton(self, mesh, res=32, pad=0.1):
        """Curve skeleton / medial axis of a mesh: the ridge (local maxima) of the interior distance field --
        the deepest, surface-equidistant points that trace the shape's backbone (M9). Returns {points, depth
        (medial radius = local thickness), res, bounds}. Built from the shared correspondence machine
        (closest_face_point) + the winding number, not a new machine. KEPT NEGATIVE: a voxel ridge, res-limited,
        not yet collapsed to a connected 1-D curve. See holographic_skeleton.mesh_skeleton."""
        from holographic.mesh_and_geometry.holographic_skeleton import mesh_skeleton as _sk
        return _sk(self._as_mesh(mesh), res=res, pad=pad)

    def skeleton_curve(self, mesh, res=32, pad=0.1, nbins=12):
        """A single-branch centerline CURVE (ordered polyline) from the medial-axis ridge -- the 1-D collapse
        of mesh_skeleton for a LIMB-LIKE shape (M9 inc 2), via principal-axis binning. Returns {curve, depth
        (medial radius along it), n_ridge}. KEPT NEGATIVE: single-branch only -- one PCA axis cuts corners on
        bent/branched shapes, which need branch segmentation first. See holographic_skeleton.skeleton_curve."""
        from holographic.mesh_and_geometry.holographic_skeleton import skeleton_curve as _sc
        return _sc(self._as_mesh(mesh), res=res, pad=pad, nbins=nbins)

    def interior_distance_field(self, mesh, res=32, pad=0.1):
        """The signed interior DEPTH of a mesh on a res^3 grid (distance-to-surface where inside, 0 outside) --
        the field whose ridge is the skeleton, also usable directly for thickness/wall analysis. Returns
        (depth, (lo,hi), cell). See holographic_skeleton.interior_distance_field."""
        from holographic.mesh_and_geometry.holographic_skeleton import interior_distance_field as _idf
        return _idf(self._as_mesh(mesh), res=res, pad=pad)

    def mesh_topology_delta(self, src_mesh, out_mesh):
        """Did an op change topology it had no business changing? Returns islands_created, holes_created,
        holes_filled, euler_changed, nonmanifold_added and a single `preserved` verdict. THE GATE THE
        SILHOUETTE CANNOT BE: an outline is blind to anything inside it -- measured, surface_retopo scored
        0.973 IoU (a clean PASS) while punching 6 boundary edges into a CLOSED box. Rules: a reducing op must
        not CREATE islands (detached geometry means it tore the surface); must not punch holes in a closed
        mesh; and must not FILL holes that existed (a scan's holes are DATA -- closing them invents surface
        that was never measured). Integers, no tolerance. Measurement, not policy: the caller decides.
        See holographic_meshtools.topology_delta."""
        from holographic.mesh_and_geometry.holographic_meshtools import topology_delta
        return topology_delta(self._as_mesh(src_mesh), self._as_mesh(out_mesh))

    def transform_mesh(self, mesh, matrix):
        """Apply a 3x3/4x4 matrix to a mesh AND flip face winding when the matrix REFLECTS (det < 0). Use this
        for any mirror, axis swap, or negative scale. WHY IT MATTERS: a reflection leaves the mesh perfectly
        self-consistent and entirely INSIDE-OUT -- measured, the naive Z-up->Y-up swap V[:,[0,2,1]] gives a box
        that reports oriented=True with 0% outward normals, and m.mesh_orient CANNOT fix it (it repairs
        neighbours DISAGREEING; global inversion has no disagreement to find). Singular matrices raise.
        See holographic_meshtools.transform_mesh."""
        from holographic.mesh_and_geometry.holographic_meshtools import transform_mesh as _tm
        return _tm(self._as_mesh(mesh), matrix)

    def convert_up_axis(self, mesh, frm="z", to="y"):
        """Re-orient a mesh between up-axis conventions (a Z-up terrain into a Y-up scene) with the winding
        kept correct -- via a PROPER rotation (det=+1), not the naive column permutation that silently turns
        the surface inside out. See holographic_meshtools.convert_up_axis."""
        from holographic.mesh_and_geometry.holographic_meshtools import convert_up_axis as _cu
        return _cu(self._as_mesh(mesh), frm=frm, to=to)

    def mesh_orient(self, mesh, seed_face=0):
        """Make face winding CONSISTENT (flood-fill 2-colouring over the dual graph): neighbours end up
        traversing their shared edge in opposite directions. Returns (mesh, report) with flipped, components,
        non_manifold_edges, non_orientable_components. THE PRECONDITION FOR FIELD WORK: cross_field /
        guided_cross_field / surface_retopo all need consistent winding and scans do not have it. Already-
        oriented meshes come back BIT-IDENTICAL. Non-manifold edges (3+ faces) are SKIPPED and counted -- that
        is a different defect, repaired by m.mesh_repair, and conflating the two reports a lie. Genuinely
        non-orientable components are left ALONE and counted. NOTE: `propagation_components` counts what the
        orientation flood could REACH (manifold-edge connectivity), NOT geometric components -- 399 vs 9 on a
        ladybird LOD, because non-manifold edges block the flood; `components` is the old misleading alias,
        deprecated. See holographic_meshtools.mesh_orient."""
        from holographic.mesh_and_geometry.holographic_meshtools import mesh_orient as _mo
        return _mo(self._as_mesh(mesh), seed_face=seed_face)

    def mesh_orientation_report(self, mesh):
        """Is every DIRECTED edge traversed exactly once, for ANY face degree? Returns oriented,
        duplicated_directed_edges, boundary_edges, faces. Unlike m.mesh_is_oriented (quad-ONLY: it indexes
        q[0..3] literally) this reads triangle meshes -- i.e. every scan and every decimation output.
        See holographic_meshtools.face_orientation_report."""
        from holographic.mesh_and_geometry.holographic_meshtools import face_orientation_report
        return face_orientation_report(self._as_mesh(mesh))

    def surface_retopo(self, mesh, density=1.0, edge_length=None, guide_dirs=None, guide_weight=5.0,
                       iterations=20, boundary="natural", silhouette=0.95, max_density=4.0, topology=True,
                       guard_iterations=None, fast=False, snap_singular=False, feature_sized=False):
        """SURFACE-ROUTE RETOPO: field-aligned quad-dominant topology whose vertices NEVER LEAVE the source
        surface, so the silhouette survives by construction. Use this for SCANS and dense meshes; auto_retopo
        voxelises and is for BLOCK-OUTS (measured: voxel_remesh alone fails the 0.95 gate at every affordable
        resolution on thin features -- an SDF cannot represent what it cannot sample).

        Chain: cross_field/guided_cross_field -> position_field (IFAM 4-PoSy) -> extract_quads (IFAM 4.4) ->
        shrinkwrap(source). `density` scales the lattice against the mean source edge (1.0 = ~one quad per
        source edge; higher = coarser). `guide_dirs` (n_faces,3) routes through guided_cross_field so a strain
        or rig signal puts loops where deformation lives. Guarded by default: if the result misses the
        silhouette floor the density is walked FINER (a LINEAR knob -- deliberately not auto_retopo's cubic
        voxel resolution) up to max_density; silhouette=None opts out. `topology=True` (default) REPORTS the
        topology delta -- islands/holes created, holes filled -- which the outline gate is structurally blind
        to; `topology="refuse"` raises instead. Reporting is the default ON PURPOSE: this operator is measured
        to punch holes (M11), and an instrument that starts refusing yesterday's work is a decision change
        wearing a measurement's clothes. Returns (mesh, report). See holographic_crossfield.surface_retopo."""
        from holographic.mesh_and_geometry.holographic_crossfield import surface_retopo as _sr
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guarded
        src = self._as_mesh(mesh)
        state = {}

        def op(knob):
            # knob = density * 100 (silhouette_guarded walks INTEGER knobs); finer = SMALLER density, so the
            # knob is inverted: walking the knob UP walks the density DOWN, which is the direction that adds
            # detail. Without the inversion the guard would walk the mesh COARSER on failure -- backwards.
            d = float(max_density) * 100.0 / float(knob)
            # H4 (measured): position_field face count / quad_fraction PLATEAU by ~5 iterations (2653 faces at
            # it=5 vs 2664 at it=20, 0.4%), but time is LINEAR in iterations (6.5s vs 15.9s). So the guard's
            # TRIAL extractions -- which only need each density's silhouette + face count, both stable early --
            # can run at guard_iterations, and only the CHOSEN density gets a full-iteration solve below.
            trial_it = int(guard_iterations) if guard_iterations is not None else int(iterations)
            out, rep = _sr(src, density=d, edge_length=edge_length, guide_dirs=guide_dirs,
                           guide_weight=guide_weight, iterations=trial_it, boundary=boundary, fast=fast,
                           snap_singular=snap_singular, feature_sized=feature_sized)
            state[id(out)] = rep
            state.setdefault("_density_of", {})[id(out)] = d
            return out

        start = int(round(float(max_density) * 100.0 / float(density)))
        out, guard = silhouette_guarded(src, op, start, min_iou=silhouette, knob_cost="linear",
                                        max_knob=int(round(float(max_density) * 100.0 / 0.25)))
        # H4 REFINE: re-solve ONCE at the chosen density with full iterations, so the shipped mesh is always
        # the accurate (full-iteration) one -- the reduced-iteration solves are ONLY ever trials used to find
        # the density. The WIN is on a guard WALK: N trials at guard_iterations + 1 full solve, instead of N
        # full solves (measured 114.9s -> 79.0s, 1.45x, identical face count). On a first-try pass it is
        # break-even (1 cheap trial + 1 full solve ~= 1 full solve), never a regression in quality. The plateau
        # (face count stable 5->20 iters) is why the trials' SILHOUETTE decision matches the full solve's.
        if guard_iterations is not None and int(guard_iterations) != int(iterations):
            chosen_d = state.get("_density_of", {}).get(id(out))
            if chosen_d is not None:
                out_full, rep_full = _sr(src, density=chosen_d, edge_length=edge_length, guide_dirs=guide_dirs,
                                         guide_weight=guide_weight, iterations=int(iterations), boundary=boundary, fast=fast,
                                         snap_singular=snap_singular, feature_sized=feature_sized)
                state[id(out_full)] = rep_full
                out = out_full
        rep = dict(state.get(id(out), {}))
        rep["silhouette_report"] = guard
        # TOPOLOGY IS REPORTED, NOT ENFORCED (M13, deliberately additive): the outline gate is blind to holes
        # and islands, and THIS operator is measured to punch holes (M11) -- so the caller must be able to SEE
        # that without the default suddenly refusing work that shipped yesterday. `topology="refuse"` opts in
        # to the second gate; the default only tells the truth. Enforcement changes decisions, and decisions
        # change in ONE place, on purpose, never as a side effect of adding an instrument.
        from holographic.mesh_and_geometry.holographic_meshtools import topology_delta
        if topology:
            rep["topology"] = topology_delta(src, out)
            if str(topology) == "refuse" and not rep["topology"]["preserved"]:
                raise ValueError("surface_retopo changed topology (islands_created=%s holes_created=%s "
                                 "holes_filled=%s); see M11 -- extract_quads drops degenerate cells and that "
                                 "punches holes. Pass topology=True to report instead of refusing."
                                 % (rep["topology"]["islands_created"], rep["topology"]["holes_created"],
                                    rep["topology"]["holes_filled"]))
        return out, rep

    def cross_field(self, mesh, solver="auto", boundary="raise"):
        """The smoothest 4-RoSy field on a CLOSED, ORIENTED surface: per-face angles, as the eigenvector of the
        smallest eigenvalue of the complex connection Laplacian (Knoppel, Crane, Pinkall & Schroder, SIGGRAPH 2013).
        It is a SOLVE, not a local iteration -- Jacobi smoothing oscillates (a torus's energy fell to 2788 by 50
        sweeps and ROSE to 2866 by 400). `solver="auto"` keeps the dense eigh below 2048 faces (bit-identical to
        history) and switches to the sparse Rayleigh-shifted inverse iteration above it: 42k faces in ~9 s where
        the dense path extrapolated to 3.4 days and a 26 GB matrix. Returns (phi, ctx); ctx["solver"] says which
        path ran. `boundary="natural"` solves OPEN meshes with free boundaries (every scan is open -- the
        ladybird has 31,932 boundary edges); the default "raise" keeps the historical error contract and
        closed meshes are bit-identical either way. See holographic_crossfield.cross_field."""
        from holographic.mesh_and_geometry.holographic_crossfield import cross_field as _cf
        return _cf(self._as_mesh(mesh), solver=solver, boundary=boundary)

    def quad_remesh(self, mesh, use_field=True, field=None, silhouette=0.95, topology=True):
        """FIELD-GUIDED tri-to-quad RETOPOLOGY: pair adjacent triangles into quads, preferring pairs whose quad edges
        align with the 4-RoSy cross field and that form a convex, near-square quad (greedy maximal matching). Returns
        a QUAD-DOMINANT mesh + report {quads, tris, quad_fraction, field_used}. Reuses cross_field, so the input wants
        a CLOSED oriented manifold triangle mesh -- run mesh_repair(..., triangulate=True) first; if the field cannot
        solve it falls back to a squareness metric (field_used=False). HONEST: this walks the field to place quads on
        the EXISTING vertices -- it does NOT move vertices or regularise valence, so it is NOT a full Instant-Meshes
        remesh (deferred). See holographic_crossfield.quad_remesh."""
        from holographic.mesh_and_geometry.holographic_crossfield import quad_remesh as _qr
        src = self._as_mesh(mesh)
        out = _qr(src, use_field=use_field, field=field)
        floor = None if silhouette in (None, False) else float(silhouette)
        if floor is not None:
            # quad pairing does not move vertices, so this normally passes at exactly 1.0 (measured) -- the
            # guard is near-free insurance. There is NO finer knob to walk, so the conservative action below
            # the floor is REFUSAL: hand back the ORIGINAL mesh with the verdict attached, and let the caller
            # force destruction explicitly with silhouette=None. Never silently ship a broken retopo.
            from holographic.rendering.holographic_render import silhouette_sweep
            qmesh = out[0] if isinstance(out, tuple) else out
            r = silhouette_sweep(src, qmesh, n_azimuth=6, size=128)
            verdict = {"min_silhouette_iou": floor, "worst": r["worst"], "worst_view": r["worst_view"],
                       "silhouette_iou": r["iou"], "refused": r["worst"] < floor}
            if verdict["refused"]:
                src.silhouette_report = verdict
                return (src, {"refused_for_silhouette": True, **verdict}) if isinstance(out, tuple) else src
            qmesh.silhouette_report = verdict
        # M13: quad_remesh restructures faces, so its topology delta is the interesting one -- pairing can
        # legally change euler bookkeeping on n-gons but must not create islands or holes.
        final = out[0] if isinstance(out, tuple) else out
        d = self._topology_check("quad_remesh", src, final, topology)
        if d is not None:
            try:
                final.topology_report = d
            except Exception:
                pass
        return out

    def guided_cross_field(self, mesh, guide_dirs, guide_weight=5.0, solver="auto", boundary="raise"):
        """A GUIDED 4-RoSy field: the smoothest field that ALSO aligns to a prescribed per-face direction where one
        is given -- field DESIGN, not just smoothing. guide_dirs is (n_faces, 3): a non-zero row guides that face (its
        length is the confidence), a zero row leaves it free. Solves the soft-constrained (L + w) u = w c (Dirichlet
        smoothness + an alignment penalty of guide_weight), a linear solve not an eigenproblem. With no guides it ==
        cross_field. Returns (phi, ctx). This is what lets retopo follow DEFORMATION (strain_directions) or curvature
        instead of only the smoothest field. Needs a CLOSED oriented manifold mesh. See guided_cross_field."""
        from holographic.mesh_and_geometry.holographic_crossfield import guided_cross_field as _gcf
        return _gcf(self._as_mesh(mesh), guide_dirs, guide_weight=guide_weight, solver=solver, boundary=boundary)

    def strain_directions(self, mesh, deformed_vertices):
        """Per-face PRINCIPAL STRETCH direction of a deformation (rest mesh -> deformed_vertices) -- the DEFORMATION
        guide for guided_cross_field, so retopo places edge loops that FOLLOW how the surface bends/stretches
        (deformation-aware topology), which a distortion-only auto-remesher cannot. Per triangle it forms the
        deformation gradient, the right Cauchy-Green C, and takes C max-stretch eigenvector back to 3-D, SCALED by the
        strain anisotropy (isotropic faces -> ~0 confidence, left free). Returns (n_faces, 3) ready as guide_dirs;
        guiding the field to it puts quad LOOPS perpendicular to the stretch -- encircling the bend. See
        strain_directions."""
        from holographic.mesh_and_geometry.holographic_crossfield import strain_directions as _sd
        return _sd(self._as_mesh(mesh), deformed_vertices)

    def position_field(self, mesh, orient, edge_length, iterations=10, seed=0):
        """IFAM POSITION FIELD (4-PoSy): optimise a per-vertex LATTICE position aligned to the orientation field, by
        the local extrinsic smoothing of Instant Field-Aligned Meshes (Jakob et al., SIGGRAPH Asia 2015). For each
        edge it forms q_ij (the point on both tangent planes), translates the neighbour position by INTEGER rho-steps
        to line up, and moves the vertex to the neighbour-weighted average -- so neighbours differ by integer lattice
        steps. THIS is the stage that regularises vertex spacing/valence (field-aligned grid). Works on the vertex
        graph (no closed mesh needed). Returns P (n_vertices, 3). HONEST: the position FIELD only -- extraction to the
        final quad mesh (IFAM sec 4.4) is the next step, not built. See position_field / position_field_regularity."""
        from holographic.mesh_and_geometry.holographic_crossfield import position_field as _pf
        return _pf(self._as_mesh(mesh), orient, edge_length, iterations=iterations, seed=seed)

    def position_field_regularity(self, mesh, P, orient, edge_length):
        """How LATTICE-REGULAR a position field is: mean per-edge residual of (p_i - p_j) after removing the nearest
        integer rho-steps along the field axes, as a fraction of rho. 0 = a perfect field-aligned grid; ~0.5 = no
        lattice structure. The honest measure that position_field converged. See position_field_regularity."""
        from holographic.mesh_and_geometry.holographic_crossfield import position_field_regularity as _pfr
        return _pfr(self._as_mesh(mesh), P, orient, edge_length)

    def face_field_to_vertex(self, mesh, phi):
        """Average a per-FACE 4-RoSy field (cross_field angles) to a per-VERTEX tangent-plane direction -- the input
        position_field wants (cross_field is per-face; the position field lives on the vertex graph). See
        face_field_to_vertex."""
        from holographic.mesh_and_geometry.holographic_crossfield import face_field_to_vertex as _ffv
        return _ffv(self._as_mesh(mesh), phi)

    def trace_streamlines(self, mesh, field, seeds=None, step=None, max_steps=200, four_rosy=True, seed=0, n_seeds=24):
        """Trace STREAMLINES (integral curves) of a per-face direction field across a triangle mesh -- walk along the
        field crossing edge to edge, until a boundary / max_steps / a closed loop. Returns a list of polylines. The
        general field->curves primitive, source-agnostic: a 4-RoSy cross_field (retopo guide-curves, hatching),
        strain_directions (deformation flow lines), an SDF gradient, or a SIMULATION velocity field (streamlines /
        pathlines). field is per-face angles (len n_faces) or per-face 3-D vectors (n_faces,3); four_rosy=True treats
        it as a 4-RoSy cross (branch nearest travel chosen so the curve does not reverse), False for a true vector
        field. Deterministic. See holographic_crossfield.trace_streamlines."""
        from holographic.mesh_and_geometry.holographic_crossfield import trace_streamlines as _ts
        return _ts(self._as_mesh(mesh), field, seeds=seeds, step=step, max_steps=max_steps, four_rosy=four_rosy,
                   seed=seed, n_seeds=n_seeds)





    def field_singularities(self, mesh, phi=None):
        """The STATELESS one-shot twin of `cross_field` + `singularity_index`: mesh in, plain data out
        (`index`, `n_singularities`, `sum_index`, `euler`, `quarter_residual`, `energy`).

        USE THIS OVER HTTP. `cross_field` returns a `ctx` whose `rho` is keyed by `(face, face)` TUPLES; serialised,
        those become the strings `"(0, 1)"`, so the payload looks like a context and cannot be fed back --
        `singularity_index` dies with `KeyError: (0, 1)`. An object that serialises into something that looks right
        but cannot be used is worse than one that raises. See holographic_crossfield.field_singularities."""
        from holographic.mesh_and_geometry.holographic_crossfield import field_singularities as _fs
        return _fs(self._as_mesh(mesh), phi=phi)

    def singularity_index(self, phi, ctx):
        """The per-vertex singularity index, EXACTLY a multiple of 1/4. sum(index) == the Euler characteristic --
        and that says NOTHING about the field: the matching integers are antisymmetric, so they cancel around every
        dual edge and what remains is mesh-only. A random field satisfies it just as exactly.
        See holographic_crossfield.singularity_index."""
        from holographic.mesh_and_geometry.holographic_crossfield import singularity_index as _si
        return _si(np.asarray(phi, float), ctx)

    def field_report(self, mesh, phi=None, ctx=None):
        """{lambda_min, energy, n_singularities, sum_index, euler, quarter_residual, poincare_hopf}. JUDGE A FIELD
        BY `n_singularities` AND `energy`, not by `poincare_hopf` -- measured, the smoothest field has 49
        singularities and energy 54.7 where a random one has 127 and 1542.2, and BOTH satisfy Poincare-Hopf
        exactly. A bar that passes for every input is not a bar. See holographic_crossfield.field_report."""
        from holographic.mesh_and_geometry.holographic_crossfield import field_report as _fr
        return _fr(self._as_mesh(mesh), phi=phi, ctx=ctx)

    # -- K8: dialect emitters -- one source of truth, two runtimes, no drift -------------------------------
    def emit_kernel(self, fn, dialect="wgsl"):
        """Emit a scalar, straight-line, float kernel into `wgsl` | `c_f64` | `c_f32` | `js` | `zig_f64` | `zig_f32`. The hand-written
        compute shader becomes a PROJECTION of the authoritative Python kernel. K10's rule: the emitter REFUSES
        rather than guesses -- an unannotated parameter, an unknown call, a `while`, a range whose bound is not a
        literal, or a missing return each raise with
        the construct named.
        BOUNDED `for i in range(N)` WITH A LITERAL BOUND *IS* SUPPORTED and emits a correct counted loop --
        this docstring previously said "a loop" was refused outright, which was wrong and hid a working
        feature. What is refused is UNBOUNDED or DATA-DEPENDENT iteration (`while`, or `range(n)` where n is
        a parameter), because a shader invocation must have a statically known trip count. A scalar
        straight-line kernel with a bounded loop is exactly the shape a compute-shader invocation runs, so
        this covers per-element maps; a CROSS-INVOCATION REDUCTION is a different problem and is not emitted.
        See holographic_emit.emit."""
        from holographic.io_and_interop.holographic_emit import emit, emit_source
        if isinstance(fn, str):
            return emit_source(fn, dialect=str(dialect))     # the kernel is text; a string is a valid kernel
        return emit(fn, dialect=str(dialect))

    def validate_kernel(self, fn, calls, dialect="c_f64"):
        """Compile the emitted C with `cc`, RUN it on `calls`, and compare to the Python original: {dialect, n,
        max_abs_diff, max_rel_diff, bit_identical}. `c_f64` comes out BIT-IDENTICAL. `c_f32` cannot -- and its
        error (2.9e-07 on an SDF) IS the tolerance a WGSL port must be judged against, because WGSL is f32 and
        NumPy is f64. That is why `c_f32` exists: so the tolerance is MEASURED, not chosen.
        Zig dialects (`zig_f64` / `zig_f32`) route to validate_zig -- compiled `-O ReleaseSafe` with the OPT-IN
        `ziglang` wheel (exactly numba's contract: everything passes without it, absence reported loudly).
        Measured: zig_f64 BIT-IDENTICAL on builtin-intrinsic kernels; std.math.pow is a declared 1-ulp negative.
        See holographic_emit.validate_c / validate_zig."""
        from holographic.io_and_interop.holographic_emit import validate_c, validate_zig
        calls = [tuple(float(x) for x in c) for c in calls]
        if str(dialect).startswith("zig"):
            return validate_zig(fn, calls, dialect=str(dialect))
        return validate_c(fn, calls, dialect=str(dialect))

    def zig_batch_eval(self, kernel, arrays, dtype="f64", simd=0, opt="safe"):
        """Compile a scalar kernel to a native shared library (content-hash cached, `ziglang` wheel, OPT-IN like
        numba) and batch-evaluate it over P same-length arrays. `opt='safe'` is deterministic (f64 scalar measured
        BIT-IDENTICAL to the NumPy evaluation); `simd=8` with dtype='f32' is the measured throughput sweet spot.
        Returns the results as a list. First call pays ~1-2 s of compiler, then ~0 -- a one-shot small-n call is a
        LOSS and this method does not pretend otherwise. See holographic_zigrun.ZigKernel."""
        from holographic.io_and_interop.holographic_zigrun import ZigKernel
        import numpy as _np
        cols = [_np.asarray(a, dtype=float) for a in arrays]
        return [float(x) for x in ZigKernel(kernel, dtype=str(dtype), simd=int(simd), opt=str(opt))(*cols)]

    def zig_regime_map(self, kernel, sizes=(1000, 100000, 1000000), repeats=5, seed=0, simd_width=8):
        """Z3's honest measurement: race numpy / zig scalar f64 / zig simd f32 across sizes. Every row carries the
        baseline, the spread, and a correctness max-abs-err -- a fast wrong answer is not a result. MEASURED verdict
        on the round-box SDF: a modest real 2-5x, peaking near n=1e5, compressing to ~2x at n=1e6 where everything
        goes memory-bandwidth bound. No order-of-magnitude win exists and none is claimed.
        See holographic_zigrun.regime_map."""
        from holographic.io_and_interop.holographic_zigrun import regime_map
        return regime_map(kernel, sizes=tuple(int(s) for s in sizes), repeats=int(repeats),
                          seed=int(seed), simd_width=int(simd_width))

    def kernel_from_description(self, text, name="scene", dialect="python"):
        """Generate a geometry KERNEL from a controlled-vocabulary description (C3): registered parametric SDF
        forms (sphere, rounded box, plane -- iq's exact formulae) composed with union/intersect/subtract. Returns
        Python source, or emit-ready source in any dialect if `dialect` names one (c_f64|c_f32|wgsl|js|zig_*).
        This is NOT free-form NL->code (an LLM's job, out of scope): outside the vocabulary it REFUSES BY NAME,
        and colour/material words are NOTED as ignored, not silently dropped -- an SDF has no colour. Closes the
        loop C1 opened: a generated kernel emits and re-explains. See holographic_codecompose.describe_to_kernel."""
        from holographic.io_and_interop.holographic_codecompose import describe_to_kernel
        src = describe_to_kernel(str(text), name=str(name))
        if str(dialect) != "python":
            from holographic.io_and_interop.holographic_emit import emit_source
            return emit_source(src, str(dialect))
        return src

    def register_geometry_form(self, name, aliases, params, body_fn, purpose, citation=""):
        """Grow C3's controlled vocabulary: register a parametric SDF form whose `body_fn(params)` returns kernel
        lines assigning the signed distance to `$d`. Additive; real published formulae + citation (panel
        discipline). See holographic_codecompose.register_form."""
        from holographic.io_and_interop.holographic_codecompose import register_form
        return register_form(str(name), tuple(aliases), dict(params), body_fn, str(purpose), str(citation))

    def translate_kernel(self, src, from_dialect, to_dialect):
        """Translate a kernel between languages -- python | c_f64 | c_f32 | wgsl | js | zig_f64 | zig_f32 --
        through the ONE shared IR (C2), so there are no pairwise paths to drift. The bar this rides on is
        executed, not asserted: round-trip BYTE-IDENTITY over all 144 dialect pairs in the codeparse selftest,
        and numeric equivalence via validate_kernel for the executable dialects. Refuses, by name, anything
        outside the kernel grammar (K10). See holographic_codeparse.translate."""
        from holographic.io_and_interop.holographic_codeparse import translate
        return translate(str(src), str(from_dialect), str(to_dialect))

    def triage_code(self, src, as_text=False):
        """Triage code in an UNRECOGNIZED language (C5): honest structural OBSERVATIONS -- ranked identifier word
        pieces (camelCase/snake_case split), literal inventory, nesting profile, and a WEAK language hint with
        its evidence -- every field checkable against the source, none claiming to know what the code does. This
        is triage, not comprehension: mind.explain_code parses and explains languages we know; this describes
        the ones we don't (and explain_code falls back here automatically on an unknown dialect). `as_text=True`
        returns the prose report. See holographic_codetriage.triage."""
        from holographic.io_and_interop.holographic_codetriage import triage, triage_report
        return triage_report(str(src)) if as_text else triage(str(src))


    def sdf_emitters_agree(self, node, points=None, tol=1e-5, seed=0):
        """DO THE TWO SDF EMITTERS COMPUTE THE SAME SHAPE? -> {glsl, c_f64, worst, agree, why}.

        holographic_sdf.to_glsl and sdfemit.sdf_dialect both emit a map() for one tree, and sdfemit's own
        header warns that two tables for one concept WILL disagree. This EXECUTES both -- the GLSL through a
        vec3 shim under g++, the C dialect under cc -- and compares each to the Python evaluation, so the
        agreement is measured rather than asserted. `points` defaults to 200 seeded samples in [-2,2]^3.
        THE BARS DIFFER ON PURPOSE: the C dialect must be EXACT; the GLSL gets `tol`, because GLSL float is
        32-bit by language definition and to_glsl writes literals to six significant digits (cos(0.7) ships
        as 0.764842). MEASURED worst case across the node zoo: 4.3e-7. See holographic_sdfemit.emitters_agree."""
        import numpy as _np
        from holographic.mesh_and_geometry.holographic_sdfemit import emitters_agree
        if points is None:
            points = _np.random.default_rng(seed).uniform(-2.0, 2.0, (200, 3))
        return emitters_agree(node, points, tol=tol)

    def sdf_validate_glsl(self, node, points=None, seed=0):
        """Compile the Shadertoy GLSL's own map() and RUN it, comparing to the Python tree ->
        {n, max_abs_diff, bit_identical, source}. The half nobody could execute before: a vec3 shim under
        g++ gives GLSL semantics without a GL runtime. Refuses on GLSL the shim does not model (mat2/mat4,
        textures) rather than comparing wrongly. See holographic_sdfemit.validate_glsl."""
        import numpy as _np
        from holographic.mesh_and_geometry.holographic_sdfemit import validate_glsl
        if points is None:
            points = _np.random.default_rng(seed).uniform(-2.0, 2.0, (200, 3))
        return validate_glsl(node, points)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p04_sdf_offset", "_UnifiedPart04")
    print("holographic_unified_p04_sdf_offset selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

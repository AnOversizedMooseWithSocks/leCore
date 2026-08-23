"""holographic_catalog_p05 -- part 5/6 of the capability registry (split from holographic_catalog).

MECHANICAL SPLIT, no edits. holographic_catalog.py hit 81% of the 1 MB agent-read cap, so the file
that makes capabilities discoverable was becoming the one file an agent could not open. The parts are
called IN ORDER by default_catalog() and the emitted catalog is byte-identical -- verified by hashing
every capability field before and after. Order matters: find_capability ranks by score and ties break
by registration order, so a reordering would silently move search results.

Add new capabilities to the LAST part, or to whichever part is topically right -- never to a new file
without registering it in default_catalog(), or it will simply not exist.
"""


def register_p05(c):
    """Register this part's capabilities on `c`. Called by default_catalog() in order."""
    c.register_capability("packet_demux", "demultiplex a PACKETIZED stream (holographic_demux): variable-length "
                          "bursts from different sources, no cyclic stride. Change-point segmentation (binary "
                          "segmentation, BIC penalty -- a homogeneous stream honestly returns no boundaries), then "
                          "NOISE-CALIBRATED assignment: split-half signatures estimate the noise floor, features "
                          "weighted by 1/noise, segments merge within 3x the floor -- no magic threshold. Returns "
                          "boundaries, assignment, and per-source reassembled streams ready for explore_series. "
                          "The continuous costume of holographic_segment's discrete branching-entropy move",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "r=np.random.default_rng(0); x=np.concatenate([r.standard_normal(60)*0.1, "
                          "3+r.standard_normal(80), r.standard_normal(50)*0.1]); m.packet_demux(x)['n_sources']",
                          native=True, aliases=("packetized stream demux", "variable length bursts",
                                                "detect packet boundaries", "burst segmentation",
                                                "assign segments to sources", "demultiplex bursts to sources"))
    c.register_capability("detect_regimes", "WHERE does a recorded series change behaviour? Located change-point "
                          "detection over a whole batch (holographic_demux.segment_stream): returns the exact "
                          "boundary indices where the statistics shift, plus each segment's start/stop/mean/std. A "
                          "homogeneous stream honestly returns NO boundaries. The OFFLINE batch twin of "
                          "regime_detector (which is causal/online) -- use it to re-fit a cache margin per regime, "
                          "split a forecast at its boundaries, or segment any recorded engine signal into spans",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "r=np.random.default_rng(0); x=np.concatenate([r.normal(0,0.2,150), r.normal(2,0.2,150), "
                          "r.normal(0,1.0,150)]); m.detect_regimes(x)['boundaries']",
                          native=True, aliases=("where does the signal change", "find regime changes offline",
                                                "locate change points in a recording", "segment a series into spans",
                                                "where did the statistics shift", "batch change point detection",
                                                "split a recorded stream at shifts", "find behaviour boundaries"))
    c.register_capability("decompose_piecewise", "decompose a PIECEWISE signal (holographic_scaffold): segment at "
                          "the statistics shifts first (segment_stream), then fit a law PER SEGMENT with "
                          "decompose_signal -- a regime-built signal fits a global formula badly (no 'switch at "
                          "t' atom in the dictionary). MEASURED vs the global baseline on a 3-regime signal: "
                          "residual RMS 0.5001 -> 0.0013, MDL bits 2723 -> 588 (4.6x better compression). The "
                          "result CARRIES its baseline, so a signal where segmentation does not pay is visible",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "y=np.concatenate([2*np.linspace(0,1,100), np.sin(4*np.pi*np.linspace(0,1,100))+3]); "
                          "d=m.decompose_piecewise(y, min_seg=24); (d['total_bits'] < d['baseline']['mdl_bits'])",
                          native=True, aliases=("piecewise decomposition", "fit a law per regime",
                                                "compress a piecewise signal", "regime by regime formula",
                                                "segment then decompose", "better compression for switching signals",
                                                "signal with multiple regimes", "decompose in pieces"))
    c.register_capability("Compression & codec", "shrink data losslessly or by rate-distortion: a sequence/entropy "
                          "codec (codec), general compression (compress), rate-distortion quantization "
                          "(ratedistortion), and content-addressed storage (storage). How the engine fits vectors into "
                          "bytes", example="from holographic.misc.holographic_codec import ...; from holographic.misc.holographic_ratedistortion import ...",
                          native=True, aliases=("compress", "compression", "codec", "entropy coding", "rate distortion",
                                                "quantize", "content addressed storage", "encode data", "shrink data", "deduplicate"))
    c.register_capability("Video (temporal)", "temporal image sequences: video compression with keyframe/delta coding "
                          "(video), temporal compression, motion/phase morph between frames (phasemorph), and frame "
                          "interpolation. Moving pictures on the substrate", example="from holographic.io_and_interop.holographic_video import ...; mind.blend_images(a, b)",
                          native=True, aliases=("video", "compress a video", "temporal compression", "frames", "motion",
                                                "interpolate frames", "keyframe", "sequence of images", "movie"))
    c.register_capability("Honesty & measurement", "measure claims honestly: error bars + significance (measure), "
                          "ablation studies (ablate), proof-of-structure against a null (structure), calibrated "
                          "detection with false-discovery control, benchmark + variance harness, and stress tests. The "
                          "engine's own truth-in-advertising tools", example="from holographic.misc.holographic_measure import ...; from holographic.misc.holographic_ablate import ...",
                          native=True, aliases=("measure", "error bars", "significance", "ablation", "false discovery rate",
                                                "calibrated", "benchmark", "variance", "stress test", "proof of structure",
                                                "honesty", "null model", "confidence interval"), module="honesty", consumes=('scalar',), produces=('scalar',))
    c.register_capability("Program & machine (VM)", "the VSA computer: a stored-program holographic machine "
                          "(machine/HoloMachine) that runs vector programs, recipes with holes / hygienic templates "
                          "(template), a content-addressed compile cache (compile), tool-orchestration planning "
                          "(orchestrator/voidsynth), and reversible computation. Programs as data. PERF: atoms are "
                          "memoised (pure derivations -- bit-identical, always on), and HoloMachine(fast_cleanup=True) "
                          "or mind.vm_fast_cleanup=True opts decode into one cached-codebook matmul per cleanup "
                          "instead of a Python cosine loop -- measured 2x end-to-end, result-identical, opt-in", example="from holographic.agents_and_reasoning.holographic_machine import HoloMachine; from holographic.simulation_and_physics.holographic_template import RecipeTemplate",
                          native=True, aliases=("virtual machine", "stored program", "run a program", "vm", "recipe",
                                                "template", "recipe with holes", "compile", "content addressed compile",
                                                "orchestrate", "plan tools", "reversible computation", "bytecode",
                                                "make the vm faster", "speed up program execution", "simd decode"))
    c.register_capability("Decoded-instruction cache (fetch/decode split from execute)", "decoding a VM "
                          "instruction is a PURE function of (program vector, address) -- it never reads the "
                          "accumulator -- so the plain interpreter re-derives eight transforms every time the "
                          "program counter revisits an address (26x redundancy measured on a 64-iteration "
                          "ITERATE over a 2-instruction body). DecodePlan decodes a whole BLOCK of addresses in "
                          "ONE batched spectral sweep and answers every later visit from a content-addressed "
                          "cache. MEASURED 6.7x-14x end-to-end; accumulators bit-identical and traces identical "
                          "across 126 programs x 3 dims x 3 seeds. Opt-in, never-flip rule",
                          example="mind.vm_decode_plan(True); mind.run_procedure([('LOAD','a'),('BIND','b'),('HALT',None)]); mind.vm_plan_stats()",
                          native=True, aliases=("decoded instruction cache", "instruction cache", "decode cache",
                                                "vectorize the interpreter", "batch decode a program",
                                                "decode once execute many times", "why is my program slow",
                                                "speed up run_procedure", "make the interpreter faster",
                                                "cache decoded instructions", "fetch decode execute"))
    # --- vendored knowledge: a real dictionary + taxonomy for contextual awareness ---
    c.register_capability("Dictionary + taxonomy (vendored)", "a comprehensive vendored English DICTIONARY (~144k "
                          "words: definition, part of speech, synonyms, example) AND an is_a TAXONOMY (encyclopedia "
                          "side: 'a dog is a kind of domestic animal...'), giving the engine real world-knowledge for "
                          "contextual awareness beyond its internal machinery. OPT-IN + lazy: it never loads from "
                          "importing leCore or building a mind -- only the first language call decompresses it (lzma, "
                          "~3.3 MB on disk) into a plain dict in RAM (~22 MB), after which lookups are instant. Control "
                          "it explicitly with holographic.misc.holographic_dictionary.is_loaded()/preload()/unload()/stats(). Stdlib-only "
                          "(lzma+json); the mind can also LEARN meaning from it. Princeton WordNet, free with attribution",
                          example="mind.lookup('gravity'); mind.word_taxonomy('dog'); import holographic.misc.holographic_dictionary as hd; hd.stats()",
                          native=True, aliases=("dictionary", "define", "definition", "word meaning", "synonyms",
                                                "encyclopedia", "taxonomy", "hypernym", "wordnet", "vocabulary",
                                                "contextual awareness", "knowledge", "lexicon", "what does word mean",
                                                "preload dictionary", "unload dictionary", "optional language"))
    c.register_capability("Semantic word index (find words by meaning)", "the fuzzy REVERSE of a dictionary: describe "
                          "an idea and get the words whose definitions mean it. mind.build_semantic_index(words=...) "
                          "places words in a meaning space by RANDOM INDEXING over their glosses, then idx.find('un"
                          "expected good luck') -> 'serendipity' and idx.similar('puppy') -> 'dog','kitten'. OPT-IN and "
                          "separate: nothing loads or builds until you call it. Approximate by design (this is where "
                          "leCore's geometry-preserving/lossy side belongs) -- reliable for the top hit, noisy in the "
                          "tail, and word-sense sensitive.",
                          example="idx = mind.build_semantic_index(words=my_vocab); idx.find('a young dog'); idx.similar('ocean')",
                          native=True, aliases=("semantic index", "find words by meaning", "reverse dictionary",
                                                "words like", "similar words", "meaning search", "word similarity",
                                                "describe a word", "what's the word for", "concept to word", "synonym search"))
    # --- material LIBRARIES: render appearance + physical properties, and the bridge between them ---
    c.register_capability("Material library (render + physical)", "the engine's material LIBRARIES, discoverable in "
                          "one place: ~141 RENDER presets (metals/gems/woods/stones/liquids/biomes -- PBR appearance) "
                          "and ~120 PHYSICAL materials in 12 categories (metals/liquids/gases/polymers/ceramics/glass/"
                          "minerals/stone/wood/biological/building/semiconductors) with density, refractive index, "
                          "viscosity, Young's modulus, sound speed, specific heat, thermal conductivity/expansion, "
                          "melting/boiling point, phase -- validated, unit-documented, for solvers/scientists. "
                          "material_info(name) gives BOTH how "
                          "a material looks AND how it behaves; find_materials()/materials() search + list across both. "
                          "Users can add their own to either library",
                          example="mind.material_info('gold'); mind.find_materials('clear liquid'); mind.materials()",
                          native=True, aliases=("material library", "materials", "physical material", "material properties",
                                                "density", "refractive index", "render material", "pbr preset", "gold",
                                                "copper", "diamond", "material data", "material list", "scientist material"))
    # --- material + shading (consolidation R3) ---
    c.register_capability("Material (channels)", "the material as a record of named channels (albedo/metallic/"
                          "roughness/normal/...) you sample per point; its position-dependent channels BAKE via the "
                          "Cache home and shade via the Shading home", example="from holographic.materials_and_texture.holographic_material import Material",
                          native=True, aliases=("material", "channels", "albedo", "roughness", "metallic", "shader"))
    c.register_capability("Iridescent thin-film tint (soap bubble / oil slick)", "MATERIAL: mind.iridescent_tint(thickness_nm, cos_theta) returns the view-dependent RGB tint of a thin film -- the soap-bubble / oil-slick / pearlescent sheen. Sweeping the angle or thickness cycles the tint through the spectrum (the hallmark of iridescence). n_film 1.33 = soapy water, 1.45 = oil. Multiply a surface's reflected colour by this; holographic_thinfilm.iridescent_socket builds the full f(points,normals,view)->rgb shader socket.",
                          example="import lecore; m=lecore.UnifiedMind(); m.iridescent_tint(thickness_nm=320.0, cos_theta=1.0)",
                          native=True, aliases=("iridescent material", "iridescence", "soap bubble colour", "soap bubble color",
                                                "oil slick sheen", "thin film interference", "pearlescent", "nacre", "rainbow sheen",
                                                "make it iridescent", "peacock colour", "beetle shell"))
    c.register_capability("Multi-material (mask-blended)", "combine N materials by per-point MASKS -- generalises the "
                          "2-way Material.blend to a weighted mix where each material's weight is a mask (a texture "
                          "graph, a field, or a constant) that varies over the surface: paint rust into metal, moss "
                          "onto stone, a decal onto a surface. 'blend' = soft weighted sum (weights normalised so "
                          "brightness stays put); 'select' = hard pick the dominant material (a material-ID / splat "
                          "map). CMP3",
                          example="mind.multi_material([metal, rust], [1.0, mind.texture_leaf('fbm', n_dims=2)]).sample('albedo', [0.3, 0.7])",
                          native=True, aliases=("multi-material", "multimaterial", "blend materials", "material mask",
                                                "material map", "splat map", "material id", "paint materials", "mix materials",
                                                "layer materials by mask"))
    c.register_capability("Layered material (order schema)", "an ORDERED stack of material layers -- base -> diffuse "
                          "-> specular/reflection -> coat/clearcoat -- where the order is a SCHEMA checked at compose "
                          "time, so you can't put a reflection under a diffuse (an out-of-order stack is refused up "
                          "front). Each layer composites OVER the one below by a coverage alpha (a number, field, or "
                          "texture graph). Honest: fixes the stacking, not the energy-conserving radiometry of a true "
                          "layered BRDF. CMP2",
                          example="mind.layered_material([mind.material_layer('base', paint), mind.material_layer('clearcoat', gloss, alpha=0.3)]).sample('albedo', [0.3, 0.7])",
                          native=True, aliases=("layered material", "material layers", "clearcoat", "coat", "layer stack",
                                                "material stack", "over compositing", "base diffuse specular coat",
                                                "stacked material", "material order"))
    c.register_capability("Shading (BRDF)", "the shade model: cook_torrance (full specular+diffuse per light), "
                          "lambert (diffuse term), sample_brdf (importance-sampled bounce) -- call these instead of "
                          "re-deriving Fresnel/GGX/diffuse", example="from holographic.rendering.holographic_brdf import cook_torrance, lambert",
                          native=True, aliases=("shade", "brdf", "cook_torrance", "lambert", "fresnel", "ggx", "specular", "diffuse"))
    c.register_capability("Standalone API service", "run the engine as a standalone DATABASE server on any OS and "
                          "talk to it over HTTP/JSON: full SQL (CREATE/INSERT/SELECT/UPDATE/DELETE/JOIN/DROP), a "
                          "GraphQL front door for nested documents, disk PERSISTENCE (data survives a restart), "
                          "capability discovery, and an optional bearer-token gate. Stdlib-only (numpy aside); a "
                          "drop-in DB replacement for other apps. Launched by serve.sh (Linux/macOS) / serve.bat (Windows)",
                          example="./serve.sh --persist mydb.json   # then: curl -X POST .../sql -d '{\"sql\":\"SELECT ...\"}'",
                          native=True, aliases=("api", "server", "service", "standalone", "http", "rest", "daemon",
                                                "database", "sql", "graphql", "persistence", "drop-in database",
                                                "run as server", "endpoint", "launch", "serve"))
    # rev. 9: the skills selftest's own route probe ("start pause resume cancel a render job") shipped RED at
    # confidence 0.565 -- the cloud-bake entry, a CLIENT of this skill, legitimately shares its vocabulary and
    # split the dominance. The verbs belong in this NAME (they are the skill), which restores the name bonus the
    # generic title gave away: 6.5 -> 9.0, confidence 0.643 -> "act". Same mechanism as the automaton and
    # describe-a-scene fixes: the ranking is fine; the entry under-stated itself.
    c.register_capability("Job lifecycle control (start / pause / resume / cancel)", "start / pause / resume / cancel long-running work (renders, "
                          "simulations, dataset processing) as CHECKPOINTABLE monoid jobs: completed buckets fold into "
                          "partials, so a job pauses at a bucket boundary, saves to disk, survives an app restart, and "
                          "resumes only the remaining buckets. Works across any coordinator backend (local pool / farm)",
                          example="from holographic.scene_and_pipeline.holographic_jobs import JobManager; m.create(id, buckets, worker); m.start(id, background=True); m.pause(id); m.resume(id)",
                          native=True, aliases=("job", "start", "pause", "resume", "cancel", "checkpoint", "render job",
                                                "long running", "background task", "resumable", "progress", "lifecycle"))
    c.register_capability("Code / file editing (agentic)", "read, view (line-numbered), write, exact-string replace, "
                          "replace-lines, insert/delete lines, grep, find-definition, list, tree, archive, move, and "
                          "UNDO -- structured source-file editing for an agent working the codebase, scoped to a project "
                          "ROOT so a path can never escape it. Atomic writes; replace requires a unique match; every "
                          "mutation is reversible with file_undo; replace_across renames a string across many files "
                          "(with a dry-run preview); python_check (syntax) and import_check (real import in a subprocess) "
                          "catch a broken edit immediately. Exposed as mind.file_* methods, so callable over the HTTP "
                          "tool protocol (GET /tools, POST /invoke) like any faculty",
                          example="mind.set_file_root('.'); mind.file_find_definition('make_cloud'); mind.file_replace('a.py', 'old()', 'new()'); mind.file_import_check('a.py'); mind.file_undo()",
                          native=True, aliases=("edit file", "edit code", "modify file", "modify code", "write file",
                                                "read file", "replace in file", "patch", "insert lines", "delete file",
                                                "archive file", "move file", "rename file", "grep", "search code",
                                                "list files", "create file", "file editing", "source editing",
                                                "undo edit", "undo my last edit", "find definition", "jump to definition",
                                                "rename symbol", "rename everywhere", "replace across files", "directory tree",
                                                "check imports", "did my edit break", "view file", "see the file"))
    c.register_capability("Affected-test selection (which tests does my change need)",
                          "answers 'why do thousands of tests run on every small commit?' -- a static import-graph "
                          "selector (pure ast, no execution/coverage tracing) that picks only the tests reachable "
                          "from changed files, or auto-detects the change from git. Fails SAFE: an unscopable "
                          "change widens to the WHOLE suite; a docs-only change selects nothing. The same "
                          "selection CI already runs on push/PR -- previously CLI-only, now a mind faculty. See "
                          "mind.affected_tests's own docstring for the full contract",
                          example="mind.set_file_root('.'); mind.affected_tests(changed_paths=['holographic/rendering/holographic_render.py'])  # or mind.affected_tests() alone to auto-detect from git",
                          native=True, aliases=("affected tests", "which tests to run", "select tests", "test selection",
                                                "run only affected tests", "skip unrelated tests", "reduce test count",
                                                "test suite too slow", "avoid full test suite", "only run changed tests",
                                                "which tests does this touch", "impacted tests", "test impact analysis",
                                                "fewer tests per commit", "why do so many tests run", "cut down tests",
                                                "duplicate tests", "too many tests"))
    c.register_capability("Background cloud bake (resumable)", "run the slow fBm noise bake behind a cloud render as a "
                          "monitorable background JOB you can pause/resume/cancel (even across a process restart), then "
                          "feed the baked grid straight into a render without re-baking. The agent-friendly way to "
                          "handle a render that takes minutes: kick it off, poll progress, do other work",
                          example="jid = mind.bake_cloud_job(radius=1.0, seed=0, background=True); mind.job_status(jid); mind.job_pause(jid); mind.job_resume(jid); grid = mind.job_result(jid)",
                          native=True, aliases=("bake cloud", "background render", "resumable render", "monitor render",
                                                "pause render", "long render", "render job", "noise bake"))
    c.register_capability("Compare rendered images (files)", "perceptual similarity in [0,1] between two images given "
                          "as FILE PATHS (e.g. two rendered PNGs) -- SSIM + colour + edge, shift/lighting-tolerant, the "
                          "on-disk companion to compare_images. The call an agent makes to check 'did my render change "
                          "or match the target?' when the images are files",
                          example="mind.compare_image_files('render_a.png', 'render_b.png')  # -> {similarity, distance, ...}",
                          native=True, aliases=("compare images", "image diff", "render diff", "compare renders",
                                                "image comparison", "did the render change", "image similarity"))
    c.register_capability("Distributed hardening (R5)", "fault tolerance + verification for untrusted farm nodes: "
                          "retry-with-backoff (a reissue reassigns a dead node\'s work), redundant computation + "
                          "majority VOTING (accept only what independent nodes agree on -- a node can\'t force a "
                          "result), canary buckets (known answers reject an untrusted node), and speculative straggler "
                          "backups. The BOINC/SETI@home discipline, mandatory before public contributors",
                          example="from holographic.misc.holographic_hardening import HardenedCoordinator; HardenedCoordinator(farm, redundancy=3).run(buckets, worker, cache, reduce, canaries=[...])",
                          native=True, aliases=("voting", "redundant compute", "retry", "fault tolerance", "canary",
                                                "untrusted node", "quorum", "straggler", "backup execution", "verify result"))
    c.register_capability("Network render farm", "run the coordinator\'s monoid workers on OTHER machines: a worker "
                          "daemon per node (stdlib http/json), the read-only cache shipped ONCE by content hash and "
                          "reused, buckets dispatched concurrently and reduced -- the same Coordinator.run as the local "
                          "pool. Buckets are data, workers are registered code; a node runs only its registered workers",
                          example="from holographic.misc.holographic_farm import WorkerDaemon, NetworkFarm; Coordinator(NetworkFarm([addr])).run(buckets, 'worker_name', cache, reduce)",
                          native=True, aliases=("render farm", "distributed", "network", "seti", "worker daemon",
                                                "remote", "cluster", "node", "another machine", "farm"))
    c.register_capability("Command runner (external tools)", "run any registered ALLOWLISTED program/script as a "
                          "task (subprocess, no shell, time-boxed) and wire it as an orchestrator Tool the Planner "
                          "can chain, with a CircuitBreaker on a flaky one -- the door to external tools and services. "
                          "SECURITY: allowlist only, never a command from untrusted input, values fill placeholders",
                          example="from holographic.scene_and_pipeline.holographic_command import CommandRunner, command_as_tool; r.register('ffmpeg', [...]); r.run('ffmpeg', args)",
                          native=True, aliases=("run command", "external tool", "subprocess", "shell", "run program",
                                                "ffmpeg", "job runner", "allowlist", "command backend"))
    c.register_capability("False-discovery gate over an ablation table", "one claim gets an honest CI from measure(); "
                          "a TABLE of ablations is a scan, and scanning enough subsystems means one clears its bar by "
                          "luck. measure.fdr_gate(rows, alpha) applies Benjamini-Yekutieli across the whole family "
                          "(paired permutation p-values, dependent=True) and reports how many survive.",
                          example="aug, n_load_bearing, n_survive = measure.fdr_gate(rows, alpha=0.1)",
                          native=True, aliases=("fdr", "false discovery", "ablation table", "multiple testing",
                                                "look elsewhere", "benjamini", "is this component load-bearing"))
    c.register_capability("Database layers: durability, locking, history, graph", "opt-in layers on the query Database: "
                          "db.snapshot(path)/Database.restore(path) and db.journal(path) for crash-safe durability; "
                          "db.writer_lock() and db.snapshot_reader(table) for one-writer/many-reader concurrency; "
                          "db.versioned(table) for committed history and time travel; db.adjacency(edges, src, dst) "
                          "for graph traversal. A plain Database pays nothing for them.",
                          example="db.snapshot('/tmp/s.json'); db2 = Database.restore('/tmp/s.json'); vt = db.versioned('shop.items')",
                          native=True, aliases=("durable database", "snapshot", "journal", "wal", "crash safe",
                                                "single writer lock", "concurrency", "time travel", "versioned table",
                                                "graph traversal", "adjacency", "database layers"))
    c.register_capability("Compose new scenes from tags (forward generation)", "the resonator run FORWARD: bind an "
                          "object's (colour, shape, texture) tags into a composite vector and superpose objects into "
                          "a scene -- composing what was never stored, rather than morphing what was. "
                          "mind.novel_object_specs() enumerates the whole generation space.",
                          example="specs = mind.novel_object_specs(); scene = mind.compose_from_tags(specs[:3])",
                          native=True, aliases=("compose a scene", "forward generation", "generate new objects",
                                                "novel combinations", "compose from tags", "procedural scene"))
    c.register_capability("Regime-shift detector (fast/slow layers)", "borrowed from ocean physics: a FAST component "
                          "tracks the present while a SLOW one holds the persistent state; when their divergence stays "
                          "high the system commits to a new LAYER. mind.regime_detector().observe(x) -> (divergence, "
                          "layer, started_new_layer). Tells a genuine regime CHANGE from a wobble.",
                          example="d = mind.regime_detector(); div, layer, new = d.observe(x)",
                          # NB: no bare "diffusion" alias -- it collides with the reaction-diffusion automaton home
                          # and displaced it for the probe "reaction diffusion cellular automaton". One token, two
                          # unrelated meanings; the specific phrase keeps this findable without stealing that query.
                          native=True, aliases=("regime shift", "change point", "drift detection", "has the data changed",
                                                "double-diffusive", "layer detection", "concept drift",
                                                "has the regime changed"))
    c.register_capability("holographic_automaton", "Turing patterns in hypervector space: a vector-valued "
                          "REACTION-DIFFUSION CELLULAR AUTOMATON. Every cell of a 2D grid holds a hypervector; "
                          "short-range activation vs long-range annular inhibition (the Turing mechanism) "
                          "self-organises noise into spots, stripes and labyrinths. Batched FFTs, pure numpy.",
                          # rev. 9: this home was AUTO-SEEDED with a thin `does` and generic aliases, scored a
                          # five-way TIE at 1.50 for "reaction diffusion cellular automaton", and lost the top-3 to
                          # `diffusion_operator`/`diffusion_transfer` PURELY ALPHABETICALLY ('d' < 'h') when those
                          # two entries landed. A curated entry with the module's own vocabulary is the fix the
                          # regime-shift note above already prescribed for this exact probe.
                          example="from holographic.misc.holographic_automaton import HyperCA; ca = HyperCA(64, dim=32, seed=0); ca.step()",
                          native=True, aliases=("cellular automaton", "reaction-diffusion", "reaction diffusion",
                                                "turing patterns", "activator inhibitor", "spots and stripes"))
    c.register_capability("Grid-free PDE solve on an SDF (Walk on Stars)", "solve Laplace or Poisson inside an SDF "
                          "domain with NO MESH, no grid and no global linear system: mind.solve_laplace(sdf, points, "
                          "boundary_value) walks from each point to the boundary and averages what it finds there. "
                          "Pointwise (evaluate only where you care), progressive (error falls as 1/sqrt(walks)), and "
                          "farm-parallel with NO seed coordination -- every random number is a pure function of "
                          "position (hash_unit). Pass dirichlet_sdf to make the rest of the boundary zero-flux "
                          "(Neumann/insulating) -- that is Walk on STARS, which vanilla Walk on Spheres cannot do.",
                          example="u = mind.solve_laplace(sdf.eval, pts, boundary_value, walks=1024, dim=3)  # + dirichlet_sdf= for insulating walls",
                          native=True, aliases=("solve laplace", "poisson equation", "pde without a mesh", "walk on spheres",
                                                "walk on stars", "grid free solver", "harmonic function", "steady state heat",
                                                "boundary value problem", "mesh free", "monte carlo pde", "diffusion curves"))
    c.register_capability("Stateless coordinate-keyed randomness (hash_unit)", "np.random carries STATE, so the n-th "
                          "draw depends on every draw before it -- fatal for farm work, where bucket order then "
                          "changes the numbers. hash_unit(x, y, walk, step, seed) makes the value a pure FUNCTION of "
                          "where and which: same inputs, same value, on any node, in any order, with no seed "
                          "coordination at all. Pure integer arithmetic, independent of PYTHONHASHSEED. "
                          "hash_direction() gives a uniform direction on the sphere or circle.",
                          example="from holographic.misc.holographic_determinism import hash_unit, hash_direction; u = hash_unit(x, y, bounce, seed)",
                          native=True, aliases=("stateless random", "hash noise", "coordinate keyed", "no seed coordination",
                                                "reproducible random", "farm parallel sampling", "hash_unit",
                                                # a GPU's per-thread RNG is exactly this: a pure function of the
                                                # thread's coordinates, with no draw counter and no seed stream.
                                                "random number per thread without a seed stream", "per thread rng",
                                                "gpu random", "philox", "counter based rng", "thread id random",
                                                "deterministic sampling", "per pixel random"))
    c.register_capability("GPU-reproducible 32-bit hash (PCG, matches GLSL)", "hash_unit is 64-bit, so a GPU shader "
                          "(GLSL ES 3.00 / WGSL, 32-bit ints) cannot reproduce it -- why value_noise could not emit. "
                          "hash32_pcg is the 32-bit companion: a PCG output hash (Jarzynski & Olano 2020) of mul/xor/"
                          "shift that wrap mod 2**32 identically in NumPy uint32 and a GLSL uint, so noise built on it "
                          "matches per-point CPU vs GPU. hash32_unit keys it on lattice coords; hash32_pcg_glsl emits "
                          "the GLSL. Coarser than hash_u64 -- reach for it only for the GPU case (it unblocked "
                          "pattern_to_glsl('noise32'/'fbm32')).",
                          example="from holographic.misc.holographic_determinism import hash32_pcg, hash32_unit; u = hash32_unit(3, 5, 7, seed=0)  # -> a deterministic [0,1) value per integer cell, identical to the GLSL PCG",
                          native=True, aliases=("gpu reproducible hash", "32 bit hash for shaders", "pcg hash",
                                                "hash that matches glsl", "hash32", "value noise hash for the gpu",
                                                "cpu gpu matching noise hash", "jarzynski olano hash", "shader hash"))
    c.register_capability("Exact periodic PDE solve (spectral Laplace)", "on a PERIODIC grid the Laplacian is a "
                          "circular convolution, so it is DIAGONAL in the Fourier basis and the solve is closed "
                          "form. mind.solve_poisson_periodic(f) inverts laplacian(u)=f in one FFT; "
                          "mind.diffuse_periodic(T, alpha, t) evolves the heat equation to ANY time t in one "
                          "evaluation (each mode decays by exp(-alpha k^2 t)) -- no time step, no stability limit, "
                          "no substepping. Measured exact to 6.7e-16 where 1000 iterative steps sit at 1.5e-4. "
                          "Periodic only: the Neumann/edge-replicated Laplacian is NOT circular.",
                          example="u = mind.solve_poisson_periodic(f, dx=1/64); T = mind.diffuse_periodic(T0, alpha=0.01, t=1e6, dx=1/64)",
                          native=True, aliases=("spectral laplace", "poisson fft", "closed form heat", "exact diffusion",
                                                "periodic pde", "fourier solve", "no time step", "steady state exact",
                                                "diagonalise the laplacian", "diffuse a field to a given time", "propagator as a transfer",
                                                "diffusion operator", "compose once apply many",
                                                "reuse a pde propagator", "exp(-alpha k^2 t)"))
    c.register_capability("Multi-way tensor compression (Tucker / TT)", "compress data with structure along SEVERAL "
                          "axes -- a field over (x,y,t), a frame stack, a BRDF table, a volume -- by factoring every "
                          "mode at once. mind.compress_tensor(X, method='tucker') uses HOSVD with a RANK GATE that "
                          "picks ranks from the singular spectrum; method='tt' uses a Tensor Train whose storage is "
                          "linear in the number of modes. Measured on a real diffusing field: 57x at rel-err 7.5e-3, "
                          "against 5.9x for a per-slice SVD (which sees structure within a frame but none across "
                          "frames). On data with NO low-rank structure the gate returns full rank -- store it raw. "
                          "Never CP: for 3+ modes a best rank-R CP approximation may not even exist.",
                          example="code = mind.compress_tensor(field, energy=0.999); X = mind.decompress_tensor(code)",
                          native=True, aliases=("tensor compression", "tucker", "hosvd", "tensor train", "low rank tensor",
                                                "compress a volume", "compress a frame stack", "multiway svd",
                                                "rank gate", "should i compress this"))
    c.register_capability("Denoise multi-way data (low-rank tensor prior)", "clean a noisy field over several axes "
                          "-- (x,y,t), a frame stack, a volume -- by projecting onto the low-rank manifold the noise "
                          "level implies. mind.denoise_tensor(X) estimates sigma itself and keeps only singular "
                          "values a noise matrix could not produce. Measured: 31.5 dB -> 48.6 dB on a real diffusing "
                          "field, where a per-slice SVD denoiser reaches 39.5 (it is blind to correlation ACROSS "
                          "slices). KEPT NEGATIVE: a low-rank prior is a claim about the signal -- on a FULL-RANK "
                          "signal it destroys the data (43 dB -> 17 dB). Check the rank gate first.",
                          example="clean, ranks, sigma = mind.denoise_tensor(noisy_field)",
                          native=True, aliases=("denoise a volume", "denoise a field", "low rank denoise",
                                                "tensor denoising", "clean a frame stack", "remove noise from a field",
                                                "multiway denoise"), module="denoise", consumes=('image', 'field'),
                          produces=('image', 'field'),
                          # POLYMORPHIC (C6): denoise_tensor(X) hands back the SAME kind it was given -- it can
                          # never turn an image into a field. Without this the cross product invented
                          # image->field / field->image edges, and suggest_pipeline("image","mesh") used the fake
                          # hop to escape into field-space and answer with this denoiser + an Aharonov-Bohm ring.
                          polymorphic=True)
    c.register_capability("Store a multi-way array (tensor-train file)", "holographic_tucker.save_tensor(X, path) "
                          "writes a volume / frame stack / BRDF table as a Tensor-Train code, and load_tensor reads "
                          "it back. Measured on a real (24,32,32) field: 4,433 bytes at rel-err 3.9e-5, against "
                          "int8's 24,576 bytes at 9.5e-3 -- 5.6x smaller AND 244x more accurate. The bar is INT8 (1 "
                          "byte/element), not float64: on data with no cross-mode structure the TT code is bigger, "
                          "and the file falls back to storing the array RAW and exact. core.save(quant='rd'/'auto') "
                          "carries the same decision for 3+ mode state arrays.",
                          example="from holographic.caching_and_storage.holographic_tucker import save_tensor, load_tensor; save_tensor(volume, 'v.tt'); X = load_tensor('v.tt')",
                          native=True, aliases=("save a volume", "store a frame stack", "tensor train file",
                                                "compress and save a field", "tt file", "multiway storage"))
    c.register_capability("Will compression pay? (area law vs volume law)", "mind.tensor_structure(X) answers, before "
                          "you pay to find out, whether a tensor factorisation can help. It compares the rank kept at "
                          "every cut (how many numbers must cross that boundary) to the most it could possibly be. "
                          "Ranks far below the bound = an AREA LAW: the cost of a cut is set by its boundary, not the "
                          "volume it encloses, and a tensor train is cheap. Ranks that saturate = a VOLUME LAW: every "
                          "degree of freedom is independent, store it raw. Measured: a diffusing field scores 0.21 "
                          "(TT: 4,394 B vs int8 24,576); white noise scores 1.00 (TT: 104,782 B).",
                          example="v = mind.tensor_structure(field); v['verdict']  # 'area-law' or 'volume-law'",
                          native=True, aliases=("will compression help", "area law", "volume law", "schmidt rank",
                                                "bond rank", "is this compressible", "should i compress this",
                                                "entanglement entropy", "structure diagnostic"))
    c.register_capability("Rate-distortion report (bits per vector at a fidelity)", "mind.rate_distortion_report("
                          "arrays, target_cos): the cheapest bit budget that stores vectors while keeping their "
                          "GEOMETRY (pairwise similarity), not just bits -- auto-KLT-rank + coarsest quantization, "
                          "rANS entropy-coded (Duda's ANS). Reports bits_per_vector against the float32 baseline, the "
                          "ratio, achieved cosine (mean+min), rank, and a `pays` flag. KEPT NEGATIVE (loud): "
                          "incompressible near-orthogonal vectors do NOT pay -- the code can be LARGER than float32 "
                          "and pays=False. Measured: low-rank ~3x (691 vs 2048 b/vec); random unit vectors 0.95x.",
                          example="import numpy as np; rng=np.random.default_rng(0); B=rng.normal(size=(3,64)); "
                                  "A=[(np.array([1,.4,-.2])+.05*rng.normal(size=3))@B for _ in range(12)]; "
                                  "r=mind.rate_distortion_report(A, target_cos=0.999); print(r['ratio'], r['pays'])",
                          native=True, aliases=("bits per vector", "how many bits to store a vector", "rate distortion",
                                                "compress a codebook", "entropy code vectors", "geometry preserving "
                                                "compression", "cheapest bit budget", "will these vectors compress",
                                                "ans entropy coding", "quantize a codebook honestly"),
                          semantic="analyze/measure", consumes=('hypervector',), produces=('scalar',), module="ratedistortion")
    c.register_capability("Shuffled-null test (score vs its own null)", "mind.permutation_null(observed, score_fn, "
                          "resample_fn, n_null, alpha, side): the SETI/particle-physics discipline as one composable "
                          "primitive -- score your real datum, re-run the IDENTICAL scoring on resamples that destroy "
                          "the structure, and report whether it stands out. Returns {p, null_mean, null_std, null_ci, "
                          "observed, collapsed, n_null}; p carries the +1 plug (never exactly 0). Generalises the "
                          "engine's five procedure-matched private nulls. KEPT NEGATIVE: a wrong resample_fn gives a "
                          "mis-calibrated null -- the procedure-match is the caller's job. Calibrated + deterministic.",
                          example="import numpy as np; cb=np.random.default_rng(0).standard_normal((20,64)); "
                                  "cb/=np.linalg.norm(cb,axis=1,keepdims=True); "
                                  "sc=lambda q: float(np.max(cb@(q/np.linalg.norm(q)))); "
                                  "rs=lambda r: r.standard_normal(64); "
                                  "print(mind.permutation_null(sc(cb[3]), sc, rs, n_null=200)['collapsed'])",
                          native=True, aliases=("permutation test", "shuffled null", "score against a null",
                                                "p value from a null distribution", "is my result better than chance",
                                                "significance test", "monte carlo p value", "prove it isn't noise",
                                                "false alarm probability", "null hypothesis test"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Documentation map (which doc answers which question)", "SIX doc generators exist -- "
                          "docgen.py (REFERENCE.md, every module), capdoc.py (CAPABILITIES.md, job-oriented), "
                          "apiquickref.py (API_QUICKREF.md, curated app surface), facultymap.py (FACULTY_MAP.md, "
                          "mind methods by topic), pipelinemap.py (PIPELINE_MAP.md, the X->Y workflow graph), "
                          "docmap.py (this map), plus tools/structure_audit.py. docs/DOC_MAP.md lists them with the "
                          "question each answers; tools/regen_docs.py is the ONE DOOR that runs them (--check for "
                          "drift). Exists because root scripts are not catalog entries, so this surface was once "
                          "UNDISCOVERABLE -- a Rule-0 miss, kept loud.",
                          example="import subprocess; print(subprocess.run(['python3','docmap.py'],capture_output=True,text=True).stdout)",
                          native=True, aliases=("where are the docs", "documentation map", "regenerate the docs",
                                                "doc generators", "which doc should i read", "how is this documented",
                                                "api reference", "quick reference", "faculty map", "doc of docs",
                                                "one line per symbol"),
                          semantic="analyze/describe", consumes=(), produces=())
    c.register_capability("What this build has (feature manifest)", "mind.features(names) -> {name: bool} answers "
                          "a preflight in ONE call ('does this build have pipeline_map?'); mind.features() maps "
                          "every public faculty to True. mind.version() -> {engine, capabilities_schema, dim, "
                          "seed} says WHICH BUILD it is. Together they replace a hardcoded client-side list of "
                          "faculty names -- which rots SILENTLY, because a missing faculty and a renamed one both "
                          "look like an absent attribute from outside. Private names are always False: they are "
                          "not part of the contract.",
                          example="import lecore; m=lecore.UnifiedMind(dim=64,seed=0); "
                                  "print(m.features(['pipeline_map','io_kinds','job_submit'])); print(m.version())",
                          native=True, aliases=("features", "feature manifest", "what features are available",
                                                "does this build have", "preflight", "version", "engine version",
                                                "capability check", "what can this engine do", "schema version"),
                          semantic="analyze/pipeline", consumes=(), produces=())
    c.register_capability("Run any faculty as a background job", "mind.job_submit(name, args) -> job_id: start "
                          "ANY public faculty as a real background job, then poll mind.job_status(id) and read "
                          "mind.job_result(id) when status is 'done'. The generic twin of bake_cloud_job, which "
                          "could only background its own bake -- so an 'async' toggle used to work for exactly "
                          "one method. ATOMIC: one bucket, so progress is 0 then 1 and pause/resume cannot split "
                          "the call. args should be JSON-safe to survive a process restart; a live object runs "
                          "fine in-process but the job records persisted=False rather than crashing.",
                          example="import lecore; m=lecore.UnifiedMind(dim=64,seed=0); "
                                  "jid=m.job_submit('infer_semantic_tag', {'name':'render_scene'}); "
                                  "print(m.job_status(jid))",
                          native=True, aliases=("job submit", "run in background", "async", "background job",
                                                "run a faculty asynchronously", "start a job", "queue work",
                                                "non-blocking call"),
                          # NOT simulate/step (my own miss-tag, caught reading the branch's members): simulate/ is
                          # "evolve a physical field over time" per SEMANTIC_TAXONOMY.md, and a background job
                          # evolves nothing -- it is dispatch infrastructure, which is what analyze/pipeline holds.
                          semantic="analyze/pipeline", consumes=(), produces=())
    c.register_capability("Call a faculty by name (JSON dispatch)", "mind.invoke(name, args): run ONE public "
                          "faculty by name with a dict of args -- the dispatch every non-HTTP client used to "
                          "re-implement. m.invoke('double', {'x':21}) -> 42. Private/unknown names raise "
                          "ValueError, never a silent wrong result. args may be a dict (kwargs), a list "
                          "(positional), or None. Returns the RAW result -- JSON coercion is the service's "
                          "boundary job. holographic_service now delegates here, so /invoke and in-process "
                          "callers share ONE set of rules instead of two copies that drift.",
                          example="import lecore; m=lecore.UnifiedMind(dim=64,seed=0); "
                                  "print(m.invoke('semantic_tag_coverage', {}))",
                          native=True, aliases=("invoke", "call by name", "dispatch", "run a faculty by name",
                                                "call a tool", "json dispatch", "call a method dynamically",
                                                "execute capability by name"),
                          semantic="analyze/pipeline", consumes=(), produces=())
    c.register_capability("JSON-drivable objects (mesh/camera coercion)", "render_mesh and friends accept PLAIN "
                          "JSON where they want live objects: mesh={'vertices','faces'} or a Mesh; "
                          "camera={'eye','target',...} or a Camera or a CameraController (coerced via its own "
                          "to_camera bridge -- it lacks projection_matrix and otherwise fails DEEP inside the "
                          "rasteriser). Real objects pass through by IDENTITY, so existing calls are unchanged. "
                          "The constructors already existed: m.render_mesh(m.mesh_box(), m.camera(...)) always "
                          "worked -- what was missing was this edge, and aliases so find_capability could "
                          "surface them. See holographic_coerce.",
                          example="import lecore; m=lecore.UnifiedMind(dim=64,seed=0); "
                                  "print(m.render_mesh({'vertices':[[0,0,0],[1,0,0],[0,1,0]],'faces':[[0,1,2]]}, "
                                  "camera={'eye':[2,2,2],'target':[0,0,0]}, width=16, height=16).shape)",
                          native=True, aliases=("render mesh from json", "mesh dict", "camera dict",
                                                "call render_mesh over http", "json client", "no imports render",
                                                "coerce mesh", "camera controller render"),
                          semantic="convert/emit", consumes=('mesh',), produces=('image',))
    c.register_capability("Semantic action menu coverage (verb tags)", "mind.semantic_tag_coverage() / "
                          "mind.infer_semantic_tag(name): browse_capabilities(by='semantic') renders the "
                          "File->Export->PNG verb tree and OMITS untagged capabilities -- so coverage IS the menu. "
                          "It was 108/2095 (5.2%%): every auto-registered faculty arrived untagged, hiding 95%% of "
                          "the engine's verb surface. The tag is now DERIVED from the verb in the name at "
                          "registration (deterministic table, no model), lifting it to ~31%% / 648 leaves across all "
                          "11 roots. ABSTAINS rather than guess; module names abstain by design (not actions).",
                          example="import lecore; m=lecore.UnifiedMind(dim=128,seed=0); "
                                  "print(m.semantic_tag_coverage()); print(m.infer_semantic_tag('render_scene'))",
                          native=True, aliases=("semantic tag coverage", "action menu coverage", "verb tags",
                                                "how many capabilities are tagged", "what verb is this",
                                                "which menu branch", "taxonomy tag", "tag a capability"),
                          semantic="analyze/pipeline", consumes=(), produces=())
    c.register_capability("Damage a vector (graceful-degradation probe)", "mind.damage_mask(destroy_fraction, "
                          "seed, dim): a keep-mask zeroing a random fraction of a vector's slots. Multiply a "
                          "stored hypervector by it to simulate a scratched plate or lossy channel, then measure "
                          "surviving recall -- how you PROVE holography degrades smoothly instead of taking "
                          "it on faith (dim=256: 20%% slots lost -> cos 0.89, 40%% -> 0.80, 80%% -> 0.54; no "
                          "cliff). Exactly int(dim*fraction) slots zeroed, deterministic in (dim, fraction, "
                          "seed) so a curve is reproducible in a test. D2 consolidation: Hologram/"
                          "HolographicImage/HolographicArchive all delegate here.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                                  "v=m.perceive('a red cube','text'); "
                                  "print(m.damage_mask(0.4).sum(), (v*m.damage_mask(0.4)).shape)",
                          native=True, aliases=("corrupt a vector for testing", "damage a hypervector",
                                                "zero out random slots", "simulate data loss",
                                                "knock out part of a vector", "robustness test mask",
                                                "graceful degradation test", "how much damage can it take",
                                                "lossy channel simulation", "corruption test"),
                          semantic="modify/perturb", consumes=(), produces=())
    c.register_capability("Edge-aware map refiner (guided filter)", "mind.guided_filter(guide, src, radius, eps): "
                          "smooth a map where the GUIDE image is smooth, keep edges where the guide has edges "
                          "(He/Sun/Tang local linear fit, O(N)). Refines ANY (H,W) map against ANY (H,W) guide: "
                          "AO, soft shadow, matte, normals-z, SSS thickness, a mask snapping to boundaries. "
                          "MEASURED vs a same-support box blur: AO RMSE 0.062->0.017, edge kept (box destroys "
                          "it). KEPT NEGATIVE: if the map IGNORES the guide it is NOT better than a box blur and "
                          "injects a spurious edge. REGIME: needs only a guide image; for G-buffer render "
                          "denoising use denoise_svgf.",
                          example="import numpy as np; g=np.zeros((48,48)); g[:,24:]=1.0; "
                                  "m=np.clip(g+0.15*np.random.default_rng(0).standard_normal((48,48)),0,1); "
                                  "print(mind.guided_filter(g, m, radius=6).shape)",
                          native=True, aliases=("edge preserving smooth", "smooth but keep edges",
                                                "refine a map so its edges follow the image", "guided filter",
                                                "edge aware upsample", "clean up a noisy depth map",
                                                "make a mask follow the picture edges", "joint bilateral filter",
                                                "snap a coarse map to object boundaries", "denoise an ao map"),
                          semantic="modify/filter", consumes=(), produces=())
    c.register_capability("N filter passes in one evaluation (shader algebra)", "a circular convolution is diagonal "
                          "in the Fourier basis, so applying it N times is just the transfer raised to the N-th "
                          "power. mind.filter_passes(field, kernel, N) costs the same whether N is 1 or 1,000,000 "
                          "(measured 1,824x faster at N=4096, exact to 2.3e-14). Two things a GPU cannot do: N may "
                          "be FRACTIONAL (half a blur pass; two halves compose to one), and N may be INFINITE -- "
                          "mind.filter_limit returns the steady state as an idempotent projection, where a literal "
                          "loop can need 200,000 passes.",
                          example="soft = mind.filter_passes(img, blur, 64); half = mind.filter_passes(img, blur, 0.5); steady = mind.filter_limit(img, blur)",
                          native=True, aliases=("many blur passes", "iterated filter", "blur n times", "fractional blur",
                                                "half a pass", "steady state filter", "filter to convergence",
                                                "shader algebra", "operator power"))
    c.register_capability("Bake a function into one vector (texture unit)", "mind.bake_field(xs, ys) stores a sampled "
                          "function as a SINGLE hypervector; mind.fetch_field(bake, x) reads it back at ANY x with one "
                          "dot product -- interpolation is built into the algebra, no grid, no lookup table. THE "
                          "ALGEBRA HAS A NYQUIST: the phasor bandwidth sets the finest detail the code can hold, and "
                          "below the signal's maximum angular frequency the bake does not blur, it returns a "
                          "confident WRONG answer and raises nothing. So the bandwidth is chosen from the data "
                          "(measured: RMS error under 0.06 at every frequency tried; half that bandwidth gives "
                          "0.09-0.30). Supplying too small a bandwidth warns.",
                          example="b = mind.bake_field(xs, ys); y = mind.fetch_field(b, 0.37)   # any x, one dot product",
                          native=True, aliases=("bake a function", "texture unit", "lookup table", "LUT", "interpolate a lookup table at arbitrary points", "approximate a function I only have samples of", "function approximation", "cache an expensive function", "memoize a continuous function", "gpu texture", "store a curve", "lookup table",
                                                "interpolate anywhere", "bandwidth", "nyquist", "sample a field",
                                                "function encoding"))
    c.register_capability("Detrend before you bake (non-periodic functions)", "the bandwidth probe is an FFT, and an "
                          "FFT treats its samples as PERIODIC. Any function whose endpoints disagree carries an "
                          "implicit jump at the wrap, and a jump has an unbounded spectrum -- so a STRAIGHT LINE "
                          "probes at 607.9 where sqrt probes at 789.7 and a real 2-cycle sine probes at 12.5, and "
                          "the bake spends its capacity on frequencies that do not exist. "
                          "mind.bake_field(xs, ys, detrend=True) subtracts the endpoint line, bakes the residual, "
                          "and restores the line analytically at fetch time. Measured absolute relative error, mean "
                          "+- sd over 12 seeds, plain vs detrended: sqrt 0.111 +- 0.038 -> 0.009 +- 0.005 (12.6x), "
                          "cube root 0.140 -> 0.017 (8.3x), f(x)=x 0.133 -> exactly 0.000. The plain bake is also "
                          "UNSTABLE (1/(x+0.05) scores 1.83 +- 4.25) because an inflated bandwidth collapses the "
                          "kernel toward a delta. It costs nothing when the endpoints already agree. RETIRED "
                          "NEGATIVE: 'near-singular functions need domain warping' -- wrong cause (the wrap, not "
                          "the singularity) and the weaker fix (warping buys 1.9x where detrending buys 8-16x).",
                          example="b = mind.bake_field(xs, ys, detrend=True); y = mind.fetch_field(b, 0.37, normalize=True)",
                          native=True, aliases=("detrend", "bake a lookup table", "bake sqrt", "non-periodic bake",
                                                "endpoint jump", "spectral leakage", "lut", "near singular function"))
    c.register_capability("Bake an N-D function into one vector (n-D texture unit)", "mind.bake_field_nd(grids, "
                          "values) stores a gridded function of several variables as a SINGLE hypervector, read back "
                          "at any point with mind.fetch_field_nd. The per-axis bandwidths are probed FROM THE DATA, "
                          "because the underlying n-D encoder's default of 3.0 measures at 1.0019 scale-free RMS on "
                          "a 2-D sine -- literally no information, silently. Probed, the same bake lands at 0.101. "
                          "There is NO capacity budget on the number of bundled points (a bundled function is only "
                          "ever summed, never unbound): at a fixed bandwidth the error is flat as the grid goes 400 "
                          "-> 6400 points (0.098 -> 0.118). BANDWIDTH IS A BIAS-VARIANCE DIAL AND dim IS THE "
                          "VARIANCE BUDGET, and the causal variable is B = margin * w_max, not margin: on a 1-cycle "
                          "sine margin 1.5 (B=9.4) is bias-limited and 16x the dimension buys nothing (0.1179 at "
                          "D=4096 vs 0.1191 at D=65536), while at B=18.8 the same signal is variance-limited and D "
                          "pays (0.122 -> 0.043). THE DIAGNOSTIC COSTS ONE EXTRA BAKE: double dim -- if the error "
                          "drops keep spending dimension, if it does not move raise the margin. KEPT NEGATIVE: at "
                          "the default margin this is a SHAPE estimator, amplitude gain 0.66; raise margin and dim "
                          "together or calibrate the gain.",
                          example="b = mind.bake_field_nd([xs, ys], V); v = mind.fetch_field_nd(b, [0.3, 0.7])",
                          native=True, aliases=("bake a 2d function", "n-d texture unit", "bake a volume",
                                                "multivariate lookup table", "encode a 2d point", "bake a grid",
                                                "n dimensional function encoding", "bake a field over a grid"))
    c.register_capability("Subdivision limit surface (closed form)", "mind.mesh_limit_surface(mesh) returns where "
                          "infinite Loop subdivision would put every vertex, plus the EXACT limit normal there -- in "
                          "O(V), performing no subdivision at all. The ring-to-ring block of the local Loop operator "
                          "is exactly a CIRCULANT, i.e. a bind operator, so iterate.transfer diagonalises it for "
                          "free: mode 0 (eigenvalue 5/8 at every valence) gives the limit position, modes +-1 span "
                          "the tangent plane so the normal is exact rather than area-weighted (0.0000 degrees "
                          "against a 6x-subdivided icosphere), and Warren's beta is read off the spectrum instead "
                          "of hard-coded. Deep subdivision converges to it: 6.0e-4 -> 3.7e-5 -> 2.3e-6 at k=4/6/8. "
                          "HONEST SCOPE: this is the k -> infinity case; a FINITE number of levels on an irregular "
                          "mesh still needs the full Stam evaluation, so use mind.mesh_subdivide(mesh, k) there.",
                          example="positions, normals = mind.mesh_limit_surface(mesh)",
                          native=True, aliases=("limit surface", "loop limit", "subdivision limit",
                                                "exact limit normal", "infinite subdivision", "smooth normals",
                                                "push vertices to the limit", "stam evaluation"))
    c.register_capability("Frequency-lifted (Gabor) splats", "mind.splat_field(img, k, basis='gabor') gives each "
                          "splat a FREQUENCY, ORIENTATION and PHASE -- a Gabor atom, seven numbers instead of four. "
                          "A Gabor atom is a BANDPASS primitive, so it buys you exactly the band it is tuned to. "
                          "Measured at equal PARAMETER budget against a jointly-refit Gaussian fit: +7.0 dB on a "
                          "narrowband oriented grating, +0.2 dB on a sharp broadband edge, +0.1 dB on noise-like "
                          "texture -- and it costs 89x the fitting time (a 196-atom dictionary per placement against "
                          "4). The extra dimensions are a levy paid up front, so the win grows with budget (+0.6 dB "
                          "at 224 numbers, +7.5 dB at 1,344). KEPT NEGATIVE, against the prediction that motivated "
                          "it: this does NOT dissolve the splatsharpen negative, which was recorded on a sharp edge "
                          "-- an edge is not a band, it is every band at once. And the Gaussian basis it was "
                          "supposed to beat was never saturated: that flat-in-K curve was greedy matching pursuit's "
                          "overlap double-counting, which splat_refit already fixed (12.9 -> 20.9 dB across K). "
                          "Use mind.spectral_detail to check whether a fit STORED the sharpness, since PSNR will "
                          "not tell you.",
                          example="atoms, img = mind.splat_field(grating, k=64, basis='gabor'); hf = mind.spectral_detail(img)",
                          native=True, aliases=("gabor splat", "gabor atom", "frequency lifted splat",
                                                "oriented splat", "fit a grating", "fit a texture with splats",
                                                "bandpass primitive", "recover high frequency detail",
                                                "does my fit store the sharpness"))
    c.register_capability("Blend M shader variants into one transfer", "an LOD stack, a multi-scale filter, an "
                          "MIS-weighted combination, a parameter sweep you intend to average -- any FIXED linear "
                          "combination of compiled pipelines is itself linear and shift-invariant, so the transfers "
                          "just add. mind.shader_combine(pipes, weights) returns one Pipeline; the cost does not "
                          "depend on M (measured exact to 2.2e-16, and 4.3x / 9.3x / 30.0x faster at M = 4 / 16 / "
                          "64 than staging the variants and blending their images). KEPT NEGATIVE: superposing the "
                          "variants under distinct keys so you can unbind any one back out does NOT work -- "
                          "unbinding recovers a variant at 1/sqrt(M), real variants are correlated copies of one "
                          "field so cleanup cannot resolve them, and the bank still pays M inverse transforms, so "
                          "it measured slower than the direct path. Superposition buys width only when items are "
                          "near-orthogonal AND a cleanup follows the readout.",
                          example="from holographic.rendering.holographic_shader import Pipeline, gauss_kernel\n"
                                  "pipes = [Pipeline(img.shape).blur(gauss_kernel(len(img), s)) for s in (2, 6, 14)]\n"
                                  "out = mind.shader_combine(pipes, [0.5, 0.3, 0.2]).apply(img)",
                          native=True, aliases=("blend filters", "combine shader variants", "lod stack",
                                                "multi-scale filter", "parameter sweep", "average many blurs",
                                                "variant bank", "mip chain"))
    c.register_capability("Gather N lookups in one dot product (superposed gather)", "a quadrature rule, a filter "
                          "stencil or a set of light samples -- sum_j w_j f(u_j) -- compiles into ONE query vector "
                          "Q = sum_j w_j Z(u_j) before the field is ever touched. mind.gather_field(bake, Q) is then "
                          "a single dot product no matter how many taps the rule has, and it is EXACT against "
                          "running the lookups separately (measured 7e-15), because a dot product is linear. There "
                          "is NO sqrt(N/D) crosstalk wall: a gather never unbinds, so more taps make it MORE "
                          "accurate as the bake's per-point errors average down (0.053 -> 0.008 RMS, N=2 -> 512). "
                          "mind.translate_rule slides the whole rule to any offset for one bind, at a cost "
                          "independent of N. Measured 190x amortised over 200 fields with a 64-tap rule. Over an "
                          "HTTP /invoke boundary the bake and the rule are live objects that do not survive JSON, "
                          "so mind.gather_samples(xs, ys, points, weights) is the stateless one-shot twin: plain "
                          "numbers in, a plain number out, no reuse win.",
                          example="b = mind.bake_field(xs, ys); Q = mind.gather_rule(b, us, ws); v = mind.gather_field(b, Q)",
                          native=True, aliases=("gather", "quadrature rule", "filter stencil", "many lookups at once",
                                                "weighted sum of samples", "compile a stencil", "slide a stencil",
                                                "superposed gather", "interpolate from many points"))
    c.register_capability("Compile a filter graph to one pass (shader pipeline)", "chain blurs, translations, gains "
                          "and unsharp blends -- every stage is linear and shift-invariant, so the WHOLE GRAPH "
                          "collapses into ONE transfer function before any data is touched. "
                          "mind.shader_pipeline(shape).blur(k, 8).translate(3).unsharp(kw, 0.6).apply(img) costs one "
                          "FFT, one multiply, one inverse FFT no matter how many stages it has. Measured exact to "
                          "6.7e-16 against running the stages, and 6.0x faster per application. Fractional passes "
                          "and sub-sample (fractional) translations are exact -- neither has a GPU analogue.",
                          example="out = mind.shader_pipeline(img.shape).blur(k, 8).translate(3).unsharp(kw, 0.6).apply(img)",
                          native=True, aliases=("filter graph", "compose filters", "shader pipeline", "multi pass",
                                                "fuse passes", "post process chain", "unsharp", "sub-pixel shift"))
    # --- ENGINE CONTRACTS (D-3). These are not user-facing features; they are the rules a CONTRIBUTOR must cite
    # instead of hand-rolling. They were unfindable, which is exactly why four modules hand-rolled the tie-break.
    c.register_capability("Deterministic tie-break (argmax_tiebreak)", "the engine's ARGMAX CONTRACT: the index of the "
                          "maximum with ties resolved to the LOWEST index (ISA-1). The argmax IS the observable "
                          "decision (which atom is recalled), and scores are not bit-stable across backends, orders "
                          "and bucket counts -- a 1e-17 delta flips the winner. Cite this rule; never call np.argmax "
                          "directly in a decision path. Adoption is enforced by tests/test_unifier_adoption.py.",
                          example="from holographic.misc.holographic_determinism import argmax_tiebreak; idx = argmax_tiebreak(codebook @ query)",
                          native=True, aliases=("break ties deterministically", "argmax", "argmax tiebreak", "tie break",
                                                "which atom wins", "deterministic decision", "lowest index wins",
                                                "bit-exact decision", "ISA-1", "cleanup decision rule"))
    c.register_capability("Closed-form operator iteration (iterate)", "a bind operator is DIAGONAL in the Fourier "
                          "basis, so iterating it k times is one closed-form evaluation (raise the transfer to the "
                          "k-th power) and the k->infinity limit is a mask -- no loop. Measured 41x (k=64) to 1059x "
                          "(k=4096); k=1,000,000 costs the same as k=1, fractional k is well defined, and a divergent "
                          "operator RAISES instead of silently overflowing to nan. Use it instead of "
                          "`for _ in range(k): x = step(x)`.",
                          example="from holographic.misc.holographic_iterate import step_k, limit; x_k = step_k(x, U, k); x_inf = limit(x, U)",
                          native=True, aliases=("iterate a linear operator many steps", "k steps at once", "operator power",
                                                # B2 (NCA backlog): SSP grid addressing IS step_k. a(i,j) = Ax^i * Ay^j,
                                                # built by `step_k(step_k(delta, Ax, i), Ay, j)` -- cosine 1.0000000000
                                                # against the loop, 249x faster at a(1000,1000). Not a new module.
                                                "address a grid cell with a vector", "grid address as a vector",
                                                "transport a code to a neighbouring cell", "shift is a binding",
                                                "spatial semantic pointer", "SSP", "convolutive power",
                                                "steady state", "fixed point", "rollout k steps", "closed form iteration",
                                                "repeat a filter n times", "propagator jump", "diffusion steady state"))
    c.register_capability("Exact order-independent sum (reduce_sum_exact / rns)", "float addition is not associative, "
                          "so a distributed SUM depends on bucket order and count (measured spread 4.6e-5). Reducing "
                          "through exact integer residue arithmetic makes the result BIT-IDENTICAL across orders and "
                          "bucket counts by construction -- the 'bit-exact distributed sum is impossible' caveat is "
                          "retired. Use it wherever a reduction must be reproducible across a farm.",
                          example="from holographic.scene_and_pipeline.holographic_distribute import reduce_sum_exact; total = reduce_sum_exact(parts, bits=40)",
                          native=True, aliases=("exact distributed sum", "bit exact sum", "order independent reduction",
                                                "reproducible sum", "float associativity", "rns", "exact integer sum",
                                                "farm reduce"))
    c.register_capability("Manifold-correct normal quantization (octnormal)", "quantize a unit normal on its own "
                          "manifold (octahedral mapping) instead of packing three floats and re-normalizing, which "
                          "distorts the sphere. The canonical home for compressing normals in meshes, g-buffers, "
                          "splats and curvature.",
                          example="from holographic.mesh_and_geometry.holographic_octnormal import oct_quantize, oct_dequantize; codes = oct_quantize(normals, bits=8)",
                          native=True, aliases=("quantize a unit normal", "compress normals", "normal packing",
                                                "octahedral normal", "unit vector quantization", "gbuffer normals"))
    c.register_capability("Distributed coordinator", "run monoid work (partition -> worker -> shared read-only cache "
                          "-> reduce) on a pluggable BACKEND: an in-process default or a persistent local process pool "
                          "(ProcessPoolExecutor + shared_memory, cache shipped ONCE, workers in separate interpreters). "
                          "Sits behind distribute; includes a margin-gated canonical tie-break so distributed results "
                          "agree on knife-edge decisions",
                          example="from holographic.scene_and_pipeline.holographic_coordinator import Coordinator, LocalPool; Coordinator(LocalPool(4)).run(buckets, worker, cache, reduce)",
                          native=True, aliases=("coordinator", "distribute compute", "process pool", "parallel",
                                                "render farm", "offload", "shared memory", "backend", "tie-break",
                                                "local pool", "worker pool", "monoid reduce"))
    c.register_capability("Graph traversal (exact)", "reachability over a table\'s edges -- neighbors, descendants, "
                          "reachable, shortest path -- what recursive SQL CTEs make painful. Uses an EXACT adjacency "
                          "index by design: the holographic graph store\'s recall collapses at scale, so traversal is "
                          "a plain deterministic graph (tombstone-aware, directed or undirected)",
                          example="from holographic.agents_and_reasoning.holographic_querygraph import EdgeGraph; EdgeGraph(t,'src','dst').path(a,b)",
                          native=True, aliases=("graph", "reachable", "descendants", "shortest path", "traversal",
                                                "adjacency", "recursive cte", "edges", "network"))
    c.register_capability("Single-writer concurrency", "B8 concurrency: one writer at a time (serialised by an "
                          "exclusive lock; a second writer waits or fails fast) plus lock-free reader SNAPSHOTS (a "
                          "consistent point-in-time view immune to later writes). MVCC deferred, stated honestly",
                          example="from holographic.agents_and_reasoning.holographic_querylock import SingleWriterLock; with lock.write(): ...",
                          native=True, aliases=("lock", "single writer", "concurrency", "snapshot read", "writer lock",
                                                "isolation", "consistent read"))
    c.register_capability("Workspace folders", "a shallow grouping tree over a database\'s tables (database > folder "
                          "> table): each table has one HOME folder (ownership -> lifecycle/tier) plus any number of "
                          "ASSOCIATION links (grouping, no deletion on unlink). Scoped search runs over just a "
                          "subtree. Folders reference existing tables, they do not copy them",
                          example="from holographic.agents_and_reasoning.holographic_queryfolder import FolderTree; ft.set_home('user.sales','reports'); ft.tables_in('reports')",
                          native=True, aliases=("folder", "group tables", "namespace tree", "organize tables",
                                                "home folder", "association folder", "scoped search", "drill down"))
    c.register_capability("VSA programs as DB objects", "installable, runnable 'stored procedures' that are "
                          "hypervectors the machine executes (LOAD/BIND/APPLY/HALT -- not arbitrary code): install, "
                          "list a queryable catalog, find a program BY MEANING (fuzzy over its doc), EXPLAIN (dry "
                          "run), and EXECUTE over query rows sandboxed to whitelisted handlers + step-bounded, result "
                          "carrying a calibrated confidence. Safer than a SQL stored procedure",
                          example="from holographic.agents_and_reasoning.holographic_queryprog import ProgramCatalog; cat.install(...); cat.find('cluster a series')",
                          native=True, aliases=("stored procedure", "install program", "execute program", "udf",
                                                "pg_proc", "find program", "run program", "vsa program", "program catalog"))
    c.register_capability("Query time-travel & audit", "git-for-data on a query table: SELECT as-of a past version "
                          "(time travel), blame a row across versions, diff two versions (added/removed/changed with "
                          "field detail), revert, branch/compare/discard, and prove/locate-tampering (Merkle root + "
                          "O(log n) which-row-changed). Wires the shipped versioning faculties into the query layer",
                          example="from holographic.agents_and_reasoning.holographic_querytime import TableHistory, select_as_of, diff_versions, prove",
                          native=True, aliases=("time travel", "point in time", "temporal", "blame", "diff versions", "revert",
                                                "branch", "git for data", "tamper", "audit", "version history", "undo"))
    c.register_capability("Workspaces (durable DB + transient sessions)", "WS3-WS6: run one persistent user database "
                          "alongside many TRANSIENT per-session workspaces (loose scratch tables + the 3D/sim/render "
                          "context) that stay isolated -- clearing or resetting one never touches the persistent DB or "
                          "a sibling. Make / switch / clear / reset-keeping-data, export/import a workspace, and combine "
                          "two with an EXPLICIT collision policy (a merge is a decision, not a guess)",
                          example="from holographic.scene_and_pipeline.holographic_workspace import WorkspaceManager; m=WorkspaceManager(); m.new_workspace('sessionA'); m.switch_workspace('sessionA')",
                          native=True, aliases=("workspace", "session", "scratch tables", "transient tables", "isolate "
                                                "session", "reset keep data", "export workspace", "combine workspaces",
                                                "per-session", "sandbox tables"))
    c.register_capability("Durability & crash recovery", "B7: make the query store survive a crash. Take a durable "
                          "SNAPSHOT of the persistent tiers (replay-based, so it rebuilds byte-identically), keep a "
                          "write-ahead JOURNAL of inserts/updates/deletes since the snapshot, and RECOVER to the last "
                          "consistent point by loading the snapshot and replaying the journal. The snapshot+WAL "
                          "discipline, on top of the plain save/load the service already exposes",
                          example="from holographic.agents_and_reasoning.holographic_query_durable import save_snapshot, Journal, recover; recover(snap_path, journal_path)",
                          native=True, aliases=("durability", "crash recovery", "journal", "write ahead log", "wal",
                                                "snapshot recover", "point in time recovery", "replay journal", "recover"))
    c.register_capability("Splat aniso-refine (re-enable)", "full-3DGS anisotropic refinement composed coarse-first: "
                          "fit cheap isotropic splats, then gradient-refine the RESIDUAL (what iso missed -- sharp / "
                          "oriented features) with anisotropic Gaussians. Strictly >= the isotropic baseline (no harm "
                          "mode); big win on sharp edges. Opt-in (no reliable cheap detector for WHEN it pays)",
                          example="from holographic.rendering.holographic_splat import fit_coarse_first; fit_coarse_first(target, K_iso, K_aniso)",
                          native=True, aliases=("splat refine", "anisotropic splat", "3dgs", "gaussian splat", "coarse "
                                                "first splat", "aniso fit", "residual refine", "gradient refine"))
    c.register_capability("Nystrom kernel (re-enable)", "apply a kernel-weighted field in O(N*m) instead of exact "
                          "O(N^2), gated by a low-rank probe: if a cheap held-out probe says the kernel is low-rank "
                          "(smooth) use Nystrom (measured 6-14x faster, near-exact), else fall back to exact. The "
                          "exact fallback is always correct, so the gate can't be wrong",
                          example="from holographic.sampling_and_signal.holographic_nystrom import apply_kernel_gated; apply_kernel_gated(points, sources, weights, sigma)",
                          native=True, aliases=("nystrom", "landmark", "low rank", "kernel", "rbf field", "large field",
                                                "spectral embedding", "quadratic cost", "smooth field"))
    c.register_capability("Lossless set-packing for image families", "single-file codecs compress every image on "
                          "its own, so a SET that shares structure (a logo suite, sprite variants, UI frames, "
                          "scanned pages) pays for the shared part in every file. mind.pack_images(images) stores "
                          "ONE reference plus per-image deltas, zlib-coded; mind.unpack_images(blob) returns them "
                          "byte for byte (the residual is mod 256, so the round trip is bit-exact). Measured on a "
                          "6-logo suite: 1,744 B against 3,553 B of per-file PNG and 3,162 B of gzip-the-whole-set. "
                          "KEPT NEGATIVE, loud: it LOSES by 16x on content that is already compressible on its own "
                          "(smooth gradients, photographs) -- 32,274 B against 1,987 B. It is content-dependent, so "
                          "mind.pack_benchmark(images) prints the table. Run it; do not guess.",
                          example="blob = mind.pack_images(logos); back = mind.unpack_images(blob)   # bit-exact",
                          native=True, aliases=("pack images", "compress a set of images", "delta compression",
                                                "sprite sheet compression", "store the diff not the frame",
                                                "image family", "lossless set packer"))
    c.register_capability("Learned navigator (adaptive search budget)", "the creature, repurposed to search the "
                          "data tree. mind.train_navigator(items) trains an agent that reads a region, senses how "
                          "confident the answer looks, and decides arrive-or-keep-moving; mind.navigator_find(cue) "
                          "searches, fronted by a ReflexCache that recognises FAMILIAR queries instantly -- it gets "
                          "faster at whatever you ask for most. WHY: a fixed beam spends the same effort on every "
                          "query, so it must be wide enough for the hard minority and overpays on the easy majority. "
                          "MEASURED against the tree's own fixed-beam curve (the strongest baseline, not a "
                          "strawman): the navigator reaches 98.0% recall at 173 comparisons; the cheapest fixed beam "
                          "matching that recall is beam 12 at 450 (2.6x more), and at the navigator's own budget the "
                          "best fixed beam reaches only 81.6%. mind.navigator_benchmark() reproduces both readings.",
                          example="mind.train_navigator(items, queries=1500)\n"
                                  "hit = mind.navigator_find(cue)          # {'index':..., 'comparisons':...}\n"
                                  "mind.navigator_benchmark()              # recall + the fixed-beam baseline",
                          native=True, aliases=("navigator", "adaptive search", "learned search", "search a tree",
                                                "nearest neighbour search", "beam search", "spend less effort on "
                                                "easy queries", "reflex cache", "find an item by cue"))
    c.register_capability("Encyclopedia (relational knowledge)", "the third rung of the dictionary -> grammar -> "
                          "encyclopedia curriculum: a dictionary tells you what a word MEANS, an encyclopedia places "
                          "it in a web of relations. mind.encyclopedia_add(concept, is_a=, has=) teaches one concept "
                          "(key them by a sense id like 'dog.n.01' so senses do not collapse); encyclopedia_is_a is "
                          "one hop with a cleanup confidence; encyclopedia_climb walks the is_a chain as a relation "
                          "ray whose throughput DECAYS with depth on purpose (a longer deduction is less certain) "
                          "and ABSTAINS rather than emit noise; encyclopedia_is_a_transitive answers taxonomic "
                          "membership; encyclopedia_siblings and encyclopedia_relatedness give relatedness from "
                          "STRUCTURE, not word overlap -- 'dog' and 'wolf' share no letters. Relatedness is "
                          "1/(1+depth_a+depth_b) to the nearest common ancestor: identical 1.000, parent 0.500, "
                          "siblings 0.333, cousins 0.200, unrelated 0.000. The state lives on the mind, so a "
                          "long-lived service accumulates knowledge across /invoke calls.",
                          example="mind.encyclopedia_add('dog.n.01', is_a='canine.n.01', has=['tail'])\n"
                                  "mind.encyclopedia_relatedness('dog.n.01', 'wolf.n.01')   # 0.333, siblings",
                          native=True, aliases=("encyclopedia", "taxonomy", "ontology", "is a hierarchy",
                                                "how are two concepts related", "relatedness between concepts",
                                                "teach the mind a fact", "what is a dog related to",
                                                "parent concept", "concept siblings", "walk up the taxonomy",
                                                "structured knowledge about a topic", "relational knowledge"))
    c.register_capability("Run an allowlisted external command", "mind.run_command(name, args) runs an external "
                          "program that an OPERATOR put on the allowlist (ffmpeg, a solver, a shell script, an API "
                          "client), returning {stdout, stderr, returncode, ok}. It joins the same VSA fabric as an "
                          "internal faculty -- mind.command_tool wraps one as an orchestrator Tool the Planner can "
                          "select and chain, with the CircuitBreaker tripping on a flaky one. SECURITY: the "
                          "allowlist is the boundary and it is set IN PROCESS (registration is private, so it is not "
                          "reachable over /invoke -- measured: an agent could register `sh` before that was fixed). "
                          "run_command can only run a name already on the list; values fill {placeholders} one token "
                          "in one token out with NO shell, so an injection attempt in a value is a literal value.",
                          example="info = mind.run_command('probe', {'path': 'clip.mp4'})  # 'probe' registered in process",
                          native=True, aliases=("run a command", "external program", "shell out", "run ffmpeg",
                                                "call an external tool", "run a script", "job runner",
                                                "wrap a program as a tool"))
    c.register_capability("Coarse-first refine (re-enable)", "run the cheap method everywhere, measure a per-cell "
                          "residual/uncertainty, and escalate to the expensive method ONLY where it's high. "
                          "mind.refine_where_uncertain(coarse, uncertainty, refine_fn, frac=0.25). Measured on "
                          "adaptive anti-aliasing of a hard edge: 6.2x fewer samples than supersampling everywhere, "
                          "for a 21% RMSE cost -- and the same budget spent at RANDOM cells is 3x worse, so it is "
                          "the SIGNAL that pays, not the budget. TWO NECESSARY CONDITIONS: (1) the uncertainty must "
                          "be CONCENTRATED -- mind.uncertainty_concentration is the free gate, and near 0 rules "
                          "coarse-first out entirely; (2) the expensive method must be priced PER CELL, because a "
                          "greedy placement method (matching pursuit) is already adaptive and a mask tells it "
                          "nothing -- measured 21.0 dB with and without, at 0.9x the speed. THE TRAP: a GREEDY "
                          "coarse pass destroys the concentration its own refinement needs (0.416 for a uniform "
                          "base, 0.106 for a greedy one). Coarse-first wants a cheap, uniform, dumb base pass. "
                          "THE LAW BOTH CONDITIONS COLLAPSE INTO: coarse-first buys adaptivity for a method that has "
                          "NONE. RETIRED CLIENTS, each already adaptive: splat (greedy placement), volint (a closed "
                          "form -- no cells to escalate), and volume_render (empty_skip + early_term ARE coarse-"
                          "first, buying 15.2x where a residual mask buys 1.0x).",
                          example="u = mind.gradient_uncertainty(coarse)\n"
                                  "if mind.uncertainty_concentration(u) > 0.3:\n"
                                  "    fine, mask, n = mind.refine_where_uncertain(coarse, u, expensive_fn, frac=0.25)",
                          native=True, aliases=("coarse first", "coarse-to-fine", "adaptive refine",
                                                "refine where uncertain", "escalate", "adaptive sampling",
                                                "uncertainty mask", "spend compute where it matters",
                                                "is adaptive refinement worth it", "adaptive antialiasing"))
    c.register_capability("Multi-scatter BRDF (re-enable)", "energy-conserving GGX for rough metals: the Kulla-Conty "
                          "multi-scatter term adds back the energy single-scatter GGX loses (white-furnace ~0.4 -> "
                          "~1.0 at high roughness), GATED by roughness so smooth surfaces skip it (the term overshoots "
                          "at low roughness). Detector is the exact material roughness",
                          example="from holographic.rendering.holographic_brdf import brdf_gated, cook_torrance_ms; brdf_gated(N,V,L,color,metallic,roughness)",
                          native=True, aliases=("multi-scatter", "multiscatter", "kulla-conty", "energy conservation",
                                                "brdf", "ggx", "rough metal", "white furnace", "roughness"))
    c.register_capability("Adaptive record (load-gated)", "a role->filler memory that picks its representation by "
                          "LOAD and FIDELITY need -- cheap real-HRR at low load, FHRR phasors past the capacity knee, "
                          "or tensor-product binding for EXACT recall (perfect to M~dim, at dim*dim storage). Uniform "
                          "add/recall; deciders are exact integers/flags, no harm mode on recall",
                          example="from holographic.simulation_and_physics.holographic_loadmemory import AdaptiveRoleFillerMemory; m=AdaptiveRoleFillerMemory(dim, pairs, exact=True)",
                          native=True, aliases=("adaptive record", "role filler memory", "fhrr", "phasor", "tensor",
                                                "exact recall", "load", "capacity", "high load recall", "bind pairs"))
    c.register_capability("Regime gate (re-enable)", "run a superior-but-niche method ONLY in its regime, behind a "
                          "cheap conservative detector, with a safe fallback everywhere else -- the pattern for "
                          "re-enabling a shelved 'kept negative' now that adaptive dispatch can spot its regime "
                          "(e.g. closed-form iterate for linear/bind operators)",
                          example="from holographic.misc.holographic_regimegate import RegimeGate; RegimeGate(name, detect, threshold, superior, fallback)",
                          native=True, aliases=("regime gate", "re-enable", "adaptive dispatch", "gate", "detector",
                                                "niche method", "fallback", "closed form iterate"))
    c.register_capability("Hypervector (datatype)", "the first-class hypervector: a raw vector + its dim / encoder / "
                          "tag, with the five verbs (bind/unbind/bundle/cleanup/permute) as methods. Encoders are the "
                          "constructors; the raw array stays one attribute away (.array / np.asarray(hv))",
                          example="from holographic.sampling_and_signal.holographic_hypervector import Hypervector; Hypervector.encode(encoder, value).bind(other)",
                          native=True, aliases=("hypervector", "datatype", "vector", "vsa", "hdvector", "symbol",
                                                "bind", "bundle", "permute", "cleanup", "encode"))
    c.register_capability("Sampling", "Monte-Carlo sampling: low-discrepancy / blue-noise patterns, cosine-hemisphere "
                          "directions, MIS weighting, firefly-clamped accumulation -- one home over the shipped samplers",
                          example="from holographic.sampling_and_signal.holographic_samplinghome import Sampling; Sampling.cosine_hemisphere(N, n, seed)",
                          native=True, aliases=("sample", "sampling", "blue_noise", "poisson", "quasi", "halton",
                                                "hemisphere", "mis", "jitter", "firefly", "accumulate"))

    # --- fields (audit named ~8) ---
    c.register_capability(
        "Field", "sample a scalar/vector field at points with ONE interface (field.sample(points)); the backend is "
        "chosen by cost: callable/oracle, dense grid, narrow-band sparse (spectral/FPE/region/dirty are backends too)",
        example="from holographic.misc.holographic_fieldhome import Field; Field.grid(arr, lo, hi).sample(pts)", native=True,
        aliases=("field", "grid", "volume", "density", "sdf", "sample", "voxel",
                # the catalog SELFTEST's own probe, re-ranked out of the top-3 when two merges added
                # ~57 capabilities. Single words lose to descriptively-titled siblings as the catalog
                # grows; the PHRASE a person types is what has to be pinned.
                "represent a density volume over space", "density volume", "volumetric field"))
    c.register_capability("holographic_sparsefield", "narrow-band sparse field -- cost scales with surface area, "
                          "not volume", example="from holographic.misc.holographic_sparsefield import ...", native=True,
                          aliases=("narrow", "band", "sparse", "field"), consumes=(), produces=('field',))
    c.register_capability("holographic_fpefield", "fractional-power-encoded N-D field (surface as one hypervector)",
                          example="from holographic.sampling_and_signal.holographic_fpefield import ...", native=True, aliases=("fpe", "field", "continuous"), consumes=(), produces=('field',))

    # --- scale / compute / the kernel verbs ---
    c.register_capability("holographic_distribute", "scale out a commutative-monoid computation: partition into "
                          "buckets, run independently, reduce (sum/min/max/bundle)", example="from holographic.scene_and_pipeline.holographic_distribute import partition, reduce_sum, reduce_min, reduce_bundle",
                          native=True, aliases=("scale", "parallel", "partition", "mapreduce", "distribute", "raid"))
    c.register_capability("holographic_fuse", "fuse a bind chain into ~2 FFTs with no Python between ops (stay "
                          "VSA-native)", example="from holographic.misc.holographic_fuse import fuse", native=True,
                          aliases=("fuse", "native", "fft", "chain", "compute"))
    c.register_capability("kernel verbs", "the five primitives: bind (attach/transform), unbind (query), bundle "
                          "(superpose/blend), permute (order), cleanup (recognise/denoise)",
                          example="from holographic.agents_and_reasoning.holographic_ai import bind, bundle; from holographic.agents_and_reasoning.holographic_ai import Vocabulary  # Vocabulary(...).cleanup(x)", native=True,
                          aliases=("bind", "unbind", "bundle", "cleanup", "permute", "superpose", "blend"))

    c.register_capability(
        "Shader-native atom families (a vocabulary that is a FUNCTION, not a table)",
        "Two zero-storage atom vocabularies from hash32_pcg -- an atom is RECOMPUTED on a GPU or in a browser, "
        "never shipped. hash_atom/encode_hash: Rademacher, integer-only, VERIFIED IN CHROME (101/101 vs f64). "
        "phasor_atom/record/query/factor/power: FHRR unit phasors; bind is phase ADDITION, unbind the "
        "conjugate, EXACT with no FFT; power is fractional-power encoding in one multiply; factor keeps the "
        "phase (0.967 vs 0.250 real-cast). canonical_terms/term_id are THE shared normalisation boundary -- "
        "apply ONCE. KEPT NEGATIVES, pinned: Rademacher does NOT bind; a record is not a key.",
        example=(
            "rec = mind.phasor_record([('colour','red'), ('size','large')])\n"
            "name, scores = mind.phasor_query(rec, 'colour', ['red','blue','large','small'])\n"
            "q = mind.encode_hash(['holographic','memory'], normalise=False)"
        ),
        aliases=[
            "atom from a hash with no storage",
            "vocabulary that works in a shader",
            "generate a codebook instead of storing it",
            "bind exactly without an FFT",
            "phasor FHRR atom vocabulary",
            "role filler record from names",
            "zero byte vocabulary",
            "recompute the same atom on the GPU",
            "browser friendly hypervector atoms",
            "one place to normalise a term",
            "same term id in both search arms",
            "factor a complex product back into its parts",
            "encode a continuous coordinate as a phase",
        ],
    )

    c.register_capability(
        "Answer, return the set, or abstain (retrieval shape policy)",
        "Decides the SHAPE of an answer instead of always forcing a ranking. If m passages contain EVERY query "
        "term they are indistinguishable to a term scorer, so above m=1 the honest output is the SET plus its "
        "size; below a null-calibrated threshold it ABSTAINS. The ambiguity count is EXACT (postings "
        "intersection), not a predicted confidence. Measured: well-posed 0.875 vs a 0.858 Bayes ceiling; "
        "ambiguous set-recall 1.000 at median 2 where top-1 gives 0.458; no-match refusal 99.2%. KEPT NEGATIVE, "
        "selftest-pinned: a near-duplicate scores HIGH, so a confidence gate does NOT detect ambiguity.",
        example=(
            "docs = [['alpha','beta','solo'], ['alpha','beta','twin'], ['zeta','kappa','unique']]\n"
            "v = mind.retrieval_verdict(['zeta','kappa'], docs)\n"
            "print(v['mode'], v['ambiguity'], v['ceiling'])"
        ),
        aliases=[
            "should I answer or refuse this query",
            "return a set instead of one result",
            "how ambiguous is this search",
            "when is a ranking meaningless",
            "abstain instead of guessing a document",
            "query performance prediction",
            "detect near duplicate documents in results",
            "bayes ceiling for a search query",
        ],
    )

    c.register_capability(
        "Verified GLSL kernels (shader source with the number that verified it)",
        "Nine GLSL kernels this engine compiled and EXECUTED against exact references: BM25 with exact containment "
        "coverage, an inverted index scattered by additive blending (63x less work, 106x wall clock on real "
        "queries), diffusion, PBD constraints, linear image formation, HDRIFT attraction and repulsion. "
        "SOURCE ONLY -- core is NumPy/Flask/stdlib, so no GL binding lives here. EVERY entry carries its KEPT "
        "NEGATIVE and the selftest REFUSES one shipped without it: scatter gives up bit-reproducibility, "
        "diffusion conserves heat only to f32, PBD is Jacobi, raster exactness depends on the scene.",
        example=(
            "print(mind.glsl_kernels())\n"
            "k = mind.glsl_kernel('scatter_bm25_vs')\n"
            "print(k['does']); print(k['verified']); print(k['source'][:200])"
        ),
        aliases=[
            "ready made glsl shader source",
            "verified shader kernels library",
            "get the glsl for bm25 scoring",
            "gpu inverted index shader",
            "diffusion step shader",
            "position based dynamics shader",
            "which shaders have been tested",
        ],
    )

    c.register_capability(
        "Save and load a retrieval index (one format, disk and browser)",
        "mind.index_save(tokens, path) writes a portable `lecore-index/1` bundle; mind.index_load(path) reads it "
        "back and REFUSES a payload whose sha256 does not match -- a truncated write parses cleanly and "
        "answers wrongly. Fills a real gap: mind.save persists the MIND, not an index. Stores the GENERATOR "
        "(packed token stream + vocabulary + global stats); postings are DERIVED on load, so duplicate state "
        "cannot drift. The SAME bytes load in a browser from IndexedDB via pages/idb_store.js.",
        example=(
            "toks = [mind.canonical_terms('the quick brown fox'), mind.canonical_terms('lazy dogs sleep')]\n"
            "man = mind.index_save(toks, '/tmp/idx.json')\n"
            "back = mind.index_load('/tmp/idx.json')\n"
            "print(back['ndocs'], back['ntok'], back['sha256'][:12])"
        ),
        aliases=[
            "save an index to disk and load it back",
            "persist memory across sessions",
            "store the corpus in the browser",
            "indexeddb local storage for leCore",
            "serialize a retrieval index",
            "keep my documents between runs",
        ],
    )

    # --- the catalog itself ---


_PART = "holographic_catalog_p05"


def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p05)
    print("%s selftest OK -- %d capabilities, no internal duplicates" % (_PART, n))


if __name__ == "__main__":
    _selftest()

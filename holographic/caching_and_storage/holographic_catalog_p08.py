"""holographic_catalog_p08.py -- catalog part 08: THE CODE-TOOLS ROBUSTNESS CARDS.

Provenance: the code/text tooling sweep. A find_capability audit for structural
codebase mapping, diagrams, spec conformance, document outlining, arbitrary-root
reference docs, and sandboxed execution returned only fallbacks -- the absence
that licensed holographic_repograph + holographic_docforge and the six faculties
carded here. Aliases below were written from a STRANGER's mouth (five phrasings
brainstormed per card before finalizing) and each card passed the 5/5
discoverability battery at close-out.
"""


def register_p08(c):
    c.register_capability(
        "repo_map",
        "Map a MIXED-LANGUAGE codebase (python/javascript/c): symbols per file, "
        "the file dependency graph from def/ref matching, deterministic "
        "PageRank ranking of which files matter, and a budgeted text skeleton "
        "(aider-style) -- large trees summarized without dumping them. "
        "focus= biases the rank toward files being worked on (personalized "
        "PageRank, the 50x aider convention). Truncation is always "
        "announced, never silent.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.repo_map('holographic/io_and_interop', "
                "budget_lines=30)['files'])",
        native=True,
        aliases=("map out a large codebase", "repo map", "what files matter most",
                 "summarize this codebase", "codebase overview",
                 "scan a javascript project", "index a c codebase",
                 "which files depend on which", "rank files by importance"),
        module="holographic_repograph")
    c.register_capability(
        "codebase_diagram",
        "Draw a codebase as DIAGRAM TEXT: mermaid flowchart or graphviz dot of "
        "the top-ranked files with reference-weighted dependency edges. Text "
        "diagrams diff, version, and render anywhere; drawing caps are "
        "announced in a note node.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.codebase_diagram('holographic/io_and_interop', "
                "fmt='mermaid', max_nodes=5).split(chr(10))[0])",
        native=True,
        aliases=("generate a codebase diagram", "draw the dependency graph",
                 "visualize my project structure", "mermaid diagram of code",
                 "architecture diagram from source", "graphviz of the repo"),
        module="holographic_repograph")
    c.register_capability(
        "spec_conformance",
        "Check a spec/SOP against a source tree WITHOUT hallucination: atomic "
        "claims, mechanical file:line evidence (every citation re-read from "
        "disk before it may appear), verdicts supported/partial/violated/"
        "unverifiable. Ungrippable prose is abstained on, never judged; an "
        "optional llm may propose finer claims but can never confirm one.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.spec_conformance('Provides `SpecChecker`.', "
                "'holographic/io_and_interop')['report'][0]['verdict'])",
        native=True,
        aliases=("check codebase against a spec", "does the code follow the sop",
                 "verify requirements are implemented", "compliance check on code",
                 "audit code against requirements", "spec coverage report"),
        module="holographic_repograph")
    c.register_capability(
        "document_outline",
        "Break a large text document into an ORGANIZED document with a table "
        "of contents: headed markdown keeps the author's structure; unheaded "
        "prose is cut at lexical-cohesion dips (TextTiling-style, "
        "deterministic). Content is reorganized, never rewritten; an optional "
        "llm may only rename section titles.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.document_outline('# A' + chr(10) + 'x' + chr(10) + "
                "chr(10) + '# B' + chr(10) + 'y')['toc'])",
        native=True,
        aliases=("break a document into sections", "table of contents",
                 "organize my research notes", "outline a long text",
                 "structure this document", "split notes into chapters"),
        module="holographic_docforge")
    c.register_capability(
        "docs_generate",
        "Generate a deterministic markdown REFERENCE for any python/js/c tree: "
        "every file, every definition with signature and line number, plus the "
        "author's own first docstring sentence where present -- docgen "
        "generalized from leCore's tree to arbitrary roots.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.docs_generate('holographic/io_and_interop')['defs'])",
        native=True,
        aliases=("generate documentation from code", "build a reference manual",
                 "document this codebase", "api reference from source",
                 "make docs for my project"),
        module="holographic_docforge")
    c.register_capability(
        "sandbox_run",
        "Run code in a SANDBOX: throttled child process with cpu/memory/"
        "file-size rlimits, scrubbed deterministic env (PYTHONHASHSEED=0), "
        "temp cwd, wall timeout, and loudly-marked output caps. Python always "
        "works; node/cc are used when installed and refused honestly when "
        "not.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.sandbox_run('print(6*7)')['stdout'].strip())",
        native=True,
        aliases=("run code in a sandbox", "execute python safely",
                 "run untrusted code", "test a snippet with limits",
                 "safe code execution", "run this script isolated"),
        module="holographic_docforge")
    c.register_capability(
        "sop_run",
        "FOLLOW ORDERS: execute an authored text SOP through the mind. A model "
        "writes the plan (## step: / invoke: / python: / shell: / verify: / "
        "on_fail: / guidance:); the substrate runs it -- invoking faculties, "
        "sandboxing code, verifying every step -- and consults the model ONLY "
        "at declared guidance/escalation points. A scriptable SOP runs with "
        "ZERO model calls (llm_calls in the result proves it); an SOP that "
        "does not fully parse is refused before step 1.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.sop_run('## step: a' + chr(10) + 'python: print(6*7)' "
                "+ chr(10) + 'verify: \"42\" in result[\"stdout\"]')['ok'])",
        native=True,
        aliases=("run an sop written by the model", "follow a step by step plan",
                 "execute orders from an llm", "run a standard operating procedure",
                 "let the llm plan and lecore execute", "run a scripted workflow",
                 "carry out a checklist automatically"),
        module="holographic_soprunner")
    c.register_capability(
        "sop_check",
        "Validate an authored SOP WITHOUT running anything: every problem "
        "named by line number, or ok with the step count. The author's "
        "(usually a model's) edit-until-clean loop before sop_run -- an order "
        "leCore cannot fully parse is an order it will refuse to follow.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.sop_check('## step: a' + chr(10) + 'python: 1+1'))",
        native=True,
        aliases=("validate a plan before running it", "check my sop syntax",
                 "lint a procedure", "will this plan run",
                 "dry run a workflow", "verify sop format"),
        module="holographic_soprunner")
    c.register_capability(
        "sop_save",
        "Save a NAMED SOP for later sop_run(name) -- the leOS macro_registry "
        "pattern on the durable KnowledgeStore, so procedures survive process "
        "restarts. Validates first (a plan that does not parse is refused, "
        "not stored); revisions are appends and the last save wins. Fetch "
        "with sop_load(name).",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.sop_save('demo', '## step: a' + chr(10) + 'python: 1'))",
        native=True,
        aliases=("save a named procedure for later", "store a reusable workflow",
                 "remember this sop", "keep a playbook across sessions",
                 "define a macro procedure", "load a saved sop"),
        module="holographic_soprunner")
    c.register_capability(
        "ask_chain",
        "MULTI-HOP CHAIN over the mind's own memory: ask_chain('tokyo', "
        "('capital','currency'), ('currency','language')) walks filler -> "
        "record -> filler hop by hop. RESTORED in sweep 63: this faculty was "
        "silently shadowed by the one-call ask() for its whole life -- the "
        "duplicate-name collision the split test guards.",
        example="import lecore; m=lecore.UnifiedMind(dim=512, seed=0); "
                "m.absorb([({'a':'x','b':'y'}, 'r1')]); "
                "print(m.ask_chain('x', ('a','b')))",
        native=True,
        aliases=("chain a question over memory", "multi hop question",
                 "follow relations hop by hop", "walk from value to value",
                 "answer through intermediate records"),
        module="holographic_unified_p02_fit_deterministic")
    c.register_capability(
        "explain_similarity",
        "WHY are two things similar: per-role comparison of two absorbed "
        "records -- for each shared role, both fillers and whether they "
        "match, with confidences. Not a bare cosine: the ROLES carry the "
        "explanation. RESTORED in sweep 63 from under explain(topic)'s "
        "shadow.",
        example="import lecore; m=lecore.UnifiedMind(dim=512, seed=0); "
                "m.absorb([({'a':'x'}, 'r1'), ({'a':'x'}, 'r2')]); "
                "print(m.explain_similarity('r1', 'r2')[0][3])",
        native=True,
        aliases=("why are these two similar", "compare two records role by role",
                 "what do these share", "per attribute comparison",
                 "explain the similarity"),
        module="holographic_unified_p02_fit_deterministic")
    c.register_capability(
        "boot",
        "BOOT THE SUBSTRATE LIKE FIRMWARE (cp34): POST -- measured self-checks "
        "with pass/fail per subsystem -- then mount memory and report readiness "
        "in one call. The in-process sibling of autoboot/agent_boot: use this "
        "when the mind already exists and you want its power-on self test. "
        "Carded explicitly in sweep 63: the docstring-derived card was "
        "alias-less and did not surface for its own name (the buried-audit "
        "dark-capability regression).",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(type(m.boot()))",
        native=True,
        # NO "start" phrasings here (sweep 65): they displaced the shipped
        # autoboot alias "how do I start" from its own top-1 -- a new card's
        # battery must also prove it did not STEAL existing aliases.
        aliases=("boot the substrate", "power on self test subsystem fail", "post checks",
                 "run startup self checks", "substrate self check",
                 "is the engine healthy on boot"),
        module="holographic_unified_p20_zoo")
    register_p08_digest(c)
    return 13


def register_p08_digest(c):
    """The document-digest card (learning augmentation) -- appended by the
    digest sweep; called from register_p08 below via the module tail hook."""
    c.register_capability(
        "Document digest (learning augmentation, zero model calls)",
        "One call digests a large text FOR LEARNING: authored TOC (# and "
        "rule+TITLE dialects), kept-negative index (citations; the text stays "
        "in its section), per-section tf*idf signatures. Budgeted markdown "
        "render: negatives funded first, all truncation declared. Runs "
        "AUTOMATICALLY at ingestion -- KnowledgeStore.add files the companion "
        "note for any document >= DIGEST_THRESHOLD; original chunks stay "
        "byte-identical (augment, never edit). mind.document_digest(text).",
        example="m.document_digest('# A\n\nKEPT NEGATIVE: x.\n')['stats']",
        aliases=("digest a big document", "organize a large text for learning",
                 "index the kept negatives", "structure a document with no llm",
                 "auto digest on ingest"),
        native=True,
    )
    c.register_capability(
        "sellmeier_ior",
        "The refractive index of a REAL GLASS at a wavelength (Sellmeier equation), plus "
        "abbe_number for how hard it disperses. mind.sellmeier_ior(nm, glass) with glass "
        "'BK7' | 'SF11' | 'fused_silica'. This is what lets a colour choose its own IOR: "
        "dispersion_spread could already split a bundle across IORs, but nothing could say "
        "which IOR a wavelength HAS, so its test hand-picked two numbers. Low Abbe = strong "
        "rainbow (SF11 25.7 flint); high = weak (BK7 64.2 crown).",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "print(round(float(m.sellmeier_ior(587.5618, 'BK7')), 4), "
                "round(m.abbe_number('SF11'), 2))",
        aliases=("index of refraction for a wavelength", "how much does glass bend blue light",
                 "refractive index of BK7", "sellmeier equation", "abbe number of a glass",
                 "pick a glass for a rainbow", "wavelength to ior"),
        module="holographic_dispersion", method="sellmeier_ior", native=True,
    )
    c.register_capability(
        "spectral_caustics",
        "A caustic WITH ITS COLOUR -- rainbow fringing on focused light. Traces mind.caustics once "
        "per hero wavelength at that wavelength's Sellmeier IOR (Hero Wavelength Sampling, Wilkie "
        "et al. EGSR 2014 -- count, u) and combines through the engine's own observer. anchors=k "
        "beats it: the layer family is rank-3, so 3 traces reconstruct all 16 at 4.9x, rel RGB "
        "error 0.0072, 100.1% of chromatic saturation. anchors=None is byte-identical to full "
        "tracing. Monochrome baseline saturation is exactly 0.0.",
        example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=64, seed=0); "
                "sph=lambda p: np.linalg.norm(p, axis=-1)-1.0; "
                "r=m.spectral_caustics(sph, glass='BK7', count=4, anchors=3, res=32, "
                "receiver_y=-1.6, extent=2.5, n_side=60); print(r['traced'], r['rgb'].shape)",
        aliases=("rainbow caustics", "chromatic aberration in caustics",
                 "coloured light through glass", "prism colours", "dispersion render",
                 "spectral rendering", "hero wavelength sampling", "why is my caustic grey",
                 "make my render show dispersion", "render dispersion", "add dispersion to a render"),
        module="holographic_dispersion", method="spectral_caustics", native=True,
    )
    c.register_capability(
        "rgb_to_spectrum",
        "RGB -> a smooth physical reflectance SPECTRUM (the inverse of spectrum_to_rgb), so ordinary "
        "RGB-authored materials can be rendered spectrally. Jakob & Hanika's sigmoid-of-a-quadratic "
        "space (Eurographics 2019): 3 coefficients, bounded in [0,1] so it cannot invent energy. The "
        "paper ships a 9 MiB table because its fit needs CERES+autodiff; this solves on demand from a "
        "closed-form seed plus Levenberg-Marquardt -- max round-trip error 9.4e-13 over 400 sRGB "
        "colours, 0 bytes of table. BAKE-ONCE at ~0.6 ms; keep the coeffs, never fit per pixel.",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "s=m.rgb_to_spectrum((0.8,0.2,0.2)); "
                "print(len(s), [round(float(v),3) for v in m.reflectance_to_rgb(s)])",
        aliases=("rgb to spectrum", "spectral upsampling", "turn a colour into a spectrum",
                 "make a material spectral", "what spectrum is this colour",
                 "render an rgb texture spectrally", "inverse of spectrum to rgb",
                 "why is my spectrum black", "reflectance from albedo"),
        module="holographic_spectralup", method="rgb_to_spectrum", native=True,
    )
    c.register_capability(
        "spectral_material_ball",
        "THE MATERIAL BALL, RENDERED SPECTRALLY -- the default preview environment (orthographic unit "
        "sphere, one directional light, Cook-Torrance) shaded once per hero wavelength instead of once "
        "in RGB, combined in linear radiance and tone-mapped once. Measured max 0.151 difference "
        "against the RGB ball on the same material: a different image, not a re-tint. KEPT NEGATIVE: "
        "there is no glass= argument -- an opaque BRDF has no transmitted path, so dispersion on a "
        "ball measured 0.001 (invisible). Use spectral_caustics for the rainbow.",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "print(m.spectral_material_ball((0.9,0.3,0.25), count=8, res=48).shape)",
        aliases=("spectral material preview", "material ball with a spectrum",
                 "preview a material spectrally", "render the preview sphere per wavelength",
                 "spectral preview sphere"),
        module="holographic_preview", method="spectral_material_ball", native=True,
    )
    c.register_capability(
        "dispersive_render",
        "DISPERSIVE GLASS IN THE VIEW PATH -- what you SEE through glass: every internally refracted "
        "edge splits into rainbow fringes. The camera-path twin of spectral_caustics (the light path). "
        "Path-traces once per hero wavelength at that wavelength's index. Name a glass, or give n_d + "
        "abbe -- the same slider Blender/LuxCore/Octane expose (cauchy_ior). YOU SEE IT ONLY WHERE THE "
        "REFRACTED IMAGE HAS EDGES: smooth sky measures saturation 0.25, studio sky with HDR panels "
        "0.46. Costs one full path trace per wavelength.",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "print(round(float(m.cauchy_ior(450.0, 1.5168, 25.0)), 4))",
        aliases=("dispersion glass material", "rainbow edges through glass",
                 "see through glass with dispersion", "glass bsdf dispersion",
                 "abbe number glass shader", "chromatic dispersion render",
                 "blender dispersion glass", "prism glass shader"),
        module="holographic_dispersion", method="dispersive_render", native=True,
    )
    c.register_capability(
        "caustic_pass",
        "Project a forward-traced caustic onto a rendered frame as a SEPARATE PASS, then composite -- "
        "how you get a visible caustic into a path-traced image. Brute force is the wrong algorithm: "
        "this tracer has no next-event estimation, so a small bright source arrives as fireflies "
        "(measured -- raising the key 95->220 made raw grain WORSE, 0.76->1.19). mind.caustics "
        "forward-traces instead, deterministic and clean. Project with the SAME window=/center=, pass "
        "occluder_sdf or the pattern paints over the glass casting it. Plane receivers only.",
        example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=64, seed=0); "
                "print(m.composite_caustic(np.full((4,4,3),0.1), np.zeros((4,4,3))).shape)",
        aliases=("caustic pass", "composite a caustic onto a render", "add caustics to a path trace",
                 "why are my caustics all fireflies", "photon map pass", "project a caustic",
                 "caustics on the floor of a render"),
        module="holographic_causticpass", method="caustic_pass", native=True,
    )
    c.register_capability(
        "render_progressive",
        "A render you can PAUSE, RESUME, WATCH WHILE IT RUNS, and finish ACROSS PROCESS RESTARTS -- so "
        "a short-lived environment stops capping the quality you can reach. Splits the SAMPLES into "
        "seeded buckets reduced by sum: Monte Carlo samples are iid, so an image is a mean of batches "
        "and order does not matter. Fixes job_submit's declared 'ATOMIC, do not expect a partial "
        "render' limit. Verified: interrupted == uninterrupted BYTE-IDENTICALLY, including across a "
        "process restart. workers=N runs buckets on separate processes (1.76x on 2 cores).",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.rendering.holographic_progressive import sample_buckets; "
                "print(len(sample_buckets(4, spp=8)), sorted({b['seed'] for b in sample_buckets(4, spp=8)}))",
        aliases=("pause and resume a render", "resume an interrupted render",
                 "render in chunks over time", "progressive render", "checkpoint a render",
                 "render bigger than my time budget", "watch a render as it converges",
                 "distributed render across processes", "finish a render after a restart"),
        module="holographic_progressive", method="render_progressive", native=True,
    )
    c.register_capability(
        "dispersion_scale",
        "The DISPERSION SLIDER: stretch a glass's index spread about its mean. 1.0 is PHYSICAL, larger "
        "is the look. Why it exists, measured: real dispersion is SMALL -- across 420-680nm fused "
        "silica spans 0.0123 of index, BK7 0.0148, SF11 (a dense flint, the hardest here) only 0.0597. "
        "The Cycles 'dispersion glass' setup people recognise stacks IOR 1.35/1.55/1.75 -- spread 0.40, "
        "6.7x SF11, needing an Abbe number near 2.5 that no real glass has. An artistic control, named "
        "as one. Accepted by dispersive_render AND spectral_caustics so one scene has one glass.",
        example="import lecore; from holographic.rendering.holographic_dispersion import exaggerate; "
                "print([round(float(v),4) for v in exaggerate([1.77,1.80,1.83], 6.0)])",
        aliases=("exaggerate dispersion", "make the rainbow stronger", "dispersion slider",
                 "my dispersion is invisible", "why is my glass not rainbow",
                 "stronger chromatic separation", "fake dispersion like blender"),
        module="holographic_dispersion", method="exaggerate_dispersion", native=True,
    )
    c.register_capability(
        "reject_outliers",
        "FIREFLY REJECTION, free from the progressive split: combine_buckets(reject=k) drops bucket "
        "values more than k MADs from the per-pixel median and averages the rest. A firefly is one "
        "rare bright path, so it lands in ONE bucket and the others disagree. Beats clamp_fireflies "
        "for spectral renders -- clamping compares a pixel to a GLOBAL percentile and cannot tell a "
        "firefly from a real highlight; this compares a pixel to ITSELF, so consensus is untouched. "
        "MAD not sigma: a 500x sample inflates sigma enough to protect itself.",
        example="import numpy as np; from holographic.rendering.holographic_progressive import combine_buckets; "
                "b=[np.ones((2,2,3)) for _ in range(5)]; b[2]=b[2].copy(); b[2][0,0]=500.0; "
                "print(round(float(combine_buckets(b)[0,0,0]),1), round(float(combine_buckets(b, reject=3.0)[0,0,0]),1))",
        aliases=("firefly removal", "reject outlier samples", "speckles in my render",
                 "bright dots in a path trace", "robust mean across buckets",
                 "salt and pepper noise in a render", "median of several renders"),
        module="holographic_progressive", method="combine_render_buckets", native=True,
    )
    c.register_capability(
        "specular_connect",
        "Solve for the point on a refractive surface connecting a LIGHT to a SHADE POINT -- the caustic "
        "connection plain next-event estimation CANNOT make: NEE uses a straight shadow ray and a "
        "caustic path goes THROUGH glass, which bends it. The gap Manifold NEE and Specular Manifold "
        "Sampling (Zeltner et al. 2020, used by Cycles) close: solve for the specular vertex instead "
        "of hoping a bounce hits it. A Newton step in the tangent plane, re-projected onto the SDF -- "
        "iterate a projection. Check `converged`. ONE interface.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                "o=m.specular_connect(sphere(1.0), np.array([[0.,1.,0.]]), np.array([[-1.,3.,0.4]]), "
                "np.array([[0.6,-2.2,-0.3]]), 1/1.5); print(bool(o['converged'][0]))",
        aliases=("manifold next event estimation", "specular manifold sampling",
                 "caustics in the path tracer", "connect through glass to a light",
                 "why are my caustics missing", "solve for a refraction point", "MNEE", "SDS path"),
        module="holographic_specularconnect", method="specular_connect", native=True,
    )
    c.register_capability(
        "mesh_query_chunk",
        "Why a big SDF bake gets OOM-KILLED, and the knob that stops it. The mesh-distance kernel held a "
        "(N,(2r+1)^3,3) neighbour block -- MEASURED 36.8 KB peak RSS per query -- so a 176^3 bake needed "
        "37 GB for a 43 MB grid. Now STREAMED: mesh_point_distance takes max_bytes (default 512 MB) and "
        "peak memory is FLAT in the query count (0.58 GB at 200k points, 0.62 GB at 800k), staying "
        "bit-identical at any chunk size. This verb reports the block a budget buys, so you can size a "
        "bake before it dies. Cost scales with RADIUS, not mesh size.",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "print(m.mesh_query_chunk(radius=2), m.mesh_query_chunk(radius=3))",
        aliases=("out of memory baking an sdf", "killed while baking a distance field",
                 "my bake gets oom killed", "how much memory does a mesh distance query need",
                 "sdf bake resolution limit", "process killed at 128 cubed",
                 "distance field too big for ram", "chunk a point to mesh query",
                 "why can't I bake at higher resolution", "memory budget for mesh distance"),
        module="holographic_meshbridge", method="mesh_query_chunk", native=True,
    )
    # ------------------------------------------------------------------------------------------------------
    # THE HOLOGRAPHIC RENDER LINEAGE, RE-CATALOGUED. Audit (sweep 149): render_baked, bake_scene,
    # radiance_transfer, render_dispatch, plan_render, holographic_fog_volume and the two field classes had
    # ZERO catalog cards -- wired as verbs, invisible to find_capability/suggest. "precomputed radiance
    # transfer" returned "Rendering (path trace)" first; "relight with a dot product" surfaced nothing of the
    # kind. Every conventional path (path_trace, caustics, spectral_caustics, progressive) HAD cards with
    # generous aliases, so the semantic surface steered every render question toward Monte Carlo -- which is
    # exactly how sweeps 138-148 walked past the substrate and rebuilt what Cycles does. By this engine's own
    # rule a capability that cannot be surfaced does not exist; these did not, and the drift followed.
    # ------------------------------------------------------------------------------------------------------
    c.register_capability(
        "Caustic as one hypervector, wavelength as an axis (holographic_caustic)",
        # NAME NOTE: capdoc.generate_json drops every card whose NAME starts with "holographic_" (it treats
        # those as auto-generated module stubs), and mind.suggest ranks over that curated set -- so a card
        # named "holographic_caustic" was findable by find_capability and INVISIBLE to suggest. Found by
        # testing the aliases; the descriptive name is the house style anyway.
        "The caustic as ONE HYPERVECTOR, wavelength as an axis -- RENDER = QUERY. Trace the light ONCE with a "
        "per-ray spectrum, BUNDLE the landings into an FPE field, READ the image at any resolution: "
        "RGB is three UNBINDS (the colour-matching integral folded into the query). .translate is a bind, .add "
        "superposes another object. Measured 0.80-0.95 agreement with the histogram (dim 1024-16384), 5x "
        "smoother at equal rays. KEPT NEGATIVE: colour separation is CAPACITY-BOUNDED (same-geometry reads at two "
        "wavelengths correlate ~0.75 at dim 2048, above a ~3% physical dispersion). Raise dim or tile.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                "h=m.holographic_caustic(sphere(0.6), light_dir=(0.35,-1,0), receiver_y=-1.2, extent=0.9, "
                "n_side=100, window=1.2, dim=1024, aim=(0,0,0)); xs=np.linspace(-1.2,1.2,32); "
                "print(h.read_rgb(xs, xs, m.wavelength_cmf).shape)",
        aliases=("caustic as a hypervector", "render the caustic from a superposition", "caustics without a histogram",
                 "read the caustic at any resolution", "wavelength as a dimension", "spectral caustic in one pass",
                 "bundle the light instead of binning it", "holographic dispersion", "render is a query",
                 "render an image by projecting from a superposition", "project a superposition to an image",
                 "image as a projection of a hypervector", "caustics without photon tracing",
                 "caustics without shooting photons", "dispersion as a bound wavelength role",
                 "wavelength as a role", "bind the wavelength", "render from the substrate"),
        module="holographic_holocaustic", method="holographic_caustic", native=True,
    )
    c.register_capability(
        "Tiled holographic caustic (break the capacity wall)",
        "holographic_caustic with the receiver split into grid x grid tile vectors, each read routing its own "
        "pixels -- the capacity wall broken the way TiledRadianceField breaks it. Build 50.7s (one dim-16384 "
        "vector) -> 4.6-11s (tiles); 0.87 agreement with the true caustic at equal ray budget. KEPT NEGATIVE: with "
        "landings per tile still above the tile's dim, each tile shows its own crosstalk texture -- a PATCHWORK at "
        "borders; use more tiles. The histogram on an analytic SDF is 2.3s: the substrate wins on resolution "
        "independence, composability and one-pass spectrum here, not raw speed.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                "t=m.holographic_caustic_tiled(sphere(0.6), light_dir=(0.35,-1,0), receiver_y=-1.2, extent=0.9, "
                "n_side=100, window=1.2, grid=2, dim=1024, aim=(0,0,0)); xs=np.linspace(-1.2,1.2,32); "
                "print(t.read_rgb(xs, xs, m.wavelength_cmf).shape)",
        aliases=("caustic too blotchy raise resolution", "holographic caustic patchwork", "tile the caustic field",
                 "caustic field capacity wall", "more landings than dimensions", "tiled hypervector caustic"),
        module="holographic_holocaustic", method="holographic_caustic_tiled", native=True,
    )
    c.register_capability(
        "Glass as a baked spectral-refractive transfer: trace once, relight by dot product",
        "bake_glass traces a dielectric's refraction ONCE per pixel for K wavelengths -- entry, interior, deterministic "
        "internal TIR bounces, exit direction, Fresnel split -- and relight_glass shades it against an env_field (the "
        "light as a hypervector over directions) in one matmul. No Monte Carlo: the tracer's only randomness was its "
        "estimator. MEASURED, 1-fold Mandelbox 320x200: bake 1.6s + relight 7s/light vs ~5 min at 192 spp, no grain. "
        "KEPT NEGATIVE: bounces past the cap are lost (3.0%%; 65.5%% before the march fix); a sharp light seen "
        "through glass reads soft.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                "cam=m.camera(eye=(0,1,4), target=(0,0,0), fov_deg=35, aspect=1.0); "
                "b=m.bake_glass(sphere(0.8), cam, 32, 32, floor_y=-0.9); "
                "e=m.env_field().add_softbox((3,4,2),(0,0,0),2.0,2.0,40.0); print(m.relight_glass(b, e).shape)",
        aliases=("render glass fast", "glass render without path tracing", "relight glass without re-rendering",
                 "bake the refraction once", "dispersion render in seconds", "why does my glass render take an hour",
                 "deterministic glass", "environment light as a hypervector", "render is a dot product",
                 "collapse the glass render", "no noise glass render"),
        module="holographic_glassbake", method="bake_glass", native=True,
    )
    c.register_capability(
        "Optics of a named material: index, dispersion, absorption in one door (glass_optics)",
        "glass_optics('ruby') -> {n_d 1.77, abbe 72.2, absorb per-RGB, tint}: everything a physically based glass "
        "render needs to refract, DISPERSE and colour a named material. The library had an index and an absorption per "
        "gem but NO dispersion, so a ruby could never split light. Catalogue values: diamond 44.3 on n 2.42 (its "
        "'fire' is a modest Abbe number on a high index), corundum 72.2, beryl 56, quartz 70, water 55.7. "
        "bake_glass(material=) / relight_glass(material=) read it; a thick ruby comes out redder than a thin edge.",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.materials_and_texture.holographic_matlib import glass_optics; "
                "print(glass_optics('diamond')['abbe'], glass_optics('ruby')['absorb'])",
        aliases=("render a ruby", "diamond dispersion", "abbe number of sapphire", "physical gem material",
                 "material index of refraction and dispersion", "make the glass a real material",
                 "how much does water disperse", "beer lambert absorption of a gem"),
        module="holographic_matlib", method="bake_glass", native=True,
    )
    c.register_capability(
        "Light a render with a real HDRI (.exr in, exact floor irradiance, the map picks the shadow)",
        "m.load_exr(path) reads OpenEXR (opt-in `pip install OpenEXR`; load_hdr stays the stdlib RGBE door) -> "
        "linear (H,W,3). m.hdri_env(img) makes it the light: .radiance(dirs) via sky_dome, plus the map's own pixels "
        "integrated once for the floor -- exact cosine-weighted irradiance, the dominant lobe, and that lobe's SHARE "
        "(0.41 for a lounge sun, 0.11 for cafeteria ceiling lights): the map decides how hard the shadow is. "
        "relight_glass(bake, m.hdri_env(img)): the background is the photograph, the gem refracts the room.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "img=np.zeros((32,64,3),np.float32); img[:16]=2.0; e=m.hdri_env(img); "
                "print(round(float(e.floor_irradiance.mean()),2), round(e.sun_share,2))",
        aliases=("load an exr", "use an hdri to light the scene", "environment map lighting", "image based lighting",
                 "light the glass with a photo", "openexr environment map", "where is the sun in my hdri",
                 "how hard should the shadow be", "poly haven hdri"),
        module="holographic_glassbake", method="hdri_env", native=True,
    )
    c.register_capability(
        "HDRI base light plus extra key/rim lobes for dispersion and caustics (env_add_light)",
        "The studio rig on top of a photograph: m.env_add_light(env, direction, irradiance=..., sigma=0.045) layers an "
        "analytic Gaussian lobe on an hdri_env. Size it by the FLOOR irradiance it adds (a fraction of "
        "env.floor_irradiance.mean() keeps the map as the base -- measured: peak radiance 3000 took a lobe to a 0.96 "
        "share and the HDRI stopped mattering). The lobe also updates floor_irradiance, dominant_dir and sun_share, so "
        "relight_glass shades and shadows consistently and the caustic aims along the strongest lobe. Brightness "
        "belongs to exposure (auto-EV from the floor median), never to the lights.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "img=np.zeros((32,64,3),np.float32); img[:16]=0.5; e=m.hdri_env(img); E0=float(e.floor_irradiance.mean()); "
                "m.env_add_light(e,(-0.45,0.8,0.4),irradiance=0.8*E0,sigma=0.045); "
                "print(round(float(e.floor_irradiance.mean())/E0,2), round(e.sun_share,2))",
        aliases=("add a key light to the hdri", "extra light on top of the environment map", "show off the caustics",
                 "rim light for the gem", "hdri plus studio lights", "without blowing out the scene",
                 "more sparkle in the crystal", "light the crystal for dispersion"),
        module="holographic_glassbake", method="env_add_light", native=True,
    )
    c.register_capability(
        "A single named crystal as a closed SDF (crystal_single)",
        "m.crystal_single('quartz', size) -- one specimen from the same builder crystal_cluster and crystal_geode "
        "place many of (habit_sdf). WHY: crystal_habit with a bare Miller list is an OPEN prism (the hexagonal "
        "(100)+(101) pair filled 16% of a probe and ran off its edge) until form=True expands each index into its "
        "whole form; this path always does. Habits: quartz, beryl, cube, octahedron, dodecahedron, needle. Hand it to "
        "bake_glass(..., material='quartz') for the gem render.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); s=m.crystal_single('quartz',0.9); "
                "g=np.linspace(-1.8,1.8,21); G=np.stack(np.meshgrid(g,g,g,indexing='ij'),-1).reshape(-1,3); "
                "d=np.asarray(s.eval(G)); print(round(float((d<0).mean()),3), float(np.abs(G[d<0]).max())<1.6)",
        aliases=("one quartz crystal", "single crystal shape", "a crystal point", "make one crystal not a cluster",
                 "closed crystal sdf", "crystal specimen"),
        module="holographic_crystalgrow", method="crystal_single", native=True,
    )
    c.register_capability(
        "Culled union: evaluate only the crystals that can be nearest (cull=True)",
        "crystal_cluster / crystal_geode / crystal_grow_on accept cull=True (default off): each placed crystal gets a "
        "bounding sphere; per point the most promising member is evaluated exactly, then only members whose bound "
        "beats that value. Lever 5, exact where it matters: same zero set, identical values near the surface, never a "
        "smaller distance (a tighter, valid sphere-tracing field). MEASURED: 60-crystal geode 2.40s -> 0.30s (8.1x) on "
        "200k points, 11-crystal cluster 2.5x. Per-CALL overhead remains -- for a render, bake_sdf the body once instead.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); Q=np.random.default_rng(0).uniform(-1,1,(2000,3)); "
                "a=np.asarray(m.crystal_cluster(count=5,seed=1).eval(Q)); b=np.asarray(m.crystal_cluster(count=5,seed=1,cull=True).eval(Q)); "
                "print(bool(np.array_equal(a<0,b<0)), bool(np.all(b>=a-1e-12)))",
        aliases=("cluster render is slow", "geode sdf too slow", "speed up the crystal union", "bounding sphere culling",
                 "many crystals evaluate faster", "spatial culling for sdf union"),
        module="holographic_crystalgrow", method="crystal_cluster", native=True,
    )
    c.register_capability(
        "Rock and glass in one bake: an opaque body beside the gem (bake_glass opaque=)",
        "bake_glass(lining, ..., opaque=rind) adds a second SDF shaded as Lambert rock: it wins the pixel where nearer "
        "than the glass, and exit/reflection rays that strike it are recorded at bake time, so relight shows the cavity "
        "wall THROUGH the crystals. relight_glass(..., opaque_albedo=) shades it: map over a cosine hemisphere + each "
        "lobe with a shadow ray + the map's sun from its floor share. Geode: crystal_geode(parts=True, clip_to_skin=True) "
        "-> (rind, lining), crystal_cut each with one plane. Before: the rind rendered as violet glass, 39% of the "
        "crystals stuck out of the nodule.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere, plane; "
                "cam=m.camera(eye=(0,3,0.01),target=(0,0,0),fov_deg=40,aspect=1.0); "
                "b=m.bake_glass(sphere(0.6),cam,24,24,n_lams=3,opaque=plane(-1.5)); "
                "print(int(b.opaque.sum())>50, bool(b.exit_opq[:,b.glass].mean()>0.5))",
        aliases=("geode with a rock rind", "opaque rock next to the glass", "crystals growing out of stone",
                 "render the geode properly", "matrix under the crystal cluster", "see the cavity wall through the crystals",
                 "rind should not be transparent", "cut geode render"),
        module="holographic_glassbake", method="bake_glass", native=True,
    )
    c.register_capability(
        "Checkerboard floor, mottled rock, grey backdrop: albedo fields and background in relight_glass",
        "relight_glass(bake, env, floor_albedo=fn, opaque_albedo=fn, background=(r,g,b)): the albedos accept a "
        "CALLABLE P(m,3)->(m,3) -- a checkerboard is light/dark by floor(x/s)+floor(z/s) parity; weathered stone is a "
        "rock colour mixed by a baked holographic fBm (procedural_noise().sample_grid_fast(48) read via GridSDF; the "
        "same field displaces the rind SDF, x0.7 to stay Lipschitz). `background` paints only pixels that see nothing; "
        "the HDRI still lights and reflects. composite_caustic(receiver_mask=bake.floor) keeps the caustic off the body.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                "cam=m.camera(eye=(0,0.8,3),target=(0,0.3,0),fov_deg=60,aspect=1.0); "
                "b=m.bake_glass(sphere(0.5),cam,24,24,n_lams=3,floor_y=-0.5); "
                "ck=lambda P: np.where(((np.floor(P[:,0]/0.5)+np.floor(P[:,2]/0.5)).astype(int)%2==0)[:,None],[[0.9]*3],[[0.1]*3]); "
                "img=m.relight_glass(b,lambda D: np.ones((len(D),3)),floor_albedo=ck,background=(0.3,0.3,0.3)).reshape(-1,3); "
                "print(bool((img[b.floor,0]>0.5).any() and (img[b.floor,0]<0.12).any()), bool(np.allclose(img[~(b.glass|b.floor)],0.3)))",
        aliases=("checkerboard floor", "solid gray background", "use the hdri for lighting but not as background",
                 "rough rock texture", "outside of the geode should be rough", "caustic drawn on top of the object",
                 "textured floor under the gem", "mottled stone albedo"),
        module="holographic_glassbake", method="relight_glass", native=True,
    )
    c.register_capability(
        "Imperfections inside a gem as one hypervector: milky quartz, inclusions, colour zoning (gem_flaw_volume)",
        "m.gem_flaw_volume(field=any P->[0,1] such as crystal_cloudiness, bounds) bundles the flaw density into ONE FPE "
        "vector; relight_glass(flaw_volume=fv, flaw_sigma, flaw_albedo) reads its integral along each pixel's recorded "
        "interior path in CLOSED FORM (holographic_volint, no marching): extinction + single-scatter glow. "
        "absorb_volume + absorb_volume_sigma(rgb) is COLOUR ZONING the same way (amethyst purple at the tips). MEASURED: "
        "vs a 400-step march 12%; crosstalk floor ~10% at dim 4096. Before, only the MC tracer had flaws.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                "cam=m.camera(eye=(0,0,3),target=(0,0,0),fov_deg=30,aspect=1.0); b=m.bake_glass(sphere(0.6),cam,16,16,n_lams=3); "
                "fv=m.gem_flaw_volume(field=lambda P: np.ones(len(P)),bounds=[(-0.7,0.7)]*3,res=6,dim=512); "
                "env=lambda D: np.ones((len(D),3)); pure=m.relight_glass(b,env); milky=m.relight_glass(b,env,flaw_volume=fv,flaw_sigma=5.0,flaw_albedo=(0.5,0.5,0.5)); "
                "print(len(b.segs)>0, not np.allclose(pure,milky))",
        aliases=("crystals are too pure", "add inclusions to the gem", "milky quartz render", "cloudy crystal base",
                 "amethyst purple only at the tips", "colour zoning in a crystal", "rutile needles inside quartz",
                 "imperfections in the glass bake", "translucent scattering gem"),
        module="holographic_gemvolume", method="gem_flaw_volume", native=True,
    )
    c.register_capability(
        "Mixed minerals in one growth: per-seed habits with their own lattices (grow_on habit=list|callable)",
        "crystal_grow_on / crystal_cluster / crystal_geode take habit=('quartz','cube','dodecahedron') (each seed draws "
        "one, seeded) or habit=lambda p: 'calcite-like cube' if a field says so else 'quartz' -- and size={habit: s} "
        "per mineral. Every habit keeps its own Bravais system and forms (HABITS), so a hexagonal point grows beside a "
        "cubic one as itself. The single-name path is byte-identical to before (pinned by test). Culling works across "
        "habits (one bounding radius each).",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); Q=np.random.default_rng(0).uniform(-1,1,(3000,3)); "
                "d=np.asarray(m.crystal_cluster(count=8,habit=('quartz','cube'),size={'quartz':0.35,'cube':0.22},seed=3,cull=True).eval(Q)); "
                "print(bool((d<0).any()))",
        aliases=("different crystal types growing together", "mixed habits in a cluster", "quartz and calcite on the same rock",
                 "which mineral grows where", "crystal lattices for mixed types", "per-seed crystal habit"),
        module="holographic_crystalgrow", method="crystal_grow_on", native=True,
    )
    c.register_capability(
        "Spectral confetti cure: wavelength stratification + spectral ray differential (lam_jitter, footprint_filter)",
        "A dispersive body at 11 hero wavelengths paints a checker as coloured confetti, and one wavelength exiting "
        "straight at the key reads its peak as a saturated speck. Two default-off cures: bake_glass(samples>1, "
        "lam_jitter=True) shifts each sub-bake's wavelength grid (3 x 13 = 39 wavelengths, no extra cost per sub-bake); "
        "relight_glass(footprint_filter=True) reads the floor over the footprint the spread between neighbouring "
        "wavelengths' exits makes, and each lobe fan-averaged at conserved energy. Zero spread = identical. Measured on "
        "a diamond blob: confetti gone, fire kept.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                "cam=m.camera(eye=(0,1.2,3),target=(0,0,0),fov_deg=35,aspect=1.0); "
                "b=m.bake_glass(sphere(0.6),cam,16,16,n_d=2.4,abbe=10.0,n_lams=5,floor_y=-0.7,samples=2,lam_jitter=True); "
                "ck=lambda P: np.where(((np.floor(P[:,0]/0.2)+np.floor(P[:,2]/0.2)).astype(int)%2==0)[:,None],[[0.9]*3],[[0.1]*3]); "
                "env=lambda D: np.ones((len(D),3)); a=m.relight_glass(b,env,floor_albedo=ck); f=m.relight_glass(b,env,floor_albedo=ck,footprint_filter=True); "
                "print(len(set(tuple(np.round(s.lams,4)) for s in b.subs))==2, bool(f.std()<=a.std()))",
        aliases=("confetti in the glass render", "coloured speckle in dispersion", "noisy rainbow pixels in the diamond",
                 "spectral aliasing", "more wavelengths without more cost", "ray differentials for refraction",
                 "checker looks noisy through the glass"),
        module="holographic_glassbake", method="bake_glass", native=True,
    )
    c.register_capability(
        "Real unit cells for crystal habits (real_cell=True): quartz's 141 deg 47' angle, not 139",
        "Every crystal in leCore is the intersection of lattice half-spaces (crystal_habit) -- no billboards, no "
        "sprites: a true convex distance field with unit gradient (pinned). But the lattice basis defaults to c = a, "
        "which is not quartz (a 4.913, c 5.405 A): the prism-rhombohedron interfacial angle came out 139.1 deg instead "
        "of the textbook 141 deg 47'. crystal_single / crystal_grow_on / crystal_cluster / crystal_geode take "
        "real_cell=True to build each habit on its mineral's axial ratio (CELLS: quartz 1.1001, beryl 0.9976). Default "
        "off keeps old fields byte-identical; new work should pass True.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_bravais import lattice_basis, reciprocal_basis; "
                "B=reciprocal_basis(lattice_basis('hexagonal',a=1.0,c=1.1001)[0]); r=np.array([1.,0,1])@B; p=np.array([1.,0,0])@B; "
                "print(round(180-np.degrees(np.arccos(r@p/np.linalg.norm(r)/np.linalg.norm(p))),1))",
        aliases=("physically accurate crystal lattice", "real quartz unit cell", "interfacial angle of quartz",
                 "are the crystals billboards", "crystals should be real geometry", "correct c/a ratio"),
        module="holographic_crystalgrow", method="crystal_single", native=True,
    )
    c.register_capability(
        "Histogram caustic baseline for spectral landings (caustic_histogram_rgb)",
        "Bin spectral_landings with colour-matching weights and a 1-px Gaussian: the ground truth the holographic "
        "caustic is measured against. MEASURED on an amethyst plate (25k landings): the tiled holographic read (grid 12, "
        "dim 2048) smeared the filaments into blotches and showed tile seams; the histogram resolved them with "
        "dispersion at the edges (test: a point source reads >3x sharper). Use it whenever the landing set exceeds the "
        "holographic capacity; feed the result to caustic_pass / composite_caustic as before.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); rng=np.random.default_rng(0); "
                "xz=rng.normal([0.3,-0.2],0.01,(2000,2)); lam=rng.uniform(420,680,2000); "
                "H=m.caustic_histogram_rgb(xz,lam,((-1,1),(-1,1)),res=64); iy,ix=np.unravel_index(np.argmax(H.mean(-1)),H.shape[:2]); "
                "print(abs(np.linspace(-1,1,64)[ix]-0.3)<0.05, abs(np.linspace(-1,1,64)[iy]+0.2)<0.05)",
        aliases=("caustic looks like a blob", "sharper caustics", "holographic caustic too blurry", "tile seams in the caustic",
                 "ground truth caustic image", "bin the landings"),
        module="holographic_holocaustic", method="caustic_histogram_rgb", native=True,
    )
    c.register_capability(
        "Bake once, relight by dot product (bake_scene / render_baked)",
        "COLLAPSE, DON'T TRACE -- the render lineage this engine started from. bake_scene(sdf, camera, w, h, methods, "
        "colors) traces primary visibility ONCE, dispatches each hit to 'collapse' (PRT) or 'trace' (a mirror bounce), "
        "and precomputes the radiance-transfer vector at every diffuse hit. render_baked(scene, light) then shades every "
        "pixel as a DOT PRODUCT -- no rays; every frame, the first included, is a relight. Measured 57x per relight "
        "over re-traced shadows; 0.0036s vs 0.0545s all-trace (15x). Break-even ~160 relights: for INTERACTIVE "
        "relighting over fixed geometry, not one still.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "class S:\n    cs=np.array([[0,0,0.],[-1.6,0,0]]); cols=np.array([[.7,.7,.7],[.8,.3,.3]])\n"
                "    def eval(s,P): return np.min(np.stack([np.linalg.norm(P-c,axis=1)-0.8 for c in s.cs]),axis=0)\n"
                "    def ids(s,P): return np.argmin(np.stack([np.linalg.norm(P-c,axis=1) for c in s.cs]),axis=0)\n"
                "cam=m.camera(eye=(0,1.2,5.5), target=(-0.8,0,0), fov_deg=40, aspect=1.0); "
                "sc=m.bake_scene(S(), cam, 24, 24, {0:'trace',1:'collapse'}, S.cols, n=120); "
                "warm=lambda w: np.clip(w@np.array([0.4,0.7,0.3]),0,1)[:,None]*np.ones(3)+0.05; "
                "print(m.render_baked(sc, warm).shape)",
        aliases=("relight without re-rendering", "render as a dot product", "bake once render many",
                 "collapse dont trace", "precomputed radiance transfer render", "interactive relighting",
                 "why is my render slow every frame", "move the light for free", "first frame is a relight"),
        module="holographic_dispatch", method="bake_scene", native=True,
    )
    c.register_capability(
        "Precomputed radiance transfer (PRT): shading as a dot product",
        "The per-point VISIBILITY INTEGRAL is what makes global illumination expensive, and for static geometry it "
        "depends only on geometry -- so precompute it ONCE as a spherical-harmonic transfer vector per surface point "
        "(radiance_transfer), project the light onto SH, and shading COLLAPSES to transfer @ light (Sloan/Kautz/Snyder "
        "2002). VSA framing: the transfer vector is a per-point codebook entry, relight a readout. Measured 3.82s "
        "precompute once, 0.0004s per relight (57x). Low-frequency, diffuse, static geometry -- the documented "
        "limits. The 'collapse the wave function' answer to path tracing.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                "pts=np.array([[0,1.0,0],[1.0,0,0]]); T=m.radiance_transfer(sphere(1.0), pts, pts, order=3, n=200); "
                "print(T.shape)",
        aliases=("precomputed radiance transfer", "PRT", "spherical harmonics lighting", "shade with a dot product",
                 "collapse the wave function instead of path tracing", "visibility integral once",
                 "global illumination without tracing every frame", "transfer vector per point"),
        module="holographic_prt", method="radiance_transfer", native=True,
    )
    c.register_capability(
        "Adaptive render: one call that picks collapse vs trace vs bake (render_adaptive / plan_render)",
        "ONE render call that ADAPTS, grounded in MEASURED break-evens: bake the SDF only when primitives or frames "
        "make it pay (bake loses under ~16 primitives single-frame, wins 6.4x at 64); collapse diffuse surfaces (PRT, "
        "free relight) and trace reflective ones, deriving each surface's method from its material; keep the exact "
        "marcher. plan_render returns the plan WITH A REASON for every choice, without rendering. This is the top of "
        "the holographic render stack -- the door to use before reaching for path_trace. Returns (frame, relight, "
        "plan).",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "from holographic.simulation_and_physics.holographic_semantic import parse_description; "
                "objs=parse_description('a red ball beside a mirror box')['objects']; "
                "print(m.plan_render(objs, frames=8, relight=True)['reasons'][:2])",
        aliases=("which render method should I use", "let the renderer choose", "adaptive rendering pipeline",
                 "auto select bake or trace", "render plan with reasons", "fast render for many frames",
                 "one render call that adapts", "relight or trace decision"),
        module="holographic_adaptive", method="render_adaptive", native=True,
    )
    c.register_capability(
        "Density field as one hypervector: closed-form ray integral, no marching (holographic_fog_volume)",
        "A traditional renderer has NO model of empty space -- it marches to find out. holographic_fog_volume carries "
        "the WHOLE density field, occupied and empty, as ONE FPE hypervector, and the line integral of density along "
        "any ray has a CLOSED FORM: one inner product per ray, no steps (the FPE basis is a phase code; the integral "
        "of a complex exponential is a complex exponential). Measured EXACT against a 160-step march (corr 1.0000), "
        "~90x faster at image scale; empty space reads tau~0 with no marching. Optical depth / extinction only -- "
        "emissive self-shadowing media still march (documented).",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "vol=m.holographic_fog_volume([[0,0,0],[1,0.5,0]], [1.0,0.6], bounds=[(-3,3)]*3, dim=1024); "
                "print(round(float(vol.optical_depth(np.array([[-3,0,0.]]), np.array([[1,0,0.]]), 6.0)[0]),3))",
        aliases=("fog without ray marching", "the field knows where empty space is", "closed form ray integral",
                 "volume as one hypervector", "optical depth in one inner product", "atmosphere in closed form",
                 "no model of empty space", "density field as a vector"),
        module="holographic_volint", method="holographic_fog_volume", native=True,
    )
    c.register_capability(
        "Gaussian blur an image (reflect or wrap borders, channels untouched)",
        "m.blur_image(image, sigma, mode='reflect'|'wrap'): the plain Gaussian low-pass every image app needs -- separable "
        "and reflect-padded for a canvas, FFT-circular for a tiling texture. WHY it is a card: the engine held four "
        "private Gaussian blurs (autobump, splatsharpen, sharpen, postfx) and exposed none, so leStudio wrote three of "
        "its own and Poly Studio one -- the most re-implemented helper across the apps (sweep 163 app_lint). Works on "
        "(H,W) and (H,W,C).",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=64,seed=0); img=np.zeros((32,32,3)); img[12:20,12:20]=1; "
                "b=m.blur_image(img, 2.0); print(b.shape, round(float(b.sum()/img.sum()),3), round(float(b[16,16,0]),3))",
        aliases=("gaussian blur an image", "blur an image", "soften an image", "low pass filter an image", "smooth pixels",
                 "separable gaussian", "blur with wrap around edges", "feather a mask"),
        module="holographic_autobump", method="blur_image", native=True,
    )
    c.register_capability(
        "Render quality gate (absolute defect thresholds, not diff-against-last-render)",
        "m.render_quality_gate(frame, limits, single_round) -> {metrics, failed, ok} (holographic_qualitygate, from Poly "
        "Studio). Each metric names the defect that shipped: terracing (second-difference steps in the floor band -- "
        "grid-baked iso-contours; good 0.018, shipped-bad 0.068), edge_tones (silhouette pixels with a partial tone "
        "nearby -- 0.94 supersampled, 0.00 jagged), fringe_ratio (chroma ADDED to edges by convergence -- a misaligned "
        "albedo). WHY absolute: a diff against the previous render cannot see a defect both frames share. A metric is "
        "never sufficient: look at the frames too.",
        example="import numpy as np, lecore; from holographic.rendering.holographic_qualitygate import _disc; m=lecore.UnifiedMind(dim=64,seed=0); "
                "good=_disc(120,160,80,60,30,ss=4,contrast=True); jag=_disc(120,160,80,60,30,ss=1,contrast=True); "
                "print(m.render_quality_gate(good)['ok'], m.render_quality_gate(jag)['failed'])",
        aliases=("did my render regress", "check a render for terracing", "jagged edges test", "aliasing regression gate",
                 "colour fringe on silhouettes", "render regression thresholds", "quality gate before shipping a render change",
                 "both renders have the same artifact"),
        module="holographic_qualitygate", method="render_quality_gate", native=True,
    )
    c.register_capability(
        "Plugins (extend a mind's verbs without growing the core)",
        "A plugin is a module with PLUGIN={name,version,does,requires,install} and register(mind, config) "
        "returning verb dicts. DISCOVERED at construction: holographic/plugins/ (bundled jit, symbolic, zig, wgsl, "
        "gpu, lean4 -- each optional dependency's verbs, named after its pip extra), LECORE_PLUGIN_PATH folders "
        "(per app), pip entry points (lecore.plugins). UnifiedMind(plugins=()) is slim; plugins=('zig',) picks by "
        "name. plugin_list() is the preflight: available/missing/install. Verbs bind AND get cards; shadowing a "
        "core faculty is refused. Loading from a caller path is private (holographic_plugin).",
        example="import lecore; m=lecore.UnifiedMind(dim=64,seed=0); "
                "print([(p['name'], p['available']) for p in m.plugin_list()]); "
                "print(hasattr(lecore.UnifiedMind(dim=64,seed=0,plugins=()), 'zig_batch_eval'))",
        aliases=("plugin system", "add a plugin", "load a plugin", "extension point", "hook system for user code",
                 "register a new tool the mind can call", "add custom functionality without changing core",
                 "third party addon", "extend lecore per app", "register an extension at runtime",
                 "optional feature as a plugin", "add a verb at runtime", "unload a plugin",
                 "what plugins are loaded", "keep optional deps out of core", "slim mind without optional stuff",
                 "which optional dependencies are installed", "plugin folder", "pip extra for a plugin"),
        module="holographic_plugin", method="plugin_list", native=True,
    )
    c.register_capability(
        "The HRR algebra for builders (bind / bundle / unbind / cosine / nearest / derived_atom)",
        "The primitives the engine is built from, in one readable module (holographic_ai): derived_atom(seed, "
        "name, dim) mints a vector that is a pure function of its name; bind(a, b) associates two (circular "
        "convolution: resembles neither, pairing recoverable); bundle(vs) superposes a set (resembles each "
        "member); unbind(c, a) recovers the other half, noisily; nearest(q, codebook) / cosine(a, b) snap and "
        "score. What a PLUGIN reaches for to add a capability ON the framework -- see "
        "holographic/plugins/_example_tags.py, a one-vector tag memory with abstention. Plate 1995.",
        example="from holographic.agents_and_reasoning import holographic_ai as A; "
                "k=A.derived_atom(0,'key',256); v=A.derived_atom(0,'value',256); "
                "t=A.bind(k,v); print(round(A.cosine(A.unbind(t,k), v), 2))",
        aliases=("bind two vectors", "bundle several vectors", "unbind a pair", "the hrr algebra",
                 "holographic reduced representation", "role filler binding", "circular convolution bind",
                 "superpose vectors", "make a vector from a name", "deterministic atom for a symbol",
                 "cosine similarity of hypervectors", "nearest vector in a codebook",
                 "how do i build on the holographic framework", "primitives for a plugin"),
        module="holographic_ai", method=None, native=True,
    )
    # Three faculties the buried audit flagged DARK (sweep 168): auto-carded from a docstring, not in
    # their own top-15, no aliases -- a user could only find them by already knowing the name.
    c.register_capability(
        "make_light",
        "mind.make_light(kind, **params) returns a path-tracer light record by NAME -- 'sun', 'point', 'spot', "
        "'area' (alias 'softbox'), 'dome' -- with sensible defaults so a scene can be lit in one call and "
        "adjusted after. The names are the ones a lighting artist uses, not the sampler's.",
        example="import lecore; m=lecore.UnifiedMind(dim=64,seed=0); print(m.make_light('sun'))",
        aliases=("add a sun light", "make a point light", "create a spotlight", "add an area light",
                 "softbox light", "dome light", "add a light to the scene", "key light", "light the scene"),
        module="holographic_lights", method="make_light", native=True,
    )
    c.register_capability(
        "lews_section",
        "mind.lews_section(kind, sid, meta, arrays) builds a section for a lews container stamped with that kind's "
        "schema version, so a reader can tell an old section from a new one and migrate rather than guess. "
        "The canonical builders for the shipped kinds live beside it (holographic_lews.make_section).",
        example="import lecore; m=lecore.UnifiedMind(dim=64,seed=0); print(m.lews_section('lecore.note', 's1', {'text': 'hi'}, {})['meta'])",
        aliases=("make a container section", "versioned section", "build a lews section", "stamp a schema version",
                 "add a section to a lews file", "typed section for the container"),
        module="holographic_lews", method="lews_section", native=True,
    )
    c.register_capability(
        "shared_workspace",
        "mind.shared_workspace() returns the swarm's shared workspace: named slots that roles read and write "
        "while collaborating on a scene, so a modeller's output is a rigger's input without a file in between. "
        "Read to inspect what the roles have exchanged; empty until a swarm runs.",
        example="import lecore; m=lecore.UnifiedMind(dim=64,seed=0); print(type(m.shared_workspace()).__name__)",
        aliases=("swarm workspace", "shared slots between roles", "what did the roles exchange",
                 "blackboard for the swarm modeller inspect", "shared scene state", "role handoff workspace"),
        module="holographic_innereye", method="shared_workspace", native=True,
    )
    c.register_capability(
        "systemone_decide",
        "TYPED decisions with honest probabilities (holographic_systemone, the native System One "
        "door -- what Jev sells as an API): answer typed questions -- choice, score, noul (yes/no) "
        "-- about one state in ONE pass. Schema validated up front so an untyped output is "
        "impossible; ties ABSTAIN (value None, why named); p appears only after labeled outcomes "
        "calibrate it. Every choice carries ranked evidence and its basis. scorer='nb' (sweeps "
        "174-175): transformed naive Bayes over the same examples, measured 0.843 vs 0.716 on AG "
        "News k=300 and 0.798 vs 0.753 on Banking77 (77 intents) k=35.",
        example="import lecore; m=lecore.UnifiedMind(dim=512,seed=0); "
        "print(m.systemone_decide('card charged twice on my invoice', {'cat': {'type': 'choice', "
        "'options': ['billing','shipping'], 'examples': {'billing': ['invoice charge refund', "
        "'card charged a fee'], 'shipping': ['package tracking courier', 'parcel lost']}}}, "
        "scorer='nb', margin=0.05))",
        aliases=("jev", "system one model", "typed decision with confidence",
                 "encode text with character ngrams",
                 "route a ticket and abstain when unsure", "yes no probability",
                 "classify with calibrated probability", "structured output guaranteed",
                 "abstain when the input matches no option", "off-manifold support floor",
                 "decision api like jev", "sentiment classification from examples",
                 "few-shot text classification with examples", "naive bayes text classifier",
                 "calibrated probability for a classifier", "risk coverage curve accuracy on answered",
                 "classify text into categories from examples", "prediction set with coverage guarantee",
                 "conformal answer set", "which options could it be with 95 percent coverage",
                 "drift report on a stream of decisions", "minimum segment length for the drift alarm",
                 "has the input distribution shifted"),
        module="holographic_systemone", method="systemone_decide", native=True,
    )
    c.register_capability(
        "systemone_map",
        "BATCH typed decisions (holographic_systemone): mind.systemone_map(states, questions, "
        "labeled=None) maps one decision schema over many states -- score every row of a table, "
        "triage a whole queue -- with row i pinned identical to the single-state call (the batch "
        "path can never fork). Fit-once for volume: so = mind.systemone(questions); so.decide_map(...); "
        "so.calibrate(labeled) adds honest probabilities; so.calibration_report(held_out) gives "
        "accuracy, Brier and ECE -- the reliability numbers a vendor claim needs before trust.",
        example="import lecore; m=lecore.UnifiedMind(dim=512,seed=0); "
        "print(m.systemone_map(['refund my invoice','parcel is lost'], {'cat': {'type': 'choice', "
        "'options': ['billing','shipping'], 'examples': {'billing': ['invoice charge refund'], "
        "'shipping': ['package tracking courier']}}})[1]['cat']['value'])",
        aliases=("map a decision over rows", "bulk classify a queue of tickets",
                 "score every row with confidence", "batch triage with abstention",
                 "calibration report brier ece", "fit once decide many"),
        module="holographic_systemone", method="systemone_map", native=True,
    )
    c.register_capability(
        "systemone_stream",
        "CLOSE THE DECISION LOOP (holographic_systemone, sweep 172 -- what a frozen hosted "
        "decision model cannot do): systemone_stream(stream, questions, lr) runs prequential "
        "test-then-train over [(state,{q:truth}),...] -- decide FIRST (the honest test), then "
        "learn via the AdaptHD miss-update; lr=0 is the frozen baseline. Returns prequential "
        "accuracy plus a two-channel drift report (label-free support + label-lagged "
        "correctness) delegated to the regime machinery. The fitted object also gains "
        "decide_or_escalate: a schema-enforced model-end seam.",
        example="import lecore; m=lecore.UnifiedMind(dim=512,seed=0); "
        "print(m.systemone_stream([('invoice charge',{'cat':'billing'}),('parcel lost',"
        "{'cat':'shipping'}),('refund me',{'cat':'billing'})], {'cat': {'type':'choice',"
        "'options':['billing','shipping'], 'examples': {'billing':['card charged fee'],"
        "'shipping':['package courier delivery']}}})['prequential_accuracy'])",
        aliases=("learn from decision outcomes online", "prequential evaluation of decisions",
                 "detect drift in a decision stream", "close the decision loop",
                 "online prototype update from mistakes", "escalate with schema enforced",
                 "continual typed decisions"),
        module="holographic_systemone", method="systemone_stream", native=True,
    )
    c.register_capability(
        "decision_tree",
        "GROW THE SUGGESTION NODE INTO A TREE (holographic_decisiontree, sweep 173). route() builds one "
        "decision node on the fly; decision_tree(context, depth, fanout) recurses it: each node ranks "
        "the live catalog, takes decide_or_abstain, and branches on what that choice RESULTS in, plus an "
        "abstain branch. The child context is the request PLUS what just happened -- measured, that edge "
        "kept 4/5 children on topic vs 0/5 for produces/consumes chaining. Returns a PlanNode; encode=True "
        "adds it as one hypervector. A 3-option node holds the right answer 0.696 vs 0.540 for top-1.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "t=m.decision_tree('turn a point cloud into a mesh', depth=2); "
        "print(t['root'].action, list(t['root'].branches), t['coverage'])",
        aliases=("decision tree from context", "build a decision tree on the fly",
                 "a decision tree that learns which option I picked", "walk the tree again and remember my choice",
                 "walk a tree of choices to pick an action", "expand a suggestion into next steps",
                 "what can I do after this step", "did you mean one of these, then what",
                 "contingency tree over the capability catalog", "menu of choices with follow-ups",
                 "route a request through a series of choices"),
        module="holographic_decisiontree", method="decision_tree", native=True,
    )
    c.register_capability(
        "decision_memory",
        "OUTCOME AUDIT LOG with an equivalence test (holographic_decisiontree, sweep 173): record that an "
        "input, through a tree, produced a result; ask which inputs reached the SAME result; compare() "
        "gives four quadrants -- consistent / equivalent / brittle (alike inputs, DIFFERENT results: the "
        "alarm) / distinct. Inputs encode by HOW THEY ROUTE (routing_fingerprint): same vs different result "
        "pairs at AUROC 0.874 vs 0.777 (bag) and 0.695 (word overlap). NOT a router or predictor -- as "
        "either it lost to a proper baseline. recall() abstains; one class needs a stated floor.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); om=m.decision_memory(); "
        "om.record('turn a point cloud into a mesh','points_to_mesh',None,label='a'); "
        "om.record('build a surface from these points','points_to_mesh',None,label='b'); "
        "print(om.classes(), om.compare('turn a point cloud into a mesh','build a surface from these points')['quadrant'])",
        aliases=("remember that this input gave this result", "which inputs lead to the same result",
                 "group inputs by outcome", "same result from different inputs",
                 "alarm when similar inputs give different results", "brittle decision detector",
                 "outcome equivalence classes", "audit log of decisions and results",
                 "record what a choice produced"),
        module="holographic_decisiontree", method="decision_memory", native=True,
    )

    c.register_capability(
        "route_tiered",
        "ROUTE WITHOUT A BARE REJECTION (sweep 176): answer / menu / clarify / refuse instead of yes-or-no. "
        "Reuses route_or_abstain's null-referenced z and find_scored's ranking; 'answer' needs z above a floor "
        "and no exact tie, 'clarify' fires when the top candidates span two families, 'menu' returns the top-k "
        "with scores, 'refuse' only in the gibberish band. Every result carries an id to report an outcome "
        "against. Measured on held-out paraphrases (ablated, 3 seeds): the old gate rejected 94.7%; tiered "
        "answers 24.9% at 0.833, menus 29.8% holding the answer 83%, refuses 34%; 0 of 7 gibberish answered.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); r=m.route_tiered('smooth a bumpy mesh'); "
        "print(r['tier'], r['answer'], r['id'], [o['name'] for o in r['options'][:3]])",
        aliases=("route without rejecting", "did you mean one of these", "answer or offer a menu",
                 "tiered routing", "never a bare no", "which capability did I mean",
                 "clarify which family I mean", "route with options instead of abstain",
                 "why was this request refused"),
        module="holographic_catalog", method="route_tiered", native=True,
    )
    c.register_capability(
        "catalog_families",
        "WHICH FAMILY IS EACH CAPABILITY IN (sweep 176): {name: (holographic/<family>/, source)} for every "
        "card. 'module' when the module folder resolves it deterministically (497 of 876), 'decided' when a typed "
        "decision over the card's own text is confident (measured 0.625 forced vs 0.280 majority, so only above "
        "the margin; +17), else None -- reported by catalog_gaps, never guessed. The family is what the tiered "
        "router's clarify tier asks about.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); f=m.catalog_families(); "
        "print(sum(1 for v in f.values() if v[0]), 'resolved of', len(f))",
        aliases=("which family is this capability in", "group capabilities by folder",
                 "unresolved capability families", "capability family map", "what folder does this tool live in"),
        module="holographic_catalog", method="catalog_families", native=True,
    )
    c.register_capability(
        "systemone_lint",
        "LINT THE QUESTION BEFORE ASKING IT (sweep 176): checks a typed-decision schema and its states -- example "
        "token budgets imbalanced more than 2x (the shortest option owns the smoothing floor: measured 1-3 of 8 "
        "tool decisions until balanced, then 8 of 8), options with fewer than 3 examples, the scorer the measured "
        "regime table recommends from k, states carrying more than one clause (returned split), and contrastive "
        "or negated states to escalate. Every failure the blueprint and 3-D experiments hit is a finding here. "
        "Reports; never rewrites.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "print(m.systemone_lint({'cat':{'type':'choice','options':['a','b'],'examples':{'a':['x y z']*3,'b':['q']*3}}}, "
        "states=['first thing. and then a second thing but not a third'])['findings'])",
        aliases=("lint a decision schema before asking", "check my examples are balanced",
                 "split a request into clauses", "is this sentence contrastive", "which scorer should I use",
                 "why did my typed decision abstain on everything", "one observation per state"),
        module="holographic_systemone", method="systemone_lint", native=True,
    )
    c.register_capability(
        "decision_outcome",
        "REPORT AN OUTCOME BY ID -- the one outcome path (sweep 176, backlog G1/G2). Every route_tiered and "
        "systemone_decide result carries an id; decision_outcome(id, truth) records what happened and, for a "
        "typed decision, forwards to the fitted SystemOne's observe() -- no teach() anywhere; the model is kept "
        "per schema so the next call decides from what it learned. Measured: a prequential stream through the "
        "doors equals systemone_stream (0.700 = 0.700, 120 AG News rows). Records are HRR-encoded; "
        "decision_records() lists them; similar() finds a like decision by cosine.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "a=m.systemone_decide('parcel lost', {'cat':{'type':'choice','options':['billing','shipping'],"
        "'examples':{'billing':['card charged twice','refund the invoice'],'shipping':['courier late','package never came']}}}, "
        "scorer='nb', margin=0.0)['cat']; print(m.decision_outcome(a['id'], 'shipping')['was_correct'], m.decision_records()['stats'])",
        aliases=("report the outcome of a decision", "tell the system what actually happened",
                 "close the loop on a decision by id", "decision ledger", "list my past decisions",
                 "was that decision right", "record what was used", "outcome feedback by id",
                 "save my decisions to memory", "remember past decisions across sessions",
                 "generate the prompt for the model end from the schema",
                 "make the reflex arc learn from use", "answer a repeated request from experience",
                 "the router learns which capability I used", "reflex bridge"),
        module="holographic_decisionrecord", method="decision_outcome", native=True,
    )
    c.register_capability(
        "systemone_batch_fdr",
        "FALSE-DISCOVERY CONTROL OVER A BATCH OF DECISIONS (sweep 176): one calibrated p per decision cannot bound "
        "the error of a thousand-row batch (the Cranmer seat, sweep 172). Each state gets a shuffle-null p-value of "
        "its margin against in-vocabulary word salad at matched length, then Benjamini-Hochberg across the batch. "
        "Measured on 150 real + 150 noise rows x 3 seeds: BH holds FDR 0.033 at q=0.05 and 0.028 at q=0.10 (real rows "
        "accepted 13-20%); the uncorrected gate lets noise through at 0.14 / 0.22; Benjamini-Yekutieli accepts nothing.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "q={'cat':{'type':'choice','options':['billing','shipping'],'examples':{'billing':['card charged twice','refund the invoice','charge on my statement'],'shipping':['parcel lost','courier late','package never came']}}}; "
        "print(m.systemone_batch_fdr(['my card was charged twice','courier lost it'], q, 'cat', alpha=0.1, n_null=32, encoder='ngram', scorer='nb'))",
        aliases=("false discovery rate over a batch of decisions", "control how many wrong accepts in a batch",
                 "benjamini hochberg over decisions", "shuffle null p value for a decision",
                 "bound the batch error of a decision stream"),
        module="holographic_systemone", method="systemone_batch_fdr", native=True,
    )
    c.register_capability(
        "systemone_absorb",
        "LEARN FROM UNLABELED TRAFFIC, GUARDED (sweep 176): semi-supervised EM over unlabeled states for the nb "
        "scorer -- E-step posteriors, M-step recount, no gradient -- on the same cached model the next decision "
        "uses. Two measured guards: refused above 10 options (EM collapsed to 0.093 on 77 intents) and kept only "
        "if held-out accuracy did not fall. Measured: plain nb on AG News, 32 examples per class plus 600 "
        "unlabeled rows, 0.697 -> 0.752, beating the transformed default's 0.713; under the transform the gain "
        "vanishes, so this door defaults to nb_transform=False.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "q={'cat':{'type':'choice','options':['billing','shipping'],'examples':{'billing':['card charged twice','refund my invoice fee','charge on my statement','double charge'],'shipping':['parcel lost in transit','courier late','package never arrived','no tracking movement']}}}; "
        "print(m.systemone_absorb(['my invoice was charged twice','the courier lost the parcel'], q, 'cat', encoder='ngram', margin=0.0))",
        aliases=("learn from unlabeled examples", "semi-supervised typed decision", "absorb unlabeled traffic",
                 "use unlabeled data to improve a classifier", "expectation maximization for naive bayes"),
        module="holographic_systemone", method="systemone_absorb", native=True,
    )
    c.register_capability(
        "swarm_step",
        "THE SWARM STEP CONTRACT (sweep 176): one shape for every message a worker sends -- state, tool, args, "
        "done_when, evidence, worker, outcome -- REFUSED without done_when and evidence, because the orchestrator "
        "failure the literature names (misclassification that compounds) starts with steps nobody can verify. "
        "Each step is a DecisionRecord in the ledger (outcome by id, compare by cosine) and is published on the "
        "mind's MessageBus. swarm_evaluate runs the audit suite (reachability_audit, catalog_gaps, skill_lint) as "
        "subprocesses and returns all_ok -- the evaluator role and the hard exit.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "s=m.swarm_step('resolve families', 'catalog_families', {'decide': False}, done_when='at least 500 resolve', "
        "evidence={'resolved': 501}, worker='w1'); print(s['id'], s['via'], len(m.bus().history('swarm')))",
        aliases=("publish a swarm step with done_when and evidence", "swarm step contract", "worker step message",
                 "run the audits as the evaluator", "hard exit for a swarm run", "step record on the bus",
                 "orchestrate workers with verifiable steps", "validated termination", "verify a step before accepting it",
                 "NOOA style typed result with evidence and a verification command",
                 "run the reachability audit", "run catalog gaps and skill lint", "run the wiring audits after a change"),
        module="holographic_decisionrecord", method="swarm_step", native=True,
    )
    c.register_capability(
        "sdf_scene_shader",
        "ONE EXACT SCENE, ONE FRAGMENT SHADER (sweep 176): a multi-part SDF scene as a WebGL2 raymarcher -- each "
        "part's map() from the engine's own emitter, combined by min, writing the nearest-part id per pixel, rays "
        "from the SAME basis Camera.ray_dirs uses, so the preview is the render's geometry. Measured on the 15-part "
        "speaker scene: silhouette IoU 0.9852 vs the engine's sphere-trace, part-id agreement 0.9924, every "
        "disagreement on an edge; 2 s per frame in a browser vs minutes for the path trace.",
        example="import lecore; from holographic.mesh_and_geometry.holographic_sdf import sphere, box; "
        "from holographic.rendering.holographic_render import Camera; m=lecore.UnifiedMind(dim=256,seed=0); "
        "sc=m.sdf_scene_shader([('ball', sphere(0.3)), ('slab', box(1,0.05,1).translate((0,-0.4,0)))], camera=Camera(eye=(0,1,3),target=(0,0,0),fov_deg=30,aspect=1.6)); "
        "print(sc['names'], 'mapAll' in sc['fragment'], sc['uniforms']['uAspect'])",
        aliases=("compile a whole scene to a shader", "webgl preview of the exact scene", "scene to glsl with part ids",
                 "raymarch the scene in the browser", "make the viewport match the render", "shader for a multi-part sdf scene"),
        module="holographic_sdfemit", method="sdf_scene_shader", native=True,
    )
    c.register_capability(
        "plan_from_request",
        "A COLD PLAN WITH NO MODEL (sweep 176): split a compound request into single-observation clauses, route "
        "each through the tiered router, and chain the steps into a PlanNode contingency plan encoded as one "
        "hypervector. Measured on compound requests of two exact aliases (3 seeds x 100): both steps recovered "
        "0.860 vs 0.540 for the whole request's menu. The first measurement (0.527) was a splitter defect found "
        "by re-testing against a single alias's routing (0.990); ', next ' is now a connective.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "p=m.plan_from_request('smooth a bumpy mesh and then grow crystals on a surface'); "
        "print([(s['tier'], s['tool'][:24] if s['tool'] else None) for s in p['steps']], p['root'].action[:24])",
        aliases=("break a request into steps", "plan a sequence of tool calls without a model", "cold plan from a request",
                 "decompose a compound request", "route each clause of a request", "multi-step plan from text"),
        module="holographic_systemone", method="plan_from_request", native=True,
    )
    c.register_capability(
        "reflex_retile",
        "RE-TILE THE EXPERIENCE TRACE AT THE MEASURED CLIFF (sweep 176): rebuild the lever-7 trace from every "
        "tile's bit-identical audit log with a new capacity advisory. Measured: one 2048-d tile reads an exact "
        "repeat back at 1.000 up to 50 writes, 0.93 at 100, ~0.6 at 150, while the old default split at 205 -- "
        "after the cliff -- so a trace loaded by boot (953 writes on 6 tiles) answered a just-taught repeat below "
        "the confidence floor; re-tiled to 14 tiles the same repeat fired via reflex. Run it after boot on a "
        "trace persisted before sweep 176; new traces tile at the cliff by default.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "r=m.route_tiered('smooth a bumpy mesh', reflex=True); m.decision_outcome(r['id'], r['answer']); "
        "print(m.reflex_retile(advisory_load=0.03), m.route_tiered('smooth a bumpy mesh', reflex=True)['via'])",
        aliases=("retile the reflex trace", "rebuild the experience trace after boot", "reflex trace past capacity",
                 "the reflex stopped firing after loading memory", "split the experience trace into more tiles"),
        module="holographic_lever7", method="reflex_retile", native=True,
    )
    c.register_capability(
        "verify_decision",
        "IS THIS ANSWER A VALID RESPONSE TO THIS INPUT? (sweep 176) leOS step 4 done holographically: read the "
        "experience trace with the STATE and check it cleans up to the answer (forward); read it with the ANSWER "
        "atom and check it points back at the state (backward -- the bidirectional lookup); the displacement "
        "profile of correct pairs; the seen gate; support against the recent stream (drift). "
        "Measured: a seen state served the recorded truth vs a wrong label -- forward and backward AUROC 1.000, "
        "verdict valid 0.99 vs 0.00. A verdict against experience, not a confidence; the margin ranks.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
        "r=m.route_tiered('smooth a bumpy mesh'); m.decision_outcome(r['id'], r['answer']); "
        "print(m.verify_decision('smooth a bumpy mesh', r['answer'], key='fingerprint')['valid'], "
        "m.verify_decision('smooth a bumpy mesh', 'Voxelization', key='fingerprint')['valid'])",
        aliases=("verify the answer matches the input", "is this result valid for this prompt",
                 "bidirectional lookup check", "does experience agree with this answer",
                 "catch an answer that contradicts what we learned", "validate a served result",
                 "check a decision against drift"),
        module="holographic_decisionrecord", method="verify_decision", native=True,
    )
    c.register_capability(
        "plan_change",
        "PLAN A CODE CHANGE -- Rule 0 as a typed decision with the evidence attached (sweep 176): what the catalog "
        "says (route_tiered tier and z), what the source says (code_search), the family, then reuse / extend / "
        "build -- the measured tier decides where it is decisive, modification intent ('add a parameter to X') "
        "turns an answer-tier hit into extend, the typed decision breaks the menu tie and learns from "
        "decision_outcome(id, what was done). Returns the build loop as steps with a done_when each. Measured on "
        "12 historical requests: 11 of 12 against today's truth (majority 0.33).",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); m.set_file_root('.'); "
        "p=m.plan_change('add a naive bayes scorer to the typed decision'); print(p['action'], p['evidence']['catalog_tier'], [s['step'] for s in p['steps']][:3])",
        aliases=("plan a code change", "should I reuse extend or build", "rule zero as a decision",
                 "where does this feature go", "which module should a new feature go in", "build loop steps with done when"),
        module="holographic_codeflow", method="plan_change", native=True,
    )
    c.register_capability(
        "edit_verified",
        "ONE EDIT UNDER VALIDATED TERMINATION (sweep 176, NOOA): file_replace, then the checks -- syntax always, "
        "import and the module selftest when asked; ANY failure undoes the edit and refuses with the failing check "
        "attached, so a broken file never survives the call; an accepted edit is a swarm step in the ledger. "
        "Measured on 200 synthetic edits: 100 of 100 broken edits refused and restored byte-identical, 100 of 100 "
        "good edits accepted.",
        example="import lecore, tempfile, os; d=tempfile.mkdtemp(); open(os.path.join(d,'m.py'),'w').write('def f(): return 1'); "
        "m=lecore.UnifiedMind(dim=256,seed=0); m.set_file_root(d); r=m.edit_verified('m.py', 'return 1', 'return ('); "
        "print(r['ok'], open(os.path.join(d,'m.py')).read()=='def f(): return 1')",
        aliases=("edit a file and verify the edit", "replace text and check it still compiles", "safe edit with undo on failure",
                 "edit with validated termination", "an edit that cannot leave the file broken"),
        module="holographic_codeflow", method="edit_verified", native=True,
    )
    c.register_capability(
        "review",
        "REVIEW A CHANGE WITH EVIDENCE (sweep 176): per file -- syntax, import in a subprocess, determinism hazards "
        "from the AST with line and reason (hash(), unseeded random, wall clock, unsorted listdir or glob, set "
        "iteration: the constitution's rules), undocumented public defs, functions over 120 lines, modules over the "
        "2,000-line part cap, a missing selftest, possible duplicates by code_similar, impure functions by "
        "function_purity, and the tests the change needs by affected_tests. merge_ready is a stated rule (no "
        "errors), never a score; the review is a record -- report merged or reverted by id.",
        example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); m.set_file_root('.'); "
        "r=m.review('holographic/agents_and_reasoning/holographic_codeflow.py'); print(r['merge_ready'], list(r['files'].values())[0]['counts'])",
        aliases=("review a code change before merging", "code review checklist", "check determinism of code hash time random",
                 "is this change merge ready", "does this new function duplicate an existing one", "which tests does my change need",
                 "find undocumented public functions", "git diff review"),
        module="holographic_codeflow", method="review", native=True,
    )

_PART = "holographic_catalog_p08"


def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p08)
    return {"part": _PART, "cards": n}


if __name__ == "__main__":
    print(_selftest())

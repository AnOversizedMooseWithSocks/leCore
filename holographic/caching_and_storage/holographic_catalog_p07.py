"""holographic_catalog_p07.py -- catalog part 07: THE ORPHAN-FAMILY CARDS.

Provenance worth keeping: this part was produced by the audit-and-improve loop with leCore as
the tool AND Claude attached as leCore's model (checkpoint 10): orphan_audit FAILED at 53
orphans against a budget of 50; the orchestrator ran the audit as INGEST steps and called the
attached model exactly twice (plan + one decision); the model's recorded decision was to
CATALOGUE the coherent query-helper family -- the audit's own 'catalogue it' arm -- because
the defect was absence from the catalog, not the functions themselves. Deletion was forbidden
by the audit's rule; bulk negative-declaration was wrong because these are useful.
"""


def register_p07(c):
    c.register_capability(
        "near_duplicates",
        "Find NEAR-DUPLICATE rows in a table by fuzzy hypervector similarity above a stated "
        "threshold -- dedupe candidates with scores, not silent merging.",
        example="from holographic.agents_and_reasoning.holographic_query import near_duplicates",
        aliases=("find duplicate rows", "dedupe candidates", "fuzzy duplicates",
                 "which rows are almost the same"), module="holographic_query")
    c.register_capability(
        "not_null",
        "Predicate builder: rows where a column IS PRESENT (the negative-space filter the "
        "query layer already prices like any other predicate).",
        example="db.not_null('email', 'phone')  # Database method: rows where columns are present",
        aliases=("filter missing values", "rows with a value", "is not null"),
        module="holographic_query")
    c.register_capability(
        "similar_to",
        "Predicate builder: rows whose encoded value is SIMILAR TO a probe above a cosine "
        "floor -- fuzzy WHERE, priced and calibrated like the rest of the query layer.",
        example="from holographic.agents_and_reasoning.holographic_query import similar_to",
        aliases=("fuzzy where clause", "rows like this value", "similarity filter"),
        module="holographic_query")
    c.register_capability(
        "run_view",
        "Execute a saved VIEW by name against the current database state (the imperative "
        "companion to create_view; refreshes exactly like the incremental machinery).",
        example="db.run_view('sales.by_region')  # Database method",
        aliases=("execute a saved view", "run a stored query"), module="holographic_query")
    c.register_capability(
        "resolve_reference",
        "Resolve a natural REFERENCE ('that one', 'the red mesh') against recent query results "
        "-- deterministic anaphora over the session's own tables.",
        example="db.resolve_reference(dest_row)  # Database method: anaphora over recent results",
        aliases=("resolve that one", "what did 'it' refer to", "anaphora over results"),
        module="holographic_query")
    c.register_capability(
        "route_question",
        "Classify a QUESTION's intent deterministically (which faculty family should answer) "
        "-- the intent half of the zoo ladder's T2 rung.",
        example="from holographic.agents_and_reasoning.holographic_intent import route_question",
        aliases=("what kind of question is this", "classify intent", "which tool answers this"),
        module="holographic_intent")
    # KEPT NEGATIVE #2 (pass 2 caught pass 1): the first version of these cards wrote
    # module-level import examples for functions that are METHODS -- skill_lint's example
    # resolver flagged all seven. The second auditor audits the first fix; examples below are
    # the true call shapes.
    # -- batch 2: TRUE orphans (kept negative, pinned: batch 1 catalogued six TEST-ONLY
    # functions after misreading the combined --list output -- the cards were worth keeping,
    # but the failing budget counts TRUE orphans only. The audit also caught the auditor:
    # chain_transport below was added by the lever-7 build itself and arrived orphaned.)
    c.register_capability(
        "chain_transport",
        "CERTIFIED warm-start transport for sphere tracing: replay a stored chain prefix on a "
        "neighbouring ray and hand back a t0 whose Lipschitz safety was checked ahead of time "
        "-- 0 mismatches over 576/576 certified pixels (lever-7 build, deep-dive Part 10).",
        example="from holographic.rendering.holographic_raymarch import chain_transport",
        aliases=("warm start a ray", "reuse a sphere trace", "certified ray warm start"),
        module="holographic_raymarch")
    c.register_capability(
        "active_workspace",
        "The CURRENT workspace handle for the scene pipeline -- which staging area edits land "
        "in when none is named.",
        example="ws.active_workspace()  # Workspace method",
        aliases=("which workspace is active", "current staging area"),
        module="holographic_workspace")
    c.register_capability(
        "cool_all",
        "Bulk-COOL every eligible entry in the cold store in one pass (compress now, inflate "
        "on get) -- the maintenance sweep the per-entry cool() implies.",
        example="store.cool_all()  # ColdStore method: bulk-compress every eligible entry",
        aliases=("compress the whole cold store", "bulk cool entries"),
        module="holographic_coldstore")
    c.register_capability(
        "close_mailbox",
        "Close a bus mailbox explicitly, releasing its queue -- the orderly-shutdown half of "
        "open_mailbox.",
        example="bus.close_mailbox('renders')  # Bus method",
        aliases=("shut a message queue", "release a mailbox"), module="holographic_bus")
    c.register_capability(
        "compress_arrays",
        "Compress the array payloads inside a recipe in place (the recipe stays replayable; "
        "the bytes shrink) -- storage hygiene for recipe libraries.",
        example="from holographic.io_and_interop.holographic_recipe import compress_arrays",
        aliases=("shrink recipe payloads", "compact stored recipes"),
        module="holographic_recipe")
    c.register_capability(
        "as_atom",
        "Encode a harmonic descriptor as a clean ATOM vector -- the bridge from the harmonic "
        "layer's structured values into bindable hypervector form.",
        example="harmonic.as_atom()  # Harmonic method: descriptor -> bindable atom",
        aliases=("harmonic to hypervector", "encode a harmonic as an atom"),
        module="holographic_harmonic")
    # -- batch 3 (pass 2 ratchet): the two module-level true orphans, then the budget drops
    c.register_capability(
        "evaluate_candidates",
        "Score CANDIDATE results against a superposed query pack in one pass -- which of these "
        "answers does the bundle actually support, with per-candidate cosines.",
        example="from holographic.misc.holographic_superposed import evaluate_candidates",
        aliases=("score candidates against a bundle", "which answer does the pack support"),
        module="holographic_superposed")
    c.register_capability(
        "export_all",
        "Bulk-EXPORT a model directory's test artifacts (probes, singular values) in one call "
        "-- the testkit's sweep counterpart to its per-item exports.",
        example="from holographic.io_and_interop.holographic_testkit import export_all",
        aliases=("export every test artifact", "bulk testkit export"),
        module="holographic_testkit")
    # -- batch 4 (pass 3, rendering + simulation; def-contexts verified module-level FIRST
    # this time -- pass 2's lesson applied prospectively, not remedially)
    c.register_capability(
        "aces_tonemap",
        "ACES filmic TONEMAP for an HDR buffer -- exposure + auto-key, the display transform "
        "a physically-lit render needs before pixels are viewable.",
        example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=64, seed=0); "
                "print(m.aces_tonemap(np.array([[[0.1, 0.5, 4.0]]]))[0][0][2] < 1.0)",
        aliases=("tonemap an hdr image", "filmic display transform", "make hdr viewable"),
        module="holographic_gbuffer", method="aces_tonemap")
    c.register_capability(
        "add_caustics",
        "Add light CAUSTICS to a rendered image from the scene + camera (light direction and "
        "receiver plane configurable) -- the focused-light pass composited after the beauty.",
        example="from holographic.rendering.holographic_gbuffer import add_caustics",
        aliases=("caustics pass", "focused light patterns", "underwater light dapple"),
        module="holographic_gbuffer")
    c.register_capability(
        "gas_pressure",
        "Ideal-gas PRESSURE from density and temperature for a named gas -- the state "
        "equation the fluid and combustion layers share.",
        example="from holographic.simulation_and_physics.holographic_gas import gas_pressure",
        aliases=("ideal gas state equation", "pressure from density and temperature"),
        module="holographic_gas")
    c.register_capability(
        "is_flammable",
        "Combustion PREDICATE: can this material burn -- the gate the fire propagation "
        "simulation consults per cell.",
        example="from holographic.simulation_and_physics.holographic_combustion import is_flammable",
        aliases=("can this material burn", "combustion check"),
        module="holographic_combustion")
    c.register_capability(
        "barrier_wall",
        "A potential BARRIER WALL for the quantum-dot simulation -- axis, position, thickness, "
        "height, optional gap (the double-slit shape is one call).",
        example="from holographic.simulation_and_physics.holographic_quantum_dot import barrier_wall",
        aliases=("quantum potential barrier", "double slit potential"),
        module="holographic_quantum_dot")
    c.register_capability(
        "from_components",
        "Build a STOKES VECTOR from its polarization components -- the constructor the "
        "polarized-light renderer reads its inputs through.",
        example="from holographic.rendering.holographic_stokes import from_components",
        aliases=("stokes vector from components", "polarization state constructor"),
        module="holographic_stokes")
    c.register_capability(
        "aniso_render",
        "ANISOTROPIC splat render: elongated gaussian splats oriented by local structure -- "
        "the quality tier above isotropic splatting for the same point set.",
        example="from holographic.rendering.holographic_splat import aniso_render",
        aliases=("anisotropic gaussian splatting", "oriented splat render"),
        module="holographic_splat")
    c.register_capability(
        "splat_denoise",
        "Edge-aware DENOISE for a splat render -- smooths the gaussian shimmer while "
        "preserving silhouettes; the cleanup pass between splatting and display.",
        example="from holographic.rendering.holographic_splat import splat_denoise",
        aliases=("denoise a splat render", "smooth gaussian shimmer"),
        module="holographic_splat")
    c.register_capability(
        "element_flame_color",
        "The FLAME COLOR an element burns with (emission spectrum -> RGB) -- the flame-test "
        "palette the combustion renderer colors its fire from.",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); "
                "print(m.element('Na')['flame_color'])",
        aliases=("what color does this element burn", "flame test color"),
        module="holographic_elements", method="element")
    c.register_capability(
        "add_velocity",
        "Inject a VELOCITY impulse into a spectral field simulation -- the stir operator for "
        "the FFT-domain fluid.",
        example="field.add_velocity(source_velocity)  # SpectralField method: stir the FFT-domain fluid",
        aliases=("stir the spectral fluid", "inject velocity impulse"),
        module="holographic_spectralfield")
    # -- batch 6 (pass 4: sql/storage/vm domain; method examples written AS methods from the
    # start -- the pass-2/3 lesson is doctrine now)
    c.register_capability(
        "merge_drift",
        "MERGE two drifted model variants through the mind's reconciliation path -- the "
        "two-replica repair the distribution layer implies.",
        example="from holographic.caching_and_storage.holographic_storeroute import merge_drift",
        aliases=("reconcile drifted replicas", "merge model variants"),
        module="holographic_storeroute")
    c.register_capability(
        "write_multichannel",
        "Steganographic MULTI-CHANNEL write: embed data across weight channels at a stated "
        "overhead and bit budget -- the substrate's covert-capacity demonstration.",
        example="from holographic.caching_and_storage.holographic_substrate import write_multichannel",
        aliases=("embed data in weights", "multichannel steganographic write"),
        module="holographic_substrate")
    c.register_capability(
        "disable_cold_storage",
        "Turn OFF the query layer's cold-storage tiering for a Database (keep everything "
        "hot) -- the benchmarking/debugging switch.",
        example="db.disable_cold_storage()  # Database method",
        aliases=("keep all rows hot", "turn off query tiering"),
        module="holographic_query")
    c.register_capability(
        "query_fuzzy",
        "FUZZY role->value scene query: rows whose encoded role is NEAR the probe value -- "
        "the scene layer's similarity WHERE.",
        example="scene_index.query_fuzzy('material', 'gold-ish')  # SceneQuery method",
        aliases=("fuzzy scene lookup", "similar-value scene query"),
        module="holographic_scene_query")
    c.register_capability(
        "redo_stack",
        "The edit history's REDO STACK (property): what undo has set aside, in order -- the "
        "other half of time travel.",
        example="history.redo_stack  # EditHistory property",
        aliases=("what can be redone", "the undone edits"),
        module="holographic_edithistory")
    c.register_capability(
        "bios_boot",
        "BOOT the substrate like firmware: POST with measured checks (levers, ladder "
        "honesty, container round-trip, calibration, Unicron spectral read), partition "
        "mount, doctrine load, machine inventory -- then os_prompt() hands the attached "
        "LLM its generated operating screen (syscalls, rules, contract).",
        example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); rep = mind.boot(partition='/data/p1'); print(mind.os_prompt(rep))",
        aliases=("boot the mind", "bios", "power on self test", "operating prompt",
                 "os for the llm"), module="holographic_bios", method="boot")
    c.register_capability(
        "doctrine_seedpack",
        "Boot a fresh mind with the DISTILLED OPERATING DOCTRINE: 14 measured, "
        "provenance-tagged lessons (ladder use, agent loops, storage law, honest "
        "benchmarks) taught through the normal reflex gate. Opt-in so cold state "
        "stays honest.",
        example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.doctrine_load()",
        aliases=("seed doctrine", "load the doctrine pack", "boot with lessons",
                 "teach the operating doctrine"), module="holographic_seedpack",
        method="doctrine_load")
    c.register_capability(
        "panel_realm",
        "Seat the expert panel in the swarm realm: each member a named resident with its "
        "own scope in one SHARED store; deliberation follows the swarm's contrast law -- "
        "consensus is silent, only disagreement is recorded, authored per-expert.",
        example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.panel_seat(); mind.panel_deliberate(q, {\'widrow\': \'a\', \'bau\': \'b\'})",
        aliases=("seat the panel", "panel swarm", "expert realm", "council realm"),
        module="holographic_unified_p20_zoo", method="panel_seat")
    c.register_capability(
        "app_substrate",
        "Give an app built on leCore its own memory PER USER: remember/recall with "
        "provenance, observe/suggest/habits (procedures mined from what the user actually "
        "does), a capability preflight, and save/load. Each (app, user) is a separate "
        "partition, so no user can appear in another's memory.",
        example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); app = mind.app_substrate(\'lestudio\', user=\'ana\'); app.suggest(\'retouch a portrait\')",
        aliases=("build on lecore", "app memory", "per user memory", "adapt to the user"),
        module="holographic_appkit")
    c.register_capability(
        "drift_sentinel",
        "leOS's displacement-drift detector on lever 7's floor: classify every "
        "task->response displacement against the neighborhood of similar past tasks. "
        "Verdicts: normal, void (honestly unexplored), echo (a non-answer restating the "
        "task), redshift (off established behaviour), blueshift (too little work), plus "
        "loop detection. teach_check() turns redshift into an IMPLICIT-CONFLICT "
        "candidate with the nearest established answers attached.",
        example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.teach_check(\'where does the deployment run\', \'it was decommissioned\')",
        aliases=("drift detection", "echo detection", "conflict candidate",
                 "stale memory", "loop detection"),
        module="holographic_drift")
    c.register_capability(
        "delegation_drift",
        "Which faculties have LOST a parameter their module function still accepts? "
        "mind.delegation_drift() audits the seam every other audit misses: a parameter added to a "
        "module and never plumbed through its wrapper leaves the capability reachable but crippled -- "
        "/tools lists it, /invoke calls it, part of it cannot be reached. Reports each faculty, its "
        "delegate, the lost parameters and the overlap, plus SUPPLIED: what a wrapper binds itself "
        "(mind=self, seed=self.seed), with its binding, because that is decided, not lost. "
        "KEPT NEG: names, not semantics. Needs a source checkout (logic is in tools/).",
        example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.delegation_drift()",
        aliases=("faculty signature drifted from the function it delegates to",
                 "a mind verb is missing a parameter the module function has",
                 "fix a wrapper that lost an argument",
                 "which faculties cannot forward everything",
                 "wrapper signature audit", "find crippled capabilities",
                 "parameter I cannot pass through the mind",
                 "check my faculty matches its module function"),
        module="delegation_drift", method="delegation_drift")
    c.register_capability(
        "applications",
        "The APPLICATIONS LIBRARY: named end-to-end programs you can run, not snippets to read. "
        "mind.apps() lists them with what each PROVES; mind.app_run(name) runs one and returns the "
        "numbers it asserts plus its runtime. 4 domains, 0.29s total: spectral_heat (a PDE advanced to "
        "any horizon in ONE exact step, 4.4e-16), interleaved_sources (the stride of an "
        "unlabelled round-robin stream recovered for 2, 3 and 4 hidden senders), request_to_record "
        "(requests parsed to {action, object, quality}, unparseable ones refused), texture_composite "
        "(fields blended to a deterministic PNG). All through faculties.",
        example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.apps(); mind.app_run('spectral_heat')",
        aliases=("run a named end to end example program",
                 "a library of runnable applications I can try",
                 "show me a worked example of what this engine does",
                 "cookbook of end to end programs",
                 "list the demo programs", "try a sample application",
                 "what can this engine actually do, show me",
                 "example programs", "run the demo"),
        module="applications", method="app_run")
    c.register_capability(
        "feedback_and_deep_zoom",
        "The demo-scene operator: ITERATE A PROJECTION. mind.feedback_step() is the video-feedback "
        "tunnel -- frame N holding a transform of frame N-1 -- and mind.deep_zoom() is a Mandelbrot "
        "deep zoom RENDERED BY it: 9.7 ms/frame at 320x180 vs 98.3 full, error ~1.1%. ONE "
        "OPERATOR, TWO COSTUMES: on a 1-D hypervector rotate becomes permute, a leaky "
        "echo-state update, and the critical decay is EXACTLY 1.0 in both whenever the transform is "
        "a permutation (mind.is_permutation). mind.zoom_floor() says where float64 ends -- 13.8 "
        "decades -- and deep_zoom STOPS there. KEPT NEG: detects that wall, cannot pass.",
        example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.deep_zoom(frames=8, band=8)",
        aliases=("feedback buffer, feed the last frame back in",
                 "zoom into a mandelbrot forever",
                 "video feedback tunnel effect", "infinite zoom demo",
                 "how deep can I zoom before float64 breaks",
                 "does my feedback buffer converge or blow up",
                 "demoscene effect", "fractal zoom in real time",
                 "as above so below, same structure at every scale",
                 "the same operator on a field and on a sequence",
                 "feedback on a vector instead of an image",
                 "leaky integrator over hypervectors",
                 "decay a sequence and bundle new content into it"),
        module="holographic_feedback", method="deep_zoom")
    return 32


_PART = "holographic_catalog_p07"


def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p07)
    return {"part": _PART, "cards": n}


if __name__ == "__main__":
    print(_selftest())
